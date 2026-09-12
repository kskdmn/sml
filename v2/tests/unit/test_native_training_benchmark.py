from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from sml.artifacts.checkpoint import open_latest_checkpoint_reader
from sml.data import pretraining as prepared_data
from sml.data.pretraining import PretrainingCursor
from sml.errors import SMLConfigurationError
from sml.training.pretrain import _restore_checkpoint
from test_benchmark_analysis import (
    _valid_raw_trial,
    _validate_trial_metadata,
    _with_trial_payload,
)

from v2.benchmarks.adapters import native_training
from v2.benchmarks.adapters import runtime as replacement
from v2.benchmarks.adapters.native import NativeRuntime
from v2.benchmarks.adapters.native_training import CheckpointRuntime, PretrainingRuntime
from v2.benchmarks.runner import measure_native_process
from v2.benchmarks.schema import METRIC_NAMES
from v2.benchmarks.workload import build_canonical_workload, fixed_canonical_rows


def _workload(*, row_count=6):
    return build_canonical_workload(
        model_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 32,
            "original_context_length": 32,
            "hidden_dropout": 0.0,
        },
        optimizer_overrides={
            "gradient_accumulation_steps": 2,
            "swag": {"gradient_accumulation_steps": 2},
        },
        loader_overrides={
            "sequence_length": 4,
            "microbatch_size": 2,
            "swag": {"sequence_length": 8, "batch_size": 1},
        },
        generation_overrides={
            "request_count": 2,
            "prompt_tokens": 4,
            "decode_chunk_size": 2,
        },
        row_count=row_count,
    )


def _record_rows(runtime, monkeypatch):
    recorded = []
    real_next = runtime._next_rows

    def next_rows():
        rows, cursor = real_next()
        recorded.append(np.array(rows))
        return rows, cursor

    monkeypatch.setattr(runtime, "_next_rows", next_rows)
    return recorded


def test_native_training_crosses_epochs_without_repeating_full_preflight(
    tmp_path, monkeypatch
):
    calls = []
    real_open = prepared_data._open_validated_prepared_resources

    def open_prepared(*args, **kwargs):
        calls.append(args[0].path)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(
        prepared_data, "_open_validated_prepared_resources", open_prepared
    )
    runtime = PretrainingRuntime("pretraining-end-to-end", _workload(), tmp_path)
    try:
        startup_calls = len(calls)
        assert startup_calls > 0
        recorded = _record_rows(runtime, monkeypatch)
        assert runtime.run(3) == 48.0
        expected = fixed_canonical_rows(row_count=6, row_width=5, vocab_size=32)
        np.testing.assert_array_equal(
            np.concatenate(recorded), np.tile(expected, (2, 1))
        )
        assert runtime.state.scalar.step == 3
        assert runtime.state.scalar.rows == 12
        assert runtime.state.scalar.cursor == PretrainingCursor(2, 0, 0)
        assert len(calls) == startup_calls
    finally:
        runtime.close()


def test_native_training_prepares_measured_order_before_timing(tmp_path, monkeypatch):
    runtime = PretrainingRuntime("pretraining-end-to-end", _workload(), tmp_path)
    native = SimpleNamespace(metric="pretraining-end-to-end", runtime=runtime)
    try:
        replacement.run_warmup("pretraining-end-to-end", native, 1)
        recorded = _record_rows(runtime, monkeypatch)

        def unexpected_preflight(*_args, **_kwargs):
            raise AssertionError("FULL verification entered the measured interval")

        monkeypatch.setattr(
            prepared_data, "_open_validated_prepared_resources", unexpected_preflight
        )
        assert replacement.run_measured("pretraining-end-to-end", native, 1) == 16.0
        expected = fixed_canonical_rows(row_count=6, row_width=5, vocab_size=32)
        np.testing.assert_array_equal(np.concatenate(recorded), expected[:4])
        assert runtime.state.scalar.step == 2
    finally:
        runtime.close()


def test_compute_runtime_uses_resident_batches_and_restarts_canonical_order(
    tmp_path, monkeypatch
):
    runtime = PretrainingRuntime("pretraining-compute", _workload(), tmp_path)
    try:
        assert runtime.stream is None
        assert all(isinstance(batch, mx.array) for batch in runtime.device_batches)
        recorded = _record_rows(runtime, monkeypatch)
        assert runtime.run(2) == 32.0
        runtime.reset_measured_order()
        assert runtime.run(1) == 16.0
        expected = fixed_canonical_rows(row_count=6, row_width=5, vocab_size=32)
        np.testing.assert_array_equal(
            np.concatenate(recorded),
            np.concatenate((expected, expected[:2], expected[:4])),
        )
        assert runtime.state.scalar.step == 3
        assert int(runtime.state.optimizer.step.item()) == 3
    finally:
        runtime.close()


def test_checkpoint_measurement_excludes_real_update_preparation_from_pause():
    clock = [0.0]
    state = SimpleNamespace(step=0, prepared=False, published=[])

    def prepare():
        assert not state.prepared
        state.step += 1
        state.prepared = True
        clock[0] += 100.0

    def publish(_metric, _native, units):
        assert units == 1
        assert state.prepared
        state.published.append(state.step)
        state.prepared = False
        clock[0] += 2.0
        return float(units)

    native = SimpleNamespace(runtime=SimpleNamespace(prepare_measured_unit=prepare))
    result = measure_native_process(
        metric="checkpoint-pause",
        adapter=SimpleNamespace(run_warmup=publish, run_measured=publish),
        native_workload=native,
        warmup_units=1,
        measured_units=2,
        synchronize=lambda: None,
        clock=lambda: clock[0],
        peak_memory=lambda: 0,
        reset_peak_memory=lambda: None,
    )
    assert state.published == [1, 2, 3, 4]
    assert clock[0] == 408.0
    assert result.compilation_seconds == 2.0
    assert result.elapsed_seconds == 4.0
    assert result.value == 2.0
    assert result.work_count == 2.0


def test_native_checkpoint_publication_restores_prepared_update_and_prunes_previous(
    tmp_path,
):
    runtime = CheckpointRuntime(_workload(), tmp_path)
    try:
        for step in (1, 2):
            runtime.prepare_measured_unit()
            assert runtime.training.state.scalar.step == step
            assert int(runtime.training.state.optimizer.step.item()) == step
            with open_latest_checkpoint_reader(runtime.path) as reader:
                assert reader.resolved.step == step - 1
            assert runtime.run(1) == 1.0
            with open_latest_checkpoint_reader(runtime.path) as reader:
                restored = _restore_checkpoint(reader)
                assert reader.resolved.step == step
                retained_path = reader.resolved.step_directory
                assert restored.scalar == runtime.training.state.scalar
                expected = dict(
                    tree_flatten(runtime.training.state.parameters.master_parameters)
                )
                actual = dict(tree_flatten(restored.parameters.master_parameters))
                assert set(actual) == set(expected)
                for name, expected_array in expected.items():
                    np.testing.assert_array_equal(
                        np.array(actual[name]), np.array(expected_array)
                    )
            assert list((runtime.path / "checkpoints").iterdir()) == [retained_path]
    finally:
        runtime.close()


@pytest.mark.parametrize("metric", METRIC_NAMES)
def test_every_native_metric_measurement_satisfies_trial_validators(metric):
    workload = _workload()
    workload = replace(
        workload,
        work_units=tuple(
            replace(unit, measured_units=2) for unit in workload.work_units
        ),
    )
    native = replacement.resolve_native_workload(metric, workload, Path.cwd())
    assert isinstance(native, replacement.NativeWorkload)
    with pytest.raises(RuntimeError, match="not been verified"):
        _ = native.execution_order_identity
    expected_work = {
        "prepared-data": 2.0,
        "pretraining-compute": 32.0,
        "pretraining-end-to-end": 32.0,
        "swag-end-to-end": 4.0,
        "inference-prefill": 8.0,
        "inference-decode": 4.0,
        "checkpoint-pause": 2.0,
        "compile-cold-start": 32.0,
        "peak-metal-memory": 32.0,
    }
    try:
        result = measure_native_process(
            metric=metric,
            adapter=replacement,
            native_workload=native,
            warmup_units=0 if metric == "compile-cold-start" else 5,
            measured_units=2,
            synchronize=mx.synchronize,
            peak_memory=mx.get_peak_memory,
            reset_peak_memory=mx.reset_peak_memory,
        )
        assert result.work_count == expected_work[metric]
        assert result.value > 0.0
        assert result.elapsed_seconds > 0.0
        reference = _with_trial_payload(
            _valid_raw_trial(workload, metric=metric),
            native_configuration=native.native_configuration,
            native_representation_identity=native.native_representation_identity,
            canonical_row_identity=native.canonical_row_identity,
            canonical_input_identity=native.canonical_input_identity,
            canonical_projection=native.canonical_projection,
            execution_order_identity=native.execution_order_identity,
            initial_parameter_identity=native.initial_parameter_identity,
            startup_verification_seconds=native.startup_verification_seconds,
            elapsed_seconds=result.elapsed_seconds,
            value=result.value,
            compilation_seconds=result.compilation_seconds,
            peak_memory_bytes=result.peak_memory_bytes,
        )
        for protocol in ("baseline", "comparison", "predecessor"):
            _validate_trial_metadata(protocol, workload, reference)
    finally:
        native.runtime.close()


@pytest.mark.parametrize(
    "section,overrides,match",
    [
        ("optimizer", {"betas": [0.9, 0.99, 0.999]}, "exactly two betas"),
        ("optimizer", {"extra": True}, "unsupported fields"),
        (
            "optimizer",
            {"gradient_accumulation_steps": 2.5},
            "gradient_accumulation_steps",
        ),
        (
            "optimizer",
            {"gradient_accumulation_steps": True},
            "gradient_accumulation_steps",
        ),
        ("optimizer", {"total_steps": 12.5}, "schedule_steps"),
        ("optimizer", {"seed": True}, "seed"),
        ("loader", {"extra": True}, "unsupported fields"),
        ("loader", {"unknown_dtype": "int32"}, "unsupported fields"),
        ("loader", {"microbatch_size": 0}, "positive integer"),
        ("loader", {"microbatch_size": 1.5}, "positive integer"),
        ("precision", {"compute_dtype": "float32"}, "precision contract"),
    ],
)
def test_native_training_factory_rejects_ignored_or_coerced_overrides(
    section, overrides, match
):
    workload = _workload()
    malformed = replace(
        workload, **{section: {**getattr(workload, section), **overrides}}
    )
    with pytest.raises((ValueError, TypeError, SMLConfigurationError), match=match):
        NativeRuntime("pretraining-compute", malformed)


def test_checkpoint_constructor_closes_training_resources_when_publication_fails(
    monkeypatch,
):
    closed = []
    real_close = PretrainingRuntime.close

    def close(runtime):
        assert runtime.stream is not None
        real_close(runtime)
        closed.append(runtime)

    def fail_publication(*_args, **_kwargs):
        raise OSError("injected checkpoint publication failure")

    monkeypatch.setattr(PretrainingRuntime, "close", close)
    monkeypatch.setattr(native_training, "publish_run", fail_publication)
    with pytest.raises(OSError, match="injected checkpoint publication failure"):
        NativeRuntime("checkpoint-pause", _workload())
    assert len(closed) == 1
    assert closed[0].stream is None
    assert not closed[0].config.output_run.parent.exists()
