from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import VerificationLevel, row_content_identity
from sml.data import pretraining as pretraining_module
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    BatchEnvelope,
    PretrainingCursor,
    PretrainingPreparationConfig,
    _windowed_row_shuffle,
    pack_token_ranges,
)

from v2.benchmarks.workload import (
    build_canonical_workload,
    fixed_canonical_rows,
    semantic_row_content_identity,
)


def _small_benchmark_workload(*, row_count: int = 32):
    return build_canonical_workload(
        model_overrides={"vocab_size": 32},
        loader_overrides={"sequence_length": 8},
        row_count=row_count,
    )


def _prefetch_threads() -> tuple[threading.Thread, ...]:
    return tuple(
        thread
        for thread in threading.enumerate()
        if thread.name == "sml-pretraining-prefetch"
    )


class _RecordingMX:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.rows: list[np.ndarray] = []

    def array(self, rows: np.ndarray):
        self.rows.append(np.array(rows, copy=True))
        return self._delegate.array(rows)

    def eval(self, *arrays) -> None:
        self._delegate.eval(*arrays)


class _FailingArrayMX:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    def array(self, _rows: np.ndarray):
        raise RuntimeError("transfer failed")

    def eval(self, *arrays) -> None:
        self._delegate.eval(*arrays)


def preparation_config(**overrides) -> PretrainingPreparationConfig:
    values = {
        "corpus": CorpusConfig(input_root=Path("corpus")),
        "tokenizer_bundle": Path("tokenizer"),
        "sequence_length": 3,
        "shuffle_window_rows": 3,
        "output_shard_rows": 2,
        "seed": 7,
    }
    values.update(overrides)
    return PretrainingPreparationConfig(**values)


def test_packing_overlaps_one_boundary_token_across_token_ranges():
    rows = list(
        pack_token_ranges(
            [[1, 2], [3, 4, 5], [], [6, 7]],
            sequence_length=3,
            vocab_size=8,
        )
    )

    assert [row.tolist() for row in rows] == [[1, 2, 3, 4], [4, 5, 6, 7]]
    assert all(row.dtype == np.dtype("<i4") and row.flags.c_contiguous for row in rows)


@pytest.mark.parametrize(
    ("token_ranges", "message"),
    [
        ([[1, -1, 2, 3]], "nonnegative"),
        ([[1, 2, 8, 3]], "smaller than vocab_size"),
        ([[1.0, 2.0, 3.0, 4.0]], "integer"),
        ([[True, False, True, False]], "integer"),
    ],
)
def test_packing_rejects_invalid_token_ranges_before_emitting_rows(
    token_ranges, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        list(
            pack_token_ranges(
                token_ranges,
                sequence_length=3,
                vocab_size=8,
            )
        )


def test_packing_discards_only_the_final_incomplete_tail():
    rows = list(pack_token_ranges([[1, 2, 3, 4, 5, 6]], sequence_length=3))

    assert [row.tolist() for row in rows] == [[1, 2, 3, 4]]


def test_window_shuffle_has_a_fixed_pcg64_order_for_full_and_partial_windows():
    rows = (np.array([index, index + 10], dtype="<i4") for index in range(8))

    shuffled = list(_windowed_row_shuffle(rows, window_rows=3, seed=7))

    assert [int(row[0]) for row in shuffled] == [0, 2, 1, 4, 5, 3, 6, 7]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"corpus": object()}, TypeError, "corpus"),
        ({"tokenizer_bundle": "tokenizer"}, TypeError, "tokenizer_bundle"),
        ({"sequence_length": 0}, ValueError, "sequence_length"),
        ({"shuffle_window_rows": 0}, ValueError, "shuffle_window_rows"),
        ({"output_shard_rows": 0}, ValueError, "output_shard_rows"),
        ({"seed": True}, TypeError, "seed"),
        (
            {"shuffle_algorithm": "windowed-row-shuffle-v2"},
            ValueError,
            "shuffle_algorithm",
        ),
    ],
)
def test_preparation_config_rejects_invalid_values(overrides, error, message):
    with pytest.raises(error, match=message):
        preparation_config(**overrides)


def test_preparation_config_expands_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = preparation_config(tokenizer_bundle=Path("~/tokenizer"))

    assert config.tokenizer_bundle == tmp_path / "tokenizer"


def test_pretraining_cursor_is_canonical_and_rejects_non_plain_integers():
    assert PretrainingCursor.initial() == PretrainingCursor(
        epoch=0,
        shard_order_position=0,
        row_offset=0,
    )

    for invalid in (True, 1.0, np.int64(1)):
        with pytest.raises(TypeError, match="epoch"):
            PretrainingCursor(invalid, 0, 0)
    with pytest.raises(ValueError, match="row_offset"):
        PretrainingCursor(0, 0, -1)


def test_batch_envelope_exposes_read_only_rows_and_releases_idempotently():
    rows = np.arange(12, dtype="<i4").reshape(3, 4)
    cursor = PretrainingCursor(0, 1, 1)
    envelope = BatchEnvelope(rows, cursor)

    with envelope as entered:
        assert entered is envelope
        assert envelope.rows.shape == (3, 4)
        assert not envelope.rows.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            envelope.rows[0, 0] = 99

    envelope.release()


def test_importing_pretraining_module_does_not_import_heavy_dependencies_itself():
    code = """
import builtins
import sml
import sys

original_import = builtins.__import__
imports = []

def tracked_import(name, *args, **kwargs):
    imports.append(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = tracked_import
try:
    import sml.data.pretraining
finally:
    builtins.__import__ = original_import
print('sentencepiece' in sys.modules, any(name == 'mlx' or name.startswith('mlx.') for name in imports))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "False False"
    assert completed.stderr == ""


def test_benchmark_factory_rejects_non_prepared_metric_before_runtime_start():
    workload = _small_benchmark_workload()
    threads_before = _prefetch_threads()

    with pytest.raises(ValueError, match="prepared-data"):
        pretraining_module.build_benchmark_workload("pretraining-compute", workload)

    assert _prefetch_threads() == threads_before


def test_benchmark_runtime_proves_both_row_identity_domains():
    workload = _small_benchmark_workload()
    runtime = pretraining_module.build_benchmark_workload("prepared-data", workload)
    try:
        canonical = fixed_canonical_rows(
            row_count=32,
            row_width=9,
            vocab_size=32,
        )
        assert runtime.canonical_row_identity == semantic_row_content_identity(
            canonical
        )
        assert runtime.bundle.manifest.row_content_identity == row_content_identity(
            canonical,
            32,
            9,
        )
        assert (
            runtime.bundle.manifest.row_content_identity
            != runtime.canonical_row_identity
        )
        assert runtime.verification_level == "full"
        assert runtime.bundle.verification is VerificationLevel.FULL
    finally:
        runtime.close()


def test_benchmark_runtime_runs_real_stream_and_resets_canonical_order():
    workload = _small_benchmark_workload(row_count=4)
    runtime = pretraining_module.build_benchmark_workload("prepared-data", workload)
    try:
        recorder = _RecordingMX(runtime._mx)
        runtime._mx = recorder
        canonical = fixed_canonical_rows(row_count=4, row_width=9, vocab_size=32)
        assert runtime.run(3) == 3.0
        assert np.concatenate(recorder.rows).tolist() == canonical[:3].tolist()
        runtime.reset_after_warmup()
        recorder.rows.clear()
        assert runtime.run(2) == 2.0
        assert np.concatenate(recorder.rows).tolist() == canonical[:2].tolist()
        runtime.reset_after_warmup()
        runtime.reset_measured_order()
        recorder.rows.clear()
        assert runtime.run(6) == 6.0
        expected = np.concatenate((canonical, canonical[:2]))
        assert np.concatenate(recorder.rows).tolist() == expected.tolist()
    finally:
        runtime.close()


def test_benchmark_runtime_closes_stream_and_temporary_tree_idempotently():
    threads_before = _prefetch_threads()
    runtime = pretraining_module.build_benchmark_workload(
        "prepared-data", _small_benchmark_workload()
    )
    temporary_root = runtime.temporary_root

    assert runtime.run(1) == 1.0
    assert _prefetch_threads() == threads_before
    runtime.close()
    runtime.close()

    assert not temporary_root.exists()
    assert _prefetch_threads() == threads_before
    with pytest.raises(RuntimeError, match="closed"):
        runtime.run(1)


def test_benchmark_runtime_closes_taken_stream_after_consumer_transfer_failure():
    threads_before = _prefetch_threads()
    runtime = pretraining_module.build_benchmark_workload(
        "prepared-data", _small_benchmark_workload()
    )
    temporary_root = runtime.temporary_root
    runtime._mx = _FailingArrayMX(runtime._mx)

    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            runtime.run(1)
        assert _prefetch_threads() == threads_before
    finally:
        runtime.close()

    assert not temporary_root.exists()
    assert _prefetch_threads() == threads_before
