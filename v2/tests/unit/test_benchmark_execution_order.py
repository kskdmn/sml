from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from sml.data import pretraining as prepared_data
from test_native_training_benchmark import _workload

from v2.benchmarks.adapters import runtime as adapter
from v2.benchmarks.adapters.execution_order import ObservedBatch, verify_input_batches
from v2.benchmarks.adapters.native_training import PretrainingRuntime
from v2.benchmarks.runner import measure_native_process
from v2.benchmarks.schema import METRIC_NAMES
from v2.benchmarks.workload import canonical_execution_order, fixed_canonical_rows


def _two_unit_workload():
    workload = _workload()
    return replace(
        workload,
        work_units=tuple(
            replace(unit, measured_units=2) for unit in workload.work_units
        ),
    )


@pytest.mark.parametrize("metric", METRIC_NAMES)
@pytest.mark.parametrize("corruption", ["reordered", "duplicated", "skipped"])
def test_each_metric_rejects_incorrect_observed_work_before_accepting_measurement(
    metric, corruption
):
    workload = _two_unit_workload()
    expected = canonical_execution_order(metric, workload)
    if corruption == "reordered":
        observed = (expected[1], expected[0], *expected[2:])
    elif corruption == "duplicated":
        observed = (expected[0], *expected[:-1])
    else:
        observed = expected[:-1]
    runtime = SimpleNamespace(
        execution_order_identity=None,
        run=lambda units: float(units),
        reset_measured_order=lambda: None,
        observed_execution_order=lambda: observed,
    )
    native = SimpleNamespace(
        metric=metric, canonical_workload=workload, runtime=runtime
    )
    with pytest.raises(RuntimeError, match="observed execution order"):
        measure_native_process(
            metric=metric,
            adapter=adapter,
            native_workload=native,
            warmup_units=0,
            measured_units=2,
            synchronize=lambda: None,
            clock=iter((0.0, 1.0, 2.0, 3.0)).__next__,
            peak_memory=lambda: 1,
            reset_peak_memory=lambda: None,
        )
    assert runtime.execution_order_identity is None


def test_order_reset_and_validation_are_outside_time_and_peak_measurements():
    events = []
    clock = [0.0]

    def timed_work(*_args):
        events.append("work")
        clock[0] += 2.0
        return 1.0

    def begin(*_args):
        events.append("begin")
        clock[0] += 100.0

    def verify(*_args):
        events.append("verify")
        clock[0] += 100.0

    measured = measure_native_process(
        metric="compile-cold-start",
        adapter=SimpleNamespace(
            run_measured=timed_work,
            begin_measured_order=begin,
            verify_measured_order=verify,
        ),
        native_workload=object(),
        warmup_units=0,
        measured_units=1,
        synchronize=lambda: None,
        clock=lambda: clock[0],
        peak_memory=lambda: events.append("peak") or 7,
        reset_peak_memory=lambda: events.append("reset-peak"),
    )
    assert events == ["begin", "reset-peak", "work", "peak", "verify"]
    assert measured.elapsed_seconds == 2.0
    assert measured.peak_memory_bytes == 7


@pytest.mark.parametrize("field", ["rows", "labels", "mask"])
def test_observed_values_are_checked_even_when_reported_ids_are_canonical(field):
    canonical = {
        "rows": np.asarray([[4, 5], [6, 7]], dtype=np.int32),
        "labels": np.asarray([0, 1], dtype=np.int32),
        "mask": np.asarray([[True, False], [False, True]]),
    }
    delivered = {name: values.copy() for name, values in canonical.items()}
    delivered[field] = delivered[field][::-1]
    with pytest.raises(RuntimeError, match="observed input values"):
        verify_input_batches(
            [ObservedBatch(delivered, work_ids=(0, 1))], canonical, batch_size=2
        )


@pytest.mark.parametrize("metric", ["prepared-data", "pretraining-end-to-end"])
@pytest.mark.parametrize("corruption", ["reordered", "duplicated", "skipped"])
def test_real_stream_wrong_rows_reject_even_with_correct_delivered_cursor(
    metric, corruption, monkeypatch
):
    workload = _two_unit_workload()
    native = adapter.resolve_native_workload(metric, workload, Path.cwd())
    canonical = fixed_canonical_rows(row_count=6, row_width=5, vocab_size=32)

    def corrupted_rows(envelope):
        cursor = envelope.cursor_after
        start = cursor.epoch * len(canonical) + cursor.row_offset - 2
        if corruption == "reordered":
            indices = (start + 1, start)
        elif corruption == "duplicated":
            indices = (start, start)
        else:
            indices = (start + 1, start + 2)
        return canonical[[index % len(canonical) for index in indices]]

    try:
        monkeypatch.setattr(
            prepared_data.BatchEnvelope, "rows", property(corrupted_rows)
        )
        adapter.begin_measured_order(metric, native)
        adapter.run_measured(metric, native, 2)
        mx.synchronize()
        with pytest.raises(RuntimeError, match="observed input values"):
            adapter.verify_measured_order(metric, native)
        assert native.runtime.execution_order_identity is None
    finally:
        native.runtime.close()


def test_compute_order_follows_delivered_device_batch_not_loop_index(tmp_path):
    runtime = PretrainingRuntime("pretraining-compute", _workload(), tmp_path)
    try:
        runtime.device_batches[0], runtime.device_batches[1] = (
            runtime.device_batches[1],
            runtime.device_batches[0],
        )
        runtime.run(1)
        with pytest.raises(RuntimeError, match="observed execution order"):
            runtime.observed_execution_order()
    finally:
        runtime.close()
