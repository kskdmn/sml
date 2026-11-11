from __future__ import annotations

import threading

import numpy as np
import pytest
from sml.artifacts.manifest import VerificationLevel, row_content_identity

from v2.benchmarks.adapters.prepared_data import build_prepared_data_benchmark_workload
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


def test_benchmark_factory_rejects_non_prepared_metric_before_runtime_start():
    workload = _small_benchmark_workload()
    threads_before = _prefetch_threads()

    with pytest.raises(ValueError, match="prepared-data"):
        build_prepared_data_benchmark_workload("pretraining-compute", workload)

    assert _prefetch_threads() == threads_before


def test_benchmark_runtime_proves_both_row_identity_domains():
    workload = _small_benchmark_workload()
    runtime = build_prepared_data_benchmark_workload("prepared-data", workload)
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
    runtime = build_prepared_data_benchmark_workload("prepared-data", workload)
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
    runtime = build_prepared_data_benchmark_workload(
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
    runtime = build_prepared_data_benchmark_workload(
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
