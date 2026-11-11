from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sml.errors import SMLDataError

from .swag_helpers import (
    VALID_ROW,
    FakeSwagProvider,
    _one_example_bundle,
    tiny_base_model,
    tiny_swag_config,
)


def _one_example_loader() -> SimpleNamespace:
    return SimpleNamespace(prefetch_depth=1, microbatch_size=1, epoch_seed=7)


def test_swag_stream_closes_owned_bundle_on_exhaustion(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    envelopes = list(stream)
    for envelope in envelopes:
        envelope.release()

    assert bundle._closed
    with pytest.raises(StopIteration):
        next(stream)


def test_swag_prefetch_finishes_after_the_requested_epoch(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    loader = SimpleNamespace(
        microbatch_size=1,
        prefetch_depth=4,
        epoch_seed=11,
    )
    with SwagBatchStream(bundle, loader, cursor=SwagCursor.initial()) as stream:
        stream._producer.join(timeout=5)
        assert not stream._producer.is_alive()
        envelopes = list(stream)
        assert len(envelopes) == 1
        assert envelopes[0].cursor_after == SwagCursor(1, 0, 0)
        envelopes[0].release()


def test_swag_epoch_plan_preserves_seeded_order_in_compact_readonly_arrays():
    from sml.data.swag import SwagBucket, _epoch_bucket_plan

    buckets = tuple(
        SwagBucket(
            length=8,
            input_ids=np.zeros((count, 4, 8), dtype=np.int32),
            valid_token_mask=np.zeros((count, 4, 8), dtype=bool),
            score_mask=np.zeros((count, 4, 8), dtype=bool),
            labels=np.zeros(count, dtype=np.int32),
        )
        for count in (4, 0, 3)
    )

    plan = _epoch_bucket_plan(buckets, epoch_seed=11, epoch=2)

    assert [(index, rows.tolist()) for index, rows in plan] == [
        (0, [3, 0, 1, 2]),
        (2, [1, 0, 2]),
    ]
    for _index, rows in plan:
        assert rows.dtype == np.dtype(np.intp)
        assert rows.flags.c_contiguous
        assert not rows.flags.writeable


@pytest.mark.parametrize("array_indices", [False, True])
@pytest.mark.parametrize("row_indices", [(3,), (3, 0, 2, 1), (3, 0, 2), (3, 1)])
def test_swag_batch_gather_preserves_order_owned_storage_and_synthetic_tails(
    row_indices,
    array_indices,
):
    from sml.data.swag import SwagBucket, _assemble_batch_arrays

    length = 8
    batch_size = 4
    ids = np.arange(4 * 4 * length, dtype=np.int32).reshape(4, 4, length)
    bucket = SwagBucket(
        length=length,
        input_ids=ids,
        valid_token_mask=ids % 2 == 0,
        score_mask=ids % 3 == 0,
        labels=np.array([1, 3, 0, 2], dtype=np.int32),
    )
    manifest = SimpleNamespace(pad_token_id=3, bos_token_id=1, eos_token_id=2)
    if array_indices:
        row_indices = np.asarray(row_indices, dtype=np.intp)
    arrays = _assemble_batch_arrays(
        bucket, row_indices, batch_size=batch_size, manifest=manifest
    )
    real_count = len(row_indices)
    indices = np.asarray(row_indices, dtype=np.intp)
    for actual, source in zip(
        arrays[:4],
        (bucket.input_ids, bucket.valid_token_mask, bucket.score_mask, bucket.labels),
        strict=True,
    ):
        np.testing.assert_array_equal(actual[:real_count], source[indices])
        assert not np.shares_memory(actual, source)
    for array in arrays:
        assert array.flags.owndata
        assert array.flags.c_contiguous
        assert not array.flags.writeable

    input_ids, valid_token_mask, score_mask, labels, example_mask = arrays
    np.testing.assert_array_equal(example_mask, np.arange(batch_size) < real_count)
    for slot in range(real_count, batch_size):
        np.testing.assert_array_equal(input_ids[slot], [[1, 2, 3, 3, 3, 3, 3, 3]] * 4)
        np.testing.assert_array_equal(
            valid_token_mask[slot],
            [[True, True, False, False, False, False, False, False]] * 4,
        )
        np.testing.assert_array_equal(
            score_mask[slot],
            [[False, True, False, False, False, False, False, False]] * 4,
        )
        assert labels[slot] == 0


def test_swag_stream_uses_owned_arrays_without_public_bucket_lease(tmp_path):
    """Internal zero-copy iteration must not enter the public borrow protocol."""
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    assert bundle._bucket_leases == 0
    envelope = next(stream)
    try:
        assert bundle._bucket_leases == 0
    finally:
        envelope.release()
        stream.close()
    assert bundle._bucket_leases == 0
    assert bundle._closed


def test_swag_stream_rejects_active_public_lease_without_changing_ownership(
    tmp_path, monkeypatch
):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    lease = bundle.borrow_buckets()
    first = int(lease.buckets[0].input_ids[0, 0, 0])
    transfer_rejected = False
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr("threading.Thread.start", lambda _thread: None)
            with pytest.raises(SMLDataError, match="active bucket leases"):
                SwagBatchStream(
                    bundle,
                    _one_example_loader(),
                    cursor=SwagCursor.initial(),
                )
        assert int(lease.buckets[0].input_ids[0, 0, 0]) == first
        assert bundle._closed is False
        transfer_rejected = True
    finally:
        lease.close()
        if not transfer_rejected:
            bundle.close()

    stream = SwagBatchStream(
        bundle,
        _one_example_loader(),
        cursor=SwagCursor.initial(),
    )
    stream.close()
    assert bundle._closed is True


def test_swag_stream_closes_owned_bundle_on_early_return_and_double_close(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    with stream:
        envelope = next(stream)
        envelope.release()
    stream.close()

    assert bundle._closed


def test_swag_stream_construction_failure_closes_owned_bundle(tmp_path, monkeypatch):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)

    def fail_start(_thread):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr("threading.Thread.start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())

    assert bundle._closed


def test_swag_stream_thread_constructor_failure_closes_owned_bundle(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)

    def fail_thread_construction(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr(swag.threading, "Thread", fail_thread_construction)
    try:
        with pytest.raises(RuntimeError, match="thread construction failed"):
            SwagBatchStream(
                bundle,
                _one_example_loader(),
                cursor=SwagCursor.initial(),
            )
        assert bundle._closed
    finally:
        bundle.close()


@pytest.mark.parametrize("position,offset", [(1, 0), (99, 0), (0, 1), (0, 99)])
def test_swag_stream_rejects_noncanonical_cursor_before_thread_start(
    tmp_path, monkeypatch, position, offset
):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)

    def forbidden_thread(*_args, **_kwargs):
        raise AssertionError("thread created before cursor validation")

    monkeypatch.setattr("sml.data.swag.threading.Thread", forbidden_thread)
    try:
        with pytest.raises(SMLDataError, match="SWAG cursor.*beyond"):
            SwagBatchStream(
                bundle,
                _one_example_loader(),
                cursor=SwagCursor(
                    epoch=0, bucket_order_position=position, row_offset=offset
                ),
            )
        assert bundle._closed
    finally:
        bundle.close()


def test_swag_stream_commits_only_delivered_monotonic_cursors(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor, prepare_swag_bundle

    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,) * 3)),
        tiny_base_model(),
        tmp_path / "swag",
    )
    with SwagBatchStream(
        bundle, _one_example_loader(), cursor=SwagCursor.initial()
    ) as stream:
        stream.commit(SwagCursor.initial())
        with pytest.raises(SMLDataError, match="not delivered"):
            stream.commit(SwagCursor(1, 0, 0))
        first = next(stream)
        second = next(stream)
        first_cursor, second_cursor = first.cursor_after, second.cursor_after
        first.release()
        second.release()
        stream.commit(second_cursor)
        stream.commit(second_cursor)
        with pytest.raises(SMLDataError, match="regress"):
            stream.commit(first_cursor)
        with pytest.raises(SMLDataError, match="not delivered"):
            stream.commit(SwagCursor(1, 0, 0))
        final = next(stream)
        final_cursor = final.cursor_after
        final.release()
        stream.commit(final_cursor)
        assert stream.committed_cursor == SwagCursor(1, 0, 0)


def test_swag_stream_rejects_an_already_closed_bundle(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    bundle.close()

    with pytest.raises(SMLDataError, match="closed"):
        SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())


def test_swag_stream_pull_preserves_producer_failure_when_close_fails(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    producer_error = SMLDataError("injected producer failure")
    cleanup_error = RuntimeError("injected stream cleanup failure")
    original_close = stream.close
    stream._stop.set()
    producer = stream._producer
    assert producer is not None
    producer.join()
    while not stream._queue.empty():
        item = stream._queue.get_nowait()
        if isinstance(item, swag.SwagBatchEnvelope):
            item.release()
    stream._queue.put(swag._ProducerFailure(producer_error))

    def fail_after_close():
        original_close()
        raise cleanup_error

    monkeypatch.setattr(stream, "close", fail_after_close)
    try:
        with pytest.raises(SMLDataError) as raised:
            stream._pull()

        assert raised.value is producer_error
        assert raised.value.__cause__ is cleanup_error
    finally:
        original_close()


def test_swag_stream_exit_preserves_body_error_when_close_fails(tmp_path, monkeypatch):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    body_error = ValueError("injected stream body failure")
    cleanup_error = RuntimeError("injected stream cleanup failure")
    original_close = stream.close

    def fail_after_close():
        original_close()
        raise cleanup_error

    monkeypatch.setattr(stream, "close", fail_after_close)
    try:
        with pytest.raises(ValueError) as raised, stream:
            raise body_error

        assert raised.value is body_error
        assert raised.value.__cause__ is cleanup_error
    finally:
        original_close()
