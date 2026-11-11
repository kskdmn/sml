from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from sml.data import swag as swag_data
from sml.data.swag import (
    SwagBatch,
    SwagBatchEnvelope,
    SwagBatchStream,
    SwagCursor,
)


def test_swag_envelope_validates_inputs_and_owns_readonly_storage():
    batch_size, candidate_count, length = 2, 4, 8
    input_ids = np.arange(
        batch_size * candidate_count * length, dtype=np.int32
    ).reshape(batch_size, candidate_count, length)
    score_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    labels = np.arange(batch_size, dtype=np.int32)
    example_mask = np.array([True, False])
    valid_token_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=1)
    envelope = SwagBatchEnvelope(
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
        cursor,
        source_epoch=0,
    )

    for array in (
        envelope.input_ids,
        envelope.score_mask,
        envelope.labels,
        envelope.example_mask,
        envelope.valid_token_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[(0,) * array.ndim] = 1
    with pytest.raises(TypeError, match="input_ids"):
        SwagBatchEnvelope(
            input_ids.tolist(),
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
            cursor,
            source_epoch=0,
        )
    with pytest.raises(ValueError, match="score_mask"):
        SwagBatchEnvelope(
            input_ids,
            score_mask.astype(np.int32),
            labels,
            example_mask,
            valid_token_mask,
            cursor,
            source_epoch=0,
        )

    original_input = int(envelope.input_ids[0, 0, 0])
    original_label = int(envelope.labels[0])
    original_example = bool(envelope.example_mask[0])
    original_score = bool(envelope.score_mask[0, 0, 0])
    original_valid = bool(envelope.valid_token_mask[0, 0, 0])
    input_ids[0, 0, 0] = original_input + 7
    labels[0] = original_label + 3
    example_mask[0] = not original_example
    score_mask[0, 0, 0] = not original_score
    valid_token_mask[0, 0, 0] = not original_valid
    assert int(envelope.input_ids[0, 0, 0]) == original_input
    assert int(envelope.labels[0]) == original_label
    assert bool(envelope.example_mask[0]) is original_example
    assert bool(envelope.score_mask[0, 0, 0]) is original_score
    assert bool(envelope.valid_token_mask[0, 0, 0]) is original_valid


def test_swag_producer_transfers_owned_envelope_arrays_without_recopying(
    tiny_swag_runtime, monkeypatch
):
    captured: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    original_assemble = swag_data._assemble_batch_arrays

    def capture_assemble(*args, **kwargs):
        arrays = original_assemble(*args, **kwargs)
        captured.append(arrays)
        return arrays

    monkeypatch.setattr(swag_data, "_assemble_batch_arrays", capture_assemble)
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=2,
        prefetch_depth=1,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        envelope = next(iter(stream))
    assert captured
    assembled = captured[0]
    pairs = (
        (assembled[0], envelope.input_ids),
        (assembled[1], envelope.valid_token_mask),
        (assembled[2], envelope.score_mask),
        (assembled[3], envelope.labels),
        (assembled[4], envelope.example_mask),
    )
    for owned, stored in pairs:
        assert stored.ctypes.data == owned.ctypes.data
        assert np.shares_memory(stored, owned)
        assert not stored.flags.writeable
        assert not owned.flags.writeable
        assert owned.flags.owndata


def test_assemble_batch_arrays_fills_staging_without_extra_copies(monkeypatch):
    length = 4
    bucket = swag_data.SwagBucket(
        length=length,
        input_ids=np.arange(2 * 4 * length, dtype=np.int32).reshape(2, 4, length),
        valid_token_mask=np.ones((2, 4, length), dtype=bool),
        score_mask=np.ones((2, 4, length), dtype=bool),
        labels=np.array([1, 3], dtype=np.int32),
    )
    manifest = SimpleNamespace(pad_token_id=3, bos_token_id=1, eos_token_id=2)
    staging: list[np.ndarray] = []
    original_empty = swag_data.np.empty
    original_zeros = swag_data.np.zeros
    original_array = swag_data.np.array
    original_owned = swag_data._owned_readonly
    copy_true_calls: list[object] = []
    owned_calls: list[int] = []

    def tracking_empty(*args, **kwargs):
        array = original_empty(*args, **kwargs)
        staging.append(array)
        return array

    def tracking_zeros(*args, **kwargs):
        array = original_zeros(*args, **kwargs)
        staging.append(array)
        return array

    def tracking_array(*args, **kwargs):
        if kwargs.get("copy") is True:
            copy_true_calls.append(args[0] if args else None)
        return original_array(*args, **kwargs)

    def tracking_owned(array):
        owned_calls.append(id(array))
        return original_owned(array)

    monkeypatch.setattr(swag_data.np, "empty", tracking_empty)
    monkeypatch.setattr(swag_data.np, "zeros", tracking_zeros)
    monkeypatch.setattr(swag_data.np, "array", tracking_array)
    monkeypatch.setattr(swag_data, "_owned_readonly", tracking_owned)

    assembled = swag_data._assemble_batch_arrays(
        bucket,
        (0,),
        batch_size=2,
        manifest=manifest,
    )
    assert copy_true_calls == []
    assert owned_calls == []
    for array in assembled:
        assert any(array.ctypes.data == staged.ctypes.data for staged in staging)
        assert not array.flags.writeable
        assert array.flags.owndata
    np.testing.assert_array_equal(assembled[0][0], bucket.input_ids[0])
    assert int(assembled[3][0]) == 1
    assert bool(assembled[4][0]) is True
    assert bool(assembled[4][1]) is False

    input_ids, valid_token_mask, score_mask, labels, example_mask = assembled
    cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=1)
    envelope = SwagBatchEnvelope._owned(
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
        cursor,
        source_epoch=0,
    )
    pairs = (
        (input_ids, envelope.input_ids),
        (valid_token_mask, envelope.valid_token_mask),
        (score_mask, envelope.score_mask),
        (labels, envelope.labels),
        (example_mask, envelope.example_mask),
    )
    for owned, stored in pairs:
        assert stored.ctypes.data == owned.ctypes.data
        assert not stored.flags.writeable

    caller_ids = np.arange(2 * 4 * length, dtype=np.int32).reshape(2, 4, length)
    caller_score = np.ones((2, 4, length), dtype=bool)
    caller_labels = np.arange(2, dtype=np.int32)
    caller_example = np.array([True, False])
    caller_valid = np.ones((2, 4, length), dtype=bool)
    public = SwagBatchEnvelope(
        caller_ids,
        caller_score,
        caller_labels,
        caller_example,
        caller_valid,
        cursor,
        source_epoch=0,
    )
    original_input = int(public.input_ids[0, 0, 0])
    caller_ids[0, 0, 0] = original_input + 7
    assert int(public.input_ids[0, 0, 0]) == original_input
    assert public.input_ids.ctypes.data != caller_ids.ctypes.data


def test_full_prefetch_queue_owns_distinct_readonly_storage_and_cannot_advance_committed_cursor(
    tiny_swag_runtime,
):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=1,
        prefetch_depth=3,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        iterator = iter(stream)
        envelopes: list[SwagBatchEnvelope] = [next(iterator)]
        while len(envelopes) < loader.prefetch_depth:
            envelopes.append(next(iterator))
        assert stream.committed_cursor == start
        pointers = []
        for envelope in envelopes:
            for array in (
                envelope.input_ids,
                envelope.score_mask,
                envelope.labels,
                envelope.example_mask,
                envelope.valid_token_mask,
            ):
                assert not array.flags.writeable
                pointers.append(array.__array_interface__["data"][0])
        assert len(set(pointers)) == len(pointers)
        for envelope in envelopes:
            envelope.release()


def test_bucket_tail_compile_shape_is_fixed(tiny_swag_runtime):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=4,
        prefetch_depth=1,
    )
    shapes = []
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        for envelope in stream:
            batch = SwagBatch.from_envelope(envelope)
            shapes.append(tuple(int(dimension) for dimension in batch.input_ids.shape))
            stream.commit(batch.cursor_after)
    assert shapes
    assert all(shape[0] == 4 for shape in shapes)
    assert len({shape[-1] for shape in shapes}) == 1


def test_stream_builds_one_permutation_per_epoch(tiny_swag_runtime, monkeypatch):
    loader = replace(tiny_swag_runtime.config.loader, microbatch_size=1)
    original_plan = swag_data._epoch_bucket_plan
    buckets = tiny_swag_runtime.bundle._owned_buckets()
    expected = [
        int(buckets[bucket_index].labels[index])
        for bucket_index, indices in original_plan(
            buckets, epoch_seed=loader.epoch_seed, epoch=0
        )
        for index in indices
    ]
    planned_epochs = []

    def record_plan(*args, **kwargs):
        planned_epochs.append(kwargs["epoch"])
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(swag_data, "_epoch_bucket_plan", record_plan)
    labels = []
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=SwagCursor.initial()
    ) as stream:
        for envelope in stream:
            with envelope:
                labels.extend(envelope.labels[envelope.example_mask].tolist())
                stream.commit(envelope.cursor_after)

    assert labels == expected
    assert planned_epochs.count(0) == 1
    assert len(planned_epochs) == len(set(planned_epochs))


def test_swag_batch_stream_accepts_structural_loader(tiny_swag_runtime):
    loader = SimpleNamespace(prefetch_depth=1, microbatch_size=2, epoch_seed=42)
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        envelope = next(iter(stream))
        assert envelope.input_ids.shape[0] == 2


def test_swag_batch_stream_rejects_loader_missing_required_attributes(
    tiny_swag_runtime,
):
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with pytest.raises(TypeError, match="prefetch_depth"):
        SwagBatchStream._borrowing_bundle(
            tiny_swag_runtime.bundle, object(), cursor=start
        )


def test_swag_batch_stream_rejects_invalid_loader_policy_types_and_ranges(
    tiny_swag_runtime,
):
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    base = {"prefetch_depth": 1, "microbatch_size": 2, "epoch_seed": 42}
    cases = (
        ({"prefetch_depth": 0}, ValueError, "prefetch_depth"),
        ({"prefetch_depth": 1.5}, TypeError, "prefetch_depth"),
        ({"microbatch_size": 0}, ValueError, "microbatch_size"),
        ({"microbatch_size": True}, TypeError, "microbatch_size"),
        ({"epoch_seed": -1}, ValueError, "epoch_seed"),
        ({"epoch_seed": 2**32}, ValueError, "epoch_seed"),
        ({"epoch_seed": True}, TypeError, "epoch_seed"),
    )
    for override, error, match in cases:
        loader = SimpleNamespace(**{**base, **override})
        with pytest.raises(error, match=match):
            SwagBatchStream._borrowing_bundle(
                tiny_swag_runtime.bundle, loader, cursor=start
            )
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle,
        SimpleNamespace(prefetch_depth=1, microbatch_size=2, epoch_seed=0),
        cursor=start,
    ) as stream:
        envelope = next(iter(stream))
        assert envelope.input_ids.shape[0] == 2
