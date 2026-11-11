from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    VerificationLevel,
    file_identity,
)
from sml.errors import SMLArtifactError, SMLDataError

from .swag_helpers import (
    VALID_ROW,
    FakeSwagProvider,
    RecordingProcessor,
    replace_row,
    tiny_base_model,
    tiny_swag_config,
)


class FixedProcessor(RecordingProcessor):
    def __init__(self, encoded: Mapping[str, tuple[int, ...]]) -> None:
        super().__init__()
        self._encoded = dict(encoded)

    def _encode_one(self, text: str) -> list[int]:
        try:
            return list(self._encoded[text])
        except KeyError as error:
            raise AssertionError(f"unexpected encode input: {text!r}") from error


def test_bucket_growth_failure_preserves_existing_rows_and_can_retry(
    tmp_path, monkeypatch
):
    from sml.data import swag

    writer = swag._BucketWriter(16, tmp_path)
    ids = np.arange(64, dtype="<i4").reshape(4, 16)
    mask = np.ones((4, 16), dtype=np.bool_)
    for index in range(8):
        writer.append(ids + index, mask, mask, index % 4)
    writer.flush()
    old_maps = dict(writer._maps)
    old_values = {name: np.array(mm) for name, mm in old_maps.items()}
    real_memmap = np.memmap
    allocated = []
    failure = OSError("disk full during bucket growth")

    def fail_second_allocation(path, *args, **kwargs):
        if len(allocated) == 1:
            Path(path).write_bytes(b"partial allocation")
            raise failure
        mm = real_memmap(path, *args, **kwargs)
        allocated.append(mm)
        return mm

    writer.append(ids + 8, mask, mask, 0)
    monkeypatch.setattr(swag.np, "memmap", fail_second_allocation)
    try:
        with pytest.raises(OSError) as raised:
            writer.flush()
        assert raised.value is failure
        assert writer.count == writer._capacity == 8
        assert all(writer._maps[name] is mm for name, mm in old_maps.items())
        assert all(not mm._mmap.closed for mm in old_maps.values())
        assert all(mm._mmap.closed for mm in allocated)
        assert not list(tmp_path.glob("*-16.dat"))
        for name, values in old_values.items():
            np.testing.assert_array_equal(writer._maps[name], values)

        monkeypatch.setattr(swag.np, "memmap", real_memmap)
        arrays = writer.arrays()
        assert writer.count == 9
        np.testing.assert_array_equal(arrays["input_ids"][-1], ids + 8)
        assert all(mm._mmap.closed for mm in old_maps.values())
    finally:
        writer.close()


def test_bucket_close_releases_every_mapping_when_flush_fails(tmp_path, monkeypatch):
    from sml.data import swag

    writer = swag._BucketWriter(16, tmp_path)
    writer._ensure_capacity(1)
    maps = list(writer._maps.values())
    real_flush = np.memmap.flush
    failure = OSError("flush failed")

    def fail_first_flush(array):
        if array is maps[0]:
            raise failure
        return real_flush(array)

    monkeypatch.setattr(np.memmap, "flush", fail_first_flush)
    with pytest.raises(OSError) as raised:
        writer.close()
    assert raised.value is failure
    assert all(mm._mmap.closed for mm in maps)
    writer.close()


def test_ingest_preserves_provider_failure_and_closes_all_buckets(
    tmp_path, monkeypatch
):
    from sml.data import swag

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 1)
    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_iter_after_rows = True
    real_close = swag._BucketWriter.close
    closed = []
    cleanup_error = OSError("bucket cleanup failed")

    def failing_close(writer):
        real_close(writer)
        closed.append(writer.length)
        if len(closed) == 1:
            raise cleanup_error

    monkeypatch.setattr(swag._BucketWriter, "close", failing_close)
    with pytest.raises(SMLDataError, match="provider unavailable") as raised:
        swag.prepare_swag_bundle(
            tiny_swag_config(provider), tiny_base_model(), tmp_path / "swag"
        )
    assert raised.value.__cause__ is cleanup_error
    assert closed == [16, 32]


def assert_eos_positions_are_scored(bucket, eos_token_id: int) -> None:
    eos = bucket.input_ids == eos_token_id
    assert eos.any()
    assert bool(np.all(bucket.score_mask[eos]))
    assert bool(np.all(bucket.valid_token_mask[eos]))


def test_context_and_endings_are_encoded_separately_and_eos_is_scored(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    recording_tokenizer = base.tokenizer
    bundle = prepare_swag_bundle(
        tiny_swag_config(provider),
        base,
        tmp_path / "swag",
    )
    try:
        assert recording_tokenizer.calls[0] == VALID_ROW["context"]
        assert tuple(recording_tokenizer.calls[1:5]) == tuple(VALID_ROW["endings"])
        with bundle.borrow_buckets() as buckets:
            bucket = buckets[0]
            assert not bucket.score_mask[:, :, 0].any()
            assert bucket.score_mask[bucket.valid_token_mask].any()
            assert (~bucket.score_mask & bucket.valid_token_mask).any()
            assert_eos_positions_are_scored(bucket, eos_token_id=2)
            assert bucket.input_ids.dtype == np.dtype("<i4")
            assert bucket.labels.dtype == np.dtype("<i4")
            assert bucket.valid_token_mask.dtype == np.dtype("bool")
            assert bucket.score_mask.dtype == np.dtype("bool")
            assert bucket.input_ids.shape == bucket.valid_token_mask.shape
            assert bucket.input_ids.shape[1:] == (4, bucket.input_ids.shape[-1])
    finally:
        bundle.close()


def test_chunk_with_multiple_rows_makes_one_tokenizer_encode_call(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    processor = RecordingProcessor()
    rows = (
        VALID_ROW,
        replace_row(label=2),
    )
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(processor=processor),
        tmp_path / "swag",
    )
    bundle.close()
    assert len(processor.encode_calls) == 1
    batched = processor.encode_calls[0]
    assert batched == [
        VALID_ROW["context"],
        *VALID_ROW["endings"],
        rows[1]["context"],
        *rows[1]["endings"],
    ]
    assert processor.calls[0] == VALID_ROW["context"]
    assert tuple(processor.calls[1:5]) == tuple(VALID_ROW["endings"])


@pytest.mark.parametrize("maximum_examples", [1, 3])
@pytest.mark.parametrize("tail", ["malformed_row", "provider_failure"])
def test_maximum_examples_stops_before_reading_unused_rows(
    tmp_path, monkeypatch, maximum_examples, tail
):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    rows = tuple(replace_row(label=index) for index in range(maximum_examples))
    provider = FakeSwagProvider(rows + (({},) if tail == "malformed_row" else ()))
    provider.fail_iter_after_rows = tail == "provider_failure"
    bundle = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=maximum_examples),
        tiny_base_model(),
        tmp_path / "swag",
    )
    try:
        assert bundle.manifest.example_count == maximum_examples
        assert bundle.manifest.dropped_overlength_rows == 0
        with bundle.borrow_buckets() as buckets:
            labels = [int(label) for bucket in buckets for label in bucket.labels]
        assert labels == list(range(maximum_examples))
    finally:
        bundle.close()


def test_maximum_examples_counts_kept_rows_after_overlength_drops(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    overlength = replace_row(context=" ".join(["word"] * 40))
    rows = (
        overlength,
        replace_row(label=0),
        overlength,
        replace_row(label=1),
        replace_row(label=2),
        {},
    )
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows), maximum_examples=3),
        tiny_base_model(),
        tmp_path / "swag",
    )
    try:
        assert bundle.manifest.example_count == 3
        assert bundle.manifest.dropped_overlength_rows == 2
        with bundle.borrow_buckets() as buckets:
            labels = [int(label) for bucket in buckets for label in bucket.labels]
        assert labels == [0, 1, 2]
    finally:
        bundle.close()


def test_npy_identity_is_hashed_while_writing(tmp_path, monkeypatch):
    from sml.data import swag

    opened: list[str] = []
    original_open = Path.open

    def tracking_open(self, mode="r", *args, **kwargs):
        if self.suffix == ".npy":
            opened.append(mode)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    path = tmp_path / "array.npy"
    reference = swag._write_npy(path, np.arange(8, dtype=np.int32), "array.npy")
    assert opened == ["xb"]
    with path.open("rb") as payload:
        assert file_identity(payload) == reference.payload.identity


def test_overlength_candidate_drops_complete_row(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    rows = (
        {
            "context": "ok",
            "endings": ("a", "b", "c", "d"),
            "label": 2,
        },
        {
            "context": "ok",
            "endings": ("a", "b", "this row is far too long for the bucket", "d"),
            "label": 0,
        },
    )
    processor = FixedProcessor(
        {
            "ok": (10,),
            "a": (11,),
            "b": (12,),
            "c": (13,),
            "d": (14,),
            "this row is far too long for the bucket": tuple(range(10, 50)),
        }
    )
    provider = FakeSwagProvider(rows)
    bundle = prepare_swag_bundle(
        tiny_swag_config(provider),
        tiny_base_model(processor=processor),
        tmp_path / "swag",
    )
    try:
        assert bundle.manifest.dropped_overlength_rows == 1
        assert bundle.manifest.example_count == 1
        with bundle.borrow_buckets() as buckets:
            assert all(
                0 <= int(label) < 4 for bucket in buckets for label in bucket.labels
            )
    finally:
        bundle.close()


def test_zero_usable_examples_are_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    processor = FixedProcessor(
        {
            "ctx": tuple(range(10, 50)),
            "a": (11,),
            "b": (12,),
            "c": (13,),
            "d": (14,),
        }
    )
    provider = FakeSwagProvider(
        ({"context": "ctx", "endings": ("a", "b", "c", "d"), "label": 0},)
    )
    with pytest.raises(SMLDataError, match="usable"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(processor=processor),
            tmp_path / "swag",
        )


@pytest.mark.parametrize(
    "overrides,message",
    [
        pytest.param(
            {"endings": ("on the mat", "in the car")}, "four", id="candidate-count"
        ),
        pytest.param({"label": 4}, "label", id="label-range"),
        pytest.param(
            {"endings": ("", "in the car", "by the door", "near a tree")},
            "scored",
            id="empty-continuation",
        ),
    ],
)
def test_invalid_rows_are_rejected_without_provider_error_wrapping(
    tmp_path, overrides, message
):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((replace_row(**overrides),))
    with pytest.raises(SMLDataError, match=message) as caught:
        prepare_swag_bundle(
            tiny_swag_config(provider), tiny_base_model(), tmp_path / "swag"
        )
    assert "unavailable" not in str(caught.value)
    assert "cache key" not in str(caught.value)
    assert "sha256:" not in str(caught.value)


def test_out_of_range_token_id_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    processor = FixedProcessor(
        {
            "the cat sat": (10, 11, 12),
            "on the mat": (90,),
            "in the car": (13, 14, 15),
            "by the door": (16, 17, 18),
            "near a tree": (19, 20, 21),
        }
    )
    provider = FakeSwagProvider((VALID_ROW,))
    with pytest.raises(SMLDataError, match="token"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(processor=processor),
            tmp_path / "swag",
        )


def test_maximum_length_beyond_effective_context_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    with pytest.raises(SMLDataError, match="effective context"):
        prepare_swag_bundle(
            tiny_swag_config(provider, maximum_length=512),
            tiny_base_model(),
            tmp_path / "swag",
        )


def test_default_preparation_keeps_candidates_beyond_pretraining_context(tmp_path):
    from sml.data.swag import (
        SwagPreparationConfig,
        SwagSourceConfig,
        prepare_swag_bundle,
    )

    provider = FakeSwagProvider(
        (VALID_ROW, replace_row(context=" ".join(["word"] * 1_100)))
    )
    config = SwagPreparationConfig(provider, SwagSourceConfig(revision="main"))
    base = tiny_base_model(model_overrides={"original_context_length": 1_024})

    with prepare_swag_bundle(config, base, tmp_path / "extended") as bundle:
        assert bundle.manifest.example_count == 2
        assert bundle.manifest.preprocessing["maximum_length"] == 8_192
        with bundle.borrow_buckets() as buckets:
            assert tuple(bucket.length for bucket in buckets) == (64, 2_048)
    assert base.model_config.effective_context_length == 1_024


def test_preparation_honors_explicit_unscaled_context(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    config = tiny_swag_config(
        provider, maximum_length=64, bucket_boundaries=(64,), rope_scaling_factor=1.0
    )
    with pytest.raises(SMLDataError, match="effective context"):
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "unscaled")
    assert provider.resolve_calls == 0


def test_prepare_rejects_non_full_base_before_provider_access(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_resolve = True
    base = replace(tiny_base_model(), verification=VerificationLevel.MANIFEST_TRUSTED)
    with pytest.raises(SMLArtifactError, match="FULL"):
        prepare_swag_bundle(tiny_swag_config(provider), base, tmp_path / "swag")
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0


def test_prepare_rejects_tokenizer_model_mismatch_before_provider_access(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_resolve = True
    base = tiny_base_model(model_overrides={"vocab_size": 32})
    with pytest.raises(SMLDataError, match="vocab|tokenizer|special"):
        prepare_swag_bundle(tiny_swag_config(provider), base, tmp_path / "swag")
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0


def test_chunked_ingest_does_not_stack_all_examples_at_once(tmp_path, monkeypatch):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    stacked_example_batches: list[int] = []
    original_stack = np.stack

    def tracking_stack(arrays, axis=0, **kwargs):
        sequence = tuple(arrays)
        first = sequence[0] if sequence else None
        if getattr(first, "ndim", 0) >= 2:
            stacked_example_batches.append(len(sequence))
        return original_stack(sequence, axis=axis, **kwargs)

    monkeypatch.setattr(swag.np, "stack", tracking_stack)
    rows = tuple(replace_row(label=index % 4) for index in range(5))
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(),
        tmp_path / "swag",
    )
    try:
        with bundle.borrow_buckets() as buckets:
            labels = [
                int(label) for bucket in buckets for label in np.asarray(bucket.labels)
            ]
        assert bundle.manifest.example_count == 5
        assert labels == [index % 4 for index in range(5)]
        assert all(count <= 2 for count in stacked_example_batches)
    finally:
        bundle.close()
