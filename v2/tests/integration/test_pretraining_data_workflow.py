from __future__ import annotations

import io
import json
import queue
import shutil
import threading
import time
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import sml.data.pretraining as pretraining_module
import zstandard as zstd
from sml.artifacts.manifest import (
    OpenedArtifact,
    PayloadRef,
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
    row_content_identity,
)
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    BatchEnvelope,
    PreparedDataBundle,
    PretrainingBatchStream,
    PretrainingCursor,
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLDataError


def _write_corpus(root: Path, texts: list[str]) -> Path:
    root.mkdir()
    midpoint = max(1, len(texts) // 2)
    partitions = (texts[:midpoint], texts[midpoint:])
    for index, partition in enumerate(partitions):
        if not partition:
            continue
        payload = b"".join(
            json.dumps({"text": text}).encode("utf-8") + b"\n" for text in partition
        )
        (root / f"tiny-00{index:02d}.jsonl.zst").write_bytes(
            zstd.ZstdCompressor().compress(payload)
        )
    return root


@pytest.fixture(scope="module")
def prepared_sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretraining-sources")
    tokenizer_corpus = _write_corpus(
        root / "tokenizer-corpus",
        [f"alpha beta gamma delta epsilon {index} " * 12 for index in range(40)],
    )
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=CorpusConfig(
                input_root=tokenizer_corpus,
                min_text_bytes=1,
                max_rows_per_file=None,
            ),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data_corpus = _write_corpus(
        root / "data-corpus",
        ["alpha beta gamma delta epsilon " * (4 + index) for index in range(12)],
    )
    return tokenizer, data_corpus


def _config(prepared_sources, **overrides):
    tokenizer, corpus = prepared_sources
    values = {
        "corpus": CorpusConfig(
            input_root=corpus,
            shuffle_files=False,
            min_text_bytes=1,
            max_rows_per_file=None,
        ),
        "tokenizer_bundle": tokenizer.path,
        "sequence_length": 8,
        "shuffle_window_rows": 5,
        "output_shard_rows": 3,
        "seed": 17,
    }
    values.update(overrides)
    return PretrainingPreparationConfig(**values)


def _load_rows(bundle_path: Path) -> np.ndarray:
    manifest = read_manifest(
        bundle_path, PretrainingDataManifest, VerificationLevel.FULL
    ).manifest
    arrays = [
        np.load(bundle_path / shard.logical_path, allow_pickle=False)
        for shard in manifest.shards
    ]
    return np.concatenate(arrays, axis=0)


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(
        logical_path=logical_path,
        identity=identity,
        byte_size=path.stat().st_size,
    )


def _write_manifest(path: Path, manifest: PretrainingDataManifest) -> None:
    path.joinpath("manifest.json").write_bytes(canonical_json_bytes(manifest))


def _replace_shards(
    bundle: PreparedDataBundle,
    arrays: tuple[np.ndarray, ...],
    *,
    declared_counts: tuple[int, ...] | None = None,
) -> PreparedDataBundle:
    shard_directory = bundle.path / "shards"
    for path in shard_directory.iterdir():
        path.unlink()

    paths = []
    for index, array in enumerate(arrays):
        path = shard_directory / f"train-{index:06d}.npy"
        with path.open("wb") as payload:
            np.save(payload, array, allow_pickle=False)
        paths.append(path)

    counts = declared_counts or tuple(array.shape[0] for array in arrays)
    references = tuple(_payload_ref(path, f"shards/{path.name}") for path in paths)
    rows = (row for array in arrays for row in array)
    manifest = replace(
        bundle.manifest,
        shard_row_counts=counts,
        shards=references,
        row_content_identity=row_content_identity(
            rows,
            sum(array.shape[0] for array in arrays),
            bundle.manifest.row_width,
        ),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    _write_manifest(bundle.path, manifest)
    return replace(bundle, manifest=manifest)


def _replace_manifest(
    bundle: PreparedDataBundle, **changes: object
) -> PreparedDataBundle:
    manifest = replace(bundle.manifest, **changes)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    _write_manifest(bundle.path, manifest)
    return replace(bundle, manifest=manifest)


def _replace_shard_payload(
    bundle: PreparedDataBundle, index: int, array: np.ndarray
) -> PreparedDataBundle:
    reference = bundle.manifest.shards[index]
    path = bundle.path / reference.logical_path
    with path.open("wb") as payload:
        np.save(payload, array, allow_pickle=False)
    references = list(bundle.manifest.shards)
    references[index] = _payload_ref(path, reference.logical_path)
    return _replace_manifest(bundle, shards=tuple(references))


def _replace_shard_bytes(
    bundle: PreparedDataBundle, index: int, payload_bytes: bytes
) -> PreparedDataBundle:
    reference = bundle.manifest.shards[index]
    path = bundle.path / reference.logical_path
    path.write_bytes(payload_bytes)
    references = list(bundle.manifest.shards)
    references[index] = _payload_ref(path, reference.logical_path)
    return _replace_manifest(bundle, shards=tuple(references))


def _rows(row_width: int, *identifiers: int) -> np.ndarray:
    return np.stack(
        [np.full(row_width, identifier, dtype="<i4") for identifier in identifiers]
    )


class _CountingMX:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.transfer_count = 0

    def array(self, rows: np.ndarray):
        self.transfer_count += 1
        return self._delegate.array(rows)

    def eval(self, *arrays) -> None:
        self._delegate.eval(*arrays)


@pytest.fixture
def prepared_bundle(prepared_sources, tmp_path) -> PreparedDataBundle:
    bundle = prepare_pretraining_bundle(
        _config(prepared_sources), tmp_path / "stream-bundle"
    )
    width = bundle.manifest.row_width
    return _replace_shards(
        bundle,
        (
            _rows(width, 0, 1),
            _rows(width, 10, 11),
            _rows(width, 20, 21),
        ),
    )


def test_benchmark_adapter_runs_real_prepared_data_workflow(monkeypatch):
    source_root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(source_root))
    from v2.benchmarks.adapters.replacement import (
        ReplacementNativeWorkload,
        resolve_native_workload,
        run_measured,
        run_warmup,
    )
    from v2.benchmarks.workload import build_canonical_workload

    workload = build_canonical_workload(
        model_overrides={"vocab_size": 32},
        loader_overrides={"sequence_length": 8},
        row_count=32,
    )
    native = resolve_native_workload("prepared-data", workload, source_root)

    assert isinstance(native, ReplacementNativeWorkload)
    recorder = _CountingMX(native.runtime._mx)
    native.runtime._mx = recorder
    try:
        assert run_warmup("prepared-data", native, 1) is None
        assert recorder.transfer_count == 1
        recorder.transfer_count = 0
        assert run_measured("prepared-data", native, 3) == 3.0
        assert recorder.transfer_count == 3
    finally:
        native.runtime.close()


def _take(stream: PretrainingBatchStream, count: int) -> list[BatchEnvelope]:
    iterator = iter(stream)
    return [next(iterator) for _ in range(count)]


def _assert_constructor_fails_before_thread_start(
    monkeypatch,
    error: type[Exception],
    message: str,
    construct,
) -> None:
    starts = 0
    original_start = threading.Thread.start

    def record_start(thread):
        nonlocal starts
        starts += 1
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", record_start)
    with pytest.raises(error, match=message):
        construct()
    assert starts == 0


def test_prepared_bundle_is_closed_self_describing_and_mmap_ready(
    prepared_sources, tmp_path
):
    output = tmp_path / "prepared"
    bundle = prepare_pretraining_bundle(_config(prepared_sources), output)

    assert bundle.path == output
    assert bundle.verification is VerificationLevel.FULL
    assert bundle.manifest.identity == bundle.manifest.recompute_identity()
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "shards",
        "tokenizer",
    }
    assert {path.name for path in (output / "tokenizer").iterdir()} == {
        "manifest.json",
        "tokenizer.model",
        "tokenizer.vocab",
    }
    nested = read_manifest(
        output / "tokenizer", TokenizerManifest, VerificationLevel.FULL
    ).manifest
    assert nested.identity == bundle.manifest.tokenizer_identity
    source_tokenizer, _corpus = prepared_sources
    for name in ("manifest.json", "tokenizer.model", "tokenizer.vocab"):
        assert (output / "tokenizer" / name).read_bytes() == (
            source_tokenizer.path / name
        ).read_bytes()
    assert [shard.logical_path for shard in bundle.manifest.shards] == [
        f"shards/train-{index:06d}.npy" for index in range(len(bundle.manifest.shards))
    ]
    assert set(bundle.manifest.row_order_policy) == {
        "algorithm",
        "shuffle_window_rows",
        "output_shard_rows",
    }
    assert set(bundle.manifest.source_summary) == {
        "corpus",
        "ordered_files",
        "physical_lines_read",
        "object_rows_read",
        "texts_used",
    }
    assert "input_root" not in bundle.manifest.source_summary["corpus"]
    assert bundle.manifest.source_summary["ordered_files"] == (
        "tiny-0000.jsonl.zst",
        "tiny-0001.jsonl.zst",
    )
    assert bundle.manifest.source_summary["physical_lines_read"] == 12
    assert bundle.manifest.source_summary["object_rows_read"] == 12
    assert bundle.manifest.source_summary["texts_used"] == 12
    assert bundle.manifest.diagnostic_source_locator == str(
        _config(prepared_sources).corpus.input_root
    )

    for count, shard in zip(
        bundle.manifest.shard_row_counts, bundle.manifest.shards, strict=True
    ):
        array = np.load(output / shard.logical_path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (count, bundle.manifest.sequence_length + 1)
        assert array.dtype == np.dtype("<i4")
        assert array.flags.c_contiguous
        assert 0 < count <= _config(prepared_sources).output_shard_rows


def test_resharding_preserves_rows_and_row_identity_but_changes_bundle_identity(
    prepared_sources, tmp_path
):
    first = prepare_pretraining_bundle(
        _config(prepared_sources, output_shard_rows=2), tmp_path / "two"
    )
    second = prepare_pretraining_bundle(
        _config(prepared_sources, output_shard_rows=7), tmp_path / "seven"
    )

    assert np.array_equal(_load_rows(first.path), _load_rows(second.path))
    assert first.manifest.row_content_identity == second.manifest.row_content_identity
    assert first.manifest.identity != second.manifest.identity


def test_seed_and_window_are_semantic_row_order_inputs(prepared_sources, tmp_path):
    original = prepare_pretraining_bundle(
        _config(prepared_sources), tmp_path / "original"
    )
    changed_seed = prepare_pretraining_bundle(
        _config(prepared_sources, seed=19), tmp_path / "seed"
    )
    changed_window = prepare_pretraining_bundle(
        _config(prepared_sources, shuffle_window_rows=7), tmp_path / "window"
    )

    assert (
        original.manifest.row_content_identity
        != changed_seed.manifest.row_content_identity
    )
    assert (
        original.manifest.row_content_identity
        != changed_window.manifest.row_content_identity
    )


def test_identical_retry_is_accepted_and_changed_config_collides(
    prepared_sources, tmp_path
):
    output = tmp_path / "prepared"
    first = prepare_pretraining_bundle(_config(prepared_sources), output)
    second = prepare_pretraining_bundle(_config(prepared_sources), output)
    assert first.manifest.identity == second.manifest.identity

    with pytest.raises(SMLArtifactError, match="collision|different identity"):
        prepare_pretraining_bundle(_config(prepared_sources, seed=18), output)


def test_retry_rejects_corrupt_existing_shard(prepared_sources, tmp_path):
    output = tmp_path / "prepared"
    bundle = prepare_pretraining_bundle(_config(prepared_sources), output)
    shard = output / bundle.manifest.shards[0].logical_path
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(
        SMLArtifactError, match="existing target failed full verification"
    ):
        prepare_pretraining_bundle(_config(prepared_sources), output)


def test_preparation_refuses_zero_complete_rows_without_visible_output(
    prepared_sources, tmp_path
):
    tokenizer, _corpus = prepared_sources
    empty = _write_corpus(tmp_path / "empty", ["x"])
    config = _config(
        prepared_sources,
        corpus=CorpusConfig(
            input_root=empty,
            shuffle_files=False,
            min_text_bytes=100,
            max_rows_per_file=None,
        ),
        tokenizer_bundle=tokenizer.path,
    )
    output = tmp_path / "prepared"

    with pytest.raises(SMLDataError, match="no complete pretraining rows"):
        prepare_pretraining_bundle(config, output)

    assert not output.exists()


@pytest.mark.parametrize("invalid_id", [-1, 10_000])
def test_preparation_rejects_invalid_processor_token_ids_before_publication(
    prepared_sources, tmp_path, monkeypatch, invalid_id
):
    from sml.data import pretraining

    loaded = pretraining.load_tokenizer_bundle(
        _config(prepared_sources).tokenizer_bundle,
        VerificationLevel.FULL,
    )

    class InvalidProcessor:
        def encode(self, _text):
            return [4, invalid_id, 5]

    monkeypatch.setattr(
        pretraining,
        "load_tokenizer_bundle",
        lambda _path, _verification: replace(loaded, processor=InvalidProcessor()),
    )
    output = tmp_path / f"invalid-{invalid_id}"

    with pytest.raises((SMLArtifactError, SMLDataError), match="token ID"):
        prepare_pretraining_bundle(_config(prepared_sources), output)

    assert not output.exists()


def test_epoch_stream_crosses_shards_and_drops_one_tail(prepared_bundle):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=2,
        cursor=PretrainingCursor.initial(),
    )
    identifiers = []
    shapes = []
    cursors = []
    with stream:
        for envelope in stream.iter_epoch(0):
            try:
                identifiers.extend(envelope.rows[:, 0].tolist())
                shapes.append(envelope.rows.shape)
                cursors.append(envelope.cursor_after)
            finally:
                envelope.release()

    assert identifiers == [10, 11, 20, 21, 0, 1]
    assert shapes == [(3, prepared_bundle.manifest.row_width)] * 2
    assert cursors == [PretrainingCursor(0, 1, 1), PretrainingCursor(1, 0, 0)]


def test_resume_stream_starts_at_normalized_cursor_without_replay(prepared_bundle):
    cursor = PretrainingCursor(epoch=0, shard_order_position=1, row_offset=2)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=2,
        seed=5,
        prefetch_depth=4,
        cursor=cursor,
    )

    with stream:
        assert stream.committed_cursor == PretrainingCursor(0, 2, 0)
        resumed = next(iter(stream))
        try:
            assert resumed.rows[:, 0].tolist() == [0, 1]
            assert resumed.cursor_after == PretrainingCursor(1, 0, 0)
        finally:
            resumed.release()


def test_stream_normalizes_exact_epoch_end_and_continues_in_next_epoch(
    prepared_bundle,
):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=2,
        cursor=PretrainingCursor(0, 3, 0),
    )

    with stream:
        assert stream.committed_cursor == PretrainingCursor(1, 0, 0)
        envelope = next(stream)
        try:
            assert envelope.rows[:, 0].tolist() == [0, 1, 20]
            assert envelope.cursor_after == PretrainingCursor(1, 1, 1)
        finally:
            envelope.release()


@pytest.mark.parametrize(
    "cursor",
    [
        PretrainingCursor(0, 4, 0),
        PretrainingCursor(0, 0, 3),
        PretrainingCursor(0, 3, 1),
    ],
)
def test_stream_rejects_cursor_beyond_epoch_order_or_shard(prepared_bundle, cursor):
    with pytest.raises(SMLDataError, match="cursor"):
        PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=cursor,
        )


def test_dropped_tail_stream_retains_commit_and_does_not_consume_next_epoch(
    prepared_bundle,
):
    width = prepared_bundle.manifest.row_width
    prepared_bundle = _replace_shards(
        prepared_bundle,
        (
            _rows(width, 0, 1),
            _rows(width, 10, 11),
            _rows(width, 20, 21, 22),
        ),
    )
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=2,
        cursor=PretrainingCursor.initial(),
    )

    epoch_zero_rows = []
    with stream:
        for envelope in stream.iter_epoch(0):
            try:
                epoch_zero_rows.extend(envelope.rows[:, 0].tolist())
                stream.commit(envelope.cursor_after)
            finally:
                envelope.release()

        assert epoch_zero_rows == [10, 11, 20, 21, 22, 0]
        assert stream.committed_cursor == PretrainingCursor(0, 2, 1)
        with pytest.raises(SMLDataError, match="delivered"):
            stream.commit(PretrainingCursor(1, 0, 0))
        resume_cursor = stream.committed_cursor

    resumed = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=2,
        cursor=resume_cursor,
    )
    resumed_rows = []
    resumed_cursors = []
    with resumed:
        for _index in range(3):
            envelope = next(resumed)
            try:
                resumed_rows.append(envelope.rows[:, 0].tolist())
                resumed_cursors.append(envelope.cursor_after)
            finally:
                envelope.release()

    assert resumed_rows == [[0, 1, 20], [21, 22, 10], [0, 1, 10]]
    assert resumed_cursors == [
        PretrainingCursor(1, 1, 1),
        PretrainingCursor(1, 2, 1),
        PretrainingCursor(2, 1, 1),
    ]


def test_prefetch_producer_position_is_not_committed(prepared_bundle):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=4,
        cursor=PretrainingCursor.initial(),
    )

    with stream:
        first, second = _take(stream, 2)
        try:
            assert stream.committed_cursor == PretrainingCursor.initial()
            stream.commit(second.cursor_after)
            assert stream.committed_cursor == second.cursor_after
            with pytest.raises(SMLDataError, match="regress"):
                stream.commit(first.cursor_after)
            with pytest.raises(SMLDataError, match="delivered"):
                stream.commit(PretrainingCursor(99, 0, 0))
        finally:
            first.release()
            second.release()


def test_stream_builds_the_shard_permutation_once_per_active_epoch(
    prepared_bundle, monkeypatch
):
    generator_constructions = 0
    numpy_generator = np.random.Generator

    def counting_generator(bit_generator):
        nonlocal generator_constructions
        generator_constructions += 1
        return numpy_generator(bit_generator)

    monkeypatch.setattr(np.random, "Generator", counting_generator)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=1,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    with stream:
        for _index in range(4):
            envelope = next(stream)
            envelope.release()

        assert generator_constructions == 1


def test_full_queue_cross_shard_envelopes_do_not_share_mutable_staging(
    prepared_bundle,
):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=4,
        cursor=PretrainingCursor.initial(),
    )
    envelopes = _take(stream, 4)
    try:
        snapshots = [envelope.rows.copy() for envelope in envelopes]
        assert all(not envelope.rows.flags.writeable for envelope in envelopes)
        assert len(
            {envelope.rows.__array_interface__["data"][0] for envelope in envelopes}
        ) == len(envelopes)
        assert [envelope.rows.tolist() for envelope in envelopes] == [
            rows.tolist() for rows in snapshots
        ]
    finally:
        for envelope in envelopes:
            envelope.release()
        stream.close()


def test_prefetch_pool_double_release_cannot_free_a_reused_live_buffer(
    prepared_bundle,
):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    iterator = iter(stream)
    first = next(iterator)
    first_pointer = first.rows.__array_interface__["data"][0]
    first.release()
    second = next(iterator)
    assert second.rows.__array_interface__["data"][0] == first_pointer

    delivered = []
    failures = []
    completed = threading.Event()

    def consume_third():
        try:
            delivered.append(next(iterator))
        except Exception as error:  # noqa: BLE001 - captured from worker thread
            failures.append(error)
        finally:
            completed.set()

    consumer = threading.Thread(target=consume_third)
    consumer.start()
    try:
        assert not completed.wait(0.15)
        first.release()
        assert not completed.wait(0.15)
        second.release()
        assert completed.wait(2)
        assert failures == []
        assert len(delivered) == 1
    finally:
        second.release()
        for envelope in delivered:
            envelope.release()
        stream.close()
        consumer.join(timeout=2)


def test_mlx_transfer_does_not_alias_released_prefetch_staging(prepared_bundle):
    import mlx.core as mx

    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    iterator = iter(stream)
    first = next(iterator)
    snapshot = first.rows.copy()
    first_pointer = first.rows.__array_interface__["data"][0]
    device_rows = mx.array(first.rows)
    first.release()
    reused = next(iterator)
    try:
        assert reused.rows.__array_interface__["data"][0] == first_pointer
        mx.eval(device_rows)
        assert device_rows.tolist() == snapshot.tolist()
    finally:
        reused.release()
        stream.close()


class _CloseCoordinatedQueue(queue.Queue):
    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.full_put_entered = threading.Event()
        self.drain_observed = threading.Event()
        self.release_put = threading.Event()

    def put(self, item, block=True, timeout=None):
        if self.full():
            self.full_put_entered.set()
            self.release_put.wait(2)
        return super().put(item, block=block, timeout=timeout)

    def get_nowait(self):
        item = super().get_nowait()
        self.drain_observed.set()
        self.release_put.set()
        return item


def test_stream_close_drains_full_queue_before_join(prepared_bundle, monkeypatch):
    monkeypatch.setattr(pretraining_module.queue, "Queue", _CloseCoordinatedQueue)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=1,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    queue_instance = stream._queue
    assert isinstance(queue_instance, _CloseCoordinatedQueue)
    assert queue_instance.full_put_entered.wait(2)

    closer = threading.Thread(target=stream.close)
    closer.start()
    drain_observed_before_emergency = queue_instance.drain_observed.wait(2)
    if not drain_observed_before_emergency:
        queue_instance.release_put.set()
    closer.join(timeout=2)

    assert drain_observed_before_emergency
    assert not closer.is_alive()
    assert stream._producer is None or not stream._producer.is_alive()
    assert stream._queue.empty()
    assert stream._owned_envelopes == {}
    stream.close()


def test_stream_close_wakes_full_queue_and_abandoned_envelope(prepared_bundle):
    full_stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=2,
        cursor=PretrainingCursor.initial(),
    )
    deadline = time.monotonic() + 2
    while not full_stream._queue.full() and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert full_stream._queue.full()
    closer = threading.Thread(target=full_stream.close)
    closer.start()
    closer.join(timeout=2)
    assert not closer.is_alive()
    full_stream.close()

    abandoned_stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    abandoned = next(abandoned_stream)
    closer = threading.Thread(target=abandoned_stream.close)
    closer.start()
    closer.join(timeout=2)
    assert not closer.is_alive()
    abandoned.release()
    abandoned_stream.close()
    with pytest.raises(StopIteration):
        next(abandoned_stream)


def test_stream_close_wakes_consumer_waiting_on_empty_queue(
    prepared_bundle, monkeypatch
):
    producer_entered = threading.Event()
    allow_producer_exit = threading.Event()

    def block_producer(_stream, cursor):
        producer_entered.set()
        allow_producer_exit.wait(2)
        return None, cursor

    monkeypatch.setattr(PretrainingBatchStream, "_next_produced", block_producer)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    assert producer_entered.wait(2)

    consumer_finished = threading.Event()

    def wait_for_batch():
        try:
            next(stream)
        except StopIteration:
            pass
        finally:
            consumer_finished.set()

    consumer = threading.Thread(target=wait_for_batch, daemon=True)
    consumer.start()
    threading.Event().wait(0.1)
    closer = threading.Thread(target=stream.close, daemon=True)
    closer.start()
    allow_producer_exit.set()
    closer.join(timeout=0.5)
    closed_without_deadlock = not closer.is_alive()

    if not closed_without_deadlock:  # Emergency cleanup for the RED implementation.
        with stream._state_condition:
            stream._closed = True
            stream._state_condition.notify_all()
    consumer.join(timeout=2)
    closer.join(timeout=2)

    assert closed_without_deadlock
    assert consumer_finished.is_set()


@pytest.mark.parametrize("consumer_api", ["next", "iter_epoch"])
def test_stream_producer_failure_does_not_deadlock_with_concurrent_close(
    prepared_bundle, monkeypatch, consumer_api
):
    producer_failed = threading.Event()

    def fail_producer(_stream, _cursor):
        producer_failed.set()
        raise RuntimeError("forced producer failure")

    monkeypatch.setattr(PretrainingBatchStream, "_next_produced", fail_producer)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    allow_failure_pull = threading.Event()
    consumer_holds_lock = threading.Event()
    close_reached_consumer_lock = threading.Event()
    consumer_finished = threading.Event()
    close_finished = threading.Event()
    consumer_errors = []
    close_errors = []
    consumer = None
    closer = None

    def emergency_notify_closed():
        with stream._state_condition:
            stream._closed = True
            stream._state_condition.notify_all()

    try:
        assert producer_failed.wait(2)
        deadline = time.monotonic() + 2
        while stream._queue.empty() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert not stream._queue.empty()

        original_pull_envelope = stream._pull_envelope

        def coordinated_pull_envelope():
            consumer_holds_lock.set()
            assert allow_failure_pull.wait(2)
            return original_pull_envelope()

        monkeypatch.setattr(stream, "_pull_envelope", coordinated_pull_envelope)
        pool = stream._pool
        assert pool is not None
        original_pool_stop = pool.stop

        def coordinated_pool_stop():
            original_pool_stop()
            close_reached_consumer_lock.set()

        monkeypatch.setattr(pool, "stop", coordinated_pool_stop)

        def consume_failure():
            try:
                if consumer_api == "next":
                    next(stream)
                else:
                    next(stream.iter_epoch(0))
            except BaseException as error:  # noqa: BLE001 - worker boundary
                consumer_errors.append(error)
            finally:
                consumer_finished.set()

        def close_stream():
            try:
                stream.close()
            except BaseException as error:  # noqa: BLE001 - worker boundary
                close_errors.append(error)
            finally:
                close_finished.set()

        consumer = threading.Thread(target=consume_failure)
        consumer.start()
        assert consumer_holds_lock.wait(2)
        closer = threading.Thread(target=close_stream)
        closer.start()
        assert close_reached_consumer_lock.wait(2)
        allow_failure_pull.set()

        failure_propagated_before_emergency = consumer_finished.wait(0.5)
        if not failure_propagated_before_emergency:
            emergency_notify_closed()
        consumer.join(timeout=2)
        closer.join(timeout=2)

        assert failure_propagated_before_emergency
        assert close_finished.is_set()
        assert not consumer.is_alive()
        assert not closer.is_alive()
        assert close_errors == []
        assert len(consumer_errors) == 1
        assert isinstance(consumer_errors[0], SMLDataError)
        assert str(consumer_errors[0]) == "pretraining batch producer failed"
        assert isinstance(consumer_errors[0].__cause__, RuntimeError)
        assert str(consumer_errors[0].__cause__) == "forced producer failure"
        assert stream._queue.empty()
        assert stream._owned_envelopes == {}
    finally:
        allow_failure_pull.set()
        if consumer is not None and consumer.is_alive():
            emergency_notify_closed()
            consumer.join(timeout=2)
        if closer is not None:
            closer.join(timeout=2)
        stream.close()


def test_stream_producer_exception_propagates_as_focused_data_error(
    prepared_bundle, monkeypatch
):
    def fail_producer(_stream, _cursor):
        raise RuntimeError("forced producer failure")

    monkeypatch.setattr(PretrainingBatchStream, "_next_produced", fail_producer)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    try:
        with pytest.raises(SMLDataError, match="producer failed") as raised:
            next(stream)
        assert isinstance(raised.value.__cause__, RuntimeError)
    finally:
        stream.close()


def test_preflight_preserves_semantic_error_when_cleanup_fails(
    prepared_bundle, monkeypatch
):
    semantic_error = SMLDataError(
        "prepared bundle does not contain one full runtime batch"
    )
    cleanup_error = RuntimeError("injected preflight postcheck failure")
    original_close = pretraining_module._close_prepared_resources

    def fail_after_close(*args, **kwargs):
        original_close(*args, **kwargs)
        raise cleanup_error

    monkeypatch.setattr(
        pretraining_module, "_close_prepared_resources", fail_after_close
    )

    with pytest.raises(SMLDataError) as raised:
        pretraining_module.preflight_pretraining_bundle(
            prepared_bundle,
            batch_size=sum(prepared_bundle.manifest.shard_row_counts) + 1,
        )

    assert str(raised.value) == str(semantic_error)
    assert raised.value.__cause__ is cleanup_error


@pytest.mark.parametrize("consumer_api", ["next", "iter_epoch"])
def test_stream_preserves_producer_failure_when_close_fails(
    prepared_bundle, monkeypatch, consumer_api
):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    producer_error = SMLDataError("injected producer failure")
    cleanup_error = RuntimeError("injected stream cleanup failure")
    original_close = stream.close

    def fail_after_close():
        original_close()
        raise cleanup_error

    monkeypatch.setattr(
        stream,
        "_pull_envelope",
        lambda: pretraining_module._ProducerFailure(producer_error),
    )
    monkeypatch.setattr(stream, "close", fail_after_close)
    try:
        with pytest.raises(SMLDataError) as raised:
            if consumer_api == "next":
                next(stream)
            else:
                next(stream.iter_epoch(0))

        assert raised.value is producer_error
        assert raised.value.__cause__ is cleanup_error
    finally:
        original_close()


def test_stream_exit_preserves_body_error_when_close_fails(
    prepared_bundle, monkeypatch
):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
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


def test_stream_maps_open_descriptors_without_numpy_path_reload(
    prepared_bundle, monkeypatch
):
    def reject_path_reload(*_args, **_kwargs):
        raise AssertionError("runtime loader must not call np.load")

    monkeypatch.setattr(np, "load", reject_path_reload)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    with stream:
        envelope = next(stream)
        try:
            assert envelope.rows[:, 0].tolist() == [10, 11, 20]
        finally:
            envelope.release()


def test_stream_uses_retained_root_after_path_replacement(prepared_bundle, monkeypatch):
    original_open = pretraining_module.open_artifact
    expected = _load_rows(prepared_bundle.path)

    def replace_after_open(path, manifest_types, verification):
        artifact = original_open(path, manifest_types, verification)
        retained = path.with_name(path.name + "-retained")
        path.rename(retained)
        shutil.copytree(retained, path)
        shard = artifact.manifest.shards[0]
        replacement = np.full(
            (artifact.manifest.shard_row_counts[0], artifact.manifest.row_width),
            99,
            dtype="<i4",
        )
        with path.joinpath(shard.logical_path).open("wb") as payload:
            np.save(payload, replacement, allow_pickle=False)
        return artifact

    monkeypatch.setattr(
        pretraining_module,
        "open_artifact",
        replace_after_open,
        raising=False,
    )
    with (
        PretrainingBatchStream(
            prepared_bundle,
            batch_size=1,
            seed=5,
            prefetch_depth=1,
            cursor=PretrainingCursor.initial(),
        ) as stream,
        next(stream) as envelope,
    ):
        assert any(np.array_equal(envelope.rows[0], row) for row in expected)


def test_stream_uses_proven_shard_after_payload_path_replacement(
    prepared_bundle, monkeypatch
):
    reference = prepared_bundle.manifest.shards[0]
    expected_rows = _load_rows(prepared_bundle.path)
    original_shard = np.load(
        prepared_bundle.path / reference.logical_path,
        allow_pickle=False,
    )
    original_open_payload = OpenedArtifact.open_payload

    def replace_after_proof(artifact, payload_reference):
        payload = original_open_payload(artifact, payload_reference)
        if payload_reference.logical_path == reference.logical_path:
            source = prepared_bundle.path / reference.logical_path
            retained = source.with_suffix(".proven.npy")
            source.rename(retained)
            with source.open("wb") as replacement:
                np.save(
                    replacement, np.full_like(original_shard, 99), allow_pickle=False
                )
        return payload

    monkeypatch.setattr(OpenedArtifact, "open_payload", replace_after_proof)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=1,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    try:
        with next(stream) as envelope:
            assert any(np.array_equal(envelope.rows[0], row) for row in expected_rows)
            assert not bool(np.all(envelope.rows[0] == 99))
    finally:
        with pytest.raises(SMLArtifactError, match="changed during use"):
            stream.close()


def test_stream_close_detects_in_place_shard_mutation(prepared_bundle):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=1,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    payload_path = (
        prepared_bundle.path / prepared_bundle.manifest.shards[0].logical_path
    )
    with payload_path.open("r+b") as payload:
        payload.seek(-1, 2)
        final_byte = payload.read(1)
        payload.seek(-1, 2)
        payload.write(bytes([final_byte[0] ^ 1]))
        payload.flush()

    with pytest.raises(SMLArtifactError, match="changed during use"):
        stream.close()
    stream.close()


def _instrument_real_prepared_cleanup(
    monkeypatch,
    *,
    fail_ndarray_index: int | None = None,
    fail_shard_open_index: int | None = None,
):
    from sml.artifacts.manifest import ArtifactRoot, VerifiedPayload

    events: list[tuple[str, int | str | None]] = []
    mappings: list[object] = []
    payloads: list[VerifiedPayload] = []
    roots: list[ArtifactRoot] = []
    array_refs: dict[int, weakref.ReferenceType[np.ndarray]] = {}
    original_mmap = pretraining_module.mmap.mmap
    original_ndarray = pretraining_module.np.ndarray
    original_open_payload = OpenedArtifact.open_payload
    original_payload_close = VerifiedPayload.close
    original_root_close = ArtifactRoot.close

    class ObservedMmap(original_mmap):
        def __new__(cls, *args, **kwargs):
            mapping = super().__new__(cls, *args, **kwargs)
            mappings.append(mapping)
            return mapping

        def close(self):
            index = mappings.index(self)
            events.append(("mmap", index))
            array_ref = array_refs.get(id(self))
            assert array_ref is None or array_ref() is None, (
                f"live prepared ndarray still exports mapping {index}"
            )
            return super().close()

    def open_payload(artifact, reference):
        if (
            reference.logical_path.startswith("shards/")
            and fail_shard_open_index is not None
            and len(payloads) == fail_shard_open_index
        ):
            raise RuntimeError("injected next-shard open failure")
        payload = original_open_payload(artifact, reference)
        if reference.logical_path.startswith("shards/"):
            payloads.append(payload)
        return payload

    def close_payload(payload):
        if payload in payloads:
            events.append(("payload", payloads.index(payload)))
        return original_payload_close(payload)

    def close_root(root):
        roots.append(root)
        events.append(("root", None))
        return original_root_close(root)

    def construct_array(*args, **kwargs):
        array = original_ndarray(*args, **kwargs)
        buffer = kwargs.get("buffer")
        if isinstance(buffer, ObservedMmap):
            array_refs[id(buffer)] = weakref.ref(array)
            if (
                fail_ndarray_index is not None
                and mappings.index(buffer) == fail_ndarray_index
            ):
                raise RuntimeError("injected ndarray construction failure")
        return array

    monkeypatch.setattr(pretraining_module.mmap, "mmap", ObservedMmap)
    monkeypatch.setattr(pretraining_module.np, "ndarray", construct_array)
    monkeypatch.setattr(OpenedArtifact, "open_payload", open_payload)
    monkeypatch.setattr(VerifiedPayload, "close", close_payload)
    monkeypatch.setattr(ArtifactRoot, "close", close_root)
    return events, mappings, payloads, roots


def _assert_real_prepared_cleanup(events, mappings, payloads, roots):
    assert events == [
        *(("mmap", index) for index in reversed(range(len(mappings)))),
        *(("payload", index) for index in reversed(range(len(payloads)))),
        ("root", None),
    ]
    assert mappings
    assert all(mapping.closed for mapping in mappings)
    assert all(payload.closed for payload in payloads)
    assert len(roots) == 1
    assert roots[0]._fd == -1


def test_prepared_ndarray_failure_closes_real_mapping_payload_and_root(
    prepared_bundle, monkeypatch
):
    events, mappings, payloads, roots = _instrument_real_prepared_cleanup(
        monkeypatch,
        fail_ndarray_index=0,
    )

    with pytest.raises(RuntimeError, match="injected ndarray construction failure"):
        pretraining_module._open_validated_prepared_resources(prepared_bundle)

    _assert_real_prepared_cleanup(events, mappings, payloads, roots)


def test_prepared_next_shard_failure_closes_real_mapping_payload_and_root(
    prepared_bundle, monkeypatch
):
    events, mappings, payloads, roots = _instrument_real_prepared_cleanup(
        monkeypatch,
        fail_shard_open_index=1,
    )

    with pytest.raises(RuntimeError, match="injected next-shard open failure"):
        pretraining_module._open_validated_prepared_resources(prepared_bundle)

    _assert_real_prepared_cleanup(events, mappings, payloads, roots)


def test_prepared_semantic_failure_closes_real_mappings_payloads_and_root(
    prepared_bundle, monkeypatch
):
    width = prepared_bundle.manifest.row_width
    prepared_bundle = _replace_shards(
        prepared_bundle,
        (_rows(width, 0, 1), _rows(width, 10, 11), _rows(width, 20, 300)),
    )
    events, mappings, payloads, roots = _instrument_real_prepared_cleanup(monkeypatch)

    with pytest.raises(SMLArtifactError, match="token IDs"):
        pretraining_module._open_validated_prepared_resources(prepared_bundle)

    _assert_real_prepared_cleanup(events, mappings, payloads, roots)


def test_prepared_detach_failure_closes_real_mappings_payloads_and_root(
    prepared_bundle, monkeypatch
):
    events, mappings, payloads, roots = _instrument_real_prepared_cleanup(monkeypatch)

    def fail_detach(_artifact):
        raise RuntimeError("injected prepared detach failure")

    monkeypatch.setattr(OpenedArtifact, "detach_root", fail_detach)
    with pytest.raises(RuntimeError, match="injected prepared detach failure"):
        pretraining_module._open_validated_prepared_resources(prepared_bundle)

    _assert_real_prepared_cleanup(events, mappings, payloads, roots)


def test_prepared_resource_cleanup_releases_views_mappings_payloads_then_root():
    events: list[str] = []
    arrays = [np.zeros((1, 1), dtype="<i4")]

    class Mapping:
        def close(self):
            assert arrays == []
            events.append("mmap")

    class Payload:
        def close(self):
            assert events == ["mmap"]
            events.append("payload")

    class Root:
        def close(self):
            assert events == ["mmap", "payload"]
            events.append("root")

    mappings = [Mapping()]
    payloads = [Payload()]
    pretraining_module._close_prepared_resources(Root(), payloads, mappings, arrays)

    assert events == ["mmap", "payload", "root"]
    assert arrays == mappings == payloads == []


def test_prepared_resource_cleanup_continues_after_mapping_failure():
    events: list[str] = []
    arrays = [np.zeros((1, 1), dtype="<i4")]

    class Mapping:
        def close(self):
            events.append("mmap")
            raise RuntimeError("mmap close failed")

    class Payload:
        def close(self):
            events.append("payload")

    class Root:
        def close(self):
            events.append("root")

    with pytest.raises(RuntimeError, match="mmap close failed"):
        pretraining_module._close_prepared_resources(
            Root(), [Payload()], [Mapping()], arrays
        )

    assert events == ["mmap", "payload", "root"]
    assert arrays == []


def test_prepared_open_failure_preserves_semantic_error_when_cleanup_fails(
    prepared_bundle, monkeypatch
):
    from sml.artifacts.manifest import ArtifactRoot

    mismatched = replace(
        prepared_bundle,
        manifest=replace(
            prepared_bundle.manifest,
            diagnostic_source_locator="mismatched-source",
        ),
    )
    original_close = ArtifactRoot.close

    def fail_after_close(root):
        original_close(root)
        raise RuntimeError("root cleanup failed")

    monkeypatch.setattr(ArtifactRoot, "close", fail_after_close)
    with pytest.raises(SMLArtifactError, match="supplied prepared bundle") as caught:
        pretraining_module._open_validated_prepared_resources(mismatched)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "root cleanup failed"


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("batch_size", True, TypeError),
        ("batch_size", 0, ValueError),
        ("prefetch_depth", 1.0, TypeError),
        ("prefetch_depth", 0, ValueError),
        ("seed", np.int64(5), TypeError),
        ("seed", -1, ValueError),
    ],
)
def test_stream_rejects_invalid_plain_integer_arguments_before_thread_start(
    prepared_bundle, monkeypatch, argument, value, error
):
    arguments = {
        "batch_size": 3,
        "seed": 5,
        "prefetch_depth": 2,
        "cursor": PretrainingCursor.initial(),
    }
    arguments[argument] = value
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        error,
        argument,
        lambda: PretrainingBatchStream(prepared_bundle, **arguments),
    )


def test_stream_requires_a_fully_verified_prepared_bundle_before_thread_start(
    prepared_bundle, monkeypatch
):
    unverified = replace(
        prepared_bundle, verification=VerificationLevel.MANIFEST_TRUSTED
    )
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "FULL",
        lambda: PretrainingBatchStream(
            unverified,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_non_prepared_bundle_before_thread_start(monkeypatch):
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        TypeError,
        "PreparedDataBundle",
        lambda: PretrainingBatchStream(
            object(),
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_noncanonical_outer_manifest_before_thread_start(
    prepared_bundle, monkeypatch
):
    manifest_path = prepared_bundle.path / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "canonical",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_corrupt_shard_bytes_before_thread_start(
    prepared_bundle, monkeypatch
):
    shard_path = prepared_bundle.path / prepared_bundle.manifest.shards[0].logical_path
    payload = bytearray(shard_path.read_bytes())
    payload[-1] ^= 1
    shard_path.write_bytes(payload)
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "identity",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


@pytest.mark.parametrize("representation", ["dtype", "shape", "count", "order"])
def test_stream_rejects_invalid_npy_metadata_before_thread_start(
    prepared_bundle, monkeypatch, representation
):
    width = prepared_bundle.manifest.row_width
    if representation == "dtype":
        prepared_bundle = _replace_shard_payload(
            prepared_bundle, 0, np.zeros((2, width), dtype="<i8")
        )
    elif representation == "shape":
        prepared_bundle = _replace_shard_payload(
            prepared_bundle, 0, np.zeros((2, width + 1), dtype="<i4")
        )
    elif representation == "count":
        prepared_bundle = _replace_shard_payload(
            prepared_bundle, 0, np.zeros((3, width), dtype="<i4")
        )
    else:
        prepared_bundle = _replace_shard_payload(
            prepared_bundle,
            0,
            np.asfortranarray(np.zeros((2, width), dtype="<i4")),
        )

    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "shard",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_invalid_npy_header_before_thread_start(
    prepared_bundle, monkeypatch
):
    reference = prepared_bundle.manifest.shards[0]
    shard_path = prepared_bundle.path / reference.logical_path
    payload = bytearray(shard_path.read_bytes())
    payload[:6] = b"broken"
    shard_path.write_bytes(payload)
    references = list(prepared_bundle.manifest.shards)
    references[0] = _payload_ref(shard_path, reference.logical_path)
    prepared_bundle = _replace_manifest(prepared_bundle, shards=tuple(references))

    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "NPY|header|shard",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


@pytest.mark.parametrize("corruption", ["version", "trailing", "short"])
def test_stream_rejects_unsupported_or_inexact_npy_payload_before_thread_start(
    prepared_bundle, monkeypatch, corruption
):
    reference = prepared_bundle.manifest.shards[0]
    source = np.load(prepared_bundle.path / reference.logical_path, allow_pickle=False)
    payload = io.BytesIO()
    np.lib.format.write_array(
        payload,
        source,
        version=(3, 0) if corruption == "version" else (1, 0),
        allow_pickle=False,
    )
    payload_bytes = payload.getvalue()
    if corruption == "trailing":
        payload_bytes += b"trailing"
    elif corruption == "short":
        payload_bytes = payload_bytes[:-1]
    prepared_bundle = _replace_shard_bytes(prepared_bundle, 0, payload_bytes)

    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "NPY|header|size|shard",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_retains_nonwriteable_descriptor_mapped_shards(prepared_bundle):
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    try:
        assert stream._shard_arrays
        assert all(not array.flags.writeable for array in stream._shard_arrays)
        with pytest.raises(ValueError, match="read-only"):
            stream._shard_arrays[0][0, 0] = 7
    finally:
        stream.close()


@pytest.mark.parametrize("invalid_token", [-1, 300])
def test_stream_rejects_bundle_wide_invalid_token_range_before_thread_start(
    prepared_bundle, monkeypatch, invalid_token
):
    width = prepared_bundle.manifest.row_width
    prepared_bundle = _replace_shards(
        prepared_bundle,
        (
            _rows(width, 0, 1),
            _rows(width, 10, 11),
            _rows(width, 20, invalid_token),
        ),
    )
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "token ID",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_binds_copied_tokenizer_manifest_to_outer_manifest_before_start(
    prepared_bundle, monkeypatch
):
    prepared_bundle = _replace_manifest(
        prepared_bundle, tokenizer_identity="sha256:" + "0" * 64
    )
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "tokenizer",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_binds_copied_tokenizer_payload_refs_before_thread_start(
    prepared_bundle, monkeypatch
):
    prepared_bundle = _replace_manifest(
        prepared_bundle,
        tokenizer_model=prepared_bundle.manifest.tokenizer_vocab,
        tokenizer_vocab=prepared_bundle.manifest.tokenizer_model,
    )
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "tokenizer",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_noncanonical_copied_tokenizer_manifest_before_start(
    prepared_bundle, monkeypatch
):
    manifest_path = prepared_bundle.path / "tokenizer" / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLArtifactError,
        "tokenizer.*canonical|canonical.*tokenizer",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )


def test_stream_rejects_bundle_too_small_for_one_batch_before_thread_start(
    prepared_bundle, monkeypatch
):
    width = prepared_bundle.manifest.row_width
    prepared_bundle = _replace_shards(
        prepared_bundle, (_rows(width, 0), _rows(width, 10))
    )
    _assert_constructor_fails_before_thread_start(
        monkeypatch,
        SMLDataError,
        "full runtime batch",
        lambda: PretrainingBatchStream(
            prepared_bundle,
            batch_size=3,
            seed=5,
            prefetch_depth=2,
            cursor=PretrainingCursor.initial(),
        ),
    )
