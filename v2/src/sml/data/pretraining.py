"""Deterministic preparation of immutable, mmap-ready pretraining data."""

from __future__ import annotations

import hashlib
import json
import mmap
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from traceback import clear_frames
from typing import Self

import numpy as np

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    ArtifactRoot,
    OpenedArtifact,
    PayloadRef,
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    VerifiedPayload,
    canonical_json_bytes,
    file_identity,
    open_artifact,
    row_content_identity,
)
from sml.data.corpus import CorpusConfig, discover_corpus_files, iter_filtered_texts
from sml.data.tokenizer import LoadedTokenizer, load_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLDataError

WINDOWED_ROW_SHUFFLE_V1 = "windowed-row-shuffle-v1"

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_TOKENIZER_MANIFEST = "manifest.json"
_TOKENIZER_MODEL = "tokenizer.model"
_TOKENIZER_VOCAB = "tokenizer.vocab"
_INT32 = np.dtype("<i4")


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class PretrainingPreparationConfig:
    """Complete configuration for deterministic pretraining-row production."""

    corpus: CorpusConfig
    tokenizer_bundle: Path
    sequence_length: int = 1_024
    shuffle_window_rows: int = 4_096
    shuffle_algorithm: str = WINDOWED_ROW_SHUFFLE_V1
    output_shard_rows: int = 4_096
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, CorpusConfig):
            raise TypeError("corpus must be a CorpusConfig")
        if not isinstance(self.tokenizer_bundle, Path):
            raise TypeError("tokenizer_bundle must be a Path")
        object.__setattr__(self, "tokenizer_bundle", self.tokenizer_bundle.expanduser())
        _require_plain_int(self.sequence_length, "sequence_length", minimum=1)
        _require_plain_int(self.shuffle_window_rows, "shuffle_window_rows", minimum=1)
        if not isinstance(self.shuffle_algorithm, str):
            raise TypeError("shuffle_algorithm must be a string")
        if self.shuffle_algorithm != WINDOWED_ROW_SHUFFLE_V1:
            raise ValueError(f"shuffle_algorithm must be {WINDOWED_ROW_SHUFFLE_V1!r}")
        _require_plain_int(self.output_shard_rows, "output_shard_rows", minimum=1)
        _require_plain_int(self.seed, "seed")


@dataclass(frozen=True, slots=True)
class PreparedDataBundle:
    path: Path
    manifest: PretrainingDataManifest
    verification: VerificationLevel


@dataclass(frozen=True, slots=True)
class PreparedDataPreflight:
    bundle: PreparedDataBundle
    row_count: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class PretrainingCursor:
    """Canonical location of the next row eligible for a runtime batch."""

    epoch: int
    shard_order_position: int
    row_offset: int

    def __post_init__(self) -> None:
        _require_plain_int(self.epoch, "epoch")
        _require_plain_int(self.shard_order_position, "shard_order_position")
        _require_plain_int(self.row_offset, "row_offset")

    @classmethod
    def initial(cls) -> PretrainingCursor:
        return cls(epoch=0, shard_order_position=0, row_offset=0)


def _epoch_shard_order(shard_count: int, *, seed: int, epoch: int) -> tuple[int, ...]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, epoch]))
    )
    return tuple(int(index) for index in generator.permutation(shard_count))


def canonicalize_pretraining_cursor(
    cursor: PretrainingCursor,
    *,
    shard_row_counts: tuple[int, ...],
    seed: int,
    shard_order: tuple[int, ...] | None = None,
) -> PretrainingCursor:
    """Return the unique cursor representation for deterministic shard geometry."""
    if not isinstance(cursor, PretrainingCursor):
        raise TypeError("cursor must be a PretrainingCursor")
    if not isinstance(shard_row_counts, tuple) or not shard_row_counts:
        raise TypeError("shard_row_counts must be a nonempty tuple")
    for row_count in shard_row_counts:
        _require_plain_int(row_count, "shard row count", minimum=1)
    _require_plain_int(seed, "seed")

    epoch = cursor.epoch
    position = cursor.shard_order_position
    offset = cursor.row_offset
    if shard_order is None:
        order = _epoch_shard_order(len(shard_row_counts), seed=seed, epoch=epoch)
    else:
        if (
            not isinstance(shard_order, tuple)
            or len(shard_order) != len(shard_row_counts)
            or set(shard_order) != set(range(len(shard_row_counts)))
        ):
            raise ValueError("shard_order must be a complete shard permutation")
        order = shard_order
    if position > len(order):
        raise SMLDataError("pretraining cursor is beyond the epoch shard order")
    if position == len(order):
        if offset != 0:
            raise SMLDataError("pretraining cursor offset is beyond the epoch")
        return PretrainingCursor(epoch + 1, 0, 0)

    row_count = shard_row_counts[order[position]]
    if offset > row_count:
        raise SMLDataError("pretraining cursor offset is beyond its shard")
    if offset < row_count:
        return cursor

    position += 1
    if position == len(order):
        return PretrainingCursor(epoch + 1, 0, 0)
    return PretrainingCursor(epoch, position, 0)


_EMPTY_ROWS = np.empty((0, 0), dtype=_INT32)
_EMPTY_ROWS.setflags(write=False)


class BatchEnvelope:
    """A read-only NumPy batch whose owned storage has explicit lifetime."""

    __slots__ = (
        "_cursor_after",
        "_release_callback",
        "_release_lock",
        "_released",
        "_rows",
        "_source_epoch",
    )

    def __init__(self, rows: np.ndarray, cursor_after: PretrainingCursor) -> None:
        self._initialize(rows, cursor_after, source_epoch=cursor_after.epoch)

    @classmethod
    def _owned(
        cls,
        rows: np.ndarray,
        cursor_after: PretrainingCursor,
        *,
        source_epoch: int,
    ) -> BatchEnvelope:
        envelope = cls.__new__(cls)
        envelope._initialize(rows, cursor_after, source_epoch=source_epoch)
        return envelope

    def _initialize(
        self,
        rows: np.ndarray,
        cursor_after: PretrainingCursor,
        *,
        source_epoch: int,
    ) -> None:
        if not isinstance(rows, np.ndarray):
            raise TypeError("rows must be a NumPy array")
        if rows.ndim != 2:
            raise ValueError("rows must be a two-dimensional array")
        if not isinstance(cursor_after, PretrainingCursor):
            raise TypeError("cursor_after must be a PretrainingCursor")
        _require_plain_int(source_epoch, "source_epoch")
        readonly_rows = rows.view()
        readonly_rows.setflags(write=False)
        self._rows = readonly_rows
        self._cursor_after = cursor_after
        self._source_epoch = source_epoch
        self._release_callback: Callable[[], None] | None = None
        self._release_lock = threading.Lock()
        self._released = False

    @property
    def rows(self) -> np.ndarray:
        return self._rows

    @property
    def cursor_after(self) -> PretrainingCursor:
        return self._cursor_after

    def _set_release_callback(self, callback: Callable[[], None]) -> None:
        self._release_callback = callback

    def release(self) -> None:
        callback: Callable[[], None] | None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            self._rows = _EMPTY_ROWS
            callback = self._release_callback
            self._release_callback = None
        if callback is not None:
            callback()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.release()


class _ProducerStopped(Exception):
    pass


class _StagingPool:
    def __init__(self, capacity: int, shape: tuple[int, int]) -> None:
        self._buffers = tuple(
            np.empty(shape, dtype=_INT32) for _index in range(capacity)
        )
        self._available = deque(range(capacity))
        self._active_generations: list[int | None] = [None] * capacity
        self._condition = threading.Condition()
        self._next_generation = 0
        self._stopped = False

    def lease(self) -> tuple[np.ndarray, int, int]:
        with self._condition:
            while not self._available and not self._stopped:
                self._condition.wait()
            if self._stopped:
                raise _ProducerStopped
            index = self._available.popleft()
            self._next_generation += 1
            generation = self._next_generation
            self._active_generations[index] = generation
            return self._buffers[index], index, generation

    def release(self, index: int, generation: int) -> None:
        with self._condition:
            if self._active_generations[index] != generation:
                return
            self._active_generations[index] = None
            self._available.append(index)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class _ProducerFailure:
    error: SMLDataError


_QUEUE_STOP = object()


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def _parsed_payload_ref(raw: object, name: str) -> PayloadRef:
    if not isinstance(raw, dict) or set(raw) != {
        "logical_path",
        "identity",
        "byte_size",
    }:
        raise ValueError(f"{name} must be an exact payload reference")
    return PayloadRef(
        logical_path=raw["logical_path"],
        identity=raw["identity"],
        byte_size=raw["byte_size"],
    )


def _parse_canonical_tokenizer_manifest(payload: bytes) -> TokenizerManifest:
    expected_fields = {
        "kind",
        "version",
        "identity",
        "algorithm",
        "training",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "model",
        "vocab",
        "diagnostic_source_locator",
    }
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_no_duplicates,
        )
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError("tokenizer manifest fields do not match the exact schema")
        manifest = TokenizerManifest(
            kind=raw["kind"],
            version=raw["version"],
            identity=raw["identity"],
            algorithm=raw["algorithm"],
            training=raw["training"],
            vocab_size=raw["vocab_size"],
            bos_token_id=raw["bos_token_id"],
            eos_token_id=raw["eos_token_id"],
            pad_token_id=raw["pad_token_id"],
            unk_token_id=raw["unk_token_id"],
            model=_parsed_payload_ref(raw["model"], "tokenizer model"),
            vocab=_parsed_payload_ref(raw["vocab"], "tokenizer vocab"),
            diagnostic_source_locator=raw["diagnostic_source_locator"],
        )
        if manifest.recompute_identity() != manifest.identity:
            raise ValueError("tokenizer manifest identity mismatch")
        if canonical_json_bytes(manifest) != payload:
            raise ValueError("copied tokenizer manifest is not canonical")
        return manifest
    except SMLArtifactError:
        raise
    except (
        KeyError,
        TypeError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise SMLArtifactError(f"invalid copied tokenizer manifest: {error}") from error


def _map_npy_payload(
    payload: VerifiedPayload,
    reference: PayloadRef,
    *,
    declared_rows: int,
    row_width: int,
) -> tuple[mmap.mmap, np.ndarray]:
    stream = payload.stream
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version: {version}")
        data_offset = stream.tell()
    except (EOFError, OSError, TypeError, ValueError) as error:
        raise SMLArtifactError(
            f"invalid prepared shard NPY header: {reference.logical_path}"
        ) from error

    expected_shape = (declared_rows, row_width)
    if shape != expected_shape:
        raise SMLArtifactError(
            "prepared shard shape mismatch: "
            f"{reference.logical_path}; expected {expected_shape}, got {shape}"
        )
    if fortran_order:
        raise SMLArtifactError(
            f"prepared shard must use C order: {reference.logical_path}"
        )
    if np.dtype(dtype).str != _INT32.str or np.dtype(dtype).hasobject:
        raise SMLArtifactError(
            f"prepared shard dtype must be <i4: {reference.logical_path}"
        )
    expected_size = data_offset + declared_rows * row_width * _INT32.itemsize
    actual_size = payload.opened_stat.st_size
    if actual_size != expected_size:
        raise SMLArtifactError(
            f"prepared shard payload size mismatch: {reference.logical_path}"
        )

    mapping: mmap.mmap | None = None
    array: np.ndarray | None = None
    try:
        mapping = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        array = np.ndarray(
            expected_shape,
            dtype=_INT32,
            buffer=mapping,
            offset=data_offset,
            order="C",
        )
        array.setflags(write=False)
        return mapping, array
    except BaseException as error:
        if error.__traceback__ is not None:
            clear_frames(error.__traceback__)
        array = None
        cleanup_error: BaseException | None = None
        if mapping is not None:
            try:
                mapping.close()
            except BaseException as caught:  # noqa: BLE001 - cleanup must complete
                cleanup_error = caught
        if isinstance(error, (BufferError, OSError, TypeError, ValueError)):
            primary: BaseException = SMLArtifactError(
                f"could not memory-map prepared shard: {reference.logical_path}"
            )
        else:
            primary = error
        if cleanup_error is not None:
            raise primary from cleanup_error
        if primary is not error:
            raise primary from error
        raise


def _validate_prepared_tokenizer_binding(
    manifest: PretrainingDataManifest,
    tokenizer: TokenizerManifest,
) -> None:
    if tokenizer.identity != manifest.tokenizer_identity:
        raise SMLArtifactError(
            "copied tokenizer identity does not match prepared manifest"
        )
    expected = (
        (
            manifest.tokenizer_model,
            tokenizer.model,
            "tokenizer/tokenizer.model",
        ),
        (
            manifest.tokenizer_vocab,
            tokenizer.vocab,
            "tokenizer/tokenizer.vocab",
        ),
    )
    for outer, nested, logical_path in expected:
        if nested.logical_path != logical_path.removeprefix("tokenizer/"):
            raise SMLArtifactError(
                "copied tokenizer payload path does not match canonical layout"
            )
        if outer != replace(nested, logical_path=logical_path):
            raise SMLArtifactError(
                "copied tokenizer payload reference does not match prepared manifest"
            )


def _close_prepared_resources(
    root: ArtifactRoot | None,
    shard_files: list[VerifiedPayload],
    mappings: list[mmap.mmap],
    shard_arrays: list[np.ndarray],
) -> None:
    shard_arrays.clear()
    errors: list[BaseException] = []
    while mappings:
        try:
            mappings.pop().close()
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(error)
    while shard_files:
        try:
            shard_files.pop().close()
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            errors.append(error)
    if root is not None:
        try:
            root.close()
        except BaseException as error:  # noqa: BLE001 - cleanup must complete
            errors.append(error)
    if errors:
        raise errors[0]


def _open_validated_prepared_resources(
    bundle: PreparedDataBundle,
) -> tuple[
    ArtifactRoot,
    list[VerifiedPayload],
    list[mmap.mmap],
    list[np.ndarray],
    PretrainingDataManifest,
]:
    artifact: OpenedArtifact[PretrainingDataManifest] = open_artifact(
        bundle.path, (PretrainingDataManifest,), bundle.verification
    )
    if artifact.manifest != bundle.manifest:
        error = SMLArtifactError(
            "supplied prepared bundle manifest does not match the verified manifest"
        )
        try:
            artifact.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise error
    manifest = artifact.manifest
    shard_files: list[VerifiedPayload] = []
    mappings: list[mmap.mmap] = []
    shard_arrays: list[np.ndarray] = []
    payload: VerifiedPayload | None = None
    mapping: mmap.mmap | None = None
    array: np.ndarray | None = None
    try:
        with artifact.root.open_payload("manifest.json") as manifest_file:
            manifest_bytes = manifest_file.read()
        if manifest_bytes != canonical_json_bytes(manifest):
            raise SMLArtifactError("prepared bundle manifest is not canonical")

        with artifact.root.open_payload("tokenizer/manifest.json") as tokenizer_file:
            tokenizer_manifest = _parse_canonical_tokenizer_manifest(
                tokenizer_file.read()
            )
        _validate_prepared_tokenizer_binding(manifest, tokenizer_manifest)

        for reference in (manifest.tokenizer_model, manifest.tokenizer_vocab):
            with artifact.open_payload(reference):
                pass

        for reference, row_count in zip(
            manifest.shards,
            manifest.shard_row_counts,
            strict=True,
        ):
            payload = artifact.open_payload(reference)
            try:
                shard_files.append(payload)
            except BaseException as error:
                try:
                    payload.close()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise
            mapping, array = _map_npy_payload(
                payload,
                reference,
                declared_rows=row_count,
                row_width=manifest.row_width,
            )
            try:
                mappings.append(mapping)
            except BaseException as error:
                if error.__traceback__ is not None:
                    clear_frames(error.__traceback__)
                array = None
                try:
                    mapping.close()
                except BaseException as cleanup_error:
                    raise error from cleanup_error
                raise
            shard_arrays.append(array)
            payload = None
            mapping = None
            array = None

        if not shard_arrays:
            raise SMLDataError("prepared bundle contains no shards")
        minimum = min(int(array.min()) for array in shard_arrays)
        maximum = max(int(array.max()) for array in shard_arrays)
        if minimum < 0 or maximum >= tokenizer_manifest.vocab_size:
            raise SMLArtifactError(
                "prepared bundle token IDs must be in "
                f"[0, {tokenizer_manifest.vocab_size})"
            )
        root = artifact.detach_root()
        return root, shard_files, mappings, shard_arrays, manifest
    except BaseException as error:
        if error.__traceback__ is not None:
            clear_frames(error.__traceback__)
        payload = None
        mapping = None
        array = None
        cleanup_errors: list[BaseException] = []
        try:
            _close_prepared_resources(None, shard_files, mappings, shard_arrays)
        except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
            cleanup_errors.append(cleanup_error)
        try:
            artifact.close()
        except BaseException as cleanup_error:  # noqa: BLE001 - cleanup completes
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise error from cleanup_errors[0]
        raise


def preflight_pretraining_bundle(
    bundle: PreparedDataBundle,
    *,
    batch_size: int,
) -> PreparedDataPreflight:
    """Synchronously prove prepared bytes and semantics without starting a stream."""
    if not isinstance(bundle, PreparedDataBundle):
        raise TypeError("bundle must be a PreparedDataBundle")
    _require_plain_int(batch_size, "batch_size", minimum=1)
    if bundle.verification is not VerificationLevel.FULL:
        raise SMLArtifactError("prepared bundle must have FULL verification")
    root = None
    shard_files: list[VerifiedPayload] = []
    mappings: list[mmap.mmap] = []
    shard_arrays: list[np.ndarray] = []
    try:
        root, shard_files, mappings, shard_arrays, manifest = (
            _open_validated_prepared_resources(bundle)
        )
        row_count = sum(manifest.shard_row_counts)
        if row_count < batch_size:
            raise SMLDataError(
                "prepared bundle does not contain one full runtime batch"
            )
        result = PreparedDataPreflight(bundle, row_count, batch_size)
    except BaseException as error:
        try:
            _close_prepared_resources(root, shard_files, mappings, shard_arrays)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        _close_prepared_resources(root, shard_files, mappings, shard_arrays)
        return result


class PretrainingBatchStream(Iterator[BatchEnvelope]):
    """Bounded deterministic NumPy prefetch over memory-mapped epochs."""

    def __init__(
        self,
        bundle: PreparedDataBundle,
        *,
        batch_size: int,
        seed: int,
        prefetch_depth: int,
        cursor: PretrainingCursor,
    ) -> None:
        if not isinstance(bundle, PreparedDataBundle):
            raise TypeError("bundle must be a PreparedDataBundle")
        _require_plain_int(batch_size, "batch_size", minimum=1)
        _require_plain_int(seed, "seed")
        _require_plain_int(prefetch_depth, "prefetch_depth", minimum=1)
        if not isinstance(cursor, PretrainingCursor):
            raise TypeError("cursor must be a PretrainingCursor")
        if bundle.verification is not VerificationLevel.FULL:
            raise SMLArtifactError("prepared bundle must have FULL verification")
        if not isinstance(bundle.path, Path):
            raise TypeError("prepared bundle path must be a Path")
        if not isinstance(bundle.manifest, PretrainingDataManifest):
            raise TypeError(
                "prepared bundle manifest must be a PretrainingDataManifest"
            )

        self._batch_size = batch_size
        self._seed = seed
        self._prefetch_depth = prefetch_depth
        self._artifact_root: ArtifactRoot | None = None
        self._shard_files: list[VerifiedPayload] = []
        self._mappings: list[mmap.mmap] = []
        self._shard_arrays: list[np.ndarray] = []
        self._manifest = bundle.manifest
        self._stop = threading.Event()
        self._queue: queue.Queue[BatchEnvelope | _ProducerFailure | object] = (
            queue.Queue(maxsize=prefetch_depth)
        )
        self._pool: _StagingPool | None = None
        self._producer: threading.Thread | None = None
        self._consumer_lock = threading.RLock()
        self._state_condition = threading.Condition()
        self._envelope_lock = threading.Lock()
        self._owned_envelopes: dict[int, BatchEnvelope] = {}
        self._pending_envelope: BatchEnvelope | None = None
        self._delivered: dict[PretrainingCursor, int] = {}
        self._delivery_sequence = 0
        self._committed_sequence = 0
        self._order_cache_epoch = -1
        self._order_cache: tuple[int, ...] = ()
        self._order_suffix_rows: tuple[int, ...] = ()
        self._closing = False
        self._closed = False

        try:
            self._open_and_validate_bundle(bundle)
            total_rows = sum(self._manifest.shard_row_counts)
            if total_rows < batch_size:
                raise SMLDataError(
                    "prepared bundle does not contain one full runtime batch"
                )
            normalized_cursor = canonicalize_pretraining_cursor(
                cursor,
                shard_row_counts=self._manifest.shard_row_counts,
                seed=self._seed,
                shard_order=self._shard_order(cursor.epoch),
            )
            self._committed_cursor = normalized_cursor
            self._producer_cursor = normalized_cursor
            self._pool = _StagingPool(
                prefetch_depth,
                (batch_size, self._manifest.row_width),
            )
            self._producer = threading.Thread(
                target=self._produce,
                name="sml-pretraining-prefetch",
                daemon=True,
            )
            self._producer.start()
        except BaseException as error:
            try:
                self._close_open_resources()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    @property
    def committed_cursor(self) -> PretrainingCursor:
        with self._state_condition:
            return self._committed_cursor

    def _open_and_validate_bundle(self, bundle: PreparedDataBundle) -> None:
        (
            self._artifact_root,
            self._shard_files,
            self._mappings,
            self._shard_arrays,
            self._manifest,
        ) = _open_validated_prepared_resources(bundle)

    def _shard_order(self, epoch: int) -> tuple[int, ...]:
        if epoch == self._order_cache_epoch:
            return self._order_cache
        order = _epoch_shard_order(
            len(self._manifest.shard_row_counts),
            seed=self._seed,
            epoch=epoch,
        )
        suffix_rows = [0] * (len(order) + 1)
        for position in range(len(order) - 1, -1, -1):
            suffix_rows[position] = (
                suffix_rows[position + 1]
                + self._manifest.shard_row_counts[order[position]]
            )
        self._order_cache_epoch = epoch
        self._order_cache = order
        self._order_suffix_rows = tuple(suffix_rows)
        return order

    def _make_envelope(
        self,
        rows: np.ndarray,
        cursor_after: PretrainingCursor,
        *,
        source_epoch: int,
        pool_lease: tuple[int, int] | None,
    ) -> BatchEnvelope:
        envelope = BatchEnvelope._owned(
            rows,
            cursor_after,
            source_epoch=source_epoch,
        )
        envelope_key = id(envelope)

        def release_owned() -> None:
            if pool_lease is not None:
                pool = self._pool
                if pool is not None:
                    pool.release(*pool_lease)
            with self._envelope_lock:
                self._owned_envelopes.pop(envelope_key, None)

        envelope._set_release_callback(release_owned)
        with self._envelope_lock:
            self._owned_envelopes[envelope_key] = envelope
        return envelope

    def _next_produced(
        self, cursor: PretrainingCursor
    ) -> tuple[BatchEnvelope | None, PretrainingCursor]:
        cursor = canonicalize_pretraining_cursor(
            cursor,
            shard_row_counts=self._manifest.shard_row_counts,
            seed=self._seed,
            shard_order=self._shard_order(cursor.epoch),
        )
        order = self._shard_order(cursor.epoch)
        remaining = (
            self._order_suffix_rows[cursor.shard_order_position] - cursor.row_offset
        )
        if remaining < self._batch_size:
            return None, PretrainingCursor(cursor.epoch + 1, 0, 0)

        needed = self._batch_size
        position = cursor.shard_order_position
        offset = cursor.row_offset
        segments: list[np.ndarray] = []
        while needed:
            shard_index = order[position]
            shard = self._shard_arrays[shard_index]
            copied = min(needed, shard.shape[0] - offset)
            segments.append(shard[offset : offset + copied])
            needed -= copied
            offset += copied
            if offset == shard.shape[0]:
                position += 1
                offset = 0

        if position == len(order):
            cursor_after = PretrainingCursor(cursor.epoch + 1, 0, 0)
        else:
            cursor_after = PretrainingCursor(cursor.epoch, position, offset)

        if len(segments) == 1:
            rows = segments[0]
            pool_lease = None
        else:
            pool = self._pool
            if pool is None:
                raise _ProducerStopped
            rows, pool_index, generation = pool.lease()
            destination = 0
            for segment in segments:
                next_destination = destination + segment.shape[0]
                rows[destination:next_destination] = segment
                destination = next_destination
            pool_lease = (pool_index, generation)

        return (
            self._make_envelope(
                rows,
                cursor_after,
                source_epoch=cursor.epoch,
                pool_lease=pool_lease,
            ),
            cursor_after,
        )

    def _put(self, item: BatchEnvelope | _ProducerFailure | object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            cursor = self._producer_cursor
            while not self._stop.is_set():
                envelope, cursor = self._next_produced(cursor)
                self._producer_cursor = cursor
                if envelope is None:
                    continue
                if not self._put(envelope):
                    envelope.release()
                    return
        except _ProducerStopped:
            return
        except BaseException as error:  # noqa: BLE001 - cross-thread error boundary
            failure = SMLDataError("pretraining batch producer failed")
            failure.__cause__ = error
            self._put(_ProducerFailure(failure))
        finally:
            self._put(_QUEUE_STOP)

    def _pull_envelope(self) -> BatchEnvelope | _ProducerFailure:
        if self._pending_envelope is not None:
            envelope = self._pending_envelope
            self._pending_envelope = None
            return envelope
        while True:
            with self._state_condition:
                if self._closed:
                    raise StopIteration
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    raise StopIteration
                producer = self._producer
                if producer is not None and not producer.is_alive():
                    raise SMLDataError(
                        "pretraining batch producer terminated without a status"
                    )
                continue
            if isinstance(item, BatchEnvelope):
                return item
            if isinstance(item, _ProducerFailure):
                return item
            if item is _QUEUE_STOP:
                with self._state_condition:
                    if self._closed or self._stop.is_set():
                        raise StopIteration

    def _record_delivery(self, envelope: BatchEnvelope) -> None:
        with self._state_condition:
            self._delivery_sequence += 1
            self._delivered[envelope.cursor_after] = self._delivery_sequence

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> BatchEnvelope:
        with self._consumer_lock:
            item = self._pull_envelope()
            if isinstance(item, BatchEnvelope):
                self._record_delivery(item)
                return item
        try:
            self.close()
        except BaseException as cleanup_error:
            raise item.error from cleanup_error
        raise item.error

    def iter_epoch(self, epoch: int) -> Iterator[BatchEnvelope]:
        _require_plain_int(epoch, "epoch")
        while True:
            with self._consumer_lock:
                item = self._pull_envelope()
                if isinstance(item, BatchEnvelope):
                    if item._source_epoch > epoch:
                        self._pending_envelope = item
                        return
                    if item._source_epoch < epoch:
                        self._pending_envelope = item
                        raise SMLDataError(
                            "requested epoch is ahead of the next prefetched batch"
                        )
                    self._record_delivery(item)
            if isinstance(item, _ProducerFailure):
                try:
                    self.close()
                except BaseException as cleanup_error:
                    raise item.error from cleanup_error
                raise item.error
            yield item

    def commit(self, cursor_after: PretrainingCursor) -> None:
        if not isinstance(cursor_after, PretrainingCursor):
            raise TypeError("cursor_after must be a PretrainingCursor")
        with self._state_condition:
            if cursor_after == self._committed_cursor:
                return
            cursor_key = (
                cursor_after.epoch,
                cursor_after.shard_order_position,
                cursor_after.row_offset,
            )
            committed_key = (
                self._committed_cursor.epoch,
                self._committed_cursor.shard_order_position,
                self._committed_cursor.row_offset,
            )
            if cursor_key < committed_key:
                raise SMLDataError("pretraining cursor commit would regress")
            sequence = self._delivered.get(cursor_after)
            if sequence is None:
                raise SMLDataError(
                    "pretraining cursor was not delivered by this stream"
                )
            if sequence <= self._committed_sequence:
                raise SMLDataError("pretraining cursor commit would regress")
            self._committed_cursor = cursor_after
            self._committed_sequence = sequence
            self._delivered = {
                cursor: delivered_sequence
                for cursor, delivered_sequence in self._delivered.items()
                if delivered_sequence > sequence
            }

    def _close_open_resources(self) -> None:
        shard_arrays = self._shard_arrays
        self._shard_arrays = []
        mappings = self._mappings
        self._mappings = []
        files = self._shard_files
        self._shard_files = []
        root = self._artifact_root
        self._artifact_root = None
        _close_prepared_resources(root, files, mappings, shard_arrays)

    def _drain_consumer_items(self) -> None:
        pending = self._pending_envelope
        self._pending_envelope = None
        if pending is not None:
            pending.release()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, BatchEnvelope):
                item.release()

    def close(self) -> None:
        with self._state_condition:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._state_condition.wait()
                return
            self._closing = True

        self._stop.set()
        pool = self._pool
        if pool is not None:
            pool.stop()

        with self._consumer_lock:
            self._drain_consumer_items()

        producer = self._producer
        if producer is not None and producer is not threading.current_thread():
            producer.join()

        with self._consumer_lock:
            self._drain_consumer_items()

        with self._envelope_lock:
            owned = tuple(self._owned_envelopes.values())
        for envelope in owned:
            envelope.release()

        try:
            self._close_open_resources()
        finally:
            with self._state_condition:
                self._closed = True
                self._closing = False
                self._state_condition.notify_all()

    def __enter__(self) -> Self:
        with self._state_condition:
            if self._closed:
                raise SMLDataError("pretraining batch stream is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if isinstance(exception, BaseException):
                raise exception from cleanup_error
            raise


def _validated_token_array(
    token_range: Iterable[int], *, vocab_size: int | None
) -> np.ndarray:
    if isinstance(token_range, np.ndarray):
        array = np.asarray(token_range)
    else:
        try:
            array = np.asarray(tuple(token_range))
        except TypeError as error:
            raise TypeError("token ranges must be iterable") from error
    if array.ndim != 1:
        raise ValueError("each token range must be one-dimensional")
    if not array.size:
        return np.empty(0, dtype=_INT32)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("token IDs must use an integer dtype")

    minimum = int(array.min())
    maximum = int(array.max())
    if minimum < 0:
        raise ValueError("token IDs must be nonnegative")
    if vocab_size is not None and maximum >= vocab_size:
        raise ValueError("token IDs must be smaller than vocab_size")
    if maximum > np.iinfo(np.int32).max:
        raise ValueError("token IDs must fit int32")
    return np.ascontiguousarray(array, dtype=_INT32)


def pack_token_ranges(
    token_ranges: Iterable[Iterable[int]],
    *,
    sequence_length: int,
    vocab_size: int | None = None,
) -> Iterator[np.ndarray]:
    """Pack a lazy token stream into overlapping fixed-width int32 rows."""
    _require_plain_int(sequence_length, "sequence_length", minimum=1)
    if vocab_size is not None:
        _require_plain_int(vocab_size, "vocab_size", minimum=1)

    row_width = sequence_length + 1
    pending = np.empty(row_width, dtype=_INT32)
    cursor = 0
    for token_range in token_ranges:
        tokens = _validated_token_array(token_range, vocab_size=vocab_size)
        source_cursor = 0
        while source_cursor < tokens.size:
            copied = min(row_width - cursor, tokens.size - source_cursor)
            pending[cursor : cursor + copied] = tokens[
                source_cursor : source_cursor + copied
            ]
            cursor += copied
            source_cursor += copied
            if cursor == row_width:
                yield pending.copy()
                pending[0] = pending[-1]
                cursor = 1


def _windowed_row_shuffle(
    rows: Iterable[np.ndarray], *, window_rows: int, seed: int
) -> Iterator[np.ndarray]:
    """Shuffle complete rows inside deterministic, bounded logical windows."""
    _require_plain_int(window_rows, "window_rows", minimum=1)
    _require_plain_int(seed, "seed")
    generator = np.random.Generator(np.random.PCG64(seed))
    window: np.ndarray | None = None
    cursor = 0

    for row in rows:
        array = np.asarray(row)
        if array.ndim != 1:
            raise ValueError("pretraining rows must be one-dimensional")
        if window is None:
            window = np.empty((window_rows, array.shape[0]), dtype=_INT32)
        elif array.shape != (window.shape[1],):
            raise ValueError("pretraining row widths must match")
        window[cursor] = np.ascontiguousarray(array, dtype=_INT32)
        cursor += 1
        if cursor == window_rows:
            for index in generator.permutation(cursor):
                yield window[index].copy()
            cursor = 0

    if window is not None and cursor:
        for index in generator.permutation(cursor):
            yield window[index].copy()


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(
        logical_path=logical_path,
        identity=identity,
        byte_size=path.stat().st_size,
    )


def _verify_copied_bytes(reference: PayloadRef, payload: bytes) -> None:
    if len(payload) != reference.byte_size:
        raise SMLArtifactError(
            f"source tokenizer {reference.logical_path} byte size changed"
        )
    identity = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if identity != reference.identity:
        raise SMLArtifactError(
            f"source tokenizer {reference.logical_path} identity changed"
        )


def _copy_verified_tokenizer(
    tokenizer: LoadedTokenizer, destination: Path
) -> tuple[PayloadRef, PayloadRef]:
    with ArtifactRoot.open(tokenizer.path, writable=False) as source:
        with source.open_payload(_TOKENIZER_MANIFEST) as manifest_file:
            manifest_bytes = manifest_file.read()
        with source.open_payload(tokenizer.manifest.model.logical_path) as model_file:
            model_bytes = model_file.read()
        with source.open_payload(tokenizer.manifest.vocab.logical_path) as vocab_file:
            vocab_bytes = vocab_file.read()

    if manifest_bytes != canonical_json_bytes(tokenizer.manifest):
        raise SMLArtifactError("source tokenizer manifest changed after verification")
    _verify_copied_bytes(tokenizer.manifest.model, model_bytes)
    _verify_copied_bytes(tokenizer.manifest.vocab, vocab_bytes)

    destination.mkdir()
    (destination / _TOKENIZER_MANIFEST).write_bytes(manifest_bytes)
    model_path = destination / _TOKENIZER_MODEL
    vocab_path = destination / _TOKENIZER_VOCAB
    model_path.write_bytes(model_bytes)
    vocab_path.write_bytes(vocab_bytes)
    return (
        _payload_ref(model_path, f"tokenizer/{_TOKENIZER_MODEL}"),
        _payload_ref(vocab_path, f"tokenizer/{_TOKENIZER_VOCAB}"),
    )


def _corpus_projection(config: CorpusConfig) -> Mapping[str, object]:
    return {
        "filename_pattern": config.filename_pattern,
        "shuffle_files": config.shuffle_files,
        "file_order_seed": config.file_order_seed,
        "text_field": config.text_field,
        "min_text_bytes": config.min_text_bytes,
        "max_text_bytes": config.max_text_bytes,
        "max_rows_per_file": config.max_rows_per_file,
    }


def _encoded_text_ranges(
    texts: Iterable[str], tokenizer: LoadedTokenizer
) -> Iterator[Iterable[int]]:
    bos = tokenizer.manifest.bos_token_id
    eos = tokenizer.manifest.eos_token_id
    for text in texts:
        try:
            encoded = tokenizer.processor.encode(text)
        except Exception as error:
            raise SMLDataError("tokenizer failed while encoding source text") from error
        try:
            yield (bos, *encoded, eos)
        except TypeError as error:
            raise SMLDataError(
                "tokenizer produced a non-iterable token ID range"
            ) from error


def _write_shard(path: Path, rows: np.ndarray) -> None:
    array = np.ascontiguousarray(rows, dtype=_INT32)
    with path.open("xb") as destination:
        np.save(destination, array, allow_pickle=False)


def _write_shards(
    rows: Iterable[np.ndarray],
    directory: Path,
    *,
    row_width: int,
    shard_rows: int,
) -> tuple[tuple[Path, ...], tuple[int, ...]]:
    directory.mkdir()
    buffer = np.empty((shard_rows, row_width), dtype=_INT32)
    paths: list[Path] = []
    counts: list[int] = []
    cursor = 0

    def flush() -> None:
        nonlocal cursor
        path = directory / f"train-{len(paths):06d}.npy"
        _write_shard(path, buffer[:cursor])
        paths.append(path)
        counts.append(cursor)
        cursor = 0

    for row in rows:
        array = np.asarray(row)
        if array.shape != (row_width,):
            raise SMLDataError(f"pretraining row width mismatch: expected {row_width}")
        buffer[cursor] = array
        cursor += 1
        if cursor == shard_rows:
            flush()
    if cursor:
        flush()
    return tuple(paths), tuple(counts)


def _saved_rows(paths: Iterable[Path], *, row_width: int) -> Iterator[np.ndarray]:
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or array.shape[1] != row_width:
            raise SMLArtifactError(f"invalid prepared shard shape: {path.name}")
        if array.dtype != _INT32 or not array.flags.c_contiguous:
            raise SMLArtifactError(
                f"invalid prepared shard representation: {path.name}"
            )
        yield from array


def prepare_pretraining_bundle(
    config: PretrainingPreparationConfig,
    output: Path,
) -> PreparedDataBundle:
    """Prepare, verify, and atomically publish deterministic int32 NPY shards."""
    if not isinstance(config, PretrainingPreparationConfig):
        raise TypeError("config must be a PretrainingPreparationConfig")
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")

    tokenizer = load_tokenizer_bundle(
        config.tokenizer_bundle,
        VerificationLevel.FULL,
    )
    row_width = config.sequence_length + 1

    def build(private_path: Path) -> PretrainingDataManifest:
        tokenizer_model, tokenizer_vocab = _copy_verified_tokenizer(
            tokenizer, private_path / "tokenizer"
        )
        files = discover_corpus_files(config.corpus)
        texts = iter_filtered_texts(config.corpus, files)
        packed = pack_token_ranges(
            _encoded_text_ranges(texts, tokenizer),
            sequence_length=config.sequence_length,
            vocab_size=tokenizer.manifest.vocab_size,
        )
        shuffled = _windowed_row_shuffle(
            packed,
            window_rows=config.shuffle_window_rows,
            seed=config.seed,
        )
        try:
            shard_paths, shard_counts = _write_shards(
                shuffled,
                private_path / "shards",
                row_width=row_width,
                shard_rows=config.output_shard_rows,
            )
        except (TypeError, ValueError) as error:
            raise SMLDataError(f"invalid token IDs: {error}") from error
        row_count = sum(shard_counts)
        if row_count == 0:
            raise SMLDataError("no complete pretraining rows were produced")

        content_identity = row_content_identity(
            _saved_rows(shard_paths, row_width=row_width),
            row_count,
            row_width,
        )
        shard_refs = tuple(
            _payload_ref(path, f"shards/{path.name}") for path in shard_paths
        )
        source_summary = {
            "corpus": _corpus_projection(config.corpus),
            "ordered_files": tuple(path.name for path in files),
            "physical_lines_read": texts.physical_lines_read,
            "object_rows_read": texts.object_rows_read,
            "texts_used": texts.texts_used,
        }
        manifest = PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity=_PLACEHOLDER_IDENTITY,
            sequence_length=config.sequence_length,
            row_width=row_width,
            dtype="int32",
            shard_row_counts=shard_counts,
            shards=shard_refs,
            preparation_seed=config.seed,
            row_order_policy={
                "algorithm": config.shuffle_algorithm,
                "shuffle_window_rows": config.shuffle_window_rows,
                "output_shard_rows": config.output_shard_rows,
            },
            tokenizer_identity=tokenizer.manifest.identity,
            tokenizer_model=tokenizer_model,
            tokenizer_vocab=tokenizer_vocab,
            source_summary=source_summary,
            diagnostic_source_locator=str(config.corpus.input_root),
            row_content_identity=content_identity,
        )
        return replace(manifest, identity=manifest.recompute_identity())

    published = publish_immutable_bundle(output, build)
    return PreparedDataBundle(
        path=published.path,
        manifest=published.manifest,
        verification=published.verification,
    )


def build_benchmark_workload(metric: str, canonical_workload: object) -> object:
    """Build the production prepared-data path for the pinned benchmark ABI."""
    from sml.data._pretraining_benchmark import (
        build_prepared_data_benchmark_workload,
    )

    return build_prepared_data_benchmark_workload(metric, canonical_workload)


__all__ = [
    "WINDOWED_ROW_SHUFFLE_V1",
    "BatchEnvelope",
    "PreparedDataBundle",
    "PreparedDataPreflight",
    "PretrainingBatchStream",
    "PretrainingCursor",
    "PretrainingPreparationConfig",
    "build_benchmark_workload",
    "canonicalize_pretraining_cursor",
    "pack_token_ranges",
    "preflight_pretraining_bundle",
    "prepare_pretraining_bundle",
]
