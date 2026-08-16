"""Deterministic preparation of immutable, mmap-ready pretraining data."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sml.artifacts.checkpoint import publish_immutable_bundle
from sml.artifacts.manifest import (
    ArtifactRoot,
    PayloadRef,
    PretrainingDataManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
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


__all__ = [
    "WINDOWED_ROW_SHUFFLE_V1",
    "PreparedDataBundle",
    "PretrainingPreparationConfig",
    "pack_token_ranges",
    "prepare_pretraining_bundle",
]
