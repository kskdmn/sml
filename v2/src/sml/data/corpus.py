"""Lazy discovery, normalization, and sampling for compressed text corpora."""

from __future__ import annotations

import hashlib
import heapq
import io
import itertools
import json
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import zstandard as zstd

from sml.errors import SMLDataError

DEFAULT_FILENAME_PATTERN = r".*\.jsonl\.zst\Z"
DEFAULT_FILE_ORDER_SEED = 42
DEFAULT_MAX_FILES = 100
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MIN_TEXT_BYTES = 100
DEFAULT_MAX_TEXT_BYTES = 16_384
DEFAULT_MAX_ROWS_PER_FILE = 8_192

# Persist only corpus semantics, excluding the relocatable source directory.
CORPUS_METADATA_FIELDS = (
    "filename_pattern",
    "shuffle_files",
    "file_order_seed",
    "text_field",
    "min_text_bytes",
    "max_text_bytes",
    "max_rows_per_file",
    "max_files",
)

_WHITESPACE = re.compile(r"\s+")


def _require_plain_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """All source-ordering and text-filtering inputs that affect training."""

    input_root: Path
    filename_pattern: str = DEFAULT_FILENAME_PATTERN
    shuffle_files: bool = True
    file_order_seed: int = DEFAULT_FILE_ORDER_SEED
    text_field: str = DEFAULT_TEXT_FIELD
    min_text_bytes: int = DEFAULT_MIN_TEXT_BYTES
    max_text_bytes: int | None = DEFAULT_MAX_TEXT_BYTES
    max_rows_per_file: int | None = DEFAULT_MAX_ROWS_PER_FILE
    max_files: int | None = DEFAULT_MAX_FILES

    def __post_init__(self) -> None:
        if not isinstance(self.input_root, Path):
            raise TypeError("input_root must be a Path")
        object.__setattr__(self, "input_root", self.input_root.expanduser())
        if not isinstance(self.filename_pattern, str):
            raise TypeError("filename_pattern must be a string")
        try:
            re.compile(self.filename_pattern)
        except re.error as error:
            raise ValueError(f"invalid filename_pattern: {error}") from error
        if not isinstance(self.shuffle_files, bool):
            raise TypeError("shuffle_files must be a bool")
        _require_plain_int(self.file_order_seed, "file_order_seed")
        if not isinstance(self.text_field, str):
            raise TypeError("text_field must be a string")
        if not self.text_field:
            raise ValueError("text_field must not be empty")
        _require_plain_int(self.min_text_bytes, "min_text_bytes", minimum=0)
        if self.max_text_bytes is not None:
            _require_plain_int(self.max_text_bytes, "max_text_bytes", minimum=0)
            if self.max_text_bytes < self.min_text_bytes:
                raise ValueError("max_text_bytes must be at least min_text_bytes")
        if self.max_rows_per_file is not None:
            _require_plain_int(
                self.max_rows_per_file,
                "max_rows_per_file",
                minimum=1,
            )
        if self.max_files is not None:
            _require_plain_int(self.max_files, "max_files", minimum=1)


def corpus_metadata(config: CorpusConfig) -> dict[str, object]:
    """Return the explicit corpus fields used in tokenizer and data identities."""
    return {name: getattr(config, name) for name in CORPUS_METADATA_FIELDS}


@dataclass(frozen=True, slots=True)
class CorpusSamplingConfig:
    """Bound a deterministic sample of the normalized, scanned source texts."""

    max_documents: int | None = 100_000
    max_bytes: int | None = 128 * 1024 * 1024
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("max_documents", "max_bytes"):
            value = getattr(self, name)
            if value is not None:
                _require_plain_int(value, name, minimum=1)
        if self.max_documents is None and self.max_bytes is None:
            raise ValueError("at least one sampling limit must be set")
        _require_plain_int(self.seed, "seed")


def discover_corpus_files(config: CorpusConfig) -> tuple[Path, ...]:
    """Return matching direct children in an isolated deterministic file order."""
    if not isinstance(config, CorpusConfig):
        raise TypeError("config must be a CorpusConfig")
    root = config.input_root
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    pattern = re.compile(config.filename_pattern)
    files = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and pattern.fullmatch(path.name) is not None
        ),
        key=lambda path: path.name,
    )
    if config.max_files is not None and len(files) > config.max_files:
        files = sorted(
            random.Random(config.file_order_seed).sample(files, config.max_files),
            key=lambda path: path.name,
        )
    if config.shuffle_files:
        random.Random(config.file_order_seed).shuffle(files)
    return tuple(files)


def _normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\x00", " ")).strip()


class _CompleteZstdReader(io.RawIOBase):
    """Stream concatenated frames, checking the last frame at physical EOF."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._decoder = zstd.ZstdDecompressor().decompressobj()
        self._compressed = b""
        self._output = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        while not self._output:
            # Bound the output of one decompress() call even for highly
            # compressible blocks; the streaming object exposes no output limit.
            compressed = self._compressed or self._source.read(1_024)
            if not compressed:
                if not self._decoder.eof:
                    raise zstd.ZstdError("incomplete zstd frame")
                return 0
            if self._decoder.eof:
                self._decoder = zstd.ZstdDecompressor().decompressobj()
            self._output = memoryview(self._decoder.decompress(compressed))
            self._compressed = self._decoder.unused_data
        size = min(len(buffer), len(self._output))
        buffer[:size] = self._output[:size]
        self._output = self._output[size:]
        return size


class FilteredTexts:
    """Single-pass-style lazy iterable with deterministic diagnostic counters."""

    def __init__(self, config: CorpusConfig, files: Sequence[Path]) -> None:
        self.config = config
        self.files = tuple(files)
        if not all(isinstance(path, Path) for path in self.files):
            raise TypeError("files must contain Path values")
        self.physical_lines_read = 0
        self.object_rows_read = 0
        self.texts_used = 0

    def __iter__(self) -> Iterator[str]:
        config = self.config
        for path in self.files:
            try:
                with (
                    path.open("rb") as compressed_stream,
                    io.BufferedReader(
                        _CompleteZstdReader(compressed_stream)
                    ) as decompressed_stream,
                    io.TextIOWrapper(
                        decompressed_stream,
                        encoding="utf-8",
                        errors="replace",
                    ) as text_stream,
                ):
                    lines = (
                        text_stream
                        if config.max_rows_per_file is None
                        else itertools.islice(text_stream, config.max_rows_per_file)
                    )
                    for line_number, line in enumerate(lines, start=1):
                        self.physical_lines_read += 1
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            row = json.loads(stripped)
                        except json.JSONDecodeError as error:
                            raise SMLDataError(
                                f"Invalid JSON in {path} at line {line_number}"
                            ) from error
                        if not isinstance(row, dict):
                            continue
                        self.object_rows_read += 1
                        value = row.get(config.text_field)
                        if not isinstance(value, str):
                            continue
                        text = _normalize_text(value)
                        byte_length = len(text.encode("utf-8"))
                        if byte_length < config.min_text_bytes:
                            continue
                        if (
                            config.max_text_bytes is not None
                            and byte_length > config.max_text_bytes
                        ):
                            continue
                        self.texts_used += 1
                        yield text
            except zstd.ZstdError as error:
                raise SMLDataError(f"zstd failed for {path}: {error}") from error
            except OSError as error:
                raise SMLDataError(
                    f"Could not read corpus file {path}: {error}"
                ) from error


def iter_filtered_texts(
    config: CorpusConfig,
    files: Sequence[Path] | None = None,
) -> FilteredTexts:
    """Build a lazy filtered-text iterable without reading corpus payloads."""
    if not isinstance(config, CorpusConfig):
        raise TypeError("config must be a CorpusConfig")
    selected_files = discover_corpus_files(config) if files is None else tuple(files)
    return FilteredTexts(config, selected_files)


def sample_texts(
    texts: Iterable[str],
    sampling: CorpusSamplingConfig,
) -> Iterator[str]:
    """Yield a bounded, unique sample in the selected texts' first-seen order."""
    if not isinstance(sampling, CorpusSamplingConfig):
        raise TypeError("sampling must be a CorpusSamplingConfig")
    # Retain UTF-8 bytes instead of Unicode strings so the byte budget also
    # bounds the sample's text payload in memory. The set shares those bytes
    # with the heap and checks actual equality, not only hash equality.
    heap: list[tuple[int, int, bytes]] = []
    retained: set[bytes] = set()
    retained_bytes = 0
    rejected_priority: int | None = None
    seed_prefix = str(sampling.seed).encode("ascii") + b"\x00"
    for sequence, text in enumerate(texts):
        encoded = text.encode("utf-8")
        if sampling.max_bytes is not None and len(encoded) > sampling.max_bytes:
            continue
        if encoded in retained:
            continue
        priority = int.from_bytes(hashlib.sha256(seed_prefix + encoded).digest())
        if rejected_priority is not None and priority >= rejected_priority:
            continue
        heapq.heappush(heap, (-priority, sequence, encoded))
        retained.add(encoded)
        retained_bytes += len(encoded)
        while (
            sampling.max_documents is not None and len(heap) > sampling.max_documents
        ) or (sampling.max_bytes is not None and retained_bytes > sampling.max_bytes):
            negative_priority, _sequence, evicted = heapq.heappop(heap)
            retained.remove(evicted)
            retained_bytes -= len(evicted)
            # Keep the random-priority prefix even when byte eviction leaves
            # spare capacity. A rejected document, including its duplicates,
            # must never reenter merely because a later copy fits that gap.
            rejected_priority = -negative_priority

    # Selection is independent of source traversal; preserve first-seen order
    # for the retained texts, including the order of small unsampled corpora.
    retained.clear()
    heap.sort(key=lambda entry: entry[1])
    for _priority, _sequence, encoded in heap:
        yield encoded.decode("utf-8")


def iter_sampled_texts(
    config: CorpusConfig,
    sampling: CorpusSamplingConfig,
    files: Sequence[Path] | None = None,
) -> Iterator[str]:
    """Sample unique normalized texts within document and UTF-8 byte limits.

    Every selected file is scanned up to its physical row limit before yielding
    the sample. File and row caps control decompression work independently of
    the sample size; the sample represents those scanned prefixes only.
    """
    if not isinstance(sampling, CorpusSamplingConfig):
        raise TypeError("sampling must be a CorpusSamplingConfig")
    return sample_texts(iter_filtered_texts(config, files), sampling)


__all__ = [
    "CORPUS_METADATA_FIELDS",
    "CorpusConfig",
    "CorpusSamplingConfig",
    "FilteredTexts",
    "corpus_metadata",
    "discover_corpus_files",
    "iter_filtered_texts",
    "iter_sampled_texts",
    "sample_texts",
]
