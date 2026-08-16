"""Lazy discovery and filtering for compressed tokenizer-training corpora."""

from __future__ import annotations

import io
import itertools
import json
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

DEFAULT_FILENAME_PATTERN = r".*-00[0-9][0-9]\.jsonl\.zst\Z"
DEFAULT_FILE_ORDER_SEED = 42
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MIN_TEXT_BYTES = 100
DEFAULT_MAX_TEXT_BYTES = 16_384
DEFAULT_MAX_ROWS_PER_FILE = 8_192

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
    if config.shuffle_files:
        random.Random(config.file_order_seed).shuffle(files)
    return tuple(files)


def _normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\x00", " ")).strip()


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

    @property
    def rows_read(self) -> int:
        """Compatibility name for JSON object rows presented to text filtering."""
        return self.object_rows_read

    def __iter__(self) -> Iterator[str]:
        config = self.config
        for path in self.files:
            try:
                with (
                    path.open("rb") as compressed_stream,
                    zstd.ZstdDecompressor().stream_reader(
                        compressed_stream
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
                            raise ValueError(
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
                raise RuntimeError(f"zstd failed for {path}: {error}") from error


def iter_filtered_texts(
    config: CorpusConfig,
    files: Sequence[Path] | None = None,
) -> FilteredTexts:
    """Build a lazy filtered-text iterable without reading corpus payloads."""
    if not isinstance(config, CorpusConfig):
        raise TypeError("config must be a CorpusConfig")
    selected_files = discover_corpus_files(config) if files is None else tuple(files)
    return FilteredTexts(config, selected_files)


__all__ = [
    "CorpusConfig",
    "FilteredTexts",
    "discover_corpus_files",
    "iter_filtered_texts",
]
