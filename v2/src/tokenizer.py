from __future__ import annotations

import io
import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import zstandard as zstd

from config import resolve_path

MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH: int | None = None
TEXT_COLUMN = "text"
TEXT_ENCODING = "utf-8"
TEXT_DECODE_ERRORS = "replace"
HIDDEN_FILE_PREFIX = "."
NULL_CHARACTER = "\x00"
FIRST_LINE_NUMBER = 1
NO_ROWS = 0
ROW_INCREMENT = 1
VOCAB_SIZE = 28_672
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 0.9995  # 1.0 for small character sets; 0.9995 for Pile/web-like datasets.
BYTE_FALLBACK = True
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
HARD_VOCAB_LIMIT = True
MAX_SENTENCE_LENGTH: int | None = MAX_TEXT_LENGTH
CONVERSATION_SPECIAL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)

WHITESPACE_PATTERN = re.compile(r"\s+")


class FilteredTextIterable:
    def __init__(
        self,
        input_files: Sequence[Path],
        max_rows_per_file: int | None = None,
    ) -> None:
        """
        Counters live on the iterable so callers can report rows read versus texts kept
        after filtering.
        """
        self.input_files = input_files
        self.max_rows_per_file = max_rows_per_file
        self.rows_read = NO_ROWS
        self.texts_used = NO_ROWS

    def __iter__(self) -> Iterator[str]:
        """
        Each shard is streamed once; rows_read counts JSON objects while texts_used
        counts only rows that survive text filters.
        """
        for input_file in self.input_files:
            for row, _line_number in iter_jsonl_records(
                input_file,
                self.max_rows_per_file,
            ):
                self.rows_read += ROW_INCREMENT
                text = filter_text(row.get(TEXT_COLUMN))
                if text is None:
                    continue

                self.texts_used += ROW_INCREMENT
                yield text


def discover_input_files(input_dir: Path, file_name_regex: str) -> tuple[Path, ...]:
    """
    Use the supplied shard naming pattern and skip hidden local metadata files.
    """
    root = resolve_path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    input_file_name_pattern = re.compile(file_name_regex)
    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(HIDDEN_FILE_PREFIX)
        and input_file_name_pattern.fullmatch(path.name) is not None
    ]
    return tuple(sorted(files, key=lambda path: path.name))


def iter_jsonl_records(
    path: Path,
    max_rows_per_file: int | None,
    start_after_line: int | None = None,
) -> Iterator[tuple[dict[str, object], int]]:
    """
    Blank lines and non-object JSON are ignored; malformed JSON includes the compressed
    file line number.
    """
    for line_number, line in iter_zstd_jsonl_lines(
        path,
        max_rows_per_file,
        start_after_line=start_after_line,
    ):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

        if isinstance(row, dict):
            yield row, line_number


def iter_zstd_jsonl_lines(
    path: Path,
    max_rows_per_file: int | None,
    start_after_line: int | None = None,
) -> Iterator[tuple[int, str]]:
    """
    Decode zstd streams as replacement-tolerant UTF-8 without materializing shards.
    """
    try:
        with path.open("rb") as compressed_stream:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(compressed_stream) as zstd_stream:
                with io.TextIOWrapper(
                    zstd_stream,
                    encoding=TEXT_ENCODING,
                    errors=TEXT_DECODE_ERRORS,
                ) as text_stream:
                    yield from enumerate_limited_lines(
                        text_stream,
                        max_rows_per_file,
                        start_after_line=start_after_line,
                    )
    except zstd.ZstdError as exc:
        raise RuntimeError(f"zstd failed for {path}: {exc}") from exc


def enumerate_limited_lines(
    stream: Iterable[str],
    max_rows_per_file: int | None,
    start_after_line: int | None = None,
) -> Iterator[tuple[int, str]]:
    """
    Line numbers stay one-based after applying the row cap for readable JSON errors.
    """
    limited_stream = (
        stream
        if max_rows_per_file is None
        else itertools.islice(stream, max_rows_per_file)
    )
    for line_number, line in enumerate(limited_stream, start=FIRST_LINE_NUMBER):
        if start_after_line is not None and line_number <= start_after_line:
            continue
        yield line_number, line


def filter_text(value: object) -> str | None:
    """
    Reject non-strings, too-short UTF-8 byte rows, and optionally too-long rows before
    SentencePiece sees them.
    """
    if not isinstance(value, str):
        return None

    text = normalize_text(value)
    text_byte_length = len(text.encode(TEXT_ENCODING))
    if text_byte_length < MIN_TEXT_LENGTH:
        return None
    if MAX_TEXT_LENGTH is not None and text_byte_length > MAX_TEXT_LENGTH:
        return None

    return text


def normalize_text(text: str) -> str:
    """
    Convert null bytes to spaces before whitespace collapse to avoid embedded
    terminators in tokenizer input.
    """
    text = text.replace(NULL_CHARACTER, " ")
    return WHITESPACE_PATTERN.sub(" ", text).strip()
