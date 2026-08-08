"""
Shared helpers for the SML tokenizer, training, fine-tuning, and inference scripts.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import mlx.core as mx
import sentencepiece as spm
import zstandard as zstd
from config import resolve_path

MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH: int | None = 16_384
TEXT_COLUMN = "text"
TEXT_ENCODING = "utf-8"
TEXT_DECODE_ERRORS = "replace"
HIDDEN_FILE_PREFIX = "."
NULL_CHARACTER = "\x00"
WHITESPACE_PATTERN = re.compile(r"\s+")


def shuffle_input_files(input_files: Iterable[Path], seed: int) -> tuple[Path, ...]:
    shuffled_files = list(input_files)
    random.Random(seed).shuffle(shuffled_files)
    return tuple(shuffled_files)


def discover_input_files(input_dir: Path, file_name_regex: str) -> tuple[Path, ...]:
    """
    Match the regex against file names rather than paths, and skip hidden files such as
    local filesystem metadata.
    """
    root = resolve_path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    pattern = re.compile(file_name_regex)
    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(HIDDEN_FILE_PREFIX)
        and pattern.fullmatch(path.name) is not None
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
        decompressor = zstd.ZstdDecompressor()
        with (
            path.open("rb") as compressed_stream,
            decompressor.stream_reader(compressed_stream) as zstd_stream,
            io.TextIOWrapper(
                zstd_stream,
                encoding=TEXT_ENCODING,
                errors=TEXT_DECODE_ERRORS,
            ) as text_stream,
        ):
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
    for line_number, line in enumerate(limited_stream, start=1):
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


def load_tokenizer(path: Path) -> spm.SentencePieceProcessor:
    """
    Fail before training starts if the configured SentencePiece model path is missing.
    """
    model_path = resolve_path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model does not exist: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def get_special_token_id(
    tokenizer: object,
    name: str,
    fallback: int | None,
) -> int | None:
    """
    SentencePiece reports negative IDs for disabled special tokens, so config fallbacks
    are used in that case.
    """
    value = getattr(tokenizer, name, None)
    if callable(value):
        value = value()
    if value is None or value < 0:
        return fallback
    return int(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    mx.random.seed(seed)


def json_ready(value: object) -> object:
    """
    Convert checkpoint metadata values into JSON-serializable equivalents.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_loss_fn(model):
    """
    Return the causal-LM training loss closure used with ``nn.value_and_grad``.
    """

    def loss_fn(input_ids, labels):
        output = model(input_ids, labels=labels)
        if output.loss is None:
            raise RuntimeError("Model did not return a training loss")
        return output.loss

    return loss_fn


def lr_lambda(
    step: int,
    total_steps: int | None,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    Warm up linearly, then either hold constant when no horizon is known or cosine-decay
    to a configured floor.
    """
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps is None:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return max(min_lr_ratio, cosine)


def build_lr_schedule(
    learning_rate: float,
    total_steps: int | None,
    warmup_steps: int,
    min_lr_ratio: float,
):
    """
    Adapt ``lr_lambda`` for MLX optimizers, which pass the step counter as a scalar
    ``mx.array`` and require an ``mx.array`` learning rate back.
    """

    def schedule(step) -> mx.array:
        multiplier = lr_lambda(int(step), total_steps, warmup_steps, min_lr_ratio)
        return mx.array(learning_rate * multiplier)

    return schedule
