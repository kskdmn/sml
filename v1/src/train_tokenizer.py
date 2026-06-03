from __future__ import annotations

import io
import itertools
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Sequence

import sentencepiece as spm
import zstandard as zstd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
INPUT_DIR = Path("~/Documents/data-common_pile/")

VOCAB_SIZE = 49_152
MAX_ROWS_PER_FILE = 10_000
MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 2_000
RANDOM_SEED = 42
NUM_THREADS = 8
INPUT_SENTENCE_SIZE = 0
SELF_TEST_SAMPLE_SIZE = 0
CHARACTER_COVERAGE = 1.0
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
SUCCESS_RETURN_CODE = 0
FIRST_LINE_NUMBER = 1
NO_ROWS = 0
ROW_INCREMENT = 1

MODEL_PREFIX_NAME = "bpe_tokenizer"
MODEL_TYPE = "bpe"
TEXT_COLUMN = "text"
TEXT_ENCODING = "utf-8"
TEXT_DECODE_ERRORS = "replace"
HIDDEN_FILE_PREFIX = "."
INPUT_FILE_NAME_PATTERN = re.compile(r".*-00[01][0-9]\.jsonl\.zst\Z")
NULL_CHARACTER = "\x00"

SHUFFLE_INPUT_SENTENCE = False
HARD_VOCAB_LIMIT = True
TRAIN_EXTREMELY_LARGE_CORPUS = True
MAX_SENTENCE_LENGTH = MAX_TEXT_LENGTH

WHITESPACE_PATTERN = re.compile(r"\s+")


class TrainingResult(NamedTuple):
    model_path: Path
    vocab_path: Path
    input_file_count: int
    rows_read: int
    texts_used: int


class FilteredTextIterable:
    def __init__(
        self,
        input_files: Sequence[Path],
        max_rows_per_file: int = MAX_ROWS_PER_FILE,
    ) -> None:
        self.input_files = input_files
        self.max_rows_per_file = max_rows_per_file
        self.rows_read = NO_ROWS
        self.texts_used = NO_ROWS

    def __iter__(self) -> Iterator[str]:
        for input_file in self.input_files:
            for row in iter_jsonl_records(input_file, self.max_rows_per_file):
                self.rows_read += ROW_INCREMENT
                text = filter_text(row.get(TEXT_COLUMN))
                if text is None:
                    continue

                self.texts_used += ROW_INCREMENT
                yield text


def resolve_path(path: Path) -> Path:
    return path.expanduser()


def discover_input_files(input_dir: Path = INPUT_DIR) -> tuple[Path, ...]:
    root = resolve_path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(HIDDEN_FILE_PREFIX)
        and INPUT_FILE_NAME_PATTERN.fullmatch(path.name) is not None
    ]
    return tuple(sorted(files, key=lambda path: path.name))


def iter_jsonl_records(path: Path, max_rows_per_file: int) -> Iterator[dict[str, object]]:
    for line_number, line in iter_jsonl_lines(path, max_rows_per_file):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

        if isinstance(row, dict):
            yield row


def iter_jsonl_lines(path: Path, max_rows_per_file: int) -> Iterator[tuple[int, str]]:
    if INPUT_FILE_NAME_PATTERN.fullmatch(path.name) is None:
        raise ValueError(f"Unsupported input file: {path}")

    yield from iter_zstd_jsonl_lines(path, max_rows_per_file)


def iter_zstd_jsonl_lines(path: Path, max_rows_per_file: int) -> Iterator[tuple[int, str]]:
    try:
        with path.open("rb") as compressed_stream:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(compressed_stream) as zstd_stream:
                with io.TextIOWrapper(
                    zstd_stream,
                    encoding=TEXT_ENCODING,
                    errors=TEXT_DECODE_ERRORS,
                ) as text_stream:
                    yield from enumerate_limited_lines(text_stream, max_rows_per_file)
    except zstd.ZstdError as exc:
        raise RuntimeError(f"zstd failed for {path}: {exc}") from exc


def enumerate_limited_lines(
    stream: Iterable[str],
    max_rows_per_file: int,
) -> Iterator[tuple[int, str]]:
    limited_stream = itertools.islice(stream, max_rows_per_file)
    yield from enumerate(limited_stream, start=FIRST_LINE_NUMBER)


def filter_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    text = normalize_text(value)
    if (
        len(text) < MIN_TEXT_LENGTH
        or len(text.encode(TEXT_ENCODING)) > MAX_TEXT_LENGTH
    ):
        return None

    return text


def normalize_text(text: str) -> str:
    text = text.replace(NULL_CHARACTER, " ")
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def require_non_empty_iterator(iterator: Iterator[str]) -> Iterator[str]:
    try:
        first_text = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            "No usable text rows found. Check TEXT_COLUMN and length constants."
        ) from exc

    return itertools.chain((first_text,), iterator)


def train_tokenizer() -> TrainingResult:
    random.seed(RANDOM_SEED)

    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(f"No supported input files found in {resolve_path(INPUT_DIR)}")

    text_iterable = FilteredTextIterable(input_files)
    sentence_iterator = require_non_empty_iterator(iter(text_iterable))

    model_prefix = output_dir / MODEL_PREFIX_NAME
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator,
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        num_threads=NUM_THREADS,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
        self_test_sample_size=SELF_TEST_SAMPLE_SIZE,
        max_sentence_length=MAX_SENTENCE_LENGTH,
        hard_vocab_limit=HARD_VOCAB_LIMIT,
        train_extremely_large_corpus=TRAIN_EXTREMELY_LARGE_CORPUS,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    return TrainingResult(
        model_path=model_prefix.with_suffix(".model"),
        vocab_path=model_prefix.with_suffix(".vocab"),
        input_file_count=len(input_files),
        rows_read=text_iterable.rows_read,
        texts_used=text_iterable.texts_used,
    )


def main() -> int:
    result = train_tokenizer()
    print(f"Model: {result.model_path}")
    print(f"Vocab: {result.vocab_path}")
    print(f"Input files: {result.input_file_count}")
    print(f"Rows read: {result.rows_read}")
    print(f"Texts used: {result.texts_used}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
