from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

import sentencepiece as spm

from config import PROJECT_DIR, resolve_path
import tokenizer

SUCCESS_RETURN_CODE = 0
INPUT_DIR = Path("~/Documents/data-common_pile/")
TOKENIZER_SHARD_NAME_REGEX = r".*-00[0-9][0-9]\.jsonl\.zst\Z"
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"
MAX_ROWS_PER_FILE = 32_768
RANDOM_SEED = 42
NUM_THREADS = 8
INPUT_SENTENCE_SIZE = 524_288
SELF_TEST_SAMPLE_SIZE = 0
SHUFFLE_INPUT_SENTENCE = True
TRAIN_EXTREMELY_LARGE_CORPUS = False  # This is only for Unigram models.


class TrainingResult(NamedTuple):
    model_path: Path
    vocab_path: Path
    input_file_count: int
    rows_read: int
    texts_used: int


def require_non_empty_iterator(iterator: Iterator[str]) -> Iterator[str]:
    """
    SentencePiece fails obscurely on empty input, so raise a project-level error before
    training starts.
    """
    try:
        first_text = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            "No usable text rows found. Check TEXT_COLUMN and length constants."
        ) from exc

    return itertools.chain((first_text,), iterator)


def train_tokenizer(
    tokenizer_model_path: Path = DEFAULT_TOKENIZER_MODEL_PATH,
) -> TrainingResult:
    """
    Feed SentencePiece from a lazy filtered iterator so compressed shards do not need to
    be materialized.
    """
    random.seed(RANDOM_SEED)

    tokenizer_model_path = resolve_path(tokenizer_model_path)
    output_dir = tokenizer_model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = tokenizer.discover_input_files(
        INPUT_DIR,
        TOKENIZER_SHARD_NAME_REGEX,
    )
    if not input_files:
        raise FileNotFoundError(
            f"No supported input files found in {resolve_path(INPUT_DIR)}"
        )

    text_iterable = tokenizer.FilteredTextIterable(
        input_files,
        max_rows_per_file=MAX_ROWS_PER_FILE,
    )
    sentence_iterator = require_non_empty_iterator(iter(text_iterable))

    model_prefix = tokenizer_model_path.with_suffix("")
    trainer_kwargs = dict(
        sentence_iterator=sentence_iterator,
        model_prefix=str(model_prefix),
        vocab_size=tokenizer.VOCAB_SIZE,
        model_type=tokenizer.MODEL_TYPE,
        character_coverage=tokenizer.CHARACTER_COVERAGE,
        byte_fallback=tokenizer.BYTE_FALLBACK,
        num_threads=NUM_THREADS,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
        self_test_sample_size=SELF_TEST_SAMPLE_SIZE,
        hard_vocab_limit=tokenizer.HARD_VOCAB_LIMIT,
        train_extremely_large_corpus=TRAIN_EXTREMELY_LARGE_CORPUS,
        unk_id=tokenizer.UNK_ID,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
        pad_id=tokenizer.PAD_ID,
        user_defined_symbols=list(tokenizer.CONVERSATION_SPECIAL_TOKENS),
    )
    if tokenizer.MAX_SENTENCE_LENGTH is not None:
        trainer_kwargs["max_sentence_length"] = tokenizer.MAX_SENTENCE_LENGTH

    spm.SentencePieceTrainer.train(**trainer_kwargs)

    return TrainingResult(
        model_path=model_prefix.with_suffix(".model"),
        vocab_path=model_prefix.with_suffix(".vocab"),
        input_file_count=len(input_files),
        rows_read=text_iterable.rows_read,
        texts_used=text_iterable.texts_used,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the SML SentencePiece tokenizer.")
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_TOKENIZER_MODEL_PATH,
        help=f"SentencePiece model path (default: {DEFAULT_TOKENIZER_MODEL_PATH})",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print generated artifacts and corpus counters for the tokenizer training CLI."""
    args = parse_args(argv)
    result = train_tokenizer(tokenizer_model_path=args.tokenizer_model)
    print(f"Model: {result.model_path}")
    print(f"Vocab: {result.vocab_path}")
    print(f"Input files: {result.input_file_count}")
    print(f"Rows read: {result.rows_read}")
    print(f"Texts used: {result.texts_used}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
