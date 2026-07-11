from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Iterator, NamedTuple

import sentencepiece as spm

from config import INPUT_DIR, OUTPUT_DIR, SUCCESS_RETURN_CODE, resolve_path
import tokenizer

VOCAB_SIZE = 28_672
RANDOM_SEED = 42
NUM_THREADS = 8
INPUT_SENTENCE_SIZE = 524_288
SELF_TEST_SAMPLE_SIZE = 0
CHARACTER_COVERAGE = 0.9995  # 1.0 for small character sets (e.g. English clean datasets), 0.9995 for Pile/web-like datasets
BYTE_FALLBACK = True
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3
SHUFFLE_INPUT_SENTENCE = True
HARD_VOCAB_LIMIT = True
TRAIN_EXTREMELY_LARGE_CORPUS = False  # This is only for Unigram models.

MODEL_PREFIX_NAME = "bpe_tokenizer"
MODEL_TYPE = "bpe"
MAX_SENTENCE_LENGTH: int | None = tokenizer.MAX_TEXT_LENGTH


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


def train_tokenizer() -> TrainingResult:
    """
    Feed SentencePiece from a lazy filtered iterator so compressed shards do not need to
    be materialized.
    """
    random.seed(RANDOM_SEED)

    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = tokenizer.discover_input_files(INPUT_DIR)
    if not input_files:
        raise FileNotFoundError(
            f"No supported input files found in {resolve_path(INPUT_DIR)}"
        )

    text_iterable = tokenizer.FilteredTextIterable(input_files)
    sentence_iterator = require_non_empty_iterator(iter(text_iterable))

    model_prefix = output_dir / MODEL_PREFIX_NAME
    trainer_kwargs = dict(
        sentence_iterator=sentence_iterator,
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        byte_fallback=BYTE_FALLBACK,
        num_threads=NUM_THREADS,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=SHUFFLE_INPUT_SENTENCE,
        self_test_sample_size=SELF_TEST_SAMPLE_SIZE,
        hard_vocab_limit=HARD_VOCAB_LIMIT,
        train_extremely_large_corpus=TRAIN_EXTREMELY_LARGE_CORPUS,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
        user_defined_symbols=list(tokenizer.CONVERSATION_SPECIAL_TOKENS),
    )
    if MAX_SENTENCE_LENGTH is not None:
        trainer_kwargs["max_sentence_length"] = MAX_SENTENCE_LENGTH

    spm.SentencePieceTrainer.train(**trainer_kwargs)

    return TrainingResult(
        model_path=model_prefix.with_suffix(".model"),
        vocab_path=model_prefix.with_suffix(".vocab"),
        input_file_count=len(input_files),
        rows_read=text_iterable.rows_read,
        texts_used=text_iterable.texts_used,
    )


def main() -> int:
    """Print generated artifacts and corpus counters for the tokenizer training CLI."""
    result = train_tokenizer()
    print(f"Model: {result.model_path}")
    print(f"Vocab: {result.vocab_path}")
    print(f"Input files: {result.input_file_count}")
    print(f"Rows read: {result.rows_read}")
    print(f"Texts used: {result.texts_used}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
