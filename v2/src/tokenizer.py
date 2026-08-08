from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from utils import MAX_TEXT_LENGTH, TEXT_COLUMN, filter_text, iter_jsonl_records

VOCAB_SIZE = 28_672
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = (
    0.9995  # 1.0 for small character sets; 0.9995 for Pile/web-like datasets.
)
BYTE_FALLBACK = True
HARD_VOCAB_LIMIT = True
MAX_SENTENCE_LENGTH: int | None = MAX_TEXT_LENGTH
CONVERSATION_SPECIAL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)


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
        self.rows_read = 0
        self.texts_used = 0

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
                self.rows_read += 1
                text = filter_text(row.get(TEXT_COLUMN))
                if text is None:
                    continue

                self.texts_used += 1
                yield text
