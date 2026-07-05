"""
Print the first lines from a compressed JSONL file.

Usage:
    uv run python scripts/peek_jsonl_zst.py path/to/file.jsonl.zst
    uv run python scripts/peek_jsonl_zst.py path/to/file.jsonl.zst --lines 10
"""

import argparse
import io
import sys
from itertools import islice
from pathlib import Path

import zstandard as zstd


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return parsed


def print_first_lines(jsonl_zst_path: Path, line_count: int) -> None:
    decompressor = zstd.ZstdDecompressor()
    with jsonl_zst_path.open("rb") as compressed:
        with decompressor.stream_reader(compressed) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in islice(text_reader, line_count):
                sys.stdout.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompress a .jsonl.zst file and print its first few lines.",
    )
    parser.add_argument("jsonl_zst_path", type=Path, help="Path to the .jsonl.zst file.")
    parser.add_argument(
        "-n",
        "--lines",
        type=positive_int,
        default=5,
        help="Number of lines to print. Defaults to 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_first_lines(args.jsonl_zst_path, args.lines)


if __name__ == "__main__":
    main()
