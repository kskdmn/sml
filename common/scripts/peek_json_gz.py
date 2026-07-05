"""
Print the first lines from a compressed JSON file.

Usage:
    python3 scripts/peek_json_gz.py path/to/file.json.gz
    python3 scripts/peek_json_gz.py path/to/file.json.gz --lines 10
"""

import argparse
import gzip
import sys
from itertools import islice
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return parsed


def print_first_lines(json_gz_path: Path, line_count: int) -> None:
    with gzip.open(json_gz_path, mode="rt", encoding="utf-8") as fh:
        for line in islice(fh, line_count):
            sys.stdout.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompress a .json.gz file and print its first few lines.",
    )
    parser.add_argument("json_gz_path", type=Path, help="Path to the .json.gz file.")
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
    print_first_lines(args.json_gz_path, args.lines)


if __name__ == "__main__":
    main()
