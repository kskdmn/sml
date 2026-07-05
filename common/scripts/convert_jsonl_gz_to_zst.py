"""
Convert JSONL stored as .json.gz to .jsonl.zst.

Usage:
    uv run python scripts/convert_jsonl_gz_to_zst.py path/to/file.json.gz
    uv run python scripts/convert_jsonl_gz_to_zst.py path/to/directory
"""

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from itertools import islice
from pathlib import Path

import zstandard as zstd


JSON_GZ_SUFFIX = ".json.gz"
JSONL_ZST_SUFFIX = ".jsonl.zst"
COPY_BUFFER_SIZE = 1024 * 1024


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return parsed


def jsonl_output_path(json_gz_path: Path) -> Path:
    if not json_gz_path.name.endswith(JSON_GZ_SUFFIX):
        raise ValueError(f"Expected a {JSON_GZ_SUFFIX} file: {json_gz_path}")
    return json_gz_path.with_name(json_gz_path.name[: -len(JSON_GZ_SUFFIX)] + JSONL_ZST_SUFFIX)


def candidate_files(path: Path) -> list[Path]:
    if path.is_file():
        if not path.name.endswith(JSON_GZ_SUFFIX):
            raise ValueError(f"Expected a {JSON_GZ_SUFFIX} file: {path}")
        return [path]
    if path.is_dir():
        return sorted(child for child in path.glob(f"*{JSON_GZ_SUFFIX}") if child.is_file())
    raise FileNotFoundError(f"No such file or directory: {path}")


def is_jsonl_gzip(path: Path, sample_lines: int) -> bool:
    valid_lines = 0
    empty_lines = 0

    with gzip.open(path, mode="rt", encoding="utf-8") as fh:
        for line in islice(fh, sample_lines):
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue

            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                return False

            valid_lines += 1

    return sample_lines == valid_lines + empty_lines and valid_lines > 0


def convert_gzip_to_zstd(source: Path, destination: Path, compression_level: int) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            compressor = zstd.ZstdCompressor(level=compression_level)
            with gzip.open(source, mode="rb") as gzip_file:
                with compressor.stream_writer(temp_file) as zstd_file:
                    shutil.copyfileobj(gzip_file, zstd_file, length=COPY_BUFFER_SIZE)

        temp_path.replace(destination)
        source.unlink()
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def process_file(path: Path, sample_lines: int, compression_level: int) -> bool:
    if not is_jsonl_gzip(path, sample_lines):
        print(f"skipped: {path} is not JSONL")
        return False

    destination = jsonl_output_path(path)
    convert_gzip_to_zstd(path, destination, compression_level)
    print(f"converted: {path} -> {destination}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .json.gz files that contain JSONL records to .jsonl.zst and delete the original .json.gz files.",
    )
    parser.add_argument("path", type=Path, help="A .json.gz file or a directory containing .json.gz files.")
    parser.add_argument(
        "--sample-lines",
        type=positive_int,
        default=5,
        help="Number of initial physical lines to inspect when detecting JSONL. Defaults to 5.",
    )
    parser.add_argument(
        "--level",
        type=positive_int,
        default=4,
        help="Zstandard compression level. Defaults to 4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        files = candidate_files(args.path)
        converted = 0
        for path in files:
            if process_file(path, args.sample_lines, args.level):
                converted += 1
    except (OSError, ValueError, gzip.BadGzipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"processed: {len(files)}, converted: {converted}, skipped: {len(files) - converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
