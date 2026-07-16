"""
Inspect pretokenized pretraining .npz shards.

Usage:
    uv run python common/scripts/peek_npz.py v2/output/pretraining_data/train-000000.npz
    uv run python common/scripts/peek_npz.py v2/output/pretraining_data/train-000000.npz --blocks 3
    uv run python common/scripts/peek_npz.py v2/output/pretraining_data/train-000000.npz --no-decode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SRC = REPO_ROOT / "v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from pretraining_format import (  # noqa: E402
    MANIFEST_NAME,
    TOKENS_ARRAY_NAME,
    load_manifest as load_manifest_file,
)

DEFAULT_TOKENIZER_MODEL_PATH = REPO_ROOT / "v2" / "output" / "bpe_tokenizer.model"
SUCCESS_RETURN_CODE = 0


def resolve_path(path: Path) -> Path:
    return path.expanduser()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be 0 or greater")
    return parsed


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    resolved_path = resolve_path(model_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Tokenizer model does not exist: {resolved_path}")

    processor = spm.SentencePieceProcessor()
    processor.load(str(resolved_path))
    return processor


def load_manifest(shard_path: Path) -> dict[str, object] | None:
    manifest_path = shard_path.parent / MANIFEST_NAME
    if not manifest_path.is_file():
        return None

    return load_manifest_file(manifest_path)


def resolve_tokenizer_model_path(
    shard_path: Path,
    tokenizer_model_path: Path | None,
) -> Path:
    if tokenizer_model_path is not None:
        return resolve_path(tokenizer_model_path)

    manifest = load_manifest(shard_path)
    if manifest is not None:
        manifest_tokenizer_path = manifest.get("tokenizer_model_path")
        if isinstance(manifest_tokenizer_path, str) and manifest_tokenizer_path:
            return resolve_path(Path(manifest_tokenizer_path))

    return resolve_path(DEFAULT_TOKENIZER_MODEL_PATH)


def format_token_preview(token_ids: list[int], limit: int = 16) -> str:
    if len(token_ids) <= limit:
        return str(token_ids)
    return f"{token_ids[:limit]} ... (+{len(token_ids) - limit} more)"


def peek_npz_shard(
    shard_path: Path,
    *,
    block_count: int = 1,
    start_block: int = 0,
    decode: bool = True,
    tokenizer_model_path: Path | None = None,
    token_preview_limit: int = 16,
) -> None:
    shard_path = resolve_path(shard_path)
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard does not exist: {shard_path}")

    with np.load(shard_path) as archive:
        array_names = list(archive.files)
        if TOKENS_ARRAY_NAME not in archive:
            raise ValueError(
                f"Expected array '{TOKENS_ARRAY_NAME}' in {shard_path}; found {array_names}"
            )
        tokens = archive[TOKENS_ARRAY_NAME]

    print(f"Shard: {shard_path}")
    print(f"Arrays: {array_names}")
    print(f"tokens: shape={tokens.shape}, dtype={tokens.dtype}")

    manifest = load_manifest(shard_path)
    if manifest is not None:
        print(f"Manifest: {shard_path.parent / MANIFEST_NAME}")
        for key in (
            "format",
            "sequence_length",
            "tokens_per_block",
            "blocks_per_shard",
            "blocks",
            "tokenizer_vocab_size",
        ):
            if key in manifest:
                print(f"  {key}: {manifest[key]}")

        shard_name = shard_path.name
        shards = manifest.get("shards")
        if isinstance(shards, list):
            for shard_entry in shards:
                if (
                    isinstance(shard_entry, dict)
                    and shard_entry.get("path") == shard_name
                ):
                    print(f"  shard_blocks: {shard_entry.get('blocks')}")
                    break

    total_blocks = int(tokens.shape[0])
    if start_block < 0 or start_block >= total_blocks:
        raise ValueError(
            f"start_block must be in [0, {total_blocks}); got {start_block}"
        )

    end_block = min(total_blocks, start_block + block_count)
    print(f"Showing blocks [{start_block}, {end_block}) of {total_blocks}")

    tokenizer = None
    if decode:
        model_path = resolve_tokenizer_model_path(shard_path, tokenizer_model_path)
        tokenizer = load_tokenizer(model_path)
        print(f"Tokenizer: {model_path}")

    for block_index in range(start_block, end_block):
        block = [int(token_id) for token_id in tokens[block_index]]
        print(f"\nBlock {block_index}:")
        print(f"  token_ids: {format_token_preview(block, token_preview_limit)}")
        if tokenizer is not None:
            print(f"  decoded: {tokenizer.decode(block)!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pretokenized uint16 pretraining .npz shards.",
    )
    parser.add_argument("npz_path", type=Path, help="Path to a .npz shard file.")
    parser.add_argument(
        "-n",
        "--blocks",
        type=positive_int,
        default=1,
        help="Number of blocks to print starting at --start-block. Defaults to 1.",
    )
    parser.add_argument(
        "--start-block",
        type=non_negative_int,
        default=0,
        help="Zero-based block index to start from. Defaults to 0.",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="Print token ids only; do not decode with the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help=(
            "SentencePiece model used to decode blocks. "
            f"Defaults to manifest tokenizer_model_path or {DEFAULT_TOKENIZER_MODEL_PATH}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    peek_npz_shard(
        args.npz_path,
        block_count=args.blocks,
        start_block=args.start_block,
        decode=not args.no_decode,
        tokenizer_model_path=args.tokenizer_model,
    )
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    sys.exit(main())
