from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    BOS_TOKEN_ID,
    DEFAULT_TOKENIZER_MODEL_PATH,
    EOS_TOKEN_ID,
    PRETRAINING_INPUT_DIR,
    PRETRAINING_INPUT_FILE_NAME_REGEX,
    OUTPUT_DIR,
    SUCCESS_RETURN_CODE,
    resolve_path,
)
from utils import (
    TEXT_COLUMN,
    discover_input_files,
    filter_text,
    get_special_token_id,
    iter_jsonl_records,
    json_ready,
    load_tokenizer,
    shuffle_input_files,
)


FORMAT_NAME = "sml-pretokenized-blocks-v1"
TOKENS_ARRAY_NAME = "tokens"
TOKEN_DTYPE_NAME = "uint16"
UINT16_VOCAB_SIZE_LIMIT = 65_536
UINT16_MAX_TOKEN_ID = UINT16_VOCAB_SIZE_LIMIT - 1
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "pretraining_data"
DEFAULT_SEQUENCE_LENGTH = 1_024
DEFAULT_BLOCKS_PER_SHARD = 8_192
DEFAULT_MAX_ROWS_PER_FILE = 40_960
DEFAULT_SEED = 42
MANIFEST_NAME = "manifest.json"
SHARD_NAME_PREFIX = "train"
SHARD_NAME_SUFFIX = ".npz"


@dataclass(frozen=True, slots=True)
class PretrainingDataConfig:
    input_dir: Path = PRETRAINING_INPUT_DIR
    input_file_name_regex: str = PRETRAINING_INPUT_FILE_NAME_REGEX
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tokenizer_model_path: Path = DEFAULT_TOKENIZER_MODEL_PATH
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    blocks_per_shard: int = DEFAULT_BLOCKS_PER_SHARD
    max_rows_per_file: int | None = DEFAULT_MAX_ROWS_PER_FILE
    shuffle_input_files: bool = True
    shuffle_blocks: bool = True
    seed: int = DEFAULT_SEED
    bos_token_id: int | None = BOS_TOKEN_ID
    eos_token_id: int | None = EOS_TOKEN_ID


@dataclass(frozen=True, slots=True)
class PreparedShard:
    path: Path
    blocks: int


@dataclass(frozen=True, slots=True)
class PreparationResult:
    manifest_path: Path
    shards: tuple[PreparedShard, ...]
    rows_read: int
    texts_used: int
    blocks: int


@dataclass(slots=True)
class PreparationCounters:
    rows_read: int = 0
    texts_used: int = 0


def validate_pretraining_data_config(config: PretrainingDataConfig) -> None:
    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if config.blocks_per_shard <= 0:
        raise ValueError("blocks_per_shard must be positive")
    if config.max_rows_per_file is not None and config.max_rows_per_file <= 0:
        raise ValueError("max_rows_per_file must be positive or None")


def validate_uint16_vocab_size(vocab_size: int) -> None:
    if vocab_size <= 0:
        raise ValueError("tokenizer vocab size must be positive")
    if vocab_size > UINT16_VOCAB_SIZE_LIMIT:
        raise ValueError(
            "uint16 pretraining shards require tokenizer vocab_size <= 65536"
        )


def checked_uint16_token_id(token_id: object) -> int:
    token_id = int(token_id)
    if token_id < 0 or token_id > UINT16_MAX_TOKEN_ID:
        raise ValueError(f"token id does not fit uint16: {token_id}")
    return token_id


def iter_packed_token_blocks(
    texts: Iterable[str],
    tokenizer,
    sequence_length: int,
    bos_token_id: int | None = BOS_TOKEN_ID,
    eos_token_id: int | None = EOS_TOKEN_ID,
) -> Iterator[list[int]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    tokens_per_block = sequence_length + 1
    buffer: list[int] = []
    bos_token_id = get_special_token_id(tokenizer, "bos_id", bos_token_id)
    eos_token_id = get_special_token_id(tokenizer, "eos_id", eos_token_id)

    def iter_ready_blocks() -> Iterator[list[int]]:
        while len(buffer) >= tokens_per_block:
            block = buffer[:tokens_per_block]
            del buffer[:sequence_length]
            yield [checked_uint16_token_id(token_id) for token_id in block]

    for text in texts:
        if bos_token_id is not None:
            buffer.append(checked_uint16_token_id(bos_token_id))
        buffer.extend(
            checked_uint16_token_id(token_id)
            for token_id in tokenizer.encode(text, out_type=int)
        )
        if eos_token_id is not None:
            buffer.append(checked_uint16_token_id(eos_token_id))
        yield from iter_ready_blocks()


def iter_filtered_texts(
    input_files: Iterable[Path],
    max_rows_per_file: int | None,
    counters: PreparationCounters,
) -> Iterator[str]:
    for input_file in input_files:
        for row, _line_number in iter_jsonl_records(input_file, max_rows_per_file):
            counters.rows_read += 1
            text = filter_text(row.get(TEXT_COLUMN))
            if text is None:
                continue
            counters.texts_used += 1
            yield text


def shard_name(shard_index: int) -> str:
    return f"{SHARD_NAME_PREFIX}-{shard_index:06d}{SHARD_NAME_SUFFIX}"


def write_token_shard(
    output_dir: Path,
    shard_index: int,
    blocks: Sequence[Sequence[int]],
) -> PreparedShard:
    if not blocks:
        raise ValueError("cannot write an empty pretraining shard")

    tokens = np.asarray(blocks, dtype=np.uint16)
    if tokens.ndim != 2:
        raise ValueError("token shard blocks must be a 2D array")

    path = output_dir / shard_name(shard_index)
    np.savez_compressed(path, **{TOKENS_ARRAY_NAME: tokens})
    return PreparedShard(path=path, blocks=int(tokens.shape[0]))


def build_manifest(
    config: PretrainingDataConfig,
    input_files: Sequence[Path],
    vocab_size: int,
    shards: Sequence[PreparedShard],
    counters: PreparationCounters,
    block_count: int,
) -> dict[str, object]:
    output_dir = resolve_path(config.output_dir)
    return {
        "format": FORMAT_NAME,
        "array_name": TOKENS_ARRAY_NAME,
        "dtype": TOKEN_DTYPE_NAME,
        "sequence_length": config.sequence_length,
        "tokens_per_block": config.sequence_length + 1,
        "blocks_per_shard": config.blocks_per_shard,
        "shuffle_input_files": config.shuffle_input_files,
        "shuffle_blocks": config.shuffle_blocks,
        "seed": config.seed,
        "tokenizer_model_path": str(resolve_path(config.tokenizer_model_path)),
        "tokenizer_vocab_size": vocab_size,
        "input_dir": str(resolve_path(config.input_dir)),
        "input_file_name_regex": config.input_file_name_regex,
        "input_files": [str(input_file) for input_file in input_files],
        "max_rows_per_file": config.max_rows_per_file,
        "rows_read": counters.rows_read,
        "texts_used": counters.texts_used,
        "blocks": block_count,
        "shards": [
            {
                "path": str(shard.path.relative_to(output_dir)),
                "blocks": shard.blocks,
            }
            for shard in shards
        ],
    }


def prepare_pretraining_data(
    config: PretrainingDataConfig | None = None,
    tokenizer=None,
) -> PreparationResult:
    config = PretrainingDataConfig() if config is None else config
    validate_pretraining_data_config(config)

    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(
        config.input_dir,
        config.input_file_name_regex,
    )
    if not input_files:
        raise FileNotFoundError(
            f"No supported input files found in {resolve_path(config.input_dir)}"
        )
    if config.shuffle_input_files:
        input_files = shuffle_input_files(input_files, seed=config.seed)

    tokenizer = (
        load_tokenizer(config.tokenizer_model_path) if tokenizer is None else tokenizer
    )
    vocab_size = int(tokenizer.get_piece_size())
    validate_uint16_vocab_size(vocab_size)

    counters = PreparationCounters()
    rng = random.Random(config.seed)
    shards: list[PreparedShard] = []
    shard_blocks: list[list[int]] = []
    block_count = 0

    def flush_shard() -> None:
        if not shard_blocks:
            return
        blocks_to_write = list(shard_blocks)
        if config.shuffle_blocks:
            rng.shuffle(blocks_to_write)
        shard = write_token_shard(output_dir, len(shards), blocks_to_write)
        shards.append(shard)
        shard_blocks.clear()

    texts = iter_filtered_texts(
        input_files,
        max_rows_per_file=config.max_rows_per_file,
        counters=counters,
    )
    for block in iter_packed_token_blocks(
        texts,
        tokenizer=tokenizer,
        sequence_length=config.sequence_length,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
    ):
        shard_blocks.append(block)
        block_count += 1
        if len(shard_blocks) >= config.blocks_per_shard:
            flush_shard()
    flush_shard()

    manifest = build_manifest(
        config=config,
        input_files=input_files,
        vocab_size=vocab_size,
        shards=shards,
        counters=counters,
        block_count=block_count,
    )
    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(json_ready(manifest), manifest_file, indent=2, sort_keys=True)

    return PreparationResult(
        manifest_path=manifest_path,
        shards=tuple(shards),
        rows_read=counters.rows_read,
        texts_used=counters.texts_used,
        blocks=block_count,
    )


def parse_optional_positive_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare pretokenized uint16 .npz shards for SML pretraining."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PRETRAINING_INPUT_DIR,
        help=f"JSONL zstd input directory (default: {PRETRAINING_INPUT_DIR})",
    )
    parser.add_argument(
        "--input-file-name-regex",
        default=PRETRAINING_INPUT_FILE_NAME_REGEX,
        help=f"input file name regex (default: {PRETRAINING_INPUT_FILE_NAME_REGEX})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"prepared shard output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=DEFAULT_TOKENIZER_MODEL_PATH,
        help=f"SentencePiece model path (default: {DEFAULT_TOKENIZER_MODEL_PATH})",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=f"training sequence length (default: {DEFAULT_SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--blocks-per-shard",
        type=int,
        default=DEFAULT_BLOCKS_PER_SHARD,
        help=f"packed blocks per .npz shard (default: {DEFAULT_BLOCKS_PER_SHARD})",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=parse_optional_positive_int,
        default=DEFAULT_MAX_ROWS_PER_FILE,
        help=(
            "maximum JSONL rows to read from each input file, or 'none' for all "
            f"rows (default: {DEFAULT_MAX_ROWS_PER_FILE})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic shuffle seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--no-shuffle-input-files",
        action="store_true",
        help="keep discovered input files in sorted order",
    )
    parser.add_argument(
        "--no-shuffle-blocks",
        action="store_true",
        help="write packed blocks in stream order within each shard",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PretrainingDataConfig:
    return PretrainingDataConfig(
        input_dir=args.input_dir,
        input_file_name_regex=args.input_file_name_regex,
        output_dir=args.output_dir,
        tokenizer_model_path=args.tokenizer_model,
        sequence_length=args.sequence_length,
        blocks_per_shard=args.blocks_per_shard,
        max_rows_per_file=args.max_rows_per_file,
        shuffle_input_files=not args.no_shuffle_input_files,
        shuffle_blocks=not args.no_shuffle_blocks,
        seed=args.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = prepare_pretraining_data(config_from_args(args))
    print(f"Manifest: {result.manifest_path}")
    print(f"Shards: {len(result.shards)}")
    print(f"Rows read: {result.rows_read}")
    print(f"Texts used: {result.texts_used}")
    print(f"Blocks: {result.blocks}")
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
