from __future__ import annotations

# Manifest validation reports malformed persisted content uniformly as ValueError.
# ruff: noqa: TRY004
import json
from pathlib import Path

from config import OUTPUT_DIR, resolve_path

FORMAT_NAME = "sml-pretokenized-blocks-v1"
TOKENS_ARRAY_NAME = "tokens"
TOKEN_DTYPE_NAME = "uint16"
MANIFEST_NAME = "manifest.json"
SHARD_NAME_PREFIX = "train"
SHARD_NAME_SUFFIX = ".npz"
DEFAULT_PRETRAINING_DATA_DIR = OUTPUT_DIR / "pretraining_data"


def shard_name(shard_index: int) -> str:
    return f"{SHARD_NAME_PREFIX}-{shard_index:06d}{SHARD_NAME_SUFFIX}"


def load_manifest(manifest_path: Path) -> dict[str, object]:
    path = resolve_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def validate_manifest_for_training(
    manifest: dict[str, object],
    *,
    manifest_dir: Path,
    sequence_length: int,
    tokenizer_vocab_size: int,
) -> tuple[Path, ...]:
    format_name = manifest.get("format")
    if format_name != FORMAT_NAME:
        raise ValueError(
            f"Unsupported pretraining format {format_name!r}; expected {FORMAT_NAME!r}"
        )
    manifest_sequence_length = manifest.get("sequence_length")
    if manifest_sequence_length != sequence_length:
        raise ValueError(
            f"TrainingConfig.sequence_length ({sequence_length}) does not match "
            f"manifest sequence_length ({manifest_sequence_length})"
        )
    tokens_per_block = manifest.get("tokens_per_block")
    if tokens_per_block != sequence_length + 1:
        raise ValueError(
            f"manifest tokens_per_block ({tokens_per_block}) does not match "
            f"sequence_length + 1 ({sequence_length + 1})"
        )
    manifest_vocab_size = manifest.get("tokenizer_vocab_size")
    if manifest_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"tokenizer vocab size ({tokenizer_vocab_size}) does not match manifest "
            f"tokenizer_vocab_size ({manifest_vocab_size})"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest shards must be a non-empty list")
    root = resolve_path(manifest_dir)
    resolved: list[Path] = []
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("manifest shard entries must be objects")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("manifest shard path must be a non-empty string")
        shard_path = root / relative
        if not shard_path.is_file():
            raise FileNotFoundError(f"Pretraining shard does not exist: {shard_path}")
        resolved.append(shard_path)
    return tuple(resolved)
