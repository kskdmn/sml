import json
import sys
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import pretraining_format as fmt  # noqa: E402


def manifest(sequence_length=4, vocab_size=128):
    return {
        "format": fmt.FORMAT_NAME,
        "sequence_length": sequence_length,
        "tokens_per_block": sequence_length + 1,
        "tokenizer_vocab_size": vocab_size,
        "shards": [{"path": "train-000000.npz", "blocks": 1}],
    }


def test_load_manifest_requires_json_object(tmp_path):
    path = tmp_path / fmt.MANIFEST_NAME
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        fmt.load_manifest(path)


def test_validate_manifest_resolves_shards(tmp_path):
    shard = tmp_path / fmt.shard_name(0)
    shard.write_bytes(b"placeholder")
    path = tmp_path / fmt.MANIFEST_NAME
    path.write_text(json.dumps(manifest()), encoding="utf-8")
    loaded = fmt.load_manifest(path)
    assert fmt.validate_manifest_for_training(
        loaded,
        manifest_dir=tmp_path,
        sequence_length=4,
        tokenizer_vocab_size=128,
    ) == (shard,)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"format": "wrong"}, "format"),
        ({"sequence_length": 8}, "sequence_length"),
        ({"tokenizer_vocab_size": 256}, "tokenizer_vocab_size"),
    ],
)
def test_validate_manifest_rejects_incompatible_metadata(tmp_path, updates, message):
    data = manifest()
    data.update(updates)
    (tmp_path / fmt.shard_name(0)).write_bytes(b"placeholder")
    with pytest.raises(ValueError, match=message):
        fmt.validate_manifest_for_training(
            data,
            manifest_dir=tmp_path,
            sequence_length=4,
            tokenizer_vocab_size=128,
        )


def test_validate_manifest_rejects_missing_shard(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-000000.npz"):
        fmt.validate_manifest_for_training(
            manifest(),
            manifest_dir=tmp_path,
            sequence_length=4,
            tokenizer_vocab_size=128,
        )
