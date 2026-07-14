import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "common" / "scripts" / "peek_npz.py"


def load_script():
    spec = importlib.util.spec_from_file_location("peek_npz", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def write_shard(path: Path, blocks: list[list[int]]) -> None:
    tokens = np.asarray(blocks, dtype=np.uint16)
    np.savez_compressed(path, tokens=tokens)


def test_peek_npz_shard_prints_summary_and_blocks(capsys, tmp_path, monkeypatch):
    peek_npz = load_script()
    shard_path = tmp_path / "train-000000.npz"
    write_shard(
        shard_path,
        [
            [1, 10, 11, 12, 2],
            [2, 1, 20, 21, 2],
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "sml-pretokenized-blocks-v1",
                "sequence_length": 4,
                "tokens_per_block": 5,
                "blocks_per_shard": 2,
                "blocks": 2,
                "tokenizer_vocab_size": 128,
                "tokenizer_model_path": str(tmp_path / "tokenizer.model"),
                "shards": [{"path": "train-000000.npz", "blocks": 2}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(peek_npz, "load_tokenizer", lambda _path: FakeTokenizer())

    peek_npz.peek_npz_shard(shard_path, block_count=2)
    output = capsys.readouterr().out

    assert "shape=(2, 5)" in output
    assert "sequence_length: 4" in output
    assert "shard_blocks: 2" in output
    assert "Block 0:" in output
    assert "decoded: '1 10 11 12 2'" in output
    assert "Block 1:" in output


def test_peek_npz_shard_rejects_missing_tokens_array(tmp_path):
    peek_npz = load_script()
    shard_path = tmp_path / "broken.npz"
    np.savez_compressed(shard_path, values=np.asarray([1, 2, 3], dtype=np.uint16))

    with pytest.raises(ValueError, match="Expected array 'tokens'"):
        peek_npz.peek_npz_shard(shard_path, decode=False)


def test_peek_npz_shard_rejects_out_of_range_start_block(tmp_path):
    peek_npz = load_script()
    shard_path = tmp_path / "train-000000.npz"
    write_shard(shard_path, [[1, 2, 3]])

    with pytest.raises(ValueError, match="start_block must be in"):
        peek_npz.peek_npz_shard(shard_path, start_block=1, decode=False)
