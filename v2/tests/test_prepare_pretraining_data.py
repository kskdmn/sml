import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPT_PATH = SRC_DIR / "prepare_pretraining_data.py"


def load_script():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "prepare_pretraining_data", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_zst_rows(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(row) for row in rows)
    compressed = zstd.ZstdCompressor().compress(text.encode("utf-8"))
    path.write_bytes(compressed)


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size

    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]

    def get_piece_size(self):
        return self.vocab_size


def test_packed_token_blocks_keep_sequence_plus_one_tokens_with_overlap():
    prepare = load_script()

    blocks = list(
        prepare.iter_packed_token_blocks(
            texts=["10 11 12", "20 21", "30 31 32 33"],
            tokenizer=FakeTokenizer(),
            sequence_length=4,
        )
    )

    assert [
        [1, 10, 11, 12, 2],
        [2, 1, 20, 21, 2],
        [2, 1, 30, 31, 32],
    ] == blocks


def test_prepare_pretraining_data_writes_uint16_npz_shards_and_manifest(tmp_path):
    prepare = load_script()
    input_file = tmp_path / "sample-0000.jsonl.zst"
    output_dir = tmp_path / "prepared"
    write_zst_rows(
        input_file,
        [
            {"text": "10 11 12 " * 25},
            {"text": "20 21 22 " * 25},
        ],
    )

    result = prepare.prepare_pretraining_data(
        prepare.PretrainingDataConfig(
            input_dir=tmp_path,
            input_file_name_regex=r"sample-0000\.jsonl\.zst\Z",
            output_dir=output_dir,
            tokenizer_model_path=tmp_path / "tokenizer.model",
            sequence_length=32,
            blocks_per_shard=3,
            max_rows_per_file=None,
            shuffle_input_files=False,
            shuffle_blocks=False,
        ),
        tokenizer=FakeTokenizer(),
    )

    assert result.manifest_path == output_dir / "manifest.json"
    assert [output_dir / "train-000000.npz", output_dir / "train-000001.npz"] == [
        shard.path for shard in result.shards
    ]

    first_shard = np.load(output_dir / "train-000000.npz")
    tokens = first_shard["tokens"]
    assert tokens.dtype == np.uint16
    assert tokens.shape == (3, 33)
    assert tokens[0][:5].tolist() == [1, 10, 11, 12, 10]

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "sml-pretokenized-blocks-v1"
    assert manifest["array_name"] == "tokens"
    assert manifest["dtype"] == "uint16"
    assert manifest["sequence_length"] == 32
    assert manifest["tokens_per_block"] == 33
    assert manifest["tokenizer_vocab_size"] == 128
    assert manifest["rows_read"] == 2
    assert manifest["texts_used"] == 2
    assert manifest["blocks"] == 4
    assert [
        {"path": "train-000000.npz", "blocks": 3},
        {"path": "train-000001.npz", "blocks": 1},
    ] == manifest["shards"]


def test_prepare_pretraining_data_rejects_vocab_too_large_for_uint16(tmp_path):
    prepare = load_script()
    input_file = tmp_path / "sample-0000.jsonl.zst"
    write_zst_rows(input_file, [{"text": "10 11 12 " * 25}])

    with pytest.raises(ValueError, match="vocab_size <= 65536"):
        prepare.prepare_pretraining_data(
            prepare.PretrainingDataConfig(
                input_dir=tmp_path,
                input_file_name_regex=r"sample-0000\.jsonl\.zst\Z",
                output_dir=tmp_path / "prepared",
                tokenizer_model_path=tmp_path / "tokenizer.model",
                sequence_length=4,
                max_rows_per_file=None,
            ),
            tokenizer=FakeTokenizer(vocab_size=65_537),
        )


def test_prepare_pretraining_data_shuffles_blocks_deterministically(tmp_path):
    prepare = load_script()
    input_file = tmp_path / "sample-0000.jsonl.zst"
    write_zst_rows(
        input_file,
        [
            {"text": "10 11 12 " * 25},
            {"text": "20 21 22 " * 25},
        ],
    )

    def build(output_name: str):
        result = prepare.prepare_pretraining_data(
            prepare.PretrainingDataConfig(
                input_dir=tmp_path,
                input_file_name_regex=r"sample-0000\.jsonl\.zst\Z",
                output_dir=tmp_path / output_name,
                tokenizer_model_path=tmp_path / "tokenizer.model",
                sequence_length=4,
                blocks_per_shard=8,
                max_rows_per_file=None,
                shuffle_input_files=False,
                shuffle_blocks=True,
                seed=99,
            ),
            tokenizer=FakeTokenizer(),
        )
        return np.load(result.shards[0].path)["tokens"].tolist()

    first = build("prepared-a")
    second = build("prepared-b")

    assert first == second
    assert first != [
        [1, 10, 11, 12, 10],
        [10, 11, 12, 10, 11],
        [11, 12, 10, 11, 12],
        [12, 10, 11, 12, 10],
        [10, 11, 12, 10, 11],
        [11, 12, 10, 11, 12],
        [12, 10, 11, 12, 10],
        [10, 11, 12, 10, 11],
    ]
