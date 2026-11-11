from __future__ import annotations

import json
from pathlib import Path

import pytest
import zstandard as zstd
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    canonical_json_bytes,
)
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle

from ._checkpoints import _checkpoint_builder, _run_manifest


@pytest.fixture
def valid_run(tmp_path: Path) -> Path:
    run = tmp_path / "run-0001"
    run.mkdir()
    (run / "checkpoints").mkdir()
    manifest = _run_manifest()
    (run / "run.json").write_bytes(canonical_json_bytes(manifest))
    with checkpoint.run_writer_lock(run):
        checkpoint.publish_checkpoint(run, _checkpoint_builder(manifest, step=1))
    return run


@pytest.fixture
def target(tmp_path):
    return tmp_path / "bundle"


@pytest.fixture(scope="package")
def prepared_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("recursive-prepared")
    corpus = root / "corpus"
    corpus.mkdir()
    rows = [
        {"text": "alpha beta gamma delta epsilon zeta eta theta " * 8}
        for _ in range(12)
    ]
    payload = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )
    (corpus / "tiny-0000.jsonl.zst").write_bytes(
        zstd.ZstdCompressor().compress(payload)
    )
    corpus_config = CorpusConfig(
        input_root=corpus,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=corpus_config,
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    prepared = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=corpus_config,
            tokenizer_bundle=tokenizer.path,
            sequence_length=4,
            shuffle_window_rows=4,
            output_shard_rows=8,
            seed=7,
        ),
        root / "prepared",
    )
    return prepared.path
