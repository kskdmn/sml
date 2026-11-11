from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import zstandard as zstd
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.inference import (
    InferenceSession,
)
from sml.model.config import ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
)
from sml.training.pretrain import resume, train


def _write_tiny_corpus(path: Path) -> Path:
    path.mkdir()
    lines = [
        json.dumps(
            {
                "text": (
                    "alpha beta gamma delta epsilon zeta eta theta " * 8
                    + f"row {index}"
                )
            }
        ).encode("utf-8")
        for index in range(24)
    ]
    payload = b"\n".join(lines) + b"\n"
    (path / "tiny-0000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


def _corpus_config(path: Path) -> CorpusConfig:
    return CorpusConfig(
        input_root=path,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )


def _prepare_tiny_data(root: Path):
    corpus = _write_tiny_corpus(root / "corpus")
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=_corpus_config(corpus),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=_corpus_config(corpus),
            tokenizer_bundle=tokenizer.path,
            sequence_length=32,
            shuffle_window_rows=5,
            output_shard_rows=3,
            seed=17,
        ),
        root / "data",
    )
    return tokenizer, data


def _tiny_pretraining_config(data_path: Path, output_run: Path, vocab_size: int):
    return PretrainingConfig(
        data=data_path,
        output_run=output_run,
        model=ModelConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_context_length=32,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=4,
            warmup_steps=0,
        ),
        loader=LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=1,
            prefetch_depth=2,
            epoch_seed=13,
        ),
        checkpoint=CheckpointPolicy(interval=1),
        maximum_steps=4,
        maximum_epochs=2,
        log_interval=1,
        seed=19,
        compile=False,
    )


@pytest.fixture(scope="session")
def _tiny_run_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inference-unit")
    tokenizer, data = _prepare_tiny_data(root)
    result = train(
        _tiny_pretraining_config(
            data.path,
            root / "run",
            tokenizer.manifest.vocab_size,
        ),
        session_steps=2,
    )
    return result.run


@pytest.fixture
def tiny_pretraining_run(tmp_path: Path, _tiny_run_template: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(_tiny_run_template, dest)
    return dest


@pytest.fixture
def tiny_session(tiny_pretraining_run: Path) -> InferenceSession:
    return InferenceSession.from_checkpoint(tiny_pretraining_run)


def publish_new_valid_step(run: Path, step: int) -> None:
    current_step = json.loads((run / "latest.json").read_bytes())["step"]
    resumed = resume(run, session_steps=step - current_step)
    assert resumed.step == step
