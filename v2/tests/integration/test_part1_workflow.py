from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
import zstandard as zstd
from sml.artifacts import VerificationResult, verify_artifact
from sml.artifacts.checkpoint import resolve_exact_step
from sml.artifacts.manifest import VerificationLevel, canonical_json_bytes
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    ResumeOverrides,
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


def _assert_model_is_exact_bf16_cast_of_master(step_directory: Path) -> None:
    model = mx.load(step_directory / "model.safetensors")
    master = mx.load(step_directory / "master.safetensors")
    assert set(model) == set(master)
    mx.eval(model, master)
    for name in sorted(master):
        assert master[name].dtype == mx.float32
        assert model[name].dtype == mx.bfloat16
        assert model[name].shape == master[name].shape
        assert mx.array_equal(model[name], master[name].astype(mx.bfloat16)).item()


def _prepare_tiny_data(tmp_path: Path):
    corpus = _write_tiny_corpus(tmp_path / "corpus")
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=_corpus_config(corpus),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        tmp_path / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=_corpus_config(corpus),
            tokenizer_bundle=tokenizer.path,
            sequence_length=4,
            shuffle_window_rows=5,
            output_shard_rows=3,
            seed=17,
        ),
        tmp_path / "data",
    )
    return tokenizer, data


def test_part1_tokenizer_to_resumed_pretraining(tmp_path: Path) -> None:
    tokenizer, data = _prepare_tiny_data(tmp_path)
    first = train(
        PretrainingConfig(
            data=data.path,
            output_run=tmp_path / "run",
            model=ModelConfig(
                vocab_size=tokenizer.manifest.vocab_size,
                hidden_size=8,
                num_layers=1,
                num_q_heads=2,
                num_kv_heads=1,
                intermediate_size=16,
                original_context_length=4,
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
            maximum_steps=2,
            maximum_epochs=2,
            log_interval=1,
            seed=19,
            compile=False,
        )
    )
    resumed = resume(
        first.run,
        data=None,
        overrides=ResumeOverrides(maximum_steps=3),
    )
    exact = resolve_exact_step(
        resumed.run,
        step=3,
        verification=VerificationLevel.FULL,
    )
    verified_tokenizer = verify_artifact(tokenizer.path, full=True)
    verified_data = verify_artifact(data.path, full=True)
    verified = verify_artifact(resumed.run, full=True)

    assert first.step == 2
    assert resumed.step == 3
    assert exact.step == 3
    assert exact.run.model["rope_scaling_factor"] == 1.0
    assert exact.run.tokenizer_identity == tokenizer.manifest.identity
    assert exact.run.data_identity == data.manifest.identity
    _assert_model_is_exact_bf16_cast_of_master(exact.step_directory)

    assert verified_tokenizer.manifest.identity == tokenizer.manifest.identity
    assert verified_tokenizer.children == ()
    assert verified_data.manifest.identity == data.manifest.identity
    assert [child.manifest.identity for child in verified_data.children] == [
        tokenizer.manifest.identity
    ]
    assert isinstance(verified, VerificationResult)
    assert verified.path == resumed.run
    assert verified.manifest.identity == exact.run.identity
    assert verified.verification is VerificationLevel.FULL
    assert {child.manifest.kind for child in verified.children} == {
        "pretraining-checkpoint",
        "tokenizer",
    }
    assert [path.name for path in (resumed.run / "checkpoints").iterdir()] == [
        "step-000000003"
    ]
    with pytest.raises(SMLArtifactError, match="does not exist"):
        resolve_exact_step(
            resumed.run,
            step=1,
            verification=VerificationLevel.FULL,
        )


def test_full_verification_rejects_mismatched_nested_tokenizer_refs(
    tmp_path: Path,
) -> None:
    _tokenizer, data = _prepare_tiny_data(tmp_path)
    manifest = data.manifest
    malformed = replace(
        manifest,
        tokenizer_model=replace(
            manifest.tokenizer_vocab,
            logical_path="tokenizer/tokenizer.vocab",
        ),
        tokenizer_vocab=replace(
            manifest.tokenizer_model,
            logical_path="tokenizer/tokenizer.model",
        ),
    )
    malformed = replace(malformed, identity=malformed.recompute_identity())
    (data.path / "manifest.json").write_bytes(canonical_json_bytes(malformed))

    with pytest.raises(SMLArtifactError, match="tokenizer payload references"):
        verify_artifact(data.path, full=True)
