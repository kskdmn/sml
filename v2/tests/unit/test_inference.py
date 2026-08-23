from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import pytest
import zstandard as zstd
from sml import inference
from sml.artifacts.checkpoint import VerifiedCheckpointContents
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import (
    GenerationRequest,
    InferenceRuntimeConfig,
    InferenceSession,
    resolve_model_artifact,
)
from sml.model.config import ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    ResumeOverrides,
)
from sml.training.pretrain import resume, train

EXPECTED_SECOND_TOKENS = (7, 11)


def raise_after_one_token(*_args, **_kwargs):
    raise SMLRuntimeError("decode failed after one token")


def deterministic_decode(*_args, **_kwargs):
    return EXPECTED_SECOND_TOKENS


def expected_second_tokens() -> tuple[int, ...]:
    return EXPECTED_SECOND_TOKENS


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
        maximum_steps=2,
        maximum_epochs=2,
        log_interval=1,
        seed=19,
        compile=False,
    )


@pytest.fixture(scope="module")
def _tiny_run_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inference-unit")
    tokenizer, data = _prepare_tiny_data(root)
    result = train(
        _tiny_pretraining_config(
            data.path,
            root / "run",
            tokenizer.manifest.vocab_size,
        )
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
    resume(run, data=None, overrides=ResumeOverrides(maximum_steps=step))


def test_session_runtime_config_rejects_non_increasing_batch_buckets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        InferenceRuntimeConfig(batch_size_buckets=(1, 2, 2, 8))


def test_session_runtime_config_rejects_non_positive_batch_buckets() -> None:
    with pytest.raises(ValueError, match="positive"):
        InferenceRuntimeConfig(batch_size_buckets=(1, 0, 4))


def test_resolve_rejects_direct_step_path(tmp_path: Path) -> None:
    step_path = tmp_path / "checkpoints" / "step-000000001"
    step_path.mkdir(parents=True)
    with pytest.raises(SMLArtifactError, match="step"):
        resolve_model_artifact(step_path, full_verify=False)


def test_resolve_rejects_historical_selector_on_step_directory(
    tiny_pretraining_run: Path,
) -> None:
    step_path = next((tiny_pretraining_run / "checkpoints").glob("step-*"))
    with pytest.raises(SMLArtifactError, match="step"):
        resolve_model_artifact(step_path, full_verify=False)


def test_verified_checkpoint_contents_mappings_are_immutable_for_session_resolve() -> (
    None
):
    scalar = {
        "kind": "pretraining-state",
        "cursor": {"epoch": 0, "shard_order_position": 0, "row_offset": 0},
    }
    inner = {"weight": mx.array([1.0], dtype=mx.float32)}
    groups = {"model.safetensors": inner}
    contents = VerifiedCheckpointContents(scalar, groups)

    with pytest.raises(TypeError):
        contents.scalar_state["kind"] = "mutated"
    with pytest.raises(TypeError):
        contents.scalar_state["cursor"]["epoch"] = 1
    with pytest.raises(TypeError):
        contents.array_groups["trainer.safetensors"] = {}
    with pytest.raises(TypeError):
        contents.array_groups["model.safetensors"]["weight"] = mx.array(
            [2.0], dtype=mx.float32
        )


def test_session_loads_latest_once_and_pins_identity(
    tiny_pretraining_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_loader = inference.load_owned_model_arrays

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(inference, "load_owned_model_arrays", counted_loader)
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    first_identity = session.model_identity
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    publish_new_valid_step(tiny_pretraining_run, step=4)
    assert session.model_identity == first_identity
    assert calls == 1


def test_failed_call_cannot_contaminate_next_call(
    tiny_session: InferenceSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tiny_session, "_decode_chunk", raise_after_one_token)
    with pytest.raises(SMLRuntimeError, match="decode"):
        tiny_session.generate("first", GenerationRequest(max_new_tokens=2))
    monkeypatch.setattr(tiny_session, "_decode_chunk", deterministic_decode)
    assert (
        tiny_session.generate("second", GenerationRequest(max_new_tokens=2)).token_ids
        == expected_second_tokens()
    )


def test_overlapping_session_call_fails_before_state_mutation(
    tiny_session: InferenceSession,
) -> None:
    with (
        tiny_session._call_guard.acquire(),
        pytest.raises(SMLRuntimeError, match="non-reentrant"),
    ):
        tiny_session.generate("blocked", GenerationRequest(max_new_tokens=1))
    assert tiny_session.buffer_pool.active_leases == 0


def test_session_generate_returns_tokens_and_releases_lease(
    tiny_session: InferenceSession,
) -> None:
    result = tiny_session.generate("alpha", GenerationRequest(max_new_tokens=1))
    assert len(result.token_ids) == 1
    assert result.model == tiny_session.model_identity
    assert tiny_session.buffer_pool.active_leases == 0
