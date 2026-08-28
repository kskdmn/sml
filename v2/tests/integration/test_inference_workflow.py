from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
import zstandard as zstd
from sml import inference
from sml.artifacts.checkpoint import resolve_latest_step, run_access_lock
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    LatestIndex,
    PayloadRef,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import (
    GenerationRequest,
    InferenceSession,
    resolve_model_artifact,
)
from sml.model.config import GenerationConfig, ModelConfig
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
    root = tmp_path_factory.mktemp("inference-integration")
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


def _latest_step_directory(run: Path) -> Path:
    resolved = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.MANIFEST_TRUSTED,
    )
    return resolved.step_directory


def _corrupt_model_payload_same_size(run: Path) -> None:
    model_path = _latest_step_directory(run) / "model.safetensors"
    payload = bytearray(model_path.read_bytes())
    header_length = int.from_bytes(payload[:8], "little")
    data_index = 8 + header_length + 32
    payload[data_index] ^= 0x01
    model_path.write_bytes(bytes(payload))


def _rewrite_working_weights_not_cast_of_master(run: Path) -> None:
    resolved = resolve_latest_step(
        run,
        writable=False,
        verification=VerificationLevel.MANIFEST_TRUSTED,
    )
    step_directory = resolved.step_directory
    model_path = step_directory / "model.safetensors"
    working = mx.load(model_path)
    first = next(iter(sorted(working)))
    working[first] = (mx.ones_like(working[first]) * 2).astype(mx.bfloat16)
    mx.save_safetensors(model_path, working)
    with model_path.open("rb") as payload:
        identity = file_identity(payload)
    model_ref = ArrayPayloadRef(
        PayloadRef("model.safetensors", identity, model_path.stat().st_size),
        tuple(
            ArraySpec(name, tuple(array.shape), "bfloat16")
            for name, array in sorted(working.items())
        ),
    )
    checkpoint = resolved.checkpoint
    updated = replace(
        checkpoint,
        identity="sha256:" + "0" * 64,
        model=model_ref,
    )
    updated = replace(updated, identity=updated.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(updated))
    latest = read_manifest(
        run,
        LatestIndex,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    repaired = replace(
        latest,
        identity="sha256:" + "0" * 64,
        checkpoint_identity=updated.identity,
    )
    repaired = replace(repaired, identity=repaired.recompute_identity())
    (run / "latest.json").write_bytes(canonical_json_bytes(repaired))


def test_read_only_stale_latest_recovery_does_not_persist(
    tiny_pretraining_run: Path,
) -> None:
    latest = tiny_pretraining_run / "latest.json"
    latest.write_bytes(b"not-json")
    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    step_dirs = list((tiny_pretraining_run / "checkpoints").glob("step-*"))
    assert len(step_dirs) == 1
    assert resolved.step == int(step_dirs[0].name.split("-")[1])
    assert latest.read_bytes() == b"not-json"


def test_resolve_holds_shared_access_lock_through_owned_array_evaluation(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []
    real_eval = inference.mx.eval

    def checking_eval(*args, **kwargs):
        with (
            pytest.raises(SMLArtifactError, match="held by"),
            run_access_lock(tiny_pretraining_run, exclusive=True),
        ):
            pass
        observed.append(True)
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(inference.mx, "eval", checking_eval)
    resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    assert observed


def test_pretraining_resolution_closes_checkpoint_owner_when_tokenizer_construction_fails(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches inference failures leaking the retained latest checkpoint reader."""
    observed_descriptors: list[int] = []
    real_open_latest = inference.open_latest_checkpoint_reader

    @contextmanager
    def record_reader(*args, **kwargs):
        reader = None
        try:
            with real_open_latest(*args, **kwargs) as reader:
                observed_descriptors.extend(
                    [
                        reader._run_descriptor,
                        reader._checkpoints_descriptor,
                        reader._owned_step.descriptor,
                    ]
                )
                yield reader
        finally:
            pass

    monkeypatch.setattr(inference, "open_latest_checkpoint_reader", record_reader)
    monkeypatch.setattr(
        inference,
        "_load_opened_tokenizer_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("tokenizer failed")
        ),
    )

    with pytest.raises(RuntimeError, match="tokenizer failed"):
        resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    for descriptor in observed_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_pretraining_resolution_uses_open_run_after_outer_name_replacement(
    tiny_pretraining_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference must read the original run inode after resolving latest."""
    displaced = tmp_path / "displaced-run"
    closed_descriptors: list[int] = []
    real_open_latest = inference.open_latest_checkpoint_reader

    @contextmanager
    def replace_outer_name(*args, **kwargs):
        with real_open_latest(*args, **kwargs) as reader:
            closed_descriptors.extend(
                [
                    reader._run_descriptor,
                    reader._checkpoints_descriptor,
                    reader._owned_step.descriptor,
                ]
            )
            tiny_pretraining_run.rename(displaced)
            tiny_pretraining_run.mkdir()
            try:
                yield reader
            finally:
                tiny_pretraining_run.rmdir()
                displaced.rename(tiny_pretraining_run)

    monkeypatch.setattr(inference, "open_latest_checkpoint_reader", replace_outer_name)

    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)

    assert resolved.artifact_kind == "pretraining-run"
    for descriptor in closed_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_pretraining_resolution_uses_loaded_checkpoint_payload_after_replacement(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved inference weights cannot be redirected by a later payload replacement."""
    closed_descriptors: list[int] = []
    real_open_latest = inference.open_latest_checkpoint_reader

    @contextmanager
    def replace_payload(*args, **kwargs):
        reader = None
        try:
            with real_open_latest(*args, **kwargs) as reader:
                closed_descriptors.extend(
                    [
                        reader._run_descriptor,
                        reader._checkpoints_descriptor,
                        reader._owned_step.descriptor,
                    ]
                )
                (reader.resolved.step_directory / "model.safetensors").write_bytes(
                    b"replacement"
                )
                yield reader
        finally:
            pass

    monkeypatch.setattr(inference, "open_latest_checkpoint_reader", replace_payload)

    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)

    assert resolved.artifact_kind == "pretraining-run"
    for descriptor in closed_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_resolve_reports_manifest_trusted_versus_full_metadata(
    tiny_pretraining_run: Path,
) -> None:
    trusted = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    full = resolve_model_artifact(tiny_pretraining_run, full_verify=True)
    assert trusted.verification is VerificationLevel.MANIFEST_TRUSTED
    assert full.verification is VerificationLevel.FULL
    assert trusted.step == full.step
    assert trusted.tokenizer_identity == full.tokenizer_identity

    _corrupt_model_payload_same_size(tiny_pretraining_run)
    still_trusted = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    assert still_trusted.verification is VerificationLevel.MANIFEST_TRUSTED
    with pytest.raises(SMLArtifactError, match="identity"):
        resolve_model_artifact(tiny_pretraining_run, full_verify=True)


def test_full_resolve_rejects_master_working_cast_mismatch(
    tiny_pretraining_run: Path,
) -> None:
    _rewrite_working_weights_not_cast_of_master(tiny_pretraining_run)
    with pytest.raises(SMLArtifactError, match="exact BF16 cast"):
        resolve_model_artifact(tiny_pretraining_run, full_verify=True)


def test_session_does_not_own_training_only_master_or_optimizer_arrays(
    tiny_session: InferenceSession,
) -> None:
    arrays = tiny_session.resolved_model.model_arrays
    names = set(arrays)
    assert "step" not in names
    assert not any(name.startswith("first_moments.") for name in names)
    assert not any(name.startswith("second_moments.") for name in names)
    assert not any(name.startswith("accumulators.") for name in names)
    assert "next_key" not in names
    assert "loss_numerator" not in names
    assert not hasattr(tiny_session.resolved_model, "master_arrays")
    assert not hasattr(tiny_session, "master_arrays")
    mx.eval(*arrays.values())
    assert all(array.dtype == mx.bfloat16 for array in arrays.values())


def test_resolved_session_checkpoint_mappings_cannot_be_mutated(
    tiny_session: InferenceSession,
) -> None:
    arrays = tiny_session.resolved_model.model_arrays
    with pytest.raises(TypeError):
        arrays["injected"] = mx.zeros((1,), dtype=mx.bfloat16)


def test_session_prompt_overflow(tiny_session: InferenceSession) -> None:
    with pytest.raises(SMLRuntimeError, match="overflow"):
        tiny_session.generate("alpha", GenerationRequest(max_new_tokens=32))


def test_session_empty_text_without_usable_bos(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tiny_session.resolved_model.tokenizer.processor,
        "bos_id",
        lambda: -1,
    )
    with pytest.raises(SMLRuntimeError, match="empty|BOS|bos"):
        tiny_session.generate("", GenerationRequest(max_new_tokens=1))


def test_session_length_buckets_are_powers_of_two_including_context(
    tiny_session: InferenceSession,
) -> None:
    assert tiny_session.length_buckets == (1, 2, 4, 8, 16, 32)


def test_session_generate_batch_restores_caller_order(
    tiny_session: InferenceSession,
) -> None:
    items = [
        ("alpha", GenerationRequest(max_new_tokens=1)),
        ("beta", GenerationRequest(max_new_tokens=2)),
        ("gamma", GenerationRequest(max_new_tokens=1)),
    ]
    serial = tuple(tiny_session.generate(text, request) for text, request in items)
    batched = tiny_session.generate_batch(items)
    assert tuple(result.token_ids for result in batched) == tuple(
        result.token_ids for result in serial
    )
    assert tiny_session.buffer_pool.active_leases == 0


def test_short_prompt_long_generation_seeded_batch_matches_serial(
    tiny_session: InferenceSession,
) -> None:
    items = [
        (
            "alpha",
            GenerationRequest(
                max_new_tokens=16,
                config=GenerationConfig(temperature=0.8, top_p=0.9, seed=101),
            ),
        ),
        (
            "alpha",
            GenerationRequest(
                max_new_tokens=16,
                config=GenerationConfig(temperature=0.8, top_p=0.9, seed=103),
            ),
        ),
    ]
    buckets = tiny_session._bucketize(items)
    assert len(buckets) == 1
    assert buckets[0].prefill_length_bucket < buckets[0].cache_capacity_bucket

    serial = tuple(tiny_session.generate(text, request) for text, request in items)
    batched = tiny_session.generate_batch(items)

    assert tuple(result.token_ids for result in batched) == tuple(
        result.token_ids for result in serial
    )
    assert tuple(result.seed for result in batched) == (101, 103)
    assert tiny_session.buffer_pool.active_leases == 0


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


def test_new_session_resolves_published_step_under_latest_only(
    tiny_pretraining_run: Path,
) -> None:
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    original_step = session.model_identity.step
    publish_new_valid_step(tiny_pretraining_run, step=original_step + 1)
    latest = InferenceSession.from_checkpoint(tiny_pretraining_run)
    assert latest.model_identity.step == original_step + 1
    step_dirs = list((tiny_pretraining_run / "checkpoints").glob("step-*"))
    assert len(step_dirs) == 1
    assert int(step_dirs[0].name.split("-")[1]) == original_step + 1
    with pytest.raises(SMLArtifactError, match="step"):
        InferenceSession.from_checkpoint(step_dirs[0])
