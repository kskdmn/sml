from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import checkpoint as checkpoint_module
from sml.artifacts import semantics as semantics_module
from sml.artifacts.manifest import (
    LatestIndex,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_checkpoint_manifest,
    read_manifest,
)
from sml.artifacts.verify import verify_artifact
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
)
from sml.training.pretrain import train

from ._payloads import (
    _array_ref,
    _payload_ref,
)


@pytest.fixture(scope="module")
def pretraining_run_template(
    tmp_path_factory: pytest.TempPathFactory,
    prepared_template: Path,
) -> Path:
    root = tmp_path_factory.mktemp("recursive-pretraining-run")
    tokenizer = read_manifest(
        prepared_template / "tokenizer",
        TokenizerManifest,
        VerificationLevel.FULL,
    ).manifest
    result = train(
        PretrainingConfig(
            data=prepared_template,
            output_run=root / "run",
            model=ModelConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=8,
                num_layers=1,
                num_q_heads=2,
                num_kv_heads=1,
                intermediate_size=16,
                original_context_length=4,
                hidden_dropout=0.1,
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
                gradient_accumulation_steps=2,
                prefetch_depth=1,
                epoch_seed=7,
            ),
            checkpoint=CheckpointPolicy(interval=1),
            maximum_steps=1,
            maximum_epochs=2,
            log_interval=1,
            seed=11,
            compile=False,
        )
    )
    return result.run


def _latest_checkpoint(run: Path) -> tuple[Path, PretrainingCheckpointManifest]:
    latest = read_manifest(run, LatestIndex, VerificationLevel.FULL).manifest
    step = run / "checkpoints" / f"step-{latest.step:09d}"
    checkpoint = read_checkpoint_manifest(
        step,
        VerificationLevel.FULL,
    ).manifest
    assert isinstance(checkpoint, PretrainingCheckpointManifest)
    return step, checkpoint


def _rebind_latest_checkpoint(
    run: Path,
    checkpoint: PretrainingCheckpointManifest,
) -> None:
    latest = read_manifest(
        run, LatestIndex, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    rebound = replace(latest, checkpoint_identity=checkpoint.identity)
    rebound = replace(rebound, identity=rebound.recompute_identity())
    (run / "latest.json").write_bytes(canonical_json_bytes(rebound))


def test_full_run_verification_does_not_materialize_checkpoint_groups(
    pretraining_run_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone recursive FULL uses headers and fixed chunks, never live groups."""

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("checkpoint group materialization is forbidden")

    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_array_stream",
        reject_materialization,
    )

    result = verify_artifact(pretraining_run_template, full=True)

    assert result.verification is VerificationLevel.FULL


def test_full_pretraining_run_rejects_resigned_model_config_leaf_disagreement(
    tmp_path: Path,
    pretraining_run_template: Path,
) -> None:
    """Changing a valid run config must not reinterpret incompatible model leaves."""
    run = tmp_path / "run"
    shutil.copytree(pretraining_run_template, run)
    original = read_manifest(
        run,
        PretrainingRunManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    model = dict(original.model)
    model["hidden_size"] = 12
    rebound_run = replace(original, model=model)
    rebound_run = replace(rebound_run, identity=rebound_run.recompute_identity())
    (run / "run.json").write_bytes(canonical_json_bytes(rebound_run))

    step, checkpoint = _latest_checkpoint(run)
    state_path = step / checkpoint.scalar_state.logical_path
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["owning_run_identity"] = rebound_run.identity
    state_path.write_bytes(canonical_json_bytes(state))
    rebound_checkpoint = replace(
        checkpoint,
        owning_run_identity=rebound_run.identity,
        scalar_state=_payload_ref(state_path, checkpoint.scalar_state.logical_path),
    )
    rebound_checkpoint = replace(
        rebound_checkpoint,
        identity=rebound_checkpoint.recompute_identity(),
    )
    (step / "checkpoint.json").write_bytes(canonical_json_bytes(rebound_checkpoint))
    latest = read_manifest(
        run, LatestIndex, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    rebound_latest = replace(
        latest,
        owning_run_identity=rebound_run.identity,
        checkpoint_identity=rebound_checkpoint.identity,
    )
    rebound_latest = replace(
        rebound_latest,
        identity=rebound_latest.recompute_identity(),
    )
    (run / "latest.json").write_bytes(canonical_json_bytes(rebound_latest))

    with pytest.raises(SMLArtifactError, match="model parameter specs"):
        verify_artifact(run, full=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("accumulation-count", "trainer accumulation must be empty"),
        ("unexpected-accumulator", "trainer keys"),
        ("next-key", "next RNG key"),
    ),
)
def test_full_pretraining_run_rejects_resigned_invalid_current_state(
    tmp_path: Path,
    pretraining_run_template: Path,
    mutation: str,
    message: str,
) -> None:
    """Current-state arrays must agree with the saved checkpoint boundary."""
    run = tmp_path / f"run-{mutation}"
    shutil.copytree(pretraining_run_template, run)
    step, checkpoint = _latest_checkpoint(run)
    trainer_path = step / checkpoint.trainer.payload.logical_path
    arrays = dict(mx.load(trainer_path))
    mx.eval(*arrays.values())
    if mutation == "accumulation-count":
        arrays["accumulation_count"] = mx.array(1, dtype=mx.int32)
    elif mutation == "unexpected-accumulator":
        arrays["accumulators.weight"] = mx.ones((1,), dtype=mx.float32)
    else:
        arrays["next_key"] = mx.random.key(999)
    mx.save_safetensors(trainer_path, arrays)
    arrays = dict(mx.load(trainer_path))
    mx.eval(*arrays.values())
    rebound = replace(
        checkpoint,
        trainer=_array_ref(
            trainer_path,
            checkpoint.trainer.payload.logical_path,
            arrays,
        ),
    )
    rebound = replace(rebound, identity=rebound.recompute_identity())
    (step / "checkpoint.json").write_bytes(canonical_json_bytes(rebound))
    _rebind_latest_checkpoint(run, rebound)

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(run, full=True)


def test_full_pretraining_run_accepts_re_signed_partial_progress(
    tmp_path: Path,
    pretraining_run_template: Path,
) -> None:
    """FULL verification accepts an epoch-ending partial accumulation window."""
    run = tmp_path / "run-partial-progress"
    shutil.copytree(pretraining_run_template, run)
    step, checkpoint = _latest_checkpoint(run)
    state_path = step / checkpoint.scalar_state.logical_path
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["step"] == 1
    assert state["microsteps"] == 2
    state["microsteps"] = 1
    state["rows"] = 1
    state_path.write_bytes(canonical_json_bytes(state))
    checkpoint = replace(
        checkpoint,
        scalar_state=_payload_ref(state_path, checkpoint.scalar_state.logical_path),
    )
    checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
    (step / "checkpoint.json").write_bytes(canonical_json_bytes(checkpoint))
    _rebind_latest_checkpoint(run, checkpoint)

    run_manifest = read_manifest(
        run,
        PretrainingRunManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    trainer_path = step / checkpoint.trainer.payload.logical_path
    trainer = dict(mx.load(trainer_path))
    mx.eval(*trainer.values())
    trainer["next_key"] = semantics_module.expected_next_key(
        seed=int(run_manifest.checkpoint["seed"]),
        microsteps=1,
        model=ModelConfig(**dict(run_manifest.model)),
    )
    mx.save_safetensors(trainer_path, trainer)
    checkpoint = replace(
        checkpoint,
        trainer=_array_ref(
            trainer_path,
            checkpoint.trainer.payload.logical_path,
            trainer,
        ),
    )
    checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
    (step / "checkpoint.json").write_bytes(canonical_json_bytes(checkpoint))
    _rebind_latest_checkpoint(run, checkpoint)

    verified = verify_artifact(run, full=True)
    assert verified.manifest.kind == "pretraining-run"
    assert verified.children[-1].manifest.step == 1


@pytest.mark.parametrize("microsteps", (0, 3))
def test_full_pretraining_run_rejects_progress_outside_step_bounds(
    tmp_path: Path,
    pretraining_run_template: Path,
    microsteps: int,
) -> None:
    """FULL verification rejects progress below or above the accumulation bounds."""
    run = tmp_path / f"run-invalid-progress-{microsteps}"
    shutil.copytree(pretraining_run_template, run)
    step, checkpoint = _latest_checkpoint(run)
    state_path = step / checkpoint.scalar_state.logical_path
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["microsteps"] = microsteps
    state["rows"] = microsteps
    state_path.write_bytes(canonical_json_bytes(state))
    rebound = replace(
        checkpoint,
        scalar_state=_payload_ref(
            state_path,
            checkpoint.scalar_state.logical_path,
        ),
    )
    rebound = replace(rebound, identity=rebound.recompute_identity())
    (step / "checkpoint.json").write_bytes(canonical_json_bytes(rebound))
    _rebind_latest_checkpoint(run, rebound)

    with pytest.raises(SMLArtifactError, match="microsteps disagree"):
        verify_artifact(run, full=True)


@pytest.mark.parametrize("full", (False, True))
def test_recursive_pretraining_run_verification_reports_recovered_latest(
    tmp_path: Path,
    pretraining_run_template: Path,
    full: bool,
) -> None:
    """A proved recovered pretraining winner remains a successful public result."""
    run = tmp_path / f"run-missing-latest-{full}"
    shutil.copytree(pretraining_run_template, run)
    (run / "latest.json").unlink()

    result = verify_artifact(run, full=full)

    assert result.latest_recovered is True
    assert result.pruning_pending is False
