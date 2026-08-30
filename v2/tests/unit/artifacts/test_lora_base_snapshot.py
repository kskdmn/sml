from __future__ import annotations

from dataclasses import replace

import pytest
from sml.artifacts.checkpoint import require_lora_base_snapshot
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    BaseSnapshotManifest,
    LoRARunManifest,
    PayloadRef,
)
from sml.errors import SMLArtifactError

IDENTITY_A = "sha256:" + "a" * 64
IDENTITY_B = "sha256:" + "b" * 64
IDENTITY_C = "sha256:" + "c" * 64
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64

_CANONICAL_PRECISION = {
    "master_parameter_dtype": "float32",
    "working_parameter_dtype": "bfloat16",
    "gradient_accumulator_dtype": "float32",
    "optimizer_state_dtype": "float32",
    "update_dtype": "float32",
    "master_weights": True,
    "dynamic_loss_scaling": False,
}
_MODEL = {"rope_scaling_factor": 1.0, "hidden_size": 8}


def _array_payload(path: str = "model.safetensors") -> ArrayPayloadRef:
    return ArrayPayloadRef(
        payload=PayloadRef(path, IDENTITY_A, 100),
        arrays=(ArraySpec("weight", (2, 3), "bfloat16"),),
    )


def _matching_snapshot_and_run() -> tuple[BaseSnapshotManifest, LoRARunManifest]:
    snapshot = BaseSnapshotManifest(
        kind="base-snapshot",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dict(_MODEL),
        precision=dict(_CANONICAL_PRECISION),
        tokenizer_identity=IDENTITY_B,
        working_weights=_array_payload(),
        diagnostic_source_run_identity=IDENTITY_C,
        diagnostic_source_step=0,
    )
    snapshot = replace(snapshot, identity=snapshot.recompute_identity())
    run = LoRARunManifest(
        kind="lora-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dict(_MODEL),
        lora={"rank": 16},
        precision={"adapter_parameter_dtype": "float32"},
        optimizer={"kind": "adam"},
        loader={"batch_size": 1},
        checkpoint={
            "interval": 5,
            "rng_schedule": "counter-addressed-forward-terminal-v1",
        },
        tokenizer_identity=IDENTITY_B,
        base_identity=snapshot.identity,
        data_identity=IDENTITY_C,
        diagnostic_data_locator="/swag-data",
    )
    run = replace(run, identity=run.recompute_identity())
    return snapshot, run


def test_require_lora_base_snapshot_accepts_matching_canonical_copy() -> None:
    snapshot, run = _matching_snapshot_and_run()
    require_lora_base_snapshot(snapshot, run)


def test_require_lora_base_snapshot_rejects_model_mismatch() -> None:
    snapshot, run = _matching_snapshot_and_run()
    snapshot = replace(snapshot, model={**dict(snapshot.model), "hidden_size": 16})
    snapshot = replace(snapshot, identity=snapshot.recompute_identity())
    run = replace(run, base_identity=snapshot.identity)
    run = replace(run, identity=run.recompute_identity())
    with pytest.raises(SMLArtifactError, match="model configuration"):
        require_lora_base_snapshot(snapshot, run)


def test_require_lora_base_snapshot_rejects_noncanonical_precision() -> None:
    snapshot, run = _matching_snapshot_and_run()
    precision = dict(snapshot.precision)
    precision["working_parameter_dtype"] = "float32"
    snapshot = replace(snapshot, precision=precision)
    snapshot = replace(snapshot, identity=snapshot.recompute_identity())
    run = replace(run, base_identity=snapshot.identity)
    run = replace(run, identity=run.recompute_identity())
    with pytest.raises(SMLArtifactError, match="canonical pretraining precision"):
        require_lora_base_snapshot(snapshot, run)


def test_require_lora_base_snapshot_rejects_tokenizer_mismatch() -> None:
    snapshot, run = _matching_snapshot_and_run()
    snapshot = replace(snapshot, tokenizer_identity=IDENTITY_A)
    snapshot = replace(snapshot, identity=snapshot.recompute_identity())
    run = replace(run, base_identity=snapshot.identity)
    run = replace(run, identity=run.recompute_identity())
    with pytest.raises(SMLArtifactError, match="tokenizer identity"):
        require_lora_base_snapshot(snapshot, run)
