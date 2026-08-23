from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts.checkpoint import resolve_exact_step
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
)
from sml.artifacts.verify import verify_artifact
from sml.errors import SMLArtifactError

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _array_ref(
    path: Path,
    logical_path: str,
    arrays: dict[str, mx.array],
    *,
    declared_shape: tuple[int, ...] | None = None,
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        tuple(
            ArraySpec(
                name,
                declared_shape if declared_shape is not None else tuple(array.shape),
                {
                    mx.bfloat16: "bfloat16",
                    mx.float32: "float32",
                    mx.int32: "int32",
                    mx.uint32: "uint32",
                }[array.dtype],
            )
            for name, array in sorted(arrays.items())
        ),
    )


def test_full_resolution_rejects_false_safetensors_metadata(tmp_path: Path) -> None:
    """Byte-valid payloads must not earn FULL when their array declaration is false."""
    run = tmp_path / "run"
    step_directory = run / "checkpoints" / "step-000000000"
    step_directory.mkdir(parents=True)
    run_manifest = PretrainingRunManifest(
        kind="pretraining-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model={"rope_scaling_factor": 1.0},
        precision={"working_parameter_dtype": "bfloat16"},
        optimizer={"kind": "adamw"},
        loader={"microbatch_size": 1},
        checkpoint={"interval": 1},
        tokenizer_identity="sha256:" + "1" * 64,
        data_identity="sha256:" + "2" * 64,
        diagnostic_data_locator=None,
    )
    run_manifest = replace(run_manifest, identity=run_manifest.recompute_identity())
    (run / "run.json").write_bytes(canonical_json_bytes(run_manifest))

    groups = {
        "model.safetensors": {"weight": mx.array([1.0], dtype=mx.bfloat16)},
        "master.safetensors": {"weight": mx.array([1.0], dtype=mx.float32)},
        "optimizer.safetensors": {
            "step": mx.array(0, dtype=mx.int32),
            "first_moments.weight": mx.array([0.0], dtype=mx.float32),
            "second_moments.weight": mx.array([0.0], dtype=mx.float32),
        },
        "trainer.safetensors": {
            "accumulation_count": mx.array(0, dtype=mx.int32),
            "next_key": mx.random.key(7),
            "loss_numerator": mx.array(0.0, dtype=mx.float32),
            "accumulators.weight": mx.array([0.0], dtype=mx.float32),
        },
    }
    references = {}
    for logical_path, arrays in groups.items():
        path = step_directory / logical_path
        mx.save_safetensors(path, arrays)
        references[logical_path] = _array_ref(
            path,
            logical_path,
            arrays,
            declared_shape=(2,) if logical_path == "model.safetensors" else None,
        )
    state_path = step_directory / "state.json"
    state_path.write_bytes(
        canonical_json_bytes(
            {
                "kind": "pretraining-state",
                "version": 1,
                "owning_run_identity": run_manifest.identity,
                "step": 0,
                "rows": 0,
                "microsteps": 0,
                "cursor": {
                    "epoch": 0,
                    "shard_order_position": 0,
                    "row_offset": 0,
                },
            }
        )
    )
    manifest = PretrainingCheckpointManifest(
        kind="pretraining-checkpoint",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        owning_run_identity=run_manifest.identity,
        step=0,
        scalar_state=_payload_ref(state_path, "state.json"),
        model=references["model.safetensors"],
        master=references["master.safetensors"],
        optimizer=references["optimizer.safetensors"],
        trainer=references["trainer.safetensors"],
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="array metadata.*model.safetensors"):
        resolve_exact_step(run, step=0, verification=VerificationLevel.FULL)

    corrected_model = _array_ref(
        step_directory / "model.safetensors",
        "model.safetensors",
        groups["model.safetensors"],
    )
    mismatched_master = {"weight": mx.array([2.0], dtype=mx.float32)}
    master_path = step_directory / "master.safetensors"
    mx.save_safetensors(master_path, mismatched_master)
    manifest = replace(
        manifest,
        identity=_PLACEHOLDER_IDENTITY,
        model=corrected_model,
        master=_array_ref(master_path, "master.safetensors", mismatched_master),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="exact BF16 cast"):
        verify_artifact(run, full=True)
