from __future__ import annotations

import dataclasses
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import arrays as arrays_module
from sml.artifacts.manifest import (
    BaseSnapshotManifest,
    ExportManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_manifest,
)
from sml.artifacts.verify import verify_artifact
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.model.language_model import model_parameter_specs
from sml.training.common import (
    PrecisionConfig,
)
from sml.training.lora import LoRAPrecisionConfig

from ._payloads import (
    _PLACEHOLDER_IDENTITY,
    _array_ref,
)


def _write_base_bundle(root: Path) -> tuple[BaseSnapshotManifest, dict[str, mx.array]]:
    root.mkdir()
    model = ModelConfig(
        vocab_size=8,
        hidden_size=4,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=8,
        original_context_length=8,
        hidden_dropout=0.0,
    )
    arrays = {
        spec.name: mx.zeros(spec.shape, dtype=mx.bfloat16)
        for spec in model_parameter_specs(model)
    }
    weights_path = root / "model.safetensors"
    mx.save_safetensors(weights_path, arrays)
    manifest = BaseSnapshotManifest(
        kind="base-snapshot",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dataclasses.asdict(model),
        precision=dataclasses.asdict(PrecisionConfig()),
        tokenizer_identity="sha256:" + "1" * 64,
        working_weights=_array_ref(weights_path, "model.safetensors", arrays),
        diagnostic_source_run_identity="sha256:" + "2" * 64,
        diagnostic_source_step=0,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest, arrays


def _write_export_bundle(
    root: Path, tokenizer_root: Path, *, rope_scaling_factor: float = 1.0
) -> ExportManifest:
    root.mkdir()
    shutil.copytree(tokenizer_root, root / "tokenizer")
    tokenizer = read_manifest(
        root / "tokenizer",
        TokenizerManifest,
        VerificationLevel.FULL,
    ).manifest
    model = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=4,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=8,
        original_context_length=8,
        rope_scaling_factor=rope_scaling_factor,
        hidden_dropout=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )
    arrays = {
        spec.name: mx.zeros(spec.shape, dtype=mx.bfloat16)
        for spec in model_parameter_specs(model)
    }
    weights_path = root / "model.safetensors"
    mx.save_safetensors(weights_path, arrays)
    manifest = ExportManifest(
        kind="export",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dataclasses.asdict(model),
        precision=dataclasses.asdict(LoRAPrecisionConfig()),
        tokenizer_identity=tokenizer.identity,
        model_weights=_array_ref(weights_path, "model.safetensors", arrays),
        tokenizer_model=replace(
            tokenizer.model,
            logical_path="tokenizer/tokenizer.model",
        ),
        tokenizer_vocab=replace(
            tokenizer.vocab,
            logical_path="tokenizer/tokenizer.vocab",
        ),
        diagnostic_source_run_identity="sha256:" + "1" * 64,
        diagnostic_source_step=0,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def test_full_base_verification_is_metadata_only(tmp_path, monkeypatch):
    """Standalone FULL verification must not materialize a live model copy."""
    root = tmp_path / "base"
    _write_base_bundle(root)

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("model materialization is forbidden in verification")

    monkeypatch.setattr(arrays_module.mx, "load", reject_materialization)

    result = verify_artifact(root, full=True)

    assert result.verification is VerificationLevel.FULL


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing-leaf", "model parameter specs"),
        ("wrong-dtype", "model parameter specs"),
        ("wrong-shape", "model parameter specs"),
        ("invalid-config", "invalid base snapshot model configuration"),
        ("scaled-rope", "base snapshot rope_scaling_factor must be exactly 1.0"),
    ),
)
def test_full_base_rejects_resigned_semantic_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Base FULL requires the exact BF16 plain-model tree and canonical config."""
    root = tmp_path / f"base-{mutation}"
    manifest, arrays = _write_base_bundle(root)
    if mutation in ("invalid-config", "scaled-rope"):
        model = dict(manifest.model)
        if mutation == "scaled-rope":
            model["rope_scaling_factor"] = 8.0
        else:
            model["hidden_size"] = 0
        manifest = replace(manifest, model=model)
    else:
        name = next(iter(arrays))
        if mutation == "missing-leaf":
            arrays.pop(name)
        elif mutation == "wrong-dtype":
            arrays[name] = arrays[name].astype(mx.float32)
        else:
            arrays[name] = mx.zeros((*arrays[name].shape, 1), dtype=mx.bfloat16)
        weights_path = root / manifest.working_weights.payload.logical_path
        mx.save_safetensors(weights_path, arrays)
        manifest = replace(
            manifest,
            working_weights=_array_ref(
                weights_path,
                manifest.working_weights.payload.logical_path,
                arrays,
            ),
        )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)


def test_full_base_rejects_resigned_extra_model_leaf(tmp_path: Path) -> None:
    """Byte-valid extra leaves must not satisfy the exact plain-model contract."""
    root = tmp_path / "base"
    root.mkdir()
    model = ModelConfig(
        vocab_size=8,
        hidden_size=4,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=8,
        original_context_length=8,
        hidden_dropout=0.0,
    )
    arrays = {
        spec.name: mx.zeros(spec.shape, dtype=mx.bfloat16)
        for spec in model_parameter_specs(model)
    }
    arrays["unexpected.weight"] = mx.zeros((1,), dtype=mx.bfloat16)
    weights_path = root / "model.safetensors"
    mx.save_safetensors(weights_path, arrays)
    manifest = BaseSnapshotManifest(
        kind="base-snapshot",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model=dataclasses.asdict(model),
        precision=dataclasses.asdict(PrecisionConfig()),
        tokenizer_identity="sha256:" + "1" * 64,
        working_weights=_array_ref(weights_path, "model.safetensors", arrays),
        diagnostic_source_run_identity="sha256:" + "2" * 64,
        diagnostic_source_step=0,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="model parameter specs"):
        verify_artifact(root, full=True)


@pytest.mark.parametrize("rope_scaling_factor", (1.0, 2.0, 8.0))
def test_full_export_verification_is_metadata_only(
    tmp_path, prepared_template, monkeypatch, rope_scaling_factor
):
    """Export FULL verification must validate leaves without loading their values."""
    root = tmp_path / "export"
    _write_export_bundle(
        root,
        prepared_template / "tokenizer",
        rope_scaling_factor=rope_scaling_factor,
    )

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("model materialization is forbidden in verification")

    monkeypatch.setattr(arrays_module.mx, "load", reject_materialization)

    result = verify_artifact(root, full=True)

    assert result.verification is VerificationLevel.FULL
    assert result.manifest.model["rope_scaling_factor"] == rope_scaling_factor


@pytest.mark.parametrize(
    "field",
    (
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ),
)
def test_full_export_rejects_resigned_tokenizer_model_disagreement(
    tmp_path: Path,
    prepared_template: Path,
    field: str,
) -> None:
    """Export FULL must compare exact vocabulary and special-ID metadata."""
    root = tmp_path / f"export-{field}"
    manifest = _write_export_bundle(root, prepared_template / "tokenizer")
    model = dict(manifest.model)
    if field == "vocab_size":
        model[field] = int(model[field]) + 1
    else:
        used = {
            int(model[name])
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        }
        model[field] = next(
            token_id
            for token_id in range(int(model["vocab_size"]))
            if token_id not in used
        )
    resigned = replace(manifest, model=model)
    resigned = replace(resigned, identity=resigned.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(
        SMLArtifactError,
        match="export tokenizer metadata does not match model",
    ):
        verify_artifact(root, full=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("outer-tokenizer-mismatch", "owned tokenizer identity"),
        ("inner-tokenizer-mismatch", "payload references"),
        ("adapter-only-leaf", "model parameter specs"),
        ("wrong-dtype", "model parameter specs"),
        ("wrong-shape", "model parameter specs"),
    ),
)
def test_full_export_rejects_resigned_tree_or_tokenizer_corruption(
    tmp_path: Path,
    prepared_template: Path,
    mutation: str,
    message: str,
) -> None:
    """Export FULL binds its child and permits only exact plain-model BF16 leaves."""
    root = tmp_path / f"export-{mutation}"
    manifest = _write_export_bundle(root, prepared_template / "tokenizer")
    if mutation == "outer-tokenizer-mismatch":
        manifest = replace(manifest, tokenizer_identity="sha256:" + "9" * 64)
    elif mutation == "inner-tokenizer-mismatch":
        manifest = replace(
            manifest,
            tokenizer_model=replace(
                manifest.tokenizer_model,
                identity="sha256:" + "8" * 64,
            ),
        )
    else:
        weights_path = root / manifest.model_weights.payload.logical_path
        arrays = dict(mx.load(weights_path))
        mx.eval(*arrays.values())
        name = next(iter(arrays))
        if mutation == "adapter-only-leaf":
            arrays[f"{name}.lora_a"] = mx.zeros((1,), dtype=mx.float32)
        elif mutation == "wrong-dtype":
            arrays[name] = arrays[name].astype(mx.float32)
        else:
            arrays[name] = mx.zeros((*arrays[name].shape, 1), dtype=mx.bfloat16)
        mx.save_safetensors(weights_path, arrays)
        manifest = replace(
            manifest,
            model_weights=_array_ref(
                weights_path,
                manifest.model_weights.payload.logical_path,
                arrays,
            ),
        )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)
