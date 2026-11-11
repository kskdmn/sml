from __future__ import annotations

import dataclasses
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import manifest as manifest_module
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    BaseSnapshotManifest,
    ExportManifest,
    LatestIndex,
    LoRACheckpointManifest,
    LoRARunManifest,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError

from ._manifests import (
    IDENTITY_A,
    IDENTITY_B,
    _write_manifest,
    _write_tokenizer_artifact,
    tokenizer_manifest_fixture,
)

IDENTITY_C = "sha256:" + "c" * 64


def _array_payload(
    path: str = "model.safetensors", *, dtype: str = "float32"
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        payload=PayloadRef(path, IDENTITY_A, 100),
        arrays=(ArraySpec("weight", (2, 3), dtype),),
    )


def manifest_fixtures() -> tuple[object, ...]:
    return (
        tokenizer_manifest_fixture(),
        PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity=IDENTITY_A,
            sequence_length=3,
            row_width=4,
            dtype="int32",
            shard_row_counts=(2,),
            shards=(PayloadRef("shards/train-000000.npy", IDENTITY_A, 128),),
            preparation_seed=1729,
            row_order_policy={"kind": "windowed-row-shuffle-v1", "rows": 32},
            tokenizer_identity=IDENTITY_B,
            tokenizer_model=PayloadRef("tokenizer/tokenizer.model", IDENTITY_B, 10),
            tokenizer_vocab=PayloadRef("tokenizer/tokenizer.vocab", IDENTITY_C, 20),
            source_summary={"files": 2, "documents": 5},
            diagnostic_source_locator="/corpus",
            row_content_identity=IDENTITY_C,
        ),
        PretrainingCheckpointManifest(
            kind="pretraining-checkpoint",
            version=1,
            identity=IDENTITY_A,
            owning_run_identity=IDENTITY_B,
            step=4,
            scalar_state=PayloadRef("state.json", IDENTITY_C, 30),
            model=_array_payload(dtype="bfloat16"),
            master=_array_payload("master.safetensors"),
            optimizer=_array_payload("optimizer.safetensors"),
            trainer=_array_payload("trainer.safetensors"),
        ),
        LoRACheckpointManifest(
            kind="lora-checkpoint",
            version=1,
            identity=IDENTITY_A,
            owning_run_identity=IDENTITY_B,
            step=4,
            scalar_state=PayloadRef("state.json", IDENTITY_C, 30),
            adapters=_array_payload("adapters.safetensors"),
            optimizer=_array_payload("optimizer.safetensors"),
            trainer=_array_payload("trainer.safetensors"),
        ),
        PretrainingRunManifest(
            kind="pretraining-run",
            version=1,
            identity=IDENTITY_A,
            model={"rope_scaling_factor": 1.0, "hidden_size": 8},
            precision={"compute": "bfloat16", "master": "float32"},
            optimizer={"kind": "adam"},
            loader={"batch_size": 1},
            checkpoint={
                "interval": 5,
                "rng_schedule": "counter-addressed-forward-terminal-v1",
            },
            tokenizer_identity=IDENTITY_B,
            data_identity=IDENTITY_C,
            diagnostic_data_locator="/data",
        ),
        LoRARunManifest(
            kind="lora-run",
            version=1,
            identity=IDENTITY_A,
            model={"rope_scaling_factor": 1.0, "hidden_size": 8},
            lora={"rank": 16},
            precision={"adapter": "float32"},
            optimizer={"kind": "adam"},
            loader={"batch_size": 1},
            checkpoint={
                "interval": 5,
                "rng_schedule": "counter-addressed-forward-terminal-v1",
            },
            tokenizer_identity=IDENTITY_B,
            base_identity=IDENTITY_A,
            data_identity=IDENTITY_C,
            diagnostic_data_locator="/swag-data",
        ),
        LatestIndex(
            kind="latest-index",
            version=1,
            identity=IDENTITY_A,
            owning_run_identity=IDENTITY_B,
            step=4,
            checkpoint_identity=IDENTITY_C,
        ),
        BaseSnapshotManifest(
            kind="base-snapshot",
            version=1,
            identity=IDENTITY_A,
            model={"rope_scaling_factor": 1.0},
            precision={"working": "bfloat16"},
            tokenizer_identity=IDENTITY_B,
            working_weights=_array_payload(),
            diagnostic_source_run_identity=IDENTITY_C,
            diagnostic_source_step=4,
        ),
        SwagDataManifest(
            kind="swag-data",
            version=1,
            identity=IDENTITY_A,
            source={"revision": "immutable-commit"},
            preprocessing={"maximum_length": 64},
            base_identity=IDENTITY_B,
            tokenizer_identity=IDENTITY_C,
            vocab_size=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
            unk_token_id=0,
            example_count=2,
            dropped_overlength_rows=1,
            buckets=(_array_payload("buckets/length-0064/arrays.safetensors"),),
        ),
        ExportManifest(
            kind="export",
            version=1,
            identity=IDENTITY_A,
            model={"rope_scaling_factor": 1.0},
            precision={"working": "bfloat16"},
            tokenizer_identity=IDENTITY_B,
            model_weights=_array_payload(),
            tokenizer_model=PayloadRef("tokenizer/tokenizer.model", IDENTITY_B, 10),
            tokenizer_vocab=PayloadRef("tokenizer/tokenizer.vocab", IDENTITY_C, 20),
            diagnostic_source_run_identity=IDENTITY_C,
            diagnostic_source_step=4,
        ),
    )


def _materialize_payloads(root: Path, value: object) -> object:
    if isinstance(value, PayloadRef):
        data = f"payload:{value.logical_path}".encode()
        path = root.joinpath(*value.logical_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return replace(
            value,
            identity=file_identity(io.BytesIO(data)),
            byte_size=len(data),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _materialize_payloads(root, getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(_materialize_payloads(root, item) for item in value)
    return value


@pytest.mark.parametrize(
    ("manifest_index", "manifest_type", "foreign_field"),
    [
        (2, PretrainingCheckpointManifest, "adapters"),
        (4, PretrainingRunManifest, "base_identity"),
    ],
)
def test_pretraining_schemas_reject_lora_only_fields(
    tmp_path, manifest_index, manifest_type, foreign_field
):
    """A foreign-kind field must fail before it can alter a frozen schema."""
    raw = json.loads(canonical_json_bytes(manifest_fixtures()[manifest_index]))
    raw[foreign_field] = raw.get("model", raw.get("master"))
    (tmp_path / manifest_type.MANIFEST_FILENAME).write_text(
        json.dumps(raw), encoding="utf-8"
    )

    with pytest.raises(SMLArtifactError, match=f"unknown field.*{foreign_field}"):
        read_manifest(tmp_path, manifest_type, VerificationLevel.MANIFEST_TRUSTED)


def test_sequence_manifests_reject_exact_duplicate_payload_paths():
    """Repeated shards or checkpoint groups must not acquire order semantics."""
    shard = PayloadRef("shards/train-000000.npy", IDENTITY_A, 128)
    with pytest.raises((SMLArtifactError, ValueError), match="duplicate.*payload"):
        PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity=IDENTITY_A,
            sequence_length=3,
            row_width=4,
            dtype="int32",
            shard_row_counts=(2, 2),
            shards=(shard, shard),
            preparation_seed=1729,
            row_order_policy={"kind": "windowed-row-shuffle-v1", "rows": 32},
            tokenizer_identity=IDENTITY_B,
            tokenizer_model=PayloadRef("tokenizer/tokenizer.model", IDENTITY_B, 10),
            tokenizer_vocab=PayloadRef("tokenizer/tokenizer.vocab", IDENTITY_C, 20),
            source_summary={"files": 2, "documents": 5},
            diagnostic_source_locator="/corpus",
            row_content_identity=IDENTITY_C,
        )

    group = _array_payload()
    with pytest.raises(SMLArtifactError, match="duplicate.*payload"):
        PretrainingCheckpointManifest(
            kind="pretraining-checkpoint",
            version=1,
            identity=IDENTITY_A,
            owning_run_identity=IDENTITY_B,
            step=4,
            scalar_state=PayloadRef("state.json", IDENTITY_C, 30),
            model=group,
            master=group,
            optimizer=_array_payload("optimizer.safetensors"),
            trainer=_array_payload("trainer.safetensors"),
        )


@pytest.mark.parametrize(
    ("manifest", "version"),
    [
        pytest.param(manifest, version, id=f"{manifest.kind}-v{version}")
        for manifest in manifest_fixtures()
        for version in (
            (1, 2)
            if isinstance(
                manifest, (PretrainingCheckpointManifest, LoRACheckpointManifest)
            )
            else (1,)
        )
    ],
)
def test_strict_manifest_parser_round_trips_supported_schemas(
    tmp_path, manifest, version
):
    manifest = _materialize_payloads(tmp_path, manifest)
    original_identity = manifest.recompute_identity()
    manifest = replace(manifest, version=version)
    expected = _write_manifest(tmp_path, manifest, manifest.MANIFEST_FILENAME)

    verified = read_manifest(
        tmp_path, type(manifest), VerificationLevel.MANIFEST_TRUSTED
    )

    assert verified.manifest == expected
    assert verified.verification is VerificationLevel.MANIFEST_TRUSTED
    if isinstance(manifest, (PretrainingCheckpointManifest, LoRACheckpointManifest)):
        assert (expected.identity == original_identity) is (version == 1)
        with pytest.raises(ValueError, match="version"):
            replace(manifest, version=3)


@pytest.mark.parametrize("recursive", [False, True], ids=("top-level", "child"))
def test_manifest_reader_rejects_noncanonical_json_encoding(tmp_path, recursive):
    """A digest-valid manifest must still use the one canonical wire encoding."""
    root = tmp_path / "bundle"
    _write_tokenizer_artifact(root)
    target = root
    if recursive:
        target = root / "tokenizer"
        _write_tokenizer_artifact(target)

    raw = json.loads((target / "manifest.json").read_bytes())
    (target / "manifest.json").write_text(
        json.dumps(raw, indent=2, sort_keys=False), encoding="utf-8"
    )

    if recursive:
        with (
            manifest_module.open_artifact(
                root, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
            ) as artifact,
            pytest.raises(SMLArtifactError, match="canonical"),
        ):
            artifact.open_child("tokenizer", (TokenizerManifest,))
    else:
        with pytest.raises(SMLArtifactError, match="canonical"):
            manifest_module.open_artifact(
                root, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
            )


def test_manifest_type_validation_rejects_invalid_inputs(tmp_path, monkeypatch):
    """Dispatch must reject malformed type sets before opening an artifact root."""
    with pytest.raises(TypeError, match="tuple"):
        manifest_module.open_artifact(  # type: ignore[arg-type]
            tmp_path, [TokenizerManifest], VerificationLevel.MANIFEST_TRUSTED
        )
    for invalid in ((), (object,)):
        with pytest.raises(SMLArtifactError, match="unsupported manifest types"):
            manifest_module.open_artifact(
                tmp_path, invalid, VerificationLevel.MANIFEST_TRUSTED
            )

    monkeypatch.setattr(PretrainingDataManifest, "MANIFEST_FILENAME", "data.json")
    with pytest.raises(SMLArtifactError, match="share one filename"):
        manifest_module._validated_manifest_types(
            (TokenizerManifest, PretrainingDataManifest)
        )
    monkeypatch.setattr(PretrainingDataManifest, "MANIFEST_FILENAME", "manifest.json")
    monkeypatch.setattr(
        PretrainingDataManifest, "EXPECTED_KIND", TokenizerManifest.EXPECTED_KIND
    )
    with pytest.raises(SMLArtifactError, match="unique kinds"):
        manifest_module._validated_manifest_types(
            (TokenizerManifest, PretrainingDataManifest)
        )


@pytest.mark.parametrize("mutation", ["unknown", "missing", "nested-unknown"])
def test_strict_manifest_parser_rejects_schema_field_mutations(tmp_path, mutation):
    """Ignoring an unknown or missing field would silently reinterpret another schema."""
    manifest = tokenizer_manifest_fixture()
    raw = json.loads(canonical_json_bytes(manifest))
    if mutation == "unknown":
        raw["legacy_format"] = 1
        message = "unknown field.*legacy_format"
    elif mutation == "missing":
        del raw["algorithm"]
        message = "missing field.*algorithm"
    else:
        raw["model"]["legacy_format"] = 1
        message = "unknown field.*legacy_format"
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SMLArtifactError, match=message):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "legacy-tokenizer", "kind"),
        ("version", 2, "version"),
        ("identity", "SHA256:" + "a" * 64, "identity"),
    ],
)
def test_strict_manifest_parser_rejects_discriminator_and_identity_mutations(
    tmp_path, field, value, message
):
    """Accepting alternate discriminators or digest syntax would split the schema."""
    raw = json.loads(canonical_json_bytes(tokenizer_manifest_fixture()))
    raw[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SMLArtifactError, match=message):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED)


def test_strict_manifest_parser_rejects_forged_manifest_identity(tmp_path):
    """Trusting the stored digest would allow semantic manifest edits to pass."""
    manifest = tokenizer_manifest_fixture()
    _write_manifest(tmp_path, manifest, "manifest.json")
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw["vocab_size"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(SMLArtifactError, match="manifest identity"):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED)


@pytest.mark.parametrize("run_type", (PretrainingRunManifest, LoRARunManifest))
@pytest.mark.parametrize("schedule", (None, "unknown-schedule"))
def test_run_manifest_requires_exact_rng_schedule(run_type, schedule):
    manifest = next(
        value for value in manifest_fixtures() if isinstance(value, run_type)
    )
    checkpoint = dict(manifest.checkpoint)
    if schedule is None:
        checkpoint.pop("rng_schedule")
    else:
        checkpoint["rng_schedule"] = schedule
    with pytest.raises(ValueError, match="rng_schedule"):
        replace(manifest, checkpoint=checkpoint)


def test_manifest_constructor_rejects_boolean_schema_version():
    """Treating True as version 1 would admit a noncanonical discriminator type."""
    with pytest.raises(ValueError, match="version"):
        tokenizer_manifest_fixture(version=True)


def test_pretraining_run_manifest_rejects_noncanonical_rope_factor():
    """Relaxing run lineage or base RoPE would make resume semantics ambiguous."""
    with pytest.raises(ValueError, match="rope_scaling_factor"):
        PretrainingRunManifest(
            kind="pretraining-run",
            version=1,
            identity=IDENTITY_A,
            model={"rope_scaling_factor": 2.0},
            precision={},
            optimizer={},
            loader={},
            checkpoint={
                "rng_schedule": "counter-addressed-forward-terminal-v1",
            },
            tokenizer_identity=IDENTITY_B,
            data_identity=IDENTITY_C,
            diagnostic_data_locator=None,
        )


@pytest.mark.parametrize("rope_scaling_factor", (1.0, 2.0, 8.0))
def test_lora_run_manifest_preserves_rope_factor(tmp_path, rope_scaling_factor):
    manifest = next(
        value for value in manifest_fixtures() if isinstance(value, LoRARunManifest)
    )
    scaled = replace(
        manifest,
        model={**dict(manifest.model), "rope_scaling_factor": rope_scaling_factor},
    )
    written = _write_manifest(tmp_path, scaled, "run.json")
    restored = read_manifest(
        tmp_path, LoRARunManifest, VerificationLevel.MANIFEST_TRUSTED
    ).manifest
    assert restored.identity == written.identity
    assert restored.model["rope_scaling_factor"] == rope_scaling_factor
    if rope_scaling_factor != 1.0:
        assert restored.identity != manifest.recompute_identity()


@pytest.mark.parametrize("rope_scaling_factor", (None, True, 8, "8.0", 0.0, 0.5))
def test_lora_run_manifest_rejects_invalid_rope_factor(rope_scaling_factor):
    manifest = next(
        value for value in manifest_fixtures() if isinstance(value, LoRARunManifest)
    )
    with pytest.raises(ValueError, match="rope_scaling_factor"):
        replace(
            manifest,
            model={**dict(manifest.model), "rope_scaling_factor": rope_scaling_factor},
        )
