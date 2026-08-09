from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
from dataclasses import fields, replace
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    BaseSnapshotManifest,
    CheckpointManifest,
    ExportManifest,
    LatestIndex,
    PayloadRef,
    PretrainingDataManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
    row_content_identity,
    structured_identity,
)
from sml.errors import SMLArtifactError

IDENTITY_A = "sha256:" + "a" * 64
IDENTITY_B = "sha256:" + "b" * 64
IDENTITY_C = "sha256:" + "c" * 64


class ExampleEnum(Enum):
    VALUE = "enum-value"


@dataclasses.dataclass(frozen=True)
class CanonicalExample:
    path: Path
    number: np.int64


def tokenizer_manifest_fixture(**overrides: object) -> TokenizerManifest:
    values: dict[str, object] = {
        "kind": "tokenizer",
        "version": 1,
        "identity": IDENTITY_A,
        "algorithm": "sentencepiece-bpe-v1",
        "training": {"normalization": "nmt_nfkc", "byte_fallback": True},
        "vocab_size": 256,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "unk_token_id": 0,
        "model": PayloadRef("tokenizer.model", IDENTITY_A, 10),
        "vocab": PayloadRef("tokenizer.vocab", IDENTITY_B, 20),
        "diagnostic_source_locator": "/source/tokenizer",
    }
    values.update(overrides)
    return TokenizerManifest(**values)


def _array_payload(path: str = "model.safetensors") -> ArrayPayloadRef:
    return ArrayPayloadRef(
        payload=PayloadRef(path, IDENTITY_A, 100),
        arrays=(ArraySpec("weight", (2, 3), "float32"),),
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
        CheckpointManifest(
            kind="checkpoint",
            version=1,
            identity=IDENTITY_A,
            owning_run_identity=IDENTITY_B,
            checkpoint_kind="pretraining",
            step=4,
            scalar_state=PayloadRef("state.json", IDENTITY_C, 30),
            arrays=(_array_payload(),),
        ),
        RunManifest(
            kind="run",
            version=1,
            identity=IDENTITY_A,
            run_kind="pretraining",
            model={"rope_scaling_factor": 1.0, "hidden_size": 8},
            precision={"compute": "bfloat16", "master": "float32"},
            optimizer={"kind": "adam"},
            loader={"batch_size": 1},
            checkpoint={"interval": 5},
            tokenizer_identity=IDENTITY_B,
            base_identity=None,
            data_identity=IDENTITY_C,
            diagnostic_data_locator="/data",
            diagnostic_source_locator="/source-run",
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


def _write_manifest(root: Path, manifest: object, filename: str) -> object:
    identified = replace(manifest, identity=manifest.recompute_identity())
    (root / filename).write_bytes(canonical_json_bytes(identified))
    return identified


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


def test_sml_json_v1_identity_vectors_are_stable():
    """Changing normalization of tuples, Unicode, or signed zero breaks identities."""
    left = {"z": -0.0, "a": ("雪", 1, 1.0)}
    right = {"a": ["雪", 1, 1.0], "z": 0.0}
    expected = b'{"a":["\xe9\x9b\xaa",1,1.0],"z":0}'

    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected


def test_canonical_identity_normalizes_supported_schema_values():
    """Dropping schema-value normalization would make equivalent configs diverge."""
    value = {
        "record": CanonicalExample(Path("nested/file"), np.int64(7)),
        "enum": ExampleEnum.VALUE,
        "tuple": (np.float32(1.5),),
    }

    assert canonical_json_bytes(value) == (
        b'{"enum":"enum-value","record":{"number":7,"path":"nested/file"},'
        b'"tuple":[1.5]}'
    )


@pytest.mark.parametrize(
    "value",
    [math.inf, -math.inf, math.nan, "\ud800", {"\udfff": "invalid"}],
)
def test_canonical_identity_rejects_nonfinite_numbers_and_surrogates(value):
    """Accepting unstable numeric or invalid UTF-8 values would create bad manifests."""
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        canonical_json_bytes({"value": value})


def test_file_and_structured_identity_use_exact_domain_separated_bytes():
    """Removing the domain separator would allow identities from unlike contracts to alias."""
    payload = b"exact\x00bytes"
    file_digest = hashlib.sha256(payload).hexdigest()
    structured_digest = hashlib.sha256(b'example-domain\0{"x":1}').hexdigest()

    assert file_identity(io.BytesIO(payload)) == f"sha256:{file_digest}"
    assert structured_identity("example-domain", {"x": 1}) == (
        f"sha256:{structured_digest}"
    )


def test_manifest_identity_ignores_diagnostic_locator_but_not_payload():
    """Including diagnostic paths or omitting payload IDs would destroy portability."""
    first = tokenizer_manifest_fixture(
        diagnostic_source_locator="/old/path",
        model=PayloadRef("tokenizer.model", IDENTITY_A, 10),
    )
    moved = replace(first, diagnostic_source_locator="/new/path")
    changed = replace(first, model=replace(first.model, identity=IDENTITY_B))

    assert first.recompute_identity() == moved.recompute_identity()
    assert first.recompute_identity() != changed.recompute_identity()


def test_all_manifest_outer_field_sets_are_frozen():
    """Adding or dropping an outer field would fork the version-1 artifact contract."""
    expected = {
        TokenizerManifest: {
            "kind",
            "version",
            "identity",
            "algorithm",
            "training",
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "model",
            "vocab",
            "diagnostic_source_locator",
        },
        PretrainingDataManifest: {
            "kind",
            "version",
            "identity",
            "sequence_length",
            "row_width",
            "dtype",
            "shard_row_counts",
            "shards",
            "preparation_seed",
            "row_order_policy",
            "tokenizer_identity",
            "tokenizer_model",
            "tokenizer_vocab",
            "source_summary",
            "diagnostic_source_locator",
            "row_content_identity",
        },
        CheckpointManifest: {
            "kind",
            "version",
            "identity",
            "owning_run_identity",
            "checkpoint_kind",
            "step",
            "scalar_state",
            "arrays",
        },
        RunManifest: {
            "kind",
            "version",
            "identity",
            "run_kind",
            "model",
            "precision",
            "optimizer",
            "loader",
            "checkpoint",
            "tokenizer_identity",
            "base_identity",
            "data_identity",
            "diagnostic_data_locator",
            "diagnostic_source_locator",
        },
        LatestIndex: {
            "kind",
            "version",
            "identity",
            "owning_run_identity",
            "step",
            "checkpoint_identity",
        },
        BaseSnapshotManifest: {
            "kind",
            "version",
            "identity",
            "model",
            "precision",
            "tokenizer_identity",
            "working_weights",
            "diagnostic_source_run_identity",
            "diagnostic_source_step",
        },
        SwagDataManifest: {
            "kind",
            "version",
            "identity",
            "source",
            "preprocessing",
            "base_identity",
            "tokenizer_identity",
            "vocab_size",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "example_count",
            "dropped_overlength_rows",
            "buckets",
        },
        ExportManifest: {
            "kind",
            "version",
            "identity",
            "model",
            "precision",
            "tokenizer_identity",
            "model_weights",
            "tokenizer_model",
            "tokenizer_vocab",
            "diagnostic_source_run_identity",
            "diagnostic_source_step",
        },
    }

    for manifest_type, expected_fields in expected.items():
        assert {field.name for field in fields(manifest_type)} == expected_fields
    for manifest in manifest_fixtures():
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.identity = IDENTITY_B


def test_strict_manifest_parser_round_trips_every_schema(tmp_path):
    """A missing schema registration would make a declared artifact unreadable."""
    filenames = (
        "manifest.json",
        "manifest.json",
        "checkpoint.json",
        "run.json",
        "latest.json",
        "manifest.json",
        "manifest.json",
        "manifest.json",
    )

    for index, (manifest, filename) in enumerate(zip(manifest_fixtures(), filenames)):
        root = tmp_path / str(index)
        root.mkdir()
        manifest = _materialize_payloads(root, manifest)
        expected = _write_manifest(root, manifest, filename)

        verified = read_manifest(
            root, type(manifest), VerificationLevel.MANIFEST_TRUSTED
        )

        assert verified.manifest == expected
        assert verified.verification is VerificationLevel.MANIFEST_TRUSTED


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


def test_manifest_verification_levels_have_only_the_two_pinned_values():
    """A third verification label could overstate what a reader checked."""
    assert {level.value for level in VerificationLevel} == {
        "manifest-trusted",
        "full",
    }


def test_manifest_constructor_rejects_boolean_schema_version():
    """Treating True as version 1 would admit a noncanonical discriminator type."""
    with pytest.raises(ValueError, match="version"):
        tokenizer_manifest_fixture(version=True)


@pytest.mark.parametrize(
    ("run_kind", "base_identity", "rope_scaling_factor", "message"),
    [
        ("pretraining", None, 2.0, "rope_scaling_factor"),
        ("pretraining", IDENTITY_A, 1.0, "base_identity"),
        ("lora", None, 1.0, "base_identity"),
        ("other", IDENTITY_A, 1.0, "run_kind"),
    ],
)
def test_run_manifest_rejects_lineage_and_pretraining_rope_mutations(
    run_kind, base_identity, rope_scaling_factor, message
):
    """Relaxing run lineage or base RoPE would make resume semantics ambiguous."""
    with pytest.raises((TypeError, ValueError, SMLArtifactError), match=message):
        RunManifest(
            kind="run",
            version=1,
            identity=IDENTITY_A,
            run_kind=run_kind,
            model={"rope_scaling_factor": rope_scaling_factor},
            precision={},
            optimizer={},
            loader={},
            checkpoint={},
            tokenizer_identity=IDENTITY_B,
            base_identity=base_identity,
            data_identity=IDENTITY_C,
            diagnostic_data_locator=None,
            diagnostic_source_locator=None,
        )


def test_row_content_identity_pins_shape_count_order_and_little_endian_int32():
    """Changing row order or metadata encoding must change the semantic data digest."""
    rows = [np.array([1, 2], dtype=np.int64), np.array([-3, 4], dtype=np.int16)]
    digest = hashlib.sha256()
    digest.update(b"sml-row-content-v1\0")
    digest.update((2).to_bytes(8, "little"))
    digest.update((2).to_bytes(8, "little"))
    digest.update(np.array([1, 2], dtype="<i4").tobytes())
    digest.update(np.array([-3, 4], dtype="<i4").tobytes())

    assert row_content_identity(rows, row_count=2, row_width=2) == (
        f"sha256:{digest.hexdigest()}"
    )
    assert row_content_identity(reversed(rows), row_count=2, row_width=2) != (
        f"sha256:{digest.hexdigest()}"
    )


@pytest.mark.parametrize(
    ("rows", "row_count", "row_width", "message"),
    [
        ([np.array([1, 2], dtype=np.int32)], 2, 2, "row count"),
        ([np.array([1, 2], dtype=np.int32)], 1, 3, "row width|shape"),
        ([np.array([[1, 2]], dtype=np.int32)], 1, 2, "shape"),
        ([np.array([1.0, 2.0])], 1, 2, "integer"),
        ([np.array([0, 2**31], dtype=np.int64)], 1, 2, "int32"),
    ],
)
def test_row_content_identity_rejects_count_width_shape_dtype_and_range_mismatches(
    rows, row_count, row_width, message
):
    """Silently coercing malformed rows would let invalid prepared data share an ID."""
    with pytest.raises((TypeError, ValueError), match=message):
        row_content_identity(rows, row_count=row_count, row_width=row_width)
