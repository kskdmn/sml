from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import stat
from dataclasses import fields, replace
from enum import Enum
from pathlib import Path

import numpy as np
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
            checkpoint={"interval": 5},
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
            checkpoint={"interval": 5},
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


def _write_tokenizer_artifact(
    root: Path,
    *,
    model_data: bytes = b"model-bytes",
    vocab_data: bytes = b"vocab-bytes",
    diagnostic_source_locator: str = "/original",
) -> TokenizerManifest:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.model").write_bytes(model_data)
    (root / "tokenizer.vocab").write_bytes(vocab_data)
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef(
            "tokenizer.model", file_identity(io.BytesIO(model_data)), len(model_data)
        ),
        vocab=PayloadRef(
            "tokenizer.vocab", file_identity(io.BytesIO(vocab_data)), len(vocab_data)
        ),
        diagnostic_source_locator=diagnostic_source_locator,
    )
    return _write_manifest(root, manifest, "manifest.json")


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
        PretrainingCheckpointManifest: {
            "kind",
            "version",
            "identity",
            "owning_run_identity",
            "step",
            "scalar_state",
            "model",
            "master",
            "optimizer",
            "trainer",
        },
        LoRACheckpointManifest: {
            "kind",
            "version",
            "identity",
            "owning_run_identity",
            "step",
            "scalar_state",
            "adapters",
            "optimizer",
            "trainer",
        },
        PretrainingRunManifest: {
            "kind",
            "version",
            "identity",
            "model",
            "precision",
            "optimizer",
            "loader",
            "checkpoint",
            "tokenizer_identity",
            "data_identity",
            "diagnostic_data_locator",
        },
        LoRARunManifest: {
            "kind",
            "version",
            "identity",
            "model",
            "lora",
            "precision",
            "optimizer",
            "loader",
            "checkpoint",
            "tokenizer_identity",
            "base_identity",
            "data_identity",
            "diagnostic_data_locator",
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


def test_run_and_checkpoint_schema_kinds_are_distinct_and_strict():
    """One optional-field container cannot freeze pretraining and LoRA contracts."""
    required = (
        "PretrainingRunManifest",
        "LoRARunManifest",
        "PretrainingCheckpointManifest",
        "LoRACheckpointManifest",
    )
    missing = [name for name in required if not hasattr(manifest_module, name)]
    assert missing == []

    run_kinds = {
        manifest_module.PretrainingRunManifest.EXPECTED_KIND,
        manifest_module.LoRARunManifest.EXPECTED_KIND,
    }
    checkpoint_kinds = {
        manifest_module.PretrainingCheckpointManifest.EXPECTED_KIND,
        manifest_module.LoRACheckpointManifest.EXPECTED_KIND,
    }
    assert run_kinds == {"pretraining-run", "lora-run"}
    assert checkpoint_kinds == {"pretraining-checkpoint", "lora-checkpoint"}


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


def test_strict_manifest_parser_round_trips_every_schema(tmp_path):
    """A missing schema registration would make a declared artifact unreadable."""
    filenames = (
        "manifest.json",
        "manifest.json",
        "checkpoint.json",
        "checkpoint.json",
        "run.json",
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


def test_open_artifact_does_not_preopen_declared_payloads(tmp_path):
    """Proof-only eager payload opens would separate later semantic use from proof."""
    expected = _write_manifest(tmp_path, tokenizer_manifest_fixture(), "manifest.json")

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        assert artifact.manifest == expected
        assert artifact.verification is VerificationLevel.FULL


def test_opened_artifact_child_uses_retained_root_after_path_replacement(tmp_path):
    """Constructing a child Path would parse a replacement instead of the retained root."""
    bundle = tmp_path / "bundle"
    _write_tokenizer_artifact(bundle, diagnostic_source_locator="/outer-original")
    _write_tokenizer_artifact(
        bundle / "tokenizer", diagnostic_source_locator="/child-original"
    )

    artifact = manifest_module.open_artifact(
        bundle, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
    )
    retained = tmp_path / "retained-bundle"
    bundle.rename(retained)
    _write_tokenizer_artifact(
        bundle / "tokenizer", diagnostic_source_locator="/child-replacement"
    )

    try:
        child = artifact.open_child("tokenizer", (TokenizerManifest,))
        artifact.close()
        with child:
            assert child.manifest.diagnostic_source_locator == "/child-original"
            with child.open_payload(child.manifest.model) as payload:
                assert payload.stream.read() == b"model-bytes"
    finally:
        artifact.close()


def test_full_payload_proof_and_reader_share_one_final_descriptor_after_swap(
    tmp_path, monkeypatch
):
    """Hashing one inode and reopening its name for use would accept swapped bytes."""
    expected_data = b"proven-model"
    _write_tokenizer_artifact(tmp_path, model_data=expected_data)
    artifact = manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    )
    real_open = os.open
    real_fstat = os.fstat
    real_file_identity = manifest_module.file_identity
    payload_descriptors: list[int] = []
    fstat_descriptors: list[int] = []
    identity_descriptors: list[int] = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "tokenizer.model":
            payload_descriptors.append(descriptor)
        return descriptor

    def recording_fstat(descriptor):
        if descriptor in payload_descriptors:
            fstat_descriptors.append(descriptor)
        return real_fstat(descriptor)

    def recording_file_identity(stream):
        identity_descriptors.append(stream.fileno())
        return real_file_identity(stream)

    monkeypatch.setattr(manifest_module.os, "open", recording_open)
    monkeypatch.setattr(manifest_module.os, "fstat", recording_fstat)
    monkeypatch.setattr(manifest_module, "file_identity", recording_file_identity)

    try:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        (tmp_path / "tokenizer.model").rename(tmp_path / "proven.model")
        (tmp_path / "tokenizer.model").write_bytes(b"replacement!")

        assert payload.stream.read() == expected_data
        assert payload_descriptors == [descriptor]
        assert identity_descriptors == [descriptor]
        assert fstat_descriptors and set(fstat_descriptors) == {descriptor}
        with pytest.raises(SMLArtifactError, match="changed during use"):
            payload.close()
        with pytest.raises(OSError):
            real_fstat(descriptor)
    finally:
        artifact.close()


@pytest.mark.parametrize("mutation", ["size", "mtime", "ctime"])
def test_verified_payload_close_rejects_every_tracked_in_place_mutation(
    tmp_path, mutation
):
    """Omitting any pinned stat field would allow an open payload to change during use."""
    _write_tokenizer_artifact(tmp_path)
    payload_path = tmp_path / "tokenizer.model"

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        before = payload_path.stat()
        if mutation == "size":
            payload_path.write_bytes(b"changed-size")
        elif mutation == "mtime":
            os.utime(
                payload_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        else:
            payload_path.chmod(stat.S_IMODE(before.st_mode) | stat.S_IXUSR)

        with pytest.raises(SMLArtifactError, match="changed during use"):
            payload.close()
        assert payload.closed is True
        payload.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_semantic_error_stays_primary_when_mutation_postcheck_also_fails(tmp_path):
    """Replacing the reader error with cleanup failure would hide the semantic cause."""
    _write_tokenizer_artifact(tmp_path)

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        with (
            pytest.raises(RuntimeError, match="semantic reader failed") as caught,
            payload,
        ):
            (tmp_path / "tokenizer.model").write_bytes(b"mutated")
            raise RuntimeError("semantic reader failed")

        assert isinstance(caught.value.__cause__, SMLArtifactError)
        assert "changed during use" in str(caught.value.__cause__)
        assert payload.closed is True
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    ("link_kind", "message"),
    [
        ("symlink", "symlink|no-follow"),
        ("internal-hard-link", "link count"),
        ("external-hard-link", "link count"),
    ],
)
def test_opened_artifact_rejects_linked_payloads(tmp_path, link_kind, message):
    """Accepting links would let payload ownership escape or alias the artifact root."""
    root = tmp_path / "artifact"
    root.mkdir()
    data = b"linked"
    if link_kind == "symlink":
        source = tmp_path / "external.bin"
        source.write_bytes(data)
        (root / "tokenizer.model").symlink_to(source)
    elif link_kind == "internal-hard-link":
        source = root / "source.bin"
        source.write_bytes(data)
        os.link(source, root / "tokenizer.model")
    else:
        source = tmp_path / "external.bin"
        source.write_bytes(data)
        os.link(source, root / "tokenizer.model")
    (root / "tokenizer.vocab").write_bytes(b"vocab")
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef("tokenizer.model", file_identity(io.BytesIO(data)), len(data)),
        vocab=PayloadRef("tokenizer.vocab", file_identity(io.BytesIO(b"vocab")), 5),
    )
    _write_manifest(root, manifest, "manifest.json")

    with (
        manifest_module.open_artifact(
            root, (TokenizerManifest,), VerificationLevel.FULL
        ) as artifact,
        pytest.raises(SMLArtifactError, match=message),
    ):
        artifact.open_payload(artifact.manifest.model)


def test_opened_artifact_rejects_distinct_paths_to_one_inode(tmp_path, monkeypatch):
    """Dropping retained-root inode tracking would accept duplicate payload aliases."""
    first = tmp_path / "tokenizer.model"
    second = tmp_path / "tokenizer.vocab"
    first.write_bytes(b"shared")
    os.link(first, second)
    aliased_inode = first.stat().st_ino
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef("tokenizer.model", file_identity(io.BytesIO(b"shared")), 6),
        vocab=PayloadRef("tokenizer.vocab", file_identity(io.BytesIO(b"shared")), 6),
    )
    _write_manifest(tmp_path, manifest, "manifest.json")
    real_fstat = os.fstat

    def single_link_fstat(descriptor):
        result = real_fstat(descriptor)
        if result.st_ino != aliased_inode:
            return result
        values = list(result)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(manifest_module.os, "fstat", single_link_fstat)
    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
    ) as artifact:
        with artifact.open_payload(artifact.manifest.model):
            pass
        with pytest.raises(SMLArtifactError, match="inode alias"):
            artifact.open_payload(artifact.manifest.vocab)


def test_payload_acquisition_rejects_hard_link_created_by_fdopen_hook(
    tmp_path, monkeypatch
):
    """Using a later baseline would accept a hard link introduced after validation."""
    _write_tokenizer_artifact(tmp_path)
    payload_path = tmp_path / "tokenizer.model"
    payload_inode = payload_path.stat().st_ino
    real_fdopen = os.fdopen
    real_fstat = os.fstat
    payload_descriptors: list[int] = []

    def linking_fdopen(descriptor, *args, **kwargs):
        stream = real_fdopen(descriptor, *args, **kwargs)
        if real_fstat(descriptor).st_ino == payload_inode:
            payload_descriptors.append(descriptor)
            os.link(payload_path, tmp_path / "late-alias.bin")
        return stream

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        monkeypatch.setattr(manifest_module.os, "fdopen", linking_fdopen)
        payload = None
        try:
            payload = artifact.open_payload(artifact.manifest.model)
        except SMLArtifactError as error:
            assert "link count" in str(error)
        else:
            payload.close()
            pytest.fail("payload acquisition accepted a late hard link")

    assert len(payload_descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(payload_descriptors[0])


@pytest.mark.parametrize("semantic_failure", [False, True])
def test_non_oserror_postcheck_always_closes_and_preserves_exception_precedence(
    tmp_path, monkeypatch, semantic_failure
):
    """A non-OSError postcheck failure must not leak or replace a reader failure."""
    _write_tokenizer_artifact(tmp_path)
    real_fstat = os.fstat

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()

        def failing_postcheck(candidate):
            if candidate == descriptor:
                raise RuntimeError("injected postcheck failure")
            return real_fstat(candidate)

        monkeypatch.setattr(manifest_module.os, "fstat", failing_postcheck)
        try:
            if semantic_failure:
                with (
                    pytest.raises(ValueError, match="semantic failure") as caught,
                    payload,
                ):
                    raise ValueError("semantic failure")
                assert isinstance(caught.value.__cause__, RuntimeError)
                assert "injected postcheck failure" in str(caught.value.__cause__)
            else:
                with pytest.raises(RuntimeError, match="injected postcheck failure"):
                    payload.close()

            assert payload.closed is True
            payload.close()
            with pytest.raises(OSError):
                real_fstat(descriptor)
        finally:
            payload.stream.close()


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
            checkpoint={},
            tokenizer_identity=IDENTITY_B,
            data_identity=IDENTITY_C,
            diagnostic_data_locator=None,
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
