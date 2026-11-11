from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    PretrainingDataManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_manifest,
    row_content_identity,
    structured_identity,
)
from sml.artifacts.verify import verify_artifact
from sml.data import swag as swag_module
from sml.errors import SMLArtifactError

from ._payloads import (
    _PLACEHOLDER_IDENTITY,
    _payload_ref,
)


def _npy_ref(path: Path, logical_path: str, name: str) -> ArrayPayloadRef:
    array = np.load(path, allow_pickle=False)
    dtype = "bool" if array.dtype == np.dtype("bool") else "int32"
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        (ArraySpec(name, tuple(array.shape), dtype),),
    )


def _write_swag_bundle(root: Path) -> SwagDataManifest:
    root.mkdir()
    bucket = root / "buckets" / "length-0008"
    bucket.mkdir(parents=True)
    input_ids = np.full((1, 4, 8), 3, dtype=np.int32)
    input_ids[:, :, :4] = np.array([1, 4, 5, 2], dtype=np.int32)
    valid = np.zeros((1, 4, 8), dtype=np.bool_)
    valid[:, :, :4] = True
    score = np.zeros((1, 4, 8), dtype=np.bool_)
    score[:, :, 2:4] = True
    labels = np.array([1], dtype=np.int32)
    arrays = {
        "input_ids": input_ids,
        "valid_token_mask": valid,
        "score_mask": score,
        "labels": labels,
    }
    references = []
    for name, array in arrays.items():
        path = bucket / f"{name}.npy"
        with path.open("wb") as destination:
            np.save(destination, array, allow_pickle=False)
        references.append(_npy_ref(path, f"buckets/length-0008/{name}.npy", name))
    manifest = SwagDataManifest(
        kind="swag-data",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        source={
            "backend": "huggingface-datasets",
            "namespace": "allenai",
            "name": "swag",
            "dataset_config": "regular",
            "revision": "deadbeef",
            "split": "train",
            "commit": "abc123",
            "provider_fingerprint": "fingerprint",
            "provider_package": "datasets",
            "provider_version": "1.0",
        },
        preprocessing={
            "schema_version": 1,
            "join_policy": "separate-context-ending-v1",
            "overlength_policy": "drop-complete-row-v1",
            "bos_policy": "context-bos-v1",
            "eos_policy": "scored-ending-eos-v1",
            "maximum_length": 8,
            "bucket_boundaries": [8],
            "maximum_examples": None,
        },
        base_identity="sha256:" + "1" * 64,
        tokenizer_identity="sha256:" + "2" * 64,
        vocab_size=8,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        example_count=1,
        dropped_overlength_rows=0,
        buckets=tuple(references),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _write_resigned_swag(root: Path, manifest: SwagDataManifest) -> None:
    resigned = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(resigned))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("unreadable-model", "invalid tokenizer model payload"),
        ("invalid-vocabulary", "invalid tokenizer vocabulary bytes"),
        ("special-token-disagreement", "special IDs"),
    ),
)
def test_full_tokenizer_rejects_resigned_semantic_corruption(
    tmp_path: Path,
    prepared_template: Path,
    mutation: str,
    message: str,
) -> None:
    """Tokenizer FULL loads both bound payloads and checks exact special IDs."""
    root = tmp_path / f"tokenizer-{mutation}"
    shutil.copytree(prepared_template / "tokenizer", root)
    manifest = read_manifest(
        root,
        TokenizerManifest,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    if mutation == "unreadable-model":
        path = root / manifest.model.logical_path
        path.write_bytes(b"not a SentencePiece model")
        manifest = replace(
            manifest,
            model=_payload_ref(path, manifest.model.logical_path),
        )
    elif mutation == "invalid-vocabulary":
        path = root / manifest.vocab.logical_path
        path.write_bytes(b"\xff")
        manifest = replace(
            manifest,
            vocab=_payload_ref(path, manifest.vocab.logical_path),
        )
    else:
        manifest = replace(manifest, bos_token_id=4)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)


def test_full_pretraining_data_rejects_resigned_wrong_row_identity(
    tmp_path: Path,
    prepared_template: Path,
) -> None:
    """Re-signing changed rows must not bypass the ordered row-content identity."""
    root = tmp_path / "prepared"
    shutil.copytree(prepared_template, root)
    verified = verify_artifact(root, full=True)
    assert isinstance(verified.manifest, PretrainingDataManifest)
    manifest = verified.manifest
    shard = manifest.shards[0]
    shard_path = root / shard.logical_path
    rows = np.load(shard_path, allow_pickle=False)
    rows[0, 0] = (int(rows[0, 0]) + 1) % verified.children[0].manifest.vocab_size
    with shard_path.open("wb") as destination:
        np.save(destination, rows, allow_pickle=False)
    resigned = replace(
        manifest,
        shards=(
            _payload_ref(shard_path, shard.logical_path),
            *manifest.shards[1:],
        ),
    )
    resigned = replace(resigned, identity=resigned.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(SMLArtifactError, match="row-content identity"):
        verify_artifact(root, full=True)


def test_full_pretraining_data_rejects_resigned_out_of_vocabulary_token(
    tmp_path: Path,
    prepared_template: Path,
) -> None:
    """A recomputed row identity cannot legitimize a token outside the vocabulary."""
    root = tmp_path / "prepared-oov"
    shutil.copytree(prepared_template, root)
    manifest = read_manifest(
        root,
        PretrainingDataManifest,
        VerificationLevel.FULL,
    ).manifest
    tokenizer = read_manifest(
        root / "tokenizer",
        TokenizerManifest,
        VerificationLevel.FULL,
    ).manifest
    first_path = root / manifest.shards[0].logical_path
    first_rows = np.load(first_path, allow_pickle=False)
    first_rows[0, 0] = tokenizer.vocab_size
    with first_path.open("wb") as destination:
        np.save(destination, first_rows, allow_pickle=False)
    shards = (
        _payload_ref(first_path, manifest.shards[0].logical_path),
        *manifest.shards[1:],
    )
    all_rows = (
        row
        for shard in shards
        for row in np.load(root / shard.logical_path, allow_pickle=False)
    )
    resigned = replace(
        manifest,
        shards=shards,
        row_content_identity=row_content_identity(
            all_rows,
            sum(manifest.shard_row_counts),
            manifest.row_width,
        ),
    )
    resigned = replace(resigned, identity=resigned.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(SMLArtifactError, match="token IDs must be in"):
        verify_artifact(root, full=True)


def test_full_swag_rejects_resigned_false_source_backend(tmp_path: Path) -> None:
    """The resolved source must retain the exact supported request backend."""
    root = tmp_path / "swag-source-backend"
    manifest = _write_swag_bundle(root)
    source = dict(manifest.source)
    source["backend"] = "local-files"
    _write_resigned_swag(root, replace(manifest, source=source))

    with pytest.raises(SMLArtifactError, match="source projection"):
        verify_artifact(root, full=True)


def test_full_swag_rejects_resigned_example_count_above_cap(tmp_path: Path) -> None:
    """A recorded preparation cap must bound the exact materialized row count."""
    root = tmp_path / "swag-cap"
    manifest = _write_swag_bundle(root)
    buckets = []
    for reference in manifest.buckets:
        path = root / reference.payload.logical_path
        array = np.load(path, allow_pickle=False)
        doubled = np.concatenate((array, array), axis=0)
        with path.open("wb") as destination:
            np.save(destination, doubled, allow_pickle=False)
        buckets.append(
            _npy_ref(
                path,
                reference.payload.logical_path,
                reference.arrays[0].name,
            )
        )
    preprocessing = dict(manifest.preprocessing)
    preprocessing["maximum_examples"] = 1
    _write_resigned_swag(
        root,
        replace(
            manifest,
            preprocessing=preprocessing,
            example_count=2,
            buckets=tuple(buckets),
        ),
    )

    with pytest.raises(SMLArtifactError, match="maximum_examples"):
        verify_artifact(root, full=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong-name", "array spec name"),
        ("multiple-specs", "exactly one array spec"),
    ),
)
def test_full_swag_rejects_resigned_array_spec_contract_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Each one-array NPY declaration must use its canonical path stem as name."""
    root = tmp_path / f"swag-spec-{mutation}"
    manifest = _write_swag_bundle(root)
    reference = manifest.buckets[0]
    if mutation == "wrong-name":
        arrays = (replace(reference.arrays[0], name="labels"),)
    else:
        arrays = (
            reference.arrays[0],
            replace(reference.arrays[0], name="unexpected"),
        )
    buckets = (replace(reference, arrays=arrays), *manifest.buckets[1:])
    _write_resigned_swag(root, replace(manifest, buckets=buckets))

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)


def test_full_swag_rejects_resigned_nondeterministic_bucket_order(
    tmp_path: Path,
) -> None:
    """Manifest order is part of the deterministic canonical SWAG tree."""
    root = tmp_path / "swag-order"
    manifest = _write_swag_bundle(root)
    buckets = list(manifest.buckets)
    buckets[0], buckets[1] = buckets[1], buckets[0]
    _write_resigned_swag(root, replace(manifest, buckets=tuple(buckets)))

    with pytest.raises(SMLArtifactError, match="deterministic bucket order"):
        verify_artifact(root, full=True)


def test_swag_semantic_validation_uses_fixed_row_chunks(monkeypatch):
    """Verifier scratch arrays must not grow with the dataset row count."""
    rows = 5_000
    input_ids = np.full((rows, 4, 4), 3, dtype=np.int32)
    input_ids[:, :, 0] = 1
    input_ids[:, :, 1] = 7
    input_ids[:, :, 2] = 2
    valid = np.zeros(input_ids.shape, dtype=np.bool_)
    valid[:, :, :3] = True
    score = np.zeros(input_ids.shape, dtype=np.bool_)
    score[:, :, 1:3] = True
    real_arange = swag_module.np.arange
    requested: list[int] = []

    def bounded_arange(stop, *args, **kwargs):
        requested.append(int(stop))
        return real_arange(stop, *args, **kwargs)

    monkeypatch.setattr(swag_module.np, "arange", bounded_arange)

    swag_module._validate_bucket_arrays(
        input_ids=input_ids,
        valid_token_mask=valid,
        score_mask=score,
        vocab_size=16,
        pad_token_id=3,
        eos_token_id=2,
        bos_token_id=1,
        maximum_length=4,
        bucket_length=4,
        bucket_boundaries=(4,),
    )

    assert max(requested) <= 1_024


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("invalid-mask-nesting", "score mask is true"),
        ("wrong-bos", "start with BOS"),
        ("wrong-eos", "end with EOS"),
    ),
)
def test_full_swag_rejects_resigned_token_layout_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """FULL reduces the exact masks and BOS/EOS positions of mapped buckets."""
    root = tmp_path / f"swag-{mutation}"
    manifest = _write_swag_bundle(root)
    buckets = list(manifest.buckets)
    target_name = (
        "valid_token_mask" if mutation == "invalid-mask-nesting" else "input_ids"
    )
    index = next(
        position
        for position, reference in enumerate(buckets)
        if reference.arrays[0].name == target_name
    )
    reference = buckets[index]
    path = root / reference.payload.logical_path
    array = np.load(path, allow_pickle=False)
    if mutation == "invalid-mask-nesting":
        array[0, 0, 2] = False
    elif mutation == "wrong-bos":
        array[0, 0, 0] = 4
    else:
        array[0, 0, 3] = 4
    with path.open("wb") as destination:
        np.save(destination, array, allow_pickle=False)
    buckets[index] = _npy_ref(path, reference.payload.logical_path, target_name)
    _write_resigned_swag(root, replace(manifest, buckets=tuple(buckets)))

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("undeclared-boundary", "declared boundary"),
        ("wrong-example-count", "example_count"),
        ("noncanonical-path", "noncanonical SWAG bucket path"),
    ),
)
def test_full_swag_rejects_resigned_boundary_count_or_path_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Bucket boundaries, counts, and paths are exact manifest semantics."""
    root = tmp_path / f"swag-{mutation}"
    manifest = _write_swag_bundle(root)
    if mutation == "undeclared-boundary":
        preprocessing = dict(manifest.preprocessing)
        preprocessing["bucket_boundaries"] = [9]
        manifest = replace(manifest, preprocessing=preprocessing)
    elif mutation == "wrong-example-count":
        manifest = replace(manifest, example_count=2)
    else:
        reference = manifest.buckets[0]
        old_path = root / reference.payload.logical_path
        logical_path = reference.payload.logical_path.replace(
            "length-0008",
            "length-00008",
        )
        new_path = root / logical_path
        new_path.parent.mkdir(parents=True)
        old_path.rename(new_path)
        buckets = (
            replace(reference, payload=_payload_ref(new_path, logical_path)),
            *manifest.buckets[1:],
        )
        manifest = replace(manifest, buckets=buckets)
    _write_resigned_swag(root, manifest)

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=True)


@pytest.mark.parametrize("field", ("base_identity", "tokenizer_identity"))
def test_full_swag_rejects_resigned_invalid_identity_field(
    tmp_path: Path,
    field: str,
) -> None:
    """SWAG provenance identities remain strict without external path resolution."""
    root = tmp_path / f"swag-{field}"
    _write_swag_bundle(root)
    raw = json.loads((root / "manifest.json").read_bytes())
    raw[field] = "sha256:not-hex"
    raw["identity"] = structured_identity(
        SwagDataManifest.IDENTITY_DOMAIN,
        {name: value for name, value in raw.items() if name != "identity"},
    )
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))

    with pytest.raises(SMLArtifactError, match=field):
        verify_artifact(root, full=True)


def test_full_swag_rejects_resigned_out_of_range_label(tmp_path: Path) -> None:
    """A re-signed label outside 0..3 must fail semantic bucket validation."""
    root = tmp_path / "swag"
    manifest = _write_swag_bundle(root)
    label_index = next(
        index
        for index, reference in enumerate(manifest.buckets)
        if reference.arrays[0].name == "labels"
    )
    reference = manifest.buckets[label_index]
    path = root / reference.payload.logical_path
    with path.open("wb") as destination:
        np.save(destination, np.array([4], dtype=np.int32), allow_pickle=False)
    buckets = list(manifest.buckets)
    buckets[label_index] = _npy_ref(path, reference.payload.logical_path, "labels")
    resigned = replace(manifest, buckets=tuple(buckets))
    resigned = replace(resigned, identity=resigned.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(resigned))

    with pytest.raises(SMLArtifactError, match="labels must be in 0..3"):
        verify_artifact(root, full=True)
