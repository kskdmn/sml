from __future__ import annotations

import dataclasses
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import zstandard as zstd
from sml.artifacts import verify as verify_module
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    BaseSnapshotManifest,
    ExportManifest,
    LatestIndex,
    OpenedArtifact,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_checkpoint_manifest,
    read_manifest,
    row_content_identity,
    structured_identity,
)
from sml.artifacts.verify import verify_artifact
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.model.language_model import model_parameter_specs
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    PretrainingConfig,
)
from sml.training.lora import LoRAPrecisionConfig
from sml.training.pretrain import train

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _array_ref(
    path: Path,
    logical_path: str,
    arrays: dict[str, mx.array],
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        tuple(
            ArraySpec(
                name,
                tuple(array.shape),
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


def _npy_ref(path: Path, logical_path: str, name: str) -> ArrayPayloadRef:
    array = np.load(path, allow_pickle=False)
    dtype = "bool" if array.dtype == np.dtype("bool") else "int32"
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        (ArraySpec(name, tuple(array.shape), dtype),),
    )


def _write_structural_tokenizer(path: Path) -> TokenizerManifest:
    path.mkdir()
    model_path = path / "tokenizer.model"
    vocab_path = path / "tokenizer.vocab"
    model_path.write_bytes(b"structural model bytes")
    vocab_path.write_bytes(b"<unk>\t0\n<s>\t0\n</s>\t0\n<pad>\t0\n")
    manifest = TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        algorithm="bpe",
        training={"intentionally": "not semantic"},
        vocab_size=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        model=_payload_ref(model_path, "tokenizer.model"),
        vocab=_payload_ref(vocab_path, "tokenizer.vocab"),
        diagnostic_source_locator=None,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def test_manifest_trusted_export_reports_owned_tokenizer_without_semantic_loading(
    tmp_path: Path,
) -> None:
    """Dropping export recursion would hide a structurally owned child artifact."""
    export = tmp_path / "export"
    export.mkdir()
    tokenizer = _write_structural_tokenizer(export / "tokenizer")
    weights_path = export / "model.safetensors"
    weights_path.write_bytes(b"structural model payload")
    manifest = ExportManifest(
        kind="export",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model={"rope_scaling_factor": 1.0},
        precision={"working_parameter_dtype": "bfloat16"},
        tokenizer_identity=tokenizer.identity,
        model_weights=ArrayPayloadRef(
            _payload_ref(weights_path, "model.safetensors"),
            (ArraySpec("weight", (1,), "bfloat16"),),
        ),
        tokenizer_model=replace(
            tokenizer.model, logical_path="tokenizer/tokenizer.model"
        ),
        tokenizer_vocab=replace(
            tokenizer.vocab, logical_path="tokenizer/tokenizer.vocab"
        ),
        diagnostic_source_run_identity="sha256:" + "1" * 64,
        diagnostic_source_step=0,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (export / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    verified = verify_artifact(export, full=False)

    assert verified.verification is VerificationLevel.MANIFEST_TRUSTED
    assert tuple(child.manifest.kind for child in verified.children) == ("tokenizer",)
    assert tuple(child.verification for child in verified.children) == (
        VerificationLevel.MANIFEST_TRUSTED,
    )


def test_recursive_dispatch_requires_exactly_one_manifest_candidate(
    tmp_path: Path,
) -> None:
    """A second regular discriminator cannot be ignored after root acquisition."""
    root = tmp_path / "ambiguous"
    _write_structural_tokenizer(root)
    (root / "run.json").write_bytes((root / "manifest.json").read_bytes())

    with pytest.raises(SMLArtifactError, match="exactly one"):
        verify_artifact(root, full=False)


@pytest.mark.parametrize("full", (False, True))
@pytest.mark.parametrize("mutation", ("ambiguous", "noncanonical"))
def test_recursive_tokenizer_dispatch_rejects_invalid_candidates_and_closes_root(
    tmp_path: Path,
    prepared_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    full: bool,
    mutation: str,
) -> None:
    """Every child root needs the same exact retained-root dispatch as its owner."""
    root = tmp_path / f"prepared-{full}-{mutation}"
    shutil.copytree(prepared_template, root)
    tokenizer = root / "tokenizer"
    if mutation == "ambiguous":
        (tokenizer / "run.json").write_bytes((tokenizer / "manifest.json").read_bytes())
        message = "exactly one"
    else:
        raw = json.loads((tokenizer / "manifest.json").read_bytes())
        (tokenizer / "manifest.json").write_text(
            json.dumps(raw, indent=2),
            encoding="utf-8",
        )
        message = "canonical JSON bytes"

    opened_fds: list[int] = []
    original_open_child = ArtifactRoot.open_child

    def recording_open_child(owner: ArtifactRoot, logical_path: str) -> ArtifactRoot:
        child = original_open_child(owner, logical_path)
        if logical_path == "tokenizer":
            opened_fds.append(child.fileno())
        return child

    monkeypatch.setattr(ArtifactRoot, "open_child", recording_open_child)

    with pytest.raises(SMLArtifactError, match=message):
        verify_artifact(root, full=full)

    assert len(opened_fds) == 1
    with pytest.raises(OSError):
        os.fstat(opened_fds[0])


def test_recursive_child_dispatch_preserves_semantic_error_over_cleanup_failure(
    tmp_path: Path,
    prepared_template: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child cleanup failure is chained behind the semantic candidate error."""
    root = tmp_path / "prepared-cleanup-failure"
    shutil.copytree(prepared_template, root)
    tokenizer = root / "tokenizer"
    (tokenizer / "run.json").write_bytes((tokenizer / "manifest.json").read_bytes())
    child_fd = -1
    original_open_child = ArtifactRoot.open_child
    original_close = ArtifactRoot.close

    def recording_open_child(owner: ArtifactRoot, logical_path: str) -> ArtifactRoot:
        nonlocal child_fd
        child = original_open_child(owner, logical_path)
        if logical_path == "tokenizer":
            child_fd = child.fileno()
        return child

    def failing_child_close(owner: ArtifactRoot) -> None:
        descriptor = owner.fileno()
        original_close(owner)
        if descriptor == child_fd:
            raise RuntimeError("injected child cleanup failure")

    monkeypatch.setattr(ArtifactRoot, "open_child", recording_open_child)
    monkeypatch.setattr(ArtifactRoot, "close", failing_child_close)

    with pytest.raises(SMLArtifactError, match="exactly one") as raised:
        verify_artifact(root, full=True)

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "injected child cleanup failure"
    with pytest.raises(OSError):
        os.fstat(child_fd)


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing-leaf", "model parameter specs"),
        ("wrong-dtype", "model parameter specs"),
        ("wrong-shape", "model parameter specs"),
        ("invalid-config", "invalid base snapshot model configuration"),
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
    if mutation == "invalid-config":
        model = dict(manifest.model)
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


@pytest.fixture(scope="module")
def prepared_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("recursive-prepared")
    corpus = root / "corpus"
    corpus.mkdir()
    rows = [
        {"text": "alpha beta gamma delta epsilon zeta eta theta " * 8}
        for _ in range(12)
    ]
    payload = b"".join(
        json.dumps(row, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )
    (corpus / "tiny-0000.jsonl.zst").write_bytes(
        zstd.ZstdCompressor().compress(payload)
    )
    corpus_config = CorpusConfig(
        input_root=corpus,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=corpus_config,
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    prepared = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=corpus_config,
            tokenizer_bundle=tokenizer.path,
            sequence_length=4,
            shuffle_window_rows=4,
            output_shard_rows=8,
            seed=7,
        ),
        root / "prepared",
    )
    return prepared.path


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


def _write_export_bundle(root: Path, tokenizer_root: Path) -> ExportManifest:
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


@pytest.mark.parametrize("replacement", ("outer", "tokenizer", "shard"))
def test_recursive_full_verification_consumes_post_open_owners(
    tmp_path: Path,
    prepared_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    """A pathname swap after owner acquisition must never redirect consumption."""
    root = tmp_path / "prepared"
    shutil.copytree(prepared_template, root)
    moved: Path | None = None

    if replacement == "outer":
        original_open = verify_module._open_artifact_once

        def swapping_open(path: Path, verification: VerificationLevel):
            nonlocal moved
            artifact = original_open(path, verification)
            moved = root.with_name("retained-prepared")
            root.rename(moved)
            root.mkdir()
            (root / "manifest.json").write_bytes(b"replacement")
            return artifact

        monkeypatch.setattr(verify_module, "_open_artifact_once", swapping_open)
    elif replacement == "tokenizer":
        original_verify = verify_module._verify_opened_tokenizer

        def swapping_verify(artifact):
            nonlocal moved
            if artifact.path == root / "tokenizer":
                moved = root / "retained-tokenizer"
                (root / "tokenizer").rename(moved)
                (root / "tokenizer").mkdir()
                (root / "tokenizer" / "manifest.json").write_bytes(b"replacement")
            return original_verify(artifact)

        monkeypatch.setattr(
            verify_module,
            "_verify_opened_tokenizer",
            swapping_verify,
        )
    else:
        original_payload = OpenedArtifact.open_payload
        swapped = False

        def swapping_payload(artifact, reference):
            nonlocal moved, swapped
            payload = original_payload(artifact, reference)
            if not swapped and reference.logical_path.startswith("shards/"):
                swapped = True
                source = root / reference.logical_path
                moved = root / "retained-shards"
                source.parent.rename(moved)
                shutil.copytree(moved, source.parent)
                source.write_bytes(b"replacement")
            return payload

        monkeypatch.setattr(OpenedArtifact, "open_payload", swapping_payload)

    result = verify_artifact(root, full=True)

    assert result.path == root
    assert result.manifest.kind == "pretraining-data"
    assert result.children[0].path == root / "tokenizer"
    assert result.children[0].manifest.kind == "tokenizer"
    assert moved is not None


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
        ("accumulator", "trainer accumulators must be empty"),
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
    if mutation == "accumulator":
        name = next(name for name in arrays if name.startswith("accumulators."))
        arrays[name] = mx.ones(arrays[name].shape, dtype=mx.float32)
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


def test_full_pretraining_run_rejects_incomplete_resigned_accumulation_progress(
    tmp_path: Path,
    pretraining_run_template: Path,
) -> None:
    """An empty checkpoint boundary must represent every configured microstep."""
    run = tmp_path / "run-incomplete-progress"
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

    with pytest.raises(SMLArtifactError, match="microsteps disagree"):
        verify_artifact(run, full=True)


@pytest.mark.parametrize("full", (False, True))
def test_recursive_run_verification_requires_a_bound_latest_index(
    tmp_path: Path,
    pretraining_run_template: Path,
    full: bool,
) -> None:
    """Verification must not report a recovered scan as a proven latest index."""
    run = tmp_path / f"run-missing-latest-{full}"
    shutil.copytree(pretraining_run_template, run)
    (run / "latest.json").unlink()

    with pytest.raises(SMLArtifactError, match="latest index"):
        verify_artifact(run, full=full)
