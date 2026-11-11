from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import manifest as manifest_module
from sml.artifacts import verify as verify_module
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    ExportManifest,
    OpenedArtifact,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
)
from sml.artifacts.verify import verify_artifact
from sml.errors import SMLArtifactError

from ._payloads import (
    _PLACEHOLDER_IDENTITY,
    _payload_ref,
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


@pytest.mark.parametrize("mutation", ("add-candidate", "replace-selected"))
def test_recursive_manifest_mutation_during_parse_rejects_and_closes_root(
    tmp_path: Path,
    prepared_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Candidate identity and cardinality must stay stable through schema parsing."""
    root = tmp_path / f"prepared-parse-{mutation}"
    shutil.copytree(prepared_template, root)
    tokenizer = root / "tokenizer"
    manifest_path = tokenizer / "manifest.json"
    opened_fds: list[int] = []
    original_open_child = ArtifactRoot.open_child
    real_parse = manifest_module._parse_manifest

    def recording_open_child(owner: ArtifactRoot, logical_path: str) -> ArtifactRoot:
        child = original_open_child(owner, logical_path)
        if logical_path == "tokenizer":
            opened_fds.append(child.fileno())
        return child

    def mutating_parse(raw, manifest_type):
        if manifest_type is TokenizerManifest:
            encoded = manifest_path.read_bytes()
            if mutation == "add-candidate":
                (tokenizer / "run.json").write_bytes(encoded)
            else:
                manifest_path.rename(tokenizer / "manifest.retained.json")
                manifest_path.write_bytes(encoded)
        return real_parse(raw, manifest_type)

    monkeypatch.setattr(ArtifactRoot, "open_child", recording_open_child)
    monkeypatch.setattr(manifest_module, "_parse_manifest", mutating_parse)

    with pytest.raises(SMLArtifactError, match="candidate.*changed"):
        verify_artifact(root, full=False)

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
