"""Typed recursive verification for public artifact roots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from sml.artifacts.arrays import verify_safetensors_metadata
from sml.artifacts.checkpoint import (
    CheckpointReader,
    open_latest_checkpoint_reader,
    require_lora_base_snapshot,
)
from sml.artifacts.dispatch import ArtifactManifest, open_dispatched_artifact
from sml.artifacts.manifest import (
    CHECKPOINT_MANIFEST_TYPES,
    RUN_MANIFEST_TYPES,
    ArtifactRoot,
    BaseSnapshotManifest,
    ExportManifest,
    LoRARunManifest,
    OpenedArtifact,
    PayloadRef,
    PretrainingDataManifest,
    PretrainingRunManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
)
from sml.artifacts.semantics import (
    validate_base_semantics,
    validate_export_semantics,
    validate_full_run_semantics,
)
from sml.data.pretraining import _verify_opened_pretraining_bundle
from sml.data.swag import _load_opened_swag_bundle
from sml.data.tokenizer import _load_opened_tokenizer_bundle
from sml.errors import SMLArtifactError


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A verified artifact plus any independently verified child artifacts."""

    path: Path
    manifest: ArtifactManifest
    verification: VerificationLevel
    children: tuple[VerificationResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("verified artifact path must be a Path")
        if not isinstance(
            self.manifest,
            (
                TokenizerManifest,
                PretrainingDataManifest,
                *CHECKPOINT_MANIFEST_TYPES,
                *RUN_MANIFEST_TYPES,
                BaseSnapshotManifest,
                SwagDataManifest,
                ExportManifest,
            ),
        ):
            raise TypeError("manifest must be a supported artifact manifest")
        if not isinstance(self.verification, VerificationLevel):
            raise TypeError("verification must be a VerificationLevel")
        if not isinstance(self.children, tuple) or not all(
            isinstance(child, VerificationResult) for child in self.children
        ):
            raise TypeError("children must be VerificationResult values")


def _open_artifact_once(
    path: Path,
    verification: VerificationLevel,
) -> OpenedArtifact[ArtifactManifest]:
    root = ArtifactRoot.open(path, writable=False)
    return open_dispatched_artifact(path, root, verification)


def _open_artifact_from_retained_root(
    path: Path,
    root: ArtifactRoot,
    verification: VerificationLevel,
    allowed_types: tuple[type, ...],
) -> OpenedArtifact[ArtifactManifest]:
    """Dispatch one already-owned root and transfer it only on success."""
    artifact = open_dispatched_artifact(path, root, verification)
    if isinstance(artifact.manifest, allowed_types):
        return artifact
    error = SMLArtifactError(
        f"owned child has unexpected artifact kind: {artifact.manifest.kind!r}"
    )
    try:
        artifact.close()
    except BaseException as cleanup_error:
        raise error from cleanup_error
    raise error


def _open_child_artifact(
    artifact: OpenedArtifact,
    logical_path: str,
    allowed_types: tuple[type, ...],
) -> OpenedArtifact[ArtifactManifest]:
    child_root = artifact.root.open_child(logical_path)
    child_path = artifact.path.joinpath(*logical_path.split("/"))
    return _open_artifact_from_retained_root(
        child_path,
        child_root,
        artifact.verification,
        allowed_types,
    )


def _open_reader_child_artifact(
    reader: CheckpointReader,
    logical_path: str,
    allowed_types: tuple[type, ...],
) -> OpenedArtifact[ArtifactManifest]:
    child_root = reader.open_run_child_root(logical_path)
    child_path = reader.resolved.step_directory.parent.parent.joinpath(
        *logical_path.split("/")
    )
    return _open_artifact_from_retained_root(
        child_path,
        child_root,
        reader.resolved.verification,
        allowed_types,
    )


def _verify_payload(artifact: OpenedArtifact, reference: PayloadRef) -> None:
    with artifact.open_payload(reference):
        pass


def _verify_structural_payloads(
    artifact: OpenedArtifact,
    references: tuple[PayloadRef, ...],
) -> None:
    for reference in references:
        _verify_payload(artifact, reference)


def _tokenizer_binding(
    outer: PretrainingDataManifest | ExportManifest,
    tokenizer: TokenizerManifest,
) -> None:
    if tokenizer.identity != outer.tokenizer_identity:
        raise SMLArtifactError("owned tokenizer identity does not match its manifest")
    expected_model = replace(
        tokenizer.model, logical_path=f"tokenizer/{tokenizer.model.logical_path}"
    )
    expected_vocab = replace(
        tokenizer.vocab, logical_path=f"tokenizer/{tokenizer.vocab.logical_path}"
    )
    if (
        tokenizer.model.logical_path != "tokenizer.model"
        or tokenizer.vocab.logical_path != "tokenizer.vocab"
        or outer.tokenizer_model != expected_model
        or outer.tokenizer_vocab != expected_vocab
    ):
        raise SMLArtifactError(
            "owned tokenizer payload references do not match the nested manifest"
        )


def _verify_opened_tokenizer(
    artifact: OpenedArtifact[TokenizerManifest],
) -> VerificationResult:
    if artifact.verification is VerificationLevel.FULL:
        _load_opened_tokenizer_bundle(artifact)
    else:
        _verify_structural_payloads(
            artifact,
            (artifact.manifest.model, artifact.manifest.vocab),
        )
    return VerificationResult(
        artifact.path,
        artifact.manifest,
        artifact.verification,
    )


def _verify_tokenizer_child(
    artifact: OpenedArtifact,
) -> VerificationResult:
    with _open_child_artifact(
        artifact,
        "tokenizer",
        (TokenizerManifest,),
    ) as tokenizer:
        return _verify_opened_tokenizer(tokenizer)


def _verify_opened_pretraining_data(
    artifact: OpenedArtifact[PretrainingDataManifest],
) -> VerificationResult:
    tokenizer = _verify_tokenizer_child(artifact)
    if not isinstance(tokenizer.manifest, TokenizerManifest):
        raise SMLArtifactError("prepared-data tokenizer has the wrong artifact kind")
    _tokenizer_binding(artifact.manifest, tokenizer.manifest)
    if artifact.verification is VerificationLevel.FULL:
        _verify_opened_pretraining_bundle(
            artifact,
            tokenizer.manifest,
            batch_size=1,
        )
    else:
        _verify_structural_payloads(artifact, artifact.manifest.shards)
    return VerificationResult(
        artifact.path,
        artifact.manifest,
        artifact.verification,
        (tokenizer,),
    )


def _verify_opened_base(
    artifact: OpenedArtifact[BaseSnapshotManifest],
) -> VerificationResult:
    manifest = artifact.manifest
    if artifact.verification is VerificationLevel.FULL:
        validate_base_semantics(manifest)
        verify_safetensors_metadata(artifact, manifest.working_weights)
    else:
        _verify_payload(artifact, manifest.working_weights.payload)
    return VerificationResult(
        artifact.path,
        manifest,
        artifact.verification,
    )


def _verify_opened_swag(
    artifact: OpenedArtifact[SwagDataManifest],
) -> VerificationResult:
    if artifact.verification is VerificationLevel.FULL:
        with _load_opened_swag_bundle(
            artifact,
            validate_projections=True,
        ) as bundle:
            return VerificationResult(
                bundle.path,
                bundle.manifest,
                bundle.verification,
            )
    _verify_structural_payloads(
        artifact,
        tuple(bucket.payload for bucket in artifact.manifest.buckets),
    )
    return VerificationResult(artifact.path, artifact.manifest, artifact.verification)


def _verify_opened_export(
    artifact: OpenedArtifact[ExportManifest],
) -> VerificationResult:
    tokenizer = _verify_tokenizer_child(artifact)
    if not isinstance(tokenizer.manifest, TokenizerManifest):
        raise SMLArtifactError("export tokenizer has the wrong artifact kind")
    _tokenizer_binding(artifact.manifest, tokenizer.manifest)
    manifest = artifact.manifest
    if artifact.verification is VerificationLevel.FULL:
        validate_export_semantics(manifest, tokenizer.manifest)
        verify_safetensors_metadata(artifact, manifest.model_weights)
    else:
        _verify_payload(artifact, manifest.model_weights.payload)
    return VerificationResult(
        artifact.path,
        manifest,
        artifact.verification,
        (tokenizer,),
    )


def _verify_opened_bundle(artifact: OpenedArtifact) -> VerificationResult:
    manifest = artifact.manifest
    if isinstance(manifest, TokenizerManifest):
        return _verify_opened_tokenizer(artifact)
    if isinstance(manifest, PretrainingDataManifest):
        return _verify_opened_pretraining_data(artifact)
    if isinstance(manifest, BaseSnapshotManifest):
        return _verify_opened_base(artifact)
    if isinstance(manifest, SwagDataManifest):
        return _verify_opened_swag(artifact)
    if isinstance(manifest, ExportManifest):
        return _verify_opened_export(artifact)
    raise SMLArtifactError(f"unsupported portable artifact kind: {manifest.kind!r}")


def _verify_opened_run(artifact: OpenedArtifact[RunManifest]) -> VerificationResult:
    level = artifact.verification
    with open_latest_checkpoint_reader(
        artifact.path,
        verification=level,
        load_array_groups=(frozenset() if level is VerificationLevel.FULL else None),
        run_descriptor=artifact.root.fileno(),
    ) as reader:
        resolved = reader.resolved
        if resolved.run != artifact.manifest:
            raise SMLArtifactError("run manifest changed during recursive verification")
        if resolved.latest_recovered:
            raise SMLArtifactError(
                "run latest index must directly bind the latest checkpoint"
            )
        with _open_reader_child_artifact(
            reader,
            "tokenizer",
            (TokenizerManifest,),
        ) as opened:
            tokenizer = _verify_opened_tokenizer(opened)
        if tokenizer.manifest.identity != resolved.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        children: list[VerificationResult] = [tokenizer]
        if isinstance(resolved.run, LoRARunManifest):
            with _open_reader_child_artifact(
                reader,
                "base",
                (BaseSnapshotManifest,),
            ) as opened:
                base = _verify_opened_base(opened)
            if base.manifest.identity != resolved.run.base_identity:
                raise SMLArtifactError(
                    "run base snapshot identity does not match run.json"
                )
            if base.manifest.tokenizer_identity != resolved.run.tokenizer_identity:
                raise SMLArtifactError(
                    "run base tokenizer identity does not match run.json"
                )
            if level is VerificationLevel.FULL:
                require_lora_base_snapshot(base.manifest, resolved.run)
            children.append(base)
        if level is VerificationLevel.FULL:
            validate_full_run_semantics(reader, tokenizer.manifest)
        checkpoint = VerificationResult(
            resolved.step_directory,
            resolved.checkpoint,
            level,
        )
        children.append(checkpoint)
        result = VerificationResult(
            artifact.path,
            resolved.run,
            level,
            tuple(children),
        )
    return result


def verify_artifact(path: Path, full: bool) -> VerificationResult:
    """Verify a supported artifact root and its owned child artifacts."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(full, bool):
        raise TypeError("full must be a bool")
    level = VerificationLevel.FULL if full else VerificationLevel.MANIFEST_TRUSTED
    with _open_artifact_once(path, level) as artifact:
        if isinstance(artifact.manifest, (PretrainingRunManifest, LoRARunManifest)):
            return _verify_opened_run(artifact)
        return _verify_opened_bundle(artifact)


__all__ = ["ArtifactManifest", "VerificationResult", "verify_artifact"]
