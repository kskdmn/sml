"""Typed recursive verification for public artifact roots."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from sml.artifacts.arrays import verify_safetensors_metadata
from sml.artifacts.checkpoint import (
    CheckpointReader,
    open_latest_checkpoint_reader,
    require_lora_base_snapshot,
)
from sml.artifacts.manifest import (
    CHECKPOINT_MANIFEST_TYPES,
    RUN_MANIFEST_TYPES,
    ArtifactRoot,
    BaseSnapshotManifest,
    CheckpointManifest,
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
    _json_object_no_duplicates,
    _parse_manifest,
    _reject_json_constant,
    canonical_json_bytes,
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

type ArtifactManifest = (
    TokenizerManifest
    | PretrainingDataManifest
    | CheckpointManifest
    | RunManifest
    | BaseSnapshotManifest
    | SwagDataManifest
    | ExportManifest
)

_BUNDLE_TYPES: tuple[type[ArtifactManifest], ...] = (
    TokenizerManifest,
    PretrainingDataManifest,
    BaseSnapshotManifest,
    SwagDataManifest,
    ExportManifest,
)
_MANIFEST_CANDIDATES: tuple[tuple[str, tuple[type[ArtifactManifest], ...]], ...] = (
    (PretrainingRunManifest.MANIFEST_FILENAME, RUN_MANIFEST_TYPES),
    (TokenizerManifest.MANIFEST_FILENAME, _BUNDLE_TYPES),
)


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


def _missing_payload(error: SMLArtifactError) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, FileNotFoundError):
            return True
        cause = cause.__cause__
    return False


def _read_candidate_bytes(
    root: ArtifactRoot,
    filename: str,
) -> bytes | None:
    try:
        stream, opened_stat = root._open_payload_with_stat(filename)
    except SMLArtifactError as error:
        if _missing_payload(error):
            return None
        raise
    try:
        encoded = stream.read()
        consumed_stat = os.fstat(stream.fileno())
        opened = (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        )
        consumed = (
            consumed_stat.st_dev,
            consumed_stat.st_ino,
            consumed_stat.st_size,
            consumed_stat.st_mtime_ns,
            consumed_stat.st_ctime_ns,
        )
        if opened != consumed:
            raise SMLArtifactError(f"manifest changed during parsing: {filename}")
    except BaseException as error:
        try:
            stream.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    else:
        stream.close()
        return encoded


def _open_artifact_once(
    path: Path,
    verification: VerificationLevel,
) -> OpenedArtifact[ArtifactManifest]:
    root = ArtifactRoot.open(path, writable=False)
    return _open_artifact_from_retained_root(
        path,
        root,
        verification,
        (*_BUNDLE_TYPES, *RUN_MANIFEST_TYPES),
    )


def _open_artifact_from_retained_root(
    path: Path,
    root: ArtifactRoot,
    verification: VerificationLevel,
    allowed_types: tuple[type, ...],
) -> OpenedArtifact[ArtifactManifest]:
    """Dispatch one already-owned root and transfer it only on success."""
    try:
        candidates: list[tuple[str, bytes, tuple[type[ArtifactManifest], ...]]] = []
        for filename, manifest_types in _MANIFEST_CANDIDATES:
            encoded = _read_candidate_bytes(root, filename)
            if encoded is not None:
                candidates.append((filename, encoded, manifest_types))
        if len(candidates) != 1:
            raise SMLArtifactError(
                "artifact root must contain exactly one of run.json or manifest.json"
            )
        filename, encoded, manifest_types = candidates[0]
        try:
            raw = json.loads(
                encoded.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_json_object_no_duplicates,
            )
            if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
                raise SMLArtifactError(
                    "artifact manifest must contain a string kind discriminator"
                )
            types_by_kind = {
                manifest_type.EXPECTED_KIND: manifest_type
                for manifest_type in manifest_types
            }
            try:
                manifest_type = types_by_kind[raw["kind"]]
            except KeyError as error:
                raise SMLArtifactError(
                    f"unsupported artifact kind in {filename}: {raw['kind']!r}"
                ) from error
            manifest = _parse_manifest(raw, manifest_type)
        except SMLArtifactError:
            raise
        except (
            KeyError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise SMLArtifactError(
                f"invalid artifact manifest at {path}: {error}"
            ) from error
        if manifest.recompute_identity() != manifest.identity:
            raise SMLArtifactError("artifact manifest identity mismatch")
        if encoded != canonical_json_bytes(manifest):
            raise SMLArtifactError("artifact manifest must use canonical JSON bytes")
        if not isinstance(manifest, allowed_types):
            raise SMLArtifactError(
                f"owned child has unexpected artifact kind: {manifest.kind!r}"
            )
        return OpenedArtifact(
            path=path,
            root=root,
            manifest=manifest,
            verification=verification,
        )
    except BaseException as error:
        try:
            root.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


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
