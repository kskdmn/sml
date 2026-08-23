"""Typed recursive verification for public artifact roots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sml.artifacts.checkpoint import (
    resolve_latest_step,
    run_access_lock,
)
from sml.artifacts.manifest import (
    CHECKPOINT_MANIFEST_TYPES,
    RUN_MANIFEST_TYPES,
    ArtifactRoot,
    BaseSnapshotManifest,
    CheckpointManifest,
    ExportManifest,
    PretrainingDataManifest,
    PretrainingRunManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    read_manifest,
)
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

_BUNDLE_TYPES: dict[str, type[ArtifactManifest]] = {
    TokenizerManifest.EXPECTED_KIND: TokenizerManifest,
    PretrainingDataManifest.EXPECTED_KIND: PretrainingDataManifest,
    BaseSnapshotManifest.EXPECTED_KIND: BaseSnapshotManifest,
    SwagDataManifest.EXPECTED_KIND: SwagDataManifest,
    ExportManifest.EXPECTED_KIND: ExportManifest,
}


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


def _manifest_kind(path: Path) -> str:
    try:
        with (
            ArtifactRoot.open(path, writable=False) as root,
            root.open_payload("manifest.json") as payload,
        ):
            raw = json.loads(payload.read().decode("utf-8"))
    except SMLArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SMLArtifactError(
            f"invalid artifact manifest at {path}: {error}"
        ) from error
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise SMLArtifactError("artifact manifest must contain a string kind")
    return raw["kind"]


def _verify_bundle(path: Path, level: VerificationLevel) -> VerificationResult:
    kind = _manifest_kind(path)
    try:
        manifest_type = _BUNDLE_TYPES[kind]
    except KeyError as error:
        raise SMLArtifactError(f"unsupported artifact kind: {kind!r}") from error
    verified = read_manifest(path, manifest_type, level)
    manifest = verified.manifest
    children: tuple[VerificationResult, ...] = ()
    if isinstance(manifest, PretrainingDataManifest):
        tokenizer = _verify_bundle(path / "tokenizer", level)
        if not isinstance(tokenizer.manifest, TokenizerManifest):
            raise SMLArtifactError(
                "prepared-data tokenizer child has the wrong artifact kind"
            )
        if tokenizer.manifest.identity != manifest.tokenizer_identity:
            raise SMLArtifactError(
                "prepared-data tokenizer identity does not match its manifest"
            )
        nested_model = tokenizer.manifest.model
        nested_vocab = tokenizer.manifest.vocab
        outer_model = manifest.tokenizer_model
        outer_vocab = manifest.tokenizer_vocab
        if (
            nested_model.logical_path != "tokenizer.model"
            or nested_vocab.logical_path != "tokenizer.vocab"
            or outer_model.logical_path != "tokenizer/tokenizer.model"
            or outer_vocab.logical_path != "tokenizer/tokenizer.vocab"
            or outer_model.identity != nested_model.identity
            or outer_model.byte_size != nested_model.byte_size
            or outer_vocab.identity != nested_vocab.identity
            or outer_vocab.byte_size != nested_vocab.byte_size
        ):
            raise SMLArtifactError(
                "prepared-data tokenizer payload references do not match "
                "the nested manifest"
            )
        children = (tokenizer,)
    return VerificationResult(path, manifest, level, children)


def _verify_run(path: Path, level: VerificationLevel) -> VerificationResult:
    with run_access_lock(path, exclusive=False):
        resolved = resolve_latest_step(
            path,
            writable=False,
            verification=level,
        )
        tokenizer = _verify_bundle(path / "tokenizer", level)
        if not isinstance(tokenizer.manifest, TokenizerManifest):
            raise SMLArtifactError("run tokenizer child has the wrong artifact kind")
        if tokenizer.manifest.identity != resolved.run.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")
        checkpoint = VerificationResult(
            resolved.step_directory,
            resolved.checkpoint,
            resolved.verification,
        )
        return VerificationResult(
            path,
            resolved.run,
            resolved.verification,
            (tokenizer, checkpoint),
        )


def verify_artifact(path: Path, full: bool) -> VerificationResult:
    """Verify a supported artifact root and its owned child artifacts."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(full, bool):
        raise TypeError("full must be a bool")
    level = VerificationLevel.FULL if full else VerificationLevel.MANIFEST_TRUSTED
    if path.joinpath(PretrainingRunManifest.MANIFEST_FILENAME).exists():
        return _verify_run(path, level)
    return _verify_bundle(path, level)


__all__ = ["ArtifactManifest", "VerificationResult", "verify_artifact"]
