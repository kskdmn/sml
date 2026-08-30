"""Exact retained-root dispatch for supported artifact manifests."""

from __future__ import annotations

import os
from pathlib import Path

from sml.artifacts.manifest import (
    RUN_MANIFEST_TYPES,
    ArtifactRoot,
    BaseSnapshotManifest,
    CheckpointManifest,
    ExportManifest,
    OpenedArtifact,
    PretrainingDataManifest,
    PretrainingRunManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    _read_manifest_from_root,
    _same_stable_entry,
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

type _Candidate = tuple[
    str,
    os.stat_result,
    tuple[type[ArtifactManifest], ...],
]


def _candidate_stats(root: ArtifactRoot) -> list[_Candidate]:
    result: list[_Candidate] = []
    for filename, manifest_types in _MANIFEST_CANDIDATES:
        candidate = root._stat_direct_payload(filename)
        if candidate is not None:
            result.append((filename, candidate, manifest_types))
    return result


def open_dispatched_artifact(
    path: Path,
    root: ArtifactRoot,
    verification: VerificationLevel,
) -> OpenedArtifact[ArtifactManifest]:
    """Dispatch exactly one candidate while retaining and transferring *root*."""
    try:
        initial = _candidate_stats(root)
        if len(initial) != 1:
            raise SMLArtifactError(
                "artifact root must contain exactly one of run.json or manifest.json"
            )
        filename, opened_expected, manifest_types = initial[0]

        def validate_opened(opened: os.stat_result) -> None:
            if not _same_stable_entry(opened_expected, opened):
                raise SMLArtifactError("artifact manifest candidate changed")

        def validate_before_close(opened: os.stat_result) -> None:
            final = _candidate_stats(root)
            if len(final) != 1 or final[0][0] != filename:
                raise SMLArtifactError("artifact manifest candidates changed")
            if not _same_stable_entry(opened, final[0][1]):
                raise SMLArtifactError("artifact manifest candidate changed")

        manifest = _read_manifest_from_root(
            root,
            path,
            manifest_types,
            validate_opened=validate_opened,
            validate_before_close=validate_before_close,
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


__all__ = ["ArtifactManifest", "open_dispatched_artifact"]
