"""Durable operational training controls, separate from immutable run configuration."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

from sml.artifacts.checkpoint import OS_FILESYSTEM
from sml.artifacts.manifest import (
    RUN_MANIFEST_TYPES,
    ArtifactRoot,
    VerificationLevel,
    _json_object_no_duplicates,
    _open_artifact_from_root,
    _open_stable_payload,
    _reject_json_constant,
    canonical_json_bytes,
    open_artifact,
)
from sml.errors import SMLArtifactError

TRAINING_CONTROLS_FILENAME = "training-controls.json"
TRAINING_CONTROL_FIELDS = frozenset(
    {"maximum_steps", "maximum_epochs", "log_interval", "checkpoint_interval"}
)


def _validated_controls(values: object) -> dict[str, int | None]:
    if not isinstance(values, Mapping) or set(values) != TRAINING_CONTROL_FIELDS:
        raise SMLArtifactError("training controls have invalid fields")
    controls = dict(values)
    for name, value in controls.items():
        if value is None and name in {"maximum_steps", "maximum_epochs"}:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 2**31 - 1
        ):
            raise SMLArtifactError(f"training control {name} must be a positive int32")
    if controls["maximum_steps"] is None and controls["maximum_epochs"] is None:
        raise SMLArtifactError("training controls require a termination limit")
    return controls


def read_training_controls(
    root: ArtifactRoot, owning_run_identity: str
) -> dict[str, int | None] | None:
    """Read optional controls through their retained run root."""
    try:
        payload = _open_stable_payload(root, TRAINING_CONTROLS_FILENAME)
        with payload as opened:
            encoded = opened.read()
    except SMLArtifactError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return None
        raise
    except OSError as error:
        raise SMLArtifactError("could not read training controls") from error
    try:
        document = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
        if (
            not isinstance(document, dict)
            or set(document) != {"kind", "version", "owning_run_identity", "controls"}
            or document["kind"] != "training-controls"
            or type(document["version"]) is not int
            or document["version"] != 1
            or canonical_json_bytes(document) != encoded
        ):
            raise SMLArtifactError("training controls have an invalid canonical schema")
        if document["owning_run_identity"] != owning_run_identity:
            raise SMLArtifactError("training controls belong to another run")
        return _validated_controls(document["controls"])
    except SMLArtifactError:
        raise
    except (UnicodeError, ValueError, TypeError) as error:
        raise SMLArtifactError("invalid training controls") from error


def load_training_controls(
    run: Path, owning_run_identity: str
) -> dict[str, int | None] | None:
    with open_artifact(
        run, RUN_MANIFEST_TYPES, VerificationLevel.MANIFEST_TRUSTED
    ) as artifact:
        if artifact.manifest.identity != owning_run_identity:
            raise SMLArtifactError("run changed while loading training controls")
        return read_training_controls(artifact.root, owning_run_identity)


def persist_training_controls(
    run: Path, owning_run_identity: str, controls: Mapping[str, int | None]
) -> None:
    """Atomically commit validated controls while the caller owns the run writer lock."""
    values = _validated_controls(controls)
    root = ArtifactRoot.open(run, writable=True)
    with _open_artifact_from_root(
        run, root, RUN_MANIFEST_TYPES, VerificationLevel.MANIFEST_TRUSTED
    ) as artifact:
        if artifact.manifest.identity != owning_run_identity:
            raise SMLArtifactError("run changed while persisting training controls")
        if read_training_controls(root, owning_run_identity) == values:
            return
        encoded = canonical_json_bytes(
            {
                "kind": "training-controls",
                "version": 1,
                "owning_run_identity": owning_run_identity,
                "controls": values,
            }
        )
        temporary_name = f".sml-tmp-training-controls-{uuid.uuid4().hex}"
        descriptor = -1
        replaced = False
        fs = OS_FILESYSTEM
        try:
            descriptor = fs.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root.fileno(),
            )
            fs.write_all(descriptor, encoded)
            fs.fsync_file(descriptor)
            os.close(descriptor)
            descriptor = -1
            fs.replace(
                temporary_name,
                TRAINING_CONTROLS_FILENAME,
                source_dir_fd=root.fileno(),
                destination_dir_fd=root.fileno(),
            )
            replaced = True
            fs.fsync_directory(root.fileno())
        except OSError as error:
            raise SMLArtifactError(
                "could not durably publish training controls"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not replaced:
                try:
                    fs.unlink(temporary_name, dir_fd=root.fileno())
                except FileNotFoundError:
                    pass
