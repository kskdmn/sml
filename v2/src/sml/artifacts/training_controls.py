"""Validate legacy training controls without changing original run configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping

from sml.artifacts.manifest import (
    ArtifactRoot,
    _json_object_no_duplicates,
    _open_stable_payload,
    _reject_json_constant,
    canonical_json_bytes,
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
