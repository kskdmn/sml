from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime

from v2.benchmarks.schema import JsonValue, RawTrial, validate_trial_payload
from v2.benchmarks.workload import structured_identity

CHILD_KIND = "sml-child-trial-measurement"
CHILD_IDENTITY_DOMAIN = "sml-child-trial-measurement-v1"
POST_EXIT_KIND = "sml-parent-post-exit-observation"
POST_EXIT_IDENTITY_DOMAIN = "sml-parent-post-exit-observation-v1"

ENVIRONMENT_FIELDS = frozenset(
    {
        "power_connected",
        "power_mode",
        "low_power_mode",
        "thermal_state",
        "thermal_state_raw_value",
        "memory_pressure",
        "memory_free_percentage",
        "competing_gpu_workload",
    }
)

_THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
_IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _identity_document(kind: str, domain: str, body: dict[str, JsonValue]) -> dict:
    return {**body, "identity": structured_identity(domain, body)}


def _validate_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 identity")
    return value


def _decode_thermal_state(raw_value: int) -> str:
    try:
        return _THERMAL_STATES[raw_value]
    except KeyError as error:
        raise ValueError("unsupported thermal state raw value") from error


def _validate_environment_status(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    if set(raw) != ENVIRONMENT_FIELDS:
        raise ValueError("environment_status has an invalid field set")
    boolean_fields = ("power_connected", "low_power_mode", "competing_gpu_workload")
    for name in boolean_fields:
        if type(raw[name]) is not bool:
            raise ValueError(f"environment_status {name} must be a boolean")
    power_mode = raw["power_mode"]
    if not isinstance(power_mode, str) or not power_mode:
        raise ValueError("environment_status power_mode must be a non-empty string")
    thermal_raw = raw["thermal_state_raw_value"]
    if type(thermal_raw) is not int or thermal_raw not in _THERMAL_STATES:
        raise ValueError("environment_status thermal_state_raw_value is invalid")
    thermal_state = raw["thermal_state"]
    if thermal_state != _decode_thermal_state(thermal_raw):
        raise ValueError("thermal state and raw value disagree")
    memory_pressure = raw["memory_pressure"]
    if memory_pressure not in ("normal", "warning", "critical"):
        raise ValueError("environment_status memory_pressure is invalid")
    free_percentage = raw["memory_free_percentage"]
    if type(free_percentage) is not int or not 0 <= free_percentage <= 100:
        raise ValueError("environment_status memory_free_percentage is invalid")
    return dict(raw)


def _validate_observation(
    raw: Mapping[str, object], *, label: str
) -> dict[str, JsonValue]:
    expected = {
        "observed_at_utc",
        "hardware",
        "environment_status",
        "software_versions",
    }
    if set(raw) != expected:
        raise ValueError(f"{label} has an invalid field set")
    observed_at_utc = raw["observed_at_utc"]
    if not isinstance(observed_at_utc, str):
        raise ValueError(  # noqa: TRY004
            f"{label} timestamp must be a UTC ISO timestamp"
        )
    try:
        observed_at = datetime.fromisoformat(observed_at_utc)
    except ValueError as error:
        raise ValueError(f"{label} timestamp must be a UTC ISO timestamp") from error
    if observed_at.tzinfo is None or observed_at.utcoffset() != UTC.utcoffset(
        observed_at
    ):
        raise ValueError(f"{label} timestamp must be a UTC ISO timestamp")
    hardware = raw["hardware"]
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError(f"{label} hardware must be a non-empty object")
    software_versions = raw["software_versions"]
    if (
        not isinstance(software_versions, dict)
        or not software_versions
        or any(
            not isinstance(value, str) or not value
            for value in software_versions.values()
        )
    ):
        raise ValueError(f"{label} software_versions must be non-empty strings")
    environment_status = raw["environment_status"]
    if not isinstance(environment_status, dict):
        raise ValueError(  # noqa: TRY004
            f"{label} environment_status must be an object"
        )
    return {
        "observed_at_utc": observed_at_utc,
        "hardware": dict(hardware),
        "environment_status": _validate_environment_status(environment_status),
        "software_versions": dict(software_versions),
    }


def build_child_trial_measurement(
    *,
    session_identity: str,
    journal_attempt_index: int,
    trial: Mapping[str, object],
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> dict[str, JsonValue]:
    body = {
        "kind": CHILD_KIND,
        "version": 1,
        "session_identity": _validate_identity(
            session_identity, label="session_identity"
        ),
        "journal_attempt_index": _validate_journal_attempt_index(journal_attempt_index),
        "trial": validate_trial_payload(trial),
        "start": _validate_observation(start, label="child-start observation"),
        "end": _validate_observation(end, label="child-end observation"),
    }
    return _identity_document(CHILD_KIND, CHILD_IDENTITY_DOMAIN, body)


def _validate_journal_attempt_index(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("journal_attempt_index must be a non-negative integer")
    return value


def validate_child_trial_measurement(
    document: Mapping[str, object],
) -> dict[str, JsonValue]:
    expected = {
        "kind",
        "version",
        "session_identity",
        "journal_attempt_index",
        "trial",
        "start",
        "end",
        "identity",
    }
    if set(document) != expected:
        raise ValueError("child measurement has an invalid field set")
    if document["kind"] != CHILD_KIND or document["version"] != 1:
        raise ValueError("unsupported child measurement kind or version")
    trial = document["trial"]
    start = document["start"]
    end = document["end"]
    if (
        not isinstance(trial, dict)
        or not isinstance(start, dict)
        or not isinstance(end, dict)
    ):
        raise ValueError(  # noqa: TRY004
            "child measurement nested fields must be objects"
        )
    body = {
        "kind": CHILD_KIND,
        "version": 1,
        "session_identity": _validate_identity(
            document["session_identity"], label="session_identity"
        ),
        "journal_attempt_index": _validate_journal_attempt_index(
            document["journal_attempt_index"]
        ),
        "trial": validate_trial_payload(trial),
        "start": _validate_observation(start, label="child-start observation"),
        "end": _validate_observation(end, label="child-end observation"),
    }
    identity = _validate_identity(
        document["identity"], label="child measurement identity"
    )
    expected_identity = structured_identity(CHILD_IDENTITY_DOMAIN, body)
    if identity != expected_identity:
        raise ValueError("child measurement identity does not match")
    return {**body, "identity": identity}


def build_post_exit_observation(
    *,
    measurement: Mapping[str, object],
    observed_at_utc: str,
    hardware: Mapping[str, object],
    environment_status: Mapping[str, object],
    software_versions: Mapping[str, object],
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    trial = child["trial"]
    observation = _validate_observation(
        {
            "observed_at_utc": observed_at_utc,
            "hardware": dict(hardware),
            "environment_status": dict(environment_status),
            "software_versions": dict(software_versions),
        },
        label="parent post-exit observation",
    )
    body = {
        "kind": POST_EXIT_KIND,
        "version": 1,
        "session_identity": child["session_identity"],
        "journal_attempt_index": child["journal_attempt_index"],
        "metric": trial["metric"],
        "pair_index": trial["pair_index"],
        "child_measurement_identity": child["identity"],
        **observation,
    }
    return _identity_document(POST_EXIT_KIND, POST_EXIT_IDENTITY_DOMAIN, body)


def validate_post_exit_observation(
    document: Mapping[str, object], *, measurement: Mapping[str, object]
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    expected = {
        "kind",
        "version",
        "session_identity",
        "journal_attempt_index",
        "metric",
        "pair_index",
        "child_measurement_identity",
        "observed_at_utc",
        "hardware",
        "environment_status",
        "software_versions",
        "identity",
    }
    if set(document) != expected:
        raise ValueError("post-exit observation has an invalid field set")
    if document["kind"] != POST_EXIT_KIND or document["version"] != 1:
        raise ValueError("unsupported post-exit observation kind or version")
    observation = _validate_observation(
        {
            name: document[name]
            for name in (
                "observed_at_utc",
                "hardware",
                "environment_status",
                "software_versions",
            )
        },
        label="parent post-exit observation",
    )
    body = {
        "kind": POST_EXIT_KIND,
        "version": 1,
        "session_identity": _validate_identity(
            document["session_identity"], label="session_identity"
        ),
        "journal_attempt_index": _validate_journal_attempt_index(
            document["journal_attempt_index"]
        ),
        "metric": document["metric"],
        "pair_index": document["pair_index"],
        "child_measurement_identity": _validate_identity(
            document["child_measurement_identity"], label="child_measurement_identity"
        ),
        **observation,
    }
    identity = _validate_identity(
        document["identity"], label="post-exit observation identity"
    )
    expected_identity = structured_identity(POST_EXIT_IDENTITY_DOMAIN, body)
    if identity != expected_identity:
        raise ValueError("post-exit observation identity does not match")
    trial = child["trial"]
    for name, expected_value in (
        ("session_identity", child["session_identity"]),
        ("journal_attempt_index", child["journal_attempt_index"]),
        ("metric", trial["metric"]),
        ("pair_index", trial["pair_index"]),
        ("child_measurement_identity", child["identity"]),
    ):
        if body[name] != expected_value:
            raise ValueError(
                f"post-exit observation {name} does not match child measurement"
            )
    return {**body, "identity": identity}


def merge_environment_status(
    start: dict, end: dict, post_exit: dict
) -> dict[str, JsonValue]:
    observations = (start, end, post_exit)
    pressure_order = {"normal": 0, "warning": 1, "critical": 2}
    thermal_raw = max(item["thermal_state_raw_value"] for item in observations)
    measurement_pressure = max(
        (start["memory_pressure"], end["memory_pressure"]),
        key=pressure_order.__getitem__,
    )
    power_modes = {item["power_mode"] for item in observations}
    return {
        "power_connected": all(item["power_connected"] for item in observations),
        "power_mode": power_modes.pop() if len(power_modes) == 1 else "changed",
        "low_power_mode": any(item["low_power_mode"] for item in observations),
        "thermal_state": _decode_thermal_state(thermal_raw),
        "thermal_state_raw_value": thermal_raw,
        "memory_pressure": post_exit["memory_pressure"],
        "memory_free_percentage": post_exit["memory_free_percentage"],
        "measurement_memory_pressure": measurement_pressure,
        "measurement_min_free_percentage": min(
            start["memory_free_percentage"], end["memory_free_percentage"]
        ),
        "competing_gpu_workload": any(
            item["competing_gpu_workload"] for item in observations
        ),
        "start": dict(start),
        "end": dict(end),
        "post_exit": dict(post_exit),
    }


def finalize_raw_trial(
    measurement: Mapping[str, object], post_exit: Mapping[str, object]
) -> RawTrial:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    start = child["start"]
    end = child["end"]
    if start["hardware"] != end["hardware"] or start["hardware"] != parent["hardware"]:
        raise ValueError("evidence hardware changed during measurement")
    if (
        start["software_versions"] != end["software_versions"]
        or start["software_versions"] != parent["software_versions"]
    ):
        raise ValueError("evidence software versions changed during measurement")
    return RawTrial.from_dict(
        {
            "schema_version": 2,
            **child["trial"],
            "software_versions": start["software_versions"],
            "hardware": start["hardware"],
            "environment_status": merge_environment_status(
                start["environment_status"],
                end["environment_status"],
                parent["environment_status"],
            ),
            "evidence_session_identity": child["session_identity"],
            "journal_attempt_index": child["journal_attempt_index"],
            "child_measurement": child,
            "post_exit_observation": parent,
        }
    )


def validate_raw_trial_evidence(trial: RawTrial) -> None:
    expected = finalize_raw_trial(trial.child_measurement, trial.post_exit_observation)
    if expected != trial:
        raise ValueError("raw trial does not match its embedded evidence")
