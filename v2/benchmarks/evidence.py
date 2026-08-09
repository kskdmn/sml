from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from v2.benchmarks.schema import JsonValue, RawTrial, validate_trial_payload
from v2.benchmarks.workload import structured_identity

CHILD_KIND = "sml-child-trial-measurement"
CHILD_IDENTITY_DOMAIN = "sml-child-trial-measurement-v1"
POST_EXIT_KIND = "sml-parent-post-exit-observation"
POST_EXIT_IDENTITY_DOMAIN = "sml-parent-post-exit-observation-v1"
RECOVERY_SAMPLE_KIND = "sml-parent-post-exit-recovery-sample"
RECOVERY_SAMPLE_IDENTITY_DOMAIN = "sml-parent-post-exit-recovery-sample-v1"
RECOVERY_KIND = "sml-parent-post-exit-recovery"
RECOVERY_IDENTITY_DOMAIN = "sml-parent-post-exit-recovery-v1"
MISSING_POST_EXIT_REASON = "missing-immediate-post-exit-evidence"
FINALIZED_REJECTION_REASONS = frozenset(
    {
        "non-normal-start-memory-pressure",
        "critical-measurement-memory-pressure",
        "persistent-post-exit-memory-pressure",
        "non-nominal-thermal",
    }
)
REJECTION_REASONS = FINALIZED_REJECTION_REASONS | {MISSING_POST_EXIT_REASON}

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
    if (
        document["kind"] != CHILD_KIND
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
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
    if (
        document["kind"] != POST_EXIT_KIND
        or type(document["version"]) is not int
        or document["version"] != 1
    ):
        raise ValueError("unsupported post-exit observation kind or version")
    if type(document["pair_index"]) is not int:
        raise ValueError("post-exit observation pair_index must be an integer")
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


def _recovery_binding(
    child: dict[str, JsonValue], parent: dict[str, JsonValue]
) -> dict:
    trial = child["trial"]
    if not isinstance(trial, dict):
        raise ValueError("child measurement trial must be an object")  # noqa: TRY004
    return {
        "session_identity": child["session_identity"],
        "journal_attempt_index": child["journal_attempt_index"],
        "metric": trial["metric"],
        "pair_index": trial["pair_index"],
        "child_measurement_identity": child["identity"],
        "post_exit_observation_identity": parent["identity"],
    }


def _validate_elapsed_seconds(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _validate_recovery_policy(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    expected = {
        "sample_interval_seconds",
        "timeout_seconds",
        "stability_seconds",
        "required_memory_pressure",
        "required_environment",
        "require_same_hardware_and_software",
    }
    if set(raw) != expected:
        raise ValueError("recovery policy has an invalid field set")
    required_environment = raw["required_environment"]
    if not isinstance(required_environment, dict) or required_environment != {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "competing_gpu_workload": False,
    }:
        raise ValueError("recovery policy required_environment drifted")
    policy = {
        "sample_interval_seconds": _validate_elapsed_seconds(
            raw["sample_interval_seconds"], label="recovery sample interval"
        ),
        "timeout_seconds": _validate_elapsed_seconds(
            raw["timeout_seconds"], label="recovery timeout"
        ),
        "stability_seconds": _validate_elapsed_seconds(
            raw["stability_seconds"], label="recovery stability window"
        ),
        "required_memory_pressure": raw["required_memory_pressure"],
        "required_environment": dict(required_environment),
        "require_same_hardware_and_software": raw["require_same_hardware_and_software"],
    }
    if policy != {
        "sample_interval_seconds": 5.0,
        "timeout_seconds": 300.0,
        "stability_seconds": 30.0,
        "required_memory_pressure": "normal",
        "required_environment": dict(required_environment),
        "require_same_hardware_and_software": True,
    }:
        raise ValueError("recovery policy drifted")
    return policy


def _validate_recovery_binding(
    document: Mapping[str, object],
    child: dict[str, JsonValue],
    parent: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    binding = _recovery_binding(child, parent)
    for name, expected_value in binding.items():
        value = document[name]
        if value != expected_value:
            raise ValueError(f"recovery {name} does not match bound evidence")
    return binding


def build_post_exit_recovery_sample(
    *,
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
    sample_index: int,
    previous_sample_identity: str | None,
    observed_at_utc: str,
    elapsed_seconds: float,
    hardware: Mapping[str, object],
    environment_status: Mapping[str, object],
    software_versions: Mapping[str, object],
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("recovery sample_index must be a non-negative integer")
    if previous_sample_identity is not None:
        previous_sample_identity = _validate_identity(
            previous_sample_identity, label="previous sample identity"
        )
    observation = _validate_observation(
        {
            "observed_at_utc": observed_at_utc,
            "hardware": dict(hardware),
            "environment_status": dict(environment_status),
            "software_versions": dict(software_versions),
        },
        label="post-exit recovery sample",
    )
    body = {
        "kind": RECOVERY_SAMPLE_KIND,
        "version": 1,
        **_recovery_binding(child, parent),
        "sample_index": sample_index,
        "previous_sample_identity": previous_sample_identity,
        "elapsed_seconds": _validate_elapsed_seconds(
            elapsed_seconds, label="recovery elapsed_seconds"
        ),
        **observation,
    }
    return _identity_document(
        RECOVERY_SAMPLE_KIND, RECOVERY_SAMPLE_IDENTITY_DOMAIN, body
    )


def validate_post_exit_recovery_sample(
    document: Mapping[str, object],
    *,
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
    previous_sample: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    expected = {
        "kind",
        "version",
        *_recovery_binding(child, parent),
        "sample_index",
        "previous_sample_identity",
        "elapsed_seconds",
        "observed_at_utc",
        "hardware",
        "environment_status",
        "software_versions",
        "identity",
    }
    if set(document) != expected:
        raise ValueError("recovery sample has an invalid field set")
    if document["kind"] != RECOVERY_SAMPLE_KIND or document["version"] != 1:
        raise ValueError("unsupported recovery sample kind or version")
    sample_index = document["sample_index"]
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError("recovery sample_index must be a non-negative integer")
    if previous_sample is None:
        expected_previous_identity = None
    else:
        expected_previous_identity = _validate_identity(
            previous_sample.get("identity"), label="previous sample identity"
        )
    if document["previous_sample_identity"] != expected_previous_identity:
        raise ValueError("recovery previous sample identity does not match")
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
        label="post-exit recovery sample",
    )
    body = {
        "kind": RECOVERY_SAMPLE_KIND,
        "version": 1,
        **_validate_recovery_binding(document, child, parent),
        "sample_index": sample_index,
        "previous_sample_identity": expected_previous_identity,
        "elapsed_seconds": _validate_elapsed_seconds(
            document["elapsed_seconds"], label="recovery elapsed_seconds"
        ),
        **observation,
    }
    identity = _validate_identity(
        document["identity"], label="recovery sample identity"
    )
    if identity != structured_identity(RECOVERY_SAMPLE_IDENTITY_DOMAIN, body):
        raise ValueError("recovery sample identity does not match")
    return {**body, "identity": identity}


def validate_post_exit_recovery_samples(
    documents: Sequence[Mapping[str, object]],
    *,
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
) -> tuple[dict[str, JsonValue], ...]:
    previous = None
    identities: set[str] = set()
    elapsed = -1.0
    samples = []
    for index, document in enumerate(documents):
        sample = validate_post_exit_recovery_sample(
            document,
            measurement=measurement,
            post_exit=post_exit,
            previous_sample=previous,
        )
        if sample["sample_index"] != index:
            raise ValueError("recovery sample indices must be contiguous")
        sample_elapsed = sample["elapsed_seconds"]
        if not isinstance(sample_elapsed, float) or sample_elapsed <= elapsed:
            raise ValueError("recovery sample elapsed_seconds must be increasing")
        if sample_elapsed > 300.0:
            raise ValueError("recovery sample exceeds the timeout")
        identity = sample["identity"]
        if not isinstance(identity, str) or identity in identities:
            raise ValueError("recovery sample identities must be unique")
        identities.add(identity)
        samples.append(sample)
        previous = sample
        elapsed = sample_elapsed
    return tuple(samples)


def _environment_matches_recovery_policy(
    environment: Mapping[str, object], policy: Mapping[str, object]
) -> bool:
    required = policy["required_environment"]
    return environment["memory_pressure"] == policy["required_memory_pressure"] and all(
        environment[name] == value for name, value in required.items()
    )


def _observation_environment_failure_fields(
    observation: Mapping[str, object],
    *,
    policy: Mapping[str, object],
    expected_hardware: Mapping[str, object],
    expected_software_versions: Mapping[str, object],
) -> list[str]:
    required = policy["required_environment"]
    status = observation["environment_status"]
    failures = {
        name
        for name, expected_value in required.items()
        if status[name] != expected_value
    }
    if policy["require_same_hardware_and_software"]:
        if observation["hardware"] != expected_hardware:
            failures.add("hardware")
        if observation["software_versions"] != expected_software_versions:
            failures.add("software_versions")
    return sorted(failures)


def _recovery_environment_failure_fields(
    samples: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
    expected_hardware: Mapping[str, object],
    expected_software_versions: Mapping[str, object],
) -> list[str]:
    return sorted(
        {
            field
            for sample in samples
            for field in _observation_environment_failure_fields(
                sample,
                policy=policy,
                expected_hardware=expected_hardware,
                expected_software_versions=expected_software_versions,
            )
        }
    )


def _validate_recovery_outcome(
    *,
    outcome: object,
    immediate: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    duration_seconds: float,
    failure_fields: list[str],
    expected_hardware: Mapping[str, object],
    expected_software_versions: Mapping[str, object],
) -> str:
    if outcome not in {
        "not-required",
        "recovered",
        "timeout",
        "critical",
        "environment-failure",
        "interrupted",
    }:
        raise ValueError("recovery outcome is invalid")
    immediate_pressure = immediate["environment_status"]["memory_pressure"]
    terminal = (
        samples[-1]["environment_status"]
        if samples
        else immediate["environment_status"]
    )
    has_critical_sample = any(
        sample["environment_status"]["memory_pressure"] == "critical"
        for sample in samples
    )
    requires_critical_outcome = immediate_pressure == "critical" or has_critical_sample
    immediate_failure_fields = _observation_environment_failure_fields(
        immediate,
        policy=policy,
        expected_hardware=expected_hardware,
        expected_software_versions=expected_software_versions,
    )
    sample_failure_fields = _recovery_environment_failure_fields(
        samples,
        policy=policy,
        expected_hardware=expected_hardware,
        expected_software_versions=expected_software_versions,
    )
    environment_failure_fields = immediate_failure_fields or sample_failure_fields
    if (
        outcome == "environment-failure"
        and failure_fields != environment_failure_fields
    ):
        raise ValueError(
            "recovery failure_fields do not match environment failure evidence"
        )
    last_nonmatching = max(
        (
            sample["elapsed_seconds"]
            for sample in samples
            if not _environment_matches_recovery_policy(
                sample["environment_status"], policy
            )
        ),
        default=0.0,
    )
    stable_samples = [
        sample
        for sample in samples
        if sample["elapsed_seconds"] >= last_nonmatching
        and _environment_matches_recovery_policy(sample["environment_status"], policy)
    ]
    has_stable_recovery_window = (
        bool(stable_samples)
        and duration_seconds - stable_samples[0]["elapsed_seconds"]
        >= policy["stability_seconds"]
    )
    if outcome == "not-required":
        valid = (
            immediate_pressure == "normal"
            and not immediate_failure_fields
            and not samples
            and duration_seconds == 0.0
        )
    elif outcome == "recovered":
        valid = (
            immediate_pressure == "warning"
            and bool(samples)
            and not requires_critical_outcome
            and not environment_failure_fields
            and _environment_matches_recovery_policy(terminal, policy)
            and has_stable_recovery_window
        )
    elif outcome == "timeout":
        valid = (
            immediate_pressure == "warning"
            and duration_seconds == policy["timeout_seconds"]
            and not requires_critical_outcome
            and not environment_failure_fields
            and not has_stable_recovery_window
        )
    elif outcome == "critical":
        valid = requires_critical_outcome and not immediate_failure_fields
    elif outcome == "environment-failure":
        valid = (
            bool(immediate_failure_fields) and not samples and duration_seconds == 0.0
        )
        valid = valid or (
            not immediate_failure_fields
            and immediate_pressure == "warning"
            and bool(samples)
            and not requires_critical_outcome
            and bool(sample_failure_fields)
        )
    else:
        valid = (
            immediate_pressure == "warning"
            and not requires_critical_outcome
            and not environment_failure_fields
        )
    if outcome != "environment-failure" and failure_fields:
        valid = False
    if not valid:
        raise ValueError("recovery outcome conflicts with the evidence")
    return outcome


def build_post_exit_recovery(
    *,
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
    outcome: str,
    duration_seconds: float,
    failure_fields: Sequence[str] = (),
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    validated_samples = validate_post_exit_recovery_samples(
        samples, measurement=child, post_exit=parent
    )
    normalized_policy = _validate_recovery_policy(policy)
    if any(not isinstance(field, str) or not field for field in failure_fields):
        raise ValueError("recovery failure_fields must be non-empty strings")
    normalized_failure_fields = sorted(set(failure_fields))
    normalized_duration = _validate_elapsed_seconds(
        duration_seconds, label="recovery duration_seconds"
    )
    if normalized_duration != (
        0.0 if not validated_samples else validated_samples[-1]["elapsed_seconds"]
    ):
        raise ValueError("recovery duration_seconds must match the terminal sample")
    outcome = _validate_recovery_outcome(
        outcome=outcome,
        immediate=parent,
        samples=validated_samples,
        policy=normalized_policy,
        duration_seconds=normalized_duration,
        failure_fields=normalized_failure_fields,
        expected_hardware=child["start"]["hardware"],
        expected_software_versions=child["start"]["software_versions"],
    )
    terminal_environment = (
        parent["environment_status"]
        if not validated_samples
        else validated_samples[-1]["environment_status"]
    )
    body = {
        "kind": RECOVERY_KIND,
        "version": 1,
        **_recovery_binding(child, parent),
        "policy": normalized_policy,
        "sample_identities": [sample["identity"] for sample in validated_samples],
        "outcome": outcome,
        "duration_seconds": normalized_duration,
        "failure_fields": normalized_failure_fields,
        "terminal_sample_identity": (
            None if not validated_samples else validated_samples[-1]["identity"]
        ),
        "terminal_environment_status": terminal_environment,
    }
    return _identity_document(RECOVERY_KIND, RECOVERY_IDENTITY_DOMAIN, body)


def validate_post_exit_recovery(
    document: Mapping[str, object],
    *,
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    validated_samples = validate_post_exit_recovery_samples(
        samples, measurement=child, post_exit=parent
    )
    expected = {
        "kind",
        "version",
        *_recovery_binding(child, parent),
        "policy",
        "sample_identities",
        "outcome",
        "duration_seconds",
        "failure_fields",
        "terminal_sample_identity",
        "terminal_environment_status",
        "identity",
    }
    if set(document) != expected:
        raise ValueError("post-exit recovery has an invalid field set")
    if document["kind"] != RECOVERY_KIND or document["version"] != 1:
        raise ValueError("unsupported post-exit recovery kind or version")
    policy = document["policy"]
    if not isinstance(policy, dict):
        raise ValueError("recovery policy must be an object")  # noqa: TRY004
    normalized_policy = _validate_recovery_policy(policy)
    sample_identities = document["sample_identities"]
    expected_sample_identities = [sample["identity"] for sample in validated_samples]
    if sample_identities != expected_sample_identities:
        raise ValueError("recovery sample identities do not match samples")
    raw_failure_fields = document["failure_fields"]
    if (
        not isinstance(raw_failure_fields, list)
        or any(not isinstance(field, str) or not field for field in raw_failure_fields)
        or raw_failure_fields != sorted(set(raw_failure_fields))
    ):
        raise ValueError("recovery failure_fields must be sorted and duplicate-free")
    duration_seconds = _validate_elapsed_seconds(
        document["duration_seconds"], label="recovery duration_seconds"
    )
    if duration_seconds != (
        0.0 if not validated_samples else validated_samples[-1]["elapsed_seconds"]
    ):
        raise ValueError("recovery duration_seconds must match the terminal sample")
    _validate_recovery_outcome(
        outcome=document["outcome"],
        immediate=parent,
        samples=validated_samples,
        policy=normalized_policy,
        duration_seconds=duration_seconds,
        failure_fields=raw_failure_fields,
        expected_hardware=child["start"]["hardware"],
        expected_software_versions=child["start"]["software_versions"],
    )
    terminal_identity = (
        None if not validated_samples else validated_samples[-1]["identity"]
    )
    if document["terminal_sample_identity"] != terminal_identity:
        raise ValueError("recovery terminal sample identity does not match")
    terminal_environment = (
        parent["environment_status"]
        if not validated_samples
        else validated_samples[-1]["environment_status"]
    )
    if document["terminal_environment_status"] != terminal_environment:
        raise ValueError("recovery terminal environment does not match")
    body = {
        "kind": RECOVERY_KIND,
        "version": 1,
        **_validate_recovery_binding(document, child, parent),
        "policy": normalized_policy,
        "sample_identities": expected_sample_identities,
        "outcome": document["outcome"],
        "duration_seconds": duration_seconds,
        "failure_fields": raw_failure_fields,
        "terminal_sample_identity": terminal_identity,
        "terminal_environment_status": terminal_environment,
    }
    identity = _validate_identity(document["identity"], label="recovery identity")
    if identity != structured_identity(RECOVERY_IDENTITY_DOMAIN, body):
        raise ValueError("recovery identity does not match")
    return {**body, "identity": identity}


def merge_environment_status(
    start: dict,
    end: dict,
    post_exit: dict,
    recovery_samples: Sequence[Mapping[str, object]],
    recovery: Mapping[str, object],
) -> dict[str, JsonValue]:
    recovery_statuses = tuple(
        sample["environment_status"] for sample in recovery_samples
    )
    observations = (start, end, post_exit, *recovery_statuses)
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
        "memory_pressure": (
            post_exit["memory_pressure"]
            if recovery["outcome"] == "not-required" or not recovery_statuses
            else recovery_statuses[-1]["memory_pressure"]
        ),
        "memory_free_percentage": (
            post_exit["memory_free_percentage"]
            if recovery["outcome"] == "not-required" or not recovery_statuses
            else recovery_statuses[-1]["memory_free_percentage"]
        ),
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
        "post_exit_recovery_outcome": recovery["outcome"],
        "post_exit_recovery_final": dict(recovery["terminal_environment_status"]),
    }


def finalized_trial_rejection_reason(
    trial: RawTrial, required_environment: Mapping[str, object]
) -> str | None:
    status = trial.environment_status
    if status["start"]["memory_pressure"] != required_environment["memory_pressure"]:
        return "non-normal-start-memory-pressure"
    if (
        status["end"]["memory_pressure"]
        not in required_environment["measurement_end_memory_pressure_allowed"]
    ):
        return "critical-measurement-memory-pressure"
    if (
        status["post_exit"]["memory_pressure"]
        != required_environment["post_exit_memory_pressure"]
    ):
        return "persistent-post-exit-memory-pressure"
    if any(
        status[name]["thermal_state"] != required_environment["thermal_state"]
        for name in ("start", "end", "post_exit")
    ):
        return "non-nominal-thermal"
    return None


def finalize_raw_trial(
    measurement: Mapping[str, object],
    post_exit: Mapping[str, object],
    recovery_samples: Sequence[Mapping[str, object]],
    recovery: Mapping[str, object],
) -> RawTrial:
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    samples = validate_post_exit_recovery_samples(
        recovery_samples, measurement=child, post_exit=parent
    )
    summary = validate_post_exit_recovery(
        recovery, measurement=child, post_exit=parent, samples=samples
    )
    return RawTrial.from_dict(
        {
            "schema_version": 3,
            **child["trial"],
            "software_versions": child["start"]["software_versions"],
            "hardware": child["start"]["hardware"],
            "environment_status": merge_environment_status(
                child["start"]["environment_status"],
                child["end"]["environment_status"],
                parent["environment_status"],
                samples,
                summary,
            ),
            "evidence_session_identity": child["session_identity"],
            "journal_attempt_index": child["journal_attempt_index"],
            "child_measurement": child,
            "post_exit_observation": parent,
            "post_exit_recovery_samples": list(samples),
            "post_exit_recovery": summary,
        }
    )


def validate_raw_trial_evidence(trial: RawTrial) -> None:
    expected = finalize_raw_trial(
        trial.child_measurement,
        trial.post_exit_observation,
        trial.post_exit_recovery_samples,
        trial.post_exit_recovery,
    )
    if expected != trial:
        raise ValueError("raw trial does not match its embedded evidence")
