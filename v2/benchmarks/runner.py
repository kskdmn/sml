from __future__ import annotations

# Benchmark artifact validation reports malformed content uniformly as ValueError.
# ruff: noqa: TRY004
import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Literal

from v2.benchmarks.analysis import analyze_pairs
from v2.benchmarks.evidence import (
    build_child_trial_measurement,
    build_post_exit_observation,
    build_post_exit_recovery,
    build_post_exit_recovery_sample,
    finalize_raw_trial,
    finalized_trial_rejection_reason,
    validate_child_trial_measurement,
    validate_raw_trial_evidence,
)
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    JournalAttempt,
    atomic_write_json,
    atomic_write_text,
    baseline_output_lock,
    baseline_session_lock,
    build_session_document,
    cleanup_orphaned_atomic_temporaries,
    cleanup_orphaned_journal_temporaries,
    read_json_object,
    require_external_state_directory,
)
from v2.benchmarks.recovery import (
    ThermalRecoveryTimeout,
    wait_for_nominal_thermal_window,
    wait_for_post_exit_memory_recovery,
)
from v2.benchmarks.schema import METRIC_NAMES, CanonicalWorkload, MetricName, RawTrial
from v2.benchmarks.workload import (
    DEFAULT_MEASURED_UNITS,
    LEGACY_PRECISION_POLICY,
    REPLACEMENT_PRECISION_POLICY,
    WARMUP_UNITS,
    build_canonical_workload,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    canonical_workload_identity,
    fixed_canonical_rows,
    harness_content_identity,
    post_exit_recovery_policy,
    structured_identity,
    write_paired_pretraining_representations,
)

Side = Literal["reference", "candidate"]


@dataclass(frozen=True, slots=True)
class TrialEnvironmentDisposition:
    outcome: Literal["accept", "thermal-reject", "memory-reject", "environment-reject"]
    reason: str | None


class MemoryPressureTrialRejected(RuntimeError):
    def __init__(self, slot: BaselineSlot, reason: str):
        self.slot = slot
        self.reason = reason
        super().__init__(f"{slot.metric} pair {slot.pair_index}: {reason}")


class EnvironmentTrialRejected(RuntimeError):
    def __init__(self, slot: BaselineSlot, reason: str):
        self.slot = slot
        self.reason = reason
        super().__init__(f"{slot.metric} pair {slot.pair_index}: {reason}")


PINNED_BASELINE_SOURCE_COMMIT = "3687f8b3214a44c675ae67af52e4997762f6c634"
COMPARISON_SCREEN = "screen"
COMPARISON_FINAL = "final"
SCREEN_PAIRS = 5
FINAL_PAIRS = 10
BOOTSTRAP_RESAMPLES = 10_000
SCREEN_MAXIMUM_DISPERSION = 0.02
FINAL_MAXIMUM_DISPERSION = 0.015
DEFAULT_MINIMUM_RATIO = 0.97
FINAL_PRETRAINING_MINIMUM_RATIO = 1.03
FINAL_METRICS: tuple[MetricName, ...] = (
    "prepared-data",
    "pretraining-end-to-end",
    "swag-end-to-end",
    "inference-prefill",
    "inference-decode",
    "checkpoint-pause",
    "compile-cold-start",
    "peak-metal-memory",
)
FINAL_PREDECESSOR_METRICS = frozenset(
    {
        "prepared-data",
        "pretraining-end-to-end",
        "swag-end-to-end",
        "inference-prefill",
        "inference-decode",
    }
)
PHASE_METRICS: dict[int, tuple[MetricName, ...]] = {
    1: ("pretraining-compute", "inference-prefill", "inference-decode"),
    2: ("prepared-data",),
    3: (
        "prepared-data",
        "pretraining-end-to-end",
        "checkpoint-pause",
        "peak-metal-memory",
    ),
    4: ("inference-prefill", "inference-decode"),
    5: ("swag-end-to-end",),
}
PHASE_PREDECESSOR_METRICS: dict[int, frozenset[MetricName]] = {
    1: frozenset(),
    2: frozenset(),
    3: frozenset({"prepared-data"}),
    4: frozenset({"inference-prefill", "inference-decode"}),
    5: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ProcessMeasurement:
    elapsed_seconds: float
    value: float
    work_count: float
    compilation_seconds: float | None
    peak_memory_bytes: int | None


def _raw_trial_identity(trial: RawTrial) -> str:
    validate_raw_trial_evidence(trial)
    return structured_identity("sml-raw-benchmark-trial-v3", trial.to_dict())


def _validate_raw_trials_evidence(trials: Sequence[RawTrial]) -> None:
    for trial in trials:
        validate_raw_trial_evidence(trial)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _metric_report_dict(report) -> dict:
    return json.loads(json.dumps(asdict(report)))


def build_baseline_manifest(
    *,
    trials: Sequence[RawTrial],
    workload: CanonicalWorkload,
    workload_identity: str,
    source_commit: str,
    harness_commit: str,
    harness_identity: str,
    command: str,
    pairs: int,
    warmup_units: int,
    measured_units: int,
    paired_representations: dict,
) -> dict:
    _validate_raw_trials_evidence(trials)
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if not trials:
        raise ValueError("baseline requires raw trials")
    metric_records = {}
    work_unit_by_metric = {unit.metric: unit for unit in workload.work_units}
    for metric in METRIC_NAMES:
        metric_trials = tuple(trial for trial in trials if trial.metric == metric)
        if not metric_trials:
            continue
        metric_records[metric] = {
            "direction": work_unit_by_metric[metric].direction,
            "work_unit": work_unit_by_metric[metric].work_unit,
            "raw_trial_identities": [
                _raw_trial_identity(trial) for trial in metric_trials
            ],
            "raw_values": [trial.value for trial in metric_trials],
            "native_configuration": metric_trials[0].native_configuration,
            "native_representation_identity": metric_trials[
                0
            ].native_representation_identity,
            "canonical_input_identity": metric_trials[0].canonical_input_identity,
            "execution_order_identity": metric_trials[0].execution_order_identity,
            "initial_parameter_identity": metric_trials[0].initial_parameter_identity,
        }
    body = {
        "kind": "sml-performance-baseline",
        "version": 1,
        "source": {"commit": source_commit, "clean": True},
        "harness": {
            "commit": harness_commit,
            "clean": True,
            "content_identity": harness_identity,
        },
        "command": command,
        "canonical_workload": workload.to_dict(),
        "canonical_workload_identity": workload_identity,
        "protocol": {
            "pairs": pairs,
            "compilation_passes": 1,
            "warmup_units": warmup_units,
            "measured_units": measured_units,
            "bootstrap_seed": 1729,
            "bootstrap_resamples": 10_000,
            "synchronization_boundaries": list(workload.synchronization_boundaries),
        },
        "hardware": trials[0].hardware,
        "environment_status": trials[0].environment_status,
        "software_versions": trials[0].software_versions,
        "semantic_identities": workload.semantic_identities,
        "paired_pretraining_representations": paired_representations,
        "metrics": metric_records,
    }
    return {
        **body,
        "identity": structured_identity("sml-performance-baseline-v1", body),
    }


def validate_baseline_trial(
    trial: RawTrial,
    *,
    workload: CanonicalWorkload,
    source_commit: str,
    harness_commit: str,
    harness_identity: str,
    expected_hardware: dict,
    expected_software_versions: dict[str, str],
    allow_rejected_environment: bool = False,
) -> None:
    validate_raw_trial_evidence(trial)
    if trial.side != "reference":
        raise ValueError("baseline raw trials must be reference-side records")
    if type(trial.attempt_index) is not int or trial.attempt_index != 0:
        raise ValueError("baseline raw trial attempt_index must be integer zero")
    if type(trial.pair_index) is not int or trial.pair_index not in range(SCREEN_PAIRS):
        raise ValueError("baseline raw trial pair_index is invalid")
    if trial.source_commit != source_commit:
        raise ValueError("raw source commit does not match baseline")
    if trial.harness_commit != harness_commit:
        raise ValueError("raw harness commit does not match baseline")
    if trial.harness_identity != harness_identity:
        raise ValueError("raw harness identity does not match baseline")
    workload_identity = canonical_workload_identity(workload)
    if trial.canonical_workload_identity != workload_identity:
        raise ValueError("raw canonical workload identity does not match baseline")
    if (
        trial.canonical_row_identity
        != workload.semantic_identities["canonical_training_rows"]
    ):
        raise ValueError("raw canonical row identity does not match baseline")
    if trial.canonical_input_identity != canonical_input_identity(
        trial.metric, workload
    ):
        raise ValueError("raw canonical input identity does not match baseline")
    expected_projection = canonical_metric_projection(trial.metric, workload)
    if trial.canonical_projection != expected_projection:
        raise ValueError("raw adapter failed canonical workload round trip")
    if trial.execution_order_identity != canonical_execution_order_identity(
        trial.metric, workload
    ):
        raise ValueError("raw adapter used the wrong logical work order")
    if trial.native_configuration.get(
        "canonical_projection_identity"
    ) != structured_identity("sml-benchmark-metric-projection-v1", expected_projection):
        raise ValueError("raw native projection identity is invalid")
    expected_units = next(
        unit.measured_units
        for unit in workload.work_units
        if unit.metric == trial.metric
    )
    expected_warmup = 0 if trial.metric == "compile-cold-start" else WARMUP_UNITS
    if type(trial.warmup_units) is not int or trial.warmup_units != expected_warmup:
        raise ValueError("raw trial has invalid warmup or measured units: warmup_units")
    if type(trial.measured_units) is not int or trial.measured_units != expected_units:
        raise ValueError(
            "raw trial has invalid warmup or measured units: measured_units"
        )
    if trial.startup_verification_seconds is None:
        raise ValueError("raw trial omitted mandatory startup verification")
    if trial.synchronization_boundaries != workload.synchronization_boundaries:
        raise ValueError("raw synchronization boundaries do not match workload")
    rope_scaling_factor = trial.native_configuration.get("rope_scaling_factor")
    if type(rope_scaling_factor) is not float or rope_scaling_factor != 1.0:
        raise ValueError("raw rope_scaling_factor must be exact float 1.0")
    if (
        isinstance(trial.value, bool)
        or not isinstance(trial.value, Real)
        or not math.isfinite(trial.value)
        or trial.value <= 0
    ):
        raise ValueError("raw benchmark value must be finite, positive, and non-bool")
    _validate_acceptance_environment(
        workload,
        trial,
        allow_rejected_environment=allow_rejected_environment,
    )
    if trial.hardware != expected_hardware:
        raise ValueError("raw hardware records are inconsistent")
    if trial.software_versions != expected_software_versions:
        raise ValueError("raw software-version records are inconsistent")
    _validate_software_versions(workload, trial.software_versions)


def validate_baseline_manifest(
    manifest: dict,
    trials: Sequence[RawTrial],
) -> None:
    _validate_raw_trials_evidence(trials)
    expected_fields = {
        "kind",
        "version",
        "identity",
        "source",
        "harness",
        "command",
        "canonical_workload",
        "canonical_workload_identity",
        "protocol",
        "hardware",
        "environment_status",
        "software_versions",
        "semantic_identities",
        "paired_pretraining_representations",
        "metrics",
    }
    if set(manifest) != expected_fields:
        raise ValueError("baseline manifest has an invalid field set")
    if manifest["kind"] != "sml-performance-baseline" or manifest["version"] != 1:
        raise ValueError("unsupported baseline manifest kind or version")
    body = {key: value for key, value in manifest.items() if key != "identity"}
    expected_identity = structured_identity("sml-performance-baseline-v1", body)
    if manifest["identity"] != expected_identity:
        raise ValueError("baseline manifest identity does not match content")
    workload_raw = manifest["canonical_workload"]
    if not isinstance(workload_raw, dict):
        raise ValueError("baseline canonical workload must be an object")
    workload = CanonicalWorkload.from_dict(workload_raw)
    if workload != build_canonical_workload():
        raise ValueError("baseline canonical workload is not the pinned workload")
    workload_identity = structured_identity(
        "sml-canonical-benchmark-workload-v1", workload.to_dict()
    )
    if manifest["canonical_workload_identity"] != workload_identity:
        raise ValueError("canonical workload identity does not match content")
    paired_representations = manifest["paired_pretraining_representations"]
    if not isinstance(paired_representations, dict):
        raise ValueError("paired pretraining representations must be an object")
    if set(paired_representations) != {
        "canonical_row_identity",
        "row_count",
        "row_width",
        "legacy_format",
        "legacy_dtype",
        "legacy_file_identity",
        "legacy_byte_size",
        "replacement_format",
        "replacement_dtype",
        "replacement_file_identity",
        "replacement_byte_size",
    }:
        raise ValueError("paired pretraining representations have invalid fields")
    if (
        paired_representations.get("canonical_row_identity")
        != workload.semantic_identities["canonical_training_rows"]
    ):
        raise ValueError("paired representations have the wrong canonical rows")
    if (
        paired_representations.get("row_count") != workload.loader["row_count"]
        or paired_representations.get("row_width")
        != int(workload.loader["sequence_length"]) + 1
        or paired_representations.get("legacy_format") != "npz"
        or paired_representations.get("legacy_dtype") != "uint16"
        or paired_representations.get("replacement_format") != "npy"
        or paired_representations.get("replacement_dtype") != "int32"
        or not isinstance(paired_representations.get("legacy_byte_size"), int)
        or paired_representations["legacy_byte_size"] <= 0
        or not isinstance(paired_representations.get("replacement_byte_size"), int)
        or paired_representations["replacement_byte_size"] <= 0
    ):
        raise ValueError("paired representation metadata is invalid")
    for identity_name in ("legacy_file_identity", "replacement_file_identity"):
        if (
            re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(paired_representations.get(identity_name))
            )
            is None
        ):
            raise ValueError("paired representation file identity is invalid")
    source = manifest["source"]
    harness = manifest["harness"]
    protocol = manifest["protocol"]
    metrics = manifest["metrics"]
    if not all(
        isinstance(value, dict) for value in (source, harness, protocol, metrics)
    ):
        raise ValueError(
            "baseline source, harness, protocol, and metrics must be objects"
        )
    if source.get("clean") is not True or harness.get("clean") is not True:
        raise ValueError("baseline checkouts must be clean")
    if source.get("commit") != PINNED_BASELINE_SOURCE_COMMIT:
        raise ValueError("baseline source must be the pinned 3687f8b commit")
    pairs = protocol.get("pairs")
    if pairs != SCREEN_PAIRS:
        raise ValueError("baseline requires exactly five fresh-process trials")
    if protocol != {
        "pairs": SCREEN_PAIRS,
        "compilation_passes": 1,
        "warmup_units": WARMUP_UNITS,
        "measured_units": DEFAULT_MEASURED_UNITS,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }:
        raise ValueError("baseline protocol does not match the pinned protocol")
    if set(metrics) != set(METRIC_NAMES):
        raise ValueError("baseline must contain every benchmark metric")
    if len(trials) != len(METRIC_NAMES) * SCREEN_PAIRS:
        raise ValueError("baseline contains duplicate, missing, or extra raw trials")
    if any(trial.metric not in metrics for trial in trials):
        raise ValueError("baseline contains an unreferenced raw trial")
    for metric, metric_record in metrics.items():
        if metric not in METRIC_NAMES or not isinstance(metric_record, dict):
            raise ValueError(f"unsupported baseline metric: {metric!r}")
        if set(metric_record) != {
            "direction",
            "work_unit",
            "raw_trial_identities",
            "raw_values",
            "native_configuration",
            "native_representation_identity",
            "canonical_input_identity",
            "execution_order_identity",
            "initial_parameter_identity",
        }:
            raise ValueError(f"baseline metric {metric} has invalid fields")
        metric_trials = sorted(
            (trial for trial in trials if trial.metric == metric),
            key=lambda trial: trial.pair_index,
        )
        if len(metric_trials) != pairs:
            raise ValueError(f"baseline metric {metric} has incomplete raw trials")
        if [trial.pair_index for trial in metric_trials] != list(range(pairs)):
            raise ValueError(f"baseline metric {metric} has invalid pair indices")
        for trial in metric_trials:
            validate_baseline_trial(
                trial,
                workload=workload,
                source_commit=source["commit"],
                harness_commit=harness["commit"],
                harness_identity=harness["content_identity"],
                expected_hardware=manifest["hardware"],
                expected_software_versions=manifest["software_versions"],
            )
        expected_trial_identities = [
            _raw_trial_identity(trial) for trial in metric_trials
        ]
        if metric_record.get("raw_trial_identities") != expected_trial_identities:
            raise ValueError(f"baseline metric {metric} raw identities do not match")
        if metric_record.get("raw_values") != [trial.value for trial in metric_trials]:
            raise ValueError(f"baseline metric {metric} raw values do not match")
        if any(
            trial.native_configuration != metric_record.get("native_configuration")
            for trial in metric_trials
        ):
            raise ValueError(f"baseline metric {metric} native configuration changed")
        if any(
            trial.native_representation_identity
            != metric_record.get("native_representation_identity")
            for trial in metric_trials
        ):
            raise ValueError(f"baseline metric {metric} native representation changed")
        for identity_field in (
            "canonical_input_identity",
            "execution_order_identity",
            "initial_parameter_identity",
        ):
            trial_field = {
                "canonical_input_identity": "canonical_input_identity",
                "execution_order_identity": "execution_order_identity",
                "initial_parameter_identity": "initial_parameter_identity",
            }[identity_field]
            if any(
                getattr(trial, trial_field) != metric_record.get(identity_field)
                for trial in metric_trials
            ):
                raise ValueError(f"baseline metric {metric} changed {identity_field}")
    legacy_identity = paired_representations.get("legacy_file_identity")
    pretraining_representation_metrics = {
        "prepared-data",
        "pretraining-compute",
        "pretraining-end-to-end",
        "checkpoint-pause",
        "compile-cold-start",
        "peak-metal-memory",
    }
    if any(
        trial.native_representation_identity != legacy_identity
        for trial in trials
        if trial.metric in pretraining_representation_metrics
    ):
        raise ValueError("baseline training trials do not use the paired legacy rows")
    if manifest["semantic_identities"] != workload.semantic_identities:
        raise ValueError("baseline semantic identities do not match the workload")
    if manifest["hardware"] != trials[0].hardware:
        raise ValueError("baseline hardware summary does not match raw trials")
    if manifest["environment_status"] != trials[0].environment_status:
        raise ValueError("baseline environment summary does not match raw trials")
    if manifest["software_versions"] != trials[0].software_versions:
        raise ValueError("baseline software summary does not match raw trials")


def _validate_observation_power(
    status: dict,
    required: dict,
    *,
    label: str,
    allowed_failures: frozenset[str] = frozenset(),
) -> None:
    for key in ("power_connected", "low_power_mode", "competing_gpu_workload"):
        observed = status.get(key)
        if type(observed) is not bool or (
            key not in allowed_failures and observed is not required[key]
        ):
            raise ValueError(f"{label} does not match required {key}")
    if (
        "power_mode" not in allowed_failures
        and status.get("power_mode") != required["power_mode"]
    ):
        raise ValueError(f"{label} does not match required power_mode")


def _validate_preflight_environment(
    status: dict,
    required: dict,
    *,
    label: str,
) -> None:
    _validate_observation_power(status, required, label=label)
    if status.get("memory_pressure") != required["memory_pressure"]:
        raise ValueError(f"{label} does not match required memory_pressure")


def classify_trial_environment(
    workload: CanonicalWorkload, trial: RawTrial
) -> TrialEnvironmentDisposition:
    validate_raw_trial_evidence(trial)
    reason = finalized_trial_rejection_reason(trial, workload.required_environment)
    if reason is None:
        return TrialEnvironmentDisposition("accept", None)
    if reason == "non-nominal-thermal":
        outcome = "thermal-reject"
    elif reason == "post-exit-recovery-environment-violation":
        outcome = "environment-reject"
    else:
        outcome = "memory-reject"
    return TrialEnvironmentDisposition(outcome, reason)


def _validate_acceptance_environment(
    workload: CanonicalWorkload,
    trial: RawTrial,
    *,
    allow_rejected_environment: bool = False,
) -> None:
    validate_raw_trial_evidence(trial)
    required = workload.required_environment
    disposition = classify_trial_environment(workload, trial)
    valid_outcomes = {
        "accept",
        "thermal-reject",
        "memory-reject",
        "environment-reject",
    }
    if disposition.outcome not in valid_outcomes:
        raise ValueError("raw environment has an unknown disposition")
    documented_recovery_failures = (
        frozenset(trial.post_exit_recovery["failure_fields"])
        if allow_rejected_environment
        and trial.post_exit_recovery["outcome"] == "environment-failure"
        else frozenset()
    )
    interrupted_sample_failures = (
        frozenset(
            field
            for sample in trial.post_exit_recovery_samples
            for field in _recovery_failure_fields(
                sample,
                expected_hardware=trial.hardware,
                expected_software_versions=trial.software_versions,
                required_environment=required,
            )
        )
        if allow_rejected_environment
        and trial.post_exit_recovery["outcome"] == "interrupted"
        else frozenset()
    )
    thermal_rejection = (
        allow_rejected_environment and disposition.outcome == "thermal-reject"
    )
    for key in ("chip", "cpu_cores", "gpu_cores", "unified_memory_bytes"):
        if trial.hardware.get(key) != required[key]:
            raise ValueError(f"raw hardware does not match required {key}")
    _validate_software_versions(workload, trial.software_versions)
    observations = (
        ("start", trial.child_measurement["start"], frozenset()),
        ("end", trial.child_measurement["end"], frozenset()),
        ("post_exit", trial.post_exit_observation, documented_recovery_failures),
        *(
            (
                f"recovery sample {index}",
                sample,
                documented_recovery_failures | interrupted_sample_failures,
            )
            for index, sample in enumerate(trial.post_exit_recovery_samples)
        ),
    )
    statuses = []
    for name, observation, allowed_failures in observations:
        status = observation["environment_status"]
        statuses.append(status)
        validate_thermal_observation(status)
        if (
            status["thermal_state"] != required["thermal_state"]
            and "thermal_state" not in allowed_failures
            and not thermal_rejection
        ):
            raise ValueError(
                f"raw {name} environment does not match required thermal_state"
            )
        _validate_observation_power(
            status,
            required,
            label=f"raw {name} environment",
            allowed_failures=allowed_failures,
        )
        if (
            observation["hardware"] != trial.hardware
            and "hardware" not in allowed_failures
        ):
            raise ValueError(f"raw {name} hardware records are inconsistent")
        if (
            observation["software_versions"] != trial.software_versions
            and "software_versions" not in allowed_failures
        ):
            raise ValueError(f"raw {name} software-version records are inconsistent")
    summary = trial.environment_status
    validate_thermal_observation(
        {
            "thermal_state": summary["thermal_state"],
            "thermal_state_raw_value": summary["thermal_state_raw_value"],
        }
    )
    expected_thermal_raw = max(status["thermal_state_raw_value"] for status in statuses)
    if summary["thermal_state_raw_value"] != expected_thermal_raw:
        raise ValueError("merged thermal state is not the worse observation")
    summary_failures = (documented_recovery_failures | interrupted_sample_failures) | (
        frozenset({"thermal_state"}) if thermal_rejection else frozenset()
    )
    _validate_observation_power(
        summary,
        required,
        label="raw environment summary",
        allowed_failures=summary_failures,
    )
    if not allow_rejected_environment and disposition.outcome != "accept":
        raise ValueError(f"raw environment rejected: {disposition.reason}")


def _validate_software_versions(
    workload: CanonicalWorkload, versions: dict[str, str]
) -> None:
    for package, requirement in workload.software_requirements.items():
        actual = versions.get(package)
        if actual is None or actual == "unavailable":
            raise ValueError(f"required software version is unavailable: {package}")
        if package == "python":
            if actual != requirement:
                raise ValueError("Python version does not match benchmark requirement")
            continue
        if not _version_satisfies(actual, str(requirement)):
            raise ValueError(
                f"software version does not satisfy benchmark requirement: {package}"
            )


def _version_satisfies(actual: str, requirement: str) -> bool:
    match = re.fullmatch(r"(\d+(?:\.\d+)*)", actual)
    if match is None:
        raise ValueError(f"invalid software version: {actual}")
    actual_parts = tuple(int(part) for part in match.group(1).split("."))
    for clause in requirement.split(","):
        clause_match = re.fullmatch(r"(>=|<=|>|<|==)(\d+(?:\.\d+)*)", clause)
        if clause_match is None:
            raise ValueError(f"invalid software requirement: {requirement}")
        operator, expected_text = clause_match.groups()
        expected_parts = tuple(int(part) for part in expected_text.split("."))
        width = max(len(actual_parts), len(expected_parts))
        left = actual_parts + (0,) * (width - len(actual_parts))
        right = expected_parts + (0,) * (width - len(expected_parts))
        if not {
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
            "==": left == right,
        }[operator]:
            return False
    return True


def build_comparison_report(
    *,
    baseline: dict,
    trials: Sequence[RawTrial],
    candidate_commit: str,
    minimum_ratio: float,
    pretraining_minimum_ratio: float | None,
    maximum_dispersion: float,
    require_lower_bound: bool,
    bootstrap_resamples: int,
    predecessor_metrics: dict,
    predecessors: dict[str, dict | None] | None = None,
    previous_comparisons: dict[str, dict] | None = None,
    comparison_mode: str = COMPARISON_SCREEN,
    pairs: int | None = None,
    warmup_units: int = WARMUP_UNITS,
    measured_units: int = DEFAULT_MEASURED_UNITS,
    cooldown_evidence: dict | None = None,
) -> dict:
    _validate_raw_trials_evidence(trials)
    workload = CanonicalWorkload.from_dict(baseline["canonical_workload"])
    baseline_commit = baseline["source"]["commit"]
    baseline_targets = {"baseline", f"baseline:{baseline['identity']}"}
    metric_records = {}
    latest_metrics = json.loads(json.dumps(predecessor_metrics))
    previous_comparisons = previous_comparisons or {}
    for metric in METRIC_NAMES:
        metric_trials = tuple(trial for trial in trials if trial.metric == metric)
        if not metric_trials:
            continue
        direction = next(
            unit.direction for unit in workload.work_units if unit.metric == metric
        )
        baseline_minimum = (
            pretraining_minimum_ratio
            if metric == "pretraining-end-to-end"
            and pretraining_minimum_ratio is not None
            else minimum_ratio
        )
        attempt_indices = sorted(
            {
                trial.attempt_index
                for trial in metric_trials
                if trial.comparison_target in baseline_targets
            }
        )
        if attempt_indices not in ([0], [0, 1]):
            raise ValueError(f"metric {metric} has invalid comparison attempts")
        attempts = []
        all_paired_trials: list[RawTrial] = []
        for attempt_index in attempt_indices:
            reference, candidate = _select_trial_pairs(
                metric_trials,
                metric=metric,
                attempt_index=attempt_index,
                reference_commit=baseline_commit,
                candidate_commit=candidate_commit,
                targets=baseline_targets,
            )
            analysis = analyze_pairs(
                [trial.value for trial in reference],
                [trial.value for trial in candidate],
                direction=direction,
                bootstrap_seed=1729,
                resamples=bootstrap_resamples,
                minimum_ratio=baseline_minimum,
                maximum_dispersion=maximum_dispersion,
                require_lower_bound=require_lower_bound,
            )
            paired_trials = (*reference, *candidate)
            all_paired_trials.extend(paired_trials)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "analysis": _metric_report_dict(analysis),
                    "raw_trial_identities": [
                        _raw_trial_identity(trial) for trial in paired_trials
                    ],
                }
            )
        baseline_analysis = attempts[-1]["analysis"]
        previous_comparison = _normalize_previous_comparison(
            previous_comparisons.get(metric)
        )
        result_body = {
            "metric": metric,
            "candidate_commit": candidate_commit,
            "baseline_identity": baseline["identity"],
            "baseline_comparison": baseline_analysis,
            "previous_comparison": previous_comparison,
            "attempts": attempts,
            "precision_policy": (
                {
                    "reference": LEGACY_PRECISION_POLICY,
                    "candidate": REPLACEMENT_PRECISION_POLICY,
                    "trajectory_equivalent": False,
                }
                if metric == "pretraining-end-to-end"
                else None
            ),
            "raw_trial_identities": [
                _raw_trial_identity(trial) for trial in all_paired_trials
            ],
        }
        result_identity = structured_identity(
            "sml-performance-metric-result-v1", result_body
        )
        metric_records[metric] = {
            **result_body,
            "result_identity": result_identity,
        }
        latest_metrics[metric] = {
            "metric": metric,
            "source_commit": candidate_commit,
            "result_identity": result_identity,
            "canonical_workload_identity": baseline["canonical_workload_identity"],
            "harness_identity": baseline["harness"]["content_identity"],
        }
    if not metric_records:
        raise ValueError("comparison contains no complete metric pairs")
    if predecessors is None:
        predecessors = {metric: None for metric in metric_records}
    if set(predecessors) != set(metric_records):
        raise ValueError("comparison requires one predecessor entry per metric")
    if pairs is None:
        pairs = 1 + max(trial.pair_index for trial in trials)
    body = {
        "kind": "sml-performance-comparison",
        "version": 1,
        "baseline_identity": baseline["identity"],
        "predecessors": predecessors,
        "harness": baseline["harness"],
        "canonical_workload_identity": baseline["canonical_workload_identity"],
        "candidate_commit": candidate_commit,
        "comparison_mode": comparison_mode,
        "protocol": {
            "pairs": pairs,
            "compilation_passes": 1,
            "warmup_units": warmup_units,
            "measured_units": measured_units,
            "bootstrap_seed": 1729,
            "bootstrap_resamples": bootstrap_resamples,
            "minimum_ratio": minimum_ratio,
            "pretraining_minimum_ratio": pretraining_minimum_ratio,
            "maximum_dispersion": maximum_dispersion,
            "require_lower_bound": require_lower_bound,
        },
        "cooldown_evidence": cooldown_evidence,
        "metrics": metric_records,
        "raw_trials": [trial.to_dict() for trial in trials],
        "latest_metrics": latest_metrics,
    }
    return {
        **body,
        "identity": structured_identity("sml-performance-comparison-v1", body),
    }


def _select_trial_pairs(
    trials: Sequence[RawTrial],
    *,
    metric: MetricName,
    attempt_index: int,
    reference_commit: str,
    candidate_commit: str,
    targets: set[str],
) -> tuple[list[RawTrial], list[RawTrial]]:
    reference = sorted(
        (
            trial
            for trial in trials
            if trial.metric == metric
            and trial.attempt_index == attempt_index
            and trial.side == "reference"
            and trial.source_commit == reference_commit
            and trial.comparison_target in targets
        ),
        key=lambda trial: trial.pair_index,
    )
    candidate = sorted(
        (
            trial
            for trial in trials
            if trial.metric == metric
            and trial.attempt_index == attempt_index
            and trial.side == "candidate"
            and trial.source_commit == candidate_commit
            and trial.comparison_target in targets
        ),
        key=lambda trial: trial.pair_index,
    )
    if not reference or len(reference) != len(candidate):
        raise ValueError(
            f"metric {metric} has incomplete attempt {attempt_index} pairs"
        )
    return reference, candidate


def _normalize_previous_comparison(record: dict | None) -> dict | None:
    if record is None or "attempts" in record:
        return record
    return {
        "predecessor_result_identity": record["predecessor_result_identity"],
        "predecessor_source_commit": record["predecessor_source_commit"],
        "analysis": record["analysis"],
        "raw_trial_identities": record["raw_trial_identities"],
        "attempts": [
            {
                "attempt_index": record["attempt_index"],
                "analysis": record["analysis"],
                "raw_trial_identities": record["raw_trial_identities"],
            }
        ],
    }


def _validate_comparison_protocol(report: dict) -> None:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("comparison protocol must be an object")
    mode = report.get("comparison_mode")
    common = {
        "compilation_passes": 1,
        "warmup_units": WARMUP_UNITS,
        "measured_units": DEFAULT_MEASURED_UNITS,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "minimum_ratio": DEFAULT_MINIMUM_RATIO,
    }
    for key, expected in common.items():
        if protocol.get(key) != expected:
            raise ValueError(f"comparison protocol has invalid {key}")
    if mode == COMPARISON_SCREEN:
        if (
            protocol.get("pairs") != SCREEN_PAIRS
            or protocol.get("maximum_dispersion") != SCREEN_MAXIMUM_DISPERSION
            or protocol.get("require_lower_bound") is not False
            or protocol.get("pretraining_minimum_ratio")
            not in (None, FINAL_PRETRAINING_MINIMUM_RATIO)
        ):
            raise ValueError("phase-screen protocol is invalid")
    elif mode == COMPARISON_FINAL:
        if (
            protocol.get("pairs") != FINAL_PAIRS
            or protocol.get("maximum_dispersion") != FINAL_MAXIMUM_DISPERSION
            or protocol.get("require_lower_bound") is not True
            or protocol.get("pretraining_minimum_ratio")
            != FINAL_PRETRAINING_MINIMUM_RATIO
        ):
            raise ValueError("final-acceptance protocol is invalid")
    else:
        raise ValueError("comparison_mode must be screen or final")


def _validate_comparison_trial_pair(
    *,
    reference_trial: RawTrial,
    candidate_trial: RawTrial,
    pair_index: int,
    attempt_index: int,
    baseline: dict,
    workload: CanonicalWorkload,
    expected_projection: dict,
    expected_input_identity: str,
    expected_warmup: int,
    expected_units: int,
) -> None:
    _validate_raw_trials_evidence((reference_trial, candidate_trial))
    if (
        reference_trial.pair_index != pair_index
        or candidate_trial.pair_index != pair_index
        or reference_trial.attempt_index != attempt_index
        or candidate_trial.attempt_index != attempt_index
    ):
        raise ValueError("comparison pair indices are not contiguous")
    order = process_order(pair_index)
    if reference_trial.process_order != order.index(
        "reference"
    ) or candidate_trial.process_order != order.index("candidate"):
        raise ValueError("comparison does not use alternating process order")
    for trial in (reference_trial, candidate_trial):
        if trial.harness_commit != baseline["harness"]["commit"]:
            raise ValueError("comparison raw harness commit does not match")
        if trial.harness_identity != baseline["harness"]["content_identity"]:
            raise ValueError("comparison raw harness identity does not match")
        if trial.canonical_workload_identity != baseline["canonical_workload_identity"]:
            raise ValueError("comparison raw workload identity does not match")
        if (
            trial.canonical_row_identity
            != workload.semantic_identities["canonical_training_rows"]
        ):
            raise ValueError("comparison raw row identity does not match")
        if trial.canonical_input_identity != expected_input_identity:
            raise ValueError("comparison raw semantic input identity does not match")
        if trial.canonical_projection != expected_projection:
            raise ValueError("comparison adapter failed canonical round trip")
        if trial.execution_order_identity != canonical_execution_order_identity(
            trial.metric, workload
        ):
            raise ValueError("comparison adapter used a different logical work order")
        if (
            trial.warmup_units != expected_warmup
            or trial.measured_units != expected_units
        ):
            raise ValueError("comparison raw trial uses invalid work-unit counts")
        if trial.startup_verification_seconds is None:
            raise ValueError("comparison raw trial omitted input verification")
        _validate_acceptance_environment(workload, trial)
        _validate_software_versions(workload, trial.software_versions)
    if (
        reference_trial.initial_parameter_identity
        != candidate_trial.initial_parameter_identity
    ):
        raise ValueError("comparison sides use different initial parameters")
    if (
        reference_trial.native_configuration.get("parameter_precision_policy")
        != LEGACY_PRECISION_POLICY
        or candidate_trial.native_configuration.get("parameter_precision_policy")
        != REPLACEMENT_PRECISION_POLICY
    ):
        raise ValueError("comparison precision-policy proof is invalid")


def _validate_predecessor_trial_pair(
    *,
    reference_trial: RawTrial,
    candidate_trial: RawTrial,
    pair_index: int,
    attempt_index: int,
    baseline: dict,
    workload: CanonicalWorkload,
    expected_projection: dict,
    expected_input_identity: str,
    expected_warmup: int,
    expected_units: int,
) -> None:
    _validate_raw_trials_evidence((reference_trial, candidate_trial))
    if (
        reference_trial.pair_index != pair_index
        or candidate_trial.pair_index != pair_index
        or reference_trial.attempt_index != attempt_index
        or candidate_trial.attempt_index != attempt_index
    ):
        raise ValueError("direct predecessor pair indices are invalid")
    order = process_order(pair_index)
    if reference_trial.process_order != order.index(
        "reference"
    ) or candidate_trial.process_order != order.index("candidate"):
        raise ValueError("direct predecessor comparison has invalid process order")
    for trial in (reference_trial, candidate_trial):
        if (
            trial.harness_commit != baseline["harness"]["commit"]
            or trial.harness_identity != baseline["harness"]["content_identity"]
            or trial.canonical_workload_identity
            != baseline["canonical_workload_identity"]
            or trial.canonical_row_identity
            != workload.semantic_identities["canonical_training_rows"]
            or trial.canonical_input_identity != expected_input_identity
            or trial.canonical_projection != expected_projection
            or trial.execution_order_identity
            != canonical_execution_order_identity(trial.metric, workload)
            or trial.warmup_units != expected_warmup
            or trial.measured_units != expected_units
            or trial.startup_verification_seconds is None
        ):
            raise ValueError("direct predecessor canonical proof is invalid")
        if (
            trial.native_configuration.get("parameter_precision_policy")
            != REPLACEMENT_PRECISION_POLICY
        ):
            raise ValueError("direct predecessor precision policy is invalid")
        _validate_acceptance_environment(workload, trial)
        _validate_software_versions(workload, trial.software_versions)
    if (
        reference_trial.initial_parameter_identity
        != candidate_trial.initial_parameter_identity
    ):
        raise ValueError("direct predecessor initial parameters differ")


def comparison_has_noise(report: dict, *, attempt_index: int = 0) -> bool:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("comparison metrics must be an object")
    for metric_record in metrics.values():
        attempts = metric_record.get("attempts")
        if (
            isinstance(attempts, list)
            and len(attempts) > attempt_index
            and attempts[attempt_index]["analysis"]["decision"] == "too-noisy"
        ):
            return True
        previous = metric_record.get("previous_comparison")
        if isinstance(previous, dict):
            previous_attempts = previous.get("attempts")
            if (
                isinstance(previous_attempts, list)
                and len(previous_attempts) > attempt_index
                and previous_attempts[attempt_index]["analysis"]["decision"]
                == "too-noisy"
            ):
                return True
    return False


def validate_cooldown_evidence(
    evidence: dict | None,
    required_environment: dict[str, object],
) -> None:
    if not isinstance(evidence, dict) or set(evidence) != {
        "duration_seconds",
        "sample_interval_seconds",
        "samples",
    }:
        raise ValueError("noisy retry requires complete cooldown evidence")
    duration = evidence["duration_seconds"]
    interval = evidence["sample_interval_seconds"]
    samples = evidence["samples"]
    if (
        not isinstance(duration, (int, float))
        or duration < 900
        or not isinstance(interval, (int, float))
        or interval <= 0
        or interval > 60
        or not isinstance(samples, list)
        or not samples
    ):
        raise ValueError("cooldown duration or sampling interval is invalid")
    prior_elapsed: float | None = None
    last_window = []
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "elapsed_seconds",
            "environment_status",
        }:
            raise ValueError("cooldown sample is invalid")
        elapsed = sample["elapsed_seconds"]
        status = sample["environment_status"]
        if (
            not isinstance(elapsed, (int, float))
            or elapsed < 0
            or not isinstance(status, dict)
        ):
            raise ValueError("cooldown sample ordering is invalid")
        elapsed = float(elapsed)
        if (prior_elapsed is None and elapsed > interval) or (
            prior_elapsed is not None
            and (elapsed <= prior_elapsed or elapsed - prior_elapsed > interval)
        ):
            raise ValueError("cooldown evidence contains a sampling gap")
        prior_elapsed = elapsed
        if (
            status.get("power_connected") is not True
            or status.get("power_mode") != required_environment["power_mode"]
            or status.get("low_power_mode") is not False
        ):
            raise ValueError("cooldown did not remain on the required power mode")
        if elapsed >= duration - 300:
            last_window.append(status)
    if prior_elapsed is None or prior_elapsed < duration or len(last_window) < 5:
        raise ValueError("cooldown does not prove the final five-minute window")
    if any(
        status.get("thermal_state") != "nominal"
        or status.get("memory_pressure") != "normal"
        or status.get("competing_gpu_workload") is not False
        for status in last_window
    ):
        raise ValueError("cooldown final window was not nominal")


def perform_cooldown(
    *,
    collect=None,
    clock=time.monotonic,
    sleep=time.sleep,
    duration_seconds: float = 900.0,
    sample_interval_seconds: float = 60.0,
) -> dict:
    if duration_seconds < 900 or not 0 < sample_interval_seconds <= 60:
        raise ValueError("cooldown must last 15 minutes with at most 60-second samples")
    if collect is None:
        collect = collect_environment
    started = clock()
    next_sample = started
    sampling_cadence = sample_interval_seconds * 0.9
    samples = []
    while True:
        _hardware, status, _software = collect()
        elapsed = clock() - started
        samples.append(
            {
                "elapsed_seconds": elapsed,
                "environment_status": status,
            }
        )
        if elapsed >= duration_seconds:
            break
        next_sample = min(
            started + duration_seconds,
            next_sample + sampling_cadence,
        )
        sleep(max(0.0, next_sample - clock()))
    return {
        "duration_seconds": samples[-1]["elapsed_seconds"],
        "sample_interval_seconds": sample_interval_seconds,
        "samples": samples,
    }


def validate_comparison_report(
    report: dict,
    baseline: dict,
    predecessor_reports: dict[str, dict | None] | None,
    *,
    _use_embedded_predecessors: bool = False,
) -> None:
    expected_fields = {
        "kind",
        "version",
        "identity",
        "baseline_identity",
        "predecessors",
        "harness",
        "canonical_workload_identity",
        "candidate_commit",
        "comparison_mode",
        "protocol",
        "cooldown_evidence",
        "metrics",
        "raw_trials",
        "latest_metrics",
    }
    if set(report) != expected_fields:
        raise ValueError("comparison report has an invalid field set")
    if report["kind"] != "sml-performance-comparison" or report["version"] != 1:
        raise ValueError("unsupported comparison report kind or version")
    body = {key: value for key, value in report.items() if key != "identity"}
    expected_identity = structured_identity("sml-performance-comparison-v1", body)
    if report["identity"] != expected_identity:
        raise ValueError("comparison report identity does not match content")
    if report["baseline_identity"] != baseline["identity"]:
        raise ValueError("comparison report names the wrong baseline")
    if report["harness"] != baseline["harness"]:
        raise ValueError("comparison harness does not match baseline")
    if report["canonical_workload_identity"] != baseline["canonical_workload_identity"]:
        raise ValueError("comparison workload does not match baseline")
    raw_trials = report["raw_trials"]
    if not isinstance(raw_trials, list):
        raise ValueError("comparison raw_trials must be a list")
    trials = tuple(
        RawTrial.from_dict(raw) if isinstance(raw, dict) else None for raw in raw_trials
    )
    if any(trial is None for trial in trials):
        raise ValueError("comparison raw trials must be objects")
    _validate_raw_trials_evidence(trials)
    candidate_commit = report["candidate_commit"]
    baseline_commit = baseline["source"]["commit"]
    baseline_targets = {"baseline", f"baseline:{baseline['identity']}"}
    protocol = report["protocol"]
    metrics = report["metrics"]
    if not isinstance(protocol, dict) or not isinstance(metrics, dict):
        raise ValueError("comparison protocol and metrics must be objects")
    _validate_comparison_protocol(report)
    if _use_embedded_predecessors:
        embedded = report.get("predecessors")
        if not isinstance(embedded, dict) or set(embedded) != set(metrics):
            raise ValueError("comparison embedded predecessors are invalid")
        predecessor_metrics = {}
        expected_predecessors = json.loads(json.dumps(embedded))
        for metric, proof in embedded.items():
            if proof is None:
                continue
            if (
                not isinstance(proof, dict)
                or set(proof) != {"report_identity", "result_identity"}
                or re.fullmatch(r"sha256:[0-9a-f]{64}", proof["report_identity"])
                is None
                or re.fullmatch(r"sha256:[0-9a-f]{64}", proof["result_identity"])
                is None
            ):
                raise ValueError("comparison embedded predecessor proof is invalid")
            metric_record = metrics.get(metric)
            previous = (
                metric_record.get("previous_comparison")
                if isinstance(metric_record, dict)
                else None
            )
            if (
                not isinstance(previous, dict)
                or previous.get("predecessor_result_identity")
                != proof["result_identity"]
                or re.fullmatch(
                    r"[0-9a-f]{40}",
                    str(previous.get("predecessor_source_commit")),
                )
                is None
            ):
                raise ValueError("comparison embedded predecessor lineage is invalid")
            predecessor_metrics[metric] = {
                "metric": metric,
                "source_commit": previous["predecessor_source_commit"],
                "result_identity": proof["result_identity"],
                "canonical_workload_identity": baseline["canonical_workload_identity"],
                "harness_identity": baseline["harness"]["content_identity"],
            }
    else:
        if predecessor_reports is None:
            predecessor_reports = {metric: None for metric in metrics}
        if set(predecessor_reports) != set(metrics):
            raise ValueError("validator requires one predecessor entry per metric")
        expected_predecessors = {}
        predecessor_metrics = {}
        for metric, predecessor_report in predecessor_reports.items():
            if predecessor_report is None:
                expected_predecessors[metric] = None
                continue
            _validate_comparison_document(predecessor_report, baseline)
            lineage = predecessor_report.get("latest_metrics", {}).get(metric)
            metric_record = predecessor_report.get("metrics", {}).get(metric)
            if (
                not isinstance(lineage, dict)
                or not isinstance(metric_record, dict)
                or lineage.get("result_identity")
                != metric_record.get("result_identity")
            ):
                raise ValueError("predecessor is stale or did not measure the metric")
            expected_predecessors[metric] = {
                "report_identity": predecessor_report["identity"],
                "result_identity": lineage["result_identity"],
            }
            predecessor_metrics[metric] = lineage
    if report["predecessors"] != expected_predecessors:
        raise ValueError("comparison report names the wrong per-metric predecessors")
    workload = CanonicalWorkload.from_dict(baseline["canonical_workload"])
    expected_latest = json.loads(json.dumps(predecessor_metrics))
    for metric, metric_record in metrics.items():
        if metric not in METRIC_NAMES or not isinstance(metric_record, dict):
            raise ValueError(f"unsupported comparison metric: {metric!r}")
        expected_projection = canonical_metric_projection(metric, workload)
        expected_input_identity = canonical_input_identity(metric, workload)
        expected_units = next(
            unit.measured_units for unit in workload.work_units if unit.metric == metric
        )
        expected_warmup = 0 if metric == "compile-cold-start" else WARMUP_UNITS
        direction = next(
            unit.direction for unit in workload.work_units if unit.metric == metric
        )
        minimum_ratio = (
            protocol["pretraining_minimum_ratio"]
            if metric == "pretraining-end-to-end"
            and protocol["pretraining_minimum_ratio"] is not None
            else protocol["minimum_ratio"]
        )
        attempt_records = metric_record.get("attempts")
        if not isinstance(attempt_records, list) or len(attempt_records) not in (1, 2):
            raise ValueError("comparison metric has an invalid attempt record")
        expected_attempt_indices = list(range(len(attempt_records)))
        if [record.get("attempt_index") for record in attempt_records] != (
            expected_attempt_indices
        ):
            raise ValueError("comparison attempt indices are invalid")
        all_baseline_trials: list[RawTrial] = []
        recomputed = None
        for attempt_record in attempt_records:
            attempt_index = attempt_record["attempt_index"]
            reference, candidate = _select_trial_pairs(
                trials,
                metric=metric,
                attempt_index=attempt_index,
                reference_commit=baseline_commit,
                candidate_commit=candidate_commit,
                targets=baseline_targets,
            )
            if len(reference) != protocol["pairs"]:
                raise ValueError("comparison metric has the wrong number of pairs")
            all_baseline_trials.extend((*reference, *candidate))
            for pair_index, (reference_trial, candidate_trial) in enumerate(
                zip(reference, candidate, strict=True)
            ):
                _validate_comparison_trial_pair(
                    reference_trial=reference_trial,
                    candidate_trial=candidate_trial,
                    pair_index=pair_index,
                    attempt_index=attempt_index,
                    baseline=baseline,
                    workload=workload,
                    expected_projection=expected_projection,
                    expected_input_identity=expected_input_identity,
                    expected_warmup=expected_warmup,
                    expected_units=expected_units,
                )
            recomputed = analyze_pairs(
                [trial.value for trial in reference],
                [trial.value for trial in candidate],
                direction=direction,
                bootstrap_seed=1729,
                resamples=protocol["bootstrap_resamples"],
                minimum_ratio=minimum_ratio,
                maximum_dispersion=protocol["maximum_dispersion"],
                require_lower_bound=protocol["require_lower_bound"],
            )
            expected_attempt_record = {
                "attempt_index": attempt_index,
                "analysis": _metric_report_dict(recomputed),
                "raw_trial_identities": [
                    _raw_trial_identity(trial) for trial in (*reference, *candidate)
                ],
            }
            if attempt_record != expected_attempt_record:
                raise ValueError("comparison attempt analysis does not match raw pairs")
        assert recomputed is not None
        if metric_record.get("baseline_comparison") != _metric_report_dict(recomputed):
            raise ValueError("comparison final analysis does not match last attempt")
        if metric_record.get("raw_trial_identities") != [
            _raw_trial_identity(trial) for trial in all_baseline_trials
        ]:
            raise ValueError("comparison raw-trial identities do not match attempts")
        expected_precision_policy = (
            {
                "reference": LEGACY_PRECISION_POLICY,
                "candidate": REPLACEMENT_PRECISION_POLICY,
                "trajectory_equivalent": False,
            }
            if metric == "pretraining-end-to-end"
            else None
        )
        if metric_record.get("precision_policy") != expected_precision_policy:
            raise ValueError("comparison precision annotation is invalid")
        predecessor = predecessor_metrics.get(metric)
        previous_comparison = metric_record.get("previous_comparison")
        if predecessor is None or predecessor["source_commit"] == baseline_commit:
            if previous_comparison is not None:
                raise ValueError("first metric appearance cannot name a predecessor")
        else:
            if not isinstance(previous_comparison, dict):
                raise ValueError("comparison is missing direct predecessor pairs")
            target = f"previous:{predecessor['result_identity']}"
            previous_attempt_records = previous_comparison.get("attempts")
            if not isinstance(previous_attempt_records, list) or len(
                previous_attempt_records
            ) != len(attempt_records):
                raise ValueError("direct predecessor attempts do not match baseline")
            previous_analysis = None
            all_previous_trials: list[RawTrial] = []
            for previous_attempt_record in previous_attempt_records:
                attempt_index = previous_attempt_record.get("attempt_index")
                previous_reference, previous_candidate = _select_trial_pairs(
                    trials,
                    metric=metric,
                    attempt_index=attempt_index,
                    reference_commit=predecessor["source_commit"],
                    candidate_commit=candidate_commit,
                    targets={target},
                )
                if len(previous_reference) != protocol["pairs"]:
                    raise ValueError("direct predecessor has the wrong pair count")
                all_previous_trials.extend((*previous_reference, *previous_candidate))
                for pair_index, (reference_trial, candidate_trial) in enumerate(
                    zip(previous_reference, previous_candidate, strict=True)
                ):
                    _validate_predecessor_trial_pair(
                        reference_trial=reference_trial,
                        candidate_trial=candidate_trial,
                        pair_index=pair_index,
                        attempt_index=attempt_index,
                        baseline=baseline,
                        workload=workload,
                        expected_projection=expected_projection,
                        expected_input_identity=expected_input_identity,
                        expected_warmup=expected_warmup,
                        expected_units=expected_units,
                    )
                previous_analysis = analyze_pairs(
                    [trial.value for trial in previous_reference],
                    [trial.value for trial in previous_candidate],
                    direction=direction,
                    bootstrap_seed=1729,
                    resamples=protocol["bootstrap_resamples"],
                    minimum_ratio=protocol["minimum_ratio"],
                    maximum_dispersion=protocol["maximum_dispersion"],
                    require_lower_bound=protocol["require_lower_bound"],
                )
                if previous_attempt_record != {
                    "attempt_index": attempt_index,
                    "analysis": _metric_report_dict(previous_analysis),
                    "raw_trial_identities": [
                        _raw_trial_identity(trial)
                        for trial in (*previous_reference, *previous_candidate)
                    ],
                }:
                    raise ValueError("direct predecessor attempt analysis is invalid")
            assert previous_analysis is not None
            expected_previous_comparison = {
                "predecessor_result_identity": predecessor["result_identity"],
                "predecessor_source_commit": predecessor["source_commit"],
                "analysis": _metric_report_dict(previous_analysis),
                "raw_trial_identities": [
                    _raw_trial_identity(trial) for trial in all_previous_trials
                ],
                "attempts": previous_attempt_records,
            }
            if previous_comparison != expected_previous_comparison:
                raise ValueError("direct predecessor analysis does not match raw pairs")
        result_body = {
            key: value
            for key, value in metric_record.items()
            if key != "result_identity"
        }
        result_identity = structured_identity(
            "sml-performance-metric-result-v1", result_body
        )
        if metric_record.get("result_identity") != result_identity:
            raise ValueError("comparison metric result identity does not match")
        expected_lineage = {
            "metric": metric,
            "source_commit": candidate_commit,
            "result_identity": result_identity,
            "canonical_workload_identity": baseline["canonical_workload_identity"],
            "harness_identity": baseline["harness"]["content_identity"],
        }
        if report["latest_metrics"].get(metric) != expected_lineage:
            raise ValueError("comparison latest-metric lineage is invalid")
        expected_latest[metric] = expected_lineage
    first_attempt_noisy = comparison_has_noise(report, attempt_index=0)
    attempt_counts = {len(record["attempts"]) for record in metrics.values()}
    if first_attempt_noisy:
        if attempt_counts != {2}:
            raise ValueError("a noisy comparison must repeat every metric exactly once")
        validate_cooldown_evidence(
            report["cooldown_evidence"], workload.required_environment
        )
    elif attempt_counts != {1} or report["cooldown_evidence"] is not None:
        raise ValueError("a nominal comparison cannot contain a retry")
    referenced_trial_identities = []
    for metric_record in metrics.values():
        for attempt in metric_record["attempts"]:
            referenced_trial_identities.extend(attempt["raw_trial_identities"])
        previous = metric_record["previous_comparison"]
        if previous is not None:
            for attempt in previous["attempts"]:
                referenced_trial_identities.extend(attempt["raw_trial_identities"])
    embedded_trial_identities = [_raw_trial_identity(trial) for trial in trials]
    if sorted(embedded_trial_identities) != sorted(referenced_trial_identities):
        raise ValueError("comparison contains unreferenced or missing raw trials")
    if report["latest_metrics"] != expected_latest:
        raise ValueError("comparison did not preserve latest per-metric lineage")


def _run_command(arguments: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_root(path: Path) -> Path:
    return Path(_run_command(("git", "rev-parse", "--show-toplevel"), cwd=path))


def _git_commit(path: Path, revision: str = "HEAD") -> str:
    commit = _run_command(("git", "rev-parse", f"{revision}^{{commit}}"), cwd=path)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"Git did not resolve a full commit: {commit!r}")
    return commit


def _require_clean_checkout(path: Path, *, label: str) -> None:
    status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=path
    )
    if status:
        raise RuntimeError(f"{label} checkout must be clean before measurement")


def validate_checkout_status(
    status: str, *, allowed_untracked_paths: frozenset[str]
) -> None:
    for line in status.splitlines():
        if not line:
            continue
        if not line.startswith("?? ") or line[3:] not in allowed_untracked_paths:
            raise ValueError("checkout must be clean before measurement")


def _system_profiler(data_type: str) -> dict:
    output = subprocess.run(
        ("system_profiler", data_type, "-json"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed = json.loads(output)
    records = parsed.get(data_type)
    if not isinstance(records, list) or not records or not isinstance(records[0], dict):
        raise RuntimeError(f"system_profiler returned no {data_type} record")
    return records[0]


THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def decode_thermal_state(raw_value: int) -> str:
    if type(raw_value) is not int or raw_value not in THERMAL_STATES:
        raise RuntimeError(f"unsupported ProcessInfo thermal state: {raw_value!r}")
    return THERMAL_STATES[raw_value]


def validate_thermal_observation(status: dict[str, object]) -> None:
    raw_value = status.get("thermal_state_raw_value")
    state = status.get("thermal_state")
    if type(raw_value) is not int or state != decode_thermal_state(raw_value):
        raise ValueError("thermal state and raw value disagree")
    for nested_name in ("start", "end", "post_exit"):
        nested = status.get(nested_name)
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError(f"thermal {nested_name} observation must be an object")
            validate_thermal_observation(nested)
    if isinstance(status.get("start"), dict) and isinstance(status.get("end"), dict):
        expected_raw = max(
            status[name]["thermal_state_raw_value"]
            for name in ("start", "end", "post_exit")
            if isinstance(status.get(name), dict)
        )
        if raw_value != expected_raw:
            raise ValueError("merged thermal state is not the worse observation")


def _thermal_state() -> tuple[str, int]:
    environment = os.environ.copy()
    cache_root = Path(tempfile.gettempdir()) / "sml-v2-swift-module-cache"
    environment["CLANG_MODULE_CACHE_PATH"] = str(cache_root / "clang")
    environment["SWIFT_MODULECACHE_PATH"] = str(cache_root / "swift")
    raw_text = subprocess.run(
        (
            "swift",
            "-e",
            "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()
    try:
        raw_value = int(raw_text)
    except ValueError as error:
        raise RuntimeError(
            f"unsupported ProcessInfo thermal state: {raw_text!r}"
        ) from error
    return decode_thermal_state(raw_value), raw_value


def decode_memory_pressure_level(level: int) -> str:
    levels = {1: "normal", 2: "warning", 4: "critical"}
    if level not in levels:
        raise RuntimeError(f"unsupported macOS VM pressure level: {level}")
    return levels[level]


def _memory_pressure() -> tuple[str, int]:
    output = subprocess.run(
        ("memory_pressure", "-Q"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if match is None:
        raise RuntimeError("memory_pressure did not report free memory percentage")
    free_percentage = int(match.group(1))
    raw_level = subprocess.run(
        ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return decode_memory_pressure_level(int(raw_level)), free_percentage


def parse_power_status(
    power_source: str, custom_settings: str
) -> tuple[bool, str, bool]:
    source_match = re.search(r"Now drawing from '([^']+)'", power_source)
    if source_match is None or source_match.group(1) not in (
        "AC Power",
        "Battery Power",
    ):
        raise RuntimeError("pmset did not report the active power source")
    active_source = source_match.group(1)
    sections = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"(?ms)^(AC Power|Battery Power):\s*\n(.*?)(?=^(?:AC Power|Battery Power):|\Z)",
            custom_settings,
        )
    }
    section = sections.get(active_source)
    if section is None:
        raise RuntimeError(f"pmset did not report settings for {active_source}")
    match = re.search(r"^\s*lowpowermode\s+(\d+)\s*$", section, re.MULTILINE)
    if match is None:
        raise RuntimeError("pmset did not report low-power mode")
    low_power = match.group(1) != "0"
    connected = active_source == "AC Power"
    return connected, "low-power" if low_power else "automatic", low_power


def _power_status() -> tuple[bool, str, bool]:
    source = subprocess.run(
        ("pmset", "-g", "ps"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    custom = subprocess.run(
        ("pmset", "-g", "custom"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_power_status(source, custom)


def detect_competing_gpu_workload(
    process_table: str,
    *,
    current_pid: int,
    parent_pid: int,
) -> bool:
    processes = {}
    for line in process_table.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            parent = int(parts[1])
        except ValueError:
            continue
        processes[pid] = (parent, parts[2])
    ignored = {current_pid, parent_pid}
    ancestor = parent_pid
    while ancestor in processes:
        ancestor = processes[ancestor][0]
        if ancestor <= 0 or ancestor in ignored:
            break
        ignored.add(ancestor)
    gpu_signatures = (
        "mlx_lm",
        "train_sml.py",
        "ft_swag.py",
        "v2.benchmarks.runner",
        "ollama",
        "llama-server",
        "llama-cli",
        "torchrun",
    )
    for pid, (_parent, command) in processes.items():
        if pid in ignored:
            continue
        lowered = command.lower()
        if "mtlcompilerservice" in lowered:
            continue
        if any(signature in lowered for signature in gpu_signatures):
            return True
    return False


def _has_competing_gpu_workload() -> bool:
    result = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,command="),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return detect_competing_gpu_workload(
        result,
        current_pid=os.getpid(),
        parent_pid=os.getppid(),
    )


def collect_environment(
    *, memory_sample: tuple[str, int] | None = None
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    if memory_sample is None:
        memory_sample = _memory_pressure()
    pressure, free_percentage = memory_sample
    hardware_record = _system_profiler("SPHardwareDataType")
    display_record = _system_profiler("SPDisplaysDataType")
    processor_match = re.search(
        r"proc\s+(\d+):", str(hardware_record.get("number_processors", ""))
    )
    if processor_match is None:
        raise RuntimeError("system_profiler did not report CPU core count")
    memory_match = re.fullmatch(
        r"(\d+) GB", str(hardware_record.get("physical_memory", ""))
    )
    if memory_match is None:
        raise RuntimeError("system_profiler did not report unified memory")
    gpu_cores = int(display_record["sppci_cores"])
    macos_build = subprocess.run(
        ("sw_vers", "-buildVersion"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hardware = {
        "chip": hardware_record["chip_type"],
        "cpu_cores": int(processor_match.group(1)),
        "gpu_cores": gpu_cores,
        "unified_memory_bytes": int(memory_match.group(1)) * 1024**3,
        "machine_model": hardware_record["machine_model"],
        "macos_build": macos_build,
    }
    connected, power_mode, low_power = _power_status()
    thermal_state, thermal_state_raw_value = _thermal_state()
    environment_status = {
        "power_connected": connected,
        "power_mode": power_mode,
        "low_power_mode": low_power,
        "thermal_state": thermal_state,
        "thermal_state_raw_value": thermal_state_raw_value,
        "memory_pressure": pressure,
        "memory_free_percentage": free_percentage,
        "competing_gpu_workload": _has_competing_gpu_workload(),
    }
    packages = ("mlx", "numpy", "sentencepiece", "datasets", "lm_eval")
    versions = {"python": platform.python_version(), "macos": platform.mac_ver()[0]}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return hardware, environment_status, versions


def collect_post_exit_environment(
    *,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> tuple[float, dict[str, object]] | None:
    started_at = clock()
    if deadline is not None and started_at > deadline:
        return None
    memory_sample = _memory_pressure()
    observed_at_utc = _utc_now_iso()
    hardware, environment_status, software_versions = collect_environment(
        memory_sample=memory_sample
    )
    return started_at, {
        "observed_at_utc": observed_at_utc,
        "hardware": hardware,
        "environment_status": environment_status,
        "software_versions": software_versions,
    }


def _recovery_failure_fields(
    observation: dict,
    *,
    expected_hardware: dict,
    expected_software_versions: dict[str, str],
    required_environment: dict,
) -> tuple[str, ...]:
    status = observation["environment_status"]
    validate_thermal_observation(status)
    failure_fields = set()
    if observation["hardware"] != expected_hardware:
        failure_fields.add("hardware")
    if observation["software_versions"] != expected_software_versions:
        failure_fields.add("software_versions")
    for field in (
        "power_connected",
        "power_mode",
        "low_power_mode",
        "thermal_state",
        "competing_gpu_workload",
    ):
        if status.get(field) != required_environment[field]:
            failure_fields.add(field)
    return tuple(sorted(failure_fields))


def _reconstruct_missing_recovery(state, workload):
    observation = {
        name: state.post_exit[name]
        for name in (
            "observed_at_utc",
            "hardware",
            "environment_status",
            "software_versions",
        )
    }
    failures = _recovery_failure_fields(
        observation,
        expected_hardware=state.measurement["start"]["hardware"],
        expected_software_versions=state.measurement["start"]["software_versions"],
        required_environment=workload.required_environment,
    )
    pressure = observation["environment_status"]["memory_pressure"]
    if failures:
        outcome = "environment-failure"
    elif pressure == "critical":
        outcome = "critical"
    elif pressure == "normal":
        outcome = "not-required"
    else:
        outcome = "interrupted"
    return build_post_exit_recovery(
        measurement=state.measurement,
        post_exit=state.post_exit,
        samples=state.recovery_samples,
        policy=post_exit_recovery_policy(workload),
        outcome=outcome,
        duration_seconds=(
            0.0
            if not state.recovery_samples
            else state.recovery_samples[-1]["elapsed_seconds"]
        ),
        failure_fields=failures,
        completion_source="crash-reconstruction",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_jsonl(path: Path, trials: Sequence[RawTrial]) -> None:
    _validate_raw_trials_evidence(trials)
    lines = "".join(
        json.dumps(trial.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for trial in trials
    )
    _atomic_write_text(path, lines)


def _read_trials(path: Path) -> tuple[RawTrial, ...]:
    trials = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"raw trial line {line_number} must be an object")
        trial = RawTrial.from_dict(raw)
        validate_raw_trial_evidence(trial)
        trials.append(trial)
    if not trials:
        raise ValueError("raw trial file is empty")
    return tuple(trials)


def _run_single_process(args: argparse.Namespace) -> int:
    harness_root = _git_root(args.harness_root)
    source_root = _git_root(args.source_root)
    _require_clean_checkout(harness_root, label="harness")
    _require_clean_checkout(source_root, label="source")
    if _git_commit(harness_root) != args.harness_commit:
        raise RuntimeError("harness checkout commit changed before measurement")
    if _git_commit(source_root) != args.source_commit:
        raise RuntimeError("source checkout commit changed before measurement")
    if harness_content_identity(harness_root) != args.harness_identity:
        raise RuntimeError("harness content identity changed before measurement")
    workload = build_canonical_workload()
    workload_identity = canonical_workload_identity(workload)
    if args.adapter == "legacy":
        from v2.benchmarks.adapters import legacy as adapter
    else:
        from v2.benchmarks.adapters import replacement as adapter
    native = adapter.resolve_native_workload(args.metric, workload, source_root)
    if type(native).__name__ == "UnavailableNativeWorkload":
        raise RuntimeError(f"replacement metric unavailable: {native.reason}")
    start_hardware, start_status, start_software_versions = collect_environment()
    start_observation = {
        "observed_at_utc": _utc_now_iso(),
        "hardware": start_hardware,
        "environment_status": start_status,
        "software_versions": start_software_versions,
    }
    import mlx.core as mx

    work_unit = next(unit for unit in workload.work_units if unit.metric == args.metric)
    measured_units = work_unit.measured_units
    process_measurement = measure_native_process(
        adapter=adapter,
        metric=args.metric,
        native_workload=native,
        warmup_units=0 if args.metric == "compile-cold-start" else args.warmup,
        measured_units=measured_units,
        synchronize=mx.synchronize,
        peak_memory=mx.get_peak_memory,
        reset_peak_memory=mx.reset_peak_memory,
    )
    end_hardware, end_status, end_software_versions = collect_environment()
    end_observation = {
        "observed_at_utc": _utc_now_iso(),
        "hardware": end_hardware,
        "environment_status": end_status,
        "software_versions": end_software_versions,
    }
    trial_payload = {
        "metric": args.metric,
        "side": args.side,
        "attempt_index": args.attempt_index,
        "pair_index": args.pair_index,
        "process_order": args.process_order,
        "source_commit": args.source_commit,
        "source_clean": True,
        "harness_commit": args.harness_commit,
        "harness_clean": True,
        "harness_identity": args.harness_identity,
        "canonical_workload_identity": workload_identity,
        "native_configuration": native.native_configuration,
        "native_representation_identity": native.native_representation_identity,
        "canonical_row_identity": native.canonical_row_identity,
        "canonical_input_identity": native.canonical_input_identity,
        "canonical_projection": native.canonical_projection,
        "execution_order_identity": native.execution_order_identity,
        "initial_parameter_identity": native.initial_parameter_identity,
        "comparison_target": args.comparison_target,
        "warmup_units": (0 if args.metric == "compile-cold-start" else args.warmup),
        "measured_units": measured_units,
        "elapsed_seconds": process_measurement.elapsed_seconds,
        "value": process_measurement.value,
        "startup_verification_seconds": native.startup_verification_seconds,
        "compilation_seconds": (
            process_measurement.elapsed_seconds
            if args.metric == "compile-cold-start"
            else process_measurement.compilation_seconds
        ),
        "peak_memory_bytes": process_measurement.peak_memory_bytes,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }
    document = build_child_trial_measurement(
        session_identity=args.evidence_session_identity,
        journal_attempt_index=args.journal_attempt_index,
        trial=trial_payload,
        start=start_observation,
        end=end_observation,
    )
    atomic_write_json(args.measurement_output, document, create_only=True)
    return 0


def _create_detached_worktree(repository: Path, commit: str, destination: Path) -> None:
    subprocess.run(
        ("git", "worktree", "add", "--detach", str(destination), commit),
        cwd=repository,
        check=True,
    )
    try:
        _require_clean_checkout(destination, label="source")
        if _git_commit(destination) != commit:
            raise RuntimeError("detached source worktree resolved the wrong commit")
    except BaseException as error:
        _cleanup_detached_worktree(repository, destination, primary_error=error)
        raise


def _remove_worktree(repository: Path, destination: Path) -> None:
    subprocess.run(
        ("git", "worktree", "remove", "--force", str(destination)),
        cwd=repository,
        check=True,
    )


def _fallback_cleanup_failed_worktree(
    repository: Path, destination: Path
) -> tuple[BaseException, ...]:
    failures: list[BaseException] = []
    try:
        if not repository.is_absolute() or not destination.is_absolute():
            raise RuntimeError("worktree cleanup paths must be absolute")
        if destination.is_symlink():
            raise RuntimeError("refusing to recursively remove a symlinked worktree")
        repository_root = repository.resolve()
        destination_root = destination.resolve()
        if (
            destination_root == Path(destination_root.anchor)
            or destination_root == repository_root
            or repository_root.is_relative_to(destination_root)
        ):
            raise RuntimeError("refusing to remove an unsafe worktree path")
        if destination.exists():
            if not destination.is_dir():
                raise RuntimeError("failed worktree path is not a directory")
            shutil.rmtree(destination)
    except BaseException as error:  # noqa: BLE001 - collect every cleanup failure
        failures.append(error)

    try:
        subprocess.run(
            ("git", "worktree", "prune"),
            cwd=repository,
            check=True,
        )
    except BaseException as error:  # noqa: BLE001 - collect every cleanup failure
        failures.append(error)
    return tuple(failures)


def _cleanup_detached_worktree(
    repository: Path,
    destination: Path,
    *,
    primary_error: BaseException | None = None,
) -> None:
    try:
        _remove_worktree(repository, destination)
    except BaseException as cleanup_error:
        fallback_failures = _fallback_cleanup_failed_worktree(repository, destination)
        if primary_error is not None:
            for failure in (cleanup_error, *fallback_failures):
                primary_error.add_note(
                    "detached worktree cleanup failed: "
                    f"{type(failure).__name__}: {failure}"
                )
            return
        for failure in fallback_failures:
            cleanup_error.add_note(
                "detached worktree fallback cleanup failed: "
                f"{type(failure).__name__}: {failure}"
            )
        raise


@contextmanager
def _managed_detached_worktree(repository: Path, commit: str, destination: Path):
    _create_detached_worktree(repository, commit, destination)
    try:
        yield destination
    except BaseException as error:
        _cleanup_detached_worktree(repository, destination, primary_error=error)
        raise
    else:
        _cleanup_detached_worktree(repository, destination)


def _launch_trial(
    *,
    harness_root: Path,
    source_root: Path,
    source_commit: str,
    harness_commit: str,
    harness_identity: str,
    adapter: str,
    metric: MetricName,
    side: Side,
    attempt_index: int,
    pair_index: int,
    order: int,
    warmup: int,
    measure: int,
    comparison_target: str,
    evidence_session_identity: str,
    journal_attempt_index: int,
    measurement_output: Path,
    post_exit_output: Path,
    recovery_samples_directory: Path,
    recovery_output: Path,
    output: Path,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RawTrial:
    command = (
        sys.executable,
        "-m",
        "v2.benchmarks.runner",
        "_run-process",
        "--harness-root",
        str(harness_root),
        "--source-root",
        str(source_root),
        "--source-commit",
        source_commit,
        "--harness-commit",
        harness_commit,
        "--harness-identity",
        harness_identity,
        "--adapter",
        adapter,
        "--metric",
        metric,
        "--side",
        side,
        "--attempt-index",
        str(attempt_index),
        "--pair-index",
        str(pair_index),
        "--process-order",
        str(order),
        "--warmup",
        str(warmup),
        "--measure",
        str(measure),
        "--comparison-target",
        comparison_target,
        "--evidence-session-identity",
        evidence_session_identity,
        "--journal-attempt-index",
        str(journal_attempt_index),
        "--measurement-output",
        str(measurement_output),
    )
    subprocess.run(command, cwd=harness_root, check=True)
    measurement = validate_child_trial_measurement(
        read_json_object(measurement_output, label="fresh child measurement")
    )
    if measurement["session_identity"] != evidence_session_identity:
        raise ValueError("fresh child measurement has the wrong session identity")
    if measurement["journal_attempt_index"] != journal_attempt_index:
        raise ValueError("fresh child measurement has the wrong journal attempt index")
    immediate_started_at, observation = collect_post_exit_environment(clock=clock)
    post_exit = build_post_exit_observation(
        measurement=measurement,
        **observation,
    )
    atomic_write_json(post_exit_output, post_exit, create_only=True)
    workload = build_canonical_workload()
    policy = post_exit_recovery_policy(workload)

    def record_sample(
        index: int,
        elapsed: float,
        collected: dict,
        previous_identity: str | None,
    ) -> dict:
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=index,
            previous_sample_identity=previous_identity,
            elapsed_seconds=elapsed,
            **collected,
        )
        atomic_write_json(
            recovery_samples_directory / f"{index}.json",
            sample,
            create_only=True,
        )
        return sample

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=observation,
        immediate_started_at=immediate_started_at,
        recovery_policy=policy,
        collect=lambda deadline: collect_post_exit_environment(
            clock=clock, deadline=deadline
        ),
        classify_nonmemory=lambda item: _recovery_failure_fields(
            item,
            expected_hardware=measurement["start"]["hardware"],
            expected_software_versions=measurement["start"]["software_versions"],
            required_environment=workload.required_environment,
        ),
        record_sample=record_sample,
        clock=clock,
        sleep=sleep,
    )
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=result.samples,
        policy=policy,
        outcome=result.outcome,
        duration_seconds=result.duration_seconds,
        failure_fields=result.failure_fields,
        completion_source="live",
    )
    atomic_write_json(recovery_output, recovery, create_only=True)
    trial = finalize_raw_trial(measurement, post_exit, result.samples, recovery)
    atomic_write_json(output, trial.to_dict(), create_only=True)
    return RawTrial.from_dict(read_json_object(output, label="fresh raw trial"))


def _next_capture_indices(
    journal: BaselineJournal, slot: BaselineSlot
) -> tuple[int, int]:
    preflight_history = journal._validate_preflight_history()
    recovery_history = journal._validate_thermal_recovery_history()
    return len(preflight_history.get(slot, ())), len(recovery_history.get(slot, ()))


def _thermal_trigger_payload(document: dict) -> tuple[tuple[str, str], dict]:
    if document["source"] == "preflight":
        preflight = document["preflight"]
        return (
            ("preflight", preflight["identity"]),
            {"source": "preflight", "preflight": preflight},
        )
    identity = document["rejected_trial_identity"]
    return (
        ("rejected-trial", identity),
        {"source": "rejected-trial", "rejected_trial_identity": identity},
    )


def _persisted_pending_thermal_triggers(
    journal: BaselineJournal,
    slots: Sequence[BaselineSlot],
    clock: Callable[[], float],
    validate_observation: Callable[[dict, dict, dict], None],
) -> dict[BaselineSlot, tuple[dict, float]]:
    expected = set(slots)
    recovery_deadlines: dict[BaselineSlot, float] = {}
    preflight_order: dict[BaselineSlot, list[tuple[tuple[str, str], dict]]] = {}
    for slot, _index, _path, document in journal._preflight_records():
        if slot not in expected:
            raise ValueError("unexpected preflight slot")
        validate_observation(
            document["hardware"],
            document["environment_status"],
            document["software_versions"],
        )
        validate_thermal_observation(document["environment_status"])
        if document["environment_status"]["thermal_state"] == "nominal":
            continue
        _start_recovery_deadline(recovery_deadlines, slot, clock)
        key = ("preflight", document["identity"])
        trigger = {"source": "preflight", "preflight": document}
        preflight_order.setdefault(slot, []).append((key, trigger))

    rejected_order: dict[BaselineSlot, list[tuple[tuple[str, str], dict]]] = {}
    for attempt, path in journal._attempt_records("rejected"):
        if attempt.slot not in expected:
            raise ValueError("unexpected rejected slot")
        document = read_json_object(path, label="rejected trial")
        if document["reason"] != "non-nominal-thermal":
            continue
        trial = RawTrial.from_dict(document["trial"])
        validate_thermal_observation(trial.environment_status)
        if trial.environment_status["thermal_state"] == "nominal":
            raise ValueError("thermal rejection contains a nominal trial")
        _start_recovery_deadline(recovery_deadlines, attempt.slot, clock)
        key = ("rejected-trial", document["identity"])
        trigger = {
            "source": "rejected-trial",
            "rejected_trial_identity": document["identity"],
        }
        rejected_order.setdefault(attempt.slot, []).append((key, trigger))

    recovery_by_slot: dict[
        BaselineSlot, list[tuple[int, Path, tuple[str, str], dict]]
    ] = {}
    used_sources: dict[BaselineSlot, set[tuple[str, str]]] = {}
    for slot, recovery_index, path in journal._thermal_recovery_records():
        if slot not in expected:
            raise ValueError("unexpected thermal recovery slot")
        for sample_path in journal._recovery_sample_paths(path):
            sample = read_json_object(sample_path, label="thermal sample")
            validate_observation(
                sample["hardware"],
                sample["environment_status"],
                sample["software_versions"],
            )
        trigger_document = read_json_object(
            path / "trigger.json", label="thermal recovery trigger"
        )
        key, trigger = _thermal_trigger_payload(trigger_document)
        _start_recovery_deadline(recovery_deadlines, slot, clock)
        recovery_by_slot.setdefault(slot, []).append(
            (recovery_index, path, key, trigger)
        )
        used_sources.setdefault(slot, set()).add(key)

    pending: dict[BaselineSlot, tuple[dict, float]] = {}
    for slot in slots:
        recoveries = sorted(recovery_by_slot.get(slot, ()))
        last_success = -1
        for recovery_index, path, _key, _trigger in recoveries:
            summary_path = path / "summary.json"
            if summary_path.exists():
                summary = read_json_object(
                    summary_path, label="thermal recovery summary"
                )
                if summary["outcome"] == "nominal-window":
                    last_success = recovery_index
        trailing = [record for record in recoveries if record[0] > last_success]
        if trailing:
            pending[slot] = (trailing[-1][3], recovery_deadlines[slot])
            continue

        used = used_sources.get(slot, set())
        unresolved_rejected = [
            trigger for key, trigger in rejected_order.get(slot, ()) if key not in used
        ]
        if unresolved_rejected:
            pending[slot] = (unresolved_rejected[-1], recovery_deadlines[slot])
            continue
        unresolved_preflight = [
            trigger for key, trigger in preflight_order.get(slot, ()) if key not in used
        ]
        if unresolved_preflight:
            pending[slot] = (unresolved_preflight[-1], recovery_deadlines[slot])
    return pending


def _start_recovery_deadline(
    deadlines: dict[BaselineSlot, float],
    slot: BaselineSlot,
    clock: Callable[[], float],
) -> None:
    if slot not in deadlines:
        deadlines[slot] = clock() + 7_200.0


def capture_baseline_trials(
    *,
    journal: BaselineJournal,
    slots: Sequence[BaselineSlot],
    launch_trial: Callable[[BaselineSlot, JournalAttempt], RawTrial],
    preflight: Callable[[], tuple[dict, dict, dict]],
    validate_preflight: Callable[[dict, dict, dict], None],
    recover: Callable[[BaselineSlot, int, float, dict], None],
    validate_trial: Callable[..., None],
    classify_trial: Callable[[RawTrial], TrialEnvironmentDisposition],
    clock: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = _utc_now_iso,
    progress: Callable[[str], None] = print,
) -> tuple[RawTrial, ...]:
    ordered_slots = tuple(slots)
    accepted = journal.load_accepted(ordered_slots)
    persisted_recoveries = _persisted_pending_thermal_triggers(
        journal, ordered_slots, clock, validate_preflight
    )
    pending_triggers = {
        slot: trigger for slot, (trigger, _deadline) in persisted_recoveries.items()
    }
    recovery_deadlines = {
        slot: deadline
        for slot, (_trigger, deadline) in persisted_recoveries.items()
        if slot not in accepted
    }

    def transition_attempt(
        attempt: JournalAttempt, trial: RawTrial, *, resumed: bool
    ) -> None:
        disposition = classify_trial(trial)
        if disposition.outcome == "accept":
            journal.accept_inflight(attempt, trial)
            accepted[attempt.slot] = trial
            prefix = "accepted in-flight" if resumed else "accepted"
            progress(
                f"{prefix} {attempt.slot.metric} pair {attempt.slot.pair_index} "
                f"({len(accepted)}/{len(ordered_slots)})"
            )
            return
        if disposition.outcome == "thermal-reject":
            _start_recovery_deadline(recovery_deadlines, attempt.slot, clock)
            journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
            rejected = read_json_object(
                journal.rejected_path(attempt.slot, attempt.journal_attempt_index),
                label="rejected trial",
            )
            pending_triggers[attempt.slot] = {
                "source": "rejected-trial",
                "rejected_trial_identity": rejected["identity"],
            }
            prefix = "rejected in-flight" if resumed else "rejected"
            progress(
                f"{prefix} {attempt.slot.metric} pair {attempt.slot.pair_index}: "
                f"thermal={trial.environment_status['thermal_state']} "
                f"raw={trial.environment_status['thermal_state_raw_value']}"
            )
            return
        if disposition.outcome == "environment-reject":
            if disposition.reason is None:
                raise AssertionError("environment rejection is missing its reason")
            journal.reject_inflight(attempt, trial, reason=disposition.reason)
            raise EnvironmentTrialRejected(attempt.slot, disposition.reason)
        if disposition.outcome != "memory-reject":
            raise AssertionError("trial classification has an unknown outcome")
        if disposition.reason is None:
            raise AssertionError("memory rejection is missing its reason")
        journal.reject_inflight(attempt, trial, reason=disposition.reason)
        raise MemoryPressureTrialRejected(attempt.slot, disposition.reason)

    for slot in ordered_slots:
        if slot not in accepted:
            continue
        trial = accepted[slot]
        validate_trial(trial, allow_rejected_environment=False)
        progress(f"resumed accepted {slot.metric} pair {slot.pair_index}")

    pending_attempts = journal.load_pending_attempts(ordered_slots)
    workload = None
    for state in pending_attempts:
        if state.post_exit is None:
            journal.reject_unfinalized(
                state.attempt,
                state.measurement,
                reason="missing-immediate-post-exit-evidence",
            )
            continue
        recovery = state.recovery
        if recovery is None:
            if workload is None:
                workload = CanonicalWorkload.from_dict(
                    journal.session["canonical_workload"]
                )
            recovery = _reconstruct_missing_recovery(state, workload)
            atomic_write_json(
                journal.recovery_path(
                    state.attempt.slot, state.attempt.journal_attempt_index
                ),
                recovery,
                create_only=True,
            )
        if state.trial is None:
            trial = finalize_raw_trial(
                state.measurement,
                state.post_exit,
                state.recovery_samples,
                recovery,
            )
            atomic_write_json(state.attempt.path, trial.to_dict(), create_only=True)
        else:
            trial = state.trial
        validate_trial(trial, allow_rejected_environment=True)
        transition_attempt(state.attempt, trial, resumed=True)

    for slot in ordered_slots:
        if slot in accepted:
            continue
        preflight_index, recovery_index = _next_capture_indices(journal, slot)
        recovery_deadline = recovery_deadlines.get(slot)
        pending_trigger = pending_triggers.get(slot)

        while slot not in accepted:
            if pending_trigger is not None:
                if recovery_deadline is None:
                    raise AssertionError("thermal recovery is missing its deadline")
                progress(
                    f"thermal recovery {slot.metric} pair {slot.pair_index} "
                    f"episode {recovery_index}"
                )
                recover(slot, recovery_index, recovery_deadline, pending_trigger)
                progress(
                    f"thermal recovery complete {slot.metric} pair {slot.pair_index} "
                    f"episode {recovery_index}"
                )
                recovery_index += 1
                pending_trigger = None

            hardware, status, software_versions = preflight()
            if status.get("thermal_state") != "nominal" and recovery_deadline is None:
                recovery_deadline = clock() + 7_200.0
                recovery_deadlines[slot] = recovery_deadline
            preflight_document = journal.record_preflight(
                slot,
                preflight_index,
                {
                    "observed_at_utc": utc_now(),
                    "hardware": hardware,
                    "environment_status": status,
                    "software_versions": software_versions,
                },
            )
            progress(
                f"recorded preflight {slot.metric} pair {slot.pair_index} "
                f"check {preflight_index}"
            )
            preflight_index += 1
            validate_preflight(hardware, status, software_versions)
            if status["thermal_state"] != "nominal":
                pending_trigger = {
                    "source": "preflight",
                    "preflight": preflight_document,
                }
                continue

            attempt = journal.next_attempt(slot)
            progress(
                f"launching {slot.metric} pair {slot.pair_index} "
                f"journal attempt {attempt.journal_attempt_index}"
            )
            launched_trial = launch_trial(slot, attempt)
            persisted = RawTrial.from_dict(
                read_json_object(attempt.path, label="inflight trial")
            )
            if persisted != launched_trial:
                raise ValueError("launched trial does not match in-flight output")
            validate_trial(persisted, allow_rejected_environment=True)
            transition_attempt(attempt, persisted, resumed=False)
            recovery_deadline = recovery_deadlines.get(slot)
            pending_trigger = pending_triggers.get(slot)

    return tuple(accepted[slot] for slot in ordered_slots)


def _validate_baseline_preflight(
    *,
    workload: CanonicalWorkload,
    hardware: dict,
    status: dict,
    software_versions: dict,
    expected_hardware: dict,
    expected_software_versions: dict[str, str],
) -> None:
    if hardware != expected_hardware:
        raise ValueError("preflight hardware does not match the baseline session")
    if software_versions != expected_software_versions:
        raise ValueError(
            "preflight software versions do not match the baseline session"
        )
    validate_thermal_observation(status)
    required = workload.required_environment
    for key in ("chip", "cpu_cores", "gpu_cores", "unified_memory_bytes"):
        if hardware.get(key) != required[key]:
            raise ValueError(f"preflight hardware does not match required {key}")
    _validate_preflight_environment(
        status,
        required,
        label="preflight environment",
    )
    _validate_software_versions(workload, software_versions)


def _allowed_checkout_output_paths(
    harness_root: Path, paths: Sequence[Path]
) -> frozenset[str]:
    allowed = set()
    for path in paths:
        try:
            allowed.add(path.resolve().relative_to(harness_root).as_posix())
        except ValueError:
            continue
    return frozenset(allowed)


def _resolve_baseline_output_paths(
    *,
    state_root: Path,
    manifest_path: Path,
    raw_output_path: Path,
) -> tuple[Path, Path]:
    state = state_root.resolve()
    manifest = manifest_path.resolve()
    raw_output = raw_output_path.resolve()
    path_pairs = (
        (state, manifest),
        (state, raw_output),
        (manifest, raw_output),
    )
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for left, right in path_pairs
    ):
        raise ValueError(
            "final output paths must be distinct and outside the state directory"
        )
    return manifest, raw_output


def _publish_final_artifact(path: Path, text: str) -> None:
    expected = text.encode("utf-8")
    try:
        atomic_write_text(path, text, create_only=True)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ValueError(
                "final artifact already exists with different content"
            ) from error
        if existing != expected:
            raise ValueError("final artifact already exists with different content")


def canonical_baseline_command(journal: BaselineJournal) -> str:
    session = journal.session
    protocol = session["protocol"]
    return shlex.join(
        (
            "uv",
            "run",
            "python",
            "-m",
            "v2.benchmarks.runner",
            "record-baseline",
            "--source-commit",
            session["source"]["commit"],
            "--manifest",
            session["manifest_path"],
            "--raw-output",
            session["raw_output_path"],
            "--state-directory",
            "SESSION_STATE_DIRECTORY",
            "--metrics",
            ",".join(METRIC_NAMES),
            "--pairs",
            str(protocol["pairs"]),
            "--warmup",
            str(protocol["warmup_units"]),
            "--measure",
            str(protocol["measured_units"]),
        )
    )


def publish_baseline_from_journal(
    *,
    journal: BaselineJournal,
    trials: Sequence[RawTrial],
    workload: CanonicalWorkload,
    workload_identity: str,
    source_commit: str,
    harness_commit: str,
    harness_identity: str,
    paired_representations: dict,
    manifest_path: Path,
    raw_output_path: Path,
) -> dict:
    _validate_raw_trials_evidence(trials)
    manifest_path, raw_output_path = _resolve_baseline_output_paths(
        state_root=journal.root,
        manifest_path=manifest_path,
        raw_output_path=raw_output_path,
    )
    if journal.session["paired_representations"] != paired_representations:
        raise ValueError("publication paired representations do not match the session")
    if journal.session["manifest_path"] != str(manifest_path.resolve()):
        raise ValueError("publication manifest path does not match the session")
    if journal.session["raw_output_path"] != str(raw_output_path.resolve()):
        raise ValueError("publication raw output path does not match the session")

    slots = BaselineJournal.expected_slots(METRIC_NAMES, SCREEN_PAIRS)
    accepted = journal.load_accepted(slots)
    if len(accepted) != len(slots):
        raise ValueError("baseline journal does not contain every accepted slot")
    ordered_trials = tuple(accepted[slot] for slot in slots)
    if tuple(trials) != ordered_trials:
        raise ValueError("publication trials do not exactly match accepted slots")

    manifest = build_baseline_manifest(
        trials=ordered_trials,
        workload=workload,
        workload_identity=workload_identity,
        source_commit=source_commit,
        harness_commit=harness_commit,
        harness_identity=harness_identity,
        command=canonical_baseline_command(journal),
        pairs=SCREEN_PAIRS,
        warmup_units=WARMUP_UNITS,
        measured_units=DEFAULT_MEASURED_UNITS,
        paired_representations=paired_representations,
    )
    validate_baseline_manifest(manifest, ordered_trials)

    raw_text = "".join(
        json.dumps(trial.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for trial in ordered_trials
    )
    manifest_text = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    _publish_final_artifact(raw_output_path, raw_text)
    _publish_final_artifact(manifest_path, manifest_text)

    completion_body = {
        "kind": "sml-baseline-journal-completion",
        "version": 1,
        "session_identity": journal.session["identity"],
        "baseline_identity": manifest["identity"],
        "manifest_path": str(manifest_path.resolve()),
        "raw_output_path": str(raw_output_path.resolve()),
        "raw_trial_identities": [
            _raw_trial_identity(trial) for trial in ordered_trials
        ],
    }
    journal.publish_completed(
        {
            **completion_body,
            "identity": structured_identity(
                "sml-baseline-journal-completion-v1", completion_body
            ),
        }
    )
    return manifest


def _record_baseline(args: argparse.Namespace) -> int:
    harness_root = _git_root(Path.cwd())
    state_root = require_external_state_directory(args.state_directory, (harness_root,))
    manifest_path, raw_output_path = _resolve_baseline_output_paths(
        state_root=state_root,
        manifest_path=args.manifest,
        raw_output_path=args.raw_output,
    )
    with (
        baseline_output_lock(manifest_path, raw_output_path),
        baseline_session_lock(state_root),
    ):
        cleanup_orphaned_journal_temporaries(state_root)
        cleanup_orphaned_atomic_temporaries((manifest_path, raw_output_path))
        return _record_baseline_locked(
            args,
            harness_root=harness_root,
            state_root=state_root,
            manifest_path=manifest_path,
            raw_output_path=raw_output_path,
        )


def _record_baseline_locked(
    args: argparse.Namespace,
    *,
    harness_root: Path,
    state_root: Path,
    manifest_path: Path,
    raw_output_path: Path,
) -> int:
    allowed_outputs = frozenset()
    session_path = state_root / "session.json"
    if session_path.is_file():
        existing_session = read_json_object(
            session_path, label="baseline journal session"
        )
        BaselineJournal.open(state_root, existing_session)
        if existing_session["manifest_path"] == str(manifest_path) and existing_session[
            "raw_output_path"
        ] == str(raw_output_path):
            allowed_outputs = _allowed_checkout_output_paths(
                harness_root, (manifest_path, raw_output_path)
            )
    harness_status = _run_command(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=harness_root,
    )
    validate_checkout_status(harness_status, allowed_untracked_paths=allowed_outputs)
    harness_commit = _git_commit(harness_root)
    harness_identity = harness_content_identity(harness_root)
    source_commit = _git_commit(harness_root, args.source_commit)
    if source_commit != PINNED_BASELINE_SOURCE_COMMIT:
        raise ValueError("record-baseline requires the pinned 3687f8b source commit")
    if (
        tuple(args.metrics) != METRIC_NAMES
        or args.pairs != SCREEN_PAIRS
        or args.warmup != WARMUP_UNITS
        or args.measure != DEFAULT_MEASURED_UNITS
    ):
        raise ValueError("record-baseline protocol is immutable")
    workload = build_canonical_workload()
    workload_identity = canonical_workload_identity(workload)
    initial_environment = collect_environment()
    initial_hardware, _initial_status, initial_software_versions = initial_environment
    protocol = {
        "pairs": args.pairs,
        "compilation_passes": 1,
        "warmup_units": args.warmup,
        "measured_units": args.measure,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }
    with tempfile.TemporaryDirectory(prefix="sml-v2-baseline-") as temporary_name:
        temporary = Path(temporary_name)
        paired_representations = write_paired_pretraining_representations(
            fixed_canonical_rows(), temporary / "paired-representations"
        )
        session = build_session_document(
            harness_commit=harness_commit,
            harness_identity=harness_identity,
            source_commit=source_commit,
            canonical_workload=workload,
            canonical_workload_identity=workload_identity,
            protocol=protocol,
            hardware=initial_hardware,
            software_versions=initial_software_versions,
            paired_representations=paired_representations,
            manifest_path=manifest_path,
            raw_output_path=raw_output_path,
        )
        journal = BaselineJournal.open(state_root, session)
        slots = BaselineJournal.expected_slots(args.metrics, args.pairs)
        source_root = temporary / "source"
        with _managed_detached_worktree(harness_root, source_commit, source_root):
            require_external_state_directory(state_root, (harness_root, source_root))
            cached_preflight = [initial_environment]

            def collect_preflight():
                if cached_preflight:
                    return cached_preflight.pop()
                return collect_environment()

            def recover_thermal(
                slot: BaselineSlot,
                recovery_index: int,
                deadline: float,
                trigger: dict,
            ) -> None:
                journal.record_recovery_trigger(slot, recovery_index, trigger)

                def record_sample(sample_index: int, sample: dict) -> None:
                    journal.record_thermal_sample(
                        slot, recovery_index, sample_index, sample
                    )
                    print(
                        f"thermal sample {slot.metric} pair {slot.pair_index} "
                        f"episode {recovery_index}: elapsed="
                        f"{sample['elapsed_seconds']:.1f}s"
                    )

                try:
                    result = wait_for_nominal_thermal_window(
                        collect=collect_environment,
                        expected_hardware=initial_hardware,
                        expected_software_versions=initial_software_versions,
                        required_environment=workload.required_environment,
                        record_sample=record_sample,
                        deadline=deadline,
                        clock=time.monotonic,
                        sleep=time.sleep,
                        utc_now=_utc_now_iso,
                    )
                except ThermalRecoveryTimeout as error:
                    journal.record_recovery_summary(
                        slot,
                        recovery_index,
                        {
                            "outcome": "timeout",
                            "duration_seconds": error.result.duration_seconds,
                            "sample_count": error.result.sample_count,
                        },
                    )
                    raise
                journal.record_recovery_summary(
                    slot,
                    recovery_index,
                    {
                        "outcome": "nominal-window",
                        "duration_seconds": result.duration_seconds,
                        "sample_count": result.sample_count,
                    },
                )

            trials = capture_baseline_trials(
                journal=journal,
                slots=slots,
                launch_trial=lambda slot, attempt: _launch_trial(
                    harness_root=harness_root,
                    source_root=source_root,
                    source_commit=source_commit,
                    harness_commit=harness_commit,
                    harness_identity=harness_identity,
                    adapter="legacy",
                    metric=slot.metric,
                    side="reference",
                    attempt_index=0,
                    pair_index=slot.pair_index,
                    order=0,
                    warmup=args.warmup,
                    measure=args.measure,
                    comparison_target="baseline",
                    evidence_session_identity=journal.session["identity"],
                    journal_attempt_index=attempt.journal_attempt_index,
                    measurement_output=journal.measurement_path(
                        slot, attempt.journal_attempt_index
                    ),
                    post_exit_output=journal.post_exit_path(
                        slot, attempt.journal_attempt_index
                    ),
                    recovery_samples_directory=journal.recovery_samples_path(
                        slot, attempt.journal_attempt_index
                    ),
                    recovery_output=journal.recovery_path(
                        slot, attempt.journal_attempt_index
                    ),
                    output=attempt.path,
                ),
                preflight=collect_preflight,
                validate_preflight=lambda hardware, status, software: (
                    _validate_baseline_preflight(
                        workload=workload,
                        hardware=hardware,
                        status=status,
                        software_versions=software,
                        expected_hardware=initial_hardware,
                        expected_software_versions=initial_software_versions,
                    )
                ),
                recover=recover_thermal,
                validate_trial=lambda trial, allow_rejected_environment: (
                    validate_baseline_trial(
                        trial,
                        workload=workload,
                        source_commit=source_commit,
                        harness_commit=harness_commit,
                        harness_identity=harness_identity,
                        expected_hardware=initial_hardware,
                        expected_software_versions=initial_software_versions,
                        allow_rejected_environment=allow_rejected_environment,
                    )
                ),
                classify_trial=lambda trial: classify_trial_environment(
                    workload, trial
                ),
            )
    publish_baseline_from_journal(
        journal=journal,
        trials=trials,
        workload=workload,
        workload_identity=workload_identity,
        source_commit=source_commit,
        harness_commit=harness_commit,
        harness_identity=harness_identity,
        paired_representations=paired_representations,
        manifest_path=manifest_path,
        raw_output_path=raw_output_path,
    )
    return 0


def _validate_baseline_files(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("baseline manifest must be an object")
    trials = _read_trials(args.raw_input)
    validate_baseline_manifest(manifest, trials)
    return 0


def _read_json_object(path: Path, *, label: str) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def _validate_baseline_document(manifest: dict) -> None:
    body = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest.get("identity") != structured_identity(
        "sml-performance-baseline-v1", body
    ):
        raise ValueError("baseline manifest identity does not match content")
    if (
        manifest.get("kind") != "sml-performance-baseline"
        or manifest.get("version") != 1
    ):
        raise ValueError("unsupported baseline manifest kind or version")
    workload_raw = manifest.get("canonical_workload")
    if not isinstance(workload_raw, dict):
        raise ValueError("baseline canonical workload must be an object")
    workload = CanonicalWorkload.from_dict(workload_raw)
    if workload != build_canonical_workload():
        raise ValueError("baseline canonical workload is not the pinned workload")
    if manifest.get("canonical_workload_identity") != canonical_workload_identity(
        workload
    ):
        raise ValueError("baseline canonical workload identity is invalid")
    source = manifest.get("source")
    harness = manifest.get("harness")
    if not isinstance(source, dict) or source.get("clean") is not True:
        raise ValueError("baseline source proof is invalid")
    if not isinstance(harness, dict) or harness.get("clean") is not True:
        raise ValueError("baseline harness proof is invalid")
    if source.get("commit") != PINNED_BASELINE_SOURCE_COMMIT:
        raise ValueError("baseline source is not the pinned 3687f8b commit")
    protocol = manifest.get("protocol")
    if protocol != {
        "pairs": SCREEN_PAIRS,
        "compilation_passes": 1,
        "warmup_units": WARMUP_UNITS,
        "measured_units": DEFAULT_MEASURED_UNITS,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }:
        raise ValueError("baseline document has the wrong protocol")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ValueError("baseline document does not contain every metric")


def _validate_comparison_document(report: dict, baseline: dict) -> None:
    validate_comparison_report(
        report,
        baseline,
        predecessor_reports=None,
        _use_embedded_predecessors=True,
    )
    validate_throughput_gates(report, label="predecessor")


def _validate_report_raw_trial_evidence(report: dict) -> None:
    raw_trials = report.get("raw_trials")
    if not isinstance(raw_trials, list):
        raise ValueError("comparison raw_trials must be a list")
    for raw in raw_trials:
        if not isinstance(raw, dict):
            raise ValueError("comparison raw trials must be objects")
        validate_raw_trial_evidence(RawTrial.from_dict(raw))


def _resolve_predecessor_mapping(
    value: str,
    repository: Path,
    baseline: dict,
    metrics: Sequence[MetricName],
    *,
    additional_search_directories: Sequence[Path] = (),
) -> tuple[dict[str, dict | None], dict[str, dict], dict[str, dict | None]]:
    mapping_path = Path(value)
    raw_mapping = (
        json.loads(mapping_path.read_text(encoding="utf-8"))
        if mapping_path.is_file()
        else json.loads(value)
    )
    if not isinstance(raw_mapping, dict) or set(raw_mapping) != set(metrics):
        raise ValueError("predecessors must map every measured metric exactly once")
    reports: dict[str, dict | None] = {}
    lineages = {}
    proof = {}
    results_directory = repository / "v2" / "benchmarks" / "results"
    search_directories = tuple(
        dict.fromkeys((results_directory, *additional_search_directories))
    )
    for metric in metrics:
        requested = raw_mapping[metric]
        if requested is None:
            reports[metric] = None
            proof[metric] = None
            continue
        if not isinstance(requested, str) or not requested:
            raise ValueError(
                "predecessor values must be a report path, identity, or null"
            )
        requested_path = Path(requested)
        candidates = [
            requested_path,
            results_directory / requested,
            results_directory / f"{requested}.json",
        ]
        if re.fullmatch(r"sha256:[0-9a-f]{64}", requested):
            for directory in search_directories:
                candidates.extend(sorted(directory.glob("*.json")))
        report = None
        for candidate in candidates:
            if not candidate.is_file():
                continue
            candidate_report = _read_json_object(
                candidate, label=f"{metric} predecessor report"
            )
            if (
                requested.startswith("sha256:")
                and candidate_report.get("identity") != requested
            ):
                continue
            report = candidate_report
            break
        if report is None:
            raise FileNotFoundError(f"predecessor report does not exist: {requested}")
        _validate_comparison_document(report, baseline)
        lineage = report.get("latest_metrics", {}).get(metric)
        metric_record = report.get("metrics", {}).get(metric)
        if (
            not isinstance(lineage, dict)
            or not isinstance(metric_record, dict)
            or lineage.get("result_identity") != metric_record.get("result_identity")
        ):
            raise ValueError(f"predecessor for {metric} is wrong-metric or stale")
        for comparison_name in ("baseline_comparison", "previous_comparison"):
            comparison = metric_record.get(comparison_name)
            if (
                comparison is not None
                and comparison.get("decision") != "pass"
                and (comparison.get("analysis", {}).get("decision") != "pass")
            ):
                raise ValueError(f"predecessor for {metric} was not accepted")
        reports[metric] = report
        lineages[metric] = lineage
        proof[metric] = {
            "report_identity": report["identity"],
            "result_identity": lineage["result_identity"],
        }
    return reports, lineages, proof


def _comparison_evidence_session_identity(
    *,
    harness_commit: str,
    harness_identity: str,
    reference_commit: str,
    candidate_commit: str,
    comparison_target: str,
    attempt_index: int,
    metrics: Sequence[MetricName],
    pairs: int,
    warmup: int,
    measure: int,
) -> str:
    return structured_identity(
        "sml-comparison-evidence-session-v1",
        {
            "harness_commit": harness_commit,
            "harness_identity": harness_identity,
            "reference_commit": reference_commit,
            "candidate_commit": candidate_commit,
            "comparison_target": comparison_target,
            "attempt_index": attempt_index,
            "metrics": list(metrics),
            "pairs": pairs,
            "warmup": warmup,
            "measure": measure,
        },
    )


def _run_paired_trials(
    *,
    harness_root: Path,
    reference_root: Path,
    candidate_root: Path,
    reference_commit: str,
    candidate_commit: str,
    harness_commit: str,
    harness_identity: str,
    reference_adapter: str,
    metrics: Sequence[MetricName],
    pairs: int,
    warmup: int,
    measure: int,
    comparison_target: str,
    attempt_index: int,
    output_directory: Path,
) -> list[RawTrial]:
    workload = build_canonical_workload()
    evidence_session_identity = _comparison_evidence_session_identity(
        harness_commit=harness_commit,
        harness_identity=harness_identity,
        reference_commit=reference_commit,
        candidate_commit=candidate_commit,
        comparison_target=comparison_target,
        attempt_index=attempt_index,
        metrics=metrics,
        pairs=pairs,
        warmup=warmup,
        measure=measure,
    )
    trials = []
    for metric in metrics:
        for pair_index in range(pairs):
            order = process_order(pair_index)
            for order_index, side in enumerate(order):
                is_reference = side == "reference"
                stem = f"{metric}-{comparison_target[-12:]}-{pair_index}-{side}"
                trial = _launch_trial(
                    harness_root=harness_root,
                    source_root=reference_root if is_reference else candidate_root,
                    source_commit=(
                        reference_commit if is_reference else candidate_commit
                    ),
                    harness_commit=harness_commit,
                    harness_identity=harness_identity,
                    adapter=reference_adapter if is_reference else "replacement",
                    metric=metric,
                    side=side,
                    attempt_index=attempt_index,
                    pair_index=pair_index,
                    order=order_index,
                    warmup=warmup,
                    measure=measure,
                    comparison_target=comparison_target,
                    evidence_session_identity=evidence_session_identity,
                    journal_attempt_index=attempt_index,
                    measurement_output=(output_directory / f"{stem}.measurement.json"),
                    post_exit_output=output_directory / f"{stem}.post-exit.json",
                    recovery_samples_directory=(
                        output_directory / f"{stem}.recovery-samples"
                    ),
                    recovery_output=output_directory / f"{stem}.recovery.json",
                    output=output_directory / f"{stem}.trial.json",
                )
                _validate_acceptance_environment(workload, trial)
                _validate_software_versions(workload, trial.software_versions)
                trials.append(trial)
    return trials


def _analyze_previous_pairs(
    *,
    metric: MetricName,
    trials: Sequence[RawTrial],
    target: str,
    reference_commit: str,
    candidate_commit: str,
    workload: CanonicalWorkload,
    minimum_ratio: float,
    maximum_dispersion: float,
    require_lower_bound: bool,
    bootstrap_resamples: int,
    predecessor_result_identity: str,
    attempt_index: int,
) -> dict:
    _validate_raw_trials_evidence(trials)
    reference = sorted(
        (
            trial
            for trial in trials
            if trial.metric == metric
            and trial.attempt_index == attempt_index
            and trial.side == "reference"
            and trial.source_commit == reference_commit
            and trial.comparison_target == target
        ),
        key=lambda trial: trial.pair_index,
    )
    candidate = sorted(
        (
            trial
            for trial in trials
            if trial.metric == metric
            and trial.attempt_index == attempt_index
            and trial.side == "candidate"
            and trial.source_commit == candidate_commit
            and trial.comparison_target == target
        ),
        key=lambda trial: trial.pair_index,
    )
    if not reference or len(reference) != len(candidate):
        raise ValueError(f"metric {metric} has incomplete predecessor pairs")
    direction = next(
        unit.direction for unit in workload.work_units if unit.metric == metric
    )
    analysis = analyze_pairs(
        [trial.value for trial in reference],
        [trial.value for trial in candidate],
        direction=direction,
        bootstrap_seed=1729,
        resamples=bootstrap_resamples,
        minimum_ratio=minimum_ratio,
        maximum_dispersion=maximum_dispersion,
        require_lower_bound=require_lower_bound,
    )
    return {
        "attempt_index": attempt_index,
        "predecessor_result_identity": predecessor_result_identity,
        "predecessor_source_commit": reference_commit,
        "analysis": _metric_report_dict(analysis),
        "raw_trial_identities": [
            _raw_trial_identity(trial) for trial in (*reference, *candidate)
        ],
    }


def _collect_comparison_attempt(
    *,
    repository: Path,
    args: argparse.Namespace,
    baseline: dict,
    predecessor_metrics: dict[str, dict],
    workload: CanonicalWorkload,
    harness_commit: str,
    harness_identity: str,
    baseline_commit: str,
    candidate_commit: str,
    attempt_index: int,
) -> tuple[list[RawTrial], dict[str, dict]]:
    with tempfile.TemporaryDirectory(
        prefix=f"sml-v2-compare-attempt-{attempt_index}-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        harness_root = temporary / "harness"
        baseline_root = temporary / "baseline-source"
        candidate_root = temporary / "candidate-source"
        created_worktrees = []
        try:
            for commit, destination in (
                (harness_commit, harness_root),
                (baseline_commit, baseline_root),
                (candidate_commit, candidate_root),
            ):
                _create_detached_worktree(repository, commit, destination)
                created_worktrees.append(destination)
            if harness_content_identity(harness_root) != harness_identity:
                raise RuntimeError(
                    "pinned harness checkout has the wrong content identity"
                )
            baseline_target = f"baseline:{baseline['identity']}"
            trials = _run_paired_trials(
                harness_root=harness_root,
                reference_root=baseline_root,
                candidate_root=candidate_root,
                reference_commit=baseline_commit,
                candidate_commit=candidate_commit,
                harness_commit=harness_commit,
                harness_identity=harness_identity,
                reference_adapter="legacy",
                metrics=args.metrics,
                pairs=args.pairs,
                warmup=args.warmup,
                measure=args.measure,
                comparison_target=baseline_target,
                attempt_index=attempt_index,
                output_directory=temporary,
            )
            previous_comparisons = {}
            if predecessor_metrics:
                roots_by_commit = {}
                for metric in args.metrics:
                    predecessor = predecessor_metrics.get(metric)
                    if predecessor is None:
                        continue
                    previous_commit = predecessor["source_commit"]
                    if previous_commit == baseline_commit:
                        continue
                    previous_root = roots_by_commit.get(previous_commit)
                    if previous_root is None:
                        previous_root = temporary / f"previous-{len(roots_by_commit)}"
                        _create_detached_worktree(
                            repository, previous_commit, previous_root
                        )
                        created_worktrees.append(previous_root)
                        roots_by_commit[previous_commit] = previous_root
                    target = f"previous:{predecessor['result_identity']}"
                    previous_trials = _run_paired_trials(
                        harness_root=harness_root,
                        reference_root=previous_root,
                        candidate_root=candidate_root,
                        reference_commit=previous_commit,
                        candidate_commit=candidate_commit,
                        harness_commit=harness_commit,
                        harness_identity=harness_identity,
                        reference_adapter="replacement",
                        metrics=(metric,),
                        pairs=args.pairs,
                        warmup=args.warmup,
                        measure=args.measure,
                        comparison_target=target,
                        attempt_index=attempt_index,
                        output_directory=temporary,
                    )
                    trials.extend(previous_trials)
                    previous_comparisons[metric] = _analyze_previous_pairs(
                        metric=metric,
                        trials=previous_trials,
                        target=target,
                        reference_commit=previous_commit,
                        candidate_commit=candidate_commit,
                        workload=workload,
                        minimum_ratio=args.minimum_ratio,
                        maximum_dispersion=args.maximum_dispersion,
                        require_lower_bound=not args.lower_bound_report_only,
                        bootstrap_resamples=args.bootstrap_resamples,
                        predecessor_result_identity=predecessor["result_identity"],
                        attempt_index=attempt_index,
                    )
            return trials, previous_comparisons
        finally:
            for destination in reversed(created_worktrees):
                _remove_worktree(repository, destination)


def _combine_previous_attempts(
    first: dict[str, dict], second: dict[str, dict] | None = None
) -> dict[str, dict]:
    combined = {}
    for metric, first_record in first.items():
        records = [first_record]
        if second is not None:
            retry = second.get(metric)
            if retry is None:
                raise ValueError("retry omitted a predecessor comparison")
            records.append(retry)
        final = records[-1]
        combined[metric] = {
            "predecessor_result_identity": final["predecessor_result_identity"],
            "predecessor_source_commit": final["predecessor_source_commit"],
            "analysis": final["analysis"],
            "raw_trial_identities": [
                identity
                for record in records
                for identity in record["raw_trial_identities"]
            ],
            "attempts": [
                {
                    "attempt_index": record["attempt_index"],
                    "analysis": record["analysis"],
                    "raw_trial_identities": record["raw_trial_identities"],
                }
                for record in records
            ],
        }
    if second is not None and set(second) != set(first):
        raise ValueError("retry predecessor metric set changed")
    return combined


def _compare(args: argparse.Namespace) -> int:
    repository = _git_root(Path.cwd())
    _require_clean_checkout(repository, label="candidate")
    comparison_mode = _resolve_comparison_mode(args)
    baseline = _read_json_object(args.baseline, label="baseline manifest")
    _validate_baseline_document(baseline)
    predecessor_reports, predecessor_metrics, predecessor_proof = (
        _resolve_predecessor_mapping(
            args.predecessors, repository, baseline, args.metrics
        )
    )
    if comparison_mode == COMPARISON_FINAL:
        supplied = {
            metric
            for metric, predecessor in predecessor_reports.items()
            if predecessor is not None
        }
        if supplied != set(FINAL_PREDECESSOR_METRICS):
            raise ValueError("final acceptance has the wrong predecessor mapping")
    harness_commit = baseline["harness"]["commit"]
    harness_identity = baseline["harness"]["content_identity"]
    baseline_commit = baseline["source"]["commit"]
    candidate_commit = _git_commit(repository, args.candidate)
    workload = CanonicalWorkload.from_dict(baseline["canonical_workload"])

    trials, first_previous = _collect_comparison_attempt(
        repository=repository,
        args=args,
        baseline=baseline,
        predecessor_metrics=predecessor_metrics,
        workload=workload,
        harness_commit=harness_commit,
        harness_identity=harness_identity,
        baseline_commit=baseline_commit,
        candidate_commit=candidate_commit,
        attempt_index=0,
    )
    previous_comparisons = _combine_previous_attempts(first_previous)
    report_arguments = {
        "baseline": baseline,
        "candidate_commit": candidate_commit,
        "minimum_ratio": args.minimum_ratio,
        "pretraining_minimum_ratio": args.pretraining_minimum_ratio,
        "maximum_dispersion": args.maximum_dispersion,
        "require_lower_bound": not args.lower_bound_report_only,
        "bootstrap_resamples": args.bootstrap_resamples,
        "predecessor_metrics": predecessor_metrics,
        "predecessors": predecessor_proof,
        "previous_comparisons": previous_comparisons,
        "comparison_mode": comparison_mode,
        "pairs": args.pairs,
        "warmup_units": args.warmup,
        "measured_units": args.measure,
    }
    provisional = build_comparison_report(trials=trials, **report_arguments)
    cooldown_evidence = None
    if comparison_has_noise(provisional):
        cooldown_evidence = perform_cooldown()
        validate_cooldown_evidence(cooldown_evidence, workload.required_environment)
        retry_trials, retry_previous = _collect_comparison_attempt(
            repository=repository,
            args=args,
            baseline=baseline,
            predecessor_metrics=predecessor_metrics,
            workload=workload,
            harness_commit=harness_commit,
            harness_identity=harness_identity,
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit,
            attempt_index=1,
        )
        trials.extend(retry_trials)
        previous_comparisons = _combine_previous_attempts(
            first_previous, retry_previous
        )
        report_arguments["previous_comparisons"] = previous_comparisons
    report = build_comparison_report(
        trials=trials,
        cooldown_evidence=cooldown_evidence,
        **report_arguments,
    )
    validate_comparison_report(report, baseline, predecessor_reports)
    if comparison_mode == COMPARISON_FINAL:
        validate_throughput_gates(report, label="final acceptance")
    if args.raw_output is not None:
        _write_jsonl(args.raw_output, trials)
    _write_json(args.output, report)
    return 0


def _validate_phase(args: argparse.Namespace) -> int:
    repository = _git_root(Path.cwd())
    baseline = _read_json_object(args.baseline, label="baseline manifest")
    _validate_baseline_document(baseline)
    report = _read_json_object(args.results, label="phase results")
    _validate_report_raw_trial_evidence(report)
    predecessor_reports, _lineages, _proof = _resolve_predecessor_mapping(
        args.predecessors,
        repository,
        baseline,
        tuple(report.get("metrics", {})),
    )
    validate_comparison_report(report, baseline, predecessor_reports)
    if report["comparison_mode"] != COMPARISON_SCREEN:
        raise ValueError("phase validation requires a screen-mode report")
    expected_metrics = PHASE_METRICS.get(args.phase)
    if expected_metrics is None:
        raise ValueError("unsupported refactor phase")
    if tuple(report["metrics"]) != expected_metrics:
        raise ValueError(f"phase {args.phase} measured the wrong metric set")
    required_predecessors = PHASE_PREDECESSOR_METRICS[args.phase]
    if {
        metric for metric, predecessor in predecessor_reports.items() if predecessor
    } != set(required_predecessors):
        raise ValueError(f"phase {args.phase} has the wrong predecessor mapping")
    validate_throughput_gates(report, label=f"phase {args.phase}")
    if args.output is not None:
        _write_json(args.output, report)
    return 0


def validate_final_report(
    report: dict,
    baseline: dict,
    predecessor_reports: dict[str, dict | None],
    raw_trials: Sequence[RawTrial],
) -> None:
    _validate_raw_trials_evidence(raw_trials)
    if [trial.to_dict() for trial in raw_trials] != report.get("raw_trials"):
        raise ValueError("final raw input does not exactly match the complete report")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or tuple(metrics) != FINAL_METRICS:
        raise ValueError("final acceptance measured the wrong metric set")
    if report.get("comparison_mode") != COMPARISON_FINAL:
        raise ValueError("final validation requires a final-mode report")
    proofs = report.get("predecessors")
    if not isinstance(proofs, dict) or set(proofs) != set(FINAL_METRICS):
        raise ValueError("final acceptance has an invalid predecessor proof set")
    supplied = {metric for metric, proof in proofs.items() if proof is not None}
    if supplied != set(FINAL_PREDECESSOR_METRICS):
        raise ValueError("final acceptance has the wrong predecessor mapping")
    validate_comparison_report(report, baseline, predecessor_reports)
    validate_throughput_gates(report, label="final acceptance")


def _validate_final(args: argparse.Namespace) -> int:
    repository = _git_root(Path.cwd())
    baseline = _read_json_object(args.baseline, label="baseline manifest")
    _validate_baseline_document(baseline)
    report = _read_json_object(args.report, label="final report")
    _validate_report_raw_trial_evidence(report)
    proofs = report.get("predecessors")
    if not isinstance(proofs, dict) or set(proofs) != set(FINAL_METRICS):
        raise ValueError("final acceptance has an invalid predecessor proof set")
    supplied = {metric for metric, proof in proofs.items() if proof is not None}
    if supplied != set(FINAL_PREDECESSOR_METRICS):
        raise ValueError("final acceptance has the wrong predecessor mapping")
    predecessor_mapping = {}
    for metric, proof in proofs.items():
        if proof is None:
            predecessor_mapping[metric] = None
        elif (
            isinstance(proof, dict)
            and set(proof) == {"report_identity", "result_identity"}
            and isinstance(proof.get("report_identity"), str)
        ):
            predecessor_mapping[metric] = proof["report_identity"]
        else:
            raise ValueError("final acceptance has an invalid predecessor proof")
    predecessor_reports, _lineages, _resolved_proof = _resolve_predecessor_mapping(
        json.dumps(predecessor_mapping),
        repository,
        baseline,
        FINAL_METRICS,
        additional_search_directories=(args.report.parent,),
    )
    raw_trials = _read_trials(args.raw_input)
    validate_final_report(report, baseline, predecessor_reports, raw_trials)
    return 0


def validate_throughput_gates(report: dict, *, label: str) -> None:
    throughput_metrics = {
        "prepared-data",
        "pretraining-compute",
        "pretraining-end-to-end",
        "swag-end-to-end",
        "inference-prefill",
        "inference-decode",
    }
    for metric, record in report["metrics"].items():
        if metric not in throughput_metrics:
            continue
        if record["baseline_comparison"]["decision"] != "pass":
            raise ValueError(f"{label} baseline gate failed for {metric}")
        previous_comparison = record["previous_comparison"]
        if (
            previous_comparison is not None
            and previous_comparison["analysis"]["decision"] != "pass"
        ):
            raise ValueError(f"{label} predecessor gate failed for {metric}")


def process_order(pair_index: int) -> tuple[Side, Side]:
    if pair_index < 0:
        raise ValueError("pair_index must be non-negative")
    if pair_index % 2 == 0:
        return ("reference", "candidate")
    return ("candidate", "reference")


def _resolve_comparison_mode(args: argparse.Namespace) -> str:
    mode = args.mode
    if mode is None:
        mode = (
            COMPARISON_FINAL
            if args.pairs == FINAL_PAIRS and not args.lower_bound_report_only
            else COMPARISON_SCREEN
        )
    protocol = {
        "pairs": args.pairs,
        "compilation_passes": 1,
        "warmup_units": args.warmup,
        "measured_units": args.measure,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": args.bootstrap_resamples,
        "minimum_ratio": args.minimum_ratio,
        "pretraining_minimum_ratio": args.pretraining_minimum_ratio,
        "maximum_dispersion": args.maximum_dispersion,
        "require_lower_bound": not args.lower_bound_report_only,
    }
    _validate_comparison_protocol({"comparison_mode": mode, "protocol": protocol})
    if mode == COMPARISON_FINAL and tuple(args.metrics) != FINAL_METRICS:
        raise ValueError("final acceptance requires the complete final metric set")
    return mode


def parse_metrics(value: str) -> tuple[MetricName, ...]:
    metrics = tuple(part.strip() for part in value.split(",") if part.strip())
    unsupported = tuple(metric for metric in metrics if metric not in METRIC_NAMES)
    if unsupported:
        raise ValueError(f"unsupported benchmark metric: {unsupported[0]}")
    if len(set(metrics)) != len(metrics):
        raise ValueError("duplicate benchmark metric")
    if not metrics:
        raise ValueError("at least one benchmark metric is required")
    return metrics


def measure_native_process(
    *,
    adapter,
    metric: MetricName,
    native_workload,
    warmup_units: int,
    measured_units: int,
    synchronize,
    clock=time.perf_counter,
    peak_memory,
    reset_peak_memory,
) -> ProcessMeasurement:
    if warmup_units < 0:
        raise ValueError("warmup_units must be non-negative")
    if measured_units <= 0:
        raise ValueError("measured_units must be positive")

    compilation_seconds: float | None = None
    if metric != "compile-cold-start":
        synchronize()
        compilation_start = clock()
        adapter.run_warmup(metric, native_workload, 1)
        synchronize()
        compilation_seconds = clock() - compilation_start
        if compilation_seconds <= 0:
            raise RuntimeError("benchmark compilation clock did not advance")
        for _ in range(warmup_units):
            adapter.run_warmup(metric, native_workload, 1)
            synchronize()

    reset_peak_memory()
    synchronize()
    start = clock()
    work_count = float(adapter.run_measured(metric, native_workload, measured_units))
    synchronize()
    elapsed = clock() - start
    if elapsed <= 0:
        raise RuntimeError("benchmark clock did not advance")
    peak = int(peak_memory())

    if metric == "peak-metal-memory":
        value = float(peak)
    elif metric in ("checkpoint-pause", "compile-cold-start"):
        value = elapsed / measured_units
    else:
        if work_count <= 0:
            raise RuntimeError("throughput benchmark produced no work")
        value = work_count / elapsed
    return ProcessMeasurement(
        elapsed_seconds=elapsed,
        value=value,
        work_count=work_count,
        compilation_seconds=compilation_seconds,
        peak_memory_bytes=peak,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the versioned v2 benchmarks.")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    baseline = subparsers.add_parser("record-baseline")
    baseline.add_argument("--source-commit", required=True)
    baseline.add_argument("--manifest", type=Path, required=True)
    baseline.add_argument("--raw-output", type=Path, required=True)
    baseline.add_argument(
        "--state-directory",
        type=Path,
        required=True,
        help="external durable journal directory for baseline resume and diagnostics",
    )
    baseline.add_argument("--metrics", type=parse_metrics, default=METRIC_NAMES)
    baseline.add_argument("--pairs", type=int, default=SCREEN_PAIRS)
    baseline.add_argument("--warmup", type=int, default=WARMUP_UNITS)
    baseline.add_argument("--measure", type=int, default=DEFAULT_MEASURED_UNITS)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--metrics", type=parse_metrics, required=True)
    compare.add_argument("--mode", choices=(COMPARISON_SCREEN, COMPARISON_FINAL))
    compare.add_argument("--pairs", type=int, default=SCREEN_PAIRS)
    compare.add_argument("--warmup", type=int, default=WARMUP_UNITS)
    compare.add_argument("--measure", type=int, default=DEFAULT_MEASURED_UNITS)
    compare.add_argument("--bootstrap-resamples", type=int, default=10_000)
    compare.add_argument("--minimum-ratio", type=float, default=0.97)
    compare.add_argument("--pretraining-minimum-ratio", type=float)
    compare.add_argument("--maximum-dispersion", type=float, default=0.02)
    compare.add_argument("--lower-bound-report-only", action="store_true")
    compare.add_argument(
        "--predecessors",
        required=True,
        help="JSON metric-to-report-path-or-null mapping, or a path containing it",
    )
    compare.add_argument("--raw-output", type=Path)
    compare.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--raw-input", type=Path, required=True)

    validate_phase = subparsers.add_parser("validate-phase")
    validate_phase.add_argument("--phase", type=int, required=True)
    validate_phase.add_argument("--baseline", type=Path, required=True)
    validate_phase.add_argument("--predecessors", required=True)
    validate_phase.add_argument("--results", type=Path, required=True)
    validate_phase.add_argument("--output", type=Path)

    validate_final = subparsers.add_parser("validate-final")
    validate_final.add_argument("--baseline", type=Path, required=True)
    validate_final.add_argument("--raw-input", type=Path, required=True)
    validate_final.add_argument("--report", type=Path, required=True)

    process = subparsers.add_parser("_run-process", help=argparse.SUPPRESS)
    process.add_argument("--harness-root", type=Path, required=True)
    process.add_argument("--source-root", type=Path, required=True)
    process.add_argument("--source-commit", required=True)
    process.add_argument("--harness-commit", required=True)
    process.add_argument("--harness-identity", required=True)
    process.add_argument("--adapter", choices=("legacy", "replacement"), required=True)
    process.add_argument("--metric", choices=METRIC_NAMES, required=True)
    process.add_argument("--side", choices=("reference", "candidate"), required=True)
    process.add_argument("--attempt-index", type=int, required=True)
    process.add_argument("--pair-index", type=int, required=True)
    process.add_argument("--process-order", type=int, required=True)
    process.add_argument("--warmup", type=int, required=True)
    process.add_argument("--measure", type=int, required=True)
    process.add_argument("--comparison-target", required=True)
    process.add_argument("--evidence-session-identity", required=True)
    process.add_argument("--journal-attempt-index", type=int, required=True)
    process.add_argument("--measurement-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "record-baseline":
        return _record_baseline(args)
    if args.operation == "validate":
        return _validate_baseline_files(args)
    if args.operation == "_run-process":
        return _run_single_process(args)
    if args.operation == "compare":
        return _compare(args)
    if args.operation == "validate-phase":
        return _validate_phase(args)
    if args.operation == "validate-final":
        return _validate_final(args)
    raise AssertionError(f"unhandled operation: {args.operation}")


if __name__ == "__main__":
    raise SystemExit(main())
