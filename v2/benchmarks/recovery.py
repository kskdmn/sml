from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

_SAMPLE_INTERVAL_SECONDS = 27.0
_REQUIRED_NOMINAL_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class ThermalRecoveryResult:
    duration_seconds: float
    sample_count: int


class ThermalRecoveryTimeout(RuntimeError):
    def __init__(self, result: ThermalRecoveryResult):
        super().__init__("thermal recovery exceeded the two-hour deadline")
        self.result = result


@dataclass(frozen=True, slots=True)
class PostExitMemoryRecoveryResult:
    outcome: Literal[
        "not-required", "recovered", "timeout", "critical", "environment-failure"
    ]
    duration_seconds: float
    samples: tuple[dict, ...]
    failure_fields: tuple[str, ...]


def _require_positive_finite(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def wait_for_post_exit_memory_recovery(
    *,
    immediate_observation: dict,
    immediate_started_at: float,
    recovery_policy: dict,
    collect: Callable[[], dict],
    classify_nonmemory: Callable[[dict], tuple[str, ...]],
    record_sample: Callable[[int, float, dict, str | None], dict],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> PostExitMemoryRecoveryResult:
    if (
        type(immediate_started_at) not in (int, float)
        or not math.isfinite(immediate_started_at)
        or immediate_started_at < 0
    ):
        raise ValueError("immediate_started_at must be a finite non-negative number")

    interval = _require_positive_finite(
        recovery_policy["sample_interval_seconds"], label="recovery sample interval"
    )
    timeout = _require_positive_finite(
        recovery_policy["timeout_seconds"], label="recovery timeout"
    )
    stability = _require_positive_finite(
        recovery_policy["stability_seconds"], label="recovery stability window"
    )
    if stability > timeout:
        raise ValueError("recovery stability window must not exceed recovery timeout")

    immediate_failures = tuple(classify_nonmemory(immediate_observation))
    immediate_pressure = immediate_observation["environment_status"]["memory_pressure"]
    if immediate_failures:
        return PostExitMemoryRecoveryResult(
            "environment-failure", 0.0, (), immediate_failures
        )
    if immediate_pressure == "critical":
        return PostExitMemoryRecoveryResult("critical", 0.0, (), ())
    if immediate_pressure == "normal":
        return PostExitMemoryRecoveryResult("not-required", 0.0, (), ())
    if immediate_pressure != "warning":
        raise ValueError("immediate post-exit memory pressure is invalid")

    deadline = immediate_started_at + timeout
    samples = []
    normal_since = None
    previous_identity = None
    last_sample_started_at = immediate_started_at

    while True:
        scheduled = min(last_sample_started_at + interval, deadline)
        sleep(max(0.0, scheduled - clock()))
        sample_started_at = clock()
        if sample_started_at > deadline:
            return PostExitMemoryRecoveryResult(
                "timeout", deadline - immediate_started_at, tuple(samples), ()
            )
        observation = collect()
        elapsed = sample_started_at - immediate_started_at
        sample = record_sample(len(samples), elapsed, observation, previous_identity)
        samples.append(sample)
        previous_identity = sample["identity"]
        last_sample_started_at = sample_started_at

        failures = tuple(classify_nonmemory(observation))
        pressure = observation["environment_status"]["memory_pressure"]
        if failures:
            return PostExitMemoryRecoveryResult(
                "environment-failure", elapsed, tuple(samples), failures
            )
        if pressure == "critical":
            return PostExitMemoryRecoveryResult("critical", elapsed, tuple(samples), ())
        if pressure == "warning":
            normal_since = None
        elif pressure == "normal":
            normal_since = elapsed if normal_since is None else normal_since
            if elapsed - normal_since >= stability:
                return PostExitMemoryRecoveryResult(
                    "recovered", elapsed, tuple(samples), ()
                )
        else:
            raise ValueError("recovery memory pressure is invalid")
        if sample_started_at >= deadline:
            return PostExitMemoryRecoveryResult("timeout", elapsed, tuple(samples), ())


def _require_exact_match(actual: dict, expected: dict, *, label: str) -> None:
    if actual == expected:
        return
    for field in sorted(set(actual) | set(expected)):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"{label} does not match expected {field}")
    raise ValueError(f"{label} does not match expected values")


def _validate_nonthermal_environment(
    status: dict[str, object], required_environment: dict[str, object]
) -> None:
    expected_booleans = {
        "power_connected": True,
        "low_power_mode": False,
        "competing_gpu_workload": False,
    }
    for field, value in expected_booleans.items():
        observed = status.get(field)
        if type(observed) is not bool or observed is not value:
            raise ValueError(f"environment status does not match required {field}")
    expected = {
        "power_mode": required_environment["power_mode"],
        "memory_pressure": "normal",
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise ValueError(f"environment status does not match required {field}")


def wait_for_nominal_thermal_window(
    *,
    collect: Callable[[], tuple[dict, dict, dict]],
    expected_hardware: dict,
    expected_software_versions: dict[str, str],
    required_environment: dict[str, object],
    record_sample: Callable[[int, dict], None],
    deadline: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    utc_now: Callable[[], str],
) -> ThermalRecoveryResult:
    from v2.benchmarks.runner import validate_thermal_observation

    started = clock()
    scheduled = started
    sample_count = 0
    nominal_since: float | None = None

    while True:
        hardware, status, software_versions = collect()
        elapsed = clock() - started
        sample = {
            "schema_version": 1,
            "observed_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "hardware": hardware,
            "environment_status": status,
            "software_versions": software_versions,
        }
        record_sample(sample_count, sample)
        sample_count += 1

        _require_exact_match(hardware, expected_hardware, label="hardware")
        _require_exact_match(
            software_versions,
            expected_software_versions,
            label="software versions",
        )
        validate_thermal_observation(status)
        _validate_nonthermal_environment(status, required_environment)

        if status["thermal_state"] == "nominal":
            if nominal_since is None:
                nominal_since = elapsed
        else:
            nominal_since = None

        if clock() >= deadline:
            raise ThermalRecoveryTimeout(ThermalRecoveryResult(elapsed, sample_count))
        if (
            nominal_since is not None
            and elapsed - nominal_since >= _REQUIRED_NOMINAL_SECONDS
        ):
            return ThermalRecoveryResult(elapsed, sample_count)

        scheduled = min(scheduled + _SAMPLE_INTERVAL_SECONDS, deadline)
        sleep(max(0.0, scheduled - clock()))
