from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class PostExitMemoryRecoveryTerminal:
    outcome: Literal[
        "not-required", "recovered", "timeout", "critical", "environment-failure"
    ]
    duration_seconds: float
    failure_fields: tuple[str, ...]
    terminal_sample_index: int | None


def _require_positive_finite(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _validated_recovery_times(
    recovery_policy: Mapping[str, object],
) -> tuple[float, ...]:
    interval = _require_positive_finite(
        recovery_policy["sample_interval_seconds"],
        label="recovery sample interval",
    )
    timeout = _require_positive_finite(
        recovery_policy["timeout_seconds"], label="recovery timeout"
    )
    stability = _require_positive_finite(
        recovery_policy["stability_seconds"],
        label="recovery stability window",
    )
    if stability > timeout:
        raise ValueError("recovery stability window must not exceed recovery timeout")
    return interval, timeout, stability


def reduce_post_exit_memory_recovery(
    *,
    immediate_observation: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
    recovery_policy: Mapping[str, object],
    classify_nonmemory: Callable[[Mapping[str, object]], Sequence[str]],
    deadline_reached: bool = False,
) -> PostExitMemoryRecoveryTerminal | None:
    interval, timeout, stability = _validated_recovery_times(recovery_policy)
    immediate_failures = tuple(classify_nonmemory(immediate_observation))
    immediate_status = immediate_observation["environment_status"]
    if not isinstance(immediate_status, Mapping):
        raise TypeError("immediate recovery environment status must be an object")
    immediate_pressure = immediate_status["memory_pressure"]
    immediate_terminal: PostExitMemoryRecoveryTerminal | None = None
    if immediate_failures:
        immediate_terminal = PostExitMemoryRecoveryTerminal(
            "environment-failure", 0.0, immediate_failures, None
        )
    elif immediate_pressure == "critical":
        immediate_terminal = PostExitMemoryRecoveryTerminal("critical", 0.0, (), None)
    elif immediate_pressure == "normal":
        immediate_terminal = PostExitMemoryRecoveryTerminal(
            "not-required", 0.0, (), None
        )
    elif immediate_pressure != "warning":
        raise ValueError("immediate post-exit memory pressure is invalid")
    if immediate_terminal is not None:
        if samples:
            raise ValueError("recovery contains samples after the first terminal event")
        return immediate_terminal

    normal_since: float | None = None
    previous_elapsed = 0.0
    for index, sample in enumerate(samples):
        elapsed = sample["elapsed_seconds"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed):
            raise ValueError("recovery sample elapsed_seconds must be finite")
        elapsed = float(elapsed)
        if elapsed < previous_elapsed + interval:
            raise ValueError("recovery sample cadence is infeasible")
        if elapsed > timeout:
            raise ValueError("recovery sample exceeds the timeout")
        previous_elapsed = elapsed

        failures = tuple(classify_nonmemory(sample))
        status = sample["environment_status"]
        if not isinstance(status, Mapping):
            raise TypeError("recovery sample environment status must be an object")
        pressure = status["memory_pressure"]
        terminal: PostExitMemoryRecoveryTerminal | None = None
        if failures:
            terminal = PostExitMemoryRecoveryTerminal(
                "environment-failure", elapsed, failures, index
            )
        elif pressure == "critical":
            terminal = PostExitMemoryRecoveryTerminal("critical", elapsed, (), index)
        elif pressure == "warning":
            normal_since = None
        elif pressure == "normal":
            normal_since = elapsed if normal_since is None else normal_since
            if elapsed - normal_since >= stability:
                terminal = PostExitMemoryRecoveryTerminal(
                    "recovered", elapsed, (), index
                )
        else:
            raise ValueError("recovery memory pressure is invalid")
        if terminal is None and elapsed == timeout:
            terminal = PostExitMemoryRecoveryTerminal("timeout", timeout, (), index)
        if terminal is not None:
            if index != len(samples) - 1:
                raise ValueError(
                    "recovery contains samples after the first terminal event"
                )
            return terminal

    if deadline_reached:
        return PostExitMemoryRecoveryTerminal("timeout", timeout, (), None)
    return None


def wait_for_post_exit_memory_recovery(
    *,
    immediate_observation: dict,
    immediate_started_at: float,
    recovery_policy: dict,
    collect: Callable[[float], tuple[float, dict] | None],
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

    interval, timeout, _stability = _validated_recovery_times(recovery_policy)
    terminal = reduce_post_exit_memory_recovery(
        immediate_observation=immediate_observation,
        samples=(),
        recovery_policy=recovery_policy,
        classify_nonmemory=classify_nonmemory,
    )
    if terminal is not None:
        return PostExitMemoryRecoveryResult(
            terminal.outcome, terminal.duration_seconds, (), terminal.failure_fields
        )

    deadline = immediate_started_at + timeout
    samples = []
    observations = []
    previous_identity = None
    last_sample_started_at = immediate_started_at

    def timeout_result() -> PostExitMemoryRecoveryResult:
        terminal = reduce_post_exit_memory_recovery(
            immediate_observation=immediate_observation,
            samples=observations,
            recovery_policy=recovery_policy,
            classify_nonmemory=classify_nonmemory,
            deadline_reached=True,
        )
        if terminal is None:
            raise AssertionError("recovery deadline did not produce a terminal state")
        return PostExitMemoryRecoveryResult(
            terminal.outcome,
            terminal.duration_seconds,
            tuple(samples),
            terminal.failure_fields,
        )

    while True:
        scheduled = min(last_sample_started_at + interval, deadline)
        sleep(max(0.0, scheduled - clock()))
        if clock() > deadline:
            return timeout_result()
        collected = collect(deadline)
        if collected is None:
            return timeout_result()
        sample_started_at, observation = collected
        if type(sample_started_at) not in (int, float) or not math.isfinite(
            sample_started_at
        ):
            raise ValueError("recovery sample start must be a finite number")
        sample_started_at = float(sample_started_at)
        if sample_started_at > deadline:
            return timeout_result()
        elapsed = sample_started_at - immediate_started_at
        sample = record_sample(len(samples), elapsed, observation, previous_identity)
        samples.append(sample)
        observations.append({**observation, "elapsed_seconds": elapsed})
        previous_identity = sample["identity"]
        last_sample_started_at = sample_started_at
        terminal = reduce_post_exit_memory_recovery(
            immediate_observation=immediate_observation,
            samples=observations,
            recovery_policy=recovery_policy,
            classify_nonmemory=classify_nonmemory,
        )
        if terminal is not None:
            return PostExitMemoryRecoveryResult(
                terminal.outcome,
                terminal.duration_seconds,
                tuple(samples),
                terminal.failure_fields,
            )


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
