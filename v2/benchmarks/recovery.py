from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
