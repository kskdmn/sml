import pytest

from v2.benchmarks.evidence import (
    build_post_exit_observation,
    build_post_exit_recovery,
    build_post_exit_recovery_sample,
    finalize_raw_trial,
    validate_raw_trial_evidence,
)
from v2.benchmarks.recovery import (
    PostExitMemoryRecoveryResult,
    ThermalRecoveryTimeout,
    wait_for_nominal_thermal_window,
    wait_for_post_exit_memory_recovery,
)
from v2.benchmarks.runner import (
    perform_cooldown,
    validate_cooldown_evidence,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
    post_exit_recovery_policy,
)

from .support import (
    _nominal_cooldown_evidence,
    _RecoveryClock,
    _valid_child_measurement,
    _valid_observation,
    _valid_recovery_policy,
)


def _recovery_collect(states):
    remaining = iter(states)
    last = [states[-1]]

    def collect():
        try:
            last[0] = next(remaining)
        except StopIteration:
            pass
        status = {
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": last[0],
            "thermal_state_raw_value": {
                "nominal": 0,
                "fair": 1,
                "serious": 2,
                "critical": 3,
            }[last[0]],
            "memory_pressure": "normal",
            "memory_free_percentage": 60,
            "competing_gpu_workload": False,
        }
        return {"chip": "Apple M5"}, status, {"python": "3.12.16"}

    return collect


def test_cooldown_uses_fake_clock_and_proves_final_nominal_window():
    now = [0.0]
    status = _nominal_cooldown_evidence()["samples"][0]["environment_status"]

    evidence = perform_cooldown(
        collect=lambda: ({}, status, {}),
        clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    validate_cooldown_evidence(
        evidence, build_canonical_workload().required_environment
    )
    assert evidence["duration_seconds"] == 900.0
    assert len(evidence["samples"]) >= 16


def test_generated_cooldown_evidence_survives_a_realistically_advancing_clock():
    now = [0.0]
    status = _nominal_cooldown_evidence()["samples"][0]["environment_status"]

    def clock():
        observed = now[0]
        now[0] += 0.001
        return observed

    evidence = perform_cooldown(
        collect=lambda: ({}, status, {}),
        clock=clock,
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert evidence["duration_seconds"] == evidence["samples"][-1]["elapsed_seconds"]
    validate_cooldown_evidence(
        evidence, build_canonical_workload().required_environment
    )


def test_cooldown_rejects_a_gap_larger_than_the_declared_interval():
    evidence = _nominal_cooldown_evidence()
    evidence["samples"].pop(1)

    with pytest.raises(ValueError, match="sampling gap"):
        validate_cooldown_evidence(
            evidence, build_canonical_workload().required_environment
        )


def test_thermal_recovery_requires_five_continuous_nominal_minutes():
    clock = _RecoveryClock()
    samples = []
    result = wait_for_nominal_thermal_window(
        collect=_recovery_collect(["nominal"] * 4 + ["fair"] + ["nominal"] * 20),
        expected_hardware={"chip": "Apple M5"},
        expected_software_versions={"python": "3.12.16"},
        required_environment=build_canonical_workload().required_environment,
        record_sample=lambda index, sample: samples.append((index, sample)),
        deadline=clock() + 7_200,
        clock=clock,
        sleep=clock.sleep,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
    )

    nominal_times = [
        sample["elapsed_seconds"]
        for _index, sample in samples
        if sample["environment_status"]["thermal_state"] == "nominal"
    ]
    assert result.duration_seconds >= 300
    assert nominal_times[-1] - nominal_times[4] >= 300


def test_thermal_recovery_times_out_without_losing_samples():
    clock = _RecoveryClock()
    samples = []
    with pytest.raises(ThermalRecoveryTimeout, match="two-hour deadline"):
        wait_for_nominal_thermal_window(
            collect=_recovery_collect(["fair"]),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.16"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: samples.append((index, sample)),
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
    assert samples[-1][1]["environment_status"]["thermal_state"] == "fair"
    assert samples[-1][1]["elapsed_seconds"] >= 120


def test_thermal_recovery_enforces_a_deadline_at_a_nominal_window_boundary():
    clock = _RecoveryClock()
    samples = []

    with pytest.raises(ThermalRecoveryTimeout, match="two-hour deadline") as raised:
        wait_for_nominal_thermal_window(
            collect=_recovery_collect(["nominal"]),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.16"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: samples.append((index, sample)),
            deadline=324.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )

    assert raised.value.result.duration_seconds == 324.0
    assert samples[-1][1]["elapsed_seconds"] == 324.0


def test_post_exit_recovery_returns_immediately_for_normal_memory():
    clock = _RecoveryClock()
    collected = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: collected.append(True),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda *args: pytest.fail("normal memory must not sample"),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result == PostExitMemoryRecoveryResult("not-required", 0.0, (), ())
    assert collected == []
    assert clock.sleeps == []


@pytest.mark.parametrize(
    ("pressure", "failure_fields", "outcome"),
    (
        ("critical", (), "critical"),
        ("normal", ("hardware",), "environment-failure"),
    ),
)
def test_post_exit_recovery_returns_immediately_for_terminal_immediate_observation(
    pressure, failure_fields, outcome
):
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = pressure

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: pytest.fail(
            "terminal immediate observation must not collect"
        ),
        classify_nonmemory=lambda observation: tuple(failure_fields),
        record_sample=lambda *args: pytest.fail(
            "terminal immediate observation must not sample"
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result == PostExitMemoryRecoveryResult(outcome, 0.0, (), failure_fields)
    assert clock.sleeps == []


def test_post_exit_recovery_requires_thirty_continuous_normal_seconds():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    observations = [
        _valid_observation(f"2026-08-09T00:00:{second:02d}+00:00")
        for second in (5, 10, 15, 20, 25, 30, 35)
    ]
    iter_observations = iter(observations)
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), next(iter_observations)),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted.append(
                {
                    "sample_index": index,
                    "elapsed_seconds": elapsed,
                    "identity": str(index),
                }
            )
            or persisted[-1]
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "recovered"
    assert result.duration_seconds == 35.0
    assert len(result.samples) == 7
    assert clock.sleeps == [5.0] * 7


def _run_scripted_memory_recovery(*, pressures, failure_fields=()):
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    scripted = []
    for index, pressure in enumerate(pressures, 1):
        observation = _valid_observation(
            f"2026-08-09T00:00:{min(index * 5, 59):02d}+00:00"
        )
        observation["environment_status"]["memory_pressure"] = pressure
        scripted.append(observation)
    iter_scripted = iter(scripted)
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), next(iter_scripted)),
        classify_nonmemory=lambda observation: (
            () if observation is immediate else tuple(failure_fields)
        ),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted.append(
                {
                    "sample_index": index,
                    "elapsed_seconds": elapsed,
                    "identity": f"sample-{index}",
                }
            )
            or persisted[-1]
        ),
        clock=clock,
        sleep=clock.sleep,
    )
    return result, persisted


def test_post_exit_recovery_warning_resets_the_normal_stability_window():
    result, persisted = _run_scripted_memory_recovery(
        pressures=("normal", "normal", "normal", "warning") + ("normal",) * 7
    )

    assert result.outcome == "recovered"
    assert result.duration_seconds == 55.0
    assert len(persisted) == 11


def test_post_exit_recovery_records_the_deadline_sample_without_extending_deadline():
    pressures = tuple(
        "warning" if elapsed % 30 == 25 else "normal" for elapsed in range(5, 301, 5)
    )

    result, persisted = _run_scripted_memory_recovery(pressures=pressures)

    assert result.outcome == "timeout"
    assert result.duration_seconds == 300.0
    assert len(persisted) == 60
    assert persisted[-1]["elapsed_seconds"] == 300.0


def test_post_exit_recovery_shifted_cadence_times_out_without_an_early_final_sample():
    clock = _RecoveryClock()
    clock.now = 6.0
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    collection_starts = []

    def collect(_deadline):
        collection_starts.append(clock())
        return clock(), immediate

    def record_sample(index, elapsed, observation, previous_identity):
        return build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=index,
            previous_sample_identity=previous_identity,
            elapsed_seconds=elapsed,
            **observation,
        )

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(workload),
        collect=collect,
        classify_nonmemory=lambda observation: (),
        record_sample=record_sample,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "timeout"
    assert result.duration_seconds == clock() == 300.0
    assert collection_starts == list(range(6, 300, 5))
    assert [sample["elapsed_seconds"] for sample in result.samples] == collection_starts
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=result.samples,
        policy=post_exit_recovery_policy(workload),
        outcome=result.outcome,
        duration_seconds=result.duration_seconds,
        completion_source="live",
    )
    trial = finalize_raw_trial(measurement, post_exit, result.samples, recovery)
    validate_raw_trial_evidence(trial)


def test_post_exit_recovery_first_wake_overshoot_finalizes_zero_sample_timeout():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"

    def oversleep(_seconds):
        clock.now = 300.001

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: pytest.fail("deadline overshoot must not collect"),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda *args: pytest.fail("deadline overshoot must not persist"),
        clock=clock,
        sleep=oversleep,
    )
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=result.samples,
        policy=post_exit_recovery_policy(build_canonical_workload()),
        outcome=result.outcome,
        duration_seconds=result.duration_seconds,
        completion_source="live",
    )
    trial = finalize_raw_trial(measurement, post_exit, result.samples, recovery)

    assert result == PostExitMemoryRecoveryResult("timeout", 300.0, (), ())
    assert recovery["terminal_sample_identity"] is None
    validate_raw_trial_evidence(trial)


@pytest.mark.parametrize(
    ("terminal_pressure", "failure_fields", "expected"),
    (
        ("critical", (), "critical"),
        ("normal", ("power_connected",), "environment-failure"),
        ("normal", ("thermal_state",), "environment-failure"),
        ("normal", ("hardware",), "environment-failure"),
        ("normal", ("software_versions",), "environment-failure"),
    ),
)
def test_post_exit_recovery_stops_on_terminal_failure(
    terminal_pressure, failure_fields, expected
):
    result, persisted = _run_scripted_memory_recovery(
        pressures=(terminal_pressure,), failure_fields=failure_fields
    )

    assert result.outcome == expected
    assert len(persisted) == 1
    assert result.failure_fields == failure_fields


def test_post_exit_recovery_persists_before_classifying_a_sample():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    observation = _valid_observation("2026-08-09T00:00:05+00:00")
    persisted = []

    def classify_nonmemory(candidate):
        if candidate is immediate:
            return ()
        assert persisted == [observation]
        return ("power_connected",)

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda _deadline: (clock(), observation),
        classify_nonmemory=classify_nonmemory,
        record_sample=lambda index, elapsed, candidate, previous: (
            persisted.append(candidate) or {"identity": "sample-0"}
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "environment-failure"
    assert result.samples == ({"identity": "sample-0"},)


def test_post_exit_recovery_does_not_overlap_a_slow_collector():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    pressures = iter(("normal", "critical"))
    collection_starts = []

    def collect(_deadline):
        sample_started_at = clock()
        collection_starts.append(sample_started_at)
        clock.now += 7.0
        observation = _valid_observation("2026-08-09T00:00:05+00:00")
        observation["environment_status"]["memory_pressure"] = next(pressures)
        return sample_started_at, observation

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=collect,
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: {
            "identity": f"sample-{index}"
        },
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "critical"
    assert collection_starts == [5.0, 12.0]
    assert clock.sleeps == [5.0, 0.0]


def test_post_exit_recovery_uses_memory_probe_start_as_sample_time():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    pressures = iter(("normal", "critical"))
    persisted_elapsed = []

    def collect(_deadline):
        clock.now += 1.0
        observation = _valid_observation("2026-08-09T00:00:05+00:00")
        observation["environment_status"]["memory_pressure"] = next(pressures)
        return clock(), observation

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=collect,
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: (
            persisted_elapsed.append(elapsed) or {"identity": f"sample-{index}"}
        ),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "critical"
    assert persisted_elapsed == [6.0, 12.0]
    assert clock.sleeps == [5.0, 5.0]


@pytest.mark.parametrize(
    ("policy_change", "message"),
    (
        (("sample_interval_seconds", 0.0), "sample interval"),
        (("timeout_seconds", float("inf")), "timeout"),
        (("stability_seconds", -1.0), "stability"),
        (("stability_seconds", 301.0), "must not exceed"),
    ),
)
def test_post_exit_recovery_validates_time_policy(policy_change, message):
    policy = _valid_recovery_policy(build_canonical_workload())
    policy[policy_change[0]] = policy_change[1]

    with pytest.raises(ValueError, match=message):
        wait_for_post_exit_memory_recovery(
            immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
            immediate_started_at=0.0,
            recovery_policy=policy,
            collect=lambda _deadline: pytest.fail("invalid policy must not collect"),
            classify_nonmemory=lambda observation: (),
            record_sample=lambda *args: pytest.fail("invalid policy must not sample"),
            clock=lambda: 0.0,
            sleep=lambda seconds: pytest.fail("invalid policy must not sleep"),
        )


@pytest.mark.parametrize("started_at", (-1.0, float("inf"), float("nan")))
def test_post_exit_recovery_validates_immediate_start_time(started_at):
    with pytest.raises(ValueError, match="immediate_started_at"):
        wait_for_post_exit_memory_recovery(
            immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
            immediate_started_at=started_at,
            recovery_policy=_valid_recovery_policy(build_canonical_workload()),
            collect=lambda _deadline: pytest.fail(
                "invalid start time must not collect"
            ),
            classify_nonmemory=lambda observation: (),
            record_sample=lambda *args: pytest.fail(
                "invalid start time must not sample"
            ),
            clock=lambda: 0.0,
            sleep=lambda seconds: pytest.fail("invalid start time must not sleep"),
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("status", "power_connected", False),
        ("status", "power_connected", 1),
        ("status", "low_power_mode", True),
        ("status", "low_power_mode", 0),
        ("status", "memory_pressure", "warning"),
        ("status", "competing_gpu_workload", True),
        ("status", "competing_gpu_workload", 0),
        ("hardware", "chip", "Apple M4"),
        ("software", "python", "3.12.12"),
    ],
)
def test_thermal_recovery_stops_on_nonthermal_changes(target, field, value):
    clock = _RecoveryClock()
    hardware = {"chip": "Apple M5"}
    status = _recovery_collect(["fair"])()[1]
    software = {"python": "3.12.16"}
    if target == "hardware":
        hardware[field] = value
    elif target == "software":
        software[field] = value
    else:
        status[field] = value

    with pytest.raises(ValueError, match=field):
        wait_for_nominal_thermal_window(
            collect=lambda: (hardware, status, software),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.16"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: None,
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
