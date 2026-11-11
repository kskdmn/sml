from dataclasses import replace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.runner import (
    classify_trial_environment,
    decode_memory_pressure_level,
    decode_thermal_state,
    detect_competing_gpu_workload,
    parse_power_status,
    validate_thermal_observation,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
)

from .support import (
    _recovery_evidence,
    _valid_raw_trial,
    _valid_recovered_raw_trial,
    _with_environment_observations,
    _with_observation_changes,
    _with_trial_payload,
)


@pytest.mark.parametrize(
    ("raw_value", "state"),
    [(0, "nominal"), (1, "fair"), (2, "serious"), (3, "critical")],
)
def test_thermal_state_retains_foundation_raw_value(raw_value, state):
    assert decode_thermal_state(raw_value) == state
    validate_thermal_observation(
        {"thermal_state": state, "thermal_state_raw_value": raw_value}
    )


def test_thermal_observation_rejects_a_mismatched_string_and_raw_value():
    with pytest.raises(ValueError, match="thermal state and raw value disagree"):
        validate_thermal_observation(
            {"thermal_state": "nominal", "thermal_state_raw_value": 1}
        )


def test_thermal_observation_rejects_a_merged_value_that_is_not_the_worst_endpoint():
    nominal = {"thermal_state": "nominal", "thermal_state_raw_value": 0}
    fair = {"thermal_state": "fair", "thermal_state_raw_value": 1}
    with pytest.raises(ValueError, match="merged thermal state"):
        validate_thermal_observation(
            {
                "thermal_state": "nominal",
                "thermal_state_raw_value": 0,
                "start": nominal,
                "end": fair,
            }
        )


def test_gpu_workload_detection_ignores_system_metal_and_unrelated_python():
    process_table = """
       10 1 /System/Library/Frameworks/Metal.framework/MTLCompilerService
       20 1 .venv/bin/python scripts/nego_golden_eval.py --suite nego
       30 1 python -m mlx_lm.server --model local
       40 1 uv run python v2/src/train_sml.py
    """

    assert detect_competing_gpu_workload(process_table, current_pid=30, parent_pid=1)
    without_gpu_jobs = "\n".join(process_table.splitlines()[:3])
    assert not detect_competing_gpu_workload(
        without_gpu_jobs, current_pid=999, parent_pid=998
    )


@pytest.mark.parametrize(
    "command",
    (
        "uv run python -m sml train --config train.json",
        "python -m sml finetune --config swag.json",
        "python -m sml infer model prompt",
        "python -m sml evaluate model",
        "uv run python -m v2.benchmarks.quality --help",
        "python -m v2.benchmarks.swag_quality run",
    ),
)
def test_gpu_workload_detection_catches_v2_commands_but_ignores_own_process(command):
    process_table = f"40 1 {command}"
    assert detect_competing_gpu_workload(process_table, current_pid=30, parent_pid=1)
    assert not detect_competing_gpu_workload(
        process_table, current_pid=40, parent_pid=1
    )


@pytest.mark.parametrize("operation", ("prepare", "verify", "train-tokenizer"))
def test_gpu_workload_detection_allows_cpu_only_v2_commands(operation):
    assert not detect_competing_gpu_workload(
        f"40 1 python -m sml {operation}", current_pid=30, parent_pid=1
    )


def test_native_memory_pressure_levels_fail_closed():
    assert decode_memory_pressure_level(1) == "normal"
    assert decode_memory_pressure_level(2) == "warning"
    assert decode_memory_pressure_level(4) == "critical"
    with pytest.raises(RuntimeError, match="unsupported"):
        decode_memory_pressure_level(0)


def test_power_status_selects_the_active_labeled_pmset_section():
    custom = """Battery Power:
 lowpowermode 1
 sleep 1
AC Power:
 lowpowermode 0
 sleep 0
"""

    assert parse_power_status("Now drawing from 'AC Power'\n", custom) == (
        True,
        "automatic",
        False,
    )
    assert parse_power_status("Now drawing from 'Battery Power'\n", custom) == (
        False,
        "low-power",
        True,
    )


def _raw_trial_for_recovery_signal(workload, signal):
    if signal == "accepted-not-required":
        return _valid_raw_trial(workload)
    if signal == "accepted-recovered":
        return _valid_recovered_raw_trial(workload)
    if signal in {"start-warning", "end-critical"}:
        trial = _valid_raw_trial(workload)
        start = dict(trial.environment_status["start"])
        end = dict(trial.environment_status["end"])
        if signal == "start-warning":
            start["memory_pressure"] = "warning"
        else:
            end["memory_pressure"] = "critical"
        return _with_environment_observations(trial, start=start, end=end)
    if signal == "recovery-timeout":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome="timeout",
        )[4]
    if signal == "recovery-critical":
        return _recovery_evidence(
            workload, immediate_pressure="critical", outcome="critical"
        )[4]
    if signal == "recovery-interrupted":
        return _recovery_evidence(
            workload, immediate_pressure="warning", outcome="interrupted"
        )[4]
    if signal == "recovery-thermal":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=(
                {"thermal_state": "fair", "thermal_state_raw_value": 1},
            ),
            outcome="environment-failure",
            failure_fields=("thermal_state",),
        )[4]
    if signal == "recovery-power":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome="environment-failure",
            failure_fields=("power_connected",),
        )[4]
    raise AssertionError(f"unknown recovery signal: {signal}")


@pytest.mark.parametrize(
    ("signal", "outcome", "reason"),
    (
        ("accepted-not-required", "accept", None),
        ("accepted-recovered", "accept", None),
        ("start-warning", "memory-reject", "non-normal-start-memory-pressure"),
        ("end-critical", "memory-reject", "critical-measurement-memory-pressure"),
        ("recovery-timeout", "memory-reject", "persistent-post-exit-memory-pressure"),
        ("recovery-critical", "memory-reject", "critical-post-exit-memory-pressure"),
        ("recovery-interrupted", "memory-reject", "interrupted-post-exit-recovery"),
        ("recovery-thermal", "thermal-reject", "non-nominal-thermal"),
        (
            "recovery-power",
            "environment-reject",
            "post-exit-recovery-environment-violation",
        ),
    ),
)
def test_trial_disposition_uses_the_complete_recovery_chain(signal, outcome, reason):
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, signal)

    disposition = classify_trial_environment(workload, trial)

    assert (disposition.outcome, disposition.reason) == (outcome, reason)


def test_trial_disposition_is_performance_independent():
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, "recovery-timeout")
    changed = _with_trial_payload(
        trial,
        value=123.0,
        elapsed_seconds=17.0,
        compilation_seconds=0.25,
        peak_memory_bytes=9_999,
    )

    assert classify_trial_environment(workload, changed) == (
        classify_trial_environment(workload, trial)
    )


@pytest.mark.parametrize("endpoint", ("start", "end", "post_exit"))
def test_trial_thermal_disposition_rejects_each_non_nominal_observation(endpoint):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses[endpoint].update(thermal_state="fair", thermal_state_raw_value=1)
    changed = _with_environment_observations(trial, **statuses)

    disposition = classify_trial_environment(workload, changed)

    assert (disposition.outcome, disposition.reason) == (
        "thermal-reject",
        "non-nominal-thermal",
    )


@pytest.mark.parametrize(
    ("endpoint", "change", "message"),
    (
        ("start", {"power_connected": False}, "power_connected"),
        ("end", {"power_mode": "changed"}, "power_mode"),
        ("start", {"competing_gpu_workload": True}, "competing_gpu_workload"),
    ),
)
def test_rejected_environment_override_never_allows_power_policy_failures(
    endpoint, change, message
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses[endpoint].update(change)
    changed = _with_environment_observations(trial, **statuses)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(
            workload, changed, allow_rejected_environment=True
        )


def test_rejected_environment_override_allows_named_immediate_failure_only():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    post_exit = {
        **trial.environment_status["post_exit"],
        "low_power_mode": True,
    }
    changed = _with_environment_observations(trial, post_exit=post_exit)

    with pytest.raises(ValueError, match="low_power_mode"):
        benchmark_runner._validate_acceptance_environment(workload, changed)
    benchmark_runner._validate_acceptance_environment(
        workload, changed, allow_rejected_environment=True
    )


@pytest.mark.parametrize(
    ("signal", "message"),
    (
        ("recovery-power", "power_connected"),
        ("recovery-thermal", "thermal_state"),
    ),
)
def test_rejected_environment_override_validates_nested_recovery_samples(
    signal, message
):
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, signal)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(workload, trial)
    benchmark_runner._validate_acceptance_environment(
        workload, trial, allow_rejected_environment=True
    )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"hardware_changes": {"chip": "unexpected"}}, "required chip"),
        ({"software_changes": {"python": "3.12.12"}}, "Python version"),
        ({"identity_mismatch": True}, "embedded evidence"),
    ),
)
def test_rejected_environment_override_never_allows_proof_failures(change, message):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    if change.get("identity_mismatch"):
        changed = replace(
            trial,
            evidence_session_identity="sha256:" + "e" * 64,
        )
    else:
        changed = _with_observation_changes(trial, **change)

    with pytest.raises(ValueError, match=message):
        benchmark_runner._validate_acceptance_environment(
            workload, changed, allow_rejected_environment=True
        )


def test_rejected_environment_override_allows_only_retryable_dispositions():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses["post_exit"]["memory_pressure"] = "warning"
    changed = _with_environment_observations(trial, **statuses)

    with pytest.raises(ValueError, match="interrupted-post-exit-recovery"):
        benchmark_runner._validate_acceptance_environment(workload, changed)
    benchmark_runner._validate_acceptance_environment(
        workload, changed, allow_rejected_environment=True
    )
