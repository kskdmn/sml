import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import v2.benchmarks.journal as baseline_journal
import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.adapters import runtime
from v2.benchmarks.evidence import (
    build_child_trial_measurement,
    finalize_raw_trial,
    validate_child_trial_measurement,
    validate_raw_trial_evidence,
)
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    atomic_write_json,
    read_json_object,
)
from v2.benchmarks.runner import (
    capture_baseline_trials,
    classify_trial_environment,
)
from v2.benchmarks.schema import METRIC_NAMES, RawTrial
from v2.benchmarks.workload import (
    build_canonical_workload,
)

from .support import (
    _baseline_validator,
    _preflight_from_trial,
    _recovery_evidence,
    _RecoveryClock,
    _session_document,
    _valid_child_measurement,
    _valid_observation,
    _valid_post_exit_observation,
    _valid_post_exit_recovery,
    _valid_raw_trial,
    _valid_trial_payload,
    _with_environment_observations,
)


def _single_process_arguments(tmp_path):
    return SimpleNamespace(
        harness_root=tmp_path / "harness",
        source_root=tmp_path / "source",
        source_commit="c" * 40,
        harness_commit="a" * 40,
        harness_identity="sha256:" + "b" * 64,
        metric="prepared-data",
        side="reference",
        attempt_index=0,
        pair_index=0,
        process_order=0,
        warmup=5,
        measure=100,
        prepared_data_measure=100,
        comparison_target="baseline",
        evidence_session_identity="sha256:" + "9" * 64,
        journal_attempt_index=0,
        measurement_output=(
            tmp_path / "state" / "measurements" / "prepared-data" / "0" / "0.json"
        ),
    )


def _launch_trial_arguments(tmp_path):
    return {
        "harness_root": tmp_path / "harness",
        "source_root": tmp_path / "source",
        "source_commit": "c" * 40,
        "harness_commit": "a" * 40,
        "harness_identity": "sha256:" + "b" * 64,
        "metric": "prepared-data",
        "side": "reference",
        "attempt_index": 0,
        "pair_index": 0,
        "order": 0,
        "warmup": 5,
        "measure": 100,
        "prepared_data_measure": 100,
        "comparison_target": "baseline",
    }


def _stub_single_process_measurement(monkeypatch, args, captured=None):
    args.harness_root.mkdir()
    args.source_root.mkdir()
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload, metric=args.metric)
    status = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 60,
        "competing_gpu_workload": False,
    }
    native = SimpleNamespace(
        native_configuration=trial.native_configuration,
        native_representation_identity=trial.native_representation_identity,
        canonical_row_identity=trial.canonical_row_identity,
        canonical_input_identity=trial.canonical_input_identity,
        canonical_projection=trial.canonical_projection,
        execution_order_identity=trial.execution_order_identity,
        initial_parameter_identity=trial.initial_parameter_identity,
        startup_verification_seconds=trial.startup_verification_seconds,
    )
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: path.resolve())
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda path, *, label: None
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_git_commit",
        lambda path: (
            args.harness_commit
            if path.resolve() == args.harness_root.resolve()
            else args.source_commit
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "harness_content_identity",
        lambda path: args.harness_identity,
    )
    monkeypatch.setattr(runtime, "resolve_native_workload", lambda *unused: native)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda: (trial.hardware, status, trial.software_versions),
    )

    def measure(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return benchmark_runner.ProcessMeasurement(
            elapsed_seconds=trial.elapsed_seconds,
            value=trial.value,
            work_count=100.0,
            compilation_seconds=trial.compilation_seconds,
            peak_memory_bytes=trial.peak_memory_bytes,
        )

    monkeypatch.setattr(benchmark_runner, "measure_native_process", measure)


@pytest.mark.parametrize(
    ("metric", "expected_warmup", "expected_measured"),
    [
        ("prepared-data", 5, 100),
        ("inference-prefill", 5, 32),
        ("compile-cold-start", 0, 1),
        ("peak-metal-memory", 5, 1),
    ],
)
def test_child_process_forwards_each_canonical_metric_count(
    tmp_path, monkeypatch, metric, expected_warmup, expected_measured
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = expected_measured
    args.measurement_output = (
        tmp_path / "state" / "measurements" / metric / "0" / "0.json"
    )
    captured = {}
    _stub_single_process_measurement(monkeypatch, args, captured)

    benchmark_runner._run_single_process(args)

    assert captured["warmup_units"] == expected_warmup
    assert captured["measured_units"] == expected_measured
    measurement = validate_child_trial_measurement(
        read_json_object(args.measurement_output, label="child measurement")
    )
    assert measurement["trial"]["warmup_units"] == expected_warmup
    assert measurement["trial"]["measured_units"] == expected_measured


@pytest.mark.parametrize(
    ("metric", "supplied"),
    (("prepared-data", 20), ("prepared-data", 99), ("pretraining-compute", 100)),
)
def test_child_rejects_noncanonical_count_before_adapter_import(
    tmp_path, monkeypatch, metric, supplied
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = supplied
    resolved = []
    _stub_single_process_measurement(monkeypatch, args)
    monkeypatch.setattr(
        runtime, "resolve_native_workload", lambda *args: resolved.append(args)
    )

    with pytest.raises(ValueError, match="child measured units do not match"):
        benchmark_runner._run_single_process(args)

    assert resolved == []


def test_completed_child_output_never_overwrites_an_existing_attempt(
    tmp_path, monkeypatch
):
    args = _single_process_arguments(tmp_path)
    _stub_single_process_measurement(monkeypatch, args)
    args.measurement_output.parent.mkdir(parents=True)
    args.measurement_output.write_bytes(b"existing immutable measurement\n")

    with pytest.raises(FileExistsError):
        benchmark_runner._run_single_process(args)

    assert args.measurement_output.read_bytes() == b"existing immutable measurement\n"


def test_completed_child_output_crosses_the_parent_fsync_boundary(
    tmp_path, monkeypatch
):
    args = _single_process_arguments(tmp_path)
    _stub_single_process_measurement(monkeypatch, args)
    synced_directories = []
    monkeypatch.setattr(
        baseline_journal,
        "_fsync_directory",
        lambda path: synced_directories.append(path.resolve()),
    )

    benchmark_runner._run_single_process(args)

    assert args.measurement_output.parent.resolve() in synced_directories
    assert (
        validate_child_trial_measurement(
            read_json_object(
                args.measurement_output, label="completed child measurement"
            )
        )["trial"]["attempt_index"]
        == 0
    )


def _observation_with_memory(pressure, free_percentage):
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    observation["environment_status"].update(
        memory_pressure=pressure,
        memory_free_percentage=free_percentage,
    )
    return observation


def test_collect_post_exit_environment_records_monotonic_time_before_memory_probe(
    monkeypatch,
):
    events = []
    observation = _valid_observation("2026-08-09T00:00:03+00:00")

    monkeypatch.setattr(
        benchmark_runner,
        "_memory_pressure",
        lambda: events.append("memory") or ("normal", 67),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_utc_now_iso",
        lambda: events.append("timestamp") or observation["observed_at_utc"],
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda *, memory_sample: (
            events.append(("environment", memory_sample))
            or (
                observation["hardware"],
                observation["environment_status"],
                observation["software_versions"],
            )
        ),
    )

    started_at, collected = benchmark_runner.collect_post_exit_environment(
        clock=lambda: events.append("clock") or 42.0,
        deadline=42.0,
    )

    assert started_at == 42.0
    assert events == ["clock", "memory", "timestamp", ("environment", ("normal", 67))]
    assert collected == observation


def test_recovery_failure_fields_ignores_memory_and_reports_identity_and_environment_drift():
    workload = build_canonical_workload()
    observation = _observation_with_memory("warning", 1)
    observation["hardware"] = {**observation["hardware"], "chip": "Other"}
    observation["software_versions"] = {
        **observation["software_versions"],
        "python": "3.12.12",
    }
    observation["environment_status"]["power_connected"] = False
    observation["environment_status"]["thermal_state"] = "fair"
    observation["environment_status"]["thermal_state_raw_value"] = 1

    assert benchmark_runner._recovery_failure_fields(
        observation,
        expected_hardware=_valid_observation("2026-08-09T00:00:00+00:00")["hardware"],
        expected_software_versions=_valid_observation("2026-08-09T00:00:00+00:00")[
            "software_versions"
        ],
        required_environment=workload.required_environment,
    ) == ("hardware", "power_connected", "software_versions", "thermal_state")


def test_recovery_failure_fields_rejects_malformed_thermal_evidence():
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    observation["environment_status"]["thermal_state"] = "fair"

    with pytest.raises(ValueError, match="thermal state and raw value disagree"):
        benchmark_runner._recovery_failure_fields(
            observation,
            expected_hardware=observation["hardware"],
            expected_software_versions=observation["software_versions"],
            required_environment=build_canonical_workload().required_environment,
        )


def test_parent_records_immediate_warning_then_every_recovery_sample_before_finalizing(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    clock = _RecoveryClock()
    events = []
    observations = [
        _observation_with_memory("warning", 36),
        *[_observation_with_memory("normal", 67) for _ in range(7)],
    ]

    def run_child(command, *, cwd, check):
        events.append("child-exited")
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (
            events.append("memory-first") or clock(),
            observations.pop(0),
        ),
    )
    original_atomic_write_json = benchmark_runner.atomic_write_json

    def record_atomic_write(path, value, *, create_only):
        events.append(f"write:{path.name}")
        return original_atomic_write_json(path, value, create_only=create_only)

    monkeypatch.setattr(benchmark_runner, "atomic_write_json", record_atomic_write)

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=measurement_path,
        post_exit_output=post_exit_path,
        recovery_samples_directory=recovery_samples_directory,
        recovery_output=recovery_path,
        output=trial_path,
        clock=clock,
        sleep=clock.sleep,
    )

    assert events[:2] == ["child-exited", "memory-first"]
    assert trial.schema_version == 3
    assert (
        trial.post_exit_observation["environment_status"]["memory_pressure"]
        == "warning"
    )
    assert trial.post_exit_recovery["outcome"] == "recovered"
    assert len(trial.post_exit_recovery_samples) == 7
    assert sorted(path.name for path in recovery_samples_directory.iterdir()) == [
        f"{index}.json" for index in range(7)
    ]
    assert recovery_path.is_file()
    assert events.index("write:post-exit.json") < events.index("write:0.json")
    assert events.index("write:6.json") < events.index("write:recovery.json")
    assert events.index("write:recovery.json") < events.index("write:trial.json")
    assert trial == RawTrial.from_dict(read_json_object(trial_path, label="trial"))


def test_parent_finalizes_and_rejects_timeout_when_deadline_sleep_overshoots(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(
        workload,
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    immediate = _observation_with_memory("warning", 36)

    class OversleepingDeadlineClock(_RecoveryClock):
        def sleep(self, seconds):
            super().sleep(seconds)
            if self.now == 300.0:
                self.now += 0.001

    clock = OversleepingDeadlineClock()
    collection_starts = []

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            journal.measurement_path(slot, 0), measurement, create_only=True
        ),
    )

    def collect_post_exit_environment(**kwargs):
        collection_starts.append(clock())
        return clock(), immediate

    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        collect_post_exit_environment,
    )

    def launch(current_slot, attempt):
        assert current_slot == slot
        return benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=journal.session["identity"],
            journal_attempt_index=attempt.journal_attempt_index,
            measurement_output=journal.measurement_path(slot, 0),
            post_exit_output=journal.post_exit_path(slot, 0),
            recovery_samples_directory=journal.recovery_samples_path(slot, 0),
            recovery_output=journal.recovery_path(slot, 0),
            output=attempt.path,
            clock=clock,
            sleep=clock.sleep,
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="persistent-post-exit-memory-pressure",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=launch,
            preflight=lambda: _preflight_from_trial(_valid_raw_trial(workload)),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
            clock=clock,
        )

    recovery = read_json_object(journal.recovery_path(slot, 0), label="recovery")
    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejection")
    assert recovery["outcome"] == "timeout"
    assert recovery["duration_seconds"] == 300.0
    assert recovery["terminal_environment_status"] == immediate["environment_status"]
    assert collection_starts == [float(value) for value in range(0, 300, 5)]
    assert rejected["reason"] == "persistent-post-exit-memory-pressure"
    assert rejected["trial"]["post_exit_recovery"] == recovery


def test_parent_aborts_recovery_collection_when_probe_start_crosses_deadline(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(
        workload,
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    immediate = _observation_with_memory("warning", 36)

    class CollectorRaceClock(_RecoveryClock):
        def __init__(self):
            super().__init__()
            self.deadline_reads = 0

        def __call__(self):
            if self.now == 300.0:
                self.deadline_reads += 1
                if self.deadline_reads == 2:
                    self.now += 0.001
            return self.now

        def sleep(self, seconds):
            self.sleeps.append(seconds)
            self.now = 300.0
            self.deadline_reads = 0

    clock = CollectorRaceClock()
    memory_probe_calls = []
    environment_collection_calls = []

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            journal.measurement_path(slot, 0), measurement, create_only=True
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_memory_pressure",
        lambda: memory_probe_calls.append(True) or ("warning", 36),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_utc_now_iso",
        lambda: immediate["observed_at_utc"],
    )

    def collect_environment(*, memory_sample):
        environment_collection_calls.append(memory_sample)
        return (
            immediate["hardware"],
            immediate["environment_status"],
            immediate["software_versions"],
        )

    monkeypatch.setattr(benchmark_runner, "collect_environment", collect_environment)

    def launch(current_slot, attempt):
        assert current_slot == slot
        return benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=journal.session["identity"],
            journal_attempt_index=attempt.journal_attempt_index,
            measurement_output=journal.measurement_path(slot, 0),
            post_exit_output=journal.post_exit_path(slot, 0),
            recovery_samples_directory=journal.recovery_samples_path(slot, 0),
            recovery_output=journal.recovery_path(slot, 0),
            output=attempt.path,
            clock=clock,
            sleep=clock.sleep,
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="persistent-post-exit-memory-pressure",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=launch,
            preflight=lambda: _preflight_from_trial(_valid_raw_trial(workload)),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
            clock=clock,
        )

    recovery = read_json_object(journal.recovery_path(slot, 0), label="recovery")
    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejection")
    assert memory_probe_calls == [True]
    assert environment_collection_calls == [("warning", 36)]
    assert not journal.recovery_samples_path(slot, 0).exists()
    assert recovery["outcome"] == "timeout"
    assert recovery["duration_seconds"] == 300.0
    assert recovery["completion_source"] == "live"
    assert rejected["reason"] == "persistent-post-exit-memory-pressure"
    assert rejected["trial"]["post_exit_recovery"] == recovery
    validate_raw_trial_evidence(RawTrial.from_dict(rejected["trial"]))


def test_parent_finalizes_immediate_hardware_drift_without_recovery_samples(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    immediate = _observation_with_memory("normal", 67)
    immediate["hardware"] = {**immediate["hardware"], "chip": "Other"}

    def run_child(command, *, cwd, check):
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (0.0, immediate),
    )

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=measurement_path,
        post_exit_output=post_exit_path,
        recovery_samples_directory=recovery_samples_directory,
        recovery_output=recovery_path,
        output=trial_path,
        clock=lambda: 0.0,
        sleep=lambda seconds: pytest.fail("immediate failure must not sleep"),
    )

    assert trial.schema_version == 3
    assert trial.post_exit_recovery_samples == ()
    assert trial.post_exit_recovery["outcome"] == "environment-failure"
    assert trial.post_exit_recovery["duration_seconds"] == 0.0
    assert trial.post_exit_recovery["failure_fields"] == ["hardware"]
    assert not recovery_samples_directory.exists()
    validate_raw_trial_evidence(trial)


def _stub_parent_observation(monkeypatch, observation):
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (0.0, observation),
    )


def test_parent_rejects_wrong_child_count_before_post_exit_publication(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = build_child_trial_measurement(
        session_identity="sha256:" + "9" * 64,
        journal_attempt_index=0,
        trial={**_valid_trial_payload(workload), "measured_units": 20},
        start=_valid_observation("2026-08-08T00:00:00+00:00"),
        end=_valid_observation("2026-08-08T00:00:01+00:00"),
    )
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, **unused: atomic_write_json(
            measurement_path, measurement, create_only=True
        ),
    )

    with pytest.raises(ValueError, match="child trial does not match"):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=tmp_path / "recovery-samples",
            recovery_output=tmp_path / "recovery.json",
            output=tmp_path / "trial.json",
        )

    assert not post_exit_path.exists()


@pytest.mark.parametrize("existing_stage", ["post-exit", "sample", "recovery", "trial"])
def test_parent_preserves_existing_evidence_and_stops_publication(
    tmp_path, monkeypatch, existing_stage
):
    measurement = _valid_child_measurement(build_canonical_workload())
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"
    existing_path = {
        "post-exit": post_exit_path,
        "sample": samples_directory / "0.json",
        "recovery": recovery_path,
        "trial": trial_path,
    }[existing_stage]
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"existing immutable evidence\n")
    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: atomic_write_json(
            measurement_path, measurement, create_only=True
        ),
    )
    observation = (
        _observation_with_memory("warning", 36)
        if existing_stage == "sample"
        else _valid_observation("2026-08-08T00:00:02+00:00")
    )
    _stub_parent_observation(monkeypatch, observation)

    with pytest.raises(FileExistsError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity=measurement["session_identity"],
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
            clock=_RecoveryClock(),
            sleep=lambda seconds: None,
        )

    assert existing_path.read_bytes() == b"existing immutable evidence\n"
    assert post_exit_path.is_file()
    assert samples_directory.exists() == (existing_stage == "sample")
    assert recovery_path.exists() == (existing_stage in {"recovery", "trial"})
    assert trial_path.exists() == (existing_stage == "trial")


def test_nonzero_child_exit_does_not_publish_parent_evidence(tmp_path, monkeypatch):
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    recovery_samples_directory = tmp_path / "recovery-samples"
    recovery_path = tmp_path / "recovery.json"
    trial_path = tmp_path / "trial.json"

    def fail_child(command, *, cwd, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", fail_child)

    with pytest.raises(subprocess.CalledProcessError):
        benchmark_runner._launch_trial(
            **_launch_trial_arguments(tmp_path),
            evidence_session_identity="sha256:" + "9" * 64,
            journal_attempt_index=0,
            measurement_output=measurement_path,
            post_exit_output=post_exit_path,
            recovery_samples_directory=recovery_samples_directory,
            recovery_output=recovery_path,
            output=trial_path,
        )

    assert not measurement_path.exists()
    assert not post_exit_path.exists()
    assert not recovery_samples_directory.exists()
    assert not recovery_path.exists()
    assert not trial_path.exists()


def test_paired_trials_stops_before_the_next_process_on_interrupted_memory(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    warning = dict(trial.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    rejected = _with_environment_observations(trial, post_exit=warning)
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        return rejected

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)

    with pytest.raises(ValueError, match="interrupted-post-exit-recovery"):
        benchmark_runner._run_paired_trials(
            harness_root=tmp_path / "harness",
            reference_root=tmp_path / "reference",
            candidate_root=tmp_path / "candidate",
            reference_commit=rejected.source_commit,
            candidate_commit="c" * 40,
            harness_commit=rejected.harness_commit,
            harness_identity=rejected.harness_identity,
            metrics=("prepared-data",),
            pairs=2,
            warmup=5,
            measure=20,
            comparison_target="baseline",
            attempt_index=0,
            output_directory=tmp_path,
        )

    assert len(launches) == 1


@pytest.mark.parametrize("outcome", ("timeout", "critical", "environment-failure"))
def test_paired_trials_rejects_every_terminal_recovery_outcome_before_next_launch(
    tmp_path, monkeypatch, outcome
):
    workload = build_canonical_workload()
    if outcome == "timeout":
        trial = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome=outcome,
        )[4]
    elif outcome == "critical":
        trial = _recovery_evidence(
            workload,
            immediate_pressure="critical",
            outcome=outcome,
        )[4]
    else:
        trial = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome=outcome,
            failure_fields=("power_connected",),
        )[4]
    launches = []
    validated = []
    original_validate = benchmark_runner._validate_acceptance_environment

    def launch(**kwargs):
        launches.append(kwargs)
        return trial

    def validate_at_boundary(current_workload, current_trial):
        validated.append(current_trial)
        original_validate(current_workload, current_trial)

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)
    monkeypatch.setattr(
        benchmark_runner, "_validate_acceptance_environment", validate_at_boundary
    )

    with pytest.raises(ValueError, match=r"raw .*environment"):
        benchmark_runner._run_paired_trials(
            harness_root=tmp_path / "harness",
            reference_root=tmp_path / "reference",
            candidate_root=tmp_path / "candidate",
            reference_commit=trial.source_commit,
            candidate_commit="c" * 40,
            harness_commit=trial.harness_commit,
            harness_identity=trial.harness_identity,
            metrics=("prepared-data",),
            pairs=2,
            warmup=5,
            measure=20,
            comparison_target="baseline",
            attempt_index=0,
            output_directory=tmp_path,
        )

    assert len(launches) == 1
    assert validated == [trial]
    assert validated[0].schema_version == 3
    assert validated[0].post_exit_recovery["outcome"] == outcome


def test_paired_trials_share_one_identity_bound_evidence_session(tmp_path, monkeypatch):
    workload = build_canonical_workload()
    launches = []

    def launch(**kwargs):
        launches.append(kwargs)
        measurement = _valid_child_measurement(
            workload,
            session_identity=kwargs["evidence_session_identity"],
            journal_attempt_index=kwargs["journal_attempt_index"],
        )
        post_exit = _valid_post_exit_observation(measurement)
        recovery = _valid_post_exit_recovery(measurement, post_exit)
        return finalize_raw_trial(measurement, post_exit, (), recovery)

    monkeypatch.setattr(benchmark_runner, "_launch_trial", launch)

    trials = benchmark_runner._run_paired_trials(
        harness_root=tmp_path / "harness",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        reference_commit="1" * 40,
        candidate_commit="2" * 40,
        harness_commit="3" * 40,
        harness_identity="sha256:" + "4" * 64,
        metrics=("prepared-data",),
        pairs=1,
        warmup=5,
        measure=20,
        comparison_target="baseline:sha256:" + "5" * 64,
        attempt_index=1,
        output_directory=tmp_path,
    )

    assert len(trials) == 2
    assert {launch["evidence_session_identity"] for launch in launches} == {
        launches[0]["evidence_session_identity"]
    }
    assert {launch["journal_attempt_index"] for launch in launches} == {1}
    assert {
        tuple(launch["measurement_output"].suffixes[-2:]) for launch in launches
    } == {(".measurement", ".json")}
    assert {tuple(launch["post_exit_output"].suffixes[-2:]) for launch in launches} == {
        (".post-exit", ".json")
    }
    assert {tuple(launch["recovery_output"].suffixes[-2:]) for launch in launches} == {
        (".recovery", ".json")
    }
    assert len({launch["recovery_output"] for launch in launches}) == 2
    assert len({launch["recovery_samples_directory"] for launch in launches}) == 2
    assert all(
        path.name.endswith(".recovery-samples")
        for path in (launch["recovery_samples_directory"] for launch in launches)
    )
    assert {tuple(launch["output"].suffixes[-2:]) for launch in launches} == {
        (".trial", ".json")
    }


def test_parent_launches_each_metric_with_its_canonical_count(tmp_path, monkeypatch):
    workload = build_canonical_workload()
    launches = []
    monkeypatch.setattr(
        benchmark_runner,
        "_launch_trial",
        lambda **kwargs: (
            launches.append(kwargs)
            or _valid_raw_trial(workload, metric=kwargs["metric"])
        ),
    )
    monkeypatch.setattr(
        benchmark_runner, "_validate_acceptance_environment", lambda *_: None
    )
    monkeypatch.setattr(
        benchmark_runner, "_validate_software_versions", lambda *_: None
    )

    benchmark_runner._run_paired_trials(
        harness_root=tmp_path / "harness",
        reference_root=tmp_path / "reference",
        candidate_root=tmp_path / "candidate",
        reference_commit="1" * 40,
        candidate_commit="2" * 40,
        harness_commit="3" * 40,
        harness_identity="sha256:" + "4" * 64,
        metrics=METRIC_NAMES,
        pairs=1,
        warmup=5,
        measure=20,
        prepared_data_measure=100,
        comparison_target="baseline:sha256:" + "5" * 64,
        attempt_index=0,
        output_directory=tmp_path,
    )

    assert {launch["metric"]: launch["measure"] for launch in launches} == {
        unit.metric: unit.measured_units for unit in workload.work_units
    }


def test_baseline_launch_routes_recovery_paths_when_journal_supports_them(
    tmp_path, monkeypatch
):
    slot = BaselineSlot("prepared-data", 0)
    attempt = SimpleNamespace(journal_attempt_index=0, path=tmp_path / "trial.json")
    journal = SimpleNamespace(
        session={"identity": "sha256:" + "9" * 64},
        measurement_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.measurement.json"
        ),
        post_exit_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.post-exit.json"
        ),
        recovery_samples_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.recovery-samples"
        ),
        recovery_path=lambda current_slot, index: (
            tmp_path / f"{current_slot.metric}-{index}.recovery.json"
        ),
    )
    captured = {}
    observation = _valid_observation("2026-08-09T00:00:00+00:00")
    arguments = SimpleNamespace(
        source_commit="c" * 40,
        metrics=METRIC_NAMES,
        pairs=benchmark_runner.SCREEN_PAIRS,
        warmup=benchmark_runner.WARMUP_UNITS,
        measure=benchmark_runner.DEFAULT_MEASURED_UNITS,
        prepared_data_measure=benchmark_runner.PREPARED_DATA_MEASURED_UNITS,
    )

    class FakeBaselineJournal:
        @staticmethod
        def open(*unused):
            return journal

        @staticmethod
        def expected_slots(*unused):
            return (slot,)

    @contextmanager
    def managed_worktree(*unused):
        yield tmp_path / "source"

    monkeypatch.setattr(benchmark_runner, "BaselineJournal", FakeBaselineJournal)
    monkeypatch.setattr(benchmark_runner, "_run_command", lambda *unused, **kwargs: "")
    monkeypatch.setattr(
        benchmark_runner, "_git_commit", lambda *unused: arguments.source_commit
    )
    monkeypatch.setattr(
        benchmark_runner, "harness_content_identity", lambda path: "harness"
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda: (
            observation["hardware"],
            observation["environment_status"],
            observation["software_versions"],
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "build_session_document",
        lambda **unused: journal.session,
    )
    monkeypatch.setattr(
        benchmark_runner, "_managed_detached_worktree", managed_worktree
    )
    monkeypatch.setattr(
        benchmark_runner,
        "capture_baseline_trials",
        lambda **kwargs: (kwargs["launch_trial"](slot, attempt),),
    )
    monkeypatch.setattr(
        benchmark_runner, "_launch_trial", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        benchmark_runner, "publish_baseline_from_journal", lambda **kwargs: {}
    )

    assert (
        benchmark_runner._record_baseline_locked(
            arguments,
            harness_root=tmp_path / "harness",
            state_root=tmp_path / "state",
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=tmp_path / "baseline.jsonl",
        )
        == 0
    )
    assert captured["recovery_samples_directory"] == (
        tmp_path / "prepared-data-0.recovery-samples"
    )
    assert captured["recovery_output"] == tmp_path / "prepared-data-0.recovery.json"
