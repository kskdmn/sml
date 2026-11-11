import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.evidence import (
    build_post_exit_recovery,
    finalize_raw_trial,
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
from v2.benchmarks.workload import (
    build_canonical_workload,
    post_exit_recovery_policy,
)

from .support import (
    _accept_trial,
    _baseline_validator,
    _journal_trial_evidence,
    _persist_journal_trial,
    _preflight_from_trial,
    _recovery_evidence,
    _session_document,
    _valid_raw_trial,
    _valid_recovered_raw_trial,
    _with_environment_observations,
    _with_thermal_state,
)


def test_interrupted_memory_rejection_stops_then_manual_resume_fills_only_missing_slot(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    first_attempt = journal.next_attempt(first)
    accepted = _persist_journal_trial(
        journal,
        first_attempt,
        _valid_raw_trial(workload, pair_index=0),
    )
    journal.accept_inflight(first_attempt, accepted)
    normal = _valid_raw_trial(workload, pair_index=1)
    warning = dict(normal.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    rejected = _with_environment_observations(normal, post_exit=warning)
    first_run_launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda slot, attempt: (
                first_run_launches.append(slot)
                or _persist_journal_trial(journal, attempt, rejected)
            ),
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "memory rejection entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert first_run_launches == [second]
    assert journal.load_accepted((first, second)) == {first: accepted}
    assert (
        read_json_object(journal.rejected_path(second, 0), label="memory rejection")[
            "reason"
        ]
        == "interrupted-post-exit-recovery"
    )

    second_run_launches = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=(first, second),
        launch_trial=lambda slot, attempt: (
            second_run_launches.append(slot)
            or _persist_journal_trial(journal, attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "memory rejection created a thermal trigger"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert second_run_launches == [second]
    assert trials[0] == accepted
    assert (trials[1].metric, trials[1].pair_index, trials[1].value) == (
        normal.metric,
        normal.pair_index,
        normal.value,
    )


@pytest.mark.parametrize(
    ("endpoint", "pressure", "reason"),
    (
        ("start", "warning", "non-normal-start-memory-pressure"),
        ("end", "critical", "critical-measurement-memory-pressure"),
    ),
)
def test_capture_stops_immediately_for_rejected_child_memory_observations(
    tmp_path, endpoint, pressure, reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    status = dict(normal.environment_status[endpoint])
    status["memory_pressure"] = pressure
    rejected = _with_environment_observations(normal, **{endpoint: status})
    launches = []

    with pytest.raises(benchmark_runner.MemoryPressureTrialRejected) as raised:
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: (
                launches.append(current_slot)
                or _persist_journal_trial(journal, attempt, rejected)
            ),
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail(
                "child memory rejection entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert raised.value.slot == slot
    assert raised.value.reason == reason
    assert launches == [slot]
    assert (
        read_json_object(journal.rejected_path(slot, 0), label="memory rejection")[
            "reason"
        ]
        == reason
    )


def test_capture_accepts_child_end_warning_when_post_exit_is_normal(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    warning = dict(normal.environment_status["end"])
    warning["memory_pressure"] = "warning"
    diagnostic_warning = _with_environment_observations(normal, end=warning)
    launches = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            launches.append(current_slot)
            or _persist_journal_trial(journal, attempt, diagnostic_warning)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail("accepted memory warning entered recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [slot]
    assert trials[0].environment_status["measurement_memory_pressure"] == "warning"
    assert journal.load_accepted((slot,)) == {slot: trials[0]}


def test_capture_rejects_measurement_only_crash_then_launches_a_new_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _persisted = _journal_trial_evidence(
        journal, attempt, normal
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    events = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: (
            events.append(("launch", current_attempt.journal_attempt_index))
            or _persist_journal_trial(journal, current_attempt, normal)
        ),
        preflight=lambda: (
            events.append(("preflight", 1)) or _preflight_from_trial(normal)
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail("missing evidence entered thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert events == [("preflight", 1), ("launch", 1)]
    assert (
        read_json_object(
            journal.rejected_path(slot, 0), label="missing evidence rejection"
        )["reason"]
        == "missing-immediate-post-exit-evidence"
    )
    assert trials[0].pair_index == 0

    resumed = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "accepted slot was relaunched after missing-evidence rejection"
        ),
        preflight=lambda: pytest.fail(
            "accepted slot reached preflight after missing-evidence rejection"
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail(
            "missing-evidence rejection created a thermal trigger on resume"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert resumed == trials


def test_capture_never_creates_raw_evidence_for_an_unstaged_launcher_result(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    normal = _valid_raw_trial(workload)

    with pytest.raises(FileNotFoundError):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, current_attempt: normal,
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail("unstaged launch entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert not journal.inflight_path(slot, 0).exists()


def test_capture_reconstructs_measurement_and_post_exit_before_replacement_launch(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, expected = _journal_trial_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    expected_recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=(),
        policy=post_exit_recovery_policy(workload),
        outcome="not-required",
        duration_seconds=0.0,
        completion_source="crash-reconstruction",
    )
    expected = finalize_raw_trial(measurement, post_exit, (), expected_recovery)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "complete staged evidence launched a replacement"
        ),
        preflight=lambda: pytest.fail(
            "complete staged evidence reached replacement preflight"
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda *args: pytest.fail(
            "complete staged evidence entered thermal recovery"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert trials == (expected,)
    assert not attempt.path.exists()
    assert journal.load_accepted((slot,)) == {slot: expected}


def test_capture_reconstructs_not_required_summary_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, expected = _journal_trial_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    expected_recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=(),
        policy=post_exit_recovery_policy(workload),
        outcome="not-required",
        duration_seconds=0.0,
        completion_source="crash-reconstruction",
    )
    expected = finalize_raw_trial(measurement, post_exit, (), expected_recovery)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    launches = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
        preflight=lambda: _preflight_from_trial(expected),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "normal pending evidence entered thermal recovery"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == []
    assert trials == (expected,)
    assert journal.recovery_path(slot, 0).is_file()
    assert (
        read_json_object(journal.recovery_path(slot, 0), label="recovery")["outcome"]
        == "not-required"
    )


def test_capture_rejects_partial_warning_recovery_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    interrupted = _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",),
        sample_elapsed=(5.0,),
        outcome="interrupted",
    )[4]
    measurement, post_exit, samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, interrupted
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "interrupted memory recovery entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert launches == []
    assert (
        read_json_object(journal.recovery_path(slot, 0), label="recovery")["outcome"]
        == "interrupted"
    )


@pytest.mark.parametrize(
    "boundary",
    ("critical", "environment-failure", "stable-completion", "deadline"),
)
def test_capture_reconstructs_decisive_warning_crash_as_interrupted(tmp_path, boundary):
    workload = build_canonical_workload()
    if boundary == "critical":
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("critical",),
            sample_elapsed=(5.0,),
            outcome="critical",
        )[4]
    elif boundary == "environment-failure":
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("critical",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome="environment-failure",
            failure_fields=("power_connected",),
        )[4]
    elif boundary == "stable-completion":
        terminal = _valid_recovered_raw_trial(workload)
    else:
        terminal = _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome="timeout",
        )[4]

    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _summary, rebound = _journal_trial_evidence(
        journal, attempt, terminal
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    for index, sample in enumerate(samples):
        atomic_write_json(
            journal.recovery_sample_path(slot, 0, index), sample, create_only=True
        )

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda *args: pytest.fail(
                "crash reconstruction launched a replacement"
            ),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda *args: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    recovery_path = journal.recovery_path(slot, 0)
    rejection_path = journal.rejected_path(slot, 0)
    recovery = read_json_object(recovery_path, label="recovery")
    rejected = read_json_object(rejection_path, label="rejection")
    assert (
        recovery["version"],
        recovery["completion_source"],
        recovery["outcome"],
    ) == (
        2,
        "crash-reconstruction",
        "interrupted",
    )
    assert rejected["reason"] == "interrupted-post-exit-recovery"
    retained_paths = tuple(
        [journal.recovery_sample_path(slot, 0, index) for index in range(len(samples))]
        + [recovery_path, rejection_path]
    )
    retained = tuple(path.read_bytes() for path in retained_paths)
    launches = []
    normal = _valid_raw_trial(workload)

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, next_attempt: (
            launches.append(next_attempt.journal_attempt_index)
            or _persist_journal_trial(journal, next_attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda *args: None,
        recover=lambda *args: pytest.fail("memory rejection created thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [1]
    assert tuple(path.read_bytes() for path in retained_paths) == retained


@pytest.mark.parametrize(
    ("trial", "exception_name", "reason"),
    (
        (
            lambda workload: _recovery_evidence(
                workload, immediate_pressure="critical", outcome="critical"
            )[4],
            "MemoryPressureTrialRejected",
            "critical-post-exit-memory-pressure",
        ),
        (
            lambda workload: _with_environment_observations(
                _valid_raw_trial(workload),
                post_exit={
                    **_valid_raw_trial(workload).environment_status["post_exit"],
                    "power_connected": False,
                },
            ),
            "EnvironmentTrialRejected",
            "post-exit-recovery-environment-violation",
        ),
    ),
)
def test_capture_reconstructs_terminal_immediate_failure_before_any_launch(
    tmp_path, trial, exception_name, reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    rejected = trial(workload)
    measurement, post_exit, _samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, rejected
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    launches = []

    exception = getattr(benchmark_runner, exception_name)
    with pytest.raises(exception, match=reason):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda *args: pytest.fail(
                "terminal immediate failure entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert launches == []
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery")[
        "outcome"
    ] in {"critical", "environment-failure"}
    assert (
        read_json_object(journal.rejected_path(slot, 0), label="rejection")["reason"]
        == reason
    )


def test_capture_resumes_after_interruption_with_only_the_next_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    interrupted = _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",),
        sample_elapsed=(5.0,),
        outcome="interrupted",
    )[4]
    measurement, post_exit, samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, interrupted
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )

    with pytest.raises(benchmark_runner.MemoryPressureTrialRejected):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda *args: pytest.fail(
                "interrupted attempt was relaunched"
            ),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda *args: None,
            recover=lambda *args: pytest.fail("memory rejection entered recovery"),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    retained_paths = (
        journal.recovery_sample_path(slot, 0, 0),
        journal.recovery_path(slot, 0),
        journal.rejected_path(slot, 0),
    )
    retained = tuple(path.read_bytes() for path in retained_paths)
    rejected_document = read_json_object(retained_paths[-1], label="rejection")
    raw_trial = rejected_document["trial"]
    launches = []
    normal = _valid_raw_trial(workload)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, next_attempt: (
            launches.append(next_attempt.journal_attempt_index)
            or _persist_journal_trial(journal, next_attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda *args: None,
        recover=lambda *args: pytest.fail("memory rejection created thermal recovery"),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == [1]
    assert len(trials) == 1
    assert tuple(path.read_bytes() for path in retained_paths) == retained
    assert read_json_object(retained_paths[-1], label="rejection")["trial"] == raw_trial


def test_capture_retries_only_the_thermal_slot_and_resumes_accepted_slots(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 0),
        BaselineSlot("prepared-data", 1),
    )
    accepted_first = _valid_raw_trial(workload, pair_index=0)
    first_attempt = journal.next_attempt(slots[0])
    accepted_first = _persist_journal_trial(journal, first_attempt, accepted_first)
    journal.accept_inflight(first_attempt, accepted_first)
    launches = []

    def launch(slot, attempt):
        launches.append(slot)
        base = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if len(launches) == 1:
            base = _with_thermal_state(base, "fair", 1)
        return _persist_journal_trial(journal, attempt, base)

    recovered = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=launch,
        preflight=lambda: _preflight_from_trial(accepted_first),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: recovered.append(slot),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slots[1], slots[1]]
    assert recovered == [slots[1]]
    assert [(trial.metric, trial.pair_index) for trial in trials] == [
        ("prepared-data", 0),
        ("prepared-data", 1),
    ]
    rejected = read_json_object(
        journal.rejected_path(slots[1], 0), label="rejected trial"
    )
    assert rejected["trial"]["environment_status"]["thermal_state"] == "fair"


def test_capture_reports_resumed_slots_in_supplied_order(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 1),
        BaselineSlot("prepared-data", 0),
    )
    for slot in reversed(slots):
        trial = _valid_raw_trial(workload, pair_index=slot.pair_index)
        attempt = journal.next_attempt(slot)
        trial = _persist_journal_trial(journal, attempt, trial)
        journal.accept_inflight(attempt, trial)
    messages = []

    capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=lambda slot, attempt: pytest.fail("accepted slot was launched"),
        preflight=lambda: pytest.fail("accepted slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
        progress=messages.append,
    )

    assert messages == [
        "resumed accepted prepared-data pair 1",
        "resumed accepted prepared-data pair 0",
    ]


def test_capture_passes_the_environment_policy_to_trial_validation_by_keyword(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    attempt = journal.next_attempt(slot)
    trial = _persist_journal_trial(journal, attempt, trial)
    policies = []

    def validate(current_trial, *, allow_rejected_environment):
        policies.append(allow_rejected_environment)

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "complete in-flight trial was relaunched"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=validate,
        classify_trial=_accept_trial,
        progress=lambda message: None,
    )

    assert policies == [True]
