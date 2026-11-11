from pathlib import Path
from types import SimpleNamespace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    read_json_object,
)
from v2.benchmarks.recovery import (
    ThermalRecoveryResult,
    ThermalRecoveryTimeout,
)
from v2.benchmarks.runner import (
    capture_baseline_trials,
    classify_trial_environment,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
)

from .support import (
    _accept_trial,
    _persist_journal_evidence,
    _persist_journal_trial,
    _preflight_from_trial,
    _session_document,
    _valid_raw_trial,
    _with_environment_observations,
    _with_thermal_state,
)


def test_capture_records_preflight_thermal_trigger_before_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal_trial = _valid_raw_trial(workload)
    fair = {
        **nominal_trial.environment_status["start"],
        "thermal_state": "fair",
        "thermal_state_raw_value": 1,
    }
    preflights = iter(
        [
            (nominal_trial.hardware, fair, nominal_trial.software_versions),
            (
                nominal_trial.hardware,
                nominal_trial.environment_status["start"],
                nominal_trial.software_versions,
            ),
        ]
    )
    triggers = []
    launches = []

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            launches.append(current_slot)
            or _persist_journal_trial(journal, attempt, nominal_trial)
        ),
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: triggers.append(
            trigger
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slot]
    assert triggers[0]["source"] == "preflight"
    assert (
        triggers[0]["preflight"]["environment_status"]["thermal_state_raw_value"] == 1
    )
    assert journal.preflight_path(slot, 0).is_file()


def test_capture_classifies_complete_inflight_before_launching(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    trial = _persist_journal_trial(journal, attempt, trial)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "resume launched a replacement for a complete in-flight trial"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda current_trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert trials == (trial,)
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_capture_timeout_preserves_prior_acceptance_and_rejected_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    accepted = _valid_raw_trial(workload, pair_index=0)
    attempt = journal.next_attempt(first)
    accepted = _persist_journal_trial(journal, attempt, accepted)
    journal.accept_inflight(attempt, accepted)
    fair_trial = _with_thermal_state(
        _valid_raw_trial(workload, pair_index=1), "fair", 1
    )

    def timeout_recovery(current_slot, recovery_index, deadline, trigger):
        raise ThermalRecoveryTimeout(ThermalRecoveryResult(7_200.0, 241))

    with pytest.raises(ThermalRecoveryTimeout):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda current_slot, current_attempt: _persist_journal_trial(
                journal, current_attempt, fair_trial
            ),
            preflight=lambda: _preflight_from_trial(accepted),
            validate_preflight=lambda hardware, status, software: None,
            recover=timeout_recovery,
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            clock=lambda: 0.0,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
            progress=lambda message: None,
        )

    assert journal.load_accepted((first, second)) == {first: accepted}
    assert journal.rejected_path(second, 0).is_file()


@pytest.mark.parametrize(
    "persisted_state",
    (
        "untriggered-preflight",
        "unrecovered-rejected",
        "unfinished-recovery",
        "timeout-recovery",
    ),
)
def test_capture_replays_persisted_thermal_state_before_preflight(
    tmp_path, persisted_state
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }

    if persisted_state == "unrecovered-rejected":
        attempt = journal.next_attempt(slot)
        fair = _persist_journal_trial(journal, attempt, fair)
        journal.reject_inflight(attempt, fair, reason="non-nominal-thermal")
        rejected = read_json_object(
            journal.rejected_path(slot, 0), label="rejected trial"
        )
        expected_trigger = {
            "source": "rejected-trial",
            "rejected_trial_identity": rejected["identity"],
        }
        expected_recovery_index = 0
    else:
        preflight_document = journal.record_preflight(slot, 0, observation)
        expected_trigger = {
            "source": "preflight",
            "preflight": preflight_document,
        }
        expected_recovery_index = 0
        if persisted_state in ("unfinished-recovery", "timeout-recovery"):
            journal.record_recovery_trigger(slot, 0, expected_trigger)
            expected_recovery_index = 1
        if persisted_state == "timeout-recovery":
            journal.record_recovery_summary(
                slot,
                0,
                {
                    "outcome": "timeout",
                    "duration_seconds": 7_200.0,
                    "sample_count": 0,
                },
            )

    events = []
    recovered = []
    clock = SimpleNamespace(now=0.0)

    def progress(message):
        clock.now += 100.0

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            events.append("launch") or _persist_journal_trial(journal, attempt, nominal)
        ),
        preflight=lambda: events.append("preflight") or _preflight_from_trial(nominal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            events.append("recover"),
            recovered.append((recovery_index, deadline, trigger)),
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        clock=lambda: clock.now,
        utc_now=lambda: "2026-08-05T00:01:00+00:00",
        progress=progress,
    )

    assert events == ["recover", "preflight", "launch"]
    assert recovered == [(expected_recovery_index, 7_200.0, expected_trigger)]


def test_capture_replays_post_exit_only_thermal_rejection_before_preflight(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    post_exit_fair = dict(nominal.environment_status["post_exit"])
    post_exit_fair.update(thermal_state="fair", thermal_state_raw_value=1)
    rejected = _with_environment_observations(nominal, post_exit=post_exit_fair)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, rejected = _persist_journal_evidence(
        journal, attempt, rejected
    )
    journal.reject_inflight(attempt, rejected, reason="non-nominal-thermal")
    persisted_document = read_json_object(
        journal.rejected_path(slot, 0), label="rejected trial"
    )
    recovered = []

    def stop_after_recovery(current_slot, recovery_index, deadline, trigger):
        recovered.append((current_slot, recovery_index, deadline, trigger))
        raise RuntimeError("stop after persisted recovery")

    with pytest.raises(RuntimeError, match="stop after persisted recovery"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, current_attempt: pytest.fail(
                "trial launched before persisted recovery"
            ),
            preflight=lambda: pytest.fail("preflight ran before persisted recovery"),
            validate_preflight=lambda hardware, status, software: None,
            recover=stop_after_recovery,
            validate_trial=lambda trial, allow_rejected_environment: None,
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            clock=lambda: 0.0,
            progress=lambda message: None,
        )

    assert recovered == [
        (
            slot,
            0,
            7_200.0,
            {
                "source": "rejected-trial",
                "rejected_trial_identity": persisted_document["identity"],
            },
        )
    ]


def _session_bound_preflight_validator(workload, trial):
    return lambda hardware, status, software: (
        benchmark_runner._validate_baseline_preflight(
            workload=workload,
            hardware=hardware,
            status=status,
            software_versions=software,
            expected_hardware=trial.hardware,
            expected_software_versions=trial.software_versions,
        )
    )


def test_capture_rejects_an_invalid_persisted_preflight_before_any_action(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(workload)
    journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": trial.hardware,
            "environment_status": {
                **trial.environment_status,
                "power_connected": False,
            },
            "software_versions": trial.software_versions,
        },
    )

    with pytest.raises(ValueError, match="power_connected"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "launch ran before persisted preflight validation"
            ),
            preflight=lambda: pytest.fail(
                "new preflight ran before persisted preflight validation"
            ),
            validate_preflight=_session_bound_preflight_validator(workload, trial),
            recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
                "recovery ran before persisted preflight validation"
            ),
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
            progress=lambda message: None,
        )


def test_capture_rejects_every_persisted_recovery_sample_before_any_action(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    preflight = journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": fair.hardware,
            "environment_status": fair.environment_status,
            "software_versions": fair.software_versions,
        },
    )
    journal.record_recovery_trigger(
        slot, 0, {"source": "preflight", "preflight": preflight}
    )
    journal.record_thermal_sample(
        slot,
        0,
        0,
        {
            "schema_version": 1,
            "observed_at_utc": "2026-08-05T00:00:01+00:00",
            "elapsed_seconds": 300.0,
            "hardware": nominal.hardware,
            "environment_status": nominal.environment_status,
            "software_versions": {
                **nominal.software_versions,
                "python": "3.12.12",
            },
        },
    )
    journal.record_recovery_summary(
        slot,
        0,
        {"outcome": "nominal-window", "duration_seconds": 300.0, "sample_count": 1},
    )

    with pytest.raises(ValueError, match="software versions"):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "launch ran before persisted recovery validation"
            ),
            preflight=lambda: pytest.fail(
                "new preflight ran before persisted recovery validation"
            ),
            validate_preflight=_session_bound_preflight_validator(workload, nominal),
            recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
                "recovery ran before persisted recovery validation"
            ),
            validate_trial=lambda current_trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
            progress=lambda message: None,
        )


def test_capture_anchors_persisted_deadline_before_later_history_reads():
    workload = build_canonical_workload()
    slot = BaselineSlot("prepared-data", 0)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    preflight_document = {
        "identity": "sha256:" + "a" * 64,
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }
    clock = SimpleNamespace(now=0.0)

    class SlowHistoryJournal:
        def load_accepted(self, slots):
            return {}

        def _preflight_records(self):
            return ((slot, 0, Path("preflight.json"), preflight_document),)

        def _attempt_records(self, category):
            clock.now += 100.0
            return ()

        def _thermal_recovery_records(self):
            clock.now += 100.0
            return ()

        def load_inflight(self, slots):
            return ()

        def load_pending_attempts(self, slots):
            return ()

        def _validate_preflight_history(self):
            clock.now += 100.0
            return {slot: ((0, preflight_document),)}

        def _validate_thermal_recovery_history(self):
            clock.now += 100.0
            return {}

    recovered_deadlines = []

    def stop_after_recovery(current_slot, recovery_index, deadline, trigger):
        recovered_deadlines.append(deadline)
        raise RuntimeError("stop after recovery")

    with pytest.raises(RuntimeError, match="stop after recovery"):
        capture_baseline_trials(
            journal=SlowHistoryJournal(),
            slots=(slot,),
            launch_trial=lambda current_slot, attempt: pytest.fail(
                "trial launched before persisted recovery"
            ),
            preflight=lambda: pytest.fail("preflight ran before persisted recovery"),
            validate_preflight=lambda hardware, status, software: None,
            recover=stop_after_recovery,
            validate_trial=lambda trial, allow_rejected_environment: None,
            classify_trial=_accept_trial,
            clock=lambda: clock.now,
            progress=lambda message: None,
        )

    assert recovered_deadlines == [7_200.0]


def test_persisted_recovery_reuses_first_recognition_deadline_for_slot():
    workload = build_canonical_workload()
    slot = BaselineSlot("prepared-data", 0)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    first = {
        "identity": "sha256:" + "a" * 64,
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }
    second = {
        "identity": "sha256:" + "b" * 64,
        "hardware": fair.hardware,
        "environment_status": fair.environment_status,
        "software_versions": fair.software_versions,
    }

    class MultiplePreflightJournal:
        def _preflight_records(self):
            return (
                (slot, 0, Path("0.json"), first),
                (slot, 1, Path("1.json"), second),
            )

        def _attempt_records(self, category):
            return ()

        def _thermal_recovery_records(self):
            return ()

    clock = SimpleNamespace(now=0.0)

    def advancing_clock():
        observed = clock.now
        clock.now += 100.0
        return observed

    persisted = benchmark_runner._persisted_pending_thermal_triggers(
        MultiplePreflightJournal(),
        (slot,),
        advancing_clock,
        lambda hardware, status, software: None,
    )

    assert persisted[slot] == (
        {"source": "preflight", "preflight": second},
        7_200.0,
    )


def test_capture_does_not_replay_thermal_state_closed_by_nominal_window(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    preflight_document = journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": fair.hardware,
            "environment_status": fair.environment_status,
            "software_versions": fair.software_versions,
        },
    )
    journal.record_recovery_trigger(
        slot, 0, {"source": "preflight", "preflight": preflight_document}
    )
    journal.record_thermal_sample(
        slot,
        0,
        0,
        {
            "schema_version": 1,
            "observed_at_utc": "2026-08-05T00:05:00+00:00",
            "elapsed_seconds": 300.0,
            "hardware": nominal.hardware,
            "environment_status": nominal.environment_status,
            "software_versions": nominal.software_versions,
        },
    )
    journal.record_recovery_summary(
        slot,
        0,
        {
            "outcome": "nominal-window",
            "duration_seconds": 300.0,
            "sample_count": 1,
        },
    )

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: _persist_journal_trial(
            journal, attempt, nominal
        ),
        preflight=lambda: _preflight_from_trial(nominal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: pytest.fail(
            "completed nominal recovery was replayed"
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=_accept_trial,
        progress=lambda message: None,
    )


@pytest.mark.parametrize("violation_source", ("preflight", "rejected-trial"))
def test_capture_deadline_starts_before_slow_thermal_transition(
    tmp_path, monkeypatch, violation_source
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    fair = _with_thermal_state(nominal, "fair", 1)
    clock = SimpleNamespace(now=0.0)
    deadlines = []

    if violation_source == "preflight":
        preflights = iter(
            (
                (fair.hardware, fair.environment_status, fair.software_versions),
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
            )
        )
        original = BaselineJournal.record_preflight

        def slow_record_preflight(self, current_slot, preflight_index, observation):
            clock.now += 100.0
            return original(self, current_slot, preflight_index, observation)

        monkeypatch.setattr(BaselineJournal, "record_preflight", slow_record_preflight)

        def launch(current_slot, attempt):
            return _persist_journal_trial(journal, attempt, nominal)

    else:
        preflights = iter(
            (
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
                (
                    nominal.hardware,
                    nominal.environment_status,
                    nominal.software_versions,
                ),
            )
        )
        launches = []
        original = BaselineJournal.reject_inflight

        def slow_reject(self, attempt, trial, *, reason):
            clock.now += 100.0
            return original(self, attempt, trial, reason=reason)

        monkeypatch.setattr(BaselineJournal, "reject_inflight", slow_reject)

        def launch(current_slot, attempt):
            launches.append(current_slot)
            if len(launches) == 1:
                clock.now = 0.0
                trial = fair
            else:
                trial = nominal
            return _persist_journal_trial(journal, attempt, trial)

    def progress(message):
        clock.now += 100.0

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=launch,
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            deadlines.append(deadline)
        ),
        validate_trial=lambda trial, allow_rejected_environment: None,
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        clock=lambda: clock.now,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=progress,
    )

    assert deadlines == [7_200.0]
