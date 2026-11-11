import json
import os

import pytest

import v2.benchmarks.journal as baseline_journal
from v2.benchmarks.evidence import (
    build_post_exit_recovery_sample,
)
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    JournalAttempt,
    atomic_write_json,
    read_json_object,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
    structured_identity,
)

from .support import (
    _journal_trial_evidence,
    _persist_journal_evidence,
    _session_document,
    _valid_child_measurement,
    _valid_observation,
    _valid_post_exit_observation,
    _valid_raw_trial,
    _with_environment_observations,
    _with_thermal_state,
)


def test_journal_retains_the_complete_recovery_chain_after_acceptance(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, recovery, trial = _persist_journal_evidence(
        journal, attempt
    )

    journal.accept_inflight(attempt, trial)

    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert journal.recovery_path(slot, 0).is_file()
    assert tuple(
        journal.recovery_sample_path(slot, 0, index).is_file()
        for index in range(len(samples))
    ) == (True,) * len(samples)
    assert journal.load_accepted((slot,)) == {slot: trial}
    assert read_json_object(journal.measurement_path(slot, 0), label="measurement") == (
        measurement
    )
    assert read_json_object(journal.post_exit_path(slot, 0), label="post-exit") == (
        post_exit
    )
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery") == (
        recovery
    )


def test_journal_reports_measurement_only_as_pending_immediate_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    pending = journal.load_pending_attempts((slot,))

    assert pending == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=None,
            recovery_samples=(),
            recovery=None,
            trial=None,
        ),
    )


def test_journal_rejects_post_exit_without_its_exact_measurement(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)

    with pytest.raises(ValueError, match="post-exit evidence has no measurement"):
        journal.load_pending_attempts((slot,))


def test_journal_reports_post_exit_without_recovery_as_pending(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, recovery=False, inflight=False
    )

    assert journal.load_pending_attempts((slot,)) == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=post_exit,
            recovery_samples=(),
            recovery=None,
            trial=None,
        ),
    )


def test_journal_rejects_recovery_sample_without_post_exit(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )

    with pytest.raises(ValueError, match="recovery sample has no post-exit evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_bound_to_another_immediate_identity(
    tmp_path,
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    other_post_exit = json.loads(json.dumps(post_exit))
    other_post_exit["observed_at_utc"] = "2026-08-09T00:00:04+00:00"
    other_body = {
        key: value for key, value in other_post_exit.items() if key != "identity"
    }
    other_post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", other_body
    )
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    foreign_sample = build_post_exit_recovery_sample(
        measurement=measurement,
        post_exit=other_post_exit,
        sample_index=0,
        previous_sample_identity=None,
        elapsed_seconds=5.0,
        **observation,
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), foreign_sample, create_only=True
    )

    with pytest.raises(ValueError, match="recovery .* does not match bound evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_index_gap(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), samples[1], create_only=True
    )

    with pytest.raises(ValueError, match="recovery sample indices have a gap"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_sample_previous_identity_fork(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    forked = {
        **samples[1],
        "previous_sample_identity": "sha256:" + "0" * 64,
    }
    forked_body = {key: value for key, value in forked.items() if key != "identity"}
    forked["identity"] = structured_identity(
        "sml-parent-post-exit-recovery-sample-v1", forked_body
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), forked, create_only=True
    )

    with pytest.raises(ValueError, match="previous sample identity"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_nonincreasing_pending_recovery_sample_elapsed_time(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    nonincreasing = {
        **samples[1],
        "elapsed_seconds": samples[0]["elapsed_seconds"],
    }
    nonincreasing_body = {
        key: value for key, value in nonincreasing.items() if key != "identity"
    }
    nonincreasing["identity"] = structured_identity(
        "sml-parent-post-exit-recovery-sample-v1", nonincreasing_body
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 1), nonincreasing, create_only=True
    )

    with pytest.raises(ValueError, match="elapsed_seconds must be increasing"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_recovery_summary_without_its_exact_samples(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    for index, sample in enumerate(samples[:-1]):
        atomic_write_json(
            journal.recovery_sample_path(slot, 0, index), sample, create_only=True
        )
    atomic_write_json(journal.recovery_path(slot, 0), recovery, create_only=True)

    with pytest.raises(ValueError, match="sample identities do not match samples"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_inflight_without_recovery_summary(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _persist_journal_evidence(journal, attempt, recovery=False)

    with pytest.raises(ValueError, match="inflight trial has no recovery summary"):
        journal.load_pending_attempts((slot,))


@pytest.mark.parametrize("outcome", ["accepted", "rejected"])
def test_journal_rejects_terminal_evidence_missing_recovery_summary(tmp_path, outcome):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    source_trial = _valid_raw_trial(workload)
    if outcome == "rejected":
        source_trial = _with_thermal_state(source_trial, "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, source_trial
    )
    if outcome == "accepted":
        accepted_path = journal.accepted_path(slot)
        accepted_path.parent.mkdir(parents=True)
        os.link(attempt.path, accepted_path)
    else:
        journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
    journal.recovery_path(slot, 0).unlink()

    with pytest.raises(
        ValueError,
        match=(
            "incomplete staged evidence|no recovery summary|"
            "does not match its post-exit recovery"
        ),
    ):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_measurement_from_another_session(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(build_canonical_workload())
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)

    with pytest.raises(ValueError, match="does not match the journal session"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_post_exit_bound_to_another_measurement(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _journal_trial_evidence(
        journal, attempt
    )
    other_measurement = json.loads(json.dumps(measurement))
    other_measurement["trial"]["value"] += 1.0
    other_body = {
        key: value for key, value in other_measurement.items() if key != "identity"
    }
    other_measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", other_body
    )
    other_post_exit = _valid_post_exit_observation(other_measurement)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(
        journal.post_exit_path(slot, 0), other_post_exit, create_only=True
    )

    with pytest.raises(
        ValueError, match="child_measurement_identity does not match child measurement"
    ):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_inflight_without_staged_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    atomic_write_json(
        attempt.path,
        _valid_raw_trial(build_canonical_workload()).to_dict(),
        create_only=True,
    )

    with pytest.raises(ValueError, match="inflight trial has no post-exit evidence"):
        journal.load_pending_attempts((slot,))


def test_journal_rejects_measurement_only_and_preserves_its_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    journal.reject_unfinalized(
        attempt,
        measurement,
        reason="missing-immediate-post-exit-evidence",
    )

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected == {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "missing-immediate-post-exit-evidence",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": None,
        "post_exit_recovery_identity": None,
        "trial": None,
        "identity": rejected["identity"],
    }
    assert journal.measurement_path(slot, 0).is_file()
    assert journal.load_pending_attempts((slot,)) == ()
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_reject_unfinalized_requires_measurement_only_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, inflight=False
    )

    with pytest.raises(ValueError, match="measurement-only evidence"):
        journal.reject_unfinalized(
            attempt,
            measurement,
            reason="missing-immediate-post-exit-evidence",
        )


def test_journal_reject_unfinalized_requires_the_exact_reason(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, post_exit=False, inflight=False
    )

    with pytest.raises(ValueError, match="invalid reason"):
        journal.reject_unfinalized(attempt, measurement, reason="process-failed")


def test_journal_memory_rejection_needs_no_thermal_trigger(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    nominal = _valid_raw_trial(workload)
    warning = dict(nominal.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    memory_trial = _with_environment_observations(nominal, post_exit=warning)
    measurement, post_exit, _samples, _recovery, persisted = _persist_journal_evidence(
        journal, attempt, memory_trial
    )

    journal.reject_inflight(attempt, persisted, reason="interrupted-post-exit-recovery")

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected["version"] == 3
    assert rejected["child_measurement_identity"] == measurement["identity"]
    assert rejected["post_exit_observation_identity"] == post_exit["identity"]
    assert (
        rejected["post_exit_recovery_identity"]
        == persisted.post_exit_recovery["identity"]
    )
    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert not (journal.root / "thermal-waits").exists()
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_reject_inflight_rejects_an_unknown_reason_before_writing(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, fair = _persist_journal_evidence(
        journal, attempt, fair
    )

    with pytest.raises(ValueError, match="rejected trial reason is invalid"):
        journal.reject_inflight(attempt, fair, reason="retry-environment-later")

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


@pytest.mark.parametrize(
    "reason",
    (
        "critical-post-exit-memory-pressure",
        "post-exit-recovery-environment-violation",
        "interrupted-post-exit-recovery",
    ),
)
def test_journal_recognizes_every_schema_v3_recovery_rejection_reason(tmp_path, reason):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, fair
    )

    with pytest.raises(ValueError, match="expected non-nominal-thermal"):
        journal.reject_inflight(attempt, trial, reason=reason)

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


def test_journal_reject_inflight_refuses_an_accept_disposition(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, nominal = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )

    with pytest.raises(ValueError, match="accepted trial cannot be rejected"):
        journal.reject_inflight(attempt, nominal, reason="non-nominal-thermal")

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


@pytest.mark.parametrize(
    ("primary_signal", "supplied_reason", "expected_reason"),
    (
        (
            "start-memory",
            "critical-measurement-memory-pressure",
            "non-normal-start-memory-pressure",
        ),
        (
            "end-memory",
            "persistent-post-exit-memory-pressure",
            "critical-measurement-memory-pressure",
        ),
        (
            "post-exit-memory",
            "non-nominal-thermal",
            "interrupted-post-exit-recovery",
        ),
        (
            "thermal",
            "persistent-post-exit-memory-pressure",
            "non-nominal-thermal",
        ),
    ),
)
def test_journal_reject_inflight_requires_the_ordered_evidence_reason(
    tmp_path, primary_signal, supplied_reason, expected_reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal = _valid_raw_trial(workload)
    start = dict(nominal.environment_status["start"])
    end = dict(nominal.environment_status["end"])
    post_exit = dict(nominal.environment_status["post_exit"])
    if primary_signal == "start-memory":
        start["memory_pressure"] = "warning"
        end["memory_pressure"] = "critical"
        post_exit["memory_pressure"] = "warning"
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    elif primary_signal == "end-memory":
        end["memory_pressure"] = "critical"
        post_exit["memory_pressure"] = "warning"
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    elif primary_signal == "post-exit-memory":
        post_exit["memory_pressure"] = "warning"
    else:
        post_exit.update(thermal_state="fair", thermal_state_raw_value=1)
    rejected = _with_environment_observations(
        nominal,
        start=start,
        end=end,
        post_exit=post_exit,
    )
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, rejected = _persist_journal_evidence(
        journal, attempt, rejected
    )

    with pytest.raises(ValueError, match=expected_reason):
        journal.reject_inflight(attempt, rejected, reason=supplied_reason)

    assert attempt.path.is_file()
    assert not journal.rejected_path(slot, 0).exists()


def _rewrite_rejected_reason(journal, slot, reason):
    path = journal.rejected_path(slot, 0)
    document = read_json_object(path, label="rejected trial")
    body = {
        **{key: value for key, value in document.items() if key != "identity"},
        "reason": reason,
    }
    path.unlink()
    atomic_write_json(
        path,
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )


@pytest.mark.parametrize(
    "mutated_reason",
    (
        "retry-environment-later",
        "missing-immediate-post-exit-evidence",
        "persistent-post-exit-memory-pressure",
    ),
)
def test_journal_load_rejects_a_resigned_wrong_finalized_reason(
    tmp_path, mutated_reason
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, fair = _persist_journal_evidence(
        journal, attempt, fair
    )
    journal.reject_inflight(attempt, fair, reason="non-nominal-thermal")
    _rewrite_rejected_reason(journal, slot, mutated_reason)

    with pytest.raises(ValueError, match="rejected trial reason"):
        journal.next_attempt(slot)


def test_journal_load_refuses_a_rejected_accept_disposition(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, recovery, nominal = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "non-nominal-thermal",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": post_exit["identity"],
        "post_exit_recovery_identity": recovery["identity"],
        "trial": nominal.to_dict(),
    }
    attempt.path.unlink()
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="accepted trial cannot be rejected"):
        journal.next_attempt(slot)


def test_journal_rejects_both_outcomes_for_one_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, recovery, trial = _persist_journal_evidence(
        journal,
        attempt,
        _with_thermal_state(_valid_raw_trial(workload), "fair", 1),
    )
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)
    body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "non-nominal-thermal",
        "child_measurement_identity": measurement["identity"],
        "post_exit_observation_identity": post_exit["identity"],
        "post_exit_recovery_identity": recovery["identity"],
        "trial": trial.to_dict(),
    }
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="both accepted and rejected outcomes"):
        journal.load_pending_attempts((slot,))

    assert attempt.path.is_file()


def test_journal_rejects_an_attempt_index_gap_in_staged_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _persist_journal_evidence(journal, forged, post_exit=False, inflight=False)

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        journal.load_pending_attempts((slot,))


@pytest.mark.parametrize(
    "category", ["measurements", "post-exit", "recovery-samples", "recovery"]
)
def test_journal_rejects_unsupported_stage_paths(tmp_path, category):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    unsupported = journal.root / category / "unsupported" / "0" / "0.json"
    atomic_write_json(unsupported, {}, create_only=True)

    with pytest.raises(ValueError, match="unsupported metric directory"):
        journal.load_pending_attempts((BaselineSlot("prepared-data", 0),))


@pytest.mark.parametrize(
    "category", ["measurements", "post-exit", "recovery-samples", "recovery"]
)
def test_journal_refuses_symlinked_stage_ancestors(tmp_path, category):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    outside = tmp_path / f"outside-{category}"
    outside.mkdir()
    stage_root = journal.root / category
    stage_root.mkdir()
    (stage_root / "prepared-data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        journal.load_pending_attempts((BaselineSlot("prepared-data", 0),))
