import json
import os

import pytest

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
    _persist_journal_evidence,
    _session_document,
    _valid_raw_trial,
    _with_environment_observations,
    _with_thermal_state,
    _with_trial_payload,
)


def test_journal_promotes_an_inflight_trial_and_resumes_the_slot(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )

    journal.accept_inflight(attempt, trial)

    assert journal.load_accepted((slot,)) == {slot: trial}
    assert not attempt.path.exists()
    with pytest.raises(ValueError, match="accepted slot is immutable"):
        journal.accept_inflight(
            JournalAttempt(slot, 1, journal.inflight_path(slot, 1)),
            _with_trial_payload(trial, value=trial.value + 1.0),
        )


def test_journal_preserves_rejected_trial_and_uses_a_new_attempt_number(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal,
        attempt,
        _with_thermal_state(_valid_raw_trial(workload), "fair", 1),
    )

    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")

    rejected = read_json_object(journal.rejected_path(slot, 0), label="rejected trial")
    assert rejected["journal_attempt_index"] == 0
    assert rejected["reason"] == "non-nominal-thermal"
    assert rejected["trial"]["environment_status"]["thermal_state_raw_value"] == 1
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_records_every_preflight_observation(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    document = journal.record_preflight(slot, 0, observation)

    assert document["identity"].startswith("sha256:")
    assert (
        read_json_object(journal.preflight_path(slot, 0), label="preflight") == document
    )


def test_journal_rejects_malformed_and_unexpected_accepted_state(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    atomic_write_json(journal.accepted_path(slot), {}, create_only=True)
    with pytest.raises(ValueError, match="raw trial has an invalid field set"):
        journal.load_accepted((slot,))

    journal.accepted_path(slot).unlink()
    unexpected = BaselineSlot("prepared-data", 1)
    atomic_write_json(journal.accepted_path(unexpected), {}, create_only=True)
    with pytest.raises(ValueError, match="unexpected accepted slot"):
        journal.load_accepted((slot,))


def test_journal_refuses_a_symlinked_preflight_ancestor(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    outside = tmp_path / "outside"
    outside.mkdir()
    preflight = journal.root / "preflight"
    preflight.mkdir()
    (preflight / slot.metric).symlink_to(outside, target_is_directory=True)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    with pytest.raises(ValueError):
        journal.record_preflight(slot, 0, observation)

    assert not list(outside.rglob("*"))


@pytest.mark.parametrize("entry_point", ["accept", "load-inflight", "load-accepted"])
def test_journal_reconciles_hard_linked_accepted_inflight_crash_split(
    tmp_path, entry_point
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    *_, trial = _persist_journal_evidence(journal, attempt, _valid_raw_trial(workload))
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(attempt.path, accepted_path)

    if entry_point == "accept":
        journal.accept_inflight(attempt, trial)
    elif entry_point == "load-inflight":
        assert journal.load_inflight((slot,)) == ()
    else:
        assert journal.load_accepted((slot,)) == {slot: trial}

    assert not attempt.path.exists()
    assert journal.load_inflight((slot,)) == ()
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_journal_replays_a_rejected_inflight_crash_split(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    fair = _with_thermal_state(_valid_raw_trial(workload), "fair", 1)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, fair
    )
    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    resumed = journal.next_attempt(slot)

    assert resumed.journal_attempt_index == 1
    assert not attempt.path.exists()


@pytest.mark.parametrize("operation", ["accept", "reject"])
def test_journal_rejects_a_mutation_that_skips_an_attempt_index(tmp_path, operation):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, forged, _valid_raw_trial(workload)
    )

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        if operation == "accept":
            journal.accept_inflight(forged, trial)
        else:
            journal.reject_inflight(forged, trial, reason="non-nominal-thermal")

    assert forged.path.exists()


def test_journal_rejects_a_preflight_index_gap_before_writing(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    with pytest.raises(ValueError, match="preflight indices have a gap"):
        journal.record_preflight(slot, 1, observation)

    assert not journal.preflight_path(slot, 1).exists()


def test_journal_rejects_a_recovery_index_gap_before_writing(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    preflight = journal.record_preflight(
        slot,
        0,
        {
            "observed_at_utc": "2026-08-05T00:00:00+00:00",
            "hardware": trial.hardware,
            "environment_status": trial.environment_status,
            "software_versions": trial.software_versions,
        },
    )

    with pytest.raises(ValueError, match="thermal recovery indices have a gap"):
        journal.record_recovery_trigger(
            slot, 1, {"source": "preflight", "preflight": preflight}
        )


def test_journal_requires_a_recovery_trigger_preflight_from_its_slot(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    other_slot = BaselineSlot("prepared-data", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }
    journal.record_preflight(slot, 0, observation)
    other_preflight = journal.record_preflight(other_slot, 0, observation)

    with pytest.raises(ValueError, match="does not match a persisted preflight"):
        journal.record_recovery_trigger(
            slot, 0, {"source": "preflight", "preflight": other_preflight}
        )


def test_journal_requires_a_recovery_trigger_rejected_identity_from_its_slot(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)

    with pytest.raises(ValueError, match="does not match a persisted rejected trial"):
        journal.record_recovery_trigger(
            slot,
            0,
            {
                "source": "rejected-trial",
                "rejected_trial_identity": "sha256:" + "0" * 64,
            },
        )


def _persist_rejected_trial_for_recovery_test(journal, slot, reason):
    attempt = journal.next_attempt(slot)
    workload = build_canonical_workload()
    if reason == "missing-immediate-post-exit-evidence":
        measurement, _post_exit, _samples, _recovery, _trial = (
            _persist_journal_evidence(journal, attempt, post_exit=False, inflight=False)
        )
        journal.reject_unfinalized(attempt, measurement, reason=reason)
    else:
        trial = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if reason == "interrupted-post-exit-recovery":
            warning = dict(trial.environment_status["post_exit"])
            warning["memory_pressure"] = "warning"
            trial = _with_environment_observations(trial, post_exit=warning)
        elif reason == "non-nominal-thermal":
            trial = _with_thermal_state(trial, "fair", 1)
        _measurement, _post_exit, _samples, _recovery, persisted = (
            _persist_journal_evidence(journal, attempt, trial)
        )
        journal.reject_inflight(attempt, persisted, reason=reason)
    return read_json_object(
        journal.rejected_path(slot, attempt.journal_attempt_index),
        label="rejected trial",
    )


@pytest.mark.parametrize(
    "reason",
    (
        "interrupted-post-exit-recovery",
        "missing-immediate-post-exit-evidence",
    ),
)
@pytest.mark.parametrize("boundary", ("record", "load"))
def test_journal_never_uses_a_nonthermal_rejection_as_a_thermal_trigger(
    tmp_path, reason, boundary
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    rejected = _persist_rejected_trial_for_recovery_test(journal, slot, reason)
    trigger_body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "rejected-trial",
        "rejected_trial_identity": rejected["identity"],
    }

    if boundary == "load":
        trigger = {
            **trigger_body,
            "identity": structured_identity(
                "sml-baseline-thermal-recovery-trigger-v1", trigger_body
            ),
        }
        atomic_write_json(
            journal.root
            / "thermal-waits"
            / slot.metric
            / str(slot.pair_index)
            / "0"
            / "trigger.json",
            trigger,
            create_only=True,
        )

    with pytest.raises(
        ValueError,
        match="requires a non-nominal-thermal rejected trial",
    ):
        if boundary == "record":
            journal.record_recovery_trigger(
                slot,
                0,
                {
                    "source": "rejected-trial",
                    "rejected_trial_identity": rejected["identity"],
                },
            )
        else:
            journal._validate_thermal_recovery_history()


def test_journal_accepts_a_non_nominal_thermal_rejection_as_a_recovery_trigger(
    tmp_path,
):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    rejected = _persist_rejected_trial_for_recovery_test(
        journal, slot, "non-nominal-thermal"
    )

    journal.record_recovery_trigger(
        slot,
        0,
        {
            "source": "rejected-trial",
            "rejected_trial_identity": rejected["identity"],
        },
    )

    assert journal._validate_thermal_recovery_history() == {slot: (0,)}


def test_journal_rejects_boolean_rejected_attempt_index(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    first = JournalAttempt(slot, 0, journal.inflight_path(slot, 0))
    second = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    first_measurement, _post_exit, _samples, _recovery, _trial = (
        _persist_journal_evidence(journal, first, post_exit=False, inflight=False)
    )
    second_measurement, _post_exit, _samples, _recovery, _trial = (
        _persist_journal_evidence(journal, second, post_exit=False, inflight=False)
    )
    zero_body = {
        "kind": "sml-baseline-rejected-trial",
        "version": 3,
        "journal_attempt_index": 0,
        "reason": "missing-immediate-post-exit-evidence",
        "child_measurement_identity": first_measurement["identity"],
        "post_exit_observation_identity": None,
        "post_exit_recovery_identity": None,
        "trial": None,
    }
    one_body = {
        **zero_body,
        "journal_attempt_index": True,
        "child_measurement_identity": second_measurement["identity"],
    }
    atomic_write_json(
        journal.rejected_path(slot, 0),
        {
            **zero_body,
            "identity": structured_identity(
                "sml-baseline-rejected-trial-v3", zero_body
            ),
        },
        create_only=True,
    )
    atomic_write_json(
        journal.rejected_path(slot, 1),
        {
            **one_body,
            "identity": structured_identity("sml-baseline-rejected-trial-v3", one_body),
        },
        create_only=True,
    )

    with pytest.raises(ValueError, match="journal attempt index must be non-negative"):
        journal.next_attempt(slot)


def test_journal_does_not_replay_independently_serialized_accepted_content(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    _measurement, _post_exit, _samples, _recovery, trial = _persist_journal_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    accepted_path.write_text(
        json.dumps(trial.to_dict(), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="accepted transition"):
        journal.load_accepted((slot,))

    assert attempt.path.exists()


def test_journal_preserves_a_gapped_inflight_split_before_reconciliation(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    forged = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _measurement, _post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, forged, _valid_raw_trial(workload)
    )
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(forged.path, accepted_path)

    with pytest.raises(ValueError, match="attempt indices have a gap"):
        journal.load_inflight((slot,))

    assert forged.path.exists()


def test_journal_preserves_multiple_inflight_split_candidates(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    first = JournalAttempt(slot, 0, journal.inflight_path(slot, 0))
    second = JournalAttempt(slot, 1, journal.inflight_path(slot, 1))
    _measurement, _post_exit, _samples, _recovery, _first_trial = (
        _persist_journal_evidence(journal, first, _valid_raw_trial(workload))
    )
    _persist_journal_evidence(journal, second, _valid_raw_trial(workload))
    accepted_path = journal.accepted_path(slot)
    accepted_path.parent.mkdir(parents=True)
    os.link(first.path, accepted_path)

    with pytest.raises(ValueError, match="multiple accepted transition candidates"):
        journal.load_inflight((slot,))

    assert first.path.exists()
    assert second.path.exists()


def test_journal_rejects_a_foreign_persisted_preflight_trigger_before_sampling(
    tmp_path,
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    other_slot = BaselineSlot("prepared-data", 1)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }
    other_preflight = journal.record_preflight(other_slot, 0, observation)
    body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "preflight",
        "preflight": other_preflight,
    }
    trigger = {
        **body,
        "identity": structured_identity(
            "sml-baseline-thermal-recovery-trigger-v1", body
        ),
    }
    trigger_path = (
        journal.root / "thermal-waits" / slot.metric / "0" / "0" / "trigger.json"
    )
    atomic_write_json(trigger_path, trigger, create_only=True)

    with pytest.raises(ValueError, match="does not match a persisted preflight"):
        journal.record_thermal_sample(
            slot,
            0,
            0,
            {
                "schema_version": 1,
                "observed_at_utc": "2026-08-05T00:00:01+00:00",
                "elapsed_seconds": 0.0,
                "hardware": trial.hardware,
                "environment_status": trial.environment_status,
                "software_versions": trial.software_versions,
            },
        )

    assert not (trigger_path.parent / "0.json").exists()


def test_journal_rejects_a_stale_persisted_rejected_trigger_before_summary(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    body = {
        "kind": "sml-baseline-thermal-recovery-trigger",
        "version": 1,
        "source": "rejected-trial",
        "rejected_trial_identity": "sha256:" + "0" * 64,
    }
    trigger = {
        **body,
        "identity": structured_identity(
            "sml-baseline-thermal-recovery-trigger-v1", body
        ),
    }
    trigger_path = (
        journal.root / "thermal-waits" / slot.metric / "0" / "0" / "trigger.json"
    )
    atomic_write_json(trigger_path, trigger, create_only=True)

    with pytest.raises(ValueError, match="does not match a persisted rejected trial"):
        journal.record_recovery_summary(
            slot,
            0,
            {"outcome": "timeout", "duration_seconds": 0.0, "sample_count": 0},
        )

    assert not (trigger_path.parent / "summary.json").exists()
