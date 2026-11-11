import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    read_json_object,
)
from v2.benchmarks.runner import (
    publish_baseline_from_journal,
    validate_baseline_manifest,
    validate_checkout_status,
)
from v2.benchmarks.schema import METRIC_NAMES, RawTrial
from v2.benchmarks.workload import (
    build_canonical_workload,
    canonical_workload_identity,
)

from .support import (
    _persist_journal_evidence,
    _session_document,
    _valid_baseline_trials,
)


def test_checkout_status_allows_only_bound_untracked_final_outputs():
    allowed = frozenset(
        {
            "v2/benchmarks/manifests/baseline-current.json",
            "v2/benchmarks/results/baseline-current.jsonl",
        }
    )
    validate_checkout_status(
        "?? v2/benchmarks/manifests/baseline-current.json\n"
        "?? v2/benchmarks/results/baseline-current.jsonl\n",
        allowed_untracked_paths=allowed,
    )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            "?? v2/benchmarks/manifests/baseline-current.json\n?? unexpected.txt\n",
            allowed_untracked_paths=allowed,
        )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            " M v2/benchmarks/runner.py\n",
            allowed_untracked_paths=allowed,
        )


def _accepted_complete_journal(
    tmp_path,
    *,
    manifest_name="baseline.json",
    raw_output_name="baseline.jsonl",
):
    workload = build_canonical_workload()
    session = _session_document(
        tmp_path,
        manifest_name=manifest_name,
        raw_output_name=raw_output_name,
    )
    journal = BaselineJournal.open(tmp_path / "state", session)
    trials = _valid_baseline_trials(workload)
    persisted_trials = []
    for trial in trials:
        slot = BaselineSlot(trial.metric, trial.pair_index)
        attempt = journal.next_attempt(slot)
        _measurement, _post_exit, _samples, _recovery, persisted = (
            _persist_journal_evidence(journal, attempt, trial)
        )
        journal.accept_inflight(attempt, persisted)
        persisted_trials.append(persisted)
    return workload, journal, tuple(persisted_trials)


def test_final_publication_uses_exactly_the_45_accepted_trials(tmp_path):
    workload, journal, trials = _accepted_complete_journal(tmp_path)
    manifest_path = tmp_path / "baseline.json"
    raw_path = tmp_path / "baseline.jsonl"

    manifest = publish_baseline_from_journal(
        journal=journal,
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        manifest_path=manifest_path,
        raw_output_path=raw_path,
    )

    raw_trials = tuple(
        RawTrial.from_dict(json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    )
    validate_baseline_manifest(manifest, raw_trials)
    assert len(raw_trials) == 45
    assert [(trial.metric, trial.pair_index) for trial in raw_trials] == [
        (metric, pair_index) for metric in METRIC_NAMES for pair_index in range(5)
    ]
    completed = read_json_object(journal.completed_path, label="completion")
    assert completed["baseline_identity"] == manifest["identity"]
    assert completed["raw_trial_identities"] == [
        benchmark_runner._raw_trial_identity(trial) for trial in raw_trials
    ]


def test_final_publication_never_overwrites_different_existing_output(tmp_path):
    workload, journal, trials = _accepted_complete_journal(tmp_path)
    raw_path = tmp_path / "baseline.jsonl"
    raw_path.write_text("different existing content\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="final artifact already exists with different content"
    ):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=raw_path,
        )

    assert raw_path.read_text(encoding="utf-8") == "different existing content\n"
    assert not journal.completed_path.exists()


def test_interrupted_manifest_publication_resumes_byte_identically_across_cli_spellings(
    tmp_path, monkeypatch
):
    workload, journal, trials = _accepted_complete_journal(tmp_path)
    manifest_path = tmp_path / "baseline.json"
    raw_path = tmp_path / "baseline.jsonl"
    original_publish_completed = BaselineJournal.publish_completed

    monkeypatch.setattr(sys, "executable", "python-relative")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v2/benchmarks/runner.py",
            "record-baseline",
            "--manifest",
            "baseline.json",
            "--raw-output",
            "baseline.jsonl",
        ],
    )
    first_command = benchmark_runner.canonical_baseline_command(journal)
    command_parts = shlex.split(first_command)
    assert command_parts[command_parts.index("--pairs") + 1] == "5"
    assert command_parts[command_parts.index("--warmup") + 1] == "5"
    assert command_parts[command_parts.index("--measure") + 1] == "20"
    relocated = BaselineJournal(tmp_path / "equivalent-state", journal.session)
    assert benchmark_runner.canonical_baseline_command(relocated) == first_command
    monkeypatch.setattr(
        BaselineJournal,
        "publish_completed",
        lambda self, document: (_ for _ in ()).throw(
            RuntimeError("interrupted before completion")
        ),
    )
    with pytest.raises(RuntimeError, match="interrupted before completion"):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            manifest_path=manifest_path,
            raw_output_path=raw_path,
        )
    first_manifest_bytes = manifest_path.read_bytes()

    monkeypatch.setattr(sys, "executable", "/absolute/venv/bin/python3")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "/absolute/checkout/v2/benchmarks/runner.py",
            "record-baseline",
            "--raw-output",
            str(raw_path),
            "--manifest",
            str(manifest_path),
        ],
    )
    monkeypatch.setattr(
        BaselineJournal, "publish_completed", original_publish_completed
    )
    resumed = BaselineJournal.open(journal.root, journal.session)
    second_command = benchmark_runner.canonical_baseline_command(resumed)
    manifest = publish_baseline_from_journal(
        journal=resumed,
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        manifest_path=manifest_path,
        raw_output_path=raw_path,
    )

    assert second_command == first_command
    assert manifest["command"] == first_command
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert resumed.completed_path.is_file()


@pytest.mark.parametrize(
    ("manifest_name", "raw_output_name"),
    (
        ("same.json", "same.json"),
        ("baseline.json", "state/completed.json"),
        ("state/final-manifest.json", "baseline.jsonl"),
        ("out", "out/raw.jsonl"),
        ("out/manifest.json", "out"),
        (".", "baseline.jsonl"),
    ),
)
def test_final_publication_rejects_colliding_or_journal_contained_outputs(
    tmp_path, manifest_name, raw_output_name
):
    workload, journal, trials = _accepted_complete_journal(
        tmp_path,
        manifest_name=manifest_name,
        raw_output_name=raw_output_name,
    )
    manifest_path = tmp_path / manifest_name
    raw_path = tmp_path / raw_output_name
    output_existence_before = {
        manifest_path: manifest_path.exists(),
        raw_path: raw_path.exists(),
    }
    journal_files_before = {
        path.relative_to(journal.root): path.read_bytes()
        for path in journal.root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="final output paths"):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            manifest_path=manifest_path,
            raw_output_path=raw_path,
        )

    assert not journal.completed_path.exists()
    for path in (manifest_path, raw_path):
        if not output_existence_before[path]:
            assert not path.exists()
    assert {
        path.relative_to(journal.root): path.read_bytes()
        for path in journal.root.rglob("*")
        if path.is_file()
    } == journal_files_before
    assert (
        BaselineJournal.open(journal.root, journal.session).session == journal.session
    )


def test_detached_worktree_cleans_registration_after_verification_error(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    destination = tmp_path / "source"
    commit = "a" * 40
    registered = set()
    commands = []

    def run(command, *, cwd, check):
        commands.append(command)
        if command[:3] == ("git", "worktree", "add"):
            registered.add(Path(command[-2]))
        elif command[:3] == ("git", "worktree", "remove"):
            registered.remove(Path(command[-1]))
        return SimpleNamespace()

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run)
    monkeypatch.setattr(
        benchmark_runner,
        "_require_clean_checkout",
        lambda path, label: (_ for _ in ()).throw(RuntimeError("verification failed")),
    )

    with pytest.raises(RuntimeError, match="verification failed"):
        benchmark_runner._create_detached_worktree(repository, commit, destination)

    assert registered == set()
    assert [command[:3] for command in commands] == [
        ("git", "worktree", "add"),
        ("git", "worktree", "remove"),
    ]


def test_detached_worktree_preserves_verification_error_when_cleanup_fails(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "source"
    commit = "a" * 40
    original_error = RuntimeError("original verification failure")
    commands = []

    def run(command, *, cwd, check):
        commands.append(command)
        if command[:3] == ("git", "worktree", "add"):
            destination.mkdir()
            (destination / "tracked.py").write_text("content\n", encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run)
    monkeypatch.setattr(
        benchmark_runner,
        "_require_clean_checkout",
        lambda path, label: (_ for _ in ()).throw(original_error),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_remove_worktree",
        lambda repository, destination: (_ for _ in ()).throw(
            RuntimeError("primary cleanup failure")
        ),
    )

    with pytest.raises(RuntimeError, match="original verification failure") as caught:
        benchmark_runner._create_detached_worktree(repository, commit, destination)

    assert caught.value is original_error
    assert not destination.exists()
    assert ("git", "worktree", "prune") in commands
    assert any("primary cleanup failure" in note for note in caught.value.__notes__)


def test_managed_worktree_preserves_capture_error_when_final_cleanup_fails(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / "tracked.py").write_text("content\n", encoding="utf-8")
    original_error = RuntimeError("outer capture failure")
    commands = []

    monkeypatch.setattr(
        benchmark_runner,
        "_create_detached_worktree",
        lambda current_repository, commit, current_destination: None,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_remove_worktree",
        lambda current_repository, current_destination: (_ for _ in ()).throw(
            RuntimeError("final cleanup failure")
        ),
    )
    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: commands.append(command) or SimpleNamespace(),
    )

    with (
        pytest.raises(RuntimeError, match="outer capture failure") as caught,
        benchmark_runner._managed_detached_worktree(repository, "a" * 40, destination),
    ):
        raise original_error

    assert caught.value is original_error
    assert not destination.exists()
    assert ("git", "worktree", "prune") in commands
    assert any("final cleanup failure" in note for note in caught.value.__notes__)
