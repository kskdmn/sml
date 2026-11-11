import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import v2.benchmarks.journal as baseline_journal
import v2.benchmarks.runner as benchmark_runner
from v2.benchmarks.journal import (
    BaselineJournal,
    BaselineSlot,
    atomic_write_json,
    atomic_write_text,
    read_json_object,
    require_external_state_directory,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
)

from .support import (
    _session_document,
    _valid_raw_trial,
)


def test_baseline_session_and_replay_bind_prepared_data_units(tmp_path):
    workload = build_canonical_workload()
    protocol = benchmark_runner._baseline_protocol(3, workload)
    session = _session_document(tmp_path, protocol=protocol)
    journal = BaselineJournal.open(tmp_path / "state", session)

    assert journal.session["protocol"]["measured_units"] == 20
    assert journal.session["protocol"]["prepared_data_measured_units"] == 100
    command = benchmark_runner.canonical_baseline_command(journal)
    assert "--measure 20" in command
    assert "--prepared-data-measure 100" in command


def test_baseline_journal_resumes_only_an_identical_session(tmp_path):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)

    first = BaselineJournal.open(state, expected)
    resumed = BaselineJournal.open(state, expected)

    assert first.session == expected
    assert resumed.session == expected
    changed = _session_document(tmp_path, harness_commit="d" * 40)
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, changed)


def test_thermal_trigger_resumes_after_crash_between_mkdir_and_publication(
    tmp_path, monkeypatch
):
    session = _session_document(tmp_path)
    journal = BaselineJournal.open(tmp_path / "state", session)
    slot = BaselineSlot("prepared-data", 0)
    trial = _valid_raw_trial(build_canonical_workload())
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
    trigger = {"source": "preflight", "preflight": preflight}
    real_write = baseline_journal._write_immutable_json

    def crash_after_directory(path, document, *, label):
        if label == "thermal recovery trigger":
            path.parent.mkdir(parents=True)
            raise KeyboardInterrupt("interrupted before trigger publication")
        return real_write(path, document, label=label)

    with monkeypatch.context() as patch:
        patch.setattr(baseline_journal, "_write_immutable_json", crash_after_directory)
        with pytest.raises(KeyboardInterrupt, match="before trigger publication"):
            journal.record_recovery_trigger(slot, 0, trigger)
    episode = journal._thermal_wait_path(slot, 0)
    assert episode.is_dir() and not tuple(episode.iterdir())

    resumed = BaselineJournal.open(journal.root, session)
    assert resumed._thermal_recovery_records() == ()
    resumed.record_recovery_trigger(slot, 0, trigger)
    resumed.record_recovery_trigger(slot, 0, trigger)
    assert resumed._validate_thermal_recovery_history() == {slot: (0,)}
    assert (
        read_json_object(episode / "trigger.json", label="trigger")["preflight"]
        == preflight
    )


def test_thermal_episode_without_trigger_preserves_nonempty_malformed_evidence(
    tmp_path,
):
    session = _session_document(tmp_path)
    journal = BaselineJournal.open(tmp_path / "state", session)
    slot = BaselineSlot("prepared-data", 0)
    evidence = journal._thermal_wait_path(slot, 0) / "0.json"
    atomic_write_json(evidence, {"partial": True}, create_only=True)

    resumed = BaselineJournal.open(journal.root, session)
    with pytest.raises(ValueError, match="thermal recovery trigger"):
        resumed._thermal_recovery_records()
    assert read_json_object(evidence, label="evidence") == {"partial": True}


@pytest.mark.parametrize(
    "changed",
    [
        {"protocol": {"pairs": 4, "warmup_units": 5, "measured_units": 20}},
        {"protocol": {"pairs": 5, "warmup_units": 20, "measured_units": 100}},
        {"hardware": {"chip": "Apple M4"}},
        {"software_versions": {"python": "3.12.12", "mlx": "0.32.4"}},
        {"manifest_name": "other.json"},
        {"raw_output_name": "other.jsonl"},
    ],
)
def test_baseline_journal_rejects_every_session_compatibility_change(tmp_path, changed):
    state = tmp_path / "state"
    BaselineJournal.open(state, _session_document(tmp_path))
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, _session_document(tmp_path, **changed))


def test_state_directory_must_be_outside_measured_checkouts(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    with pytest.raises(ValueError, match="outside measured checkouts"):
        require_external_state_directory(harness / "state", (harness,))

    external = tmp_path / "external"
    assert require_external_state_directory(external, (harness,)) == external.resolve()


def test_journal_never_adopts_a_nonempty_directory_without_a_session(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "orphan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty state directory has no session"):
        BaselineJournal.open(state, _session_document(tmp_path))


def _orphan_atomic_temporary(destination, token="a" * 32):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (f".{destination.name}.sml-atomic-{token}.tmp")
    temporary.write_text("orphan\n", encoding="utf-8")
    return temporary


def test_session_lock_is_same_thread_reentrant_and_excludes_a_concurrent_invocation(
    tmp_path,
):
    state = tmp_path / "state"
    outcomes = []

    with baseline_journal.baseline_session_lock(state):
        with baseline_journal.baseline_session_lock(state):
            pass

        def contend():
            try:
                with baseline_journal.baseline_session_lock(state):
                    outcomes.append("entered")
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert outcomes == ["baseline session is already locked"]
    assert (state / ".baseline-session.lock").is_file()


def test_session_lock_is_released_when_an_owner_process_crashes(tmp_path):
    state = tmp_path / "state"
    script = (
        "import os, pathlib; "
        "from v2.benchmarks.journal import baseline_session_lock; "
        f"state = pathlib.Path({str(state)!r}); "
        "lock = baseline_session_lock(state); lock.__enter__(); os._exit(0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with baseline_journal.baseline_session_lock(state):
        pass


def test_output_lock_excludes_a_process_with_a_different_temporary_directory(
    tmp_path,
):
    manifest = tmp_path / "outputs" / "baseline.json"
    raw_output = tmp_path / "outputs" / "baseline.jsonl"
    alternate_temporary_directory = tmp_path / "alternate-tmp"
    alternate_temporary_directory.mkdir()
    script = f"""
from pathlib import Path

from v2.benchmarks.journal import baseline_output_lock

manifest = Path({str(manifest)!r})
raw_output = Path({str(raw_output)!r})
try:
    with baseline_output_lock(manifest, raw_output):
        raise SystemExit("entered")
except RuntimeError as error:
    print(error)
"""

    with baseline_journal.baseline_output_lock(manifest, raw_output):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[4],
            env={**os.environ, "TMPDIR": str(alternate_temporary_directory)},
            check=False,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "baseline final outputs are already locked"


@pytest.mark.parametrize(
    ("first_name", "alias_name"),
    [
        ("Case/Manifest.json", "case/manifest.JSON"),
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}.json",
            "cafe\N{COMBINING ACUTE ACCENT}.json",
        ),
    ],
)
def test_output_lock_conservatively_serializes_case_and_unicode_aliases(
    tmp_path, first_name, alias_name
):
    outcomes = []
    first_manifest = tmp_path / first_name
    alias_manifest = tmp_path / alias_name

    with baseline_journal.baseline_output_lock(
        first_manifest, tmp_path / "first.jsonl"
    ):

        def contend():
            try:
                with baseline_journal.baseline_output_lock(
                    alias_manifest, tmp_path / "second.jsonl"
                ):
                    outcomes.append("entered")
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert outcomes == ["baseline final outputs are already locked"]


def test_output_lock_root_is_private_and_rejects_unsafe_permissions(
    tmp_path, monkeypatch
):
    secure_root = tmp_path / "secure-locks"
    monkeypatch.setattr(
        baseline_journal, "_baseline_output_lock_root", lambda: secure_root
    )

    with baseline_journal.baseline_output_lock(
        tmp_path / "manifest.json", tmp_path / "raw.jsonl"
    ):
        pass

    assert stat.S_IMODE(secure_root.stat().st_mode) == 0o700

    unsafe_root = tmp_path / "unsafe-locks"
    unsafe_root.mkdir(mode=0o700)
    unsafe_root.chmod(0o777)
    monkeypatch.setattr(
        baseline_journal, "_baseline_output_lock_root", lambda: unsafe_root
    )

    with (
        pytest.raises(ValueError, match="lock directory permissions"),
        baseline_journal.baseline_output_lock(
            tmp_path / "other-manifest.json", tmp_path / "other-raw.jsonl"
        ),
    ):
        pass


def test_durable_directory_creation_tolerates_a_concurrent_creator(
    tmp_path, monkeypatch
):
    target = tmp_path / "parent" / "child"
    original_mkdir = baseline_journal.os.mkdir
    raced = False

    def mkdir_after_competitor(path, mode=0o777):
        nonlocal raced
        if Path(path) == target and not raced:
            raced = True
            original_mkdir(path, mode)
            raise FileExistsError
        return original_mkdir(path, mode)

    monkeypatch.setattr(baseline_journal.os, "mkdir", mkdir_after_competitor)

    baseline_journal._create_durable_directory(target)

    assert target.is_dir()


def test_journal_never_treats_a_symlink_as_the_persistent_lock_inode(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_text("outside\n", encoding="utf-8")
    (state / ".baseline-session.lock").symlink_to(outside)

    with pytest.raises(ValueError, match="lock"):
        BaselineJournal.open(state, _session_document(tmp_path))

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_locked_resume_removes_every_recognized_journal_orphan_and_fsyncs_parents(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected_session = _session_document(tmp_path)
    BaselineJournal.open(state, expected_session)
    destinations = (
        state / ".baseline-session-initializing",
        state / "session.json",
        state / "completed.json",
        state / "accepted" / "prepared-data" / "0.json",
        state / "measurements" / "prepared-data" / "0" / "0.json",
        state / "post-exit" / "prepared-data" / "0" / "0.json",
        state / "recovery-samples" / "prepared-data" / "0" / "0" / "0.json",
        state / "recovery" / "prepared-data" / "0" / "0.json",
        state / "inflight" / "prepared-data" / "0" / "0.json",
        state / "rejected" / "prepared-data" / "0" / "0.json",
        state / "preflight" / "prepared-data" / "0" / "0.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "trigger.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "0.json",
        state / "thermal-waits" / "prepared-data" / "0" / "0" / "summary.json",
    )
    temporaries = tuple(_orphan_atomic_temporary(path) for path in destinations)
    marker = state / ".baseline-session-initializing"
    marker.write_text("", encoding="utf-8")
    synced = []
    original_fsync = baseline_journal._fsync_directory

    with baseline_journal.baseline_session_lock(state):
        monkeypatch.setattr(
            baseline_journal,
            "_fsync_directory",
            lambda path: synced.append(path.resolve()),
        )
        baseline_journal.cleanup_orphaned_journal_temporaries(state)
        monkeypatch.setattr(baseline_journal, "_fsync_directory", original_fsync)

    assert not marker.exists()
    assert not any(path.exists() for path in temporaries)
    assert {path.parent.resolve() for path in temporaries} <= set(synced)
    resumed = BaselineJournal.open(state, expected_session)
    assert (
        resumed.next_attempt(BaselineSlot("prepared-data", 0)).journal_attempt_index
        == 0
    )


def test_locked_resume_recovers_an_interrupted_root_initialization(tmp_path):
    state = tmp_path / "state"
    marker = state / ".baseline-session-initializing"
    marker_temp = _orphan_atomic_temporary(marker)
    session_temp = _orphan_atomic_temporary(state / "session.json", "b" * 32)
    marker.write_text("", encoding="utf-8")

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)

    assert not marker.exists()
    assert not marker_temp.exists()
    assert not session_temp.exists()
    BaselineJournal.open(state, _session_document(tmp_path))


def test_orphan_cleanup_preserves_hidden_wrong_pattern_and_symlink_nodes(tmp_path):
    state = tmp_path / "state"
    arbitrary = state / ".arbitrary-hidden"
    wrong_pattern = state / ".session.json.sml-atomic-not-a-token.tmp"
    wrong_destination = state / f".foreign.json.sml-atomic-{'b' * 32}.tmp"
    arbitrary.parent.mkdir(parents=True)
    arbitrary.write_text("keep\n", encoding="utf-8")
    wrong_pattern.write_text("keep\n", encoding="utf-8")
    wrong_destination.write_text("keep\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("outside\n", encoding="utf-8")
    symlink = state / f".session.json.sml-atomic-{'c' * 32}.tmp"
    symlink.symlink_to(outside)

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)

    assert arbitrary.read_text(encoding="utf-8") == "keep\n"
    assert wrong_pattern.read_text(encoding="utf-8") == "keep\n"
    assert wrong_destination.read_text(encoding="utf-8") == "keep\n"
    assert symlink.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside\n"
    with pytest.raises(ValueError, match="unexpected content|non-empty"):
        BaselineJournal.open(state, _session_document(tmp_path))


def test_a_live_atomic_temporary_cannot_be_cleaned_by_a_concurrent_invocation(
    tmp_path,
):
    state = tmp_path / "state"
    destination = state / "session.json"
    outcomes = []

    with baseline_journal.baseline_session_lock(state):
        live = _orphan_atomic_temporary(destination)

        def contend():
            try:
                with baseline_journal.baseline_session_lock(state):
                    baseline_journal.cleanup_orphaned_journal_temporaries(state)
            except RuntimeError as error:
                outcomes.append(str(error))

        thread = threading.Thread(target=contend)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert live.is_file()

    assert outcomes == ["baseline session is already locked"]
    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_journal_temporaries(state)
    assert not live.exists()


def test_locked_final_output_cleanup_removes_only_exact_regular_temporaries(tmp_path):
    state = tmp_path / "state"
    manifest = tmp_path / "checkout" / "baseline.json"
    raw = tmp_path / "results" / "baseline.jsonl"
    manifest_temp = _orphan_atomic_temporary(manifest, "d" * 32)
    raw_temp = _orphan_atomic_temporary(raw, "e" * 32)
    wrong = manifest.parent / f".other.json.sml-atomic-{'f' * 32}.tmp"
    wrong.write_text("keep\n", encoding="utf-8")
    outside = tmp_path / "outside-final"
    outside.write_text("outside\n", encoding="utf-8")
    linked = raw.parent / f".{raw.name}.sml-atomic-{'0' * 32}.tmp"
    linked.symlink_to(outside)

    with baseline_journal.baseline_session_lock(state):
        baseline_journal.cleanup_orphaned_atomic_temporaries((manifest, raw))

    assert not manifest_temp.exists()
    assert not raw_temp.exists()
    assert wrong.read_text(encoding="utf-8") == "keep\n"
    assert linked.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_record_baseline_holds_the_session_lock_across_cleanup_and_capture_entry(
    tmp_path, monkeypatch
):
    harness = tmp_path / "harness"
    harness.mkdir()
    args = SimpleNamespace(
        state_directory=tmp_path / "state",
        manifest=tmp_path / "outputs" / "baseline.json",
        raw_output=tmp_path / "outputs" / "baseline.jsonl",
    )
    manifest_temp = _orphan_atomic_temporary(args.manifest)
    raw_temp = _orphan_atomic_temporary(args.raw_output, "b" * 32)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    failures = []

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: harness)

    def capture(current_args, **paths):
        calls.append(paths)
        assert not manifest_temp.exists()
        assert not raw_temp.exists()
        entered.set()
        assert release.wait(timeout=5)
        return 0

    monkeypatch.setattr(benchmark_runner, "_record_baseline_locked", capture)

    def first_invocation():
        try:
            benchmark_runner._record_baseline(args)
        except Exception as error:  # noqa: BLE001 - surface thread failures to test
            failures.append(error)

    thread = threading.Thread(target=first_invocation)
    thread.start()
    assert entered.wait(timeout=5)
    with pytest.raises(RuntimeError, match="already locked"):
        benchmark_runner._record_baseline(args)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert len(calls) == 1


@pytest.mark.parametrize("second_raw_name", ["baseline.jsonl", "other.jsonl"])
def test_different_state_roots_cannot_clean_a_live_shared_output_temporary(
    tmp_path, monkeypatch, second_raw_name
):
    harness = tmp_path / "harness"
    harness.mkdir()
    manifest = tmp_path / "outputs" / "baseline.json"
    raw_output = tmp_path / "outputs" / "baseline.jsonl"
    first_args = SimpleNamespace(
        state_directory=tmp_path / "first-state",
        manifest=manifest,
        raw_output=raw_output,
    )
    second_args = SimpleNamespace(
        state_directory=tmp_path / "second-state",
        manifest=manifest,
        raw_output=tmp_path / "outputs" / second_raw_name,
    )
    entered = threading.Event()
    release = threading.Event()
    failures = []
    live_temporary = []

    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: harness)

    def capture(current_args, **paths):
        if paths["state_root"] == first_args.state_directory.resolve():
            live_temporary.append(_orphan_atomic_temporary(manifest))
            entered.set()
            assert release.wait(timeout=5)
        return 0

    monkeypatch.setattr(benchmark_runner, "_record_baseline_locked", capture)

    def first_invocation():
        try:
            benchmark_runner._record_baseline(first_args)
        except Exception as error:  # noqa: BLE001 - surface thread failures to test
            failures.append(error)

    thread = threading.Thread(target=first_invocation)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="final outputs are already locked"):
            benchmark_runner._record_baseline(second_args)
        assert live_temporary[0].is_file()
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []


def test_journal_rejects_an_orphan_written_during_session_initialization(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    original_write = baseline_journal.atomic_write_json

    def write_session_after_orphan(path, value, *, create_only=False):
        (state / "orphan.json").parent.mkdir(parents=True, exist_ok=True)
        (state / "orphan.json").write_text("{}\n", encoding="utf-8")
        return original_write(path, value, create_only=create_only)

    monkeypatch.setattr(
        baseline_journal, "atomic_write_json", write_session_after_orphan
    )

    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)

    assert (state / "orphan.json").is_file()
    assert (state / "session.json").is_file()
    assert {path.name for path in state.iterdir()} == {
        ".baseline-session-initializing",
        "orphan.json",
        "session.json",
    }
    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)


def test_journal_rejects_later_resume_after_orphan_follows_publication_scan(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    original_iterdir = Path.iterdir

    def entries_before_late_orphan(path):
        entries = tuple(original_iterdir(path))
        if path == state and (state / "session.json") in entries:
            (state / "orphan.json").write_text("{}\n", encoding="utf-8")
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", entries_before_late_orphan)

    first = BaselineJournal.open(state, expected)

    assert first.session == expected
    assert (state / "orphan.json").is_file()
    with pytest.raises(ValueError, match="unexpected content"):
        BaselineJournal.open(state, expected)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda session: session.__setitem__("unexpected", True),
        lambda session: session.pop("protocol"),
        lambda session: session.__setitem__("identity", "sha256:" + "0" * 64),
    ],
)
def test_journal_rejects_corrupt_persisted_session_documents(tmp_path, corrupt):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)
    BaselineJournal.open(state, expected)
    persisted = read_json_object(state / "session.json", label="test session")
    corrupt(persisted)
    (state / "session.json").write_text(json.dumps(persisted), encoding="utf-8")

    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, expected)


def test_atomic_write_text_replaces_content_and_cleans_up_temporary_files(tmp_path):
    path = tmp_path / "record.txt"
    path.write_text("old\n", encoding="utf-8")

    atomic_write_text(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".record.txt.*"))


def test_atomic_write_json_create_only_never_overwrites_and_cleans_up(tmp_path):
    path = tmp_path / "record.json"
    atomic_write_json(path, {"generation": 1}, create_only=True)

    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"generation": 2}, create_only=True)

    assert read_json_object(path, label="record") == {"generation": 1}
    assert not list(tmp_path.glob(".record.json.*"))
