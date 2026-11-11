from __future__ import annotations

import fcntl
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts.checkpoint import (
    publication_lock,
    run_access_lock,
    run_writer_lock,
)
from sml.errors import SMLArtifactError

from ._publication import (
    _ForwardingFilesystemOps,
)


class _DelayedMetadataFilesystemOps(_ForwardingFilesystemOps):
    def __init__(self, inner, *, transition) -> None:
        super().__init__(inner)
        self._transition = transition
        self._claimed = False

    def write_all(self, descriptor, data):
        delay = False
        if b"protected_path" in data and not self._claimed:
            self._claimed = True
            delay = True
        if delay:
            self._transition.send("metadata-started")
            if not self._transition.poll(10):
                raise RuntimeError("timed out delaying owner diagnostics")
            if self._transition.recv() != "allow-metadata":
                raise RuntimeError("unexpected owner diagnostic release message")
        self._inner.write_all(descriptor, data)


class _DelayedReleaseFilesystemOps(_ForwardingFilesystemOps):
    def __init__(self, inner, *, transition) -> None:
        super().__init__(inner)
        self._transition = transition
        self._creator_pid = os.getpid()
        self._empty_metadata_descriptor: int | None = None
        self._delayed = False

    def write_all(self, descriptor, data):
        self._inner.write_all(descriptor, data)
        if data == b'{"holders":[]}':
            self._empty_metadata_descriptor = descriptor

    def flock(self, descriptor, operation):
        self._inner.flock(descriptor, operation)
        if (
            not self._delayed
            and os.getpid() != self._creator_pid
            and descriptor == self._empty_metadata_descriptor
            and operation == fcntl.LOCK_UN
        ):
            self._delayed = True
            self._transition.send("transition-started")
            if not self._transition.poll(10):
                raise RuntimeError("timed out delaying release transition")
            if self._transition.recv() != "allow-transition":
                raise RuntimeError("unexpected release transition message")


def _acquire_publication_lock_then_exit(target: str, connection) -> None:
    with publication_lock(Path(target)):
        connection.send("locked")
        connection.close()
        os._exit(0)


def _hold_run_writer(run: str, connection) -> None:
    try:
        with run_writer_lock(Path(run)):
            connection.send("holder-ready")
            if not connection.poll(10):
                raise RuntimeError("timed out waiting to release run writer")
            if connection.recv() != "release-holder":
                raise RuntimeError("unexpected run writer release message")
    finally:
        connection.close()


def _hold_run_writer_with_fs(run: str, connection, fs) -> None:
    checkpoint.OS_FILESYSTEM = fs
    try:
        _hold_run_writer(run, connection)
    finally:
        fs._transition.close()


def _hold_run_access(run: str, connection) -> None:
    try:
        with run_access_lock(Path(run), exclusive=False):
            connection.send("holder-ready")
            if not connection.poll(10):
                raise RuntimeError("timed out waiting to release run access")
            if connection.recv() != "release-holder":
                raise RuntimeError("unexpected run access release message")
    finally:
        connection.close()


def _send_if_open(connection, message: str) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _join_or_terminate(process) -> tuple[int | None, bool]:
    terminated = False
    process.join(timeout=10)
    if process.is_alive():
        terminated = True
        process.terminate()
        process.join(timeout=5)
    return process.exitcode, terminated


def _acquire_run_access_then_exit(run: str, connection) -> None:
    with run_access_lock(Path(run), exclusive=False):
        connection.send("locked")
        connection.close()
        os._exit(0)


def _conflicting_run_writer_message(run: Path) -> str:
    try:
        with run_writer_lock(run):
            raise AssertionError("conflicting writer entered protected operation")
    except SMLArtifactError as error:
        return str(error)


def _run_writer_outcome(run: Path) -> tuple[str, str]:
    try:
        with run_writer_lock(run):
            return "acquired", ""
    except SMLArtifactError as error:
        return "conflict", str(error)


def test_conflicting_writer_reports_owner(tmp_path):
    """A conflicting run writer needs the live owner PID and protected run."""
    run = tmp_path / "run-0001"
    run.mkdir()

    with (
        run_writer_lock(run),
        pytest.raises(SMLArtifactError) as conflict,
        run_writer_lock(run),
    ):
        pytest.fail("conflicting writer entered the protected operation")

    message = str(conflict.value)
    assert str(os.getpid()) in message
    assert str(run) in message


@pytest.mark.parametrize("category", ("run-writer", "publication", "run-access"))
@pytest.mark.parametrize("existing", (False, True))
def test_case_aliases_share_artifact_locks(tmp_path, category, existing):
    """Creation and existing-artifact locks follow the parent's name semantics."""
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if os.fpathconf(descriptor, 11):
            pytest.skip("requires a case-insensitive APFS volume")
    finally:
        os.close(descriptor)
    target = tmp_path / "Run"
    alias = tmp_path / "run"
    if existing:
        target.mkdir()
        assert target.samefile(alias)
    with (
        checkpoint._protected_lock(target, category=category, exclusive=True),
        pytest.raises(SMLArtifactError, match="held by"),
        checkpoint._protected_lock(alias, category=category, exclusive=True),
    ):
        pytest.fail("a case alias bypassed the protected artifact lock")


def test_case_sensitive_volume_keeps_distinct_creation_locks(tmp_path, monkeypatch):
    """Names that can denote separate artifacts must remain independently lockable."""
    monkeypatch.setattr(checkpoint.os, "fpathconf", lambda _fd, _key: 1)
    with publication_lock(tmp_path / "Run"), publication_lock(tmp_path / "run"):
        pass


def test_unicode_aliases_share_creation_lock(tmp_path):
    with (
        publication_lock(tmp_path / "caf\u00e9"),
        pytest.raises(SMLArtifactError, match="held by"),
        publication_lock(tmp_path / "cafe\u0301"),
    ):
        pytest.fail("a normalization alias bypassed the publication lock")


def test_conflicting_process_reports_live_owner(tmp_path):
    """Conflict diagnostics must identify the live holder in another process."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_hold_run_writer,
        args=(str(run), child_connection),
    )
    process.start()
    child_connection.close()
    child_exitcode = None
    child_terminated = False
    try:
        assert parent_connection.poll(10), "child did not acquire the run writer lock"
        assert parent_connection.recv() == "holder-ready"
        with (
            pytest.raises(SMLArtifactError) as conflict,
            run_writer_lock(run),
        ):
            pytest.fail("conflicting writer entered the protected operation")
        message = str(conflict.value)
        assert str(process.pid) in message
        assert str(run) in message
    finally:
        _send_if_open(parent_connection, "release-holder")
        child_exitcode, child_terminated = _join_or_terminate(process)
        parent_connection.close()
        process.close()
    assert child_terminated is False
    assert child_exitcode == 0


def test_conflict_waits_for_complete_owner_diagnostics(tmp_path, monkeypatch):
    """A contender must not read owner metadata during truncate/write transition."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    metadata_parent, metadata_child = context.Pipe()
    holder_parent, holder_child = context.Pipe()
    delayed_fs = _DelayedMetadataFilesystemOps(
        checkpoint.OS_FILESYSTEM,
        transition=metadata_child,
    )
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", delayed_fs)
    process = context.Process(
        target=_hold_run_writer_with_fs,
        args=(str(run), holder_child, delayed_fs),
    )
    process.start()
    metadata_child.close()
    holder_child.close()
    child_exitcode = None
    child_terminated = False
    try:
        assert metadata_parent.poll(10), "owner did not begin diagnostic transition"
        assert metadata_parent.recv() == "metadata-started"
        with ThreadPoolExecutor(max_workers=1) as executor:
            conflict = executor.submit(_conflicting_run_writer_message, run)
            try:
                time.sleep(0.15)
                assert not conflict.done()
            finally:
                _send_if_open(metadata_parent, "allow-metadata")
            assert holder_parent.poll(10), (
                "owner did not complete diagnostic transition"
            )
            assert holder_parent.recv() == "holder-ready"
            message = conflict.result(timeout=5)
        assert str(process.pid) in message
        assert str(run) in message
    finally:
        _send_if_open(metadata_parent, "allow-metadata")
        _send_if_open(holder_parent, "release-holder")
        child_exitcode, child_terminated = _join_or_terminate(process)
        metadata_parent.close()
        holder_parent.close()
        process.close()
    assert child_terminated is False
    assert child_exitcode == 0


def test_release_transition_never_exposes_empty_owner_diagnostics(
    tmp_path, monkeypatch
):
    """Metadata removal and protected unlock must be one serialized transition."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    holder_parent, holder_child = context.Pipe()
    transition_parent, transition_child = context.Pipe()
    delayed_fs = _DelayedReleaseFilesystemOps(
        checkpoint.OS_FILESYSTEM,
        transition=transition_child,
    )
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", delayed_fs)
    process = context.Process(
        target=_hold_run_writer_with_fs,
        args=(str(run), holder_child, delayed_fs),
    )
    process.start()
    holder_child.close()
    transition_child.close()
    child_exitcode = None
    child_terminated = False
    try:
        assert holder_parent.poll(10), "child did not acquire the run writer lock"
        assert holder_parent.recv() == "holder-ready"
        holder_parent.send("release-holder")
        assert transition_parent.poll(10), "child did not reach release transition"
        assert transition_parent.recv() == "transition-started"
        outcome, message = _run_writer_outcome(run)
        if outcome == "conflict":
            assert "owner diagnostics unavailable" not in message
            assert str(process.pid) in message
            assert str(run) in message
        else:
            assert outcome == "acquired"
    finally:
        _send_if_open(transition_parent, "allow-transition")
        _send_if_open(holder_parent, "release-holder")
        child_exitcode, child_terminated = _join_or_terminate(process)
        holder_parent.close()
        transition_parent.close()
        process.close()
    assert child_terminated is False
    assert child_exitcode == 0


def test_dead_shared_holder_is_pruned_before_conflict_diagnostics(tmp_path):
    """An abrupt shared-holder exit must not displace the live holder diagnostic."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    dead_holder = context.Process(
        target=_acquire_run_access_then_exit,
        args=(str(run), sending),
    )
    dead_holder.start()
    sending.close()
    dead_pid = dead_holder.pid
    dead_exitcode = None
    dead_terminated = False
    try:
        assert receiving.poll(10), "abrupt holder did not acquire shared access"
        assert receiving.recv() == "locked"
    finally:
        dead_exitcode, dead_terminated = _join_or_terminate(dead_holder)
        receiving.close()
        dead_holder.close()
    assert dead_terminated is False
    assert dead_exitcode == 0

    live_parent, live_child = context.Pipe()
    live_holder = context.Process(
        target=_hold_run_access,
        args=(str(run), live_child),
    )
    live_holder.start()
    live_child.close()
    live_exitcode = None
    live_terminated = False
    try:
        assert live_parent.poll(10), "live holder did not acquire shared access"
        assert live_parent.recv() == "holder-ready"
        with (
            pytest.raises(SMLArtifactError) as conflict,
            run_access_lock(run, exclusive=True),
        ):
            pytest.fail("exclusive contender entered while shared holder was live")
        message = str(conflict.value)
        assert str(live_holder.pid) in message
        assert str(dead_pid) not in message
        assert str(run) in message
    finally:
        _send_if_open(live_parent, "release-holder")
        live_exitcode, live_terminated = _join_or_terminate(live_holder)
        live_parent.close()
        live_holder.close()
    assert live_terminated is False
    assert live_exitcode == 0


def test_lock_is_released_on_process_exit(tmp_path):
    """Only kernel flock ownership may determine whether a stale sidecar is live."""
    target = tmp_path / "bundle"
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(
        target=_acquire_publication_lock_then_exit,
        args=(str(target), sending),
    )
    process.start()
    sending.close()
    try:
        assert receiving.poll(10), "child did not acquire the publication lock"
        assert receiving.recv() == "locked"
        process.join(timeout=10)
        assert process.exitcode == 0

        with publication_lock(target):
            pass
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        receiving.close()
        process.close()


def test_lock_probe_failure_closes_parent_descriptor(tmp_path, monkeypatch):
    """An unexpected APFS authority failure must not leak the parent descriptor."""
    target = tmp_path / "bundle"
    probed_descriptors: list[int] = []

    def failing_probe(descriptor: int) -> bool:
        probed_descriptors.append(descriptor)
        raise RuntimeError("probe failed")

    monkeypatch.setattr(
        checkpoint,
        "_descriptor_filesystem_capability",
        failing_probe,
    )

    with (
        pytest.raises(RuntimeError, match="probe failed"),
        publication_lock(target),
    ):
        pytest.fail("lock entered without writable filesystem authority")

    assert len(probed_descriptors) == 1
    descriptor = probed_descriptors[0]
    try:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
