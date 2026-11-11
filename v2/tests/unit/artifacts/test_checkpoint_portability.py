from __future__ import annotations

import errno
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    VerificationLevel,
    canonical_json_bytes,
)
from sml.errors import SMLArtifactError

from ._checkpoints import (
    RecordingFilesystemOps,
    _publish_step,
)


class _FailRunAccessSidecarOpenFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, error_number: int) -> None:
        super().__init__(run)
        self._error_number = error_number

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            dir_fd is not None
            and os.fspath(path).startswith(".sml-run-access-")
            and flags & os.O_RDWR
        ):
            raise OSError(self._error_number, os.strerror(self._error_number))
        return super().open(path, flags, mode, dir_fd=dir_fd)


def _copy_portable_run(valid_run: Path, *, parent_name: str) -> Path:
    parent = valid_run.parent / parent_name
    parent.mkdir()
    run = parent / valid_run.name
    shutil.copytree(valid_run, run)
    return run


def _force_filesystem_capability(
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
) -> None:
    def classify(_descriptor: int):
        return checkpoint._FilesystemCapability(capability)

    monkeypatch.setattr(
        checkpoint,
        "_descriptor_filesystem_capability",
        classify,
        raising=False,
    )


@pytest.mark.parametrize("alias", [".", "..", "../.."])
def test_current_and_parent_aliases_share_run_locks_and_read_latest(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    working_directory = valid_run
    for index, _parent in enumerate(Path(alias).parts):
        working_directory /= f"child-{index}"
    working_directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(working_directory)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="run-writer lock"),
        checkpoint.run_writer_lock(Path(alias)),
    ):
        pytest.fail("directory alias bypassed the existing run writer lock")

    with checkpoint.open_latest_checkpoint_reader(
        Path(alias), verification=VerificationLevel.MANIFEST_TRUSTED
    ) as reader:
        assert reader.resolved.step == 1


@pytest.mark.parametrize("exclusive", [False, True], ids=["reader", "retention"])
def test_run_writer_does_not_block_run_access(tmp_path: Path, exclusive: bool) -> None:
    run = tmp_path / "run-0001"
    run.mkdir()

    def access_from_another_thread() -> str:
        with checkpoint.run_access_lock(run, exclusive=exclusive):
            return "acquired"

    with checkpoint.run_writer_lock(run), ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            executor.submit(access_from_another_thread).result(timeout=5) == "acquired"
        )


def test_exclusive_run_access_conflicts_with_shared_access_holder(
    tmp_path: Path,
) -> None:
    """Separate access sidecars must still serialize readers against deletion."""
    run = tmp_path / "run-0001"
    run.mkdir()

    with (
        checkpoint.run_access_lock(run, exclusive=False),
        pytest.raises(
            SMLArtifactError,
            match="access lock|held by",
        ),
        checkpoint.run_access_lock(run, exclusive=True),
    ):
        pytest.fail("exclusive access entered while a shared reader was live")


def test_local_non_apfs_latest_reader_is_portable_and_never_mutates_sidecars(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing a local foreign filesystem through APFS locks breaks portability."""
    _publish_step(valid_run, 2)
    (valid_run / "latest.json").write_text("not-json", encoding="utf-8")
    run = _copy_portable_run(valid_run, parent_name="local-other-parent")
    _force_filesystem_capability(monkeypatch, "local-other")
    parent_entries = sorted(path.name for path in run.parent.iterdir())
    latest_bytes = (run / "latest.json").read_bytes()
    checkpoint_entries = sorted(path.name for path in (run / "checkpoints").iterdir())
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with checkpoint.open_latest_checkpoint_reader(
            run,
            verification=VerificationLevel.FULL,
            load_array_groups=frozenset(),
            run_descriptor=descriptor,
        ) as reader:
            reader.read_contents()
            assert reader.resolved.step == 2
            assert reader.resolved.latest_recovered is True
            assert reader.resolved.latest_repair_persisted is False
            assert reader.resolved.pruning_pending is True
        os.fstat(descriptor)
    finally:
        os.close(descriptor)

    assert sorted(path.name for path in run.parent.iterdir()) == parent_entries
    assert (run / "latest.json").read_bytes() == latest_bytes
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == (
        checkpoint_entries
    )


@pytest.mark.parametrize("sidecar_error", (errno.EACCES, errno.EPERM, errno.EROFS))
def test_local_apfs_latest_reader_falls_back_before_nonwritable_sidecar_creation(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_error: int,
) -> None:
    """Removing the permission fallback would reject a valid read-only copy."""
    run = _copy_portable_run(valid_run, parent_name="nonwritable-parent")
    parent = run.parent
    original_mode = stat.S_IMODE(parent.stat().st_mode)
    _force_filesystem_capability(monkeypatch, "local-apfs")
    monkeypatch.setattr(
        checkpoint,
        "OS_FILESYSTEM",
        _FailRunAccessSidecarOpenFilesystemOps(run, sidecar_error),
    )
    parent.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with checkpoint.open_latest_checkpoint_reader(
            run,
            verification=VerificationLevel.FULL,
            load_array_groups=frozenset(),
        ) as reader:
            assert reader.resolved.step == 1
    finally:
        parent.chmod(original_mode)

    assert sorted(path.name for path in parent.iterdir()) == [run.name]


@pytest.mark.parametrize(
    "mutation_stage",
    ("lock-created", "diagnostic-created", "stale-owner-write", "new-owner-write"),
)
@pytest.mark.parametrize("sidecar_error", (errno.EACCES, errno.EPERM, errno.EROFS))
def test_latest_reader_rejects_authority_failure_after_sidecar_mutation(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_stage: str,
    sidecar_error: int,
) -> None:
    """Partial lock setup must never be downgraded to an unlocked latest reader."""
    run = _copy_portable_run(valid_run, parent_name="mutated-sidecar-parent")
    lock_name = checkpoint._lock_name(run.name, "run-access")
    diagnostic_name = checkpoint._diagnostic_name(run.name, "run-access")
    lock_path = run.parent / lock_name
    diagnostic_path = run.parent / diagnostic_name
    if mutation_stage != "lock-created":
        lock_path.write_bytes(b"")
    if mutation_stage in {"stale-owner-write", "new-owner-write"}:
        holders = (
            [{"pid": 0, "protected_path": str(run), "token": "stale-owner"}]
            if mutation_stage == "stale-owner-write"
            else []
        )
        diagnostic_path.write_bytes(canonical_json_bytes({"holders": holders}))

    class FailAfterSidecarMutation(RecordingFilesystemOps):
        diagnostic_descriptor: int | None = None
        fault_raised = False

        def fail(self) -> None:
            self.fault_raised = True
            raise OSError(sidecar_error, os.strerror(sidecar_error))

        def open(self, path, flags, mode=0o777, *, dir_fd=None):
            if mutation_stage == "lock-created" and os.fspath(path) == diagnostic_name:
                assert lock_path.is_file()
                self.fail()
            descriptor = super().open(path, flags, mode, dir_fd=dir_fd)
            if os.fspath(path) == diagnostic_name:
                self.diagnostic_descriptor = descriptor
            return descriptor

        def stat(self, path, *, dir_fd=None, follow_symlinks=False):
            if (
                mutation_stage == "diagnostic-created"
                and self.diagnostic_descriptor is not None
                and path == self.diagnostic_descriptor
            ):
                assert diagnostic_path.is_file()
                self.fail()
            return super().stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        def write_all(self, descriptor, data):
            if descriptor == self.diagnostic_descriptor:
                # The real diagnostic writer has already truncated the file.
                assert os.fstat(descriptor).st_size == 0
                if mutation_stage == "stale-owner-write":
                    assert data == b'{"holders":[]}'
                else:
                    assert mutation_stage == "new-owner-write"
                    assert str(os.getpid()).encode() in data
                self.fail()
            super().write_all(descriptor, data)

    fs = FailAfterSidecarMutation(run)
    _force_filesystem_capability(monkeypatch, "local-apfs")
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", fs)

    def forbidden_recovery(*_args, **_kwargs):
        pytest.fail("latest recovery entered after sidecar mutation")

    monkeypatch.setattr(checkpoint, "_recover_latest_open", forbidden_recovery)
    with (
        pytest.raises(SMLArtifactError, match="run-access lock") as caught,
        checkpoint.open_latest_checkpoint_reader(run),
    ):
        pytest.fail("mutated sidecar protocol yielded a reduced-mode reader")

    assert fs.fault_raised
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == sidecar_error
    if mutation_stage in {"stale-owner-write", "new-owner-write"}:
        assert diagnostic_path.read_bytes() == b""


@pytest.mark.parametrize(
    "operation",
    (
        "writable-recovery",
        "checkpoint-publication",
        "pruning",
        "shared-run-access",
        "run-writer",
        "immutable-publication",
    ),
)
def test_local_non_apfs_writable_operations_remain_rejected(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Reusing the reduced reader mode for mutation would weaken APFS guarantees."""
    run = _copy_portable_run(valid_run, parent_name=f"writable-{operation}")
    _force_filesystem_capability(monkeypatch, "local-other")
    monkeypatch.setattr(
        checkpoint,
        "_descriptor_is_local_apfs",
        lambda _descriptor: False,
    )

    with pytest.raises(SMLArtifactError, match="local APFS"):
        if operation == "writable-recovery":
            checkpoint.recover_latest_index(
                run,
                writable=True,
                verification=VerificationLevel.FULL,
            )
        elif operation == "checkpoint-publication":
            checkpoint.publish_checkpoint(
                run,
                lambda _private: pytest.fail("builder ran off local APFS"),
            )
        elif operation == "pruning":
            checkpoint.prune_to_latest(run)
        elif operation == "shared-run-access":
            with checkpoint.run_access_lock(run, exclusive=False):
                pytest.fail("run-access lock entered off local APFS")
        elif operation == "run-writer":
            with checkpoint.run_writer_lock(run):
                pytest.fail("run-writer lock entered off local APFS")
        else:
            assert operation == "immutable-publication"
            with checkpoint.publication_lock(run.parent / "bundle"):
                pytest.fail("publication lock entered off local APFS")


def test_non_local_latest_reader_fails_before_descriptor_traversal_or_sidecars(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating non-local storage as reduced local mode would overstate safety."""
    run = _copy_portable_run(valid_run, parent_name="non-local-parent")
    _force_filesystem_capability(monkeypatch, "non-local")
    monkeypatch.setattr(
        checkpoint,
        "_descriptor_is_local_apfs",
        lambda _descriptor: True,
    )
    parent_entries = sorted(path.name for path in run.parent.iterdir())
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with (
            pytest.raises(SMLArtifactError, match="non-local"),
            checkpoint.open_latest_checkpoint_reader(
                run,
                verification=VerificationLevel.FULL,
                run_descriptor=descriptor,
            ),
        ):
            pytest.fail("non-local latest reader entered")
        os.fstat(descriptor)
    finally:
        os.close(descriptor)

    assert sorted(path.name for path in run.parent.iterdir()) == parent_entries


def test_latest_reader_does_not_downgrade_lock_contention(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catching lock contention as no-lock mode would race checkpoint pruning."""
    _force_filesystem_capability(monkeypatch, "local-apfs")

    @contextmanager
    def contended_lock(*_args, **_kwargs):
        raise checkpoint._LockUnavailable("injected run-access contention")
        yield

    def forbidden_recovery(*_args, **_kwargs):
        pytest.fail("latest recovery entered after lock contention")

    monkeypatch.setattr(checkpoint, "_protected_lock", contended_lock)
    monkeypatch.setattr(checkpoint, "_recover_latest_open", forbidden_recovery)

    with (
        pytest.raises(checkpoint._LockUnavailable, match="contention"),
        checkpoint.open_latest_checkpoint_reader(valid_run),
    ):
        pytest.fail("contended latest reader entered reduced mode")


@pytest.mark.parametrize("sidecar_kind", ("symlink", "hard-link"))
def test_latest_reader_does_not_downgrade_corrupt_sidecars(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_kind: str,
) -> None:
    """Ignoring structural sidecar corruption would bypass lock ownership proof."""
    run = _copy_portable_run(valid_run, parent_name=f"corrupt-{sidecar_kind}")
    _force_filesystem_capability(monkeypatch, "local-apfs")
    monkeypatch.setattr(
        checkpoint,
        "_descriptor_is_local_apfs",
        lambda _descriptor: True,
    )
    sidecar = run.parent / checkpoint._lock_name(run.name, "run-access")
    if sidecar_kind == "symlink":
        sidecar.symlink_to(run / "latest.json")
    else:
        source = run.parent / "shared-sidecar-inode"
        source.write_bytes(b"")
        os.link(source, sidecar)

    def forbidden_recovery(*_args, **_kwargs):
        pytest.fail("latest recovery entered after sidecar corruption")

    monkeypatch.setattr(checkpoint, "_recover_latest_open", forbidden_recovery)
    with (
        pytest.raises(SMLArtifactError, match="lock"),
        checkpoint.open_latest_checkpoint_reader(run),
    ):
        pytest.fail("corrupt sidecar was downgraded to reduced mode")


def test_latest_reader_does_not_downgrade_unexpected_lock_io(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating EIO as read-only authority loss would hide storage failure."""
    run = _copy_portable_run(valid_run, parent_name="unexpected-io-parent")
    _force_filesystem_capability(monkeypatch, "local-apfs")
    monkeypatch.setattr(
        checkpoint,
        "_descriptor_is_local_apfs",
        lambda _descriptor: True,
    )
    monkeypatch.setattr(
        checkpoint,
        "OS_FILESYSTEM",
        _FailRunAccessSidecarOpenFilesystemOps(run, errno.EIO),
    )

    def forbidden_recovery(*_args, **_kwargs):
        pytest.fail("latest recovery entered after unexpected lock I/O")

    monkeypatch.setattr(checkpoint, "_recover_latest_open", forbidden_recovery)
    with (
        pytest.raises(SMLArtifactError, match="run-access lock") as caught,
        checkpoint.open_latest_checkpoint_reader(run),
    ):
        pytest.fail("unexpected lock I/O was downgraded to reduced mode")

    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.EIO
