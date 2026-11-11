from __future__ import annotations

import os
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import checkpoint
from sml.artifacts import manifest as artifact_manifest
from sml.artifacts.manifest import (
    VerificationLevel,
)
from sml.errors import SMLArtifactError

from ._checkpoints import (
    InjectedFailure,
    RecordingFilesystemOps,
    _checkpoint_builder,
    _publish_step,
    _run_manifest,
)


class _SwapRunDuringRecoveryFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, replacement: Path) -> None:
        super().__init__(run)
        self._replacement = replacement
        self.swapped = False

    def listdir(self, descriptor: int) -> list[str]:
        if descriptor == self._checkpoints_descriptor and not self.swapped:
            self._run.rename(self._run.with_name("moved-original-run"))
            self._replacement.rename(self._run)
            self.swapped = True
        return super().listdir(descriptor)


class _SwapCandidateAtDetachFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, *, candidate: str, replacement: Path) -> None:
        super().__init__(run)
        self._candidate = candidate
        self._replacement = replacement
        self.swapped = False

    def rename(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        if (
            not self.swapped
            and source_dir_fd == self._checkpoints_descriptor
            and os.fspath(source) == self._candidate
            and os.fspath(destination).startswith(".sml-tmp-step-")
        ):
            candidate = self._run / "checkpoints" / self._candidate
            candidate.rename(candidate.with_name("parked-original-candidate"))
            self._replacement.rename(candidate)
            self.swapped = True
        super().rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )


class _SwapLatestIntoCandidateFilesystemOps(RecordingFilesystemOps):
    def __init__(
        self,
        run: Path,
        *,
        candidate: str,
        latest: str,
        latest_replacement: Path,
    ) -> None:
        super().__init__(run)
        self._candidate = candidate
        self._latest = latest
        self._latest_replacement = latest_replacement
        self._candidate_open_count = 0
        self._swapped = False

    def _swap(self) -> None:
        if self._swapped:
            return
        self._swapped = True
        checkpoints = self._run / "checkpoints"
        (checkpoints / self._candidate).rename(
            checkpoints / "parked-original-candidate"
        )
        (checkpoints / self._latest).rename(checkpoints / self._candidate)
        self._latest_replacement.rename(checkpoints / self._latest)

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            dir_fd == self._checkpoints_descriptor
            and os.fspath(path) == self._candidate
        ):
            self._candidate_open_count += 1
            if self._candidate_open_count == 2:
                self._swap()
        return super().open(path, flags, mode, dir_fd=dir_fd)

    def rename(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        if (
            source_dir_fd == self._checkpoints_descriptor
            and os.fspath(source) == self._candidate
            and os.fspath(destination).startswith(".sml-tmp-step-")
        ):
            self._swap()
        super().rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    @property
    def swapped(self) -> bool:
        return self._swapped


class _SwapLatestIntoCleanupTemporaryFilesystemOps(RecordingFilesystemOps):
    def __init__(
        self,
        run: Path,
        *,
        temporary: str,
        latest: str,
        latest_replacement: Path,
    ) -> None:
        super().__init__(run)
        self._temporary = temporary
        self._latest = latest
        self._latest_replacement = latest_replacement
        self.swapped = False

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            not self.swapped
            and dir_fd == self._checkpoints_descriptor
            and os.fspath(path) == self._temporary
        ):
            checkpoints = self._run / "checkpoints"
            (checkpoints / self._temporary).rename(
                checkpoints / "parked-original-cleanup-temporary"
            )
            (checkpoints / self._latest).rename(checkpoints / self._temporary)
            self._latest_replacement.rename(checkpoints / self._latest)
            self.swapped = True
        return super().open(path, flags, mode, dir_fd=dir_fd)


class _MutateLatestBeforeFreshProofFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, *, latest: str) -> None:
        super().__init__(run)
        self._latest = latest
        self._latest_open_count = 0

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd == self._checkpoints_descriptor and os.fspath(path) == self._latest:
            self._latest_open_count += 1
            if self._latest_open_count == 2:
                (
                    self._run / "checkpoints" / self._latest / "model.safetensors"
                ).write_bytes(b"mutated-before-fresh-proof")
        return super().open(path, flags, mode, dir_fd=dir_fd)


class _FailAfterFirstUnlinkFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path) -> None:
        super().__init__(run)
        self._failed = False

    def unlink(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        super().unlink(path, dir_fd=dir_fd)
        if not self._failed:
            self._failed = True
            raise InjectedFailure("mid-delete")


class _SignalExclusiveAccessAttemptFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, attempted: threading.Event) -> None:
        super().__init__(run)
        self._attempted = attempted

    def flock(self, descriptor: int, operation: int) -> None:
        if (
            ".sml-run-access-lock-" in self._descriptor_names.get(descriptor, "")
            and operation & checkpoint.fcntl.LOCK_EX
            and operation & checkpoint.fcntl.LOCK_NB
        ):
            self._attempted.set()
        super().flock(descriptor, operation)


def test_publish_and_prune_reuses_only_unchanged_checkpoint_proofs(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proofs: Counter[int] = Counter()
    hashes: Counter[str] = Counter()
    real_verify = checkpoint._verify_checkpoint_semantics
    real_identity = artifact_manifest.file_identity

    def record_proof(descriptor, manifest, verification, **kwargs):
        proofs[manifest.step] += 1
        return real_verify(descriptor, manifest, verification, **kwargs)

    def record_hash(stream):
        identity = real_identity(stream)
        hashes[identity] += 1
        return identity

    monkeypatch.setattr(checkpoint, "_verify_checkpoint_semantics", record_proof)
    monkeypatch.setattr(artifact_manifest, "file_identity", record_hash)
    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.publish_and_prune_checkpoint(
            valid_run, _checkpoint_builder(_run_manifest(), step=2)
        )

    assert proofs == {1: 1, 2: 2}
    for reference in checkpoint.checkpoint_array_payloads(retained.checkpoint):
        assert hashes[reference.payload.identity] == 2
    assert retained.pruning_pending is False
    assert [path.name for path in (valid_run / "checkpoints").iterdir()] == [
        "step-000000002"
    ]


@pytest.mark.parametrize("changed_step", (1, 2))
def test_publish_and_prune_rejects_changed_cached_payload_before_deletion(
    valid_run: Path,
    changed_step: int,
) -> None:
    class MutatingFilesystemOps(RecordingFilesystemOps):
        def _complete(self, stage):
            super()._complete(stage)
            if stage == "latest-parent-fsynced":
                payload = (
                    valid_run
                    / "checkpoints"
                    / f"step-{changed_step:09d}"
                    / "optimizer.safetensors"
                )
                original = payload.read_bytes()
                payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    fs = MutatingFilesystemOps(valid_run)
    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity mismatch"),
    ):
        checkpoint.publish_and_prune_checkpoint(
            valid_run, _checkpoint_builder(_run_manifest(), step=2), fs=fs
        )

    assert fs.delete_count == 0
    assert all(
        (valid_run / "checkpoints" / f"step-{step:09d}").is_dir() for step in (1, 2)
    )


def test_publish_and_prune_reproves_replaced_identical_payload(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReplacingFilesystemOps(RecordingFilesystemOps):
        def _complete(self, stage):
            super()._complete(stage)
            if stage == "latest-parent-fsynced":
                payload = (
                    valid_run / "checkpoints" / "step-000000002" / "model.safetensors"
                )
                replacement = payload.with_suffix(".replacement")
                replacement.write_bytes(payload.read_bytes())
                replacement.replace(payload)

    real_verify = checkpoint._verify_checkpoint_semantics
    proofs: Counter[int] = Counter()

    def record_proof(descriptor, manifest, verification, **kwargs):
        proofs[manifest.step] += 1
        return real_verify(descriptor, manifest, verification, **kwargs)

    monkeypatch.setattr(checkpoint, "_verify_checkpoint_semantics", record_proof)
    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.publish_and_prune_checkpoint(
            valid_run,
            _checkpoint_builder(_run_manifest(), step=2),
            fs=ReplacingFilesystemOps(valid_run),
        )

    assert proofs == {1: 1, 2: 3}
    assert retained.step == 2
    assert retained.pruning_pending is False


def test_read_only_resolution_reports_pending_pruning_until_retention_finishes(
    valid_run: Path,
) -> None:
    """Published older steps remain visible as read-only operational state."""
    _publish_step(valid_run, 2)

    resolved = checkpoint.resolve_latest_step(
        valid_run,
        writable=False,
        verification=VerificationLevel.FULL,
    )

    assert resolved.latest_recovered is False
    assert resolved.pruning_pending is True

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run)

    assert retained.pruning_pending is False
    assert (
        checkpoint.resolve_latest_step(
            valid_run,
            writable=False,
            verification=VerificationLevel.FULL,
        ).pruning_pending
        is False
    )


def test_retention_waits_for_active_reader(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive retention must wait until every shared reader releases."""
    _publish_step(valid_run, 2)
    reader_ready = threading.Event()
    release_reader = threading.Event()
    exclusive_attempted = threading.Event()
    signaling_fs = _SignalExclusiveAccessAttemptFilesystemOps(
        valid_run,
        exclusive_attempted,
    )
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", signaling_fs)

    def hold_reader() -> None:
        with checkpoint.run_access_lock(valid_run, exclusive=False):
            reader_ready.set()
            if not release_reader.wait(10):
                raise RuntimeError("timed out waiting to release reader")

    def retain_owned() -> checkpoint.ResolvedStep:
        with checkpoint.run_writer_lock(valid_run):
            return checkpoint.prune_to_latest(valid_run)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reader = executor.submit(hold_reader)
        assert reader_ready.wait(5)
        retention = executor.submit(retain_owned)
        try:
            assert exclusive_attempted.wait(5)
            assert not retention.done()
            assert (valid_run / "checkpoints" / "step-000000001").is_dir()
        finally:
            release_reader.set()
        reader.result(timeout=5)
        result = retention.result(timeout=5)

    assert result.step == 2


def test_retention_never_deletes_latest(valid_run: Path) -> None:
    """Finite retention must preserve the authoritative latest checkpoint."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run)

    assert retained.step == 3
    assert sorted(path.name for path in (valid_run / "checkpoints").iterdir()) == [
        "step-000000003"
    ]


@pytest.mark.parametrize("checkpoint_count", (2, 4))
def test_checkpoint_maintenance_keeps_tensor_memory_bounded(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_count: int,
) -> None:
    """Publication, resolution, and retention need proofs without full snapshots."""

    def reject_tensor_load(*_args, **_kwargs):
        pytest.fail("checkpoint maintenance materialized safetensors payloads")

    monkeypatch.setattr(mx, "load", reject_tensor_load)
    for step in range(2, checkpoint_count + 1):
        _publish_step(valid_run, step)

    real_delete = checkpoint._detach_and_delete_owned_step
    deleted_steps: list[int] = []

    def check_retained_contents(*args, candidate, latest, **kwargs):
        for owned in (candidate, latest):
            assert owned.contents is not None
            assert owned.contents.array_groups == {}
            assert owned.contents.payload_bytes == {}
            assert owned.contents.boundary_state is not None
        deleted_steps.append(candidate.resolved.step)
        return real_delete(*args, candidate=candidate, latest=latest, **kwargs)

    monkeypatch.setattr(
        checkpoint,
        "_detach_and_delete_owned_step",
        check_retained_contents,
    )
    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run)

    assert deleted_steps == list(range(1, checkpoint_count))
    assert retained.step == checkpoint_count
    assert retained.verification is VerificationLevel.FULL
    assert sorted(path.name for path in (valid_run / "checkpoints").iterdir()) == [
        f"step-{checkpoint_count:09d}"
    ]


@pytest.mark.parametrize("corrupt_step", (1, 2, 3))
def test_streaming_retention_rejects_equal_size_payload_corruption(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_step: int,
) -> None:
    """Removing tensor materialization must preserve every candidate's full hash."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    payload = (
        valid_run / "checkpoints" / f"step-{corrupt_step:09d}" / "optimizer.safetensors"
    )
    encoded = payload.read_bytes()
    payload.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))

    def reject_tensor_load(*_args, **_kwargs):
        pytest.fail("checkpoint retention materialized safetensors payloads")

    monkeypatch.setattr(mx, "load", reject_tensor_load)
    fs = RecordingFilesystemOps(valid_run)
    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity mismatch"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.delete_count == 0
    assert all(
        (valid_run / "checkpoints" / f"step-{step:09d}").is_dir() for step in (1, 2, 3)
    )


def test_retention_reports_persisted_latest_recovery(valid_run: Path) -> None:
    """Retention must preserve the recovery outcome returned by its first phase."""
    _publish_step(valid_run, 2)
    (valid_run / "latest.json").unlink()

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run)

    assert retained.step == 2
    assert retained.latest_recovered is True
    assert retained.latest_repair_persisted is True


def test_retention_run_path_swap_uses_retained_original_descriptor(
    valid_run: Path,
) -> None:
    """A real path swap during recovery must not redirect later retention work."""
    _publish_step(valid_run, 2)
    replacement = valid_run.with_name("replacement-run")
    shutil.copytree(valid_run, replacement)
    sentinel = replacement / "decoy-must-survive.bin"
    sentinel.write_bytes(b"decoy-run")
    fs = _SwapRunDuringRecoveryFilesystemOps(valid_run, replacement)

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run, fs=fs)

    assert retained.step == 2
    assert fs.swapped is True
    moved_original = valid_run.with_name("moved-original-run")
    assert sorted(path.name for path in (moved_original / "checkpoints").iterdir()) == [
        "step-000000002"
    ]
    assert sorted(path.name for path in (valid_run / "checkpoints").iterdir()) == [
        "step-000000001",
        "step-000000002",
    ]
    assert (valid_run / sentinel.name).read_bytes() == b"decoy-run"


def test_retention_candidate_swap_at_detach_preserves_replacement(
    valid_run: Path,
) -> None:
    """A real canonical swap at detach must fail before replacement deletion."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    replacement = valid_run.with_name("candidate-replacement")
    replacement.mkdir()
    sentinel = replacement / "must-survive.bin"
    sentinel.write_bytes(b"outside-candidate")
    fs = _SwapCandidateAtDetachFilesystemOps(
        valid_run,
        candidate="step-000000001",
        replacement=replacement,
    )

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="inode|bound|swapped|candidate"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.swapped is True
    assert (valid_run / "checkpoints" / "parked-original-candidate").is_dir()
    surviving_sentinels = list((valid_run / "checkpoints").glob("*/must-survive.bin"))
    assert len(surviving_sentinels) == 1
    assert surviving_sentinels[0].read_bytes() == b"outside-candidate"


def test_retention_latest_swap_never_deletes_proved_latest(valid_run: Path) -> None:
    """Latest must stay inode-bound across candidate detach and recursive deletion."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    latest_path = valid_run / "checkpoints" / "step-000000003"
    latest_stat = latest_path.stat()
    latest_inode = (latest_stat.st_dev, latest_stat.st_ino)
    replacement = valid_run.with_name("latest-replacement")
    replacement.mkdir()
    (replacement / "sentinel.bin").write_bytes(b"replacement-latest")
    fs = _SwapLatestIntoCandidateFilesystemOps(
        valid_run,
        candidate="step-000000001",
        latest="step-000000003",
        latest_replacement=replacement,
    )

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="inode|bound|swapped|latest|candidate"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.swapped is True
    surviving_inodes = {
        (entry.stat().st_dev, entry.stat().st_ino)
        for entry in (valid_run / "checkpoints").iterdir()
        if entry.is_dir()
    }
    assert latest_inode in surviving_inodes


def test_retention_cleanup_never_deletes_latest_swapped_under_exact_temp(
    valid_run: Path,
) -> None:
    """Retry cleanup must reject a temp rebound to the retained latest inode."""
    _publish_step(valid_run, 2)
    checkpoints = valid_run / "checkpoints"
    temporary_name = ".sml-tmp-step-" + "a" * 32
    cleanup_temporary = checkpoints / temporary_name
    cleanup_temporary.mkdir()
    (cleanup_temporary / "old-temp.bin").write_bytes(b"old-temp")
    latest_path = checkpoints / "step-000000002"
    latest_stat = latest_path.stat()
    latest_inode = (latest_stat.st_dev, latest_stat.st_ino)
    latest_replacement = valid_run.with_name("cleanup-latest-replacement")
    latest_replacement.mkdir()
    (latest_replacement / "replacement.bin").write_bytes(b"replacement")
    fs = _SwapLatestIntoCleanupTemporaryFilesystemOps(
        valid_run,
        temporary=temporary_name,
        latest="step-000000002",
        latest_replacement=latest_replacement,
    )

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="inode|bound|swapped|latest|temporary"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.swapped is True
    assert fs.delete_count == 0
    surviving_inodes = {
        (entry.stat().st_dev, entry.stat().st_ino)
        for entry in checkpoints.iterdir()
        if entry.is_dir()
    }
    assert latest_inode in surviving_inodes


def test_retention_mutated_latest_before_fresh_proof_deletes_nothing(
    valid_run: Path,
) -> None:
    """The fresh delete-authorizing proof must occur after candidate prevalidation."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    fs = _MutateLatestBeforeFreshProofFilesystemOps(
        valid_run,
        latest="step-000000003",
    )

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity|byte size"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.delete_count == 0
    assert (valid_run / "checkpoints" / "step-000000001").is_dir()


def test_retention_mid_delete_failure_detaches_and_retry_cleans(
    valid_run: Path,
) -> None:
    """A delete failure may leave only an owned temp and must be retryable."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    fs = _FailAfterFirstUnlinkFilesystemOps(valid_run)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(InjectedFailure, match="mid-delete"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    checkpoint_names = sorted(
        entry.name for entry in (valid_run / "checkpoints").iterdir()
    )
    assert "step-000000001" not in checkpoint_names
    assert any(name.startswith(".sml-tmp-step-") for name in checkpoint_names)
    assert fs.checkpoints_fsync_count >= 1
    for name in checkpoint_names:
        if name.startswith("step-"):
            checkpoint.resolve_exact_step(
                valid_run,
                step=int(name.removeprefix("step-")),
                verification=VerificationLevel.FULL,
            )

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.prune_to_latest(valid_run)

    assert retained.step == 3
    assert sorted(entry.name for entry in (valid_run / "checkpoints").iterdir()) == [
        "step-000000003"
    ]


def test_retention_requires_current_full_proof_before_first_delete(
    valid_run: Path,
) -> None:
    """Corrupt retained bytes must abort retention before any deletion syscall."""
    _publish_step(valid_run, 2)
    (valid_run / "checkpoints" / "step-000000002" / "model.safetensors").write_bytes(
        b"tampered"
    )
    fs = RecordingFilesystemOps(valid_run)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity|byte size"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.delete_count == 0
    assert (valid_run / "checkpoints" / "step-000000001").is_dir()
    assert (valid_run / "checkpoints" / "step-000000002").is_dir()


def test_retention_validates_all_candidates_before_first_delete(
    valid_run: Path,
) -> None:
    """A corrupt eligible old step must prevent every planned deletion."""
    _publish_step(valid_run, 2)
    _publish_step(valid_run, 3)
    (valid_run / "checkpoints" / "step-000000002" / "state.json").write_bytes(
        b"corrupt-state"
    )
    fs = RecordingFilesystemOps(valid_run)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity|byte size"),
    ):
        checkpoint.prune_to_latest(valid_run, fs=fs)

    assert fs.delete_count == 0
    assert all(
        (valid_run / "checkpoints" / f"step-{step:09d}").is_dir() for step in (1, 2, 3)
    )
