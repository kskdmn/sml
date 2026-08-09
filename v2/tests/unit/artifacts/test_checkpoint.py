from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    CheckpointManifest,
    PayloadRef,
    RunManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
)
from sml.errors import SMLArtifactError

CHECKPOINT_STAGES = (
    "arrays-written",
    "scalar-state-written",
    "checkpoint-manifest-written",
    "step-directory-fsynced",
    "step-directory-renamed",
    "step-parent-fsynced",
    "latest-temporary-fsynced",
    "latest-replaced",
    "latest-parent-fsynced",
)


class InjectedFailure(RuntimeError):
    pass


class RecordingFilesystemOps:
    """Test-only wrapper mapping faults to completed checkpoint operations."""

    def __init__(
        self,
        run: Path,
        *,
        failure_stage: str | None = None,
    ) -> None:
        self._inner = checkpoint.OS_FILESYSTEM
        self._run = run
        self._failure_stage = failure_stage
        self._raised = False
        self._run_descriptor: int | None = None
        self._checkpoints_descriptor: int | None = None
        self._temporary_step_descriptor: int | None = None
        self._descriptor_names: dict[int, str] = {}
        self._step_directory_fsync_count = 0
        self.completed_stages: list[str] = []
        self.delete_count = 0

    @classmethod
    def raise_after(cls, stage: str, *, run: Path) -> RecordingFilesystemOps:
        return cls(run, failure_stage=stage)

    def _complete(self, stage: str) -> None:
        self.completed_stages.append(stage)
        if not self._raised and stage == self._failure_stage:
            self._raised = True
            raise InjectedFailure(stage)

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = self._inner.open(path, flags, mode, dir_fd=dir_fd)
        path_text = os.fspath(path)
        name = Path(path_text).name
        self._descriptor_names[descriptor] = name
        if dir_fd is None and Path(path_text) == self._run:
            self._run_descriptor = descriptor
        elif (
            dir_fd == self._run_descriptor
            and name == "checkpoints"
            and flags & os.O_DIRECTORY
        ):
            self._checkpoints_descriptor = descriptor
        elif (
            dir_fd == self._checkpoints_descriptor
            and name.startswith(".sml-tmp-step-")
            and flags & os.O_DIRECTORY
        ):
            self._temporary_step_descriptor = descriptor
        return descriptor

    def mkdir(
        self,
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int,
    ) -> None:
        self._inner.mkdir(path, mode, dir_fd=dir_fd)

    def write_all(self, descriptor: int, data: bytes) -> None:
        self._inner.write_all(descriptor, data)

    def fsync_file(self, descriptor: int) -> None:
        self._inner.fsync_file(descriptor)
        name = self._descriptor_names.get(descriptor, "")
        if name == "checkpoint.json":
            self._complete("checkpoint-manifest-written")
        elif name.startswith(".sml-tmp-latest-"):
            self._complete("latest-temporary-fsynced")

    def fsync_directory(self, descriptor: int) -> None:
        self._inner.fsync_directory(descriptor)
        if descriptor == self._temporary_step_descriptor:
            self._step_directory_fsync_count += 1
            stage = {
                1: "arrays-written",
                2: "scalar-state-written",
                3: "step-directory-fsynced",
            }.get(self._step_directory_fsync_count)
            if stage is not None:
                self._complete(stage)
        elif descriptor == self._checkpoints_descriptor:
            self._complete("step-parent-fsynced")
        elif descriptor == self._run_descriptor:
            self._complete("latest-parent-fsynced")

    def rename(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        self._inner.rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )
        self._complete("step-directory-renamed")

    def replace(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        self._inner.replace(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )
        self._complete("latest-replaced")

    def unlink(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        self.delete_count += 1
        self._inner.unlink(path, dir_fd=dir_fd)

    def rmdir(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        self.delete_count += 1
        self._inner.rmdir(path, dir_fd=dir_fd)

    def stat(
        self,
        path: int | str | os.PathLike[str],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = False,
    ) -> os.stat_result:
        return self._inner.stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def listdir(self, descriptor: int) -> list[str]:
        return self._inner.listdir(descriptor)

    def flock(self, descriptor: int, operation: int) -> None:
        self._inner.flock(descriptor, operation)


class _SwapRunOnSecondOpenFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, replacement: Path) -> None:
        super().__init__(run)
        self._replacement = replacement
        self._run_open_count = 0

    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None and Path(path) == self._run:
            self._run_open_count += 1
            if self._run_open_count == 2:
                self._run.rename(self._run.with_name("moved-original-run"))
                self._replacement.rename(self._run)
        return super().open(path, flags, mode, dir_fd=dir_fd)


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _run_manifest() -> RunManifest:
    manifest = RunManifest(
        kind="run",
        version=1,
        identity="sha256:" + "0" * 64,
        run_kind="pretraining",
        model={"rope_scaling_factor": 1.0},
        precision={"working": "bfloat16"},
        optimizer={"name": "adamw"},
        loader={"batch_size": 2},
        checkpoint={"keep_last": None},
        tokenizer_identity="sha256:" + "1" * 64,
        base_identity=None,
        data_identity="sha256:" + "2" * 64,
        diagnostic_data_locator="/relocatable/data",
        diagnostic_source_locator="/relocatable/run",
    )
    return replace(manifest, identity=manifest.recompute_identity())


def _checkpoint_builder(
    run_manifest: RunManifest,
    *,
    step: int,
    array_bytes: bytes | None = None,
) -> Callable[[Path], CheckpointManifest]:
    materialized_array_bytes = array_bytes or f"array-{step}".encode()

    def build(private_step: Path) -> CheckpointManifest:
        arrays_path = private_step / "arrays.safetensors"
        state_path = private_step / "state.json"
        arrays_path.write_bytes(materialized_array_bytes)
        state_path.write_bytes(f'{{"step":{step}}}'.encode())
        manifest = CheckpointManifest(
            kind="checkpoint",
            version=1,
            identity="sha256:" + "0" * 64,
            owning_run_identity=run_manifest.identity,
            checkpoint_kind=run_manifest.run_kind,
            step=step,
            scalar_state=_payload_ref(state_path, "state.json"),
            arrays=(
                ArrayPayloadRef(
                    payload=_payload_ref(arrays_path, "arrays.safetensors"),
                    arrays=(ArraySpec("model.weight", (1,), "float32"),),
                ),
            ),
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return build


@pytest.fixture
def valid_run(tmp_path: Path) -> Path:
    run = tmp_path / "run-0001"
    run.mkdir()
    (run / "checkpoints").mkdir()
    manifest = _run_manifest()
    (run / "run.json").write_bytes(canonical_json_bytes(manifest))
    with checkpoint.run_writer_lock(run):
        checkpoint.publish_checkpoint(run, _checkpoint_builder(manifest, step=1))
    return run


def _publish_step(run: Path, step: int) -> None:
    run_manifest = checkpoint.resolve_exact_step(
        run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run
    with checkpoint.run_writer_lock(run):
        checkpoint.publish_checkpoint(
            run,
            _checkpoint_builder(run_manifest, step=step),
        )


def _resolve_latest_owned(run: Path, *, writable: bool) -> checkpoint.ResolvedStep:
    if not writable:
        return checkpoint.resolve_latest_step(
            run,
            writable=False,
            verification=VerificationLevel.FULL,
        )
    with checkpoint.run_writer_lock(run):
        return checkpoint.resolve_latest_step(
            run,
            writable=True,
            verification=VerificationLevel.FULL,
        )


def test_run_writer_can_coexist_with_shared_run_access(tmp_path: Path) -> None:
    """Training ownership must not block a reader of immutable checkpoints."""
    run = tmp_path / "run-0001"
    run.mkdir()

    def read_from_another_thread() -> str:
        with checkpoint.run_access_lock(run, exclusive=False):
            return "readable"

    with checkpoint.run_writer_lock(run), ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(read_from_another_thread).result(timeout=5) == "readable"


def test_run_writer_does_not_block_exclusive_run_access(tmp_path: Path) -> None:
    """Access exclusion must contend only with access holders, not the writer."""
    run = tmp_path / "run-0001"
    run.mkdir()

    with (
        checkpoint.run_writer_lock(run),
        checkpoint.run_access_lock(
            run,
            exclusive=True,
        ),
    ):
        pass


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


def test_checkpoint_stage_tuple_is_exact_and_ordered() -> None:
    """Changing a checkpoint durability boundary must break the public mapping."""
    assert checkpoint.CHECKPOINT_PUBLICATION_STAGES == CHECKPOINT_STAGES


def test_exact_step_ignores_latest_and_malformed_newer_step(valid_run: Path) -> None:
    """Exact resolution must not consult derived or unrelated checkpoint state."""
    (valid_run / "latest.json").write_text("not-json", encoding="utf-8")
    (valid_run / "checkpoints" / "step-000000009").mkdir()

    resolved = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    )

    assert resolved.step == 1
    assert resolved.latest_recovered is False
    assert resolved.latest_repair_persisted is False
    assert resolved.verification is VerificationLevel.FULL


def test_exact_resolution_keeps_run_descriptor_bound(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run ownership proof and checkpoint traversal must share one run inode."""
    replacement = valid_run.with_name("replacement-run")
    shutil.copytree(valid_run, replacement)
    original = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    )
    replacement_run = replace(
        original.run,
        identity="sha256:" + "0" * 64,
        optimizer={"name": "different"},
    )
    replacement_run = replace(
        replacement_run,
        identity=replacement_run.recompute_identity(),
    )
    (replacement / "run.json").write_bytes(canonical_json_bytes(replacement_run))
    fs = _SwapRunOnSecondOpenFilesystemOps(valid_run, replacement)
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", fs)

    resolved = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    )

    assert resolved.run.identity == original.run.identity
    assert not valid_run.with_name("moved-original-run").exists()


def test_step_rename_before_latest_replace_recovers_new_step(valid_run: Path) -> None:
    """The step rename, rather than latest replacement, is the commit point."""
    run_manifest = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run
    fs = RecordingFilesystemOps.raise_after(
        "step-directory-renamed",
        run=valid_run,
    )

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(InjectedFailure, match="step-directory-renamed"),
    ):
        checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(run_manifest, step=2),
            fs=fs,
        )

    resolved = _resolve_latest_owned(valid_run, writable=True)
    assert resolved.step == 2
    assert resolved.latest_recovered is True
    assert resolved.latest_repair_persisted is True


@pytest.mark.parametrize("stage", CHECKPOINT_STAGES)
def test_checkpoint_interruption_recovery(stage: str, valid_run: Path) -> None:
    """Every fault boundary must recover exactly the last committed step."""
    run_manifest = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run
    fs = RecordingFilesystemOps.raise_after(stage, run=valid_run)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(InjectedFailure, match=stage),
    ):
        checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(run_manifest, step=2),
            fs=fs,
        )

    completed_index = CHECKPOINT_STAGES.index(stage)
    assert fs.completed_stages == list(CHECKPOINT_STAGES[: completed_index + 1])
    resolved = _resolve_latest_owned(
        valid_run,
        writable=stage not in CHECKPOINT_STAGES[:4],
    )
    expected_step = 1 if completed_index < 4 else 2
    assert resolved.step == expected_step
    assert resolved.verification is VerificationLevel.FULL


def test_read_only_recovery_never_persists_latest(valid_run: Path) -> None:
    """Read-only recovery must report derived repair state without writing it."""
    _publish_step(valid_run, 2)
    latest = valid_run / "latest.json"
    latest.write_text("not-json", encoding="utf-8")
    original_bytes = latest.read_bytes()

    resolved = checkpoint.resolve_latest_step(
        valid_run,
        writable=False,
        verification=VerificationLevel.FULL,
    )

    assert resolved.step == 2
    assert resolved.latest_recovered is True
    assert resolved.latest_repair_persisted is False
    assert latest.read_bytes() == original_bytes


def test_malformed_required_scan_candidate_fails(valid_run: Path) -> None:
    """A newer published candidate cannot be silently omitted from recovery."""
    malformed = valid_run / "checkpoints" / "step-000000002"
    malformed.mkdir()
    (malformed / "checkpoint.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(SMLArtifactError, match="candidate|checkpoint|manifest"):
        checkpoint.resolve_latest_step(
            valid_run,
            writable=False,
            verification=VerificationLevel.FULL,
        )


def test_older_idempotent_publication_cannot_move_latest_backward(
    valid_run: Path,
) -> None:
    """Replaying an old identical step must never regress the latest index."""
    _publish_step(valid_run, 2)
    run_manifest = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="older|backward|latest"),
    ):
        checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(run_manifest, step=1),
        )

    assert (
        checkpoint.resolve_latest_step(
            valid_run,
            writable=False,
            verification=VerificationLevel.FULL,
        ).step
        == 2
    )


def test_retention_waits_for_active_reader(valid_run: Path) -> None:
    """Exclusive retention must wait until every shared reader releases."""
    _publish_step(valid_run, 2)
    reader_ready = threading.Event()
    release_reader = threading.Event()

    def hold_reader() -> None:
        with checkpoint.run_access_lock(valid_run, exclusive=False):
            reader_ready.set()
            if not release_reader.wait(10):
                raise RuntimeError("timed out waiting to release reader")

    def retain_owned() -> checkpoint.ResolvedStep:
        with checkpoint.run_writer_lock(valid_run):
            return checkpoint.apply_retention(valid_run, keep_last=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reader = executor.submit(hold_reader)
        assert reader_ready.wait(5)
        retention = executor.submit(retain_owned)
        try:
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
        retained = checkpoint.apply_retention(valid_run, keep_last=1)

    assert retained.step == 3
    assert sorted(path.name for path in (valid_run / "checkpoints").iterdir()) == [
        "step-000000003"
    ]


def test_retention_reports_persisted_latest_recovery(valid_run: Path) -> None:
    """Retention must preserve the recovery outcome returned by its first phase."""
    _publish_step(valid_run, 2)
    (valid_run / "latest.json").unlink()

    with checkpoint.run_writer_lock(valid_run):
        retained = checkpoint.apply_retention(valid_run, keep_last=1)

    assert retained.step == 2
    assert retained.latest_recovered is True
    assert retained.latest_repair_persisted is True


def test_retention_requires_current_full_proof_before_first_delete(
    valid_run: Path,
) -> None:
    """Corrupt retained bytes must abort retention before any deletion syscall."""
    _publish_step(valid_run, 2)
    (valid_run / "checkpoints" / "step-000000002" / "arrays.safetensors").write_bytes(
        b"tampered"
    )
    fs = RecordingFilesystemOps(valid_run)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity|byte size"),
    ):
        checkpoint.apply_retention(valid_run, keep_last=1, fs=fs)

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
        checkpoint.apply_retention(valid_run, keep_last=1, fs=fs)

    assert fs.delete_count == 0
    assert all(
        (valid_run / "checkpoints" / f"step-{step:09d}").is_dir() for step in (1, 2, 3)
    )


@pytest.mark.parametrize("keep_last", [0, -1, True, 1.5])
def test_retention_rejects_nonpositive_or_noninteger_keep_last(
    valid_run: Path,
    keep_last: object,
) -> None:
    """Invalid retention limits must fail before inspecting deletion targets."""
    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises((TypeError, ValueError)),
    ):
        checkpoint.apply_retention(valid_run, keep_last=keep_last)  # type: ignore[arg-type]
