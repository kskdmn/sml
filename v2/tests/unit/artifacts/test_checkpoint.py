from __future__ import annotations

import os
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from inspect import signature
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    LatestIndex,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
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


def test_checkpoint_public_retention_surface_is_latest_only() -> None:
    """Callers must not be able to select arbitrary checkpoint history depth."""
    assert "apply_retention" not in checkpoint.__all__
    assert not hasattr(checkpoint, "apply_retention")
    assert "keep_last" not in signature(checkpoint.prune_to_latest).parameters


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
        self._descriptor_paths: dict[int, tuple[str, ...]] = {}
        self._descriptor_inodes: dict[int, tuple[int, int]] = {}
        self._step_directory_fsync_count = 0
        self.completed_stages: list[str] = []
        self.fsynced_files: list[tuple[str, ...]] = []
        self.fsynced_directories: list[tuple[str, ...]] = []
        self.checkpoints_fsync_count = 0
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
        parent_path = None
        if dir_fd is not None and dir_fd in self._descriptor_paths:
            try:
                parent_stat = self._inner.stat(dir_fd)
            except OSError:
                pass
            else:
                if (
                    parent_stat.st_dev,
                    parent_stat.st_ino,
                ) == self._descriptor_inodes.get(dir_fd):
                    parent_path = self._descriptor_paths[dir_fd]
        if (
            parent_path is None
            and dir_fd is not None
            and self._temporary_step_descriptor is not None
        ):
            try:
                parent_stat = self._inner.stat(dir_fd)
                temporary_stat = self._inner.stat(self._temporary_step_descriptor)
            except OSError:
                pass
            else:
                if (parent_stat.st_dev, parent_stat.st_ino) == (
                    temporary_stat.st_dev,
                    temporary_stat.st_ino,
                ):
                    parent_path = ()
        descriptor = self._inner.open(path, flags, mode, dir_fd=dir_fd)
        descriptor_stat = self._inner.stat(descriptor)
        self._descriptor_inodes[descriptor] = (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        )
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
            self._descriptor_paths[descriptor] = ()
        elif parent_path is not None:
            self._descriptor_paths[descriptor] = parent_path + (name,)
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
        descriptor_stat = self._inner.stat(descriptor)
        if descriptor in self._descriptor_paths and self._descriptor_inodes.get(
            descriptor
        ) == (descriptor_stat.st_dev, descriptor_stat.st_ino):
            self.fsynced_files.append(self._descriptor_paths[descriptor])
        name = self._descriptor_names.get(descriptor, "")
        if name == "checkpoint.json":
            self._complete("checkpoint-manifest-written")
        elif name.startswith(".sml-tmp-latest-"):
            self._complete("latest-temporary-fsynced")

    def fsync_directory(self, descriptor: int) -> None:
        self._inner.fsync_directory(descriptor)
        descriptor_stat = self._inner.stat(descriptor)
        if descriptor in self._descriptor_paths and self._descriptor_inodes.get(
            descriptor
        ) == (descriptor_stat.st_dev, descriptor_stat.st_ino):
            self.fsynced_directories.append(self._descriptor_paths[descriptor])
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
            self.checkpoints_fsync_count += 1
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
        if str(source).startswith(".sml-tmp-step-") and str(destination).startswith(
            "step-"
        ):
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


class _MutateAfterStepParentFsyncFilesystemOps(RecordingFilesystemOps):
    def __init__(self, run: Path, *, step: int) -> None:
        super().__init__(run)
        self._step = step
        self._mutated = False

    def fsync_directory(self, descriptor: int) -> None:
        super().fsync_directory(descriptor)
        if descriptor == self._checkpoints_descriptor and not self._mutated:
            self._mutated = True
            (
                self._run
                / "checkpoints"
                / f"step-{self._step:09d}"
                / "model.safetensors"
            ).write_bytes(b"mutated-after-commit")


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


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _run_manifest() -> PretrainingRunManifest:
    manifest = PretrainingRunManifest(
        kind="pretraining-run",
        version=1,
        identity="sha256:" + "0" * 64,
        model={"rope_scaling_factor": 1.0},
        precision={"working": "bfloat16"},
        optimizer={"name": "adamw"},
        loader={"batch_size": 2},
        checkpoint={
            "interval": 1,
            "rng_schedule": "counter-addressed-forward-terminal-v1",
        },
        tokenizer_identity="sha256:" + "1" * 64,
        data_identity="sha256:" + "2" * 64,
        diagnostic_data_locator="/relocatable/data",
    )
    return replace(manifest, identity=manifest.recompute_identity())


def _checkpoint_builder(
    run_manifest: PretrainingRunManifest,
    *,
    step: int,
    scalar_logical_path: str = "state.json",
) -> Callable[[Path], PretrainingCheckpointManifest]:

    def build(private_step: Path) -> PretrainingCheckpointManifest:
        state_path = private_step / scalar_logical_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        working = {"model.weight": mx.array([step], dtype=mx.bfloat16)}
        master = {"model.weight": working["model.weight"].astype(mx.float32)}
        optimizer = {
            "step": mx.array(step, dtype=mx.int32),
            "first_moments.model.weight": mx.zeros((1,), dtype=mx.float32),
            "second_moments.model.weight": mx.zeros((1,), dtype=mx.float32),
        }
        trainer = {
            "accumulation_count": mx.array(0, dtype=mx.int32),
            "next_key": mx.random.key(step),
            "loss_numerator": mx.array(0.0, dtype=mx.float32),
            "accumulators.model.weight": mx.zeros((1,), dtype=mx.float32),
        }
        references: dict[str, ArrayPayloadRef] = {}
        for logical_path, arrays in (
            ("model.safetensors", working),
            ("master.safetensors", master),
            ("optimizer.safetensors", optimizer),
            ("trainer.safetensors", trainer),
        ):
            path = private_step / logical_path
            mx.save_safetensors(path, arrays)
            references[logical_path] = ArrayPayloadRef(
                payload=_payload_ref(path, logical_path),
                arrays=tuple(
                    ArraySpec(
                        name,
                        tuple(array.shape),
                        {
                            mx.bfloat16: "bfloat16",
                            mx.float32: "float32",
                            mx.int32: "int32",
                            mx.uint32: "uint32",
                        }[array.dtype],
                    )
                    for name, array in sorted(arrays.items())
                ),
            )
        state_path.write_bytes(
            canonical_json_bytes(
                {
                    "kind": "pretraining-state",
                    "version": 1,
                    "owning_run_identity": run_manifest.identity,
                    "step": step,
                    "rows": step,
                    "microsteps": step,
                    "cursor": {
                        "epoch": 0,
                        "shard_order_position": 0,
                        "row_offset": 0,
                    },
                }
            )
        )
        manifest = PretrainingCheckpointManifest(
            kind="pretraining-checkpoint",
            version=1,
            identity="sha256:" + "0" * 64,
            owning_run_identity=run_manifest.identity,
            step=step,
            scalar_state=_payload_ref(state_path, scalar_logical_path),
            model=references["model.safetensors"],
            master=references["master.safetensors"],
            optimizer=references["optimizer.safetensors"],
            trainer=references["trainer.safetensors"],
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


@pytest.mark.parametrize("semantic_failure", [False, True])
def test_checkpoint_manifest_descriptor_is_stable_through_schema_validation(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_failure: bool,
) -> None:
    """Manifest parsing and its cleanup must share one stable descriptor lifetime."""
    step = valid_run / "checkpoints" / "step-000000001"
    manifest_path = step / "checkpoint.json"
    descriptor = os.open(step, os.O_RDONLY | os.O_DIRECTORY)
    real_parse = checkpoint._parse_manifest
    primary = SMLArtifactError("injected checkpoint manifest semantic failure")

    def mutating_parse(raw, manifest_type):
        before = manifest_path.stat()
        os.utime(
            manifest_path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
        )
        if semantic_failure:
            raise primary
        return real_parse(raw, manifest_type)

    monkeypatch.setattr(checkpoint, "_parse_manifest", mutating_parse)
    try:
        with pytest.raises(SMLArtifactError) as caught:
            checkpoint._read_manifest_from_descriptor(
                descriptor,
                PretrainingCheckpointManifest,
                VerificationLevel.MANIFEST_TRUSTED,
                context="injected checkpoint",
            )
    finally:
        os.close(descriptor)

    if semantic_failure:
        assert caught.value is primary
        assert isinstance(caught.value.__cause__, SMLArtifactError)
        assert "changed during use" in str(caught.value.__cause__)
    else:
        assert "changed during use" in str(caught.value)


@pytest.mark.parametrize("semantic_failure", [False, True])
def test_checkpoint_scalar_descriptor_is_stable_through_schema_validation(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_failure: bool,
) -> None:
    """Scalar schema checks must finish before postcheck and deterministic close."""
    step = valid_run / "checkpoints" / "step-000000001"
    descriptor = os.open(step, os.O_RDONLY | os.O_DIRECTORY)
    try:
        manifest = checkpoint._read_manifest_from_descriptor(
            descriptor,
            PretrainingCheckpointManifest,
            VerificationLevel.MANIFEST_TRUSTED,
            context="checkpoint scalar fixture",
        )
    finally:
        os.close(descriptor)

    state_path = step / "state.json"
    real_plain_nonnegative = checkpoint._plain_nonnegative
    mutated = False
    primary = SMLArtifactError("injected checkpoint scalar semantic failure")

    def mutating_validation(value, name):
        nonlocal mutated
        if not mutated:
            mutated = True
            before = state_path.stat()
            os.utime(
                state_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            if semantic_failure:
                raise primary
        return real_plain_nonnegative(value, name)

    monkeypatch.setattr(checkpoint, "_plain_nonnegative", mutating_validation)
    with (
        checkpoint.ArtifactRoot.open(step, writable=False) as root,
        pytest.raises(SMLArtifactError) as caught,
    ):
        checkpoint._read_checkpoint_scalar_payload(root, manifest, full=False)

    if semantic_failure:
        assert caught.value is primary
        assert isinstance(caught.value.__cause__, SMLArtifactError)
        assert "changed during use" in str(caught.value.__cause__)
    else:
        assert "changed during use" in str(caught.value)


@pytest.mark.parametrize("latest", [False, True])
def test_reader_cleanup_keeps_semantic_failure_primary_and_closes_all_fds(
    valid_run: Path, monkeypatch: pytest.MonkeyPatch, latest: bool
) -> None:
    """Every reader descriptor closes even when owned-step cleanup also fails."""
    original_close = checkpoint._OwnedStep.close
    descriptors: list[int] = []
    close_calls = 0

    def close_then_fail(owned):
        nonlocal close_calls
        close_calls += 1
        original_close(owned)
        if close_calls == (2 if latest else 1):
            raise RuntimeError("owned step cleanup failed")

    monkeypatch.setattr(checkpoint._OwnedStep, "close", close_then_fail)
    semantic = InjectedFailure("semantic body failure")
    opener = (
        checkpoint.open_latest_checkpoint_reader
        if latest
        else lambda run: checkpoint.open_checkpoint_reader(run, step=1)
    )

    with pytest.raises(InjectedFailure) as raised, opener(valid_run) as reader:
        descriptors.extend(
            [
                reader._run_descriptor,
                reader._checkpoints_descriptor,
                reader._owned_step.descriptor,
            ]
        )
        raise semantic

    assert raised.value is semantic
    assert isinstance(raised.value.__cause__, RuntimeError)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


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
    assert {
        ("model.safetensors",),
        ("master.safetensors",),
        ("optimizer.safetensors",),
        ("trainer.safetensors",),
    }.issubset(fs.fsynced_files)
    assert () in fs.fsynced_directories
    if completed_index >= 1:
        assert ("state.json",) in fs.fsynced_files
        assert fs.fsynced_directories.count(()) >= 2
    resolved = _resolve_latest_owned(
        valid_run,
        writable=stage not in CHECKPOINT_STAGES[:4],
    )
    expected_step = 1 if completed_index < 4 else 2
    assert resolved.step == expected_step
    assert resolved.verification is VerificationLevel.FULL


def test_checkpoint_scalar_state_must_be_named_state_json(valid_run: Path) -> None:
    """Generic scalar aliases must not weaken the checkpoint layout contract."""
    run_manifest = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises((SMLArtifactError, ValueError), match="state.json|scalar"),
    ):
        checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(
                run_manifest,
                step=2,
                scalar_logical_path="scalar.json",
            ),
        )


def test_post_commit_mutation_never_publishes_latest_or_returns_full(
    valid_run: Path,
) -> None:
    """Committed visibility must be FULL-proved again before latest advances."""
    run_manifest = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).run
    fs = _MutateAfterStepParentFsyncFilesystemOps(valid_run, step=2)

    with (
        checkpoint.run_writer_lock(valid_run),
        pytest.raises(SMLArtifactError, match="payload identity|byte size|FULL|full"),
    ):
        checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(run_manifest, step=2),
            fs=fs,
        )

    latest = read_manifest(
        valid_run,
        LatestIndex,
        VerificationLevel.MANIFEST_TRUSTED,
    ).manifest
    assert latest.step == 1


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


def test_latest_reader_full_proves_only_selected_winner_once(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate selection is metadata-only; the retained winner gets one proof."""
    _publish_step(valid_run, 2)
    real_verify = checkpoint._verify_checkpoint_semantics
    proved_steps: list[int] = []

    def record_proof(descriptor, manifest, verification, **kwargs):
        proved_steps.append(manifest.step)
        return real_verify(descriptor, manifest, verification, **kwargs)

    monkeypatch.setattr(checkpoint, "_verify_checkpoint_semantics", record_proof)

    with checkpoint.open_latest_checkpoint_reader(
        valid_run,
        verification=VerificationLevel.FULL,
        load_array_groups=frozenset(),
    ) as reader:
        assert reader.resolved.step == 2

    assert proved_steps == [2]


@pytest.mark.parametrize(
    ("verification", "groups"),
    (
        (VerificationLevel.FULL, frozenset()),
        (
            VerificationLevel.MANIFEST_TRUSTED,
            frozenset({"model.safetensors"}),
        ),
    ),
)
def test_selected_checkpoint_payloads_are_opened_once_for_proof_and_use(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    verification: VerificationLevel,
    groups: frozenset[str],
) -> None:
    """Structural proof must not reopen payloads owned by semantic proof/use."""
    opens: Counter[str] = Counter()
    payload_names = {
        "state.json",
        "model.safetensors",
        "master.safetensors",
        "optimizer.safetensors",
        "trainer.safetensors",
    }
    real_payload_open = ArtifactRoot._open_payload_with_stat
    real_opened_entry = checkpoint._opened_entry

    def counted_payload_open(owner, logical_path):
        if logical_path in payload_names:
            opens[logical_path] += 1
        return real_payload_open(owner, logical_path)

    def counted_opened_entry(
        fs,
        name,
        *,
        parent_descriptor,
        flags,
    ):
        if name in payload_names:
            opens[name] += 1
        return real_opened_entry(
            fs,
            name,
            parent_descriptor=parent_descriptor,
            flags=flags,
        )

    monkeypatch.setattr(
        ArtifactRoot,
        "_open_payload_with_stat",
        counted_payload_open,
    )
    monkeypatch.setattr(checkpoint, "_opened_entry", counted_opened_entry)
    with checkpoint.open_latest_checkpoint_reader(
        valid_run,
        verification=verification,
        load_array_groups=groups,
    ) as reader:
        reader.read_contents()

    assert opens["state.json"] == 1
    for name in (
        "model.safetensors",
        "master.safetensors",
        "optimizer.safetensors",
        "trainer.safetensors",
    ):
        assert opens[name] == 1


def test_non_deferred_trusted_resolution_retains_one_structural_content_proof(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted metadata-only resolution still proves content when no use follows."""
    opens: Counter[str] = Counter()
    payload_names = {
        "state.json",
        "model.safetensors",
        "master.safetensors",
        "optimizer.safetensors",
        "trainer.safetensors",
    }
    real_payload_open = ArtifactRoot._open_payload_with_stat
    real_opened_entry = checkpoint._opened_entry

    def counted_payload_open(owner, logical_path):
        if logical_path in payload_names:
            opens[logical_path] += 1
        return real_payload_open(owner, logical_path)

    def counted_opened_entry(fs, name, *, parent_descriptor, flags):
        if name in payload_names:
            opens[name] += 1
        return real_opened_entry(
            fs,
            name,
            parent_descriptor=parent_descriptor,
            flags=flags,
        )

    monkeypatch.setattr(
        ArtifactRoot,
        "_open_payload_with_stat",
        counted_payload_open,
    )
    monkeypatch.setattr(checkpoint, "_opened_entry", counted_opened_entry)

    resolved = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.MANIFEST_TRUSTED,
    )

    assert resolved.step == 1
    assert opens == Counter({name: 1 for name in payload_names})


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("symlink", "symlink|special|closed-world|no-follow"),
        ("hard-link", "hard-linked|link count|closed-world"),
        ("wrong-size", "byte size|closed-world"),
        ("case-fold-collision", "normalized|case-folded|closed-world"),
        ("unexpected-file", "closed-world"),
        ("missing-file", "closed-world"),
    ),
)
def test_stat_only_closed_world_rejects_checkpoint_namespace_corruption(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    """Namespace proof rejects corruption before any payload content is opened."""
    step = next((valid_run / "checkpoints").glob("step-*"))
    model = step / "model.safetensors"
    master = step / "master.safetensors"
    fs = checkpoint.OS_FILESYSTEM
    if mutation == "symlink":
        model.unlink()
        model.symlink_to(master.name)
    elif mutation == "hard-link":
        model.unlink()
        os.link(master, model)
    elif mutation == "wrong-size":
        with model.open("ab") as payload:
            payload.write(b"x")
    elif mutation == "case-fold-collision":

        class CaseFoldCollisionFilesystemOps(RecordingFilesystemOps):
            def listdir(self, descriptor: int) -> list[str]:
                names = super().listdir(descriptor)
                if "model.safetensors" in names:
                    names.append("MODEL.SAFETENSORS")
                return names

        fs = CaseFoldCollisionFilesystemOps(valid_run)
    elif mutation == "unexpected-file":
        (step / "unexpected.bin").write_bytes(b"unexpected")
    else:
        assert mutation == "missing-file"
        model.unlink()

    payload_names = {
        "state.json",
        "model.safetensors",
        "master.safetensors",
        "optimizer.safetensors",
        "trainer.safetensors",
    }
    real_payload_open = ArtifactRoot._open_payload_with_stat
    real_opened_entry = checkpoint._opened_entry

    def reject_payload_open(owner, logical_path):
        if logical_path in payload_names:
            pytest.fail(f"stat-only proof opened payload content: {logical_path}")
        return real_payload_open(owner, logical_path)

    def reject_opened_entry(fs, name, *, parent_descriptor, flags):
        if name in payload_names:
            pytest.fail(f"stat-only proof opened payload content: {name}")
        return real_opened_entry(
            fs,
            name,
            parent_descriptor=parent_descriptor,
            flags=flags,
        )

    monkeypatch.setattr(ArtifactRoot, "_open_payload_with_stat", reject_payload_open)
    monkeypatch.setattr(checkpoint, "_opened_entry", reject_opened_entry)
    with (
        pytest.raises(SMLArtifactError, match=match),
        checkpoint.open_latest_checkpoint_reader(
            valid_run,
            verification=VerificationLevel.FULL,
            fs=fs,
        ),
    ):
        pass


def test_trusted_requested_group_does_not_reduce_trainer_state(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted materialization validates metadata without unrelated value reads."""

    def reject_boundary_reduction(*_args, **_kwargs):
        raise AssertionError("trusted trainer/optimizer reduction is forbidden")

    monkeypatch.setattr(
        checkpoint,
        "_checkpoint_boundary_state",
        reject_boundary_reduction,
    )

    with checkpoint.open_checkpoint_reader(
        valid_run,
        step=1,
        verification=VerificationLevel.MANIFEST_TRUSTED,
        load_array_groups=frozenset({"model.safetensors"}),
    ) as reader:
        assert set(reader.read_contents().array_groups) == {"model.safetensors"}


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
