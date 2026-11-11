from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    VerificationLevel,
    canonical_json_bytes,
)

from ._payloads import _array_ref, _payload_ref


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
            references[logical_path] = _array_ref(path, logical_path, arrays)
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
