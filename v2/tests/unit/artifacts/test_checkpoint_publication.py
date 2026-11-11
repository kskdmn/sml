from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts import manifest as artifact_manifest
from sml.artifacts.manifest import (
    LatestIndex,
    VerificationLevel,
    read_manifest,
)
from sml.errors import SMLArtifactError

from ._checkpoints import (
    InjectedFailure,
    RecordingFilesystemOps,
    _checkpoint_builder,
    _publish_step,
    _run_manifest,
)

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


def test_publication_hashes_payloads_once_per_verification_boundary(
    valid_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-rename and post-rename proofs each read tensor bytes exactly once."""
    hashes: Counter[str] = Counter()
    original_identity = artifact_manifest.file_identity

    def counted_identity(stream):
        identity = original_identity(stream)
        hashes[identity] += 1
        return identity

    monkeypatch.setattr(artifact_manifest, "file_identity", counted_identity)
    with checkpoint.run_writer_lock(valid_run):
        published = checkpoint.publish_checkpoint(
            valid_run,
            _checkpoint_builder(_run_manifest(), step=2),
        )

    for reference in checkpoint.checkpoint_array_payloads(published.checkpoint):
        assert hashes[reference.payload.identity] == 2


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
