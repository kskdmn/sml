from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts.manifest import (
    ArtifactRoot,
    PretrainingCheckpointManifest,
    VerificationLevel,
    canonical_json_bytes,
)
from sml.errors import SMLArtifactError

from ._checkpoints import (
    InjectedFailure,
    RecordingFilesystemOps,
    _publish_step,
)


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


def test_latest_reader_still_rejects_a_symlink_run(valid_run: Path) -> None:
    alias = valid_run.with_name("linked-run")
    alias.symlink_to(valid_run, target_is_directory=True)

    with (
        pytest.raises(SMLArtifactError, match="run directory"),
        checkpoint.open_latest_checkpoint_reader(alias),
    ):
        pytest.fail("latest reader followed a symlink run")


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


def test_materializing_reader_rechecks_equal_length_replaced_publisher_manifest(
    valid_run: Path,
) -> None:
    """Stat-only size proof must not accept a replaced named checkpoint manifest."""
    original = checkpoint.resolve_exact_step(
        valid_run,
        step=1,
        verification=VerificationLevel.FULL,
    ).checkpoint
    assert isinstance(original, PretrainingCheckpointManifest)
    digest = original.model.payload.identity.removeprefix("sha256:")
    replacement_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
    replacement_payload = replace(
        original.model.payload,
        identity=f"sha256:{replacement_digest}",
    )
    replacement = replace(
        original,
        identity="sha256:" + "0" * 64,
        model=replace(original.model, payload=replacement_payload),
    )
    replacement = replace(replacement, identity=replacement.recompute_identity())
    original_bytes = canonical_json_bytes(original)
    replacement_bytes = canonical_json_bytes(replacement)
    assert replacement.identity != original.identity
    assert len(replacement_bytes) == len(original_bytes)

    step = valid_run / "checkpoints" / "step-000000001"
    manifest_path = step / PretrainingCheckpointManifest.MANIFEST_FILENAME

    class ReplaceManifestDuringClosedWorld(checkpoint._OSFilesystemOps):
        def __init__(self) -> None:
            self.swapped = False

        def listdir(self, descriptor: int) -> list[str]:
            names = super().listdir(descriptor)
            if not self.swapped and "checkpoint.json" in names:
                temporary = step / ".replacement-checkpoint.json"
                temporary.write_bytes(replacement_bytes)
                os.replace(temporary, manifest_path)
                self.swapped = True
            return names

    fs = ReplaceManifestDuringClosedWorld()
    with (
        pytest.raises(SMLArtifactError, match="publisher-owned manifest changed"),
        checkpoint.open_checkpoint_reader(
            valid_run,
            step=1,
            verification=VerificationLevel.FULL,
            fs=fs,
        ) as reader,
    ):
        reader.read_contents()

    assert fs.swapped is True


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
