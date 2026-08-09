from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts.checkpoint import (
    IMMUTABLE_PUBLICATION_STAGES,
    OS_FILESYSTEM,
    FilesystemOps,
    publish_immutable_bundle,
)
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError


class InjectedFailure(RuntimeError):
    pass


class RecordingFilesystemOps:
    """Test-only filesystem wrapper that fails after completed real operations."""

    def __init__(
        self,
        *,
        failure_stage: str | None = None,
        target: Path | None = None,
        swap_candidate: tuple[Path, Path] | None = None,
        mutate_after_payload_fsync: Callable[[], None] | None = None,
    ) -> None:
        self._inner: FilesystemOps = OS_FILESYSTEM
        self._failure_stage = failure_stage
        self._target = target
        self._raised = False
        self._parent_descriptor: int | None = None
        self._temporary_descriptor: int | None = None
        self._manifest_descriptors: set[int] = set()
        self._temporary_fsyncs: dict[int, int] = {}
        self._swap_candidate = swap_candidate
        self._swapped = False
        self._mutate_after_payload_fsync = mutate_after_payload_fsync
        self._mutated = False
        self.completed_stages: list[str] = []

    @classmethod
    def raise_after(cls, stage: str, *, target: Path) -> RecordingFilesystemOps:
        return cls(failure_stage=stage, target=target)

    def _complete(self, stage: str) -> None:
        self.completed_stages.append(stage)
        if not self._raised and self._failure_stage == stage:
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
        path_text = os.fspath(path)
        if (
            self._swap_candidate is not None
            and not self._swapped
            and dir_fd is not None
            and Path(path_text).name == self._swap_candidate[0].name
            and flags & os.O_DIRECTORY
        ):
            candidate, replacement = self._swap_candidate
            candidate.rename(candidate.with_name(candidate.name + "-moved"))
            candidate.symlink_to(replacement, target_is_directory=True)
            self._swapped = True
        descriptor = self._inner.open(path, flags, mode, dir_fd=dir_fd)
        name = Path(path_text).name
        if (
            self._target is not None
            and dir_fd is None
            and Path(path_text) == self._target.parent
            and flags & os.O_DIRECTORY
        ):
            self._parent_descriptor = descriptor
        elif (
            dir_fd == self._parent_descriptor
            and name.startswith(".sml-tmp-")
            and flags & os.O_DIRECTORY
        ):
            self._temporary_descriptor = descriptor
        if name in {"manifest.json", "checkpoint.json", "run.json"}:
            self._manifest_descriptors.add(descriptor)
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
        if descriptor in self._manifest_descriptors:
            self._complete("manifest-written")
        elif self._mutate_after_payload_fsync is not None and not self._mutated:
            self._mutated = True
            self._mutate_after_payload_fsync()

    def fsync_directory(self, descriptor: int) -> None:
        self._inner.fsync_directory(descriptor)
        if descriptor == self._temporary_descriptor:
            count = self._temporary_fsyncs.get(descriptor, 0) + 1
            self._temporary_fsyncs[descriptor] = count
            self._complete(
                "payloads-written" if count == 1 else "temporary-directory-fsynced"
            )
        elif descriptor == self._parent_descriptor:
            self._complete("parent-directory-fsynced")

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
        self._complete("directory-renamed")

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

    def unlink(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        self._inner.unlink(path, dir_fd=dir_fd)

    def rmdir(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        self._inner.rmdir(path, dir_fd=dir_fd)

    def stat(
        self,
        path: int | str | os.PathLike[str],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = False,
    ) -> os.stat_result:
        return self._inner.stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def listdir(self, descriptor: int) -> list[str]:
        return self._inner.listdir(descriptor)

    def flock(self, descriptor: int, operation: int) -> None:
        self._inner.flock(descriptor, operation)


def _tokenizer_manifest_with_payloads(root, *, payload_directory: str | None = None):
    payload_root = root
    logical_prefix = ""
    if payload_directory is not None:
        payload_root = root / payload_directory
        payload_root.mkdir()
        logical_prefix = f"{payload_directory}/"
    model_path = payload_root / "tokenizer.model"
    vocab_path = payload_root / "tokenizer.vocab"
    model_path.write_bytes(b"model bytes")
    vocab_path.write_bytes(b"vocab bytes")
    with model_path.open("rb") as file:
        model_identity = file_identity(file)
    with vocab_path.open("rb") as file:
        vocab_identity = file_identity(file)
    manifest = TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity="sha256:" + "0" * 64,
        algorithm="sentencepiece-bpe-v1",
        training={"byte_fallback": True},
        vocab_size=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        model=PayloadRef(
            f"{logical_prefix}tokenizer.model",
            model_identity,
            model_path.stat().st_size,
        ),
        vocab=PayloadRef(
            f"{logical_prefix}tokenizer.vocab",
            vocab_identity,
            vocab_path.stat().st_size,
        ),
        diagnostic_source_locator="/source/tokenizer",
    )
    return replace(manifest, identity=manifest.recompute_identity())


def _published_tokenizer_manifest(root):
    manifest = _tokenizer_manifest_with_payloads(root)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _immutable_bundle_builder(private_path: Path) -> TokenizerManifest:
    return _tokenizer_manifest_with_payloads(
        private_path,
        payload_directory="payloads",
    )


@pytest.mark.parametrize("stage", IMMUTABLE_PUBLICATION_STAGES)
def test_interrupted_bundle_never_exposes_partial_target(stage, tmp_path):
    """Every completed durability boundary must leave absence or a full bundle."""
    target = tmp_path / "bundle"
    fs = RecordingFilesystemOps.raise_after(stage, target=target)

    with pytest.raises(InjectedFailure, match=stage):
        publish_immutable_bundle(target, _immutable_bundle_builder, fs=fs)

    completed_index = IMMUTABLE_PUBLICATION_STAGES.index(stage)
    assert fs.completed_stages == list(
        IMMUTABLE_PUBLICATION_STAGES[: completed_index + 1]
    )

    if target.exists():
        verified = read_manifest(target, TokenizerManifest, VerificationLevel.FULL)
        assert verified.verification is VerificationLevel.FULL
    assert not any(
        child.name.startswith(".sml-tmp-") and child.is_dir()
        for child in tmp_path.iterdir()
    )


def test_cleanup_accepts_only_exact_target_digest_marker(tmp_path):
    """Stale cleanup must never claim another target or a malformed candidate."""
    target = tmp_path / "bundle"
    digest = hashlib.sha256(target.name.encode("utf-8")).hexdigest()
    owned = tmp_path / f".sml-tmp-{digest}-{'a' * 32}"
    wrong_digest = tmp_path / f".sml-tmp-{'b' * 64}-{'c' * 32}"
    malformed_suffix = tmp_path / f".sml-tmp-{digest}-not-hex"
    for candidate in (owned, wrong_digest, malformed_suffix):
        candidate.mkdir()
        (candidate / "stale.bin").write_bytes(b"stale")

    publish_immutable_bundle(target, _immutable_bundle_builder)

    assert not owned.exists()
    assert wrong_digest.is_dir()
    assert malformed_suffix.is_dir()


def test_cleanup_revalidates_candidate_before_descriptor_relative_delete(tmp_path):
    """Swapping a checked stale name must not redirect deletion outside its inode."""
    target = tmp_path / "bundle"
    digest = hashlib.sha256(target.name.encode("utf-8")).hexdigest()
    candidate = tmp_path / f".sml-tmp-{digest}-{'d' * 32}"
    candidate.mkdir()
    (candidate / "stale.bin").write_bytes(b"stale")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "must-survive.bin"
    sentinel.write_bytes(b"outside")
    fs = RecordingFilesystemOps(swap_candidate=(candidate, external))

    with pytest.raises(SMLArtifactError, match="cleanup|symlink|revalid"):
        publish_immutable_bundle(target, _immutable_bundle_builder, fs=fs)

    assert sentinel.read_bytes() == b"outside"
    assert not target.exists()


def test_payload_mutation_during_durability_is_revalidated_before_commit(tmp_path):
    """FULL must be established after durability work, immediately before rename."""
    target = tmp_path / "bundle"
    private_paths: list[Path] = []

    def builder(private_path: Path) -> TokenizerManifest:
        private_paths.append(private_path)
        return _immutable_bundle_builder(private_path)

    def mutate_payload() -> None:
        (private_paths[0] / "payloads" / "tokenizer.model").write_bytes(
            b"mutated-bytes"
        )

    fs = RecordingFilesystemOps(mutate_after_payload_fsync=mutate_payload)

    with pytest.raises(SMLArtifactError, match="payload identity|payload byte size"):
        publish_immutable_bundle(target, builder, fs=fs)

    assert not target.exists()


def test_full_manifest_verification_rehashes_payloads(tmp_path):
    """Reporting full without hashing changed bytes would overstate integrity."""
    expected = _published_tokenizer_manifest(tmp_path)

    verified = read_manifest(tmp_path, TokenizerManifest, VerificationLevel.FULL)

    assert verified.manifest == expected
    assert verified.verification is VerificationLevel.FULL

    (tmp_path / "tokenizer.model").write_bytes(b"tampered!!!")
    with pytest.raises(SMLArtifactError, match="payload identity"):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.FULL)


def test_manifest_trusted_verification_does_not_claim_or_perform_full_rehash(tmp_path):
    """A read-only trusted open must remain distinct from content verification."""
    expected = _published_tokenizer_manifest(tmp_path)
    (tmp_path / "tokenizer.model").write_bytes(b"tampered!!!")

    verified = read_manifest(
        tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED
    )

    assert verified.manifest == expected
    assert verified.verification is VerificationLevel.MANIFEST_TRUSTED


def test_manifest_trusted_verification_still_checks_payload_metadata(tmp_path):
    """Skipping metadata with the hash would let truncated payloads pass as trusted."""
    _published_tokenizer_manifest(tmp_path)
    (tmp_path / "tokenizer.model").write_bytes(b"short")

    with pytest.raises(SMLArtifactError, match="payload byte size"):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED)


def test_read_manifest_rejects_symlinked_manifest(tmp_path):
    """Opening the manifest by path would follow an attacker-controlled symlink."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    _published_tokenizer_manifest(external)
    (bundle / "manifest.json").symlink_to(external / "manifest.json")

    with pytest.raises(SMLArtifactError, match="symlink|no-follow"):
        read_manifest(bundle, TokenizerManifest, VerificationLevel.FULL)


def test_full_verification_rejects_external_hard_link(tmp_path):
    """Accepting multi-link payloads would allow bytes outside the artifact to mutate."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    expected = _published_tokenizer_manifest(bundle)
    external_payload = tmp_path / "external.model"
    external_payload.write_bytes(b"model bytes")
    (bundle / "tokenizer.model").unlink()
    os.link(external_payload, bundle / "tokenizer.model")

    with pytest.raises(SMLArtifactError, match="link count"):
        read_manifest(bundle, type(expected), VerificationLevel.FULL)
