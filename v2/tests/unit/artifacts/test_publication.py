from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts import manifest as manifest_module
from sml.artifacts.checkpoint import (
    publish_immutable_bundle,
)
from sml.artifacts.manifest import (
    PayloadRef,
    PretrainingDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    read_manifest,
)
from sml.errors import SMLArtifactError

from ._payloads import _payload_ref
from ._publication import (
    _ForwardingFilesystemOps,
)


class _SignalingLockAttemptFilesystemOps(_ForwardingFilesystemOps):
    def __init__(self, inner, *, second_attempted: threading.Event) -> None:
        super().__init__(inner)
        self._second_attempted = second_attempted
        self._attempt_count = 0
        self._attempt_lock = threading.Lock()

    def flock(self, descriptor, operation):
        try:
            self._inner.flock(descriptor, operation)
        finally:
            if operation & fcntl.LOCK_NB:
                with self._attempt_lock:
                    self._attempt_count += 1
                    if self._attempt_count == 2:
                        self._second_attempted.set()


class _SwappingRenameFilesystemOps(_ForwardingFilesystemOps):
    def __init__(self, inner, *, private_paths: list[Path]) -> None:
        super().__init__(inner)
        self._private_paths = private_paths

    def rename(
        self,
        source,
        destination,
        *,
        source_dir_fd,
        destination_dir_fd,
    ):
        private_path = self._private_paths[0]
        private_path.rename(private_path.with_name(private_path.name + "-verified"))
        private_path.mkdir()
        (private_path / "corrupt.bin").write_bytes(b"unverified replacement")
        self._inner.rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )


def _bundle_builder(model_bytes: bytes = b"model bytes"):
    def build(private_path: Path) -> TokenizerManifest:
        model_path = private_path / "tokenizer.model"
        vocab_path = private_path / "tokenizer.vocab"
        model_path.write_bytes(model_bytes)
        vocab_path.write_bytes(b"vocab bytes")
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
            model=_payload_ref(model_path, "tokenizer.model"),
            vocab=_payload_ref(vocab_path, "tokenizer.vocab"),
            diagnostic_source_locator="/source/tokenizer",
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return build


def _prepared_bundle_builder(*, mutation: str | None = None):
    def build(private_path: Path) -> PretrainingDataManifest:
        tokenizer_path = private_path / "tokenizer"
        tokenizer_path.mkdir()
        tokenizer_manifest = _bundle_builder()(tokenizer_path)
        if mutation == "nested-model-logical-path":
            wrong_model = tokenizer_path / "wrong.model"
            wrong_model.write_bytes((tokenizer_path / "tokenizer.model").read_bytes())
            tokenizer_manifest = replace(
                tokenizer_manifest,
                identity="sha256:" + "0" * 64,
                model=_payload_ref(wrong_model, "wrong.model"),
            )
            tokenizer_manifest = replace(
                tokenizer_manifest,
                identity=tokenizer_manifest.recompute_identity(),
            )
        elif mutation == "nested-vocab-logical-path":
            wrong_vocab = tokenizer_path / "wrong.vocab"
            wrong_vocab.write_bytes((tokenizer_path / "tokenizer.vocab").read_bytes())
            tokenizer_manifest = replace(
                tokenizer_manifest,
                identity="sha256:" + "0" * 64,
                vocab=_payload_ref(wrong_vocab, "wrong.vocab"),
            )
            tokenizer_manifest = replace(
                tokenizer_manifest,
                identity=tokenizer_manifest.recompute_identity(),
            )
        (tokenizer_path / "manifest.json").write_bytes(
            canonical_json_bytes(tokenizer_manifest)
        )

        shards_path = private_path / "shards"
        shards_path.mkdir()
        shard_path = shards_path / "train-000000.npy"
        shard_path.write_bytes(b"npy bytes")

        tokenizer_identity = tokenizer_manifest.identity
        tokenizer_model = PayloadRef(
            "tokenizer/tokenizer.model",
            tokenizer_manifest.model.identity,
            tokenizer_manifest.model.byte_size,
        )
        tokenizer_vocab = PayloadRef(
            "tokenizer/tokenizer.vocab",
            tokenizer_manifest.vocab.identity,
            tokenizer_manifest.vocab.byte_size,
        )
        if mutation == "outer-tokenizer-identity":
            tokenizer_identity = "sha256:" + "f" * 64
        elif mutation == "outer-model-ref":
            tokenizer_model = replace(
                tokenizer_model,
                identity="sha256:" + "f" * 64,
            )
        elif mutation == "noncanonical-nested-manifest":
            (tokenizer_path / "manifest.json").write_text(
                canonical_json_bytes(tokenizer_manifest).decode("utf-8") + "\n",
                encoding="utf-8",
            )

        manifest = PretrainingDataManifest(
            kind="pretraining-data",
            version=1,
            identity="sha256:" + "0" * 64,
            sequence_length=3,
            row_width=4,
            dtype="int32",
            shard_row_counts=(1,),
            shards=(_payload_ref(shard_path, "shards/train-000000.npy"),),
            preparation_seed=42,
            row_order_policy={"algorithm": "windowed-row-shuffle-v1"},
            tokenizer_identity=tokenizer_identity,
            tokenizer_model=tokenizer_model,
            tokenizer_vocab=tokenizer_vocab,
            source_summary={},
            diagnostic_source_locator=None,
            row_content_identity="sha256:" + "a" * 64,
        )
        return replace(manifest, identity=manifest.recompute_identity())

    return build


@pytest.fixture
def bundle_builder():
    return _bundle_builder()


def test_publication_reuses_payload_hashes_only_within_its_transaction(
    tmp_path, monkeypatch
):
    target = tmp_path / "bundle"
    hashes = Counter()
    real_identity = manifest_module.file_identity

    def count_identity(stream):
        info = os.fstat(stream.fileno())
        hashes[info.st_dev, info.st_ino] += 1
        return real_identity(stream)

    monkeypatch.setattr(manifest_module, "file_identity", count_identity)
    published = publish_immutable_bundle(target, _bundle_builder())
    assert published.verification is VerificationLevel.FULL
    for name in ("tokenizer.model", "tokenizer.vocab"):
        info = (target / name).stat()
        assert hashes[info.st_dev, info.st_ino] == 1

    # A later publication must independently prove an existing destination.
    publish_immutable_bundle(target, _bundle_builder())
    for name in ("tokenizer.model", "tokenizer.vocab"):
        info = (target / name).stat()
        assert hashes[info.st_dev, info.st_ino] > 1


@pytest.mark.parametrize("stage", ("durability", "rename"))
@pytest.mark.parametrize("replace_inode", (False, True))
def test_publication_revalidates_same_size_payload_changes(
    tmp_path, stage, replace_inode
):
    target = tmp_path / "bundle"
    private_paths = []
    build_bundle = _bundle_builder()

    def build(private):
        private_paths.append(private)
        return build_bundle(private)

    class MutatingFilesystem(_ForwardingFilesystemOps):
        mutated = False

        def mutate(self, path):
            original = path.stat()
            changed = path.with_suffix(".replacement") if replace_inode else path
            changed.write_bytes(b"MODEL BYTES")
            os.utime(changed, ns=(original.st_atime_ns, original.st_mtime_ns))
            if replace_inode:
                changed.replace(path)
            self.mutated = True

        def fsync_file(self, descriptor):
            super().fsync_file(descriptor)
            if stage == "durability" and not self.mutated and private_paths:
                self.mutate(private_paths[0] / "tokenizer.model")

        def rename(self, source, destination, *, source_dir_fd, destination_dir_fd):
            super().rename(
                source,
                destination,
                source_dir_fd=source_dir_fd,
                destination_dir_fd=destination_dir_fd,
            )
            if stage == "rename" and not self.mutated:
                self.mutate(target / "tokenizer.model")

    fs = MutatingFilesystem(checkpoint.OS_FILESYSTEM)
    with pytest.raises(SMLArtifactError, match="payload identity"):
        publish_immutable_bundle(target, build, fs=fs)
    assert fs.mutated
    if stage == "durability":
        assert not target.exists()


def test_concurrent_identical_publication_is_idempotent(bundle_builder, target):
    """Serializing identical contenders must return one fully verified identity."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_immutable_bundle, target, bundle_builder)
            for _ in range(2)
        ]
        first, second = (future.result(timeout=10) for future in futures)

    assert first.path == second.path == target
    assert first.manifest.identity == second.manifest.identity
    assert first.verification is VerificationLevel.FULL
    assert second.verification is VerificationLevel.FULL
    verified = read_manifest(target, TokenizerManifest, VerificationLevel.FULL)
    assert verified.verification is VerificationLevel.FULL


def test_contending_identical_publication_waits_until_owner_releases(
    target, monkeypatch
):
    """Publication correctness must not expire while a valid owner is building."""
    first_builder_entered = threading.Event()
    release_first_builder = threading.Event()
    builder_call_count = 0
    builder_call_lock = threading.Lock()
    base_builder = _bundle_builder()
    second_lock_attempted = threading.Event()
    signaling_fs = _SignalingLockAttemptFilesystemOps(
        checkpoint.OS_FILESYSTEM,
        second_attempted=second_lock_attempted,
    )
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", signaling_fs)

    def coordinated_builder(private_path: Path) -> TokenizerManifest:
        nonlocal builder_call_count
        with builder_call_lock:
            builder_call_count += 1
            call_number = builder_call_count
        if call_number == 1:
            first_builder_entered.set()
            if not release_first_builder.wait(10):
                raise RuntimeError("timed out waiting to release first builder")
        return base_builder(private_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            publish_immutable_bundle,
            target,
            coordinated_builder,
        )
        assert first_builder_entered.wait(5)
        second = executor.submit(
            publish_immutable_bundle,
            target,
            coordinated_builder,
        )
        try:
            assert second_lock_attempted.wait(5)
            assert not second.done()
            assert builder_call_count == 1
        finally:
            release_first_builder.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert first_result.manifest.identity == second_result.manifest.identity
    assert second_result.verification is VerificationLevel.FULL


def test_named_temporary_swap_at_rename_boundary_never_returns_full(target):
    """The inode verified through the owned fd must be the inode committed by rename."""
    private_paths: list[Path] = []
    builder = _bundle_builder()

    def records_private_path(private_path: Path) -> TokenizerManifest:
        private_paths.append(private_path)
        return builder(private_path)

    fs = _SwappingRenameFilesystemOps(
        checkpoint.OS_FILESYSTEM,
        private_paths=private_paths,
    )

    with pytest.raises(SMLArtifactError, match="committed|corrupt|inode|swapped"):
        publish_immutable_bundle(target, records_private_path, fs=fs)

    assert target.is_dir()
    with pytest.raises(SMLArtifactError):
        read_manifest(target, TokenizerManifest, VerificationLevel.FULL)


def test_different_existing_target_is_collision(target):
    """Replacing an existing bundle would violate immutable identity ownership."""
    first = publish_immutable_bundle(target, _bundle_builder(b"first model"))

    with pytest.raises(SMLArtifactError, match="collision|different identity"):
        publish_immutable_bundle(target, _bundle_builder(b"second model"))

    verified = read_manifest(target, TokenizerManifest, VerificationLevel.FULL)
    assert verified.manifest.identity == first.manifest.identity


def test_identical_existing_target_requires_full_verification(bundle_builder, target):
    """Identity equality must not accept changed bytes without a full rehash."""
    publish_immutable_bundle(target, bundle_builder)
    (target / "tokenizer.model").write_bytes(b"tamper bytes")

    with pytest.raises(SMLArtifactError, match="full verification|payload identity"):
        publish_immutable_bundle(target, bundle_builder)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("manifest.json", "builder.*manifest"),
        ("unreferenced.bin", "closed-world|unreferenced"),
        ("unreferenced/", "closed-world|unreferenced"),
        ("checkpoint.json", "closed-world|unreferenced|manifest"),
    ],
    ids=["own-manifest", "file", "directory", "foreign-manifest"],
)
def test_builder_rejects_entries_outside_declared_payloads(target, entry, message):
    builder = _bundle_builder()

    def adds_unowned_entry(private_path: Path) -> TokenizerManifest:
        manifest = builder(private_path)
        path = private_path / entry
        if entry.endswith("/"):
            path.mkdir()
        else:
            path.write_bytes(b"not owned by manifest")
        return manifest

    with pytest.raises(SMLArtifactError, match=message):
        publish_immutable_bundle(target, adds_unowned_entry)
    assert not target.exists()


def test_pretraining_publication_owns_a_fully_bound_nested_tokenizer_manifest(
    target,
):
    """Removing child-manifest ownership would reject the portable tokenizer copy."""
    published = publish_immutable_bundle(target, _prepared_bundle_builder())

    assert published.verification is VerificationLevel.FULL
    assert (target / "tokenizer" / "manifest.json").is_file()
    assert {path.name for path in (target / "tokenizer").iterdir()} == {
        "manifest.json",
        "tokenizer.model",
        "tokenizer.vocab",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("outer-tokenizer-identity", "nested tokenizer.*identity|identity.*nested"),
        ("outer-model-ref", "nested tokenizer.*model|model.*nested tokenizer"),
        (
            "noncanonical-nested-manifest",
            "nested tokenizer.*canonical|canonical.*nested",
        ),
        ("nested-model-logical-path", "nested tokenizer.*model logical path"),
        ("nested-vocab-logical-path", "nested tokenizer.*vocab logical path"),
    ],
)
def test_pretraining_publication_rejects_unbound_nested_tokenizer_manifest(
    target,
    mutation,
    message,
):
    """Weak child ownership could publish tokenizer metadata unrelated to its bytes."""
    with pytest.raises(SMLArtifactError, match=message):
        publish_immutable_bundle(
            target,
            _prepared_bundle_builder(mutation=mutation),
        )

    assert not target.exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing-nested-manifest", "corrupt-nested-manifest", "unknown-file"],
)
def test_existing_pretraining_target_requires_complete_closed_world(
    target,
    mutation,
):
    """An identical retry must FULL-verify child metadata and the whole tree."""
    publish_immutable_bundle(target, _prepared_bundle_builder())
    if mutation == "missing-nested-manifest":
        (target / "tokenizer" / "manifest.json").unlink()
    elif mutation == "corrupt-nested-manifest":
        (target / "tokenizer" / "manifest.json").write_bytes(b"not json")
    else:
        (target / "unknown.bin").write_bytes(b"not owned")

    with pytest.raises(
        SMLArtifactError, match="existing target failed full verification"
    ):
        publish_immutable_bundle(target, _prepared_bundle_builder())


def test_temporary_name_uses_exact_target_digest(bundle_builder, target):
    """A cleanup marker derived from anything else could cross target ownership."""
    observed_names: list[str] = []

    def observes_private_path(private_path: Path) -> TokenizerManifest:
        observed_names.append(private_path.name)
        return bundle_builder(private_path)

    publish_immutable_bundle(target, observes_private_path)

    digest = hashlib.sha256(target.name.encode("utf-8")).hexdigest()
    assert len(observed_names) == 1
    assert observed_names[0].startswith(f".sml-tmp-{digest}-")
