from __future__ import annotations

import hashlib
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from sml.artifacts import checkpoint
from sml.artifacts.checkpoint import (
    Published,
    publication_lock,
    publish_immutable_bundle,
    run_access_lock,
    run_writer_lock,
)
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


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


@pytest.fixture
def bundle_builder():
    return _bundle_builder()


@pytest.fixture
def target(tmp_path):
    return tmp_path / "bundle"


def _acquire_publication_lock_then_exit(target: str, connection) -> None:
    with publication_lock(Path(target)):
        connection.send("locked")
        connection.close()
        os._exit(0)


def test_concurrent_identical_publication_is_idempotent(bundle_builder, target):
    """Serializing identical contenders must return one fully verified identity."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_immutable_bundle, target, bundle_builder)
            for _ in range(2)
        ]
        first, second = (future.result(timeout=10) for future in futures)

    assert first.manifest.identity == second.manifest.identity
    assert first.verification is VerificationLevel.FULL
    assert second.verification is VerificationLevel.FULL
    verified = read_manifest(target, TokenizerManifest, VerificationLevel.FULL)
    assert verified.verification is VerificationLevel.FULL


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


def test_conflicting_writer_reports_owner(tmp_path):
    """A conflicting run accessor needs the live writer PID and protected run."""
    run = tmp_path / "run-0001"
    run.mkdir()

    with (
        run_writer_lock(run),
        pytest.raises(SMLArtifactError) as conflict,
        run_access_lock(run, exclusive=False),
    ):
        pytest.fail("conflicting accessor entered the protected operation")

    message = str(conflict.value)
    assert str(os.getpid()) in message
    assert str(run) in message


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

    monkeypatch.setattr(checkpoint, "_descriptor_is_local_apfs", failing_probe)

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


def test_published_is_frozen_slotted_and_reports_full(bundle_builder, target):
    """A writer result must not be mutable or overstate a weaker verification."""
    published = publish_immutable_bundle(target, bundle_builder)

    assert isinstance(published, Published)
    assert published.path == target
    assert published.verification is VerificationLevel.FULL
    assert not hasattr(published, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        published.path = target.parent / "other"


def test_builder_cannot_publish_its_own_manifest(target):
    """Letting the callback create the manifest would break manifest-last ordering."""
    builder = _bundle_builder()

    def writes_manifest(private_path: Path) -> TokenizerManifest:
        manifest = builder(private_path)
        (private_path / manifest.MANIFEST_FILENAME).write_bytes(b"not canonical")
        return manifest

    with pytest.raises(SMLArtifactError, match="builder.*manifest"):
        publish_immutable_bundle(target, writes_manifest)


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
