from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
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


class _ForwardingFilesystemOps:
    def __init__(self, inner) -> None:
        self._inner = inner

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        return self._inner.open(path, flags, mode, dir_fd=dir_fd)

    def mkdir(self, path, mode=0o777, *, dir_fd):
        self._inner.mkdir(path, mode, dir_fd=dir_fd)

    def write_all(self, descriptor, data):
        self._inner.write_all(descriptor, data)

    def fsync_file(self, descriptor):
        self._inner.fsync_file(descriptor)

    def fsync_directory(self, descriptor):
        self._inner.fsync_directory(descriptor)

    def rename(
        self,
        source,
        destination,
        *,
        source_dir_fd,
        destination_dir_fd,
    ):
        self._inner.rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    def replace(
        self,
        source,
        destination,
        *,
        source_dir_fd,
        destination_dir_fd,
    ):
        self._inner.replace(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    def unlink(self, path, *, dir_fd):
        self._inner.unlink(path, dir_fd=dir_fd)

    def rmdir(self, path, *, dir_fd):
        self._inner.rmdir(path, dir_fd=dir_fd)

    def stat(self, path, *, dir_fd=None, follow_symlinks=False):
        return self._inner.stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def listdir(self, descriptor):
        return self._inner.listdir(descriptor)

    def flock(self, descriptor, operation):
        self._inner.flock(descriptor, operation)


class _DelayedMetadataFilesystemOps(_ForwardingFilesystemOps):
    def __init__(self, inner, *, started, release, claimed) -> None:
        super().__init__(inner)
        self._started = started
        self._release = release
        self._claimed = claimed

    def write_all(self, descriptor, data):
        delay = False
        if b"protected_path" in data:
            with self._claimed.get_lock():
                if self._claimed.value == 0:
                    self._claimed.value = 1
                    delay = True
        if delay:
            self._started.set()
            if not self._release.wait(10):
                raise RuntimeError("timed out delaying owner diagnostics")
        self._inner.write_all(descriptor, data)


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


def _hold_run_writer(run: str, ready, release) -> None:
    with run_writer_lock(Path(run)):
        ready.set()
        if not release.wait(10):
            raise RuntimeError("timed out waiting to release run writer")


def _hold_run_writer_with_fs(run: str, ready, release, fs) -> None:
    checkpoint.OS_FILESYSTEM = fs
    _hold_run_writer(run, ready, release)


def _conflicting_run_access_message(run: Path) -> str:
    try:
        with run_access_lock(run, exclusive=False):
            raise AssertionError("conflicting accessor entered protected operation")
    except SMLArtifactError as error:
        return str(error)


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


def test_contending_identical_publication_waits_until_owner_releases(
    target, monkeypatch
):
    """Publication correctness must not expire while a valid owner is building."""
    first_builder_entered = threading.Event()
    release_first_builder = threading.Event()
    builder_call_count = 0
    builder_call_lock = threading.Lock()
    base_builder = _bundle_builder()

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

    monkeypatch.setattr(checkpoint, "_LOCK_RETRY_SECONDS", 0.05, raising=False)
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
            time.sleep(0.15)
            assert not second.done()
            assert builder_call_count == 1
        finally:
            release_first_builder.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)

    assert first_result.manifest.identity == second_result.manifest.identity
    assert second_result.verification is VerificationLevel.FULL


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


def test_conflicting_process_reports_live_owner(tmp_path):
    """Conflict diagnostics must identify the live holder in another process."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_run_writer,
        args=(str(run), ready, release),
    )
    process.start()
    try:
        assert ready.wait(10), "child did not acquire the run writer lock"
        with (
            pytest.raises(SMLArtifactError) as conflict,
            run_access_lock(run, exclusive=False),
        ):
            pytest.fail("conflicting accessor entered the protected operation")
        message = str(conflict.value)
        assert str(process.pid) in message
        assert str(run) in message
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()


def test_conflict_waits_for_complete_owner_diagnostics(tmp_path, monkeypatch):
    """A contender must not read owner metadata during truncate/write transition."""
    run = tmp_path / "run-0001"
    run.mkdir()
    context = multiprocessing.get_context("spawn")
    metadata_started = context.Event()
    allow_metadata = context.Event()
    holder_ready = context.Event()
    holder_release = context.Event()
    claimed = context.Value("i", 0)
    delayed_fs = _DelayedMetadataFilesystemOps(
        checkpoint.OS_FILESYSTEM,
        started=metadata_started,
        release=allow_metadata,
        claimed=claimed,
    )
    monkeypatch.setattr(checkpoint, "OS_FILESYSTEM", delayed_fs)
    process = context.Process(
        target=_hold_run_writer_with_fs,
        args=(str(run), holder_ready, holder_release, delayed_fs),
    )
    process.start()
    try:
        assert metadata_started.wait(10), "owner did not begin diagnostic transition"
        with ThreadPoolExecutor(max_workers=1) as executor:
            conflict = executor.submit(_conflicting_run_access_message, run)
            try:
                time.sleep(0.15)
                assert not conflict.done()
            finally:
                allow_metadata.set()
            assert holder_ready.wait(10), "owner did not complete diagnostic transition"
            message = conflict.result(timeout=5)
        assert str(process.pid) in message
        assert str(run) in message
    finally:
        allow_metadata.set()
        holder_release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()


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


def test_builder_cannot_add_unreferenced_regular_file(target):
    """Returning FULL must reject regular bytes outside the manifest's closed world."""
    builder = _bundle_builder()

    def adds_unreferenced_file(private_path: Path) -> TokenizerManifest:
        manifest = builder(private_path)
        (private_path / "unreferenced.bin").write_bytes(b"not owned by manifest")
        return manifest

    with pytest.raises(SMLArtifactError, match="closed-world|unreferenced"):
        publish_immutable_bundle(target, adds_unreferenced_file)
    assert not target.exists()


def test_builder_cannot_add_unreferenced_empty_directory(target):
    """Returning FULL must reject directories that are not payload ancestors."""
    builder = _bundle_builder()

    def adds_empty_directory(private_path: Path) -> TokenizerManifest:
        manifest = builder(private_path)
        (private_path / "unreferenced").mkdir()
        return manifest

    with pytest.raises(SMLArtifactError, match="closed-world|unreferenced"):
        publish_immutable_bundle(target, adds_empty_directory)
    assert not target.exists()


def test_builder_cannot_add_another_manifest_filename(target):
    """Only the publisher-created manifest for the returned type may be present."""
    builder = _bundle_builder()

    def adds_other_manifest(private_path: Path) -> TokenizerManifest:
        manifest = builder(private_path)
        (private_path / "checkpoint.json").write_bytes(b"foreign manifest")
        return manifest

    with pytest.raises(SMLArtifactError, match="closed-world|unreferenced|manifest"):
        publish_immutable_bundle(target, adds_other_manifest)
    assert not target.exists()


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
