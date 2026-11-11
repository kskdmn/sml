from __future__ import annotations

import io
import os
import stat

import pytest
from sml.artifacts import manifest as manifest_module
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
    file_identity,
)
from sml.errors import SMLArtifactError

from ._manifests import (
    _write_manifest,
    _write_tokenizer_artifact,
    tokenizer_manifest_fixture,
)


def test_open_artifact_does_not_preopen_declared_payloads(tmp_path):
    """Proof-only eager payload opens would separate later semantic use from proof."""
    expected = _write_manifest(tmp_path, tokenizer_manifest_fixture(), "manifest.json")

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        assert artifact.manifest == expected
        assert artifact.verification is VerificationLevel.FULL


def test_opened_artifact_child_uses_retained_root_after_path_replacement(tmp_path):
    """Constructing a child Path would parse a replacement instead of the retained root."""
    bundle = tmp_path / "bundle"
    _write_tokenizer_artifact(bundle, diagnostic_source_locator="/outer-original")
    _write_tokenizer_artifact(
        bundle / "tokenizer", diagnostic_source_locator="/child-original"
    )

    artifact = manifest_module.open_artifact(
        bundle, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
    )
    retained = tmp_path / "retained-bundle"
    bundle.rename(retained)
    _write_tokenizer_artifact(
        bundle / "tokenizer", diagnostic_source_locator="/child-replacement"
    )

    try:
        child = artifact.open_child("tokenizer", (TokenizerManifest,))
        artifact.close()
        with child:
            assert child.manifest.diagnostic_source_locator == "/child-original"
            with child.open_payload(child.manifest.model) as payload:
                assert payload.stream.read() == b"model-bytes"
    finally:
        artifact.close()


@pytest.mark.parametrize("recursive", [False, True], ids=("top-level", "child"))
def test_manifest_reader_rejects_in_place_mutation_during_schema_parse(
    tmp_path, monkeypatch, recursive
):
    """The manifest descriptor must remain pinned through schema validation."""
    root = tmp_path / "bundle"
    _write_tokenizer_artifact(root)
    target = root
    if recursive:
        target = root / "tokenizer"
        _write_tokenizer_artifact(target)
    manifest_path = target / "manifest.json"
    real_parse = manifest_module._parse_manifest
    parse_count = 0

    def mutating_parse(raw, manifest_type):
        nonlocal parse_count
        parse_count += 1
        if parse_count == (2 if recursive else 1):
            before = manifest_path.stat()
            os.utime(
                manifest_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        return real_parse(raw, manifest_type)

    monkeypatch.setattr(manifest_module, "_parse_manifest", mutating_parse)

    if recursive:
        with (
            manifest_module.open_artifact(
                root, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
            ) as artifact,
            pytest.raises(SMLArtifactError, match="changed during use"),
        ):
            artifact.open_child("tokenizer", (TokenizerManifest,))
    else:
        with pytest.raises(SMLArtifactError, match="changed during use"):
            manifest_module.open_artifact(
                root, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
            )


def test_opened_artifact_detach_root_is_one_shot_and_invalidates_owner(tmp_path):
    """Root ownership transfer must be explicit, unique, and leave no split owner."""
    _write_tokenizer_artifact(tmp_path)
    artifact = manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
    )

    root = artifact.detach_root()
    descriptor = root.fileno()
    os.fstat(descriptor)
    with pytest.raises(SMLArtifactError, match="already transferred"):
        artifact.detach_root()
    with pytest.raises(SMLArtifactError, match="closed"):
        artifact.open_payload(artifact.manifest.model)
    with pytest.raises(SMLArtifactError, match="closed"):
        artifact.open_child("child", (TokenizerManifest,))
    artifact.close()
    os.fstat(descriptor)
    root.close()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_payload_acquisition_preserves_primary_when_cleanup_also_fails(
    tmp_path, monkeypatch
):
    """Cleanup failure must be attached without replacing corruption evidence."""
    (tmp_path / "payload.bin").write_bytes(b"payload")
    real_fdopen = os.fdopen
    real_fstat = os.fstat

    class FailingClose:
        def __init__(self, stream):
            self._stream = stream

        def fileno(self):
            return self._stream.fileno()

        def close(self):
            self._stream.close()
            raise RuntimeError("injected payload close failure")

    def wrapped_fdopen(*args, **kwargs):
        return FailingClose(real_fdopen(*args, **kwargs))

    def linked_fstat(descriptor):
        result = real_fstat(descriptor)
        if stat.S_ISREG(result.st_mode):
            values = list(result)
            values[3] = 2
            return os.stat_result(values)
        return result

    monkeypatch.setattr(manifest_module.os, "fdopen", wrapped_fdopen)
    monkeypatch.setattr(manifest_module.os, "fstat", linked_fstat)
    with (
        manifest_module.ArtifactRoot.open(tmp_path, writable=False) as root,
        pytest.raises(SMLArtifactError, match="link count") as caught,
    ):
        root._open_payload_with_stat("payload.bin")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "injected payload close failure" in str(caught.value.__cause__)


def test_artifact_acquisition_preserves_parse_error_when_root_close_fails(
    tmp_path, monkeypatch
):
    """A root-close failure must not replace the original manifest failure."""
    primary = ValueError("injected manifest parse failure")

    class FailingRoot:
        def close(self):
            raise RuntimeError("injected root close failure")

    def fail_read(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(manifest_module, "_read_manifest_from_root", fail_read)
    with pytest.raises(ValueError, match="manifest parse failure") as caught:
        manifest_module._open_artifact_from_root(
            tmp_path,
            FailingRoot(),
            (TokenizerManifest,),
            VerificationLevel.MANIFEST_TRUSTED,
        )

    assert caught.value is primary
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "injected root close failure" in str(caught.value.__cause__)


def test_full_payload_proof_and_reader_share_one_final_descriptor_after_swap(
    tmp_path, monkeypatch
):
    """Hashing one inode and reopening its name for use would accept swapped bytes."""
    expected_data = b"proven-model"
    _write_tokenizer_artifact(tmp_path, model_data=expected_data)
    artifact = manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    )
    real_open = os.open
    real_fstat = os.fstat
    real_file_identity = manifest_module.file_identity
    payload_descriptors: list[int] = []
    fstat_descriptors: list[int] = []
    identity_descriptors: list[int] = []

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "tokenizer.model":
            payload_descriptors.append(descriptor)
        return descriptor

    def recording_fstat(descriptor):
        if descriptor in payload_descriptors:
            fstat_descriptors.append(descriptor)
        return real_fstat(descriptor)

    def recording_file_identity(stream):
        identity_descriptors.append(stream.fileno())
        return real_file_identity(stream)

    monkeypatch.setattr(manifest_module.os, "open", recording_open)
    monkeypatch.setattr(manifest_module.os, "fstat", recording_fstat)
    monkeypatch.setattr(manifest_module, "file_identity", recording_file_identity)

    try:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        (tmp_path / "tokenizer.model").rename(tmp_path / "proven.model")
        (tmp_path / "tokenizer.model").write_bytes(b"replacement!")

        assert payload.stream.read() == expected_data
        assert payload_descriptors == [descriptor]
        assert identity_descriptors == [descriptor]
        assert fstat_descriptors and set(fstat_descriptors) == {descriptor}
        with pytest.raises(SMLArtifactError, match="changed during use"):
            payload.close()
        with pytest.raises(OSError):
            real_fstat(descriptor)
    finally:
        artifact.close()


@pytest.mark.parametrize("mutation", ["size", "mtime", "ctime"])
def test_verified_payload_close_rejects_every_tracked_in_place_mutation(
    tmp_path, mutation
):
    """Omitting any pinned stat field would allow an open payload to change during use."""
    _write_tokenizer_artifact(tmp_path)
    payload_path = tmp_path / "tokenizer.model"

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        before = payload_path.stat()
        if mutation == "size":
            payload_path.write_bytes(b"changed-size")
        elif mutation == "mtime":
            os.utime(
                payload_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        else:
            payload_path.chmod(stat.S_IMODE(before.st_mode) | stat.S_IXUSR)

        with pytest.raises(SMLArtifactError, match="changed during use"):
            payload.close()
        assert payload.closed is True
        payload.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_semantic_error_stays_primary_when_mutation_postcheck_also_fails(tmp_path):
    """Replacing the reader error with cleanup failure would hide the semantic cause."""
    _write_tokenizer_artifact(tmp_path)

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()
        with (
            pytest.raises(RuntimeError, match="semantic reader failed") as caught,
            payload,
        ):
            (tmp_path / "tokenizer.model").write_bytes(b"mutated")
            raise RuntimeError("semantic reader failed")

        assert isinstance(caught.value.__cause__, SMLArtifactError)
        assert "changed during use" in str(caught.value.__cause__)
        assert payload.closed is True
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    ("link_kind", "message"),
    [
        ("symlink", "symlink|no-follow"),
        ("internal-hard-link", "link count"),
        ("external-hard-link", "link count"),
    ],
)
def test_opened_artifact_rejects_linked_payloads(tmp_path, link_kind, message):
    """Accepting links would let payload ownership escape or alias the artifact root."""
    root = tmp_path / "artifact"
    root.mkdir()
    data = b"linked"
    if link_kind == "symlink":
        source = tmp_path / "external.bin"
        source.write_bytes(data)
        (root / "tokenizer.model").symlink_to(source)
    elif link_kind == "internal-hard-link":
        source = root / "source.bin"
        source.write_bytes(data)
        os.link(source, root / "tokenizer.model")
    else:
        source = tmp_path / "external.bin"
        source.write_bytes(data)
        os.link(source, root / "tokenizer.model")
    (root / "tokenizer.vocab").write_bytes(b"vocab")
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef("tokenizer.model", file_identity(io.BytesIO(data)), len(data)),
        vocab=PayloadRef("tokenizer.vocab", file_identity(io.BytesIO(b"vocab")), 5),
    )
    _write_manifest(root, manifest, "manifest.json")

    with (
        manifest_module.open_artifact(
            root, (TokenizerManifest,), VerificationLevel.FULL
        ) as artifact,
        pytest.raises(SMLArtifactError, match=message),
    ):
        artifact.open_payload(artifact.manifest.model)


def test_opened_artifact_rejects_distinct_paths_to_one_inode(tmp_path, monkeypatch):
    """Dropping retained-root inode tracking would accept duplicate payload aliases."""
    first = tmp_path / "tokenizer.model"
    second = tmp_path / "tokenizer.vocab"
    first.write_bytes(b"shared")
    os.link(first, second)
    aliased_inode = first.stat().st_ino
    manifest = tokenizer_manifest_fixture(
        model=PayloadRef("tokenizer.model", file_identity(io.BytesIO(b"shared")), 6),
        vocab=PayloadRef("tokenizer.vocab", file_identity(io.BytesIO(b"shared")), 6),
    )
    _write_manifest(tmp_path, manifest, "manifest.json")
    real_fstat = os.fstat

    def single_link_fstat(descriptor):
        result = real_fstat(descriptor)
        if result.st_ino != aliased_inode:
            return result
        values = list(result)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(manifest_module.os, "fstat", single_link_fstat)
    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.MANIFEST_TRUSTED
    ) as artifact:
        with artifact.open_payload(artifact.manifest.model):
            pass
        with pytest.raises(SMLArtifactError, match="inode alias"):
            artifact.open_payload(artifact.manifest.vocab)


def test_payload_acquisition_rejects_hard_link_created_by_fdopen_hook(
    tmp_path, monkeypatch
):
    """Using a later baseline would accept a hard link introduced after validation."""
    _write_tokenizer_artifact(tmp_path)
    payload_path = tmp_path / "tokenizer.model"
    payload_inode = payload_path.stat().st_ino
    real_fdopen = os.fdopen
    real_fstat = os.fstat
    payload_descriptors: list[int] = []

    def linking_fdopen(descriptor, *args, **kwargs):
        stream = real_fdopen(descriptor, *args, **kwargs)
        if real_fstat(descriptor).st_ino == payload_inode:
            payload_descriptors.append(descriptor)
            os.link(payload_path, tmp_path / "late-alias.bin")
        return stream

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        monkeypatch.setattr(manifest_module.os, "fdopen", linking_fdopen)
        payload = None
        try:
            payload = artifact.open_payload(artifact.manifest.model)
        except SMLArtifactError as error:
            assert "link count" in str(error)
        else:
            payload.close()
            pytest.fail("payload acquisition accepted a late hard link")

    assert len(payload_descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(payload_descriptors[0])


@pytest.mark.parametrize("semantic_failure", [False, True])
def test_non_oserror_postcheck_always_closes_and_preserves_exception_precedence(
    tmp_path, monkeypatch, semantic_failure
):
    """A non-OSError postcheck failure must not leak or replace a reader failure."""
    _write_tokenizer_artifact(tmp_path)
    real_fstat = os.fstat

    with manifest_module.open_artifact(
        tmp_path, (TokenizerManifest,), VerificationLevel.FULL
    ) as artifact:
        payload = artifact.open_payload(artifact.manifest.model)
        descriptor = payload.stream.fileno()

        def failing_postcheck(candidate):
            if candidate == descriptor:
                raise RuntimeError("injected postcheck failure")
            return real_fstat(candidate)

        monkeypatch.setattr(manifest_module.os, "fstat", failing_postcheck)
        try:
            if semantic_failure:
                with (
                    pytest.raises(ValueError, match="semantic failure") as caught,
                    payload,
                ):
                    raise ValueError("semantic failure")
                assert isinstance(caught.value.__cause__, RuntimeError)
                assert "injected postcheck failure" in str(caught.value.__cause__)
            else:
                with pytest.raises(RuntimeError, match="injected postcheck failure"):
                    payload.close()

            assert payload.closed is True
            payload.close()
            with pytest.raises(OSError):
                real_fstat(descriptor)
        finally:
            payload.stream.close()
