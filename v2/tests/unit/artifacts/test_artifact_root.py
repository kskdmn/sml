from __future__ import annotations

import io
import os

import pytest
from sml.artifacts import manifest as artifacts
from sml.errors import SMLArtifactError


def _payload_ref(logical_path: str, data: bytes) -> artifacts.PayloadRef:
    return artifacts.PayloadRef(
        logical_path=logical_path,
        identity=artifacts.file_identity(io.BytesIO(data)),
        byte_size=len(data),
    )


@pytest.mark.parametrize(
    "logical_path",
    ["/absolute", "../escape", "a//b", "a/./b", "a\\b", "trail."],
)
def test_logical_paths_are_rejected_before_open(tmp_path, logical_path):
    """Relaxing lexical validation would let manifest names escape the root."""
    with (
        artifacts.ArtifactRoot.open(tmp_path, writable=False) as root,
        pytest.raises(SMLArtifactError, match="logical path"),
    ):
        root.open_payload(logical_path)


def test_logical_path_components_are_nfkc_normalized():
    """Omitting NFKC would give compatibility-equivalent names distinct identities."""
    assert artifacts.parse_logical_path("nested/ｍodel-１.bin") == (
        "nested",
        "model-1.bin",
    )


def test_unicode_normalized_paths_collide(tmp_path):
    """Dropping normalized collision checks would make one payload have two names."""
    data = b"weights"
    (tmp_path / "model.bin").write_bytes(data)
    references = (
        _payload_ref("model.bin", data),
        _payload_ref("ｍodel.bin", data),
    )

    with (
        artifacts.ArtifactRoot.open(tmp_path, writable=False) as root,
        pytest.raises(SMLArtifactError, match="normalized path collision"),
    ):
        root.verify_payloads(references, full=False)


def test_casefolded_paths_collide(tmp_path):
    """Omitting case-fold collision checks would make artifacts host-dependent."""
    data = b"weights"
    (tmp_path / "Model.bin").write_bytes(data)
    (tmp_path / "model.bin").write_bytes(data)
    references = (
        _payload_ref("Model.bin", data),
        _payload_ref("model.bin", data),
    )

    with (
        artifacts.ArtifactRoot.open(tmp_path, writable=False) as root,
        pytest.raises(SMLArtifactError, match="case-folded path collision"),
    ):
        root.verify_payloads(references, full=False)


def test_two_logical_paths_cannot_share_inode(tmp_path, monkeypatch):
    """Omitting inode tracking would let distinct names alias one payload object."""
    data = b"weights"
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(data)
    os.link(first, second)
    aliased_inode = first.stat().st_ino
    real_fstat = os.fstat

    def single_link_fstat(fd: int) -> os.stat_result:
        result = real_fstat(fd)
        if result.st_ino != aliased_inode:
            return result
        values = list(result)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(artifacts.os, "fstat", single_link_fstat)
    references = (
        _payload_ref("first.bin", data),
        _payload_ref("second.bin", data),
    )

    with (
        artifacts.ArtifactRoot.open(tmp_path, writable=False) as root,
        pytest.raises(SMLArtifactError, match="inode alias"),
    ):
        root.verify_payloads(references, full=False)


def test_open_payload_rejects_symlink_swap_and_hard_link_alias(tmp_path):
    """Reopening by path would follow a swapped directory or accept linked bytes."""
    bundle = tmp_path / "bundle"
    nested = bundle / "nested"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"original")
    external = tmp_path / "external"
    external.mkdir()
    (external / "model.safetensors").write_bytes(b"outside")

    with artifacts.ArtifactRoot.open(bundle, writable=False) as root:
        nested.rename(bundle / "nested-old")
        nested.symlink_to(external, target_is_directory=True)
        with pytest.raises(SMLArtifactError, match="symlink|no-follow"):
            root.open_payload("nested/model.safetensors")

    linked_bundle = tmp_path / "linked-bundle"
    linked_bundle.mkdir()
    external_payload = tmp_path / "external.bin"
    external_payload.write_bytes(b"shared")
    os.link(external_payload, linked_bundle / "payload.bin")
    reference = _payload_ref("payload.bin", b"shared")

    with (
        artifacts.ArtifactRoot.open(linked_bundle, writable=False) as root,
        pytest.raises(SMLArtifactError, match="link count"),
    ):
        root.verify_payloads((reference,), full=True)


def test_root_descriptor_lives_until_context_exit(tmp_path):
    """Closing the owned root early would make later relative opens unsafe."""
    (tmp_path / "payload.bin").write_bytes(b"payload")
    with artifacts.ArtifactRoot.open(tmp_path, writable=False) as root:
        descriptor = root._fd
        os.fstat(descriptor)
        with root.open_payload("payload.bin") as payload:
            assert payload.read() == b"payload"
        os.fstat(descriptor)

    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_payload_descriptor_is_owned_by_returned_stream(tmp_path):
    """Closing a successful final descriptor early would invalidate its stream."""
    (tmp_path / "payload.bin").write_bytes(b"payload")
    with artifacts.ArtifactRoot.open(tmp_path, writable=False) as root:
        payload = root.open_payload("payload.bin")
        descriptor = payload.fileno()
        os.fstat(descriptor)
        payload.close()
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("existing_payload", [True, False])
def test_intermediate_descriptor_closes_after_payload_open(
    tmp_path, monkeypatch, existing_payload
):
    """Leaking a walked directory descriptor would exhaust long-lived readers."""
    nested = tmp_path / "nested"
    nested.mkdir()
    if existing_payload:
        (nested / "payload.bin").write_bytes(b"payload")
    real_open = os.open
    intermediate_descriptors: list[int] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "nested":
            intermediate_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(artifacts.os, "open", recording_open)
    with artifacts.ArtifactRoot.open(tmp_path, writable=False) as root:
        if existing_payload:
            with root.open_payload("nested/payload.bin") as payload:
                assert payload.read() == b"payload"
        else:
            with pytest.raises(SMLArtifactError, match="no-follow"):
                root.open_payload("nested/payload.bin")

        assert len(intermediate_descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(intermediate_descriptors[0])


def test_non_apfs_writer_is_rejected(tmp_path, monkeypatch):
    """Granting write authority off local APFS would break publication guarantees."""
    monkeypatch.setattr(
        artifacts, "_descriptor_is_local_apfs", lambda _descriptor: False, raising=False
    )

    with pytest.raises(SMLArtifactError, match="local APFS"):
        artifacts.ArtifactRoot.open(tmp_path, writable=True)


def test_non_apfs_read_only_open_reports_no_writer_guarantee(tmp_path, monkeypatch):
    """Reporting APFS guarantees for a foreign reader would overstate its authority."""
    monkeypatch.setattr(
        artifacts, "_descriptor_is_local_apfs", lambda _descriptor: False, raising=False
    )

    with artifacts.ArtifactRoot.open(tmp_path, writable=False) as root:
        assert root.local_apfs is False
