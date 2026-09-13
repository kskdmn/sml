from __future__ import annotations

import hashlib
import io
from contextlib import contextmanager

import numpy as np
import pytest
from sml.artifacts import npy as npy_module
from sml.artifacts.manifest import (
    ArtifactRoot,
    PayloadRef,
    VerificationLevel,
    _open_verified_payload,
)
from sml.artifacts.npy import VerifiedNpyMapping
from sml.errors import SMLArtifactError


@contextmanager
def _payload(tmp_path, array, *, version=(1, 0), verification=VerificationLevel.FULL):
    encoded = io.BytesIO()
    np.lib.format.write_array(encoded, array, version=version, allow_pickle=False)
    data = encoded.getvalue()
    path = tmp_path / "array.npy"
    path.write_bytes(data)
    reference = PayloadRef(
        path.name, "sha256:" + hashlib.sha256(data).hexdigest(), len(data)
    )
    with ArtifactRoot.open(tmp_path, writable=False) as root:
        payload = _open_verified_payload(root, reference, verification)
        try:
            yield payload
        finally:
            payload.close()


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
@pytest.mark.parametrize("dtype", ["<i4", "bool"])
@pytest.mark.parametrize(
    "verification", [VerificationLevel.FULL, VerificationLevel.MANIFEST_TRUSTED]
)
def test_npy_mapping_retains_verified_descriptor_and_readonly_storage(
    tmp_path, version, dtype, verification
):
    expected = np.array([[0, 1], [1, 0]], dtype=dtype)
    with _payload(
        tmp_path, expected, version=version, verification=verification
    ) as payload:
        with VerifiedNpyMapping.from_payload(
            payload,
            logical_path="array.npy",
            expected_shape=expected.shape,
            expected_dtype=expected.dtype,
            description="test array",
        ) as owner:
            assert owner.payload is payload
            assert owner.payload.verification is verification
            assert owner.array.base is owner.mapping
            assert owner.array.flags.c_contiguous
            assert not owner.array.flags.writeable
            np.testing.assert_array_equal(owner.array, expected)
        assert owner.mapping.closed
        assert payload.closed
        owner.close()


@pytest.mark.parametrize("failure_phase", ["mapping", "array"])
def test_npy_acquisition_errors_preserve_domain_error_and_close_payload(
    tmp_path, monkeypatch, failure_phase
):
    expected = np.array([[7]], dtype="<i4")
    acquisition_error = OSError("injected acquisition failure")
    mappings = []
    original_mmap = npy_module.mmap.mmap

    def open_mapping(*args, **kwargs):
        if failure_phase == "mapping":
            raise acquisition_error
        mapping = original_mmap(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    def fail_array(*_args, **_kwargs):
        raise acquisition_error

    with _payload(tmp_path, expected) as payload:
        monkeypatch.setattr(npy_module.mmap, "mmap", open_mapping)
        if failure_phase == "array":
            monkeypatch.setattr(npy_module.np, "ndarray", fail_array)
        with pytest.raises(SMLArtifactError, match="could not memory-map") as caught:
            VerifiedNpyMapping.from_payload(
                payload,
                logical_path="array.npy",
                expected_shape=expected.shape,
                expected_dtype=expected.dtype,
                description="test array",
            )
        assert caught.value.__cause__ is acquisition_error
        assert all(mapping.closed for mapping in mappings)
        assert payload.closed
