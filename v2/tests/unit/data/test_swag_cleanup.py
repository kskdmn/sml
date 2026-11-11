from __future__ import annotations

import io
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from sml.artifacts import npy as npy_module
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    OpenedArtifact,
    PayloadRef,
    VerificationLevel,
)
from sml.errors import SMLArtifactError, SMLDataError

from .swag_helpers import (
    IDENTITY_A,
    VALID_ROW,
    FakeSwagProvider,
    _array_logical_path,
    _npy_bytes,
    _one_example_bundle,
    _replace_array_bytes,
    tiny_base_model,
    tiny_swag_config,
)


def test_swag_bundle_close_releases_views_mappings_payloads_then_root_once(
    tmp_path, monkeypatch
):
    from sml.artifacts.manifest import ArtifactRoot
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    events: list[str] = []
    original_release = swag._OwnedNpyMapping.release_view
    original_mapping_close = swag._OwnedNpyMapping.close_mapping
    original_payload_close = swag._OwnedNpyMapping.close_payload
    original_root_close = ArtifactRoot.close

    def release(owner):
        events.append("view")
        original_release(owner)

    def close_mapping(owner):
        events.append("mmap")
        original_mapping_close(owner)

    def close_payload(owner):
        events.append("payload")
        original_payload_close(owner)

    def close_root(root):
        events.append("root")
        original_root_close(root)

    monkeypatch.setattr(swag._OwnedNpyMapping, "release_view", release)
    monkeypatch.setattr(swag._OwnedNpyMapping, "close_mapping", close_mapping)
    monkeypatch.setattr(swag._OwnedNpyMapping, "close_payload", close_payload)
    monkeypatch.setattr(ArtifactRoot, "close", close_root)
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        tmp_path / "swag",
    )
    events.clear()
    owner_count = len(bundle._mappings)
    bundle.close()
    bundle.close()

    assert events == [
        *(["view"] * owner_count),
        *(["mmap"] * owner_count),
        *(["payload"] * owner_count),
        "root",
    ]
    with pytest.raises(SMLDataError, match="closed"):
        bundle.borrow_buckets()


def _instrument_real_swag_cleanup(monkeypatch, *, fail_open_index: int | None = None):
    from sml.artifacts.manifest import ArtifactRoot
    from sml.data import swag

    events: list[tuple[str, str | None]] = []
    owners: list[object] = []
    roots: list[ArtifactRoot] = []
    array_refs: dict[int, weakref.ReferenceType[np.ndarray]] = {}
    original_open = swag._OwnedNpyMapping.open.__func__
    original_release = swag._OwnedNpyMapping.release_view
    original_mapping_close = swag._OwnedNpyMapping.close_mapping
    original_payload_close = swag._OwnedNpyMapping.close_payload
    original_root_close = ArtifactRoot.close

    def open_mapping(cls, artifact, reference):
        if fail_open_index is not None and len(owners) == fail_open_index:
            raise RuntimeError("injected mapping acquisition failure")
        owner = original_open(cls, artifact, reference)
        owners.append(owner)
        array_refs[id(owner)] = weakref.ref(owner.array)
        return owner

    def release(owner):
        events.append(("view", owner.logical_path))
        original_release(owner)

    def close_mapping(owner):
        events.append(("mmap", owner.logical_path))
        assert array_refs[id(owner)]() is None, (
            f"live ndarray still exports {owner.logical_path}"
        )
        original_mapping_close(owner)

    def close_payload(owner):
        events.append(("payload", owner.logical_path))
        original_payload_close(owner)

    def close_root(root):
        roots.append(root)
        events.append(("root", None))
        original_root_close(root)

    monkeypatch.setattr(swag._OwnedNpyMapping, "open", classmethod(open_mapping))
    monkeypatch.setattr(swag._OwnedNpyMapping, "release_view", release)
    monkeypatch.setattr(swag._OwnedNpyMapping, "close_mapping", close_mapping)
    monkeypatch.setattr(swag._OwnedNpyMapping, "close_payload", close_payload)
    monkeypatch.setattr(ArtifactRoot, "close", close_root)
    return events, owners, roots


def _assert_real_swag_cleanup(events, owners, roots):
    logical_paths = [owner.logical_path for owner in reversed(owners)]
    assert events == [
        *(("view", path) for path in logical_paths),
        *(("mmap", path) for path in logical_paths),
        *(("payload", path) for path in logical_paths),
        ("root", None),
    ]
    assert owners
    assert all(owner.mapping.closed for owner in owners)
    assert all(owner.payload.closed for owner in owners)
    assert len(roots) == 1
    assert roots[0]._fd == -1


def test_swag_bundle_construction_failure_closes_partial_mapping_then_root(
    tmp_path, monkeypatch
):
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(
        monkeypatch, fail_open_index=1
    )

    with pytest.raises(RuntimeError, match="injected mapping acquisition failure"):
        load_swag_bundle(path, VerificationLevel.FULL)

    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_registration_failure_closes_pending_real_mapping_phase_wide(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    reference = next(
        reference
        for reference in prepared.manifest.buckets
        if reference.payload.logical_path.endswith("/input_ids.npy")
    )
    length = reference.arrays[0].shape[-1]
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)
    semantic_error = RuntimeError("injected owner registration failure")
    cleanup_error = RuntimeError("injected payload postcheck failure")
    instrumented_payload_close = swag._OwnedNpyMapping.close_payload

    class FailingRegistrationKey(str):
        def __hash__(self):
            raise semantic_error

    class SingleReferenceLookup:
        def __getitem__(self, _name):
            return reference

    def fail_after_payload_close(owner):
        instrumented_payload_close(owner)
        raise cleanup_error

    monkeypatch.setattr(
        swag,
        "_ARRAY_NAMES",
        (FailingRegistrationKey("input_ids"),),
    )
    monkeypatch.setattr(
        swag,
        "_group_manifest_buckets",
        lambda _manifest: ((length, SingleReferenceLookup()),),
    )
    monkeypatch.setattr(
        swag._OwnedNpyMapping,
        "close_payload",
        fail_after_payload_close,
    )

    with pytest.raises(RuntimeError, match="owner registration") as caught:
        load_swag_bundle(path, VerificationLevel.FULL)

    assert caught.value is semantic_error
    assert caught.value.__cause__ is cleanup_error
    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_pre_registration_failure_closes_pending_real_mapping_phase_wide(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)
    semantic_error = RuntimeError("injected pre-registration failure")
    instrumented_open = swag._OwnedNpyMapping.open.__func__
    registration_pending = False
    previous_trace = sys.gettrace()

    def open_then_arm_registration_failure(cls, artifact, reference):
        nonlocal registration_pending
        owner = instrumented_open(cls, artifact, reference)
        registration_pending = True
        return owner

    def fail_before_registration(frame, event, _argument):
        nonlocal registration_pending
        if event == "line" and frame.f_code is swag._open_buckets.__code__:
            if not registration_pending:
                return fail_before_registration
            registration_pending = False
            sys.settrace(previous_trace)
            raise semantic_error
        return fail_before_registration

    monkeypatch.setattr(
        swag._OwnedNpyMapping,
        "open",
        classmethod(open_then_arm_registration_failure),
    )
    sys.settrace(fail_before_registration)
    try:
        with pytest.raises(RuntimeError, match="pre-registration") as caught:
            load_swag_bundle(path, VerificationLevel.FULL)
    finally:
        sys.settrace(previous_trace)

    assert caught.value is semantic_error
    assert caught.value.__cause__ is None
    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_post_append_failure_closes_real_mapping_exactly_once(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)
    semantic_error = RuntimeError("injected post-append failure")
    cleanup_error = RuntimeError("injected payload postcheck failure")
    instrumented_open = swag._OwnedNpyMapping.open.__func__
    instrumented_payload_close = swag._OwnedNpyMapping.close_payload
    registration_pending = False
    registration_line_count = 0
    observed_pending_list_overlap = False
    previous_trace = sys.gettrace()

    def open_then_arm_registration_failure(cls, artifact, reference):
        nonlocal registration_pending
        owner = instrumented_open(cls, artifact, reference)
        registration_pending = True
        return owner

    def fail_after_list_registration(frame, event, _argument):
        nonlocal observed_pending_list_overlap
        nonlocal registration_pending, registration_line_count
        if event == "line" and frame.f_code is swag._open_buckets.__code__:
            if not registration_pending:
                return fail_after_list_registration
            registration_line_count += 1
            if registration_line_count == 2:
                observed_pending_list_overlap = (
                    frame.f_locals["mappings"][-1] is frame.f_locals["pending_owner"]
                )
                registration_pending = False
                sys.settrace(previous_trace)
                raise semantic_error
        return fail_after_list_registration

    def fail_after_payload_close(owner):
        instrumented_payload_close(owner)
        raise cleanup_error

    monkeypatch.setattr(
        swag._OwnedNpyMapping,
        "open",
        classmethod(open_then_arm_registration_failure),
    )
    monkeypatch.setattr(
        swag._OwnedNpyMapping,
        "close_payload",
        fail_after_payload_close,
    )
    sys.settrace(fail_after_list_registration)
    try:
        with pytest.raises(RuntimeError, match="post-append") as caught:
            load_swag_bundle(path, VerificationLevel.FULL)
    finally:
        sys.settrace(previous_trace)

    assert registration_line_count == 2
    assert observed_pending_list_overlap
    assert caught.value is semantic_error
    assert caught.value.__cause__ is cleanup_error
    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_semantic_failure_closes_real_mmaps_phase_wide(tmp_path, monkeypatch):
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    manifest = prepared.manifest
    with prepared.borrow_buckets() as buckets:
        labels = np.array(buckets[0].labels)
    prepared.close()
    labels[0] = 4
    _replace_array_bytes(
        path,
        manifest,
        _array_logical_path(manifest, "labels"),
        _npy_bytes(labels),
    )
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)

    with pytest.raises(SMLArtifactError, match="labels must be") as caught:
        load_swag_bundle(path, VerificationLevel.FULL)

    assert not isinstance(caught.value, BufferError)
    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_bundle_final_construction_failure_closes_all_resources(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)

    def fail_bundle_construction(*_args, **_kwargs):
        raise RuntimeError("bundle construction failed")

    monkeypatch.setattr(swag, "SwagDataBundle", fail_bundle_construction)

    with pytest.raises(RuntimeError, match="bundle construction failed"):
        load_swag_bundle(path, VerificationLevel.FULL)

    _assert_real_swag_cleanup(events, owners, roots)


def test_swag_bundle_root_detach_failure_closes_all_resources(tmp_path, monkeypatch):
    from sml.artifacts.manifest import OpenedArtifact
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events, owners, roots = _instrument_real_swag_cleanup(monkeypatch)

    def fail_detach(_artifact):
        raise RuntimeError("root detach failed")

    monkeypatch.setattr(OpenedArtifact, "detach_root", fail_detach)

    with pytest.raises(RuntimeError, match="root detach failed"):
        load_swag_bundle(path, VerificationLevel.FULL)

    _assert_real_swag_cleanup(events, owners, roots)


def test_owned_npy_mapping_close_attempts_every_phase_and_preserves_primary(
    monkeypatch,
):
    from sml.data import swag

    events: list[str] = []

    class FailingMapping:
        def close(self):
            events.append("mmap")
            raise RuntimeError("mmap close failed")

    class FailingPayload:
        def close(self):
            events.append("payload")
            raise RuntimeError("payload close failed")

    owner = swag._OwnedNpyMapping(
        "array.npy",
        FailingPayload(),
        FailingMapping(),
        np.zeros((1,), dtype="<i4"),
    )

    def fail_release(_owner):
        events.append("view")
        raise RuntimeError("view release failed")

    monkeypatch.setattr(swag._OwnedNpyMapping, "release_view", fail_release)
    with pytest.raises(RuntimeError, match="view release failed"):
        owner.close()

    assert events == ["view", "mmap", "payload"]
    assert owner._closed


def test_owned_npy_mapping_open_failure_clears_view_mapping_and_payload(
    monkeypatch,
):
    from sml.data import swag

    payload_bytes = _npy_bytes(np.array([7], dtype="<i4"))
    events: list[str] = []

    class FakeArray:
        dtype = np.dtype("<i4")

        def setflags(self, *, write):
            assert write is False

    class FakeMapping:
        def close(self):
            events.append("mmap")
            raise RuntimeError("mmap cleanup failed")

    class FakeStream(io.BytesIO):
        def fileno(self):
            return 42

    class FakePayload:
        stream = FakeStream(payload_bytes)
        opened_stat = SimpleNamespace(st_size=len(payload_bytes))

        def close(self):
            events.append("payload")
            raise RuntimeError("payload cleanup failed")

    class FakeArtifact:
        def open_payload(self, _reference):
            return FakePayload()

    reference = ArrayPayloadRef(
        payload=PayloadRef("array.npy", IDENTITY_A, len(payload_bytes)),
        arrays=(ArraySpec(name="array", shape=(1,), dtype="int32"),),
    )

    def fail_construction(*_args, **_kwargs):
        raise RuntimeError("owner construction failed")

    monkeypatch.setattr(
        npy_module.mmap, "mmap", lambda *_args, **_kwargs: FakeMapping()
    )
    monkeypatch.setattr(npy_module.np, "ndarray", lambda *_args, **_kwargs: FakeArray())
    monkeypatch.setattr(swag._OwnedNpyMapping, "__init__", fail_construction)

    with pytest.raises(RuntimeError, match="owner construction failed") as caught:
        swag._OwnedNpyMapping.open(FakeArtifact(), reference)

    assert events == ["mmap", "payload"]
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "mmap cleanup failed"


@pytest.mark.parametrize("failure_phase", ["ndarray", "owner"])
def test_owned_npy_mapping_real_acquisition_failure_closes_every_resource(
    tmp_path, monkeypatch, failure_phase
):
    from sml.artifacts.manifest import ArtifactRoot, VerifiedPayload
    from sml.data import swag
    from sml.data.swag import load_swag_bundle

    prepared = _one_example_bundle(tmp_path)
    path = prepared.path
    prepared.close()
    events: list[str] = []
    mappings: list[object] = []
    payloads: list[VerifiedPayload] = []
    roots: list[ArtifactRoot] = []
    array_refs: dict[int, weakref.ReferenceType[np.ndarray]] = {}
    original_mmap = npy_module.mmap.mmap
    original_ndarray = npy_module.np.ndarray
    original_init = swag._OwnedNpyMapping.__init__
    original_open_payload = OpenedArtifact.open_payload
    original_payload_close = VerifiedPayload.close
    original_root_close = ArtifactRoot.close

    class ObservedMmap(original_mmap):
        def __new__(cls, *args, **kwargs):
            mapping = super().__new__(cls, *args, **kwargs)
            mappings.append(mapping)
            return mapping

        def close(self):
            events.append("mmap")
            array_ref = array_refs.get(id(self))
            assert array_ref is None or array_ref() is None, (
                "live SWAG ndarray still exports a failed acquisition mapping"
            )
            return super().close()

    def construct_array(*args, **kwargs):
        array = original_ndarray(*args, **kwargs)
        mapping = kwargs.get("buffer")
        if isinstance(mapping, ObservedMmap):
            array_refs[id(mapping)] = weakref.ref(array)
            if failure_phase == "ndarray":
                raise RuntimeError("injected ndarray construction failure")
        return array

    def construct_owner(owner, logical_path, payload, mapping, array):
        if failure_phase == "owner":
            array_refs[id(mapping)] = weakref.ref(array)
            raise RuntimeError("injected owner construction failure")
        original_init(owner, logical_path, payload, mapping, array)

    def open_payload(artifact, reference):
        payload = original_open_payload(artifact, reference)
        payloads.append(payload)
        return payload

    def close_payload(payload):
        events.append("payload")
        return original_payload_close(payload)

    def close_root(root):
        roots.append(root)
        events.append("root")
        return original_root_close(root)

    monkeypatch.setattr(npy_module.mmap, "mmap", ObservedMmap)
    monkeypatch.setattr(npy_module.np, "ndarray", construct_array)
    monkeypatch.setattr(swag._OwnedNpyMapping, "__init__", construct_owner)
    monkeypatch.setattr(OpenedArtifact, "open_payload", open_payload)
    monkeypatch.setattr(VerifiedPayload, "close", close_payload)
    monkeypatch.setattr(ArtifactRoot, "close", close_root)

    with pytest.raises(RuntimeError, match=f"injected {failure_phase}") as caught:
        load_swag_bundle(path, VerificationLevel.FULL)

    assert caught.value.__cause__ is None
    assert events == ["mmap", "payload", "root"]
    assert len(mappings) == len(payloads) == len(roots) == 1
    assert mappings[0].closed
    assert payloads[0].closed
    assert roots[0]._fd == -1
    assert array_refs[id(mappings[0])]() is None
