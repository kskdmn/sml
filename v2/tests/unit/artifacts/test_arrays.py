from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import arrays as arrays_module
from sml.artifacts import load_safetensors_payload
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    BaseSnapshotManifest,
    PayloadRef,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    open_artifact,
)
from sml.errors import SMLArtifactError

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_TOKENIZER_IDENTITY = "sha256:" + "1" * 64
_RUN_IDENTITY = "sha256:" + "2" * 64


class InjectedFailure(RuntimeError):
    pass


def _write_array_artifact(
    root: Path,
    arrays: dict[str, mx.array],
    specs: tuple[ArraySpec, ...],
) -> ArrayPayloadRef:
    root.mkdir()
    payload_path = root / "model.safetensors"
    mx.save_safetensors(payload_path, arrays)
    with payload_path.open("rb") as payload:
        payload_ref = PayloadRef(
            logical_path="model.safetensors",
            identity=file_identity(payload),
            byte_size=payload_path.stat().st_size,
        )
    reference = ArrayPayloadRef(payload=payload_ref, arrays=specs)
    manifest = BaseSnapshotManifest(
        kind="base-snapshot",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model={"rope_scaling_factor": 1.0},
        precision={"working_parameter_dtype": "bfloat16"},
        tokenizer_identity=_TOKENIZER_IDENTITY,
        working_weights=reference,
        diagnostic_source_run_identity=_RUN_IDENTITY,
        diagnostic_source_step=0,
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return reference


def _open_array_artifact(root: Path):
    return open_artifact(
        root,
        (BaseSnapshotManifest,),
        VerificationLevel.FULL,
    )


def test_exact_array_contract_returns_eager_sorted_fresh_mapping(
    tmp_path: Path,
) -> None:
    """Dropping exact validation or sorting would expose ambiguous model leaves."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {
            "z.weight": mx.array([3.0, 4.0], dtype=mx.float32),
            "a.weight": mx.array([1.0, 2.0], dtype=mx.float32),
        },
        (
            ArraySpec("z.weight", (2,), "float32"),
            ArraySpec("a.weight", (2,), "float32"),
        ),
    )

    with _open_array_artifact(root) as artifact:
        loaded = load_safetensors_payload(artifact, reference)

    assert list(loaded) == ["a.weight", "z.weight"]
    assert mx.array_equal(loaded["a.weight"], mx.array([1.0, 2.0]))
    assert mx.array_equal(loaded["z.weight"], mx.array([3.0, 4.0]))


@pytest.mark.parametrize(
    ("arrays", "specs", "message"),
    [
        (
            {"weight": mx.array([1.0, 2.0], dtype=mx.float32)},
            (
                ArraySpec("weight", (2,), "float32"),
                ArraySpec("missing", (1,), "float32"),
            ),
            "keys",
        ),
        (
            {
                "weight": mx.array([1.0, 2.0], dtype=mx.float32),
                "extra": mx.array([3.0], dtype=mx.float32),
            },
            (ArraySpec("weight", (2,), "float32"),),
            "keys",
        ),
        (
            {"weight": mx.array([1.0, 2.0], dtype=mx.float32)},
            (ArraySpec("weight", (2,), "bfloat16"),),
            "metadata|dtype",
        ),
        (
            {"weight": mx.array([1.0, 2.0], dtype=mx.float32)},
            (ArraySpec("weight", (3,), "float32"),),
            "metadata|shape",
        ),
    ],
    ids=("missing-leaf", "extra-leaf", "wrong-dtype", "wrong-shape"),
)
def test_array_contract_rejects_false_leaf_declarations(
    tmp_path: Path,
    arrays: dict[str, mx.array],
    specs: tuple[ArraySpec, ...],
    message: str,
) -> None:
    """Accepting a false manifest contract would misbind persisted parameters."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(root, arrays, specs)

    with (
        _open_array_artifact(root) as artifact,
        pytest.raises(SMLArtifactError, match=message),
    ):
        load_safetensors_payload(artifact, reference)


@pytest.mark.parametrize(
    "loaded",
    [mx.array([1.0], dtype=mx.float32), {1: mx.array([1.0], dtype=mx.float32)}],
    ids=("not-a-mapping", "non-string-key"),
)
def test_loader_requires_a_string_keyed_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded: object,
) -> None:
    """Permitting non-name results would make exact leaf validation undefined."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"weight": mx.array([1.0], dtype=mx.float32)},
        (ArraySpec("weight", (1,), "float32"),),
    )
    monkeypatch.setattr(arrays_module.mx, "load", lambda *_args, **_kwargs: loaded)

    with (
        _open_array_artifact(root) as artifact,
        pytest.raises(SMLArtifactError, match="string-keyed mapping"),
    ):
        load_safetensors_payload(artifact, reference)


def test_full_load_consumes_retained_payload_after_root_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reopening the diagnostic path after proof would load replacement bytes."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"weight": mx.array([1.0, 2.0], dtype=mx.float32)},
        (ArraySpec("weight", (2,), "float32"),),
    )
    real_load = arrays_module.mx.load

    def replacing_load(stream, *, format):
        retained = tmp_path / "retained-bundle"
        root.rename(retained)
        root.mkdir()
        mx.save_safetensors(
            root / "model.safetensors",
            {"weight": mx.array([9.0, 9.0], dtype=mx.float32)},
        )
        return real_load(stream, format=format)

    monkeypatch.setattr(arrays_module.mx, "load", replacing_load)

    with _open_array_artifact(root) as artifact:
        loaded = load_safetensors_payload(artifact, reference)

    assert mx.array_equal(loaded["weight"], mx.array([1.0, 2.0]))


def test_in_place_mutation_during_mx_load_fails_payload_postcheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the postcheck would accept bytes changed during semantic load."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"weight": mx.array([1.0, 2.0], dtype=mx.float32)},
        (ArraySpec("weight", (2,), "float32"),),
    )
    real_load = arrays_module.mx.load

    def mutating_load(stream, *, format):
        loaded = real_load(stream, format=format)
        payload_path = root / "model.safetensors"
        before = payload_path.stat()
        os.utime(
            payload_path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
        )
        return loaded

    monkeypatch.setattr(arrays_module.mx, "load", mutating_load)

    with (
        _open_array_artifact(root) as artifact,
        pytest.raises(SMLArtifactError, match="changed during use"),
    ):
        load_safetensors_payload(artifact, reference)


def test_lazy_arrays_are_evaluated_in_sorted_order_before_payload_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferring mx.eval would let lazy arrays first touch a closed payload."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"z": mx.array([2.0]), "a": mx.array([1.0])},
        (
            ArraySpec("z", (1,), "float32"),
            ArraySpec("a", (1,), "float32"),
        ),
    )
    payload_stream = None
    evaluated: list[str] = []
    lazy_arrays = {}

    class LazyArray:
        shape = (1,)
        dtype = mx.float32

        def __init__(self, name: str) -> None:
            self.name = name

    def fake_load(stream, *, format):
        nonlocal payload_stream
        assert format == "safetensors"
        payload_stream = stream
        lazy_arrays.update({"z": LazyArray("z"), "a": LazyArray("a")})
        return lazy_arrays

    def fake_eval(*values):
        assert payload_stream is not None
        os.fstat(payload_stream.fileno())
        for value in values:
            evaluated.append(value.name)

    monkeypatch.setattr(arrays_module.mx, "load", fake_load)
    monkeypatch.setattr(arrays_module.mx, "eval", fake_eval)

    with _open_array_artifact(root) as artifact:
        loaded = load_safetensors_payload(artifact, reference)

    assert list(loaded) == ["a", "z"]
    assert loaded is not lazy_arrays
    assert evaluated == ["a", "z"]
    assert payload_stream is not None and payload_stream.closed


@pytest.mark.parametrize("failure_stage", ["load", "validation", "eval"])
def test_loader_failures_close_payload_but_leave_root_owned_by_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Any semantic failure must close the payload without stealing root ownership."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"weight": mx.array([1.0], dtype=mx.float32)},
        (ArraySpec("weight", (1,), "float32"),),
    )
    artifact = _open_array_artifact(root)
    root_descriptor = artifact.root._fd
    opened_payloads = []
    real_open_payload = artifact.open_payload

    def recording_open_payload(payload_reference):
        payload = real_open_payload(payload_reference)
        opened_payloads.append(payload)
        return payload

    class InvalidMetadataArray:
        dtype = mx.float32

        @property
        def shape(self):
            raise InjectedFailure("injected validation failure")

    def fake_load(*_args, **_kwargs):
        if failure_stage == "load":
            raise InjectedFailure("injected load failure")
        if failure_stage == "validation":
            return {"weight": InvalidMetadataArray()}
        return {"weight": mx.array([1.0], dtype=mx.float32)}

    def fake_eval(*_values):
        if failure_stage == "eval":
            raise InjectedFailure("injected eval failure")

    monkeypatch.setattr(artifact, "open_payload", recording_open_payload)
    monkeypatch.setattr(arrays_module.mx, "load", fake_load)
    monkeypatch.setattr(arrays_module.mx, "eval", fake_eval)

    with artifact:
        with pytest.raises(SMLArtifactError) as raised:
            load_safetensors_payload(artifact, reference)
        assert isinstance(raised.value.__cause__, InjectedFailure)
        assert "injected" in str(raised.value.__cause__)
        assert len(opened_payloads) == 1
        assert opened_payloads[0].closed is True
        assert opened_payloads[0].stream.closed is True
        os.fstat(root_descriptor)

    with pytest.raises(OSError):
        os.fstat(root_descriptor)


def test_malformed_array_metadata_is_normalized_and_closes_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed lazy metadata object must not escape as a public AttributeError."""
    root = tmp_path / "bundle"
    reference = _write_array_artifact(
        root,
        {"weight": mx.array([1.0], dtype=mx.float32)},
        (ArraySpec("weight", (1,), "float32"),),
    )
    artifact = _open_array_artifact(root)
    root_descriptor = artifact.root._fd
    opened_payloads = []
    real_open_payload = artifact.open_payload

    class MissingShape:
        dtype = mx.float32

        @property
        def shape(self):
            raise AttributeError("injected missing shape")

    def recording_open_payload(payload_reference):
        payload = real_open_payload(payload_reference)
        opened_payloads.append(payload)
        return payload

    monkeypatch.setattr(artifact, "open_payload", recording_open_payload)
    monkeypatch.setattr(
        arrays_module.mx,
        "load",
        lambda *_args, **_kwargs: {"weight": MissingShape()},
    )

    with artifact:
        with pytest.raises(SMLArtifactError) as raised:
            load_safetensors_payload(artifact, reference)
        assert isinstance(raised.value.__cause__, AttributeError)
        assert "injected missing shape" in str(raised.value.__cause__)
        assert len(opened_payloads) == 1
        assert opened_payloads[0].closed is True
        assert opened_payloads[0].stream.closed is True
        os.fstat(root_descriptor)

    with pytest.raises(OSError):
        os.fstat(root_descriptor)
