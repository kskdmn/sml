from __future__ import annotations

import io
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    OpenedArtifact,
    SwagDataManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError, SMLDataError

from .swag_helpers import (
    VALID_ROW,
    FakeSwagProvider,
    _array_logical_path,
    _npy_bytes,
    _one_example_bundle,
    _replace_array_bytes,
    tiny_base_model,
    tiny_swag_config,
)


def test_load_swag_bundle_reopens_without_provider(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    try:
        assert loaded.manifest.identity == prepared.manifest.identity
        with (
            loaded.borrow_buckets() as loaded_buckets,
            prepared.borrow_buckets() as prepared_buckets,
        ):
            assert (
                loaded_buckets[0].input_ids.shape == prepared_buckets[0].input_ids.shape
            )
    finally:
        loaded.close()
        prepared.close()


def _rewrite_swag_manifest(path: Path, **overrides: object) -> None:
    verified = read_manifest(path, SwagDataManifest, VerificationLevel.FULL)
    manifest = replace(verified.manifest, **overrides)
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _rewrite_swag_array(path: Path, logical_path: str, mutate) -> None:
    verified = read_manifest(path, SwagDataManifest, VerificationLevel.FULL)
    array_path = path / logical_path
    mutated = np.ascontiguousarray(
        mutate(np.array(np.load(array_path, allow_pickle=False)))
    )
    with array_path.open("wb") as destination:
        np.save(destination, mutated, allow_pickle=False)
    with array_path.open("rb") as payload:
        identity = file_identity(payload)
    updated = []
    for reference in verified.manifest.buckets:
        if reference.payload.logical_path != logical_path:
            updated.append(reference)
            continue
        spec = reference.arrays[0]
        updated.append(
            ArrayPayloadRef(
                payload=replace(
                    reference.payload,
                    identity=identity,
                    byte_size=array_path.stat().st_size,
                ),
                arrays=(replace(spec, shape=tuple(int(dim) for dim in mutated.shape)),),
            )
        )
    manifest = replace(verified.manifest, buckets=tuple(updated))
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def _object_npy_bytes(_array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(
        payload,
        np.array(["object"], dtype=object),
        version=(1, 0),
        allow_pickle=True,
    )
    return payload.getvalue()


@pytest.mark.parametrize(
    ("name", "payload_factory", "message"),
    [
        (
            "unsupported-version",
            lambda array: _npy_bytes(array, version=(3, 0)),
            "version",
        ),
        (
            "fortran-order",
            lambda array: _npy_bytes(np.asfortranarray(array)),
            "C order",
        ),
        (
            "dtype",
            lambda array: _npy_bytes(array.astype("<i8")),
            "dtype",
        ),
        ("object-dtype", _object_npy_bytes, "dtype"),
        (
            "shape",
            lambda array: _npy_bytes(array.reshape(-1)),
            "shape",
        ),
        (
            "trailing-bytes",
            lambda array: _npy_bytes(array) + b"trailing",
            "size",
        ),
        (
            "short-bytes",
            lambda array: _npy_bytes(array)[:-1],
            "size",
        ),
        (
            "invalid-data-offset",
            lambda array: _npy_bytes(array)[:9] + b"\xff\xff" + _npy_bytes(array)[11:],
            "header",
        ),
    ],
)
def test_load_swag_bundle_rejects_invalid_descriptor_bound_npy(
    tmp_path, name, payload_factory, message
):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    output = tmp_path / name
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        output,
    )
    manifest = prepared.manifest
    logical_path = _array_logical_path(manifest, "input_ids")
    with prepared.borrow_buckets() as buckets:
        original = np.array(buckets[0].input_ids)
    prepared.close()
    _replace_array_bytes(
        output,
        manifest,
        logical_path,
        payload_factory(original),
    )

    with pytest.raises(SMLArtifactError, match=message):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_swag_bundle_uses_retained_root_after_path_replacement(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        output,
    )
    with prepared.borrow_buckets() as buckets:
        expected = np.array(buckets[0].input_ids)
    prepared.close()
    original_open = swag.open_artifact

    def replace_after_open(path, manifest_types, verification):
        artifact = original_open(path, manifest_types, verification)
        retained = output.with_name("retained-swag")
        output.rename(retained)
        shutil.copytree(retained, output)
        logical_path = _array_logical_path(artifact.manifest, "input_ids")
        replacement = np.zeros_like(expected)
        _replace_array_bytes(
            output, artifact.manifest, logical_path, _npy_bytes(replacement)
        )
        return artifact

    monkeypatch.setattr(swag, "open_artifact", replace_after_open, raising=False)
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    try:
        with loaded.borrow_buckets() as buckets:
            assert np.array_equal(buckets[0].input_ids, expected)
    finally:
        loaded.close()


@pytest.mark.parametrize(
    "array_name", ("input_ids", "valid_token_mask", "score_mask", "labels")
)
def test_load_swag_bundle_uses_proven_payload_after_path_replacement(
    tmp_path, monkeypatch, array_name
):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        output,
    )
    manifest = prepared.manifest
    with prepared.borrow_buckets() as buckets:
        expected = np.array(getattr(buckets[0], array_name))
    prepared.close()
    logical_path = _array_logical_path(manifest, array_name)
    original_open_payload = OpenedArtifact.open_payload

    def replace_after_proof(artifact, reference):
        payload = original_open_payload(artifact, reference)
        if reference.logical_path == logical_path:
            source = output / logical_path
            retained = source.with_suffix(".proven.npy")
            source.rename(retained)
            replacement = np.zeros_like(expected)
            source.write_bytes(_npy_bytes(replacement))
        return payload

    monkeypatch.setattr(OpenedArtifact, "open_payload", replace_after_proof)
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    with loaded.borrow_buckets() as buckets:
        assert np.array_equal(getattr(buckets[0], array_name), expected)
    with pytest.raises(SMLArtifactError, match="changed during use"):
        loaded.close()


def test_swag_mapping_is_read_only_and_close_detects_in_place_mutation(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        output,
    )
    manifest = prepared.manifest
    prepared.close()
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    with (
        loaded.borrow_buckets() as buckets,
        pytest.raises(ValueError, match="read-only"),
    ):
        buckets[0].input_ids[0, 0, 0] = 7

    logical_path = _array_logical_path(manifest, "input_ids")
    payload_path = output / logical_path
    with payload_path.open("r+b") as payload:
        payload.seek(-1, 2)
        final_byte = payload.read(1)
        payload.seek(-1, 2)
        payload.write(bytes([final_byte[0] ^ 1]))
        payload.flush()
    with pytest.raises(SMLArtifactError, match="changed during use"):
        loaded.close()
    with pytest.raises(SMLDataError, match="closed"):
        loaded.borrow_buckets()


def test_public_swag_array_remains_safe_after_bundle_close_in_subprocess(tmp_path):
    """Retaining a public bucket array must never leave a dangling mmap view."""
    from sml.data.swag import prepare_swag_bundle

    output = tmp_path / "swag-public-borrower"
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        output,
    )
    with prepared.borrow_buckets() as buckets:
        expected = int(buckets[0].input_ids[0, 0, 0])
    prepared.close()

    program = """
import sys
from pathlib import Path

from sml.artifacts.manifest import VerificationLevel
from sml.data.swag import load_swag_bundle
from sml.errors import SMLDataError

bundle = load_swag_bundle(Path(sys.argv[1]), VerificationLevel.FULL)
lease = bundle.borrow_buckets()
retained = lease.buckets[0].input_ids
try:
    bundle.close()
except SMLDataError:
    pass
else:
    raise AssertionError("bundle close accepted a live public bucket lease")
print(int(retained[0, 0, 0]))
lease.close()
try:
    retained[0, 0, 0]
except SMLDataError:
    pass
else:
    raise AssertionError("closed bucket lease remained addressable")
bundle.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(expected)


def test_load_swag_bundle_rejects_unsupported_policy_before_opening_arrays(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    preprocessing = dict(prepared.manifest.preprocessing)
    prepared.close()
    preprocessing["join_policy"] = "other-join-v1"
    _rewrite_swag_manifest(output, preprocessing=preprocessing)

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("arrays opened before projection validation")

    monkeypatch.setattr(swag, "_open_buckets", fail_open)
    with pytest.raises(SMLArtifactError, match="join"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_swag_bundle_rejects_unsupported_schema_before_opening_arrays(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    preprocessing = dict(prepared.manifest.preprocessing)
    prepared.close()
    preprocessing["schema_version"] = 2
    _rewrite_swag_manifest(output, preprocessing=preprocessing)

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("arrays opened before projection validation")

    monkeypatch.setattr(swag, "_open_buckets", fail_open)
    with pytest.raises(SMLArtifactError, match="schema"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_swag_bundle_rejects_bucket_policy_that_misses_maximum_length(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    preprocessing = dict(prepared.manifest.preprocessing)
    preprocessing["maximum_length"] = 10_000
    prepared.close()
    _rewrite_swag_manifest(output, preprocessing=preprocessing)

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("arrays opened before projection validation")

    monkeypatch.setattr(swag, "_open_buckets", fail_open)
    with pytest.raises(SMLArtifactError, match="bucket|maximum_length"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_swag_bundle_rejects_duplicate_special_ids_before_opening_arrays(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    duplicate_id = prepared.manifest.bos_token_id
    prepared.close()

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("arrays opened before projection validation")

    monkeypatch.setattr(swag, "_open_buckets", fail_open)
    _rewrite_swag_manifest(output, pad_token_id=duplicate_id)
    with pytest.raises(SMLArtifactError, match="special"):
        load_swag_bundle(output, VerificationLevel.FULL)


@pytest.mark.parametrize(
    "array_name,position,value,message",
    [
        pytest.param(
            "input_ids", "bos", "vocab_size", "vocab", id="token-outside-vocabulary"
        ),
        pytest.param("score_mask", "padding", True, "score", id="scored-padding"),
        pytest.param(
            "input_ids", "padding", 10, "padding|valid", id="non-padding-token"
        ),
        pytest.param(
            "valid_token_mask",
            "mask-hole",
            False,
            "padding|valid",
            id="valid-mask-hole",
        ),
        pytest.param(
            "score_mask", "candidate", False, "scored", id="no-scored-continuation"
        ),
        pytest.param("score_mask", "eos", False, "eos|EOS", id="unscored-eos"),
        pytest.param("input_ids", "eos", 10, "eos|EOS", id="missing-eos"),
        pytest.param(
            "score_mask",
            "score-hole",
            False,
            "score|suffix|contiguous",
            id="score-mask-hole",
        ),
        pytest.param("input_ids", "bos", 10, "BOS|bos", id="missing-bos"),
    ],
)
def test_load_rejects_invalid_candidate_arrays(
    tmp_path, array_name, position, value, message
):
    from sml.data.swag import load_swag_bundle

    with _one_example_bundle(tmp_path) as prepared:
        output = prepared.path
        manifest = prepared.manifest
    valid = np.load(output / _array_logical_path(manifest, "valid_token_mask"))
    last = int(valid[0, 0].sum()) - 1
    positions = {
        "bos": 0,
        "padding": -1,
        "mask-hole": 1,
        "candidate": slice(None),
        "eos": last,
        "score-hole": last - 1,
    }
    replacement = manifest.vocab_size if value == "vocab_size" else value

    def mutate(array):
        array[0, 0, positions[position]] = replacement
        return array

    _rewrite_swag_array(output, _array_logical_path(manifest, array_name), mutate)
    with pytest.raises(SMLArtifactError, match=message):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_full_swag_reduction_operands_are_bounded_to_1024_rows(tmp_path, monkeypatch):
    from sml.data import swag
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    row_count = 2_049
    for name in ("input_ids", "valid_token_mask", "score_mask", "labels"):
        logical = _array_logical_path(manifest, name)
        _rewrite_swag_array(
            output,
            logical,
            lambda array: np.repeat(array, row_count, axis=0),
        )
    _rewrite_swag_manifest(output, example_count=row_count)

    recorded: list[tuple[int, ...]] = []
    real_np = np

    class RecordingNumpy:
        def __getattr__(self, name):
            return getattr(real_np, name)

        def _record(self, operation, operand, *args, **kwargs):
            recorded.append(np.shape(operand))
            return operation(operand, *args, **kwargs)

        def min(self, operand, *args, **kwargs):
            return self._record(real_np.min, operand, *args, **kwargs)

        def max(self, operand, *args, **kwargs):
            return self._record(real_np.max, operand, *args, **kwargs)

        def any(self, operand, *args, **kwargs):
            return self._record(real_np.any, operand, *args, **kwargs)

        def all(self, operand, *args, **kwargs):
            return self._record(real_np.all, operand, *args, **kwargs)

    monkeypatch.setattr(swag, "np", RecordingNumpy())
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    loaded.close()

    row_operands = [shape for shape in recorded if shape and shape[0] > 1]
    assert row_operands
    assert (1_024,) in row_operands
    assert max(shape[0] for shape in row_operands) <= 1_024


def test_load_rejects_bucket_length_that_is_not_a_declared_boundary(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    with prepared.borrow_buckets() as buckets:
        bucket_length = buckets[0].length
    preprocessing = dict(prepared.manifest.preprocessing)
    prepared.close()
    preprocessing["bucket_boundaries"] = [8, 32]
    assert bucket_length not in preprocessing["bucket_boundaries"]
    _rewrite_swag_manifest(output, preprocessing=preprocessing)
    with pytest.raises(SMLArtifactError, match="bucket"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_valid_sequence_longer_than_maximum_length(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    valid = np.load(output / _array_logical_path(manifest, "valid_token_mask"))
    valid_length = int(np.asarray(valid[0, 0]).sum())
    assert valid_length > 4
    preprocessing = dict(manifest.preprocessing)
    preprocessing["maximum_length"] = 4
    _rewrite_swag_manifest(output, preprocessing=preprocessing)
    with pytest.raises(SMLArtifactError, match="maximum_length"):
        load_swag_bundle(output, VerificationLevel.FULL)
