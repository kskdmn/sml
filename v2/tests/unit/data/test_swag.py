from __future__ import annotations

import inspect
import io
import shutil
import subprocess
import sys
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    OpenedArtifact,
    PayloadRef,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
    read_manifest,
)
from sml.errors import SMLArtifactError, SMLDataError
from sml.inference import ResolvedModel
from sml.model.config import ModelConfig

IDENTITY_A = "sha256:" + "a" * 64
IDENTITY_B = "sha256:" + "b" * 64
IDENTITY_C = "sha256:" + "c" * 64

VALID_ROW: dict[str, object] = {
    "context": "the cat sat",
    "endings": ("on the mat", "in the car", "by the door", "near a tree"),
    "label": 1,
}


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.encode_calls: list[object] = []

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        self.encode_calls.append(text)
        if isinstance(text, str):
            self.calls.append(text)
            return self._encode_one(text)
        encoded = []
        for item in text:
            self.calls.append(item)
            encoded.append(self._encode_one(item))
        return encoded

    @staticmethod
    def _encode_one(text: str) -> list[int]:
        words = text.split()
        if not words:
            return []
        return [10 + (index % 20) for index, _ in enumerate(words)]


class FixedProcessor:
    def __init__(self, encoded: Mapping[str, tuple[int, ...]]) -> None:
        self._encoded = dict(encoded)
        self.calls: list[str] = []
        self.encode_calls: list[object] = []

    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        self.encode_calls.append(text)
        if isinstance(text, str):
            self.calls.append(text)
            return self._encode_one(text)
        encoded = []
        for item in text:
            self.calls.append(item)
            encoded.append(self._encode_one(item))
        return encoded

    def _encode_one(self, text: str) -> list[int]:
        try:
            return list(self._encoded[text])
        except KeyError as error:
            raise AssertionError(f"unexpected encode input: {text!r}") from error


class FakeSwagProvider:
    def __init__(
        self,
        rows: tuple[Mapping[str, object], ...],
        *,
        commit: str = "abc123def456",
        fingerprint: str = "fingerprint-v1",
        package: str = "datasets",
        version: str = "2.0.0",
    ) -> None:
        self.rows = rows
        self.commit = commit
        self.fingerprint = fingerprint
        self.package = package
        self.version = version
        self.resolve_calls = 0
        self.iter_calls = 0
        self.fail_resolve = False
        self.fail_iter = False
        self.fail_iter_after_rows = False
        self.fail_iter_data_error = False
        self.fail_iter_after_rows_data_error = False
        self.resolved_namespace = None

    def resolve(self, source):
        self.resolve_calls += 1
        if self.fail_resolve:
            raise RuntimeError("provider unavailable")
        from sml.data.swag import ResolvedSwagSource

        namespace = (
            source.namespace
            if self.resolved_namespace is None
            else self.resolved_namespace
        )
        return ResolvedSwagSource(
            backend=source.backend,
            namespace=namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit=self.commit,
            provider_fingerprint=self.fingerprint,
            provider_package=self.package,
            provider_version=self.version,
        )

    def iter_rows(self, resolved) -> Iterator[Mapping[str, object]]:
        self.iter_calls += 1
        if self.fail_iter:
            raise RuntimeError("provider unavailable")
        if self.fail_iter_data_error:
            raise SMLDataError("the datasets package is required")
        yield from self.rows
        if self.fail_iter_after_rows:
            raise RuntimeError("provider unavailable")
        if self.fail_iter_after_rows_data_error:
            raise SMLDataError("the datasets package is required")


class RecordingTokenizer:
    def __init__(self, manifest: TokenizerManifest, processor: object) -> None:
        self.path = Path("recording-tokenizer")
        self.manifest = manifest
        self.verification = VerificationLevel.FULL
        self.processor = processor

    @property
    def calls(self) -> list[str]:
        return self.processor.calls


def tokenizer_manifest(**overrides: object) -> TokenizerManifest:
    values: dict[str, object] = {
        "kind": "tokenizer",
        "version": 1,
        "identity": IDENTITY_A,
        "algorithm": "sentencepiece-bpe-v1",
        "training": {"normalization": "nmt_nfkc"},
        "vocab_size": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "unk_token_id": 0,
        "model": PayloadRef("tokenizer.model", IDENTITY_A, 10),
        "vocab": PayloadRef("tokenizer.vocab", IDENTITY_B, 20),
        "diagnostic_source_locator": "/source/tokenizer",
    }
    values.update(overrides)
    return TokenizerManifest(**values)


def tiny_model_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 32,
        "original_context_length": 32,
        "hidden_dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def tiny_base_model(
    *,
    processor: object | None = None,
    tokenizer_manifest_overrides: Mapping[str, object] | None = None,
    model_overrides: Mapping[str, object] | None = None,
) -> ResolvedModel:
    processor = RecordingProcessor() if processor is None else processor
    tokenizer = RecordingTokenizer(
        tokenizer_manifest(**dict(tokenizer_manifest_overrides or {})),
        processor,
    )
    return ResolvedModel(
        artifact_kind="pretraining-checkpoint",
        run_identity=IDENTITY_B,
        step=1,
        checkpoint_identity=IDENTITY_C,
        run_step_identity=IDENTITY_A,
        verification=VerificationLevel.FULL,
        model_config=tiny_model_config(**dict(model_overrides or {})),
        tokenizer=tokenizer,
        model_arrays={},
    )


def tiny_swag_config(provider, **overrides):
    from sml.data.swag import SwagPreparationConfig, SwagSourceConfig

    values = {
        "provider": provider,
        "source": SwagSourceConfig(revision="deadbeef" * 5),
        "maximum_length": 32,
        "bucket_boundaries": (16, 32),
    }
    values.update(overrides)
    return SwagPreparationConfig(**values)


def change_identity_field(config, field: str):
    replacements = {
        "namespace": "other-org",
        "name": "other-swag",
        "dataset_config": "full",
        "revision": "cafebabe" * 5,
        "split": "val",
        "preprocessing_schema_version": 2,
        "join_policy": "other-join-v1",
        "overlength_policy": "other-drop-v1",
        "bos_policy": "other-bos-v1",
        "eos_policy": "other-eos-v1",
        "maximum_length": 24,
        "bucket_boundaries": (32,),
    }
    if field not in replacements:
        raise AssertionError(f"unknown identity field: {field}")
    value = replacements[field]
    if field in {"namespace", "name", "dataset_config", "revision", "split"}:
        return replace(config, source=replace(config.source, **{field: value}))
    return replace(config, **{field: value})


def assert_eos_positions_are_scored(bucket, eos_token_id: int) -> None:
    eos = bucket.input_ids == eos_token_id
    assert eos.any()
    assert bool(np.all(bucket.score_mask[eos]))
    assert bool(np.all(bucket.valid_token_mask[eos]))


def replace_row(**overrides: object) -> dict[str, object]:
    row = dict(VALID_ROW)
    row.update(overrides)
    return row


def test_importing_swag_module_does_not_import_datasets(monkeypatch):
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.delitem(sys.modules, "sml.data.swag", raising=False)
    from sml.data import swag

    assert "datasets" not in sys.modules
    assert swag.HuggingFaceDatasetsSwagProvider is not None


def test_context_and_endings_are_encoded_separately_and_eos_is_scored(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    recording_tokenizer = base.tokenizer
    bundle = prepare_swag_bundle(
        tiny_swag_config(provider),
        base,
        tmp_path / "swag",
    )
    try:
        assert recording_tokenizer.calls[0] == VALID_ROW["context"]
        assert tuple(recording_tokenizer.calls[1:5]) == tuple(VALID_ROW["endings"])
        with bundle.borrow_buckets() as buckets:
            bucket = buckets[0]
            assert not bucket.score_mask[:, :, 0].any()
            assert bucket.score_mask[bucket.valid_token_mask].any()
            assert (~bucket.score_mask & bucket.valid_token_mask).any()
            assert_eos_positions_are_scored(bucket, eos_token_id=2)
            assert bucket.input_ids.dtype == np.dtype("<i4")
            assert bucket.labels.dtype == np.dtype("<i4")
            assert bucket.valid_token_mask.dtype == np.dtype("bool")
            assert bucket.score_mask.dtype == np.dtype("bool")
            assert bucket.input_ids.shape == bucket.valid_token_mask.shape
            assert bucket.input_ids.shape[1:] == (4, bucket.input_ids.shape[-1])
    finally:
        bundle.close()


def test_chunk_with_multiple_rows_makes_one_tokenizer_encode_call(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    processor = RecordingProcessor()
    rows = (
        VALID_ROW,
        replace_row(label=2),
    )
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(processor=processor),
        tmp_path / "swag",
    )
    bundle.close()
    assert len(processor.encode_calls) == 1
    batched = processor.encode_calls[0]
    assert batched == [
        VALID_ROW["context"],
        *VALID_ROW["endings"],
        rows[1]["context"],
        *rows[1]["endings"],
    ]
    assert processor.calls[0] == VALID_ROW["context"]
    assert tuple(processor.calls[1:5]) == tuple(VALID_ROW["endings"])


def test_npy_identity_is_hashed_while_writing(tmp_path, monkeypatch):
    from sml.artifacts.manifest import file_identity
    from sml.data import swag

    opened: list[str] = []
    original_open = Path.open

    def tracking_open(self, mode="r", *args, **kwargs):
        if self.suffix == ".npy":
            opened.append(mode)
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    path = tmp_path / "array.npy"
    reference = swag._write_npy(path, np.arange(8, dtype=np.int32), "array.npy")
    assert opened == ["xb"]
    with path.open("rb") as payload:
        assert file_identity(payload) == reference.payload.identity


def test_every_preprocessing_field_changes_bundle_identity(tmp_path, monkeypatch):
    from sml.data import swag
    from sml.data.swag import SWAG_IDENTITY_FIELDS, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    config = tiny_swag_config(provider)
    monkeypatch.setattr(swag, "_validate_recorded_projections", lambda _manifest: None)
    first = prepare_swag_bundle(config, base, tmp_path / "first")
    try:
        for field in SWAG_IDENTITY_FIELDS:
            changed = prepare_swag_bundle(
                change_identity_field(config, field),
                base,
                tmp_path / field,
            )
            try:
                assert changed.manifest.identity != first.manifest.identity, field
            finally:
                changed.close()
    finally:
        first.close()


def test_overlength_candidate_drops_complete_row(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    rows = (
        {
            "context": "ok",
            "endings": ("a", "b", "c", "d"),
            "label": 2,
        },
        {
            "context": "ok",
            "endings": ("a", "b", "this row is far too long for the bucket", "d"),
            "label": 0,
        },
    )
    processor = FixedProcessor(
        {
            "ok": (10,),
            "a": (11,),
            "b": (12,),
            "c": (13,),
            "d": (14,),
            "this row is far too long for the bucket": tuple(range(10, 50)),
        }
    )
    provider = FakeSwagProvider(rows)
    bundle = prepare_swag_bundle(
        tiny_swag_config(provider),
        tiny_base_model(processor=processor),
        tmp_path / "swag",
    )
    try:
        assert bundle.manifest.dropped_overlength_rows == 1
        assert bundle.manifest.example_count == 1
        with bundle.borrow_buckets() as buckets:
            assert all(
                0 <= int(label) < 4 for bucket in buckets for label in bucket.labels
            )
    finally:
        bundle.close()


def test_zero_usable_examples_are_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    processor = FixedProcessor(
        {
            "ctx": tuple(range(10, 50)),
            "a": (11,),
            "b": (12,),
            "c": (13,),
            "d": (14,),
        }
    )
    provider = FakeSwagProvider(
        ({"context": "ctx", "endings": ("a", "b", "c", "d"), "label": 0},)
    )
    with pytest.raises(SMLDataError, match="usable"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(processor=processor),
            tmp_path / "swag",
        )


def test_not_four_candidates_are_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider(
        (
            {
                "context": "the cat sat",
                "endings": ("on the mat", "in the car"),
                "label": 0,
            },
        )
    )
    with pytest.raises(SMLDataError, match="four"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(),
            tmp_path / "swag",
        )


def test_out_of_range_label_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((replace_row(label=4),))
    with pytest.raises(SMLDataError, match="label"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(),
            tmp_path / "swag",
        )


def test_out_of_range_token_id_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    processor = FixedProcessor(
        {
            "the cat sat": (10, 11, 12),
            "on the mat": (90,),
            "in the car": (13, 14, 15),
            "by the door": (16, 17, 18),
            "near a tree": (19, 20, 21),
        }
    )
    provider = FakeSwagProvider((VALID_ROW,))
    with pytest.raises(SMLDataError, match="token"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(processor=processor),
            tmp_path / "swag",
        )


def test_missing_scored_continuation_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider(
        (
            {
                "context": "the cat sat",
                "endings": ("", "in the car", "by the door", "near a tree"),
                "label": 1,
            },
        )
    )
    with pytest.raises(SMLDataError, match="scored"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(),
            tmp_path / "swag",
        )


def test_maximum_length_beyond_effective_context_is_rejected(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    with pytest.raises(SMLDataError, match="effective context"):
        prepare_swag_bundle(
            tiny_swag_config(provider, maximum_length=64),
            tiny_base_model(),
            tmp_path / "swag",
        )


def test_unavailable_uncached_provider_names_source_and_cache_key(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_resolve = True
    config = tiny_swag_config(provider)
    with pytest.raises(SMLDataError) as caught:
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "swag")
    message = str(caught.value)
    assert config.source.namespace in message
    assert config.source.dataset_config in message
    assert config.source.revision in message
    assert "sha256:" in message


def test_existing_bundle_is_reused_without_resolving_provider(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    output = tmp_path / "swag"
    first = prepare_swag_bundle(tiny_swag_config(provider), base, output)
    second = None
    try:
        provider.fail_resolve = True
        provider.resolve_calls = 0
        provider.iter_calls = 0
        second = prepare_swag_bundle(tiny_swag_config(provider), base, output)
        assert provider.resolve_calls == 0
        assert provider.iter_calls == 0
        assert second.manifest.identity == first.manifest.identity
    finally:
        if second is not None:
            second.close()
        first.close()


def test_existing_bundle_with_different_identity_is_a_collision(tmp_path):
    from sml.data.swag import SwagSourceConfig, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    output = tmp_path / "swag"
    first = prepare_swag_bundle(tiny_swag_config(provider), base, output)
    try:
        with pytest.raises(SMLArtifactError, match="collision"):
            prepare_swag_bundle(
                tiny_swag_config(
                    provider,
                    source=SwagSourceConfig(
                        revision="deadbeef" * 5,
                        namespace="other-org",
                    ),
                ),
                base,
                output,
            )
    finally:
        first.close()


def test_existing_bundle_rejects_tokenizer_identity_mismatch(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    first = prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)
    try:
        other_base = tiny_base_model(
            tokenizer_manifest_overrides={"identity": IDENTITY_C}
        )
        with pytest.raises(SMLArtifactError, match="collision"):
            prepare_swag_bundle(tiny_swag_config(provider), other_base, output)
    finally:
        first.close()


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


def test_runtime_provider_is_not_part_of_bundle_identity(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    first_provider = FakeSwagProvider((VALID_ROW,), fingerprint="fingerprint-v1")
    second_provider = FakeSwagProvider((VALID_ROW,), fingerprint="fingerprint-v1")
    base = tiny_base_model()
    first = prepare_swag_bundle(
        tiny_swag_config(first_provider), base, tmp_path / "first"
    )
    second = prepare_swag_bundle(
        tiny_swag_config(second_provider), base, tmp_path / "second"
    )
    try:
        assert first.manifest.identity == second.manifest.identity
        assert "provider" not in first.manifest.source
        assert first.manifest.source["provider_fingerprint"] == "fingerprint-v1"
        assert first.manifest.source["commit"] == first_provider.commit
        assert first.manifest.source["provider_package"] == first_provider.package
        assert first.manifest.source["provider_version"] == first_provider.version
    finally:
        second.close()
        first.close()


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


def _array_logical_path(manifest: SwagDataManifest, name: str) -> str:
    suffix = f"/{name}.npy"
    for reference in manifest.buckets:
        if reference.payload.logical_path.endswith(suffix):
            return reference.payload.logical_path
    raise AssertionError(f"missing SWAG array {name}")


def _npy_bytes(array: np.ndarray, *, version: tuple[int, int] = (1, 0)) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(payload, array, version=version, allow_pickle=False)
    return payload.getvalue()


def _object_npy_bytes(_array: np.ndarray) -> bytes:
    payload = io.BytesIO()
    np.lib.format.write_array(
        payload,
        np.array(["object"], dtype=object),
        version=(1, 0),
        allow_pickle=True,
    )
    return payload.getvalue()


def _replace_array_bytes(
    path: Path,
    manifest: SwagDataManifest,
    logical_path: str,
    payload_bytes: bytes,
) -> SwagDataManifest:
    array_path = path / logical_path
    array_path.write_bytes(payload_bytes)
    with array_path.open("rb") as payload:
        identity = file_identity(payload)
    references = tuple(
        replace(
            reference,
            payload=replace(
                reference.payload,
                identity=identity,
                byte_size=len(payload_bytes),
            ),
        )
        if reference.payload.logical_path == logical_path
        else reference
        for reference in manifest.buckets
    )
    updated = replace(manifest, buckets=references)
    updated = replace(updated, identity=updated.recompute_identity())
    (path / "manifest.json").write_bytes(canonical_json_bytes(updated))
    return updated


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


def test_swag_bundle_close_releases_views_mappings_payloads_then_root_once(
    tmp_path, monkeypatch
):
    from sml.artifacts.manifest import ArtifactRoot
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    events: list[str] = []
    original_release = swag._OwnedNpyMapping._release_view
    original_mapping_close = swag._OwnedNpyMapping._close_mapping
    original_payload_close = swag._OwnedNpyMapping._close_payload
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

    monkeypatch.setattr(swag._OwnedNpyMapping, "_release_view", release)
    monkeypatch.setattr(swag._OwnedNpyMapping, "_close_mapping", close_mapping)
    monkeypatch.setattr(swag._OwnedNpyMapping, "_close_payload", close_payload)
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


def _one_example_bundle(tmp_path: Path):
    from sml.data.swag import prepare_swag_bundle

    return prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider((VALID_ROW,))),
        tiny_base_model(),
        tmp_path / "swag",
    )


def _one_example_loader() -> SimpleNamespace:
    return SimpleNamespace(prefetch_depth=1, microbatch_size=1, epoch_seed=7)


def test_swag_stream_closes_owned_bundle_on_exhaustion(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    envelopes = list(stream)
    for envelope in envelopes:
        envelope.release()

    assert bundle._closed
    with pytest.raises(StopIteration):
        next(stream)


def test_swag_stream_uses_owned_arrays_without_public_bucket_lease(tmp_path):
    """Internal zero-copy iteration must not enter the public borrow protocol."""
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    assert bundle._bucket_leases == 0
    envelope = next(stream)
    try:
        assert bundle._bucket_leases == 0
    finally:
        envelope.release()
        stream.close()
    assert bundle._bucket_leases == 0
    assert bundle._closed


def test_swag_stream_rejects_active_public_lease_without_changing_ownership(
    tmp_path, monkeypatch
):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    lease = bundle.borrow_buckets()
    first = int(lease.buckets[0].input_ids[0, 0, 0])
    transfer_rejected = False
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr("threading.Thread.start", lambda _thread: None)
            with pytest.raises(SMLDataError, match="active bucket leases"):
                SwagBatchStream(
                    bundle,
                    _one_example_loader(),
                    cursor=SwagCursor.initial(),
                )
        assert int(lease.buckets[0].input_ids[0, 0, 0]) == first
        assert bundle._closed is False
        transfer_rejected = True
    finally:
        lease.close()
        if not transfer_rejected:
            bundle.close()

    stream = SwagBatchStream(
        bundle,
        _one_example_loader(),
        cursor=SwagCursor.initial(),
    )
    stream.close()
    assert bundle._closed is True


def test_swag_stream_closes_owned_bundle_on_early_return_and_double_close(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    with stream:
        envelope = next(stream)
        envelope.release()
    stream.close()

    assert bundle._closed


def test_swag_stream_construction_failure_closes_owned_bundle(tmp_path, monkeypatch):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)

    def fail_start(_thread):
        raise RuntimeError("thread start failed")

    monkeypatch.setattr("threading.Thread.start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())

    assert bundle._closed


def test_swag_stream_thread_constructor_failure_closes_owned_bundle(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)

    def fail_thread_construction(*_args, **_kwargs):
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr(swag.threading, "Thread", fail_thread_construction)
    try:
        with pytest.raises(RuntimeError, match="thread construction failed"):
            SwagBatchStream(
                bundle,
                _one_example_loader(),
                cursor=SwagCursor.initial(),
            )
        assert bundle._closed
    finally:
        bundle.close()


def test_swag_stream_past_end_cursor_closes_on_later_epoch_exhaustion(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(
        bundle,
        _one_example_loader(),
        cursor=SwagCursor(epoch=0, bucket_order_position=99, row_offset=0),
    )
    try:
        with pytest.raises(StopIteration):
            next(stream)
        assert stream._closed
        assert bundle._closed
    finally:
        stream.close()


def test_swag_stream_rejects_an_already_closed_bundle(tmp_path):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    bundle.close()

    with pytest.raises(SMLDataError, match="closed"):
        SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())


def test_swag_stream_public_constructor_does_not_expose_borrowed_ownership():
    from sml.data.swag import SwagBatchStream

    assert tuple(inspect.signature(SwagBatchStream).parameters) == (
        "bundle",
        "loader",
        "cursor",
    )


def test_swag_stream_pull_preserves_producer_failure_when_close_fails(
    tmp_path, monkeypatch
):
    from sml.data import swag
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    producer_error = SMLDataError("injected producer failure")
    cleanup_error = RuntimeError("injected stream cleanup failure")
    original_close = stream.close
    stream._stop.set()
    producer = stream._producer
    assert producer is not None
    producer.join()
    while not stream._queue.empty():
        item = stream._queue.get_nowait()
        if isinstance(item, swag.SwagBatchEnvelope):
            item.release()
    stream._queue.put(swag._ProducerFailure(producer_error))

    def fail_after_close():
        original_close()
        raise cleanup_error

    monkeypatch.setattr(stream, "close", fail_after_close)
    try:
        with pytest.raises(SMLDataError) as raised:
            stream._pull()

        assert raised.value is producer_error
        assert raised.value.__cause__ is cleanup_error
    finally:
        original_close()


def test_swag_stream_exit_preserves_body_error_when_close_fails(tmp_path, monkeypatch):
    from sml.data.swag import SwagBatchStream, SwagCursor

    bundle = _one_example_bundle(tmp_path)
    stream = SwagBatchStream(bundle, _one_example_loader(), cursor=SwagCursor.initial())
    body_error = ValueError("injected stream body failure")
    cleanup_error = RuntimeError("injected stream cleanup failure")
    original_close = stream.close

    def fail_after_close():
        original_close()
        raise cleanup_error

    monkeypatch.setattr(stream, "close", fail_after_close)
    try:
        with pytest.raises(ValueError) as raised, stream:
            raise body_error

        assert raised.value is body_error
        assert raised.value.__cause__ is cleanup_error
    finally:
        original_close()


def _instrument_real_swag_cleanup(monkeypatch, *, fail_open_index: int | None = None):
    from sml.artifacts.manifest import ArtifactRoot
    from sml.data import swag

    events: list[tuple[str, str | None]] = []
    owners: list[object] = []
    roots: list[ArtifactRoot] = []
    array_refs: dict[int, weakref.ReferenceType[np.ndarray]] = {}
    original_open = swag._OwnedNpyMapping.open.__func__
    original_release = swag._OwnedNpyMapping._release_view
    original_mapping_close = swag._OwnedNpyMapping._close_mapping
    original_payload_close = swag._OwnedNpyMapping._close_payload
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
    monkeypatch.setattr(swag._OwnedNpyMapping, "_release_view", release)
    monkeypatch.setattr(swag._OwnedNpyMapping, "_close_mapping", close_mapping)
    monkeypatch.setattr(swag._OwnedNpyMapping, "_close_payload", close_payload)
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
    instrumented_payload_close = swag._OwnedNpyMapping._close_payload

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
        "_close_payload",
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
    instrumented_payload_close = swag._OwnedNpyMapping._close_payload
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
        "_close_payload",
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

    monkeypatch.setattr(swag._OwnedNpyMapping, "_release_view", fail_release)
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

    monkeypatch.setattr(swag.mmap, "mmap", lambda *_args, **_kwargs: FakeMapping())
    monkeypatch.setattr(swag.np, "ndarray", lambda *_args, **_kwargs: FakeArray())
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
    original_mmap = swag.mmap.mmap
    original_ndarray = swag.np.ndarray
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

    monkeypatch.setattr(swag.mmap, "mmap", ObservedMmap)
    monkeypatch.setattr(swag.np, "ndarray", construct_array)
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


def test_huggingface_provider_pins_main_revision_to_commit_sha(monkeypatch):
    from sml.data import swag
    from sml.data.swag import HuggingFaceDatasetsSwagProvider, SwagSourceConfig

    pinned = "abcdef0123456789abcdef0123456789abcdef01"
    load_revisions: list[object] = []

    class FakeApi:
        def dataset_info(self, repo_id, revision=None, timeout=None, **kwargs):
            assert repo_id == "allenai/swag"
            assert revision == "main"
            return SimpleNamespace(sha=pinned)

    class FakeDataset:
        _fingerprint = "fp-1"

        def __iter__(self):
            return iter(())

    def fake_load_dataset(path, name, split=None, revision=None, **kwargs):
        del path, name, split, kwargs
        load_revisions.append(revision)
        return FakeDataset()

    monkeypatch.setattr(
        swag,
        "_import_huggingface_hub",
        lambda: SimpleNamespace(HfApi=FakeApi),
        raising=False,
    )
    monkeypatch.setattr(
        swag,
        "_import_datasets",
        lambda: SimpleNamespace(load_dataset=fake_load_dataset),
    )

    provider = HuggingFaceDatasetsSwagProvider()
    source = SwagSourceConfig(revision="main")
    resolved = provider.resolve(source)
    assert resolved.commit == pinned
    assert resolved.commit != source.revision
    assert resolved.revision == "main"
    list(provider.iter_rows(resolved))
    assert load_revisions
    assert all(revision == pinned for revision in load_revisions)


def test_resolved_source_mismatch_is_rejected_before_rows(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.resolved_namespace = "normalized-org"
    with pytest.raises(SMLDataError, match="correspond"):
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(),
            tmp_path / "swag",
        )
    assert provider.iter_calls == 0


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


@pytest.mark.parametrize(
    ("field", "message"),
    (("join_policy", "join"), ("preprocessing_schema_version", "schema")),
)
def test_prepare_swag_bundle_validates_newly_published_projections(
    tmp_path, field, message
):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    with pytest.raises(SMLArtifactError, match=message):
        prepare_swag_bundle(
            change_identity_field(tiny_swag_config(provider), field),
            tiny_base_model(),
            tmp_path / field,
        )


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


def test_prepare_rejects_non_full_base_before_provider_access(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_resolve = True
    base = replace(tiny_base_model(), verification=VerificationLevel.MANIFEST_TRUSTED)
    with pytest.raises(SMLArtifactError, match="FULL"):
        prepare_swag_bundle(tiny_swag_config(provider), base, tmp_path / "swag")
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0


def test_prepare_rejects_tokenizer_model_mismatch_before_provider_access(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_resolve = True
    base = tiny_base_model(model_overrides={"vocab_size": 32})
    with pytest.raises(SMLDataError, match="vocab|tokenizer|special"):
        prepare_swag_bundle(tiny_swag_config(provider), base, tmp_path / "swag")
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0


def test_iter_rows_start_failure_names_source_and_cache_key(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_iter = True
    config = tiny_swag_config(provider)
    with pytest.raises(SMLDataError) as caught:
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "swag")
    message = str(caught.value)
    assert config.source.namespace in message
    assert config.source.dataset_config in message
    assert config.source.revision in message
    assert "sha256:" in message


def test_iter_rows_consumption_failure_names_source_and_cache_key(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_iter_after_rows = True
    config = tiny_swag_config(provider)
    with pytest.raises(SMLDataError) as caught:
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "swag")
    message = str(caught.value)
    assert config.source.namespace in message
    assert config.source.dataset_config in message
    assert config.source.revision in message
    assert "sha256:" in message
    assert "four" not in message
    assert "usable" not in message


def test_iter_rows_start_data_error_is_wrapped_with_source_and_cache_key(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_iter_data_error = True
    config = tiny_swag_config(provider)
    with pytest.raises(SMLDataError) as caught:
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "swag")
    message = str(caught.value)
    assert config.source.namespace in message
    assert config.source.dataset_config in message
    assert config.source.revision in message
    assert "sha256:" in message
    assert caught.value.__cause__ is not None


def test_iter_rows_consumption_data_error_is_wrapped_with_source_and_cache_key(
    tmp_path,
):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    provider.fail_iter_after_rows_data_error = True
    config = tiny_swag_config(provider)
    with pytest.raises(SMLDataError) as caught:
        prepare_swag_bundle(config, tiny_base_model(), tmp_path / "swag")
    message = str(caught.value)
    assert config.source.namespace in message
    assert config.source.dataset_config in message
    assert config.source.revision in message
    assert "sha256:" in message
    assert "four" not in message
    assert caught.value.__cause__ is not None


def test_parse_errors_are_not_wrapped_as_provider_failures(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((replace_row(endings=("on the mat", "in the car")),))
    with pytest.raises(SMLDataError, match="four") as caught:
        prepare_swag_bundle(
            tiny_swag_config(provider),
            tiny_base_model(),
            tmp_path / "swag",
        )
    message = str(caught.value)
    assert "unavailable" not in message
    assert "cache key" not in message
    assert "sha256:" not in message


def test_different_maximum_examples_at_existing_path_is_a_collision(
    tmp_path, monkeypatch
):
    from sml.data.swag import SwagDataBundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW, replace_row(label=0)))
    base = tiny_base_model()
    output = tmp_path / "swag"
    first = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=2),
        base,
        output,
    )
    assert first.manifest.preprocessing["maximum_examples"] == 2
    closed: list[SwagDataBundle] = []
    original_close = SwagDataBundle.close

    def record_close(bundle):
        closed.append(bundle)
        original_close(bundle)

    monkeypatch.setattr(SwagDataBundle, "close", record_close)
    try:
        with pytest.raises(SMLArtifactError, match="collision"):
            prepare_swag_bundle(
                tiny_swag_config(provider, maximum_examples=1),
                base,
                output,
            )
        assert len(closed) == 1
        assert closed[0] is not first
        assert closed[0]._closed
    finally:
        first.close()


def test_none_versus_integer_maximum_examples_is_a_collision(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    output = tmp_path / "swag"
    uncapped = prepare_swag_bundle(tiny_swag_config(provider), base, output)
    try:
        assert uncapped.manifest.preprocessing["maximum_examples"] is None
        with pytest.raises(SMLArtifactError, match="collision"):
            prepare_swag_bundle(
                tiny_swag_config(provider, maximum_examples=1),
                base,
                output,
            )
    finally:
        uncapped.close()


def test_matching_maximum_examples_reuses_existing_bundle(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW, replace_row(label=2)))
    base = tiny_base_model()
    output = tmp_path / "swag"
    first = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=1),
        base,
        output,
    )
    provider.fail_resolve = True
    provider.resolve_calls = 0
    provider.iter_calls = 0
    second = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=1),
        base,
        output,
    )
    try:
        assert provider.resolve_calls == 0
        assert provider.iter_calls == 0
        assert second.manifest.identity == first.manifest.identity
    finally:
        second.close()
        first.close()


def test_maximum_examples_changes_bundle_identity(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    uncapped = prepare_swag_bundle(
        tiny_swag_config(provider),
        base,
        tmp_path / "uncapped",
    )
    capped = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=1),
        base,
        tmp_path / "capped",
    )
    try:
        assert capped.manifest.identity != uncapped.manifest.identity
    finally:
        capped.close()
        uncapped.close()


def test_load_rejects_token_id_outside_vocabulary(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "input_ids")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, 0] = manifest.vocab_size
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="vocab"):
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


def test_load_rejects_score_mask_true_where_valid_mask_is_false(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "score_mask")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, -1] = True
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="score"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_padding_inconsistent_with_valid_mask(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "input_ids")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, -1] = 10
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="padding|valid"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_hole_in_valid_mask(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "valid_token_mask")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, 1] = False
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="padding|valid"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_candidate_without_scored_continuation(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "score_mask")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, :] = False
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="scored"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_unscored_eos_when_it_is_the_last_valid_token(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    valid = np.load(output / _array_logical_path(manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    assert last >= 0
    input_ids = np.load(output / _array_logical_path(manifest, "input_ids"))
    assert int(input_ids[0, 0, last]) == manifest.eos_token_id
    logical = _array_logical_path(manifest, "score_mask")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, last] = False
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="eos|EOS"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_last_valid_token_that_is_not_eos(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    valid = np.load(output / _array_logical_path(manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    logical = _array_logical_path(manifest, "input_ids")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, last] = 10
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="eos|EOS"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_score_mask_hole_in_valid_prefix(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    valid = np.load(output / _array_logical_path(manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    logical = _array_logical_path(manifest, "score_mask")
    score = np.load(output / logical)
    hole = last - 1
    assert hole >= 1
    assert bool(score[0, 0, hole])
    assert bool(score[0, 0, last])

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, hole] = False
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="score|suffix|contiguous"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_candidate_that_does_not_start_with_bos(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    manifest = prepared.manifest
    prepared.close()
    logical = _array_logical_path(manifest, "input_ids")
    input_ids = np.load(output / logical)
    assert int(input_ids[0, 0, 0]) == manifest.bos_token_id

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, 0] = 10
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="BOS|bos"):
        load_swag_bundle(output, VerificationLevel.FULL)


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


def test_chunked_ingest_does_not_stack_all_examples_at_once(tmp_path, monkeypatch):
    from sml.data import swag
    from sml.data.swag import prepare_swag_bundle

    monkeypatch.setattr(swag, "_INGEST_CHUNK_SIZE", 2)
    stacked_example_batches: list[int] = []
    original_stack = np.stack

    def tracking_stack(arrays, axis=0, **kwargs):
        sequence = tuple(arrays)
        first = sequence[0] if sequence else None
        if getattr(first, "ndim", 0) >= 2:
            stacked_example_batches.append(len(sequence))
        return original_stack(sequence, axis=axis, **kwargs)

    monkeypatch.setattr(swag.np, "stack", tracking_stack)
    rows = tuple(replace_row(label=index % 4) for index in range(5))
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(),
        tmp_path / "swag",
    )
    try:
        with bundle.borrow_buckets() as buckets:
            labels = [
                int(label) for bucket in buckets for label in np.asarray(bucket.labels)
            ]
        assert bundle.manifest.example_count == 5
        assert labels == [index % 4 for index in range(5)]
        assert all(count <= 2 for count in stacked_example_batches)
    finally:
        bundle.close()


def test_swag_data_module_does_not_import_training():
    from sml.data import swag as data_swag

    source = inspect.getsource(data_swag)
    assert "sml.training" not in source
    assert "LoaderConfig" not in source
