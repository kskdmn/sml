from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sml.artifacts.manifest import (
    ArrayPayloadRef,
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

    assert recording_tokenizer.calls[0] == VALID_ROW["context"]
    assert tuple(recording_tokenizer.calls[1:5]) == tuple(VALID_ROW["endings"])
    bucket = bundle.buckets[0]
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
    prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(processor=processor),
        tmp_path / "swag",
    )
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


def test_every_preprocessing_field_changes_bundle_identity(tmp_path):
    from sml.data.swag import SWAG_IDENTITY_FIELDS, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    config = tiny_swag_config(provider)
    first = prepare_swag_bundle(config, base, tmp_path / "first")
    for field in SWAG_IDENTITY_FIELDS:
        changed = prepare_swag_bundle(
            change_identity_field(config, field),
            base,
            tmp_path / field,
        )
        assert changed.manifest.identity != first.manifest.identity, field


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
    assert bundle.manifest.dropped_overlength_rows == 1
    assert bundle.manifest.example_count == 1
    assert all(
        0 <= int(label) < 4 for bucket in bundle.buckets for label in bucket.labels
    )


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
    provider.fail_resolve = True
    provider.resolve_calls = 0
    provider.iter_calls = 0
    second = prepare_swag_bundle(tiny_swag_config(provider), base, output)
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0
    assert second.manifest.identity == first.manifest.identity


def test_existing_bundle_with_different_identity_is_a_collision(tmp_path):
    from sml.data.swag import SwagSourceConfig, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    output = tmp_path / "swag"
    prepare_swag_bundle(tiny_swag_config(provider), base, output)
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


def test_existing_bundle_rejects_tokenizer_identity_mismatch(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output)
    other_base = tiny_base_model(tokenizer_manifest_overrides={"identity": IDENTITY_C})
    with pytest.raises(SMLArtifactError, match="collision"):
        prepare_swag_bundle(tiny_swag_config(provider), other_base, output)


def test_load_swag_bundle_reopens_without_provider(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    loaded = load_swag_bundle(output, VerificationLevel.FULL)
    assert loaded.manifest.identity == prepared.manifest.identity
    assert loaded.buckets[0].input_ids.shape == prepared.buckets[0].input_ids.shape


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
    assert first.manifest.identity == second.manifest.identity
    assert "provider" not in first.manifest.source
    assert first.manifest.source["provider_fingerprint"] == "fingerprint-v1"
    assert first.manifest.source["commit"] == first_provider.commit
    assert first.manifest.source["provider_package"] == first_provider.package
    assert first.manifest.source["provider_version"] == first_provider.version


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
    prepare_swag_bundle(
        change_identity_field(tiny_swag_config(provider), "join_policy"),
        tiny_base_model(),
        output,
    )

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
    prepare_swag_bundle(
        change_identity_field(
            tiny_swag_config(provider), "preprocessing_schema_version"
        ),
        tiny_base_model(),
        output,
    )

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

    def fail_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("arrays opened before projection validation")

    monkeypatch.setattr(swag, "_open_buckets", fail_open)
    _rewrite_swag_manifest(output, pad_token_id=prepared.manifest.bos_token_id)
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


def test_different_maximum_examples_at_existing_path_is_a_collision(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW, replace_row(label=0)))
    base = tiny_base_model()
    output = tmp_path / "swag"
    first = prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=2),
        base,
        output,
    )
    assert first.manifest.preprocessing["maximum_examples"] == 2
    with pytest.raises(SMLArtifactError, match="collision"):
        prepare_swag_bundle(
            tiny_swag_config(provider, maximum_examples=1),
            base,
            output,
        )


def test_none_versus_integer_maximum_examples_is_a_collision(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    output = tmp_path / "swag"
    uncapped = prepare_swag_bundle(tiny_swag_config(provider), base, output)
    assert uncapped.manifest.preprocessing["maximum_examples"] is None
    with pytest.raises(SMLArtifactError, match="collision"):
        prepare_swag_bundle(
            tiny_swag_config(provider, maximum_examples=1),
            base,
            output,
        )


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
    assert provider.resolve_calls == 0
    assert provider.iter_calls == 0
    assert second.manifest.identity == first.manifest.identity


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
    assert capped.manifest.identity != uncapped.manifest.identity


def test_load_rejects_token_id_outside_vocabulary(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    logical = _array_logical_path(prepared.manifest, "input_ids")

    def mutate(array: np.ndarray) -> np.ndarray:
        mutated = np.array(array, copy=True)
        mutated[0, 0, 0] = prepared.manifest.vocab_size
        return mutated

    _rewrite_swag_array(output, logical, mutate)
    with pytest.raises(SMLArtifactError, match="vocab"):
        load_swag_bundle(output, VerificationLevel.FULL)


def test_load_rejects_score_mask_true_where_valid_mask_is_false(tmp_path):
    from sml.data.swag import load_swag_bundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    prepared = prepare_swag_bundle(
        tiny_swag_config(provider), tiny_base_model(), output
    )
    logical = _array_logical_path(prepared.manifest, "score_mask")

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
    logical = _array_logical_path(prepared.manifest, "input_ids")

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
    logical = _array_logical_path(prepared.manifest, "valid_token_mask")

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
    logical = _array_logical_path(prepared.manifest, "score_mask")

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
    valid = np.load(output / _array_logical_path(prepared.manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    assert last >= 0
    input_ids = np.load(output / _array_logical_path(prepared.manifest, "input_ids"))
    assert int(input_ids[0, 0, last]) == prepared.manifest.eos_token_id
    logical = _array_logical_path(prepared.manifest, "score_mask")

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
    valid = np.load(output / _array_logical_path(prepared.manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    logical = _array_logical_path(prepared.manifest, "input_ids")

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
    valid = np.load(output / _array_logical_path(prepared.manifest, "valid_token_mask"))
    last = int(np.asarray(valid[0, 0]).sum()) - 1
    logical = _array_logical_path(prepared.manifest, "score_mask")
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
    logical = _array_logical_path(prepared.manifest, "input_ids")
    input_ids = np.load(output / logical)
    assert int(input_ids[0, 0, 0]) == prepared.manifest.bos_token_id

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
    bucket_length = prepared.buckets[0].length
    preprocessing = dict(prepared.manifest.preprocessing)
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
    valid = np.load(output / _array_logical_path(prepared.manifest, "valid_token_mask"))
    valid_length = int(np.asarray(valid[0, 0]).sum())
    assert valid_length > 4
    preprocessing = dict(prepared.manifest.preprocessing)
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
    labels = [
        int(label) for bucket in bundle.buckets for label in np.asarray(bucket.labels)
    ]
    assert bundle.manifest.example_count == 5
    assert labels == [index % 4 for index in range(5)]
    assert all(count <= 2 for count in stacked_example_batches)
