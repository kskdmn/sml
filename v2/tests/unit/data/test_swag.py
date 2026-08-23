from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    PayloadRef,
    TokenizerManifest,
    VerificationLevel,
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

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        words = text.split()
        if not words:
            return []
        return [10 + (index % 20) for index, _ in enumerate(words)]


class FixedProcessor:
    def __init__(self, encoded: Mapping[str, tuple[int, ...]]) -> None:
        self._encoded = dict(encoded)
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
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

    def resolve(self, source):
        self.resolve_calls += 1
        if self.fail_resolve:
            raise RuntimeError("provider unavailable")
        from sml.data.swag import ResolvedSwagSource

        return ResolvedSwagSource(
            backend=source.backend,
            namespace=source.namespace,
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
        yield from self.rows


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
