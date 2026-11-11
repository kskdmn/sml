from __future__ import annotations

from dataclasses import replace

import pytest
from sml.errors import SMLArtifactError

from .swag_helpers import (
    IDENTITY_C,
    VALID_ROW,
    FakeSwagProvider,
    replace_row,
    tiny_base_model,
    tiny_swag_config,
)


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
        "maximum_examples": 1,
        "bucket_boundaries": (32,),
    }
    if field not in replacements:
        raise AssertionError(f"unknown identity field: {field}")
    value = replacements[field]
    if field in {"namespace", "name", "dataset_config", "revision", "split"}:
        return replace(config, source=replace(config.source, **{field: value}))
    return replace(config, **{field: value})


def test_every_configurable_preprocessing_field_changes_bundle_identity(tmp_path):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    base = tiny_base_model()
    config = tiny_swag_config(provider)
    first = prepare_swag_bundle(config, base, tmp_path / "first")
    try:
        for field in (
            "namespace",
            "name",
            "dataset_config",
            "revision",
            "split",
            "maximum_length",
            "maximum_examples",
            "bucket_boundaries",
        ):
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


@pytest.mark.parametrize("maximum_examples", [None, 1], ids=["uncapped", "capped"])
def test_existing_bundle_is_reused_without_resolving_provider(
    tmp_path, maximum_examples
):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW, replace_row(label=2)))
    base = tiny_base_model()
    output = tmp_path / "swag"
    config = tiny_swag_config(provider, maximum_examples=maximum_examples)
    with prepare_swag_bundle(config, base, output) as first:
        provider.fail_resolve = True
        provider.resolve_calls = provider.iter_calls = 0
        with prepare_swag_bundle(config, base, output) as second:
            assert provider.resolve_calls == provider.iter_calls == 0
            assert second.manifest.identity == first.manifest.identity


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


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("join_policy", "join"),
        ("overlength_policy", "overlength"),
        ("bos_policy", "bos"),
        ("eos_policy", "eos"),
        ("preprocessing_schema_version", "schema"),
    ),
)
def test_unsupported_preprocessing_is_rejected_before_publication(
    tmp_path, field, message
):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    output = tmp_path / "swag"
    with pytest.raises(ValueError, match=message):
        prepare_swag_bundle(
            change_identity_field(tiny_swag_config(provider), field),
            tiny_base_model(),
            output,
        )
    assert provider.resolve_calls == provider.iter_calls == 0
    assert not output.exists()
    with prepare_swag_bundle(tiny_swag_config(provider), tiny_base_model(), output):
        assert output.is_dir()


@pytest.mark.parametrize("saved_limit", [None, 2], ids=["uncapped", "different-cap"])
def test_changed_maximum_examples_collides_and_closes_reopened_bundle(
    tmp_path, monkeypatch, saved_limit
):
    from sml.data.swag import SwagDataBundle, prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW, replace_row(label=0)))
    base = tiny_base_model()
    output = tmp_path / "swag"
    with prepare_swag_bundle(
        tiny_swag_config(provider, maximum_examples=saved_limit), base, output
    ) as first:
        assert first.manifest.preprocessing["maximum_examples"] == saved_limit
        closed = []
        original_close = SwagDataBundle.close

        def record_close(bundle):
            closed.append(bundle)
            original_close(bundle)

        monkeypatch.setattr(SwagDataBundle, "close", record_close)
        with pytest.raises(SMLArtifactError, match="collision"):
            prepare_swag_bundle(
                tiny_swag_config(provider, maximum_examples=1), base, output
            )
        assert len(closed) == 1
        assert closed[0] is not first
        assert closed[0]._closed
