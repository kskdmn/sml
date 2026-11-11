from __future__ import annotations

from types import SimpleNamespace

import pytest
from sml.errors import SMLDataError

from .swag_helpers import (
    VALID_ROW,
    FakeSwagProvider,
    tiny_base_model,
    tiny_swag_config,
)


@pytest.mark.parametrize(
    "failure",
    [
        "fail_resolve",
        "fail_iter",
        "fail_iter_after_rows",
        "fail_iter_data_error",
        "fail_iter_after_rows_data_error",
    ],
)
def test_provider_failures_name_source_and_cache_key(tmp_path, failure):
    from sml.data.swag import prepare_swag_bundle

    provider = FakeSwagProvider((VALID_ROW,))
    setattr(provider, failure, True)
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
    assert caught.value.__cause__ is not None


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
