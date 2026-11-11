from __future__ import annotations

from pathlib import Path

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    evaluate,
)
from sml_test_support.evaluation import (
    tiny_evaluation_config,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [(1.0, "1.0"), (2, "2")],
)
def test_task_metadata_version_canonicalizes_finite_provider_scalars(
    version: float,
    expected: str,
) -> None:
    from sml import evaluation

    assert evaluation._metadata_version({"metadata": {"version": version}}) == expected


@pytest.mark.parametrize("version", [True, float("nan"), float("inf")])
def test_task_metadata_version_rejects_nonfinite_or_boolean_provider_values(
    version: object,
) -> None:
    from sml import evaluation

    with pytest.raises(SMLRuntimeError, match="metadata version"):
        evaluation._metadata_version({"metadata": {"version": version}})


def test_provider_versions_require_all_five_evaluation_packages() -> None:
    """Catches provider provenance that silently drops datasets or another required package."""
    from importlib import metadata

    from sml import evaluation

    assert evaluation._provider_versions() == tuple(
        sorted(
            (name, metadata.version(name))
            for name in ("datasets", "lm-eval", "mlx", "numpy", "sentencepiece")
        )
    )


def test_lm_eval_source_commit_uses_public_pep_610_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches private distribution-path access instead of the public metadata API."""
    from importlib import metadata

    from sml import evaluation

    class PublicDistribution:
        def __init__(self, direct_url: str | None) -> None:
            self._direct_url = direct_url

        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return self._direct_url

    distribution = PublicDistribution(None)
    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: distribution,
    )

    assert evaluation._lm_eval_source_commit() is None
    distribution._direct_url = (
        '{"url":"https://example.invalid/lm-eval.whl","archive_info":{}}'
    )
    assert evaluation._lm_eval_source_commit() is None
    distribution._direct_url = (
        '{"url":"file:///tmp/lm-eval","dir_info":{"editable":true}}'
    )
    assert evaluation._lm_eval_source_commit() is None
    distribution._direct_url = (
        '{"url":"https://github.com/EleutherAI/lm-evaluation-harness",'
        '"vcs_info":{"vcs":"git","commit_id":"0123456789abcdef"}}'
    )
    assert evaluation._lm_eval_source_commit() == "0123456789abcdef"


@pytest.mark.parametrize(
    "payload",
    (
        "not-json",
        "[]",
        '{"vcs_info":{"vcs":"git","commit_id":"abc"}}',
        '{"url":"","vcs_info":{"vcs":"git","commit_id":"abc"}}',
        '{"url":3,"vcs_info":{"vcs":"git","commit_id":"abc"}}',
        '{"url":"https://example.invalid","vcs_info":"not-a-mapping"}',
        '{"url":"https://example.invalid","vcs_info":{"commit_id":"abc"}}',
        '{"url":"https://example.invalid","vcs_info":{"vcs":"","commit_id":"abc"}}',
        '{"url":"https://example.invalid","vcs_info":{"vcs":"git"}}',
        '{"url":"https://example.invalid","vcs_info":{"vcs":"git","commit_id":""}}',
        '{"url":"https://example.invalid"}',
        '{"url":"https://example.invalid","archive_info":"not-a-mapping"}',
        '{"url":"https://example.invalid","dir_info":[]}',
        '{"url":"https://example.invalid","archive_info":{},"dir_info":{}}',
        '{"url":"https://example.invalid","vcs_info":{"vcs":"git","commit_id":"abc"},"archive_info":{}}',
    ),
)
def test_lm_eval_source_commit_fails_closed_for_malformed_pep_610_metadata(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    """Catches acceptance of malformed direct_url metadata as trustworthy provenance."""
    from importlib import metadata

    from sml import evaluation

    class PublicDistribution:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return payload

    monkeypatch.setattr(
        metadata,
        "distribution",
        lambda name: PublicDistribution(),
    )

    with pytest.raises(SMLRuntimeError, match="direct_url|PEP 610|source commit"):
        evaluation._lm_eval_source_commit()


def test_lm_eval_source_commit_normalizes_undecodable_public_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches leaked UnicodeDecodeError from Distribution.read_text()."""
    from importlib import metadata

    from sml import evaluation

    class PublicDistribution:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(metadata, "distribution", lambda name: PublicDistribution())

    with pytest.raises(SMLRuntimeError, match="direct_url"):
        evaluation._lm_eval_source_commit()


@pytest.mark.parametrize(
    ("package", "failure"),
    tuple(
        (package, failure)
        for package in ("datasets", "lm-eval", "mlx", "numpy", "sentencepiece")
        for failure in ("missing", "empty")
    ),
)
def test_evaluation_fails_before_publication_for_each_invalid_provider_version(
    package: str,
    failure: str,
    tiny_pretraining_run: Path,
    fake_lm_eval,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catches publication after any required provider version is missing or empty."""
    from importlib import metadata

    original_version = metadata.version

    def invalid_version(name: str) -> str:
        if name == package:
            if failure == "missing":
                raise metadata.PackageNotFoundError(name)
            return ""
        return original_version(name)

    monkeypatch.setattr(metadata, "version", invalid_version)
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="required evaluation provider package"):
        evaluate(config)

    assert not config.output.exists()
