from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import VerificationLevel
from sml.errors import SMLArtifactError
from sml.evaluation_result import (
    EvaluationProviderVersion,
    EvaluationResult,
    EvaluationSourceIdentity,
    EvaluationTaskRecord,
    evaluation_result_identity,
    evaluation_task_identity,
    normalize_json_value,
)
from sml.inference import ModelIdentity


def model_identity() -> ModelIdentity:
    return ModelIdentity(
        artifact_kind="export",
        run_identity=None,
        step=7,
        checkpoint_identity=None,
        run_step_identity=None,
        tokenizer_identity="sha256:" + "1" * 64,
        verification=VerificationLevel.FULL,
    )


def make_task_record(
    *,
    metric_payload: object = {"acc,none": 0.5},
) -> EvaluationTaskRecord:
    return EvaluationTaskRecord(
        task_name="hellaswag",
        task_identity="sha256:" + "0" * 64,
        task_yaml=EvaluationSourceIdentity(
            logical_name="tasks/hellaswag.yaml",
            content_identity="sha256:" + "2" * 64,
        ),
        include_template_closure=(
            EvaluationSourceIdentity(
                logical_name="tasks/common.yaml",
                content_identity="sha256:" + "3" * 64,
            ),
        ),
        task_metadata_version="1.0",
        prompt_config={"description": "multiple choice"},
        few_shot_config={"num_fewshot": 5},
        generation_config={"temperature": 0.0},
        metric_normalization_config={"acc,none": "mean"},
        seeds={"fewshot": 1234},
        limit=10,
        ordered_request_identity="sha256:" + "4" * 64,
        lm_eval_package_version="0.4.12",
        lm_eval_source_commit=None,
        dataset_revision="revision-1",
        dataset_fingerprint="fingerprint-1",
        provider_versions=(
            EvaluationProviderVersion(name="datasets", version="3.0.0"),
            EvaluationProviderVersion(name="lm-eval", version="0.4.12"),
        ),
        metric_payload=metric_payload,
    )


def make_result(
    *, model: ModelIdentity, tasks: tuple[EvaluationTaskRecord, ...]
) -> EvaluationResult:
    return EvaluationResult(
        kind="evaluation-result",
        version=1,
        identity="sha256:" + "0" * 64,
        model=model,
        tasks=tasks,
        provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
    )


def test_json_normalization_is_closed_finite_and_deeply_immutable() -> None:
    normalized = normalize_json_value(
        {"z": np.int64(3), "a": [True, np.float32(1.25), None]},
        context="provider result",
    )
    assert tuple(normalized) == ("a", "z")
    assert normalized["a"] == (True, 1.25, None)
    with pytest.raises(TypeError):
        normalized["z"] = 4  # type: ignore[index]
    for invalid in (float("nan"), float("inf"), b"bytes", Path("local"), {1: "x"}):
        with pytest.raises((TypeError, ValueError)):
            normalize_json_value(invalid, context="provider result")


def test_source_identity_requires_a_portable_logical_path() -> None:
    for logical_name in ("/private/tmp/task.yaml", "C:\\task.yaml", "../task.yaml"):
        with pytest.raises(SMLArtifactError):
            EvaluationSourceIdentity(
                logical_name=logical_name,
                content_identity="sha256:" + "2" * 64,
            )


def test_task_and_result_identities_cover_metrics_requests_and_model() -> None:
    task = make_task_record(metric_payload={"acc,none": 0.5})
    task = replace(task, task_identity=evaluation_task_identity(task))
    result = make_result(model=model_identity(), tasks=(task,))
    result = replace(result, identity=evaluation_result_identity(result))
    changed_metric = replace(task, metric_payload={"acc,none": 0.75})
    changed_request = replace(task, ordered_request_identity="sha256:" + "2" * 64)
    assert (
        evaluation_result_identity(
            replace(result, identity="sha256:" + "0" * 64, tasks=(changed_metric,))
        )
        != result.identity
    )
    assert evaluation_task_identity(changed_request) != task.task_identity
    assert (
        evaluation_result_identity(
            replace(
                result,
                identity="sha256:" + "0" * 64,
                model=replace(model_identity(), step=8),
            )
        )
        != result.identity
    )


def test_records_reject_unordered_or_invalid_values() -> None:
    with pytest.raises(ValueError):
        EvaluationProviderVersion(name="lm-eval", version="")
    with pytest.raises(ValueError):
        replace(make_task_record(), limit=0)
    with pytest.raises(ValueError):
        replace(
            make_task_record(),
            provider_versions=(
                EvaluationProviderVersion(name="lm-eval", version="0.4.12"),
                EvaluationProviderVersion(name="datasets", version="3.0.0"),
            ),
        )
    with pytest.raises(ValueError):
        make_result(model=model_identity(), tasks=())
    with pytest.raises(ValueError):
        replace(
            make_result(model=model_identity(), tasks=(make_task_record(),)),
            version=1.0,
        )
