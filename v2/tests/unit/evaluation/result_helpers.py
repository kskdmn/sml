from __future__ import annotations

from dataclasses import replace

from sml.artifacts.manifest import (
    VerificationLevel,
)
from sml.evaluation_result import (
    EvaluationProviderVersion,
    EvaluationResult,
    EvaluationSourceIdentity,
    EvaluationTaskRecord,
    evaluation_result_identity,
    evaluation_task_identity,
)
from sml.inference import ModelIdentity


def model_identity() -> ModelIdentity:
    return ModelIdentity(
        artifact_kind="export",
        artifact_identity="sha256:" + "a" * 64,
        run_identity=None,
        step=7,
        checkpoint_identity=None,
        run_step_identity=None,
        tokenizer_identity="sha256:" + "1" * 64,
        verification=VerificationLevel.FULL,
        latest_recovered=False,
        pruning_pending=False,
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
        version=3,
        identity="sha256:" + "0" * 64,
        model=model,
        tasks=tasks,
        provider_result={
            "results": {task.task_name: task.metric_payload for task in tasks},
            "configs": {task.task_name: {} for task in tasks},
        },
    )


def identified_result(
    *, metric_payload: object = {"acc,none": 0.5}
) -> EvaluationResult:
    task = make_task_record(metric_payload=metric_payload)
    task = replace(task, task_identity=evaluation_task_identity(task))
    result = make_result(model=model_identity(), tasks=(task,))
    return replace(result, identity=evaluation_result_identity(result))


def with_provider_metadata(result: EvaluationResult, **metadata) -> EvaluationResult:
    changed = replace(result, provider_result={**result.provider_result, **metadata})
    return replace(changed, identity=evaluation_result_identity(changed))
