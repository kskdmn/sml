# ruff: noqa: F811
from __future__ import annotations

import sys
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

_UNIT_DIR = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIT_DIR))

import pytest
from sml.artifacts.manifest import (
    ExportManifest,
    VerificationLevel,
    read_checkpoint_manifest,
    read_manifest,
    read_run_manifest,
    structured_identity,
)
from sml.data.swag import SwagPreparationConfig, SwagSourceConfig, prepare_swag_bundle
from sml.evaluation import evaluate, read_evaluation_result
from sml.evaluation_result import normalize_json_value
from sml.inference import (
    GenerationRequest,
    InferenceSession,
    ModelIdentity,
    resolve_model_artifact,
)
from sml.model.config import ModelConfig
from sml.training.common import ResumeOverrides
from sml.training.swag import export_merged, finetune, resume_finetune
from test_evaluation import (  # noqa: F401
    fake_lm_eval,
    fake_provider,
    tiny_evaluation_config,
)
from test_swag_workflow import (  # noqa: F401
    FakeSwagProvider,
    _swag_rows,
    _tiny_base_template,
    tiny_base_run,
    tiny_swag_training_config,
)


def tiny_swag_preparation_config(provider: FakeSwagProvider) -> SwagPreparationConfig:
    return SwagPreparationConfig(
        provider=provider,
        source=SwagSourceConfig(revision="deadbeef" * 5),
        maximum_length=32,
        bucket_boundaries=(16, 32),
    )


@pytest.fixture
def fake_swag_provider() -> FakeSwagProvider:
    return FakeSwagProvider(_swag_rows(5))


def read_export_manifest(path: Path) -> SimpleNamespace:
    verified = read_manifest(path, ExportManifest, VerificationLevel.FULL)
    return SimpleNamespace(model_config=ModelConfig(**dict(verified.manifest.model)))


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _expected_provider_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, metadata.version(name))
            for name in ("datasets", "lm-eval", "mlx", "numpy", "sentencepiece")
        )
    )


def _expected_provider_samples(task_name: str) -> dict[str, object]:
    return {
        task_name: [
            {
                "request_type": "loglikelihood",
                "doc_id": 2,
                "repeats": 1,
                "arguments": ["alpha", " beta"],
            },
            {
                "request_type": "loglikelihood",
                "doc_id": 1,
                "repeats": 1,
                "arguments": ["gamma", " delta"],
            },
            {
                "request_type": "generate_until",
                "doc_id": 3,
                "repeats": 1,
                "arguments": [
                    "alpha",
                    {"max_gen_toks": 1, "until": ["omega"]},
                ],
            },
        ]
    }


def assert_complete_evaluation(
    path: Path,
    *,
    artifact_kind: str,
    step: int,
    run_identity: bool,
    checkpoint: Path,
) -> None:
    result = read_evaluation_result(path)
    assert result.kind == "evaluation-result"
    assert result.version == 2
    assert result.identity.startswith("sha256:")
    if run_identity:
        run = read_run_manifest(checkpoint, VerificationLevel.MANIFEST_TRUSTED).manifest
        checkpoint_manifest = read_checkpoint_manifest(
            checkpoint / "checkpoints" / f"step-{result.model.step:09d}",
            VerificationLevel.MANIFEST_TRUSTED,
        ).manifest
        assert result.model == ModelIdentity(
            artifact_kind=run.kind,
            run_identity=run.identity,
            step=checkpoint_manifest.step,
            checkpoint_identity=checkpoint_manifest.identity,
            run_step_identity=structured_identity(
                "sml-run-step-v1",
                {
                    "run_identity": run.identity,
                    "checkpoint_identity": checkpoint_manifest.identity,
                },
            ),
            tokenizer_identity=run.tokenizer_identity,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
    else:
        assert (
            result.model
            == resolve_model_artifact(checkpoint, full_verify=False).identity()
        )
    assert result.model.artifact_kind == artifact_kind
    assert result.model.step == step
    if run_identity:
        assert result.model.run_identity is not None
        assert result.model.run_identity.startswith("sha256:")
        assert result.model.checkpoint_identity is not None
        assert result.model.checkpoint_identity.startswith("sha256:")
        assert result.model.run_step_identity is not None
        assert result.model.run_step_identity.startswith("sha256:")
    else:
        assert result.model.run_identity is None
        assert result.model.checkpoint_identity is None
        assert result.model.run_step_identity is None
    assert result.model.tokenizer_identity.startswith("sha256:")
    assert result.model.verification is VerificationLevel.MANIFEST_TRUSTED
    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.task_name == "hellaswag"
    assert task.task_identity.startswith("sha256:")
    assert task.task_yaml.logical_name == "tasks/hellaswag.yaml"
    assert task.task_yaml.content_identity.startswith("sha256:")
    assert [item.logical_name for item in task.include_template_closure] == [
        "tasks/common.yaml"
    ]
    assert all(
        item.content_identity.startswith("sha256:")
        for item in task.include_template_closure
    )
    assert task.task_metadata_version == "1.0"
    assert task.prompt_config == {
        "adapter_padding": "right",
        "apply_chat_template": False,
        "description": "offline task",
        "doc_to_choice": None,
        "doc_to_target": "offline.doc_to_target",
        "doc_to_text": "offline.doc_to_text",
        "fewshot_delimiter": "\n\n",
        "gen_prefix": "",
        "output_type": "loglikelihood",
        "process_docs": "offline.process_docs",
        "system_instruction": None,
        "target_delimiter": " ",
    }
    assert task.few_shot_config == {
        "fewshot_as_multiturn": True,
        "fewshot_config": {
            "doc_to_choice": "offline.serialized._offline_doc_to_choice",
            "doc_to_target": "offline.serialized._offline_doc_to_target",
            "doc_to_text": "offline.serialized._offline_doc_to_text",
            "process_docs": "offline.serialized._offline_process_docs",
            "split": "fewshot-config",
        },
        "fewshot_split": "train",
        "num_fewshot": 0,
    }
    assert task.generation_config == {
        "generation_kwargs": {"temperature": 0.0},
        "provider_gen_kwargs": None,
    }
    assert task.metric_normalization_config == {
        "bootstrap_iters": 100000,
        "doc_to_decontamination_query": None,
        "filter_list": ({"filter": ({"function": "take_first"},), "name": "none"},),
        "log_samples": True,
        "metric_list": ({"metric": "acc"},),
        "predict_only": False,
        "repeats": 1,
        "should_decontaminate": False,
    }
    assert task.seeds == {
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
    }
    assert task.limit is None
    assert task.ordered_request_identity.startswith("sha256:")
    assert task.lm_eval_package_version == "0.4.12"
    assert task.lm_eval_source_commit is None
    assert task.dataset_revision == "version:1.0.0"
    assert task.dataset_fingerprint.startswith("sha256:")
    assert tuple((item.name, item.version) for item in task.provider_versions) == (
        _expected_provider_versions()
    )
    assert result.provider_result["results"] == {"hellaswag": task.metric_payload}
    assert set(result.provider_result) == {
        "results",
        "configs",
        "versions",
        "n-shot",
        "higher_is_better",
        "n-samples",
        "samples",
        "config",
        "git_hash",
        "date",
    }
    assert set(result.provider_result["configs"]) == {"hellaswag"}
    provider_config = result.provider_result["configs"]["hellaswag"]
    assert set(provider_config) == {
        "task",
        "metadata",
        "output_type",
        "description",
        "process_docs",
        "doc_to_text",
        "doc_to_target",
        "doc_to_choice",
        "target_delimiter",
        "fewshot_delimiter",
        "gen_prefix",
        "padding",
        "num_fewshot",
        "fewshot_split",
        "fewshot_config",
        "generation_kwargs",
        "metric_list",
        "repeats",
        "should_decontaminate",
        "doc_to_decontamination_query",
        "validation_split",
        "test_split",
        "dataset_kwargs",
    }
    assert provider_config["task"] == task.task_name
    assert provider_config["metadata"] == {"version": 1.0}
    assert provider_config["output_type"] == task.prompt_config["output_type"]
    assert provider_config["description"] == task.prompt_config["description"]
    assert provider_config["process_docs"] == task.prompt_config["process_docs"]
    assert provider_config["doc_to_text"] == task.prompt_config["doc_to_text"]
    assert provider_config["doc_to_target"] == task.prompt_config["doc_to_target"]
    assert provider_config["doc_to_choice"] == task.prompt_config["doc_to_choice"]
    assert provider_config["target_delimiter"] == task.prompt_config["target_delimiter"]
    assert (
        provider_config["fewshot_delimiter"] == task.prompt_config["fewshot_delimiter"]
    )
    assert provider_config["gen_prefix"] == task.prompt_config["gen_prefix"]
    assert provider_config["padding"] == "left"
    assert provider_config["num_fewshot"] == 0
    assert provider_config["fewshot_split"] == task.few_shot_config["fewshot_split"]
    assert provider_config["fewshot_config"] == task.few_shot_config["fewshot_config"]
    assert (
        provider_config["generation_kwargs"]
        == task.generation_config["generation_kwargs"]
    )
    assert (
        provider_config["metric_list"]
        == task.metric_normalization_config["metric_list"]
    )
    assert provider_config["repeats"] == task.metric_normalization_config["repeats"]
    assert provider_config["should_decontaminate"] is False
    assert provider_config["doc_to_decontamination_query"] is None
    assert provider_config["validation_split"] == "validation"
    assert provider_config["test_split"] == "test"
    assert provider_config["dataset_kwargs"] == {}
    assert result.provider_result["versions"] == {"hellaswag": "1.0.0"}
    assert result.provider_result["n-shot"] == {"hellaswag": 0}
    assert result.provider_result["higher_is_better"] == {
        "hellaswag": {"acc,none": True}
    }
    assert result.provider_result["n-samples"] == {
        "hellaswag": {"effective": 2, "original": 2}
    }
    assert normalize_json_value(
        result.provider_result["samples"], context="persisted provider samples"
    ) == normalize_json_value(
        _expected_provider_samples("hellaswag"), context="expected provider samples"
    )
    assert result.provider_result["config"] == {
        "bootstrap_iters": 100000,
        "log_samples": True,
    }
    assert result.provider_result["git_hash"] == "offline-lm-eval-commit"
    assert result.provider_result["date"] == "2026-08-25T00:00:00Z"


def test_encoded_swag_to_exported_evaluation(
    tiny_base_run, fake_swag_provider, fake_lm_eval, tmp_path
):
    data = prepare_swag_bundle(
        tiny_swag_preparation_config(fake_swag_provider),
        resolve_model_artifact(tiny_base_run, full_verify=True),
        tmp_path / "swag",
    )
    tuned = finetune(
        tiny_swag_training_config(
            tiny_base_run, data, tmp_path / "run", maximum_steps=1
        )
    )
    resumed = resume_finetune(
        tuned.run, data=data.path, overrides=ResumeOverrides(maximum_steps=2)
    )
    lora_path = tmp_path / "lora-evaluation.json"
    lora_evaluation = evaluate(
        replace(
            tiny_evaluation_config(resumed.run, tmp_path, tasks=("hellaswag",)),
            output=lora_path,
        )
    )
    exported = export_merged(resumed.run, tmp_path / "export")
    inference_result = InferenceSession.from_checkpoint(
        exported.path, full_verify=True
    ).generate("prompt", GenerationRequest(max_new_tokens=2))
    evaluation_result = evaluate(
        tiny_evaluation_config(exported.path, tmp_path, tasks=("hellaswag",))
    )
    assert inference_result.model.verification is VerificationLevel.FULL
    assert evaluation_result.model.artifact_kind == "export"
    assert evaluation_result.model.verification is VerificationLevel.MANIFEST_TRUSTED
    assert lora_evaluation.model.artifact_kind == "lora-run"
    assert_complete_evaluation(
        lora_path,
        artifact_kind="lora-run",
        step=2,
        run_identity=True,
        checkpoint=resumed.run,
    )
    assert_complete_evaluation(
        tmp_path / "evaluation.json",
        artifact_kind="export",
        step=2,
        run_identity=False,
        checkpoint=exported.path,
    )
    assert read_export_manifest(exported.path).model_config.rope_scaling_factor == 1.0
