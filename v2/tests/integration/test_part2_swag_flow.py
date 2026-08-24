# ruff: noqa: F811
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_UNIT_DIR = Path(__file__).resolve().parents[1] / "unit"
if str(_UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIT_DIR))

import pytest
from sml.artifacts.manifest import ExportManifest, VerificationLevel, read_manifest
from sml.data.swag import SwagPreparationConfig, SwagSourceConfig, prepare_swag_bundle
from sml.evaluation import evaluate
from sml.inference import GenerationRequest, InferenceSession, resolve_model_artifact
from sml.model.config import ModelConfig
from sml.training.common import ResumeOverrides
from sml.training.swag import export_merged, finetune, resume_finetune
from test_evaluation import fake_lm_eval, tiny_evaluation_config  # noqa: F401
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
    exported = export_merged(resumed.run, tmp_path / "export")
    inference_result = InferenceSession.from_checkpoint(
        exported.path, full_verify=True
    ).generate("prompt", GenerationRequest(max_new_tokens=2))
    evaluation_result = evaluate(
        tiny_evaluation_config(exported.path, tmp_path, tasks=("hellaswag",))
    )
    assert inference_result.model.verification is VerificationLevel.FULL
    assert evaluation_result.model.artifact_kind == "export"
    assert read_export_manifest(exported.path).model_config.rope_scaling_factor == 1.0
