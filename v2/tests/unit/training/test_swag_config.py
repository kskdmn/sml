from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig
from sml.training.swag import (
    SwagTrainingConfig,
    default_swag_optimizer_config,
)


def test_finetuning_defaults_to_eightfold_context():
    config = SwagTrainingConfig(
        base_checkpoint=Path("base"), data=Path("data"), output_run=Path("run")
    )
    assert config.rope_scaling_factor == 8.0
    assert (
        replace(
            ModelConfig(), rope_scaling_factor=config.rope_scaling_factor
        ).effective_context_length
        == 8_192
    )


@pytest.mark.parametrize("factor", [True, False, 0.5, float("inf"), float("nan"), "8"])
def test_finetuning_rejects_invalid_rope_scaling(factor):
    with pytest.raises(SMLConfigurationError, match="rope_scaling_factor"):
        SwagTrainingConfig(
            base_checkpoint=Path("base"),
            data=Path("data"),
            output_run=Path("run"),
            rope_scaling_factor=factor,
        )


def test_default_swag_optimizer_preserves_legacy_learning_rate_and_zero_adapter_decay():
    config = default_swag_optimizer_config()
    assert config.learning_rate == 1e-4
    assert config.schedule_steps == 8_192
    assert config.weight_decay.lora_a == 0.0
    assert config.weight_decay.lora_b == 0.0
