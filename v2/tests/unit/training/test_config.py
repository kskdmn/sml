from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    PretrainingConfig,
    WeightDecayPolicy,
    build_weight_decay_tree,
    learning_rate_at,
)

from .helpers import (
    assert_close,
    numpy_schedule_oracle,
)


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=1.0,
        hidden_dropout=0.0,
    )


def test_pretraining_config_pins_standard_rope_and_composed_defaults(tmp_path):
    """A non-standard pretraining RoPE factor would break saved-run compatibility."""
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=replace(tiny_model_config(), rope_scaling_factor=1.0),
    )

    assert config.model.rope_scaling_factor == 1.0
    assert config.loader.microbatch_size == 1
    assert config.loader.gradient_accumulation_steps == 8
    assert config.optimizer.bias_correction is False
    assert config.optimizer.schedule_steps is None
    assert config.optimizer.warmup_steps is None
    with pytest.raises(
        SMLConfigurationError, match="pretraining.*rope_scaling_factor.*1.0"
    ):
        replace(config, model=replace(config.model, rope_scaling_factor=2.0))


def test_training_configs_reject_invalid_controls_before_array_allocation(tmp_path):
    """Invalid controls must fail on the host instead of reaching device updates."""
    with pytest.raises(SMLConfigurationError, match="learning_rate"):
        OptimizerConfig(learning_rate=math.nan)
    with pytest.raises(SMLConfigurationError, match="warmup_steps"):
        OptimizerConfig(schedule_steps=10, warmup_steps=10)
    with pytest.raises(SMLConfigurationError, match="microbatch_size"):
        LoaderConfig(microbatch_size=0)
    with pytest.raises(SMLConfigurationError, match="interval"):
        CheckpointPolicy(interval=0)
    with pytest.raises(SMLConfigurationError, match="master_parameter_dtype"):
        PrecisionConfig(master_parameter_dtype="bfloat16")  # type: ignore[arg-type]
    with pytest.raises(SMLConfigurationError, match="termination"):
        PretrainingConfig(
            data=Path(tmp_path / "data"),
            output_run=Path(tmp_path / "run"),
            model=tiny_model_config(),
            maximum_steps=None,
            maximum_epochs=None,
        )
    with pytest.raises(SMLConfigurationError, match="maximum_steps"):
        PretrainingConfig(
            data=Path(tmp_path / "data"),
            output_run=Path(tmp_path / "run"),
            model=tiny_model_config(),
            maximum_steps=2**31,
        )


def test_learning_rate_schedule_matches_scalar_oracle():
    """Warmup and cosine decay must agree with the reference at their boundaries."""
    config = OptimizerConfig(
        schedule_steps=100,
        warmup_steps=10,
        minimum_learning_rate_ratio=0.1,
    )
    steps = mx.array([0, 9, 10, 55, 100], dtype=mx.int32)

    actual = mx.vmap(lambda step: learning_rate_at(step, config))(steps)
    expected = numpy_schedule_oracle(steps.tolist(), config)

    assert_close(actual, expected, atol=1e-8, rtol=1e-8)


def test_weight_decay_policy_classifies_tied_embeddings_norms_projections_and_lora():
    """A misplaced decay rate would silently change the saved optimizer policy."""
    named_parameters = {
        "embed_tokens": {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
        "layers": [
            {
                "input_norm": {"weight": mx.ones((2,), dtype=mx.bfloat16)},
                "self_attn": {
                    "q_proj": {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
                    "k_proj": {"lora_A": mx.ones((2, 1), dtype=mx.bfloat16)},
                    "v_proj": {"lora_B": mx.ones((1, 2), dtype=mx.bfloat16)},
                },
            }
        ],
        "lm_head": {"weight": mx.ones((2, 2), dtype=mx.bfloat16)},
        "other": {"weight": mx.ones((2,), dtype=mx.bfloat16)},
    }
    policy = WeightDecayPolicy(
        embed_tokens=0.11,
        lm_head=0.22,
        q_proj=0.21,
        lora_a=0.03,
        lora_b=0.04,
        other=0.07,
    )

    decay = dict(tree_flatten(build_weight_decay_tree(named_parameters, policy)))

    assert decay["embed_tokens.weight"] == 0.11
    assert decay["layers.0.input_norm.weight"] == 0.0
    assert decay["layers.0.self_attn.q_proj.weight"] == 0.21
    assert decay["layers.0.self_attn.k_proj.lora_A"] == 0.03
    assert decay["layers.0.self_attn.v_proj.lora_B"] == 0.04
    assert decay["lm_head.weight"] == 0.22
    assert decay["other.weight"] == 0.07
