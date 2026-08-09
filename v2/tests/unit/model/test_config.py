from __future__ import annotations

import dataclasses
import math

import pytest
from sml.model.config import GenerationConfig, InitializerConfig, ModelConfig


def test_model_config_is_frozen_and_derives_context():
    """Changing head geometry or scale must update the exposed derived values."""
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=2.0,
    )

    assert config.head_dim == 4
    assert config.effective_context_length == 16
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.hidden_size = 32


def test_model_config_derives_residual_initializers_from_its_depth():
    """A stale residual scale after changing depth would destabilize residual paths."""
    config = ModelConfig(num_layers=8, initializer_range=0.04)

    assert config.initializers == InitializerConfig(
        embed_tokens=0.04,
        lm_head=0.04,
        q_proj=0.04,
        k_proj=0.04,
        v_proj=0.04,
        o_proj=0.01,
        gate_proj=0.04,
        up_proj=0.04,
        down_proj=0.01,
        other=0.04,
    )


def test_model_config_normalizes_initializer_mapping_but_preserves_explicit_config():
    """Consumers must receive one immutable initializer type without copying it."""
    explicit = InitializerConfig(q_proj=0.03)

    normalized = ModelConfig(initializers={"q_proj": 0.03})
    preserved = ModelConfig(initializers=explicit)

    assert normalized.initializers == explicit
    assert isinstance(normalized.initializers, InitializerConfig)
    assert preserved.initializers is explicit


@pytest.mark.parametrize(
    ("overrides", "message", "error_type"),
    [
        ({"hidden_size": 15, "num_q_heads": 4}, "divisible", ValueError),
        ({"hidden_size": 15, "num_q_heads": 3}, "even", ValueError),
        ({"num_q_heads": 4, "num_kv_heads": 3}, "divisible", ValueError),
        ({"rope_theta": math.inf}, "rope_theta", ValueError),
        ({"rope_scaling_factor": 0.5}, "rope_scaling_factor", ValueError),
        ({"yarn_mscale": 1.0}, "both be set", ValueError),
        ({"pad_token_id": 64, "vocab_size": 64}, "pad_token_id", ValueError),
        ({"pad_token_id": 1, "bos_token_id": 1}, "unique", ValueError),
        ({"use_cache": 1}, "use_cache", TypeError),
    ],
)
def test_model_config_rejects_invalid_geometry_scales_and_special_tokens(
    overrides, message, error_type
):
    """Malformed metadata must fail before it reaches attention or embeddings."""
    with pytest.raises(error_type, match=message):
        ModelConfig(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "message", "error_type"),
    [
        ({"temperature": math.nan}, "temperature", ValueError),
        ({"temperature": "cold"}, "temperature", TypeError),
        ({"top_p": 0.0}, "top_p", ValueError),
        ({"repetition_penalty": 0.0}, "repetition_penalty", ValueError),
        ({"no_repeat_ngram_size": -1}, "no_repeat_ngram_size", ValueError),
        ({"seed": -1}, "seed", ValueError),
        ({"seed": 2**32}, "seed", ValueError),
        ({"seed": 1.5}, "seed", ValueError),
    ],
)
def test_generation_config_rejects_invalid_decoding_controls(
    kwargs, message, error_type
):
    """Invalid controls must be rejected rather than changing generation silently."""
    with pytest.raises(error_type, match=message):
        GenerationConfig(**kwargs)


def test_generation_config_accepts_the_uint32_seed_boundary():
    assert GenerationConfig(seed=2**32 - 1).seed == 2**32 - 1
