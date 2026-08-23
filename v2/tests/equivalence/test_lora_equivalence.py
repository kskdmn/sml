from __future__ import annotations

import mlx.core as mx
import pytest
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.lora import LoRAConfig, apply_lora, merged_model_weights


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


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


def tiny_lora_config(**overrides: object) -> LoRAConfig:
    values: dict[str, object] = {
        "rank": 2,
        "alpha": 4.0,
        "scaling_mode": "lora",
        "dropout": 0.0,
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    }
    values.update(overrides)
    return LoRAConfig(**values)


@pytest.fixture(name="tiny_model_config")
def tiny_model_config_fixture() -> ModelConfig:
    return tiny_model_config()


@pytest.fixture(name="tiny_lora_config")
def tiny_lora_config_fixture():
    return tiny_lora_config


def test_lora_forward_matches_weight_pinned_legacy_reference(
    tiny_model_config,
    tiny_lora_config,
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
    load_legacy_lora_state,
):
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(4))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    adapted = apply_lora(model, tiny_lora_config(dropout=0.0), key=mx.random.key(5))
    load_legacy_lora_state(adapted, legacy_arrays, legacy_control)

    live_logits = adapted(legacy_arrays["lora.input_ids"], training=False).logits
    mx.eval(live_logits)
    assert_close(
        live_logits, legacy_arrays["lora.forward_logits"], atol=2e-2, rtol=2e-2
    )

    merged = merged_model_weights(adapted)
    inference = SMLLanguageModel(tiny_model_config, key=mx.random.key(9))
    inference.load_weights(list(merged.items()), strict=True)
    mx.eval(inference.parameters())
    merged_logits = inference(legacy_arrays["lora.input_ids"], training=False).logits
    mx.eval(merged_logits)
    assert_close(
        merged_logits, legacy_arrays["lora.merged_logits"], atol=2e-2, rtol=2e-2
    )


def test_live_and_merged_layer_outputs_match_captured_adapters_within_bf16_tolerance(
    tiny_model_config,
    tiny_lora_config,
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
    load_legacy_lora_state,
):
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(4))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    adapted = apply_lora(model, tiny_lora_config(dropout=0.0), key=mx.random.key(5))
    load_legacy_lora_state(adapted, legacy_arrays, legacy_control)
    layer = adapted.layers[0].self_attn.q_proj
    x = mx.ones((2, 4, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    live, _next_key = layer(x, key=mx.random.key(8), training=False)
    merged = merged_model_weights(adapted)["layers.0.self_attn.q_proj.weight"]
    merged_output = (x @ merged.T).astype(mx.bfloat16)
    mx.eval(live, merged_output)
    assert_close(live, merged_output, atol=2e-2, rtol=2e-2)
