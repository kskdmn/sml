from __future__ import annotations

import inspect
import math
from dataclasses import replace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.model.layers import _Linear
from sml.training import lora as lora_module
from sml.training.common import PrecisionConfig
from sml.training.lora import (
    LoRAConfig,
    LoRAInitializerConfig,
    LoRALinear,
    LoRAPrecisionConfig,
    apply_lora,
    load_lora_state_dict,
    lora_state_dict,
    merged_model_weights,
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


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def assert_lora_state_equal(actual: dict[str, mx.array], expected: dict[str, mx.array]):
    assert set(actual) == set(expected)
    mx.eval(*actual.values(), *expected.values())
    for name, value in actual.items():
        assert bool(mx.array_equal(value, expected[name]).item()), name


@pytest.fixture(name="tiny_model_config")
def tiny_model_config_fixture() -> ModelConfig:
    return tiny_model_config()


@pytest.fixture(name="tiny_lora_config")
def tiny_lora_config_fixture():
    return tiny_lora_config


@pytest.fixture
def tiny_model(tiny_model_config: ModelConfig) -> SMLLanguageModel:
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(3))
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny_adapted_model(
    tiny_model: SMLLanguageModel,
    tiny_lora_config,
) -> SMLLanguageModel:
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(
            dropout=0.0,
            initializer=LoRAInitializerConfig(lora_a=0.01, lora_b=0.05),
        ),
        key=mx.random.key(6),
    )
    mx.eval(adapted.parameters())
    return adapted


def test_lora_config_keeps_captured_legacy_defaults():
    config = LoRAConfig()

    assert config.rank == 16
    assert config.alpha == 32.0
    assert config.scaling_mode == "rslora"
    assert config.dropout == 0.05
    assert config.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert config.initializer == LoRAInitializerConfig(lora_a=0.01, lora_b=0.0)


def test_lora_precision_is_distinct_from_pretraining_master_weights():
    config = LoRAPrecisionConfig()

    assert config.frozen_base_dtype == "bfloat16"
    assert config.adapter_parameter_dtype == "float32"
    assert config.gradient_accumulator_dtype == "float32"
    assert config.optimizer_state_dtype == "float32"
    assert config.update_dtype == "float32"
    assert config.dynamic_loss_scaling is False
    assert "master_weights" not in LoRAPrecisionConfig.__dataclass_fields__
    assert PrecisionConfig().master_weights is True
    with pytest.raises(SMLConfigurationError, match="frozen_base_dtype"):
        LoRAPrecisionConfig(frozen_base_dtype="float32")  # type: ignore[arg-type]


def test_lora_config_rejects_invalid_rank_alpha_dropout_scaling_and_targets():
    with pytest.raises(SMLConfigurationError, match="rank"):
        LoRAConfig(rank=0)
    with pytest.raises(SMLConfigurationError, match="alpha"):
        LoRAConfig(alpha=0.0)
    with pytest.raises(SMLConfigurationError, match="dropout"):
        LoRAConfig(dropout=-0.1)
    with pytest.raises(SMLConfigurationError, match="dropout"):
        LoRAConfig(dropout=1.0)
    with pytest.raises(SMLConfigurationError, match="scaling_mode"):
        LoRAConfig(scaling_mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(SMLConfigurationError, match="target_modules"):
        LoRAConfig(target_modules=())
    with pytest.raises(SMLConfigurationError, match="target_modules"):
        LoRAConfig(target_modules=("q_proj", "q_proj"))
    with pytest.raises(SMLConfigurationError, match="target_modules"):
        LoRAConfig(target_modules=("embed_tokens",))  # type: ignore[arg-type]
    with pytest.raises(SMLConfigurationError, match="lora_a"):
        LoRAInitializerConfig(lora_a=-0.1)
    with pytest.raises(SMLConfigurationError, match="lora_b"):
        LoRAInitializerConfig(lora_b=math.inf)


def test_lora_linear_uses_configured_scaling_modes():
    linear = _Linear(8, 4)
    linear.weight = mx.ones((4, 8), dtype=mx.bfloat16)
    lora = LoRALinear(
        linear,
        replace(tiny_lora_config(), rank=4, alpha=8.0, scaling_mode="lora"),
        key=mx.random.key(1),
    )
    rslora = LoRALinear(
        _Linear(8, 4),
        replace(tiny_lora_config(), rank=4, alpha=8.0, scaling_mode="rslora"),
        key=mx.random.key(2),
    )

    mx.eval(lora.scale, rslora.scale)
    assert_close(
        lora.scale.astype(mx.float32),
        mx.array(2.0, dtype=mx.float32),
        atol=0.0,
        rtol=0.0,
    )
    assert_close(
        rslora.scale.astype(mx.float32),
        mx.array(4.0, dtype=mx.float32),
        atol=0.0,
        rtol=0.0,
    )


def test_lora_dtype_boundaries_and_live_formula(tiny_model):
    adapted = apply_lora(
        tiny_model, tiny_lora_config(dropout=0.0), key=mx.random.key(4)
    )
    layer = adapted.layers[0].self_attn.q_proj
    x = mx.ones((1, 2, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    actual, _next_key = layer(x, key=mx.random.key(8), training=False)
    expected_adapter = layer.scale.astype(mx.float32) * (
        (x.astype(mx.float32) @ layer.lora_a.T) @ layer.lora_b.T
    )
    expected = layer.base(x) + expected_adapter.astype(mx.bfloat16)
    mx.eval(actual, expected)
    assert layer.base.weight.dtype == mx.bfloat16
    assert layer.lora_a.dtype == mx.float32
    assert layer.lora_b.dtype == mx.float32
    assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_merged_weight_matches_exact_array_formula_without_mutation(tiny_adapted_model):
    before = lora_state_dict(tiny_adapted_model)
    merged = merged_model_weights(tiny_adapted_model)
    module = tiny_adapted_model.layers[0].self_attn.q_proj
    expected = (
        module.base.weight.astype(mx.float32)
        + module.scale.astype(mx.float32) * (module.lora_b @ module.lora_a)
    ).astype(mx.bfloat16)
    assert mx.array_equal(merged["layers.0.self_attn.q_proj.weight"], expected).item()
    assert_lora_state_equal(lora_state_dict(tiny_adapted_model), before)


def test_merged_weights_use_plain_inference_parameter_names(tiny_adapted_model):
    merged = merged_model_weights(tiny_adapted_model)
    names = set(merged)

    assert "layers.0.self_attn.q_proj.weight" in names
    assert "layers.0.mlp.gate_proj.weight" in names
    assert "embed_tokens.weight" in names
    assert not any(name.endswith(".base.weight") for name in names)
    assert not any(name.endswith((".lora_a", ".lora_b")) for name in names)


def test_apply_lora_wraps_only_configured_linear_targets(tiny_model, tiny_lora_config):
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(target_modules=("q_proj", "gate_proj")),
        key=mx.random.key(4),
    )

    assert isinstance(adapted.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(adapted.layers[0].mlp.gate_proj, LoRALinear)
    assert not isinstance(adapted.layers[0].self_attn.k_proj, LoRALinear)
    assert not isinstance(adapted.layers[0].mlp.up_proj, LoRALinear)
    assert isinstance(adapted.layers[0].self_attn.k_proj, _Linear)


def test_apply_lora_leaves_only_adapter_arrays_trainable(tiny_adapted_model):
    trainable = dict(tree_flatten(tiny_adapted_model.trainable_parameters()))
    parameters = dict(tree_flatten(tiny_adapted_model.parameters()))

    assert set(trainable) == {
        name for name in parameters if name.endswith((".lora_a", ".lora_b"))
    }
    assert all(value.dtype == mx.float32 for value in trainable.values())
    assert all(
        value.dtype == mx.bfloat16
        for name, value in parameters.items()
        if not name.endswith((".lora_a", ".lora_b"))
    )


def test_lora_dropout_replays_from_the_same_explicit_key(tiny_model, tiny_lora_config):
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(
            dropout=0.5,
            initializer=LoRAInitializerConfig(lora_a=0.01, lora_b=0.05),
        ),
        key=mx.random.key(4),
    )
    layer = adapted.layers[0].self_attn.q_proj
    x = mx.ones((4, 8, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    key = mx.random.key(11)

    first, first_next_key = layer(x, key=key, training=True)
    replay, replay_next_key = layer(x, key=key, training=True)
    idle, idle_next_key = layer(x, key=key, training=False)

    assert_close(first, replay, atol=0.0, rtol=0.0)
    assert_close(first_next_key, replay_next_key, atol=0.0, rtol=0.0)
    mx.eval(first_next_key, idle_next_key, key, first, idle)
    assert not bool(mx.array_equal(first_next_key, key).item())
    assert bool(mx.array_equal(idle_next_key, key).item())
    assert not bool(mx.allclose(first, idle, atol=0.0, rtol=0.0).item())


def test_live_and_merged_outputs_match_within_pinned_bf16_tolerance(tiny_adapted_model):
    layer = tiny_adapted_model.layers[0].self_attn.q_proj
    x = mx.ones((2, 3, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    live, _next_key = layer(x, key=mx.random.key(8), training=False)
    merged = merged_model_weights(tiny_adapted_model)[
        "layers.0.self_attn.q_proj.weight"
    ]
    merged_output = (x @ merged.T).astype(mx.bfloat16)

    assert_close(live, merged_output, atol=2e-2, rtol=2e-2)


def test_load_lora_state_dict_rejects_missing_additional_wrong_shape_and_dtype(
    tiny_adapted_model,
):
    state = lora_state_dict(tiny_adapted_model)
    first_key = next(iter(state))

    missing = dict(state)
    missing.pop(first_key)
    with pytest.raises(SMLConfigurationError, match="missing"):
        load_lora_state_dict(tiny_adapted_model, missing)

    additional = dict(state)
    additional["unexpected.lora_a"] = mx.zeros_like(state[first_key])
    with pytest.raises(SMLConfigurationError, match="additional"):
        load_lora_state_dict(tiny_adapted_model, additional)

    wrong_shape = dict(state)
    wrong_shape[first_key] = mx.zeros((1, 1), dtype=mx.float32)
    with pytest.raises(SMLConfigurationError, match="shape"):
        load_lora_state_dict(tiny_adapted_model, wrong_shape)

    wrong_dtype = dict(state)
    wrong_dtype[first_key] = state[first_key].astype(mx.bfloat16)
    with pytest.raises(SMLConfigurationError, match="dtype"):
        load_lora_state_dict(tiny_adapted_model, wrong_dtype)


def test_lora_state_dict_round_trip_preserves_adapter_arrays(tiny_model_config):
    source = SMLLanguageModel(tiny_model_config, key=mx.random.key(13))
    target = SMLLanguageModel(tiny_model_config, key=mx.random.key(17))
    config = tiny_lora_config(
        initializer=LoRAInitializerConfig(lora_a=0.01, lora_b=0.05)
    )
    apply_lora(source, config, key=mx.random.key(4))
    apply_lora(target, config, key=mx.random.key(5))

    load_lora_state_dict(target, lora_state_dict(source))

    assert_lora_state_equal(lora_state_dict(target), lora_state_dict(source))


def test_lora_uses_keyed_dropout_rather_than_nn_dropout():
    source = inspect.getsource(lora_module)

    assert "keyed_dropout" in source
    assert "nn.Dropout" not in source
