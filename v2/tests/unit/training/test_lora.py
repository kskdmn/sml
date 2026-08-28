from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, replace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from sml.errors import SMLConfigurationError
from sml.model import LoRAAdapterSpec, LoRAForwardPolicy
from sml.model import layers as layers_module
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.model.layers import _Linear, _linear, keyed_dropout
from sml.training.common import PrecisionConfig
from sml.training.lora import (
    LoRAConfig,
    LoRAInitializerConfig,
    LoRALinear,
    LoRAPrecisionConfig,
    apply_lora,
    load_lora_state_dict,
    lora_config_from_mapping,
    lora_parameter_specs,
    lora_state_dict,
    merged_model_weights,
    split_adapter_parameters,
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


class _StringSubclass(str):
    pass


class _LoRAAdapterLookalike:
    module_path = "layers.0.self_attn.q_proj"


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
            rank=3,
            alpha=1.0,
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


def test_lora_config_reconstructs_canonical_zero_dropout_for_static_policy(
    tiny_model,
):
    config = lora_config_from_mapping(
        {
            "rank": 2,
            "alpha": 4.0,
            "scaling_mode": "lora",
            "dropout": 0,
            "target_modules": ["q_proj"],
            "initializer": {"lora_a": 0.01, "lora_b": 0},
        }
    )

    adapted = apply_lora(tiny_model, config, key=mx.random.key(4))

    assert adapted.lora_forward_policy is not None
    assert all(
        type(spec.dropout) is float and spec.dropout == 0.0
        for spec in adapted.lora_forward_policy.adapters
    )


def test_lora_linear_uses_configured_static_scaling_modes():
    linear = _Linear(8, 4)
    linear.weight = mx.ones((4, 8), dtype=mx.bfloat16)
    lora = LoRALinear(
        linear,
        tiny_lora_config(rank=4),
        module_path="test.lora",
        spec=LoRAAdapterSpec(module_path="test.lora", scale=2.0, dropout=0.0),
        key=mx.random.key(1),
    )
    rslora = LoRALinear(
        _Linear(8, 4),
        tiny_lora_config(rank=4),
        module_path="test.rslora",
        spec=LoRAAdapterSpec(module_path="test.rslora", scale=4.0, dropout=0.0),
        key=mx.random.key(2),
    )

    assert lora.spec.scale == 2.0
    assert rslora.spec.scale == 4.0
    assert lora.module_path == "test.lora"
    assert rslora.module_path == "test.rslora"


def test_lora_dtype_boundaries_and_live_formula(tiny_model):
    adapted = apply_lora(
        tiny_model, tiny_lora_config(dropout=0.0), key=mx.random.key(4)
    )
    layer = adapted.layers[0].self_attn.q_proj
    x = mx.ones((1, 2, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    actual, _next_key = layer(x, key=mx.random.key(8), training=False)
    expected_adapter = mx.array(layer.spec.scale, dtype=mx.float32) * (
        (x.astype(mx.float32) @ layer.lora_a.T) @ layer.lora_b.T
    )
    expected = layer.base(x) + expected_adapter.astype(mx.bfloat16)
    mx.eval(actual, expected)
    assert layer.base.weight.dtype == mx.bfloat16
    assert layer.lora_a.dtype == mx.float32
    assert layer.lora_b.dtype == mx.float32
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("targets", "rank", "expected_names"),
    [
        (
            ("q_proj",),
            2,
            (
                "layers.0.self_attn.q_proj.lora_a",
                "layers.0.self_attn.q_proj.lora_b",
                "layers.1.self_attn.q_proj.lora_a",
                "layers.1.self_attn.q_proj.lora_b",
            ),
        ),
        (
            ("k_proj", "down_proj"),
            3,
            (
                "layers.0.mlp.down_proj.lora_a",
                "layers.0.mlp.down_proj.lora_b",
                "layers.0.self_attn.k_proj.lora_a",
                "layers.0.self_attn.k_proj.lora_b",
                "layers.1.mlp.down_proj.lora_a",
                "layers.1.mlp.down_proj.lora_b",
                "layers.1.self_attn.k_proj.lora_a",
                "layers.1.self_attn.k_proj.lora_b",
            ),
        ),
    ],
)
def test_lora_parameter_specs_match_adapter_state(
    tiny_model_config,
    targets,
    rank,
    expected_names,
):
    """Catches adapter manifest metadata drifting from transformed-model leaves."""
    config = tiny_lora_config(target_modules=targets, rank=rank)
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(43))
    apply_lora(model, config, key=mx.random.key(47))

    actual = {
        spec.name: (spec.shape, spec.dtype)
        for spec in lora_parameter_specs(tiny_model_config, config)
    }
    expected = {
        name: (tuple(value.shape), str(value.dtype).removeprefix("mlx.core."))
        for name, value in lora_state_dict(model).items()
    }

    assert tuple(actual) == expected_names
    assert actual == expected
    assert all(name.endswith((".lora_a", ".lora_b")) for name in actual)
    assert all(dtype == "float32" for _shape, dtype in actual.values())
    assert actual[expected_names[0]][0][0] == rank
    assert actual[expected_names[1]][0][1] == rank


def test_shared_linear_formula_matches_live_wrapper_and_manual_dropout_oracle():
    module_path = "layers.0.self_attn.q_proj"
    spec = LoRAAdapterSpec(module_path=module_path, scale=1.0 / 3.0, dropout=0.5)
    linear = _Linear(4, 3)
    linear.weight = mx.array(
        [[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0], [2.0, 0.0, 1.0, -0.5]],
        dtype=mx.bfloat16,
    )
    layer = LoRALinear(
        linear,
        tiny_lora_config(rank=2),
        module_path=module_path,
        spec=spec,
        key=mx.random.key(7),
    )
    layer.lora_a = mx.array(
        [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 1.0]],
        dtype=mx.float32,
    )
    layer.lora_b = mx.array(
        [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
        dtype=mx.float32,
    )
    x = mx.array(
        [[[1.0, -0.5, 2.0, 0.25], [0.75, 1.5, -1.0, 2.0]]],
        dtype=mx.bfloat16,
    )
    key = mx.random.key(19)
    adapter_input, expected_key = keyed_dropout(x.astype(mx.float32), spec.dropout, key)
    expected_adapter = mx.array(1.0 / 3.0, dtype=mx.float32) * (
        (adapter_input @ layer.lora_a.T) @ layer.lora_b.T
    )
    expected = (x @ linear.weight.T + expected_adapter.astype(mx.bfloat16)).astype(
        mx.bfloat16
    )
    policy = LoRAForwardPolicy((spec,))

    pure, pure_key = _linear(
        x,
        layer.parameters(),
        module_path=module_path,
        lora_policy=policy,
        training=True,
        key=key,
    )
    live, live_key = layer(x, training=True, key=key)

    mx.eval(expected, pure, live, expected_key, pure_key, live_key)
    assert pure.dtype == live.dtype == mx.bfloat16
    assert bool(mx.array_equal(pure, expected).item())
    assert bool(mx.array_equal(live, expected).item())
    assert bool(mx.array_equal(pure_key, expected_key).item())
    assert bool(mx.array_equal(live_key, expected_key).item())


def test_merged_weight_matches_exact_array_formula_without_mutation(tiny_adapted_model):
    module = tiny_adapted_model.layers[0].self_attn.q_proj
    module.base.weight = mx.zeros_like(module.base.weight)
    module.lora_a = mx.zeros_like(module.lora_a).at[0, 0].add(3.0087)
    module.lora_b = mx.zeros_like(module.lora_b).at[0, 0].add(1.0)
    before = lora_state_dict(tiny_adapted_model)
    merged = merged_model_weights(tiny_adapted_model)
    spec = module.spec
    scale = mx.array(spec.scale, dtype=mx.float32)
    expected = (
        module.base.weight.astype(mx.float32) + scale * (module.lora_b @ module.lora_a)
    ).astype(mx.bfloat16)
    bf16_scale_result = (
        module.base.weight.astype(mx.float32)
        + scale.astype(mx.bfloat16).astype(mx.float32) * (module.lora_b @ module.lora_a)
    ).astype(mx.bfloat16)
    mx.eval(expected, bf16_scale_result)
    assert not bool(mx.array_equal(expected, bf16_scale_result).item())
    assert bool(
        mx.array_equal(
            merged["layers.0.self_attn.q_proj.weight"],
            expected,
        ).item()
    )
    assert_lora_state_equal(lora_state_dict(tiny_adapted_model), before)


def test_merged_weights_use_plain_inference_parameter_names(tiny_adapted_model):
    merged = merged_model_weights(tiny_adapted_model)
    names = set(merged)

    assert "layers.0.self_attn.q_proj.weight" in names
    assert "layers.0.mlp.gate_proj.weight" in names
    assert "embed_tokens.weight" in names
    assert not any(name.endswith(".base.weight") for name in names)
    assert not any(name.endswith((".lora_a", ".lora_b", ".scale")) for name in names)


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


def test_lora_policy_is_canonical_static_and_scale_is_not_a_parameter(
    tiny_model,
) -> None:
    """Catches policy order following traversal or config order instead of execution."""
    config = tiny_lora_config(
        rank=3,
        alpha=1.0,
        scaling_mode="lora",
        dropout=0.25,
        target_modules=("down_proj", "v_proj", "q_proj"),
    )
    adapted = apply_lora(tiny_model, config, key=mx.random.key(4))
    policy = adapted.lora_forward_policy
    assert isinstance(policy, LoRAForwardPolicy)
    assert tuple(spec.module_path for spec in policy.adapters) == (
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.mlp.down_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.mlp.down_proj",
    )
    assert all(spec.dropout == 0.25 for spec in policy.adapters)
    parameters = dict(tree_flatten(adapted.parameters()))
    trainable = dict(tree_flatten(adapted.trainable_parameters()))
    assert not any(name.endswith(".scale") for name in parameters)
    assert set(trainable) == {
        name for name in parameters if name.endswith((".lora_a", ".lora_b"))
    }
    scale_fp32 = mx.array(policy.adapters[0].scale, dtype=mx.float32)
    scale_bf16_round_trip = scale_fp32.astype(mx.bfloat16).astype(mx.float32)
    mx.eval(scale_fp32, scale_bf16_round_trip)
    assert not bool(mx.array_equal(scale_fp32, scale_bf16_round_trip).item())


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("module_path", ""),
        ("module_path", 1),
        ("module_path", b"layers.0.self_attn.q_proj"),
        ("module_path", _StringSubclass("layers.0.self_attn.q_proj")),
        ("scale", True),
        ("scale", 1),
        ("scale", "1.0"),
        ("scale", math.inf),
        ("scale", 0.0),
        ("dropout", False),
        ("dropout", 0),
        ("dropout", "0.0"),
        ("dropout", math.inf),
        ("dropout", 1.0),
    ),
)
def test_lora_adapter_spec_rejects_noncanonical_static_fields(
    field_name: str,
    value: object,
) -> None:
    """Catches coercing non-float policy values into static adapter behavior."""
    fields: dict[str, object] = {
        "module_path": "layers.0.self_attn.q_proj",
        "scale": 1.0,
        "dropout": 0.0,
    }
    fields[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        LoRAAdapterSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "adapters",
    (
        [],
        (),
        (
            LoRAAdapterSpec("layers.0.self_attn.q_proj", 1.0, 0.0),
            LoRAAdapterSpec("layers.0.self_attn.q_proj", 2.0, 0.0),
        ),
        (_LoRAAdapterLookalike(),),
    ),
)
def test_lora_forward_policy_rejects_noncanonical_adapter_collections(
    adapters: object,
) -> None:
    """Catches mutable, empty, duplicate, or duck-typed policy adapters."""
    with pytest.raises(ValueError, match="LoRA policy"):
        LoRAForwardPolicy(adapters)  # type: ignore[arg-type]


@pytest.mark.parametrize("module_path", ("", 1, b"layers.0.self_attn.q_proj"))
def test_lora_forward_policy_rejects_noncanonical_lookup_paths(
    module_path: object,
) -> None:
    """Catches invalid lookup paths being silently treated as missing adapters."""
    policy = LoRAForwardPolicy(
        (LoRAAdapterSpec("layers.0.self_attn.q_proj", 1.0, 0.0),)
    )

    with pytest.raises(ValueError, match="module_path"):
        policy.for_module(module_path)  # type: ignore[arg-type]


def test_lora_forward_policy_is_recursively_immutable() -> None:
    """Catches mutable policy adapters changing static forward behavior after setup."""
    policy = LoRAForwardPolicy(
        (LoRAAdapterSpec("layers.0.self_attn.q_proj", 1.0, 0.0),)
    )

    with pytest.raises(FrozenInstanceError):
        policy.adapters[0].scale = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        policy.adapters[0] = LoRAAdapterSpec(  # type: ignore[index]
            "layers.0.self_attn.v_proj", 2.0, 0.0
        )


def test_split_adapter_parameters_contains_only_fp32_adapters_and_bf16_base(
    tiny_model,
) -> None:
    """Catches scale leaking into adapter or frozen parameter trees."""
    adapted = apply_lora(tiny_model, tiny_lora_config(), key=mx.random.key(5))
    adapters, frozen = split_adapter_parameters(adapted.parameters())
    adapter_leaves = dict(tree_flatten(adapters))
    frozen_leaves = dict(tree_flatten(frozen))
    assert adapter_leaves
    assert frozen_leaves
    assert all(array.dtype == mx.float32 for array in adapter_leaves.values())
    assert all(array.dtype == mx.bfloat16 for array in frozen_leaves.values())
    assert not any(name.endswith("scale") for name in (*adapter_leaves, *frozen_leaves))


def test_apply_lora_leaves_only_adapter_arrays_trainable(tiny_adapted_model):
    trainable = dict(tree_flatten(tiny_adapted_model.trainable_parameters()))
    parameters = dict(tree_flatten(tiny_adapted_model.parameters()))

    assert not any(name.endswith(".scale") for name in parameters)
    assert set(trainable) == {
        name for name in parameters if name.endswith((".lora_a", ".lora_b"))
    }
    assert all(value.dtype == mx.float32 for value in trainable.values())
    assert all(
        value.dtype == mx.bfloat16
        for name, value in parameters.items()
        if not name.endswith((".lora_a", ".lora_b"))
    )


def test_adapted_language_model_forward_returns_bf16_logits(
    tiny_model, tiny_lora_config
):
    adapted = apply_lora(
        tiny_model, tiny_lora_config(dropout=0.0), key=mx.random.key(4)
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    output = adapted(input_ids, training=False)
    mx.eval(output.logits)

    assert output.logits.dtype == mx.bfloat16
    assert tuple(output.logits.shape) == (1, 4, tiny_model.config.vocab_size)


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


def test_pure_array_lora_dropout_replays_and_advances_key(
    tiny_model,
    tiny_lora_config,
) -> None:
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(
            dropout=0.5,
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        key=mx.random.key(4),
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    key = mx.random.key(19)
    first, first_cache, first_key = adapted.forward_arrays(
        adapted.parameters(),
        input_ids,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )
    replay, replay_cache, replay_key = adapted.forward_arrays(
        adapted.parameters(),
        input_ids,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )
    mx.eval(first, replay, first_key, replay_key)
    assert first_cache is replay_cache is None
    assert bool(mx.array_equal(first, replay).item())
    assert bool(mx.array_equal(first_key, replay_key).item())
    assert not bool(mx.array_equal(first_key, key).item())


def test_inference_and_zero_dropout_consume_no_adapter_key(
    tiny_model,
    tiny_lora_config,
) -> None:
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(dropout=0.0),
        key=mx.random.key(4),
    )
    key = mx.random.key(23)
    input_ids = mx.array([[1, 2]], dtype=mx.int32)
    _, _, returned = adapted.forward_arrays(
        adapted.parameters(),
        input_ids,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )
    _, _, inference_returned = adapted.forward_arrays(
        adapted.parameters(),
        input_ids,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=False,
        key=key,
    )
    mx.eval(returned, inference_returned, key)
    assert bool(mx.array_equal(returned, key).item())
    assert bool(mx.array_equal(inference_returned, key).item())


def test_pure_array_lora_and_hidden_dropout_use_canonical_key_order(
    monkeypatch,
    tiny_lora_config,
) -> None:
    config = replace(tiny_model_config(), hidden_dropout=0.18)
    model = SMLLanguageModel(config, key=mx.random.key(3))
    adapted = apply_lora(
        model,
        tiny_lora_config(
            dropout=0.1,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        key=mx.random.key(4),
    )
    dropout_by_module = {
        "q_proj": 0.11,
        "k_proj": 0.12,
        "v_proj": 0.13,
        "o_proj": 0.14,
        "gate_proj": 0.15,
        "up_proj": 0.16,
        "down_proj": 0.17,
    }
    original_policy = adapted.lora_forward_policy
    assert original_policy is not None
    adapted.lora_forward_policy = LoRAForwardPolicy(
        tuple(
            LoRAAdapterSpec(
                module_path=spec.module_path,
                scale=spec.scale,
                dropout=dropout_by_module[spec.module_path.rsplit(".", 1)[-1]],
            )
            for spec in original_policy.adapters
        )
    )
    observed_probabilities: list[float] = []
    real_keyed_dropout = layers_module.keyed_dropout

    def record_keyed_dropout(x, probability, key):
        observed_probabilities.append(probability)
        return real_keyed_dropout(x, probability, key)

    monkeypatch.setattr(layers_module, "keyed_dropout", record_keyed_dropout)
    key = mx.random.key(29)
    _logits, _cache, returned_key = adapted.forward_arrays(
        adapted.parameters(),
        mx.array([[1, 2, 3]], dtype=mx.int32),
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )

    per_layer = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
    assert observed_probabilities == per_layer * config.num_layers
    expected_key = key
    for _dropout_site in observed_probabilities:
        expected_key = mx.random.split(expected_key)[0]
    mx.eval(returned_key, expected_key)
    assert bool(mx.array_equal(returned_key, expected_key).item())


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
    source = inspect.getsource(layers_module)

    assert "keyed_dropout" in source
    assert "nn.Dropout" not in source
