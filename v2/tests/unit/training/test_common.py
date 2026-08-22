from __future__ import annotations

import inspect
import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig
from sml.training import common as common_module
from sml.training.common import (
    AdamState,
    BaseParameterState,
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PrecisionConfig,
    PretrainingConfig,
    TrainerState,
    WeightDecayPolicy,
    accumulate_fp32,
    adamw_mixed_precision_update,
    adamw_mixed_precision_update_tree,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    learning_rate_at,
    normalize_and_clip,
    resolved_warmup_steps,
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


def optimizer_config(**overrides: object) -> OptimizerConfig:
    return OptimizerConfig(schedule_steps=100, warmup_steps=0, **overrides)


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def assert_tree_close(actual: dict, expected: dict, *, atol: float, rtol: float):
    actual_leaves = dict(tree_flatten(actual))
    expected_leaves = dict(tree_flatten(expected))
    assert set(actual_leaves) == set(expected_leaves)
    for name, actual_leaf in actual_leaves.items():
        assert_close(
            actual_leaf,
            expected_leaves[name],
            atol=atol,
            rtol=rtol,
        )


def numpy_schedule_oracle(steps: list[int], config: OptimizerConfig) -> mx.array:
    warmup_steps = resolved_warmup_steps(config)
    values = []
    for step in steps:
        warmup = (step + 1) / max(1, warmup_steps)
        if step < warmup_steps:
            values.append(config.learning_rate * warmup)
        elif config.schedule_steps is None:
            values.append(config.learning_rate)
        else:
            decay_steps = config.schedule_steps - warmup_steps
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = (
                config.minimum_learning_rate_ratio
                + (1.0 - config.minimum_learning_rate_ratio) * cosine
            )
            values.append(config.learning_rate * ratio)
    return mx.array(values, dtype=mx.float32)


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
    assert config.optimizer.warmup_steps is None
    assert resolved_warmup_steps(config.optimizer) == 2_680
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


def test_learning_rate_schedule_matches_scalar_oracle_without_host_step_conversion():
    """A host conversion would make the compiled schedule synchronize per update."""
    config = OptimizerConfig(
        schedule_steps=100,
        warmup_steps=10,
        minimum_learning_rate_ratio=0.1,
    )
    steps = mx.array([0, 9, 10, 55, 100], dtype=mx.int32)

    actual = mx.vmap(lambda step: learning_rate_at(step, config))(steps)
    expected = numpy_schedule_oracle(steps.tolist(), config)

    assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    source = inspect.getsource(common_module)
    assert "int(step)" not in source
    assert ".item(" not in source
    assert "mx.eval(" not in source


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


def test_accumulation_normalization_and_clipping_use_fp32_once():
    """Repeated division or BF16 accumulation would perturb the clipped update."""
    accumulators = {"weight": mx.array([1.0, -2.0], dtype=mx.float32)}
    gradients = {"weight": mx.array([3.0, -2.0], dtype=mx.bfloat16)}

    accumulated = accumulate_fp32(accumulators, gradients)
    normalized = normalize_and_clip(
        accumulated,
        mx.array(2, dtype=mx.int32),
        gradient_clip_norm=1.0,
    )

    assert accumulated["weight"].dtype == mx.float32
    assert normalized["weight"].dtype == mx.float32
    assert_close(
        normalized["weight"],
        mx.array([2.0, -2.0], dtype=mx.float32) / math.sqrt(8.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_compiled_invalid_normalization_count_fails_closed_with_zero_gradients():
    """A traced empty accumulation must not emit divide-by-zero values."""

    @mx.compile
    def normalize(gradients, count):
        return normalize_and_clip(gradients, count, gradient_clip_norm=1.0)

    normalized = normalize(
        {"weight": mx.array([2.0, -4.0], dtype=mx.float32)},
        mx.array(0, dtype=mx.int32),
    )

    assert normalized["weight"].dtype == mx.float32
    assert_close(
        normalized["weight"],
        mx.zeros((2,), dtype=mx.float32),
        atol=0.0,
        rtol=0.0,
    )


def test_adam_keeps_fp32_masters_bf16_working_parameters_and_fp32_moments():
    """Casting the authoritative state to BF16 would lose optimizer precision."""
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0, -2.0], dtype=mx.bfloat16)},
    )
    gradients = {"weight": mx.array([0.25, -0.5], dtype=mx.bfloat16)}
    state = initialize_adam_state(parameter_state.master_parameters)

    masters, working, state = adamw_mixed_precision_update(
        parameter_state.master_parameters,
        gradients,
        state,
        optimizer_config(weight_decay=WeightDecayPolicy(other=0.1)),
        {"weight": True},
    )

    mx.eval(masters, working, state.to_tree())
    assert masters["weight"].dtype == mx.float32
    assert working["weight"].dtype == mx.bfloat16
    assert mx.array_equal(
        working["weight"], masters["weight"].astype(mx.bfloat16)
    ).item()
    assert state.first_moments["weight"].dtype == mx.float32
    assert state.second_moments["weight"].dtype == mx.float32
    assert state.step.dtype == mx.int32


def test_sub_bf16_ulp_update_survives_in_master_state():
    """A sub-BF16-ULP update must remain in the FP32 master for the next step."""
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0], dtype=mx.bfloat16)},
    )

    masters, working, _state = adamw_mixed_precision_update(
        parameter_state.master_parameters,
        {"weight": mx.array([1.0], dtype=mx.bfloat16)},
        initialize_adam_state(parameter_state.master_parameters),
        optimizer_config(learning_rate=1e-4, beta1=0.0, beta2=0.0, epsilon=1e-8),
        {"weight": False},
    )

    mx.eval(masters, working)
    assert not mx.array_equal(
        masters["weight"], parameter_state.master_parameters["weight"]
    ).item()
    assert mx.array_equal(
        working["weight"], parameter_state.working_parameters["weight"]
    ).item()


def test_state_wrappers_require_exact_tree_keys_shapes_and_dtypes():
    """Silently coercing checkpoint trees would corrupt a resumed optimizer state."""
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0], dtype=mx.bfloat16)},
    )
    optimizer_state = initialize_adam_state(parameter_state.master_parameters)
    trainer_state = TrainerState(
        accumulators={"weight": mx.zeros((1,), dtype=mx.float32)},
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=mx.random.key(4),
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )

    assert (
        BaseParameterState.from_tree(parameter_state.to_tree()).to_tree()
        == parameter_state.to_tree()
    )
    assert (
        AdamState.from_tree(optimizer_state.to_tree()).to_tree()
        == optimizer_state.to_tree()
    )
    assert (
        TrainerState.from_tree(trainer_state.to_tree()).to_tree()
        == trainer_state.to_tree()
    )
    with pytest.raises(SMLConfigurationError, match="bfloat16"):
        BaseParameterState.from_tree(
            (parameter_state.master_parameters, parameter_state.master_parameters)
        )
    with pytest.raises(SMLConfigurationError, match="int32"):
        AdamState.from_tree(
            (
                mx.array(0, dtype=mx.int64),
                optimizer_state.first_moments,
                optimizer_state.second_moments,
            )
        )
    with pytest.raises(SMLConfigurationError, match="float32"):
        TrainerState.from_tree(
            (
                {"different": mx.zeros((1,), dtype=mx.bfloat16)},
                trainer_state.accumulation_count,
                trainer_state.next_key,
                trainer_state.loss_numerator,
            )
        )


def test_trainer_state_requires_fp32_scalar_loss_numerator():
    """A non-FP32 loss numerator would make device-side metric reduction drift."""
    accumulators = {"weight": mx.zeros((1,), dtype=mx.float32)}

    state = TrainerState(
        accumulators=accumulators,
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=mx.random.key(4),
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )

    assert state.to_tree()[3].dtype == mx.float32
    assert state.to_tree()[3].shape == ()
    with pytest.raises(SMLConfigurationError, match="loss_numerator"):
        TrainerState(
            accumulators,
            mx.array(0, dtype=mx.int32),
            mx.random.key(4),
            mx.array(0.0, dtype=mx.bfloat16),
        )


def test_tree_native_adam_update_carries_array_state_through_compilation():
    """A custom Adam wrapper in the traced body would break a pure array boundary."""
    masters = {"weight": mx.array([1.0], dtype=mx.float32)}
    gradients = {"weight": mx.array([1.0], dtype=mx.bfloat16)}
    config = OptimizerConfig(
        schedule_steps=None,
        warmup_steps=0,
        learning_rate=0.1,
        beta1=0.0,
        beta2=0.0,
    )

    @mx.compile
    def update(parameters, adam_tree):
        return adamw_mixed_precision_update_tree(
            parameters, gradients, adam_tree, config, {"weight": False}
        )

    first_masters, _first_working, first_adam = update(
        masters, initialize_adam_state(masters).to_tree()
    )
    second_masters, _second_working, second_adam = update(first_masters, first_adam)

    mx.eval(second_masters, second_adam)
    assert int(second_adam[0].item()) == 2
    assert_close(
        second_masters["weight"],
        mx.array([0.8], dtype=mx.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def _run_two_updates(*, compiled: bool):
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0, -2.0], dtype=mx.bfloat16)},
    )
    optimizer_state = initialize_adam_state(parameter_state.master_parameters)
    gradients = {"weight": mx.array([0.25, -0.5], dtype=mx.bfloat16)}
    config = optimizer_config(weight_decay=WeightDecayPolicy(other=0.0))
    weight_decay_tree = {"weight": False}

    def update(master_parameters, state_tree):
        state = AdamState.from_tree(state_tree)
        masters, working, next_state = adamw_mixed_precision_update(
            master_parameters,
            gradients,
            state,
            config,
            weight_decay_tree,
        )
        return masters, working, next_state.to_tree()

    update_fn = mx.compile(update) if compiled else update
    masters, working, state_tree = update_fn(
        parameter_state.master_parameters, optimizer_state.to_tree()
    )
    masters, working, state_tree = update_fn(masters, state_tree)
    return BaseParameterState(masters, working), AdamState.from_tree(state_tree)


def test_compiled_second_update_observes_first_state():
    """Dropping returned compiled state would replay the first optimizer update."""
    eager_parameters, eager_optimizer = _run_two_updates(compiled=False)
    compiled_parameters, compiled_optimizer = _run_two_updates(compiled=True)

    assert_tree_close(
        compiled_parameters.master_parameters,
        eager_parameters.master_parameters,
        atol=1e-6,
        rtol=1e-6,
    )
    assert_tree_close(
        compiled_parameters.working_parameters,
        eager_parameters.working_parameters,
        atol=0.0,
        rtol=0.0,
    )
    assert_tree_close(
        compiled_optimizer.first_moments,
        eager_optimizer.first_moments,
        atol=1e-6,
        rtol=1e-6,
    )
    assert int(compiled_optimizer.step.item()) == 2


def adamw_scalar_oracle(
    parameter: float,
    gradients: list[float],
    config: OptimizerConfig,
    decay: float,
) -> float:
    first_moment = 0.0
    second_moment = 0.0
    for step, gradient in enumerate(gradients, start=1):
        first_moment = config.beta1 * first_moment + (1.0 - config.beta1) * gradient
        second_moment = (
            config.beta2 * second_moment + (1.0 - config.beta2) * gradient * gradient
        )
        if config.bias_correction:
            first_for_update = first_moment / (1.0 - config.beta1**step)
            second_for_update = second_moment / (1.0 - config.beta2**step)
        else:
            first_for_update = first_moment
            second_for_update = second_moment
        parameter -= config.learning_rate * (
            first_for_update / (math.sqrt(second_for_update) + config.epsilon)
            + decay * parameter
        )
    return parameter


@pytest.mark.parametrize("bias_correction", [False, True])
def test_adamw_matches_independent_two_step_oracle_with_epsilon_and_decay(
    bias_correction,
):
    """Moving epsilon, decay, or bias correction must change this two-step result."""
    config = OptimizerConfig(
        learning_rate=0.03,
        beta1=0.8,
        beta2=0.6,
        epsilon=0.2,
        bias_correction=bias_correction,
        schedule_steps=None,
        warmup_steps=0,
    )
    masters = {"weight": mx.array([1.25], dtype=mx.float32)}
    state = initialize_adam_state(masters)
    for gradient in (0.75, -0.5):
        masters, _working, state = adamw_mixed_precision_update(
            masters,
            {"weight": mx.array([gradient], dtype=mx.bfloat16)},
            state,
            config,
            {"weight": 0.15},
        )

    assert_close(
        masters["weight"],
        mx.array(
            [adamw_scalar_oracle(1.25, [0.75, -0.5], config, 0.15)],
            dtype=mx.float32,
        ),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("bias_correction", [False, True])
def test_adamw_matches_independent_one_step_oracle_with_epsilon_and_decay(
    bias_correction,
):
    """The first update must use the saved beta, epsilon, bias, and decay rules."""
    config = OptimizerConfig(
        learning_rate=0.03,
        beta1=0.8,
        beta2=0.6,
        epsilon=0.2,
        bias_correction=bias_correction,
        schedule_steps=None,
        warmup_steps=0,
    )
    gradient = 0.75
    first_moment = (1.0 - config.beta1) * gradient
    second_moment = (1.0 - config.beta2) * gradient * gradient
    if bias_correction:
        first_moment /= 1.0 - config.beta1
        second_moment /= 1.0 - config.beta2
    expected = 1.25 - config.learning_rate * (
        first_moment / (math.sqrt(second_moment) + config.epsilon) + 0.15 * 1.25
    )

    masters, _working, state = adamw_mixed_precision_update(
        {"weight": mx.array([1.25], dtype=mx.float32)},
        {"weight": mx.array([gradient], dtype=mx.bfloat16)},
        initialize_adam_state({"weight": mx.array([1.25], dtype=mx.float32)}),
        config,
        {"weight": 0.15},
    )

    assert_close(
        masters["weight"],
        mx.array([expected], dtype=mx.float32),
        atol=2e-6,
        rtol=2e-6,
    )
    assert int(state.step.item()) == 1


def test_fp32_normalized_gradients_flow_into_adam_update():
    """Rejecting normalized FP32 gradients breaks the accumulation-to-Adam boundary."""
    masters = {"weight": mx.array([1.0], dtype=mx.float32)}
    normalized = normalize_and_clip(
        accumulate_fp32(
            {"weight": mx.zeros((1,), dtype=mx.float32)},
            {"weight": mx.array([0.5], dtype=mx.bfloat16)},
        ),
        mx.array(1, dtype=mx.int32),
        gradient_clip_norm=1.0,
    )

    updated, _working, _state = adamw_mixed_precision_update(
        masters,
        normalized,
        initialize_adam_state(masters),
        optimizer_config(beta1=0.0, beta2=0.0, learning_rate=0.1),
        {"weight": False},
    )

    assert_close(
        updated["weight"],
        mx.array([0.9], dtype=mx.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_counter_boundaries_reject_invalid_loaded_values_and_overflow():
    """Invalid counters must fail instead of altering a resumed update position."""
    with pytest.raises(SMLConfigurationError, match="gradient_accumulation_steps"):
        LoaderConfig(gradient_accumulation_steps=2**31)
    with pytest.raises(SMLConfigurationError, match="Adam step"):
        AdamState(
            mx.array(-1, dtype=mx.int32),
            {"weight": mx.zeros((1,), dtype=mx.float32)},
            {"weight": mx.zeros((1,), dtype=mx.float32)},
        )
    with pytest.raises(SMLConfigurationError, match="accumulation_count"):
        TrainerState(
            {"weight": mx.zeros((1,), dtype=mx.float32)},
            mx.array(-1, dtype=mx.int32),
            mx.random.key(7),
            mx.array(0.0, dtype=mx.float32),
        )
    with pytest.raises(SMLConfigurationError, match="normalization_count"):
        normalize_and_clip(
            {"weight": mx.ones((1,), dtype=mx.float32)},
            mx.array(0, dtype=mx.int32),
            gradient_clip_norm=1.0,
        )

    masters = {"weight": mx.array([1.0], dtype=mx.float32)}
    with pytest.raises(SMLConfigurationError, match="Adam step"):
        adamw_mixed_precision_update(
            masters,
            {"weight": mx.ones((1,), dtype=mx.bfloat16)},
            AdamState(
                mx.array(2**31 - 1, dtype=mx.int32),
                {"weight": mx.zeros((1,), dtype=mx.float32)},
                {"weight": mx.zeros((1,), dtype=mx.float32)},
            ),
            optimizer_config(),
            {"weight": False},
        )


def test_compiled_maximum_adam_step_preserves_every_optimizer_leaf():
    """A traced counter boundary must not mutate moments while retaining its step."""
    masters = {"weight": mx.array([1.0], dtype=mx.float32)}
    state = AdamState(
        mx.array(2**31 - 1, dtype=mx.int32),
        {"weight": mx.zeros((1,), dtype=mx.float32)},
        {"weight": mx.zeros((1,), dtype=mx.float32)},
    )

    @mx.compile
    def update(parameters, state_tree):
        next_masters, _working, next_state = adamw_mixed_precision_update_tree(
            parameters,
            {"weight": mx.ones((1,), dtype=mx.bfloat16)},
            state_tree,
            optimizer_config(),
            {"weight": False},
        )
        return next_masters, next_state

    next_masters, next_state_tree = update(masters, state.to_tree())

    assert_tree_close(next_masters, masters, atol=0.0, rtol=0.0)
    assert_tree_close(next_state_tree[1], state.first_moments, atol=0.0, rtol=0.0)
    assert_tree_close(next_state_tree[2], state.second_moments, atol=0.0, rtol=0.0)
    assert int(next_state_tree[0].item()) == 2**31 - 1


def test_state_tree_validation_preserves_nested_containers_keys_shapes_and_random_key():
    """Flattened paths must not make malformed checkpoint trees appear equivalent."""
    master = {
        "weight": mx.ones((2,), dtype=mx.float32),
        "nested": {},
    }
    working = {
        "weight": mx.ones((2,), dtype=mx.bfloat16),
        "nested": [],
    }
    with pytest.raises(SMLConfigurationError, match="structure|keys"):
        BaseParameterState(master, working)
    with pytest.raises(SMLConfigurationError, match="structure|keys"):
        BaseParameterState(
            {"a.b": mx.ones((1,), dtype=mx.float32)},
            {"a": {"b": mx.ones((1,), dtype=mx.bfloat16)}},
        )
    with pytest.raises(SMLConfigurationError, match="keys"):
        BaseParameterState(
            {"weight": mx.ones((1,), dtype=mx.float32)},
            {"other": mx.ones((1,), dtype=mx.bfloat16)},
        )
    with pytest.raises(SMLConfigurationError, match="shapes"):
        BaseParameterState(
            {"weight": mx.ones((2,), dtype=mx.float32)},
            {"weight": mx.ones((3,), dtype=mx.bfloat16)},
        )
    with pytest.raises(SMLConfigurationError, match="top-level dict"):
        initialize_base_parameter_state([mx.ones((1,), dtype=mx.bfloat16)])
    with pytest.raises(SMLConfigurationError, match="next_key"):
        TrainerState(
            {"weight": mx.zeros((1,), dtype=mx.float32)},
            mx.array(0, dtype=mx.int32),
            mx.zeros((1,), dtype=mx.uint32),
            mx.array(0.0, dtype=mx.float32),
        )

    tied = dict(
        tree_flatten(
            build_weight_decay_tree(
                {"embed_tokens": {"weight": mx.ones((1,), dtype=mx.bfloat16)}},
                WeightDecayPolicy(embed_tokens=0.11, lm_head=0.22),
            )
        )
    )
    assert tied == {"embed_tokens.weight": 0.11}
