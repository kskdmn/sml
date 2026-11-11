from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from sml.training.common import (
    AdamState,
    BaseParameterState,
    OptimizerConfig,
    WeightDecayPolicy,
    accumulate_fp32,
    adamw_mixed_precision_update,
    adamw_mixed_precision_update_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    normalize_and_clip,
)

from .helpers import (
    assert_close,
    numpy_schedule_oracle,
    optimizer_config,
)


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


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
def test_gradient_clipping_is_exactly_invariant_to_checkpoint_tree_order(compiled):
    """Large and small norm contributions must survive a checkpoint tree reorder."""
    fresh = {
        "z": {"weight": mx.array([10_000.0], dtype=mx.float32)},
        "a": {
            f"p{index}": mx.array([2.0], dtype=mx.float32)
            for index in reversed(range(16))
        },
    }
    restored = tree_unflatten(sorted(tree_flatten(fresh), key=lambda item: item[0]))

    def clip(gradients):
        return normalize_and_clip(
            gradients,
            mx.array(1, dtype=mx.int32),
            gradient_clip_norm=1.0,
        )

    clip_fn = mx.compile(clip) if compiled else clip
    fresh_clipped = dict(tree_flatten(clip_fn(fresh)))
    restored_clipped = dict(tree_flatten(clip_fn(restored)))
    mx.eval(fresh_clipped, restored_clipped)

    assert float(fresh_clipped["z.weight"].item()) < 1.0
    for path, gradient in fresh_clipped.items():
        assert bool(mx.array_equal(gradient, restored_clipped[path])), path


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
    gradients: tuple[float, ...],
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
def test_adamw_matches_scalar_oracle_after_each_update(
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
    gradients = (0.75, -0.5)
    for step, gradient in enumerate(gradients, start=1):
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
                [adamw_scalar_oracle(1.25, gradients[:step], config, 0.15)],
                dtype=mx.float32,
            ),
            atol=2e-6,
            rtol=2e-6,
        )
        assert int(state.step.item()) == step


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


def _numpy_leaf(value: object) -> np.ndarray:
    if isinstance(value, mx.array):
        return np.array(value.astype(mx.float32), dtype=np.float64)
    raise TypeError(f"oracle expected an MLX array leaf, got {type(value)!r}")


def _decay_rate(flag: object, config: OptimizerConfig) -> float:
    if flag is True:
        return float(config.weight_decay.other)
    if flag is False:
        return 0.0
    if isinstance(flag, (int, float)) and not isinstance(flag, bool):
        rate = float(flag)
        if math.isfinite(rate) and rate >= 0.0:
            return rate
    raise AssertionError(
        "oracle weight-decay leaves must be bool or non-negative floats"
    )


def _adamw_leaf(
    parameter: object,
    gradient: object,
    first: object,
    second: object,
    decay_flag: object,
    *,
    config: OptimizerConfig,
    learning_rate: float,
    completed: float,
) -> tuple[mx.array, mx.array, mx.array]:
    parameter_np = _numpy_leaf(parameter)
    gradient_np = _numpy_leaf(gradient)
    first_np = _numpy_leaf(first)
    second_np = _numpy_leaf(second)
    next_first_np = config.beta1 * first_np + (1.0 - config.beta1) * gradient_np
    next_second_np = config.beta2 * second_np + (1.0 - config.beta2) * np.square(
        gradient_np
    )
    first_for_update = next_first_np
    second_for_update = next_second_np
    if config.bias_correction:
        first_for_update = next_first_np / (1.0 - config.beta1**completed)
        second_for_update = next_second_np / (1.0 - config.beta2**completed)
    decay = _decay_rate(decay_flag, config)
    updated_np = parameter_np - learning_rate * (
        first_for_update / (np.sqrt(second_for_update) + config.epsilon)
        + decay * parameter_np
    )
    return (
        mx.array(updated_np.astype(np.float32), dtype=mx.float32),
        mx.array(next_first_np.astype(np.float32), dtype=mx.float32),
        mx.array(next_second_np.astype(np.float32), dtype=mx.float32),
    )


def _map_adamw_oracle(
    parameters: object,
    gradients: object,
    first_moments: object,
    second_moments: object,
    weight_decay_tree: object,
    *,
    config: OptimizerConfig,
    learning_rate: float,
    completed: float,
) -> tuple[object, object, object]:
    if isinstance(parameters, dict):
        updated = {}
        next_first = {}
        next_second = {}
        for key, parameter in parameters.items():
            updated[key], next_first[key], next_second[key] = _map_adamw_oracle(
                parameter,
                gradients[key],
                first_moments[key],
                second_moments[key],
                weight_decay_tree[key],
                config=config,
                learning_rate=learning_rate,
                completed=completed,
            )
        return updated, next_first, next_second
    return _adamw_leaf(
        parameters,
        gradients,
        first_moments,
        second_moments,
        weight_decay_tree,
        config=config,
        learning_rate=learning_rate,
        completed=completed,
    )


def direct_fp32_adamw_oracle(
    parameters: dict,
    gradients: dict,
    state: AdamState,
    config: OptimizerConfig,
    weight_decay_tree: dict,
) -> tuple[dict, AdamState]:
    """Independent NumPy AdamW using the saved config and boolean decay tree."""
    step = int(np.array(state.step.astype(mx.int32)))
    learning_rate = float(np.array(numpy_schedule_oracle([step], config)).reshape(()))
    completed = float(step + 1)
    updated, next_first, next_second = _map_adamw_oracle(
        parameters,
        gradients,
        state.first_moments,
        state.second_moments,
        weight_decay_tree,
        config=config,
        learning_rate=learning_rate,
        completed=completed,
    )
    return updated, AdamState(
        step=mx.array(step + 1, dtype=mx.int32),
        first_moments=next_first,
        second_moments=next_second,
    )


def test_fp32_adam_preserves_adapter_dtype_and_formula():
    from sml.training.common import adamw_fp32_update

    parameters = {"lora_a": mx.array([[1.0, -2.0]], dtype=mx.float32)}
    gradients = {"lora_a": mx.array([[0.25, -0.5]], dtype=mx.float32)}
    state = initialize_adam_state(parameters)
    config = optimizer_config(weight_decay=WeightDecayPolicy(other=0.1))
    weight_decay_tree = {"lora_a": True}
    updated, next_state = adamw_fp32_update(
        parameters,
        gradients,
        state,
        config,
        weight_decay_tree,
    )
    expected, expected_state = direct_fp32_adamw_oracle(
        parameters,
        gradients,
        state,
        config,
        weight_decay_tree,
    )
    mx.eval(updated, next_state.to_tree(), expected, expected_state.to_tree())
    assert updated["lora_a"].dtype == mx.float32
    assert next_state.first_moments["lora_a"].dtype == mx.float32
    assert next_state.second_moments["lora_a"].dtype == mx.float32
    assert_tree_close(updated, expected, atol=1e-6, rtol=1e-6)
    assert_tree_close(
        next_state.first_moments, expected_state.first_moments, atol=1e-6, rtol=1e-6
    )
    assert_tree_close(
        next_state.second_moments, expected_state.second_moments, atol=1e-6, rtol=1e-6
    )
