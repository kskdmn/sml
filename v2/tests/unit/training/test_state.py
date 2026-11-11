from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from sml.errors import SMLConfigurationError
from sml.training.common import (
    AdamState,
    BaseParameterState,
    LoaderConfig,
    TrainerState,
    WeightDecayPolicy,
    adamw_mixed_precision_update,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    normalize_and_clip,
)

from .helpers import (
    optimizer_config,
)


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


@pytest.mark.parametrize("valid", [True, False])
def test_master_cast_validation_matches_paths_independent_of_dict_order(valid):
    masters = {"a": mx.array([1.0]), "nested": {"b": mx.array([2.0])}}
    working = {
        "nested": {"b": mx.array([2.0 if valid else 1.0], dtype=mx.bfloat16)},
        "a": mx.array([1.0 if valid else 2.0], dtype=mx.bfloat16),
    }
    if valid:
        assert BaseParameterState(masters, working).working_parameters is working
    else:
        with pytest.raises(SMLConfigurationError, match="exact bfloat16 casts"):
            BaseParameterState(masters, working)


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
