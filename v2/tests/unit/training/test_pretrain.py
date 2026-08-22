from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import sml.training.pretrain as pretrain_module
from mlx.utils import tree_flatten, tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    BaseParameterState,
    LoaderConfig,
    PretrainingConfig,
    TrainerState,
    adamw_mixed_precision_update,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    normalize_and_clip,
)
from sml.training.pretrain import build_pretraining_kernels


def assert_tree_close(
    actual: dict, expected: dict, *, atol: float, rtol: float
) -> None:
    actual_leaves = dict(tree_flatten(actual))
    expected_leaves = dict(tree_flatten(expected))
    assert actual_leaves.keys() == expected_leaves.keys()
    mx.eval(*actual_leaves.values(), *expected_leaves.values())
    for path, actual_leaf in actual_leaves.items():
        assert bool(
            mx.allclose(actual_leaf, expected_leaves[path], atol=atol, rtol=rtol).item()
        ), path


def all_builtin_array_tree_leaves(*trees: object) -> bool:
    def check(tree: object) -> bool:
        if isinstance(tree, mx.array):
            return True
        if isinstance(tree, dict):
            return all(check(value) for value in tree.values())
        if isinstance(tree, (list, tuple)):
            return all(check(value) for value in tree)
        return False

    return all(check(tree) for tree in trees)


def source_has_none_of(module: object, forbidden: list[str]) -> bool:
    source = inspect.getsource(module)
    return all(token not in source for token in forbidden)


@dataclass(frozen=True)
class TinyRuntime:
    config: PretrainingConfig
    parameters: BaseParameterState
    trainer: TrainerState
    optimizer: object
    kernels: object
    weight_decay_tree: dict
    rows: np.ndarray

    def microstep(self, parameters: BaseParameterState, trainer: TrainerState, rows):
        return self.kernels.microstep(parameters, trainer, rows)


def build_tiny_runtime(tmp_path: Path, *, dropout: float = 0.0) -> TinyRuntime:
    model_config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=4,
        rope_scaling_factor=1.0,
        hidden_dropout=dropout,
    )
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=model_config,
        loader=LoaderConfig(gradient_accumulation_steps=4),
    )
    model = SMLLanguageModel(model_config, key=mx.random.key(9))
    parameters = initialize_base_parameter_state(model.parameters())
    trainer = TrainerState(
        accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=mx.random.key(11),
    )
    weight_decay_tree = build_weight_decay_tree(
        parameters.working_parameters,
        config.optimizer.weight_decay,
    )
    return TinyRuntime(
        config=config,
        parameters=parameters,
        trainer=trainer,
        optimizer=initialize_adam_state(parameters.master_parameters),
        kernels=build_pretraining_kernels(model, config, weight_decay_tree),
        weight_decay_tree=weight_decay_tree,
        rows=np.arange(10, dtype=np.int32).reshape(2, 5) % model_config.vocab_size,
    )


@pytest.fixture
def tiny_runtime(tmp_path: Path) -> TinyRuntime:
    return build_tiny_runtime(tmp_path)


def test_microstep_transfers_rows_once_and_keeps_state_on_device(tiny_runtime):
    """A host sync in the hot path would serialize every microbatch."""
    state = tiny_runtime.microstep(
        tiny_runtime.parameters, tiny_runtime.trainer, tiny_runtime.rows
    )

    assert state.trainer.accumulation_count.dtype == mx.int32
    assert state.trainer.accumulators["embed_tokens"]["weight"].dtype == mx.float32
    assert source_has_none_of(
        pretrain_module, ["loss.item(", ".tolist(", "mx.eval(loss"]
    )
    assert (
        inspect.getsource(tiny_runtime.kernels.microstep).count("mx.array(rows)") == 1
    )
    assert "mx.eval(" not in inspect.getsource(tiny_runtime.kernels.microstep)


def test_compiled_cores_use_only_builtin_array_trees(tiny_runtime):
    """A wrapper object at a compile boundary would capture mutable host state."""
    rows = mx.array(tiny_runtime.rows)
    working_parameters, trainer_tree = tiny_runtime.kernels.compiled_microstep_core(
        tiny_runtime.parameters.working_parameters,
        tiny_runtime.trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )

    assert isinstance(working_parameters, dict)
    assert isinstance(trainer_tree, tuple)
    assert all_builtin_array_tree_leaves(working_parameters, trainer_tree)
    assert source_has_none_of(
        pretrain_module,
        [
            "mx.compile(BaseParameterState",
            "mx.compile(AdamState",
            "mx.compile(TrainerState",
            "mx.compile(KVCache",
            "model.update(",
        ],
    )


def test_optimizer_step_resets_partial_window_using_actual_microbatch_count(
    tiny_runtime,
):
    """Dividing by configured accumulation would under-scale an epoch-tail update."""
    compiled_trainer = tiny_runtime.trainer
    eager_trainer_tree = tiny_runtime.trainer.to_tree()
    eager_working = tiny_runtime.parameters.working_parameters
    for _ in range(3):
        compiled_trainer = tiny_runtime.microstep(
            tiny_runtime.parameters, compiled_trainer, tiny_runtime.rows
        ).trainer
        eager_working, eager_trainer_tree = tiny_runtime.kernels.eager_microstep_core(
            eager_working,
            eager_trainer_tree,
            mx.array(tiny_runtime.rows)[:, :-1],
            mx.array(tiny_runtime.rows)[:, 1:],
        )

    compiled = tiny_runtime.kernels.optimizer_step(
        tiny_runtime.parameters, tiny_runtime.optimizer, compiled_trainer
    )
    expected_gradients = normalize_and_clip(
        eager_trainer_tree[0],
        eager_trainer_tree[1],
        gradient_clip_norm=tiny_runtime.config.optimizer.gradient_clip_norm,
    )
    expected_masters, expected_working, expected_adam = adamw_mixed_precision_update(
        tiny_runtime.parameters.master_parameters,
        expected_gradients,
        tiny_runtime.optimizer,
        tiny_runtime.config.optimizer,
        tiny_runtime.weight_decay_tree,
    )

    assert_tree_close(
        compiled.parameters.master_parameters,
        expected_masters,
        atol=1e-6,
        rtol=1e-6,
    )
    assert_tree_close(
        compiled.parameters.working_parameters,
        expected_working,
        atol=0.0,
        rtol=0.0,
    )
    assert int(compiled.trainer.accumulation_count.item()) == 0
    assert int(compiled.optimizer.step.item()) == 1
    assert_tree_close(
        compiled.optimizer.first_moments,
        expected_adam.first_moments,
        atol=1e-6,
        rtol=1e-6,
    )

    second_microstep = tiny_runtime.microstep(
        compiled.parameters, compiled.trainer, tiny_runtime.rows
    )
    second = tiny_runtime.kernels.optimizer_step(
        second_microstep.parameters,
        compiled.optimizer,
        second_microstep.trainer,
    )
    assert int(second.optimizer.step.item()) == 2
    assert_tree_close(
        second.parameters.working_parameters,
        tree_map(
            lambda master: master.astype(mx.bfloat16),
            second.parameters.master_parameters,
        ),
        atol=0.0,
        rtol=0.0,
    )


def test_disabled_dropout_preserves_explicit_prng_key(tiny_runtime):
    """Advancing a disabled dropout key would break deterministic resume state."""
    state = tiny_runtime.microstep(
        tiny_runtime.parameters, tiny_runtime.trainer, tiny_runtime.rows
    )

    mx.eval(state.trainer.next_key, tiny_runtime.trainer.next_key)
    assert bool(mx.array_equal(state.trainer.next_key, tiny_runtime.trainer.next_key))


def test_enabled_dropout_advances_explicit_prng_key(tmp_path: Path):
    """Reusing a dropout key would repeat masks after every resumed microstep."""
    runtime = build_tiny_runtime(tmp_path, dropout=0.2)

    state = runtime.microstep(runtime.parameters, runtime.trainer, runtime.rows)

    mx.eval(state.trainer.next_key, runtime.trainer.next_key)
    assert not bool(mx.array_equal(state.trainer.next_key, runtime.trainer.next_key))
