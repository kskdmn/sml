from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import sml.training.pretrain as pretrain_module
from mlx.utils import tree_flatten, tree_map
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.training.common import (
    BaseParameterState,
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerState,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
)
from sml.training.pretrain import build_pretraining_kernels


def assert_tree_close(
    actual: object, expected: object, *, atol: float, rtol: float
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
    model: SMLLanguageModel
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
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=None,
            warmup_steps=0,
            gradient_clip_norm=1.0,
        ),
    )
    model = SMLLanguageModel(model_config, key=mx.random.key(9))
    parameters = initialize_base_parameter_state(model.parameters())
    trainer = TrainerState(
        accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=mx.random.key(11),
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )
    weight_decay_tree = build_weight_decay_tree(
        parameters.working_parameters,
        config.optimizer.weight_decay,
    )
    return TinyRuntime(
        config=config,
        model=model,
        parameters=parameters,
        trainer=trainer,
        optimizer=initialize_adam_state(parameters.master_parameters),
        kernels=build_pretraining_kernels(model, config, weight_decay_tree),
        weight_decay_tree=weight_decay_tree,
        rows=np.arange(10, dtype=np.int32).reshape(2, 5) % model_config.vocab_size,
    )


def test_retention_handoff_requires_run_and_checkpoint_identity():
    published = SimpleNamespace(
        step=7,
        run=SimpleNamespace(identity="run-a"),
        checkpoint=SimpleNamespace(identity="checkpoint-a"),
    )
    substituted = SimpleNamespace(
        step=7,
        run=SimpleNamespace(identity="run-b"),
        checkpoint=SimpleNamespace(identity="checkpoint-b"),
    )

    with pytest.raises(SMLArtifactError, match="identity"):
        pretrain_module._require_retained_publication(published, substituted)


@pytest.fixture
def tiny_runtime(tmp_path: Path) -> TinyRuntime:
    return build_tiny_runtime(tmp_path)


def test_microstep_transfers_rows_once_and_keeps_state_on_device(tiny_runtime):
    """A host sync in the hot path would serialize every microbatch."""
    state = tiny_runtime.microstep(
        tiny_runtime.parameters, tiny_runtime.trainer, tiny_runtime.rows
    )

    assert state.trainer.accumulation_count.dtype == mx.int32
    assert state.trainer.loss_numerator.dtype == mx.float32
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


def _reference_partial_window_update(runtime: TinyRuntime, trainer_tree: tuple):
    accumulators, accumulation_count, next_key, loss_numerator = trainer_tree
    count = accumulation_count.astype(mx.float32)
    normalized = tree_map(lambda value: value.astype(mx.float32) / count, accumulators)
    squared_norm = sum(
        (mx.sum(mx.square(value)) for _, value in tree_flatten(normalized)),
        mx.array(0.0, dtype=mx.float32),
    )
    global_norm = mx.sqrt(squared_norm)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(runtime.config.optimizer.gradient_clip_norm, dtype=mx.float32)
        / mx.maximum(global_norm, mx.array(1e-12, dtype=mx.float32)),
    )
    gradients = tree_map(lambda value: value * scale, normalized)
    beta1 = runtime.config.optimizer.beta1
    beta2 = runtime.config.optimizer.beta2
    first_moments = tree_map(
        lambda moment, gradient: beta1 * moment + (1.0 - beta1) * gradient,
        runtime.optimizer.first_moments,
        gradients,
    )
    second_moments = tree_map(
        lambda moment, gradient: beta2 * moment + (1.0 - beta2) * mx.square(gradient),
        runtime.optimizer.second_moments,
        gradients,
    )
    learning_rate = mx.array(runtime.config.optimizer.learning_rate, dtype=mx.float32)
    masters = tree_map(
        lambda master, first, second, decay: (
            master
            - learning_rate
            * (
                first / (mx.sqrt(second) + runtime.config.optimizer.epsilon)
                + float(decay) * master
            )
        ),
        runtime.parameters.master_parameters,
        first_moments,
        second_moments,
        runtime.weight_decay_tree,
    )
    working = tree_map(lambda master: master.astype(mx.bfloat16), masters)
    return (
        masters,
        working,
        (
            runtime.optimizer.step + mx.array(1, dtype=mx.int32),
            first_moments,
            second_moments,
        ),
        (
            tree_map(lambda value: value - value, accumulators),
            accumulation_count - accumulation_count,
            next_key,
            loss_numerator - loss_numerator,
        ),
        {
            "learning_rate": learning_rate,
            "loss_numerator": loss_numerator,
            "loss": loss_numerator / count,
            "accumulation_count": accumulation_count,
        },
    )


def test_optimizer_step_resets_partial_window_using_actual_microbatch_count(
    tiny_runtime,
):
    """Dividing by configured accumulation would under-scale an epoch-tail update."""
    rows = mx.array(tiny_runtime.rows)
    logits, cache_state, _next_key = tiny_runtime.model.forward_arrays(
        tiny_runtime.parameters.working_parameters,
        rows[:, :-1],
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=tiny_runtime.trainer.next_key,
    )
    assert cache_state is None
    single_loss_numerator = causal_lm_loss(
        logits,
        rows[:, 1:],
        rows[:, 1:] != tiny_runtime.model.config.pad_token_id,
    )
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
    expected = _reference_partial_window_update(tiny_runtime, eager_trainer_tree)
    (
        expected_masters,
        expected_working,
        expected_adam,
        expected_trainer,
        expected_metrics,
    ) = expected

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
    assert_tree_close(compiled.optimizer.to_tree(), expected_adam, atol=1e-6, rtol=1e-6)
    assert_tree_close(compiled.trainer.to_tree(), expected_trainer, atol=0.0, rtol=0.0)
    assert_tree_close(compiled.metrics, expected_metrics, atol=1e-6, rtol=1e-6)
    assert_tree_close(
        {"loss_numerator": compiled.metrics["loss_numerator"]},
        {"loss_numerator": 3.0 * single_loss_numerator},
        atol=1e-6,
        rtol=1e-6,
    )
    assert_tree_close(
        {"loss": compiled.metrics["loss"]},
        {"loss": single_loss_numerator},
        atol=1e-6,
        rtol=1e-6,
    )


def test_consecutive_eager_and_compiled_transitions_match_every_state_tree(
    tmp_path: Path,
):
    """Returning stale state would only become visible on the second transition."""
    runtime = build_tiny_runtime(tmp_path, dropout=0.2)
    rows = mx.array(runtime.rows)

    def run(microstep_core, optimizer_step_core):
        masters = runtime.parameters.master_parameters
        working = runtime.parameters.working_parameters
        adam_tree = runtime.optimizer.to_tree()
        trainer_tree = runtime.trainer.to_tree()
        for _ in range(2):
            working, trainer_tree = microstep_core(
                working, trainer_tree, rows[:, :-1], rows[:, 1:]
            )
            masters, working, adam_tree, trainer_tree, metrics = optimizer_step_core(
                masters, working, adam_tree, trainer_tree
            )
        return masters, working, adam_tree, trainer_tree, metrics

    eager = run(
        runtime.kernels.eager_microstep_core,
        runtime.kernels.eager_optimizer_step_core,
    )
    compiled = run(
        runtime.kernels.compiled_microstep_core,
        runtime.kernels.compiled_optimizer_step_core,
    )

    for actual, expected in zip(compiled, eager, strict=True):
        assert_tree_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert int(compiled[2][0].item()) == 2
    assert not bool(mx.array_equal(compiled[3][2], runtime.trainer.next_key))


def test_optimizer_core_uses_no_adam_wrapper_in_traced_body(tiny_runtime):
    """Custom wrapper construction in a traced core would violate its array boundary."""
    source = inspect.getsource(tiny_runtime.kernels.eager_optimizer_step_core)

    assert "AdamState" not in source


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


def test_resume_overrides_and_checkpoint_policy_expose_only_reviewed_controls():
    """A retention override would reintroduce unsupported checkpoint history."""
    from sml.training import common as common_module

    ResumeOverrides = common_module.ResumeOverrides
    assert tuple(ResumeOverrides.__dataclass_fields__) == (
        "maximum_steps",
        "maximum_epochs",
        "log_interval",
        "checkpoint_interval",
    )
    assert ResumeOverrides() == ResumeOverrides(None, None, None, None)
    assert tuple(CheckpointPolicy.__dataclass_fields__) == ("interval",)
    assert tuple(pretrain_module.TrainingResult.__dataclass_fields__) == (
        "run",
        "step",
        "epoch",
        "rows",
    )
