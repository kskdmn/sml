from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import sml.training.pretrain as pretrain_module
from mlx.utils import tree_flatten, tree_map
from sml.data.pretraining import PretrainingCursor
from sml.errors import SMLConfigurationError, SMLRuntimeError
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.training.common import (
    BaseParameterState,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerState,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
    resolved_warmup_steps,
)
from sml.training.pretrain import build_pretraining_kernels
from sml.training.random import counter_random_key


@pytest.mark.parametrize(
    (
        "row_count",
        "microbatch_size",
        "accumulation_steps",
        "maximum_steps",
        "maximum_epochs",
        "expected_steps",
    ),
    (
        (17, 3, 2, None, 3, 9),
        (17, 3, 2, 5, 3, 5),
        (17, 3, 2, 20, 3, 9),
        (17, 3, 2, 20, None, 20),
        (17, 3, 8, None, 1, 1),
        (40_000, 2, 4, None, 2, 10_000),
        (2**40, 1, 8, 100_000, 2**31 - 1, 100_000),
    ),
)
def test_automatic_schedule_matches_planned_optimizer_updates(
    tmp_path,
    row_count,
    microbatch_size,
    accumulation_steps,
    maximum_steps,
    maximum_epochs,
    expected_steps,
):
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=ModelConfig(),
        loader=LoaderConfig(
            microbatch_size=microbatch_size,
            gradient_accumulation_steps=accumulation_steps,
        ),
        maximum_steps=maximum_steps,
        maximum_epochs=maximum_epochs,
    )

    resolved = pretrain_module._resolved_fresh_config(config, row_count=row_count)

    assert config.optimizer.schedule_steps is None
    assert resolved.optimizer.schedule_steps == expected_steps
    assert resolved_warmup_steps(resolved.optimizer) == int(0.01 * expected_steps)


def test_explicit_schedule_and_warmup_remain_authoritative(tmp_path):
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=ModelConfig(),
        optimizer=OptimizerConfig(schedule_steps=1_000, warmup_steps=200),
        maximum_steps=2,
    )

    assert pretrain_module._resolved_fresh_config(config, row_count=8) is config


def test_automatic_schedule_rejects_explicit_warmup_longer_than_run(tmp_path):
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=ModelConfig(),
        optimizer=OptimizerConfig(warmup_steps=2),
    )

    with pytest.raises(SMLConfigurationError, match="warmup_steps.*schedule_steps"):
        pretrain_module._resolved_fresh_config(config, row_count=8)


def test_automatic_schedule_rejects_update_counter_overflow(tmp_path):
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=ModelConfig(),
    )

    with pytest.raises(SMLConfigurationError, match="schedule_steps.*int32"):
        pretrain_module._resolved_fresh_config(config, row_count=8 * 2**31)


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
        rows=np.arange(4, 14, dtype=np.int32).reshape(2, 5),
    )


@pytest.fixture
def tiny_runtime(tmp_path: Path) -> TinyRuntime:
    return build_tiny_runtime(tmp_path)


def test_microstep_preserves_parameters_and_accumulates_fp32_state(tiny_runtime):
    """Microbatches must retain working weights and accumulate with full precision."""
    state = tiny_runtime.microstep(
        tiny_runtime.parameters, tiny_runtime.trainer, tiny_runtime.rows
    )

    assert state.parameters is tiny_runtime.parameters
    assert state.trainer.accumulation_count.dtype == mx.int32
    assert state.trainer.loss_numerator.dtype == mx.float32
    assert state.trainer.accumulators["embed_tokens"]["weight"].dtype == mx.float32


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize(
    "failure",
    ("nan-loss", "infinite-gradient", "parameter-overflow", "epsilon-underflow"),
)
def test_optimizer_rejects_numerical_failure_without_changing_input_state(
    tiny_runtime, compiled, failure
):
    runtime = tiny_runtime
    config = replace(runtime.config, compile=compiled)
    trainer = runtime.microstep(
        runtime.parameters, runtime.trainer, runtime.rows
    ).trainer
    if failure == "nan-loss":
        trainer = TrainerState.from_compiled_tree(
            (
                trainer.accumulators,
                trainer.accumulation_count,
                trainer.next_key,
                mx.array(float("nan"), dtype=mx.float32),
            )
        )
    elif failure == "infinite-gradient":
        accumulators = tree_map(mx.array, trainer.accumulators)
        accumulators["norm"]["weight"] = mx.full_like(
            accumulators["norm"]["weight"], float("inf")
        )
        trainer = replace(trainer, accumulators=accumulators)
    elif failure == "parameter-overflow":
        config = replace(
            config,
            optimizer=replace(
                config.optimizer, learning_rate=3e38, beta1=0.9, beta2=0.999
            ),
        )
    else:
        config = replace(config, optimizer=replace(config.optimizer, epsilon=1e-100))
    kernels = build_pretraining_kernels(
        runtime.model, config, runtime.weight_decay_tree
    )
    before = dict(tree_flatten(runtime.parameters.working_parameters))
    with pytest.raises(SMLRuntimeError, match="nonfinite"):
        kernels.optimizer_step(runtime.parameters, runtime.optimizer, trainer)
    assert int(runtime.optimizer.step.item()) == 0
    assert all(
        value is before[name]
        for name, value in tree_flatten(runtime.parameters.working_parameters)
    )


@pytest.mark.parametrize("accumulation_steps", (1, 4))
def test_accumulation_submits_complete_microsteps_without_host_waits(
    tiny_runtime, monkeypatch, accumulation_steps
):
    """Multi-batch windows must run before the eventual optimizer boundary."""
    config = replace(
        tiny_runtime.config,
        loader=replace(
            tiny_runtime.config.loader,
            gradient_accumulation_steps=accumulation_steps,
        ),
    )
    kernels = build_pretraining_kernels(
        tiny_runtime.model, config, tiny_runtime.weight_decay_tree
    )
    submitted = []
    submit = mx.async_eval

    def record_submission(tree):
        submitted.append(dict(tree_flatten(tree)))
        submit(tree)

    def reject_host_wait(*_args):
        raise AssertionError("microstep blocked the host before the optimizer update")

    trainer = tiny_runtime.trainer
    with monkeypatch.context() as patch:
        patch.setattr(mx, "async_eval", record_submission)
        patch.setattr(mx, "eval", reject_host_wait)
        for index in range(accumulation_steps):
            result = kernels.microstep(
                tiny_runtime.parameters, trainer, tiny_runtime.rows
            )
            trainer = result.trainer
            assert result.parameters is tiny_runtime.parameters
            if accumulation_steps > 1:
                assert len(submitted) == index + 1
                returned = dict(tree_flatten(trainer.to_tree()))
                assert submitted[-1].keys() == returned.keys()
                assert all(
                    submitted[-1][name] is value for name, value in returned.items()
                )
            else:
                assert not submitted

    assert int(trainer.accumulation_count.item()) == accumulation_steps
    assert int(tiny_runtime.optimizer.step.item()) == 0
    assert float(trainer.loss_numerator.item()) > 0.0


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
            "finite": mx.array(True),
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


def test_disabled_dropout_preserves_explicit_prng_key(tiny_runtime):
    """Advancing a disabled dropout key would break deterministic resume state."""
    state = tiny_runtime.microstep(
        tiny_runtime.parameters, tiny_runtime.trainer, tiny_runtime.rows
    )

    mx.eval(state.trainer.next_key, tiny_runtime.trainer.next_key)
    assert bool(mx.array_equal(state.trainer.next_key, tiny_runtime.trainer.next_key))


def test_enabled_dropout_advances_explicit_prng_key(tmp_path: Path):
    """Enabled dropout preserves the terminal key returned by the forward."""
    runtime = build_tiny_runtime(tmp_path, dropout=0.2)
    expected, _unused = mx.random.split(runtime.trainer.next_key)

    state = runtime.microstep(runtime.parameters, runtime.trainer, runtime.rows)

    mx.eval(state.trainer.next_key, expected)
    assert bool(mx.array_equal(state.trainer.next_key, expected))


def test_training_loop_publishes_exact_forward_returned_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = build_tiny_runtime(tmp_path, dropout=0.2)
    config = replace(
        runtime.config,
        loader=replace(
            runtime.config.loader,
            microbatch_size=2,
            gradient_accumulation_steps=1,
        ),
        maximum_steps=1,
        maximum_epochs=None,
        seed=23,
    )
    terminal = mx.array([101, 202], dtype=mx.uint32)
    captured = []

    class ControlledKernels:
        @staticmethod
        def microstep(parameters, trainer, _rows):
            addressed = counter_random_key(config.seed, 0)
            mx.eval(trainer.next_key, addressed)
            assert bool(mx.array_equal(trainer.next_key, addressed))
            trainer_tree = trainer.to_tree()
            returned = TrainerState.from_compiled_tree(
                (
                    trainer_tree[0],
                    trainer_tree[1],
                    terminal,
                    trainer_tree[3],
                )
            )
            return pretrain_module.MicrostepState(parameters, returned)

        @staticmethod
        def optimizer_step(parameters, optimizer, trainer):
            assert trainer.next_key is terminal
            return pretrain_module.OptimizerStepState(
                parameters,
                optimizer,
                trainer,
                {},
            )

    class SingleEnvelope:
        rows = runtime.rows
        cursor_after = PretrainingCursor(0, 0, 1)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class SingleStream:
        def iter_epoch(self, _epoch):
            yield SingleEnvelope()

        def commit(self, _cursor):
            return None

    def capture_publication(_run, _manifest, state):
        captured.append(state)
        return SimpleNamespace()

    monkeypatch.setattr(
        pretrain_module,
        "build_pretraining_kernels",
        lambda *_args, **_kwargs: ControlledKernels(),
    )
    monkeypatch.setattr(
        pretrain_module,
        "_publish_training_state",
        capture_publication,
    )
    restored = pretrain_module._RestoredTrainingState(
        runtime.parameters,
        runtime.optimizer,
        runtime.trainer,
        pretrain_module.ScalarTrainingState(
            0,
            0,
            0,
            PretrainingCursor.initial(),
        ),
    )

    result = pretrain_module._run_training(
        tmp_path / "run",
        SimpleNamespace(),
        config,
        runtime.model,
        restored,
        SingleStream(),
    )

    assert result.step == 1
    assert len(captured) == 1
    assert captured[0].trainer.next_key is terminal
    assert bool(mx.array_equal(captured[0].trainer.next_key, terminal))
