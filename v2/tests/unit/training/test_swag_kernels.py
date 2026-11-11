from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map
from sml.data.swag import (
    SwagBatch,
    SwagBatchStream,
    SwagCursor,
)
from sml.errors import SMLArtifactError, SMLConfigurationError, SMLRuntimeError
from sml.model.language_model import SMLLanguageModel
from sml.training import swag as swag_module
from sml.training.common import (
    WeightDecayPolicy,
    build_weight_decay_tree,
    initialize_adam_state,
)
from sml.training.lora import LoRAConfig, LoRAInitializerConfig, apply_lora
from sml.training.random import counter_random_key
from sml.training.swag import (
    SwagKernelConfig,
    build_swag_kernels,
)

from .swag_helpers import (
    TinySwagRuntime,
    _first_padded_batch,
    assert_tree_close,
    clone_tree,
    split_adapter_parameters,
    tiny_model_config,
)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("valid_count", 1, "accumulation must be empty"),
        ("loss_numerator", float("nan"), "loss numerator must be empty"),
        ("correct_count", 1, "correct count must be empty"),
        ("accumulators", float("nan"), "accumulators must be empty"),
        ("accumulators", 1.0, "accumulators must be empty"),
    ],
)
def test_checkpoint_boundary_rejects_nonempty_or_nonfinite_trainer_state(
    field, value, message
):
    trainer = swag_module.initial_swag_trainer_state(
        {"first": mx.zeros((2,)), "nested": {"last": mx.zeros((3,))}},
        key=mx.random.key(3),
    )
    swag_module._require_empty_trainer_state(trainer)
    changed = (
        {
            "first": trainer.accumulators["first"],
            "nested": {"last": mx.full((3,), value)},
        }
        if field == "accumulators"
        else mx.array(value, dtype=getattr(trainer, field).dtype)
    )
    with pytest.raises(SMLArtifactError, match=message):
        swag_module._require_empty_trainer_state(replace(trainer, **{field: changed}))


def assert_tree_equal(actual: dict, expected: dict):
    actual_leaves = dict(tree_flatten(actual))
    expected_leaves = dict(tree_flatten(expected))
    assert set(actual_leaves) == set(expected_leaves)
    mx.eval(*actual_leaves.values(), *expected_leaves.values())
    for name, actual_leaf in actual_leaves.items():
        assert bool(mx.array_equal(actual_leaf, expected_leaves[name]).item()), name


def assert_tree_dtypes(tree: object, dtype: mx.Dtype):
    for name, leaf in tree_flatten(tree):
        assert leaf.dtype == dtype, name


def assert_builtin_array_tree(tree: object) -> None:
    def check(node: object) -> bool:
        if isinstance(node, mx.array):
            return True
        if isinstance(node, dict):
            return all(check(value) for value in node.values())
        if isinstance(node, (list, tuple)):
            return all(check(value) for value in node)
        return False

    assert check(tree)


def test_swag_kernels_require_a_nonempty_static_lora_policy(tiny_swag_runtime):
    unadapted = SMLLanguageModel(tiny_model_config(), key=mx.random.key(17))

    with pytest.raises(SMLConfigurationError, match="nonempty LoRA forward policy"):
        build_swag_kernels(unadapted, tiny_swag_runtime.config, {})


@pytest.mark.parametrize("compiled", (False, True))
def test_swag_kernels_reject_lora_policy_reassignment_before_execution(
    tiny_swag_runtime,
    compiled: bool,
):
    kernels = tiny_swag_runtime.kernels(compiled=compiled)
    policy = tiny_swag_runtime.model.lora_forward_policy
    assert policy is not None
    tiny_swag_runtime.model.lora_forward_policy = type(policy)(policy.adapters)

    with pytest.raises(
        SMLConfigurationError, match="policy changed after kernel build"
    ):
        kernels.compiled_ranking_microstep_core(*tiny_swag_runtime.core_inputs())


@pytest.mark.parametrize("accumulation_steps", (1, 4))
def test_swag_accumulation_submits_masked_microsteps_without_host_waits(
    tiny_swag_runtime, monkeypatch, accumulation_steps
):
    """Padded batches must submit gradients and counts before the window ends."""
    runtime = tiny_swag_runtime
    batch = _first_padded_batch(runtime)
    real_examples = int(batch.example_mask.sum().item())
    config = replace(
        runtime.config,
        loader=replace(
            runtime.config.loader,
            gradient_accumulation_steps=accumulation_steps,
        ),
    )
    kernels = build_swag_kernels(runtime.model, config, runtime.weight_decay_tree)
    trainer = swag_module.initial_swag_trainer_state(
        runtime.initial_adapters, key=mx.random.key(11)
    )
    submitted = []
    submit = mx.async_eval

    def record_submission(tree):
        submitted.append(dict(tree_flatten(tree)))
        submit(tree)

    def reject_host_wait(*_args):
        raise AssertionError("microstep blocked the host before the optimizer update")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "async_eval", record_submission)
        patch.setattr(mx, "eval", reject_host_wait)
        for index in range(accumulation_steps):
            trainer = kernels.ranking_microstep(
                runtime.initial_adapters, runtime.frozen_base, trainer, batch
            )
            if accumulation_steps > 1:
                assert len(submitted) == index + 1
                returned = dict(tree_flatten(trainer.to_tree()))
                assert submitted[-1].keys() == returned.keys()
                assert all(
                    submitted[-1][name] is value for name, value in returned.items()
                )
            else:
                assert not submitted

    assert int(trainer.valid_count.item()) == accumulation_steps * real_examples
    assert float(trainer.loss_numerator.item()) > 0.0


def test_compiled_swag_optimizer_core_consumes_builtin_adapter_and_adam_trees(
    tiny_swag_runtime,
):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    ranking_inputs = tiny_swag_runtime.core_inputs()
    adapters = ranking_inputs[0]
    assert_builtin_array_tree(ranking_inputs)
    adapter_paths = [path for path, _leaf in tree_flatten(ranking_inputs[0])]
    frozen_paths = [path for path, _leaf in tree_flatten(ranking_inputs[1])]
    assert all(
        path.rsplit(".", 1)[-1] in {"lora_a", "lora_b"} for path in adapter_paths
    )
    assert all(path.rsplit(".", 1)[-1] != "scale" for path in frozen_paths)
    trainer_tree = kernels.compiled_ranking_microstep_core(*ranking_inputs)
    mx.eval(trainer_tree)
    assert_builtin_array_tree(trainer_tree)

    optimizer = initialize_adam_state(adapters)
    outputs = kernels.compiled_optimizer_step_core(
        adapters,
        optimizer.to_tree(),
        trainer_tree,
    )
    mx.eval(outputs)
    assert_builtin_array_tree(outputs)
    next_adapters, next_adam_tree, next_trainer_tree, finite = outputs
    assert bool(finite)
    assert_tree_dtypes(next_adapters, mx.float32)
    assert_tree_dtypes(next_adam_tree[1], mx.float32)
    assert_tree_dtypes(next_adam_tree[2], mx.float32)
    changed = False
    for (_, before), (_, after) in zip(
        tree_flatten(adapters), tree_flatten(next_adapters), strict=True
    ):
        if not bool(mx.array_equal(before, after).item()):
            changed = True
            break
    assert changed
    assert int(next_trainer_tree[1].item()) == 0


@pytest.mark.parametrize("compiled", (False, True))
@pytest.mark.parametrize(
    "failure",
    ("nan-loss", "infinite-gradient", "parameter-overflow", "epsilon-underflow"),
)
def test_swag_optimizer_rejects_numerical_failure_without_changing_input_state(
    tiny_swag_runtime, compiled, failure
):
    runtime = tiny_swag_runtime
    config = replace(runtime.config, compile=compiled)
    adapters = runtime.initial_adapters
    optimizer = initialize_adam_state(adapters)
    trainer = replace(
        swag_module.initial_swag_trainer_state(adapters, key=mx.random.key(11)),
        accumulators=tree_map(mx.ones_like, adapters),
        valid_count=mx.array(1, dtype=mx.int32),
        loss_numerator=mx.array(1.0, dtype=mx.float32),
    )
    if failure == "nan-loss":
        trainer = replace(
            trainer, loss_numerator=mx.array(float("nan"), dtype=mx.float32)
        )
    elif failure == "infinite-gradient":
        trainer = replace(
            trainer,
            accumulators=tree_map(
                lambda value: mx.full_like(value, float("inf")), adapters
            ),
        )
    elif failure == "parameter-overflow":
        config = replace(
            config,
            optimizer=replace(
                config.optimizer,
                learning_rate=3e38,
                schedule_steps=None,
                warmup_steps=0,
            ),
        )
    else:
        config = replace(config, optimizer=replace(config.optimizer, epsilon=1e-100))
    kernels = build_swag_kernels(runtime.model, config, runtime.weight_decay_tree)
    before = dict(tree_flatten(adapters))
    with pytest.raises(SMLRuntimeError, match="nonfinite"):
        kernels.optimizer_step(adapters, optimizer, trainer)
    assert int(optimizer.step.item()) == 0
    assert all(value is before[name] for name, value in tree_flatten(adapters))


def test_second_compiled_step_sees_returned_adapter_optimizer_and_key_state(
    tiny_swag_runtime,
):
    model = SMLLanguageModel(
        tiny_model_config(hidden_dropout=0.1), key=mx.random.key(3)
    )
    apply_lora(
        model,
        LoRAConfig(
            rank=2,
            alpha=1.0,
            scaling_mode="lora",
            dropout=0.5,
            target_modules=("q_proj", "v_proj", "down_proj"),
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        key=mx.random.key(5),
    )
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    mx.eval(adapters, frozen_base)
    before_base = clone_tree(frozen_base)
    runtime = TinySwagRuntime(
        bundle=tiny_swag_runtime.bundle,
        model=model,
        frozen_base=frozen_base,
        initial_adapters=clone_tree(adapters),
        weight_decay_tree=build_weight_decay_tree(adapters, WeightDecayPolicy()),
        config=tiny_swag_runtime.config,
    )

    def run_two_steps(*, compiled: bool):
        kernels = runtime.kernels(compiled=compiled)
        adapters = clone_tree(runtime.initial_adapters)
        optimizer = initialize_adam_state(adapters)
        key0 = mx.random.key(11)
        trainer = swag_module.initial_swag_trainer_state(adapters, key=key0)
        loader = replace(
            runtime.config.loader,
            microbatch_size=1,
            prefetch_depth=2,
            gradient_accumulation_steps=1,
        )
        keys = [key0]
        start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
        with SwagBatchStream._borrowing_bundle(
            runtime.bundle, loader, cursor=start
        ) as stream:
            iterator = iter(stream)
            for _step in range(2):
                batch = SwagBatch.from_envelope(next(iterator))
                trainer = kernels.ranking_microstep(
                    adapters,
                    runtime.frozen_base,
                    trainer,
                    batch,
                )
                keys.append(trainer.next_key)
                adapters, optimizer, trainer = kernels.optimizer_step(
                    adapters,
                    optimizer,
                    trainer,
                )
                stream.commit(batch.cursor_after)
        mx.eval(adapters, optimizer.to_tree(), trainer.to_tree(), *keys)
        return adapters, optimizer, trainer, keys

    compiled_adapters, compiled_optimizer, compiled_trainer, compiled_keys = (
        run_two_steps(compiled=True)
    )
    eager_adapters, eager_optimizer, eager_trainer, eager_keys = run_two_steps(
        compiled=False
    )
    expected_keys = [eager_keys[0]]
    for _microstep in range(2):
        terminal = expected_keys[-1]
        for _site in range(8):
            terminal, _unused = mx.random.split(terminal)
        expected_keys.append(terminal)
    mx.eval(
        compiled_adapters,
        compiled_optimizer.to_tree(),
        compiled_trainer.to_tree(),
        eager_adapters,
        eager_optimizer.to_tree(),
        eager_trainer.to_tree(),
        frozen_base,
        before_base,
        *compiled_keys,
        *eager_keys,
        *expected_keys,
    )
    assert int(np.array(compiled_optimizer.step)) == 2
    assert int(np.array(eager_optimizer.step)) == 2
    assert bool(mx.array_equal(compiled_keys[0], eager_keys[0]).item())
    assert bool(mx.array_equal(eager_keys[1], expected_keys[1]).item())
    assert bool(mx.array_equal(eager_keys[2], expected_keys[2]).item())
    assert bool(mx.array_equal(compiled_keys[1], expected_keys[1]).item())
    assert bool(mx.array_equal(compiled_keys[2], expected_keys[2]).item())
    assert bool(mx.array_equal(compiled_keys[1], eager_keys[1]).item())
    assert bool(mx.array_equal(compiled_keys[2], eager_keys[2]).item())
    assert_tree_dtypes(frozen_base, mx.bfloat16)
    assert_tree_dtypes(compiled_adapters, mx.float32)
    assert_tree_dtypes(compiled_optimizer.first_moments, mx.float32)
    assert_tree_dtypes(compiled_optimizer.second_moments, mx.float32)
    assert_tree_equal(frozen_base, before_base)
    assert_tree_close(compiled_adapters, eager_adapters, atol=1e-5, rtol=1e-5)
    assert_tree_close(
        compiled_optimizer.to_tree(), eager_optimizer.to_tree(), atol=1e-5, rtol=1e-5
    )
    assert_tree_close(
        compiled_trainer.to_tree(), eager_trainer.to_tree(), atol=1e-5, rtol=1e-5
    )
    compiled_trainer_tree = compiled_trainer.to_tree()
    eager_trainer_tree = eager_trainer.to_tree()
    for index in (1, 2, 4):
        assert bool(
            mx.array_equal(
                compiled_trainer_tree[index], eager_trainer_tree[index]
            ).item()
        )


def test_training_loop_publishes_exact_forward_returned_key(
    tiny_swag_runtime,
    tmp_path,
    monkeypatch,
):
    config = replace(
        tiny_swag_runtime.config,
        lora=replace(tiny_swag_runtime.config.lora, dropout=0.5),
        loader=replace(
            tiny_swag_runtime.config.loader,
            microbatch_size=1,
            gradient_accumulation_steps=1,
        ),
        maximum_steps=1,
        maximum_epochs=None,
        seed=29,
    )
    adapters = clone_tree(tiny_swag_runtime.initial_adapters)
    optimizer = initialize_adam_state(adapters)
    trainer = swag_module.initial_swag_trainer_state(
        adapters,
        key=mx.random.key(7),
    )
    terminal = mx.array([303, 404], dtype=mx.uint32)
    captured = []

    class ControlledKernels:
        @staticmethod
        def ranking_microstep(_adapters, _frozen_base, trainer, _batch):
            addressed = counter_random_key(config.seed, 0)
            mx.eval(trainer.next_key, addressed)
            assert bool(mx.array_equal(trainer.next_key, addressed))
            trainer_tree = trainer.to_tree()
            return swag_module.SwagTrainerState.from_compiled_tree(
                (
                    trainer_tree[0],
                    mx.array(1, dtype=mx.int32),
                    terminal,
                    trainer_tree[3],
                    trainer_tree[4],
                )
            )

        @staticmethod
        def optimizer_step(adapters, optimizer, trainer):
            assert trainer.next_key is terminal
            return adapters, optimizer, trainer

    def capture_publication(_run, _manifest, state):
        captured.append(state)
        return SimpleNamespace()

    monkeypatch.setattr(
        swag_module,
        "build_swag_kernels",
        lambda *_args, **_kwargs: ControlledKernels(),
    )
    monkeypatch.setattr(
        swag_module,
        "_publish_training_state",
        capture_publication,
    )
    restored = swag_module._RestoredSwagState(
        adapters,
        tiny_swag_runtime.frozen_base,
        optimizer,
        trainer,
        swag_module.ScalarSwagState(0, 0, 0, SwagCursor.initial()),
    )

    result = swag_module._run_training(
        tmp_path / "run",
        SimpleNamespace(),
        config,
        tiny_swag_runtime.model,
        restored,
        tiny_swag_runtime.bundle,
    )

    assert result.step == 1
    assert len(captured) == 1
    assert captured[0].trainer.next_key is terminal
    assert bool(mx.array_equal(captured[0].trainer.next_key, terminal))


def test_kernel_config_is_frozen_from_loader_optimizer_and_compile(tiny_swag_runtime):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    config = kernels.kernel_config
    assert isinstance(config, SwagKernelConfig)
    assert (
        config.accumulation_steps
        == tiny_swag_runtime.config.loader.gradient_accumulation_steps
    )
    assert (
        config.gradient_clip_norm
        == tiny_swag_runtime.config.optimizer.gradient_clip_norm
    )
    assert config.compile is True
