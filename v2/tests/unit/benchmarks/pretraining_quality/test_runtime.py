from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerState,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
)

import v2.benchmarks.quality as quality_module


def test_fp32_oracle_is_real_fp32_compute_with_matching_dropout_key():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=4,
        hidden_dropout=0.2,
    )
    model = SMLLanguageModel(config, key=mx.random.key(7))
    bf16_parameters = model.parameters()
    fp32_parameters = tree_map(lambda value: value.astype(mx.float32), bf16_parameters)
    rows = mx.array([[1, 4, 5, 2]], dtype=mx.int32)
    key = mx.random.key(11)

    candidate_logits, candidate_cache, candidate_key = model.forward_arrays(
        bf16_parameters,
        rows,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=True,
        key=key,
    )
    oracle_logits, oracle_key = quality_module._fp32_forward_arrays(
        model,
        fp32_parameters,
        rows,
        training=True,
        key=key,
    )
    mx.eval(candidate_logits, oracle_logits, candidate_key, oracle_key)

    assert candidate_cache is None
    assert candidate_logits.dtype == mx.bfloat16
    assert oracle_logits.dtype == mx.float32
    assert bool(mx.array_equal(candidate_key, oracle_key))
    assert "lm_head" not in fp32_parameters
    assert bool(mx.all(mx.isfinite(oracle_logits)).item())


def _tiny_runtime(tmp_path: Path):
    model_config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=4,
        hidden_dropout=0.2,
    )
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=model_config,
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=2,
            warmup_steps=0,
        ),
        loader=LoaderConfig(gradient_accumulation_steps=1),
        maximum_steps=2,
        maximum_epochs=None,
        compile=False,
    )
    model_key, trainer_key = mx.random.split(mx.random.key(13))
    model = SMLLanguageModel(model_config, key=model_key)
    parameters = initialize_base_parameter_state(model.parameters())
    optimizer = initialize_adam_state(parameters.master_parameters)
    trainer = TrainerState(
        accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
        accumulation_count=mx.array(0, dtype=mx.int32),
        next_key=trainer_key,
        loss_numerator=mx.array(0.0, dtype=mx.float32),
    )
    decay = build_weight_decay_tree(
        parameters.working_parameters,
        config.optimizer.weight_decay,
    )
    return config, model, parameters, optimizer, trainer, decay


def test_oracle_transition_keeps_fp32_working_state_and_matching_key(tmp_path):
    config, model, parameters, optimizer, trainer, decay = _tiny_runtime(tmp_path)
    rows = mx.array([[1, 4, 5, 2, 6]], dtype=mx.int32)
    candidate = quality_module._build_candidate_kernels(model, config, decay)
    oracle = quality_module._build_oracle_kernels(model, config, decay)

    candidate_working, candidate_trainer = candidate.microstep_core(
        parameters.working_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    oracle_working, oracle_trainer = oracle.microstep_core(
        parameters.master_parameters,
        trainer.to_tree(),
        rows[:, :-1],
        rows[:, 1:],
    )
    oracle_result = oracle.optimizer_step_core(
        parameters.master_parameters,
        oracle_working,
        optimizer.to_tree(),
        oracle_trainer,
    )
    mx.eval(candidate_working, candidate_trainer, oracle_result)

    assert bool(mx.array_equal(candidate_trainer[2], oracle_trainer[2]))
    assert int(oracle_result[2][0].item()) == 1
    for (_master_path, master), (_working_path, working) in zip(
        quality_module.tree_flatten(oracle_result[0]),
        quality_module.tree_flatten(oracle_result[1]),
        strict=True,
    ):
        assert working.dtype == mx.float32
        assert bool(mx.array_equal(master, working))


def test_update_statistics_prove_sub_bf16_ulp_master_survival():
    previous_master = {"norm": {"weight": mx.array([1.0], dtype=mx.float32)}}
    previous_working = tree_map(
        lambda value: value.astype(mx.bfloat16), previous_master
    )
    updated_master = {"norm": {"weight": mx.array([1.0001], dtype=mx.float32)}}
    updated_working = tree_map(lambda value: value.astype(mx.bfloat16), updated_master)

    observation = quality_module._observe_update(
        previous_master,
        updated_master,
        previous_working,
        updated_working,
        {"norm.weight": mx.array(0, dtype=mx.int32)},
        step=1,
    )
    statistics, survived = quality_module._materialize_update_observation(observation)

    assert len(statistics) == 1
    statistic = statistics[0]
    assert statistic.path == "norm.weight"
    assert statistic.shape == (1,)
    assert statistic.value_count == 1
    assert statistic.nonzero_update_count == 1
    assert statistic.sub_bf16_ulp_update_count == 1
    assert statistic.survived_sub_bf16_ulp_count == 1
    assert statistic.changed_bf16_working_count == 0
    assert statistic.first_sub_bf16_survival_step == 1
    assert statistic.minimum_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.mean_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.maximum_update_to_bf16_ulp == 0.0128021240234375
    assert statistic.before_master_identity != statistic.after_master_identity
    assert statistic.before_bf16_working_identity == (
        statistic.after_bf16_working_identity
    )
    assert survived is True


@pytest.mark.parametrize("accumulation_steps", (1, 2))
def test_multi_step_runtime_submits_accumulation_and_optimizer_boundaries(
    monkeypatch, tmp_path, accumulation_steps
):
    config, model, parameters, optimizer, trainer, decay = _tiny_runtime(tmp_path)
    kernels = quality_module._build_candidate_kernels(model, config, decay)
    rows = np.asarray([[1, 4, 5, 2, 6]], dtype=np.int32)
    received_keys = []
    real_microstep = kernels.microstep_core

    def record_microstep(working, trainer_tree, input_ids, labels):
        received_keys.append(trainer_tree[2])
        return real_microstep(working, trainer_tree, input_ids, labels)

    kernels = replace(kernels, microstep_core=record_microstep)
    state = quality_module._RuntimeLoopState(
        masters=parameters.master_parameters,
        working=parameters.working_parameters,
        adam_tree=optimizer.to_tree(),
        trainer_tree=trainer.to_tree(),
        cumulative_survival={
            path: mx.array(False, dtype=mx.bool_)
            for path, _value in quality_module.tree_flatten(
                parameters.master_parameters
            )
        },
        observations=(),
        metrics={"loss": mx.array(0.0, dtype=mx.float32)},
        microstep_index=0,
    )
    real_async_eval = mx.async_eval
    submissions = []

    def recording_async_eval(*trees):
        submissions.append(trees)
        real_async_eval(*trees)

    monkeypatch.setattr(mx, "async_eval", recording_async_eval)

    result = quality_module._execute_training_steps(
        kernels=kernels,
        runtime="candidate",
        gradient_accumulation_steps=accumulation_steps,
        ordered_batches=((0,),) * (3 * accumulation_steps),
        training_rows=rows,
        start_step=0,
        stop_step=3,
        state=state,
        training_seed=config.seed,
        dropout_enabled=True,
    )
    mx.eval(result.masters, result.adam_tree, result.trainer_tree)

    expected_submissions = 3 * (accumulation_steps + 1 if accumulation_steps > 1 else 1)
    assert len(submissions) == expected_submissions
    if accumulation_steps > 1:
        for step in range(3):
            for microstep in range(accumulation_steps):
                (submitted_trainer,) = submissions[
                    step * (accumulation_steps + 1) + microstep
                ]
                assert int(submitted_trainer[1].item()) == microstep + 1
    assert int(result.adam_tree[0].item()) == 3
    assert result.microstep_index == 3 * accumulation_steps
    for index, key in enumerate(received_keys):
        assert bool(
            mx.array_equal(key, quality_module.counter_random_key(config.seed, index))
        )
    assert not bool(mx.array_equal(result.trainer_tree[2], trainer.next_key))
