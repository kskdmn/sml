"""Explicit MLX pretraining kernels with device-resident mutable state."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_map

from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.training.common import (
    AdamState,
    BaseParameterState,
    PretrainingConfig,
    TrainerState,
    accumulate_fp32,
    adamw_mixed_precision_update_tree,
    learning_rate_at,
    normalize_and_clip,
)


@dataclass(frozen=True, slots=True)
class MicrostepState:
    parameters: BaseParameterState
    trainer: TrainerState


@dataclass(frozen=True, slots=True)
class OptimizerStepState:
    parameters: BaseParameterState
    optimizer: AdamState
    trainer: TrainerState
    metrics: dict


@dataclass(frozen=True, slots=True)
class PretrainingKernels:
    compiled_microstep_core: object
    compiled_optimizer_step_core: object
    eager_microstep_core: object
    eager_optimizer_step_core: object

    def microstep(
        self,
        parameters: BaseParameterState,
        trainer: TrainerState,
        rows: object,
    ) -> MicrostepState:
        rows_array = mx.array(rows)
        next_working, next_trainer_tree = self.compiled_microstep_core(
            parameters.working_parameters,
            trainer.to_tree(),
            rows_array[:, :-1],
            rows_array[:, 1:],
        )
        return MicrostepState(
            parameters=BaseParameterState.from_compiled_tree(
                (parameters.master_parameters, next_working)
            ),
            trainer=TrainerState.from_compiled_tree(next_trainer_tree),
        )

    def optimizer_step(
        self,
        parameters: BaseParameterState,
        optimizer: AdamState,
        trainer: TrainerState,
    ) -> OptimizerStepState:
        (
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        ) = self.compiled_optimizer_step_core(
            parameters.master_parameters,
            parameters.working_parameters,
            optimizer.to_tree(),
            trainer.to_tree(),
        )
        mx.eval(
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        )
        return OptimizerStepState(
            parameters=BaseParameterState.from_compiled_tree(
                (next_masters, next_working)
            ),
            optimizer=AdamState.from_compiled_tree(next_adam_tree),
            trainer=TrainerState.from_compiled_tree(next_trainer_tree),
            metrics=metrics,
        )


def build_pretraining_kernels(
    model: SMLLanguageModel,
    config: PretrainingConfig,
    weight_decay_tree: dict,
) -> PretrainingKernels:
    """Build eager and compiled kernels over explicit, built-in state trees."""

    def loss_with_key(
        working_parameters: dict,
        input_ids: mx.array,
        labels: mx.array,
        key: mx.array,
    ) -> tuple[mx.array, mx.array]:
        logits, cache_state, next_key = model.forward_arrays(
            working_parameters,
            input_ids,
            attention_mask=None,
            positions=None,
            cache_state=None,
            training=True,
            key=key,
        )
        assert cache_state is None
        if next_key is None:
            raise RuntimeError("training forward did not return a PRNG key")
        valid_mask = labels != model.config.pad_token_id
        return causal_lm_loss(logits, labels, valid_mask), next_key

    loss_and_grad = mx.value_and_grad(loss_with_key)

    def microstep_core(
        working_parameters: dict,
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
        input_ids: mx.array,
        labels: mx.array,
    ) -> tuple[dict, tuple[dict, mx.array, mx.array, mx.array]]:
        accumulators, accumulation_count, key, loss_numerator = trainer_tree
        (loss, next_key), gradients = loss_and_grad(
            working_parameters, input_ids, labels, key
        )
        next_accumulators = accumulate_fp32(accumulators, gradients)
        next_count = (accumulation_count + mx.array(1, dtype=mx.int32)).astype(mx.int32)
        next_loss_numerator = (loss_numerator + loss.astype(mx.float32)).astype(
            mx.float32
        )
        return working_parameters, (
            next_accumulators,
            next_count,
            next_key,
            next_loss_numerator,
        )

    def optimizer_step_core(
        master_parameters: dict,
        working_parameters: dict,
        adam_tree: tuple[mx.array, dict, dict],
        trainer_tree: tuple[dict, mx.array, mx.array, mx.array],
    ) -> tuple[
        dict,
        dict,
        tuple[mx.array, dict, dict],
        tuple[dict, mx.array, mx.array, mx.array],
        dict,
    ]:
        del working_parameters
        accumulators, accumulation_count, next_key, loss_numerator = trainer_tree
        gradients = normalize_and_clip(
            accumulators,
            accumulation_count,
            gradient_clip_norm=config.optimizer.gradient_clip_norm,
        )
        next_masters, next_working, next_adam_tree = adamw_mixed_precision_update_tree(
            master_parameters,
            gradients,
            adam_tree,
            config.optimizer,
            weight_decay_tree,
        )
        reset_accumulators = tree_map(
            lambda accumulator: accumulator - accumulator,
            accumulators,
        )
        next_trainer_tree = (
            reset_accumulators,
            (accumulation_count - accumulation_count).astype(mx.int32),
            next_key,
            (loss_numerator - loss_numerator).astype(mx.float32),
        )
        safe_count = mx.maximum(
            accumulation_count.astype(mx.float32), mx.array(1.0, dtype=mx.float32)
        )
        metrics = {
            "learning_rate": learning_rate_at(adam_tree[0], config.optimizer),
            "accumulation_count": accumulation_count,
            "loss_numerator": loss_numerator,
            "loss": (loss_numerator / safe_count).astype(mx.float32),
        }
        return (
            next_masters,
            next_working,
            next_adam_tree,
            next_trainer_tree,
            metrics,
        )

    return PretrainingKernels(
        compiled_microstep_core=mx.compile(microstep_core),
        compiled_optimizer_step_core=mx.compile(optimizer_step_core),
        eager_microstep_core=microstep_core,
        eager_optimizer_step_core=optimizer_step_core,
    )


__all__ = (
    "MicrostepState",
    "OptimizerStepState",
    "PretrainingKernels",
    "build_pretraining_kernels",
)
