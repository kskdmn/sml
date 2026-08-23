"""Compiled mean-normalized SWAG ranking kernels and FP32 adapter updates."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_map

from sml.data.swag import SwagBatch
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    AdamState,
    LoaderConfig,
    OptimizerConfig,
    accumulate_fp32,
    adamw_fp32_update_tree,
    normalize_and_clip,
)


def _merge_adapter_parameters(adapters: object, frozen_base: object) -> object:
    if isinstance(frozen_base, dict):
        adapter_map = adapters if isinstance(adapters, dict) else {}
        merged: dict[object, object] = {}
        for key in set(frozen_base) | set(adapter_map):
            if key in frozen_base and key in adapter_map:
                merged[key] = _merge_adapter_parameters(
                    adapter_map[key], frozen_base[key]
                )
            elif key in adapter_map:
                merged[key] = adapter_map[key]
            else:
                merged[key] = frozen_base[key]
        return merged
    if isinstance(frozen_base, list):
        adapter_items = (
            adapters if isinstance(adapters, list) else [{} for _ in frozen_base]
        )
        return [
            _merge_adapter_parameters(adapter, frozen)
            for adapter, frozen in zip(adapter_items, frozen_base, strict=True)
        ]
    if isinstance(frozen_base, tuple):
        adapter_items = (
            adapters if isinstance(adapters, tuple) else tuple({} for _ in frozen_base)
        )
        return tuple(
            _merge_adapter_parameters(adapter, frozen)
            for adapter, frozen in zip(adapter_items, frozen_base, strict=True)
        )
    return frozen_base


def score_candidates(
    logits: mx.array,
    input_ids: mx.array,
    score_mask: mx.array,
) -> mx.array:
    """Return FP32 mean continuation-token log-likelihood, including EOS."""

    shifted_logits = logits[..., :-1, :].astype(mx.float32)
    targets = input_ids[..., 1:]
    mask = score_mask[..., 1:].astype(mx.float32)
    target_logit = mx.take_along_axis(
        shifted_logits, mx.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(-1)
    token_ll = target_logit - mx.logsumexp(shifted_logits, axis=-1)
    counted = mx.maximum(mask.sum(axis=-1), mx.array(1.0, dtype=mx.float32))
    return ((token_ll * mask).sum(axis=-1) / counted).astype(mx.float32)


def default_swag_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(learning_rate=1e-4, schedule_steps=8_192)


@dataclass(frozen=True, slots=True)
class SwagTrainingConfig:
    loader: LoaderConfig
    optimizer: OptimizerConfig
    compile: bool = False
    seed: int = 0


@dataclass(frozen=True, slots=True)
class SwagKernelConfig:
    accumulation_steps: int
    gradient_clip_norm: float
    compile: bool


@dataclass(frozen=True, slots=True)
class SwagTrainerState:
    accumulators: dict
    valid_count: mx.array
    next_key: mx.array
    loss_numerator: mx.array
    correct_count: mx.array

    def to_tree(self) -> tuple[dict, mx.array, mx.array, mx.array, mx.array]:
        return (
            self.accumulators,
            self.valid_count,
            self.next_key,
            self.loss_numerator,
            self.correct_count,
        )

    @classmethod
    def from_compiled_tree(cls, tree: object) -> SwagTrainerState:
        """Wrap a compiled trainer tree without synchronizing counters."""

        (
            accumulators,
            valid_count,
            next_key,
            loss_numerator,
            correct_count,
        ) = tree
        instance = object.__new__(cls)
        object.__setattr__(instance, "accumulators", accumulators)
        object.__setattr__(instance, "valid_count", valid_count)
        object.__setattr__(instance, "next_key", next_key)
        object.__setattr__(instance, "loss_numerator", loss_numerator)
        object.__setattr__(instance, "correct_count", correct_count)
        return instance


def initial_swag_trainer_state(adapters: dict, *, key: mx.array) -> SwagTrainerState:
    zeros = tree_map(
        lambda parameter: mx.zeros_like(parameter).astype(mx.float32), adapters
    )
    return SwagTrainerState(
        accumulators=zeros,
        valid_count=mx.array(0, dtype=mx.int32),
        next_key=key,
        loss_numerator=mx.array(0.0, dtype=mx.float32),
        correct_count=mx.array(0, dtype=mx.int32),
    )


@dataclass(frozen=True, slots=True)
class SwagKernels:
    compiled_ranking_microstep_core: object
    compiled_optimizer_step_core: object
    kernel_config: SwagKernelConfig

    def ranking_microstep(
        self,
        adapters: dict,
        frozen_base: dict,
        trainer: SwagTrainerState,
        batch: SwagBatch,
    ) -> SwagTrainerState:
        next_tree = self.compiled_ranking_microstep_core(
            adapters,
            frozen_base,
            trainer.to_tree(),
            batch.input_ids,
            batch.score_mask,
            batch.labels,
            batch.example_mask,
            batch.valid_token_mask,
        )
        return SwagTrainerState.from_compiled_tree(next_tree)

    def optimizer_step(
        self,
        adapters: dict,
        optimizer: AdamState,
        trainer: SwagTrainerState,
    ) -> tuple[dict, AdamState, SwagTrainerState]:
        next_adapters, next_adam, next_trainer = self.compiled_optimizer_step_core(
            adapters,
            optimizer.to_tree(),
            trainer.to_tree(),
        )
        mx.eval(next_adapters, next_adam, next_trainer)
        return (
            next_adapters,
            AdamState.from_compiled_tree(next_adam),
            SwagTrainerState.from_compiled_tree(next_trainer),
        )


def build_swag_kernels(
    model: SMLLanguageModel,
    config: SwagTrainingConfig,
    weight_decay_tree: dict,
) -> SwagKernels:
    """Build host wrappers around eager or compiled ranking and optimizer cores."""

    kernel_config = SwagKernelConfig(
        accumulation_steps=config.loader.gradient_accumulation_steps,
        gradient_clip_norm=config.optimizer.gradient_clip_norm,
        compile=config.compile,
    )

    def adapter_loss(adapters, frozen_base, batch_arrays, key):
        input_ids, score_mask, labels, example_mask, valid_token_mask = batch_arrays
        parameters = _merge_adapter_parameters(adapters, frozen_base)
        batch_size, candidate_count, length = input_ids.shape
        flat_ids = input_ids.reshape((batch_size * candidate_count, length))
        flat_valid = valid_token_mask.reshape((batch_size * candidate_count, length))
        flat_score = score_mask.reshape((batch_size * candidate_count, length))
        logits, _cache_state, next_key = model.forward_arrays(
            parameters,
            flat_ids,
            attention_mask=flat_valid,
            positions=None,
            cache_state=None,
            training=True,
            key=key,
        )
        if next_key is None:
            raise RuntimeError("training forward did not return a PRNG key")
        scores = score_candidates(logits, flat_ids, flat_score)
        scores = scores.reshape((batch_size, candidate_count))
        label_scores = mx.take_along_axis(
            scores, mx.expand_dims(labels, axis=-1), axis=-1
        ).squeeze(-1)
        per_slot = mx.logsumexp(scores, axis=-1) - label_scores
        weights = example_mask.astype(mx.float32)
        loss_sum = (per_slot * weights).sum().astype(mx.float32)
        predictions = mx.argmax(scores, axis=-1)
        correct = (
            (predictions == labels).astype(mx.int32) * example_mask.astype(mx.int32)
        ).sum()
        valid = example_mask.astype(mx.int32).sum()
        return loss_sum, (next_key, correct.astype(mx.int32), valid.astype(mx.int32))

    loss_and_grad = mx.value_and_grad(adapter_loss, argnums=0)

    def ranking_microstep_core(
        adapters,
        frozen_base,
        trainer_tree,
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
    ):
        accumulators, valid_count, key, loss_numerator, correct_count = trainer_tree
        batch_arrays = (
            input_ids,
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
        )
        (loss_sum, (next_key, correct, valid)), gradients = loss_and_grad(
            adapters, frozen_base, batch_arrays, key
        )
        next_accumulators = accumulate_fp32(accumulators, gradients)
        next_valid = (valid_count + valid).astype(mx.int32)
        next_loss = (loss_numerator + loss_sum.astype(mx.float32)).astype(mx.float32)
        next_correct = (correct_count + correct).astype(mx.int32)
        return (
            next_accumulators,
            next_valid,
            next_key,
            next_loss,
            next_correct,
        )

    def swag_optimizer_step_core(adapters, adam_tree, trainer_tree):
        accumulators, valid_count, next_key, loss_numerator, correct_count = (
            trainer_tree
        )
        gradients = normalize_and_clip(
            accumulators,
            valid_count,
            gradient_clip_norm=kernel_config.gradient_clip_norm,
        )
        next_adapters, next_adam = adamw_fp32_update_tree(
            adapters,
            gradients,
            adam_tree,
            config.optimizer,
            weight_decay_tree,
        )
        reset_accumulators = tree_map(
            lambda accumulator: accumulator - accumulator, accumulators
        )
        next_trainer = (
            reset_accumulators,
            (valid_count - valid_count).astype(mx.int32),
            next_key,
            (loss_numerator - loss_numerator).astype(mx.float32),
            (correct_count - correct_count).astype(mx.int32),
        )
        return next_adapters, next_adam, next_trainer

    return SwagKernels(
        compiled_ranking_microstep_core=(
            mx.compile(ranking_microstep_core)
            if config.compile
            else ranking_microstep_core
        ),
        compiled_optimizer_step_core=(
            mx.compile(swag_optimizer_step_core)
            if config.compile
            else swag_optimizer_step_core
        ),
        kernel_config=kernel_config,
    )


__all__ = (
    "SwagKernelConfig",
    "SwagKernels",
    "SwagTrainerState",
    "SwagTrainingConfig",
    "build_swag_kernels",
    "default_swag_optimizer_config",
    "initial_swag_trainer_state",
    "score_candidates",
)
