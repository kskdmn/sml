from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from sml.data.swag import (
    SwagBatch,
    SwagBatchStream,
    SwagCursor,
)
from sml.training import swag as swag_module
from sml.training.swag import (
    score_candidates,
)

from .swag_helpers import (
    TinySwagRuntime,
    _first_padded_batch,
    assert_close,
    assert_tree_close,
    clone_tree,
)


def _example_mask_host(batch: SwagBatch) -> np.ndarray:
    return np.array(batch.example_mask, dtype=bool)


def _select_examples(batch: SwagBatch, mask: np.ndarray) -> SwagBatch:
    return SwagBatch(
        mx.array(np.array(batch.input_ids)[mask]),
        mx.array(np.array(batch.score_mask)[mask]),
        mx.array(np.array(batch.labels)[mask]),
        mx.array(np.array(batch.example_mask)[mask]),
        mx.array(np.array(batch.valid_token_mask)[mask]),
        batch.cursor_after,
    )


def _ranking_trainer(runtime: TinySwagRuntime, batch: SwagBatch):
    kernels = runtime.kernels(compiled=False)
    adapters = clone_tree(runtime.initial_adapters)
    trainer = swag_module.initial_swag_trainer_state(adapters, key=mx.random.key(11))
    trainer = kernels.ranking_microstep(
        adapters,
        runtime.frozen_base,
        trainer,
        batch,
    )
    mx.eval(trainer.to_tree())
    return trainer


def logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(
        np.sum(np.exp(values - peak), axis=axis)
    )


def fixed_logits(*, dtype) -> mx.array:
    vocab = 8
    length = 6
    logits = mx.arange(4 * length * vocab, dtype=mx.float32).reshape(4, length, vocab)
    logits = (logits - mx.mean(logits)) / 10.0
    return logits.astype(dtype)


def score_fixture_with_unequal_lengths_and_eos() -> tuple[mx.array, mx.array]:
    pad = 3
    eos = 2
    input_ids = mx.array(
        [
            [1, 4, 5, eos, pad, pad],
            [1, 4, 6, 7, eos, pad],
            [1, 4, 0, eos, pad, pad],
            [1, 4, 5, 6, 7, eos],
        ],
        dtype=mx.int32,
    )
    score_mask = mx.array(
        [
            [False, False, True, True, False, False],
            [False, False, True, True, True, False],
            [False, False, True, True, False, False],
            [False, False, True, True, True, True],
        ]
    )
    return input_ids, score_mask


def direct_fp32_target_logit_minus_logsumexp_mean(
    logits: mx.array,
    input_ids: mx.array,
    score_mask: mx.array,
) -> mx.array:
    shift_logits = np.array(logits[..., :-1, :].astype(mx.float32), dtype=np.float64)
    targets = np.array(input_ids[..., 1:], dtype=np.int64)
    mask = np.array(score_mask[..., 1:], dtype=np.float64)
    target_logit = np.take_along_axis(shift_logits, targets[..., None], axis=-1)[..., 0]
    token_ll = target_logit - logsumexp_np(shift_logits, axis=-1)
    counted = np.maximum(mask.sum(axis=-1), 1.0)
    means = (token_ll * mask).sum(axis=-1) / counted
    return mx.array(means.astype(np.float32), dtype=mx.float32)


def test_candidate_score_is_fp32_mean_including_eos():
    logits = fixed_logits(dtype=mx.bfloat16)
    input_ids, score_mask = score_fixture_with_unequal_lengths_and_eos()
    scores = score_candidates(logits, input_ids, score_mask)
    expected = direct_fp32_target_logit_minus_logsumexp_mean(
        logits, input_ids, score_mask
    )
    mx.eval(scores, expected)
    assert scores.dtype == mx.float32
    assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_padded_tail_matches_unpadded_example_weighted_update(tiny_swag_runtime):
    padded = tiny_swag_runtime.train_one_epoch(fixed_batch_size=4)
    eager = tiny_swag_runtime.train_one_epoch(fixed_batch_size=1)
    assert padded.real_examples == eager.real_examples == 5
    initial = dict(tree_flatten(tiny_swag_runtime.initial_adapters))
    assert any(
        not bool(mx.array_equal(value, initial[name]).item())
        for name, value in tree_flatten(padded.adapters)
    )
    assert_tree_close(padded.adapters, eager.adapters, atol=1e-5, rtol=1e-5)
    assert (
        padded.cursor
        == eager.cursor
        == SwagCursor(epoch=1, bucket_order_position=0, row_offset=0)
    )


def test_per_slot_ranking_losses_are_finite_before_masking(tiny_swag_runtime):
    padded = _first_padded_batch(tiny_swag_runtime)
    real_mask = _example_mask_host(padded)
    assert real_mask.any()
    assert (~real_mask).any()
    unpadded = _select_examples(padded, real_mask)
    padded_trainer = _ranking_trainer(tiny_swag_runtime, padded)
    reference_trainer = _ranking_trainer(tiny_swag_runtime, unpadded)
    assert bool(mx.isfinite(padded_trainer.loss_numerator).item())
    assert int(padded_trainer.valid_count) == int(real_mask.sum())
    assert int(padded_trainer.valid_count) == int(reference_trainer.valid_count)
    assert int(padded_trainer.correct_count) == int(reference_trainer.correct_count)
    assert_close(
        padded_trainer.loss_numerator,
        reference_trainer.loss_numerator,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_tree_close(
        padded_trainer.accumulators,
        reference_trainer.accumulators,
        atol=1e-5,
        rtol=1e-5,
    )


def test_synthetic_slots_have_no_loss_accuracy_gradient_progress_or_cursor_contribution(
    tiny_swag_runtime,
):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=4,
        prefetch_depth=1,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        batches = [SwagBatch.from_envelope(envelope) for envelope in stream]
    assert any(not bool(np.array(batch.example_mask).all()) for batch in batches)
    tail = next(
        batch for batch in batches if not bool(np.array(batch.example_mask).all())
    )
    pad_slots = ~_example_mask_host(tail)
    real_slots = _example_mask_host(tail)
    assert pad_slots.any()
    assert bool(np.array(tail.valid_token_mask)[pad_slots].any())
    assert bool(np.array(tail.score_mask)[pad_slots].any())
    real_seen = sum(int(np.array(batch.example_mask).sum()) for batch in batches)
    assert real_seen == 5
    assert batches[-1].cursor_after == SwagCursor(
        epoch=1, bucket_order_position=0, row_offset=0
    )

    unpadded = _select_examples(tail, real_slots)
    synthetic = _select_examples(tail, pad_slots)
    padded_trainer = _ranking_trainer(tiny_swag_runtime, tail)
    reference_trainer = _ranking_trainer(tiny_swag_runtime, unpadded)
    synthetic_trainer = _ranking_trainer(tiny_swag_runtime, synthetic)
    assert int(synthetic_trainer.valid_count) == 0
    assert int(synthetic_trainer.correct_count) == 0
    assert float(np.array(synthetic_trainer.loss_numerator)) == 0.0
    for _name, leaf in tree_flatten(synthetic_trainer.accumulators):
        assert bool(mx.allclose(leaf, mx.zeros_like(leaf)).item()), _name
    assert int(padded_trainer.valid_count) == int(reference_trainer.valid_count)
    assert int(padded_trainer.correct_count) == int(reference_trainer.correct_count)
    assert_close(
        padded_trainer.loss_numerator,
        reference_trainer.loss_numerator,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_tree_close(
        padded_trainer.accumulators,
        reference_trainer.accumulators,
        atol=1e-5,
        rtol=1e-5,
    )
