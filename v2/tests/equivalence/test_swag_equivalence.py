from __future__ import annotations

import mlx.core as mx
import numpy as np
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.swag import score_candidates


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=1.0,
        hidden_dropout=0.0,
    )


def logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(
        np.sum(np.exp(values - peak), axis=axis)
    )


def direct_fp32_mean_oracle(
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


def test_mean_normalized_scores_match_fp32_oracle_for_unequal_endings():
    model = SMLLanguageModel(tiny_model_config(), key=mx.random.key(0))
    input_ids = mx.array(
        [
            [1, 29, 30, 2, 3, 3],
            [1, 29, 32, 33, 34, 2],
            [1, 29, 35, 36, 2, 3],
            [1, 29, 37, 38, 39, 2],
        ],
        dtype=mx.int32,
    )
    pad = 3
    score_mask = (input_ids != pad) & (mx.arange(input_ids.shape[-1])[None, :] >= 2)
    logits, _cache, _key = model.forward_arrays(
        model.parameters(),
        input_ids,
        attention_mask=input_ids != pad,
        positions=None,
        cache_state=None,
        training=False,
        key=None,
    )
    scores = score_candidates(logits, input_ids, score_mask)
    expected = direct_fp32_mean_oracle(logits, input_ids, score_mask)
    mx.eval(scores, expected)
    assert scores.dtype == mx.float32
    assert_close(scores, expected, atol=1e-6, rtol=1e-6)
