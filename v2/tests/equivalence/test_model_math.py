from __future__ import annotations

import math

import mlx.core as mx
import pytest
from sml.model.cache import KVCache
from sml.model.config import GenerationConfig, ModelConfig
from sml.model.generation import select_next_token
from sml.model.language_model import SMLLanguageModel, causal_lm_loss
from sml.model.rope import (
    RotaryEmbedding,
    find_correction_dimension,
    find_correction_range,
    resolve_attention_factor,
)


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


@pytest.mark.parametrize(
    ("factor", "suffix"),
    [(2.0, ""), (4.0, ".factor_4")],
)
def test_rope_caches_match_captured_reference(legacy_arrays, factor, suffix):
    """A cache formula drift would affect all rotated attention keys and queries."""
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=factor,
        hidden_dropout=0.0,
    )

    rope = RotaryEmbedding(config)

    mx.eval(rope.cos_cached, rope.sin_cached)
    assert rope.cos_cached.dtype == mx.float32
    assert rope.sin_cached.dtype == mx.float32
    assert_close(
        rope.cos_cached,
        legacy_arrays[f"rope.cos_cache{suffix}"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert_close(
        rope.sin_cached,
        legacy_arrays[f"rope.sin_cache{suffix}"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_yarn_correction_math_matches_captured_factor_two_metadata(legacy_control):
    """Cutoff or attention-scale changes would select incorrect YaRN frequency bands."""
    factor_metadata = legacy_control["cases"]["rope"]["factors"]["2.0"]

    low, high = find_correction_range(32.0, 1.0, 4, 10_000.0, 8, truncate=True)

    assert (low, high) == tuple(factor_metadata["correction_range"])
    assert find_correction_dimension(32.0, 4, 10_000.0, 8) < 0.0
    assert resolve_attention_factor(2.0) == pytest.approx(1.0 + 0.1 * math.log(2.0))


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=2.0,
        hidden_dropout=0.0,
    )


def test_grouped_query_attention_matches_captured_fused_gqa_output(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(41))
    load_legacy_model_state(model, legacy_arrays, legacy_control)

    actual = model.layers[0].self_attn(legacy_arrays["gqa.input"])

    assert actual.dtype == mx.bfloat16
    assert_close(actual, legacy_arrays["gqa.output"], atol=2e-2, rtol=2e-2)


def test_model_logits_and_fp32_loss_match_captured_reference(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(43))
    load_legacy_model_state(model, legacy_arrays, legacy_control)

    logits = model(legacy_arrays["model.input_ids"], training=False).logits
    loss = causal_lm_loss(
        logits,
        legacy_arrays["model.labels"],
        legacy_arrays["model.labels"] != model.config.pad_token_id,
    )

    assert logits.dtype == mx.bfloat16
    assert loss.dtype == mx.float32
    assert_close(logits, legacy_arrays["model.logits"], atol=2e-2, rtol=2e-2)
    assert_close(
        loss,
        legacy_arrays["model.loss"].astype(mx.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_fixed_capacity_cache_matches_full_sequential_and_chunked_references(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(47))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    input_ids = legacy_arrays["cache.input_ids"]
    capacity = input_ids.shape[1]

    full_logits = model(input_ids, training=False).logits
    sequential_cache = KVCache.allocate(
        model.config,
        batch_size=1,
        capacity=capacity,
        dtype=mx.bfloat16,
    )
    sequential_logits = mx.concatenate(
        [
            model(
                input_ids[:, index : index + 1],
                cache=sequential_cache,
                training=False,
            ).logits
            for index in range(capacity)
        ],
        axis=1,
    )
    chunked_cache = KVCache.allocate(
        model.config,
        batch_size=1,
        capacity=capacity,
        dtype=mx.bfloat16,
    )
    chunked_logits = mx.concatenate(
        [
            model(input_ids[:, :2], cache=chunked_cache, training=False).logits,
            model(input_ids[:, 2:4], cache=chunked_cache, training=False).logits,
            model(input_ids[:, 4:], cache=chunked_cache, training=False).logits,
        ],
        axis=1,
    )

    assert_close(full_logits, legacy_arrays["cache.full_logits"], atol=2e-2, rtol=2e-2)
    assert_close(
        sequential_logits,
        legacy_arrays["cache.sequential_logits"],
        atol=2e-2,
        rtol=2e-2,
    )
    assert_close(
        chunked_logits,
        legacy_arrays["cache.chunked_logits"],
        atol=2e-2,
        rtol=2e-2,
    )
    assert_close(full_logits, sequential_logits, atol=2e-2, rtol=2e-2)
    assert_close(full_logits, chunked_logits, atol=2e-2, rtol=2e-2)
    keys, values, lengths = chunked_cache.state
    assert_close(lengths, mx.array([capacity], dtype=mx.int32), atol=0.0, rtol=0.0)
    for layer_index in range(model.config.num_layers):
        assert_close(
            keys[layer_index],
            legacy_arrays[f"cache.chunked_key.{layer_index}"],
            atol=2e-2,
            rtol=2e-2,
        )
        assert_close(
            values[layer_index],
            legacy_arrays[f"cache.chunked_value.{layer_index}"],
            atol=2e-2,
            rtol=2e-2,
        )


def test_explicit_parameter_forward_supports_consecutive_compiled_fixture_calls(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(53))
    load_legacy_model_state(model, legacy_arrays, legacy_control)

    @mx.compile
    def compiled_forward(parameters, input_ids):
        logits, cache_state, next_key = model.forward_arrays(
            parameters,
            input_ids,
            attention_mask=None,
            positions=None,
            cache_state=None,
            training=False,
            key=None,
        )
        assert cache_state is None
        assert next_key is None
        return logits

    first = compiled_forward(
        model.parameters(), legacy_arrays["compiled_state.input_ids.0"]
    )
    mx.eval(first)
    second = compiled_forward(
        model.parameters(), legacy_arrays["compiled_state.input_ids.1"]
    )

    assert_close(first, legacy_arrays["compiled_state.logits.0"], atol=2e-2, rtol=2e-2)
    assert_close(second, legacy_arrays["compiled_state.logits.1"], atol=2e-2, rtol=2e-2)


def test_generation_sampling_matches_captured_legacy_primitive(legacy_arrays):
    """Seeded nucleus sampling must preserve the accepted legacy token choice."""
    result = select_next_token(
        legacy_arrays["generation.logits"],
        GenerationConfig(temperature=0.8, top_p=0.9),
        mx.random.key(1234),
    )
    expected = mx.squeeze(legacy_arrays["generation.sampled_token"], axis=-1)

    assert result.token_ids.shape == (1,)
    assert result.token_ids.dtype == mx.uint32
    assert_close(result.token_ids, expected, atol=0.0, rtol=0.0)
