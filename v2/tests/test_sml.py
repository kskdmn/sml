import sys

import pytest

try:
    import mlx.core as mx
    import mlx.nn  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytestmark = pytest.mark.skip(reason=f"mlx is not available: {exc}")


class TestSMLConfig:
    def test_default_config_targets_small_model_size(self):
        from sml import SMLConfig, count_parameters, create_model, estimate_model_size

        config = SMLConfig()

        assert 28672 == config.vocab_size
        assert 768 == config.hidden_size
        assert 12 == config.num_layers
        assert 12 == config.num_q_heads
        assert 3 == config.num_kv_heads
        assert 2176 == config.intermediate_size
        assert 99896064 == estimate_model_size(config)
        total_params, _ = count_parameters(create_model(config))
        assert total_params == estimate_model_size(config)
        assert 0 == config.unk_token_id
        assert 1 == config.bos_token_id
        assert 2 == config.eos_token_id
        assert 3 == config.pad_token_id
        assert 0 == config.head_dim % 2
        assert 1024 == config.original_max_position_embeddings
        assert 4.0 == config.rope_scaling_factor
        assert 32.0 == config.yarn_beta_fast
        assert 1.0 == config.yarn_beta_slow
        assert 4096 == config.effective_max_position_embeddings
        assert not hasattr(config, "attention_dropout")
        assert not hasattr(config, "gradient_checkpointing")

    def test_default_initializer_ranges_depth_scale_residual_outputs(self):
        import math

        from sml import SMLConfig

        config = SMLConfig(num_layers=8)
        initializer_range = config.parameter_initializer_range
        residual_initializer_range = config.initializer_range / math.sqrt(
            2 * config.num_layers
        )

        assert initializer_range.embed_tokens == pytest.approx(0.02)
        assert initializer_range.lm_head == pytest.approx(0.02)
        assert initializer_range.q_proj == pytest.approx(0.02)
        assert initializer_range.k_proj == pytest.approx(0.02)
        assert initializer_range.v_proj == pytest.approx(0.02)
        assert initializer_range.o_proj == pytest.approx(residual_initializer_range)
        assert initializer_range.gate_proj == pytest.approx(0.02)
        assert initializer_range.up_proj == pytest.approx(0.02)
        assert initializer_range.down_proj == pytest.approx(residual_initializer_range)
        assert initializer_range.other == pytest.approx(0.02)

    def test_parameter_initializer_range_config_rejects_unset_values(self):
        from sml import ParameterInitializerRangeConfig

        with pytest.raises(ValueError, match="q_proj"):
            ParameterInitializerRangeConfig(q_proj=None)

    def test_parameter_initializer_range_config_validates_initializer_range_with_member_method(
        self,
    ):
        from sml import ParameterInitializerRangeConfig

        config = ParameterInitializerRangeConfig()

        config.validate_initializer_range(0.0, "q_proj")
        with pytest.raises(ValueError, match="q_proj"):
            config.validate_initializer_range(float("inf"), "q_proj")

    def test_invalid_attention_shape_is_rejected(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match="hidden_size"):
            SMLConfig(hidden_size=30, num_q_heads=8)

        with pytest.raises(ValueError, match="num_q_heads"):
            SMLConfig(num_q_heads=6, num_kv_heads=4)

    def test_invalid_positive_shape_fields_are_rejected(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match="num_layers"):
            SMLConfig(num_layers=0)

        with pytest.raises(ValueError, match="num_q_heads"):
            SMLConfig(num_q_heads=0)

        with pytest.raises(ValueError, match="num_kv_heads"):
            SMLConfig(num_kv_heads=0)

        with pytest.raises(ValueError, match="rope_theta"):
            SMLConfig(rope_theta=0.0)

        with pytest.raises(ValueError, match="hidden_dropout"):
            SMLConfig(hidden_dropout=-0.1)

    def test_model_config_for_training_disables_yarn(self):
        from sml import SMLConfig
        from train_sml import model_config_for_training

        config = SMLConfig(rope_scaling_factor=2.0)
        training_config = model_config_for_training(config)

        assert 2.0 == config.rope_scaling_factor
        assert 2048 == config.effective_max_position_embeddings
        assert 1.0 == training_config.rope_scaling_factor
        assert 1024 == training_config.effective_max_position_embeddings

    def test_invalid_context_scaling_is_rejected(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match="original_max_position_embeddings"):
            SMLConfig(original_max_position_embeddings=0)

        with pytest.raises(ValueError, match="rope_scaling_factor"):
            SMLConfig(rope_scaling_factor=0.0)

        with pytest.raises(ValueError, match="yarn_beta_fast"):
            SMLConfig(yarn_beta_fast=1.0, yarn_beta_slow=1.0)

    def test_yarn_mscale_fields_must_be_set_together(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match="yarn_mscale and yarn_mscale_all_dim"):
            SMLConfig(yarn_mscale=1.0)

    def test_effective_max_position_embeddings_scales_with_large_rope_factor(self):
        from sml import SMLConfig

        config = SMLConfig(
            original_max_position_embeddings=1_024,
            rope_scaling_factor=8.0,
        )

        assert 8192 == config.effective_max_position_embeddings


class TestSMLModel:
    def tiny_config(self):
        from sml import SMLConfig

        return SMLConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            original_max_position_embeddings=32,
            rope_scaling_factor=2.0,
            hidden_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

    def test_model_uses_per_parameter_initializer_ranges(self, monkeypatch):
        legacy = sys.modules["sml._legacy"]
        from sml import (
            ParameterInitializerRangeConfig,
            SMLConfig,
            SMLLanguageModel,
        )

        def fake_normal(*, shape, scale):
            return mx.full(shape, scale)

        def assert_full(array, value):
            assert bool(mx.allclose(array, mx.full(array.shape, value)).item())

        monkeypatch.setattr(legacy.mx.random, "normal", fake_normal)
        config = SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=16,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
            pad_token_id=None,
            tie_word_embeddings=False,
            parameter_initializer_range=ParameterInitializerRangeConfig(
                embed_tokens=0.011,
                lm_head=0.012,
                q_proj=0.013,
                k_proj=0.014,
                v_proj=0.015,
                o_proj=0.016,
                gate_proj=0.017,
                up_proj=0.018,
                down_proj=0.019,
            ),
        )

        model = SMLLanguageModel(config)
        block = model.layers[0]

        assert_full(model.embed_tokens.weight, 0.011)
        assert_full(model.lm_head.weight, 0.012)
        assert_full(block.self_attn.q_proj.weight, 0.013)
        assert_full(block.self_attn.k_proj.weight, 0.014)
        assert_full(block.self_attn.v_proj.weight, 0.015)
        assert_full(block.self_attn.o_proj.weight, 0.016)
        assert_full(block.mlp.gate_proj.weight, 0.017)
        assert_full(block.mlp.up_proj.weight, 0.018)
        assert_full(block.mlp.down_proj.weight, 0.019)
        assert_full(block.input_norm.weight, 1.0)

    def test_forward_returns_logits_and_causal_lm_loss(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = mx.random.randint(0, config.vocab_size, shape=(2, 8))

        output = model(input_ids, labels=input_ids)
        mx.eval(output.logits, output.loss)

        assert (2, 8, config.vocab_size) == output.logits.shape
        assert output.loss is not None
        assert () == output.loss.shape

    def test_rotary_cache_covers_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        rope = model.layers[0].self_attn.rope

        assert config.effective_max_position_embeddings == rope.cos_cached.shape[0]
        assert config.effective_max_position_embeddings == rope.sin_cached.shape[0]

    def test_forward_accepts_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = mx.random.randint(
            0,
            config.vocab_size,
            shape=(1, config.effective_max_position_embeddings),
        )

        output = model(input_ids)
        mx.eval(output.logits)

        assert (
            1,
            config.effective_max_position_embeddings,
            config.vocab_size,
        ) == output.logits.shape

    def test_forward_rejects_out_of_vocabulary_input_ids(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)

        with pytest.raises(ValueError, match="input_ids must be within"):
            model(mx.array([[10, config.vocab_size]], dtype=mx.int32))
        with pytest.raises(ValueError, match="input_ids must be within"):
            model(mx.array([[-1, 10]], dtype=mx.int32))

    def test_forward_rejects_out_of_vocabulary_labels(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = mx.array([[10, 20]], dtype=mx.int32)

        with pytest.raises(ValueError, match="labels must be within"):
            model(input_ids, labels=mx.array([[20, config.vocab_size]], dtype=mx.int32))

    def test_forward_rejects_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = mx.random.randint(
            0,
            config.vocab_size,
            shape=(1, config.effective_max_position_embeddings + 1),
        )

        with pytest.raises(ValueError, match="effective_max_position_embeddings"):
            model(input_ids)

    def test_embedding_and_lm_head_weights_are_tied(self):
        from sml import SMLLanguageModel

        model = SMLLanguageModel(self.tiny_config())

        assert model.lm_head.weight is model.embed_tokens.weight

    def test_generate_appends_requested_tokens_and_uses_cache(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)

        generated = model.generate(prompt, max_new_tokens=4)
        mx.eval(generated)

        assert (1, 7) == generated.shape
        assert bool(mx.all(prompt == generated[:, : prompt.shape[1]]).item())

    def test_generate_preallocates_kv_cache_to_requested_length(self, monkeypatch):
        legacy = sys.modules["sml._legacy"]
        from sml import SMLLanguageModel

        created_cache_lengths = []
        original_kv_cache = legacy.KVCache

        class RecordingKVCache(original_kv_cache):
            def __init__(self, max_seq_len=None):
                created_cache_lengths.append(max_seq_len)
                super().__init__(max_seq_len=max_seq_len)

        monkeypatch.setattr(legacy, "KVCache", RecordingKVCache)
        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)

        generated = model.generate(prompt, max_new_tokens=4)
        mx.eval(generated)

        assert [7] == created_cache_lengths

    def test_kv_cache_preallocates_capacity_and_tracks_logical_length(self):
        from sml import KVCache

        cache = KVCache(max_seq_len=8)
        key = mx.ones((1, 2, 3, 4))
        value = mx.full((1, 2, 3, 4), 2.0)

        cached_key, cached_value = cache.update(0, key, value)
        mx.eval(cached_key, cached_value)

        assert 3 == cache.get_seq_len(0)
        assert 8 == cache.key_cache[0].shape[2]
        assert (1, 2, 3, 4) == cached_key.shape
        assert (1, 2, 3, 4) == cached_value.shape

        next_key = mx.full((1, 2, 1, 4), 3.0)
        next_value = mx.full((1, 2, 1, 4), 4.0)
        cached_key, cached_value = cache.update(0, next_key, next_value)
        mx.eval(cached_key, cached_value)

        assert 4 == cache.get_seq_len(0)
        assert 8 == cache.key_cache[0].shape[2]
        assert (1, 2, 4, 4) == cached_key.shape
        assert bool(mx.allclose(cached_key[:, :, :3, :], key).item())
        assert bool(mx.allclose(cached_key[:, :, 3:, :], next_key).item())

    def test_cached_multi_token_chunk_matches_sequential_decode(self):
        from sml import KVCache, SMLLanguageModel

        mx.random.seed(0)
        config = self.tiny_config()
        model = SMLLanguageModel(config)
        # Token ids must stay below config.vocab_size; out-of-range ids make the
        # GPU embedding lookup read undefined memory and the test nondeterministic.
        prompt = mx.array([[10, 20, 30]], dtype=mx.int32)
        chunk = mx.array([[25, 28]], dtype=mx.int32)

        kv_sequential = KVCache()
        model(prompt, kv_cache=kv_sequential)
        logits_first = model(mx.array([[25]]), kv_cache=kv_sequential).logits[0, 0]
        logits_second = model(mx.array([[28]]), kv_cache=kv_sequential).logits[0, 0]

        kv_chunk = KVCache()
        model(prompt, kv_cache=kv_chunk)
        chunk_logits = model(chunk, kv_cache=kv_chunk).logits
        mx.eval(logits_first, logits_second, chunk_logits)

        assert bool(
            mx.allclose(logits_first, chunk_logits[0, 0], rtol=1e-5, atol=1e-3).item()
        )
        assert bool(
            mx.allclose(logits_second, chunk_logits[0, 1], rtol=1e-5, atol=1e-3).item()
        )

    def test_generate_rejects_requested_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.full(
            (1, config.effective_max_position_embeddings),
            config.bos_token_id,
            dtype=mx.int32,
        )

        with pytest.raises(ValueError, match="effective_max_position_embeddings"):
            model.generate(prompt, max_new_tokens=1)

    def test_repetition_penalty_reduces_repeated_token_logit(self):
        from sml import apply_repetition_penalty

        logits = mx.array([[1.0, 2.0, 3.0]])
        input_ids = mx.array([[1, 1, 2]])
        adjusted = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        mx.eval(adjusted)

        assert 1.0 == pytest.approx(adjusted[0, 0].item())
        assert 1.0 == pytest.approx(adjusted[0, 1].item())
        assert 1.5 == pytest.approx(adjusted[0, 2].item())

    def test_no_repeat_ngram_blocks_repeated_trigram(self):
        from sml import apply_no_repeat_ngram

        logits = mx.array([[0.0, 0.0, 0.0, 5.0]])
        generated = mx.array([[0, 1, 2, 0, 1]])
        adjusted = apply_no_repeat_ngram(logits, generated, ngram_size=3)
        mx.eval(adjusted)

        assert bool(mx.isinf(adjusted[0, 2]).item())
        assert 5.0 == pytest.approx(adjusted[0, 3].item())

    def test_no_repeat_ngram_blocks_repeated_unigrams(self):
        from sml import apply_no_repeat_ngram

        logits = mx.array([[0.0, 1.0, 2.0, 3.0]])
        generated = mx.array([[1, 3]])
        adjusted = apply_no_repeat_ngram(logits, generated, ngram_size=1)
        mx.eval(adjusted)

        assert 0.0 == pytest.approx(adjusted[0, 0].item())
        assert bool(mx.isinf(adjusted[0, 1]).item())
        assert 2.0 == pytest.approx(adjusted[0, 2].item())
        assert bool(mx.isinf(adjusted[0, 3]).item())

    def test_generate_uses_repetition_penalty_without_sampling(self):
        from sml import GenerationConfig, SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)

        greedy = model.generate(
            prompt,
            max_new_tokens=3,
            generation_config=GenerationConfig(repetition_penalty=1.0),
        )
        penalized = model.generate(
            prompt,
            max_new_tokens=3,
            generation_config=GenerationConfig(repetition_penalty=10.0),
        )
        mx.eval(greedy, penalized)

        assert (1, 6) == greedy.shape
        assert (1, 6) == penalized.shape
        assert not bool(mx.all(greedy == penalized).item())

    def test_generate_sampling_is_reproducible_with_seed(self):
        from sml import GenerationConfig, SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)
        generation_config = GenerationConfig(temperature=0.8, top_p=0.9, seed=123)

        first = model.generate(
            prompt,
            max_new_tokens=4,
            generation_config=generation_config,
        )
        second = model.generate(
            prompt,
            max_new_tokens=4,
            generation_config=generation_config,
        )
        mx.eval(first, second)

        assert bool(mx.all(first == second).item())

    def test_forward_accepts_large_rope_scaling_factor(self):
        from dataclasses import replace

        from sml import SMLLanguageModel

        config = replace(
            self.tiny_config(),
            original_max_position_embeddings=16,
            rope_scaling_factor=4.0,
        )
        model = SMLLanguageModel(config)
        input_ids = mx.random.randint(
            0,
            config.vocab_size,
            shape=(1, config.effective_max_position_embeddings),
        )

        output = model(input_ids)
        mx.eval(output.logits)

        assert (
            1,
            config.effective_max_position_embeddings,
            config.vocab_size,
        ) == output.logits.shape
        assert 64 == model.layers[0].self_attn.rope.cos_cached.shape[0]

    def test_resolve_yarn_attention_factor_supports_mscale_ratio(self):
        from sml import resolve_yarn_attention_factor, yarn_get_mscale

        factor = 8.0
        expected = yarn_get_mscale(factor, 1.0) / yarn_get_mscale(factor, 0.5)

        assert resolve_yarn_attention_factor(
            factor, mscale=1.0, mscale_all_dim=0.5
        ) == pytest.approx(expected)

    def test_yarn_find_correction_range_clamps_to_rotary_band_count(self):
        from sml import yarn_find_correction_range

        low, high = yarn_find_correction_range(
            low_rot=1.0,
            high_rot=1.0,
            rotary_dim=16,
            base=10_000.0,
            original_max_position_embeddings=1_024,
        )

        assert low >= 0
        assert high <= 7

    def test_yarn_find_correction_range_preserves_fractional_bounds(self):
        from sml import yarn_find_correction_dim, yarn_find_correction_range

        expected_low = yarn_find_correction_dim(32.0, 64, 10_000.0, 1_024)
        expected_high = yarn_find_correction_dim(1.0, 64, 10_000.0, 1_024)
        low, high = yarn_find_correction_range(
            low_rot=32.0,
            high_rot=1.0,
            rotary_dim=64,
            base=10_000.0,
            original_max_position_embeddings=1_024,
            truncate=False,
        )

        assert low == pytest.approx(expected_low)
        assert high == pytest.approx(expected_high)
        assert not low.is_integer()
        assert not high.is_integer()

    def test_causal_lm_loss_uses_aligned_next_token_labels(self):
        from sml import compute_causal_lm_loss

        logits = mx.array(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0],
                    [-10.0, -10.0, 10.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0, 10.0, -10.0],
                ]
            ]
        )
        labels = mx.array([[1, 2, 3]], dtype=mx.int32)

        loss = compute_causal_lm_loss(logits, labels, pad_token_id=4)
        mx.eval(loss)

        assert loss.item() < 0.001
