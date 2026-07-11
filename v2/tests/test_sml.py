import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import mlx.core as mx
    import mlx.nn
except (ImportError, RuntimeError) as exc:
    pytestmark = pytest.mark.skip(reason=f"mlx is not available: {exc}")


class TestSMLConfig:
    def test_default_config_targets_small_model_size(self):
        from sml import estimate_model_size
        from sml import SMLConfig

        config = SMLConfig()

        assert 24576 == config.vocab_size
        assert 512 == config.hidden_size
        assert 12 == config.num_layers
        assert 8 == config.num_q_heads
        assert 2 == config.num_kv_heads
        assert 1536 == config.intermediate_size
        assert 48771584 == estimate_model_size(config)
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
        assert not config.gradient_checkpointing

    def test_invalid_attention_shape_is_rejected(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match='hidden_size'):
            SMLConfig(hidden_size=30, num_q_heads=8)

        with pytest.raises(ValueError, match='num_q_heads'):
            SMLConfig(num_q_heads=6, num_kv_heads=4)

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

        with pytest.raises(ValueError, match='original_max_position_embeddings'):
            SMLConfig(original_max_position_embeddings=0)

        with pytest.raises(ValueError, match='rope_scaling_factor'):
            SMLConfig(rope_scaling_factor=0.0)

        with pytest.raises(ValueError, match='yarn_beta_fast'):
            SMLConfig(yarn_beta_fast=1.0, yarn_beta_slow=1.0)

    def test_yarn_mscale_fields_must_be_set_together(self):
        from sml import SMLConfig

        with pytest.raises(ValueError, match='yarn_mscale and yarn_mscale_all_dim'):
            SMLConfig(yarn_mscale=1.0)

    def test_effective_max_position_embeddings_scales_with_large_rope_factor(self):
        from sml import SMLConfig

        config = SMLConfig(
            original_max_position_embeddings=1_024,
            rope_scaling_factor=8.0,
        )

        assert 8192 == config.effective_max_position_embeddings


class TestSMLSchedule:
    def test_lr_schedule_stays_constant_after_warmup_without_total_steps(self):
        from sml import lr_lambda

        assert 1.0 == lr_lambda(step=1000, total_steps=None, warmup_steps=100, min_lr_ratio=0.1)


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
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_checkpointing=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

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

        assert (1, config.effective_max_position_embeddings, config.vocab_size) == output.logits.shape

    def test_forward_rejects_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = mx.random.randint(
            0,
            config.vocab_size,
            shape=(1, config.effective_max_position_embeddings + 1),
        )

        with pytest.raises(ValueError, match='effective_max_position_embeddings'):
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

    def test_generate_rejects_requested_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = mx.full(
            (1, config.effective_max_position_embeddings),
            config.bos_token_id,
            dtype=mx.int32,
        )

        with pytest.raises(ValueError, match='effective_max_position_embeddings'):
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

    def test_generate_uses_repetition_penalty_without_sampling(self):
        from sml import GenerationConfig
        from sml import SMLLanguageModel

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
        from sml import GenerationConfig
        from sml import SMLLanguageModel

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

        assert (1, config.effective_max_position_embeddings, config.vocab_size) == output.logits.shape
        assert 64 == model.layers[0].self_attn.rope.cos_cached.shape[0]

    def test_resolve_yarn_attention_factor_supports_mscale_ratio(self):
        from sml import resolve_yarn_attention_factor, yarn_get_mscale

        factor = 8.0
        expected = yarn_get_mscale(factor, 1.0) / yarn_get_mscale(factor, 0.5)

        assert resolve_yarn_attention_factor(factor, mscale=1.0, mscale_all_dim=0.5) == pytest.approx(expected)

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
