import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import torch
except ImportError:  # pragma: no cover - exercised only before torch is installed
    torch = None


class SMLConfigTest(unittest.TestCase):
    def test_default_config_targets_small_model_size(self):
        from sml import estimate_model_size
        from sml_config import SMLConfig

        config = SMLConfig()

        self.assertEqual(24_576, config.vocab_size)
        self.assertEqual(512, config.hidden_size)
        self.assertEqual(12, config.num_layers)
        self.assertEqual(8, config.num_q_heads)
        self.assertEqual(2, config.num_kv_heads)
        self.assertEqual(1_536, config.intermediate_size)
        self.assertEqual(48_771_584, estimate_model_size(config))
        self.assertEqual(0, config.unk_token_id)
        self.assertEqual(1, config.bos_token_id)
        self.assertEqual(2, config.eos_token_id)
        self.assertEqual(3, config.pad_token_id)
        self.assertEqual(0, config.head_dim % 2)
        self.assertEqual(1_024, config.original_max_position_embeddings)
        self.assertEqual(2.0, config.rope_scaling_factor)
        self.assertEqual(32.0, config.yarn_beta_fast)
        self.assertEqual(1.0, config.yarn_beta_slow)
        self.assertEqual(2_048, config.effective_max_position_embeddings)
        self.assertFalse(config.gradient_checkpointing)

    def test_invalid_attention_shape_is_rejected(self):
        from sml_config import SMLConfig

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            SMLConfig(hidden_size=30, num_q_heads=8)

        with self.assertRaisesRegex(ValueError, "num_q_heads"):
            SMLConfig(num_q_heads=6, num_kv_heads=4)

    def test_model_config_for_training_disables_yarn(self):
        from sml_config import SMLConfig, model_config_for_training

        config = SMLConfig(rope_scaling_factor=2.0)
        training_config = model_config_for_training(config)

        self.assertEqual(2.0, config.rope_scaling_factor)
        self.assertEqual(2_048, config.effective_max_position_embeddings)
        self.assertEqual(1.0, training_config.rope_scaling_factor)
        self.assertEqual(1_024, training_config.effective_max_position_embeddings)

    def test_invalid_context_scaling_is_rejected(self):
        from sml_config import SMLConfig

        with self.assertRaisesRegex(ValueError, "original_max_position_embeddings"):
            SMLConfig(original_max_position_embeddings=0)

        with self.assertRaisesRegex(ValueError, "rope_scaling_factor"):
            SMLConfig(rope_scaling_factor=0.0)

        with self.assertRaisesRegex(ValueError, "yarn_beta_fast"):
            SMLConfig(yarn_beta_fast=1.0, yarn_beta_slow=1.0)

    def test_yarn_mscale_fields_must_be_set_together(self):
        from sml_config import SMLConfig

        with self.assertRaisesRegex(ValueError, "yarn_mscale and yarn_mscale_all_dim"):
            SMLConfig(yarn_mscale=1.0)

    def test_effective_max_position_embeddings_scales_with_large_rope_factor(self):
        from sml_config import SMLConfig

        config = SMLConfig(
            original_max_position_embeddings=1_024,
            rope_scaling_factor=8.0,
        )

        self.assertEqual(8_192, config.effective_max_position_embeddings)


class SMLScheduleTest(unittest.TestCase):
    def test_lr_schedule_stays_constant_after_warmup_without_total_steps(self):
        from sml import lr_lambda

        self.assertEqual(
            1.0,
            lr_lambda(
                step=1_000,
                total_steps=None,
                warmup_steps=100,
                min_lr_ratio=0.1,
            ),
        )


@unittest.skipIf(torch is None, "torch is not installed")
class SMLModelTest(unittest.TestCase):
    def tiny_config(self):
        from sml_config import SMLConfig

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
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids, labels=input_ids)

        self.assertEqual((2, 8, config.vocab_size), tuple(output.logits.shape))
        self.assertEqual(0, output.loss.dim())

    def test_rotary_cache_covers_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        rope = model.layers[0].self_attn.rope

        self.assertEqual(
            config.effective_max_position_embeddings,
            rope.cos_cached.shape[0],
        )
        self.assertEqual(
            config.effective_max_position_embeddings,
            rope.sin_cached.shape[0],
        )

    def test_forward_accepts_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (1, config.effective_max_position_embeddings),
        )

        output = model(input_ids)

        self.assertEqual(
            (1, config.effective_max_position_embeddings, config.vocab_size),
            tuple(output.logits.shape),
        )

    def test_forward_rejects_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (1, config.effective_max_position_embeddings + 1),
        )

        with self.assertRaisesRegex(ValueError, "effective_max_position_embeddings"):
            model(input_ids)

    def test_embedding_and_lm_head_weights_are_tied(self):
        from sml import SMLLanguageModel

        model = SMLLanguageModel(self.tiny_config())

        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)

    def test_training_forward_uses_gradient_checkpointing_when_enabled(self):
        import torch.utils.checkpoint
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        with mock.patch(
            "torch.utils.checkpoint.checkpoint",
            wraps=torch.utils.checkpoint.checkpoint,
        ) as checkpoint:
            output = model(input_ids, labels=input_ids)
            self.assertEqual(config.num_layers, checkpoint.call_count)

        self.assertEqual(0, output.loss.dim())
        output.loss.backward()
        self.assertIsNotNone(model.embed_tokens.weight.grad)
        for call in checkpoint.call_args_list:
            self.assertIs(False, call.kwargs["use_reentrant"])

    def test_generate_appends_requested_tokens_and_uses_cache(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = torch.tensor([[config.bos_token_id, 5, 6]], dtype=torch.long)

        generated = model.generate(prompt, max_new_tokens=4)

        self.assertEqual((1, 7), tuple(generated.shape))
        self.assertTrue(torch.equal(prompt, generated[:, : prompt.shape[1]]))

    def test_generate_rejects_requested_tokens_beyond_scaled_context_length(self):
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        prompt = torch.full(
            (1, config.effective_max_position_embeddings),
            config.bos_token_id,
            dtype=torch.long,
        )

        with self.assertRaisesRegex(ValueError, "effective_max_position_embeddings"):
            model.generate(prompt, max_new_tokens=1)

    def test_forward_accepts_large_rope_scaling_factor(self):
        from dataclasses import replace

        from sml import SMLLanguageModel

        config = replace(
            self.tiny_config(),
            original_max_position_embeddings=16,
            rope_scaling_factor=4.0,
        )
        model = SMLLanguageModel(config)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (1, config.effective_max_position_embeddings),
        )

        output = model(input_ids)

        self.assertEqual(
            (1, config.effective_max_position_embeddings, config.vocab_size),
            tuple(output.logits.shape),
        )
        self.assertEqual(64, model.layers[0].self_attn.rope.cos_cached.shape[0])

    def test_resolve_yarn_attention_factor_supports_mscale_ratio(self):
        from sml import resolve_yarn_attention_factor, yarn_get_mscale

        factor = 8.0
        expected = yarn_get_mscale(factor, 1.0) / yarn_get_mscale(factor, 0.5)

        self.assertAlmostEqual(
            expected,
            resolve_yarn_attention_factor(
                factor,
                mscale=1.0,
                mscale_all_dim=0.5,
            ),
        )

    def test_yarn_find_correction_range_clamps_to_rotary_band_count(self):
        from sml import yarn_find_correction_range

        low, high = yarn_find_correction_range(
            low_rot=1.0,
            high_rot=1.0,
            rotary_dim=16,
            base=10_000.0,
            original_max_position_embeddings=1_024,
        )

        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 7)

    def test_causal_lm_loss_uses_aligned_next_token_labels(self):
        from sml import compute_causal_lm_loss

        logits = torch.full((1, 3, 5), -10.0)
        labels = torch.tensor([[1, 2, 3]], dtype=torch.long)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 3] = 10.0

        loss = compute_causal_lm_loss(logits, labels, pad_token_id=4)

        self.assertLess(loss.item(), 0.001)


if __name__ == "__main__":
    unittest.main()
