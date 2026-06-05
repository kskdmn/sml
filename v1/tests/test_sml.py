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
    def test_default_config_matches_existing_tokenizer_vocab(self):
        from sml_config import SMLConfig

        config = SMLConfig()

        self.assertEqual(49_152, config.vocab_size)
        self.assertEqual(0, config.unk_token_id)
        self.assertEqual(1, config.bos_token_id)
        self.assertEqual(2, config.eos_token_id)
        self.assertEqual(3, config.pad_token_id)
        self.assertEqual(0, config.head_dim % 2)
        self.assertTrue(config.gradient_checkpointing)

    def test_invalid_attention_shape_is_rejected(self):
        from sml_config import SMLConfig

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            SMLConfig(hidden_size=30, num_q_heads=8)

        with self.assertRaisesRegex(ValueError, "num_q_heads"):
            SMLConfig(num_q_heads=6, num_kv_heads=4)


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
            max_position_embeddings=32,
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
