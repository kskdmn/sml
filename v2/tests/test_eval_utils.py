import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class FakeTokenizer:
    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]

    def decode(self, ids):
        return " ".join(str(token_id) for token_id in ids)


class BoundaryTokenizer:
    def encode(self, text, out_type=int):
        del out_type
        if text == "a":
            return [10]
        if text == "ab":
            return [10, 20]
        if text == "b":
            return [99]
        raise AssertionError(f"unexpected text: {text!r}")

    def decode(self, ids):
        return " ".join(str(token_id) for token_id in ids)


class FakeModel:
    def __init__(self, effective_max_position_embeddings=8):
        self.config = SimpleNamespace(
            effective_max_position_embeddings=effective_max_position_embeddings,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
        self.calls = []
        self.max_new_tokens = None

    def __call__(self, input_ids):
        self.calls.append(input_ids.detach().cpu().tolist())
        logits = torch.full(
            (*input_ids.shape, 128),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for position in range(input_ids.shape[1] - 1):
            next_token_id = int(input_ids[0, position + 1])
            logits[0, position, next_token_id] = 100.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens, eos_token_id):
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        continuation = torch.arange(
            6,
            6 + max_new_tokens,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        return torch.cat((input_ids, continuation), dim=1)


class EvalUtilsTest(unittest.TestCase):
    def test_loglikelihood_scores_only_continuation_tokens(self):
        import eval_utils

        model = FakeModel()
        lm = eval_utils.SMLEvalLM(
            model=model,
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(args=("4 5", " 6 7"))

        result = lm.loglikelihood([request])

        self.assertEqual(1, len(result))
        logprob, is_greedy = result[0]
        self.assertGreater(logprob, -0.001)
        self.assertTrue(is_greedy)
        self.assertEqual([[[1, 4, 5, 6, 7]]], model.calls)

    def test_loglikelihood_tokenizes_context_and_continuation_together(self):
        import eval_utils

        model = FakeModel()
        lm = eval_utils.SMLEvalLM(
            model=model,
            tokenizer=BoundaryTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(args=("a", "b"))

        result = lm.loglikelihood([request])

        self.assertGreater(result[0][0], -0.001)
        self.assertTrue(result[0][1])
        self.assertEqual([[[1, 10, 20]]], model.calls)

    def test_loglikelihood_rejects_sequences_beyond_checkpoint_context(self):
        import eval_utils

        lm = eval_utils.SMLEvalLM(
            model=FakeModel(effective_max_position_embeddings=3),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(args=("4 5", " 6"))

        with self.assertRaisesRegex(ValueError, "prompt plus continuation"):
            lm.loglikelihood([request])

    def test_generate_until_caps_completion_and_applies_earliest_stop(self):
        import eval_utils

        model = FakeModel(effective_max_position_embeddings=6)
        lm = eval_utils.SMLEvalLM(
            model=model,
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(
            args=(
                "4 5",
                {
                    "max_gen_toks": 10,
                    "until": [" 7", " 8"],
                    "do_sample": False,
                },
            )
        )

        result = lm.generate_until([request])

        self.assertEqual(["6"], result)
        self.assertEqual(3, model.max_new_tokens)
        self.assertEqual(2, model.eos_token_id)

    def test_evaluate_lm_passes_common_lm_eval_options(self):
        import eval_utils

        lm = mock.Mock()
        expected = {"results": {"hellaswag": {"acc,none": 0.0}}}
        with mock.patch.object(
            eval_utils,
            "simple_evaluate",
            return_value=expected,
        ) as simple_evaluate:
            result = eval_utils.evaluate_lm(
                lm=lm,
                checkpoint_path=Path("v1/output/sml.pt"),
                tasks=["hellaswag"],
                limit=2,
            )

        self.assertIs(expected, result)
        simple_evaluate.assert_called_once_with(
            model=lm,
            model_args={"path": "v1/output/sml.pt"},
            tasks=["hellaswag"],
            num_fewshot=0,
            batch_size=1,
            limit=2,
            log_samples=False,
        )

    def test_write_results_creates_parent_directory_and_serializes_paths(self):
        import eval_utils

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "results.json"

            eval_utils.write_results(
                output_path,
                {"checkpoint": Path("v1/output/sml.pt")},
            )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("v1/output/sml.pt", saved["checkpoint"])


if __name__ == "__main__":
    unittest.main()
