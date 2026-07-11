import sys
import unittest
import os
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


class FakeModel:
    def __init__(self, effective_max_position_embeddings=6):
        self.config = SimpleNamespace(
            effective_max_position_embeddings=effective_max_position_embeddings,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
        self.max_new_tokens = None

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


class HumanEvalAdapterTest(unittest.TestCase):
    def test_generate_until_caps_completion_and_applies_earliest_stop(self):
        import eval_humaneval

        model = FakeModel(effective_max_position_embeddings=6)
        lm = eval_humaneval.SMLHumanEvalLM(
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

    def test_generate_until_rejects_prompt_beyond_checkpoint_context(self):
        import eval_humaneval

        lm = eval_humaneval.SMLHumanEvalLM(
            model=FakeModel(effective_max_position_embeddings=3),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(
            args=("4 5 6", {"max_gen_toks": 1, "until": [], "do_sample": False})
        )

        with self.assertRaisesRegex(ValueError, "HumanEval prompt"):
            lm.generate_until([request])

    def test_generate_until_rejects_sampling(self):
        import eval_humaneval

        lm = eval_humaneval.SMLHumanEvalLM(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        request = SimpleNamespace(
            args=("4", {"max_gen_toks": 1, "until": [], "do_sample": True})
        )

        with self.assertRaisesRegex(ValueError, "greedy generation"):
            lm.generate_until([request])

    def test_evaluate_humaneval_confirms_unsafe_code_execution(self):
        import eval_humaneval

        lm = mock.Mock()
        expected = {"results": {"humaneval": {"pass@1,none": 0.0}}}

        def evaluate(**kwargs):
            self.assertEqual("1", os.environ.get("HF_ALLOW_CODE_EVAL"))
            return expected

        with mock.patch.object(
            eval_humaneval,
            "evaluate_lm",
            side_effect=evaluate,
        ) as evaluate_lm:
            result = eval_humaneval.evaluate_humaneval(
                lm=lm,
                checkpoint_path=Path("v1/output/sml.pt"),
                limit=2,
            )

        self.assertIs(expected, result)
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=Path("v1/output/sml.pt"),
            tasks=["humaneval"],
            limit=2,
            confirm_run_unsafe_code=True,
        )

    def test_main_loads_default_checkpoint_and_runs_limited_evaluation(self):
        import eval_humaneval
        from config import TOKENIZER_MODEL_PATH

        lm = mock.Mock()
        results = {"results": {"humaneval": {"pass@1,none": 0.0}}}
        with (
            mock.patch.object(
                eval_humaneval.SMLHumanEvalLM,
                "from_checkpoint",
                return_value=lm,
            ) as from_checkpoint,
            mock.patch.object(
                eval_humaneval,
                "evaluate_humaneval",
                return_value=results,
            ) as evaluate_humaneval,
            mock.patch.object(
                eval_humaneval,
                "write_results",
            ) as write_results,
            mock.patch.object(
                eval_humaneval,
                "make_table",
                return_value="table",
            ),
            mock.patch("builtins.print") as print_output,
        ):
            return_code = eval_humaneval.main(["--device", "cpu", "--limit", "2"])

        self.assertEqual(0, return_code)
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_humaneval.DEFAULT_CHECKPOINT_PATH,
            tokenizer_model_path=TOKENIZER_MODEL_PATH,
            device_name="cpu",
        )
        evaluate_humaneval.assert_called_once_with(
            lm=lm,
            checkpoint_path=eval_humaneval.DEFAULT_CHECKPOINT_PATH,
            limit=2,
        )
        write_results.assert_called_once_with(
            eval_humaneval.DEFAULT_RESULTS_PATH,
            results,
        )
        print_output.assert_called_once_with("table")


if __name__ == "__main__":
    unittest.main()
