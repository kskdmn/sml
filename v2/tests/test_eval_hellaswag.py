import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class HellaSwagEvalTest(unittest.TestCase):
    def test_evaluate_hellaswag_uses_lm_eval_task(self):
        import eval_hellaswag

        lm = mock.Mock()
        expected = {"results": {"hellaswag": {"acc,none": 0.0}}}
        with mock.patch.object(
            eval_hellaswag,
            "evaluate_lm",
            return_value=expected,
        ) as evaluate_lm:
            result = eval_hellaswag.evaluate_hellaswag(
                lm=lm,
                checkpoint_path=Path("v1/output/sml.pt"),
                limit=3,
            )

        self.assertIs(expected, result)
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=Path("v1/output/sml.pt"),
            tasks=["hellaswag"],
            limit=3,
        )

    def test_main_loads_default_checkpoint_and_writes_results(self):
        import eval_hellaswag

        lm = mock.Mock()
        results = {"results": {"hellaswag": {"acc,none": 0.0}}}
        with (
            mock.patch.object(
                eval_hellaswag.SMLEvalLM,
                "from_checkpoint",
                return_value=lm,
            ) as from_checkpoint,
            mock.patch.object(
                eval_hellaswag,
                "evaluate_hellaswag",
                return_value=results,
            ) as evaluate_hellaswag,
            mock.patch.object(eval_hellaswag, "write_results") as write_results,
            mock.patch.object(eval_hellaswag, "make_table", return_value="table"),
            mock.patch("builtins.print") as print_output,
        ):
            return_code = eval_hellaswag.main(["--device", "cpu", "--limit", "2"])

        self.assertEqual(0, return_code)
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_hellaswag.DEFAULT_CHECKPOINT_PATH,
            tokenizer_model_path=eval_hellaswag.TOKENIZER_MODEL_PATH,
            device_name="cpu",
        )
        evaluate_hellaswag.assert_called_once_with(
            lm=lm,
            checkpoint_path=eval_hellaswag.DEFAULT_CHECKPOINT_PATH,
            limit=2,
        )
        write_results.assert_called_once_with(
            eval_hellaswag.DEFAULT_RESULTS_PATH,
            results,
        )
        print_output.assert_called_once_with("table")


if __name__ == "__main__":
    unittest.main()
