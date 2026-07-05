import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class WinograndeEvalTest(unittest.TestCase):
    def test_evaluate_winogrande_uses_lm_eval_task(self):
        import eval_winogrande

        lm = mock.Mock()
        expected = {"results": {"winogrande": {"acc,none": 0.0}}}
        with mock.patch.object(
            eval_winogrande,
            "evaluate_lm",
            return_value=expected,
        ) as evaluate_lm:
            result = eval_winogrande.evaluate_winogrande(
                lm=lm,
                checkpoint_path=Path("v1/output/sml.pt"),
                limit=3,
            )

        self.assertIs(expected, result)
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=Path("v1/output/sml.pt"),
            tasks=["winogrande"],
            limit=3,
        )

    def test_main_loads_default_checkpoint_and_writes_results(self):
        import eval_winogrande

        lm = mock.Mock()
        results = {"results": {"winogrande": {"acc,none": 0.0}}}
        with (
            mock.patch.object(
                eval_winogrande.SMLEvalLM,
                "from_checkpoint",
                return_value=lm,
            ) as from_checkpoint,
            mock.patch.object(
                eval_winogrande,
                "evaluate_winogrande",
                return_value=results,
            ) as evaluate_winogrande,
            mock.patch.object(eval_winogrande, "write_results") as write_results,
            mock.patch.object(eval_winogrande, "make_table", return_value="table"),
            mock.patch("builtins.print") as print_output,
        ):
            return_code = eval_winogrande.main(["--device", "cpu", "--limit", "2"])

        self.assertEqual(0, return_code)
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_winogrande.DEFAULT_CHECKPOINT_PATH,
            tokenizer_model_path=eval_winogrande.TOKENIZER_MODEL_PATH,
            device_name="cpu",
        )
        evaluate_winogrande.assert_called_once_with(
            lm=lm,
            checkpoint_path=eval_winogrande.DEFAULT_CHECKPOINT_PATH,
            limit=2,
        )
        write_results.assert_called_once_with(
            eval_winogrande.DEFAULT_RESULTS_PATH,
            results,
        )
        print_output.assert_called_once_with("table")


if __name__ == "__main__":
    unittest.main()
