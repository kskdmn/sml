import sys
from pathlib import Path

from helpers import Spy


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class TestHellaSwagEval:
    def test_evaluate_hellaswag_uses_lm_eval_task(self, monkeypatch):
        import eval_hellaswag

        lm = object()
        expected = {"results": {"hellaswag": {"acc,none": 0.0}}}
        evaluate_lm = Spy(return_value=expected)
        monkeypatch.setattr(eval_hellaswag, "evaluate_lm", evaluate_lm)

        result = eval_hellaswag.evaluate_hellaswag(
            lm=lm,
            checkpoint_path=Path("v2/output/sml"),
            limit=3,
        )

        assert expected is result
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=Path("v2/output/sml"),
            tasks=["hellaswag"],
            limit=3,
        )

    def test_main_loads_default_checkpoint_and_writes_results(self, monkeypatch):
        import eval_hellaswag

        lm = object()
        results = {"results": {"hellaswag": {"acc,none": 0.0}}}
        from_checkpoint = Spy(return_value=lm)
        evaluate_hellaswag = Spy(return_value=results)
        write_results = Spy()
        print_output = Spy()
        monkeypatch.setattr(
            eval_hellaswag.SMLEvalLM,
            "from_checkpoint",
            from_checkpoint,
        )
        monkeypatch.setattr(eval_hellaswag, "evaluate_hellaswag", evaluate_hellaswag)
        monkeypatch.setattr(eval_hellaswag, "write_results", write_results)
        monkeypatch.setattr(eval_hellaswag, "make_table", Spy(return_value="table"))
        monkeypatch.setattr("builtins.print", print_output)

        return_code = eval_hellaswag.main(["--limit", "2"])

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_hellaswag.DEFAULT_CHECKPOINT_PATH,
            tokenizer_model_path=eval_hellaswag.TOKENIZER_MODEL_PATH,
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
