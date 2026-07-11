import sys
from pathlib import Path

from helpers import Spy


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class TestWinograndeEval:
    def test_evaluate_winogrande_uses_lm_eval_task(self, monkeypatch):
        import eval_winogrande

        lm = object()
        expected = {"results": {"winogrande": {"acc,none": 0.0}}}
        evaluate_lm = Spy(return_value=expected)
        monkeypatch.setattr(eval_winogrande, "evaluate_lm", evaluate_lm)

        result = eval_winogrande.evaluate_winogrande(
            lm=lm,
            checkpoint_path=Path("v2/output/sml"),
            limit=3,
        )

        assert expected is result
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=Path("v2/output/sml"),
            tasks=["winogrande"],
            limit=3,
        )

    def test_main_loads_default_checkpoint_and_writes_results(self, monkeypatch):
        import eval_winogrande

        lm = object()
        results = {"results": {"winogrande": {"acc,none": 0.0}}}
        from_checkpoint = Spy(return_value=lm)
        evaluate_winogrande = Spy(return_value=results)
        write_results = Spy()
        print_output = Spy()
        monkeypatch.setattr(
            eval_winogrande.SMLEvalLM,
            "from_checkpoint",
            from_checkpoint,
        )
        monkeypatch.setattr(eval_winogrande, "evaluate_winogrande", evaluate_winogrande)
        monkeypatch.setattr(eval_winogrande, "write_results", write_results)
        monkeypatch.setattr(eval_winogrande, "make_table", Spy(return_value="table"))
        monkeypatch.setattr("builtins.print", print_output)

        return_code = eval_winogrande.main(["--limit", "2"])

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_winogrande.DEFAULT_MODEL_PATH,
            tokenizer_model_path=eval_winogrande.DEFAULT_TOKENIZER_MODEL_PATH,
        )
        evaluate_winogrande.assert_called_once_with(
            lm=lm,
            checkpoint_path=eval_winogrande.DEFAULT_MODEL_PATH,
            limit=2,
        )
        write_results.assert_called_once_with(
            eval_winogrande.DEFAULT_RESULTS_PATH,
            results,
        )
        print_output.assert_called_once_with("table")

    def test_main_accepts_model_path_alias(self, monkeypatch):
        import eval_winogrande

        lm = object()
        results = {"results": {"winogrande": {"acc,none": 0.0}}}
        from_checkpoint = Spy(return_value=lm)
        monkeypatch.setattr(
            eval_winogrande.SMLEvalLM,
            "from_checkpoint",
            from_checkpoint,
        )
        monkeypatch.setattr(eval_winogrande, "evaluate_winogrande", Spy(return_value=results))
        monkeypatch.setattr(eval_winogrande, "write_results", Spy())
        monkeypatch.setattr(eval_winogrande, "make_table", Spy(return_value="table"))
        monkeypatch.setattr("builtins.print", Spy())

        return_code = eval_winogrande.main(["--model", "/tmp/custom-sml"])

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=Path("/tmp/custom-sml"),
            tokenizer_model_path=eval_winogrande.DEFAULT_TOKENIZER_MODEL_PATH,
        )

    def test_main_accepts_tokenizer_model_path(self, monkeypatch):
        import eval_winogrande

        lm = object()
        results = {"results": {"winogrande": {"acc,none": 0.0}}}
        from_checkpoint = Spy(return_value=lm)
        monkeypatch.setattr(
            eval_winogrande.SMLEvalLM,
            "from_checkpoint",
            from_checkpoint,
        )
        monkeypatch.setattr(eval_winogrande, "evaluate_winogrande", Spy(return_value=results))
        monkeypatch.setattr(eval_winogrande, "write_results", Spy())
        monkeypatch.setattr(eval_winogrande, "make_table", Spy(return_value="table"))
        monkeypatch.setattr("builtins.print", Spy())

        return_code = eval_winogrande.main(
            ["--tokenizer-model", "/tmp/custom-tokenizer.model"]
        )

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=eval_winogrande.DEFAULT_MODEL_PATH,
            tokenizer_model_path=Path("/tmp/custom-tokenizer.model"),
        )
