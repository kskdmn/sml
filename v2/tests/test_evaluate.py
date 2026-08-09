import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import Spy

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - depends on host
    mx = None
    _MLX_IMPORT_ERROR = exc
else:
    _MLX_IMPORT_ERROR = None


def require_mlx_runtime():
    if _MLX_IMPORT_ERROR is not None:
        pytest.skip(f"mlx is not available: {_MLX_IMPORT_ERROR}")
    try:
        import mlx.nn  # noqa: F401

        mx.eval(mx.array([0]))
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - depends on host
        pytest.skip(f"mlx is not available: {exc}")
    return mx


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
        mx = require_mlx_runtime()
        mx.eval(input_ids)
        self.calls.append(input_ids.tolist())
        next_token_ids = mx.concatenate(
            [
                input_ids[:, 1:],
                mx.full((input_ids.shape[0], 1), 0, dtype=mx.int32),
            ],
            axis=1,
        )
        vocab_ids = mx.arange(128, dtype=mx.int32).reshape(1, 1, 128)
        logits = mx.where(
            vocab_ids == next_token_ids[:, :, None],
            mx.full(
                (*input_ids.shape, 128),
                100.0,
                dtype=mx.float32,
            ),
            mx.full(
                (*input_ids.shape, 128),
                -100.0,
                dtype=mx.float32,
            ),
        )
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, max_new_tokens, eos_token_id):
        mx = require_mlx_runtime()
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        continuation = mx.arange(
            6,
            6 + max_new_tokens,
            dtype=mx.int32,
        ).reshape(1, max_new_tokens)
        return mx.concatenate((input_ids, continuation), axis=1)


class TestSMLEvalLM:
    def test_loglikelihood_scores_only_continuation_tokens(self):
        require_mlx_runtime()
        import evaluate_sml

        model = FakeModel()
        lm = evaluate_sml.SMLEvalLM(
            model=model,
            tokenizer=FakeTokenizer(),
        )
        request = SimpleNamespace(args=("4 5", " 6 7"))

        result = lm.loglikelihood([request])

        assert 1 == len(result)
        logprob, is_greedy = result[0]
        assert logprob > -0.001
        assert is_greedy
        assert [[[1, 4, 5, 6, 7]]] == model.calls

    def test_loglikelihood_tokenizes_context_and_continuation_together(self):
        require_mlx_runtime()
        import evaluate_sml

        model = FakeModel()
        lm = evaluate_sml.SMLEvalLM(
            model=model,
            tokenizer=BoundaryTokenizer(),
        )
        request = SimpleNamespace(args=("a", "b"))

        result = lm.loglikelihood([request])

        assert result[0][0] > -0.001
        assert result[0][1]
        assert [[[1, 10, 20]]] == model.calls

    def test_loglikelihood_batches_padded_requests(self):
        require_mlx_runtime()
        import evaluate_sml

        model = FakeModel()
        lm = evaluate_sml.SMLEvalLM(
            model=model,
            tokenizer=FakeTokenizer(),
        )
        requests = [
            SimpleNamespace(args=("4 5", " 6")),
            SimpleNamespace(args=("7", " 8 9")),
        ]

        result = lm.loglikelihood(requests)

        assert 2 == len(result)
        assert result[0][0] > -0.001
        assert result[0][1]
        assert result[1][0] > -0.001
        assert result[1][1]
        assert [[[1, 4, 5, 6], [1, 7, 8, 9]]] == model.calls

    def test_loglikelihood_rejects_sequences_beyond_checkpoint_context(self):
        import evaluate_sml

        lm = evaluate_sml.SMLEvalLM(
            model=FakeModel(effective_max_position_embeddings=3),
            tokenizer=FakeTokenizer(),
        )
        request = SimpleNamespace(args=("4 5", " 6"))

        with pytest.raises(ValueError, match="prompt plus continuation"):
            lm.loglikelihood([request])

    def test_generate_until_caps_completion_and_applies_earliest_stop(self):
        require_mlx_runtime()
        import evaluate_sml

        model = FakeModel(effective_max_position_embeddings=6)
        lm = evaluate_sml.SMLEvalLM(
            model=model,
            tokenizer=FakeTokenizer(),
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

        assert ["6"] == result
        assert 3 == model.max_new_tokens
        assert 2 == model.eos_token_id

    def test_evaluate_lm_passes_common_lm_eval_options(self, monkeypatch):
        import evaluate_sml

        lm = object()
        expected = {"results": {"hellaswag": {"acc,none": 0.0}}}
        simple_evaluate = Spy(return_value=expected)
        monkeypatch.setattr(evaluate_sml, "simple_evaluate", simple_evaluate)

        result = evaluate_sml.evaluate_lm(
            lm=lm,
            checkpoint_path=Path("v2/output/sml"),
            tasks=["hellaswag"],
            limit=2,
        )

        assert expected is result
        simple_evaluate.assert_called_once_with(
            model=lm,
            model_args={"path": "v2/output/sml"},
            tasks=["hellaswag"],
            num_fewshot=0,
            batch_size=evaluate_sml.DEFAULT_LOGLIKELIHOOD_BATCH_SIZE,
            limit=2,
            log_samples=False,
        )

    def test_write_results_creates_parent_directory_and_serializes_paths(self):
        import evaluate_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "results.json"

            evaluate_sml.write_results(
                output_path,
                {"checkpoint": Path("v2/output/sml")},
            )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        assert "v2/output/sml" == saved["checkpoint"]


class TestEvalCli:
    def patch_eval_pipeline(self, monkeypatch, lm, results):
        import evaluate_sml

        from_checkpoint = Spy(return_value=lm)
        evaluate_lm = Spy(return_value=results)
        write_results = Spy()
        print_output = Spy()
        monkeypatch.setattr(evaluate_sml.SMLEvalLM, "from_checkpoint", from_checkpoint)
        monkeypatch.setattr(evaluate_sml, "evaluate_lm", evaluate_lm)
        monkeypatch.setattr(evaluate_sml, "write_results", write_results)
        monkeypatch.setattr(evaluate_sml, "make_table", Spy(return_value="table"))
        monkeypatch.setattr("builtins.print", print_output)
        return from_checkpoint, evaluate_lm, write_results, print_output

    @pytest.mark.parametrize("benchmark", ["hellaswag", "winogrande"])
    def test_main_loads_default_checkpoint_and_writes_results(
        self, monkeypatch, benchmark
    ):
        import evaluate_sml
        from config import DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_MODEL_PATH, OUTPUT_DIR

        lm = object()
        results = {"results": {benchmark: {"acc,none": 0.0}}}
        from_checkpoint, evaluate_lm, write_results, print_output = (
            self.patch_eval_pipeline(monkeypatch, lm, results)
        )

        return_code = evaluate_sml.main(["--benchmark", benchmark, "--limit", "2"])

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=DEFAULT_MODEL_PATH,
            tokenizer_model_path=DEFAULT_TOKENIZER_MODEL_PATH,
        )
        evaluate_lm.assert_called_once_with(
            lm=lm,
            checkpoint_path=DEFAULT_MODEL_PATH,
            tasks=[benchmark],
            limit=2,
        )
        write_results.assert_called_once_with(
            OUTPUT_DIR / f"{benchmark}.json",
            results,
        )
        print_output.assert_called_once_with("table")

    def test_main_accepts_model_path_alias(self, monkeypatch):
        import evaluate_sml
        from config import DEFAULT_TOKENIZER_MODEL_PATH

        lm = object()
        results = {"results": {"hellaswag": {"acc,none": 0.0}}}
        from_checkpoint, _, _, _ = self.patch_eval_pipeline(monkeypatch, lm, results)

        return_code = evaluate_sml.main(
            ["--benchmark", "hellaswag", "--model", "/tmp/custom-sml"]
        )

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=Path("/tmp/custom-sml"),
            tokenizer_model_path=DEFAULT_TOKENIZER_MODEL_PATH,
        )

    def test_main_accepts_tokenizer_model_path(self, monkeypatch):
        import evaluate_sml
        from config import DEFAULT_MODEL_PATH

        lm = object()
        results = {"results": {"hellaswag": {"acc,none": 0.0}}}
        from_checkpoint, _, _, _ = self.patch_eval_pipeline(monkeypatch, lm, results)

        return_code = evaluate_sml.main(
            ["--benchmark", "hellaswag", "--tokenizer-model", "/tmp/custom.model"]
        )

        assert 0 == return_code
        from_checkpoint.assert_called_once_with(
            checkpoint_path=DEFAULT_MODEL_PATH,
            tokenizer_model_path=Path("/tmp/custom.model"),
        )

    def test_main_accepts_output_path(self, monkeypatch):
        import evaluate_sml

        lm = object()
        results = {"results": {"winogrande": {"acc,none": 0.0}}}
        _, _, write_results, _ = self.patch_eval_pipeline(monkeypatch, lm, results)

        return_code = evaluate_sml.main(
            ["--benchmark", "winogrande", "--output", "/tmp/custom-results.json"]
        )

        assert 0 == return_code
        write_results.assert_called_once_with(
            Path("/tmp/custom-results.json"),
            results,
        )

    def test_main_requires_benchmark_option(self, monkeypatch, capsys):
        import evaluate_sml

        with pytest.raises(SystemExit):
            evaluate_sml.parse_args([])
        assert "--benchmark" in capsys.readouterr().err

    def test_main_rejects_unknown_benchmark(self, monkeypatch, capsys):
        import evaluate_sml

        with pytest.raises(SystemExit):
            evaluate_sml.parse_args(["--benchmark", "mmlu"])
        assert "invalid choice" in capsys.readouterr().err
