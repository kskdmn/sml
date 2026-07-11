import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from helpers import Spy
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


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


class TestEvalUtils:
    def test_loglikelihood_scores_only_continuation_tokens(self):
        require_mlx_runtime()
        import eval_utils

        model = FakeModel()
        lm = eval_utils.SMLEvalLM(
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
        import eval_utils

        model = FakeModel()
        lm = eval_utils.SMLEvalLM(
            model=model,
            tokenizer=BoundaryTokenizer(),
        )
        request = SimpleNamespace(args=("a", "b"))

        result = lm.loglikelihood([request])

        assert result[0][0] > -0.001
        assert result[0][1]
        assert [[[1, 10, 20]]] == model.calls

    def test_loglikelihood_rejects_sequences_beyond_checkpoint_context(self):
        import eval_utils

        lm = eval_utils.SMLEvalLM(
            model=FakeModel(effective_max_position_embeddings=3),
            tokenizer=FakeTokenizer(),
        )
        request = SimpleNamespace(args=("4 5", " 6"))

        with pytest.raises(ValueError, match="prompt plus continuation"):
            lm.loglikelihood([request])

    def test_generate_until_caps_completion_and_applies_earliest_stop(self):
        require_mlx_runtime()
        import eval_utils

        model = FakeModel(effective_max_position_embeddings=6)
        lm = eval_utils.SMLEvalLM(
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
        import eval_utils

        lm = object()
        expected = {"results": {"hellaswag": {"acc,none": 0.0}}}
        simple_evaluate = Spy(return_value=expected)
        monkeypatch.setattr(eval_utils, "simple_evaluate", simple_evaluate)

        result = eval_utils.evaluate_lm(
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
            batch_size=1,
            limit=2,
            log_samples=False,
        )

    def test_build_eval_parser_accepts_model_option_for_checkpoint_path(self):
        import eval_utils

        parser = eval_utils.build_eval_parser(
            description="Evaluate.",
            default_results_path=Path("results.json"),
            limit_help="limit examples",
        )

        args = parser.parse_args(["--model", "/tmp/custom-sml"])

        assert Path("/tmp/custom-sml") == args.checkpoint

    def test_build_eval_parser_accepts_tokenizer_model_option(self):
        import eval_utils

        parser = eval_utils.build_eval_parser(
            description="Evaluate.",
            default_results_path=Path("results.json"),
            limit_help="limit examples",
        )

        args = parser.parse_args(["--tokenizer-model", "/tmp/custom-tokenizer.model"])

        assert Path("/tmp/custom-tokenizer.model") == args.tokenizer_model

    def test_write_results_creates_parent_directory_and_serializes_paths(self):
        import eval_utils

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "results.json"

            eval_utils.write_results(
                output_path,
                {"checkpoint": Path("v2/output/sml")},
            )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        assert "v2/output/sml" == saved["checkpoint"]
