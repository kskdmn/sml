import io
import json
import sys
import tempfile
from dataclasses import asdict
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


class FakeGenerationModel:
    def __init__(self, generated_ids):
        self.config = SimpleNamespace(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
            effective_max_position_embeddings=16,
        )
        self.generate = Spy(return_value=generated_ids)


class TestInferenceInterface:
    def tiny_config(self):
        from sml import SMLConfig

        return SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=16,
            rope_scaling_factor=2.0,
            hidden_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

    def test_encode_prompt_adds_bos_and_batch_dimension(self):
        mx = require_mlx_runtime()
        import infer_sml

        input_ids = infer_sml.encode_prompt(
            FakeTokenizer(),
            "4 5",
            bos_token_id=1,
        )

        mx.eval(input_ids)
        assert [[1, 4, 5]] == input_ids.tolist()
        assert mx.int32 == input_ids.dtype

    def test_decode_token_ids_omits_bos_pad_and_stops_at_eos(self):
        import infer_sml

        text = infer_sml.decode_token_ids(
            FakeTokenizer(),
            [1, 4, 3, 5, 2, 6],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

        assert "4 5" == text

    def test_load_model_restores_checkpoint_config_and_weights(self):
        mx = require_mlx_runtime()
        import infer_sml
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml"
            checkpoint_path.mkdir()
            (checkpoint_path / "metadata.json").write_text(
                json.JSONEncoder().encode({"model_config": asdict(config)}),
                encoding="utf-8",
            )
            model.save_weights(str(checkpoint_path / "model.safetensors"))

            loaded = infer_sml.load_model(checkpoint_path)

        assert not loaded.training
        output = loaded(mx.array([[1, 4, 5]], dtype=mx.int32))
        mx.eval(output.logits)
        assert (1, 3, config.vocab_size) == tuple(output.logits.shape)

    def test_load_checkpoint_metadata_requires_dictionary(self):
        import infer_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml"
            checkpoint_path.mkdir()
            (checkpoint_path / "metadata.json").write_text("[]", encoding="utf-8")

            with pytest.raises(ValueError, match="metadata must contain a dictionary"):
                infer_sml.load_checkpoint_metadata(checkpoint_path)

    def test_generate_text_omits_prompt_by_default(self, monkeypatch):
        mx = require_mlx_runtime()
        import infer_sml

        model = FakeGenerationModel(mx.array([[1, 4, 5, 6, 2]], dtype=mx.int32))
        monkeypatch.setattr(
            infer_sml,
            "load_tokenizer",
            Spy(return_value=FakeTokenizer()),
        )
        monkeypatch.setattr(infer_sml, "load_model", Spy(return_value=model))

        text = infer_sml.generate_text("4 5", max_new_tokens=2)

        assert "6" == text

    def test_resolve_max_new_tokens_uses_remaining_context_window_by_default(self):
        import infer_sml

        assert 13 == infer_sml.resolve_max_new_tokens(None, 16, 3)
        assert 5 == infer_sml.resolve_max_new_tokens(5, 16, 3)
        assert 0 == infer_sml.resolve_max_new_tokens(None, 16, 20)

    def test_generate_text_uses_remaining_context_window_by_default(self, monkeypatch):
        mx = require_mlx_runtime()
        import infer_sml

        model = FakeGenerationModel(mx.array([[1, 4, 5, 6, 2]], dtype=mx.int32))
        monkeypatch.setattr(
            infer_sml,
            "load_tokenizer",
            Spy(return_value=FakeTokenizer()),
        )
        monkeypatch.setattr(infer_sml, "load_model", Spy(return_value=model))

        text = infer_sml.generate_text("4 5")

        assert "6" == text
        assert 13 == model.generate.call_args.kwargs["max_new_tokens"]

    def test_main_omits_prompt_by_default(self, monkeypatch):
        import infer_sml

        generation_config = object()
        generate_text = Spy(return_value="hello world")
        print_ = Spy()
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr(
            infer_sml,
            "generation_config_from_args",
            Spy(return_value=generation_config),
        )
        monkeypatch.setattr("builtins.print", print_)

        exit_code = infer_sml.main(
            [
                "hello",
                "--max-new-tokens",
                "3",
            ]
        )

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            checkpoint_path=infer_sml.DEFAULT_MODEL_PATH,
            tokenizer_model_path=infer_sml.DEFAULT_TOKENIZER_MODEL_PATH,
            max_new_tokens=3,
            include_prompt=False,
            generation_config=generation_config,
        )
        print_.assert_called_once_with("hello world")

    def test_main_can_include_prompt_from_cli_args(self, monkeypatch):
        import infer_sml

        generation_config = object()
        generate_text = Spy(return_value="hello world")
        print_ = Spy()
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr(
            infer_sml,
            "generation_config_from_args",
            Spy(return_value=generation_config),
        )
        monkeypatch.setattr("builtins.print", print_)

        exit_code = infer_sml.main(["hello", "--include-prompt"])

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            checkpoint_path=infer_sml.DEFAULT_MODEL_PATH,
            tokenizer_model_path=infer_sml.DEFAULT_TOKENIZER_MODEL_PATH,
            max_new_tokens=None,
            include_prompt=True,
            generation_config=generation_config,
        )
        print_.assert_called_once_with("hello world")

    def test_main_passes_model_path_from_cli_args(self, monkeypatch):
        import infer_sml

        generation_config = object()
        generate_text = Spy(return_value="hello world")
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr(
            infer_sml,
            "generation_config_from_args",
            Spy(return_value=generation_config),
        )
        monkeypatch.setattr("builtins.print", Spy())

        exit_code = infer_sml.main(["hello", "--model", "/tmp/custom-sml"])

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            checkpoint_path=Path("/tmp/custom-sml"),
            tokenizer_model_path=infer_sml.DEFAULT_TOKENIZER_MODEL_PATH,
            max_new_tokens=None,
            include_prompt=False,
            generation_config=generation_config,
        )

    def test_main_passes_tokenizer_model_path_from_cli_args(self, monkeypatch):
        import infer_sml

        generation_config = object()
        generate_text = Spy(return_value="hello world")
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr(
            infer_sml,
            "generation_config_from_args",
            Spy(return_value=generation_config),
        )
        monkeypatch.setattr("builtins.print", Spy())

        exit_code = infer_sml.main(
            ["hello", "--tokenizer-model", "/tmp/custom-tokenizer.model"]
        )

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            checkpoint_path=infer_sml.DEFAULT_MODEL_PATH,
            tokenizer_model_path=Path("/tmp/custom-tokenizer.model"),
            max_new_tokens=None,
            include_prompt=False,
            generation_config=generation_config,
        )

    def test_parse_args_accepts_decoding_flags(self):
        import infer_sml

        args = infer_sml.parse_args(
            [
                "hello",
                "--temperature",
                "0.8",
                "--top-p",
                "0.9",
                "--repetition-penalty",
                "1.2",
                "--no-repeat-ngram-size",
                "3",
                "--seed",
                "7",
            ]
        )

        assert 0.8 == args.temperature
        assert 0.9 == args.top_p
        assert 1.2 == args.repetition_penalty
        assert 3 == args.no_repeat_ngram_size
        assert 7 == args.seed

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("--completion-only", None),
        ],
    )
    def test_parse_args_rejects_fixed_inference_defaults(
        self,
        option,
        value,
        monkeypatch,
    ):
        import infer_sml

        argv = ["hello", option] if value is None else ["hello", option, value]
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        with pytest.raises(SystemExit):
            infer_sml.parse_args(argv)
