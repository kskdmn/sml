import io
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
    import torch
except ImportError:  # pragma: no cover - exercised only before torch is installed
    torch = None


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


@pytest.mark.skipif(torch is None, reason="torch is not installed")
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
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_checkpointing=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

    def test_encode_prompt_adds_bos_and_batch_dimension(self):
        import infer_sml

        input_ids = infer_sml.encode_prompt(
            FakeTokenizer(),
            "4 5",
            bos_token_id=1,
            device=torch.device("cpu"),
        )

        assert torch.equal(torch.tensor([[1, 4, 5]]), input_ids)

    def test_decode_token_ids_omits_bos_pad_and_stops_at_eos(self):
        import infer_sml

        text = infer_sml.decode_token_ids(
            FakeTokenizer(),
            [1, 4, 3, 5, 2, 6],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

        assert '4 5' == text

    def test_load_model_restores_checkpoint_config_and_weights(self):
        import infer_sml
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml.pt"
            torch.save(
                {
                    "model_config": asdict(config),
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            loaded = infer_sml.load_model(checkpoint_path, torch.device("cpu"))

        assert not loaded.training
        output = loaded(torch.tensor([[1, 4, 5]]))
        assert (1, 3, config.vocab_size) == tuple(output.logits.shape)

    def test_load_model_maps_legacy_max_position_embeddings_config(self):
        import infer_sml
        from sml import SMLLanguageModel

        config = self.tiny_config()
        model = SMLLanguageModel(config)
        legacy_model_config = asdict(config)
        legacy_model_config["max_position_embeddings"] = legacy_model_config.pop(
            "original_max_position_embeddings"
        )
        legacy_model_config.pop("rope_scaling_factor")

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml.pt"
            torch.save(
                {
                    "model_config": legacy_model_config,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            loaded = infer_sml.load_model(checkpoint_path, torch.device("cpu"))

        assert config.original_max_position_embeddings == loaded.config.original_max_position_embeddings
        assert 2.0 == loaded.config.rope_scaling_factor

    def test_generate_text_omits_prompt_by_default(self, monkeypatch):
        import infer_sml

        model = FakeGenerationModel(torch.tensor([[1, 4, 5, 6, 2]]))
        monkeypatch.setattr(
            infer_sml,
            "resolve_device",
            Spy(return_value=torch.device("cpu")),
        )
        monkeypatch.setattr(
            infer_sml,
            "load_tokenizer",
            Spy(return_value=FakeTokenizer()),
        )
        monkeypatch.setattr(infer_sml, "load_model", Spy(return_value=model))

        text = infer_sml.generate_text("4 5", max_new_tokens=2)

        assert '6' == text

    def test_resolve_max_new_tokens_uses_remaining_context_window_by_default(self):
        import infer_sml

        assert 13 == infer_sml.resolve_max_new_tokens(None, 16, 3)
        assert 5 == infer_sml.resolve_max_new_tokens(5, 16, 3)
        assert 0 == infer_sml.resolve_max_new_tokens(None, 16, 20)

    def test_generate_text_uses_remaining_context_window_by_default(self, monkeypatch):
        import infer_sml

        model = FakeGenerationModel(torch.tensor([[1, 4, 5, 6, 2]]))
        monkeypatch.setattr(
            infer_sml,
            "resolve_device",
            Spy(return_value=torch.device("cpu")),
        )
        monkeypatch.setattr(
            infer_sml,
            "load_tokenizer",
            Spy(return_value=FakeTokenizer()),
        )
        monkeypatch.setattr(infer_sml, "load_model", Spy(return_value=model))

        text = infer_sml.generate_text("4 5")

        assert '6' == text
        assert 13 == model.generate.call_args.kwargs['max_new_tokens']

    def test_create_completion_response_uses_openai_compatible_shape(self, monkeypatch):
        import infer_sml

        generate_text = Spy(return_value="6")
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)

        response = infer_sml.create_completion_response(
            {
                "model": "sml-test",
                "prompt": "4 5",
                "max_tokens": 2,
            }
        )

        generate_text.assert_called_once_with(
            prompt="4 5",
            max_new_tokens=2,
            include_prompt=False,
        )
        assert 'text_completion' == response['object']
        assert 'sml-test' == response['model']
        assert {'index': 0, 'text': '6', 'finish_reason': 'length'} == response['choices'][0]
        assert {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3} == response['usage']

    def test_create_chat_completion_response_formats_messages(self, monkeypatch):
        import infer_sml

        generate_text = Spy(return_value="6")
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)

        response = infer_sml.create_chat_completion_response(
            {
                "model": "sml-test",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "4 5"},
                ],
                "max_tokens": 3,
            }
        )

        generate_text.assert_called_once_with(
            prompt="system: be terse\nuser: 4 5\nassistant:",
            max_new_tokens=3,
            include_prompt=False,
        )
        assert 'chat.completion' == response['object']
        assert 'sml-test' == response['model']
        assert {'index': 0, 'message': {'role': 'assistant', 'content': '6'}, 'finish_reason': 'length'} == response['choices'][0]
        assert {'prompt_tokens': 7, 'completion_tokens': 1, 'total_tokens': 8} == response['usage']

    def test_create_models_response_lists_default_model(self):
        import infer_sml

        response = infer_sml.create_models_response("sml-test")

        assert 'list' == response['object']
        assert 'sml-test' == response['data'][0]['id']
        assert 'model' == response['data'][0]['object']

    def test_route_openai_request_dispatches_chat_completions(self, monkeypatch):
        import infer_sml

        generate_text = Spy(return_value="6")
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)

        status_code, response = infer_sml.route_openai_request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "sml-test",
                "messages": [{"role": "user", "content": "4 5"}],
                "max_tokens": 2,
            },
        )

        assert 200 == status_code
        generate_text.assert_called_once_with(
            prompt="user: 4 5\nassistant:",
            max_new_tokens=2,
            include_prompt=False,
        )
        assert 'chat.completion' == response['object']

    def test_route_openai_request_returns_404_for_unknown_paths(self):
        import infer_sml

        status_code, response = infer_sml.route_openai_request(
            "GET",
            "/v1/unknown",
        )

        assert 404 == status_code
        assert 'not_found' == response['error']['type']

    def test_main_omits_prompt_by_default(self, monkeypatch):
        import infer_sml
        from sml import GenerationConfig

        generate_text = Spy(return_value="hello world")
        print_ = Spy()
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
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
            max_new_tokens=3,
            device_name="auto",
            include_prompt=False,
            generation_config=GenerationConfig(),
        )
        print_.assert_called_once_with("hello world")

    def test_main_passes_device_from_cli_args(self, monkeypatch):
        import infer_sml
        from sml import GenerationConfig

        generate_text = Spy(return_value="hello world")
        print_ = Spy()
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr("builtins.print", print_)

        exit_code = infer_sml.main(["hello", "--device", "cuda:0"])

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            max_new_tokens=None,
            device_name="cuda:0",
            include_prompt=False,
            generation_config=GenerationConfig(),
        )
        print_.assert_called_once_with("hello world")

    def test_main_can_include_prompt_from_cli_args(self, monkeypatch):
        import infer_sml
        from sml import GenerationConfig

        generate_text = Spy(return_value="hello world")
        print_ = Spy()
        monkeypatch.setattr(infer_sml, "generate_text", generate_text)
        monkeypatch.setattr("builtins.print", print_)

        exit_code = infer_sml.main(["hello", "--include-prompt"])

        assert 0 == exit_code
        generate_text.assert_called_once_with(
            prompt="hello",
            max_new_tokens=None,
            device_name="auto",
            include_prompt=True,
            generation_config=GenerationConfig(),
        )
        print_.assert_called_once_with("hello world")

    def test_parse_args_accepts_decoding_flags(self):
        import infer_sml
        from sml import GenerationConfig

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

        assert GenerationConfig(temperature=0.8, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3, seed=7) == infer_sml.generation_config_from_args(args)

    def test_main_can_start_openai_compatible_server(self, monkeypatch):
        import infer_sml

        run_server = Spy()
        monkeypatch.setattr(infer_sml, "run_openai_compatible_server", run_server)

        exit_code = infer_sml.main(
            [
                "--serve",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--model",
                "sml-test",
            ]
        )

        assert 0 == exit_code
        run_server.assert_called_once_with(
            host="0.0.0.0",
            port=9000,
            model_name="sml-test",
            device_name="auto",
        )

    def test_main_passes_device_to_openai_compatible_server(self, monkeypatch):
        import infer_sml

        run_server = Spy()
        monkeypatch.setattr(infer_sml, "run_openai_compatible_server", run_server)

        exit_code = infer_sml.main(
            [
                "--serve",
                "--device",
                "cpu",
            ]
        )

        assert 0 == exit_code
        run_server.assert_called_once_with(
            host="127.0.0.1",
            port=8000,
            model_name=infer_sml.DEFAULT_MODEL_NAME,
            device_name="cpu",
        )

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("--checkpoint", "model.pt"),
            ("--tokenizer", "tokenizer.model"),
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

    def test_parse_args_defaults_device_to_auto(self):
        import infer_sml

        args = infer_sml.parse_args(["hello"])

        assert 'auto' == args.device
