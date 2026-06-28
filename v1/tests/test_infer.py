import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock


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


@unittest.skipIf(torch is None, "torch is not installed")
class InferenceInterfaceTest(unittest.TestCase):
    def tiny_config(self):
        from sml_config import SMLConfig

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

        self.assertTrue(torch.equal(torch.tensor([[1, 4, 5]]), input_ids))

    def test_decode_token_ids_omits_bos_pad_and_stops_at_eos(self):
        import infer_sml

        text = infer_sml.decode_token_ids(
            FakeTokenizer(),
            [1, 4, 3, 5, 2, 6],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

        self.assertEqual("4 5", text)

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

        self.assertFalse(loaded.training)
        output = loaded(torch.tensor([[1, 4, 5]]))
        self.assertEqual((1, 3, config.vocab_size), tuple(output.logits.shape))

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

        self.assertEqual(
            config.original_max_position_embeddings,
            loaded.config.original_max_position_embeddings,
        )
        self.assertEqual(2.0, loaded.config.rope_scaling_factor)

    def test_generate_text_omits_prompt_by_default(self):
        import infer_sml

        model = mock.Mock()
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.config.pad_token_id = 3
        model.generate.return_value = torch.tensor([[1, 4, 5, 6, 2]])

        with (
            mock.patch.object(infer_sml, "resolve_device", return_value=torch.device("cpu")),
            mock.patch.object(infer_sml, "load_tokenizer", return_value=FakeTokenizer()),
            mock.patch.object(infer_sml, "load_model", return_value=model),
        ):
            text = infer_sml.generate_text("4 5", max_new_tokens=2)

        self.assertEqual("6", text)

    def test_create_completion_response_uses_openai_compatible_shape(self):
        import infer_sml

        with mock.patch.object(
            infer_sml,
            "generate_text",
            return_value="6",
        ) as generate_text:
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
        self.assertEqual("text_completion", response["object"])
        self.assertEqual("sml-test", response["model"])
        self.assertEqual(
            {
                "index": 0,
                "text": "6",
                "finish_reason": "length",
            },
            response["choices"][0],
        )
        self.assertEqual(
            {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
            response["usage"],
        )

    def test_create_chat_completion_response_formats_messages(self):
        import infer_sml

        with mock.patch.object(
            infer_sml,
            "generate_text",
            return_value="6",
        ) as generate_text:
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
        self.assertEqual("chat.completion", response["object"])
        self.assertEqual("sml-test", response["model"])
        self.assertEqual(
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "6",
                },
                "finish_reason": "length",
            },
            response["choices"][0],
        )
        self.assertEqual(
            {
                "prompt_tokens": 7,
                "completion_tokens": 1,
                "total_tokens": 8,
            },
            response["usage"],
        )

    def test_create_models_response_lists_default_model(self):
        import infer_sml

        response = infer_sml.create_models_response("sml-test")

        self.assertEqual("list", response["object"])
        self.assertEqual("sml-test", response["data"][0]["id"])
        self.assertEqual("model", response["data"][0]["object"])

    def test_route_openai_request_dispatches_chat_completions(self):
        import infer_sml

        with mock.patch.object(
            infer_sml,
            "generate_text",
            return_value="6",
        ) as generate_text:
            status_code, response = infer_sml.route_openai_request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "sml-test",
                    "messages": [{"role": "user", "content": "4 5"}],
                    "max_tokens": 2,
                },
            )

        self.assertEqual(200, status_code)
        generate_text.assert_called_once_with(
            prompt="user: 4 5\nassistant:",
            max_new_tokens=2,
            include_prompt=False,
        )
        self.assertEqual("chat.completion", response["object"])

    def test_route_openai_request_returns_404_for_unknown_paths(self):
        import infer_sml

        status_code, response = infer_sml.route_openai_request(
            "GET",
            "/v1/unknown",
        )

        self.assertEqual(404, status_code)
        self.assertEqual("not_found", response["error"]["type"])

    def test_main_omits_prompt_by_default(self):
        import infer_sml

        with (
            mock.patch.object(
                infer_sml,
                "generate_text",
                return_value="hello world",
            ) as generate_text,
            mock.patch("builtins.print") as print_,
        ):
            exit_code = infer_sml.main(
                [
                    "hello",
                    "--max-new-tokens",
                    "3",
                ]
            )

        self.assertEqual(0, exit_code)
        generate_text.assert_called_once_with(
            prompt="hello",
            max_new_tokens=3,
            include_prompt=False,
        )
        print_.assert_called_once_with("hello world")

    def test_main_can_include_prompt_from_cli_args(self):
        import infer_sml

        with (
            mock.patch.object(
                infer_sml,
                "generate_text",
                return_value="hello world",
            ) as generate_text,
            mock.patch("builtins.print") as print_,
        ):
            exit_code = infer_sml.main(["hello", "--include-prompt"])

        self.assertEqual(0, exit_code)
        generate_text.assert_called_once_with(
            prompt="hello",
            max_new_tokens=infer_sml.DEFAULT_MAX_NEW_TOKENS,
            include_prompt=True,
        )
        print_.assert_called_once_with("hello world")

    def test_main_can_start_openai_compatible_server(self):
        import infer_sml

        with mock.patch.object(infer_sml, "run_openai_compatible_server") as run_server:
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

        self.assertEqual(0, exit_code)
        run_server.assert_called_once_with(
            host="0.0.0.0",
            port=9000,
            model_name="sml-test",
        )

    def test_parse_args_rejects_fixed_inference_defaults(self):
        import infer_sml

        for option, value in (
            ("--checkpoint", "model.pt"),
            ("--tokenizer", "tokenizer.model"),
            ("--device", "cpu"),
            ("--completion-only", None),
        ):
            with self.subTest(option=option):
                argv = ["hello", option] if value is None else ["hello", option, value]
                with (
                    mock.patch("sys.stderr"),
                    self.assertRaises(SystemExit),
                ):
                    infer_sml.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
