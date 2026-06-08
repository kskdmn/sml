from __future__ import annotations

import argparse
import json
import pathlib
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch.serialization import safe_globals

from config import OUTPUT_DIR, SUCCESS_RETURN_CODE, TOKENIZER_MODEL_PATH
from sml import SMLLanguageModel
from sml_config import SMLConfig
from train_sml import load_tokenizer, resolve_device
from train_tokenizer import resolve_path


DEFAULT_CHECKPOINT_PATH = OUTPUT_DIR / "sml.pt"
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_MODEL_NAME = "sml"


class InferenceTokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    with safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> SMLLanguageModel:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")

    model = SMLLanguageModel(SMLConfig(**model_config))
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model


def encode_prompt(
    tokenizer: InferenceTokenizer,
    prompt: str,
    bos_token_id: int | None,
    device: torch.device,
) -> torch.Tensor:
    token_ids = tokenizer.encode(prompt, out_type=int)
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def decode_token_ids(
    tokenizer: InferenceTokenizer,
    token_ids: Sequence[int],
    bos_token_id: int | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> str:
    decoded_ids: list[int] = []
    skipped_ids = {
        token_id for token_id in (bos_token_id, pad_token_id) if token_id is not None
    }
    for token_id in token_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if token_id in skipped_ids:
            continue
        decoded_ids.append(int(token_id))
    return tokenizer.decode(decoded_ids)


def generate_text(
    prompt: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device_name: str = "auto",
    include_prompt: bool = False,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    device = resolve_device(device_name)
    tokenizer = load_tokenizer(tokenizer_model_path)
    model = load_model(checkpoint_path, device)
    input_ids = encode_prompt(
        tokenizer,
        prompt,
        bos_token_id=model.config.bos_token_id,
        device=device,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=model.config.eos_token_id,
        )

    start_index = 0 if include_prompt else input_ids.shape[1]
    generated_ids = generated[0, start_index:].detach().cpu().tolist()
    return decode_token_ids(
        tokenizer,
        generated_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )


def estimate_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return len(text.split())


def resolve_max_tokens(request: Mapping[str, Any]) -> int:
    max_tokens = request.get("max_tokens", request.get("max_new_tokens"))
    if max_tokens is None:
        return DEFAULT_MAX_NEW_TOKENS
    if not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    return max_tokens


def resolve_model_name(request: Mapping[str, Any]) -> str:
    model = request.get("model", DEFAULT_MODEL_NAME)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    return model


def create_usage(prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = estimate_token_count(prompt)
    completion_tokens = estimate_token_count(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def format_chat_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
        raise ValueError("messages must contain at least one message")

    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("each message must include a non-empty role")
        if not isinstance(content, str):
            raise ValueError("SML chat completions currently support string content only")
        lines.append(f"{role}: {content}")

    if messages[-1].get("role") != "assistant":
        lines.append("assistant:")
    return "\n".join(lines)


def resolve_generator(
    generator: Callable[..., str] | None,
) -> Callable[..., str]:
    return generate_text if generator is None else generator


def create_completion_response(
    request: Mapping[str, Any],
    generator: Callable[..., str] | None = None,
) -> dict[str, Any]:
    prompt = request.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")

    model = resolve_model_name(request)
    max_tokens = resolve_max_tokens(request)
    completion = resolve_generator(generator)(
        prompt=prompt,
        max_new_tokens=max_tokens,
        include_prompt=False,
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": completion,
                "finish_reason": "length",
            },
        ],
        "usage": create_usage(prompt, completion),
    }


def create_chat_completion_response(
    request: Mapping[str, Any],
    generator: Callable[..., str] | None = None,
) -> dict[str, Any]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    prompt = format_chat_messages(messages)
    model = resolve_model_name(request)
    max_tokens = resolve_max_tokens(request)
    completion = resolve_generator(generator)(
        prompt=prompt,
        max_new_tokens=max_tokens,
        include_prompt=False,
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion,
                },
                "finish_reason": "length",
            },
        ],
        "usage": create_usage(prompt, completion),
    }


def create_models_response(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    if not model_name:
        raise ValueError("model_name must be non-empty")
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sml",
            },
        ],
    }


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": None,
        },
    }


def route_openai_request(
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    generator: Callable[..., str] | None = None,
) -> tuple[int, dict[str, Any]]:
    method = method.upper()
    path = path.rstrip("/") or "/"

    if method == "GET" and path == "/v1/models":
        return 200, create_models_response(model_name)

    if method == "POST" and path == "/v1/completions":
        if payload is None:
            payload = {}
        try:
            return 200, create_completion_response(payload, generator=generator)
        except ValueError as exc:
            return 400, create_error_response(str(exc))

    if method == "POST" and path == "/v1/chat/completions":
        if payload is None:
            payload = {}
        try:
            return 200, create_chat_completion_response(payload, generator=generator)
        except ValueError as exc:
            return 400, create_error_response(str(exc))

    return 404, create_error_response(
        f"Unknown endpoint: {method} {path}",
        error_type="not_found",
    )


class OpenAICompatibleHTTPHandler(BaseHTTPRequestHandler):
    model_name = DEFAULT_MODEL_NAME
    generator: Callable[..., str] | None = None

    def do_GET(self) -> None:
        status_code, response = route_openai_request(
            self.command,
            self.path,
            model_name=self.model_name,
            generator=self.generator,
        )
        self.send_json(status_code, response)

    def do_POST(self) -> None:
        payload = self.read_json_payload()
        if not isinstance(payload, dict):
            self.send_json(
                400,
                create_error_response("request body must be a JSON object"),
            )
            return

        status_code, response = route_openai_request(
            self.command,
            self.path,
            payload=payload,
            model_name=self.model_name,
            generator=self.generator,
        )
        self.send_json(status_code, response)

    def read_json_payload(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_openai_compatible_handler(
    model_name: str,
    generator: Callable[..., str] | None = None,
) -> type[OpenAICompatibleHTTPHandler]:
    class ConfiguredOpenAICompatibleHTTPHandler(OpenAICompatibleHTTPHandler):
        pass

    ConfiguredOpenAICompatibleHTTPHandler.model_name = model_name
    ConfiguredOpenAICompatibleHTTPHandler.generator = generator
    return ConfiguredOpenAICompatibleHTTPHandler


def run_openai_compatible_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    handler_class = make_openai_compatible_handler(model_name)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"Serving SML OpenAI-compatible API at http://{host}:{port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from an SML checkpoint."
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text to continue.")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve a vLLM-style OpenAI-compatible HTTP API.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --serve. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve. Defaults to 8000.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"OpenAI-compatible model name. Defaults to {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated tokens. Defaults to {DEFAULT_MAX_NEW_TOKENS}.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print the prompt with the generated completion.",
    )
    args = parser.parse_args(argv)
    if not args.serve and args.prompt is None:
        parser.error("prompt is required unless --serve is set")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.serve:
        run_openai_compatible_server(
            host=args.host,
            port=args.port,
            model_name=args.model,
        )
        return SUCCESS_RETURN_CODE

    text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        include_prompt=args.include_prompt,
    )
    print(text)
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
