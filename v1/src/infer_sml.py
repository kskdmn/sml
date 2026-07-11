"""
Inference entrypoint for SML checkpoints.

Default files (under ``v1/output/``): checkpoint ``sml.pt``, tokenizer
``bpe_tokenizer.model``. Both must exist before inference.

One-shot CLI generation accepts decoding flags documented on
``GenerationConfig`` in ``sml_config.py`` and exposed as ``--temperature``,
``--top-p``, ``--repetition-penalty``, ``--no-repeat-ngram-size``, and
``--seed``. Those flags do not apply to the OpenAI-compatible HTTP API yet.

``--serve`` exposes a vLLM-style API at ``/v1/models``, ``/v1/completions``,
and ``/v1/chat/completions``. Streaming is not implemented. Chat message
``content`` must be a string. Role markers are plain text labels, not special
tokens; see ``format_chat_messages``. ``max_tokens`` maps to ``max_new_tokens``.
Token usage is a whitespace-split estimate, not tokenizer-exact accounting.
"""

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
from sml_config import GenerationConfig
from sml_config import SMLConfig
from train_sml import load_tokenizer, resolve_device
from train_tokenizer import resolve_path


DEFAULT_CHECKPOINT_PATH = OUTPUT_DIR / "sml.pt"
DEFAULT_MODEL_NAME = "sml"


class InferenceTokenizer(Protocol):
    def encode(self, text: str, out_type: type = int) -> list[int]:
        """
        Inference relies on the SentencePiece-style `out_type=int` contract and does not
        need the concrete tokenizer type.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode receives integer IDs after caller-side special-token filtering."""
        ...


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """
    Load a checkpoint dict, allowlisting POSIX paths in older payloads.

    Maps tensor storage to the selected torch device. PyTorch 2.6+ defaults
    ``torch.load`` to ``weights_only=True``; this helper uses ``safe_globals``
    for trusted local checkpoints. Typical keys include ``step``, ``model_config``,
    ``training_config``, ``input_files``, ``data_state``, ``model_state_dict``,
    ``optimizer_state_dict``, and ``scheduler_state_dict``.
    """
    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    with safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def normalize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """
    Older checkpoints used `max_position_embeddings` and may omit newer RoPE fields;
    rename legacy keys and let dataclass defaults fill gaps.
    """
    normalized = dict(model_config)
    if "max_position_embeddings" in normalized:
        legacy_max_position_embeddings = normalized.pop("max_position_embeddings")
        normalized.setdefault(
            "original_max_position_embeddings",
            legacy_max_position_embeddings,
        )
    return normalized


def load_model(checkpoint_path: Path, device: torch.device) -> SMLLanguageModel:
    """
    Checkpoint files must contain both shape config and weights; returned models are
    moved to the requested device and switched to eval mode.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")

    model = SMLLanguageModel(SMLConfig(**normalize_model_config(model_config)))
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
    """
    Insert BOS before batching because training may have taught the model to expect an
    explicit document-start token.
    """
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
    """
    Generation can include echoed BOS, padding, or EOS; remove those control tokens
    before handing IDs back to SentencePiece.
    """
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


def resolve_max_new_tokens(
    max_new_tokens: int | None,
    max_length: int,
    input_length: int,
) -> int:
    """
    Fill the remaining context window when callers omit an explicit token budget.

    When ``max_new_tokens`` is ``None``, return ``max_length - input_length``.
    """
    if max_new_tokens is None:
        return max(0, max_length - input_length)
    return max_new_tokens


def generate_text(
    prompt: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    tokenizer_model_path: Path = TOKENIZER_MODEL_PATH,
    max_new_tokens: int | None = None,
    device_name: str = "auto",
    include_prompt: bool = False,
    generation_config: GenerationConfig | None = None,
) -> str:
    """
    This one-shot path loads model and tokenizer per call, then decodes only the
    continuation unless the caller asks to include the prompt.
    """
    if max_new_tokens is not None and max_new_tokens < 0:
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
    max_length = model.config.effective_max_position_embeddings
    input_length = input_ids.shape[1]
    if input_length > max_length:
        raise ValueError(
            "prompt exceeds the checkpoint context window: "
            f"{input_length} > {max_length}"
        )
    resolved_max_new_tokens = resolve_max_new_tokens(
        max_new_tokens,
        max_length,
        input_length,
    )

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=resolved_max_new_tokens,
            eos_token_id=model.config.eos_token_id,
            generation_config=generation_config,
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
    """
    The OpenAI-compatible shim reports approximate usage without keeping a tokenizer
    object in the response-construction layer.
    """
    if not text.strip():
        return 0
    return len(text.split())


def resolve_max_tokens(request: Mapping[str, Any]) -> int | None:
    """
    Accept OpenAI's `max_tokens` and the local `max_new_tokens` alias.

    When neither is present, return ``None`` so generation can use the remaining
    context window.
    """
    max_tokens = request.get("max_tokens", request.get("max_new_tokens"))
    if max_tokens is None:
        return None
    if not isinstance(max_tokens, int):
        raise ValueError("max_tokens must be an integer")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    return max_tokens


def resolve_model_name(request: Mapping[str, Any]) -> str:
    """
    The model field is echoed in responses for client compatibility; it does not select
    among multiple local checkpoints.
    """
    model = request.get("model", DEFAULT_MODEL_NAME)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    return model


def create_usage(prompt: str, completion: str) -> dict[str, int]:
    """
    Use the same lightweight token estimate for every OpenAI-compatible response shape.
    """
    prompt_tokens = estimate_token_count(prompt)
    completion_tokens = estimate_token_count(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def format_chat_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """
    Flatten role/content pairs into a deterministic transcript and append an assistant
    cue when the client has not already provided one.

    Example output::

        system: Be concise.
        user: Explain SML in one sentence.
        assistant:

    ``system``, ``user``, and ``assistant`` are encoded as ordinary text. The only
    special token added for inference is BOS at the start of the whole prompt
    (see ``encode_prompt``).
    """
    if not messages:
        raise ValueError("messages must contain at least one message")

    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError("each message must include a non-empty role")
        if not isinstance(content, str):
            raise ValueError(
                "SML chat completions currently support string content only"
            )
        lines.append(f"{role}: {content}")

    if messages[-1].get("role") != "assistant":
        lines.append("assistant:")
    return "\n".join(lines)


def resolve_generator(
    generator: Callable[..., str] | None,
) -> Callable[..., str]:
    """
    Tests and HTTP handlers can inject a generator, while production requests fall back
    to checkpoint-backed generation.
    """
    return generate_text if generator is None else generator


def create_completion_response(
    request: Mapping[str, Any],
    generator: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """
    Mirror the legacy completions schema while forcing generation to return completion
    text rather than prompt-plus-completion.

    Generated text is returned at ``choices[0].text``.
    """
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
    """
    Convert chat messages into the plain-text prompt format SML can consume, then wrap
    the result in OpenAI's chat response shape.

    The assistant message is returned at ``choices[0].message.content``.
    """
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
    """
    Expose a single configured model name so OpenAI clients can discover the local
    endpoint.
    """
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
    """
    Keep unused `param` and `code` keys so simple OpenAI clients can parse errors
    without special casing this server.
    """
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
    """
    Normalize paths and centralize endpoint validation so the HTTP handler and tests
    share identical response behavior.
    """
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
        """
        BaseHTTPRequestHandler provides method and path state; this method delegates all
        endpoint decisions to the shared router.
        """
        status_code, response = route_openai_request(
            self.command,
            self.path,
            model_name=self.model_name,
            generator=self.generator,
        )
        self.send_json(status_code, response)

    def do_POST(self) -> None:
        """
        Reject invalid or non-object JSON before endpoint-specific validation so every
        POST route receives a mapping.
        """
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
        """
        Treat an empty request body as `{}` so missing required fields produce the same
        validation errors as empty JSON.
        """
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        """
        Set Content-Length explicitly before writing bytes for compatibility with simple
        HTTP clients.
        """
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
    """
    Bind model name and generator on a fresh subclass so separate servers do not mutate
    shared handler state.
    """

    class ConfiguredOpenAICompatibleHTTPHandler(OpenAICompatibleHTTPHandler):
        pass

    ConfiguredOpenAICompatibleHTTPHandler.model_name = model_name
    ConfiguredOpenAICompatibleHTTPHandler.generator = generator
    return ConfiguredOpenAICompatibleHTTPHandler


def run_openai_compatible_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = DEFAULT_MODEL_NAME,
    device_name: str = "auto",
) -> None:
    """
    ThreadingHTTPServer allows concurrent local requests; KeyboardInterrupt is swallowed
    so the CLI exits cleanly.
    """

    def generate_with_device(**kwargs: Any) -> str:
        return generate_text(**kwargs, device_name=device_name)

    handler_class = make_openai_compatible_handler(
        model_name,
        generator=generate_with_device,
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"Serving SML OpenAI-compatible API at http://{host}:{port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def generation_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    """
    Build decoding settings from CLI flags while keeping greedy decoding as default.

    ``infer_sml.py`` exposes these knobs as ``--temperature``, ``--top-p``,
    ``--repetition-penalty``, ``--no-repeat-ngram-size``, and ``--seed``.
    """
    return GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        seed=args.seed,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Serving and one-shot generation share a parser, but prompts are required only
    outside server mode.
    """
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
        default=None,
        help=(
            "Maximum generated tokens. Defaults to the remaining context window "
            "(effective_max_position_embeddings minus prompt length)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print the prompt with the generated completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature; 0 keeps greedy decoding (default). "
            "With sampling, try 0.7-1.0; 0.8 is a common start. "
            "See GenerationConfig in sml_config.py."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help=(
            "Nucleus sampling cutoff in (0, 1]; 1.0 disables (default). "
            "Ignored when --temperature is 0. With sampling, try 0.9-0.95."
        ),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=(
            "Down-weight tokens already in the prefix; 1.0 disables (default). "
            "For phrase loops, try 1.05-1.25; start at 1.15."
        ),
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help=(
            "Hard-block tokens that would repeat an n-gram of this length; "
            "0 disables (default). Try 3 or 4; 3 is stricter than 4."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling. Ignored when --temperature is 0.",
    )
    args = parser.parse_args(argv)
    if not args.serve and args.prompt is None:
        parser.error("prompt is required unless --serve is set")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """
    Choose between server mode and one-shot generation after parsing, then return the
    project success code.
    """
    args = parse_args(argv)
    if args.serve:
        run_openai_compatible_server(
            host=args.host,
            port=args.port,
            model_name=args.model,
            device_name=args.device,
        )
        return SUCCESS_RETURN_CODE

    text = generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device_name=args.device,
        include_prompt=args.include_prompt,
        generation_config=generation_config_from_args(args),
    )
    print(text)
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
