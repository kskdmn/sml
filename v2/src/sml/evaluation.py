"""Batched lm-eval scoring and atomically persisted evaluation metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sml.artifacts.manifest import VerificationLevel
from sml.errors import SMLRuntimeError
from sml.inference import (
    GenerationConfig,
    GenerationRequest,
    InferenceRuntimeConfig,
    InferenceSession,
    ModelIdentity,
)

_ALLOWED_TASKS = frozenset({"hellaswag", "winogrande"})
_PROVIDER_PACKAGES = ("lm-eval", "mlx", "numpy", "sentencepiece")


@dataclass(frozen=True, slots=True)
class LoglikelihoodRequest:
    context: str
    continuation: str


@dataclass(frozen=True, slots=True)
class LoglikelihoodResult:
    log_likelihood: float
    greedy_match: bool


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    checkpoint: Path
    tasks: tuple[Literal["hellaswag", "winogrande"], ...]
    output: Path
    full_verify: bool = False
    padding: Literal["left", "right"] = "right"
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a Path")
        if not isinstance(self.output, Path):
            raise TypeError("output must be a Path")
        if not isinstance(self.full_verify, bool):
            raise TypeError("full_verify must be a bool")
        if self.padding not in ("left", "right"):
            raise ValueError("padding must be 'left' or 'right'")
        if not isinstance(self.runtime, InferenceRuntimeConfig):
            raise TypeError("runtime must be an InferenceRuntimeConfig")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError("tasks must contain at least one task")
        for task in self.tasks:
            if task not in _ALLOWED_TASKS:
                raise ValueError("tasks must be hellaswag or winogrande")
        if self.limit is not None and (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit <= 0
        ):
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    output: Path
    model: ModelIdentity
    tasks: tuple[str, ...]
    provider_versions: tuple[tuple[str, str], ...]


def _import_lm_eval():
    from lm_eval import simple_evaluate

    return _LMEvalAdapter(simple_evaluate)


class _LMEvalAdapter:
    def __init__(self, simple_evaluate) -> None:
        self.simple_evaluate = simple_evaluate


def _encode_text(processor, text: str) -> list[int]:
    return [int(token) for token in processor.encode(text)]


def _encode_loglikelihood_request(
    session: InferenceSession,
    request: LoglikelihoodRequest,
) -> tuple[tuple[int, ...], int]:
    processor = session.resolved_model.tokenizer.processor
    context_ids = _encode_text(processor, request.context)
    full_ids = _encode_text(processor, request.context + request.continuation)
    if full_ids[: len(context_ids)] != context_ids:
        continuation_ids = _encode_text(processor, request.continuation)
        continuation_start = len(full_ids) - len(continuation_ids)
    else:
        continuation_start = len(context_ids)

    bos_id = int(processor.bos_id())
    if bos_id >= 0:
        prefix_id = session.resolved_model.model_config.bos_token_id
        full_ids = [prefix_id, *full_ids]
        continuation_start += 1
    elif continuation_start == 0:
        raise SMLRuntimeError("empty context requires a usable prefix token")

    if continuation_start < 0:
        raise SMLRuntimeError("invalid continuation boundary")
    if continuation_start >= len(full_ids) and request.continuation:
        raise SMLRuntimeError("continuation produced no tokens")
    if not full_ids:
        raise SMLRuntimeError("empty context requires a usable prefix token")
    return tuple(full_ids), continuation_start


def score_loglikelihood_batch(
    session: InferenceSession,
    requests: Sequence[LoglikelihoodRequest],
    *,
    padding: str,
) -> tuple[LoglikelihoodResult, ...]:
    if padding not in ("left", "right"):
        raise ValueError("padding must be 'left' or 'right'")
    encoded = tuple(
        _encode_loglikelihood_request(session, request) for request in requests
    )
    raw = session.score_encoded_loglikelihoods(encoded, padding=padding)
    return tuple(
        LoglikelihoodResult(log_likelihood=log_likelihood, greedy_match=greedy_match)
        for log_likelihood, greedy_match in raw
    )


def _truncate_at_stop(text: str, until: object) -> str:
    if isinstance(until, str):
        stop_sequences = [until]
    else:
        stop_sequences = list(until or [])
    positions = [
        position
        for stop in stop_sequences
        if stop and (position := text.find(stop)) >= 0
    ]
    return text[: min(positions)] if positions else text


class SMLEvalLM:
    def __init__(
        self,
        session: InferenceSession,
        *,
        padding: Literal["left", "right"] = "right",
    ) -> None:
        self._session = session
        self._padding = padding

    def loglikelihood(self, requests):
        scored = score_loglikelihood_batch(
            self._session,
            [
                LoglikelihoodRequest(
                    context=request.args[0],
                    continuation=request.args[1],
                )
                for request in requests
            ],
            padding=self._padding,
        )
        return [(result.log_likelihood, result.greedy_match) for result in scored]

    def generate_until(self, requests) -> list[str]:
        items: list[tuple[str, GenerationRequest]] = []
        stop_lists: list[object] = []
        for request in requests:
            context, generation_kwargs = request.args
            if generation_kwargs.get("do_sample", False):
                raise SMLRuntimeError("SMLEvalLM only supports greedy generation")
            max_new_tokens = int(generation_kwargs.get("max_gen_toks", 256))
            items.append(
                (
                    context,
                    GenerationRequest(
                        max_new_tokens=max_new_tokens,
                        config=GenerationConfig(),
                    ),
                )
            )
            stop_lists.append(generation_kwargs.get("until") or [])
        generated = self._session.generate_batch(items)
        return [
            _truncate_at_stop(result.text, until)
            for result, until in zip(generated, stop_lists, strict=True)
        ]

    def loglikelihood_rolling(self, requests):
        del requests
        raise SMLRuntimeError("unsupported evaluation method: loglikelihood_rolling")


def _provider_versions() -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for name in _PROVIDER_PACKAGES:
        try:
            found.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            continue
    return tuple(sorted(found, key=lambda item: item[0]))


def _result_payload(result: EvaluationResult) -> dict[str, object]:
    model = result.model
    return {
        "output": str(result.output),
        "model": {
            "artifact_kind": model.artifact_kind,
            "run_identity": model.run_identity,
            "step": model.step,
            "checkpoint_identity": model.checkpoint_identity,
            "run_step_identity": model.run_step_identity,
            "tokenizer_identity": model.tokenizer_identity,
            "verification": model.verification.value,
        },
        "tasks": list(result.tasks),
        "provider_versions": [list(pair) for pair in result.provider_versions],
    }


def _result_from_payload(payload: dict[str, object]) -> EvaluationResult:
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise SMLRuntimeError("evaluation result model metadata is invalid")
    return EvaluationResult(
        output=Path(str(payload["output"])),
        model=ModelIdentity(
            artifact_kind=str(model_payload["artifact_kind"]),
            run_identity=model_payload["run_identity"],
            step=model_payload["step"],
            checkpoint_identity=model_payload["checkpoint_identity"],
            run_step_identity=model_payload["run_step_identity"],
            tokenizer_identity=str(model_payload["tokenizer_identity"]),
            verification=VerificationLevel(model_payload["verification"]),
        ),
        tasks=tuple(payload["tasks"]),  # type: ignore[arg-type]
        provider_versions=tuple(
            (str(name), str(version))
            for name, version in payload["provider_versions"]  # type: ignore[misc]
        ),
    )


def _dumps_result(result: EvaluationResult) -> str:
    return json.dumps(_result_payload(result), indent=2, sort_keys=True) + "\n"


def read_evaluation_result(path: Path) -> EvaluationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _result_from_payload(payload)


def _persist_evaluation_result(result: EvaluationResult) -> None:
    text = _dumps_result(result)
    path = result.output
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise SMLRuntimeError(f"evaluation output collision: {path}")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def evaluate(config: EvaluationConfig) -> EvaluationResult:
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    session = InferenceSession.from_checkpoint(
        config.checkpoint,
        full_verify=config.full_verify,
        runtime=config.runtime,
    )
    lm_eval = _import_lm_eval()
    lm = SMLEvalLM(session, padding=config.padding)
    lm_eval.simple_evaluate(
        model=lm,
        tasks=list(config.tasks),
        num_fewshot=0,
        limit=config.limit,
        log_samples=False,
    )
    result = EvaluationResult(
        output=config.output,
        model=session.model_identity,
        tasks=config.tasks,
        provider_versions=_provider_versions(),
    )
    _persist_evaluation_result(result)
    return result


__all__ = (
    "EvaluationConfig",
    "EvaluationResult",
    "LoglikelihoodRequest",
    "LoglikelihoodResult",
    "SMLEvalLM",
    "evaluate",
    "read_evaluation_result",
    "score_loglikelihood_batch",
)
