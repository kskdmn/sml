"""Persistent non-reentrant latest-only inference session loading."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import mlx.core as mx
from mlx.utils import tree_unflatten

from sml.artifacts.checkpoint import (
    ResolvedStep,
    open_checkpoint_reader,
    recover_latest_index,
    run_access_lock,
)
from sml.artifacts.manifest import (
    PretrainingRunManifest,
    VerificationLevel,
)
from sml.data.tokenizer import LoadedTokenizer, load_tokenizer_bundle
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.model.cache import allocate_kv_state
from sml.model.config import GenerationConfig, ModelConfig
from sml.model.generation import (
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    select_next_token_arrays,
)
from sml.model.language_model import SMLLanguageModel

_MODEL_GROUP = "model.safetensors"
_MASTER_GROUP = "master.safetensors"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    artifact_kind: str
    run_identity: str | None
    step: int | None
    checkpoint_identity: str | None
    run_step_identity: str | None
    tokenizer_identity: str
    verification: VerificationLevel


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    seed: int | None
    model: ModelIdentity


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    max_new_tokens: int
    config: GenerationConfig = field(default_factory=GenerationConfig)
    include_prompt: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be non-negative")
        if not isinstance(self.config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig")
        if not isinstance(self.include_prompt, bool):
            raise TypeError("include_prompt must be a bool")


@dataclass(frozen=True, slots=True)
class InferenceRuntimeConfig:
    batch_size_buckets: tuple[int, ...] = (1, 2, 4, 8, 16)
    decode_chunk_size: int = 8

    def __post_init__(self) -> None:
        buckets = self.batch_size_buckets
        if not isinstance(buckets, tuple) or not buckets:
            raise ValueError("batch_size_buckets must be a nonempty tuple")
        previous = 0
        for bucket in buckets:
            if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0:
                raise ValueError("batch size buckets must be positive")
            if bucket <= previous:
                raise ValueError("batch size buckets must be strictly increasing")
            previous = bucket
        if (
            isinstance(self.decode_chunk_size, bool)
            or not isinstance(self.decode_chunk_size, int)
            or self.decode_chunk_size <= 0
        ):
            raise ValueError("decode_chunk_size must be positive")


_DEFAULT_RUNTIME = InferenceRuntimeConfig()


@dataclass(frozen=True, slots=True)
class GenerationKernelKey:
    temperature: float
    top_p: float
    repetition_penalty: float
    no_repeat_ngram_size: int


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    artifact_kind: str
    run_identity: str | None
    step: int | None
    checkpoint_identity: str | None
    run_step_identity: str | None
    verification: VerificationLevel
    model_config: ModelConfig
    tokenizer: LoadedTokenizer
    model_arrays: Mapping[str, mx.array]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_arrays",
            MappingProxyType(dict(self.model_arrays)),
        )

    @property
    def tokenizer_identity(self) -> str:
        return self.tokenizer.manifest.identity

    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            artifact_kind=self.artifact_kind,
            run_identity=self.run_identity,
            step=self.step,
            checkpoint_identity=self.checkpoint_identity,
            run_step_identity=self.run_step_identity,
            tokenizer_identity=self.tokenizer_identity,
            verification=self.verification,
        )


class _CallGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self):
        if not self._lock.acquire(blocking=False):
            raise SMLRuntimeError("session is non-reentrant")
        try:
            yield
        finally:
            self._lock.release()


class _Lease:
    def __init__(
        self,
        pool: BufferPool,
        token_storage: mx.array,
        cache_state: object,
        pool_key: tuple[int, ...],
    ) -> None:
        self.token_storage = token_storage
        self.cache_state = cache_state
        self._pool = pool
        self._pool_key = pool_key
        self._released = False

    def discard(self) -> None:
        if self._released:
            return
        self._pool._return(self)
        self._released = True
        self.token_storage = None
        self.cache_state = None


class BufferPool:
    def __init__(self) -> None:
        self._active = 0
        self._free: dict[tuple[int, ...], list[tuple[mx.array, object]]] = {}

    @property
    def active_leases(self) -> int:
        return self._active

    def lease(
        self,
        *,
        batch_size: int,
        capacity: int,
        config: ModelConfig,
    ) -> _Lease:
        key = (
            batch_size,
            capacity,
            config.num_layers,
            config.num_kv_heads,
            config.head_dim,
        )
        self._active += 1
        available = self._free.get(key)
        if available:
            token_storage, cache_state = available.pop()
        else:
            token_storage = mx.zeros((batch_size, capacity), dtype=mx.int32)
            cache_state = allocate_kv_state(config, batch_size, capacity, mx.bfloat16)
            mx.eval(token_storage, cache_state)
        return _Lease(self, token_storage, cache_state, key)

    def release(self, lease: _Lease) -> None:
        lease.discard()

    def _reset_cache(self, cache_state: object) -> object:
        keys, values, lengths = cache_state
        reset = (
            tuple(mx.zeros_like(layer) for layer in keys),
            tuple(mx.zeros_like(layer) for layer in values),
            mx.zeros_like(lengths),
        )
        mx.eval(reset)
        return reset

    def _return(self, lease: _Lease) -> None:
        self._active = max(0, self._active - 1)
        self._free.setdefault(lease._pool_key, []).append(
            (lease.token_storage, self._reset_cache(lease.cache_state))
        )


def _length_buckets(effective_context_length: int) -> tuple[int, ...]:
    buckets: list[int] = []
    size = 1
    while size < effective_context_length:
        buckets.append(size)
        size *= 2
    buckets.append(effective_context_length)
    return tuple(buckets)


def _require_run_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if path.name.startswith("step-"):
        raise SMLArtifactError("direct step-* paths are rejected")
    if (path / "checkpoint.json").exists():
        raise SMLArtifactError("direct checkpoint step paths are rejected")
    return path


def load_owned_model_arrays(
    run: Path,
    *,
    full_verify: bool,
) -> tuple[ResolvedStep, Mapping[str, mx.array]]:
    if not isinstance(full_verify, bool):
        raise TypeError("full_verify must be a bool")
    verification = (
        VerificationLevel.FULL if full_verify else VerificationLevel.MANIFEST_TRUSTED
    )
    load_groups = frozenset(
        {_MODEL_GROUP, _MASTER_GROUP} if full_verify else {_MODEL_GROUP}
    )
    with run_access_lock(run, exclusive=False):
        recovered = recover_latest_index(
            run,
            writable=False,
            verification=VerificationLevel.MANIFEST_TRUSTED,
        )
        with open_checkpoint_reader(
            run,
            step=recovered.step,
            expected_checkpoint_identity=recovered.checkpoint.identity,
            verification=verification,
            load_array_groups=load_groups,
            hold_lock=False,
        ) as reader:
            contents = reader.read_contents()
            model_group = contents.array_groups[_MODEL_GROUP]
            owned = {name: model_group[name] for name in model_group}
            mx.eval(*owned.values())
            return reader.resolved, MappingProxyType(owned)


def resolve_model_artifact(path: Path, *, full_verify: bool) -> ResolvedModel:
    path = _require_run_path(path)
    if not isinstance(full_verify, bool):
        raise TypeError("full_verify must be a bool")
    resolved_step, owned = load_owned_model_arrays(path, full_verify=full_verify)
    if not isinstance(resolved_step.run, PretrainingRunManifest):
        raise SMLArtifactError("model resolution currently supports pretraining runs")
    rope_factor = resolved_step.run.model.get("rope_scaling_factor")
    if rope_factor != 1.0:
        raise SMLArtifactError("pretraining rope_scaling_factor must be exactly 1.0")
    model_config = ModelConfig(**dict(resolved_step.run.model))
    if model_config.rope_scaling_factor != 1.0:
        raise SMLArtifactError("resolution never substitutes rope_scaling_factor")
    verification = resolved_step.verification
    tokenizer = load_tokenizer_bundle(path / "tokenizer", verification)
    if tokenizer.manifest.identity != resolved_step.run.tokenizer_identity:
        raise SMLArtifactError("run tokenizer identity does not match run.json")
    return ResolvedModel(
        artifact_kind=resolved_step.run.kind,
        run_identity=resolved_step.run.identity,
        step=resolved_step.step,
        checkpoint_identity=resolved_step.checkpoint.identity,
        run_step_identity=resolved_step.run_step_identity,
        verification=verification,
        model_config=model_config,
        tokenizer=tokenizer,
        model_arrays=owned,
    )


class InferenceSession:
    def __init__(
        self,
        resolved: ResolvedModel,
        runtime: InferenceRuntimeConfig,
    ) -> None:
        self._resolved = resolved
        self._runtime = runtime
        self._call_guard = _CallGuard()
        self.buffer_pool = BufferPool()
        self._model = SMLLanguageModel(resolved.model_config, key=mx.random.key(0))
        self._parameters = tree_unflatten(sorted(resolved.model_arrays.items()))
        self._compiled: dict[tuple[object, ...], object] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        full_verify: bool = False,
        runtime: InferenceRuntimeConfig = _DEFAULT_RUNTIME,
    ) -> InferenceSession:
        if not isinstance(runtime, InferenceRuntimeConfig):
            raise TypeError("runtime must be an InferenceRuntimeConfig")
        resolved = resolve_model_artifact(path, full_verify=full_verify)
        return cls(resolved, runtime)

    @property
    def resolved_model(self) -> ResolvedModel:
        return self._resolved

    @property
    def model_identity(self) -> ModelIdentity:
        return self._resolved.identity()

    @property
    def length_buckets(self) -> tuple[int, ...]:
        return _length_buckets(self._resolved.model_config.effective_context_length)

    def generate(self, text: str, request: GenerationRequest) -> GenerationResult:
        self._require_generate_args(text, request)
        with self._call_guard.acquire():
            return self._generate_one(text, request)

    def generate_batch(
        self,
        items: Sequence[tuple[str, GenerationRequest]],
    ) -> tuple[GenerationResult, ...]:
        with self._call_guard.acquire():
            return tuple(self._generate_one(text, request) for text, request in items)

    def _require_generate_args(self, text: object, request: object) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest")

    def _generate_one(self, text: str, request: GenerationRequest) -> GenerationResult:
        lease = None
        try:
            prompt_ids = self._encode_prompt(text)
            required = len(prompt_ids) + request.max_new_tokens
            capacity = self._select_length_bucket(required)
            lease = self.buffer_pool.lease(
                batch_size=1,
                capacity=capacity,
                config=self._resolved.model_config,
            )
            generated = self._decode_chunk(prompt_ids, request, lease, capacity)
            token_ids = tuple(int(token) for token in generated)
            if request.include_prompt:
                token_ids = prompt_ids + token_ids
            decoded = self._resolved.tokenizer.processor.decode(list(token_ids))
            return GenerationResult(
                text=decoded,
                token_ids=token_ids,
                seed=request.config.seed,
                model=self.model_identity,
            )
        finally:
            if lease is not None:
                self.buffer_pool.release(lease)

    def _encode_prompt(self, text: str) -> tuple[int, ...]:
        processor = self._resolved.tokenizer.processor
        bos_id = int(processor.bos_id())
        encoded = [int(token) for token in processor.encode(text)]
        if bos_id >= 0 and (not encoded or encoded[0] != bos_id):
            encoded = [bos_id, *encoded]
        if not encoded:
            raise SMLRuntimeError(
                "prompt encodes to no tokens without a usable BOS token"
            )
        return tuple(encoded)

    def _select_length_bucket(self, required: int) -> int:
        for bucket in self.length_buckets:
            if required <= bucket:
                return bucket
        raise SMLRuntimeError(
            "prompt overflow: required capacity exceeds effective context length"
        )

    def _compiled_forward(
        self,
        length_bucket: int,
        batch_size_bucket: int,
        kernel_key: GenerationKernelKey,
    ):
        key = (length_bucket, batch_size_bucket, kernel_key)
        compiled = self._compiled.get(key)
        if compiled is None:
            model = self._model

            def _forward(
                parameters,
                input_ids,
                attention_mask,
                positions,
                cache_state,
            ):
                logits, cache_state, _next_key = model.forward_arrays(
                    parameters,
                    input_ids,
                    attention_mask=attention_mask,
                    positions=positions,
                    cache_state=cache_state,
                    training=False,
                    key=None,
                )
                return logits, cache_state

            compiled = mx.compile(_forward)
            self._compiled[key] = compiled
        return compiled

    def _decode_chunk(
        self,
        prompt_ids,
        request: GenerationRequest,
        lease: _Lease,
        length_bucket: int,
    ):
        config = request.config
        kernel_key = GenerationKernelKey(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
        )
        forward = self._compiled_forward(length_bucket, 1, kernel_key)
        parameters = self._parameters
        prompt_len = len(prompt_ids)
        prompt = mx.array(list(prompt_ids), dtype=mx.int32)
        lease.token_storage[0, :prompt_len] = prompt
        prompt_mask = mx.ones((1, prompt_len), dtype=mx.bool_)
        logits, cache_state = forward(
            parameters,
            lease.token_storage[:, :prompt_len],
            prompt_mask,
            None,
            lease.cache_state,
        )
        lease.cache_state = cache_state
        generated: list[int] = []
        eos_id = self._resolved.model_config.eos_token_id
        rng = mx.random.key(0 if config.seed is None else config.seed)
        next_logits = logits[0, prompt_len - 1]
        chunk_size = self._runtime.decode_chunk_size
        remaining = request.max_new_tokens
        logical = prompt_len
        while remaining > 0:
            steps = min(remaining, chunk_size)
            chunk_tokens = []
            for step in range(steps):
                lengths = mx.array([logical], dtype=mx.int32)
                scored = next_logits.astype(mx.float32)[None, :]
                scored = apply_repetition_penalty(
                    scored,
                    lease.token_storage,
                    lengths,
                    config.repetition_penalty,
                )
                scored = apply_no_repeat_ngram(
                    scored,
                    lease.token_storage,
                    lengths,
                    config.no_repeat_ngram_size,
                )
                token_id, rng = select_next_token_arrays(
                    scored[0],
                    rng,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                lease.token_storage[0, logical] = token_id
                chunk_tokens.append(token_id)
                logical += 1
                if step + 1 < steps or remaining > steps:
                    step_ids = mx.reshape(token_id, (1, 1))
                    step_mask = mx.ones((1, 1), dtype=mx.bool_)
                    logits, cache_state = forward(
                        parameters,
                        step_ids,
                        step_mask,
                        None,
                        cache_state,
                    )
                    lease.cache_state = cache_state
                    next_logits = logits[0, 0]
            stacked = mx.stack(chunk_tokens)
            mx.eval(stacked, cache_state, rng, lease.token_storage)
            for token in stacked.tolist():
                host_token = int(token)
                generated.append(host_token)
                if host_token == eos_id:
                    return tuple(generated)
            remaining -= steps
        return tuple(generated)


__all__ = (
    "BufferPool",
    "GenerationKernelKey",
    "GenerationRequest",
    "GenerationResult",
    "InferenceRuntimeConfig",
    "InferenceSession",
    "ModelIdentity",
    "ResolvedModel",
    "load_owned_model_arrays",
    "resolve_model_artifact",
)
