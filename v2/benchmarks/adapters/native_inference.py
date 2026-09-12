"""Fixed-work inference measurements using the production model array kernels."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from sml.model.cache import allocate_kv_state
from sml.model.config import GenerationConfig, ModelConfig
from sml.model.generation import (
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    select_next_token_arrays,
)
from sml.model.language_model import SMLLanguageModel

from v2.benchmarks.adapters.execution_order import ObservedBatch, verify_input_batches
from v2.benchmarks.parameters import initialize_parameters
from v2.benchmarks.schema import CanonicalWorkload
from v2.benchmarks.workload import (
    canonical_json_bytes,
    file_identity,
    fixed_canonical_rows,
    fixed_inference_requests,
    semantic_array_identity,
)


class _InferenceRuntime:
    def __init__(
        self,
        metric: str,
        workload: CanonicalWorkload,
        directory: Path,
        generation: GenerationConfig,
    ) -> None:
        self.metric = metric
        self.config = ModelConfig(**workload.model)
        if not self.config.use_cache or self.config.rope_scaling_factor != 1.0:
            raise ValueError(
                "native inference requires cached inference with unit RoPE scaling"
            )
        self.model = SMLLanguageModel(
            self.config, key=mx.random.key(int(workload.optimizer["seed"]))
        )
        self.model.eval()
        self.initial_parameter_identity = initialize_parameters(self.model, workload)
        self.parameters = self.model.parameters()
        rows = fixed_canonical_rows(
            row_count=int(workload.loader["row_count"]),
            row_width=int(workload.loader["sequence_length"]) + 1,
            vocab_size=self.config.vocab_size,
        )
        requests = fixed_inference_requests(workload, rows)
        if (
            requests.identity
            != workload.semantic_identities["canonical_inference_requests"]
        ):
            raise ValueError("native inference requests differ from canonical inputs")
        path = directory / "native-inference-requests.npz"
        policy = canonical_json_bytes(
            {
                key: value
                for key, value in workload.generation.items()
                if key != "request_count"
            }
        )
        np.savez(
            path,
            request_ids=np.asarray(requests.request_ids, dtype="<i4"),
            prompt_ids=requests.prompt_ids.astype("<i4", copy=False),
            generation_policy_json=np.frombuffer(policy, dtype=np.uint8),
        )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        self.canonical_input_identity = semantic_array_identity(
            "sml-benchmark-inference-requests-v1", arrays
        )
        if self.canonical_input_identity != requests.identity:
            raise ValueError(
                "persisted native inference requests changed canonical inputs"
            )
        self.native_representation_identity = file_identity(path)
        self.prompts = arrays["prompt_ids"]
        self._canonical_prompts = requests.prompt_ids
        self.request_ids = tuple(int(value) for value in arrays["request_ids"])
        self.prompt_tokens = int(workload.generation["prompt_tokens"])
        self.chunk_size = requests.decode_tokens
        self.capacity = self.config.original_context_length
        if self.prompt_tokens + self.chunk_size > self.capacity:
            raise ValueError("fixed inference work exceeds cache capacity")
        self.generation = generation
        self._prefill = mx.compile(self._forward)
        self._decode = mx.compile(self._decode_arrays)
        self.request_index = 0
        self.ordered_request_ids: list[int] = []
        self._observed_requests: list[tuple[int, object]] = []
        # All measured decode requests start at the same logical boundary as the
        # reference: a filled prompt cache and its already selected first token.
        self._states = (
            [self._prepare_state(index) for index in self.request_ids]
            if metric == "inference-decode"
            else []
        )

    def _forward(self, parameters, input_ids, cache_state):
        logits, state, _ = self.model.forward_arrays(
            parameters,
            input_ids,
            attention_mask=None,
            positions=None,
            cache_state=cache_state,
            training=False,
            key=None,
            logits_positions=mx.full(
                (input_ids.shape[0], 1), input_ids.shape[1] - 1, dtype=mx.int32
            ),
        )
        return logits, state

    def _select(self, logits, tokens, lengths, key):
        scored = apply_repetition_penalty(
            logits.astype(mx.float32),
            tokens,
            lengths,
            self.generation.repetition_penalty,
        )
        scored = apply_no_repeat_ngram(
            scored, tokens, lengths, self.generation.no_repeat_ngram_size
        )
        selected, key = select_next_token_arrays(
            scored[0],
            key,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
        )
        return selected.reshape((1, 1)).astype(mx.int32), key

    def _new_cache(self):
        return allocate_kv_state(self.config, 1, self.capacity, mx.bfloat16)

    def _prepare_state(self, index: int):
        prompt = mx.array(self.prompts[index][None, :], dtype=mx.int32)
        logits, cache = self._prefill(self.parameters, prompt, self._new_cache())
        tokens = mx.pad(prompt, ((0, 0), (0, self.capacity + 1 - self.prompt_tokens)))
        lengths = mx.array([self.prompt_tokens], dtype=mx.int32)
        key = mx.random.key((self.generation.seed + index) % (2**32))
        selected, key = self._select(logits[:, -1, :], tokens, lengths, key)
        tokens = mx.where(
            mx.arange(tokens.shape[1])[None, :] == lengths[:, None], selected, tokens
        )
        state = cache, selected, tokens, lengths + 1, key
        mx.eval(state)
        return state

    def _decode_arrays(self, parameters, cache, selected, tokens, lengths, key):
        # The benchmark counts a fixed number of forwards even if EOS appears.
        # This matches the reference workload, whose stopping rule is its chunk.
        for _ in range(self.chunk_size):
            logits, cache = self._forward(parameters, selected, cache)
            selected, key = self._select(logits[:, -1, :], tokens, lengths, key)
            tokens = mx.where(
                mx.arange(tokens.shape[1])[None, :] == lengths[:, None],
                selected,
                tokens,
            )
            lengths = lengths + 1
        return cache, selected, tokens, lengths, key

    def run(self, units: int) -> float:
        if type(units) is not int or units < 0:
            raise ValueError("inference work units must be nonnegative integers")
        for _ in range(units):
            index = self.request_index % len(self.request_ids)
            self.request_index += 1
            if self.metric == "inference-prefill":
                inputs = mx.array(self.prompts[index][None, :], dtype=mx.int32)
                self.last_output = self._prefill(
                    self.parameters, inputs, self._new_cache()
                )
            else:
                inputs = self._states[index]
                self.last_output = self._decode(self.parameters, *inputs)
            mx.eval(self.last_output)
            self.ordered_request_ids.append(self.request_ids[index])
            self._observed_requests.append((self.request_ids[index], inputs))
        tokens = (
            self.prompt_tokens
            if self.metric == "inference-prefill"
            else self.chunk_size
        )
        return float(units * tokens)

    def reset_measured_order(self) -> None:
        self.request_index = 0
        self.ordered_request_ids.clear()
        self._observed_requests.clear()

    def observed_execution_order(self) -> tuple[int, ...]:
        batches = [
            ObservedBatch(
                {
                    "prompt_ids": inputs
                    if self.metric == "inference-prefill"
                    else inputs[2][:, : self.prompt_tokens]
                },
                work_ids=(request_id,),
            )
            for request_id, inputs in self._observed_requests
        ]
        return verify_input_batches(
            batches, {"prompt_ids": self._canonical_prompts}, batch_size=1
        )

    def reset_after_warmup(self) -> None:
        self.reset_measured_order()


def make_runtime(metric: str, workload: CanonicalWorkload, directory: Path):
    if metric not in ("inference-prefill", "inference-decode"):
        raise ValueError(f"unsupported native inference metric: {metric}")
    generation_values = workload.generation
    policy_fields = {
        "temperature",
        "top_p",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "seed",
    }
    if set(generation_values) != policy_fields | {
        "request_count",
        "prompt_tokens",
        "decode_chunk_size",
        "request_order",
    }:
        raise ValueError("unsupported native inference generation fields")
    for field in ("request_count", "prompt_tokens", "decode_chunk_size"):
        if type(generation_values[field]) is not int or generation_values[field] < 1:
            raise ValueError(f"inference {field} must be a positive integer")
    for field in ("row_count", "sequence_length"):
        if type(workload.loader[field]) is not int or workload.loader[field] < 1:
            raise ValueError(f"inference loader {field} must be a positive integer")
    if generation_values["request_order"] != "fixed-canonical-order-v1":
        raise ValueError("native inference requires fixed canonical request order")
    if generation_values["prompt_tokens"] > workload.loader["sequence_length"] + 1:
        raise ValueError("inference prompt length exceeds canonical row width")
    generation = GenerationConfig(
        **{field: generation_values[field] for field in policy_fields}
    )
    if generation.seed is None:
        raise ValueError("native inference requires an explicit generation seed")
    if generation.temperature != 0.0:
        raise ValueError(
            "native inference benchmarks currently support only greedy temperature=0"
        )
    return _InferenceRuntime(metric, workload, directory, generation)
