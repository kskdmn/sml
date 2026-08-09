from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCRIPT = Path(__file__).resolve()
SCHEMA_TEST = CAPTURE_SCRIPT.with_name("test_legacy_fixture_integrity.py")
PINNED_SOURCE_COMMIT = "3687f8b"
CAPTURE_IDENTITY_FILES = (CAPTURE_SCRIPT, SCHEMA_TEST)

PYTHON_SEED = 20_260_801
NUMPY_SEED = 20_260_802
MODEL_SEED = 7
LORA_SEED = 11
SAMPLING_SEED = 1_234
COMPILED_INPUT_SEED = 19

PRETRAINING_MODEL_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_layers": 2,
    "num_q_heads": 4,
    "num_kv_heads": 2,
    "intermediate_size": 32,
    "original_max_position_embeddings": 8,
    "rope_theta": 10_000.0,
    "rope_scaling_factor": 1.0,
    "yarn_beta_fast": 32.0,
    "yarn_beta_slow": 1.0,
    "yarn_attention_factor": None,
    "yarn_mscale": None,
    "yarn_mscale_all_dim": None,
    "yarn_truncate": True,
    "rms_norm_eps": 1e-6,
    "hidden_dropout": 0.0,
    "initializer_range": 0.02,
    "pad_token_id": 3,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "unk_token_id": 0,
    "tie_word_embeddings": True,
    "use_cache": True,
}

LORA_CONFIG = {
    "rank": 2,
    "alpha": 4.0,
    "scaling_mode": "lora",
    "dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_a_initializer_range": 0.01,
    "lora_b_initializer_range": 0.01,
}

MODEL_INPUT_IDS = [[1, 4, 5, 6, 7, 2], [1, 8, 9, 10, 11, 2]]
MODEL_LABELS = [[4, 5, 6, 7, 2, 3], [8, 9, 10, 11, 2, 3]]
CACHE_INPUT_IDS = [[1, 12, 13, 14, 15, 16]]
GREEDY_PROMPT = [[1, 17, 18]]
SAMPLED_PROMPT = [[1, 19, 20]]
UNEQUAL_PROMPTS = [[1, 21, 22], [1, 23]]
PADDED_UNEQUAL_PROMPTS = [[1, 21, 22], [1, 23, 3]]
MAX_NEW_TOKENS = 3

ROPE_FACTORS = (2.0, 4.0)
ROPE_POSITIONS = [0, 5, 12]
ROPE_Q_VALUES = [((index % 13) - 6) / 8.0 for index in range(1 * 4 * 3 * 4)]
ROPE_K_VALUES = [((index % 11) - 5) / 7.0 for index in range(1 * 2 * 3 * 4)]
GQA_INPUT_VALUES = [((index % 17) - 8) / 9.0 for index in range(2 * 4 * 16)]

GENERATION_LOGITS = [
    -1.25,
    0.75,
    1.5,
    -0.5,
    2.25,
    0.0,
    1.0,
    -2.0,
]
GENERATION_CONFIGS = {
    "greedy": {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "seed": None,
    },
    "seeded": {
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2,
        "seed": SAMPLING_SEED,
    },
}

LOGLIKELIHOOD_INPUTS = [[1, 24, 25, 26], [1, 27, 28]]
LOGLIKELIHOOD_LABELS = [[24, 25, 26, 2], [27, 28, 2]]
LOGLIKELIHOOD_PADDED_INPUTS = [[1, 24, 25, 26], [1, 27, 28, 3]]
LOGLIKELIHOOD_PADDED_LABELS = [[24, 25, 26, 2], [27, 28, 2, 3]]

SWAG_INPUT_IDS = [
    [
        [1, 29, 30, 31],
        [1, 29, 32, 33],
        [1, 29, 34, 35],
        [1, 29, 36, 37],
    ]
]
SWAG_LABELS = [
    [
        [3, 30, 31, 2],
        [3, 32, 33, 2],
        [3, 34, 35, 2],
        [3, 36, 37, 2],
    ]
]


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_commit(root: Path, commit: str) -> str:
    return _run_git(root, "rev-parse", f"{commit}^{{commit}}")


def assert_clean_checkout(root: Path, expected_commit: str) -> str:
    root = root.resolve()
    actual = _resolve_commit(root, "HEAD")
    try:
        expected = _resolve_commit(root, expected_commit)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"checkout {root} expected commit {expected_commit} does not resolve"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"checkout {root} is at {actual}, expected commit {expected_commit} ({expected})"
        )
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"checkout {root} is not clean:\n{status}")
    return actual


def capture_tool_identity() -> str:
    digest = hashlib.sha256()
    digest.update(b"sml-legacy-capture-tool-v1\0")
    for path in CAPTURE_IDENTITY_FILES:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


@contextlib.contextmanager
def detached_worktree(project_root: Path, commit: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="sml-legacy-source-") as temporary:
        source_root = Path(temporary) / "source"
        subprocess.run(
            ["git", "worktree", "add", "--quiet", "--detach", str(source_root), commit],
            cwd=project_root,
            check=True,
        )
        try:
            yield source_root
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(source_root)],
                cwd=project_root,
                check=True,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture deterministic legacy v2 model equivalence fixtures."
    )
    parser.add_argument("--capture-tool-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-control", type=Path, required=True)
    parser.add_argument("--output-arrays", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--capture-tool-identity", help=argparse.SUPPRESS)
    parser.add_argument("--resolved-capture-tool-commit", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_worker_from_external_script(
    args: argparse.Namespace,
    *,
    source_root: Path,
    resolved_capture_tool_commit: str,
    tool_identity: str,
) -> None:
    environment = os.environ.copy()
    legacy_source = str((source_root / "v2" / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        [legacy_source, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    command = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        "--worker",
        "--capture-tool-commit",
        args.capture_tool_commit,
        "--resolved-capture-tool-commit",
        resolved_capture_tool_commit,
        "--capture-tool-identity",
        tool_identity,
        "--source-commit",
        args.source_commit,
        "--source-root",
        str(source_root),
        "--output-control",
        str(args.output_control.resolve()),
        "--output-arrays",
        str(args.output_arrays.resolve()),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _raw_array_bytes(array, mx, np) -> bytes:
    if array.dtype == mx.bfloat16:
        host = np.asarray(array.view(mx.uint16))
    else:
        host = np.asarray(array)
    return np.ascontiguousarray(host).tobytes(order="C")


def _array_payload_identity(array, mx, np) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(_raw_array_bytes(array, mx, np))
    return f"sha256:{digest.hexdigest()}"


def _json_array_values(array, mx, np):
    return np.asarray(array.astype(mx.float32)).tolist()


def _state_payload_identity(mappings: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"sml-legacy-parameter-state-v1\0")
    for mapping in mappings:
        digest.update(str(mapping["destination"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(mapping["payload_identity"]).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _mapping_record(source: str, destination: str, array, mx, np):
    return {
        "source": source,
        "destination": destination,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "payload_identity": _array_payload_identity(array, mx, np),
    }


def _flatten_parameters(model, tree_flatten):
    return sorted(tree_flatten(model.parameters()), key=lambda item: item[0])


def _store_model_state(arrays, model, mx, np, tree_flatten):
    flattened = _flatten_parameters(model, tree_flatten)
    values = dict(flattened)
    tied_names = ["embed_tokens.weight", "lm_head.weight"]
    if any(name not in values for name in tied_names):
        raise RuntimeError("legacy tied parameter leaves are incomplete")
    tied_identical = _raw_array_bytes(
        values[tied_names[0]], mx, np
    ) == _raw_array_bytes(values[tied_names[1]], mx, np)
    if not tied_identical:
        raise RuntimeError("legacy tied vocabulary leaves have different bytes")

    mappings = []
    for source, value in flattened:
        if source == "lm_head.weight":
            continue
        destination = source
        key = f"model_state.{destination}"
        if key in arrays:
            if _raw_array_bytes(arrays[key], mx, np) != _raw_array_bytes(value, mx, np):
                raise RuntimeError(f"parameter alias {source} has different bytes")
            continue
        arrays[key] = value
        mappings.append(_mapping_record(source, destination, value, mx, np))
    return mappings, tied_identical


def _store_lora_state(arrays, model, mx, np, tree_flatten):
    base_mappings = []
    adapter_mappings = []
    flattened = _flatten_parameters(model, tree_flatten)
    parameter_values = dict(flattened)
    for source, value in flattened:
        is_adapter = source.endswith((".lora_A", ".lora_B"))
        if source == "lm_head.weight":
            embedding = parameter_values["embed_tokens.weight"]
            if _raw_array_bytes(value, mx, np) != _raw_array_bytes(embedding, mx, np):
                raise RuntimeError("LoRA tied vocabulary leaves have different bytes")
            continue
        namespace = "lora_state" if is_adapter else "lora_base_state"
        arrays[f"{namespace}.{source}"] = value
        mapping = _mapping_record(source, source, value, mx, np)
        (adapter_mappings if is_adapter else base_mappings).append(mapping)
    return base_mappings, adapter_mappings


def _add_array(arrays, name: str, value) -> None:
    if name in arrays:
        raise RuntimeError(f"duplicate fixture array name: {name}")
    arrays[name] = value


def _generated_with_margins(model, sml, mx, prompt, config, max_new_tokens: int):
    generated = prompt
    cache = sml.KVCache(max_seq_len=prompt.shape[-1] + max_new_tokens)
    key = mx.random.key(config.seed) if config.seed is not None else None
    margins = []
    for _ in range(max_new_tokens):
        model_input = generated[:, -1:] if cache.key_cache else generated
        output = model(model_input, kv_cache=cache)
        logits = output.logits[:, -1, :]
        logits = sml.apply_repetition_penalty(
            logits, generated, config.repetition_penalty
        )
        logits = sml.apply_no_repeat_ngram(
            logits, generated, config.no_repeat_ngram_size
        )
        ordered = mx.sort(logits, axis=-1)
        margins.append(ordered[:, -1] - ordered[:, -2])
        sample_key = None
        if key is not None:
            keys = mx.random.split(key)
            key, sample_key = keys[0], keys[1]
        next_token = sml.select_next_token(logits, config, key=sample_key).astype(
            prompt.dtype
        )
        generated = mx.concatenate([generated, next_token], axis=1)
    margins_array = mx.stack(margins, axis=1)
    mx.eval(generated, margins_array)
    return generated, margins_array


def _sequence_loglikelihood(model, mx, input_ids, labels, pad_token_id: int):
    logits = model(input_ids).logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    gathered = mx.squeeze(
        mx.take_along_axis(log_probs, mx.expand_dims(labels, -1), axis=-1), axis=-1
    )
    return mx.sum(mx.where(labels != pad_token_id, gathered, 0.0), axis=-1)


def _legacy_forward_without_validation(model, input_ids):
    hidden = model.embed_tokens(input_ids)
    for layer in model.layers:
        hidden = layer(hidden)
    hidden = model.norm(hidden)
    return model.lm_head(hidden)


def _capture_references(args, modules):
    mx, nn, np, tree_flatten, sml, train, lora, ft_swag, utilities = modules
    random.seed(PYTHON_SEED)
    np.random.seed(NUMPY_SEED)
    utilities.set_seed(MODEL_SEED)

    config = sml.SMLConfig(**PRETRAINING_MODEL_CONFIG)
    model = sml.SMLLanguageModel(config)
    train.apply_model_dtype(model, "bfloat16")
    model.eval()
    mx.eval(model.parameters())

    arrays = {}
    parameter_mapping, tied_identical = _store_model_state(
        arrays, model, mx, np, tree_flatten
    )

    input_ids = mx.array(MODEL_INPUT_IDS, dtype=mx.int32)
    labels = mx.array(MODEL_LABELS, dtype=mx.int32)
    model_output = model(input_ids, labels=labels)
    mx.eval(model_output.logits, model_output.loss)
    _add_array(arrays, "model.input_ids", input_ids)
    _add_array(arrays, "model.labels", labels)
    _add_array(arrays, "model.logits", model_output.logits)
    _add_array(arrays, "model.loss", model_output.loss)

    loss_and_grad = nn.value_and_grad(
        model,
        lambda current_input, current_labels: (
            model(current_input, labels=current_labels).loss
        ),
    )
    tied_loss, tied_gradients = loss_and_grad(input_ids, labels)
    flattened_gradients = dict(tree_flatten(tied_gradients))
    embedding_grad = flattened_gradients["embed_tokens.weight"]
    head_grad = flattened_gradients["lm_head.weight"]
    mx.eval(tied_loss, embedding_grad, head_grad)
    _add_array(arrays, "tied.embedding_grad", embedding_grad)
    _add_array(arrays, "tied.head_grad", head_grad)
    _add_array(arrays, "tied.loss", tied_loss)

    rope_config_record = {}
    rope_q = (
        mx.array(ROPE_Q_VALUES, dtype=mx.float32)
        .reshape((1, 4, 3, 4))
        .astype(mx.bfloat16)
    )
    rope_k = (
        mx.array(ROPE_K_VALUES, dtype=mx.float32)
        .reshape((1, 2, 3, 4))
        .astype(mx.bfloat16)
    )
    rope_positions = mx.array(ROPE_POSITIONS, dtype=mx.int32)
    _add_array(arrays, "rope.q", rope_q)
    _add_array(arrays, "rope.k", rope_k)
    _add_array(arrays, "rope.positions", rope_positions)
    for factor in ROPE_FACTORS:
        rope = sml.RotaryEmbedding(
            dim=4,
            original_max_position_embeddings=8,
            effective_max_position_embeddings=int(8 * factor),
            base=10_000.0,
            rope_scaling_factor=factor,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            yarn_truncate=True,
        )
        low, high = sml.yarn_find_correction_range(
            32.0, 1.0, 4, 10_000.0, 8, truncate=True
        )
        valid_positions = [
            position for position in ROPE_POSITIONS if position < 8 * factor
        ]
        selected_q = rope_q[:, :, : len(valid_positions), :]
        selected_k = rope_k[:, :, : len(valid_positions), :]
        selected_positions = mx.array(valid_positions, dtype=mx.int32)
        cos = rope.cos_cached[selected_positions]
        sin = rope.sin_cached[selected_positions]
        output_q, output_k = sml.apply_rotary_pos_emb(selected_q, selected_k, cos, sin)
        mx.eval(rope.cos_cached, rope.sin_cached, output_q, output_k)
        suffix = "" if factor == 2.0 else f".factor_{int(factor)}"
        _add_array(arrays, f"rope.output_q{suffix}", output_q)
        _add_array(arrays, f"rope.output_k{suffix}", output_k)
        _add_array(arrays, f"rope.cos_cache{suffix}", rope.cos_cached)
        _add_array(arrays, f"rope.sin_cache{suffix}", rope.sin_cached)
        rope_config_record[str(factor)] = {
            "rope_scaling_factor": factor,
            "correction_range": [low, high],
            "positions": valid_positions,
        }

    gqa_input = (
        mx.array(GQA_INPUT_VALUES, dtype=mx.float32)
        .reshape((2, 4, 16))
        .astype(mx.bfloat16)
    )
    gqa_output = model.layers[0].self_attn(gqa_input)
    mx.eval(gqa_output)
    _add_array(arrays, "gqa.input", gqa_input)
    _add_array(arrays, "gqa.output", gqa_output)

    cache_input = mx.array(CACHE_INPUT_IDS, dtype=mx.int32)
    full_logits = model(cache_input).logits
    sequential_cache = sml.KVCache(max_seq_len=cache_input.shape[-1])
    sequential_parts = [
        model(cache_input[:, index : index + 1], kv_cache=sequential_cache).logits
        for index in range(cache_input.shape[-1])
    ]
    sequential_logits = mx.concatenate(sequential_parts, axis=1)
    chunked_cache = sml.KVCache(max_seq_len=cache_input.shape[-1])
    chunked_logits = mx.concatenate(
        [
            model(cache_input[:, :2], kv_cache=chunked_cache).logits,
            model(cache_input[:, 2:4], kv_cache=chunked_cache).logits,
            model(cache_input[:, 4:], kv_cache=chunked_cache).logits,
        ],
        axis=1,
    )
    mx.eval(full_logits, sequential_logits, chunked_logits)
    _add_array(arrays, "cache.input_ids", cache_input)
    _add_array(arrays, "cache.full_logits", full_logits)
    _add_array(arrays, "cache.sequential_logits", sequential_logits)
    _add_array(arrays, "cache.chunked_logits", chunked_logits)
    for layer_index, (key_array, value_array) in enumerate(
        zip(chunked_cache.key_cache, chunked_cache.value_cache, strict=True)
    ):
        _add_array(arrays, f"cache.chunked_key.{layer_index}", key_array)
        _add_array(arrays, f"cache.chunked_value.{layer_index}", value_array)

    generation_record = {}
    for name, prompt_values in (
        ("greedy", GREEDY_PROMPT),
        ("seeded", SAMPLED_PROMPT),
    ):
        prompt = mx.array(prompt_values, dtype=mx.int32)
        generation_config = sml.GenerationConfig(**GENERATION_CONFIGS[name])
        generated, margins = _generated_with_margins(
            model, sml, mx, prompt, generation_config, MAX_NEW_TOKENS
        )
        native_generated = model.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            generation_config=generation_config,
        )
        mx.eval(native_generated)
        if not bool(mx.array_equal(generated, native_generated).item()):
            raise RuntimeError(
                f"manual {name} generation diverged from legacy generate"
            )
        _add_array(arrays, f"generation.{name}_prompt", prompt)
        _add_array(arrays, f"generation.{name}_tokens", generated)
        _add_array(arrays, f"generation.{name}_winning_margins", margins)
        generation_record[name] = {
            **GENERATION_CONFIGS[name],
            "max_new_tokens": MAX_NEW_TOKENS,
            "winning_margins": _json_array_values(margins, mx, np),
        }

    literal_logits = mx.array([GENERATION_LOGITS], dtype=mx.float32)
    sampled_token = sml.select_next_token(
        literal_logits,
        sml.GenerationConfig(temperature=0.8, top_p=0.9),
        key=mx.random.key(SAMPLING_SEED),
    )
    mx.eval(sampled_token)
    _add_array(arrays, "generation.logits", literal_logits)
    _add_array(arrays, "generation.sampled_token", sampled_token)

    serial_tokens = []
    serial_margins = []
    for index, prompt_values in enumerate(UNEQUAL_PROMPTS):
        prompt = mx.array([prompt_values], dtype=mx.int32)
        tokens, margins = _generated_with_margins(
            model,
            sml,
            mx,
            prompt,
            sml.GenerationConfig(),
            MAX_NEW_TOKENS,
        )
        serial_tokens.append(tokens)
        serial_margins.append(margins)
        _add_array(arrays, f"generation.serial_prompt.{index}", prompt)
        _add_array(arrays, f"generation.serial_tokens.{index}", tokens)
        _add_array(arrays, f"generation.serial_winning_margins.{index}", margins)
    padded_prompts = mx.array(PADDED_UNEQUAL_PROMPTS, dtype=mx.int32)
    padded_tokens, padded_margins = _generated_with_margins(
        model, sml, mx, padded_prompts, sml.GenerationConfig(), MAX_NEW_TOKENS
    )
    _add_array(arrays, "generation.padded_unequal_prompts", padded_prompts)
    _add_array(arrays, "generation.padded_unequal_tokens", padded_tokens)
    _add_array(arrays, "generation.padded_unequal_winning_margins", padded_margins)

    serial_loglikelihoods = []
    for index, (input_values, label_values) in enumerate(
        zip(LOGLIKELIHOOD_INPUTS, LOGLIKELIHOOD_LABELS, strict=True)
    ):
        ll_input = mx.array([input_values], dtype=mx.int32)
        ll_labels = mx.array([label_values], dtype=mx.int32)
        score = _sequence_loglikelihood(model, mx, ll_input, ll_labels, 3)
        mx.eval(score)
        serial_loglikelihoods.append(score)
        _add_array(arrays, f"loglikelihood.serial_input_ids.{index}", ll_input)
        _add_array(arrays, f"loglikelihood.serial_labels.{index}", ll_labels)
        _add_array(arrays, f"loglikelihood.serial_score.{index}", score)
    padded_ll_input = mx.array(LOGLIKELIHOOD_PADDED_INPUTS, dtype=mx.int32)
    padded_ll_labels = mx.array(LOGLIKELIHOOD_PADDED_LABELS, dtype=mx.int32)
    padded_scores = _sequence_loglikelihood(
        model, mx, padded_ll_input, padded_ll_labels, 3
    )
    mx.eval(padded_scores)
    _add_array(arrays, "loglikelihood.padded_input_ids", padded_ll_input)
    _add_array(arrays, "loglikelihood.padded_labels", padded_ll_labels)
    _add_array(arrays, "loglikelihood.padded_scores", padded_scores)

    swag_input_ids = mx.array(SWAG_INPUT_IDS, dtype=mx.int32)
    swag_labels = mx.array(SWAG_LABELS, dtype=mx.int32)
    swag_sums = ft_swag.score_swag_candidates(model, swag_input_ids, swag_labels, 3)
    mx.eval(swag_sums)
    _add_array(arrays, "swag_legacy_sum.input_ids", swag_input_ids)
    _add_array(arrays, "swag_legacy_sum.labels", swag_labels)
    _add_array(arrays, "swag_legacy_sum.scores", swag_sums)

    utilities.set_seed(LORA_SEED)
    lora_model = deepcopy(model)
    lora.apply_lora(
        lora_model,
        lora.LoRAConfig(
            rank=LORA_CONFIG["rank"],
            alpha=LORA_CONFIG["alpha"],
            scaling_mode=LORA_CONFIG["scaling_mode"],
            dropout=LORA_CONFIG["dropout"],
            target_modules=tuple(LORA_CONFIG["target_modules"]),
        ),
        parameter_initializer_range={
            "lora_a": LORA_CONFIG["lora_a_initializer_range"],
            "lora_b": LORA_CONFIG["lora_b_initializer_range"],
        },
    )
    lora_model.eval()
    mx.eval(lora_model.parameters())
    lora_base_mapping, lora_adapter_mapping = _store_lora_state(
        arrays, lora_model, mx, np, tree_flatten
    )
    lora_input = mx.array([[1, 38, 39, 40]], dtype=mx.int32)
    lora_forward = lora_model(lora_input).logits
    merged_model = deepcopy(lora_model)
    lora.merge_lora(merged_model)
    merged_model.eval()
    mx.eval(merged_model.parameters())
    lora_merged = merged_model(lora_input).logits
    mx.eval(lora_forward, lora_merged)
    _add_array(arrays, "lora.input_ids", lora_input)
    _add_array(arrays, "lora.forward_logits", lora_forward)
    _add_array(arrays, "lora.merged_logits", lora_merged)

    compiled_input_a = mx.array([[1, 41, 42]], dtype=mx.int32)
    compiled_input_b = mx.array([[1, 43, 44, 45]], dtype=mx.int32)
    compiled_forward = mx.compile(
        lambda current_input: _legacy_forward_without_validation(model, current_input)
    )
    compiled_output_a = compiled_forward(compiled_input_a)
    mx.eval(compiled_output_a)
    compiled_output_b = compiled_forward(compiled_input_b)
    mx.eval(compiled_output_b)
    _add_array(arrays, "compiled_state.input_ids.0", compiled_input_a)
    _add_array(arrays, "compiled_state.logits.0", compiled_output_a)
    _add_array(arrays, "compiled_state.input_ids.1", compiled_input_b)
    _add_array(arrays, "compiled_state.logits.1", compiled_output_b)

    arrays = dict(sorted(arrays.items()))
    mx.eval(arrays)
    array_metadata = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "payload_identity": _array_payload_identity(value, mx, np),
        }
        for name, value in arrays.items()
    }
    cases = {
        "model": {
            "arrays": [name for name in arrays if name.startswith("model.")],
            "input_ids": MODEL_INPUT_IDS,
            "labels": MODEL_LABELS,
        },
        "parameter_state": {
            "arrays": [name for name in arrays if name.startswith("model_state.")]
        },
        "rope": {
            "arrays": [name for name in arrays if name.startswith("rope.")],
            "factors": rope_config_record,
        },
        "gqa": {"arrays": [name for name in arrays if name.startswith("gqa.")]},
        "cache": {
            "arrays": [name for name in arrays if name.startswith("cache.")],
            "chunk_sizes": [2, 2, 2],
        },
        "generation": {
            "arrays": [name for name in arrays if name.startswith("generation.")],
            "configurations": generation_record,
            "unequal_prompts": UNEQUAL_PROMPTS,
            "padded_unequal_prompts": PADDED_UNEQUAL_PROMPTS,
            "primitive_sampling_seed": SAMPLING_SEED,
        },
        "loglikelihood": {
            "arrays": [name for name in arrays if name.startswith("loglikelihood.")],
            "serial_inputs": LOGLIKELIHOOD_INPUTS,
            "serial_labels": LOGLIKELIHOOD_LABELS,
            "padded_inputs": LOGLIKELIHOOD_PADDED_INPUTS,
            "padded_labels": LOGLIKELIHOOD_PADDED_LABELS,
        },
        "tied_gradient_leaves": {
            "arrays": [name for name in arrays if name.startswith("tied.")]
        },
        "lora": {
            "arrays": [
                name
                for name in arrays
                if name.startswith(("lora.", "lora_base_state.", "lora_state."))
            ],
            "config": LORA_CONFIG,
        },
        "compiled_state": {
            "arrays": [name for name in arrays if name.startswith("compiled_state.")],
            "consecutive_calls": 2,
            "input_seed": COMPILED_INPUT_SEED,
        },
        "swag_legacy_sum": {
            "arrays": [name for name in arrays if name.startswith("swag_legacy_sum.")],
            "reduction": "sum",
        },
    }
    control = {
        "schema": "sml-legacy-equivalence-v1",
        "source_commit": args.source_commit,
        "source_resolved_commit": _resolve_commit(args.source_root, "HEAD"),
        "capture_tool_commit": args.resolved_capture_tool_commit,
        "capture_tool_identity": args.capture_tool_identity,
        "capture_tool_identity_files": [
            path.relative_to(PROJECT_ROOT).as_posix() for path in CAPTURE_IDENTITY_FILES
        ],
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "mlx_version": _distribution_version("mlx"),
        "parameter_dtype": "bfloat16",
        "dropout": 0.0,
        "seeds": {
            "python": PYTHON_SEED,
            "numpy": NUMPY_SEED,
            "model": MODEL_SEED,
            "lora": LORA_SEED,
            "sampling": SAMPLING_SEED,
            "compiled_input": COMPILED_INPUT_SEED,
        },
        "pretraining_model_config": PRETRAINING_MODEL_CONFIG,
        "parameter_state": {
            "mapping": parameter_mapping,
            "tied_source_names": ["embed_tokens.weight", "lm_head.weight"],
            "canonical_destination": "embed_tokens.weight",
            "tied_alias_byte_identical": tied_identical,
            "payload_identity": _state_payload_identity(parameter_mapping),
        },
        "lora_parameter_state": {
            "base_mapping": lora_base_mapping,
            "adapter_mapping": lora_adapter_mapping,
            "base_payload_identity": _state_payload_identity(lora_base_mapping),
            "adapter_payload_identity": _state_payload_identity(lora_adapter_mapping),
        },
        "cases": cases,
        "arrays": array_metadata,
    }
    return arrays, control


def _load_legacy_modules(source_root: Path):
    sys.dont_write_bytecode = True
    source_directory = (source_root / "v2" / "src").resolve()
    source_text = str(source_directory)
    sys.path[:] = [source_text, *[entry for entry in sys.path if entry != source_text]]

    import ft_swag
    import lora
    import mlx.core as mx
    import numpy as np
    import sml
    import train_sml as train
    import utils as utilities
    from mlx import nn
    from mlx.utils import tree_flatten

    for module in (sml, train, lora, ft_swag, utilities):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(source_directory):
            raise RuntimeError(
                f"legacy module {module.__name__!r} resolved outside source checkout: "
                f"{module_path}"
            )
    return mx, nn, np, tree_flatten, sml, train, lora, ft_swag, utilities


def capture_from_loaded_legacy_source(args: argparse.Namespace) -> None:
    if args.source_root is None:
        raise RuntimeError("worker requires --source-root")
    expected_identity = capture_tool_identity()
    if args.capture_tool_identity != expected_identity:
        raise RuntimeError("capture-tool content identity changed before worker launch")
    if args.resolved_capture_tool_commit != _resolve_commit(PROJECT_ROOT, "HEAD"):
        raise RuntimeError("worker capture-tool commit does not match driver checkout")
    modules = _load_legacy_modules(args.source_root)
    arrays, control = _capture_references(args, modules)
    mx = modules[0]

    args.output_control.parent.mkdir(parents=True, exist_ok=True)
    args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output_arrays), arrays)
    args.output_control.write_text(
        json.dumps(
            control, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"source_commit={control['source_resolved_commit']}")
    print(f"capture_tool_commit={control['capture_tool_commit']}")
    print(f"capture_tool_identity={control['capture_tool_identity']}")
    print(f"array_count={len(arrays)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        if args.source_commit != PINNED_SOURCE_COMMIT:
            raise RuntimeError(f"worker source commit must be {PINNED_SOURCE_COMMIT}")
        assert_clean_checkout(args.source_root, expected_commit=PINNED_SOURCE_COMMIT)
        capture_from_loaded_legacy_source(args)
        return 0

    if args.source_commit != PINNED_SOURCE_COMMIT:
        raise RuntimeError(f"source commit must be {PINNED_SOURCE_COMMIT}")
    resolved_capture_tool_commit = assert_clean_checkout(
        PROJECT_ROOT, expected_commit=args.capture_tool_commit
    )
    tool_identity = capture_tool_identity()
    print(f"driver_commit={resolved_capture_tool_commit}")
    print("driver_clean=true")
    print(f"capture_tool_identity={tool_identity}")
    with detached_worktree(PROJECT_ROOT, commit=PINNED_SOURCE_COMMIT) as source_root:
        source_commit = assert_clean_checkout(
            source_root, expected_commit=PINNED_SOURCE_COMMIT
        )
        print(f"source_worktree_commit={source_commit}")
        print("source_worktree_clean=true")
        run_worker_from_external_script(
            args,
            source_root=source_root,
            resolved_capture_tool_commit=resolved_capture_tool_commit,
            tool_identity=tool_identity,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
