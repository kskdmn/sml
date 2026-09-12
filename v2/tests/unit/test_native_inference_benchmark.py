import mlx.core as mx
import numpy as np
import pytest
from sml.model.cache import KVCache
from sml.model.generation import (
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    select_next_token,
)

from v2.benchmarks.adapters.native_inference import make_runtime
from v2.benchmarks.workload import build_canonical_workload


@pytest.fixture
def workload():
    return build_canonical_workload(
        model_overrides={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_layers": 1,
            "num_q_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 32,
            "original_context_length": 32,
            "hidden_dropout": 0.1,
        },
        loader_overrides={"sequence_length": 8},
        generation_overrides={
            "request_count": 2,
            "prompt_tokens": 4,
            "decode_chunk_size": 2,
        },
        row_count=32,
    )


@pytest.mark.parametrize("metric", ["inference-prefill", "inference-decode"])
def test_native_inference_repeats_canonical_work_after_warmup(
    metric, workload, tmp_path
):
    runtime = make_runtime(metric, workload, tmp_path)
    runtime.run(3)
    first = runtime.last_output
    runtime.reset_after_warmup()
    assert runtime.ordered_request_ids == []
    work = runtime.run(3)
    assert work == 3 * (4 if metric == "inference-prefill" else 2)
    assert runtime.ordered_request_ids == [0, 1, 0]
    runtime.reset_measured_order()
    runtime.run(1)
    if metric == "inference-prefill":
        np.testing.assert_array_equal(
            runtime.last_output[0].astype(mx.float32), first[0].astype(mx.float32)
        )
        assert runtime.last_output[1][2].tolist() == [4]
    else:
        np.testing.assert_array_equal(runtime.last_output[2], first[2])
        assert runtime.last_output[0][2].tolist() == [6]
        assert runtime.last_output[3].tolist() == [7]
        # Warmup and previous measured chunks must not alter initial cache state.
        assert runtime._states[0][0][2].tolist() == [4]


def test_native_prefill_matches_public_cached_model(workload, tmp_path):
    runtime = make_runtime("inference-prefill", workload, tmp_path)
    cache = KVCache.allocate(runtime.config, 1, 32, mx.bfloat16)
    expected = runtime.model(
        mx.array(runtime.prompts[0][None, :], dtype=mx.int32), cache=cache
    )
    runtime.run(1)
    assert runtime.last_output[0].shape == (1, 1, runtime.config.vocab_size)
    np.testing.assert_allclose(
        np.asarray(runtime.last_output[0].astype(mx.float32)),
        np.asarray(expected.logits[:, -1:, :].astype(mx.float32)),
        rtol=0.02,
        atol=0.002,
    )
    assert runtime.last_output[1][2].tolist() == cache.state[2].tolist()


@pytest.mark.parametrize("penalty,ngram", [(1.0, 0), (1.2, 2)])
def test_native_decode_matches_eager_fixed_chunk(penalty, ngram, workload, tmp_path):
    generation = dict(workload.generation)
    generation.update(repetition_penalty=penalty, no_repeat_ngram_size=ngram)
    workload = build_canonical_workload(
        model_overrides=workload.model,
        loader_overrides=workload.loader,
        generation_overrides=generation,
        row_count=32,
    )
    runtime = make_runtime("inference-decode", workload, tmp_path)
    cache = KVCache.allocate(runtime.config, 1, 32, mx.bfloat16)
    tokens = mx.array(runtime.prompts[0][None, :], dtype=mx.int32)
    logits = runtime.model(tokens, cache=cache).logits[:, -1, :]
    key = mx.random.key(runtime.generation.seed)
    # One initial selection belongs to prefill, then each measured step performs
    # another forward and selection, matching the legacy benchmark boundary.
    for step in range(runtime.chunk_size + 1):
        lengths = mx.array([tokens.shape[1]], dtype=mx.int32)
        scores = apply_repetition_penalty(logits, tokens, lengths, penalty)
        scores = apply_no_repeat_ngram(scores, tokens, lengths, ngram)
        selection = select_next_token(scores, runtime.generation, key)
        selected = selection.token_ids.reshape((1, 1)).astype(mx.int32)
        key = selection.next_key
        tokens = mx.concatenate([tokens, selected], axis=1)
        if step < runtime.chunk_size:
            logits = runtime.model(selected, cache=cache).logits[:, -1, :]
    runtime.run(1)
    np.testing.assert_array_equal(runtime.last_output[2][:, : tokens.shape[1]], tokens)
    assert runtime.last_output[0][2].tolist() == cache.state[2].tolist()


def test_inference_metrics_use_identical_initial_parameters(workload, tmp_path):
    prefill = make_runtime("inference-prefill", workload, tmp_path)
    decode = make_runtime("inference-decode", workload, tmp_path)
    assert prefill.initial_parameter_identity == decode.initial_parameter_identity


def test_native_inference_rejects_unpaired_sampling(workload, tmp_path):
    from dataclasses import replace

    sampled = replace(workload, generation={**workload.generation, "temperature": 0.5})
    with pytest.raises(ValueError, match="only greedy"):
        make_runtime("inference-decode", sampled, tmp_path)


@pytest.mark.parametrize("metric", ["inference-prefill", "inference-decode"])
def test_native_inference_rejects_truncated_canonical_prompts(
    metric, workload, tmp_path
):
    from dataclasses import replace

    truncated = replace(
        workload, generation={**workload.generation, "prompt_tokens": 10}
    )
    with pytest.raises(ValueError, match="exceeds canonical row width"):
        make_runtime(metric, truncated, tmp_path)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"request_order": "reverse"}, "fixed canonical request order"),
        ({"stop_on_eos": False}, "unsupported.*generation fields"),
        ({"prompt_tokens": 4.5}, "positive integer"),
        ({"decode_chunk_size": True}, "positive integer"),
        ({"no_repeat_ngram_size": 1.5}, "non-negative"),
    ],
)
def test_native_inference_rejects_ignored_or_coerced_generation_overrides(
    overrides, match, workload, tmp_path
):
    from dataclasses import replace

    changed = replace(workload, generation={**workload.generation, **overrides})
    with pytest.raises(ValueError, match=match):
        make_runtime("inference-decode", changed, tmp_path)
