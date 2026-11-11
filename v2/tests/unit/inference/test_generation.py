from __future__ import annotations

import mlx.core as mx
import pytest
from sml.inference import (
    GenerationRequest,
    InferenceRuntimeConfig,
    InferenceSession,
)
from sml.model.config import GenerationConfig


@pytest.mark.parametrize("max_new_tokens", [1, 2, 7, 8, 9, 15, 16, 17])
def test_decode_bounds_forward_work_and_omits_terminal_prediction(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    max_new_tokens: int,
) -> None:
    forward_lengths: list[int] = []
    forward = tiny_session._model.forward_arrays

    def controlled_forward(parameters, input_ids, **kwargs):
        forward_lengths.append(input_ids.shape[1])
        logits, cache_state, key = forward(parameters, input_ids, **kwargs)
        forced_logits = mx.where(
            mx.arange(logits.shape[-1])[None, None, :] == 4,
            0.0,
            -100.0,
        )
        return mx.broadcast_to(forced_logits, logits.shape), cache_state, key

    monkeypatch.setattr(tiny_session._model, "forward_arrays", controlled_forward)
    result = tiny_session.generate("alpha", GenerationRequest(max_new_tokens))

    assert result.token_ids == (4,) * max_new_tokens
    assert forward_lengths == [
        tiny_session._select_length_bucket(len(tiny_session._encode_prompt("alpha"))),
        *([1] * (max_new_tokens - 1)),
    ]
    assert tiny_session.buffer_pool.active_leases == 0


@pytest.mark.parametrize("include_prompt", [False, True])
def test_zero_token_generation_skips_model_and_buffers(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    include_prompt: bool,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("a zero-token request does not need model computation or buffers")

    monkeypatch.setattr(tiny_session.buffer_pool, "lease", forbidden)
    monkeypatch.setattr(tiny_session._model, "forward_arrays", forbidden)
    result = tiny_session.generate(
        "alpha",
        GenerationRequest(
            0, config=GenerationConfig(seed=19), include_prompt=include_prompt
        ),
    )
    expected_ids = tiny_session._encode_prompt("alpha") if include_prompt else ()
    assert result.token_ids == expected_ids
    assert result.text == tiny_session.resolved_model.tokenizer.processor.decode(
        list(expected_ids)
    )
    assert result.seed == 19
    assert result.model == tiny_session.model_identity


@pytest.mark.parametrize("sampled", [False, True])
def test_tail_chunks_preserve_tokens_across_chunk_sizes(
    tiny_session: InferenceSession,
    sampled: bool,
) -> None:
    single_step = InferenceSession(
        tiny_session.resolved_model,
        InferenceRuntimeConfig(decode_chunk_size=1),
    )
    config = GenerationConfig(
        temperature=0.8 if sampled else 0.0,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        seed=43,
    )
    items = [
        ("alpha", GenerationRequest(max_new_tokens=count, config=config))
        for count in (1, 9, 10, 12, 17)
    ]

    expected = single_step.generate_batch(items)
    actual = tiny_session.generate_batch(items)

    assert actual == expected


@pytest.mark.parametrize(
    ("stop_after", "expected_chunks"),
    [((1,), 1), ((1, 5), 3), ((1, 8), 4)],
)
def test_decode_stops_when_all_real_requests_finish(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
    stop_after: tuple[int, ...],
    expected_chunks: int,
) -> None:
    tiny_session._runtime = InferenceRuntimeConfig(
        batch_size_buckets=(4,), decode_chunk_size=2
    )
    prompt = "alpha"
    prompt_length = len(tiny_session._encode_prompt(prompt))
    config = tiny_session.resolved_model.model_config
    eos_id = config.eos_token_id
    other_id = 4
    thresholds = mx.array(
        [prompt_length + count - 2 for count in stop_after]
        + [0] * (4 - len(stop_after)),
        dtype=mx.int32,
    )
    forward = tiny_session._model.forward_arrays

    def controlled_forward(parameters, input_ids, **kwargs):
        _logits, cache, key = forward(parameters, input_ids, **kwargs)
        selected = mx.where(
            kwargs["positions"] >= thresholds[:, None], eos_id, other_id
        )
        logits = mx.where(
            mx.arange(config.vocab_size)[None, None, :] == selected[:, :, None],
            0.0,
            -100.0,
        )
        if kwargs.get("logits_positions") is not None:
            logits = mx.take_along_axis(
                logits, kwargs["logits_positions"][:, :, None], axis=1
            )
        return logits, cache, key

    chunks = 0
    compiled_kernels = tiny_session._compiled_kernels

    def counted_kernels(*args):
        prefill, decode = compiled_kernels(*args)

        def counted_decode(*values):
            nonlocal chunks
            chunks += 1
            return decode(*values)

        return prefill, counted_decode

    monkeypatch.setattr(tiny_session._model, "forward_arrays", controlled_forward)
    monkeypatch.setattr(tiny_session, "_compiled_kernels", counted_kernels)
    results = tiny_session.generate_batch(
        [(prompt, GenerationRequest(max_new_tokens=7)) for _ in stop_after]
    )

    assert chunks == expected_chunks
    for result, count in zip(results, stop_after, strict=True):
        expected = (other_id,) * min(count - 1, 7)
        if count <= 7:
            expected += (eos_id,)
        assert result.token_ids == expected
    assert tiny_session.buffer_pool.active_leases == 0
