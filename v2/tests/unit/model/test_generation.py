from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
from sml.model.config import GenerationConfig
from sml.model.generation import (
    TokenSelection,
    apply_no_repeat_ngram,
    apply_repetition_penalty,
    select_next_token,
    select_next_token_arrays,
)


def _assert_array_equal(actual: mx.array, expected: mx.array) -> None:
    mx.eval(actual, expected)
    assert bool(mx.array_equal(actual, expected).item())


def test_greedy_selection_returns_argmax_without_advancing_key():
    """Greedy decoding must not consume randomness or retain a sample axis."""
    key = mx.random.key(11)

    selected = select_next_token(
        mx.array(
            [[0.0, 3.0, 1.0], [4.0, -1.0, 2.0]],
            dtype=mx.float32,
        ),
        GenerationConfig(temperature=0.0),
        key,
    )

    assert isinstance(selected, TokenSelection)
    assert [field.name for field in fields(TokenSelection)] == [
        "token_ids",
        "next_key",
    ]
    assert selected.token_ids.shape == (2,)
    assert selected.token_ids.dtype == mx.uint32
    assert selected.next_key.shape == (2,)
    assert selected.next_key.dtype == mx.uint32
    _assert_array_equal(selected.token_ids, mx.array([1, 0], dtype=mx.uint32))
    _assert_array_equal(selected.next_key, key)


def test_repetition_penalty_uses_logit_sign_and_ignores_inactive_storage():
    """Padding tokens must not be penalized, and negative logits move downward."""
    logits = mx.array(
        [
            [4.0, -2.0, 3.0, -1.0, 0.5, -0.5],
            [-4.0, 2.0, 6.0, -3.0, 1.5, -0.25],
        ],
        dtype=mx.float32,
    )
    tokens = mx.array(
        [
            [0, 1, 4, 5],
            [2, 3, 1, 0],
        ],
        dtype=mx.int32,
    )
    logical_lengths = mx.array([2, 3], dtype=mx.int32)

    actual = apply_repetition_penalty(logits, tokens, logical_lengths, 2.0)

    expected = mx.array(
        [
            [2.0, -4.0, 3.0, -1.0, 0.5, -0.5],
            [-4.0, 1.0, 3.0, -6.0, 1.5, -0.25],
        ],
        dtype=mx.float32,
    )
    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    _assert_array_equal(actual, expected)


def test_disabled_processors_preserve_logits():
    """Disabled generation controls must be mathematical identities."""
    logits = mx.array([[1.0, -2.0, 3.0, 0.5]], dtype=mx.float32)
    tokens = mx.array([[1, 3, 2, 0]], dtype=mx.int32)
    lengths = mx.array([2], dtype=mx.int32)

    repeated = apply_repetition_penalty(logits, tokens, lengths, 1.0)
    no_repeat = apply_no_repeat_ngram(logits, tokens, lengths, 0)

    _assert_array_equal(repeated, logits)
    _assert_array_equal(no_repeat, logits)


def test_no_repeat_unigram_masks_only_tokens_before_each_logical_length():
    """Unigram blocking must ignore fixed-capacity padding independently per row."""
    logits = mx.zeros((2, 7), dtype=mx.float32)
    tokens = mx.array(
        [
            [1, 4, 6, 5, 0],
            [2, 3, 2, 6, 1],
        ],
        dtype=mx.int32,
    )
    lengths = mx.array([2, 4], dtype=mx.int32)

    actual = apply_no_repeat_ngram(logits, tokens, lengths, 1)

    expected = mx.array(
        [
            [0.0, float("-inf"), 0.0, 0.0, float("-inf"), 0.0, 0.0],
            [
                0.0,
                0.0,
                float("-inf"),
                float("-inf"),
                0.0,
                0.0,
                float("-inf"),
            ],
        ],
        dtype=mx.float32,
    )
    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    _assert_array_equal(actual, expected)


def test_no_repeat_trigram_uses_each_rows_active_suffix_only():
    """A matching active suffix must ban its successor without reading capacity tail."""
    logits = mx.zeros((2, 8), dtype=mx.float32)
    tokens = mx.array(
        [
            [1, 2, 3, 1, 2, 7, 7],
            [4, 5, 4, 6, 4, 6, 7],
        ],
        dtype=mx.int32,
    )
    lengths = mx.array([5, 4], dtype=mx.int32)

    actual = apply_no_repeat_ngram(logits, tokens, lengths, 3)

    expected = mx.array(
        [
            [0.0, 0.0, 0.0, float("-inf"), 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=mx.float32,
    )
    _assert_array_equal(actual, expected)


def test_same_sampling_key_replays_token_and_next_key():
    """Explicit sampling state must replay both the draw and future stream."""
    logits = mx.array([0.0, 0.5, 1.0, 1.5], dtype=mx.float32)
    key = mx.random.key(29)

    first = select_next_token_arrays(logits, key, temperature=0.8, top_p=0.9)
    replay = select_next_token_arrays(logits, key, temperature=0.8, top_p=0.9)

    mx.eval(first, replay)
    assert first[0].shape == ()
    assert first[0].dtype == mx.uint32
    assert first[1].shape == (2,)
    assert first[1].dtype == mx.uint32
    _assert_array_equal(first[0], replay[0])
    _assert_array_equal(first[1], replay[1])
    _assert_array_equal(first[1], mx.random.split(key)[0])


def test_top_p_uses_stable_order_for_equal_logits():
    """A one-token nucleus over tied logits must retain the first vocabulary ID."""
    token_id, _next_key = select_next_token_arrays(
        mx.zeros((4,), dtype=mx.float32),
        mx.random.key(31),
        temperature=1.0,
        top_p=0.2,
    )

    _assert_array_equal(token_id, mx.array(0, dtype=mx.uint32))


def test_bfloat16_top_p_uses_fp32_probability_cutoff_at_base_vocabulary(
    monkeypatch,
):
    """BF16 model logits must not truncate a production-size nucleus early."""
    captured_logits = []
    categorical = mx.random.categorical

    def capture_sampling_logits(logits, *, key):
        captured_logits.append(logits)
        return categorical(logits, key=key)

    monkeypatch.setattr(mx.random, "categorical", capture_sampling_logits)

    select_next_token_arrays(
        mx.zeros((28_672,), dtype=mx.bfloat16),
        mx.random.key(35),
        temperature=1.0,
        top_p=0.9,
    )

    assert len(captured_logits) == 1
    masked_logits = captured_logits[0]
    masked_count = mx.sum(masked_logits == float("-inf"))
    mx.eval(masked_count)
    assert masked_logits.dtype == mx.bfloat16
    assert int(masked_count.item()) == 2_867


def test_compiled_vmap_accepts_one_distinct_key_per_row():
    """Batched compilation must never pass a stacked key to categorical sampling."""
    logits = mx.array(
        [
            [0.0, 0.5, 1.0, 1.5],
            [1.5, 1.0, 0.5, 0.0],
            [0.0, 1.5, 0.5, 1.0],
        ],
        dtype=mx.float32,
    )
    keys = mx.stack([mx.random.key(37), mx.random.key(41), mx.random.key(43)])

    def select_row(logits_row, key):
        return select_next_token_arrays(
            logits_row,
            key,
            temperature=0.8,
            top_p=0.9,
        )

    compiled_select = mx.compile(mx.vmap(select_row, in_axes=(0, 0)))
    token_ids, next_keys = compiled_select(logits, keys)

    mx.eval(token_ids, next_keys)
    assert token_ids.shape == (3,)
    assert token_ids.dtype == mx.uint32
    assert next_keys.shape == (3, 2)
    assert next_keys.dtype == mx.uint32
    _assert_array_equal(next_keys, mx.vmap(lambda key: mx.random.split(key)[0])(keys))


def test_generation_kernels_contain_no_host_conversion_or_synchronization():
    """Processor kernels must remain traceable when generation moves under compile."""
    module_path = Path(inspect.getfile(select_next_token_arrays))
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    forbidden_attributes = {"item", "tolist", "eval", "numpy"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name.split(".")[0] for alias in node.names]
            assert "numpy" not in imported
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_attributes
