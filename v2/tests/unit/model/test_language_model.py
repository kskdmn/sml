from __future__ import annotations

import ast
import inspect
from dataclasses import fields, replace
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from sml.model.cache import KVCache
from sml.model.config import ModelConfig
from sml.model.language_model import (
    ForwardOutput,
    SMLLanguageModel,
    causal_lm_loss,
    model_parameter_specs,
)
from sml.model.layers import LoRAAdapterSpec, LoRAForwardPolicy


def _tiny_model_config(**overrides) -> ModelConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_layers": 2,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "intermediate_size": 32,
        "original_context_length": 8,
        "rope_scaling_factor": 2.0,
        "hidden_dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _assert_close(
    actual: mx.array,
    expected: mx.array,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def _causal_batch_loss(model, input_ids, labels):
    output = model(input_ids, training=False)
    return causal_lm_loss(output.logits, labels, labels != model.config.pad_token_id)


def test_tied_model_registers_one_vocabulary_parameter_and_sums_gradients(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(
        replace(_tiny_model_config(), tie_word_embeddings=True),
        key=mx.random.key(7),
    )
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    names = [name for name, _value in tree_flatten(model.trainable_parameters())]
    assert names.count("embed_tokens.weight") == 1
    assert "lm_head.weight" not in names
    loss_and_grad = nn.value_and_grad(model, _causal_batch_loss)

    _loss, grads = loss_and_grad(
        model,
        legacy_arrays["model.input_ids"],
        legacy_arrays["model.labels"],
    )

    expected = legacy_arrays["tied.embedding_grad"] + legacy_arrays["tied.head_grad"]
    _assert_close(grads["embed_tokens"]["weight"], expected, atol=2e-2, rtol=2e-2)


def test_base_precision_boundaries():
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(9))

    output = model(mx.array([[1, 2, 3]], dtype=mx.int32), training=False)

    mx.eval(output.logits)
    assert all(
        value.dtype == mx.bfloat16 for _name, value in tree_flatten(model.parameters())
    )
    assert output.logits.dtype == mx.bfloat16
    assert model.layers[0].self_attn.rope.cos_cached.dtype == mx.float32


def test_untied_model_registers_an_independent_vocabulary_head():
    model = SMLLanguageModel(
        _tiny_model_config(tie_word_embeddings=False),
        key=mx.random.key(11),
    )
    parameters = dict(tree_flatten(model.trainable_parameters()))

    assert parameters["embed_tokens.weight"].shape == (64, 16)
    assert parameters["lm_head.weight"].shape == (64, 16)
    assert parameters["embed_tokens.weight"] is not parameters["lm_head.weight"]
    assert parameters["lm_head.weight"].dtype == mx.bfloat16


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_model_parameter_specs_match_plain_model_state(tie_word_embeddings):
    """Catches artifact metadata drifting from the persisted model leaves."""
    config = _tiny_model_config(tie_word_embeddings=tie_word_embeddings)
    model = SMLLanguageModel(config, key=mx.random.key(41))
    actual = {
        spec.name: (spec.shape, spec.dtype) for spec in model_parameter_specs(config)
    }
    expected = {
        name: (tuple(value.shape), str(value.dtype).removeprefix("mlx.core."))
        for name, value in tree_flatten(model.parameters())
    }

    assert actual == expected
    assert actual["embed_tokens.weight"] == ((64, 16), "bfloat16")
    assert ("lm_head.weight" in actual) is (not tie_word_embeddings)


def test_model_config_rejects_invalid_parameter_projection_dimensions():
    """Catches accepting a configuration that cannot describe model leaves."""
    with pytest.raises(ValueError, match="divisible"):
        _tiny_model_config(hidden_size=15)


def test_initialization_is_replayable_and_replaces_parameters_explicitly():
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(13))
    original = dict(tree_flatten(model.parameters()))

    result = model.initialize_state(mx.random.key(29))
    reinitialized = dict(tree_flatten(model.parameters()))
    replay = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(29))
    replayed = dict(tree_flatten(replay.parameters()))

    assert result is None
    assert original.keys() == reinitialized.keys() == replayed.keys()
    assert any(original[name] is not reinitialized[name] for name in original)
    for name, value in reinitialized.items():
        _assert_close(value, replayed[name])


def test_forward_arrays_uses_explicit_parameters_without_installing_them():
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(17))
    registered_before = dict(tree_flatten(model.parameters()))
    zero_parameters = tree_map(mx.zeros_like, model.parameters())
    input_ids = mx.array([[1, 4, 5]], dtype=mx.int32)

    logits, cache_state, next_key = model.forward_arrays(
        zero_parameters,
        input_ids,
        attention_mask=None,
        positions=None,
        cache_state=None,
        training=False,
        key=None,
    )

    mx.eval(logits)
    registered_after = dict(tree_flatten(model.parameters()))
    assert cache_state is None
    assert next_key is None
    assert all(
        registered_after[name] is registered_before[name] for name in registered_before
    )
    _assert_close(logits, mx.zeros_like(logits))


def test_forward_arrays_rejects_lora_policy_entry_for_plain_projection():
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(17))
    model.lora_forward_policy = LoRAForwardPolicy(
        (LoRAAdapterSpec("layers.0.self_attn.q_proj", 1.0, 0.0),)
    )

    with pytest.raises(ValueError, match="plain linear parameters"):
        model.forward_arrays(
            model.parameters(),
            mx.array([[1, 4, 5]], dtype=mx.int32),
            attention_mask=None,
            positions=None,
            cache_state=None,
            training=False,
            key=None,
        )


def test_attention_mask_and_positions_make_padding_inert(
    legacy_arrays,
    legacy_control,
    load_legacy_model_state,
):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(19))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    padded_ids = mx.array([[1, 4, 5], [1, 6, 3]], dtype=mx.int32)
    valid_mask = mx.array([[True, True, True], [True, True, False]])
    positions = mx.array([[0, 1, 2], [0, 1, 0]], dtype=mx.int32)

    padded = model(
        padded_ids,
        attention_mask=valid_mask,
        positions=positions,
        training=False,
    ).logits
    serial = model(
        mx.array([[1, 6]], dtype=mx.int32),
        training=False,
    ).logits

    _assert_close(padded[1, :2], serial[0], atol=2e-2, rtol=2e-2)


def test_training_forward_splits_one_key_per_active_layer_dropout():
    config = _tiny_model_config(hidden_dropout=0.5)
    model = SMLLanguageModel(config, key=mx.random.key(23))
    input_ids = mx.array([[1, 4, 5]], dtype=mx.int32)
    key = mx.random.key(31)

    first = model(input_ids, training=True, key=key)
    replay = model(input_ids, training=True, key=key)

    _assert_close(first.logits, replay.logits)
    _assert_close(first.next_key, replay.next_key)
    expected_key = key
    for _layer_index in range(config.num_layers):
        expected_key = mx.random.split(expected_key)[0]
    _assert_close(first.next_key, expected_key)


def test_causal_lm_loss_masks_invalid_tokens_and_returns_float32():
    logits = mx.array(
        [
            [
                [4.0, 1.0, -2.0],
                [0.0, 2.0, 1.0],
                [-3.0, 0.5, 3.0],
            ]
        ],
        dtype=mx.bfloat16,
    )
    labels = mx.array([[0, 1, 2]], dtype=mx.int32)
    valid_mask = mx.array([[True, False, True]])

    actual = causal_lm_loss(logits, labels, valid_mask)
    logits_fp32 = logits.astype(mx.float32)
    token_losses = nn.losses.cross_entropy(
        logits_fp32.reshape((-1, 3)),
        labels.reshape((-1,)),
        reduction="none",
    ).reshape(labels.shape)
    expected = (token_losses[0, 0] + token_losses[0, 2]) / 2

    assert actual.dtype == mx.float32
    _assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_forward_does_not_synchronize_for_token_validation(monkeypatch):
    model = SMLLanguageModel(_tiny_model_config(), key=mx.random.key(37))
    calls = []

    def record_eval(*arrays):
        calls.append(arrays)

    monkeypatch.setattr(mx, "eval", record_eval)

    output = model(mx.array([[1, 2, 3]], dtype=mx.int32), training=False)

    assert output.logits.shape == (1, 3, 64)
    assert calls == []


def test_model_outputs_are_host_wrappers_but_compiled_boundaries_are_array_trees():
    assert [field.name for field in fields(ForwardOutput)] == [
        "logits",
        "cache",
        "next_key",
    ]

    module_paths = {
        Path(inspect.getfile(KVCache)),
        Path(inspect.getfile(SMLLanguageModel)),
    }
    forbidden_types = {"KVCache", "KVView", "ForwardOutput"}
    for module_path in module_paths:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_compiled = any(
                "compile" in ast.unparse(decorator) for decorator in node.decorator_list
            )
            if not is_compiled:
                continue
            signature = " ".join(
                ast.unparse(annotation)
                for annotation in [
                    *(argument.annotation for argument in node.args.args),
                    node.returns,
                ]
                if annotation is not None
            )
            assert forbidden_types.isdisjoint(signature.split())


def test_forward_kernels_contain_no_host_validation_or_parameter_installation():
    source = inspect.getsource(SMLLanguageModel.forward_arrays)

    assert ".item(" not in source
    assert ".tolist(" not in source
    assert "mx.eval(" not in source
    assert ".update(" not in source
