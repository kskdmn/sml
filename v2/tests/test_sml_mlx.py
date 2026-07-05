import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import mlx.core as mx
    import mlx.nn

    mx.eval(mx.array([0]))
except (ImportError, RuntimeError) as exc:  # pragma: no cover - depends on host Metal access
    pytestmark = pytest.mark.skip(reason=f"mlx is not available: {exc}")


def tiny_config():
    from sml import SMLConfig

    return SMLConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_max_position_embeddings=32,
        rope_scaling_factor=2.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_checkpointing=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )


def test_mlx_forward_returns_logits_and_causal_lm_loss():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    input_ids = mx.random.randint(0, config.vocab_size, shape=(2, 8))

    output = model(input_ids, labels=input_ids)
    mx.eval(output.logits, output.loss)

    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.loss is not None
    assert output.loss.shape == ()


def test_mlx_rotary_cache_covers_scaled_context_length():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    rope = model.layers[0].self_attn.rope

    assert rope.cos_cached.shape[0] == config.effective_max_position_embeddings
    assert rope.sin_cached.shape[0] == config.effective_max_position_embeddings


def test_mlx_forward_accepts_scaled_context_length():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    input_ids = mx.random.randint(
        0,
        config.vocab_size,
        shape=(1, config.effective_max_position_embeddings),
    )

    output = model(input_ids)
    mx.eval(output.logits)

    assert output.logits.shape == (
        1,
        config.effective_max_position_embeddings,
        config.vocab_size,
    )


def test_mlx_forward_rejects_tokens_beyond_scaled_context_length():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    input_ids = mx.random.randint(
        0,
        config.vocab_size,
        shape=(1, config.effective_max_position_embeddings + 1),
    )

    with pytest.raises(ValueError, match="effective_max_position_embeddings"):
        model(input_ids)


def test_mlx_embedding_and_lm_head_weights_are_tied():
    from sml_mlx import SMLLanguageModel

    model = SMLLanguageModel(tiny_config())

    assert model.lm_head.weight is model.embed_tokens.weight


def test_mlx_generate_appends_requested_tokens_and_uses_cache():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)

    generated = model.generate(prompt, max_new_tokens=4)
    mx.eval(generated)

    assert generated.shape == (1, 7)
    assert bool(mx.array_equal(prompt, generated[:, : prompt.shape[1]]).item())


def test_mlx_generate_rejects_requested_tokens_beyond_scaled_context_length():
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    prompt = mx.full(
        (1, config.effective_max_position_embeddings),
        config.bos_token_id,
        dtype=mx.int32,
    )

    with pytest.raises(ValueError, match="effective_max_position_embeddings"):
        model.generate(prompt, max_new_tokens=1)


def test_mlx_repetition_penalty_reduces_repeated_token_logit():
    from sml_mlx import apply_repetition_penalty

    logits = mx.array([[1.0, 2.0, 3.0]])
    input_ids = mx.array([[1, 1, 2]])
    adjusted = apply_repetition_penalty(logits, input_ids, penalty=2.0)
    mx.eval(adjusted)

    assert adjusted[0, 0].item() == pytest.approx(1.0)
    assert adjusted[0, 1].item() == pytest.approx(1.0)
    assert adjusted[0, 2].item() == pytest.approx(1.5)


def test_mlx_no_repeat_ngram_blocks_repeated_trigram():
    from sml_mlx import apply_no_repeat_ngram

    logits = mx.array([[0.0, 0.0, 0.0, 5.0]])
    generated = mx.array([[0, 1, 2, 0, 1]])
    adjusted = apply_no_repeat_ngram(logits, generated, ngram_size=3)
    mx.eval(adjusted)

    assert bool(mx.isinf(adjusted[0, 2]).item())
    assert adjusted[0, 3].item() == pytest.approx(5.0)


def test_mlx_generate_sampling_is_reproducible_with_seed():
    from sml import GenerationConfig
    from sml_mlx import SMLLanguageModel

    config = tiny_config()
    model = SMLLanguageModel(config)
    prompt = mx.array([[config.bos_token_id, 5, 6]], dtype=mx.int32)
    generation_config = GenerationConfig(temperature=0.8, top_p=0.9, seed=123)

    first = model.generate(
        prompt,
        max_new_tokens=4,
        generation_config=generation_config,
    )
    second = model.generate(
        prompt,
        max_new_tokens=4,
        generation_config=generation_config,
    )
    mx.eval(first, second)

    assert bool(mx.array_equal(first, second).item())


def test_mlx_create_model_uses_default_config():
    from sml import SMLConfig
    from sml_mlx import SMLLanguageModel, create_model

    model = create_model()

    assert isinstance(model, SMLLanguageModel)
    assert isinstance(model.config, SMLConfig)


def test_mlx_count_parameters_deduplicates_tied_embeddings():
    from sml import estimate_model_size
    from sml_mlx import SMLLanguageModel, count_parameters

    config = tiny_config()
    model = SMLLanguageModel(config)
    qkv_bias_params = config.num_layers * (
        config.num_q_heads * config.head_dim
        + 2 * config.num_kv_heads * config.head_dim
    )

    total, trainable = count_parameters(model)

    assert total == estimate_model_size(config) + qkv_bias_params
    assert trainable == total
