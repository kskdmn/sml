from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import zstandard as zstd
from sml import inference
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    prepare_pretraining_bundle,
)
from sml.data.tokenizer import TokenizerTrainingConfig, train_tokenizer_bundle
from sml.inference import GenerationRequest, InferenceSession
from sml.model.config import GenerationConfig, ModelConfig
from sml.model.layers import RMSNorm
from sml.training.common import (
    CheckpointPolicy,
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
)
from sml.training.pretrain import train


def _write_tiny_corpus(path: Path) -> Path:
    path.mkdir()
    lines = [
        json.dumps(
            {
                "text": (
                    "alpha beta gamma delta epsilon zeta eta theta " * 8
                    + f"row {index}"
                )
            }
        ).encode("utf-8")
        for index in range(24)
    ]
    payload = b"\n".join(lines) + b"\n"
    (path / "tiny-0000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


def _corpus_config(path: Path) -> CorpusConfig:
    return CorpusConfig(
        input_root=path,
        shuffle_files=False,
        min_text_bytes=1,
        max_rows_per_file=None,
    )


def _prepare_tiny_data(root: Path):
    corpus = _write_tiny_corpus(root / "corpus")
    tokenizer = train_tokenizer_bundle(
        TokenizerTrainingConfig(
            corpus=_corpus_config(corpus),
            vocab_size=300,
            hard_vocab_limit=False,
            num_threads=1,
        ),
        root / "tokenizer",
    )
    data = prepare_pretraining_bundle(
        PretrainingPreparationConfig(
            corpus=_corpus_config(corpus),
            tokenizer_bundle=tokenizer.path,
            sequence_length=32,
            shuffle_window_rows=5,
            output_shard_rows=3,
            seed=17,
        ),
        root / "data",
    )
    return tokenizer, data


def _tiny_pretraining_config(data_path: Path, output_run: Path, vocab_size: int):
    return PretrainingConfig(
        data=data_path,
        output_run=output_run,
        model=ModelConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_context_length=32,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            beta1=0.5,
            beta2=0.5,
            schedule_steps=4,
            warmup_steps=0,
        ),
        loader=LoaderConfig(
            microbatch_size=1,
            gradient_accumulation_steps=1,
            prefetch_depth=2,
            epoch_seed=13,
        ),
        checkpoint=CheckpointPolicy(interval=1),
        maximum_steps=2,
        maximum_epochs=2,
        log_interval=1,
        seed=19,
        compile=False,
    )


@pytest.fixture(scope="module")
def _tiny_run_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("inference-equivalence")
    tokenizer, data = _prepare_tiny_data(root)
    result = train(
        _tiny_pretraining_config(
            data.path,
            root / "run",
            tokenizer.manifest.vocab_size,
        )
    )
    return result.run


@pytest.fixture
def tiny_pretraining_run(tmp_path: Path, _tiny_run_template: Path) -> Path:
    dest = tmp_path / "run"
    shutil.copytree(_tiny_run_template, dest)
    return dest


POLICY_PROMPT = "alpha beta alpha beta"
# Last prompt bigram continues with 275; 264 is seen but not that successor.
GREEDY_WINNER = 275
REPETITION_WINNER = 40
NGRAM_WINNER = 264
# Seeded top-p first token after the designed logits (probed, then hardcoded).
SAMPLED_WINNER = 219
EOS_PROMPTS = ("zeta", "eta theta")


@pytest.fixture
def tiny_session(tiny_pretraining_run: Path) -> InferenceSession:
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    _install_distinct_policy_logits(session, POLICY_PROMPT)
    _install_early_eos_logits(session, POLICY_PROMPT)
    return session


def _last_hidden(session: InferenceSession, token_ids: list[int]) -> mx.array:
    params = session._parameters
    tokens = mx.array([token_ids], dtype=mx.int32)
    mask = mx.ones(tokens.shape, dtype=mx.bool_)
    relative = mx.cumsum(mask.astype(mx.int32), axis=1) - 1
    positions = mx.where(mask, relative, mx.zeros_like(relative)).astype(mx.int32)
    hidden = params["embed_tokens"]["weight"][tokens]
    for layer_index, layer in enumerate(session._model.layers):
        hidden, _, _ = layer.forward_arrays(
            params["layers"][layer_index],
            hidden,
            attention_mask=mask,
            positions=positions,
            cache_state=None,
            training=False,
            key=None,
        )
    hidden = RMSNorm.forward_arrays(
        params["norm"]["weight"],
        hidden,
        session._model.norm.epsilon,
    )
    return hidden[0, -1].astype(mx.float32)


def _set_token_logit(
    weight: mx.array, hidden: mx.array, token: int, target: float
) -> None:
    row = weight[token].astype(mx.float32)
    current = mx.sum(hidden * row)
    denom = mx.maximum(mx.sum(mx.square(hidden)), 1e-6)
    weight[token] = (row + ((target - current) / denom) * hidden).astype(weight.dtype)


def _set_token_logit_for_hiddens(
    weight: mx.array,
    hiddens: list[mx.array],
    token: int,
    targets: list[float],
) -> None:
    stacked = np.stack(
        [np.array(hidden.astype(mx.float32)) for hidden in hiddens]
    ).astype(np.float32)
    row = np.array(weight[token].astype(mx.float32), dtype=np.float32)
    residual = np.asarray(targets, dtype=np.float32) - stacked @ row
    gram = stacked @ stacked.T + (1e-6 * np.eye(len(targets), dtype=np.float32))
    delta = np.linalg.solve(gram, residual) @ stacked
    weight[token] = mx.array(row + delta).astype(weight.dtype)


def _install_distinct_policy_logits(session: InferenceSession, prompt: str) -> None:
    """Give greedy, repetition, n-gram, and sampling distinct safe-margin winners."""
    hidden = _last_hidden(session, list(session._encode_prompt(prompt)))
    mx.eval(hidden)
    weight = session._parameters["embed_tokens"]["weight"]
    # First-token split on POLICY_PROMPT:
    # greedy keeps 275; n-gram size 2 bans that successor so 264 wins;
    # repetition 1.2 drops seen 275/264 below unseen 40 (2.6).
    for token, target in (
        (GREEDY_WINNER, 3.0),
        (NGRAM_WINNER, 2.8),
        (REPETITION_WINNER, 2.6),
    ):
        _set_token_logit(weight, hidden, token, target)
    mx.eval(weight)


def _install_early_eos_logits(session: InferenceSession, policy_prompt: str) -> None:
    """Force EOS-only prompts to emit eos_token_id without stealing other winners."""
    eos_id = session.resolved_model.model_config.eos_token_id
    weight = session._parameters["embed_tokens"]["weight"]
    prompts = (*EOS_PROMPTS, policy_prompt)
    hiddens = [
        _last_hidden(session, list(session._encode_prompt(prompt)))
        for prompt in prompts
    ]
    mx.eval(*hiddens)
    targets = [8.0] * len(EOS_PROMPTS) + [-2.0]
    _set_token_logit_for_hiddens(weight, hiddens, eos_id, targets)
    mx.eval(weight)


def assert_results_margin_aware_equal(serial, batched) -> None:
    assert len(serial) == len(batched)
    for index, (left, right) in enumerate(zip(serial, batched, strict=True)):
        assert left.seed == right.seed, index
        assert left.model == right.model, index
        if left.token_ids == right.token_ids:
            assert left.text == right.text, index
            continue
        raise AssertionError(
            f"result {index} tokens differ outside a captured boundary: "
            f"{left.token_ids!r} != {right.token_ids!r}"
        )


def _sampled(seed: int, max_new_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(
        max_new_tokens=max_new_tokens,
        config=GenerationConfig(temperature=0.8, top_p=0.9, seed=seed),
    )


@pytest.fixture
def requests_for_mode(mode: str) -> list[tuple[str, GenerationRequest]]:
    if mode == "greedy":
        return [
            (
                "alpha",
                GenerationRequest(max_new_tokens=2, config=GenerationConfig(seed=1)),
            ),
            (
                "alpha beta",
                GenerationRequest(max_new_tokens=3, config=GenerationConfig(seed=2)),
            ),
            (
                "short",
                GenerationRequest(max_new_tokens=1, config=GenerationConfig(seed=3)),
            ),
        ]
    if mode == "sampled":
        return [
            ("alpha", _sampled(11, 2)),
            ("alpha beta", _sampled(13, 3)),
            ("short", _sampled(17, 1)),
        ]
    if mode == "repetition":
        return [
            (
                "alpha alpha alpha",
                GenerationRequest(
                    max_new_tokens=3,
                    config=GenerationConfig(repetition_penalty=1.2, seed=3),
                ),
            ),
            (
                "beta gamma",
                GenerationRequest(
                    max_new_tokens=2,
                    config=GenerationConfig(repetition_penalty=1.5, seed=5),
                ),
            ),
        ]
    if mode == "no-repeat-ngram":
        return [
            (
                "alpha beta alpha",
                GenerationRequest(
                    max_new_tokens=4,
                    config=GenerationConfig(no_repeat_ngram_size=2, seed=7),
                ),
            ),
            (
                "gamma",
                GenerationRequest(
                    max_new_tokens=3,
                    config=GenerationConfig(no_repeat_ngram_size=2, seed=8),
                ),
            ),
        ]
    if mode == "eos":
        return [
            (
                "zeta",
                GenerationRequest(max_new_tokens=4, config=GenerationConfig(seed=9)),
            ),
            (
                "eta theta",
                GenerationRequest(max_new_tokens=6, config=GenerationConfig(seed=10)),
            ),
        ]
    raise AssertionError(f"unknown mode {mode}")


def long_seeded_request(seed: int) -> tuple[str, GenerationRequest]:
    return ("alpha beta gamma delta", _sampled(seed, 4))


def medium_seeded_request(seed: int) -> tuple[str, GenerationRequest]:
    return ("alpha beta gamma", _sampled(seed, 4))


def unseeded_request(text: str) -> tuple[str, GenerationRequest]:
    return (
        text,
        GenerationRequest(
            max_new_tokens=4,
            config=GenerationConfig(temperature=0.8, top_p=0.9),
        ),
    )


def replace_request_seed(request: GenerationRequest, seed: int) -> GenerationRequest:
    return GenerationRequest(
        max_new_tokens=request.max_new_tokens,
        config=replace(request.config, seed=seed),
        include_prompt=request.include_prompt,
    )


@pytest.mark.parametrize(
    "mode", ["greedy", "sampled", "repetition", "no-repeat-ngram", "eos"]
)
def test_heterogeneous_batch_matches_serial_with_margin_rule(
    tiny_session, requests_for_mode, mode
):
    serial = tuple(
        tiny_session.generate(text, request) for text, request in requests_for_mode
    )
    batched = tiny_session.generate_batch(requests_for_mode)
    assert_results_margin_aware_equal(serial, batched)
    if mode == "eos":
        eos_id = tiny_session.resolved_model.model_config.eos_token_id
        for result, (_text, request) in zip(serial, requests_for_mode, strict=True):
            assert eos_id in result.token_ids
            assert result.token_ids[-1] == eos_id
            assert len(result.token_ids) < request.max_new_tokens


def test_seed_stream_is_invariant_to_bucket_neighbors(tiny_session):
    target = (
        "short",
        GenerationRequest(
            max_new_tokens=4,
            config=GenerationConfig(temperature=0.8, top_p=0.9, seed=17),
        ),
    )
    alone = tiny_session.generate_batch([target])[0]
    mixed = tiny_session.generate_batch(
        [long_seeded_request(99), target, medium_seeded_request(33)]
    )[1]
    assert mixed.seed == alone.seed == 17
    assert mixed.token_ids == alone.token_ids


def test_omitted_seed_is_allocated_before_reordering(
    tiny_session, monkeypatch: pytest.MonkeyPatch
):
    allocated: list[int] = []
    planned = (101, 102, 103)
    remaining = iter(planned)

    def allocator() -> int:
        seed = next(remaining)
        allocated.append(seed)
        return seed

    monkeypatch.setattr(inference, "allocate_generation_seed", allocator)
    items = [
        unseeded_request("a"),
        (
            "alpha beta gamma delta",
            GenerationRequest(
                max_new_tokens=4,
                config=GenerationConfig(temperature=0.8, top_p=0.9),
            ),
        ),
        unseeded_request("b"),
    ]
    result = tiny_session.generate_batch(items)
    assert allocated == list(planned)
    assert [item.seed for item in result] == list(planned)
    replay = tiny_session.generate(
        "a", replace_request_seed(items[0][1], result[0].seed)
    )
    assert replay.token_ids == result[0].token_ids


def test_one_batch_preserves_distinct_generation_policies(tiny_session):
    prompt = POLICY_PROMPT
    items = [
        (prompt, GenerationRequest(4, GenerationConfig(temperature=0.0, seed=1))),
        (
            prompt,
            GenerationRequest(4, GenerationConfig(temperature=0.8, top_p=0.9, seed=2)),
        ),
        (
            prompt,
            GenerationRequest(
                4, GenerationConfig(temperature=0.0, repetition_penalty=1.2, seed=3)
            ),
        ),
        (
            prompt,
            GenerationRequest(
                4, GenerationConfig(temperature=0.0, no_repeat_ngram_size=2, seed=4)
            ),
        ),
    ]
    serial = tuple(tiny_session.generate(text, request) for text, request in items)
    assert [result.token_ids[0] for result in serial] == [
        GREEDY_WINNER,
        SAMPLED_WINNER,
        REPETITION_WINNER,
        NGRAM_WINNER,
    ]
    assert len({result.token_ids for result in serial}) == len(items)
    batched = tiny_session.generate_batch(items)
    assert_results_margin_aware_equal(serial, batched)
