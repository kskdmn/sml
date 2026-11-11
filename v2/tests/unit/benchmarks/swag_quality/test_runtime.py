from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.lora import LoRAConfig, apply_lora, split_adapter_parameters

from v2.benchmarks import swag_quality
from v2.benchmarks.swag_quality import (
    MODEL_SEED,
)

from . import ROOT


@pytest.fixture
def tiny_quality_runtime():
    config = ModelConfig(
        vocab_size=300,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=64,
        hidden_dropout=0.5,
    )
    model = SMLLanguageModel(config, key=mx.random.key(3))
    apply_lora(model, LoRAConfig(rank=2, dropout=0.5), key=mx.random.key(4))
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    adapters = tree_map(lambda array: mx.ones_like(array) * 0.1, adapters)
    encoded = {
        name: value[:2]
        for name, value in swag_quality._load_encoded_arrays(
            ROOT / swag_quality.VALIDATION_FIXTURE
        ).items()
    }
    return config, model, adapters, frozen_base, encoded


def test_validation_disables_dropout_and_does_not_differentiate(
    monkeypatch, tiny_quality_runtime
):
    config, model, adapters, frozen_base, encoded = tiny_quality_runtime
    real_forward = SMLLanguageModel.forward_arrays
    calls = []

    def record_forward(self, *args, **kwargs):
        calls.append((kwargs["training"], kwargs["key"]))
        return real_forward(self, *args, **kwargs)

    def forbid_gradients(*args, **kwargs):
        raise AssertionError("quality validation must not compute gradients")

    monkeypatch.setattr(SMLLanguageModel, "forward_arrays", record_forward)
    monkeypatch.setattr(mx, "value_and_grad", forbid_gradients)
    mx.random.seed(11)
    first = swag_quality._evaluate_validation(
        model, adapters, frozen_base, encoded, model_config=config
    )
    mx.random.seed(29)
    second = swag_quality._evaluate_validation(
        model, adapters, frozen_base, encoded, model_config=config
    )

    assert first == second
    assert first[2] == 2
    assert np.isfinite(first[0])
    assert calls == [(False, None)] * 4


def test_swag_quality_training_uses_production_counter_keys(
    monkeypatch, tiny_quality_runtime, canonical_workload
):
    config, model, adapters, frozen_base, encoded = tiny_quality_runtime
    real_build = swag_quality.build_swag_kernels
    received_keys = []

    def recording_kernels(*args, **kwargs):
        kernels = real_build(*args, **kwargs)
        real_microstep = kernels.compiled_ranking_microstep_core

        def record_microstep(adapters, base, trainer_tree, *arrays):
            received_keys.append(trainer_tree[2])
            return real_microstep(adapters, base, trainer_tree, *arrays)

        return replace(kernels, compiled_ranking_microstep_core=record_microstep)

    monkeypatch.setattr(swag_quality, "build_swag_kernels", recording_kernels)
    swag_quality._run_runtime(
        runtime="candidate",
        compile=False,
        workload=replace(canonical_workload, ordered_batches=((0,), (1,))),
        model=model,
        frozen_base=frozen_base,
        adapters=adapters,
        trainer_key=mx.random.key(999),
        training=encoded,
        validation=encoded,
        model_config=config,
    )

    assert len(received_keys) == 2
    for index, key in enumerate(received_keys):
        assert bool(
            mx.array_equal(
                key,
                swag_quality.counter_random_key(canonical_workload.model_seed, index),
            )
        )


def test_verified_source_snapshot_fully_verifies_pretraining_checkpoint(
    monkeypatch, tmp_path
):
    published = tmp_path / "pretraining-run"
    published.mkdir()
    called: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        swag_quality,
        "_publish_source_pretraining_run",
        lambda *_args, **_kwargs: published,
        raising=False,
    )

    def fake_resolve(path, *, full_verify):
        called.append((path, full_verify))
        raise RuntimeError("verified")

    monkeypatch.setattr(
        swag_quality, "resolve_model_artifact", fake_resolve, raising=False
    )
    with pytest.raises(RuntimeError, match="verified"):
        swag_quality._verified_source_snapshot(ModelConfig(), LoRAConfig(), MODEL_SEED)
    assert called == [(published, True)]
