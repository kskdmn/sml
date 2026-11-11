from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest
from sml.artifacts import semantics as semantics_module
from sml.errors import SMLArtifactError
from sml.model.config import ModelConfig
from sml.training.lora import LoRAConfig
from sml.training.random import counter_random_key


@pytest.mark.parametrize("field", ("maximum_steps", "maximum_epochs", "log_interval"))
def test_full_checkpoint_configuration_rejects_values_above_signed_int32(field):
    """Saved runtime counters must obey the same signed-int32 bound as training."""
    values = {
        "interval": 1,
        "maximum_steps": 1,
        "maximum_epochs": 1,
        "log_interval": 1,
        "seed": 0,
        "compile": False,
        "rng_schedule": "counter-addressed-forward-terminal-v1",
    }
    values[field] = 2**31

    with pytest.raises(SMLArtifactError, match="checkpoint config"):
        semantics_module.checkpoint_configuration(values, context="run")


@pytest.mark.parametrize(
    ("hidden_dropout", "lora_dropout", "targets", "expected_splits"),
    (
        (0.2, 0.0, (), 2),
        (0.0, 0.3, ("q_proj", "v_proj"), 4),
        (0.2, 0.3, ("q_proj", "v_proj"), 6),
        (0.0, 0.0, ("q_proj", "v_proj"), 0),
    ),
)
def test_expected_training_key_uses_only_current_microstep_sites(
    monkeypatch,
    hidden_dropout,
    lora_dropout,
    targets,
    expected_splits,
):
    """FULL verification must reconstruct only the terminal forward key."""
    split_calls = 0
    real_split = semantics_module.mx.random.split

    def counted_split(key):
        nonlocal split_calls
        split_calls += 1
        return real_split(key)

    monkeypatch.setattr(semantics_module.mx.random, "split", counted_split)
    model = replace(
        ModelConfig(num_layers=2),
        hidden_dropout=hidden_dropout,
    )
    lora = None
    if targets:
        lora = LoRAConfig(dropout=lora_dropout, target_modules=targets)

    actual = semantics_module.expected_next_key(
        seed=17,
        microsteps=9_000_000_000_000,
        model=model,
        lora=lora,
    )
    mx.eval(actual)
    assert split_calls == expected_splits


@pytest.mark.parametrize(
    ("model", "lora"),
    (
        (ModelConfig(hidden_dropout=0.2), None),
        (
            ModelConfig(hidden_dropout=0.0),
            LoRAConfig(dropout=0.3, target_modules=("q_proj", "v_proj")),
        ),
    ),
)
def test_expected_training_key_at_zero_microsteps_is_initial_counter_key(model, lora):
    actual = semantics_module.expected_next_key(
        seed=17,
        microsteps=0,
        model=model,
        lora=lora,
    )

    assert bool(mx.array_equal(actual, counter_random_key(17, 0)))


def test_expected_training_key_with_disabled_dropout_is_initial_counter_key():
    actual = semantics_module.expected_next_key(
        seed=17,
        microsteps=9_000_000_000_000,
        model=ModelConfig(hidden_dropout=0.0),
        lora=LoRAConfig(
            dropout=0.0,
            target_modules=("q_proj", "v_proj"),
        ),
    )

    assert bool(mx.array_equal(actual, counter_random_key(17, 0)))
