from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sml.data.swag import prepare_swag_bundle
from sml.errors import SMLConfigurationError
from sml.model.config import ModelConfig
from sml.training import swag as swag_module
from sml.training.common import LoaderConfig, resolved_warmup_steps
from sml.training.swag import (
    SwagTrainingConfig,
    default_swag_optimizer_config,
)

from .swag_helpers import (
    VALID_ROW,
    FakeSwagProvider,
    tiny_base_model,
    tiny_swag_config,
)


def test_finetuning_defaults_to_eightfold_context():
    config = SwagTrainingConfig(
        base_checkpoint=Path("base"), data=Path("data"), output_run=Path("run")
    )
    assert config.rope_scaling_factor == 8.0
    assert config.maximum_steps is None
    assert config.maximum_epochs == 3
    assert (
        replace(
            ModelConfig(), rope_scaling_factor=config.rope_scaling_factor
        ).effective_context_length
        == 8_192
    )


@pytest.mark.parametrize("factor", [True, False, 0.5, float("inf"), float("nan"), "8"])
def test_finetuning_rejects_invalid_rope_scaling(factor):
    with pytest.raises(SMLConfigurationError, match="rope_scaling_factor"):
        SwagTrainingConfig(
            base_checkpoint=Path("base"),
            data=Path("data"),
            output_run=Path("run"),
            rope_scaling_factor=factor,
        )


def test_default_swag_optimizer_preserves_legacy_learning_rate_and_zero_adapter_decay():
    config = default_swag_optimizer_config()
    assert config.learning_rate == 1e-4
    assert config.schedule_steps is None
    assert config.warmup_steps is None
    assert config.weight_decay.lora_a == 0.0
    assert config.weight_decay.lora_b == 0.0


@pytest.fixture
def schedule_bundle(tmp_path):
    rows = tuple(
        {**VALID_ROW, "context": "word " * length} for length in (3, 3, 3, 20, 20, 20)
    )
    with prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(rows)),
        tiny_base_model(),
        tmp_path / "swag",
    ) as bundle:
        with bundle.borrow_buckets() as buckets:
            assert [bucket.input_ids.shape[0] for bucket in buckets] == [3, 3]
        yield bundle


@pytest.mark.parametrize(
    (
        "microbatch_size",
        "accumulation_steps",
        "maximum_steps",
        "maximum_epochs",
        "expected_steps",
    ),
    (
        (1, 8, None, 3, 3),
        (2, 3, None, 3, 6),
        (2, 3, 4, 3, 4),
        (2, 3, 10, 3, 6),
        (2, 3, 10, None, 10),
    ),
)
def test_automatic_schedule_counts_bucket_padding_and_epoch_tails(
    tmp_path,
    schedule_bundle,
    microbatch_size,
    accumulation_steps,
    maximum_steps,
    maximum_epochs,
    expected_steps,
):
    config = SwagTrainingConfig(
        base_checkpoint=tmp_path / "base",
        data=schedule_bundle.path,
        output_run=tmp_path / "run",
        loader=LoaderConfig(
            microbatch_size=microbatch_size,
            gradient_accumulation_steps=accumulation_steps,
        ),
        maximum_steps=maximum_steps,
        maximum_epochs=maximum_epochs,
    )

    resolved = swag_module._resolved_fresh_config(config, schedule_bundle)

    assert config.optimizer.schedule_steps is None
    assert resolved.optimizer.schedule_steps == expected_steps
    assert resolved_warmup_steps(resolved.optimizer) == int(0.01 * expected_steps)


def test_explicit_schedule_and_warmup_remain_authoritative(tmp_path, schedule_bundle):
    config = SwagTrainingConfig(
        base_checkpoint=tmp_path / "base",
        data=schedule_bundle.path,
        output_run=tmp_path / "run",
        optimizer=replace(
            default_swag_optimizer_config(), schedule_steps=100, warmup_steps=10
        ),
        maximum_steps=1,
    )

    assert swag_module._resolved_fresh_config(config, schedule_bundle) is config


def test_automatic_schedule_rejects_warmup_longer_than_run(tmp_path, schedule_bundle):
    config = SwagTrainingConfig(
        base_checkpoint=tmp_path / "base",
        data=schedule_bundle.path,
        output_run=tmp_path / "run",
        optimizer=replace(default_swag_optimizer_config(), warmup_steps=3),
    )

    with pytest.raises(SMLConfigurationError, match="warmup_steps.*schedule_steps"):
        swag_module._resolved_fresh_config(config, schedule_bundle)
