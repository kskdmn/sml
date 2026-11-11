from __future__ import annotations

import math

import mlx.core as mx
from sml.training.common import (
    OptimizerConfig,
    resolved_warmup_steps,
)


def optimizer_config(**overrides: object) -> OptimizerConfig:
    return OptimizerConfig(schedule_steps=100, warmup_steps=0, **overrides)


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def numpy_schedule_oracle(steps: list[int], config: OptimizerConfig) -> mx.array:
    warmup_steps = resolved_warmup_steps(config)
    values = []
    for step in steps:
        warmup = (step + 1) / max(1, warmup_steps)
        if step < warmup_steps:
            values.append(config.learning_rate * warmup)
        elif config.schedule_steps is None:
            values.append(config.learning_rate)
        else:
            decay_steps = config.schedule_steps - warmup_steps
            progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = (
                config.minimum_learning_rate_ratio
                + (1.0 - config.minimum_learning_rate_ratio) * cosine
            )
            values.append(config.learning_rate * ratio)
    return mx.array(values, dtype=mx.float32)
