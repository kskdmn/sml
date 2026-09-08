"""Deterministic starting weights for comparisons between current revisions.

This protocol belongs to the measurement harness. It does not change the production initializer. New harness identities require fresh
baseline evidence when this protocol changes.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten


def initialize_parameters(model, workload) -> str:
    """Give compared revisions exactly the same initial values."""
    seed = int(workload.optimizer["seed"])
    initializers = workload.model["initializers"]
    leaves = dict(tree_flatten(model.parameters()))
    weights = []
    for name, previous in sorted(leaves.items()):
        logical = name
        base_name = logical.replace(".base.weight", ".weight")
        key_bytes = hashlib.sha256(
            seed.to_bytes(4, "little") + base_name.encode("utf-8")
        ).digest()[:8]
        key = mx.array(np.frombuffer(key_bytes, dtype="<u4"), dtype=mx.uint32)
        if logical.endswith(".lora_b"):
            value = mx.zeros(previous.shape, dtype=mx.bfloat16)
        elif logical.endswith(".lora_a"):
            value = mx.random.normal(shape=previous.shape, scale=0.01, key=key).astype(
                mx.bfloat16
            )
        elif base_name.endswith("norm.weight"):
            value = mx.ones(previous.shape, dtype=mx.bfloat16)
        else:
            module = base_name.split(".")[-2]
            scale = float(initializers.get(module, initializers["other"]))
            value = mx.random.normal(shape=previous.shape, scale=scale, key=key).astype(
                mx.bfloat16
            )
            if base_name == "embed_tokens.weight":
                value = mx.where(
                    mx.arange(value.shape[0])[:, None]
                    == int(workload.model["pad_token_id"]),
                    mx.zeros_like(value),
                    value,
                )
        value = value.astype(previous.dtype)
        weights.append((name, value))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return parameter_identity(model)


def parameter_identity(model) -> str:
    digest = hashlib.sha256(b"sml-benchmark-canonical-parameters-v2\0")
    leaves = {}
    for name, value in tree_flatten(model.parameters()):
        logical = name
        if logical in leaves:
            raise ValueError(f"duplicate canonical parameter: {logical}")
        leaves[logical] = value
    for name, value in sorted(leaves.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        array = np.asarray(value.astype(mx.float32), dtype="<f4")
        digest.update(len(array.shape).to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(dimension.to_bytes(8, "little"))
        digest.update(array.tobytes())
    return "sha256:" + digest.hexdigest()
