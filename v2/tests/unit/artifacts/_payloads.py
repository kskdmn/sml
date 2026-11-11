from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    PayloadRef,
    file_identity,
)

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _array_ref(
    path: Path,
    logical_path: str,
    arrays: dict[str, mx.array],
    *,
    declared_shape: tuple[int, ...] | None = None,
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        tuple(
            ArraySpec(
                name,
                declared_shape if declared_shape is not None else tuple(array.shape),
                {
                    mx.bfloat16: "bfloat16",
                    mx.float32: "float32",
                    mx.int32: "int32",
                    mx.uint32: "uint32",
                }[array.dtype],
            )
            for name, array in sorted(arrays.items())
        ),
    )
