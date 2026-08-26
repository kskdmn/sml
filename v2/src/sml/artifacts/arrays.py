"""Descriptor-bound MLX array payload loading."""

from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx

from sml.artifacts.manifest import ArrayPayloadRef, OpenedArtifact
from sml.errors import SMLArtifactError


def load_safetensors_payload(
    artifact: OpenedArtifact,
    reference: ArrayPayloadRef,
) -> dict[str, mx.array]:
    """Load, exactly validate, and eagerly materialize one safetensors payload."""
    if not isinstance(artifact, OpenedArtifact):
        raise TypeError("artifact must be an OpenedArtifact")
    if not isinstance(reference, ArrayPayloadRef):
        raise TypeError("reference must be an ArrayPayloadRef")

    logical_path = reference.payload.logical_path
    try:
        with artifact.open_payload(reference.payload) as payload:
            arrays = mx.load(payload.stream, format="safetensors")
            if not isinstance(arrays, Mapping) or not all(
                isinstance(name, str) for name in arrays
            ):
                raise SMLArtifactError(
                    f"safetensors payload must be a string-keyed mapping: {logical_path}"
                )

            expected = {spec.name: spec for spec in reference.arrays}
            if set(arrays) != set(expected):
                raise SMLArtifactError(
                    f"safetensors array keys mismatch: {logical_path}"
                )

            names = sorted(expected)
            for name in names:
                array = arrays[name]
                spec = expected[name]
                dtype_name = str(array.dtype).removeprefix("mlx.core.")
                if tuple(array.shape) != spec.shape or dtype_name != spec.dtype:
                    raise SMLArtifactError(
                        f"safetensors array metadata mismatch: {logical_path}:{name}"
                    )

            mx.eval(*(arrays[name] for name in names))
            return {name: arrays[name] for name in names}
    except SMLArtifactError:
        raise
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise SMLArtifactError(
            f"invalid safetensors payload: {logical_path}"
        ) from error


__all__ = ["load_safetensors_payload"]
