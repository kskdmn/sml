"""Descriptor-bound MLX array payload loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import BinaryIO

import mlx.core as mx

from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    OpenedArtifact,
    _json_object_no_duplicates,
    _reject_json_constant,
)
from sml.errors import SMLArtifactError

_SAFETENSORS_DTYPES = {
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "BOOL": ("bool", 1),
}
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000


@dataclass(frozen=True, slots=True)
class TensorSlice:
    spec: ArraySpec
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SafetensorsLayout:
    tensors: Mapping[str, TensorSlice]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


def _plain_offset(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SMLArtifactError(f"{context} must be a non-negative integer")
    return value


def read_safetensors_layout(
    stream: BinaryIO,
    reference: ArrayPayloadRef,
) -> SafetensorsLayout:
    """Read and exactly validate safetensors metadata without tensor materialization."""
    logical_path = reference.payload.logical_path
    try:
        stream.seek(0)
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise SMLArtifactError(
                f"safetensors payload has a truncated header length: {logical_path}"
            )
        header_length = int.from_bytes(encoded_length, byteorder="little")
        if header_length > _MAX_SAFETENSORS_HEADER_BYTES:
            raise SMLArtifactError(
                f"safetensors header exceeds parser limit: {logical_path}"
            )
        if header_length > reference.payload.byte_size - 8:
            raise SMLArtifactError(
                f"safetensors header exceeds payload bytes: {logical_path}"
            )
        encoded_header = stream.read(header_length)
        if len(encoded_header) != header_length:
            raise SMLArtifactError(
                f"safetensors payload has a truncated header: {logical_path}"
            )
        raw = json.loads(
            encoded_header.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_no_duplicates,
        )
        if not isinstance(raw, dict):
            raise SMLArtifactError(
                f"safetensors header must be an object: {logical_path}"
            )
        metadata = raw.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in metadata.items()
            )
        ):
            raise SMLArtifactError(
                f"safetensors metadata must be string-keyed strings: {logical_path}"
            )

        expected = {spec.name: spec for spec in reference.arrays}
        if set(raw) != set(expected):
            raise SMLArtifactError(f"safetensors array keys mismatch: {logical_path}")
        data_start = 8 + header_length
        slices: dict[str, TensorSlice] = {}
        relative_ranges: list[tuple[int, int, str]] = []
        for name, spec in expected.items():
            entry = raw[name]
            if not isinstance(entry, dict) or set(entry) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise SMLArtifactError(
                    f"safetensors tensor metadata is invalid: {logical_path}:{name}"
                )
            try:
                dtype_name, item_size = _SAFETENSORS_DTYPES[entry["dtype"]]
            except (KeyError, TypeError) as error:
                raise SMLArtifactError(
                    f"safetensors tensor dtype is unsupported: {logical_path}:{name}"
                ) from error
            shape = entry["shape"]
            if not isinstance(shape, list):
                raise SMLArtifactError(
                    f"safetensors tensor shape is invalid: {logical_path}:{name}"
                )
            parsed_shape = tuple(
                _plain_offset(dimension, context="safetensors shape dimension")
                for dimension in shape
            )
            offsets = entry["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise SMLArtifactError(
                    f"safetensors tensor offsets are invalid: {logical_path}:{name}"
                )
            start = _plain_offset(offsets[0], context="safetensors data offset")
            end = _plain_offset(offsets[1], context="safetensors data offset")
            element_count = 1
            for dimension in parsed_shape:
                element_count *= dimension
            if (
                end < start
                or end - start != element_count * item_size
                or data_start + end > reference.payload.byte_size
            ):
                raise SMLArtifactError(
                    f"safetensors tensor byte range is invalid: {logical_path}:{name}"
                )
            if parsed_shape != spec.shape or dtype_name != spec.dtype:
                raise SMLArtifactError(
                    f"safetensors array metadata mismatch: {logical_path}:{name}"
                )
            slices[name] = TensorSlice(spec, data_start + start, data_start + end)
            relative_ranges.append((start, end, name))

        cursor = 0
        for start, end, _name in sorted(relative_ranges):
            if start != cursor:
                raise SMLArtifactError(
                    f"safetensors tensor ranges are not contiguous: {logical_path}"
                )
            cursor = end
        if data_start + cursor != reference.payload.byte_size:
            raise SMLArtifactError(
                f"safetensors tensor ranges do not cover payload: {logical_path}"
            )
        return SafetensorsLayout(slices)
    except SMLArtifactError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise SMLArtifactError(
            f"invalid safetensors payload: {logical_path}"
        ) from error


def verify_safetensors_metadata(
    artifact: OpenedArtifact,
    reference: ArrayPayloadRef,
) -> SafetensorsLayout:
    """Prove one payload and validate only its exact safetensors metadata."""
    if not isinstance(artifact, OpenedArtifact):
        raise TypeError("artifact must be an OpenedArtifact")
    if not isinstance(reference, ArrayPayloadRef):
        raise TypeError("reference must be an ArrayPayloadRef")
    with artifact.open_payload(reference.payload) as payload:
        return read_safetensors_layout(payload.stream, reference)


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
