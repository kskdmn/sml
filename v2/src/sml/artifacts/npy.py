"""Read-only NPY mappings that own their verified payload descriptors."""

from __future__ import annotations

import mmap
from math import prod
from traceback import clear_frames
from typing import Self

import numpy as np

from sml.artifacts.manifest import VerifiedPayload
from sml.errors import SMLArtifactError


class VerifiedNpyMapping:
    """One validated array, mapping, and payload with ordered resource release."""

    __slots__ = ("_closed", "array", "logical_path", "mapping", "payload")

    def __init__(
        self,
        logical_path: str,
        payload: VerifiedPayload,
        mapping: mmap.mmap,
        array: np.ndarray,
    ) -> None:
        self.logical_path = logical_path
        self.payload = payload
        self.mapping = mapping
        self.array = array
        self._closed = False

    @classmethod
    def from_payload(
        cls,
        payload: VerifiedPayload,
        *,
        logical_path: str,
        expected_shape: tuple[int, ...],
        expected_dtype: np.dtype,
        description: str,
    ) -> Self:
        """Take payload ownership, including when parsing or acquisition fails."""
        mapping: mmap.mmap | None = None
        array: np.ndarray | None = None
        try:
            stream = payload.stream
            try:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
                else:
                    raise ValueError(f"unsupported NPY version: {version}")
                data_offset = stream.tell()
            except (EOFError, OSError, TypeError, ValueError) as error:
                raise SMLArtifactError(
                    f"invalid {description} NPY header: {logical_path}: {error}"
                ) from error

            parsed_dtype = np.dtype(dtype)
            if fortran_order:
                raise SMLArtifactError(
                    f"{description} must use C order: {logical_path}"
                )
            if parsed_dtype.hasobject or parsed_dtype != expected_dtype:
                raise SMLArtifactError(
                    f"{description} dtype mismatch: {logical_path}; "
                    f"expected {expected_dtype.str}, got {parsed_dtype.str}"
                )
            if tuple(shape) != expected_shape:
                raise SMLArtifactError(
                    f"{description} shape mismatch: {logical_path}; "
                    f"expected {expected_shape}, got {shape}"
                )
            expected_size = data_offset + prod(expected_shape) * parsed_dtype.itemsize
            if expected_size != payload.opened_stat.st_size:
                raise SMLArtifactError(
                    f"{description} payload size mismatch: {logical_path}"
                )

            try:
                mapping = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
                array = np.ndarray(
                    expected_shape,
                    dtype=parsed_dtype,
                    buffer=mapping,
                    offset=data_offset,
                    order="C",
                )
                array.setflags(write=False)
            except (BufferError, OSError, TypeError, ValueError) as error:
                if error.__traceback__ is not None:
                    clear_frames(error.__traceback__)
                raise SMLArtifactError(
                    f"could not memory-map {description}: {logical_path}"
                ) from error
            return cls(logical_path, payload, mapping, array)
        except BaseException as error:
            if error.__traceback__ is not None:
                clear_frames(error.__traceback__)
            array = None
            cleanup_errors: list[BaseException] = []
            try:
                if mapping is not None:
                    mapping.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
                cleanup_errors.append(cleanup_error)
            try:
                payload.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - cleanup continues
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise error from cleanup_errors[0]
            raise

    def release_view(self) -> None:
        """Drop the owner's array before closing its backing mapping."""
        self.array = np.empty((0,), dtype=self.array.dtype)
        self.array.setflags(write=False)

    def close_mapping(self) -> None:
        self.mapping.close()

    def close_payload(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.payload.close()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for release in (self.release_view, self.close_mapping, self.close_payload):
            try:
                release()
            except BaseException as error:  # noqa: BLE001 - cleanup must continue
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self) -> Self:
        if self._closed:
            raise SMLArtifactError("NPY mapping is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if isinstance(exception, BaseException):
                raise exception from close_error
            raise
