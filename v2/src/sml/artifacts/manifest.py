from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar, Self, cast

import numpy as np

from sml.errors import SMLArtifactError

_IDENTITY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PORTABLE_COMPONENT_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?\Z"
)
_FILE_READ_SIZE = 1024 * 1024
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_OPEN_FLAGS = _OPEN_FLAGS | os.O_DIRECTORY
_PAYLOAD_OPEN_FLAGS = _OPEN_FLAGS | os.O_NONBLOCK
_MNT_LOCAL = 0x00001000
TRAINING_RNG_SCHEDULE = "counter-addressed-forward-terminal-v1"


class _DarwinFsid(ctypes.Structure):
    _fields_ = [("values", ctypes.c_int32 * 2)]


class _DarwinStatfs(ctypes.Structure):
    _fields_ = [
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", _DarwinFsid),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


class _FilesystemCapability(Enum):
    LOCAL_APFS = "local-apfs"
    LOCAL_OTHER = "local-other"
    NON_LOCAL = "non-local"


def _descriptor_filesystem_capability(descriptor: int) -> _FilesystemCapability:
    if sys.platform != "darwin":
        return _FilesystemCapability.NON_LOCAL
    filesystem = _DarwinStatfs()
    libc = ctypes.CDLL(None, use_errno=True)
    fstatfs = libc.fstatfs
    fstatfs.argtypes = (ctypes.c_int, ctypes.POINTER(_DarwinStatfs))
    fstatfs.restype = ctypes.c_int
    if fstatfs(descriptor, ctypes.byref(filesystem)) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    filesystem_type = bytes(filesystem.f_fstypename).split(b"\0", 1)[0]
    if not filesystem.f_flags & _MNT_LOCAL:
        return _FilesystemCapability.NON_LOCAL
    if filesystem_type == b"apfs":
        return _FilesystemCapability.LOCAL_APFS
    return _FilesystemCapability.LOCAL_OTHER


def _descriptor_is_local_apfs(descriptor: int) -> bool:
    return (
        _descriptor_filesystem_capability(descriptor)
        is _FilesystemCapability.LOCAL_APFS
    )


def _validated_string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("canonical JSON strings must contain valid Unicode") from error
    return value


def _normalize(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, PurePath):
        return _validated_string(str(value))
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return 0 if value == 0.0 else value
    if isinstance(value, str):
        return _validated_string(value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized[_validated_string(key)] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode *value* using the sole project-specific ``sml-json-v1`` format."""
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_identity(file: BinaryIO) -> str:
    """Return the SHA-256 identity of bytes read from the file's current position."""
    digest = hashlib.sha256()
    while chunk := file.read(_FILE_READ_SIZE):
        if not isinstance(chunk, bytes):
            raise TypeError("file_identity requires a binary file")
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def structured_identity(domain_tag: str, value: object) -> str:
    if not isinstance(domain_tag, str):
        raise TypeError("domain_tag must be a string")
    digest = hashlib.sha256()
    digest.update(_validated_string(domain_tag).encode())
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return f"sha256:{digest.hexdigest()}"


def _require_plain_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return _validated_string(value)


def _require_optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must match sha256:[0-9a-f]{{64}}")
    return value


def _freeze_normalized(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_normalized(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


def _freeze_json_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a canonical JSON object")
    normalized = _normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a canonical JSON object")
    return cast(Mapping[str, object], _freeze_normalized(normalized))


def _require_training_rng_schedule(checkpoint: Mapping[str, object]) -> None:
    if checkpoint.get("rng_schedule") != TRAINING_RNG_SCHEDULE:
        raise ValueError(f"checkpoint rng_schedule must be {TRAINING_RNG_SCHEDULE!r}")


def _require_tuple(value: object, name: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    return value


def row_content_identity(
    rows: Iterable[np.ndarray], row_count: int, row_width: int
) -> str:
    """Hash ordered token rows independently of their shard representation."""
    _require_plain_int(row_count, "row count")
    _require_plain_int(row_width, "row width", minimum=1)
    if row_count >= 2**64 or row_width >= 2**64:
        raise ValueError("row count and row width must fit unsigned 64-bit integers")

    digest = hashlib.sha256()
    digest.update(b"sml-row-content-v1\0")
    digest.update(row_count.to_bytes(8, "little", signed=False))
    digest.update(row_width.to_bytes(8, "little", signed=False))

    actual_count = 0
    int32 = np.iinfo(np.int32)
    for row in rows:
        array = np.asarray(row)
        if array.ndim != 1 or array.shape != (row_width,):
            raise ValueError(
                f"row shape mismatch: expected ({row_width},), got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("row dtype must be an integer dtype")
        if array.size and (
            int(array.min()) < int32.min or int(array.max()) > int32.max
        ):
            raise ValueError("row values must fit int32")
        canonical = np.ascontiguousarray(array, dtype=np.dtype("<i4"))
        digest.update(canonical.tobytes(order="C"))
        actual_count += 1

    if actual_count != row_count:
        raise ValueError(
            f"row count mismatch: expected {row_count}, got {actual_count}"
        )
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class PayloadRef:
    logical_path: str
    identity: str
    byte_size: int

    def __post_init__(self) -> None:
        _require_string(self.logical_path, "logical_path")
        _require_identity(self.identity, "payload identity")
        _require_plain_int(self.byte_size, "byte_size")


def parse_logical_path(value: str) -> tuple[str, ...]:
    """Return a normalized portable artifact path without filesystem access."""
    if not isinstance(value, str):
        raise TypeError("logical path must be a string")
    if not value or value.startswith("/") or "\\" in value:
        raise SMLArtifactError(f"invalid logical path: {value!r}")
    raw_components = value.split("/")
    if any(component in {"", ".", ".."} for component in raw_components):
        raise SMLArtifactError(f"invalid logical path: {value!r}")

    components = tuple(
        unicodedata.normalize("NFKC", component) for component in raw_components
    )
    if any(
        "/" in component
        or "\\" in component
        or _PORTABLE_COMPONENT_PATTERN.fullmatch(component) is None
        for component in components
    ):
        raise SMLArtifactError(f"invalid logical path: {value!r}")
    return components


def _logical_path_collision_key(logical_path: str) -> tuple[str, ...]:
    return tuple(component.casefold() for component in parse_logical_path(logical_path))


def _stable_stat_fields(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_stable_entry(
    expected: os.stat_result,
    actual: os.stat_result,
) -> bool:
    return _stable_stat_fields(expected) == _stable_stat_fields(actual)


def _cleanup_exception(
    actions: Sequence[Callable[[], object]],
) -> BaseException | None:
    """Run every cleanup action and retain all failures for exception chaining."""
    failures: list[BaseException] = []
    for action in actions:
        try:
            action()
        except BaseException as error:  # noqa: BLE001 - cleanup is exhaustive
            failures.append(error)
    if not failures:
        return None
    if len(failures) == 1:
        return failures[0]
    return BaseExceptionGroup("multiple artifact cleanup failures", failures)


def _validated_payload_references(
    references: Sequence[PayloadRef],
) -> tuple[PayloadRef, ...]:
    parsed_references: list[PayloadRef] = []
    collision_paths: dict[tuple[str, ...], tuple[str, tuple[str, ...]]] = {}
    for reference in references:
        if not isinstance(reference, PayloadRef):
            raise TypeError("references must contain PayloadRef values")
        components = parse_logical_path(reference.logical_path)
        collision_key = tuple(component.casefold() for component in components)
        previous = collision_paths.get(collision_key)
        if previous is not None:
            if previous[0] == reference.logical_path:
                raise SMLArtifactError(
                    f"duplicate logical payload path: {reference.logical_path!r}"
                )
            category = (
                "normalized path collision"
                if previous[1] == components
                else "case-folded path collision"
            )
            raise SMLArtifactError(
                f"{category}: {previous[0]!r} and {reference.logical_path!r}"
            )
        collision_paths[collision_key] = (reference.logical_path, components)
        parsed_references.append(reference)
    return tuple(parsed_references)


def _require_unique_payload_paths(
    references: Sequence[PayloadRef | ArrayPayloadRef],
    name: str,
) -> None:
    seen: dict[tuple[str, ...], str] = {}
    for value in references:
        reference = value.payload if isinstance(value, ArrayPayloadRef) else value
        if not isinstance(reference, PayloadRef):
            raise TypeError(f"{name} must contain payload references")
        key = _logical_path_collision_key(reference.logical_path)
        previous = seen.get(key)
        if previous is not None:
            raise SMLArtifactError(
                f"duplicate or colliding payload path in {name}: "
                f"{previous!r} and {reference.logical_path!r}"
            )
        seen[key] = reference.logical_path


class ArtifactRoot:
    """An artifact directory owned and traversed through open descriptors."""

    def __init__(self, descriptor: int, *, local_apfs: bool) -> None:
        self._fd = descriptor
        self._local_apfs = local_apfs
        self._inode_paths: dict[tuple[int, int], tuple[str, ...]] = {}

    @classmethod
    def open(cls, path: Path, *, writable: bool) -> ArtifactRoot:
        if not isinstance(path, Path):
            raise TypeError("artifact root path must be a Path")
        if not isinstance(writable, bool):
            raise TypeError("writable must be a bool")
        descriptor = -1
        try:
            descriptor = os.open(path, _DIRECTORY_OPEN_FLAGS)
            root_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(root_stat.st_mode):
                raise SMLArtifactError("artifact root is not a directory")
            local_apfs = _descriptor_is_local_apfs(descriptor)
            if writable and not local_apfs:
                raise SMLArtifactError(
                    "writable artifact roots require a local APFS filesystem"
                )
            root = cls(descriptor, local_apfs=local_apfs)
            descriptor = -1
            return root
        except SMLArtifactError:
            raise
        except OSError as error:
            raise SMLArtifactError(
                f"could not open artifact root with no-follow semantics: {path}"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @property
    def local_apfs(self) -> bool:
        return self._local_apfs

    def duplicate(self) -> ArtifactRoot:
        """Return an independently owned descriptor for this same root inode."""
        return ArtifactRoot(os.dup(self._fd), local_apfs=self._local_apfs)

    def fileno(self) -> int:
        """Expose the owned directory descriptor for descriptor-relative consumers."""
        return self._fd

    def close(self) -> None:
        descriptor = self._fd
        self._fd = -1
        if descriptor >= 0:
            os.close(descriptor)

    def __enter__(self) -> Self:
        if self._fd < 0:
            raise SMLArtifactError("artifact root is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()

    def _open_directory_descriptor(
        self,
        components: tuple[str, ...],
        *,
        logical_path: str,
        context: str,
    ) -> int:
        if self._fd < 0:
            raise SMLArtifactError("artifact root is closed")
        descriptor = -1
        try:
            descriptor = os.dup(self._fd)
            for component in components:
                next_descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
                try:
                    component_stat = os.fstat(next_descriptor)
                    if not stat.S_ISDIR(component_stat.st_mode):
                        raise SMLArtifactError(
                            "no-follow traversal found a non-directory component "
                            f"in {context}: {logical_path}"
                        )
                except BaseException:
                    os.close(next_descriptor)
                    raise
                previous_descriptor = descriptor
                descriptor = next_descriptor
                os.close(previous_descriptor)
            result = descriptor
            descriptor = -1
            return result
        except SMLArtifactError:
            raise
        except OSError as error:
            raise SMLArtifactError(
                f"no-follow traversal failed for {context}: {logical_path}"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def open_child(self, logical_path: str) -> ArtifactRoot:
        components = parse_logical_path(logical_path)
        descriptor = self._open_directory_descriptor(
            components,
            logical_path=logical_path,
            context="child root",
        )
        try:
            return ArtifactRoot(descriptor, local_apfs=self._local_apfs)
        except BaseException:
            os.close(descriptor)
            raise

    def _stat_direct_payload(self, logical_path: str) -> os.stat_result | None:
        components = parse_logical_path(logical_path)
        if len(components) != 1:
            raise SMLArtifactError(
                f"direct payload must be one portable component: {logical_path!r}"
            )
        if self._fd < 0:
            raise SMLArtifactError("artifact root is closed")
        try:
            payload_stat = os.stat(
                components[0],
                dir_fd=self._fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            raise SMLArtifactError(
                f"could not stat direct artifact payload: {logical_path}"
            ) from error
        if not stat.S_ISREG(payload_stat.st_mode):
            raise SMLArtifactError(
                f"direct artifact payload is not a regular file: {logical_path}"
            )
        if payload_stat.st_nlink != 1:
            raise SMLArtifactError(
                f"direct artifact payload link count must be exactly one: {logical_path}"
            )
        return payload_stat

    def _open_payload_with_stat(
        self, logical_path: str
    ) -> tuple[BinaryIO, os.stat_result]:
        components = parse_logical_path(logical_path)

        current_descriptor = -1
        final_descriptor = -1
        stream: BinaryIO | None = None
        result: tuple[BinaryIO, os.stat_result] | None = None
        primary_error: BaseException | None = None
        try:
            current_descriptor = self._open_directory_descriptor(
                components[:-1],
                logical_path=logical_path,
                context="payload",
            )

            final_descriptor = os.open(
                components[-1],
                _PAYLOAD_OPEN_FLAGS,
                dir_fd=current_descriptor,
            )
            stream = cast(BinaryIO, os.fdopen(final_descriptor, "rb"))
            final_descriptor = -1
            payload_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(payload_stat.st_mode):
                raise SMLArtifactError(f"payload is not a regular file: {logical_path}")
            if payload_stat.st_nlink != 1:
                raise SMLArtifactError(
                    f"payload link count must be exactly one: {logical_path}"
                )

            inode = (payload_stat.st_dev, payload_stat.st_ino)
            previous_path = self._inode_paths.get(inode)
            if previous_path is not None and previous_path != components:
                raise SMLArtifactError(
                    "distinct logical paths resolve to one inode alias: "
                    f"{'/'.join(previous_path)!r} and {logical_path!r}"
                )
            self._inode_paths[inode] = components

            result_stream = stream
            stream = None
            result = (result_stream, payload_stat)
        except SMLArtifactError as error:
            primary_error = error
        except OSError as error:
            primary_error = SMLArtifactError(
                f"no-follow traversal failed for payload: {logical_path}"
            )
            primary_error.__cause__ = error
        except BaseException as error:  # noqa: BLE001 - preserve unexpected primary
            primary_error = error

        cleanup_actions: list[Callable[[], object]] = []
        if stream is not None:
            cleanup_actions.append(stream.close)
        if final_descriptor >= 0:
            cleanup_actions.append(lambda: os.close(final_descriptor))
        if current_descriptor >= 0:
            cleanup_actions.append(lambda: os.close(current_descriptor))
        cleanup_error = _cleanup_exception(cleanup_actions)

        if primary_error is not None:
            if cleanup_error is not None:
                raise primary_error from cleanup_error
            raise primary_error
        if cleanup_error is not None:
            assert result is not None
            escaped_stream = result[0]
            secondary_cleanup = _cleanup_exception((escaped_stream.close,))
            if secondary_cleanup is not None:
                raise cleanup_error from secondary_cleanup
            raise cleanup_error
        assert result is not None
        return result

    def open_payload(self, logical_path: str) -> BinaryIO:
        stream, _opened_stat = self._open_payload_with_stat(logical_path)
        return stream

    def verify_payloads(self, references: Sequence[PayloadRef], *, full: bool) -> None:
        if not isinstance(full, bool):
            raise TypeError("full must be a bool")
        verification = (
            VerificationLevel.FULL if full else VerificationLevel.MANIFEST_TRUSTED
        )
        for reference in _validated_payload_references(references):
            with _open_verified_payload(self, reference, verification):
                pass


@dataclass(frozen=True, slots=True)
class ArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        _require_string(self.name, "array name")
        _require_string(self.dtype, "array dtype")
        shape = _require_tuple(self.shape, "array shape")
        for dimension in shape:
            _require_plain_int(dimension, "array shape dimension")


@dataclass(frozen=True, slots=True)
class ArrayPayloadRef:
    payload: PayloadRef
    arrays: tuple[ArraySpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, PayloadRef):
            raise TypeError("payload must be a PayloadRef")
        arrays = _require_tuple(self.arrays, "arrays")
        if not all(isinstance(array, ArraySpec) for array in arrays):
            raise TypeError("arrays must contain ArraySpec values")
        names = [array.name for array in arrays]
        if len(names) != len(set(names)):
            raise ValueError("array names must be unique within a payload")


class VerificationLevel(Enum):
    MANIFEST_TRUSTED = "manifest-trusted"
    FULL = "full"


def _close_descriptor_bound_stream(
    stream: BinaryIO,
    opened_stat: os.stat_result,
    *,
    logical_path: str,
) -> None:
    postcheck_error: BaseException | None = None
    try:
        current_stat = os.fstat(stream.fileno())
        if _stable_stat_fields(current_stat) != _stable_stat_fields(opened_stat):
            postcheck_error = SMLArtifactError(
                f"payload changed during use: {logical_path}"
            )
    except OSError as error:
        postcheck_error = SMLArtifactError(
            f"could not perform payload stability postcheck: {logical_path}"
        )
        postcheck_error.__cause__ = error
    except BaseException as error:  # noqa: BLE001 - cleanup must still run
        postcheck_error = error

    close_error = _cleanup_exception((stream.close,))
    if postcheck_error is not None:
        if close_error is not None:
            raise postcheck_error from close_error
        raise postcheck_error
    if close_error is not None:
        raise close_error


class _StablePayload:
    """A raw payload descriptor pinned through a caller's semantic consumption."""

    def __init__(
        self,
        *,
        logical_path: str,
        stream: BinaryIO,
        opened_stat: os.stat_result,
    ) -> None:
        self.logical_path = logical_path
        self.stream = stream
        self.opened_stat = opened_stat
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            _close_descriptor_bound_stream(
                self.stream,
                self.opened_stat,
                logical_path=self.logical_path,
            )
        finally:
            self.closed = True

    def read(self) -> bytes:
        if self.closed:
            raise SMLArtifactError("stable payload is closed")
        encoded = self.stream.read()
        consumed_stat = os.fstat(self.stream.fileno())
        if _stable_stat_fields(consumed_stat) != _stable_stat_fields(self.opened_stat):
            raise SMLArtifactError(
                f"payload changed while being read: {self.logical_path}"
            )
        return encoded

    def __enter__(self) -> Self:
        if self.closed:
            raise SMLArtifactError("stable payload is closed")
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


def _open_stable_payload(root: ArtifactRoot, logical_path: str) -> _StablePayload:
    stream, opened_stat = root._open_payload_with_stat(logical_path)
    return _StablePayload(
        logical_path=logical_path,
        stream=stream,
        opened_stat=opened_stat,
    )


class VerifiedPayload:
    """One proven payload descriptor retained through semantic consumption."""

    def __init__(
        self,
        *,
        reference: PayloadRef,
        verification: VerificationLevel,
        stream: BinaryIO,
        opened_stat: os.stat_result,
    ) -> None:
        self.reference = reference
        self.verification = verification
        self.stream = stream
        self.opened_stat = opened_stat
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            _close_descriptor_bound_stream(
                self.stream,
                self.opened_stat,
                logical_path=self.reference.logical_path,
            )
        finally:
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise SMLArtifactError("verified payload is closed")
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


def _open_verified_payload(
    root: ArtifactRoot,
    reference: PayloadRef,
    verification: VerificationLevel,
) -> VerifiedPayload:
    if not isinstance(reference, PayloadRef):
        raise TypeError("reference must be a PayloadRef")
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")

    stream, opened_stat = root._open_payload_with_stat(reference.logical_path)
    payload: VerifiedPayload | None = None
    try:
        payload = VerifiedPayload(
            reference=reference,
            verification=verification,
            stream=stream,
            opened_stat=opened_stat,
        )
        if opened_stat.st_size != reference.byte_size:
            raise SMLArtifactError(
                f"payload byte size mismatch: {reference.logical_path}"
            )
        if (
            verification is VerificationLevel.FULL
            and file_identity(stream) != reference.identity
        ):
            raise SMLArtifactError(
                f"payload identity mismatch: {reference.logical_path}"
            )
        stream.seek(0)
        return payload
    except BaseException as error:
        if payload is None:
            stream.close()
        else:
            try:
                payload.close()
            except BaseException as close_error:
                raise error from close_error
        raise


@dataclass(frozen=True, slots=True)
class Verified[M]:
    manifest: M
    verification: VerificationLevel

    def __post_init__(self) -> None:
        if not isinstance(self.verification, VerificationLevel):
            raise TypeError("verification must be a VerificationLevel")


class _Manifest:
    EXPECTED_KIND: ClassVar[str]
    EXPECTED_VERSION: ClassVar[int] = 1
    IDENTITY_DOMAIN: ClassVar[str]
    MANIFEST_FILENAME: ClassVar[str] = "manifest.json"

    kind: str
    version: int
    identity: str

    def _validate_common(self) -> None:
        _require_string(self.kind, "kind")
        if self.kind != self.EXPECTED_KIND:
            raise ValueError(f"kind must be {self.EXPECTED_KIND!r}")
        _require_plain_int(self.version, "version", minimum=1)
        if self.version != self.EXPECTED_VERSION:
            raise ValueError(f"version must be {self.EXPECTED_VERSION}")
        _require_identity(self.identity, "manifest identity")

    def identity_projection(self) -> Mapping[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name != "identity" and not field.name.startswith("diagnostic_")
        }

    def recompute_identity(self) -> str:
        return structured_identity(self.IDENTITY_DOMAIN, self.identity_projection())


@dataclass(frozen=True, slots=True)
class TokenizerManifest(_Manifest):
    kind: str
    version: int
    identity: str
    algorithm: str
    training: Mapping[str, object]
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_token_id: int
    model: PayloadRef
    vocab: PayloadRef
    diagnostic_source_locator: str | None

    EXPECTED_KIND: ClassVar[str] = "tokenizer"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-tokenizer-manifest-v1"

    def __post_init__(self) -> None:
        self._validate_common()
        _require_string(self.algorithm, "algorithm")
        object.__setattr__(
            self, "training", _freeze_json_mapping(self.training, "training")
        )
        _require_plain_int(self.vocab_size, "vocab_size", minimum=1)
        token_ids = (
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id,
        )
        for token_id in token_ids:
            _require_plain_int(token_id, "special token ID")
            if token_id >= self.vocab_size:
                raise ValueError("special token ID must be smaller than vocab_size")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("special token IDs must be unique")
        if not isinstance(self.model, PayloadRef) or not isinstance(
            self.vocab, PayloadRef
        ):
            raise TypeError("model and vocab must be PayloadRef values")
        _require_optional_string(
            self.diagnostic_source_locator, "diagnostic_source_locator"
        )


@dataclass(frozen=True, slots=True)
class PretrainingDataManifest(_Manifest):
    kind: str
    version: int
    identity: str
    sequence_length: int
    row_width: int
    dtype: str
    shard_row_counts: tuple[int, ...]
    shards: tuple[PayloadRef, ...]
    preparation_seed: int
    row_order_policy: Mapping[str, object]
    tokenizer_identity: str
    tokenizer_model: PayloadRef
    tokenizer_vocab: PayloadRef
    source_summary: Mapping[str, object]
    diagnostic_source_locator: str | None
    row_content_identity: str

    EXPECTED_KIND: ClassVar[str] = "pretraining-data"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-pretraining-data-manifest-v1"

    def __post_init__(self) -> None:
        self._validate_common()
        _require_plain_int(self.sequence_length, "sequence_length", minimum=1)
        _require_plain_int(self.row_width, "row_width", minimum=1)
        if self.row_width != self.sequence_length + 1:
            raise ValueError("row_width must equal sequence_length + 1")
        if self.dtype != "int32":
            raise ValueError("dtype must be 'int32'")
        counts = _require_tuple(self.shard_row_counts, "shard_row_counts")
        shards = _require_tuple(self.shards, "shards")
        if len(counts) != len(shards):
            raise ValueError("shard row counts must match ordered shard refs")
        for count in counts:
            _require_plain_int(count, "shard row count", minimum=1)
        if not all(isinstance(shard, PayloadRef) for shard in shards):
            raise TypeError("shards must contain PayloadRef values")
        _require_unique_payload_paths(shards, "shards")
        _require_plain_int(self.preparation_seed, "preparation_seed")
        object.__setattr__(
            self,
            "row_order_policy",
            _freeze_json_mapping(self.row_order_policy, "row_order_policy"),
        )
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        if not isinstance(self.tokenizer_model, PayloadRef) or not isinstance(
            self.tokenizer_vocab, PayloadRef
        ):
            raise TypeError(
                "tokenizer_model and tokenizer_vocab must be PayloadRef values"
            )
        object.__setattr__(
            self,
            "source_summary",
            _freeze_json_mapping(self.source_summary, "source_summary"),
        )
        _require_optional_string(
            self.diagnostic_source_locator, "diagnostic_source_locator"
        )
        _require_identity(self.row_content_identity, "row_content_identity")


def _require_array_payload_path(
    value: ArrayPayloadRef,
    name: str,
    logical_path: str,
) -> None:
    if not isinstance(value, ArrayPayloadRef):
        raise TypeError(f"{name} must be an ArrayPayloadRef")
    if value.payload.logical_path != logical_path:
        raise ValueError(f"{name} logical path must be exactly {logical_path!r}")


@dataclass(frozen=True, slots=True)
class PretrainingCheckpointManifest(_Manifest):
    kind: str
    version: int
    identity: str
    owning_run_identity: str
    step: int
    scalar_state: PayloadRef
    model: ArrayPayloadRef
    master: ArrayPayloadRef
    optimizer: ArrayPayloadRef
    trainer: ArrayPayloadRef

    EXPECTED_KIND: ClassVar[str] = "pretraining-checkpoint"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-pretraining-checkpoint-manifest-v1"
    MANIFEST_FILENAME: ClassVar[str] = "checkpoint.json"

    def __post_init__(self) -> None:
        self._validate_common()
        _require_identity(self.owning_run_identity, "owning_run_identity")
        _require_plain_int(self.step, "step")
        if not isinstance(self.scalar_state, PayloadRef):
            raise TypeError("scalar_state must be a PayloadRef")
        if self.scalar_state.logical_path != "state.json":
            raise ValueError("scalar_state logical path must be exactly 'state.json'")
        _require_unique_payload_paths(
            (self.model, self.master, self.optimizer, self.trainer),
            "pretraining checkpoint array groups",
        )
        for name, logical_path in (
            ("model", "model.safetensors"),
            ("master", "master.safetensors"),
            ("optimizer", "optimizer.safetensors"),
            ("trainer", "trainer.safetensors"),
        ):
            _require_array_payload_path(getattr(self, name), name, logical_path)


@dataclass(frozen=True, slots=True)
class LoRACheckpointManifest(_Manifest):
    kind: str
    version: int
    identity: str
    owning_run_identity: str
    step: int
    scalar_state: PayloadRef
    adapters: ArrayPayloadRef
    optimizer: ArrayPayloadRef
    trainer: ArrayPayloadRef

    EXPECTED_KIND: ClassVar[str] = "lora-checkpoint"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-lora-checkpoint-manifest-v1"
    MANIFEST_FILENAME: ClassVar[str] = "checkpoint.json"

    def __post_init__(self) -> None:
        self._validate_common()
        _require_identity(self.owning_run_identity, "owning_run_identity")
        _require_plain_int(self.step, "step")
        if not isinstance(self.scalar_state, PayloadRef):
            raise TypeError("scalar_state must be a PayloadRef")
        if self.scalar_state.logical_path != "state.json":
            raise ValueError("scalar_state logical path must be exactly 'state.json'")
        _require_unique_payload_paths(
            (self.adapters, self.optimizer, self.trainer),
            "LoRA checkpoint array groups",
        )
        for name, logical_path in (
            ("adapters", "adapters.safetensors"),
            ("optimizer", "optimizer.safetensors"),
            ("trainer", "trainer.safetensors"),
        ):
            _require_array_payload_path(getattr(self, name), name, logical_path)


type CheckpointManifest = PretrainingCheckpointManifest | LoRACheckpointManifest


@dataclass(frozen=True, slots=True)
class PretrainingRunManifest(_Manifest):
    kind: str
    version: int
    identity: str
    model: Mapping[str, object]
    precision: Mapping[str, object]
    optimizer: Mapping[str, object]
    loader: Mapping[str, object]
    checkpoint: Mapping[str, object]
    tokenizer_identity: str
    data_identity: str
    diagnostic_data_locator: str | None

    EXPECTED_KIND: ClassVar[str] = "pretraining-run"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-pretraining-run-manifest-v1"
    MANIFEST_FILENAME: ClassVar[str] = "run.json"

    def __post_init__(self) -> None:
        self._validate_common()
        for name in ("model", "precision", "optimizer", "loader", "checkpoint"):
            object.__setattr__(
                self, name, _freeze_json_mapping(getattr(self, name), name)
            )
        _require_training_rng_schedule(self.checkpoint)
        rope_factor = self.model.get("rope_scaling_factor")
        if not isinstance(rope_factor, float) or rope_factor != 1.0:
            raise ValueError(
                "pretraining model rope_scaling_factor must be exactly 1.0"
            )
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        _require_identity(self.data_identity, "data_identity")
        _require_optional_string(
            self.diagnostic_data_locator, "diagnostic_data_locator"
        )


@dataclass(frozen=True, slots=True)
class LoRARunManifest(_Manifest):
    kind: str
    version: int
    identity: str
    model: Mapping[str, object]
    lora: Mapping[str, object]
    precision: Mapping[str, object]
    optimizer: Mapping[str, object]
    loader: Mapping[str, object]
    checkpoint: Mapping[str, object]
    tokenizer_identity: str
    base_identity: str
    data_identity: str
    diagnostic_data_locator: str | None

    EXPECTED_KIND: ClassVar[str] = "lora-run"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-lora-run-manifest-v1"
    MANIFEST_FILENAME: ClassVar[str] = "run.json"

    def __post_init__(self) -> None:
        self._validate_common()
        for name in (
            "model",
            "lora",
            "precision",
            "optimizer",
            "loader",
            "checkpoint",
        ):
            object.__setattr__(
                self, name, _freeze_json_mapping(getattr(self, name), name)
            )
        _require_training_rng_schedule(self.checkpoint)
        rope_factor = self.model.get("rope_scaling_factor")
        if not isinstance(rope_factor, float) or rope_factor != 1.0:
            raise ValueError("LoRA model rope_scaling_factor must be exactly 1.0")
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        _require_identity(self.base_identity, "base_identity")
        _require_identity(self.data_identity, "data_identity")
        _require_optional_string(
            self.diagnostic_data_locator, "diagnostic_data_locator"
        )


type RunManifest = PretrainingRunManifest | LoRARunManifest


@dataclass(frozen=True, slots=True)
class LatestIndex(_Manifest):
    kind: str
    version: int
    identity: str
    owning_run_identity: str
    step: int
    checkpoint_identity: str

    EXPECTED_KIND: ClassVar[str] = "latest-index"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-latest-index-v1"
    MANIFEST_FILENAME: ClassVar[str] = "latest.json"

    def __post_init__(self) -> None:
        self._validate_common()
        _require_identity(self.owning_run_identity, "owning_run_identity")
        _require_plain_int(self.step, "step")
        _require_identity(self.checkpoint_identity, "checkpoint_identity")


@dataclass(frozen=True, slots=True)
class BaseSnapshotManifest(_Manifest):
    kind: str
    version: int
    identity: str
    model: Mapping[str, object]
    precision: Mapping[str, object]
    tokenizer_identity: str
    working_weights: ArrayPayloadRef
    diagnostic_source_run_identity: str
    diagnostic_source_step: int

    EXPECTED_KIND: ClassVar[str] = "base-snapshot"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-base-snapshot-manifest-v1"

    def __post_init__(self) -> None:
        self._validate_common()
        object.__setattr__(self, "model", _freeze_json_mapping(self.model, "model"))
        object.__setattr__(
            self, "precision", _freeze_json_mapping(self.precision, "precision")
        )
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        if not isinstance(self.working_weights, ArrayPayloadRef):
            raise TypeError("working_weights must be an ArrayPayloadRef")
        _require_identity(
            self.diagnostic_source_run_identity, "diagnostic_source_run_identity"
        )
        _require_plain_int(self.diagnostic_source_step, "diagnostic_source_step")


@dataclass(frozen=True, slots=True)
class SwagDataManifest(_Manifest):
    kind: str
    version: int
    identity: str
    source: Mapping[str, object]
    preprocessing: Mapping[str, object]
    base_identity: str
    tokenizer_identity: str
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    unk_token_id: int
    example_count: int
    dropped_overlength_rows: int
    buckets: tuple[ArrayPayloadRef, ...]

    EXPECTED_KIND: ClassVar[str] = "swag-data"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-swag-data-manifest-v1"

    def __post_init__(self) -> None:
        self._validate_common()
        object.__setattr__(self, "source", _freeze_json_mapping(self.source, "source"))
        object.__setattr__(
            self,
            "preprocessing",
            _freeze_json_mapping(self.preprocessing, "preprocessing"),
        )
        _require_identity(self.base_identity, "base_identity")
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        _require_plain_int(self.vocab_size, "vocab_size", minimum=1)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            token_id = _require_plain_int(getattr(self, name), name)
            if token_id >= self.vocab_size:
                raise ValueError(f"{name} must be smaller than vocab_size")
        _require_plain_int(self.example_count, "example_count", minimum=1)
        _require_plain_int(self.dropped_overlength_rows, "dropped_overlength_rows")
        buckets = _require_tuple(self.buckets, "buckets")
        if not buckets or not all(
            isinstance(bucket, ArrayPayloadRef) for bucket in buckets
        ):
            raise TypeError(
                "buckets must be a nonempty tuple of ArrayPayloadRef values"
            )
        _require_unique_payload_paths(buckets, "buckets")


@dataclass(frozen=True, slots=True)
class ExportManifest(_Manifest):
    kind: str
    version: int
    identity: str
    model: Mapping[str, object]
    precision: Mapping[str, object]
    tokenizer_identity: str
    model_weights: ArrayPayloadRef
    tokenizer_model: PayloadRef
    tokenizer_vocab: PayloadRef
    diagnostic_source_run_identity: str
    diagnostic_source_step: int

    EXPECTED_KIND: ClassVar[str] = "export"
    IDENTITY_DOMAIN: ClassVar[str] = "sml-export-manifest-v1"

    def __post_init__(self) -> None:
        self._validate_common()
        object.__setattr__(self, "model", _freeze_json_mapping(self.model, "model"))
        object.__setattr__(
            self, "precision", _freeze_json_mapping(self.precision, "precision")
        )
        _require_identity(self.tokenizer_identity, "tokenizer_identity")
        if not isinstance(self.model_weights, ArrayPayloadRef):
            raise TypeError("model_weights must be an ArrayPayloadRef")
        if not isinstance(self.tokenizer_model, PayloadRef) or not isinstance(
            self.tokenizer_vocab, PayloadRef
        ):
            raise TypeError(
                "tokenizer_model and tokenizer_vocab must be PayloadRef values"
            )
        _require_identity(
            self.diagnostic_source_run_identity, "diagnostic_source_run_identity"
        )
        _require_plain_int(self.diagnostic_source_step, "diagnostic_source_step")


_MANIFEST_TYPES: tuple[type[_Manifest], ...] = (
    TokenizerManifest,
    PretrainingDataManifest,
    PretrainingCheckpointManifest,
    LoRACheckpointManifest,
    PretrainingRunManifest,
    LoRARunManifest,
    LatestIndex,
    BaseSnapshotManifest,
    SwagDataManifest,
    ExportManifest,
)

RUN_MANIFEST_TYPES = (PretrainingRunManifest, LoRARunManifest)
CHECKPOINT_MANIFEST_TYPES = (
    PretrainingCheckpointManifest,
    LoRACheckpointManifest,
)


class OpenedArtifact[M: _Manifest]:
    """A strict manifest and the retained root descriptor that owns it."""

    def __init__(
        self,
        *,
        path: Path,
        root: ArtifactRoot,
        manifest: M,
        verification: VerificationLevel,
    ) -> None:
        self.path = path
        self.root = root
        self.manifest = manifest
        self.verification = verification
        self.closed = False
        self._owns_root = True

    def open_payload(self, reference: PayloadRef) -> VerifiedPayload:
        if self.closed or not self._owns_root:
            raise SMLArtifactError("opened artifact is closed")
        return _open_verified_payload(self.root, reference, self.verification)

    def open_child[ChildManifest: _Manifest](
        self,
        logical_path: str,
        manifest_types: tuple[type[ChildManifest], ...],
    ) -> OpenedArtifact[ChildManifest]:
        if self.closed or not self._owns_root:
            raise SMLArtifactError("opened artifact is closed")
        components = parse_logical_path(logical_path)
        child_root = self.root.open_child(logical_path)
        child_path = self.path.joinpath(*components)
        try:
            return _open_artifact_from_root(
                child_path,
                child_root,
                manifest_types,
                self.verification,
            )
        except BaseException:
            child_root.close()
            raise

    def detach_root(self) -> ArtifactRoot:
        if self.closed or not self._owns_root:
            raise SMLArtifactError("artifact root ownership was already transferred")
        self._owns_root = False
        self.closed = True
        return self.root

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._owns_root:
            self._owns_root = False
            self.root.close()

    def __enter__(self) -> Self:
        if self.closed or not self._owns_root:
            raise SMLArtifactError("opened artifact is closed")
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()


def _strict_keys(raw: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(raw)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise SMLArtifactError(f"{context} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise SMLArtifactError(f"{context} has missing field(s): {', '.join(missing)}")


def _parse_payload_ref(raw: object) -> PayloadRef:
    if not isinstance(raw, Mapping):
        raise SMLArtifactError("payload ref must be an object")
    _strict_keys(raw, {"logical_path", "identity", "byte_size"}, "payload ref")
    return PayloadRef(
        logical_path=_require_string(raw["logical_path"], "logical_path"),
        identity=_require_identity(raw["identity"], "payload identity"),
        byte_size=_require_plain_int(raw["byte_size"], "byte_size"),
    )


def _parse_array_spec(raw: object) -> ArraySpec:
    if not isinstance(raw, Mapping):
        raise SMLArtifactError("array spec must be an object")
    _strict_keys(raw, {"name", "shape", "dtype"}, "array spec")
    shape = raw["shape"]
    if not isinstance(shape, list):
        raise SMLArtifactError("array shape must be a JSON array")
    return ArraySpec(
        name=_require_string(raw["name"], "array name"),
        shape=tuple(
            _require_plain_int(item, "array shape dimension") for item in shape
        ),
        dtype=_require_string(raw["dtype"], "array dtype"),
    )


def _parse_array_payload_ref(raw: object) -> ArrayPayloadRef:
    if not isinstance(raw, Mapping):
        raise SMLArtifactError("array payload ref must be an object")
    _strict_keys(raw, {"payload", "arrays"}, "array payload ref")
    arrays = raw["arrays"]
    if not isinstance(arrays, list):
        raise SMLArtifactError("arrays must be a JSON array")
    return ArrayPayloadRef(
        payload=_parse_payload_ref(raw["payload"]),
        arrays=tuple(_parse_array_spec(item) for item in arrays),
    )


def _parse_payload_tuple(raw: object, name: str) -> tuple[PayloadRef, ...]:
    if not isinstance(raw, list):
        raise SMLArtifactError(f"{name} must be a JSON array")
    return tuple(_parse_payload_ref(item) for item in raw)


def _parse_array_payload_tuple(raw: object, name: str) -> tuple[ArrayPayloadRef, ...]:
    if not isinstance(raw, list):
        raise SMLArtifactError(f"{name} must be a JSON array")
    return tuple(_parse_array_payload_ref(item) for item in raw)


def _parse_int_tuple(raw: object, name: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise SMLArtifactError(f"{name} must be a JSON array")
    return tuple(_require_plain_int(item, name) for item in raw)


def _parse_manifest[M: _Manifest](raw: object, manifest_type: type[M]) -> M:
    if manifest_type not in _MANIFEST_TYPES:
        raise SMLArtifactError(f"unsupported manifest type: {manifest_type!r}")
    if not isinstance(raw, Mapping):
        raise SMLArtifactError("manifest must be a JSON object")
    expected_fields = {field.name for field in dataclasses.fields(manifest_type)}
    _strict_keys(raw, expected_fields, manifest_type.__name__)
    if not isinstance(raw["kind"], str) or raw["kind"] != manifest_type.EXPECTED_KIND:
        raise SMLArtifactError(f"manifest kind must be {manifest_type.EXPECTED_KIND!r}")
    if (
        isinstance(raw["version"], bool)
        or not isinstance(raw["version"], int)
        or raw["version"] != manifest_type.EXPECTED_VERSION
    ):
        raise SMLArtifactError(
            f"manifest version must be {manifest_type.EXPECTED_VERSION}"
        )
    _require_identity(raw["identity"], "manifest identity")

    common = {
        "kind": raw["kind"],
        "version": raw["version"],
        "identity": raw["identity"],
    }
    if manifest_type is TokenizerManifest:
        manifest = TokenizerManifest(
            **common,
            algorithm=raw["algorithm"],
            training=raw["training"],
            vocab_size=raw["vocab_size"],
            bos_token_id=raw["bos_token_id"],
            eos_token_id=raw["eos_token_id"],
            pad_token_id=raw["pad_token_id"],
            unk_token_id=raw["unk_token_id"],
            model=_parse_payload_ref(raw["model"]),
            vocab=_parse_payload_ref(raw["vocab"]),
            diagnostic_source_locator=raw["diagnostic_source_locator"],
        )
    elif manifest_type is PretrainingDataManifest:
        manifest = PretrainingDataManifest(
            **common,
            sequence_length=raw["sequence_length"],
            row_width=raw["row_width"],
            dtype=raw["dtype"],
            shard_row_counts=_parse_int_tuple(
                raw["shard_row_counts"], "shard_row_counts"
            ),
            shards=_parse_payload_tuple(raw["shards"], "shards"),
            preparation_seed=raw["preparation_seed"],
            row_order_policy=raw["row_order_policy"],
            tokenizer_identity=raw["tokenizer_identity"],
            tokenizer_model=_parse_payload_ref(raw["tokenizer_model"]),
            tokenizer_vocab=_parse_payload_ref(raw["tokenizer_vocab"]),
            source_summary=raw["source_summary"],
            diagnostic_source_locator=raw["diagnostic_source_locator"],
            row_content_identity=raw["row_content_identity"],
        )
    elif manifest_type is PretrainingCheckpointManifest:
        manifest = PretrainingCheckpointManifest(
            **common,
            owning_run_identity=raw["owning_run_identity"],
            step=raw["step"],
            scalar_state=_parse_payload_ref(raw["scalar_state"]),
            model=_parse_array_payload_ref(raw["model"]),
            master=_parse_array_payload_ref(raw["master"]),
            optimizer=_parse_array_payload_ref(raw["optimizer"]),
            trainer=_parse_array_payload_ref(raw["trainer"]),
        )
    elif manifest_type is LoRACheckpointManifest:
        manifest = LoRACheckpointManifest(
            **common,
            owning_run_identity=raw["owning_run_identity"],
            step=raw["step"],
            scalar_state=_parse_payload_ref(raw["scalar_state"]),
            adapters=_parse_array_payload_ref(raw["adapters"]),
            optimizer=_parse_array_payload_ref(raw["optimizer"]),
            trainer=_parse_array_payload_ref(raw["trainer"]),
        )
    elif manifest_type is PretrainingRunManifest:
        manifest = PretrainingRunManifest(
            **common,
            model=raw["model"],
            precision=raw["precision"],
            optimizer=raw["optimizer"],
            loader=raw["loader"],
            checkpoint=raw["checkpoint"],
            tokenizer_identity=raw["tokenizer_identity"],
            data_identity=raw["data_identity"],
            diagnostic_data_locator=raw["diagnostic_data_locator"],
        )
    elif manifest_type is LoRARunManifest:
        manifest = LoRARunManifest(
            **common,
            model=raw["model"],
            lora=raw["lora"],
            precision=raw["precision"],
            optimizer=raw["optimizer"],
            loader=raw["loader"],
            checkpoint=raw["checkpoint"],
            tokenizer_identity=raw["tokenizer_identity"],
            base_identity=raw["base_identity"],
            data_identity=raw["data_identity"],
            diagnostic_data_locator=raw["diagnostic_data_locator"],
        )
    elif manifest_type is LatestIndex:
        manifest = LatestIndex(
            **common,
            owning_run_identity=raw["owning_run_identity"],
            step=raw["step"],
            checkpoint_identity=raw["checkpoint_identity"],
        )
    elif manifest_type is BaseSnapshotManifest:
        manifest = BaseSnapshotManifest(
            **common,
            model=raw["model"],
            precision=raw["precision"],
            tokenizer_identity=raw["tokenizer_identity"],
            working_weights=_parse_array_payload_ref(raw["working_weights"]),
            diagnostic_source_run_identity=raw["diagnostic_source_run_identity"],
            diagnostic_source_step=raw["diagnostic_source_step"],
        )
    elif manifest_type is SwagDataManifest:
        manifest = SwagDataManifest(
            **common,
            source=raw["source"],
            preprocessing=raw["preprocessing"],
            base_identity=raw["base_identity"],
            tokenizer_identity=raw["tokenizer_identity"],
            vocab_size=raw["vocab_size"],
            bos_token_id=raw["bos_token_id"],
            eos_token_id=raw["eos_token_id"],
            pad_token_id=raw["pad_token_id"],
            unk_token_id=raw["unk_token_id"],
            example_count=raw["example_count"],
            dropped_overlength_rows=raw["dropped_overlength_rows"],
            buckets=_parse_array_payload_tuple(raw["buckets"], "buckets"),
        )
    else:
        manifest = ExportManifest(
            **common,
            model=raw["model"],
            precision=raw["precision"],
            tokenizer_identity=raw["tokenizer_identity"],
            model_weights=_parse_array_payload_ref(raw["model_weights"]),
            tokenizer_model=_parse_payload_ref(raw["tokenizer_model"]),
            tokenizer_vocab=_parse_payload_ref(raw["tokenizer_vocab"]),
            diagnostic_source_run_identity=raw["diagnostic_source_run_identity"],
            diagnostic_source_step=raw["diagnostic_source_step"],
        )
    return cast(M, manifest)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _payload_refs(value: object) -> Iterator[PayloadRef]:
    if isinstance(value, PayloadRef):
        yield value
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _payload_refs(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _payload_refs(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _payload_refs(item)


def _manifest_type_for_raw[M: _Manifest](
    raw: object,
    manifest_types: tuple[type[M], ...],
) -> type[M]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("kind"), str):
        raise SMLArtifactError("manifest must contain a string kind discriminator")
    by_kind = {
        manifest_type.EXPECTED_KIND: manifest_type for manifest_type in manifest_types
    }
    try:
        return by_kind[raw["kind"]]
    except KeyError as error:
        raise SMLArtifactError(
            f"unsupported manifest kind for this owner: {raw['kind']!r}"
        ) from error


def _validated_manifest_types[M: _Manifest](
    manifest_types: tuple[type[M], ...],
) -> tuple[type[M], ...]:
    if not isinstance(manifest_types, tuple):
        raise TypeError("manifest_types must be a tuple")
    if not manifest_types or any(
        manifest_type not in _MANIFEST_TYPES for manifest_type in manifest_types
    ):
        raise SMLArtifactError(f"unsupported manifest types: {manifest_types!r}")
    filenames = {manifest_type.MANIFEST_FILENAME for manifest_type in manifest_types}
    if len(filenames) != 1:
        raise SMLArtifactError("discriminated manifest types must share one filename")
    kinds = [manifest_type.EXPECTED_KIND for manifest_type in manifest_types]
    if len(kinds) != len(set(kinds)):
        raise SMLArtifactError("manifest_types must contain unique kinds")
    return manifest_types


def _parse_and_validate_manifest_bytes[M: _Manifest](
    encoded: bytes,
    path: Path,
    manifest_types: tuple[type[M], ...],
) -> M:
    manifest_types = _validated_manifest_types(manifest_types)
    try:
        text = encoded.decode("utf-8")
        raw = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_json_object_no_duplicates,
        )
        manifest_type = _manifest_type_for_raw(raw, manifest_types)
        manifest = _parse_manifest(raw, manifest_type)

        recomputed = manifest.recompute_identity()
        if recomputed != manifest.identity:
            raise SMLArtifactError(
                "manifest identity mismatch: "
                f"stored {manifest.identity}, recomputed {recomputed}"
            )

        canonical = canonical_json_bytes(manifest)
        if encoded != canonical:
            raise SMLArtifactError(f"manifest is not canonical JSON bytes: {path}")
    except SMLArtifactError as error:
        raise SMLArtifactError(f"invalid manifest at {path}: {error}") from error
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise SMLArtifactError(f"invalid manifest at {path}: {error}") from error

    return manifest


def _read_manifest_from_root[M: _Manifest](
    root: ArtifactRoot,
    path: Path,
    manifest_types: tuple[type[M], ...],
    *,
    validate_opened: Callable[[os.stat_result], None] | None = None,
    validate_before_close: Callable[[os.stat_result], None] | None = None,
) -> M:
    manifest_types = _validated_manifest_types(manifest_types)
    filename = next(
        iter({manifest_type.MANIFEST_FILENAME for manifest_type in manifest_types})
    )
    with _open_stable_payload(root, filename) as payload:
        if validate_opened is not None:
            validate_opened(payload.opened_stat)
        encoded = payload.read()
        manifest = _parse_and_validate_manifest_bytes(
            encoded,
            path / filename,
            manifest_types,
        )
        if validate_before_close is not None:
            validate_before_close(payload.opened_stat)
        return manifest


def _open_artifact_from_root[M: _Manifest](
    path: Path,
    root: ArtifactRoot,
    manifest_types: tuple[type[M], ...],
    verification: VerificationLevel,
) -> OpenedArtifact[M]:
    try:
        manifest = _read_manifest_from_root(root, path, manifest_types)
        return OpenedArtifact(
            path=path,
            root=root,
            manifest=manifest,
            verification=verification,
        )
    except BaseException as error:
        try:
            root.close()
        except BaseException as close_error:
            raise error from close_error
        raise


def open_artifact[M: _Manifest](
    path: Path,
    manifest_types: tuple[type[M], ...],
    verification: VerificationLevel,
) -> OpenedArtifact[M]:
    """Open one strict artifact while retaining its root descriptor."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    _validated_manifest_types(manifest_types)
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    root = ArtifactRoot.open(path, writable=False)
    return _open_artifact_from_root(path, root, manifest_types, verification)


def _read_manifest_types[M: _Manifest](
    root: Path,
    manifest_types: tuple[type[M], ...],
    verification: VerificationLevel,
) -> Verified[M]:
    with open_artifact(root, manifest_types, verification) as artifact:
        references = _validated_payload_references(
            tuple(_payload_refs(artifact.manifest))
        )
        for reference in references:
            with artifact.open_payload(reference):
                pass
        return Verified(manifest=artifact.manifest, verification=verification)


def read_manifest[M: _Manifest](
    root: Path, manifest_type: type[M], verification: VerificationLevel
) -> Verified[M]:
    """Read one exact version-1 schema and verify its structured identity."""
    return _read_manifest_types(root, (manifest_type,), verification)


def read_run_manifest(
    root: Path, verification: VerificationLevel
) -> Verified[RunManifest]:
    """Dispatch one strict run kind from ``run.json``."""
    return _read_manifest_types(root, RUN_MANIFEST_TYPES, verification)


def read_checkpoint_manifest(
    root: Path, verification: VerificationLevel
) -> Verified[CheckpointManifest]:
    """Dispatch one strict checkpoint kind from ``checkpoint.json``."""
    return _read_manifest_types(root, CHECKPOINT_MANIFEST_TYPES, verification)


__all__ = [
    "CHECKPOINT_MANIFEST_TYPES",
    "RUN_MANIFEST_TYPES",
    "TRAINING_RNG_SCHEDULE",
    "ArrayPayloadRef",
    "ArraySpec",
    "ArtifactRoot",
    "BaseSnapshotManifest",
    "CheckpointManifest",
    "ExportManifest",
    "LatestIndex",
    "LoRACheckpointManifest",
    "LoRARunManifest",
    "OpenedArtifact",
    "PayloadRef",
    "PretrainingCheckpointManifest",
    "PretrainingDataManifest",
    "PretrainingRunManifest",
    "RunManifest",
    "SwagDataManifest",
    "TokenizerManifest",
    "VerificationLevel",
    "Verified",
    "VerifiedPayload",
    "canonical_json_bytes",
    "file_identity",
    "open_artifact",
    "parse_logical_path",
    "read_checkpoint_manifest",
    "read_manifest",
    "read_run_manifest",
    "row_content_identity",
    "structured_identity",
]
