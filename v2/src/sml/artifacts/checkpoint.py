from __future__ import annotations

import ctypes
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArtifactRoot,
    BaseSnapshotManifest,
    CheckpointManifest,
    ExportManifest,
    LatestIndex,
    PayloadRef,
    PretrainingDataManifest,
    RunManifest,
    SwagDataManifest,
    TokenizerManifest,
    VerificationLevel,
    _descriptor_is_local_apfs,
    _json_object_no_duplicates,
    _parse_manifest,
    _reject_json_constant,
    canonical_json_bytes,
    parse_logical_path,
    structured_identity,
)
from sml.errors import SMLArtifactError

IMMUTABLE_PUBLICATION_STAGES = (
    "payloads-written",
    "manifest-written",
    "temporary-directory-fsynced",
    "directory-renamed",
    "parent-directory-fsynced",
)

CHECKPOINT_PUBLICATION_STAGES = (
    "arrays-written",
    "scalar-state-written",
    "checkpoint-manifest-written",
    "step-directory-fsynced",
    "step-directory-renamed",
    "step-parent-fsynced",
    "latest-temporary-fsynced",
    "latest-replaced",
    "latest-parent-fsynced",
)

_OPEN_READ = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_OPEN_DIRECTORY = _OPEN_READ | os.O_DIRECTORY
_OPEN_PAYLOAD = _OPEN_READ | os.O_NONBLOCK
_OPEN_LOCK = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_OPEN_MANIFEST = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_TEMPORARY_SUFFIX_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_TEMPORARY_STEP_PATTERN = re.compile(r"\.sml-tmp-step-([0-9a-f]{32})\Z")
_LOCK_RETRY_INITIAL_SECONDS = 0.001
_LOCK_RETRY_MAX_SECONDS = 0.1
_RENAME_EXCL = 0x00000004

_MANIFEST_TYPES = (
    TokenizerManifest,
    PretrainingDataManifest,
    CheckpointManifest,
    RunManifest,
    LatestIndex,
    BaseSnapshotManifest,
    SwagDataManifest,
    ExportManifest,
)


@runtime_checkable
class FilesystemOps(Protocol):
    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int: ...

    def mkdir(
        self,
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int,
    ) -> None: ...

    def write_all(self, descriptor: int, data: bytes) -> None: ...

    def fsync_file(self, descriptor: int) -> None: ...

    def fsync_directory(self, descriptor: int) -> None: ...

    def rename(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None: ...

    def replace(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None: ...

    def unlink(self, path: str | os.PathLike[str], *, dir_fd: int) -> None: ...

    def rmdir(self, path: str | os.PathLike[str], *, dir_fd: int) -> None: ...

    def stat(
        self,
        path: int | str | os.PathLike[str],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = False,
    ) -> os.stat_result: ...

    def listdir(self, descriptor: int) -> list[str]: ...

    def flock(self, descriptor: int, operation: int) -> None: ...


class _OSFilesystemOps:
    def open(
        self,
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        return os.open(path, flags, mode, dir_fd=dir_fd)

    def mkdir(
        self,
        path: str | os.PathLike[str],
        mode: int = 0o777,
        *,
        dir_fd: int,
    ) -> None:
        os.mkdir(path, mode, dir_fd=dir_fd)

    def write_all(self, descriptor: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError(errno.EIO, "zero-byte filesystem write")
            view = view[written:]

    def fsync_file(self, descriptor: int) -> None:
        os.fsync(descriptor)

    def fsync_directory(self, descriptor: int) -> None:
        os.fsync(descriptor)

    def rename(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        if sys.platform != "darwin":
            os.rename(
                source,
                destination,
                src_dir_fd=source_dir_fd,
                dst_dir_fd=destination_dir_fd,
            )
            return
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = libc.renameatx_np
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            source_dir_fd,
            os.fsencode(source),
            destination_dir_fd,
            os.fsencode(destination),
            _RENAME_EXCL,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))

    def replace(
        self,
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        source_dir_fd: int,
        destination_dir_fd: int,
    ) -> None:
        os.replace(
            source,
            destination,
            src_dir_fd=source_dir_fd,
            dst_dir_fd=destination_dir_fd,
        )

    def unlink(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        os.unlink(path, dir_fd=dir_fd)

    def rmdir(self, path: str | os.PathLike[str], *, dir_fd: int) -> None:
        os.rmdir(path, dir_fd=dir_fd)

    def stat(
        self,
        path: int | str | os.PathLike[str],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = False,
    ) -> os.stat_result:
        if isinstance(path, int):
            if dir_fd is not None:
                raise ValueError("dir_fd cannot be used when statting a descriptor")
            return os.fstat(path)
        return os.stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def listdir(self, descriptor: int) -> list[str]:
        return os.listdir(descriptor)

    def flock(self, descriptor: int, operation: int) -> None:
        fcntl.flock(descriptor, operation)


OS_FILESYSTEM: FilesystemOps = _OSFilesystemOps()


@dataclass(frozen=True, slots=True)
class Published[M]:
    path: Path
    manifest: M
    verification: VerificationLevel

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("published path must be a Path")
        if not isinstance(self.verification, VerificationLevel):
            raise TypeError("verification must be a VerificationLevel")


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    run: RunManifest
    checkpoint: CheckpointManifest
    step_directory: Path
    run_step_identity: str
    verification: VerificationLevel
    latest_recovered: bool
    latest_repair_persisted: bool

    @property
    def step(self) -> int:
        return self.checkpoint.step


@dataclass(slots=True)
class _OwnedStep:
    resolved: ResolvedStep
    name: str
    descriptor: int
    opened_stat: os.stat_result

    def close(self) -> None:
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)


class _LockUnavailable(SMLArtifactError):
    pass


def _path_parts(protected: Path) -> tuple[Path, str]:
    if not isinstance(protected, Path):
        raise TypeError("protected path must be a Path")
    name = protected.name
    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise SMLArtifactError("protected path must have one direct-child name")
    return protected.parent, name


def _open_writable_parent(parent: Path, fs: FilesystemOps) -> int:
    descriptor = -1
    try:
        descriptor = fs.open(parent, _OPEN_DIRECTORY)
        parent_stat = fs.stat(descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise SMLArtifactError("artifact parent is not a directory")
        if not _descriptor_is_local_apfs(descriptor):
            raise SMLArtifactError(
                "writable artifact parents require a local APFS filesystem"
            )
        return descriptor
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, SMLArtifactError):
            raise
        if isinstance(error, OSError):
            raise SMLArtifactError(
                "could not open writable artifact parent with no-follow semantics: "
                f"{parent}"
            ) from error
        raise


def _name_digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _lock_name(protected_name: str, category: str) -> str:
    return f".sml-{category}-lock-{_name_digest(protected_name)}"


def _diagnostic_name(protected_name: str, category: str) -> str:
    return f".sml-{category}-diagnostic-{_name_digest(protected_name)}"


def _read_lock_owners(descriptor: int) -> list[dict[str, object]]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw_bytes = bytearray()
        while chunk := os.read(descriptor, 4096):
            raw_bytes.extend(chunk)
            if len(raw_bytes) > 1024 * 1024:
                return []
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict) or set(raw) != {"holders"}:
        return []
    holders = raw["holders"]
    if not isinstance(holders, list):
        return []
    validated: list[dict[str, object]] = []
    for holder in holders:
        if (
            not isinstance(holder, dict)
            or set(holder) != {"pid", "protected_path", "token"}
            or isinstance(holder["pid"], bool)
            or not isinstance(holder["pid"], int)
            or not isinstance(holder["protected_path"], str)
            or not isinstance(holder["token"], str)
        ):
            return []
        validated.append(holder)
    return validated


def _write_lock_owners(
    descriptor: int,
    holders: list[dict[str, object]],
) -> None:
    document = json.dumps(
        {"holders": holders},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    OS_FILESYSTEM.write_all(descriptor, document)
    OS_FILESYSTEM.fsync_file(descriptor)


def _owner_diagnostics(holders: list[dict[str, object]]) -> str:
    if not holders:
        return "owner diagnostics unavailable"
    return json.dumps(
        {"holders": holders},
        sort_keys=True,
        separators=(",", ":"),
    )


def _pid_may_be_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _prune_lock_owners(
    holders: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [holder for holder in holders if _pid_may_be_alive(holder["pid"])]


def _open_lock_sidecar(sidecar: str, parent_descriptor: int) -> int:
    existing_flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return OS_FILESYSTEM.open(
            sidecar,
            _OPEN_LOCK | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
    except FileExistsError:
        return OS_FILESYSTEM.open(
            sidecar,
            existing_flags,
            dir_fd=parent_descriptor,
        )


@contextmanager
def _protected_lock(
    protected: Path,
    *,
    category: str,
    exclusive: bool,
    wait: bool = False,
) -> Iterator[None]:
    parent, protected_name = _path_parts(protected)
    parent_descriptor = _open_writable_parent(parent, OS_FILESYSTEM)
    lock_descriptor = -1
    diagnostic_descriptor = -1
    acquired = False
    protected_released = False
    owner_token = uuid.uuid4().hex
    try:
        sidecar = _lock_name(protected_name, category)
        lock_descriptor = _open_lock_sidecar(sidecar, parent_descriptor)
        diagnostic_descriptor = _open_lock_sidecar(
            _diagnostic_name(protected_name, category),
            parent_descriptor,
        )
        for descriptor in (lock_descriptor, diagnostic_descriptor):
            sidecar_stat = OS_FILESYSTEM.stat(descriptor)
            if not stat.S_ISREG(sidecar_stat.st_mode) or sidecar_stat.st_nlink != 1:
                raise SMLArtifactError(
                    "lock sidecars must be singly linked regular files"
                )
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        retry_delay = _LOCK_RETRY_INITIAL_SECONDS
        while not acquired:
            OS_FILESYSTEM.flock(diagnostic_descriptor, fcntl.LOCK_EX)
            conflict: _LockUnavailable | None = None
            try:
                stored_holders = _read_lock_owners(diagnostic_descriptor)
                holders = _prune_lock_owners(stored_holders)
                if holders != stored_holders:
                    _write_lock_owners(diagnostic_descriptor, holders)
                try:
                    OS_FILESYSTEM.flock(
                        lock_descriptor,
                        operation | fcntl.LOCK_NB,
                    )
                except OSError as error:
                    if error.errno not in {
                        errno.EACCES,
                        errno.EAGAIN,
                        errno.EWOULDBLOCK,
                    }:
                        raise
                    conflict = _LockUnavailable(
                        f"{category} lock for {protected} is held by "
                        f"{_owner_diagnostics(holders)}"
                    )
                else:
                    holders = [] if exclusive else holders
                    holders.append(
                        {
                            "pid": os.getpid(),
                            "protected_path": str(protected),
                            "token": owner_token,
                        }
                    )
                    _write_lock_owners(diagnostic_descriptor, holders)
                    OS_FILESYSTEM.fsync_directory(parent_descriptor)
                    acquired = True
            finally:
                OS_FILESYSTEM.flock(diagnostic_descriptor, fcntl.LOCK_UN)

            if conflict is not None:
                if not wait:
                    raise conflict from None
                time.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * 2,
                    _LOCK_RETRY_MAX_SECONDS,
                )
        yield
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError(
            f"could not manage {category} lock for {protected}"
        ) from error
    finally:
        if acquired and diagnostic_descriptor >= 0:
            diagnostic_locked = False
            try:
                OS_FILESYSTEM.flock(diagnostic_descriptor, fcntl.LOCK_EX)
                diagnostic_locked = True
                try:
                    remaining = [
                        holder
                        for holder in _prune_lock_owners(
                            _read_lock_owners(diagnostic_descriptor)
                        )
                        if holder["token"] != owner_token
                    ]
                    _write_lock_owners(diagnostic_descriptor, remaining)
                finally:
                    if lock_descriptor >= 0:
                        try:
                            OS_FILESYSTEM.flock(lock_descriptor, fcntl.LOCK_UN)
                            protected_released = True
                        except OSError:
                            try:
                                os.close(lock_descriptor)
                            finally:
                                lock_descriptor = -1
            except OSError:
                pass
            finally:
                if diagnostic_locked:
                    try:
                        OS_FILESYSTEM.flock(
                            diagnostic_descriptor,
                            fcntl.LOCK_UN,
                        )
                    except OSError:
                        pass
        if lock_descriptor >= 0:
            if not protected_released:
                try:
                    OS_FILESYSTEM.flock(lock_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lock_descriptor)
        if diagnostic_descriptor >= 0:
            os.close(diagnostic_descriptor)
        os.close(parent_descriptor)


def publication_lock(target: Path) -> Iterator[None]:
    return _protected_lock(target, category="publication", exclusive=True)


def run_writer_lock(run: Path) -> Iterator[None]:
    return _protected_lock(run, category="run-writer", exclusive=True)


def run_access_lock(run: Path, *, exclusive: bool) -> Iterator[None]:
    if not isinstance(exclusive, bool):
        raise TypeError("exclusive must be a bool")
    return _protected_lock(run, category="run-access", exclusive=exclusive)


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _safe_child_name(name: str) -> bool:
    return (
        bool(name) and name not in {".", ".."} and "/" not in name and "\0" not in name
    )


def _stat_if_present(
    fs: FilesystemOps,
    name: str,
    *,
    parent_descriptor: int,
) -> os.stat_result | None:
    try:
        return fs.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _require_named_directory_inode(
    fs: FilesystemOps,
    name: str,
    *,
    parent_descriptor: int,
    directory_descriptor: int,
    context: str,
) -> None:
    try:
        named_directory = fs.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        opened_directory = fs.stat(directory_descriptor)
    except OSError as error:
        raise SMLArtifactError(
            f"{context} is no longer bound to the verified directory inode"
        ) from error
    if not stat.S_ISDIR(named_directory.st_mode) or not _same_inode(
        named_directory,
        opened_directory,
    ):
        raise SMLArtifactError(
            f"{context} was swapped from the verified directory inode"
        )


def _temporary_prefix(target_name: str) -> str:
    return f".sml-tmp-{_name_digest(target_name)}-"


def _is_owned_temporary_name(target_name: str, candidate_name: str) -> bool:
    prefix = _temporary_prefix(target_name)
    return (
        candidate_name.startswith(prefix)
        and _TEMPORARY_SUFFIX_PATTERN.fullmatch(candidate_name[len(prefix) :])
        is not None
    )


def _opened_entry(
    fs: FilesystemOps,
    name: str,
    *,
    parent_descriptor: int,
    flags: int,
) -> tuple[int, os.stat_result]:
    descriptor = fs.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened_stat = fs.stat(descriptor)
        named_stat = fs.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not _same_inode(opened_stat, named_stat):
            raise SMLArtifactError(
                f"cleanup revalidation detected an entry swap: {name}"
            )
        return descriptor, opened_stat
    except BaseException:
        os.close(descriptor)
        raise


def _delete_directory_contents(fs: FilesystemOps, directory_descriptor: int) -> None:
    for name in sorted(fs.listdir(directory_descriptor)):
        if not _safe_child_name(name):
            raise SMLArtifactError("cleanup found an invalid directory entry name")
        entry_stat = fs.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISREG(entry_stat.st_mode):
            if entry_stat.st_nlink != 1:
                raise SMLArtifactError(f"cleanup rejects hard-linked file: {name}")
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_PAYLOAD,
            )
            try:
                if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                    raise SMLArtifactError(f"cleanup rejects non-regular file: {name}")
                current_stat = fs.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(opened_stat, current_stat):
                    raise SMLArtifactError(
                        f"cleanup revalidation detected a file swap: {name}"
                    )
                fs.unlink(name, dir_fd=directory_descriptor)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(entry_stat.st_mode):
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_DIRECTORY,
            )
            try:
                if not stat.S_ISDIR(opened_stat.st_mode):
                    raise SMLArtifactError(f"cleanup rejects non-directory: {name}")
                _delete_directory_contents(fs, descriptor)
                current_stat = fs.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(opened_stat, current_stat):
                    raise SMLArtifactError(
                        f"cleanup revalidation detected a directory swap: {name}"
                    )
                fs.rmdir(name, dir_fd=directory_descriptor)
            finally:
                os.close(descriptor)
        else:
            raise SMLArtifactError(f"cleanup rejects symlink or special file: {name}")


def _cleanup_temporary(
    fs: FilesystemOps,
    parent_descriptor: int,
    target_name: str,
    candidate_name: str,
) -> None:
    if not _is_owned_temporary_name(target_name, candidate_name):
        return
    initial_stat = _stat_if_present(
        fs, candidate_name, parent_descriptor=parent_descriptor
    )
    if initial_stat is None:
        return
    if not stat.S_ISDIR(initial_stat.st_mode):
        raise SMLArtifactError(
            f"cleanup rejects a non-directory temporary candidate: {candidate_name}"
        )
    descriptor = -1
    try:
        descriptor, opened_stat = _opened_entry(
            fs,
            candidate_name,
            parent_descriptor=parent_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        if not _same_inode(initial_stat, opened_stat):
            raise SMLArtifactError(
                f"cleanup revalidation detected a candidate swap: {candidate_name}"
            )
        _delete_directory_contents(fs, descriptor)
        current_stat = fs.stat(
            candidate_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not _is_owned_temporary_name(target_name, candidate_name)
            or not stat.S_ISDIR(current_stat.st_mode)
            or not _same_inode(opened_stat, current_stat)
        ):
            raise SMLArtifactError(
                f"cleanup revalidation rejected candidate: {candidate_name}"
            )
        fs.rmdir(candidate_name, dir_fd=parent_descriptor)
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError(
            f"cleanup rejected unsafe temporary candidate: {candidate_name}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _cleanup_stale_temporaries(
    fs: FilesystemOps,
    parent_descriptor: int,
    target_name: str,
) -> None:
    for candidate_name in sorted(fs.listdir(parent_descriptor)):
        if _is_owned_temporary_name(target_name, candidate_name):
            _cleanup_temporary(
                fs,
                parent_descriptor,
                target_name,
                candidate_name,
            )


def _payload_references(value: object) -> Iterator[PayloadRef]:
    if isinstance(value, PayloadRef):
        yield value
        return
    if isinstance(value, ArrayPayloadRef):
        yield value.payload
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _payload_references(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _payload_references(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _payload_references(item)


def _expected_closed_world(
    references: Sequence[PayloadRef],
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    expected_files: set[tuple[str, ...]] = set()
    expected_directories: set[tuple[str, ...]] = set()
    for reference in references:
        components = parse_logical_path(reference.logical_path)
        expected_files.add(components)
        expected_directories.update(
            components[:depth] for depth in range(1, len(components))
        )
    if expected_files & expected_directories:
        raise SMLArtifactError(
            "closed-world payload paths cannot be both files and directories"
        )
    return expected_files, expected_directories


def _scan_closed_world(
    fs: FilesystemOps,
    directory_descriptor: int,
    *,
    prefix: tuple[str, ...] = (),
    files: set[tuple[str, ...]] | None = None,
    directories: set[tuple[str, ...]] | None = None,
    collision_paths: dict[tuple[str, ...], tuple[str, ...]] | None = None,
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    if files is None:
        files = set()
    if directories is None:
        directories = set()
    if collision_paths is None:
        collision_paths = {}
    for name in sorted(fs.listdir(directory_descriptor)):
        if not _safe_child_name(name):
            raise SMLArtifactError("closed-world tree contains an invalid entry name")
        normalized_name = parse_logical_path(name)
        if len(normalized_name) != 1:
            raise SMLArtifactError("closed-world entry must be one direct child")
        logical_path = prefix + normalized_name
        collision_key = tuple(component.casefold() for component in logical_path)
        previous = collision_paths.get(collision_key)
        if previous is not None:
            raise SMLArtifactError(
                "closed-world tree contains a normalized or case-folded collision: "
                f"{'/'.join(previous)!r} and {'/'.join(logical_path)!r}"
            )
        collision_paths[collision_key] = logical_path
        entry_stat = fs.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISREG(entry_stat.st_mode):
            if entry_stat.st_nlink != 1:
                raise SMLArtifactError(
                    f"closed-world payload is hard-linked: {'/'.join(logical_path)}"
                )
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_PAYLOAD,
            )
            try:
                if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                    raise SMLArtifactError(
                        f"closed-world payload is not a regular file: "
                        f"{'/'.join(logical_path)}"
                    )
                files.add(logical_path)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(entry_stat.st_mode):
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_DIRECTORY,
            )
            try:
                if not stat.S_ISDIR(opened_stat.st_mode):
                    raise SMLArtifactError(
                        f"closed-world component is not a directory: "
                        f"{'/'.join(logical_path)}"
                    )
                directories.add(logical_path)
                _scan_closed_world(
                    fs,
                    descriptor,
                    prefix=logical_path,
                    files=files,
                    directories=directories,
                    collision_paths=collision_paths,
                )
            finally:
                os.close(descriptor)
        else:
            raise SMLArtifactError(
                f"closed-world tree rejects symlink or special file: "
                f"{'/'.join(logical_path)}"
            )
    return files, directories


def _format_logical_paths(paths: set[tuple[str, ...]]) -> list[str]:
    return ["/".join(path) for path in sorted(paths)]


def _verify_pretraining_nested_tokenizer(
    fs: FilesystemOps,
    temporary_descriptor: int,
    manifest: PretrainingDataManifest,
) -> None:
    expected_model_path = "tokenizer/tokenizer.model"
    expected_vocab_path = "tokenizer/tokenizer.vocab"
    if manifest.tokenizer_model.logical_path != expected_model_path:
        raise SMLArtifactError(
            "nested tokenizer model must be bound to tokenizer/tokenizer.model"
        )
    if manifest.tokenizer_vocab.logical_path != expected_vocab_path:
        raise SMLArtifactError(
            "nested tokenizer vocab must be bound to tokenizer/tokenizer.vocab"
        )

    descriptor, _opened_stat = _opened_entry(
        fs,
        "tokenizer",
        parent_descriptor=temporary_descriptor,
        flags=_OPEN_DIRECTORY,
    )
    try:
        try:
            nested = _read_manifest_from_descriptor(
                descriptor,
                TokenizerManifest,
                VerificationLevel.FULL,
                context="nested tokenizer manifest",
            )
        except SMLArtifactError as error:
            raise SMLArtifactError(
                f"invalid nested tokenizer manifest: {error}"
            ) from error
        _require_named_directory_inode(
            fs,
            "tokenizer",
            parent_descriptor=temporary_descriptor,
            directory_descriptor=descriptor,
            context="nested tokenizer directory",
        )
    finally:
        os.close(descriptor)

    if nested.model.logical_path != "tokenizer.model":
        raise SMLArtifactError(
            "nested tokenizer model logical path must be tokenizer.model"
        )
    if nested.vocab.logical_path != "tokenizer.vocab":
        raise SMLArtifactError(
            "nested tokenizer vocab logical path must be tokenizer.vocab"
        )
    if nested.identity != manifest.tokenizer_identity:
        raise SMLArtifactError(
            "nested tokenizer identity does not match pretraining manifest"
        )
    if (
        nested.model.identity != manifest.tokenizer_model.identity
        or nested.model.byte_size != manifest.tokenizer_model.byte_size
    ):
        raise SMLArtifactError(
            "nested tokenizer model does not match pretraining manifest"
        )
    if (
        nested.vocab.identity != manifest.tokenizer_vocab.identity
        or nested.vocab.byte_size != manifest.tokenizer_vocab.byte_size
    ):
        raise SMLArtifactError(
            "nested tokenizer vocab does not match pretraining manifest"
        )


def _verify_closed_world(
    fs: FilesystemOps,
    temporary_descriptor: int,
    manifest: object,
    *,
    manifest_present: bool,
    full: bool = True,
) -> None:
    references = tuple(_payload_references(manifest))
    expected_files, expected_directories = _expected_closed_world(references)
    if isinstance(manifest, PretrainingDataManifest):
        _verify_pretraining_nested_tokenizer(fs, temporary_descriptor, manifest)
        expected_files.add(("tokenizer", TokenizerManifest.MANIFEST_FILENAME))
    if manifest_present:
        expected_files.add(parse_logical_path(manifest.MANIFEST_FILENAME))
    actual_files, actual_directories = _scan_closed_world(
        fs,
        temporary_descriptor,
    )
    if actual_files != expected_files or actual_directories != expected_directories:
        raise SMLArtifactError(
            "artifact closed-world mismatch: "
            f"unreferenced files={_format_logical_paths(actual_files - expected_files)}, "
            f"missing files={_format_logical_paths(expected_files - actual_files)}, "
            "unreferenced directories="
            f"{_format_logical_paths(actual_directories - expected_directories)}, "
            "missing directories="
            f"{_format_logical_paths(expected_directories - actual_directories)}"
        )

    with ArtifactRoot(os.dup(temporary_descriptor), local_apfs=True) as artifact_root:
        artifact_root.verify_payloads(references, full=full)
        if manifest_present:
            with artifact_root.open_payload(
                manifest.MANIFEST_FILENAME
            ) as manifest_file:
                if manifest_file.read() != canonical_json_bytes(manifest):
                    raise SMLArtifactError(
                        "publisher-owned manifest changed before immutable commit"
                    )
    if isinstance(manifest, PretrainingDataManifest):
        _verify_pretraining_nested_tokenizer(fs, temporary_descriptor, manifest)


def _make_payload_tree_durable(fs: FilesystemOps, directory_descriptor: int) -> None:
    for name in sorted(fs.listdir(directory_descriptor)):
        if not _safe_child_name(name):
            raise SMLArtifactError("payload tree contains an invalid entry name")
        entry_stat = fs.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISREG(entry_stat.st_mode):
            if entry_stat.st_nlink != 1:
                raise SMLArtifactError(
                    f"payload link count must be exactly one: {name}"
                )
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_PAYLOAD,
            )
            try:
                if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                    raise SMLArtifactError(f"payload is not a regular file: {name}")
                fs.fsync_file(descriptor)
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(entry_stat.st_mode):
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_DIRECTORY,
            )
            try:
                if not stat.S_ISDIR(opened_stat.st_mode):
                    raise SMLArtifactError(
                        f"payload component is not a directory: {name}"
                    )
                _make_payload_tree_durable(fs, descriptor)
            finally:
                os.close(descriptor)
        else:
            raise SMLArtifactError(
                f"payload tree rejects symlink or special file: {name}"
            )
    fs.fsync_directory(directory_descriptor)


def _validate_builder_result(
    manifest: object,
    temporary_descriptor: int,
    fs: FilesystemOps,
) -> object:
    if type(manifest) not in _MANIFEST_TYPES:
        raise SMLArtifactError("builder returned an unsupported version-1 manifest")
    if manifest.identity != manifest.recompute_identity():
        raise SMLArtifactError("builder returned a manifest identity mismatch")
    if (
        _stat_if_present(
            fs,
            manifest.MANIFEST_FILENAME,
            parent_descriptor=temporary_descriptor,
        )
        is not None
    ):
        raise SMLArtifactError("builder must not write the manifest")
    _verify_closed_world(
        fs,
        temporary_descriptor,
        manifest,
        manifest_present=False,
    )
    return manifest


def _write_manifest_last(
    fs: FilesystemOps,
    temporary_descriptor: int,
    manifest: object,
) -> None:
    descriptor = -1
    try:
        descriptor = fs.open(
            manifest.MANIFEST_FILENAME,
            _OPEN_MANIFEST,
            0o600,
            dir_fd=temporary_descriptor,
        )
        fs.write_all(descriptor, canonical_json_bytes(manifest))
        fs.fsync_file(descriptor)
    except SMLArtifactError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise SMLArtifactError("could not write canonical manifest last") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _accept_existing[M](
    target: Path,
    manifest: M,
    *,
    fs: FilesystemOps,
    parent_descriptor: int,
    target_name: str,
) -> Published[M]:
    target_descriptor = -1
    try:
        target_descriptor, opened_stat = _opened_entry(
            fs,
            target_name,
            parent_descriptor=parent_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SMLArtifactError("existing publication target is not a directory")
        existing_manifest = _read_manifest_from_descriptor(
            target_descriptor,
            type(manifest),
            VerificationLevel.FULL,
            context=str(target),
        )
        _verify_closed_world(
            fs,
            target_descriptor,
            existing_manifest,
            manifest_present=True,
        )
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=target_descriptor,
            context="existing publication target after full verification",
        )
    except (OSError, SMLArtifactError) as error:
        raise SMLArtifactError(
            f"existing target failed full verification: {target}"
        ) from error
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
    if type(existing_manifest) is not type(manifest):
        raise SMLArtifactError(f"existing target manifest type collision: {target}")
    if existing_manifest.identity != manifest.identity:
        raise SMLArtifactError(
            f"existing target has a different identity collision: {target}"
        )
    return Published(
        path=target,
        manifest=existing_manifest,
        verification=VerificationLevel.FULL,
    )


def _publish_with_lock[M](
    target: Path,
    build: Callable[[Path], M],
    *,
    fs: FilesystemOps,
) -> Published[M]:
    parent, target_name = _path_parts(target)
    parent_descriptor = _open_writable_parent(parent, fs)
    temporary_name = f"{_temporary_prefix(target_name)}{uuid.uuid4().hex}"
    temporary_descriptor = -1
    committed = False
    created = False
    try:
        _cleanup_stale_temporaries(fs, parent_descriptor, target_name)
        fs.mkdir(temporary_name, 0o700, dir_fd=parent_descriptor)
        created = True
        temporary_descriptor = fs.open(
            temporary_name,
            _OPEN_DIRECTORY,
            dir_fd=parent_descriptor,
        )
        temporary_path = parent / temporary_name
        manifest = _validate_builder_result(
            build(temporary_path), temporary_descriptor, fs
        )
        _make_payload_tree_durable(fs, temporary_descriptor)
        _write_manifest_last(fs, temporary_descriptor, manifest)
        fs.fsync_directory(temporary_descriptor)

        existing = _stat_if_present(
            fs, target_name, parent_descriptor=parent_descriptor
        )
        if existing is not None:
            _verify_closed_world(
                fs,
                temporary_descriptor,
                manifest,
                manifest_present=True,
            )
            return _accept_existing(
                target,
                manifest,
                fs=fs,
                parent_descriptor=parent_descriptor,
                target_name=target_name,
            )

        _verify_closed_world(
            fs,
            temporary_descriptor,
            manifest,
            manifest_present=True,
        )
        _require_named_directory_inode(
            fs,
            temporary_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=temporary_descriptor,
            context="temporary publication directory",
        )
        try:
            fs.rename(
                temporary_name,
                target_name,
                source_dir_fd=parent_descriptor,
                destination_dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise SMLArtifactError(
                f"publication collision while committing immutable target: {target}"
            ) from error
        committed = True
        fs.fsync_directory(parent_descriptor)
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=temporary_descriptor,
            context="committed publication target",
        )
        _verify_closed_world(
            fs,
            temporary_descriptor,
            manifest,
            manifest_present=True,
        )
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=temporary_descriptor,
            context="committed publication target after full verification",
        )
        return Published(
            path=target,
            manifest=manifest,
            verification=VerificationLevel.FULL,
        )
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError(f"immutable publication failed for {target}") from error
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        try:
            if created and not committed:
                _cleanup_temporary(
                    fs,
                    parent_descriptor,
                    target_name,
                    temporary_name,
                )
        finally:
            os.close(parent_descriptor)


def publish_immutable_bundle[M](
    target: Path,
    build: Callable[[Path], M],
    *,
    fs: FilesystemOps = OS_FILESYSTEM,
) -> Published[M]:
    if not callable(build):
        raise TypeError("build must be callable")
    if not isinstance(fs, FilesystemOps):
        raise TypeError("fs must implement FilesystemOps")
    with _protected_lock(
        target,
        category="publication",
        exclusive=True,
        wait=True,
    ):
        return _publish_with_lock(target, build, fs=fs)


def _validate_private_run(
    run: Path,
    run_descriptor: int,
    expected: RunManifest,
    *,
    fs: FilesystemOps,
) -> RunManifest:
    expected_entries = {"run.json", "tokenizer", "latest.json", "checkpoints"}
    actual_entries = set(fs.listdir(run_descriptor))
    if actual_entries != expected_entries:
        raise SMLArtifactError(
            "private run has an invalid closed-world layout: "
            f"missing={sorted(expected_entries - actual_entries)}, "
            f"unexpected={sorted(actual_entries - expected_entries)}"
        )
    manifest = _read_manifest_from_descriptor(
        run_descriptor,
        RunManifest,
        VerificationLevel.FULL,
        context=str(run / RunManifest.MANIFEST_FILENAME),
    )
    if manifest != expected:
        raise SMLArtifactError("private run manifest changed during creation")

    tokenizer_descriptor = -1
    checkpoints_descriptor = -1
    try:
        tokenizer_descriptor, tokenizer_stat = _opened_entry(
            fs,
            "tokenizer",
            parent_descriptor=run_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        if not stat.S_ISDIR(tokenizer_stat.st_mode):
            raise SMLArtifactError("run tokenizer is not a directory")
        tokenizer = _read_manifest_from_descriptor(
            tokenizer_descriptor,
            TokenizerManifest,
            VerificationLevel.FULL,
            context=str(run / "tokenizer"),
        )
        _verify_closed_world(
            fs,
            tokenizer_descriptor,
            tokenizer,
            manifest_present=True,
            full=True,
        )
        if tokenizer.identity != manifest.tokenizer_identity:
            raise SMLArtifactError("run tokenizer identity does not match run.json")

        checkpoints_descriptor = _open_checkpoints_directory(
            run,
            fs,
            run_descriptor,
            writable=True,
        )
        latest = _recover_latest_open(
            run,
            manifest,
            run_descriptor,
            checkpoints_descriptor,
            writable=False,
            verification=VerificationLevel.FULL,
            fs=fs,
            allow_empty=False,
        )
        if latest is None or latest.step != 0:
            raise SMLArtifactError("fresh run must publish checkpoint step zero")
        if latest.latest_recovered:
            raise SMLArtifactError("fresh run latest index does not name step zero")
        if set(fs.listdir(checkpoints_descriptor)) != {_step_name(0)}:
            raise SMLArtifactError("fresh run must contain only checkpoint step zero")
        return manifest
    finally:
        if checkpoints_descriptor >= 0:
            os.close(checkpoints_descriptor)
        if tokenizer_descriptor >= 0:
            os.close(tokenizer_descriptor)


def publish_run(
    target: Path,
    build: Callable[[Path], RunManifest],
    *,
    fs: FilesystemOps = OS_FILESYSTEM,
) -> Published[RunManifest]:
    """Atomically create a new run containing its complete step-zero state.

    The caller owns the nonblocking run-writer lock for ``target``. Unlike an
    immutable bundle publication, an existing target is always an error.
    """
    if not isinstance(target, Path):
        raise TypeError("target must be a Path")
    if not callable(build):
        raise TypeError("build must be callable")
    if not isinstance(fs, FilesystemOps):
        raise TypeError("fs must implement FilesystemOps")

    parent, target_name = _path_parts(target)
    parent_descriptor = _open_writable_parent(parent, fs)
    temporary_name = f"{_temporary_prefix(target_name)}{uuid.uuid4().hex}"
    temporary_descriptor = -1
    created = False
    committed = False
    try:
        _cleanup_stale_temporaries(fs, parent_descriptor, target_name)
        if _stat_if_present(fs, target_name, parent_descriptor=parent_descriptor):
            raise SMLArtifactError(f"fresh run target already exists: {target}")
        fs.mkdir(temporary_name, 0o700, dir_fd=parent_descriptor)
        created = True
        temporary_descriptor = fs.open(
            temporary_name,
            _OPEN_DIRECTORY,
            dir_fd=parent_descriptor,
        )
        temporary_path = parent / temporary_name
        manifest = build(temporary_path)
        if not isinstance(manifest, RunManifest):
            raise SMLArtifactError("run builder returned the wrong manifest type")
        if manifest.identity != manifest.recompute_identity():
            raise SMLArtifactError("run builder returned a manifest identity mismatch")
        validated = _validate_private_run(
            temporary_path,
            temporary_descriptor,
            manifest,
            fs=fs,
        )
        _make_payload_tree_durable(fs, temporary_descriptor)
        _require_named_directory_inode(
            fs,
            temporary_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=temporary_descriptor,
            context="private run directory",
        )
        if _stat_if_present(fs, target_name, parent_descriptor=parent_descriptor):
            raise SMLArtifactError(f"fresh run target appeared concurrently: {target}")
        try:
            fs.rename(
                temporary_name,
                target_name,
                source_dir_fd=parent_descriptor,
                destination_dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise SMLArtifactError(
                f"fresh run target appeared concurrently: {target}"
            ) from error
        committed = True
        fs.fsync_directory(parent_descriptor)
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=parent_descriptor,
            directory_descriptor=temporary_descriptor,
            context="committed fresh run",
        )
        _validate_private_run(target, temporary_descriptor, validated, fs=fs)
        return Published(
            path=target,
            manifest=validated,
            verification=VerificationLevel.FULL,
        )
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError(f"fresh run publication failed: {target}") from error
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        try:
            if created and not committed:
                _cleanup_temporary(
                    fs,
                    parent_descriptor,
                    target_name,
                    temporary_name,
                )
        finally:
            os.close(parent_descriptor)


def _require_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be nonnegative")
    return step


def _step_name(step: int) -> str:
    return f"step-{_require_step(step):09d}"


def _parse_step_name(name: str) -> int | None:
    if not name.startswith("step-"):
        return None
    digits = name.removeprefix("step-")
    if not digits or not digits.isascii() or not digits.isdecimal():
        return None
    step = int(digits)
    return step if name == _step_name(step) else None


def _is_temporary_step_name(name: str) -> bool:
    return _TEMPORARY_STEP_PATTERN.fullmatch(name) is not None


def _temporary_step_name() -> str:
    return f".sml-tmp-step-{uuid.uuid4().hex}"


def _open_directory(
    path: Path,
    fs: FilesystemOps,
    *,
    writable: bool,
    context: str,
) -> int:
    descriptor = -1
    try:
        descriptor = fs.open(path, _OPEN_DIRECTORY)
        opened_stat = fs.stat(descriptor)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SMLArtifactError(f"{context} is not a directory")
        if writable and not _descriptor_is_local_apfs(descriptor):
            raise SMLArtifactError(f"writable {context} requires local APFS")
        return descriptor
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, (SMLArtifactError, TypeError, ValueError)):
            raise
        if isinstance(error, OSError):
            raise SMLArtifactError(
                f"could not open {context} with no-follow semantics: {path}"
            ) from error
        raise


def _read_manifest_from_descriptor[M](
    descriptor: int,
    manifest_type: type[M],
    verification: VerificationLevel,
    *,
    context: str,
) -> M:
    try:
        local_apfs = _descriptor_is_local_apfs(descriptor)
        with ArtifactRoot(os.dup(descriptor), local_apfs=local_apfs) as artifact_root:
            with artifact_root.open_payload(
                manifest_type.MANIFEST_FILENAME
            ) as manifest_file:
                manifest_bytes = manifest_file.read()
            raw = json.loads(
                manifest_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_json_object_no_duplicates,
            )
            manifest = _parse_manifest(raw, manifest_type)
            if manifest.recompute_identity() != manifest.identity:
                raise SMLArtifactError(f"manifest identity mismatch in {context}")
            if manifest_bytes != canonical_json_bytes(manifest):
                raise SMLArtifactError(f"noncanonical manifest bytes in {context}")
            artifact_root.verify_payloads(
                tuple(_payload_references(manifest)),
                full=verification is VerificationLevel.FULL,
            )
            return cast(M, manifest)
    except SMLArtifactError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        raise SMLArtifactError(f"invalid manifest in {context}: {error}") from error


def _validate_checkpoint_owner(
    run_manifest: RunManifest,
    manifest: CheckpointManifest,
    *,
    expected_step: int,
) -> None:
    if manifest.owning_run_identity != run_manifest.identity:
        raise SMLArtifactError("checkpoint belongs to a different run")
    if manifest.checkpoint_kind != run_manifest.run_kind:
        raise SMLArtifactError("checkpoint kind does not match the owning run")
    if manifest.step != expected_step:
        raise SMLArtifactError(
            "checkpoint logical step does not match its canonical directory"
        )
    if manifest.scalar_state.logical_path != "state.json":
        raise SMLArtifactError(
            "checkpoint scalar state logical path must be exactly 'state.json'"
        )


def _resolved_step(
    run_path: Path,
    run_manifest: RunManifest,
    checkpoint_manifest: CheckpointManifest,
    verification: VerificationLevel,
    *,
    latest_recovered: bool,
    latest_repair_persisted: bool,
) -> ResolvedStep:
    return ResolvedStep(
        run=run_manifest,
        checkpoint=checkpoint_manifest,
        step_directory=run_path / "checkpoints" / _step_name(checkpoint_manifest.step),
        run_step_identity=structured_identity(
            "sml-run-step-v1",
            {
                "run_identity": run_manifest.identity,
                "checkpoint_identity": checkpoint_manifest.identity,
            },
        ),
        verification=verification,
        latest_recovered=latest_recovered,
        latest_repair_persisted=latest_repair_persisted,
    )


def _open_checkpoints_directory(
    run: Path,
    fs: FilesystemOps,
    run_descriptor: int,
    *,
    writable: bool,
) -> int:
    descriptor = -1
    try:
        descriptor = fs.open(
            "checkpoints",
            _OPEN_DIRECTORY,
            dir_fd=run_descriptor,
        )
        opened_stat = fs.stat(descriptor)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SMLArtifactError("checkpoints entry is not a directory")
        if writable and not _descriptor_is_local_apfs(descriptor):
            raise SMLArtifactError("writable checkpoints directory requires local APFS")
        _require_named_directory_inode(
            fs,
            "checkpoints",
            parent_descriptor=run_descriptor,
            directory_descriptor=descriptor,
            context="checkpoints directory",
        )
        return descriptor
    except BaseException as error:
        if descriptor >= 0:
            os.close(descriptor)
        if isinstance(error, SMLArtifactError):
            raise
        if isinstance(error, OSError):
            raise SMLArtifactError(
                f"could not open checkpoints directory for run: {run}"
            ) from error
        raise


def _open_verified_step_from_descriptor(
    run: Path,
    run_manifest: RunManifest,
    checkpoints_descriptor: int,
    *,
    step: int,
    verification: VerificationLevel,
    fs: FilesystemOps,
) -> _OwnedStep:
    name = _step_name(step)
    descriptor = -1
    try:
        descriptor, opened_stat = _opened_entry(
            fs,
            name,
            parent_descriptor=checkpoints_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SMLArtifactError(f"checkpoint candidate is not a directory: {name}")
        manifest = _read_manifest_from_descriptor(
            descriptor,
            CheckpointManifest,
            verification,
            context=str(run / "checkpoints" / name),
        )
        _validate_checkpoint_owner(
            run_manifest,
            manifest,
            expected_step=step,
        )
        _verify_closed_world(
            fs,
            descriptor,
            manifest,
            manifest_present=True,
            full=verification is VerificationLevel.FULL,
        )
        _require_named_directory_inode(
            fs,
            name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=descriptor,
            context=f"checkpoint step {step}",
        )
        owned = _OwnedStep(
            resolved=_resolved_step(
                run,
                run_manifest,
                manifest,
                verification,
                latest_recovered=False,
                latest_repair_persisted=False,
            ),
            name=name,
            descriptor=descriptor,
            opened_stat=opened_stat,
        )
        descriptor = -1
        return owned
    except FileNotFoundError as error:
        raise SMLArtifactError(f"checkpoint step does not exist: {step}") from error
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError(f"could not resolve checkpoint step: {step}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _resolve_step_from_descriptor(
    run: Path,
    run_manifest: RunManifest,
    checkpoints_descriptor: int,
    *,
    step: int,
    verification: VerificationLevel,
    fs: FilesystemOps,
) -> ResolvedStep:
    owned = _open_verified_step_from_descriptor(
        run,
        run_manifest,
        checkpoints_descriptor,
        step=step,
        verification=verification,
        fs=fs,
    )
    try:
        return owned.resolved
    finally:
        owned.close()


def resolve_exact_step(
    run: Path,
    *,
    step: int,
    verification: VerificationLevel,
) -> ResolvedStep:
    requested_step = _require_step(step)
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    run_descriptor = _open_directory(
        run,
        OS_FILESYSTEM,
        writable=False,
        context="run directory",
    )
    checkpoints_descriptor = -1
    try:
        run_manifest = _read_manifest_from_descriptor(
            run_descriptor,
            RunManifest,
            verification,
            context=str(run / RunManifest.MANIFEST_FILENAME),
        )
        checkpoints_descriptor = _open_checkpoints_directory(
            run,
            OS_FILESYSTEM,
            run_descriptor,
            writable=False,
        )
        return _resolve_step_from_descriptor(
            run,
            run_manifest,
            checkpoints_descriptor,
            step=requested_step,
            verification=verification,
            fs=OS_FILESYSTEM,
        )
    finally:
        if checkpoints_descriptor >= 0:
            os.close(checkpoints_descriptor)
        os.close(run_descriptor)


def _latest_index_for(resolved: ResolvedStep) -> LatestIndex:
    index = LatestIndex(
        kind="latest-index",
        version=1,
        identity="sha256:" + "0" * 64,
        owning_run_identity=resolved.run.identity,
        step=resolved.step,
        checkpoint_identity=resolved.checkpoint.identity,
    )
    return dataclasses.replace(index, identity=index.recompute_identity())


def _persist_latest_index(
    run_descriptor: int,
    resolved: ResolvedStep,
    fs: FilesystemOps,
) -> None:
    temporary_name = f".sml-tmp-latest-{uuid.uuid4().hex}"
    descriptor = -1
    replaced = False
    try:
        descriptor = fs.open(
            temporary_name,
            _OPEN_MANIFEST,
            0o600,
            dir_fd=run_descriptor,
        )
        fs.write_all(descriptor, canonical_json_bytes(_latest_index_for(resolved)))
        fs.fsync_file(descriptor)
        os.close(descriptor)
        descriptor = -1
        fs.replace(
            temporary_name,
            LatestIndex.MANIFEST_FILENAME,
            source_dir_fd=run_descriptor,
            destination_dir_fd=run_descriptor,
        )
        replaced = True
        fs.fsync_directory(run_descriptor)
    except SMLArtifactError:
        raise
    except OSError as error:
        raise SMLArtifactError("could not durably publish latest index") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not replaced:
            try:
                fs.unlink(temporary_name, dir_fd=run_descriptor)
            except FileNotFoundError:
                pass


def _stored_latest(
    run: Path,
    run_manifest: RunManifest,
    run_descriptor: int,
    checkpoints_descriptor: int,
    verification: VerificationLevel,
    fs: FilesystemOps,
) -> tuple[LatestIndex | None, ResolvedStep | None]:
    try:
        index = _read_manifest_from_descriptor(
            run_descriptor,
            LatestIndex,
            VerificationLevel.MANIFEST_TRUSTED,
            context=str(run / LatestIndex.MANIFEST_FILENAME),
        )
    except SMLArtifactError:
        return None, None
    if index.owning_run_identity != run_manifest.identity:
        return None, None
    try:
        pointed = _resolve_step_from_descriptor(
            run,
            run_manifest,
            checkpoints_descriptor,
            step=index.step,
            verification=verification,
            fs=fs,
        )
    except SMLArtifactError:
        return None, None
    if pointed.checkpoint.identity != index.checkpoint_identity:
        return None, None
    return index, pointed


def _malformed_candidate_could_be_newer(name: str, lower_bound: int | None) -> bool:
    if lower_bound is None:
        return True
    if not name.startswith("step-"):
        return True
    digits = name.removeprefix("step-")
    if not digits or not digits.isascii() or not digits.isdecimal():
        return True
    return int(digits) > lower_bound


def _scan_checkpoint_candidates(
    run: Path,
    run_manifest: RunManifest,
    checkpoints_descriptor: int,
    *,
    lower_bound: int | None,
    verification: VerificationLevel,
    fs: FilesystemOps,
) -> list[ResolvedStep]:
    candidates: list[ResolvedStep] = []
    for name in sorted(fs.listdir(checkpoints_descriptor)):
        if _is_temporary_step_name(name):
            continue
        step = _parse_step_name(name)
        if step is None:
            if _malformed_candidate_could_be_newer(name, lower_bound):
                raise SMLArtifactError(
                    f"malformed required checkpoint candidate: {name}"
                )
            continue
        if lower_bound is not None and step <= lower_bound:
            continue
        candidates.append(
            _resolve_step_from_descriptor(
                run,
                run_manifest,
                checkpoints_descriptor,
                step=step,
                verification=verification,
                fs=fs,
            )
        )
    return candidates


def _recover_latest_open(
    run: Path,
    run_manifest: RunManifest,
    run_descriptor: int,
    checkpoints_descriptor: int,
    *,
    writable: bool,
    verification: VerificationLevel,
    fs: FilesystemOps,
    allow_empty: bool,
) -> ResolvedStep | None:
    stored_index, pointed = _stored_latest(
        run,
        run_manifest,
        run_descriptor,
        checkpoints_descriptor,
        verification,
        fs,
    )
    lower_bound = pointed.step if pointed is not None else None
    candidates = _scan_checkpoint_candidates(
        run,
        run_manifest,
        checkpoints_descriptor,
        lower_bound=lower_bound,
        verification=verification,
        fs=fs,
    )
    available = ([pointed] if pointed is not None else []) + candidates
    if not available:
        if allow_empty:
            return None
        raise SMLArtifactError("run has no published checkpoints")
    selected = max(available, key=lambda candidate: candidate.step)
    exact_stored_match = (
        stored_index is not None
        and stored_index.step == selected.step
        and stored_index.checkpoint_identity == selected.checkpoint.identity
    )
    recovered = not exact_stored_match
    selected = dataclasses.replace(
        selected,
        latest_recovered=recovered,
        latest_repair_persisted=False,
    )
    if writable and recovered:
        _persist_latest_index(run_descriptor, selected, fs)
        selected = dataclasses.replace(
            selected,
            latest_repair_persisted=True,
        )
    return selected


def recover_latest_index(
    run: Path,
    *,
    writable: bool,
    verification: VerificationLevel,
    fs: FilesystemOps = OS_FILESYSTEM,
) -> ResolvedStep:
    if not isinstance(run, Path):
        raise TypeError("run must be a Path")
    if not isinstance(writable, bool):
        raise TypeError("writable must be a bool")
    if not isinstance(verification, VerificationLevel):
        raise TypeError("verification must be a VerificationLevel")
    if not isinstance(fs, FilesystemOps):
        raise TypeError("fs must implement FilesystemOps")
    if writable and verification is not VerificationLevel.FULL:
        raise SMLArtifactError("writable latest recovery requires FULL verification")
    run_descriptor = _open_directory(
        run,
        fs,
        writable=writable,
        context="run directory",
    )
    checkpoints_descriptor = -1
    try:
        run_manifest = _read_manifest_from_descriptor(
            run_descriptor,
            RunManifest,
            verification,
            context=str(run / RunManifest.MANIFEST_FILENAME),
        )
        checkpoints_descriptor = _open_checkpoints_directory(
            run,
            fs,
            run_descriptor,
            writable=writable,
        )
        resolved = _recover_latest_open(
            run,
            run_manifest,
            run_descriptor,
            checkpoints_descriptor,
            writable=writable,
            verification=verification,
            fs=fs,
            allow_empty=False,
        )
        if resolved is None:
            raise SMLArtifactError("run has no published checkpoints")
        return resolved
    finally:
        if checkpoints_descriptor >= 0:
            os.close(checkpoints_descriptor)
        os.close(run_descriptor)


def resolve_latest_step(
    run: Path,
    *,
    writable: bool,
    verification: VerificationLevel,
) -> ResolvedStep:
    return recover_latest_index(
        run,
        writable=writable,
        verification=verification,
    )


def _fsync_payload_reference(
    fs: FilesystemOps,
    root_descriptor: int,
    reference: PayloadRef,
) -> None:
    components = parse_logical_path(reference.logical_path)
    current_descriptor = os.dup(root_descriptor)
    payload_descriptor = -1
    try:
        for component in components[:-1]:
            next_descriptor = fs.open(
                component,
                _OPEN_DIRECTORY,
                dir_fd=current_descriptor,
            )
            component_stat = fs.stat(next_descriptor)
            if not stat.S_ISDIR(component_stat.st_mode):
                os.close(next_descriptor)
                raise SMLArtifactError(
                    f"checkpoint payload component is not a directory: {component}"
                )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        payload_descriptor, payload_stat = _opened_entry(
            fs,
            components[-1],
            parent_descriptor=current_descriptor,
            flags=_OPEN_PAYLOAD,
        )
        if not stat.S_ISREG(payload_stat.st_mode) or payload_stat.st_nlink != 1:
            raise SMLArtifactError(
                f"checkpoint payload is not a singly linked file: {reference.logical_path}"
            )
        fs.fsync_file(payload_descriptor)
    finally:
        if payload_descriptor >= 0:
            os.close(payload_descriptor)
        os.close(current_descriptor)


def _fsync_directory_tree(fs: FilesystemOps, directory_descriptor: int) -> None:
    for name in sorted(fs.listdir(directory_descriptor)):
        entry_stat = fs.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(entry_stat.st_mode):
            continue
        child_descriptor, opened_stat = _opened_entry(
            fs,
            name,
            parent_descriptor=directory_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        try:
            if not stat.S_ISDIR(opened_stat.st_mode):
                raise SMLArtifactError(
                    f"checkpoint payload component is not a directory: {name}"
                )
            _fsync_directory_tree(fs, child_descriptor)
        finally:
            os.close(child_descriptor)
    fs.fsync_directory(directory_descriptor)


def _make_checkpoint_group_durable(
    fs: FilesystemOps,
    temporary_descriptor: int,
    references: Sequence[PayloadRef],
) -> None:
    for reference in references:
        _fsync_payload_reference(fs, temporary_descriptor, reference)
    _fsync_directory_tree(fs, temporary_descriptor)


def _cleanup_checkpoint_temporary(
    fs: FilesystemOps,
    checkpoints_descriptor: int,
    temporary_name: str,
) -> None:
    if not _is_temporary_step_name(temporary_name):
        return
    candidate_stat = _stat_if_present(
        fs,
        temporary_name,
        parent_descriptor=checkpoints_descriptor,
    )
    if candidate_stat is None:
        return
    descriptor, opened_stat = _opened_entry(
        fs,
        temporary_name,
        parent_descriptor=checkpoints_descriptor,
        flags=_OPEN_DIRECTORY,
    )
    try:
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SMLArtifactError("temporary checkpoint is not a directory")
        _delete_directory_contents(fs, descriptor)
        _require_named_directory_inode(
            fs,
            temporary_name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=descriptor,
            context="temporary checkpoint directory",
        )
        fs.rmdir(temporary_name, dir_fd=checkpoints_descriptor)
    finally:
        os.close(descriptor)


def publish_checkpoint(
    run: Path,
    build: Callable[[Path], CheckpointManifest],
    *,
    fs: FilesystemOps = OS_FILESYSTEM,
) -> ResolvedStep:
    if not isinstance(run, Path):
        raise TypeError("run must be a Path")
    if not callable(build):
        raise TypeError("build must be callable")
    if not isinstance(fs, FilesystemOps):
        raise TypeError("fs must implement FilesystemOps")
    run_descriptor = _open_directory(
        run,
        fs,
        writable=True,
        context="run directory",
    )
    checkpoints_descriptor = -1
    temporary_descriptor = -1
    temporary_name = ""
    committed = False
    try:
        run_manifest = _read_manifest_from_descriptor(
            run_descriptor,
            RunManifest,
            VerificationLevel.FULL,
            context=str(run / RunManifest.MANIFEST_FILENAME),
        )
        checkpoints_descriptor = _open_checkpoints_directory(
            run,
            fs,
            run_descriptor,
            writable=True,
        )
        temporary_name = _temporary_step_name()
        fs.mkdir(temporary_name, 0o700, dir_fd=checkpoints_descriptor)
        temporary_descriptor = fs.open(
            temporary_name,
            _OPEN_DIRECTORY,
            dir_fd=checkpoints_descriptor,
        )
        temporary_path = run / "checkpoints" / temporary_name
        manifest = build(temporary_path)
        if not isinstance(manifest, CheckpointManifest):
            raise SMLArtifactError(
                "checkpoint builder returned the wrong manifest type"
            )
        _validate_checkpoint_owner(
            run_manifest,
            manifest,
            expected_step=manifest.step,
        )
        validated_manifest = _validate_builder_result(
            manifest,
            temporary_descriptor,
            fs,
        )
        if validated_manifest is not manifest:
            raise SMLArtifactError(
                "checkpoint builder result changed during validation"
            )

        current_latest = _recover_latest_open(
            run,
            run_manifest,
            run_descriptor,
            checkpoints_descriptor,
            writable=True,
            verification=VerificationLevel.FULL,
            fs=fs,
            allow_empty=True,
        )
        if current_latest is not None and manifest.step < current_latest.step:
            raise SMLArtifactError(
                "older checkpoint publication cannot move latest backward"
            )

        target_name = _step_name(manifest.step)
        if (
            _stat_if_present(
                fs,
                target_name,
                parent_descriptor=checkpoints_descriptor,
            )
            is not None
        ):
            existing = _resolve_step_from_descriptor(
                run,
                run_manifest,
                checkpoints_descriptor,
                step=manifest.step,
                verification=VerificationLevel.FULL,
                fs=fs,
            )
            if existing.checkpoint.identity != manifest.identity:
                raise SMLArtifactError(
                    "existing checkpoint step has a different identity collision"
                )
            if current_latest is None or current_latest.step != existing.step:
                _persist_latest_index(run_descriptor, existing, fs)
                return dataclasses.replace(
                    existing,
                    latest_recovered=True,
                    latest_repair_persisted=True,
                )
            return current_latest

        array_references = tuple(array.payload for array in manifest.arrays)
        _make_checkpoint_group_durable(
            fs,
            temporary_descriptor,
            array_references,
        )
        _make_checkpoint_group_durable(
            fs,
            temporary_descriptor,
            (manifest.scalar_state,),
        )
        _write_manifest_last(fs, temporary_descriptor, manifest)
        fs.fsync_directory(temporary_descriptor)
        _verify_closed_world(
            fs,
            temporary_descriptor,
            manifest,
            manifest_present=True,
            full=True,
        )
        _require_named_directory_inode(
            fs,
            temporary_name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=temporary_descriptor,
            context="private checkpoint directory",
        )
        try:
            fs.rename(
                temporary_name,
                target_name,
                source_dir_fd=checkpoints_descriptor,
                destination_dir_fd=checkpoints_descriptor,
            )
        except FileExistsError as error:
            raise SMLArtifactError(
                f"checkpoint step collision while committing: {manifest.step}"
            ) from error
        committed = True
        fs.fsync_directory(checkpoints_descriptor)
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=temporary_descriptor,
            context="committed checkpoint directory",
        )
        committed_manifest = _read_manifest_from_descriptor(
            temporary_descriptor,
            CheckpointManifest,
            VerificationLevel.FULL,
            context=str(run / "checkpoints" / target_name),
        )
        _validate_checkpoint_owner(
            run_manifest,
            committed_manifest,
            expected_step=manifest.step,
        )
        if committed_manifest.identity != manifest.identity:
            raise SMLArtifactError(
                "committed checkpoint manifest changed after step rename"
            )
        _verify_closed_world(
            fs,
            temporary_descriptor,
            committed_manifest,
            manifest_present=True,
            full=True,
        )
        _require_named_directory_inode(
            fs,
            target_name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=temporary_descriptor,
            context="committed checkpoint after full verification",
        )
        published = _resolved_step(
            run,
            run_manifest,
            manifest,
            VerificationLevel.FULL,
            latest_recovered=False,
            latest_repair_persisted=False,
        )
        _persist_latest_index(run_descriptor, published, fs)
        return published
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if checkpoints_descriptor >= 0:
            try:
                if temporary_name and not committed:
                    _cleanup_checkpoint_temporary(
                        fs,
                        checkpoints_descriptor,
                        temporary_name,
                    )
            finally:
                os.close(checkpoints_descriptor)
        os.close(run_descriptor)


def _retention_entries(
    checkpoints_descriptor: int,
    *,
    fs: FilesystemOps,
) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for name in sorted(fs.listdir(checkpoints_descriptor)):
        if _is_temporary_step_name(name):
            continue
        step = _parse_step_name(name)
        if step is None:
            raise SMLArtifactError(f"malformed retention candidate: {name}")
        entries.append((step, name))
    return entries


def _delete_directory_contents_durable(
    fs: FilesystemOps,
    directory_descriptor: int,
    *,
    after_destructive_transition: Callable[[], None] | None = None,
) -> None:
    for name in sorted(fs.listdir(directory_descriptor)):
        if not _safe_child_name(name):
            raise SMLArtifactError("retention found an invalid directory entry name")
        entry_stat = fs.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISREG(entry_stat.st_mode):
            if entry_stat.st_nlink != 1:
                raise SMLArtifactError(f"retention rejects hard-linked file: {name}")
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_PAYLOAD,
            )
            try:
                if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                    raise SMLArtifactError(
                        f"retention rejects non-regular file: {name}"
                    )
                current_stat = fs.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(opened_stat, current_stat):
                    raise SMLArtifactError(f"retention detected a file swap: {name}")
                fs.unlink(name, dir_fd=directory_descriptor)
                if after_destructive_transition is not None:
                    after_destructive_transition()
            finally:
                os.close(descriptor)
        elif stat.S_ISDIR(entry_stat.st_mode):
            descriptor, opened_stat = _opened_entry(
                fs,
                name,
                parent_descriptor=directory_descriptor,
                flags=_OPEN_DIRECTORY,
            )
            try:
                if not stat.S_ISDIR(opened_stat.st_mode):
                    raise SMLArtifactError(f"retention rejects non-directory: {name}")
                _delete_directory_contents_durable(
                    fs,
                    descriptor,
                    after_destructive_transition=after_destructive_transition,
                )
                current_stat = fs.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if not _same_inode(opened_stat, current_stat):
                    raise SMLArtifactError(
                        f"retention detected a directory swap: {name}"
                    )
                fs.rmdir(name, dir_fd=directory_descriptor)
                if after_destructive_transition is not None:
                    after_destructive_transition()
            finally:
                os.close(descriptor)
        else:
            raise SMLArtifactError(f"retention rejects symlink or special file: {name}")
    fs.fsync_directory(directory_descriptor)


def _cleanup_retention_temporaries(
    checkpoints_descriptor: int,
    *,
    latest: _OwnedStep,
    fs: FilesystemOps,
) -> None:
    for name in sorted(fs.listdir(checkpoints_descriptor)):
        if not _is_temporary_step_name(name):
            continue
        _require_named_directory_inode(
            fs,
            latest.name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=latest.descriptor,
            context="retained latest checkpoint before temporary cleanup",
        )
        descriptor, opened_stat = _opened_entry(
            fs,
            name,
            parent_descriptor=checkpoints_descriptor,
            flags=_OPEN_DIRECTORY,
        )
        try:
            if not stat.S_ISDIR(opened_stat.st_mode):
                raise SMLArtifactError(
                    f"retention temporary is not a directory: {name}"
                )
            latest_stat = fs.stat(latest.descriptor)
            if _same_inode(opened_stat, latest_stat):
                raise SMLArtifactError(
                    "retention temporary is bound to the retained latest checkpoint"
                )
            _require_named_directory_inode(
                fs,
                latest.name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=latest.descriptor,
                context="retained latest checkpoint before temporary deletion",
            )
            _require_named_directory_inode(
                fs,
                name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=descriptor,
                context=f"retention temporary {name}",
            )
            _delete_directory_contents_durable(
                fs,
                descriptor,
                after_destructive_transition=lambda: _require_named_directory_inode(
                    fs,
                    latest.name,
                    parent_descriptor=checkpoints_descriptor,
                    directory_descriptor=latest.descriptor,
                    context="retained latest checkpoint during temporary deletion",
                ),
            )
            _require_named_directory_inode(
                fs,
                latest.name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=latest.descriptor,
                context="retained latest checkpoint before temporary removal",
            )
            _require_named_directory_inode(
                fs,
                name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=descriptor,
                context=f"emptied retention temporary {name}",
            )
            fs.rmdir(name, dir_fd=checkpoints_descriptor)
            _require_named_directory_inode(
                fs,
                latest.name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=latest.descriptor,
                context="retained latest checkpoint after temporary removal",
            )
            fs.fsync_directory(checkpoints_descriptor)
            _require_named_directory_inode(
                fs,
                latest.name,
                parent_descriptor=checkpoints_descriptor,
                directory_descriptor=latest.descriptor,
                context="retained latest checkpoint after temporary cleanup",
            )
        finally:
            os.close(descriptor)


def _detach_and_delete_owned_step(
    checkpoints_descriptor: int,
    *,
    candidate: _OwnedStep,
    latest: _OwnedStep,
    fs: FilesystemOps,
) -> None:
    if candidate.name == latest.name or _parse_step_name(candidate.name) is None:
        raise SMLArtifactError("retention refuses to delete latest or malformed step")
    _require_named_directory_inode(
        fs,
        candidate.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=candidate.descriptor,
        context=f"retention candidate {candidate.name}",
    )
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint",
    )
    temporary_name = _temporary_step_name()
    try:
        fs.rename(
            candidate.name,
            temporary_name,
            source_dir_fd=checkpoints_descriptor,
            destination_dir_fd=checkpoints_descriptor,
        )
    except FileExistsError as error:
        raise SMLArtifactError(
            f"retention temporary collision for {candidate.name}"
        ) from error
    _require_named_directory_inode(
        fs,
        temporary_name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=candidate.descriptor,
        context=f"detached retention candidate {candidate.name}",
    )
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint after candidate detach",
    )
    fs.fsync_directory(checkpoints_descriptor)
    _require_named_directory_inode(
        fs,
        temporary_name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=candidate.descriptor,
        context=f"durable detached candidate {candidate.name}",
    )
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint before recursive delete",
    )
    _delete_directory_contents_durable(
        fs,
        candidate.descriptor,
        after_destructive_transition=lambda: _require_named_directory_inode(
            fs,
            latest.name,
            parent_descriptor=checkpoints_descriptor,
            directory_descriptor=latest.descriptor,
            context="retained latest checkpoint during candidate deletion",
        ),
    )
    _require_named_directory_inode(
        fs,
        temporary_name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=candidate.descriptor,
        context=f"emptied retention candidate {candidate.name}",
    )
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint before temporary removal",
    )
    fs.rmdir(temporary_name, dir_fd=checkpoints_descriptor)
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint after candidate removal",
    )
    fs.fsync_directory(checkpoints_descriptor)
    _require_named_directory_inode(
        fs,
        latest.name,
        parent_descriptor=checkpoints_descriptor,
        directory_descriptor=latest.descriptor,
        context="retained latest checkpoint after candidate cleanup",
    )


def apply_retention(
    run: Path,
    *,
    keep_last: int | None,
    fs: FilesystemOps = OS_FILESYSTEM,
) -> ResolvedStep:
    if not isinstance(run, Path):
        raise TypeError("run must be a Path")
    if keep_last is not None:
        if isinstance(keep_last, bool) or not isinstance(keep_last, int):
            raise TypeError("keep_last must be a positive integer or None")
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
    if not isinstance(fs, FilesystemOps):
        raise TypeError("fs must implement FilesystemOps")

    with _protected_lock(
        run,
        category="run-access",
        exclusive=True,
        wait=True,
    ):
        run_descriptor = _open_directory(
            run,
            fs,
            writable=True,
            context="run directory",
        )
        checkpoints_descriptor = -1
        owned_deletions: list[_OwnedStep] = []
        owned_latest: _OwnedStep | None = None
        try:
            run_manifest = _read_manifest_from_descriptor(
                run_descriptor,
                RunManifest,
                VerificationLevel.FULL,
                context=str(run / RunManifest.MANIFEST_FILENAME),
            )
            checkpoints_descriptor = _open_checkpoints_directory(
                run,
                fs,
                run_descriptor,
                writable=True,
            )
            latest = _recover_latest_open(
                run,
                run_manifest,
                run_descriptor,
                checkpoints_descriptor,
                writable=True,
                verification=VerificationLevel.FULL,
                fs=fs,
                allow_empty=False,
            )
            if latest is None:
                raise SMLArtifactError("run has no published checkpoints")
            if keep_last is not None:
                entries = _retention_entries(checkpoints_descriptor, fs=fs)
                retained_names = {name for _step, name in sorted(entries)[-keep_last:]}
                latest_name = _step_name(latest.step)
                retained_names.add(latest_name)
                deletions = [
                    (step, name)
                    for step, name in entries
                    if name not in retained_names and name != latest_name
                ]
                for step, _name in deletions:
                    owned_deletions.append(
                        _open_verified_step_from_descriptor(
                            run,
                            latest.run,
                            checkpoints_descriptor,
                            step=step,
                            verification=VerificationLevel.FULL,
                            fs=fs,
                        )
                    )

            owned_latest = _open_verified_step_from_descriptor(
                run,
                latest.run,
                checkpoints_descriptor,
                step=latest.step,
                verification=VerificationLevel.FULL,
                fs=fs,
            )
            latest_proof = owned_latest.resolved
            if latest_proof.checkpoint.identity != latest.checkpoint.identity:
                raise SMLArtifactError(
                    "fresh retained latest proof changed checkpoint identity"
                )
            latest = dataclasses.replace(
                latest_proof,
                latest_recovered=latest.latest_recovered,
                latest_repair_persisted=latest.latest_repair_persisted,
            )
            _cleanup_retention_temporaries(
                checkpoints_descriptor,
                latest=owned_latest,
                fs=fs,
            )
            if keep_last is None:
                return latest
            for candidate in owned_deletions:
                _detach_and_delete_owned_step(
                    checkpoints_descriptor,
                    candidate=candidate,
                    latest=owned_latest,
                    fs=fs,
                )
            return latest
        finally:
            if owned_latest is not None:
                owned_latest.close()
            for candidate in owned_deletions:
                candidate.close()
            if checkpoints_descriptor >= 0:
                os.close(checkpoints_descriptor)
            os.close(run_descriptor)


__all__ = [
    "CHECKPOINT_PUBLICATION_STAGES",
    "IMMUTABLE_PUBLICATION_STAGES",
    "OS_FILESYSTEM",
    "FilesystemOps",
    "Published",
    "ResolvedStep",
    "apply_retention",
    "publication_lock",
    "publish_checkpoint",
    "publish_immutable_bundle",
    "publish_run",
    "recover_latest_index",
    "resolve_exact_step",
    "resolve_latest_step",
    "run_access_lock",
    "run_writer_lock",
]
