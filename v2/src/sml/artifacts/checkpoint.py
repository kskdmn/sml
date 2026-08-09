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
from typing import Protocol, runtime_checkable

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
    canonical_json_bytes,
    parse_logical_path,
    read_manifest,
)
from sml.errors import SMLArtifactError

IMMUTABLE_PUBLICATION_STAGES = (
    "payloads-written",
    "manifest-written",
    "temporary-directory-fsynced",
    "directory-renamed",
    "parent-directory-fsynced",
)

_OPEN_READ = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_OPEN_DIRECTORY = _OPEN_READ | os.O_DIRECTORY
_OPEN_PAYLOAD = _OPEN_READ | os.O_NONBLOCK
_OPEN_LOCK = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_OPEN_MANIFEST = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
)
_TEMPORARY_SUFFIX_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
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
    return _protected_lock(run, category="run", exclusive=True)


def run_access_lock(run: Path, *, exclusive: bool) -> Iterator[None]:
    if not isinstance(exclusive, bool):
        raise TypeError("exclusive must be a bool")
    return _protected_lock(run, category="run", exclusive=exclusive)


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


def _verify_closed_world(
    fs: FilesystemOps,
    temporary_descriptor: int,
    manifest: object,
    *,
    manifest_present: bool,
) -> None:
    references = tuple(_payload_references(manifest))
    expected_files, expected_directories = _expected_closed_world(references)
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
        artifact_root.verify_payloads(references, full=True)
        if manifest_present:
            with artifact_root.open_payload(
                manifest.MANIFEST_FILENAME
            ) as manifest_file:
                if manifest_file.read() != canonical_json_bytes(manifest):
                    raise SMLArtifactError(
                        "publisher-owned manifest changed before immutable commit"
                    )


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


def _accept_existing[M](target: Path, manifest: M) -> Published[M]:
    try:
        verified = read_manifest(target, type(manifest), VerificationLevel.FULL)
    except SMLArtifactError as error:
        raise SMLArtifactError(
            f"existing target failed full verification: {target}"
        ) from error
    if type(verified.manifest) is not type(manifest):
        raise SMLArtifactError(f"existing target manifest type collision: {target}")
    if verified.manifest.identity != manifest.identity:
        raise SMLArtifactError(
            f"existing target has a different identity collision: {target}"
        )
    return Published(
        path=target,
        manifest=verified.manifest,
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
            return _accept_existing(target, manifest)

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


__all__ = [
    "IMMUTABLE_PUBLICATION_STAGES",
    "OS_FILESYSTEM",
    "FilesystemOps",
    "Published",
    "publication_lock",
    "publish_immutable_bundle",
    "run_access_lock",
    "run_writer_lock",
]
