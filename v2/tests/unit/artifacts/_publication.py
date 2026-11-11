from __future__ import annotations


class _ForwardingFilesystemOps:
    def __init__(self, inner) -> None:
        self._inner = inner

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        return self._inner.open(path, flags, mode, dir_fd=dir_fd)

    def mkdir(self, path, mode=0o777, *, dir_fd):
        self._inner.mkdir(path, mode, dir_fd=dir_fd)

    def write_all(self, descriptor, data):
        self._inner.write_all(descriptor, data)

    def fsync_file(self, descriptor):
        self._inner.fsync_file(descriptor)

    def fsync_directory(self, descriptor):
        self._inner.fsync_directory(descriptor)

    def rename(
        self,
        source,
        destination,
        *,
        source_dir_fd,
        destination_dir_fd,
    ):
        self._inner.rename(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    def replace(
        self,
        source,
        destination,
        *,
        source_dir_fd,
        destination_dir_fd,
    ):
        self._inner.replace(
            source,
            destination,
            source_dir_fd=source_dir_fd,
            destination_dir_fd=destination_dir_fd,
        )

    def unlink(self, path, *, dir_fd):
        self._inner.unlink(path, dir_fd=dir_fd)

    def rmdir(self, path, *, dir_fd):
        self._inner.rmdir(path, dir_fd=dir_fd)

    def stat(self, path, *, dir_fd=None, follow_symlinks=False):
        return self._inner.stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )

    def listdir(self, descriptor):
        return self._inner.listdir(descriptor)

    def flock(self, descriptor, operation):
        self._inner.flock(descriptor, operation)
