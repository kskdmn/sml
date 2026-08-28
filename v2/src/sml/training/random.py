"""Counter-addressed random-key schedule for resumable training."""

from __future__ import annotations

import hashlib

import mlx.core as mx

from sml.errors import SMLConfigurationError

_DOMAIN = b"sml-training-counter-key-v1\0"
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def counter_random_key(seed: int, microstep: int) -> mx.array:
    """Derive the exact training key for ``(seed, microstep)`` in O(1) work."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= _UINT32_MAX
    ):
        raise SMLConfigurationError("training random seed must be uint32")
    if (
        isinstance(microstep, bool)
        or not isinstance(microstep, int)
        or not 0 <= microstep <= _UINT64_MAX
    ):
        raise SMLConfigurationError("training random microstep must be uint64")
    digest = hashlib.sha256(
        _DOMAIN
        + seed.to_bytes(4, byteorder="little")
        + microstep.to_bytes(8, byteorder="little")
    ).digest()
    return mx.array(
        (
            int.from_bytes(digest[:4], byteorder="little"),
            int.from_bytes(digest[4:8], byteorder="little"),
        ),
        dtype=mx.uint32,
    )


__all__ = ["counter_random_key"]
