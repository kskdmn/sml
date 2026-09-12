"""Verify delivered benchmark inputs after the measured interval has ended."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservedBatch:
    arrays: Mapping[str, object]
    work_ids: tuple[int, ...] | None = None
    cursor: object | None = None


def verify_input_batches(
    observed: list[ObservedBatch],
    canonical: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> tuple[int, ...]:
    """Check actual transferred values as well as their delivered IDs/cursors.

    Observations retain only input arrays. Host materialization for validation,
    comparisons, and identity hashing happen after timing and peak-memory sampling.
    """
    row_count = len(next(iter(canonical.values())))
    ordered: list[int] = []
    for batch in observed:
        if batch.cursor is not None:
            cursor = batch.cursor
            position = getattr(
                cursor,
                "shard_order_position",
                getattr(cursor, "bucket_order_position", None),
            )
            stop = cursor.epoch * row_count + cursor.row_offset
            if position != 0 or stop != len(ordered) + batch_size:
                raise RuntimeError(
                    "benchmark observed execution order has an invalid cursor"
                )
            work_ids = tuple(
                index % row_count for index in range(stop - batch_size, stop)
            )
        else:
            work_ids = batch.work_ids
        expected = tuple(
            (len(ordered) + offset) % row_count for offset in range(batch_size)
        )
        if work_ids != expected:
            raise RuntimeError(
                "benchmark observed execution order differs from canonical work"
            )
        if set(batch.arrays) != set(canonical):
            raise RuntimeError(
                "benchmark observed input fields differ from canonical work"
            )
        for name, actual in batch.arrays.items():
            values = np.asarray(actual)
            reference = canonical[name][list(work_ids)]
            if values.dtype != reference.dtype or not np.array_equal(values, reference):
                raise RuntimeError(
                    "benchmark observed input values differ from canonical work"
                )
        ordered.extend(work_ids)
    return tuple(ordered)
