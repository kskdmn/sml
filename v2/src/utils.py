from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path


def shuffle_input_files(input_files: Iterable[Path], seed: int) -> tuple[Path, ...]:
    shuffled_files = list(input_files)
    random.Random(seed).shuffle(shuffled_files)
    return tuple(shuffled_files)
