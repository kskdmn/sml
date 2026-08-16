from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sml.data.corpus import CorpusConfig
from sml.data.pretraining import (
    PretrainingPreparationConfig,
    _windowed_row_shuffle,
    pack_token_ranges,
)


def preparation_config(**overrides) -> PretrainingPreparationConfig:
    values = {
        "corpus": CorpusConfig(input_root=Path("corpus")),
        "tokenizer_bundle": Path("tokenizer"),
        "sequence_length": 3,
        "shuffle_window_rows": 3,
        "output_shard_rows": 2,
        "seed": 7,
    }
    values.update(overrides)
    return PretrainingPreparationConfig(**values)


def test_packing_overlaps_one_boundary_token_across_token_ranges():
    rows = list(
        pack_token_ranges(
            [[1, 2], [3, 4, 5], [], [6, 7]],
            sequence_length=3,
            vocab_size=8,
        )
    )

    assert [row.tolist() for row in rows] == [[1, 2, 3, 4], [4, 5, 6, 7]]
    assert all(row.dtype == np.dtype("<i4") and row.flags.c_contiguous for row in rows)


@pytest.mark.parametrize(
    ("token_ranges", "message"),
    [
        ([[1, -1, 2, 3]], "nonnegative"),
        ([[1, 2, 8, 3]], "smaller than vocab_size"),
        ([[1.0, 2.0, 3.0, 4.0]], "integer"),
        ([[True, False, True, False]], "integer"),
    ],
)
def test_packing_rejects_invalid_token_ranges_before_emitting_rows(
    token_ranges, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        list(
            pack_token_ranges(
                token_ranges,
                sequence_length=3,
                vocab_size=8,
            )
        )


def test_packing_discards_only_the_final_incomplete_tail():
    rows = list(pack_token_ranges([[1, 2, 3, 4, 5, 6]], sequence_length=3))

    assert [row.tolist() for row in rows] == [[1, 2, 3, 4]]


def test_window_shuffle_has_a_fixed_pcg64_order_for_full_and_partial_windows():
    rows = (np.array([index, index + 10], dtype="<i4") for index in range(8))

    shuffled = list(_windowed_row_shuffle(rows, window_rows=3, seed=7))

    assert [int(row[0]) for row in shuffled] == [0, 2, 1, 4, 5, 3, 6, 7]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"corpus": object()}, TypeError, "corpus"),
        ({"tokenizer_bundle": "tokenizer"}, TypeError, "tokenizer_bundle"),
        ({"sequence_length": 0}, ValueError, "sequence_length"),
        ({"shuffle_window_rows": 0}, ValueError, "shuffle_window_rows"),
        ({"output_shard_rows": 0}, ValueError, "output_shard_rows"),
        ({"seed": True}, TypeError, "seed"),
        (
            {"shuffle_algorithm": "windowed-row-shuffle-v2"},
            ValueError,
            "shuffle_algorithm",
        ),
    ],
)
def test_preparation_config_rejects_invalid_values(overrides, error, message):
    with pytest.raises(error, match=message):
        preparation_config(**overrides)


def test_preparation_config_expands_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = preparation_config(tokenizer_bundle=Path("~/tokenizer"))

    assert config.tokenizer_bundle == tmp_path / "tokenizer"


def test_importing_pretraining_module_does_not_import_sentencepiece():
    code = (
        "import sys; import sml.data.pretraining; print('sentencepiece' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "False"
    assert completed.stderr == ""
