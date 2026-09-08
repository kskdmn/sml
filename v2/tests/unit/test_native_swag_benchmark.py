from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from v2.benchmarks.adapters.native_swag import make_runtime
from v2.benchmarks.parameters import parameter_identity
from v2.benchmarks.workload import build_canonical_workload, fixed_swag_examples


@pytest.fixture
def tiny_workload():
    return build_canonical_workload(
        model_overrides={
            "vocab_size": 64,
            "hidden_size": 8,
            "num_layers": 1,
            "num_q_heads": 2,
            "num_kv_heads": 1,
            "intermediate_size": 16,
            "original_context_length": 8,
            "hidden_dropout": 0.2,
        },
        optimizer_overrides={
            "swag": {
                "gradient_accumulation_steps": 2,
                "warmup_steps": 0,
                "total_steps": 8,
                "lora": {"rank": 2},
            }
        },
        loader_overrides={
            "sequence_length": 8,
            "swag": {"sequence_length": 8, "example_count": 3, "batch_size": 2},
        },
        generation_overrides={
            "prompt_tokens": 2,
            "decode_chunk_size": 1,
            "request_count": 2,
        },
        row_count=4,
    )


def test_native_swag_trains_adapters_and_counts_actual_canonical_examples(
    tiny_workload, tmp_path
):
    runtime = make_runtime("swag-end-to-end", tiny_workload, tmp_path)
    try:
        initial_base = {
            name: np.asarray(value.astype(mx.float32)).copy()
            for name, value in tree_flatten(runtime.frozen_base)
        }
        initial_identity = runtime.initial_parameter_identity
        assert initial_identity == parameter_identity(runtime.model)
        assert runtime.run(2) == 8.0
        assert runtime.measured_work_ids == [0, 1, 2, 0, 1, 2, 0, 1]
        assert int(runtime.optimizer.step.item()) == 2
        assert runtime.microstep_index == 4
        assert int(runtime.trainer.valid_count.item()) == 0
        assert parameter_identity(runtime.model) != initial_identity
        for name, value in tree_flatten(runtime.frozen_base):
            np.testing.assert_array_equal(value.astype(mx.float32), initial_base[name])
        assert all(
            value.dtype == mx.float32 for _, value in tree_flatten(runtime.adapters)
        )
        assert all(
            value.dtype == mx.float32
            for _, value in tree_flatten(runtime.optimizer.first_moments)
        )

        runtime.reset_measured_order()
        assert runtime.run(1) == 4.0
        assert runtime.measured_work_ids == [0, 1, 2, 0]
        assert int(runtime.optimizer.step.item()) == 3
        assert runtime.microstep_index == 6
    finally:
        runtime.close()


def test_native_swag_batch_preserves_labels_masks_and_wrapped_order(
    tiny_workload, tmp_path
):
    runtime = make_runtime("swag-end-to-end", tiny_workload, tmp_path)
    examples = fixed_swag_examples(tiny_workload, np.empty((0, 0), dtype=np.int32))
    try:
        runtime._next_batch()
        batch = runtime._next_batch()
        np.testing.assert_array_equal(batch.input_ids, examples.input_ids[[2, 0]])
        np.testing.assert_array_equal(batch.labels, examples.candidate_labels[[2, 0]])
        np.testing.assert_array_equal(batch.score_mask, examples.labels[[2, 0]] != 0)
        np.testing.assert_array_equal(
            batch.valid_token_mask, examples.input_ids[[2, 0]] != 0
        )
        assert np.asarray(batch.example_mask).tolist() == [True, True]
        assert runtime.canonical_input_identity == examples.identity
    finally:
        runtime.close()


def test_native_swag_rejects_changed_canonical_input_identity(tiny_workload, tmp_path):
    forged = replace(
        tiny_workload,
        semantic_identities={
            **tiny_workload.semantic_identities,
            "canonical_swag_examples": "sha256:" + "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="canonical SWAG examples"):
        make_runtime("swag-end-to-end", forged, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shuffle_examples", True),
        ("dataset_revision", "other"),
        ("cache_format", "other"),
        ("example_count", 3.5),
        ("sequence_length", 8.5),
    ],
)
def test_native_swag_rejects_unsupported_loader_protocol(
    tiny_workload, tmp_path, field, value
):
    changed = replace(
        tiny_workload,
        loader={
            **tiny_workload.loader,
            "swag": {**tiny_workload.loader["swag"], field: value},
        },
    )
    with pytest.raises(ValueError, match=field):
        make_runtime("swag-end-to-end", changed, tmp_path)
