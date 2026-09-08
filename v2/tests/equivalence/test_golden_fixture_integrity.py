from __future__ import annotations

import hashlib
import json

import mlx.core as mx
import numpy as np


def _array_payload_identity(array: mx.array) -> str:
    host = np.asarray(array.view(mx.uint16) if array.dtype == mx.bfloat16 else array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(host).tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def test_golden_metadata_accounts_for_every_tensor(legacy_control, legacy_arrays):
    metadata = legacy_control["arrays"]
    assert set(metadata) == set(legacy_arrays)
    for name, array in legacy_arrays.items():
        record = metadata[name]
        assert record["shape"] == list(array.shape)
        assert record["dtype"] == str(array.dtype)
        assert record["payload_identity"] == _array_payload_identity(array)


def test_golden_model_mapping_has_one_canonical_tied_embedding(
    legacy_control, legacy_arrays
):
    mappings = legacy_control["parameter_state"]["mapping"]
    names = [mapping["destination"] for mapping in mappings]
    assert len(names) == len(set(names))
    assert set(names) == {
        name.removeprefix("model_state.")
        for name in legacy_arrays
        if name.startswith("model_state.")
    }
    assert "embed_tokens.weight" in names
    assert "lm_head.weight" not in names
    for mapping in mappings:
        assert mapping["payload_identity"] == _array_payload_identity(
            legacy_arrays[f"model_state.{mapping['destination']}"]
        )


def test_golden_lora_mapping_uses_current_base_and_adapter_names(
    legacy_control, legacy_arrays
):
    state = legacy_control["lora_parameter_state"]
    base_names = {mapping["destination"] for mapping in state["base_mapping"]}
    adapter_names = {mapping["destination"] for mapping in state["adapter_mapping"]}
    assert not base_names & adapter_names
    assert all(name.endswith((".lora_a", ".lora_b")) for name in adapter_names)
    for namespace, mapping_name in (
        ("lora_base_state", "base_mapping"),
        ("lora_state", "adapter_mapping"),
    ):
        mappings = state[mapping_name]
        names = [mapping["destination"] for mapping in mappings]
        assert len(names) == len(set(names))
        assert set(names) == {
            name.removeprefix(f"{namespace}.")
            for name in legacy_arrays
            if name.startswith(f"{namespace}.")
        }
        for mapping in mappings:
            assert mapping["payload_identity"] == _array_payload_identity(
                legacy_arrays[f"{namespace}.{mapping['destination']}"]
            )
