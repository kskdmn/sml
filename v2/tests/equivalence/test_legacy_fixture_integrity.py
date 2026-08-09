from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_SCRIPT = Path(__file__).with_name("capture_legacy.py")
EXPECTED_CASES = {
    "model",
    "parameter_state",
    "rope",
    "gqa",
    "cache",
    "generation",
    "loglikelihood",
    "tied_gradient_leaves",
    "lora",
    "compiled_state",
    "swag_legacy_sum",
}


def _capture_tool_commit() -> str:
    relative_script = CAPTURE_SCRIPT.relative_to(PROJECT_ROOT)
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative_script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_capture_module():
    spec = importlib.util.spec_from_file_location("capture_legacy", CAPTURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load capture driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_array_bytes(array: mx.array) -> bytes:
    if array.dtype == mx.bfloat16:
        host_array = np.asarray(array.view(mx.uint16))
    else:
        host_array = np.asarray(array)
    return np.ascontiguousarray(host_array).tobytes(order="C")


def _array_payload_identity(array: mx.array) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(_raw_array_bytes(array))
    return f"sha256:{digest.hexdigest()}"


def test_legacy_fixture_records_source_and_required_cases(legacy_control):
    assert legacy_control["source_commit"] == "3687f8b"
    assert legacy_control["capture_tool_commit"] == _capture_tool_commit()
    assert legacy_control["capture_tool_identity"].startswith("sha256:")
    assert legacy_control["pretraining_model_config"]["rope_scaling_factor"] == 1.0
    assert set(legacy_control["cases"]) == EXPECTED_CASES
    assert legacy_control["dropout"] == 0.0
    assert legacy_control["parameter_state"]["tied_source_names"] == [
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    assert (
        legacy_control["parameter_state"]["canonical_destination"]
        == "embed_tokens.weight"
    )
    assert legacy_control["parameter_state"]["payload_identity"].startswith("sha256:")


def test_generation_references_serialize_complete_named_configurations(
    legacy_control,
):
    generation = legacy_control["cases"]["generation"]
    assert generation["configurations"] == {
        "greedy": {
            "generation_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "seed": None,
            },
            "max_new_tokens": 3,
        },
        "padded_unequal": {
            "generation_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "seed": None,
            },
            "max_new_tokens": 3,
        },
        "primitive_sampling": {
            "generation_config": {
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "seed": None,
            },
            "key_seed": 1234,
            "max_new_tokens": 1,
        },
        "seeded": {
            "generation_config": {
                "temperature": 0.8,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
                "seed": 1234,
            },
            "max_new_tokens": 3,
        },
        "serial_unequal": {
            "generation_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "seed": None,
            },
            "max_new_tokens": 3,
        },
    }
    assert generation["references"] == {
        "greedy": {
            "configuration": "greedy",
            "arrays": [
                "generation.greedy_prompt",
                "generation.greedy_tokens",
                "generation.greedy_winning_margins",
            ],
        },
        "padded_unequal": {
            "configuration": "padded_unequal",
            "arrays": [
                "generation.padded_unequal_prompts",
                "generation.padded_unequal_tokens",
                "generation.padded_unequal_winning_margins",
            ],
        },
        "primitive_sampling": {
            "configuration": "primitive_sampling",
            "arrays": ["generation.logits", "generation.sampled_token"],
        },
        "seeded": {
            "configuration": "seeded",
            "arrays": [
                "generation.seeded_prompt",
                "generation.seeded_tokens",
                "generation.seeded_winning_margins",
            ],
        },
        "serial_unequal": {
            "configuration": "serial_unequal",
            "arrays": [
                "generation.serial_prompt.0",
                "generation.serial_prompt.1",
                "generation.serial_tokens.0",
                "generation.serial_tokens.1",
                "generation.serial_winning_margins.0",
                "generation.serial_winning_margins.1",
            ],
        },
    }


def test_capture_driver_rejects_an_uncommitted_or_wrong_checkout(tmp_path):
    assert_clean_checkout = _load_capture_module().assert_clean_checkout

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "capture-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Capture Test"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert_clean_checkout(tmp_path, expected_commit=commit)

    tracked.write_text("dirty\n", encoding="utf-8")
    try:
        assert_clean_checkout(tmp_path, expected_commit=commit)
    except RuntimeError as error:
        assert "not clean" in str(error)
    else:
        raise AssertionError("dirty checkout was accepted")

    tracked.write_text("committed\n", encoding="utf-8")
    try:
        assert_clean_checkout(tmp_path, expected_commit="0" * 40)
    except RuntimeError as error:
        assert "expected commit" in str(error)
    else:
        raise AssertionError("wrong source commit was accepted")


def test_capture_driver_converts_bfloat16_control_values_through_float32():
    capture_module = _load_capture_module()
    values = mx.array([[1.5, -2.25]], dtype=mx.bfloat16)

    assert capture_module._json_array_values(values, mx, np) == [[1.5, -2.25]]


def test_capture_driver_raw_forward_supports_consecutive_compiled_calls():
    from sml import SMLConfig, SMLLanguageModel

    capture_module = _load_capture_module()
    model = SMLLanguageModel(
        SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=8,
            rope_scaling_factor=1.0,
            hidden_dropout=0.0,
        )
    )
    model.eval()
    compiled = mx.compile(
        lambda token_ids: capture_module._legacy_forward_without_validation(
            model, token_ids
        )
    )

    first = compiled(mx.array([[1, 4]], dtype=mx.int32))
    mx.eval(first)
    second = compiled(mx.array([[1, 5, 6]], dtype=mx.int32))
    mx.eval(second)

    assert first.shape == (1, 2, 16)
    assert second.shape == (1, 3, 16)


def test_capture_driver_records_mlx_distribution_version():
    capture_module = _load_capture_module()

    assert capture_module._distribution_version("mlx") == importlib.metadata.version(
        "mlx"
    )


def test_legacy_fixture_accounts_for_every_payload(legacy_control, legacy_arrays):
    metadata = legacy_control["arrays"]
    assert list(metadata) == sorted(metadata)
    assert set(metadata) == set(legacy_arrays)
    assert any(name.startswith("model_state.") for name in metadata)
    assert any(name.startswith("lora_base_state.") for name in metadata)
    assert any(name.startswith("lora_state.") for name in metadata)

    for name, array in legacy_arrays.items():
        record = metadata[name]
        assert record["shape"] == list(array.shape)
        assert record["dtype"] == str(array.dtype)
        assert record["payload_identity"] == _array_payload_identity(array)


def test_parameter_mappings_are_complete_ordered_and_unambiguous(
    legacy_control, legacy_arrays
):
    parameter_state = legacy_control["parameter_state"]
    mappings = parameter_state["mapping"]
    source_names = [mapping["source"] for mapping in mappings]
    destinations = [mapping["destination"] for mapping in mappings]
    assert source_names == sorted(source_names)
    assert len(destinations) == len(set(destinations))
    assert destinations == [
        name.removeprefix("model_state.")
        for name in legacy_arrays
        if name.startswith("model_state.")
    ]
    assert parameter_state["tied_alias_byte_identical"] is True

    for mapping in mappings:
        array = legacy_arrays[f"model_state.{mapping['destination']}"]
        assert mapping["shape"] == list(array.shape)
        assert mapping["dtype"] == str(array.dtype)
        assert mapping["payload_identity"] == _array_payload_identity(array)


def test_lora_mappings_account_for_exact_base_and_adapter_leaves(
    legacy_control, legacy_arrays
):
    lora_state = legacy_control["lora_parameter_state"]
    base_mappings = lora_state["base_mapping"]
    adapter_mappings = lora_state["adapter_mapping"]
    base_destinations = [mapping["destination"] for mapping in base_mappings]
    adapter_destinations = [mapping["destination"] for mapping in adapter_mappings]
    assert base_destinations == sorted(base_destinations)
    assert adapter_destinations == sorted(adapter_destinations)
    assert not set(base_destinations) & set(adapter_destinations)
    assert "embed_tokens.weight" in base_destinations
    assert "lm_head.weight" not in base_destinations
    assert base_destinations == [
        name.removeprefix("lora_base_state.")
        for name in legacy_arrays
        if name.startswith("lora_base_state.")
    ]
    assert adapter_destinations == [
        name.removeprefix("lora_state.")
        for name in legacy_arrays
        if name.startswith("lora_state.")
    ]
    assert all(name.endswith((".lora_A", ".lora_B")) for name in adapter_destinations)
