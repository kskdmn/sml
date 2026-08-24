import importlib
import subprocess
import sys

import pytest

EXPECTED_LEGACY_BRIDGE_EXPORTS = {
    "ParameterInitializerRangeConfig",
    "SMLConfig",
    "GenerationConfig",
    "SMLForwardOutput",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_get_mscale",
    "resolve_yarn_attention_factor",
    "yarn_linear_ramp_mask",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_repetition_penalty",
    "apply_no_repeat_ngram",
    "select_next_token",
    "RMSNorm",
    "RotaryEmbedding",
    "KVCache",
    "GroupedQueryAttention",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "SMLLanguageModel",
    "compute_causal_lm_loss",
    "count_parameters",
    "create_model",
    "estimate_model_size",
}


def test_package_wins_over_legacy_module():
    import sml

    assert sml.__file__.endswith("sml/__init__.py")
    assert sml.SMLLanguageModel.__name__ == "SMLLanguageModel"


def test_bridge_covers_every_unmigrated_flat_import():
    import sml

    assert set(sml.LEGACY_BRIDGE_EXPORTS) == EXPECTED_LEGACY_BRIDGE_EXPORTS
    assert not hasattr(sml, "mx")
    legacy = sys.modules["sml._legacy"]
    for name in EXPECTED_LEGACY_BRIDGE_EXPORTS:
        assert getattr(sml, name) is getattr(legacy, name)


@pytest.mark.parametrize(
    "module_name", ["train_sml", "infer_sml", "evaluate_sml", "lora", "ft_swag"]
)
def test_every_unmigrated_flat_module_imports_through_bridge(module_name):
    importlib.import_module(module_name)


def test_module_entrypoint_is_available():
    completed = subprocess.run(
        [sys.executable, "-m", "sml", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "SML workflows" in completed.stdout


def test_cli_requires_a_command_for_non_help_dispatch(capsys):
    from sml.cli import main

    assert main([]) == 2
    captured = capsys.readouterr()
    assert "SMLConfigurationError: a command is required" in captured.err
    assert "Traceback" not in captured.err


def test_empty_module_entrypoint_renders_configuration_error_without_traceback():
    completed = subprocess.run(
        [sys.executable, "-m", "sml"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "SMLConfigurationError: a command is required" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_error_types_are_distinct_sml_exceptions():
    from sml.errors import (
        SMLArtifactError,
        SMLConfigurationError,
        SMLDataError,
        SMLRuntimeError,
    )

    assert (
        len({SMLConfigurationError, SMLArtifactError, SMLDataError, SMLRuntimeError})
        == 4
    )
    assert all(
        issubclass(error, Exception)
        for error in (
            SMLConfigurationError,
            SMLArtifactError,
            SMLDataError,
            SMLRuntimeError,
        )
    )
