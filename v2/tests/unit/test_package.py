import subprocess
import sys

EXPECTED_PUBLIC_TYPES = {
    "CheckpointPolicy": "sml.training.common",
    "CorpusConfig": "sml.data.corpus",
    "EvaluationConfig": "sml.evaluation",
    "EvaluationResult": "sml.evaluation",
    "ExportResult": "sml.training.swag",
    "GenerationConfig": "sml.model.config",
    "GenerationRequest": "sml.inference",
    "GenerationResult": "sml.inference",
    "InferenceConfig": "sml.inference",
    "InferenceRuntimeConfig": "sml.inference",
    "InferenceSession": "sml.inference",
    "InitializerConfig": "sml.model.config",
    "LoaderConfig": "sml.training.common",
    "LoRAConfig": "sml.training.lora",
    "LoRAInitializerConfig": "sml.training.lora",
    "LoRAPrecisionConfig": "sml.training.lora",
    "ModelConfig": "sml.model.config",
    "OptimizerConfig": "sml.training.common",
    "PrecisionConfig": "sml.training.common",
    "PreparedDataBundle": "sml.data.pretraining",
    "PretrainingConfig": "sml.training.common",
    "PretrainingPreparationConfig": "sml.data.pretraining",
    "ResumeOverrides": "sml.training.common",
    "SMLArtifactError": "sml.errors",
    "SMLConfigurationError": "sml.errors",
    "SMLDataError": "sml.errors",
    "SMLRuntimeError": "sml.errors",
    "SwagDataBundle": "sml.data.swag",
    "SwagPreparationConfig": "sml.data.swag",
    "SwagSourceConfig": "sml.data.swag",
    "SwagTrainingConfig": "sml.training.swag",
    "SwagTrainingResult": "sml.training.swag",
    "TokenizerBundle": "sml.data.tokenizer",
    "TokenizerTrainingConfig": "sml.data.tokenizer",
    "TrainingResult": "sml.training.pretrain",
    "VerificationResult": "sml.artifacts.verify",
    "WeightDecayPolicy": "sml.training.common",
}


def test_package_exports_only_supported_domain_types():
    import importlib

    import sml

    assert sml.__file__.endswith("sml/__init__.py")
    assert set(sml.__all__) == set(EXPECTED_PUBLIC_TYPES)
    for name, module_name in EXPECTED_PUBLIC_TYPES.items():
        owner = importlib.import_module(module_name)
        assert getattr(sml, name) is getattr(owner, name)


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
