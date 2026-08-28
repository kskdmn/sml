import argparse
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.cli import main, parse_command
from sml.errors import (
    SMLArtifactError,
    SMLConfigurationError,
    SMLDataError,
    SMLRuntimeError,
)


def test_main_closes_descriptor_owning_dispatch_results(monkeypatch) -> None:
    """Catches CLI output leaving a prepared SWAG bundle live."""
    closed: list[bool] = []

    class Result:
        def close(self) -> None:
            closed.append(True)

    class Command:
        def dispatch(self) -> Result:
            return Result()

    monkeypatch.setattr("sml.cli.parse_command", lambda _arguments: Command())

    assert main(["verify", "run"]) == 0
    assert closed == [True]


def test_main_preserves_dispatch_error_when_result_cleanup_also_fails(
    monkeypatch,
) -> None:
    """Catches cleanup masking a semantic error raised while printing a result."""

    class Result:
        def __str__(self) -> str:
            raise SMLArtifactError("semantic failure")

        def close(self) -> None:
            raise RuntimeError("cleanup failure")

    class Command:
        def dispatch(self) -> Result:
            return Result()

    monkeypatch.setattr("sml.cli.parse_command", lambda _arguments: Command())

    with pytest.raises(SMLArtifactError, match="semantic failure") as caught:
        main(["verify", "run"])
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_cli_precedence_is_defaults_then_toml_then_explicit(tmp_path):
    config = tmp_path / "train.toml"
    config.write_text(
        "[train]\nmicrobatch_size = 2\nmaximum_steps = 8\n"
        "[train.model]\nrope_scaling_factor = 1.0\n",
        encoding="utf-8",
    )

    parsed = parse_command(
        [
            "train",
            "--config",
            str(config),
            "--data",
            "data",
            "--output",
            "run",
            "--maximum-steps",
            "3",
        ]
    )
    domain = parsed.to_domain()

    assert domain.loader.microbatch_size == 2
    assert domain.maximum_steps == 3
    assert domain.loader.gradient_accumulation_steps == 8
    with pytest.raises(FrozenInstanceError):
        parsed.values = {"data": Path("other")}


def test_nested_dto_values_are_recursively_frozen(tmp_path):
    config = tmp_path / "train.toml"
    config.write_text(
        '[train]\ndata = "data"\noutput = "run"\n'
        "[train.model]\ninitializer_range = 0.03\n",
        encoding="utf-8",
    )
    parsed = parse_command(["train", "--config", str(config)])
    before = parsed.to_domain()
    model_values = parsed.values["model"]

    with pytest.raises(TypeError):
        model_values["initializer_range"] = 0.5

    assert parsed.values["model"]["initializer_range"] == 0.03
    assert parsed.to_domain() == before


def test_model_initializer_defaults_use_final_overlay_values(tmp_path):
    from sml.model.config import InitializerConfig

    config = tmp_path / "train.toml"
    config.write_text(
        '[train]\ndata = "data"\noutput = "run"\n'
        "[train.model]\nnum_layers = 6\ninitializer_range = 0.03\n",
        encoding="utf-8",
    )

    model = parse_command(["train", "--config", str(config)]).to_domain().model

    assert model.initializers == InitializerConfig.depth_scaled(0.03, 6)


@pytest.mark.parametrize(
    ("argv", "module_name", "workflow"),
    [
        (
            ["tokenize", "--input", "corpus", "--output", "tok"],
            "sml.data.tokenizer",
            "train_tokenizer_bundle",
        ),
        (
            [
                "prepare",
                "pretraining",
                "--input",
                "corpus",
                "--tokenizer",
                "tok",
                "--output",
                "data",
            ],
            "sml.data.pretraining",
            "prepare_pretraining_bundle",
        ),
        (
            [
                "prepare",
                "swag",
                "--checkpoint",
                "base",
                "--revision",
                "0123456789abcdef",
                "--output",
                "swag",
            ],
            "sml.data.swag",
            "prepare_swag_bundle",
        ),
        (
            ["train", "--data", "data", "--output", "run"],
            "sml.training.pretrain",
            "train",
        ),
        (
            ["infer", "--checkpoint", "run", "prompt"],
            "sml.inference",
            "infer",
        ),
        (
            [
                "evaluate",
                "--checkpoint",
                "run",
                "--task",
                "hellaswag",
                "--output",
                "eval.json",
            ],
            "sml.evaluation",
            "evaluate",
        ),
        (
            [
                "finetune",
                "--checkpoint",
                "run",
                "--data",
                "swag",
                "--output",
                "ft",
            ],
            "sml.training.swag",
            "finetune",
        ),
        (
            ["export", "--checkpoint", "ft", "--output", "export"],
            "sml.training.swag",
            "export_merged",
        ),
        (
            ["verify", "--full", "run"],
            "sml.artifacts.verify",
            "verify_artifact",
        ),
    ],
)
def test_each_subcommand_lazy_dispatches_typed_config(
    argv, module_name, workflow, monkeypatch, capsys
):
    module = __import__(module_name, fromlist=[workflow])
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(path=Path("result"))

    monkeypatch.setattr(module, workflow, fake)
    if workflow == "prepare_swag_bundle":
        import sml.inference

        monkeypatch.setattr(
            sml.inference,
            "resolve_model_artifact",
            lambda path, *, full_verify: SimpleNamespace(path=path),
        )

    assert main(argv) == 0
    assert len(calls) == 1
    assert calls[0][0]
    assert not isinstance(calls[0][0][0], argparse.Namespace)
    assert "result" in capsys.readouterr().out


def test_repeated_supported_evaluation_tasks_are_preserved():
    parsed = parse_command(
        [
            "evaluate",
            "--checkpoint",
            "run",
            "--task",
            "hellaswag",
            "--task",
            "winogrande",
            "--output",
            "results.json",
        ]
    )

    assert parsed.to_domain().tasks == ("hellaswag", "winogrande")


def test_nested_dto_values_map_to_domain_policies(tmp_path):
    config = tmp_path / "finetune.toml"
    config.write_text(
        """
[finetune]
checkpoint = "base"
data = "swag"
output = "run"
microbatch_size = 3
gradient_accumulation_steps = 4
prefetch_depth = 5
learning_rate = 0.0002
checkpoint_interval = 12

[finetune.optimizer.weight_decay]
lora_a = 0.01

[finetune.lora]
rank = 8
target_modules = ["q_proj", "v_proj"]

[finetune.lora.initializer]
lora_a = 0.02
lora_b = 0.0
""".lstrip(),
        encoding="utf-8",
    )

    domain = parse_command(["finetune", "--config", str(config)]).to_domain()

    assert domain.loader.microbatch_size == 3
    assert domain.loader.gradient_accumulation_steps == 4
    assert domain.loader.prefetch_depth == 5
    assert domain.optimizer.learning_rate == 0.0002
    assert domain.optimizer.weight_decay.lora_a == 0.01
    assert domain.checkpoint.interval == 12
    assert domain.lora.rank == 8
    assert domain.lora.target_modules == ("q_proj", "v_proj")
    assert domain.lora.initializer.lora_a == 0.02


def test_prepare_swag_revision_maps_to_source(tmp_path):
    config = tmp_path / "swag.toml"
    config.write_text(
        """
[prepare.swag]
checkpoint = "base"
revision = "0123456789abcdef"
output = "swag"

[prepare.swag.source]
namespace = "example"
name = "dataset"
""".lstrip(),
        encoding="utf-8",
    )

    domain = parse_command(["prepare", "swag", "--config", str(config)]).to_domain()

    assert domain.source.revision == "0123456789abcdef"
    assert domain.source.namespace == "example"


def test_resume_pretraining_uses_only_semantic_overrides(monkeypatch):
    import sml.training.pretrain

    calls = []
    monkeypatch.setattr(
        sml.training.pretrain,
        "resume",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "resumed",
    )

    assert (
        main(
            [
                "train",
                "--resume",
                "run",
                "--maximum-steps",
                "9",
                "--checkpoint-interval",
                "2",
            ]
        )
        == 0
    )
    args, kwargs = calls[0]
    assert args == (Path("run"),)
    assert kwargs["data"] is None
    assert kwargs["overrides"].maximum_steps == 9
    assert kwargs["overrides"].checkpoint_interval == 2


@pytest.mark.parametrize(
    ("extra_args", "expected_data"),
    [
        ([], None),
        (["--data", "swag"], Path("swag")),
    ],
)
def test_resume_finetune_dispatches_optional_data(
    extra_args, expected_data, monkeypatch, capsys
):
    import sml.training.swag

    calls = []
    monkeypatch.setattr(
        sml.training.swag,
        "resume_finetune",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "resumed",
    )

    assert main(["finetune", "--resume", "run", *extra_args]) == 0
    args, kwargs = calls[0]
    assert args == (Path("run"),)
    assert kwargs["data"] == expected_data
    assert "resumed" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["train", "finetune"])
def test_resume_rejects_fresh_only_configuration(command):
    with pytest.raises(SMLConfigurationError, match="resume"):
        parse_command([command, "--resume", "run", "--output", "other"])


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--data", "data", "--output", "run", "--step", "1"],
        ["infer", "--checkpoint", "run/step-1", "prompt"],
        ["export", "--checkpoint", "step-2", "--output", "export"],
    ],
)
def test_historical_step_selection_is_rejected(argv):
    with pytest.raises(SMLConfigurationError, match="step"):
        parse_command(argv)


def test_train_rejects_non_unit_rope_scaling_factor(tmp_path):
    config = tmp_path / "train.toml"
    config.write_text(
        '[train]\ndata = "data"\noutput = "run"\n'
        "[train.model]\nrope_scaling_factor = 2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(SMLConfigurationError, match="rope_scaling_factor"):
        parse_command(["train", "--config", str(config)])


@pytest.mark.parametrize(
    "error_type",
    [SMLConfigurationError, SMLArtifactError, SMLDataError, SMLRuntimeError],
)
def test_focused_domain_errors_are_concise(error_type, monkeypatch, capsys):
    import sml.artifacts.verify

    def fail(*args, **kwargs):
        raise error_type("focused failure")

    monkeypatch.setattr(sml.artifacts.verify, "verify_artifact", fail)

    assert main(["verify", "run"]) != 0
    captured = capsys.readouterr()
    assert "focused failure" in captured.err
    assert "Traceback" not in captured.err


def test_unexpected_exceptions_propagate(monkeypatch):
    import sml.artifacts.verify

    monkeypatch.setattr(
        sml.artifacts.verify,
        "verify_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        main(["verify", "run"])
