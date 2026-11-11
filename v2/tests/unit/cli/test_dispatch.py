import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.cli import main, parse_command
from sml.errors import (
    SMLArtifactError,
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


@pytest.mark.parametrize("command", ("train", "finetune"))
def test_resume_dispatches_only_the_run_path(command, monkeypatch, capsys):
    module_name = "sml.training.pretrain" if command == "train" else "sml.training.swag"
    workflow = "resume" if command == "train" else "resume_finetune"
    module = __import__(module_name, fromlist=[workflow])
    calls = []
    monkeypatch.setattr(
        module,
        workflow,
        lambda *args, **kwargs: calls.append((args, kwargs)) or "resumed",
    )

    assert main([command, "--resume", "run"]) == 0
    assert calls == [((Path("run"),), {})]
    assert "resumed" in capsys.readouterr().out


@pytest.mark.parametrize("command", ("train", "finetune"))
def test_session_steps_dispatches_without_changing_saved_training_budget(
    tmp_path, monkeypatch, command
):
    module_name = "sml.training.pretrain" if command == "train" else "sml.training.swag"
    workflow = "train" if command == "train" else "finetune"
    module = __import__(module_name, fromlist=[workflow])
    calls = []
    monkeypatch.setattr(
        module,
        workflow,
        lambda *args, **kwargs: calls.append((args, kwargs)) or "paused",
    )
    argv = [command, "--maximum-steps", "100", "--session-steps", "7"]
    argv += ["--data", str(tmp_path / "data"), "--output", str(tmp_path / "run")]
    if command == "finetune":
        argv += ["--checkpoint", str(tmp_path / "base")]

    parsed = parse_command(argv)
    domain = parsed.to_domain()
    assert domain.maximum_steps == 100
    assert not hasattr(domain, "session_steps")
    assert parsed.dispatch() == "paused"
    assert calls[0][1]["session_steps"] == 7


@pytest.mark.parametrize("command", ("train", "finetune"))
def test_toml_session_steps_can_be_overridden_without_changing_total_budget(
    tmp_path, monkeypatch, command
):
    config = tmp_path / "session.toml"
    config.write_text(
        f'[{command}]\ndata = "data"\noutput = "run"\n'
        + ('checkpoint = "base"\n' if command == "finetune" else "")
        + "maximum_steps = 100\nsession_steps = 3\n",
        encoding="utf-8",
    )
    module_name = "sml.training.pretrain" if command == "train" else "sml.training.swag"
    workflow = "train" if command == "train" else "finetune"
    module = __import__(module_name, fromlist=[workflow])
    calls = []
    monkeypatch.setattr(
        module,
        workflow,
        lambda *args, **kwargs: calls.append((args, kwargs)) or "paused",
    )
    assert (
        parse_command(
            [command, "--config", str(config), "--session-steps", "7"]
        ).dispatch()
        == "paused"
    )
    assert calls[0][1]["session_steps"] == 7
    assert calls[0][0][0].maximum_steps == 100
