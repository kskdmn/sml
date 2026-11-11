import pytest
from sml.cli import main, parse_command
from sml.errors import (
    SMLArtifactError,
    SMLConfigurationError,
    SMLDataError,
    SMLRuntimeError,
)


def test_finetune_resume_rejects_rope_override():
    with pytest.raises(SMLConfigurationError, match="rope_scaling_factor"):
        parse_command(["finetune", "--resume", "run", "--rope-scaling-factor", "1"])


@pytest.mark.parametrize("command", ("train", "finetune"))
@pytest.mark.parametrize("session_steps", ("0", "-1", str(2**31)))
def test_invalid_session_steps_fail_before_dispatch(command, session_steps):
    with pytest.raises(SMLConfigurationError, match="session_steps"):
        parse_command(
            [
                command,
                "--data",
                "data",
                "--output",
                "run",
                "--session-steps",
                session_steps,
            ]
        )


@pytest.mark.parametrize("session_steps", ("true", "1.5", '"2"'))
def test_toml_session_steps_requires_a_positive_integer(tmp_path, session_steps):
    config = tmp_path / "session.toml"
    config.write_text(
        f'[train]\ndata = "data"\noutput = "run"\nsession_steps = {session_steps}\n',
        encoding="utf-8",
    )
    with pytest.raises(SMLConfigurationError, match="session_steps"):
        parse_command(["train", "--config", str(config)])


@pytest.mark.parametrize("command", ("train", "finetune"))
@pytest.mark.parametrize("before_resume", (False, True))
@pytest.mark.parametrize(
    "extra_args",
    [
        ["--maximum-steps", "9"],
        ["--maximum-epochs", "2"],
        ["--log-interval", "1"],
        ["--checkpoint-interval", "2"],
        ["--data", "data"],
        ["--output", "other"],
        ["--session-steps", "7"],
        ["--microbatch-size", "1"],
        ["--gradient-accumulation-steps", "8"],
        ["--prefetch-depth", "2"],
        ["--learning-rate", "0.001"],
        ["--seed", "0"],
        ["--compile"],
        ["--no-compile"],
        ["--config", "missing.toml"],
        ["--maximum-steps=9"],
    ],
)
def test_resume_rejects_every_other_option(command, before_resume, extra_args):
    options = ["--resume", "run"]
    options = extra_args + options if before_resume else options + extra_args
    with pytest.raises(SMLConfigurationError, match="resume.*other options"):
        parse_command([command, *options])


@pytest.mark.parametrize("command", ("train", "finetune"))
@pytest.mark.parametrize(
    "settings", ("", "maximum_steps = 9\n", "[{command}.optimizer]\nbeta1 = 0.9\n")
)
def test_resume_rejects_configuration_files(tmp_path, command, settings):
    config = tmp_path / "resume.toml"
    config.write_text(
        f'[{command}]\nresume = "run"\n' + settings.format(command=command),
        encoding="utf-8",
    )

    with pytest.raises(SMLConfigurationError, match="resume.*config"):
        parse_command([command, "--config", str(config)])


@pytest.mark.parametrize(
    "option",
    ["--step", "--step=1", "--unknown"],
)
def test_unsupported_options_use_argument_parser_errors(option):
    with pytest.raises(SMLConfigurationError, match="unrecognized arguments"):
        parse_command(["infer", "--checkpoint", "run", "prompt", option])


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


def test_writer_lock_contention_returns_artifact_exit_code(tmp_path, capsys):
    from sml.artifacts.checkpoint import run_writer_lock

    output = tmp_path / "run"
    with run_writer_lock(output):
        assert main(["train", "--data", "unused-data", "--output", str(output)]) == 3

    captured = capsys.readouterr()
    assert "run-writer lock" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("argv", "contents", "message"),
    [
        (
            ["tokenize"],
            '[tokenize]\ninput = "corpus"\noutput = 123\n',
            "tokenize.output must be a path string",
        ),
        (
            ["prepare", "pretraining"],
            (
                '[prepare.pretraining]\ninput = "corpus"\ntokenizer = "tok"\n'
                "output = false\n"
            ),
            "prepare.pretraining.output must be a path string",
        ),
        (
            ["prepare", "swag"],
            (
                '[prepare.swag]\ncheckpoint = []\nrevision = "0123456789abcdef"\n'
                'output = "swag"\n'
            ),
            "prepare.swag.checkpoint must be a path string",
        ),
        (
            ["export"],
            '[export]\ncheckpoint = "run"\noutput = true\n',
            "export.output must be a path string",
        ),
        (
            ["verify"],
            "[verify]\npath = 42\n",
            "verify.path must be a path string",
        ),
        (
            ["verify"],
            '[verify]\npath = "run"\nfull = "yes"\n',
            "verify.full must be a boolean",
        ),
        (
            ["evaluate"],
            (
                '[evaluate]\ncheckpoint = "run"\noutput = "result.json"\n'
                'tasks = [["hellaswag"]]\n'
            ),
            "evaluation task must be hellaswag or winogrande",
        ),
    ],
)
def test_malformed_toml_values_fail_before_dispatch(
    tmp_path, monkeypatch, capsys, argv, contents, message
):
    import sml.cli

    config = tmp_path / "invalid.toml"
    config.write_text(contents, encoding="utf-8")

    def unexpected_dispatch(_command):
        pytest.fail("invalid TOML values reached workflow dispatch")

    for command_type in sml.cli._DTO_TYPES.values():
        monkeypatch.setattr(command_type, "dispatch", unexpected_dispatch)

    assert main([*argv, "--config", str(config)]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
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
