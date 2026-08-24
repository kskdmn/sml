from pathlib import Path

import pytest
from sml.cli import parse_command
from sml.errors import SMLConfigurationError


@pytest.mark.parametrize(
    ("argv", "contents"),
    [
        (
            ["train"],
            '[infer]\ncheckpoint = "run"\nprompt = "hello"\n',
        ),
        (
            ["prepare", "pretraining"],
            '[prepare]\ninput = "corpus"\ntokenizer = "tok"\noutput = "data"\n',
        ),
        (
            ["verify"],
            '[verify]\npath = "run"\n[train]\ndata = "data"\noutput = "other"\n',
        ),
    ],
)
def test_config_rejects_wrong_or_sibling_root_tables(tmp_path, argv, contents):
    config = tmp_path / "config.toml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(SMLConfigurationError, match="table"):
        parse_command([*argv, "--config", str(config)])


@pytest.mark.parametrize(
    "contents",
    [
        '[train]\ndata = "data"\noutput = "run"\nunknown = 1\n',
        ('[train]\ndata = "data"\noutput = "run"\n[train.optimizer]\nunknown = 1\n'),
        (
            '[train]\ndata = "data"\noutput = "run"\n'
            "[train.loader]\nmicrobatch_size = 2\n"
        ),
        ('[train]\ndata = "data"\noutput = "run"\n[train.checkpoint]\ninterval = 2\n'),
    ],
)
def test_config_rejects_unknown_or_duplicate_nested_fields(tmp_path, contents):
    config = tmp_path / "train.toml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(SMLConfigurationError, match="unknown|flat|not allowed"):
        parse_command(["train", "--config", str(config)])


def test_config_accepts_all_documented_command_tables(tmp_path):
    cases = [
        ("tokenize", ["tokenize"], '[tokenize]\ninput = "corpus"\noutput = "tok"\n'),
        (
            "prepare-pretraining",
            ["prepare", "pretraining"],
            (
                '[prepare.pretraining]\ninput = "corpus"\ntokenizer = "tok"\n'
                'output = "data"\n'
            ),
        ),
        (
            "prepare-swag",
            ["prepare", "swag"],
            (
                '[prepare.swag]\ncheckpoint = "base"\n'
                'revision = "0123456789abcdef"\noutput = "swag"\n'
            ),
        ),
        ("train", ["train"], '[train]\ndata = "data"\noutput = "run"\n'),
        (
            "infer",
            ["infer"],
            '[infer]\ncheckpoint = "run"\nprompt = "hello"\n',
        ),
        (
            "evaluate",
            ["evaluate"],
            (
                '[evaluate]\ncheckpoint = "run"\ntasks = ["hellaswag"]\n'
                'output = "eval.json"\n'
            ),
        ),
        (
            "finetune",
            ["finetune"],
            ('[finetune]\ncheckpoint = "base"\ndata = "swag"\noutput = "ft"\n'),
        ),
        (
            "export",
            ["export"],
            '[export]\ncheckpoint = "ft"\noutput = "merged"\n',
        ),
        ("verify", ["verify"], '[verify]\npath = "run"\nfull = true\n'),
    ]

    for name, argv, contents in cases:
        config = tmp_path / f"{name}.toml"
        config.write_text(contents, encoding="utf-8")
        parsed = parse_command([*argv, "--config", str(config)])
        assert parsed is not None


def test_cli_value_overrides_nested_toml_mapping(tmp_path):
    config = tmp_path / "finetune.toml"
    config.write_text(
        """
[finetune]
checkpoint = "base"
data = "swag"
output = "run"
microbatch_size = 2
learning_rate = 0.0002

[finetune.lora]
rank = 4
""".lstrip(),
        encoding="utf-8",
    )

    domain = parse_command(
        [
            "finetune",
            "--config",
            str(config),
            "--microbatch-size",
            "7",
            "--learning-rate",
            "0.0003",
        ]
    ).to_domain()

    assert domain.base_checkpoint == Path("base")
    assert domain.loader.microbatch_size == 7
    assert domain.optimizer.learning_rate == 0.0003
    assert domain.lora.rank == 4


def test_inference_request_tables_map_to_domain_fields(tmp_path):
    config = tmp_path / "infer.toml"
    config.write_text(
        """
[infer]
checkpoint = "run"
prompt = "hello"

[infer.request]
max_new_tokens = 17
include_prompt = true

[infer.request.config]
temperature = 0.5
top_p = 0.8
""".lstrip(),
        encoding="utf-8",
    )

    domain = parse_command(["infer", "--config", str(config)]).to_domain()

    assert domain.request.max_new_tokens == 17
    assert domain.request.include_prompt is True
    assert domain.request.config.temperature == 0.5
    assert domain.request.config.top_p == 0.8


def test_invalid_evaluation_task_is_rejected(tmp_path):
    config = tmp_path / "evaluate.toml"
    config.write_text(
        ('[evaluate]\ncheckpoint = "run"\ntasks = ["unknown"]\noutput = "eval.json"\n'),
        encoding="utf-8",
    )

    with pytest.raises(SMLConfigurationError, match="task"):
        parse_command(["evaluate", "--config", str(config)])


def test_fresh_run_output_collision_is_rejected(tmp_path):
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(SMLConfigurationError, match="already exists"):
        parse_command(["train", "--data", "data", "--output", str(output)])


def test_config_must_exist(tmp_path):
    with pytest.raises(SMLConfigurationError, match="config"):
        parse_command(["verify", "--config", str(tmp_path / "missing.toml"), "run"])
