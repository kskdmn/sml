from pathlib import Path

import pytest
from sml.cli import parse_command
from sml.errors import SMLConfigurationError


@pytest.mark.parametrize(
    "field, first, second",
    [("max_new_tokens", "7", "19"), ("include_prompt", "false", "true")],
)
def test_inference_rejects_duplicate_semantic_config_fields(
    tmp_path, field, first, second
):
    config = tmp_path / "infer.toml"
    config.write_text(
        f'[infer]\ncheckpoint="run"\nprompt="hello"\n{field}={first}\n'
        f"[infer.request]\n{field}={second}\n"
    )
    with pytest.raises(SMLConfigurationError, match="duplicate inference"):
        parse_command(["infer", "--config", str(config)])


def test_inference_cli_overrides_nested_request_config(tmp_path):
    config = tmp_path / "infer.toml"
    config.write_text(
        '[infer]\ncheckpoint="run"\nprompt="hello"\n[infer.request]\nmax_new_tokens=19\n'
    )
    command = parse_command(["infer", "--config", str(config), "--max-new-tokens", "7"])
    assert command.to_domain().request.max_new_tokens == 7


def test_inference_accepts_step_named_ancestors_and_literal_prompt():
    command = parse_command(
        ["infer", "--checkpoint", "step-experiments/run", "--", "--step"]
    )
    assert command.to_domain().prompt == "--step"
    assert command.to_domain().checkpoint == Path("step-experiments/run")


def test_tokenizer_sampling_cli_overrides_only_selected_nested_fields(tmp_path):
    config = tmp_path / "tokenize.toml"
    config.write_text(
        '[tokenize]\ninput="corpus"\noutput="tokenizer"\n'
        "[tokenize.corpus]\nmax_files=20\nmax_rows_per_file=1000\n"
        "[tokenize.sampling]\nmax_documents=100\nmax_bytes=10000\nseed=17\n"
    )
    command = parse_command(
        [
            "tokenize",
            "--config",
            str(config),
            "--max-documents",
            "50",
            "--max-corpus-bytes",
            "5000",
            "--max-files",
            "10",
        ]
    )
    domain = command.to_domain()
    assert domain.sampling.max_documents == 50
    assert domain.sampling.max_bytes == 5000
    assert domain.sampling.seed == 17
    assert domain.corpus.max_files == 10
    assert domain.corpus.max_rows_per_file == 1000


def test_pretraining_corpus_budget_is_independent_of_tokenizer_sample(tmp_path):
    config = tmp_path / "prepare.toml"
    config.write_text(
        '[prepare.pretraining]\ninput="corpus"\noutput="prepared"\ntokenizer="tok"\n'
        "[prepare.pretraining.corpus]\nmax_files=300\nmax_rows_per_file=32768\n"
    )
    domain = parse_command(
        ["prepare", "pretraining", "--config", str(config)]
    ).to_domain()
    assert domain.corpus.max_files == 300
    assert domain.corpus.max_rows_per_file == 32768


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
