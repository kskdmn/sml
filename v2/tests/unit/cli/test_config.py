from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.cli import parse_command


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


def test_integer_rope_scale_in_toml_produces_a_valid_run_manifest(tmp_path):
    from sml.training.pretrain import _run_manifest

    config = tmp_path / "train.toml"
    config.write_text(
        '[train]\ndata = "data"\noutput = "run"\n'
        "[train.model]\nrope_scaling_factor = 1\n",
        encoding="utf-8",
    )
    domain = parse_command(["train", "--config", str(config)]).to_domain()
    data = SimpleNamespace(
        manifest=SimpleNamespace(
            tokenizer_identity="sha256:" + "1" * 64,
            identity="sha256:" + "2" * 64,
        )
    )

    manifest = _run_manifest(domain, data)

    assert manifest.model["rope_scaling_factor"] == 1.0
    assert manifest.identity == manifest.recompute_identity()


def test_finetune_defaults_to_eightfold_context_and_preserves_batch_defaults():
    domain = parse_command(
        ["finetune", "--checkpoint", "base", "--data", "swag", "--output", "ft"]
    ).to_domain()
    assert domain.rope_scaling_factor == 8.0
    assert domain.loader.microbatch_size == 1
    assert domain.loader.gradient_accumulation_steps == 8


@pytest.mark.parametrize("command", ["finetune", "prepare.swag", "infer", "evaluate"])
def test_rope_scale_cli_overrides_toml(tmp_path, command):
    required = {
        "finetune": 'checkpoint = "base"\ndata = "swag"\noutput = "ft"\n',
        "prepare.swag": 'checkpoint = "base"\nrevision = "main"\noutput = "swag"\n',
        "infer": 'checkpoint = "base"\nprompt = "hello"\n',
        "evaluate": 'checkpoint = "base"\ntasks = ["hellaswag"]\noutput = "eval.json"\n',
    }
    runtime = command in {"infer", "evaluate"}
    config = tmp_path / "config.toml"
    config.write_text(
        f"[{command}]\n"
        + required[command]
        + (f"[{command}.runtime]\n" if runtime else "")
        + "rope_scaling_factor = 2.0\n",
        encoding="utf-8",
    )
    argv = command.split(".") + ["--config", str(config)]
    configured = parse_command(argv).to_domain()
    overridden = parse_command(argv + ["--rope-scaling-factor", "4"]).to_domain()
    if runtime:
        configured, overridden = configured.runtime, overridden.runtime
    assert configured.rope_scaling_factor == 2.0
    assert overridden.rope_scaling_factor == 4.0


def test_prepare_swag_defaults_cover_extended_context():
    domain = parse_command(
        [
            "prepare",
            "swag",
            "--checkpoint",
            "base",
            "--revision",
            "main",
            "--output",
            "swag",
        ]
    ).to_domain()
    assert domain.rope_scaling_factor == 8.0
    assert domain.maximum_length == 8_192
    assert domain.bucket_boundaries == (64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192)


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


@pytest.mark.parametrize("prompt", ["--step", "--step=1"])
def test_prompt_after_option_terminator_is_preserved(prompt):
    config = parse_command(["infer", "--checkpoint", "run", "--", prompt]).to_domain()
    assert config.prompt == prompt


@pytest.mark.parametrize(
    "argv,field,expected",
    [
        (
            ["infer", "--checkpoint", "step-experiment", "prompt"],
            "checkpoint",
            "step-experiment",
        ),
        (
            [
                "export",
                "--checkpoint",
                "checkpoints/step-experiment/run",
                "--output",
                "export",
            ],
            "checkpoint",
            "checkpoints/step-experiment/run",
        ),
        (["train", "--resume", "step-experiment"], "resume", "step-experiment"),
    ],
)
def test_artifact_path_names_do_not_determine_their_kind(argv, field, expected):
    assert parse_command(argv).values[field] == Path(expected)
