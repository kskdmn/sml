from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import zstandard as zstd

V2_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_STUBS = V2_ROOT / "tests" / "fixtures" / "provider_stubs"
RESOLVED_SWAG_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@dataclass(slots=True)
class CLIWorkspace:
    root: Path
    corpus: Path
    tokenizer: Path
    pretraining_data: Path
    base_run: Path
    base_evaluation: Path
    swag_data: Path
    lora_run: Path
    export: Path
    export_evaluation: Path
    configs: dict[str, Path]
    results: dict[str, subprocess.CompletedProcess[str]] = field(default_factory=dict)

    def run(
        self,
        *arguments: object,
        check: bool = True,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        python_paths = [str(PROVIDER_STUBS), str(V2_ROOT / "src")]
        if existing := environment.get("PYTHONPATH"):
            python_paths.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["HF_DATASETS_OFFLINE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        if extra_env:
            environment.update(extra_env)
        completed = subprocess.run(
            [sys.executable, "-m", "sml", *(str(value) for value in arguments)],
            cwd=V2_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and completed.returncode != 0:
            pytest.fail(
                f"CLI exited {completed.returncode}: {arguments!r}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed


def _toml_string(value: Path) -> str:
    return json.dumps(str(value))


def _write_config(path: Path, contents: str) -> Path:
    path.write_text(contents.lstrip(), encoding="utf-8")
    return path


def _write_corpus(path: Path) -> Path:
    path.mkdir()
    rows = [
        {
            "text": (
                "alpha beta gamma delta epsilon zeta eta theta " * 32
                + f"sample {index}"
            )
        }
        for index in range(16)
    ]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    (path / "tiny-0000.jsonl.zst").write_bytes(zstd.ZstdCompressor().compress(payload))
    return path


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory: pytest.TempPathFactory) -> CLIWorkspace:
    root = tmp_path_factory.mktemp("cli-workflows")
    corpus = _write_corpus(root / "corpus")
    tokenizer = root / "tokenizer"
    pretraining_data = root / "pretraining-data"
    base_run = root / "base-run"
    base_evaluation = root / "base-evaluation.json"
    swag_data = root / "swag-data"
    lora_run = root / "lora-run"
    export = root / "export"
    export_evaluation = root / "export-evaluation.json"

    configs = {
        "tokenize": _write_config(
            root / "tokenize.toml",
            f"""
            [tokenize]
            input = {_toml_string(corpus)}
            output = {_toml_string(tokenizer)}
            vocab_size = 300
            hard_vocab_limit = false
            num_threads = 1

            [tokenize.corpus]
            shuffle_files = false
            min_text_bytes = 1
            """,
        ),
        "prepare_pretraining": _write_config(
            root / "prepare-pretraining.toml",
            f"""
            [prepare.pretraining]
            input = {_toml_string(corpus)}
            tokenizer = {_toml_string(tokenizer)}
            output = {_toml_string(pretraining_data)}
            sequence_length = 8
            shuffle_window_rows = 16
            output_shard_rows = 8
            seed = 13

            [prepare.pretraining.corpus]
            shuffle_files = false
            min_text_bytes = 1
            """,
        ),
        "train": _write_config(
            root / "train.toml",
            f"""
            [train]
            data = {_toml_string(pretraining_data)}
            output = {_toml_string(base_run)}
            microbatch_size = 1
            gradient_accumulation_steps = 1
            prefetch_depth = 2
            checkpoint_interval = 1
            maximum_steps = 1
            maximum_epochs = 4
            log_interval = 1
            seed = 19
            compile = false

            [train.model]
            vocab_size = 300
            hidden_size = 8
            num_layers = 1
            num_q_heads = 2
            num_kv_heads = 1
            intermediate_size = 16
            original_context_length = 8
            hidden_dropout = 0.0
            """,
        ),
        "prepare_swag": _write_config(
            root / "prepare-swag.toml",
            f"""
            [prepare.swag]
            checkpoint = {_toml_string(base_run)}
            revision = "offline-swag-v1"
            output = {_toml_string(swag_data)}
            maximum_length = 8
            bucket_boundaries = [8]
            maximum_examples = 4
            """,
        ),
        "finetune": _write_config(
            root / "finetune.toml",
            f"""
            [finetune]
            checkpoint = {_toml_string(base_run)}
            data = {_toml_string(swag_data)}
            output = {_toml_string(lora_run)}
            microbatch_size = 2
            gradient_accumulation_steps = 1
            prefetch_depth = 2
            checkpoint_interval = 1
            maximum_steps = 1
            maximum_epochs = 4
            log_interval = 1
            seed = 23
            compile = false

            [finetune.lora]
            rank = 2
            alpha = 4.0
            dropout = 0.0
            target_modules = ["q_proj", "v_proj"]
            """,
        ),
    }
    return CLIWorkspace(
        root=root,
        corpus=corpus,
        tokenizer=tokenizer,
        pretraining_data=pretraining_data,
        base_run=base_run,
        base_evaluation=base_evaluation,
        swag_data=swag_data,
        lora_run=lora_run,
        export=export,
        export_evaluation=export_evaluation,
        configs=configs,
    )


def _manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _latest_step(path: Path) -> int:
    return int(json.loads((path / "latest.json").read_text(encoding="utf-8"))["step"])


@pytest.fixture(scope="module")
def completed_cli_workspace(cli_workspace: CLIWorkspace) -> CLIWorkspace:
    workspace = cli_workspace
    workspace.results["tokenize"] = workspace.run(
        "tokenize", "--config", workspace.configs["tokenize"]
    )
    workspace.results["prepare_pretraining"] = workspace.run(
        "prepare",
        "pretraining",
        "--config",
        workspace.configs["prepare_pretraining"],
    )
    workspace.results["train"] = workspace.run(
        "train", "--config", workspace.configs["train"]
    )
    workspace.results["base_infer_default"] = workspace.run(
        "infer",
        "--checkpoint",
        workspace.base_run,
        "--max-new-tokens",
        2,
        "alpha beta",
    )
    workspace.results["base_infer_full"] = workspace.run(
        "infer",
        "--checkpoint",
        workspace.base_run,
        "--full",
        "--max-new-tokens",
        2,
        "alpha beta",
    )
    workspace.results["base_evaluate"] = workspace.run(
        "evaluate",
        "--checkpoint",
        workspace.base_run,
        "--task",
        "hellaswag",
        "--output",
        workspace.base_evaluation,
    )
    workspace.results["prepare_swag"] = workspace.run(
        "prepare", "swag", "--config", workspace.configs["prepare_swag"]
    )
    workspace.results["finetune"] = workspace.run(
        "finetune", "--config", workspace.configs["finetune"]
    )
    workspace.results["export"] = workspace.run(
        "export",
        "--checkpoint",
        workspace.lora_run,
        "--output",
        workspace.export,
    )
    workspace.results["export_infer"] = workspace.run(
        "infer",
        "--checkpoint",
        workspace.export,
        "--max-new-tokens",
        2,
        "alpha beta",
    )
    workspace.results["export_evaluate"] = workspace.run(
        "evaluate",
        "--checkpoint",
        workspace.export,
        "--full",
        "--task",
        "winogrande",
        "--output",
        workspace.export_evaluation,
    )
    workspace.results["verify"] = workspace.run("verify", "--full", workspace.export)
    workspace.results["train_resume"] = workspace.run(
        "train",
        "--resume",
        workspace.base_run,
        "--maximum-steps",
        2,
        "--checkpoint-interval",
        1,
        "--log-interval",
        1,
    )
    workspace.results["finetune_resume"] = workspace.run(
        "finetune",
        "--resume",
        workspace.lora_run,
        "--maximum-steps",
        2,
        "--checkpoint-interval",
        1,
        "--log-interval",
        1,
    )
    return workspace


def test_every_cli_workflow_runs_offline_in_subprocesses(
    completed_cli_workspace: CLIWorkspace,
) -> None:
    workspace = completed_cli_workspace

    assert _manifest(workspace.tokenizer)["kind"] == "tokenizer"
    assert _manifest(workspace.pretraining_data)["kind"] == "pretraining-data"
    assert _latest_step(workspace.base_run) == 2
    assert _manifest(workspace.swag_data)["kind"] == "swag-data"
    assert _manifest(workspace.swag_data)["source"]["commit"] == RESOLVED_SWAG_COMMIT
    assert _latest_step(workspace.lora_run) == 2
    assert _manifest(workspace.export)["kind"] == "export"

    base_evaluation = json.loads(workspace.base_evaluation.read_text(encoding="utf-8"))
    assert base_evaluation["model"]["artifact_kind"] == "pretraining-run"
    assert base_evaluation["model"]["step"] == 1
    assert base_evaluation["model"]["verification"] == "manifest-trusted"
    assert base_evaluation["tasks"] == ["hellaswag"]

    export_evaluation = json.loads(
        workspace.export_evaluation.read_text(encoding="utf-8")
    )
    assert export_evaluation["model"]["artifact_kind"] == "export"
    assert export_evaluation["model"]["verification"] == "full"
    assert export_evaluation["tasks"] == ["winogrande"]
    assert "manifest-trusted" in workspace.results["base_infer_default"].stdout
    assert "VerificationLevel.FULL" in workspace.results["base_infer_full"].stdout
    assert "artifact_kind='export'" in workspace.results["export_infer"].stdout
    assert "VerificationLevel.FULL" in workspace.results["verify"].stdout


def test_cli_resume_accepts_relocated_identity_matching_data(
    completed_cli_workspace: CLIWorkspace,
) -> None:
    workspace = completed_cli_workspace
    relocated_root = workspace.root / "relocated"
    relocated_root.mkdir()

    moved_base_run = relocated_root / "base-run"
    shutil.copytree(workspace.base_run, moved_base_run)
    relocated_pretraining = relocated_root / "pretraining-data"
    workspace.pretraining_data.rename(relocated_pretraining)
    assert not workspace.pretraining_data.exists()

    try:
        workspace.run(
            "train",
            "--resume",
            moved_base_run,
            "--data",
            relocated_pretraining,
            "--maximum-steps",
            3,
            "--checkpoint-interval",
            1,
            "--log-interval",
            1,
        )
    finally:
        relocated_pretraining.rename(workspace.pretraining_data)
    assert _latest_step(moved_base_run) == 3

    moved_lora_run = relocated_root / "lora-run"
    shutil.copytree(workspace.lora_run, moved_lora_run)
    relocated_swag = relocated_root / "swag-data"
    workspace.swag_data.rename(relocated_swag)
    assert not workspace.swag_data.exists()

    try:
        workspace.run(
            "finetune",
            "--resume",
            moved_lora_run,
            "--data",
            relocated_swag,
            "--maximum-steps",
            3,
            "--checkpoint-interval",
            1,
            "--log-interval",
            1,
        )
    finally:
        relocated_swag.rename(workspace.swag_data)
    assert _latest_step(moved_lora_run) == 3


def test_cli_expected_domain_errors_have_stable_exit_codes(
    completed_cli_workspace: CLIWorkspace,
) -> None:
    workspace = completed_cli_workspace
    latest_step_path = next((workspace.base_run / "checkpoints").glob("step-*"))
    cases = [
        (
            workspace.run(
                "infer",
                "--checkpoint",
                workspace.base_run,
                "--step",
                1,
                "alpha",
                check=False,
            ),
            2,
            "SMLConfigurationError",
        ),
        (
            workspace.run(
                "infer",
                "--checkpoint",
                latest_step_path,
                "alpha",
                check=False,
            ),
            2,
            "SMLConfigurationError",
        ),
        (
            workspace.run("verify", workspace.root / "missing", check=False),
            3,
            "SMLArtifactError",
        ),
        (
            workspace.run(
                "prepare",
                "swag",
                "--checkpoint",
                workspace.base_run,
                "--revision",
                "offline-failure",
                "--output",
                workspace.root / "failed-swag",
                check=False,
                extra_env={"SML_TEST_DATASETS_FAIL": "1"},
            ),
            4,
            "SMLDataError",
        ),
        (
            workspace.run(
                "infer",
                "--checkpoint",
                workspace.base_run,
                "--max-new-tokens",
                1,
                "alpha " * 128,
                check=False,
            ),
            5,
            "SMLRuntimeError",
        ),
    ]

    for completed, exit_code, error_name in cases:
        assert completed.returncode == exit_code
        assert completed.stderr.startswith(f"{error_name}:")
        assert "Traceback" not in completed.stderr
