from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import pytest
import zstandard as zstd
from sml.artifacts.manifest import VerificationLevel
from sml.evaluation import read_evaluation_result
from sml.evaluation_result import normalize_json_value
from sml.inference import ModelIdentity, resolve_model_artifact

V2_ROOT = Path(__file__).resolve().parents[2]
PROVIDER_STUBS = V2_ROOT / "tests" / "fixtures" / "provider_stubs"
RESOLVED_SWAG_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _expected_provider_config(task_name: str) -> dict[str, object]:
    return {
        "task": task_name,
        "metadata": {"version": 1.0},
        "output_type": "loglikelihood",
        "description": "offline task",
        "process_docs": "offline.process_docs",
        "doc_to_text": "offline.doc_to_text",
        "doc_to_target": "offline.doc_to_target",
        "doc_to_choice": None,
        "target_delimiter": " ",
        "fewshot_delimiter": "\n\n",
        "gen_prefix": "",
        "padding": "left",
        "num_fewshot": 0,
        "fewshot_split": "train",
        "fewshot_config": {
            "split": "fewshot-config",
            "process_docs": "offline.serialized._offline_process_docs",
            "doc_to_text": "offline.serialized._offline_doc_to_text",
            "doc_to_target": "offline.serialized._offline_doc_to_target",
            "doc_to_choice": "offline.serialized._offline_doc_to_choice",
        },
        "generation_kwargs": {"temperature": 0.0},
        "metric_list": [{"metric": "acc"}],
        "repeats": 1,
        "should_decontaminate": False,
        "doc_to_decontamination_query": None,
        "validation_split": "validation",
        "test_split": "test",
        "dataset_kwargs": {},
    }


def _expected_provider_samples(task_name: str) -> dict[str, object]:
    return {
        task_name: [
            {
                "request_type": "loglikelihood",
                "doc_id": 2,
                "repeats": 1,
                "arguments": ["alpha", " beta"],
            },
            {
                "request_type": "loglikelihood",
                "doc_id": 1,
                "repeats": 1,
                "arguments": ["gamma", " delta"],
            },
            {
                "request_type": "generate_until",
                "doc_id": 3,
                "repeats": 1,
                "arguments": [
                    "alpha",
                    {"max_gen_toks": 1, "until": ["omega"]},
                ],
            },
        ]
    }


def _expected_provider_versions() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (name, metadata.version(name))
            for name in ("datasets", "lm-eval", "mlx", "numpy", "sentencepiece")
        )
    )


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _assert_complete_evaluation(
    path: Path,
    *,
    task_name: str,
    artifact_kind: str,
    verification: VerificationLevel,
    step: int,
    run_identity: bool,
    expected_model: ModelIdentity,
) -> None:
    result = read_evaluation_result(path)
    assert result.kind == "evaluation-result"
    assert result.version == 1
    assert result.identity.startswith("sha256:")
    assert result.model == expected_model
    assert result.model.artifact_kind == artifact_kind
    if run_identity:
        assert result.model.run_identity is not None
        assert result.model.run_identity.startswith("sha256:")
    else:
        assert result.model.run_identity is None
    assert result.model.step == step
    if run_identity:
        assert result.model.checkpoint_identity is not None
        assert result.model.checkpoint_identity.startswith("sha256:")
        assert result.model.run_step_identity is not None
        assert result.model.run_step_identity.startswith("sha256:")
    else:
        assert result.model.checkpoint_identity is None
        assert result.model.run_step_identity is None
    assert result.model.tokenizer_identity.startswith("sha256:")
    assert result.model.verification is verification

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task.task_name == task_name
    assert task.task_identity.startswith("sha256:")
    assert task.task_yaml.logical_name == f"tasks/{task_name}.yaml"
    assert task.task_yaml.content_identity.startswith("sha256:")
    assert [source.logical_name for source in task.include_template_closure] == [
        "tasks/common.yaml"
    ]
    assert all(
        source.content_identity.startswith("sha256:")
        for source in task.include_template_closure
    )
    assert task.task_metadata_version == "1.0"
    assert task.prompt_config == {
        "adapter_padding": "right",
        "apply_chat_template": False,
        "description": "offline task",
        "doc_to_choice": None,
        "doc_to_target": "offline.doc_to_target",
        "doc_to_text": "offline.doc_to_text",
        "fewshot_delimiter": "\n\n",
        "gen_prefix": "",
        "output_type": "loglikelihood",
        "process_docs": "offline.process_docs",
        "system_instruction": None,
        "target_delimiter": " ",
    }
    assert task.few_shot_config == {
        "fewshot_as_multiturn": True,
        "fewshot_config": _expected_provider_config(task_name)["fewshot_config"],
        "fewshot_split": "train",
        "num_fewshot": 0,
    }
    assert task.generation_config == {
        "generation_kwargs": {"temperature": 0.0},
        "provider_gen_kwargs": None,
    }
    assert task.metric_normalization_config == {
        "bootstrap_iters": 100000,
        "doc_to_decontamination_query": None,
        "filter_list": ({"filter": ({"function": "take_first"},), "name": "none"},),
        "log_samples": True,
        "metric_list": ({"metric": "acc"},),
        "predict_only": False,
        "repeats": 1,
        "should_decontaminate": False,
    }
    assert task.seeds == {
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
    }
    assert task.limit is None
    assert task.ordered_request_identity.startswith("sha256:")
    assert task.lm_eval_package_version == "0.4.12"
    assert task.lm_eval_source_commit is None
    assert task.dataset_revision == "version:1.0.0"
    assert task.dataset_fingerprint.startswith("sha256:")
    assert tuple((item.name, item.version) for item in task.provider_versions) == (
        _expected_provider_versions()
    )
    assert "acc,none" in task.metric_payload
    expected_provider_result = {
        "results": {task_name: task.metric_payload},
        "configs": {task_name: _expected_provider_config(task_name)},
        "versions": {task_name: "1.0.0"},
        "n-shot": {task_name: 0},
        "higher_is_better": {task_name: {"acc,none": True}},
        "n-samples": {task_name: {"effective": 2, "original": 2}},
        "samples": _expected_provider_samples(task_name),
        "config": {"bootstrap_iters": 100000, "log_samples": True},
        "git_hash": "offline-lm-eval-commit",
        "date": "2026-08-25T00:00:00Z",
    }
    assert normalize_json_value(
        result.provider_result, context="persisted provider result"
    ) == normalize_json_value(
        expected_provider_result, context="expected provider result"
    )


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
    lora_evaluation: Path
    export: Path
    export_evaluation: Path
    configs: dict[str, Path]
    results: dict[str, subprocess.CompletedProcess[str]] = field(default_factory=dict)
    evaluation_models: dict[str, ModelIdentity] = field(default_factory=dict)

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
    lora_evaluation = root / "lora-evaluation.json"
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
        lora_evaluation=lora_evaluation,
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
    workspace.evaluation_models["base"] = resolve_model_artifact(
        workspace.base_run, full_verify=False
    ).identity()
    workspace.results["prepare_swag"] = workspace.run(
        "prepare", "swag", "--config", workspace.configs["prepare_swag"]
    )
    workspace.results["finetune"] = workspace.run(
        "finetune", "--config", workspace.configs["finetune"]
    )
    workspace.results["lora_evaluate"] = workspace.run(
        "evaluate",
        "--checkpoint",
        workspace.lora_run,
        "--task",
        "hellaswag",
        "--output",
        workspace.lora_evaluation,
    )
    workspace.evaluation_models["lora"] = resolve_model_artifact(
        workspace.lora_run, full_verify=False
    ).identity()
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
    workspace.evaluation_models["export"] = resolve_model_artifact(
        workspace.export, full_verify=True
    ).identity()
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

    _assert_complete_evaluation(
        workspace.base_evaluation,
        task_name="hellaswag",
        artifact_kind="pretraining-run",
        verification=VerificationLevel.MANIFEST_TRUSTED,
        step=1,
        run_identity=True,
        expected_model=workspace.evaluation_models["base"],
    )
    assert (
        str(workspace.base_evaluation.parent).encode()
        not in workspace.base_evaluation.read_bytes()
    )

    _assert_complete_evaluation(
        workspace.lora_evaluation,
        task_name="hellaswag",
        artifact_kind="lora-run",
        verification=VerificationLevel.MANIFEST_TRUSTED,
        step=1,
        run_identity=True,
        expected_model=workspace.evaluation_models["lora"],
    )
    assert (
        str(workspace.lora_evaluation.parent).encode()
        not in workspace.lora_evaluation.read_bytes()
    )

    _assert_complete_evaluation(
        workspace.export_evaluation,
        task_name="winogrande",
        artifact_kind="export",
        verification=VerificationLevel.FULL,
        step=1,
        run_identity=False,
        expected_model=workspace.evaluation_models["export"],
    )
    assert (
        str(workspace.export_evaluation.parent).encode()
        not in workspace.export_evaluation.read_bytes()
    )
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
