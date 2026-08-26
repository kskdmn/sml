from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

__version__ = "0.4.12"


class LM:
    def __init__(self) -> None:
        pass


api = ModuleType("lm_eval.api")
model = ModuleType("lm_eval.api.model")
model.LM = LM
api.model = model
sys.modules["lm_eval.api"] = api
sys.modules["lm_eval.api.model"] = model


class _Request:
    def __init__(
        self,
        args: tuple[object, ...],
        *,
        task_name: str,
        doc_id: int,
        repeats: int,
    ) -> None:
        self.args = args
        self.task_name = task_name
        self.doc_id = doc_id
        self.repeats = repeats


@dataclass(frozen=True, slots=True)
class _Info:
    version: str


@dataclass(frozen=True, slots=True)
class _Split:
    _fingerprint: str
    info: _Info


class _Config:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def to_dict(self) -> dict[str, object]:
        return dict(self._values)

    def serialize_function(self, value: object) -> str:
        name = getattr(value, "__name__", type(value).__name__)
        return f"offline.serialized.{name}"


class _Task:
    def __init__(self, task_name: str, values: dict[str, object]) -> None:
        self.task_name = task_name
        self.config = _Config(values)
        self.instances = (
            _Request(
                ("alpha", " beta"),
                task_name=task_name,
                doc_id=2,
                repeats=1,
            ),
            _Request(
                ("gamma", " delta"),
                task_name=task_name,
                doc_id=1,
                repeats=1,
            ),
        )
        self.dataset = {
            "validation": _Split("validation-fingerprint", _Info("1.0.0")),
            "test": _Split("test-fingerprint", _Info("2.0.0")),
            "train": _Split("train-fingerprint", _Info("1.0.0")),
            "fewshot-config": _Split("fewshot-config-fingerprint", _Info("1.0.0")),
        }
        # 0.4.12 fingerprints the objects it actually iterates: test_docs()
        # applies process_docs and fewshot_docs() applies the few-shot processor.
        # Keep these identities distinct from the unprocessed dataset mapping so
        # tests reject provenance collected from task.dataset directly.
        self.eval_docs = _Split("processed-test-fingerprint", _Info("1.0.0"))
        self._fewshot_docs = _Split(
            "processed-fewshot-config-fingerprint", _Info("1.0.0")
        )

    def fewshot_docs(self) -> _Split:
        return self._fewshot_docs


@dataclass(frozen=True, slots=True)
class _Entry:
    yaml_path: Path | None


class TaskManager:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parent / "tasks"
        self.task_index = {
            "hellaswag": _Entry(root / "hellaswag.yaml"),
            "winogrande": _Entry(root / "winogrande.yaml"),
        }

    def load(self, task_list: list[str]) -> dict[str, object]:
        tasks = {name: _Task(name, _task_config(name)) for name in task_list}
        return {"tasks": tasks, "groups": {}, "group_map": {}}


def _task_config(task_name: str) -> dict[str, object]:
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
        "num_fewshot": 1,
        "fewshot_split": "train",
        "fewshot_config": {
            "split": "fewshot-config",
            "process_docs": _offline_process_docs,
            "doc_to_text": _offline_doc_to_text,
            "doc_to_target": _offline_doc_to_target,
            "doc_to_choice": _offline_doc_to_choice,
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


def _offline_process_docs(value: object) -> object:
    return value


def _offline_doc_to_text(value: object) -> str:
    return str(value)


def _offline_doc_to_target(value: object) -> str:
    return str(value)


def _offline_doc_to_choice(value: object) -> list[str]:
    return [str(value)]


def load_yaml(
    path: str | Path,
    *,
    resolve_func: bool = True,
    recursive: bool = True,
) -> dict[str, Any]:
    del resolve_func, recursive
    value = yaml.safe_load(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise TypeError("offline YAML must contain a mapping")
    return value


tasks_module = ModuleType("lm_eval.tasks")
tasks_module.TaskManager = TaskManager
yaml_loader = ModuleType("lm_eval.tasks._yaml_loader")
yaml_loader.load_yaml = load_yaml
tasks_module._yaml_loader = yaml_loader
sys.modules["lm_eval.tasks"] = tasks_module
sys.modules["lm_eval.tasks._yaml_loader"] = yaml_loader
tasks = tasks_module


def simple_evaluate(
    *,
    model: LM,
    tasks: list[str],
    num_fewshot: int,
    limit: int | None,
    log_samples: bool,
    task_manager: TaskManager | None = None,
    **_kwargs: object,
) -> dict[str, object]:
    if not isinstance(model, LM):
        raise TypeError("offline lm-eval requires an LM adapter")
    if num_fewshot != 0 or not isinstance(log_samples, bool):
        raise RuntimeError("unexpected offline lm-eval policy")
    if limit is not None and limit <= 0:
        raise RuntimeError("offline lm-eval limit must be positive")

    manager = task_manager or TaskManager()
    loaded = manager.load(tasks)
    loaded_tasks = loaded["tasks"]
    if not isinstance(loaded_tasks, dict):
        raise TypeError("offline lm-eval tasks are malformed")
    for loaded_task in loaded_tasks.values():
        if not isinstance(loaded_task, _Task):
            raise TypeError("offline lm-eval task is malformed")
        if loaded_task.config._values["num_fewshot"] != 0:
            loaded_task.config._values["num_fewshot"] = num_fewshot
    results: dict[str, object] = {}
    samples: dict[str, object] = {}
    for task, loaded_task in loaded_tasks.items():
        scored = model.loglikelihood(list(loaded_task.instances))
        generation_requests = [
            _Request(
                ("alpha", {"max_gen_toks": 1, "until": ["omega"]}),
                task_name=task,
                doc_id=3,
                repeats=1,
            )
        ]
        generated = model.generate_until(generation_requests)
        results[task] = {
            "acc,none": sum(bool(item[1]) for item in scored) / len(scored),
            "generated": generated,
        }
        if log_samples:
            samples[task] = [
                {
                    "request_type": "loglikelihood",
                    "doc_id": request.doc_id,
                    "repeats": request.repeats,
                    "arguments": list(request.args),
                }
                for request in loaded_task.instances
            ] + [
                {
                    "request_type": "generate_until",
                    "doc_id": request.doc_id,
                    "repeats": request.repeats,
                    "arguments": list(request.args),
                }
                for request in generation_requests
            ]
    result: dict[str, object] = {
        "results": results,
        "configs": {
            task: loaded_task.config.to_dict()
            for task, loaded_task in loaded_tasks.items()
        },
        "versions": {task: "1.0.0" for task in tasks},
        "n-shot": {
            task: loaded_task.config.to_dict()["num_fewshot"]
            for task, loaded_task in loaded_tasks.items()
        },
        "higher_is_better": {task: {"acc,none": True} for task in tasks},
        "n-samples": {task: {"effective": 2, "original": 2} for task in tasks},
        "config": {"bootstrap_iters": 100000, "log_samples": log_samples},
        "git_hash": "offline-lm-eval-commit",
        "date": "2026-08-25T00:00:00Z",
    }
    if log_samples:
        result["samples"] = samples
    return result
