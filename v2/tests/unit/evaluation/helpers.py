from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from sml.errors import SMLRuntimeError
from sml.model.config import ModelConfig


def _snapshot_fifo_source(path: Path, package_root: Path, outcomes) -> None:
    from sml import evaluation

    try:
        evaluation._source_snapshot(path, package_root)
    except SMLRuntimeError:
        outcomes.put("rejected")
    else:
        outcomes.put("accepted")


def _copied_task_source_provider(fake_provider, tmp_path: Path):
    root = tmp_path / "lm_eval"
    tasks = root / "tasks"
    shutil.copytree(fake_provider.package_root / "tasks", tasks)

    class CopiedSourceTaskManager(fake_provider.task_manager_type):
        def __init__(self) -> None:
            super().__init__()
            for task_name in ("hellaswag", "winogrande"):
                self.task_index[task_name] = SimpleNamespace(
                    yaml_path=tasks / f"{task_name}.yaml"
                )

    return (
        replace(
            fake_provider,
            task_manager_type=CopiedSourceTaskManager,
            package_root=root,
        ),
        tasks,
    )


def _task_record(
    fake_provider,
    *,
    provider_result: Mapping[str, object] | None = None,
):
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    tuple(task.doc_iterator())
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    return evaluation._resolve_task_record(
        model_config=ModelConfig(),
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result=provider_result
        or {"results": {"hellaswag": {"acc,none": 0.5}}, "git_hash": "abc123"},
        limit=2,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )


class _DuplicateMetricMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        if key != "acc,none":
            raise KeyError(key)
        return 0.5

    def __iter__(self) -> Iterator[str]:
        return iter(("acc,none", "acc,none"))

    def __len__(self) -> int:
        return 2

    def items(self):
        return (("acc,none", 0.5), ("acc,none", 0.75))


class _DuplicateIterationMapping(Mapping[str, object]):
    """A hostile Mapping whose sole key iteration contains duplicates."""

    def __init__(self, entries: tuple[tuple[str, object], ...]) -> None:
        self._entries = entries

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
