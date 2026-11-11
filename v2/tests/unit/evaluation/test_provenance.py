from __future__ import annotations

import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError
from sml.inference import InferenceSession
from sml.model.config import ModelConfig

from .helpers import (
    _DuplicateMetricMapping,
    _snapshot_fifo_source,
    _task_record,
)


def test_task_provenance_serializes_nested_provider_callables(fake_provider) -> None:
    record = _task_record(fake_provider)
    nested = record.few_shot_config["fewshot_config"]
    assert nested["process_docs"] == "offline.serialized._offline_process_docs"
    assert nested["doc_to_text"] == "offline.serialized._offline_doc_to_text"
    assert nested["doc_to_target"] == "offline.serialized._offline_doc_to_target"
    assert nested["doc_to_choice"] == "offline.serialized._offline_doc_to_choice"


def test_task_provenance_prefers_effective_fewshot_config_split(fake_provider) -> None:
    from sml import evaluation
    from sml.artifacts.manifest import structured_identity

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    tuple(task.doc_iterator())
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    record = evaluation._resolve_task_record(
        model_config=ModelConfig(),
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
        limit=None,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )
    assert record.dataset_fingerprint == structured_identity(
        "sml-evaluation-dataset-v1",
        (
            ("test", "evaluated-test-fingerprint"),
            ("fewshot-config", "retained-fewshot-config-fingerprint"),
        ),
    )


def test_offline_provider_task4_invocation_records_effective_zero_shot_provenance(
    fake_provider,
    tiny_session: InferenceSession,
) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    recorder = evaluation._EvaluationRequestRecorder()
    raw_result = fake_provider.simple_evaluate(
        model=evaluation._lm_eval_model(tiny_session, "right", recorder),
        tasks=["hellaswag", "winogrande"],
        num_fewshot=0,
        limit=2,
        log_samples=True,
        task_manager=manager,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=True,
        gen_kwargs=None,
        bootstrap_iters=100000,
        predict_only=False,
        **evaluation._evaluation_seeds(),
    )
    loaded = manager.loaded["tasks"]
    assert raw_result["config"]["log_samples"] is True
    expected_samples = {
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
        for task_name in ("hellaswag", "winogrande")
    }
    assert raw_result["samples"] == expected_samples
    assert raw_result["n-shot"] == {"hellaswag": 0, "winogrande": 0}
    assert all(config["num_fewshot"] == 0 for config in raw_result["configs"].values())
    records = tuple(
        evaluation._resolve_task_record(
            model_config=tiny_session.resolved_model.model_config,
            task_name=task_name,
            task=loaded[task_name],
            provider=fake_provider,
            manager=manager,
            recorder=recorder,
            provider_result=raw_result,
            limit=2,
            padding="right",
            seeds=evaluation._evaluation_seeds(),
            provider_versions=evaluation._provider_versions(),
        )
        for task_name in ("hellaswag", "winogrande")
    )
    assert all(record.few_shot_config["num_fewshot"] == 0 for record in records)
    assert all(record.dataset_revision == "version:1.0.0" for record in records)
    assert all(
        record.ordered_request_identity.startswith("sha256:") for record in records
    )


def test_task_provenance_hashes_yaml_include_config_and_dataset(fake_provider) -> None:
    record = _task_record(fake_provider)
    assert record.task_yaml.logical_name == "tasks/hellaswag.yaml"
    assert [source.logical_name for source in record.include_template_closure] == [
        "tasks/common.yaml"
    ]
    assert record.dataset_revision == "version:1.0.0"
    assert record.dataset_fingerprint.startswith("sha256:")
    assert record.metric_payload == {"acc,none": 0.5}
    assert record.prompt_config == {
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
    assert record.few_shot_config == {
        "fewshot_as_multiturn": True,
        "fewshot_config": {
            "doc_to_choice": "offline.serialized._offline_doc_to_choice",
            "doc_to_target": "offline.serialized._offline_doc_to_target",
            "doc_to_text": "offline.serialized._offline_doc_to_text",
            "process_docs": "offline.serialized._offline_process_docs",
            "split": "fewshot-config",
        },
        "fewshot_split": "train",
        "num_fewshot": 1,
    }
    assert record.generation_config == {
        "generation_kwargs": {"temperature": 0.0},
        "provider_gen_kwargs": None,
        "inference_runtime": {
            "rope_scaling_factor": 1.0,
            "original_context_length": 1_024,
            "effective_context_length": 1_024,
        },
    }
    assert record.metric_normalization_config == {
        "bootstrap_iters": 100000,
        "doc_to_decontamination_query": None,
        "filter_list": (
            {
                "filter": ({"function": "take_first"},),
                "name": "none",
            },
        ),
        "log_samples": True,
        "metric_list": ({"metric": "acc"},),
        "predict_only": False,
        "repeats": 1,
        "should_decontaminate": False,
    }
    assert str(fake_provider.package_root) not in str(record)


def test_recording_manager_captures_one_exact_valid_load(fake_provider) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    assert manager.loaded is loaded
    with pytest.raises(SMLRuntimeError, match="second|once"):
        manager.load(["hellaswag"])


def test_recording_manager_rejects_malformed_load_result() -> None:
    from sml import evaluation

    class MalformedTaskManager:
        def load(self, _task_list: object) -> dict[str, object]:
            return {"groups": {}, "group_map": {}}

    manager = evaluation._RecordingTaskManager(MalformedTaskManager())
    with pytest.raises(SMLRuntimeError, match="malformed"):
        manager.load(["hellaswag"])


@pytest.mark.parametrize("first_outcome", ("delegate-error", "missing", "malformed"))
def test_recording_manager_rejects_second_load_after_every_failed_first_attempt(
    first_outcome: str,
) -> None:
    """Catches a manager that records only successful loads and permits retries."""
    from sml import evaluation

    if first_outcome == "delegate-error":

        class Inner:
            def load(self, _task_list: object) -> object:
                raise ValueError("provider failed")

        message = "provider failed"
    elif first_outcome == "missing":

        class Inner:
            pass

        message = "no load"
    else:

        class Inner:
            def load(self, _task_list: object) -> object:
                return {"groups": {}, "group_map": {}}

        message = "malformed"

    manager = evaluation._RecordingTaskManager(Inner())
    with pytest.raises((SMLRuntimeError, ValueError), match=message):
        manager.load(["hellaswag"])
    with pytest.raises(SMLRuntimeError, match="second|once"):
        manager.load(["hellaswag"])


def test_task_provenance_records_processed_eval_and_fewshot_dataset_objects(
    fake_provider,
) -> None:
    """Catches provenance that reads raw task.dataset splits instead of iterated docs."""
    from sml.artifacts.manifest import structured_identity

    record = _task_record(fake_provider)

    assert record.dataset_fingerprint == structured_identity(
        "sml-evaluation-dataset-v1",
        (
            ("test", "evaluated-test-fingerprint"),
            ("fewshot-config", "retained-fewshot-config-fingerprint"),
        ),
    )
    assert record.dataset_revision == "version:1.0.0"


def test_task_provenance_serializes_effective_default_filter_pipeline(
    fake_provider,
) -> None:
    """Catches omission of 0.4.12's default none/take_first filter pipeline."""
    record = _task_record(fake_provider)

    assert record.metric_normalization_config["filter_list"] == (
        {
            "filter": ({"function": "take_first"},),
            "name": "none",
        },
    )


def test_task_provenance_preserves_explicit_provider_filter_pipeline(
    fake_provider,
) -> None:
    """Catches replacement of an explicit provider-owned filter pipeline with defaults."""
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    tuple(task.doc_iterator())
    task.config._values["filter_list"] = [
        {
            "name": "exact-match",
            "filter": [
                {"function": "regex", "regex_pattern": "^A$"},
                {"function": "take_first"},
            ],
        }
    ]
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)

    record = evaluation._resolve_task_record(
        model_config=ModelConfig(),
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
        limit=None,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )

    assert record.metric_normalization_config["filter_list"] == (
        {
            "filter": (
                {"function": "regex", "regex_pattern": "^A$"},
                {"function": "take_first"},
            ),
            "name": "exact-match",
        },
    )


def test_provider_iteration_captures_first_effective_eval_dataset(
    fake_provider,
    tiny_session: InferenceSession,
) -> None:
    """Catches provenance using task_docs or a post-run eval_docs recomputation."""
    from sml import evaluation
    from sml.artifacts.manifest import structured_identity

    class IteratingTaskManager(fake_provider.task_manager_type):
        def load(self, task_list: list[str]) -> dict[str, object]:
            loaded = super().load(task_list)
            task = loaded["tasks"]["hellaswag"]
            task._eval_docs_sequence = (
                SimpleNamespace(
                    _fingerprint="provider-first-eval",
                    info=SimpleNamespace(version="1.0.0"),
                ),
                SimpleNamespace(
                    _fingerprint="provider-recomputed-eval",
                    info=SimpleNamespace(version="1.0.0"),
                ),
            )
            return loaded

    manager = evaluation._RecordingTaskManager(IteratingTaskManager(), fake_provider)
    recorder = evaluation._EvaluationRequestRecorder()
    raw_result = fake_provider.simple_evaluate(
        model=evaluation._lm_eval_model(tiny_session, "right", recorder),
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=1,
        log_samples=True,
        task_manager=manager,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=True,
        gen_kwargs=None,
        bootstrap_iters=100000,
        predict_only=False,
        **evaluation._evaluation_seeds(),
    )
    task = manager.loaded["tasks"]["hellaswag"]
    first_iteration = tuple(task.doc_iterator())
    second_iteration = tuple(task.doc_iterator())
    record = evaluation._resolve_task_record(
        model_config=tiny_session.resolved_model.model_config,
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result=raw_result,
        limit=1,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )

    assert first_iteration[0][1] is second_iteration[0][1]
    assert task.eval_docs_calls == 1
    assert record.dataset_fingerprint == structured_identity(
        "sml-evaluation-dataset-v1",
        (("test", "provider-first-eval"),),
    )


@pytest.mark.parametrize(
    ("available", "expected_split"),
    (
        (
            {"fewshot_config": {"split": "nested"}, "fewshot_split": "top-level"},
            "nested",
        ),
        ({"fewshot_split": "top-level"}, "top-level"),
        ({"training_split": "train"}, "train"),
        ({"validation_split": "validation"}, "validation"),
        ({"test_split": "test"}, "test"),
    ),
)
def test_dataset_provenance_mirrors_provider_fewshot_fallback_order(
    available: dict[str, str],
    expected_split: str,
) -> None:
    """Catches rejection of 0.4.12's training/validation/test few-shot fallback."""
    from sml import evaluation
    from sml.artifacts.manifest import structured_identity

    task = SimpleNamespace(
        _sml_effective_eval_docs=SimpleNamespace(
            _fingerprint="evaluated-eval", info=SimpleNamespace(version="1.0.0")
        ),
        sampler=SimpleNamespace(
            df=SimpleNamespace(
                _fingerprint=f"retained-{expected_split}",
                info=SimpleNamespace(version="1.0.0"),
            )
        ),
    )
    config = {
        "test_split": "test",
        "num_fewshot": 1,
        "fewshot_config": {},
        "fewshot_split": None,
        "dataset_kwargs": {},
        **available,
    }

    _, fingerprint = evaluation._dataset_identity(task, config)

    assert fingerprint == structured_identity(
        "sml-evaluation-dataset-v1",
        (("test", "evaluated-eval"), (expected_split, f"retained-{expected_split}")),
    )


def test_task_provenance_rejects_missing_index_yaml_path(fake_provider) -> None:
    from sml import evaluation

    inner = fake_provider.make_task_manager()
    inner.task_index["hellaswag"] = SimpleNamespace(yaml_path=None)
    manager = evaluation._RecordingTaskManager(inner, fake_provider)
    with pytest.raises(SMLRuntimeError, match="YAML|yaml"):
        manager.load(["hellaswag"])


def test_task_provenance_rejects_missing_indexed_yaml_file(fake_provider) -> None:
    from sml import evaluation

    inner = fake_provider.make_task_manager()
    inner.task_index["hellaswag"] = SimpleNamespace(
        yaml_path=fake_provider.package_root / "tasks" / "missing.yaml"
    )
    manager = evaluation._RecordingTaskManager(inner, fake_provider)
    with pytest.raises(SMLRuntimeError, match="YAML|yaml"):
        manager.load(["hellaswag"])


def test_task_source_snapshot_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    """A special .yaml source must fail closed without waiting for a writer."""
    package_root = tmp_path / "lm_eval"
    package_root.mkdir()
    fifo = package_root / "task.yaml"
    os.mkfifo(fifo)
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    process = context.Process(
        target=_snapshot_fifo_source,
        args=(fifo, package_root, outcomes),
    )
    try:
        process.start()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("task YAML snapshot blocked while opening a FIFO")
        assert process.exitcode == 0
        assert outcomes.get(timeout=1) == "rejected"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        outcomes.close()
        outcomes.join_thread()


@pytest.mark.parametrize(
    ("primary", "included", "message"),
    [
        (
            "include: common.yaml\ntask: hellaswag\n",
            "include: hellaswag.yaml\n",
            "cycle",
        ),
        ("include: ../../outside.yaml\ntask: hellaswag\n", "", "outside|escape"),
    ],
)
def test_task_provenance_rejects_unsafe_include_closure(
    fake_provider,
    tmp_path: Path,
    primary: str,
    included: str,
    message: str,
) -> None:
    from sml import evaluation

    root = tmp_path / "lm_eval"
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "hellaswag.yaml").write_text(primary, encoding="utf-8")
    if included:
        (tasks / "common.yaml").write_text(included, encoding="utf-8")
    (tmp_path / "outside.yaml").write_text("metadata: {}\n", encoding="utf-8")
    provider = replace(fake_provider, package_root=root)
    inner = provider.make_task_manager()
    inner.task_index["hellaswag"] = SimpleNamespace(yaml_path=tasks / "hellaswag.yaml")
    manager = evaluation._RecordingTaskManager(inner, provider)
    with pytest.raises(SMLRuntimeError, match=message):
        manager.load(["hellaswag"])


@pytest.mark.parametrize(
    ("dataset_object", "attribute", "value", "message"),
    [
        ("eval_docs", "_fingerprint", "", "fingerprint"),
        ("eval_docs", "info", SimpleNamespace(version=""), "version"),
        ("sampler_docs", "_fingerprint", "", "fingerprint"),
    ],
)
def test_task_provenance_requires_selected_dataset_identities(
    fake_provider,
    dataset_object: str,
    attribute: str,
    value: object,
    message: str,
) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    dataset = task._eval_docs if dataset_object == "eval_docs" else task.sampler.df
    object.__setattr__(dataset, attribute, value)
    tuple(task.doc_iterator())
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    with pytest.raises(SMLRuntimeError, match=message):
        evaluation._resolve_task_record(
            model_config=ModelConfig(),
            task_name="hellaswag",
            task=task,
            provider=fake_provider,
            manager=manager,
            recorder=recorder,
            provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
            limit=None,
            padding="right",
            seeds=evaluation._evaluation_seeds(),
            provider_versions=evaluation._provider_versions(),
        )


def test_task_provenance_selects_test_when_validation_is_not_declared(
    fake_provider,
) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(
        fake_provider.make_task_manager(), fake_provider
    )
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    task.config._values["validation_split"] = None
    task.config._values["num_fewshot"] = 0
    object.__setattr__(task._eval_docs, "info", SimpleNamespace(version="3.0.0"))
    tuple(task.doc_iterator())
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    record = evaluation._resolve_task_record(
        model_config=ModelConfig(),
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
        limit=None,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )
    assert record.dataset_revision == "version:3.0.0"


def test_task_provenance_rejects_missing_task_metric(fake_provider) -> None:
    with pytest.raises(SMLRuntimeError, match="metric|result"):
        _task_record(fake_provider, provider_result={"results": {}})


def test_task_provenance_rejects_ambiguous_duplicate_metric_keys(fake_provider) -> None:
    with pytest.raises(SMLRuntimeError, match="duplicate|ambiguous"):
        _task_record(
            fake_provider,
            provider_result={"results": {"hellaswag": _DuplicateMetricMapping()}},
        )
