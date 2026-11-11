from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    evaluate,
    read_evaluation_result,
)
from sml.evaluation_result import evaluation_result_identity
from sml_test_support.evaluation import (
    tiny_evaluation_config,
)

from .helpers import (
    _copied_task_source_provider,
    _DuplicateIterationMapping,
)


def test_evaluate_does_not_use_the_network(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    result = evaluate(tiny_evaluation_config(tiny_pretraining_run, tmp_path))
    assert read_evaluation_result(tmp_path / "evaluation.json") == result
    assert fake_lm_eval.calls


@pytest.mark.parametrize("source_name", ("hellaswag.yaml", "common.yaml"))
def test_evaluate_rejects_task_source_mutation_before_publication(
    source_name: str,
    tiny_pretraining_run: Path,
    fake_provider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Task source bytes used for construction must remain stable through publish."""
    from sml import evaluation

    provider, tasks = _copied_task_source_provider(fake_provider, tmp_path)
    source = tasks / source_name

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        raw_result = fake_provider.simple_evaluate(**kwargs)
        source.write_bytes(source.read_bytes() + b"\n# concurrent replacement\n")
        return raw_result

    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(provider, simple_evaluate=simple_evaluate),
    )
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="source|YAML|changed"):
        evaluate(config)

    assert not config.output.exists()


def test_task_source_snapshot_rejects_mutation_during_provider_parse(
    fake_provider,
    tmp_path: Path,
) -> None:
    """One snapshot cannot mix bytes read before and during provider parsing."""
    from sml import evaluation

    provider, tasks = _copied_task_source_provider(fake_provider, tmp_path)
    primary = tasks / "hellaswag.yaml"
    real_load_yaml = provider.load_yaml
    mutated = False

    def mutating_load_yaml(path: Path, **kwargs: object) -> object:
        nonlocal mutated
        if Path(path).resolve() == primary.resolve() and not mutated:
            mutated = True
            primary.write_bytes(primary.read_bytes() + b"\n# parser race\n")
        return real_load_yaml(path, **kwargs)

    provider = replace(provider, load_yaml=mutating_load_yaml)

    with pytest.raises(SMLRuntimeError, match="source|YAML|changed"):
        evaluation._snapshot_yaml_sources(provider, primary)


def test_evaluate_preserves_complete_provider_result_and_task_metrics(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    result = evaluate(config)

    persisted = read_evaluation_result(config.output)
    assert persisted == result
    nested_config = fake_lm_eval.result["configs"]["hellaswag"]["fewshot_config"]
    assert callable(nested_config["process_docs"])
    assert set(result.provider_result) == set(fake_lm_eval.result)
    assert result.provider_result["results"] == fake_lm_eval.result["results"]
    assert result.provider_result["configs"]["hellaswag"]["fewshot_config"] == {
        "doc_to_choice": "offline.serialized._offline_doc_to_choice",
        "doc_to_target": "offline.serialized._offline_doc_to_target",
        "doc_to_text": "offline.serialized._offline_doc_to_text",
        "process_docs": "offline.serialized._offline_process_docs",
        "split": "fewshot-config",
    }
    assert result.tasks[0].metric_payload == {
        "acc,none": 0.5,
        "acc_stderr,none": 0.01,
    }
    assert str(tmp_path).encode() not in config.output.read_bytes()
    assert result.identity == evaluation_result_identity(result)


def test_evaluate_fails_before_publish_when_provider_result_is_incomplete(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    fake_lm_eval.result = {"configs": {}}
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="results"):
        evaluate(config)

    assert not config.output.exists()


def test_evaluate_rejects_duplicate_top_level_provider_keys_before_publish(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    """Breaks if copying a provider Mapping collapses duplicate iteration keys."""
    fake_lm_eval.result = _DuplicateIterationMapping(
        (
            ("results", {"hellaswag": {"acc,none": 0.5}}),
            ("configs", {"hellaswag": {}}),
            ("results", {"hellaswag": {"acc,none": 0.75}}),
        )
    )
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="duplicate"):
        evaluate(config)

    assert not config.output.exists()


def test_evaluate_rejects_duplicate_nested_config_keys_before_serialization(
    tiny_pretraining_run: Path,
    fake_provider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Breaks if recursive config copying serializes before duplicate rejection."""
    from sml import evaluation

    serialization_calls: list[object] = []

    class TrackingTaskManager(fake_provider.task_manager_type):
        def load(self, task_list):
            loaded = super().load(task_list)

            def serialize_function(value: object) -> str:
                serialization_calls.append(value)
                return "serialized"

            loaded["tasks"]["hellaswag"].config.serialize_function = serialize_function
            return loaded

    def nested_callable() -> None:
        raise AssertionError("provider callables must only be serialized")

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        raw_result = fake_provider.simple_evaluate(**kwargs)
        raw_result["results"] = {"hellaswag": {"acc,none": 0.5}}
        configs = raw_result["configs"]
        assert isinstance(configs, dict)
        task_config = configs["hellaswag"]
        assert isinstance(task_config, dict)
        task_config["fewshot_config"] = _DuplicateIterationMapping(
            (
                ("process_docs", nested_callable),
                ("process_docs", nested_callable),
            )
        )
        return raw_result

    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(
            fake_provider,
            simple_evaluate=simple_evaluate,
            task_manager_type=TrackingTaskManager,
        ),
    )
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="duplicate"):
        evaluate(config)

    assert serialization_calls == []
    assert not config.output.exists()


def test_evaluate_fails_before_publish_for_mismatched_loaded_task_ownership(
    tiny_pretraining_run: Path,
    fake_provider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sml import evaluation

    class MismatchedTaskManager(fake_provider.task_manager_type):
        def load(self, task_list):
            loaded = super().load(task_list)
            loaded["tasks"]["hellaswag"].task_name = "winogrande"
            return loaded

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        manager = kwargs["task_manager"]
        loaded = manager.load(kwargs["tasks"])
        task = loaded["tasks"]["hellaswag"]
        kwargs["model"].loglikelihood(task.instances)
        return {
            "results": {"hellaswag": {"acc,none": 0.5}},
            "configs": {"hellaswag": {}},
        }

    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(
            fake_provider,
            simple_evaluate=simple_evaluate,
            task_manager_type=MismatchedTaskManager,
        ),
    )
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="task_name|ownership"):
        evaluate(config)

    assert not config.output.exists()


def test_evaluate_fails_before_publish_for_extra_recorded_task_requests(
    tiny_pretraining_run: Path,
    fake_provider,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sml import evaluation

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        manager = kwargs["task_manager"]
        loaded = manager.load(kwargs["tasks"])
        task = loaded["tasks"]["hellaswag"]
        kwargs["model"].loglikelihood(task.instances)
        kwargs["model"].loglikelihood(
            (
                SimpleNamespace(
                    args=("alpha", " beta"),
                    task_name="winogrande",
                    doc_id=99,
                    repeats=1,
                ),
            )
        )
        return {
            "results": {"hellaswag": {"acc,none": 0.5}},
            "configs": {"hellaswag": {}},
        }

    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(fake_provider, simple_evaluate=simple_evaluate),
    )
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    with pytest.raises(SMLRuntimeError, match="recorded.*tasks|requests"):
        evaluate(config)

    assert not config.output.exists()


def test_evaluate_passes_installed_lm_eval_lm(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    from lm_eval.api.model import LM

    evaluate(tiny_evaluation_config(tiny_pretraining_run, tmp_path))
    model = fake_lm_eval.calls[0]["model"]
    assert isinstance(model, LM)
