# ruff: noqa: F811
from __future__ import annotations

import importlib
import math
import os
import socket
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sml.errors import SMLRuntimeError
from sml.evaluation import (
    EvaluationConfig,
    LoglikelihoodRequest,
    SMLEvalLM,
    evaluate,
    read_evaluation_result,
    score_loglikelihood_batch,
)
from sml.evaluation_result import evaluation_result_identity
from sml.inference import GenerationResult, InferenceSession
from test_inference import (  # noqa: F401
    _tiny_run_template,
    publish_new_valid_step,
    tiny_pretraining_run,
    tiny_session,
)

IN_VOCAB_CONTEXTS = ("alpha", "alpha beta", "alpha beta gamma")
IN_VOCAB_CONTINUATIONS = (" beta", " gamma", " delta")
_PROVIDER_STUBS = Path(__file__).resolve().parents[1] / "fixtures" / "provider_stubs"


@dataclass(frozen=True, slots=True)
class _ScoringCompileKey:
    length_bucket: int
    batch_size_bucket: int
    padding: str


class _ScoringCompileSpy:
    def __init__(self, session: InferenceSession) -> None:
        self._session = session

    @property
    def scoring_keys(self) -> set[_ScoringCompileKey]:
        compiled = getattr(self._session, "_scoring_compiled", {})
        keys: set[_ScoringCompileKey] = set()
        for key in compiled:
            if isinstance(key, tuple) and len(key) == 3:
                keys.add(_ScoringCompileKey(key[0], key[1], key[2]))
            else:
                keys.add(
                    _ScoringCompileKey(
                        key.length_bucket,
                        key.batch_size_bucket,
                        key.padding,
                    )
                )
        return keys


@pytest.fixture
def compile_spy(tiny_session: InferenceSession) -> _ScoringCompileSpy:
    return _ScoringCompileSpy(tiny_session)


def scoring_requests(count: int) -> tuple[LoglikelihoodRequest, ...]:
    return tuple(
        LoglikelihoodRequest(context="alpha", continuation=" beta")
        for _ in range(count)
    )


def heterogeneous_scoring_requests() -> tuple[LoglikelihoodRequest, ...]:
    return tuple(
        LoglikelihoodRequest(context=context, continuation=continuation)
        for context, continuation in zip(
            IN_VOCAB_CONTEXTS,
            IN_VOCAB_CONTINUATIONS,
            strict=True,
        )
    )


def assert_loglikelihood_results_close(
    left: tuple,
    right: tuple,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert len(left) == len(right)
    for actual, expected in zip(left, right, strict=True):
        assert actual.greedy_match is expected.greedy_match
        assert actual.log_likelihood == pytest.approx(
            expected.log_likelihood,
            abs=atol,
            rel=rtol,
        )


def tiny_evaluation_config(
    run: Path,
    tmp_path: Path,
    tasks: tuple[str, ...] = ("hellaswag",),
) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=run,
        tasks=tasks,
        output=tmp_path / "evaluation.json",
    )


@pytest.fixture
def fake_lm_eval(fake_provider, monkeypatch: pytest.MonkeyPatch):
    from sml import evaluation

    calls: list[dict[str, object]] = []

    def simple_evaluate(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        raw_result = fake_provider.simple_evaluate(**kwargs)
        if module.result is not None:
            return module.result
        raw_result = dict(raw_result)
        raw_result["results"] = {
            task_name: {"acc,none": 0.5, "acc_stderr,none": 0.01}
            for task_name in raw_result["results"]
        }
        module.result = raw_result
        return raw_result

    module = SimpleNamespace(result=None, calls=calls)
    monkeypatch.setattr(
        evaluation,
        "_import_lm_eval",
        lambda: replace(fake_provider, simple_evaluate=simple_evaluate),
    )
    return module


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch):
    """Load the checked-in 0.4.12-shaped provider without using the network."""
    from sml import evaluation

    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "lm_eval" or name.startswith("lm_eval.")
    }
    for name in tuple(original):
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(_PROVIDER_STUBS))
    importlib.invalidate_caches()
    module = importlib.import_module("lm_eval")
    task_module = importlib.import_module("lm_eval.tasks")
    loader_module = importlib.import_module("lm_eval.tasks._yaml_loader")
    try:
        yield evaluation._LMEvalProvider(
            simple_evaluate=module.simple_evaluate,
            task_manager_type=task_module.TaskManager,
            load_yaml=loader_module.load_yaml,
            package_root=Path(module.__file__).resolve().parent,
            package_version=module.__version__,
        )
    finally:
        for name in tuple(sys.modules):
            if name == "lm_eval" or name.startswith("lm_eval."):
                del sys.modules[name]
        sys.modules.update(original)


def _task_record(
    fake_provider,
    *,
    provider_result: Mapping[str, object] | None = None,
):
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    return evaluation._resolve_task_record(
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


@pytest.mark.parametrize(
    ("version", "expected"),
    [(1.0, "1.0"), (2, "2")],
)
def test_task_metadata_version_canonicalizes_finite_provider_scalars(
    version: float,
    expected: str,
) -> None:
    from sml import evaluation

    assert evaluation._metadata_version({"metadata": {"version": version}}) == expected


@pytest.mark.parametrize("version", [True, float("nan"), float("inf")])
def test_task_metadata_version_rejects_nonfinite_or_boolean_provider_values(
    version: object,
) -> None:
    from sml import evaluation

    with pytest.raises(SMLRuntimeError, match="metadata version"):
        evaluation._metadata_version({"metadata": {"version": version}})


@pytest.mark.parametrize("task_name", ["hellaswag", "winogrande"])
def test_installed_task_config_keeps_numeric_metadata_and_nested_callables(
    task_name: str,
) -> None:
    import lm_eval
    from lm_eval.config.task import TaskConfig
    from lm_eval.tasks._yaml_loader import load_yaml

    task_directory = Path(lm_eval.__file__).resolve().parent / "tasks" / task_name
    filename = "hellaswag.yaml" if task_name == "hellaswag" else "default.yaml"
    config = TaskConfig(**load_yaml(task_directory / filename))
    values = config.to_dict()
    assert values["metadata"]["version"] == 1.0
    assert any(callable(value) for value in values["fewshot_config"].values())


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

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    record = evaluation._resolve_task_record(
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
            ("validation", "validation-fingerprint"),
            ("fewshot-config", "fewshot-config-fingerprint"),
        ),
    )


def test_offline_provider_task4_invocation_records_effective_zero_shot_provenance(
    fake_provider,
    tiny_session: InferenceSession,
) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
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
    assert raw_result["n-shot"] == {"hellaswag": 0, "winogrande": 0}
    assert all(config["num_fewshot"] == 0 for config in raw_result["configs"].values())
    records = tuple(
        evaluation._resolve_task_record(
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


def test_request_recorder_hashes_actual_order_per_task() -> None:
    from sml import evaluation

    first = SimpleNamespace(
        task_name="hellaswag", doc_id=2, repeats=1, args=("a", " b")
    )
    second = SimpleNamespace(
        task_name="hellaswag", doc_id=1, repeats=1, args=("c", " d")
    )
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", (first, second))
    reversed_recorder = evaluation._EvaluationRequestRecorder()
    reversed_recorder.record("loglikelihood", (second, first))
    assert recorder.identity_for("hellaswag") != reversed_recorder.identity_for(
        "hellaswag"
    )


def test_request_recorder_requires_unambiguous_task_ownership() -> None:
    from sml import evaluation

    recorder = evaluation._EvaluationRequestRecorder()
    with pytest.raises(SMLRuntimeError, match="task_name|ownership"):
        recorder.record(
            "loglikelihood",
            (SimpleNamespace(doc_id=1, repeats=1, args=("a", " b")),),
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
    }
    assert record.metric_normalization_config == {
        "bootstrap_iters": 100000,
        "doc_to_decontamination_query": None,
        "filter_list": ({"filter": "none"},),
        "log_samples": True,
        "metric_list": ({"metric": "acc"},),
        "predict_only": False,
        "repeats": 1,
        "should_decontaminate": False,
    }
    assert str(fake_provider.package_root) not in str(record)


def test_recording_manager_captures_one_exact_valid_load(fake_provider) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
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


def test_task_provenance_rejects_missing_index_yaml_path(fake_provider) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    manager.task_index["hellaswag"] = SimpleNamespace(yaml_path=None)
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", loaded["tasks"]["hellaswag"].instances)
    with pytest.raises(SMLRuntimeError, match="YAML|yaml"):
        evaluation._resolve_task_record(
            task_name="hellaswag",
            task=loaded["tasks"]["hellaswag"],
            provider=fake_provider,
            manager=manager,
            recorder=recorder,
            provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
            limit=None,
            padding="right",
            seeds=evaluation._evaluation_seeds(),
            provider_versions=evaluation._provider_versions(),
        )


def test_task_provenance_rejects_missing_indexed_yaml_file(fake_provider) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    manager.task_index["hellaswag"] = SimpleNamespace(
        yaml_path=fake_provider.package_root / "tasks" / "missing.yaml"
    )
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", loaded["tasks"]["hellaswag"].instances)
    with pytest.raises(SMLRuntimeError, match="YAML|yaml"):
        evaluation._resolve_task_record(
            task_name="hellaswag",
            task=loaded["tasks"]["hellaswag"],
            provider=fake_provider,
            manager=manager,
            recorder=recorder,
            provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
            limit=None,
            padding="right",
            seeds=evaluation._evaluation_seeds(),
            provider_versions=evaluation._provider_versions(),
        )


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
    manager = evaluation._RecordingTaskManager(provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    manager.task_index["hellaswag"] = SimpleNamespace(
        yaml_path=tasks / "hellaswag.yaml"
    )
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", loaded["tasks"]["hellaswag"].instances)
    with pytest.raises(SMLRuntimeError, match=message):
        evaluation._resolve_task_record(
            task_name="hellaswag",
            task=loaded["tasks"]["hellaswag"],
            provider=provider,
            manager=manager,
            recorder=recorder,
            provider_result={"results": {"hellaswag": {"acc,none": 0.5}}},
            limit=None,
            padding="right",
            seeds=evaluation._evaluation_seeds(),
            provider_versions=evaluation._provider_versions(),
        )


@pytest.mark.parametrize(
    ("split", "attribute", "value", "message"),
    [
        ("validation", "_fingerprint", "", "fingerprint"),
        ("validation", "info", SimpleNamespace(version=""), "version"),
        ("fewshot-config", "_fingerprint", "", "fingerprint"),
    ],
)
def test_task_provenance_requires_selected_dataset_identities(
    fake_provider,
    split: str,
    attribute: str,
    value: object,
    message: str,
) -> None:
    from sml import evaluation

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    object.__setattr__(task.dataset[split], attribute, value)
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    with pytest.raises(SMLRuntimeError, match=message):
        evaluation._resolve_task_record(
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

    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    task.config._values["validation_split"] = None
    task.config._values["num_fewshot"] = 0
    task.dataset["test"] = replace(
        task.dataset["test"], info=SimpleNamespace(version="3.0.0")
    )
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", task.instances)
    record = evaluation._resolve_task_record(
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


def test_task_provenance_rejects_ambiguous_duplicate_metric_keys(fake_provider) -> None:
    with pytest.raises(SMLRuntimeError, match="duplicate|ambiguous"):
        _task_record(
            fake_provider,
            provider_result={"results": {"hellaswag": _DuplicateMetricMapping()}},
        )


def test_sml_eval_lm_records_both_methods_in_dispatch_order(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sml import evaluation

    recorder = evaluation._EvaluationRequestRecorder()
    lm = SMLEvalLM(tiny_session, recorder=recorder)
    monkeypatch.setattr(
        tiny_session,
        "generate_batch",
        lambda items: tuple(
            GenerationResult("ok", (), None, tiny_session.model_identity) for _ in items
        ),
    )
    generated = SimpleNamespace(
        task_name="hellaswag",
        doc_id=3,
        repeats=1,
        args=("alpha", {"max_gen_toks": 1, "until": []}),
    )
    scored = SimpleNamespace(
        task_name="hellaswag", doc_id=2, repeats=1, args=("alpha", " beta")
    )
    lm.generate_until([generated])
    lm.loglikelihood([scored])
    expected = evaluation._EvaluationRequestRecorder()
    expected.record("generate_until", (generated,))
    expected.record("loglikelihood", (scored,))
    assert recorder.identity_for("hellaswag") == expected.identity_for("hellaswag")


def test_score_cardinality_reuses_fixed_compiled_bucket(
    tiny_session: InferenceSession,
    compile_spy: _ScoringCompileSpy,
) -> None:
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    compiled_after_three = set(compile_spy.scoring_keys)
    score_loglikelihood_batch(tiny_session, scoring_requests(4), padding="right")
    assert set(compile_spy.scoring_keys) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4


def test_continuation_only_scoring_is_finite(tiny_session: InferenceSession) -> None:
    result = score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="", continuation="alpha")],
        padding="right",
    )[0]
    assert math.isfinite(result.log_likelihood)
    assert result.greedy_match in (True, False)


def test_greedy_match_is_boolean_and_stable(tiny_session: InferenceSession) -> None:
    request = LoglikelihoodRequest(context="alpha", continuation=" beta")
    first = score_loglikelihood_batch(tiny_session, [request], padding="right")[0]
    second = score_loglikelihood_batch(tiny_session, [request], padding="right")[0]
    assert first.greedy_match in (True, False)
    assert first.greedy_match is second.greedy_match
    assert first.log_likelihood == pytest.approx(
        second.log_likelihood, abs=2e-2, rel=2e-2
    )


def test_empty_context_inserts_prefix_token(tiny_session: InferenceSession) -> None:
    processor = tiny_session.resolved_model.tokenizer.processor
    assert int(processor.bos_id()) >= 0
    result = score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="", continuation=" beta")],
        padding="left",
    )[0]
    assert math.isfinite(result.log_likelihood)


def test_empty_context_without_prefix_fails_before_bucketing(
    tiny_session: InferenceSession,
    compile_spy: _ScoringCompileSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tiny_session.resolved_model.tokenizer.processor,
        "bos_id",
        lambda: -1,
    )
    with pytest.raises(SMLRuntimeError, match="prefix"):
        score_loglikelihood_batch(
            tiny_session,
            [LoglikelihoodRequest(context="", continuation="alpha")],
            padding="right",
        )
    assert compile_spy.scoring_keys == set()


def test_nonempty_in_vocab_context_does_not_prepend_prefix(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[int, ...], int]] = []
    original = tiny_session.score_encoded_loglikelihoods

    def spy(items, *, padding: str):
        captured.extend(items)
        return original(items, padding=padding)

    monkeypatch.setattr(tiny_session, "score_encoded_loglikelihoods", spy)
    score_loglikelihood_batch(
        tiny_session,
        [LoglikelihoodRequest(context="alpha", continuation=" beta")],
        padding="right",
    )
    assert captured
    token_ids, continuation_start = captured[0]
    processor = tiny_session.resolved_model.tokenizer.processor
    context_ids = tuple(int(token) for token in processor.encode("alpha"))
    assert token_ids[: len(context_ids)] == context_ids
    assert continuation_start == len(context_ids)


def test_generate_until_truncates_at_stop_strings(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = tiny_session.model_identity

    def fake_generate_batch(items):
        return tuple(
            GenerationResult(
                text="hello world STOP more",
                token_ids=(1, 2, 3),
                seed=None,
                model=identity,
            )
            for _ in items
        )

    monkeypatch.setattr(tiny_session, "generate_batch", fake_generate_batch)
    lm = SMLEvalLM(tiny_session)
    completions = lm.generate_until(
        [
            SimpleNamespace(
                args=(
                    "alpha",
                    {"until": ["STOP"], "max_gen_toks": 8, "do_sample": False},
                )
            )
        ]
    )
    assert completions == ["hello world "]


def test_generate_until_batches_requests(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real_generate_batch = tiny_session.generate_batch

    def recording_generate_batch(items):
        calls.append(items)
        return real_generate_batch(items)

    monkeypatch.setattr(tiny_session, "generate_batch", recording_generate_batch)
    lm = SMLEvalLM(tiny_session)
    requests = [
        SimpleNamespace(args=("alpha", {"max_gen_toks": 1, "until": []})),
        SimpleNamespace(args=("beta", {"max_gen_toks": 1, "until": []})),
    ]
    completions = lm.generate_until(requests)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert len(completions) == 2
    assert all(isinstance(completion, str) for completion in completions)


def test_generate_until_rejects_sampling(tiny_session: InferenceSession) -> None:
    lm = SMLEvalLM(tiny_session)
    with pytest.raises(SMLRuntimeError, match="greedy|sample"):
        lm.generate_until(
            [SimpleNamespace(args=("alpha", {"do_sample": True, "max_gen_toks": 1}))]
        )


def test_unsupported_eval_methods_raise(tiny_session: InferenceSession) -> None:
    lm = SMLEvalLM(tiny_session)
    with pytest.raises(SMLRuntimeError, match="unsupported"):
        lm.loglikelihood_rolling([])


def test_evaluation_config_requires_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=(),
            output=tmp_path / "out.json",
        )


def test_evaluation_config_requires_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=("hellaswag",),
            output=tmp_path / "out.json",
            limit=0,
        )


def test_evaluation_config_rejects_duplicate_task_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=("hellaswag", "winogrande", "hellaswag"),
            output=tmp_path / "out.json",
        )


def test_evaluation_config_rejects_unknown_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hellaswag|winogrande"):
        EvaluationConfig(
            checkpoint=tmp_path,
            tasks=("mmlu",),
            output=tmp_path / "out.json",
        )


def test_importing_evaluation_does_not_import_lm_eval() -> None:
    from sml import evaluation

    assert "lm_eval" not in evaluation.__dict__
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    tree_import_lines = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert all("lm_eval" not in line for line in tree_import_lines)


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


def test_evaluate_preserves_complete_provider_result_and_task_metrics(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)

    result = evaluate(config)

    persisted = read_evaluation_result(config.output)
    assert persisted == result
    assert result.provider_result["configs"] == fake_lm_eval.result["configs"]
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


def test_sml_eval_lm_loglikelihood_uses_batch_scorer(
    tiny_session: InferenceSession,
) -> None:
    lm = SMLEvalLM(tiny_session)
    request = SimpleNamespace(args=("alpha", " beta"))
    scored = lm.loglikelihood([request])
    assert len(scored) == 1
    log_likelihood, greedy_match = scored[0]
    assert math.isfinite(log_likelihood)
    assert greedy_match in (True, False)


def test_evaluate_passes_installed_lm_eval_lm(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
) -> None:
    from lm_eval.api.model import LM

    evaluate(tiny_evaluation_config(tiny_pretraining_run, tmp_path))
    model = fake_lm_eval.calls[0]["model"]
    assert isinstance(model, LM)


def test_evaluation_publish_does_not_replace_destination(
    tiny_pretraining_run: Path,
    fake_lm_eval,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replaced: list[Path] = []
    real_replace = os.replace

    def tracking_replace(src, dst, *args, **kwargs):
        replaced.append(Path(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", tracking_replace)
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)
    evaluate(config)
    assert all(path != config.output for path in replaced)
    assert config.output.exists()


def test_compiled_scoring_kernel_receives_request_mask(
    tiny_session: InferenceSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original = tiny_session._compiled_scoring_kernel

    def wrapping(length_bucket: int, batch_size_bucket: int, padding: str):
        compiled = original(length_bucket, batch_size_bucket, padding)

        def spy(*args, **kwargs):
            captured.append((args, kwargs))
            return compiled(*args, **kwargs)

        return spy

    monkeypatch.setattr(tiny_session, "_compiled_scoring_kernel", wrapping)
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    assert captured
    args, kwargs = captured[0]
    values = list(args) + list(kwargs.values())
    found = False
    for value in values:
        if getattr(value, "ndim", None) != 1:
            continue
        converted = value.tolist()
        if converted == [True, True, True, False]:
            found = True
            break
    assert found, "compiled scoring kernel did not receive a boolean request mask"
