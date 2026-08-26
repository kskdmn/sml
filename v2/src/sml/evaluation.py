"""Batched lm-eval scoring and strict evaluation artifact publication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from sml.artifacts.manifest import structured_identity
from sml.errors import SMLRuntimeError
from sml.evaluation_result import (
    EvaluationProviderVersion,
    EvaluationResult,
    EvaluationSourceIdentity,
    EvaluationTaskRecord,
    evaluation_result_identity,
    evaluation_task_identity,
    normalize_json_value,
    publish_evaluation_result,
    read_evaluation_result,
)
from sml.inference import (
    GenerationConfig,
    GenerationRequest,
    InferenceRuntimeConfig,
    InferenceSession,
)

_ALLOWED_TASKS = frozenset({"hellaswag", "winogrande"})
_PROVIDER_PACKAGES = ("datasets", "lm-eval", "mlx", "numpy", "sentencepiece")
_EVALUATION_SEEDS = {
    "random_seed": 0,
    "numpy_random_seed": 1234,
    "torch_random_seed": 1234,
    "fewshot_random_seed": 1234,
}
_BOOTSTRAP_ITERS = 100000


@dataclass(frozen=True, slots=True)
class LoglikelihoodRequest:
    context: str
    continuation: str


@dataclass(frozen=True, slots=True)
class LoglikelihoodResult:
    log_likelihood: float
    greedy_match: bool


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    checkpoint: Path
    tasks: tuple[Literal["hellaswag", "winogrande"], ...]
    output: Path
    full_verify: bool = False
    padding: Literal["left", "right"] = "right"
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    limit: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a Path")
        if not isinstance(self.output, Path):
            raise TypeError("output must be a Path")
        if not isinstance(self.full_verify, bool):
            raise TypeError("full_verify must be a bool")
        if self.padding not in ("left", "right"):
            raise ValueError("padding must be 'left' or 'right'")
        if not isinstance(self.runtime, InferenceRuntimeConfig):
            raise TypeError("runtime must be an InferenceRuntimeConfig")
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError("tasks must contain at least one task")
        for task in self.tasks:
            if task not in _ALLOWED_TASKS:
                raise ValueError("tasks must be hellaswag or winogrande")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must not contain duplicate task names")
        if self.limit is not None and (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit <= 0
        ):
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class _LMEvalProvider:
    simple_evaluate: Callable[..., object]
    task_manager_type: type
    load_yaml: Callable[..., object]
    package_root: Path
    package_version: str

    def make_task_manager(self) -> object:
        return self.task_manager_type()


def _import_lm_eval() -> _LMEvalProvider:
    import lm_eval
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml

    module_path = getattr(lm_eval, "__file__", None)
    if not isinstance(module_path, str):
        raise SMLRuntimeError("lm-eval package root cannot be resolved")
    try:
        package_version = importlib.metadata.version("lm-eval")
    except importlib.metadata.PackageNotFoundError as error:
        raise SMLRuntimeError("lm-eval package version cannot be resolved") from error
    return _LMEvalProvider(
        simple_evaluate=simple_evaluate,
        task_manager_type=TaskManager,
        load_yaml=load_yaml,
        package_root=Path(module_path).resolve().parent,
        package_version=package_version,
    )


class _RecordingTaskManager:
    """Retain the exact task mapping consumed by one provider evaluation."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self._loaded: Mapping[str, object] | None = None
        self._attempted = False

    @property
    def loaded(self) -> Mapping[str, object]:
        if self._loaded is None:
            raise SMLRuntimeError("lm-eval task manager has not loaded tasks")
        return self._loaded

    def load(self, task_list: object) -> Mapping[str, object]:
        if self._attempted:
            raise SMLRuntimeError("lm-eval task manager may load tasks only once")
        self._attempted = True
        loader = getattr(self._inner, "load", None)
        if not callable(loader):
            raise SMLRuntimeError("lm-eval task manager has no load method")
        loaded = loader(task_list)
        if not isinstance(loaded, Mapping) or not isinstance(
            loaded.get("tasks"), Mapping
        ):
            raise SMLRuntimeError(
                "lm-eval task manager returned malformed task mapping"
            )
        self._loaded = loaded
        return loaded

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _EvaluationRequestRecorder:
    """Capture normalized lm-eval requests in actual dispatch order."""

    def __init__(self) -> None:
        self._requests: dict[str, list[dict[str, object]]] = {}

    def record(self, request_type: str, requests: Sequence[object]) -> None:
        if request_type not in {"loglikelihood", "generate_until"}:
            raise SMLRuntimeError(
                f"unsupported evaluation request type: {request_type}"
            )
        for request in requests:
            task_name = getattr(request, "task_name", None)
            doc_id = getattr(request, "doc_id", None)
            repeats = getattr(request, "repeats", None)
            args = getattr(request, "args", None)
            if not isinstance(task_name, str) or not task_name:
                raise SMLRuntimeError("evaluation request has no unambiguous task_name")
            if isinstance(doc_id, bool) or not isinstance(doc_id, int):
                raise SMLRuntimeError("evaluation request has no integer doc_id")
            if (
                isinstance(repeats, bool)
                or not isinstance(repeats, int)
                or repeats <= 0
            ):
                raise SMLRuntimeError("evaluation request has no positive repeats")
            if not isinstance(args, tuple):
                raise SMLRuntimeError("evaluation request has malformed args")
            normalized_args = normalize_json_value(
                args, context="evaluation request args"
            )
            self._requests.setdefault(task_name, []).append(
                {
                    "request_type": request_type,
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "repeats": repeats,
                    "args": normalized_args,
                }
            )

    def identity_for(self, task_name: str) -> str:
        if not isinstance(task_name, str) or not task_name:
            raise SMLRuntimeError("evaluation task name is invalid")
        requests = self._requests.get(task_name)
        if not requests:
            raise SMLRuntimeError(f"no recorded evaluation requests for {task_name}")
        return structured_identity("sml-evaluation-requests-v1", tuple(requests))

    @property
    def task_names(self) -> frozenset[str]:
        return frozenset(self._requests)


def _evaluation_seeds() -> dict[str, int]:
    return dict(_EVALUATION_SEEDS)


def _source_identity(path: Path, package_root: Path) -> EvaluationSourceIdentity:
    try:
        logical_name = path.relative_to(package_root).as_posix()
    except ValueError as error:
        raise SMLRuntimeError("lm-eval task YAML escapes the package root") from error
    if path.suffix != ".yaml" or not path.is_file():
        raise SMLRuntimeError("lm-eval task YAML path is missing or invalid")
    return EvaluationSourceIdentity(
        logical_name=logical_name,
        content_identity=f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
    )


def _resolve_yaml_sources(
    provider: _LMEvalProvider,
    yaml_path: object,
) -> tuple[EvaluationSourceIdentity, tuple[EvaluationSourceIdentity, ...]]:
    if not isinstance(yaml_path, Path):
        raise SMLRuntimeError("lm-eval task index has no YAML path")
    package_root = provider.package_root.resolve()
    primary_path = yaml_path.resolve()
    primary = _source_identity(primary_path, package_root)
    closure: list[EvaluationSourceIdentity] = []
    seen = {primary_path}

    def declared_includes(path: Path) -> tuple[object, ...]:
        try:
            loaded = provider.load_yaml(path, resolve_func=False, recursive=False)
        except Exception as error:
            raise SMLRuntimeError(f"cannot load lm-eval task YAML: {path}") from error
        if not isinstance(loaded, Mapping):
            raise SMLRuntimeError("lm-eval task YAML is not a mapping")
        includes = loaded.get("include")
        if includes is None:
            return ()
        if isinstance(includes, (str, Path)):
            return (includes,)
        if isinstance(includes, list):
            return tuple(includes)
        raise SMLRuntimeError("lm-eval task YAML include is malformed")

    def visit(path: Path, ancestry: frozenset[Path]) -> None:
        for included in declared_includes(path):
            if not isinstance(included, (str, Path)) or not str(included):
                raise SMLRuntimeError("lm-eval task YAML include is malformed")
            candidate = Path(included)
            candidate = (
                candidate if candidate.is_absolute() else path.parent / candidate
            )
            resolved = candidate.resolve()
            try:
                resolved.relative_to(package_root)
            except ValueError as error:
                raise SMLRuntimeError(
                    "lm-eval task YAML include escapes the package root"
                ) from error
            if resolved in ancestry:
                raise SMLRuntimeError("lm-eval task YAML include cycle")
            if resolved in seen:
                continue
            source = _source_identity(resolved, package_root)
            seen.add(resolved)
            closure.append(source)
            visit(resolved, ancestry | {resolved})

    visit(primary_path, frozenset({primary_path}))
    return primary, tuple(closure)


def _task_config(task: object) -> tuple[Mapping[str, object], object]:
    config = getattr(task, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise SMLRuntimeError("lm-eval task config cannot be resolved")
    values = to_dict()
    if not isinstance(values, Mapping):
        raise SMLRuntimeError("lm-eval task config is malformed")
    return values, config


def _metadata_version(config: Mapping[str, object]) -> str:
    metadata = config.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SMLRuntimeError("lm-eval task metadata version is missing")
    version = metadata.get("version")
    try:
        normalized = normalize_json_value(version, context="lm-eval metadata version")
    except (TypeError, ValueError) as error:
        raise SMLRuntimeError("lm-eval task metadata version is invalid") from error
    if isinstance(normalized, bool) or not isinstance(normalized, (str, int, float)):
        raise SMLRuntimeError("lm-eval task metadata version is missing")
    text = str(normalized)
    if not text:
        raise SMLRuntimeError("lm-eval task metadata version is missing")
    return text


def _serialize_provider_value(config: object, value: object) -> object:
    if callable(value):
        serializer = getattr(config, "serialize_function", None)
        if not callable(serializer):
            raise SMLRuntimeError("lm-eval task callable serializer is unavailable")
        serialized = serializer(value)
        if callable(serialized):
            raise SMLRuntimeError("lm-eval task callable serialization is malformed")
        return _serialize_provider_value(config, serialized)
    if isinstance(value, Mapping):
        serialized_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SMLRuntimeError("lm-eval task config key is malformed")
            serialized_mapping[key] = _serialize_provider_value(config, item)
        return serialized_mapping
    if isinstance(value, (list, tuple)):
        return tuple(_serialize_provider_value(config, item) for item in value)
    return value


def _dataset_identity(
    task: object,
    config: Mapping[str, object],
) -> tuple[str, str]:
    test_split = config.get("test_split")
    validation_split = config.get("validation_split")
    evaluation_split = test_split if test_split is not None else validation_split
    if not isinstance(evaluation_split, str) or not evaluation_split:
        raise SMLRuntimeError("lm-eval task has no selected evaluation split")
    evaluation_docs = getattr(task, "eval_docs", None)
    if evaluation_docs is None:
        raise SMLRuntimeError("lm-eval task effective evaluation docs are unavailable")
    selected_datasets: list[tuple[str, object]] = [(evaluation_split, evaluation_docs)]
    num_fewshot = config.get("num_fewshot")
    if isinstance(num_fewshot, bool) or not isinstance(num_fewshot, int):
        raise SMLRuntimeError("lm-eval task num_fewshot is malformed")
    if num_fewshot > 0:
        fewshot_config = config.get("fewshot_config")
        nested_split = (
            fewshot_config.get("split") if isinstance(fewshot_config, Mapping) else None
        )
        fewshot_split = (
            nested_split if nested_split is not None else config.get("fewshot_split")
        )
        if not isinstance(fewshot_split, str) or not fewshot_split:
            raise SMLRuntimeError("lm-eval task few-shot split is missing")
        fewshot_docs = getattr(task, "fewshot_docs", None)
        if not callable(fewshot_docs):
            raise SMLRuntimeError(
                "lm-eval task effective few-shot docs are unavailable"
            )
        selected_datasets.append((fewshot_split, fewshot_docs()))

    fingerprints: list[tuple[str, str]] = []
    versions: list[str] = []
    for split_name, effective_dataset in selected_datasets:
        fingerprint = getattr(effective_dataset, "_fingerprint", None)
        version = getattr(getattr(effective_dataset, "info", None), "version", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise SMLRuntimeError(
                f"lm-eval task dataset split fingerprint is missing: {split_name}"
            )
        version_text = str(version) if version is not None else ""
        if not version_text:
            raise SMLRuntimeError(
                f"lm-eval task dataset split version is missing: {split_name}"
            )
        fingerprints.append((split_name, fingerprint))
        versions.append(version_text)

    dataset_kwargs = config.get("dataset_kwargs")
    if dataset_kwargs is None:
        dataset_kwargs = {}
    if not isinstance(dataset_kwargs, Mapping):
        raise SMLRuntimeError("lm-eval task dataset kwargs are malformed")
    revision = dataset_kwargs.get("revision")
    if revision is not None:
        if not isinstance(revision, str) or not revision:
            raise SMLRuntimeError("lm-eval task dataset revision is malformed")
        dataset_revision = revision
    else:
        if len(set(versions)) != 1:
            raise SMLRuntimeError("lm-eval task dataset split versions disagree")
        dataset_revision = f"version:{versions[0]}"
    return (
        dataset_revision,
        structured_identity("sml-evaluation-dataset-v1", tuple(fingerprints)),
    )


def _effective_filter_pipeline(
    config: Mapping[str, object],
    config_object: object,
) -> object:
    filter_list = config.get("filter_list")
    if filter_list is None:
        return (
            {
                "name": "none",
                "filter": ({"function": "take_first"},),
            },
        )
    return _serialize_provider_value(config_object, filter_list)


def _lm_eval_source_commit() -> str | None:
    try:
        distribution = importlib.metadata.distribution("lm-eval")
    except importlib.metadata.PackageNotFoundError as error:
        raise SMLRuntimeError(
            "lm-eval distribution metadata cannot be resolved"
        ) from error
    distribution_path = getattr(distribution, "_path", None)
    if not isinstance(distribution_path, Path):
        raise SMLRuntimeError("lm-eval distribution metadata path is malformed")
    direct_url_path = distribution_path / "direct_url.json"
    if not direct_url_path.is_file():
        return None
    try:
        direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SMLRuntimeError("lm-eval direct_url metadata is malformed") from error
    if not isinstance(direct_url, Mapping):
        raise SMLRuntimeError("lm-eval direct_url metadata is malformed")
    if "vcs_info" not in direct_url:
        return None
    vcs_info = direct_url["vcs_info"]
    if not isinstance(vcs_info, Mapping):
        raise SMLRuntimeError("lm-eval direct_url VCS metadata is malformed")
    commit_id = vcs_info.get("commit_id")
    if not isinstance(commit_id, str) or not commit_id:
        raise SMLRuntimeError("lm-eval direct_url source commit is malformed")
    return commit_id


def _metric_payload(
    provider_result: Mapping[str, object],
    task_name: str,
) -> dict[str, object]:
    results = provider_result.get("results")
    if not isinstance(results, Mapping) or task_name not in results:
        raise SMLRuntimeError(f"lm-eval metric result is missing for task: {task_name}")
    metrics = results[task_name]
    if not isinstance(metrics, Mapping):
        raise SMLRuntimeError(
            f"lm-eval metric result is malformed for task: {task_name}"
        )
    payload: dict[str, object] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise SMLRuntimeError("lm-eval metric key is malformed")
        if key in payload:
            raise SMLRuntimeError("lm-eval metric keys are ambiguous or duplicate")
        payload[key] = normalize_json_value(value, context=f"lm-eval metric {key}")
    if not payload:
        raise SMLRuntimeError(f"lm-eval metric result is empty for task: {task_name}")
    return payload


def _resolve_task_record(
    *,
    task_name: str,
    task: object,
    provider: _LMEvalProvider,
    manager: _RecordingTaskManager,
    recorder: _EvaluationRequestRecorder,
    provider_result: Mapping[str, object],
    limit: int | None,
    padding: Literal["left", "right"],
    seeds: Mapping[str, object],
    provider_versions: tuple[tuple[str, str], ...],
    lm_eval_source_commit: str | None = None,
) -> EvaluationTaskRecord:
    if not isinstance(task_name, str) or not task_name:
        raise SMLRuntimeError("lm-eval task name is invalid")
    task_index = getattr(manager, "task_index", None)
    if not isinstance(task_index, Mapping) or task_name not in task_index:
        raise SMLRuntimeError(f"lm-eval task index entry is missing: {task_name}")
    entry = task_index[task_name]
    task_yaml, include_closure = _resolve_yaml_sources(
        provider, getattr(entry, "yaml_path", None)
    )
    config, config_object = _task_config(task)
    task_metadata_version = _metadata_version(config)
    dataset_revision, dataset_fingerprint = _dataset_identity(task, config)
    if not isinstance(provider_result, Mapping):
        raise SMLRuntimeError("lm-eval provider result is malformed")
    provider_records = tuple(
        EvaluationProviderVersion(name=name, version=version)
        for name, version in provider_versions
    )
    prompt_config = {
        "output_type": config.get("output_type"),
        "description": config.get("description"),
        "process_docs": config.get("process_docs"),
        "doc_to_text": config.get("doc_to_text"),
        "doc_to_target": config.get("doc_to_target"),
        "doc_to_choice": config.get("doc_to_choice"),
        "target_delimiter": config.get("target_delimiter"),
        "fewshot_delimiter": config.get("fewshot_delimiter"),
        "gen_prefix": config.get("gen_prefix"),
        "system_instruction": None,
        "apply_chat_template": False,
        "adapter_padding": padding,
    }
    few_shot_config = {
        "num_fewshot": config.get("num_fewshot"),
        "fewshot_split": config.get("fewshot_split"),
        "fewshot_config": _serialize_provider_value(
            config_object, config.get("fewshot_config")
        ),
        "fewshot_as_multiturn": True,
    }
    generation_config = {
        "generation_kwargs": config.get("generation_kwargs"),
        "provider_gen_kwargs": None,
    }
    metric_normalization_config = {
        "metric_list": config.get("metric_list"),
        "filter_list": _effective_filter_pipeline(config, config_object),
        "repeats": config.get("repeats"),
        "should_decontaminate": config.get("should_decontaminate"),
        "doc_to_decontamination_query": config.get("doc_to_decontamination_query"),
        "bootstrap_iters": _BOOTSTRAP_ITERS,
        "log_samples": True,
        "predict_only": False,
    }
    provisional = EvaluationTaskRecord(
        task_name=task_name,
        task_identity="sha256:" + "0" * 64,
        task_yaml=task_yaml,
        include_template_closure=include_closure,
        task_metadata_version=task_metadata_version,
        prompt_config=prompt_config,
        few_shot_config=few_shot_config,
        generation_config=generation_config,
        metric_normalization_config=metric_normalization_config,
        seeds=dict(seeds),
        limit=limit,
        ordered_request_identity=recorder.identity_for(task_name),
        lm_eval_package_version=provider.package_version,
        lm_eval_source_commit=lm_eval_source_commit,
        dataset_revision=dataset_revision,
        dataset_fingerprint=dataset_fingerprint,
        provider_versions=provider_records,
        metric_payload=_metric_payload(provider_result, task_name),
    )
    return replace(provisional, task_identity=evaluation_task_identity(provisional))


def _encode_text(processor, text: str) -> list[int]:
    return [int(token) for token in processor.encode(text)]


def _encode_loglikelihood_request(
    session: InferenceSession,
    request: LoglikelihoodRequest,
) -> tuple[tuple[int, ...], int]:
    processor = session.resolved_model.tokenizer.processor
    context_ids = _encode_text(processor, request.context)
    full_ids = _encode_text(processor, request.context + request.continuation)
    if full_ids[: len(context_ids)] != context_ids:
        continuation_ids = _encode_text(processor, request.continuation)
        continuation_start = len(full_ids) - len(continuation_ids)
    else:
        continuation_start = len(context_ids)

    if continuation_start == 0:
        bos_id = int(processor.bos_id())
        prefix_id = session.resolved_model.model_config.bos_token_id
        if bos_id < 0:
            raise SMLRuntimeError("empty context requires a usable prefix token")
        full_ids = [prefix_id, *full_ids]
        continuation_start += 1

    if continuation_start < 0:
        raise SMLRuntimeError("invalid continuation boundary")
    if continuation_start >= len(full_ids) and request.continuation:
        raise SMLRuntimeError("continuation produced no tokens")
    if not full_ids:
        raise SMLRuntimeError("empty context requires a usable prefix token")
    return tuple(full_ids), continuation_start


def score_loglikelihood_batch(
    session: InferenceSession,
    requests: Sequence[LoglikelihoodRequest],
    *,
    padding: str,
) -> tuple[LoglikelihoodResult, ...]:
    if padding not in ("left", "right"):
        raise ValueError("padding must be 'left' or 'right'")
    encoded = tuple(
        _encode_loglikelihood_request(session, request) for request in requests
    )
    raw = session.score_encoded_loglikelihoods(encoded, padding=padding)
    return tuple(
        LoglikelihoodResult(log_likelihood=log_likelihood, greedy_match=greedy_match)
        for log_likelihood, greedy_match in raw
    )


def _truncate_at_stop(text: str, until: object) -> str:
    if isinstance(until, str):
        stop_sequences = [until]
    else:
        stop_sequences = list(until or [])
    positions = [
        position
        for stop in stop_sequences
        if stop and (position := text.find(stop)) >= 0
    ]
    return text[: min(positions)] if positions else text


class SMLEvalLM:
    def __init__(
        self,
        session: InferenceSession,
        *,
        padding: Literal["left", "right"] = "right",
        recorder: _EvaluationRequestRecorder | None = None,
    ) -> None:
        self._session = session
        self._padding = padding
        self._recorder = recorder

    def loglikelihood(self, requests):
        requests = tuple(requests)
        if self._recorder is not None:
            self._recorder.record("loglikelihood", requests)
        scored = score_loglikelihood_batch(
            self._session,
            [
                LoglikelihoodRequest(
                    context=request.args[0],
                    continuation=request.args[1],
                )
                for request in requests
            ],
            padding=self._padding,
        )
        return [(result.log_likelihood, result.greedy_match) for result in scored]

    def generate_until(self, requests) -> list[str]:
        requests = tuple(requests)
        if self._recorder is not None:
            self._recorder.record("generate_until", requests)
        items: list[tuple[str, GenerationRequest]] = []
        stop_lists: list[object] = []
        for request in requests:
            context, generation_kwargs = request.args
            if generation_kwargs.get("do_sample", False):
                raise SMLRuntimeError("SMLEvalLM only supports greedy generation")
            max_new_tokens = int(generation_kwargs.get("max_gen_toks", 256))
            items.append(
                (
                    context,
                    GenerationRequest(
                        max_new_tokens=max_new_tokens,
                        config=GenerationConfig(),
                    ),
                )
            )
            stop_lists.append(generation_kwargs.get("until") or [])
        generated = self._session.generate_batch(items)
        return [
            _truncate_at_stop(result.text, until)
            for result, until in zip(generated, stop_lists, strict=True)
        ]

    def loglikelihood_rolling(self, requests):
        del requests
        raise SMLRuntimeError("unsupported evaluation method: loglikelihood_rolling")


def _lm_eval_model(
    session: InferenceSession,
    padding: Literal["left", "right"],
    recorder: _EvaluationRequestRecorder | None = None,
):
    from lm_eval.api.model import LM

    class _SMLEvalLMAdapter(LM):
        def __init__(self, inner: SMLEvalLM) -> None:
            super().__init__()
            self._inner = inner

        def loglikelihood(self, requests):
            return self._inner.loglikelihood(requests)

        def generate_until(self, requests):
            return self._inner.generate_until(requests)

        def loglikelihood_rolling(self, requests):
            return self._inner.loglikelihood_rolling(requests)

    return _SMLEvalLMAdapter(SMLEvalLM(session, padding=padding, recorder=recorder))


def _provider_versions() -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for name in _PROVIDER_PACKAGES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise SMLRuntimeError(
                f"required evaluation provider package is missing: {name}"
            ) from error
        if not isinstance(version, str) or not version:
            raise SMLRuntimeError(
                f"required evaluation provider package version is missing: {name}"
            )
        found.append((name, version))
    return tuple(sorted(found, key=lambda item: item[0]))


def _exact_task_mapping(
    value: object,
    requested_tasks: tuple[str, ...],
    *,
    context: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SMLRuntimeError(f"lm-eval provider {context} is not a mapping")
    task_names = tuple(value)
    if (
        any(not isinstance(task_name, str) or not task_name for task_name in task_names)
        or len(task_names) != len(set(task_names))
        or set(task_names) != set(requested_tasks)
    ):
        raise SMLRuntimeError(
            f"lm-eval provider {context} must contain exactly the requested task keys"
        )
    return value


def _validate_provider_result(
    raw_result: object,
    requested_tasks: tuple[str, ...],
) -> Mapping[str, object]:
    if not isinstance(raw_result, Mapping):
        raise SMLRuntimeError("lm-eval provider result is not a mapping")
    results = _exact_task_mapping(
        raw_result.get("results"), requested_tasks, context="results"
    )
    for task_name in requested_tasks:
        metrics = results.get(task_name)
        if not isinstance(metrics, Mapping):
            raise SMLRuntimeError(
                f"lm-eval metric result is malformed for task: {task_name}"
            )
    return raw_result


def _serialize_provider_configs(
    raw_result: Mapping[str, object],
    requested_tasks: tuple[str, ...],
    loaded_tasks: tuple[object, ...],
) -> Mapping[str, object]:
    configs = _exact_task_mapping(
        raw_result.get("configs"), requested_tasks, context="configs"
    )
    serialized = dict(raw_result)
    serialized["configs"] = {
        task_name: _serialize_provider_value(
            getattr(task, "config", None), configs[task_name]
        )
        for task_name, task in zip(requested_tasks, loaded_tasks, strict=True)
    }
    return serialized


def _loaded_tasks(
    manager: _RecordingTaskManager,
    requested_tasks: tuple[str, ...],
) -> tuple[object, ...]:
    loaded_tasks = manager.loaded.get("tasks")
    if not isinstance(loaded_tasks, Mapping):
        raise SMLRuntimeError("lm-eval task manager returned malformed task mapping")
    loaded_task_names = tuple(loaded_tasks)
    if (
        any(
            not isinstance(task_name, str) or not task_name
            for task_name in loaded_task_names
        )
        or len(loaded_task_names) != len(set(loaded_task_names))
        or set(loaded_task_names) != set(requested_tasks)
    ):
        raise SMLRuntimeError(
            "lm-eval loaded tasks must contain exactly the requested task keys"
        )
    ordered_tasks = tuple(loaded_tasks[task_name] for task_name in requested_tasks)
    for task_name, task in zip(requested_tasks, ordered_tasks, strict=True):
        if getattr(task, "task_name", None) != task_name:
            raise SMLRuntimeError(
                "lm-eval loaded task task_name does not match its mapping key"
            )
    return ordered_tasks


def _validate_recorded_task_names(
    recorder: _EvaluationRequestRecorder,
    requested_tasks: tuple[str, ...],
) -> None:
    if recorder.task_names != frozenset(requested_tasks):
        raise SMLRuntimeError(
            "lm-eval recorded requests must contain exactly the requested tasks"
        )


def evaluate(config: EvaluationConfig) -> EvaluationResult:
    if not isinstance(config, EvaluationConfig):
        raise TypeError("config must be an EvaluationConfig")
    session = InferenceSession.from_checkpoint(
        config.checkpoint,
        full_verify=config.full_verify,
        runtime=config.runtime,
    )
    provider = _import_lm_eval()
    manager = _RecordingTaskManager(provider.make_task_manager())
    recorder = _EvaluationRequestRecorder()
    lm = _lm_eval_model(session, padding=config.padding, recorder=recorder)
    raw_result = provider.simple_evaluate(
        model=lm,
        tasks=list(config.tasks),
        num_fewshot=0,
        batch_size=None,
        device=None,
        limit=config.limit,
        task_manager=manager,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=True,
        gen_kwargs=None,
        bootstrap_iters=_BOOTSTRAP_ITERS,
        log_samples=True,
        predict_only=False,
        **_evaluation_seeds(),
    )
    provider_result = _validate_provider_result(raw_result, config.tasks)
    loaded_tasks = _loaded_tasks(manager, config.tasks)
    provider_result = _serialize_provider_configs(
        provider_result, config.tasks, loaded_tasks
    )
    _validate_recorded_task_names(recorder, config.tasks)
    provider_versions = _provider_versions()
    lm_eval_source_commit = _lm_eval_source_commit()
    task_records = tuple(
        _resolve_task_record(
            task_name=task_name,
            task=task,
            provider=provider,
            manager=manager,
            recorder=recorder,
            provider_result=provider_result,
            limit=config.limit,
            padding=config.padding,
            seeds=_evaluation_seeds(),
            provider_versions=provider_versions,
            lm_eval_source_commit=lm_eval_source_commit,
        )
        for task_name, task in zip(config.tasks, loaded_tasks, strict=True)
    )
    result = EvaluationResult(
        kind="evaluation-result",
        version=1,
        identity="sha256:" + "0" * 64,
        model=session.model_identity,
        tasks=task_records,
        provider_result=normalize_json_value(provider_result, context="lm-eval result"),
    )
    result = replace(result, identity=evaluation_result_identity(result))
    publish_evaluation_result(config.output, result)
    return result


__all__ = (
    "EvaluationConfig",
    "EvaluationProviderVersion",
    "EvaluationResult",
    "EvaluationSourceIdentity",
    "EvaluationTaskRecord",
    "LoglikelihoodRequest",
    "LoglikelihoodResult",
    "SMLEvalLM",
    "evaluate",
    "read_evaluation_result",
    "score_loglikelihood_batch",
)
