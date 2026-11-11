from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sml.artifacts.manifest import (
    canonical_json_bytes,
    structured_identity,
)
from sml.errors import SMLArtifactError
from sml.evaluation_result import (
    EvaluationProviderVersion,
    EvaluationResult,
    EvaluationSourceIdentity,
    evaluation_result_bytes,
    evaluation_result_identity,
    evaluation_task_identity,
    normalize_json_value,
    publish_evaluation_result,
    read_evaluation_result,
)

from .result_helpers import (
    identified_result,
    make_result,
    make_task_record,
    model_identity,
)


class DuplicateTaskKeyMapping(Mapping[str, object]):
    """A hostile Mapping whose iteration exposes the same task key twice."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __getitem__(self, key: str) -> object:
        if key != "hellaswag":
            raise KeyError(key)
        return self._value

    def __iter__(self):
        yield "hellaswag"
        yield "hellaswag"

    def __len__(self) -> int:
        return 2


def test_result_identity_binds_export_manifest(tmp_path: Path) -> None:
    result = identified_result()
    first = replace(
        result,
        version=3,
        model=replace(result.model, artifact_identity="sha256:" + "a" * 64),
    )
    first = replace(first, identity=evaluation_result_identity(first))
    second = replace(
        first,
        model=replace(first.model, artifact_identity="sha256:" + "b" * 64),
    )
    second = replace(second, identity=evaluation_result_identity(second))

    assert first.identity != second.identity
    assert first.tasks == second.tasks
    path = tmp_path / "v3.json"
    publish_evaluation_result(path, first)
    restored = read_evaluation_result(path)
    assert restored == first
    assert restored.model.artifact_identity == "sha256:" + "a" * 64


@pytest.mark.parametrize("artifact_identity", (None, "invalid"))
def test_export_requires_manifest_identity(artifact_identity: object) -> None:
    result = identified_result()
    with pytest.raises((TypeError, ValueError), match="artifact_identity"):
        replace(
            result,
            version=3,
            model=replace(result.model, artifact_identity=artifact_identity),
        )


@pytest.mark.parametrize("version", (1, 2))
def test_obsolete_evaluation_versions_are_rejected(
    tmp_path: Path, version: int
) -> None:
    result = identified_result()
    with pytest.raises(ValueError, match="version must be 3"):
        replace(result, version=version)
    payload = json.loads(evaluation_result_bytes(result))
    payload["version"] = version
    path = tmp_path / "obsolete.json"
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(SMLArtifactError, match="invalid evaluation result"):
        read_evaluation_result(path)


def test_json_normalization_is_closed_finite_and_deeply_immutable() -> None:
    normalized = normalize_json_value(
        {"z": np.int64(3), "a": [True, np.float32(1.25), None]},
        context="provider result",
    )
    assert tuple(normalized) == ("a", "z")
    assert normalized["a"] == (True, 1.25, None)
    with pytest.raises(TypeError):
        normalized["z"] = 4  # type: ignore[index]
    for invalid in (float("nan"), float("inf"), b"bytes", Path("local"), {1: "x"}):
        with pytest.raises((TypeError, ValueError)):
            normalize_json_value(invalid, context="provider result")


@pytest.mark.parametrize("value", ("bad\ud800value", {"bad\ud800key": "value"}))
def test_normalization_rejects_strings_that_cannot_be_encoded_as_utf8(
    value: object,
) -> None:
    """Breaks if a persisted JSON string survives normalization but cannot encode."""
    with pytest.raises(ValueError, match="UTF-8"):
        normalize_json_value(value, context="provider result")


def test_record_strings_reject_values_that_cannot_be_encoded_as_utf8() -> None:
    """Breaks if schema strings defer UTF-8 failure until identity serialization."""
    with pytest.raises(ValueError, match="UTF-8"):
        replace(make_task_record(), task_metadata_version="bad\ud800version")


def test_numpy_normalization_accepts_only_lossless_json_scalar_kinds() -> None:
    minimum = np.int64(np.iinfo(np.int64).min)
    maximum = np.int64(np.iinfo(np.int64).max)
    assert normalize_json_value(np.bool_(True), context="provider result") is True
    assert normalize_json_value(minimum, context="provider result") == int(minimum)
    assert normalize_json_value(maximum, context="provider result") == int(maximum)
    assert (
        normalize_json_value(
            np.float64(np.finfo(np.float64).max), context="provider result"
        )
        == np.finfo(np.float64).max
    )
    assert normalize_json_value(np.longdouble("1.5"), context="provider result") == 1.5

    with np.errstate(over="ignore"):
        outside_float_range = np.longdouble(np.finfo(float).max) * 2
    for invalid in (
        outside_float_range,
        np.datetime64("2026-08-25"),
        np.complex128(1 + 2j),
        np.str_("provider string"),
    ):
        with pytest.raises((TypeError, ValueError)):
            normalize_json_value(invalid, context="provider result")


def test_source_identity_requires_a_portable_logical_path() -> None:
    for logical_name in ("/private/tmp/task.yaml", "C:\\task.yaml", "../task.yaml"):
        with pytest.raises(SMLArtifactError):
            EvaluationSourceIdentity(
                logical_name=logical_name,
                content_identity="sha256:" + "2" * 64,
            )


def test_identity_projections_are_pinned_selective_and_complete() -> None:
    task = make_task_record()
    expected_task_identity = (
        "sha256:d429fd4328a410283b8d0c490f03a3c9fba8975837ecfe088aad5072ef1b6b65"
    )
    assert evaluation_task_identity(task) == expected_task_identity
    assert (
        evaluation_task_identity(
            replace(
                task,
                task_identity="sha256:" + "5" * 64,
                metric_payload={"acc,none": 0.75},
            )
        )
        == expected_task_identity
    )

    for changed_task in (
        replace(task, task_name="winogrande"),
        replace(
            task,
            task_yaml=EvaluationSourceIdentity(
                logical_name="tasks/hellaswag.yaml",
                content_identity="sha256:" + "6" * 64,
            ),
        ),
        replace(task, include_template_closure=()),
        replace(task, task_metadata_version="2.0"),
        replace(task, prompt_config={"description": "changed"}),
        replace(task, few_shot_config={"num_fewshot": 0}),
        replace(task, generation_config={"temperature": 1.0}),
        replace(task, metric_normalization_config={"acc,none": "sum"}),
        replace(task, seeds={"fewshot": 4321}),
        replace(task, limit=None),
        replace(task, ordered_request_identity="sha256:" + "7" * 64),
        replace(task, lm_eval_package_version="0.4.13"),
        replace(task, lm_eval_source_commit="commit-1"),
        replace(task, dataset_revision="revision-2"),
        replace(task, dataset_fingerprint="fingerprint-2"),
        replace(
            task,
            provider_versions=(
                EvaluationProviderVersion(name="datasets", version="3.1.0"),
                EvaluationProviderVersion(name="lm-eval", version="0.4.12"),
            ),
        ),
    ):
        assert evaluation_task_identity(changed_task) != expected_task_identity

    identified_task = replace(task, task_identity=expected_task_identity)
    result = make_result(model=model_identity(), tasks=(identified_task,))
    expected_result_identity = (
        "sha256:aa6359861ef530588dbadec386ae216eebb8d932aed8dc0a60576eba91d7180a"
    )
    assert evaluation_result_identity(result) == expected_result_identity
    for changed_result in (
        replace(result, model=replace(model_identity(), step=8)),
        replace(
            result,
            model=replace(model_identity(), latest_recovered=True),
        ),
        replace(
            result,
            model=replace(model_identity(), pruning_pending=True),
        ),
        replace(
            make_result(
                model=model_identity(),
                tasks=(replace(identified_task, metric_payload={"acc,none": 0.75}),),
            ),
            identity="sha256:" + "0" * 64,
        ),
    ):
        assert evaluation_result_identity(changed_result) != expected_result_identity


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_kind", ""),
        ("run_identity", "not-an-identity"),
        ("step", True),
        ("step", -1),
        ("checkpoint_identity", "not-an-identity"),
        ("run_step_identity", "not-an-identity"),
        ("tokenizer_identity", "not-an-identity"),
        ("verification", "full"),
        ("latest_recovered", 1),
        ("pruning_pending", None),
    ),
)
def test_result_rejects_malformed_model_identity(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_result(
            model=replace(model_identity(), **{field: value}),
            tasks=(make_task_record(),),
        )


def test_records_reject_unordered_or_invalid_values() -> None:
    with pytest.raises(ValueError):
        EvaluationProviderVersion(name="lm-eval", version="")
    with pytest.raises(ValueError):
        replace(make_task_record(), limit=0)
    with pytest.raises(ValueError):
        replace(
            make_task_record(),
            provider_versions=(
                EvaluationProviderVersion(name="lm-eval", version="0.4.12"),
                EvaluationProviderVersion(name="datasets", version="3.0.0"),
            ),
        )
    with pytest.raises(ValueError):
        make_result(model=model_identity(), tasks=())
    with pytest.raises(ValueError):
        replace(
            make_result(model=model_identity(), tasks=(make_task_record(),)),
            version=3.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latest_recovered", None),
        ("latest_recovered", 0),
        ("pruning_pending", None),
        ("pruning_pending", 0),
    ),
)
def test_reader_requires_boolean_operational_status(
    tmp_path: Path, field: str, value: object
) -> None:
    """Breaks if an operational status can be null or merely truthy/falsy."""
    document = json.loads(evaluation_result_bytes(identified_result()))
    document["model"][field] = value
    projection = dict(document)
    projection.pop("identity")
    document["identity"] = structured_identity("sml-evaluation-result-v3", projection)
    path = tmp_path / f"invalid-status-{field}-{value!r}.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(SMLArtifactError):
        read_evaluation_result(path)


def test_reader_rejects_strict_schema_and_tampered_identities(tmp_path: Path) -> None:
    """Breaks if the reader accepts malformed or altered persisted results."""
    result = identified_result()
    canonical = evaluation_result_bytes(result)
    path = tmp_path / "evaluation.json"
    path.write_bytes(canonical)
    assert read_evaluation_result(path) == result

    decoded = json.loads(canonical)
    boolean_version = dict(decoded)
    boolean_version["version"] = True
    boolean_step = json.loads(canonical)
    boolean_step["model"]["step"] = True
    boolean_limit = json.loads(canonical)
    boolean_limit["tasks"][0]["limit"] = False
    invalid_verification = json.loads(canonical)
    invalid_verification["model"]["verification"] = "invalid"

    documents = (
        canonical.replace(b'"version":3', b'"version":3,"extra":true'),
        canonical.replace(
            b'"kind":"evaluation-result"',
            b'"kind":"evaluation-result","kind":"evaluation-result"',
        ),
        canonical.replace(b'"provider_result":', b'"extra":true,"provider_result":'),
        json.dumps(json.loads(canonical), indent=2, sort_keys=True).encode() + b"\n",
        canonical.replace(result.identity.encode(), ("sha256:" + "f" * 64).encode()),
        canonical.replace(b"0.5", b"NaN", 1),
        canonical_json_bytes(boolean_version) + b"\n",
        canonical_json_bytes(boolean_step) + b"\n",
        canonical_json_bytes(boolean_limit) + b"\n",
        canonical_json_bytes(invalid_verification) + b"\n",
    )
    for index, payload in enumerate(documents):
        bad = tmp_path / f"bad-{index}.json"
        bad.write_bytes(payload)
        with pytest.raises(SMLArtifactError):
            read_evaluation_result(bad)


@pytest.mark.parametrize(
    ("record_name", "fields"),
    (
        (
            "result",
            ("kind", "version", "identity", "model", "tasks", "provider_result"),
        ),
        (
            "model",
            (
                "artifact_kind",
                "artifact_identity",
                "run_identity",
                "step",
                "checkpoint_identity",
                "run_step_identity",
                "tokenizer_identity",
                "verification",
                "latest_recovered",
                "pruning_pending",
            ),
        ),
        (
            "task",
            (
                "task_name",
                "task_identity",
                "task_yaml",
                "include_template_closure",
                "task_metadata_version",
                "prompt_config",
                "few_shot_config",
                "generation_config",
                "metric_normalization_config",
                "seeds",
                "limit",
                "ordered_request_identity",
                "lm_eval_package_version",
                "lm_eval_source_commit",
                "dataset_revision",
                "dataset_fingerprint",
                "provider_versions",
                "metric_payload",
            ),
        ),
        ("primary source", ("logical_name", "content_identity")),
        ("include source", ("logical_name", "content_identity")),
        ("provider version", ("name", "version")),
    ),
)
def test_reader_requires_exact_fields_for_every_persisted_record_shape(
    tmp_path: Path, record_name: str, fields: tuple[str, ...]
) -> None:
    """Breaks if any persisted nested record accepts a missing or extra field."""
    canonical = evaluation_result_bytes(identified_result())

    def target(document: dict[str, object]) -> dict[str, object]:
        task = document["tasks"][0]  # type: ignore[index]
        if record_name == "result":
            return document
        if record_name == "model":
            return document["model"]  # type: ignore[return-value]
        if record_name == "task":
            return task
        if record_name == "primary source":
            return task["task_yaml"]  # type: ignore[index, return-value]
        if record_name == "include source":
            return task["include_template_closure"][0]  # type: ignore[index, return-value]
        return task["provider_versions"][0]  # type: ignore[index, return-value]

    for field in fields:
        document = json.loads(canonical)
        record = target(document)
        record.pop(field)
        path = tmp_path / f"{record_name}-missing-{field}.json"
        path.write_bytes(canonical_json_bytes(document) + b"\n")
        with pytest.raises(SMLArtifactError):
            read_evaluation_result(path)

    document = json.loads(canonical)
    target(document)["unexpected"] = True
    path = tmp_path / f"{record_name}-extra.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    with pytest.raises(SMLArtifactError):
        read_evaluation_result(path)


@pytest.mark.parametrize(
    "provider_result",
    (
        {"results": {"hellaswag": {"acc,none": 0.5}}},
        {
            "results": {"hellaswag": {"acc,none": 0.5}},
            "configs": {"other": {}},
        },
        {
            "results": {"hellaswag": {"acc,none": 0.5}, "other": {}},
            "configs": {"hellaswag": {}, "other": {}},
        },
        {
            "results": {"hellaswag": {"acc,none": 0.75}},
            "configs": {"hellaswag": {}},
        },
    ),
)
def test_result_rejects_provider_task_key_or_metric_contradictions(
    provider_result: object,
) -> None:
    """Breaks if a constructible result can contradict its task records."""
    with pytest.raises((TypeError, ValueError)):
        EvaluationResult(
            kind="evaluation-result",
            version=3,
            identity="sha256:" + "0" * 64,
            model=model_identity(),
            tasks=(make_task_record(),),
            provider_result=provider_result,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "expected,contradictory",
    [
        pytest.param(0.5, 0.75, id="different-value"),
        pytest.param(1, True, id="different-json-type"),
    ],
)
def test_reader_rejects_self_identifying_metric_contradictions(
    tmp_path, expected, contradictory
):
    document = json.loads(
        evaluation_result_bytes(identified_result(metric_payload={"m": expected}))
    )
    document["provider_result"]["results"]["hellaswag"]["m"] = contradictory
    projection = dict(document)
    projection.pop("identity")
    document["identity"] = structured_identity("sml-evaluation-result-v3", projection)
    path = tmp_path / "contradictory.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(SMLArtifactError):
        read_evaluation_result(path)


def test_result_rejects_type_distinct_provider_metrics() -> None:
    """Breaks if JSON true can satisfy a task metric whose value is the number 1."""
    with pytest.raises(ValueError, match="metric_payload"):
        EvaluationResult(
            kind="evaluation-result",
            version=3,
            identity="sha256:" + "0" * 64,
            model=model_identity(),
            tasks=(make_task_record(metric_payload={"m": 1}),),
            provider_result={
                "results": {"hellaswag": {"m": True}},
                "configs": {"hellaswag": {}},
            },
        )


@pytest.mark.parametrize("mapping_name", ("results", "configs"))
def test_result_rejects_duplicate_task_keys_from_hostile_mappings(
    mapping_name: str,
) -> None:
    """Breaks if normalization collapses duplicate Mapping iteration keys."""
    provider_result: dict[str, object] = {
        "results": {"hellaswag": {"acc,none": 0.5}},
        "configs": {"hellaswag": {}},
    }
    provider_result[mapping_name] = DuplicateTaskKeyMapping(
        {"acc,none": 0.5} if mapping_name == "results" else {}
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        EvaluationResult(
            kind="evaluation-result",
            version=3,
            identity="sha256:" + "0" * 64,
            model=model_identity(),
            tasks=(make_task_record(),),
            provider_result=provider_result,  # type: ignore[arg-type]
        )
