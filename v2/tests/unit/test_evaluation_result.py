from __future__ import annotations

import json
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import sml.evaluation_result as evaluation_result_module
from sml.artifacts.manifest import (
    VerificationLevel,
    canonical_json_bytes,
    structured_identity,
)
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.evaluation_result import (
    EvaluationProviderVersion,
    EvaluationResult,
    EvaluationSourceIdentity,
    EvaluationTaskRecord,
    evaluation_result_bytes,
    evaluation_result_identity,
    evaluation_task_identity,
    normalize_json_value,
    publish_evaluation_result,
    read_evaluation_result,
)
from sml.inference import ModelIdentity


def model_identity() -> ModelIdentity:
    return ModelIdentity(
        artifact_kind="export",
        run_identity=None,
        step=7,
        checkpoint_identity=None,
        run_step_identity=None,
        tokenizer_identity="sha256:" + "1" * 64,
        verification=VerificationLevel.FULL,
    )


def make_task_record(
    *,
    metric_payload: object = {"acc,none": 0.5},
) -> EvaluationTaskRecord:
    return EvaluationTaskRecord(
        task_name="hellaswag",
        task_identity="sha256:" + "0" * 64,
        task_yaml=EvaluationSourceIdentity(
            logical_name="tasks/hellaswag.yaml",
            content_identity="sha256:" + "2" * 64,
        ),
        include_template_closure=(
            EvaluationSourceIdentity(
                logical_name="tasks/common.yaml",
                content_identity="sha256:" + "3" * 64,
            ),
        ),
        task_metadata_version="1.0",
        prompt_config={"description": "multiple choice"},
        few_shot_config={"num_fewshot": 5},
        generation_config={"temperature": 0.0},
        metric_normalization_config={"acc,none": "mean"},
        seeds={"fewshot": 1234},
        limit=10,
        ordered_request_identity="sha256:" + "4" * 64,
        lm_eval_package_version="0.4.12",
        lm_eval_source_commit=None,
        dataset_revision="revision-1",
        dataset_fingerprint="fingerprint-1",
        provider_versions=(
            EvaluationProviderVersion(name="datasets", version="3.0.0"),
            EvaluationProviderVersion(name="lm-eval", version="0.4.12"),
        ),
        metric_payload=metric_payload,
    )


def make_result(
    *, model: ModelIdentity, tasks: tuple[EvaluationTaskRecord, ...]
) -> EvaluationResult:
    return EvaluationResult(
        kind="evaluation-result",
        version=1,
        identity="sha256:" + "0" * 64,
        model=model,
        tasks=tasks,
        provider_result={
            "results": {task.task_name: task.metric_payload for task in tasks},
            "configs": {task.task_name: {} for task in tasks},
        },
    )


def identified_result(
    *, metric_payload: object = {"acc,none": 0.5}
) -> EvaluationResult:
    task = make_task_record(metric_payload=metric_payload)
    task = replace(task, task_identity=evaluation_task_identity(task))
    result = make_result(model=model_identity(), tasks=(task,))
    return replace(result, identity=evaluation_result_identity(result))


def _publish_fifo_result(path: Path, outcomes) -> None:
    try:
        publish_evaluation_result(path, identified_result())
    except SMLRuntimeError as error:
        outcomes.put(str(error))
    else:
        outcomes.put("accepted")


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


def test_task_and_result_identities_cover_metrics_requests_and_model() -> None:
    task = make_task_record(metric_payload={"acc,none": 0.5})
    task = replace(task, task_identity=evaluation_task_identity(task))
    result = make_result(model=model_identity(), tasks=(task,))
    result = replace(result, identity=evaluation_result_identity(result))
    changed_metric = replace(task, metric_payload={"acc,none": 0.75})
    changed_request = replace(task, ordered_request_identity="sha256:" + "2" * 64)
    assert (
        evaluation_result_identity(
            replace(
                make_result(model=model_identity(), tasks=(changed_metric,)),
                identity="sha256:" + "0" * 64,
            )
        )
        != result.identity
    )
    assert evaluation_task_identity(changed_request) != task.task_identity
    assert (
        evaluation_result_identity(
            replace(
                result,
                identity="sha256:" + "0" * 64,
                model=replace(model_identity(), step=8),
            )
        )
        != result.identity
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
        "sha256:cdd96169a9ea984c07b065378d465e3ca160db14209761b4d64cbad94f8ad6f8"
    )
    assert evaluation_result_identity(result) == expected_result_identity
    for changed_result in (
        replace(result, model=replace(model_identity(), step=8)),
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
            version=1.0,
        )


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
        canonical.replace(b'"version":1', b'"version":1,"extra":true'),
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
                "run_identity",
                "step",
                "checkpoint_identity",
                "run_step_identity",
                "tokenizer_identity",
                "verification",
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
            version=1,
            identity="sha256:" + "0" * 64,
            model=model_identity(),
            tasks=(make_task_record(),),
            provider_result=provider_result,  # type: ignore[arg-type]
        )


def test_reader_rejects_a_freshly_identifiable_provider_metric_contradiction(
    tmp_path: Path,
) -> None:
    """Breaks if strict reading accepts a self-identifying contradictory artifact."""
    document = json.loads(evaluation_result_bytes(identified_result()))
    document["provider_result"]["results"]["hellaswag"]["acc,none"] = 0.75
    projection = dict(document)
    projection.pop("identity")
    document["identity"] = structured_identity("sml-evaluation-result-v1", projection)
    path = tmp_path / "contradictory.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")

    with pytest.raises(SMLArtifactError):
        read_evaluation_result(path)


def test_concurrent_publication_is_atomic_idempotent_and_never_overwrites(
    tmp_path: Path,
) -> None:
    """Breaks if publishing can expose partial bytes or replace a winner."""
    path = tmp_path / "nested" / "evaluation.json"
    path.parent.mkdir()
    first_result = identified_result()
    second_result = identified_result(metric_payload={"acc,none": 0.75})
    start = threading.Barrier(2)

    def publish_simultaneously(result: EvaluationResult) -> EvaluationResult:
        start.wait()
        publish_evaluation_result(path, result)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(publish_simultaneously, first_result),
            executor.submit(publish_simultaneously, second_result),
        )
        outcomes = []
        collisions = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except SMLRuntimeError as error:
                collisions.append(error)

    assert len(outcomes) == 1
    assert len(collisions) == 1
    assert "collision" in str(collisions[0])
    winner = outcomes[0]
    persisted = path.read_bytes()
    assert read_evaluation_result(path) == winner
    assert persisted == evaluation_result_bytes(winner)
    assert str(tmp_path).encode() not in persisted

    publish_evaluation_result(path, winner)
    assert path.read_bytes() == persisted
    with pytest.raises(SMLRuntimeError, match="collision"):
        publish_evaluation_result(
            path, second_result if winner == first_result else first_result
        )
    assert path.read_bytes() == persisted


def test_identical_concurrent_publication_is_idempotent_for_every_writer(
    tmp_path: Path,
) -> None:
    """Breaks if equal concurrent writers spuriously report a collision."""
    path = tmp_path / "evaluation.json"
    result = identified_result()
    start = threading.Barrier(4)

    def publish_simultaneously() -> None:
        start.wait()
        publish_evaluation_result(path, result)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = tuple(executor.submit(publish_simultaneously) for _ in range(4))
        for future in futures:
            future.result()

    assert read_evaluation_result(path) == result


@pytest.mark.parametrize("operation", ("stat", "fstat", "read"))
def test_collision_validation_normalizes_destination_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Breaks if destination-race I/O errors leak a non-collision exception."""
    path = tmp_path / "evaluation.json"
    winner = identified_result()
    publish_evaluation_result(path, winner)
    loser = identified_result(metric_payload={"acc,none": 0.75})

    def fail_collision_validation(*args: object, **kwargs: object) -> object:
        raise OSError("injected destination race")

    monkeypatch.setattr(
        evaluation_result_module.os, operation, fail_collision_validation
    )
    with pytest.raises(SMLRuntimeError) as error:
        publish_evaluation_result(path, loser)
    assert str(error.value) == f"evaluation output collision: {path}"


@pytest.mark.parametrize("already_published", (False, True))
def test_publication_unlinks_temporary_entry_before_parent_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_published: bool
) -> None:
    """Breaks if cleanup is not durably committed with publication or reuse."""
    path = tmp_path / "evaluation.json"
    result = identified_result()
    if already_published:
        publish_evaluation_result(path, result)
    events: list[str] = []
    real_unlink = os.unlink
    real_fsync = os.fsync

    def record_unlink(name: str, *, dir_fd: int | None = None) -> None:
        events.append("unlink")
        real_unlink(name, dir_fd=dir_fd)

    def record_fsync(descriptor: int) -> None:
        events.append("fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "unlink", record_unlink)
    monkeypatch.setattr(evaluation_result_module.os, "fsync", record_fsync)
    publish_evaluation_result(path, result)

    assert events.index("unlink") < len(events) - 1
    assert events[-1] == "fsync"


def test_publication_closes_parent_when_temporary_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a temporary unlink error leaks the retained parent descriptor."""
    path = tmp_path / "evaluation.json"
    events: list[tuple[str, int]] = []
    parent_descriptors: list[int] = []
    real_close = os.close

    def fail_unlink(name: str, *, dir_fd: int | None = None) -> None:
        assert dir_fd is not None
        parent_descriptors.append(dir_fd)
        events.append(("unlink", dir_fd))
        raise OSError("injected cleanup failure")

    def record_close(descriptor: int) -> None:
        events.append(("close", descriptor))
        real_close(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "unlink", fail_unlink)
    monkeypatch.setattr(evaluation_result_module.os, "close", record_close)
    with pytest.raises(OSError, match="injected cleanup failure"):
        publish_evaluation_result(path, identified_result())
    cleanup_index = (
        len(events) - 1 - events[::-1].index(("unlink", parent_descriptors[-1]))
    )
    assert ("close", parent_descriptors[-1]) in events[cleanup_index + 1 :]


def test_publication_recovers_when_another_writer_creates_missing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a missing-parent mkdir race is reported as a collision."""
    path = tmp_path / "racing-parent" / "evaluation.json"
    real_mkdir = os.mkdir

    def create_then_report_race(
        name: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        real_mkdir(name, mode, dir_fd=dir_fd)
        raise FileExistsError(name)

    monkeypatch.setattr(evaluation_result_module.os, "mkdir", create_then_report_race)
    result = identified_result()
    publish_evaluation_result(path, result)
    assert read_evaluation_result(path) == result


def test_publication_durably_creates_each_missing_parent_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a crash can lose a newly created parent directory entry."""
    path = tmp_path / "first" / "second" / "evaluation.json"
    observed_fsyncs: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        observed_fsyncs.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "fsync", record_fsync)
    publish_evaluation_result(path, identified_result())

    containing_directories = (tmp_path, tmp_path / "first")
    assert {
        (directory.stat().st_dev, directory.stat().st_ino)
        for directory in containing_directories
    } <= observed_fsyncs


def test_publication_rejects_a_symlinked_parent_without_external_write(
    tmp_path: Path,
) -> None:
    """Breaks if publication follows a parent symlink outside its namespace."""
    external = tmp_path / "external"
    external.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(external, target_is_directory=True)
    path = linked_parent / "evaluation.json"

    with pytest.raises(SMLRuntimeError) as error:
        publish_evaluation_result(path, identified_result())

    assert str(error.value) == f"evaluation output collision: {path}"
    assert not (external / "evaluation.json").exists()


def test_publication_rejects_an_equal_destination_symlink(tmp_path: Path) -> None:
    """Breaks if an externally retargetable destination counts as idempotent."""
    result = identified_result()
    backing = tmp_path / "backing.json"
    backing.write_bytes(evaluation_result_bytes(result))
    path = tmp_path / "evaluation.json"
    path.symlink_to(backing)

    with pytest.raises(SMLRuntimeError) as error:
        publish_evaluation_result(path, result)

    assert str(error.value) == f"evaluation output collision: {path}"
    assert path.is_symlink()
    assert backing.read_bytes() == evaluation_result_bytes(result)


def test_publication_rejects_a_fifo_destination_without_blocking(
    tmp_path: Path,
) -> None:
    """Breaks if collision validation follows a FIFO and can block indefinitely."""
    path = tmp_path / "evaluation.fifo"
    os.mkfifo(path)
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    process = context.Process(target=_publish_fifo_result, args=(path, outcomes))
    try:
        process.start()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("publication blocked while reading a FIFO collision")
        assert process.exitcode == 0
        assert outcomes.get(timeout=1) == f"evaluation output collision: {path}"
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        outcomes.close()
        outcomes.join_thread()
