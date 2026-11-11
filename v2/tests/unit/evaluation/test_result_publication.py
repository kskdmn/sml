from __future__ import annotations

import multiprocessing
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
import sml.evaluation_result as evaluation_result_module
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.evaluation_result import (
    EvaluationResult,
    evaluation_result_bytes,
    evaluation_result_identity,
    evaluation_task_identity,
    publish_evaluation_result,
    read_evaluation_result,
)

from .result_helpers import (
    identified_result,
    with_provider_metadata,
)


def _publish_fifo_result(path: Path, outcomes) -> None:
    try:
        publish_evaluation_result(path, identified_result())
    except SMLRuntimeError as error:
        outcomes.put(str(error))
    else:
        outcomes.put("accepted")


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


@pytest.mark.parametrize("metadata", ({"date": 200.0}, {"date": None}, {}))
def test_date_only_retry_returns_original_artifact(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    path = tmp_path / "evaluation.json"
    original = identified_result()
    first = with_provider_metadata(original, date=100.0)
    retry = with_provider_metadata(original, **metadata)
    assert first.identity != retry.identity
    assert publish_evaluation_result(path, first) == first
    saved_bytes = path.read_bytes()

    reused = publish_evaluation_result(path, retry)

    assert reused == first == read_evaluation_result(path)
    assert path.read_bytes() == saved_bytes == evaluation_result_bytes(first)
    assert retry.provider_result.get("date") == metadata.get("date")
    assert retry.identity == evaluation_result_identity(retry)
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize(
    "dates",
    [
        pytest.param((None,) * 4, id="identical-writers"),
        pytest.param((100.0, 200.0), id="different-execution-dates"),
    ],
)
def test_equivalent_concurrent_writers_return_the_saved_result(tmp_path, dates):
    path = tmp_path / "evaluation.json"
    original = identified_result()
    results = tuple(
        original if date is None else with_provider_metadata(original, date=date)
        for date in dates
    )
    start = threading.Barrier(len(results))

    def publish_simultaneously(result):
        start.wait()
        return publish_evaluation_result(path, result)

    with ThreadPoolExecutor(max_workers=len(results)) as executor:
        saved = tuple(executor.map(publish_simultaneously, results))

    winner = read_evaluation_result(path)
    assert winner in results
    assert all(result == winner for result in saved)
    assert path.read_bytes() == evaluation_result_bytes(winner)
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize(
    "change", ("metrics", "model", "task", "metadata", "nested_date", "scalar_type")
)
def test_date_retry_still_rejects_other_content_changes(tmp_path: Path, change: str):
    path = tmp_path / "evaluation.json"
    first = with_provider_metadata(
        identified_result(), date=100.0, metadata=True, samples=({"date": 1},)
    )
    publish_evaluation_result(path, first)
    retry = with_provider_metadata(first, date=200.0)
    if change == "metrics":
        retry = with_provider_metadata(
            identified_result(metric_payload={"acc,none": 0.75}),
            date=200.0,
            metadata=True,
            samples=({"date": 1},),
        )
    elif change == "model":
        retry = replace(
            retry,
            model=replace(retry.model, artifact_identity="sha256:" + "b" * 64),
        )
    elif change == "task":
        task = replace(retry.tasks[0], dataset_revision="revision-2")
        task = replace(task, task_identity=evaluation_task_identity(task))
        retry = replace(retry, tasks=(task,))
    elif change == "metadata":
        retry = with_provider_metadata(retry, metadata=False)
    elif change == "nested_date":
        retry = with_provider_metadata(retry, samples=({"date": 2},))
    else:
        retry = with_provider_metadata(retry, metadata=1)
    retry = replace(retry, identity=evaluation_result_identity(retry))

    with pytest.raises(SMLRuntimeError, match="collision"):
        publish_evaluation_result(path, retry)

    assert path.read_bytes() == evaluation_result_bytes(first)
    assert tuple(tmp_path.iterdir()) == (path,)


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

    with monkeypatch.context() as collision_patch:
        collision_patch.setattr(
            evaluation_result_module.os, operation, fail_collision_validation
        )
        with pytest.raises(SMLRuntimeError) as error:
            publish_evaluation_result(path, loser)
    assert str(error.value) == f"evaluation output collision: {path}"


def test_collision_reader_preserves_artifact_error_when_descriptor_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if collision descriptor cleanup replaces its primary artifact error."""
    path = tmp_path / "evaluation.json"
    path.write_bytes(b"not an evaluation result")
    parent = os.open(tmp_path, evaluation_result_module._DIRECTORY_OPEN_FLAGS)
    real_close = os.close
    real_fstat = os.fstat
    collision_closes: list[int] = []

    def fail_collision_close(descriptor: int) -> None:
        if stat.S_ISREG(real_fstat(descriptor).st_mode):
            collision_closes.append(descriptor)
            real_close(descriptor)
            raise OSError("injected collision descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "close", fail_collision_close)
    try:
        with pytest.raises(
            SMLArtifactError, match="invalid evaluation result"
        ) as error:
            evaluation_result_module._read_existing_result(parent, path.name)
    finally:
        real_close(parent)

    evidence = [
        repr(error.value.__cause__),
        repr(error.value.__context__),
        *getattr(error.value, "__notes__", ()),
    ]
    assert any(
        "injected collision descriptor close failure" in item for item in evidence
    )
    assert len(collision_closes) == 1


def test_collision_descriptor_close_only_error_normalizes_public_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a sole collision descriptor close failure escapes as raw OSError."""
    path = tmp_path / "evaluation.json"
    publish_evaluation_result(path, identified_result())
    real_close = os.close
    real_fstat = os.fstat
    collision_closes: list[int] = []

    def fail_collision_close(descriptor: int) -> None:
        if stat.S_ISREG(real_fstat(descriptor).st_mode):
            collision_closes.append(descriptor)
            real_close(descriptor)
            raise OSError("injected collision descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "close", fail_collision_close)
    with pytest.raises(SMLRuntimeError) as error:
        publish_evaluation_result(
            path, identified_result(metric_payload={"acc,none": 0.75})
        )

    assert str(error.value) == f"evaluation output collision: {path}"
    assert len(collision_closes) == 1


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


@pytest.mark.parametrize("missing_temporary", (False, True))
def test_collision_cleanup_unlinks_then_fsyncs_parent_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_temporary: bool
) -> None:
    """Breaks if collision cleanup leaves a successful/no-entry unlink non-durable."""
    path = tmp_path / "evaluation.json"
    publish_evaluation_result(path, identified_result())
    events: list[str] = []
    real_close = os.close
    real_fstat = os.fstat
    real_fsync = os.fsync
    real_unlink = os.unlink

    def record_unlink(name: str, *, dir_fd: int | None = None) -> None:
        events.append("unlink")
        real_unlink(name, dir_fd=dir_fd)
        if missing_temporary:
            raise FileNotFoundError(name)

    def record_fsync(descriptor: int) -> None:
        kind = "parent" if stat.S_ISDIR(real_fstat(descriptor).st_mode) else "file"
        events.append(f"fsync:{kind}")
        real_fsync(descriptor)

    def record_close(descriptor: int) -> None:
        kind = "parent" if stat.S_ISDIR(real_fstat(descriptor).st_mode) else "file"
        events.append(f"close:{kind}")
        real_close(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "unlink", record_unlink)
    monkeypatch.setattr(evaluation_result_module.os, "fsync", record_fsync)
    monkeypatch.setattr(evaluation_result_module.os, "close", record_close)
    with pytest.raises(SMLRuntimeError, match="collision"):
        publish_evaluation_result(
            path, identified_result(metric_payload={"acc,none": 0.75})
        )

    cleanup_index = len(events) - 1 - events[::-1].index("unlink")
    assert events[cleanup_index:] == ["unlink", "fsync:parent", "close:parent"]


def test_collision_preserves_primary_error_and_notes_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a cleanup failure replaces or hides the primary collision error."""
    path = tmp_path / "evaluation.json"
    publish_evaluation_result(path, identified_result())
    events: list[str] = []
    real_close = os.close
    real_fstat = os.fstat
    real_fsync = os.fsync

    def fail_parent_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(real_fstat(descriptor).st_mode):
            events.append("fsync:parent")
            raise OSError("injected cleanup fsync failure")
        real_fsync(descriptor)

    def record_close(descriptor: int) -> None:
        kind = "parent" if stat.S_ISDIR(real_fstat(descriptor).st_mode) else "file"
        events.append(f"close:{kind}")
        real_close(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "fsync", fail_parent_fsync)
    monkeypatch.setattr(evaluation_result_module.os, "close", record_close)
    with pytest.raises(SMLRuntimeError, match="collision") as error:
        publish_evaluation_result(
            path, identified_result(metric_payload={"acc,none": 0.75})
        )

    assert any(
        "injected cleanup fsync failure" in note for note in error.value.__notes__
    )
    cleanup_index = len(events) - 1 - events[::-1].index("fsync:parent")
    assert events[cleanup_index:] == ["fsync:parent", "close:parent"]


def test_publication_recovers_when_another_writer_creates_missing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Breaks if a raced mkdir descends before fsyncing its exact predecessor."""
    path = tmp_path / "racing-parent" / "evaluation.json"
    real_mkdir = os.mkdir
    real_fsync = os.fsync
    predecessor = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    events: list[tuple[str, int, int]] = []

    def create_then_report_race(
        name: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        assert dir_fd is not None
        metadata = os.fstat(dir_fd)
        events.append(("mkdir-race", metadata.st_dev, metadata.st_ino))
        real_mkdir(name, mode, dir_fd=dir_fd)
        raise FileExistsError(name)

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        events.append(("fsync", metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(evaluation_result_module.os, "mkdir", create_then_report_race)
    monkeypatch.setattr(evaluation_result_module.os, "fsync", record_fsync)
    result = identified_result()
    publish_evaluation_result(path, result)

    race = ("mkdir-race", *predecessor)
    race_index = events.index(race)
    assert events[race_index : race_index + 2] == [
        race,
        ("fsync", *predecessor),
    ]
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
