from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import v2.benchmarks.quality as quality_module
from v2.benchmarks.quality import (
    CANONICAL_STEPS,
)

from . import ROOT


def _test_publication(tmp_path):
    destinations = quality_module._EvidenceDestinations(
        manifest=tmp_path / quality_module.RECORD_MANIFEST_PATH,
        raw_output=tmp_path / quality_module.RECORD_RAW_PATH,
        report=tmp_path / quality_module.RECORD_REPORT_PATH,
    )
    owner = quality_module._publication_owner_document(
        session_identity="sha256:" + "1" * 64,
        source_commit="a" * 40,
        workload_identity="sha256:" + "2" * 64,
        destinations=destinations,
    )
    recovery = tmp_path / "results" / ".quality-recording"
    publication = quality_module._prepare_evidence_publication(
        recovery, destinations, owner
    )
    payloads = {
        "manifest": b'{"artifact":"manifest"}',
        "raw": b'{"artifact":"raw"}\n',
        "report": b'{"artifact":"report"}',
    }
    quality_module._stage_evidence_publication(publication, payloads)
    return publication, payloads, owner


def test_evidence_publication_never_replaces_a_raced_target(tmp_path, monkeypatch):
    publication, payloads, _owner = _test_publication(tmp_path)
    real_link = os.link
    raced = False

    def race_first_target(source, destination, **kwargs):
        nonlocal raced
        destination = Path(destination)
        if destination == publication.destinations.raw_output and not raced:
            raced = True
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"unrelated-raced-content")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", race_first_target)

    with pytest.raises(FileExistsError, match="conflicting"):
        quality_module._publish_staged_evidence(publication)

    assert publication.destinations.raw_output.read_bytes() == (
        b"unrelated-raced-content"
    )
    assert not publication.destinations.manifest.exists()
    assert not publication.destinations.report.exists()
    assert publication.recovery_directory.is_dir()
    assert not (publication.recovery_directory / "completed.json").exists()
    assert payloads["raw"] != publication.destinations.raw_output.read_bytes()


def test_interrupted_owned_publication_resumes_and_fsyncs_completion_last(
    tmp_path,
    monkeypatch,
):
    publication, payloads, owner = _test_publication(tmp_path)
    real_link = os.link
    link_count = 0

    def interrupt_second_target(source, destination, **kwargs):
        nonlocal link_count
        destination = Path(destination)
        if destination.name != "completed.json":
            link_count += 1
            if link_count == 2:
                raise RuntimeError("injected publication interruption")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", interrupt_second_target)
    with pytest.raises(RuntimeError, match="injected publication interruption"):
        quality_module._publish_staged_evidence(publication)

    assert publication.destinations.raw_output.read_bytes() == payloads["raw"]
    assert publication.recovery_directory.is_dir()
    assert not (publication.recovery_directory / "completed.json").exists()

    monkeypatch.setattr(os, "link", real_link)
    real_fsync_directory = quality_module._fsync_directory
    fsynced = []

    def record_fsync(path):
        fsynced.append(Path(path).resolve())
        real_fsync_directory(path)

    monkeypatch.setattr(quality_module, "_fsync_directory", record_fsync)
    resumed = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    quality_module._publish_staged_evidence(resumed)

    assert publication.destinations.raw_output.read_bytes() == payloads["raw"]
    assert publication.destinations.manifest.read_bytes() == payloads["manifest"]
    assert publication.destinations.report.read_bytes() == payloads["report"]
    assert not publication.recovery_directory.exists()
    assert publication.destinations.raw_output.parent.resolve() in fsynced
    assert publication.destinations.manifest.parent.resolve() in fsynced
    assert publication.recovery_directory.parent.resolve() in fsynced


@pytest.mark.parametrize("completed_stage_count", [1, 2, 3, 4, 5])
def test_interrupted_staging_prefix_is_identity_bound_and_safely_retryable(
    tmp_path,
    monkeypatch,
    completed_stage_count,
):
    publication, payloads, owner = _test_publication(tmp_path)
    quality_module._remove_owned_recovery(publication)
    publication = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    real_stage = quality_module._stage_named_payload
    staged = 0

    def interrupt_stage(publication, name, payload):
        nonlocal staged
        real_stage(publication, name, payload)
        staged += 1
        if staged == completed_stage_count:
            raise RuntimeError("injected staging interruption")

    monkeypatch.setattr(quality_module, "_stage_named_payload", interrupt_stage)
    with pytest.raises(RuntimeError, match="injected staging interruption"):
        quality_module._stage_evidence_publication(publication, payloads)
    assert (publication.recovery_directory / "plan.payload").is_file()

    monkeypatch.setattr(quality_module, "_stage_named_payload", real_stage)
    retried = quality_module._prepare_evidence_publication(
        publication.recovery_directory,
        publication.destinations,
        owner,
    )
    assert retried.staged is (completed_stage_count == 5)
    if not retried.staged:
        assert {path.name for path in retried.recovery_directory.iterdir()} == {
            "owner.json"
        }
        retried = quality_module._stage_evidence_publication(retried, payloads)
    quality_module._publish_staged_evidence(retried)
    assert not publication.recovery_directory.exists()


@pytest.mark.parametrize(
    "interrupted_name",
    [
        "owner.json",
        "plan.payload",
        "raw.payload",
        "report.payload",
        "manifest.payload",
        "completion.payload",
    ],
)
def test_partial_staging_write_recovers_after_process_exit(tmp_path, interrupted_name):
    publication, payloads, owner = _test_publication(tmp_path)
    quality_module._remove_owned_recovery(publication)
    arguments = {
        "recovery": str(publication.recovery_directory),
        "destinations": {
            "manifest": str(publication.destinations.manifest),
            "raw_output": str(publication.destinations.raw_output),
            "report": str(publication.destinations.report),
        },
        "owner": owner,
        "payloads": {
            name: payload.decode("utf-8") for name, payload in payloads.items()
        },
        "interrupted_name": interrupted_name,
    }
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from v2.benchmarks import quality

arguments = json.loads(sys.argv[1])
atomic_write = quality.atomic_write_text
real_fdopen = os.fdopen

class InterruptedOutput:
    def __init__(self, descriptor, *args, **kwargs):
        self.output = real_fdopen(descriptor, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.output.__exit__(*args)

    def write(self, text):
        self.output.write(text[:max(1, len(text) // 2)])
        self.output.flush()
        os.fsync(self.output.fileno())
        os._exit(73)

def interrupt_write(path, text, *, create_only=False):
    if path.name != arguments['interrupted_name']:
        return atomic_write(path, text, create_only=create_only)
    with patch('v2.benchmarks.journal.os.fdopen', InterruptedOutput):
        return atomic_write(path, text, create_only=create_only)

quality.atomic_write_text = interrupt_write
publication = quality._prepare_evidence_publication(
    Path(arguments['recovery']),
    quality._EvidenceDestinations(**{
        name: Path(value) for name, value in arguments['destinations'].items()
    }),
    arguments['owner'],
)
quality._stage_evidence_publication(
    publication,
    {name: value.encode('utf-8') for name, value in arguments['payloads'].items()},
)
""",
            json.dumps(arguments),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 73, completed.stdout + completed.stderr
    assert not (publication.recovery_directory / interrupted_name).exists()
    temporaries = list(publication.recovery_directory.glob(".*.sml-atomic-*.tmp"))
    assert len(temporaries) == 1
    assert temporaries[0].stat().st_size > 0
    assert not any(path.exists() for _name, path in publication.destinations.ordered())

    resumed = quality_module._prepare_evidence_publication(
        publication.recovery_directory, publication.destinations, owner
    )
    staged = quality_module._stage_evidence_publication(resumed, payloads)
    quality_module._publish_staged_evidence(staged)
    for name, destination in publication.destinations.ordered():
        assert destination.read_bytes() == payloads[name]
    assert not publication.recovery_directory.exists()


@pytest.mark.parametrize("entry_kind", ["unknown", "symlink"])
def test_ownerless_recovery_preserves_unrecognized_or_nonregular_entries(
    tmp_path, entry_kind
):
    publication, _payloads, owner = _test_publication(tmp_path)
    quality_module._remove_owned_recovery(publication)
    publication.recovery_directory.mkdir()
    if entry_kind == "unknown":
        entry = publication.recovery_directory / "unrelated"
        entry.write_bytes(b"unrelated")
    else:
        entry = publication.recovery_directory / (
            ".owner.json.sml-atomic-" + "1" * 32 + ".tmp"
        )
        target = tmp_path / "external-owner"
        target.write_bytes(b"unrelated")
        entry.symlink_to(target)

    with pytest.raises(ValueError, match="no durable owner"):
        quality_module._prepare_evidence_publication(
            publication.recovery_directory, publication.destinations, owner
        )
    assert entry.read_bytes() == b"unrelated"
    assert entry.is_symlink() is (entry_kind == "symlink")
    assert not (publication.recovery_directory / "owner.json").exists()


@pytest.mark.parametrize(
    "interruption",
    [
        "before_completion_link",
        "completed.json",
        "completion.payload",
        "report.payload",
        "manifest.payload",
        "raw.payload",
        "plan.payload",
        "owner.json",
    ],
)
def test_record_resumes_completed_evidence_after_publication_or_cleanup_interruption(
    tmp_path,
    monkeypatch,
    interruption,
):
    root = tmp_path
    destinations = quality_module._canonical_evidence_destinations(
        root,
        root / quality_module.RECORD_MANIFEST_PATH,
        root / quality_module.RECORD_RAW_PATH,
        root / quality_module.RECORD_REPORT_PATH,
    )
    source_commit = "a" * 40
    workload = SimpleNamespace(identity="sha256:" + "2" * 64)
    command = quality_module._recording_command_document(root, destinations)
    session_identity = quality_module._recording_session_identity(
        source_commit, workload.identity, command
    )
    owner = quality_module._publication_owner_document(
        session_identity=session_identity,
        source_commit=source_commit,
        workload_identity=workload.identity,
        destinations=destinations,
    )
    recovery = root / quality_module.RECOVERY_PATH
    publication = quality_module._prepare_evidence_publication(
        recovery, destinations, owner
    )
    payloads = {
        "manifest": b'{"artifact":"manifest"}',
        "raw": b'{"artifact":"raw"}\n',
        "report": b'{"artifact":"report"}',
    }
    publication = quality_module._stage_evidence_publication(publication, payloads)
    real_link = os.link
    real_unlink = Path.unlink

    def interrupt_completion(source, destination, **kwargs):
        if (
            interruption == "before_completion_link"
            and Path(destination).name == "completed.json"
        ):
            raise RuntimeError("interrupted before completion link")
        return real_link(source, destination, **kwargs)

    def interrupt_cleanup(path, *args, **kwargs):
        result = real_unlink(path, *args, **kwargs)
        if path == recovery / interruption:
            raise RuntimeError("interrupted after cleanup unlink")
        return result

    monkeypatch.setattr(os, "link", interrupt_completion)
    monkeypatch.setattr(Path, "unlink", interrupt_cleanup)
    with pytest.raises(RuntimeError, match="interrupted"):
        quality_module._publish_staged_evidence(publication)
    assert all(path.is_file() for _name, path in destinations.ordered())
    assert recovery.is_dir()

    monkeypatch.setattr(os, "link", real_link)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    monkeypatch.setattr(quality_module, "_root", lambda: root)
    monkeypatch.setattr(
        quality_module, "_require_clean_recording_checkout", lambda *_args: None
    )
    monkeypatch.setattr(quality_module, "_git", lambda *_args: source_commit)
    monkeypatch.setattr(
        quality_module, "build_pretraining_quality_workload", lambda _root: workload
    )
    validated_before_cleanup = []

    def validate_before_cleanup(_root, _destinations, **_kwargs):
        validated_before_cleanup.append(recovery.exists())
        for name, destination in destinations.ordered():
            assert destination.read_bytes() == payloads[name]
        return "pass"

    monkeypatch.setattr(
        quality_module, "_validate_evidence_files", validate_before_cleanup
    )
    result = quality_module._record(
        SimpleNamespace(
            manifest=destinations.manifest,
            raw_output=destinations.raw_output,
            output=destinations.report,
            steps=CANONICAL_STEPS,
        )
    )

    assert result == 0
    assert validated_before_cleanup == [True]
    for name, destination in destinations.ordered():
        assert destination.read_bytes() == payloads[name]
    assert not recovery.exists()


@pytest.mark.parametrize("entry_kind", ["regular", "symlink"])
def test_owned_recovery_cleanup_preserves_and_rejects_unknown_entries(
    tmp_path,
    entry_kind,
):
    publication, _payloads, _owner = _test_publication(tmp_path)
    unknown = publication.recovery_directory / "unrelated"
    if entry_kind == "regular":
        unknown.write_bytes(b"unrelated")
    else:
        os.symlink(publication.recovery_directory / "raw.payload", unknown)

    with pytest.raises(ValueError, match="unexpected recovery entry"):
        quality_module._remove_owned_recovery(publication)

    assert unknown.exists() or unknown.is_symlink()
    assert publication.recovery_directory.is_dir()


def test_owned_recovery_cleanup_preserves_and_rejects_a_swapped_owner_symlink(
    tmp_path,
):
    publication, _payloads, _owner = _test_publication(tmp_path)
    owner_path = publication.recovery_directory / "owner.json"
    external_owner = tmp_path / "external-owner.json"
    external_owner.write_bytes(owner_path.read_bytes())
    owner_path.unlink()
    os.symlink(external_owner, owner_path)

    with pytest.raises(ValueError, match="owner.*regular file"):
        quality_module._remove_owned_recovery(publication)

    assert owner_path.is_symlink()
    assert external_owner.is_file()
    assert publication.recovery_directory.is_dir()


def test_owned_recovery_resume_preserves_and_rejects_a_swapped_payload_symlink(
    tmp_path,
):
    publication, payloads, owner = _test_publication(tmp_path)
    raw_path = publication.recovery_directory / "raw.payload"
    external_raw = tmp_path / "external-raw.jsonl"
    external_raw.write_bytes(payloads["raw"])
    raw_path.unlink()
    os.symlink(external_raw, raw_path)

    with pytest.raises(ValueError, match="staged raw.*regular file"):
        quality_module._prepare_evidence_publication(
            publication.recovery_directory,
            publication.destinations,
            owner,
        )

    assert raw_path.is_symlink()
    assert external_raw.read_bytes() == payloads["raw"]
    assert publication.recovery_directory.is_dir()
    assert not any(path.exists() for _name, path in publication.destinations.ordered())
