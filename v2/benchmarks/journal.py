from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from v2.benchmarks.schema import CanonicalWorkload, JsonValue
from v2.benchmarks.workload import structured_identity

SESSION_FIELDS = {
    "kind",
    "version",
    "identity",
    "harness",
    "source",
    "canonical_workload",
    "canonical_workload_identity",
    "protocol",
    "hardware",
    "software_versions",
    "paired_representations",
    "manifest_path",
    "raw_output_path",
}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(path: Path, text: str, *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        if create_only:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict, *, create_only: bool = False) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        create_only=create_only,
    )


def read_json_object(path: Path, *, label: str) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def require_external_state_directory(
    state_directory: Path, checkouts: Sequence[Path]
) -> Path:
    state = state_directory.resolve()
    for checkout_directory in checkouts:
        checkout = checkout_directory.resolve()
        if state == checkout or state.is_relative_to(checkout):
            raise ValueError("state directory must be outside measured checkouts")
    return state


def build_session_document(
    *,
    harness_commit: str,
    harness_identity: str,
    source_commit: str,
    canonical_workload: CanonicalWorkload,
    canonical_workload_identity: str,
    protocol: dict[str, JsonValue],
    hardware: dict[str, JsonValue],
    software_versions: dict[str, str],
    paired_representations: dict[str, JsonValue],
    manifest_path: Path,
    raw_output_path: Path,
) -> dict:
    body = {
        "kind": "sml-baseline-journal-session",
        "version": 1,
        "harness": {
            "commit": harness_commit,
            "content_identity": harness_identity,
        },
        "source": {"commit": source_commit},
        "canonical_workload": canonical_workload.to_dict(),
        "canonical_workload_identity": canonical_workload_identity,
        "protocol": protocol,
        "hardware": hardware,
        "software_versions": software_versions,
        "paired_representations": paired_representations,
        "manifest_path": str(manifest_path.resolve()),
        "raw_output_path": str(raw_output_path.resolve()),
    }
    return {
        **body,
        "identity": structured_identity("sml-baseline-journal-session-v1", body),
    }


def _validate_session_document(session: dict) -> None:
    if set(session) != SESSION_FIELDS:
        raise ValueError("session does not match expected session")
    body = {key: value for key, value in session.items() if key != "identity"}
    if session["identity"] != structured_identity(
        "sml-baseline-journal-session-v1", body
    ):
        raise ValueError("session does not match expected session")


@dataclass(frozen=True, slots=True)
class BaselineJournal:
    root: Path
    session: dict

    @classmethod
    def open(cls, root: Path, expected_session: dict) -> BaselineJournal:
        _validate_session_document(expected_session)
        state = root.resolve()
        session_path = state / "session.json"
        if session_path.exists():
            session = read_json_object(session_path, label="baseline journal session")
        else:
            if state.exists() and any(state.iterdir()):
                raise ValueError("non-empty state directory has no session")
            try:
                atomic_write_json(session_path, expected_session, create_only=True)
            except FileExistsError:
                pass
            session = read_json_object(session_path, label="baseline journal session")
        _validate_session_document(session)
        if session != expected_session:
            raise ValueError("session does not match expected session")
        return cls(root=state, session=session)
