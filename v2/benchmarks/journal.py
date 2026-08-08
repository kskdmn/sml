from __future__ import annotations

# Journal validation reports malformed persisted content uniformly as ValueError.
# ruff: noqa: TRY004
import fcntl
import json
import math
import os
import re
import secrets
import stat
import threading
import unicodedata
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from v2.benchmarks.evidence import (
    finalize_raw_trial,
    validate_child_trial_measurement,
    validate_post_exit_observation,
    validate_raw_trial_evidence,
)
from v2.benchmarks.schema import (
    METRIC_NAMES,
    CanonicalWorkload,
    JsonValue,
    MetricName,
    RawTrial,
)
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
INITIALIZATION_MARKER = ".baseline-session-initializing"
LOCK_FILE_NAME = ".baseline-session.lock"
OUTPUT_LOCK_DIRECTORY_NAME = "sml-v2-baseline-output-locks"
STATE_DIRECTORY_NAMES = frozenset(
    {
        "accepted",
        "rejected",
        "inflight",
        "measurements",
        "post-exit",
        "preflight",
        "thermal-waits",
    }
)
STATE_FILE_NAMES = frozenset({"session.json", "completed.json"})
DECIMAL_INDEX = re.compile(r"(?:0|[1-9][0-9]*)")
IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")
ATOMIC_TEMPORARY = re.compile(
    r"^\.(?P<destination>.+)\.sml-atomic-(?P<token>[0-9a-f]{32})\.tmp$"
)
_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, tuple[int, int, int]] = {}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lstat_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"journal path contains a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"journal path component is not a directory: {path}")


def _create_durable_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("journal path must be absolute")
    current = Path(path.anchor)
    _lstat_directory(current)
    for component in path.parts[1:]:
        current /= component
        try:
            _lstat_directory(current)
        except FileNotFoundError:
            try:
                os.mkdir(current)
            except FileExistsError:
                _lstat_directory(current)
            else:
                _fsync_directory(current.parent)
                _fsync_directory(current)


def atomic_write_text(path: Path, text: str, *, create_only: bool = False) -> None:
    _create_durable_directory(path.parent)
    _fsync_directory(path.parent)
    while True:
        temporary = path.parent / (
            f".{path.name}.sml-atomic-{secrets.token_hex(16)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        break
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
            _fsync_directory(path.parent)


def atomic_write_json(path: Path, value: dict, *, create_only: bool = False) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        create_only=create_only,
    )


def _json_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _json_bytes(value: dict) -> bytes:
    return _json_text(value).encode("utf-8")


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


@contextmanager
def _exclusive_file_lock(
    lock_path: Path,
    *,
    conflict_message: str,
    invalid_message: str,
    required_owner: int | None = None,
):
    _create_durable_directory(lock_path.parent)
    owner = threading.get_ident()
    descriptor: int | None = None
    with _PROCESS_LOCK_GUARD:
        held = _PROCESS_LOCKS.get(lock_path)
        if held is not None:
            held_descriptor, held_owner, depth = held
            if held_owner != owner:
                raise RuntimeError(conflict_message)
            _PROCESS_LOCKS[lock_path] = (held_descriptor, held_owner, depth + 1)
        else:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(invalid_message)
                if required_owner is not None and metadata.st_uid != required_owner:
                    raise ValueError("baseline lock must be owned by the current user")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(conflict_message) from error
                os.fsync(descriptor)
                _fsync_directory(lock_path.parent)
            except BaseException:
                os.close(descriptor)
                raise
            _PROCESS_LOCKS[lock_path] = (descriptor, owner, 1)
    try:
        yield
    finally:
        descriptor_to_close: int | None = None
        with _PROCESS_LOCK_GUARD:
            held_descriptor, held_owner, depth = _PROCESS_LOCKS[lock_path]
            if held_owner != owner:
                raise RuntimeError("baseline lock owner changed")
            if depth == 1:
                del _PROCESS_LOCKS[lock_path]
                descriptor_to_close = held_descriptor
            else:
                _PROCESS_LOCKS[lock_path] = (held_descriptor, held_owner, depth - 1)
        if descriptor_to_close is not None:
            try:
                fcntl.flock(descriptor_to_close, fcntl.LOCK_UN)
            finally:
                os.close(descriptor_to_close)


@contextmanager
def baseline_session_lock(root: Path):
    state = root.resolve()
    with _exclusive_file_lock(
        state / LOCK_FILE_NAME,
        conflict_message="baseline session is already locked",
        invalid_message="baseline session lock must be a regular file",
    ):
        yield state


def _baseline_output_lock_root() -> Path:
    return Path("/tmp").resolve() / f"{OUTPUT_LOCK_DIRECTORY_NAME}-{os.getuid()}"


def _prepare_private_lock_directory(path: Path) -> None:
    _create_durable_directory(path.parent)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    else:
        _fsync_directory(path.parent)
        _fsync_directory(path)
    _lstat_directory(path)
    metadata = os.lstat(path)
    if metadata.st_uid != os.getuid():
        raise ValueError("baseline final output lock directory has the wrong owner")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("baseline final output lock directory permissions are unsafe")


def _canonical_output_lock_key(path: Path) -> str:
    return unicodedata.normalize("NFC", str(path.resolve())).casefold()


@contextmanager
def baseline_output_lock(manifest_path: Path, raw_output_path: Path):
    destinations = (manifest_path.resolve(), raw_output_path.resolve())
    if destinations[0] == destinations[1]:
        raise ValueError("baseline final output paths must be distinct")
    lock_root = _baseline_output_lock_root()
    _prepare_private_lock_directory(lock_root)
    lock_paths = set()
    for destination in destinations:
        identity = structured_identity(
            "sml-baseline-final-output-destination-lock-v1",
            _canonical_output_lock_key(destination),
        ).removeprefix("sha256:")
        lock_paths.add(lock_root / f"{identity}.lock")
    with ExitStack() as stack:
        for lock_path in sorted(lock_paths, key=str):
            stack.enter_context(
                _exclusive_file_lock(
                    lock_path,
                    conflict_message="baseline final outputs are already locked",
                    invalid_message=(
                        "baseline final output lock must be a regular file"
                    ),
                    required_owner=os.getuid(),
                )
            )
        yield


def _atomic_temporary_destination(path: Path) -> Path | None:
    match = ATOMIC_TEMPORARY.fullmatch(path.name)
    if match is None:
        return None
    return path.parent / match.group("destination")


def _is_journal_destination(root: Path, destination: Path) -> bool:
    try:
        parts = destination.relative_to(root).parts
    except ValueError:
        return False
    if len(parts) == 1:
        return parts[0] in {INITIALIZATION_MARKER, "session.json", "completed.json"}
    if len(parts) == 3 and parts[0] == "accepted":
        return (
            parts[1] in METRIC_NAMES
            and parts[2].endswith(".json")
            and DECIMAL_INDEX.fullmatch(parts[2][:-5]) is not None
        )
    if len(parts) == 4 and parts[0] in {
        "inflight",
        "measurements",
        "post-exit",
        "rejected",
        "preflight",
    }:
        return (
            parts[1] in METRIC_NAMES
            and DECIMAL_INDEX.fullmatch(parts[2]) is not None
            and parts[3].endswith(".json")
            and DECIMAL_INDEX.fullmatch(parts[3][:-5]) is not None
        )
    if len(parts) == 5 and parts[0] == "thermal-waits":
        final_name = parts[4]
        return (
            parts[1] in METRIC_NAMES
            and DECIMAL_INDEX.fullmatch(parts[2]) is not None
            and DECIMAL_INDEX.fullmatch(parts[3]) is not None
            and (
                final_name in {"trigger.json", "summary.json"}
                or (
                    final_name.endswith(".json")
                    and DECIMAL_INDEX.fullmatch(final_name[:-5]) is not None
                )
            )
        )
    return False


def _unlink_regular_orphan(path: Path, modified: set[Path]) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        return
    path.unlink()
    modified.add(path.parent)


def cleanup_orphaned_atomic_temporaries(destinations: Sequence[Path]) -> None:
    modified: set[Path] = set()
    for destination in destinations:
        expected = destination.parent / (f".{destination.name}.sml-atomic-")
        if not destination.parent.exists():
            continue
        _require_directory(destination.parent, label="atomic output parent")
        for entry in destination.parent.iterdir():
            recovered = _atomic_temporary_destination(entry)
            if recovered == destination and entry.name.startswith(expected.name):
                _unlink_regular_orphan(entry, modified)
    for directory in modified:
        _fsync_directory(directory)


def cleanup_orphaned_journal_temporaries(root: Path) -> None:
    state = root.resolve()
    if not state.exists():
        return
    _require_directory(state, label="state directory")
    modified: set[Path] = set()
    for directory_name, _subdirectories, filenames in os.walk(state, followlinks=False):
        directory = Path(directory_name)
        for filename in filenames:
            path = directory / filename
            destination = _atomic_temporary_destination(path)
            if destination is not None and _is_journal_destination(state, destination):
                _unlink_regular_orphan(path, modified)
    marker = state / INITIALIZATION_MARKER
    _unlink_regular_orphan(marker, modified)
    for directory in modified:
        _fsync_directory(directory)


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
    if type(session["version"]) is not int or session["version"] != 1:
        raise ValueError("session does not match expected session")
    body = {key: value for key, value in session.items() if key != "identity"}
    if session["identity"] != structured_identity(
        "sml-baseline-journal-session-v1", body
    ):
        raise ValueError("session does not match expected session")


def _state_entries(state: Path) -> tuple[Path, ...]:
    if not state.exists():
        return ()
    if state.is_symlink() or not state.is_dir():
        raise ValueError("state directory must be a directory")
    return tuple(state.iterdir())


def _require_non_negative_index(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be non-negative")


def _parse_decimal_index(name: str, *, label: str) -> int:
    if DECIMAL_INDEX.fullmatch(name) is None:
        raise ValueError(f"{label} must use a canonical decimal index")
    return int(name)


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a directory")


def _require_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a file")


def _regular_file_identity(path: Path, *, label: str) -> tuple[int, int]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return metadata.st_dev, metadata.st_ino


def _require_object_fields(raw: dict, fields: set[str], *, label: str) -> None:
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"{label} has an invalid field set")


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_observation(
    observation: dict, *, label: str, require_schema_version: bool
) -> None:
    fields = {
        "observed_at_utc",
        "hardware",
        "environment_status",
        "software_versions",
    }
    if require_schema_version:
        fields |= {"schema_version", "elapsed_seconds"}
    _require_object_fields(observation, fields, label=label)
    _require_nonempty_string(observation["observed_at_utc"], label="observed_at_utc")
    for name in ("hardware", "environment_status", "software_versions"):
        if not isinstance(observation[name], dict):
            raise ValueError(f"{name} must be an object")
    if not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in observation["software_versions"].items()
    ):
        raise ValueError("software_versions must be a string mapping")
    if require_schema_version:
        if (
            type(observation["schema_version"]) is not int
            or observation["schema_version"] != 1
        ):
            raise ValueError("unsupported thermal sample schema version")
        elapsed = observation["elapsed_seconds"]
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative")


def _validate_identity_document(
    document: dict,
    *,
    kind: str,
    identity_domain: str,
    label: str,
    expected_version: int = 1,
) -> dict:
    if (
        document.get("kind") != kind
        or type(document.get("version")) is not int
        or document["version"] != expected_version
    ):
        raise ValueError(f"{label} has an invalid kind or version")
    identity = document.get("identity")
    body = {key: value for key, value in document.items() if key != "identity"}
    if identity != structured_identity(identity_domain, body):
        raise ValueError(f"{label} has an invalid identity")
    return body


def _validate_rejected_document(
    document: dict, *, slot: BaselineSlot, journal_attempt_index: int
) -> tuple[dict, RawTrial | None]:
    _require_object_fields(
        document,
        {
            "kind",
            "version",
            "journal_attempt_index",
            "reason",
            "child_measurement_identity",
            "post_exit_observation_identity",
            "trial",
            "identity",
        },
        label="rejected trial",
    )
    body = _validate_identity_document(
        document,
        kind="sml-baseline-rejected-trial",
        identity_domain="sml-baseline-rejected-trial-v2",
        label="rejected trial",
        expected_version=2,
    )
    _require_non_negative_index(
        body["journal_attempt_index"], label="journal attempt index"
    )
    if body["journal_attempt_index"] != journal_attempt_index:
        raise ValueError("rejected trial attempt index does not match its filename")
    _require_nonempty_string(body["reason"], label="rejected trial reason")
    if (
        not isinstance(body["child_measurement_identity"], str)
        or IDENTITY.fullmatch(body["child_measurement_identity"]) is None
    ):
        raise ValueError("rejected trial child measurement identity is invalid")
    post_exit_identity = body["post_exit_observation_identity"]
    if post_exit_identity is not None and (
        not isinstance(post_exit_identity, str)
        or IDENTITY.fullmatch(post_exit_identity) is None
    ):
        raise ValueError("rejected trial post-exit observation identity is invalid")
    raw_trial = body["trial"]
    if raw_trial is None:
        return body, None
    if not isinstance(raw_trial, dict):
        raise ValueError("rejected trial must contain a raw trial object or null")
    return body, _parse_trial_for_slot(raw_trial, slot=slot, label="rejected trial")


def _validate_preflight_document(
    document: dict, *, slot: BaselineSlot | None = None
) -> dict:
    _require_object_fields(
        document,
        {
            "kind",
            "version",
            "metric",
            "pair_index",
            "preflight_index",
            "observed_at_utc",
            "hardware",
            "environment_status",
            "software_versions",
            "identity",
        },
        label="preflight",
    )
    body = _validate_identity_document(
        document,
        kind="sml-baseline-preflight",
        identity_domain="sml-baseline-preflight-v1",
        label="preflight",
    )
    _require_non_negative_index(body["preflight_index"], label="preflight index")
    document_slot = BaselineSlot(body["metric"], body["pair_index"])
    if slot is not None and document_slot != slot:
        raise ValueError("preflight metric or pair index does not match its slot")
    _require_observation(
        {
            key: body[key]
            for key in (
                "observed_at_utc",
                "hardware",
                "environment_status",
                "software_versions",
            )
        },
        label="preflight",
        require_schema_version=False,
    )
    return body


def _validate_trigger_document(document: dict) -> dict:
    body = _validate_identity_document(
        document,
        kind="sml-baseline-thermal-recovery-trigger",
        identity_domain="sml-baseline-thermal-recovery-trigger-v1",
        label="thermal recovery trigger",
    )
    source = body.get("source")
    if source == "preflight":
        _require_object_fields(
            body,
            {"kind", "version", "source", "preflight"},
            label="thermal recovery trigger",
        )
        if not isinstance(body["preflight"], dict):
            raise ValueError("thermal recovery trigger preflight must be an object")
        _validate_preflight_document(body["preflight"])
    elif source == "rejected-trial":
        _require_object_fields(
            body,
            {"kind", "version", "source", "rejected_trial_identity"},
            label="thermal recovery trigger",
        )
        if (
            not isinstance(body["rejected_trial_identity"], str)
            or IDENTITY.fullmatch(body["rejected_trial_identity"]) is None
        ):
            raise ValueError("thermal recovery trigger rejected identity is invalid")
    else:
        raise ValueError("thermal recovery trigger has an unsupported source")
    return body


def _validate_sample_document(document: dict, *, sample_index: int) -> dict:
    _require_object_fields(
        document,
        {
            "kind",
            "version",
            "sample_index",
            "schema_version",
            "observed_at_utc",
            "elapsed_seconds",
            "hardware",
            "environment_status",
            "software_versions",
            "identity",
        },
        label="thermal sample",
    )
    body = _validate_identity_document(
        document,
        kind="sml-baseline-thermal-sample",
        identity_domain="sml-baseline-thermal-sample-v1",
        label="thermal sample",
    )
    _require_non_negative_index(body["sample_index"], label="thermal sample index")
    if body["sample_index"] != sample_index:
        raise ValueError("thermal sample index does not match its filename")
    _require_observation(
        {
            key: body[key]
            for key in (
                "schema_version",
                "observed_at_utc",
                "elapsed_seconds",
                "hardware",
                "environment_status",
                "software_versions",
            )
        },
        label="thermal sample",
        require_schema_version=True,
    )
    return body


def _validate_summary_document(document: dict, *, sample_count: int) -> dict:
    _require_object_fields(
        document,
        {
            "kind",
            "version",
            "outcome",
            "duration_seconds",
            "sample_count",
            "identity",
        },
        label="thermal recovery summary",
    )
    body = _validate_identity_document(
        document,
        kind="sml-baseline-thermal-recovery-summary",
        identity_domain="sml-baseline-thermal-recovery-summary-v1",
        label="thermal recovery summary",
    )
    if body["outcome"] not in ("nominal-window", "timeout"):
        raise ValueError("thermal recovery summary outcome is invalid")
    duration = body["duration_seconds"]
    if (
        type(duration) not in (int, float)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("thermal recovery summary duration is invalid")
    _require_non_negative_index(body["sample_count"], label="thermal sample count")
    if body["sample_count"] != sample_count:
        raise ValueError("thermal recovery summary sample count does not match")
    return body


def _validate_completion_document(document: dict, *, session_identity: str) -> None:
    _require_object_fields(
        document,
        {
            "kind",
            "version",
            "session_identity",
            "baseline_identity",
            "manifest_path",
            "raw_output_path",
            "raw_trial_identities",
            "identity",
        },
        label="baseline journal completion",
    )
    body = _validate_identity_document(
        document,
        kind="sml-baseline-journal-completion",
        identity_domain="sml-baseline-journal-completion-v1",
        label="baseline journal completion",
    )
    if body["session_identity"] != session_identity:
        raise ValueError("baseline journal completion does not match the session")
    for name in ("baseline_identity", "session_identity"):
        if not isinstance(body[name], str) or IDENTITY.fullmatch(body[name]) is None:
            raise ValueError(f"baseline journal completion {name} is invalid")
    for name in ("manifest_path", "raw_output_path"):
        _require_nonempty_string(
            body[name], label=f"baseline journal completion {name}"
        )
    identities = body["raw_trial_identities"]
    if not isinstance(identities, list) or not all(
        isinstance(identity, str) and IDENTITY.fullmatch(identity) is not None
        for identity in identities
    ):
        raise ValueError("baseline journal completion raw trial identities are invalid")


@dataclass(frozen=True, slots=True, order=True)
class BaselineSlot:
    metric: MetricName
    pair_index: int

    def __post_init__(self) -> None:
        if self.metric not in METRIC_NAMES:
            raise ValueError(f"unsupported baseline slot metric: {self.metric!r}")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("baseline slot pair index must be non-negative")


@dataclass(frozen=True, slots=True)
class JournalAttempt:
    slot: BaselineSlot
    journal_attempt_index: int
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.slot, BaselineSlot):
            raise ValueError("journal attempt slot must be a baseline slot")
        _require_non_negative_index(
            self.journal_attempt_index, label="journal attempt index"
        )
        if not isinstance(self.path, Path):
            raise ValueError("journal attempt path must be a path")


@dataclass(frozen=True, slots=True)
class JournalAttemptEvidence:
    attempt: JournalAttempt
    measurement: dict
    post_exit: dict | None
    trial: RawTrial | None


def _parse_trial_for_slot(raw: dict, *, slot: BaselineSlot, label: str) -> RawTrial:
    trial = RawTrial.from_dict(raw)
    validate_raw_trial_evidence(trial)
    if type(raw["pair_index"]) is not int:
        raise ValueError("raw trial pair index must be an integer")
    if type(raw["attempt_index"]) is not int or trial.attempt_index != 0:
        raise ValueError("baseline raw trial attempt_index must be zero")
    if trial.metric != slot.metric or trial.pair_index != slot.pair_index:
        raise ValueError(f"{label} metric or pair index does not match its slot")
    return trial


def _validate_root_layout(state: Path) -> tuple[Path, ...]:
    entries = _state_entries(state)
    for entry in entries:
        if entry.name == INITIALIZATION_MARKER:
            raise ValueError("state directory contains unexpected content")
        if entry.name in STATE_DIRECTORY_NAMES:
            _require_directory(entry, label=f"state {entry.name}")
        elif entry.name in STATE_FILE_NAMES or entry.name == LOCK_FILE_NAME:
            _require_file(entry, label=f"state {entry.name}")
        else:
            raise ValueError("state directory contains unexpected content")
    return entries


def _entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _write_immutable_json(path: Path, document: dict, *, label: str) -> None:
    expected = _json_bytes(document)
    try:
        atomic_write_json(path, document, create_only=True)
    except FileExistsError:
        try:
            _require_file(path, label=label)
            existing = path.read_bytes()
        except OSError as error:
            raise ValueError(f"{label} is not readable") from error
        if existing != expected:
            raise ValueError(f"{label} is immutable")


@dataclass(frozen=True, slots=True)
class BaselineJournal:
    root: Path
    session: dict

    @classmethod
    def open(cls, root: Path, expected_session: dict) -> BaselineJournal:
        _validate_session_document(expected_session)
        state = root.resolve()
        session_path = state / "session.json"
        marker_path = state / INITIALIZATION_MARKER
        entries = _state_entries(state)
        for entry in entries:
            if entry.name == LOCK_FILE_NAME:
                _require_file(entry, label="baseline session lock")
        resumed = session_path in entries
        if resumed:
            entries = _validate_root_layout(state)
            session = read_json_object(session_path, label="baseline journal session")
        else:
            non_lock_entries = tuple(
                entry for entry in entries if entry.name != LOCK_FILE_NAME
            )
            if non_lock_entries:
                if marker_path in non_lock_entries:
                    raise ValueError("state directory contains unexpected content")
                raise ValueError("non-empty state directory has no session")
            try:
                atomic_write_text(marker_path, "", create_only=True)
            except FileExistsError as error:
                raise ValueError(
                    "state directory contains unexpected content"
                ) from error
            lock_path = state / LOCK_FILE_NAME
            initialization_snapshot = set(_state_entries(state))
            initialization_entries = {marker_path}
            if lock_path in initialization_snapshot:
                initialization_entries.add(lock_path)
            if initialization_snapshot != initialization_entries:
                raise ValueError("state directory contains unexpected content")
            try:
                atomic_write_json(session_path, expected_session, create_only=True)
            except FileExistsError as error:
                raise ValueError(
                    "state directory contains unexpected content"
                ) from error
            initialized_snapshot = set(_state_entries(state))
            initialized_entries = {marker_path, session_path}
            if lock_path in initialized_snapshot:
                initialized_entries.add(lock_path)
            if initialized_snapshot != initialized_entries:
                raise ValueError("state directory contains unexpected content")
            marker_path.unlink()
            _fsync_directory(state)
            session = read_json_object(session_path, label="baseline journal session")
        _validate_session_document(session)
        if session != expected_session:
            raise ValueError("session does not match expected session")
        completed_path = state / "completed.json"
        if resumed and completed_path in _validate_root_layout(state):
            _validate_completion_document(
                read_json_object(completed_path, label="baseline journal completion"),
                session_identity=session["identity"],
            )
        return cls(root=state, session=session)

    @staticmethod
    def expected_slots(
        metrics: Sequence[MetricName], pairs: int
    ) -> tuple[BaselineSlot, ...]:
        _require_non_negative_index(pairs, label="baseline slot pair count")
        slots: list[BaselineSlot] = []
        seen_metrics: set[MetricName] = set()
        for metric in metrics:
            if metric in seen_metrics:
                raise ValueError("baseline slot metrics must not contain duplicates")
            seen_metrics.add(metric)
            slots.extend(
                BaselineSlot(metric, pair_index) for pair_index in range(pairs)
            )
        return tuple(slots)

    @property
    def completed_path(self) -> Path:
        return self.root / "completed.json"

    def accepted_path(self, slot: BaselineSlot) -> Path:
        self._require_slot(slot)
        return self.root / "accepted" / slot.metric / f"{slot.pair_index}.json"

    def inflight_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(
            journal_attempt_index, label="journal attempt index"
        )
        return (
            self.root
            / "inflight"
            / slot.metric
            / str(slot.pair_index)
            / f"{journal_attempt_index}.json"
        )

    def measurement_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(
            journal_attempt_index, label="journal attempt index"
        )
        return (
            self.root
            / "measurements"
            / slot.metric
            / str(slot.pair_index)
            / f"{journal_attempt_index}.json"
        )

    def post_exit_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(
            journal_attempt_index, label="journal attempt index"
        )
        return (
            self.root
            / "post-exit"
            / slot.metric
            / str(slot.pair_index)
            / f"{journal_attempt_index}.json"
        )

    def rejected_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(
            journal_attempt_index, label="journal attempt index"
        )
        return (
            self.root
            / "rejected"
            / slot.metric
            / str(slot.pair_index)
            / f"{journal_attempt_index}.json"
        )

    def preflight_path(self, slot: BaselineSlot, preflight_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(preflight_index, label="preflight index")
        return (
            self.root
            / "preflight"
            / slot.metric
            / str(slot.pair_index)
            / f"{preflight_index}.json"
        )

    def _thermal_wait_path(self, slot: BaselineSlot, recovery_index: int) -> Path:
        self._require_slot(slot)
        _require_non_negative_index(recovery_index, label="thermal recovery index")
        return (
            self.root
            / "thermal-waits"
            / slot.metric
            / str(slot.pair_index)
            / str(recovery_index)
        )

    @staticmethod
    def _require_slot(slot: BaselineSlot) -> None:
        if not isinstance(slot, BaselineSlot):
            raise ValueError("journal slot must be a baseline slot")

    @staticmethod
    def _expected_slot_set(
        expected_slots: Sequence[BaselineSlot],
    ) -> set[BaselineSlot]:
        slots = tuple(expected_slots)
        if not all(isinstance(slot, BaselineSlot) for slot in slots):
            raise ValueError("expected slots must be baseline slots")
        if len(set(slots)) != len(slots):
            raise ValueError("expected slots must not contain duplicates")
        return set(slots)

    def _category_metric_directories(
        self, category: str
    ) -> tuple[tuple[MetricName, Path], ...]:
        path = self.root / category
        if not _entry_exists(path):
            return ()
        _require_directory(path, label=f"{category} state")
        directories: list[tuple[MetricName, Path]] = []
        for metric_path in path.iterdir():
            if metric_path.name not in METRIC_NAMES:
                raise ValueError(f"{category} contains an unsupported metric directory")
            _require_directory(metric_path, label=f"{category} metric state")
            directories.append((metric_path.name, metric_path))
        return tuple(directories)

    def _accepted_records(self) -> tuple[tuple[BaselineSlot, Path], ...]:
        records: list[tuple[BaselineSlot, Path]] = []
        for metric, metric_path in self._category_metric_directories("accepted"):
            for path in metric_path.iterdir():
                _require_file(path, label="accepted slot")
                if path.suffix != ".json":
                    raise ValueError("accepted slot filename must end in .json")
                pair_index = _parse_decimal_index(
                    path.stem, label="accepted slot filename"
                )
                records.append((BaselineSlot(metric, pair_index), path))
        return tuple(records)

    def _attempt_records(
        self, category: str
    ) -> tuple[tuple[JournalAttempt, Path], ...]:
        path_for_category = {
            "inflight": self.inflight_path,
            "measurements": self.measurement_path,
            "post-exit": self.post_exit_path,
            "rejected": self.rejected_path,
        }
        try:
            canonical_path = path_for_category[category]
        except KeyError as error:
            raise ValueError(f"unsupported attempt category: {category}") from error
        records: list[tuple[JournalAttempt, Path]] = []
        for metric, metric_path in self._category_metric_directories(category):
            for pair_path in metric_path.iterdir():
                _require_directory(pair_path, label=f"{category} slot state")
                pair_index = _parse_decimal_index(
                    pair_path.name, label=f"{category} pair index"
                )
                slot = BaselineSlot(metric, pair_index)
                for path in pair_path.iterdir():
                    _require_file(path, label=f"{category} attempt")
                    if path.suffix != ".json":
                        raise ValueError(
                            f"{category} attempt filename must end in .json"
                        )
                    attempt_index = _parse_decimal_index(
                        path.stem, label=f"{category} attempt filename"
                    )
                    attempt_path = canonical_path(slot, attempt_index)
                    if path != attempt_path:
                        raise ValueError(f"{category} attempt has a non-canonical path")
                    records.append(
                        (
                            JournalAttempt(
                                slot,
                                attempt_index,
                                self.inflight_path(slot, attempt_index),
                            ),
                            path,
                        )
                    )
        return tuple(records)

    def _validate_attempt_history(
        self,
    ) -> tuple[dict[BaselineSlot, tuple[int, ...]], tuple[JournalAttemptEvidence, ...]]:
        measurements: dict[tuple[BaselineSlot, int], dict] = {}
        attempts: dict[tuple[BaselineSlot, int], JournalAttempt] = {}
        for attempt, path in self._attempt_records("measurements"):
            document = read_json_object(path, label="child measurement")
            measurement = validate_child_trial_measurement(document)
            trial_payload = measurement["trial"]
            if measurement["session_identity"] != self.session["identity"]:
                raise ValueError("child measurement does not match the journal session")
            if (
                trial_payload["metric"] != attempt.slot.metric
                or type(trial_payload["pair_index"]) is not int
                or trial_payload["pair_index"] != attempt.slot.pair_index
            ):
                raise ValueError(
                    "child measurement metric or pair index does not match its slot"
                )
            if (
                type(trial_payload["attempt_index"]) is not int
                or trial_payload["attempt_index"] != 0
            ):
                raise ValueError(
                    "baseline child measurement attempt_index must be zero"
                )
            if measurement["journal_attempt_index"] != attempt.journal_attempt_index:
                raise ValueError(
                    "child measurement attempt index does not match its filename"
                )
            key = (attempt.slot, attempt.journal_attempt_index)
            measurements[key] = measurement
            attempts[key] = attempt

        post_exit: dict[tuple[BaselineSlot, int], dict] = {}
        for attempt, path in self._attempt_records("post-exit"):
            key = (attempt.slot, attempt.journal_attempt_index)
            measurement = measurements.get(key)
            if measurement is None:
                raise ValueError("post-exit evidence has no measurement")
            document = read_json_object(path, label="post-exit observation")
            post_exit[key] = validate_post_exit_observation(
                document, measurement=measurement
            )
            attempts[key] = attempt

        inflight: dict[tuple[BaselineSlot, int], tuple[dict, RawTrial]] = {}
        for attempt, path in self._attempt_records("inflight"):
            key = (attempt.slot, attempt.journal_attempt_index)
            if key not in post_exit:
                raise ValueError("inflight trial has no post-exit evidence")
            raw = read_json_object(path, label="inflight trial")
            trial = _parse_trial_for_slot(
                raw, slot=attempt.slot, label="inflight trial"
            )
            expected_trial = finalize_raw_trial(measurements[key], post_exit[key])
            if raw != expected_trial.to_dict() or trial != expected_trial:
                raise ValueError("inflight trial does not match its staged evidence")
            inflight[key] = (raw, trial)
            attempts[key] = attempt

        rejected: dict[
            tuple[BaselineSlot, int], tuple[dict, dict, RawTrial | None]
        ] = {}
        for attempt, path in self._attempt_records("rejected"):
            document = read_json_object(path, label="rejected trial")
            body, trial = _validate_rejected_document(
                document,
                slot=attempt.slot,
                journal_attempt_index=attempt.journal_attempt_index,
            )
            key = (attempt.slot, attempt.journal_attempt_index)
            measurement = measurements.get(key)
            if measurement is None:
                raise ValueError("rejected trial has no child measurement")
            if body["child_measurement_identity"] != measurement["identity"]:
                raise ValueError("rejected trial does not match its child measurement")
            parent = post_exit.get(key)
            expected_parent_identity = None if parent is None else parent["identity"]
            if body["post_exit_observation_identity"] != expected_parent_identity:
                raise ValueError(
                    "rejected trial does not match its post-exit observation"
                )
            if trial is None:
                if body["reason"] != "missing-immediate-post-exit-evidence":
                    raise ValueError("unfinalized rejection has an invalid reason")
                if parent is not None or key in inflight:
                    raise ValueError(
                        "unfinalized rejection contains finalized stage evidence"
                    )
            else:
                if parent is None:
                    raise ValueError("rejected trial has no post-exit evidence")
                expected_trial = finalize_raw_trial(measurement, parent)
                if body["trial"] != expected_trial.to_dict() or trial != expected_trial:
                    raise ValueError(
                        "rejected trial does not match its staged evidence"
                    )
            rejected[key] = (document, body, trial)
            attempts[key] = attempt

        accepted: dict[tuple[BaselineSlot, int], tuple[Path, dict, RawTrial]] = {}
        for slot, path in self._accepted_records():
            raw = read_json_object(path, label="accepted trial")
            trial = _parse_trial_for_slot(raw, slot=slot, label="accepted trial")
            key = (slot, trial.journal_attempt_index)
            if key in accepted:
                raise ValueError("accepted attempt is duplicated")
            measurement = measurements.get(key)
            parent = post_exit.get(key)
            if measurement is None or parent is None:
                raise ValueError("accepted trial has incomplete staged evidence")
            expected_trial = finalize_raw_trial(measurement, parent)
            if raw != expected_trial.to_dict() or trial != expected_trial:
                raise ValueError("accepted trial does not match its staged evidence")
            accepted[key] = (
                path,
                raw,
                trial,
            )
            attempts[key] = JournalAttempt(
                slot,
                trial.journal_attempt_index,
                self.inflight_path(slot, trial.journal_attempt_index),
            )

        if accepted.keys() & rejected.keys():
            raise ValueError("journal attempt has both accepted and rejected outcomes")

        rejected_splits: list[JournalAttempt] = []
        for key in rejected.keys() & inflight.keys():
            rejected_document, _body, rejected_trial = rejected[key]
            raw, inflight_trial = inflight[key]
            if (
                rejected_trial is None
                or raw != rejected_document["trial"]
                or inflight_trial != rejected_trial
            ):
                raise ValueError(
                    "inflight trial does not match its rejected transition"
                )
            rejected_splits.append(attempts[key])

        all_keys = (
            set(measurements)
            | set(post_exit)
            | set(inflight)
            | set(rejected)
            | set(accepted)
        )
        histories: dict[BaselineSlot, list[int]] = {}
        for slot, index in all_keys:
            histories.setdefault(slot, []).append(index)
        for slot, indices in histories.items():
            if sorted(indices) != list(range(len(indices))):
                raise ValueError(
                    f"journal attempt indices have a gap for {slot.metric}"
                )

        accepted_splits: list[JournalAttempt] = []
        for key, (accepted_path, accepted_raw, accepted_trial) in accepted.items():
            slot, _index = key
            candidates = [
                (candidate_key, record)
                for candidate_key, record in inflight.items()
                if candidate_key not in rejected and candidate_key[0] == slot
            ]
            if not candidates:
                continue
            if len(candidates) != 1 or candidates[0][0] != key:
                raise ValueError("multiple accepted transition candidates")
            _candidate_key, (raw, inflight_trial) = candidates[0]
            if raw != accepted_raw or inflight_trial != accepted_trial:
                raise ValueError(
                    "inflight trial does not match its accepted transition"
                )
            attempt = attempts[key]
            if _regular_file_identity(
                accepted_path, label="accepted slot"
            ) != _regular_file_identity(attempt.path, label="inflight trial"):
                raise ValueError(
                    "inflight trial does not match its accepted transition"
                )
            accepted_splits.append(attempt)

        for attempt in (*rejected_splits, *accepted_splits):
            attempt.path.unlink()
            _fsync_directory(attempt.path.parent)
            del inflight[(attempt.slot, attempt.journal_attempt_index)]

        pending = tuple(
            JournalAttemptEvidence(
                attempt=attempts[key],
                measurement=measurements[key],
                post_exit=post_exit.get(key),
                trial=None if key not in inflight else inflight[key][1],
            )
            for key in all_keys - set(accepted) - set(rejected)
        )
        return (
            {slot: tuple(sorted(indices)) for slot, indices in histories.items()},
            pending,
        )

    def _preflight_records(
        self,
    ) -> tuple[tuple[BaselineSlot, int, Path, dict], ...]:
        records: list[tuple[BaselineSlot, int, Path, dict]] = []
        for metric, metric_path in self._category_metric_directories("preflight"):
            for pair_path in metric_path.iterdir():
                _require_directory(pair_path, label="preflight slot state")
                slot = BaselineSlot(
                    metric,
                    _parse_decimal_index(pair_path.name, label="preflight pair index"),
                )
                for path in pair_path.iterdir():
                    _require_file(path, label="preflight")
                    if path.suffix != ".json":
                        raise ValueError("preflight filename must end in .json")
                    index = _parse_decimal_index(path.stem, label="preflight filename")
                    if path != self.preflight_path(slot, index):
                        raise ValueError("preflight has a non-canonical path")
                    document = read_json_object(path, label="preflight")
                    body = _validate_preflight_document(document, slot=slot)
                    if body["preflight_index"] != index:
                        raise ValueError("preflight index does not match its filename")
                    records.append((slot, index, path, document))
        return tuple(records)

    def _validate_preflight_history(
        self,
    ) -> dict[BaselineSlot, tuple[tuple[int, dict], ...]]:
        histories: dict[BaselineSlot, list[tuple[int, dict]]] = {}
        for slot, index, _path, document in self._preflight_records():
            histories.setdefault(slot, []).append((index, document))
        for slot, records in histories.items():
            if sorted(index for index, _document in records) != list(
                range(len(records))
            ):
                raise ValueError(f"preflight indices have a gap for {slot.metric}")
        return {
            slot: tuple(sorted(records, key=lambda record: record[0]))
            for slot, records in histories.items()
        }

    def _thermal_recovery_records(
        self,
    ) -> tuple[tuple[BaselineSlot, int, Path], ...]:
        records: list[tuple[BaselineSlot, int, Path]] = []
        for metric, metric_path in self._category_metric_directories("thermal-waits"):
            for pair_path in metric_path.iterdir():
                _require_directory(pair_path, label="thermal recovery slot state")
                slot = BaselineSlot(
                    metric,
                    _parse_decimal_index(
                        pair_path.name, label="thermal recovery pair index"
                    ),
                )
                for path in pair_path.iterdir():
                    _require_directory(path, label="thermal recovery state")
                    index = _parse_decimal_index(
                        path.name, label="thermal recovery index"
                    )
                    if path != self._thermal_wait_path(slot, index):
                        raise ValueError("thermal recovery has a non-canonical path")
                    trigger_path = path / "trigger.json"
                    _require_file(trigger_path, label="thermal recovery trigger")
                    self._validate_trigger_source(
                        slot,
                        read_json_object(
                            trigger_path, label="thermal recovery trigger"
                        ),
                    )
                    samples = self._recovery_sample_paths(path)
                    summary_path = path / "summary.json"
                    if _entry_exists(summary_path):
                        _require_file(summary_path, label="thermal recovery summary")
                        _validate_summary_document(
                            read_json_object(
                                summary_path, label="thermal recovery summary"
                            ),
                            sample_count=len(samples),
                        )
                    records.append((slot, index, path))
        return tuple(records)

    def _validate_trigger_source(self, slot: BaselineSlot, trigger: dict) -> None:
        body = _validate_trigger_document(trigger)
        if body["source"] == "preflight":
            persisted = {
                _json_text(document)
                for _index, document in self._validate_preflight_history().get(slot, ())
            }
            if _json_text(body["preflight"]) not in persisted:
                raise ValueError(
                    "thermal recovery trigger does not match a persisted preflight"
                )
            return

        reasons_by_identity: dict[str, str] = {}
        for attempt, path in self._attempt_records("rejected"):
            if attempt.slot != slot:
                continue
            rejected = read_json_object(path, label="rejected trial")
            rejected_body, _trial = _validate_rejected_document(
                rejected,
                slot=attempt.slot,
                journal_attempt_index=attempt.journal_attempt_index,
            )
            reasons_by_identity[rejected["identity"]] = rejected_body["reason"]
        rejected_identity = body["rejected_trial_identity"]
        if rejected_identity not in reasons_by_identity:
            raise ValueError(
                "thermal recovery trigger does not match a persisted rejected trial"
            )
        if reasons_by_identity[rejected_identity] != "non-nominal-thermal":
            raise ValueError(
                "thermal recovery trigger requires a non-nominal-thermal rejected trial"
            )

    def _validate_thermal_recovery_history(
        self,
    ) -> dict[BaselineSlot, tuple[int, ...]]:
        histories: dict[BaselineSlot, list[int]] = {}
        for slot, index, _path in self._thermal_recovery_records():
            histories.setdefault(slot, []).append(index)
        for slot, indices in histories.items():
            if sorted(indices) != list(range(len(indices))):
                raise ValueError(
                    f"thermal recovery indices have a gap for {slot.metric}"
                )
        return {slot: tuple(sorted(indices)) for slot, indices in histories.items()}

    def load_accepted(
        self, expected_slots: Sequence[BaselineSlot]
    ) -> dict[BaselineSlot, RawTrial]:
        expected = self._expected_slot_set(expected_slots)
        records = self._accepted_records()
        recorded_slots: set[BaselineSlot] = set()
        for slot, _path in records:
            if slot not in expected:
                raise ValueError("unexpected accepted slot")
            if slot in recorded_slots:
                raise ValueError("accepted slot is duplicated")
            recorded_slots.add(slot)

        self._validate_attempt_history()
        accepted: dict[BaselineSlot, RawTrial] = {}
        for slot, path in records:
            accepted[slot] = _parse_trial_for_slot(
                read_json_object(path, label="accepted trial"),
                slot=slot,
                label="accepted trial",
            )
        return accepted

    def next_attempt(self, slot: BaselineSlot) -> JournalAttempt:
        self._require_slot(slot)
        histories, _inflight = self._validate_attempt_history()
        indices = histories.get(slot, ())
        return JournalAttempt(
            slot, len(indices), self.inflight_path(slot, len(indices))
        )

    def load_inflight(
        self, expected_slots: Sequence[BaselineSlot]
    ) -> tuple[JournalAttempt, ...]:
        expected = self._expected_slot_set(expected_slots)
        _histories, pending = self._validate_attempt_history()
        inflight = tuple(state.attempt for state in pending if state.trial is not None)
        for attempt in inflight:
            if attempt.slot not in expected:
                raise ValueError("unexpected inflight slot")
        order = {slot: index for index, slot in enumerate(expected_slots)}
        return tuple(
            sorted(
                inflight,
                key=lambda attempt: (
                    order[attempt.slot],
                    attempt.journal_attempt_index,
                ),
            )
        )

    def load_pending_attempts(
        self, expected_slots: Sequence[BaselineSlot]
    ) -> tuple[JournalAttemptEvidence, ...]:
        expected = self._expected_slot_set(expected_slots)
        _histories, pending = self._validate_attempt_history()
        for state in pending:
            if state.attempt.slot not in expected:
                raise ValueError("unexpected pending slot")
        order = {slot: index for index, slot in enumerate(expected_slots)}
        return tuple(
            sorted(
                pending,
                key=lambda state: (
                    order[state.attempt.slot],
                    state.attempt.journal_attempt_index,
                ),
            )
        )

    def _require_canonical_attempt(self, attempt: JournalAttempt) -> None:
        if not isinstance(attempt, JournalAttempt):
            raise ValueError("journal attempt must be a JournalAttempt")
        expected = self.inflight_path(attempt.slot, attempt.journal_attempt_index)
        if attempt.path != expected:
            raise ValueError("journal attempt path does not match its slot")

    @staticmethod
    def _trial_for_attempt(attempt: JournalAttempt, trial: RawTrial) -> RawTrial:
        if not isinstance(trial, RawTrial):
            raise ValueError("journal trial must be a RawTrial")
        return _parse_trial_for_slot(
            trial.to_dict(), slot=attempt.slot, label="journal trial"
        )

    def accept_inflight(self, attempt: JournalAttempt, trial: RawTrial) -> None:
        self._require_canonical_attempt(attempt)
        validated_trial = self._trial_for_attempt(attempt, trial)
        histories, _inflight = self._validate_attempt_history()
        accepted_path = self.accepted_path(attempt.slot)
        expected_bytes = _json_bytes(validated_trial.to_dict())
        if _entry_exists(accepted_path):
            _require_file(accepted_path, label="accepted slot")
            if accepted_path.read_bytes() != expected_bytes:
                raise ValueError("accepted slot is immutable")
            if not _entry_exists(attempt.path):
                return
        _require_file(attempt.path, label="inflight trial")
        inflight_trial = _parse_trial_for_slot(
            read_json_object(attempt.path, label="inflight trial"),
            slot=attempt.slot,
            label="inflight trial",
        )
        if inflight_trial != validated_trial:
            raise ValueError("inflight trial does not match the supplied trial")
        indices = histories.get(attempt.slot, ())
        if not indices or indices[-1] != attempt.journal_attempt_index:
            raise ValueError("journal attempt indices have a gap")
        _create_durable_directory(accepted_path.parent)
        _fsync_directory(accepted_path.parent)
        try:
            os.link(attempt.path, accepted_path, follow_symlinks=False)
        except FileExistsError:
            _require_file(accepted_path, label="accepted slot")
            if accepted_path.read_bytes() != expected_bytes:
                raise ValueError("accepted slot is immutable")
        else:
            _fsync_directory(accepted_path.parent)
        attempt.path.unlink()
        _fsync_directory(attempt.path.parent)

    def reject_inflight(
        self, attempt: JournalAttempt, trial: RawTrial, *, reason: str
    ) -> None:
        self._require_canonical_attempt(attempt)
        validated_trial = self._trial_for_attempt(attempt, trial)
        _require_nonempty_string(reason, label="rejected trial reason")
        histories, _inflight = self._validate_attempt_history()
        measurement = validated_trial.child_measurement
        post_exit = validated_trial.post_exit_observation
        body = {
            "kind": "sml-baseline-rejected-trial",
            "version": 2,
            "journal_attempt_index": attempt.journal_attempt_index,
            "reason": reason,
            "child_measurement_identity": measurement["identity"],
            "post_exit_observation_identity": post_exit["identity"],
            "trial": validated_trial.to_dict(),
        }
        document = {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v2", body),
        }
        rejected_path = self.rejected_path(attempt.slot, attempt.journal_attempt_index)
        if _entry_exists(rejected_path) and not _entry_exists(attempt.path):
            _write_immutable_json(rejected_path, document, label="rejected trial")
            return
        _require_file(attempt.path, label="inflight trial")
        inflight_trial = _parse_trial_for_slot(
            read_json_object(attempt.path, label="inflight trial"),
            slot=attempt.slot,
            label="inflight trial",
        )
        if inflight_trial != validated_trial:
            raise ValueError("inflight trial does not match the supplied trial")
        indices = histories.get(attempt.slot, ())
        if not indices or indices[-1] != attempt.journal_attempt_index:
            raise ValueError("journal attempt indices have a gap")
        _write_immutable_json(rejected_path, document, label="rejected trial")
        attempt.path.unlink()
        _fsync_directory(attempt.path.parent)

    def reject_unfinalized(
        self, attempt: JournalAttempt, measurement: dict, *, reason: str
    ) -> None:
        self._require_canonical_attempt(attempt)
        if reason != "missing-immediate-post-exit-evidence":
            raise ValueError("unfinalized rejection has an invalid reason")
        validated_measurement = validate_child_trial_measurement(measurement)
        payload = validated_measurement["trial"]
        if validated_measurement["session_identity"] != self.session["identity"]:
            raise ValueError("child measurement does not match the journal session")
        if (
            validated_measurement["journal_attempt_index"]
            != attempt.journal_attempt_index
        ):
            raise ValueError("child measurement does not match the journal attempt")
        if (
            payload["metric"] != attempt.slot.metric
            or type(payload["pair_index"]) is not int
            or payload["pair_index"] != attempt.slot.pair_index
        ):
            raise ValueError(
                "child measurement metric or pair index does not match its slot"
            )
        histories, pending = self._validate_attempt_history()
        expected_pending = tuple(
            state
            for state in pending
            if state.attempt.slot == attempt.slot
            and state.attempt.journal_attempt_index == attempt.journal_attempt_index
        )
        rejected_path = self.rejected_path(attempt.slot, attempt.journal_attempt_index)
        if not _entry_exists(rejected_path):
            if len(expected_pending) != 1:
                raise ValueError("unfinalized rejection has no pending measurement")
            state = expected_pending[0]
            if (
                state.measurement != validated_measurement
                or state.post_exit is not None
                or state.trial is not None
            ):
                raise ValueError(
                    "unfinalized rejection requires measurement-only evidence"
                )
            indices = histories.get(attempt.slot, ())
            if not indices or indices[-1] != attempt.journal_attempt_index:
                raise ValueError("journal attempt indices have a gap")
        body = {
            "kind": "sml-baseline-rejected-trial",
            "version": 2,
            "journal_attempt_index": attempt.journal_attempt_index,
            "reason": reason,
            "child_measurement_identity": validated_measurement["identity"],
            "post_exit_observation_identity": None,
            "trial": None,
        }
        document = {
            **body,
            "identity": structured_identity("sml-baseline-rejected-trial-v2", body),
        }
        _write_immutable_json(rejected_path, document, label="rejected trial")

    def record_preflight(
        self, slot: BaselineSlot, preflight_index: int, observation: dict
    ) -> dict:
        self._require_slot(slot)
        _require_non_negative_index(preflight_index, label="preflight index")
        histories = self._validate_preflight_history()
        indices = histories.get(slot, ())
        if preflight_index > len(indices):
            raise ValueError("preflight indices have a gap")
        _require_observation(
            observation, label="preflight observation", require_schema_version=False
        )
        body = {
            "kind": "sml-baseline-preflight",
            "version": 1,
            "metric": slot.metric,
            "pair_index": slot.pair_index,
            "preflight_index": preflight_index,
            **observation,
        }
        document = {
            **body,
            "identity": structured_identity("sml-baseline-preflight-v1", body),
        }
        _write_immutable_json(
            self.preflight_path(slot, preflight_index), document, label="preflight"
        )
        return document

    def _recovery_sample_paths(self, recovery_path: Path) -> tuple[Path, ...]:
        if not _entry_exists(recovery_path):
            return ()
        _require_directory(recovery_path, label="thermal recovery state")
        paths: dict[int, Path] = {}
        for path in recovery_path.iterdir():
            if path.name in {"trigger.json", "summary.json"}:
                continue
            _require_file(path, label="thermal sample")
            if path.suffix != ".json":
                raise ValueError("thermal sample filename must end in .json")
            index = _parse_decimal_index(path.stem, label="thermal sample filename")
            if index in paths:
                raise ValueError("thermal sample is duplicated")
            paths[index] = path
        if sorted(paths) != list(range(len(paths))):
            raise ValueError("thermal sample indices have a gap")
        for index, path in paths.items():
            _validate_sample_document(
                read_json_object(path, label="thermal sample"), sample_index=index
            )
        return tuple(paths[index] for index in range(len(paths)))

    def record_recovery_trigger(
        self, slot: BaselineSlot, recovery_index: int, trigger: dict
    ) -> None:
        recovery_path = self._thermal_wait_path(slot, recovery_index)
        if not isinstance(trigger, dict):
            raise ValueError("thermal recovery trigger must be an object")
        self._validate_attempt_history()
        self._validate_preflight_history()
        recovery_histories = self._validate_thermal_recovery_history()
        recovery_indices = recovery_histories.get(slot, ())
        if recovery_index > len(recovery_indices):
            raise ValueError("thermal recovery indices have a gap")
        body = {
            "kind": "sml-baseline-thermal-recovery-trigger",
            "version": 1,
            **trigger,
        }
        document = {
            **body,
            "identity": structured_identity(
                "sml-baseline-thermal-recovery-trigger-v1", body
            ),
        }
        self._validate_trigger_source(slot, document)
        trigger_path = recovery_path / "trigger.json"
        if _entry_exists(recovery_path) and not _entry_exists(trigger_path):
            _require_directory(recovery_path, label="thermal recovery state")
            if tuple(recovery_path.iterdir()):
                raise ValueError("thermal recovery trigger must be recorded first")
        _write_immutable_json(trigger_path, document, label="thermal recovery trigger")

    def record_thermal_sample(
        self,
        slot: BaselineSlot,
        recovery_index: int,
        sample_index: int,
        sample: dict,
    ) -> None:
        recovery_path = self._thermal_wait_path(slot, recovery_index)
        _require_non_negative_index(sample_index, label="thermal sample index")
        self._validate_attempt_history()
        self._validate_preflight_history()
        recovery_histories = self._validate_thermal_recovery_history()
        if recovery_index not in recovery_histories.get(slot, ()):
            raise ValueError("thermal recovery index does not exist")
        _require_observation(
            sample, label="thermal sample", require_schema_version=True
        )
        trigger_path = recovery_path / "trigger.json"
        _require_file(trigger_path, label="thermal recovery trigger")
        self._validate_trigger_source(
            slot,
            read_json_object(trigger_path, label="thermal recovery trigger"),
        )
        samples = self._recovery_sample_paths(recovery_path)
        if sample_index != len(samples):
            raise ValueError("thermal sample indices have a gap")
        summary_path = recovery_path / "summary.json"
        if _entry_exists(summary_path):
            raise ValueError("thermal recovery summary already exists")
        body = {
            "kind": "sml-baseline-thermal-sample",
            "version": 1,
            "sample_index": sample_index,
            **sample,
        }
        document = {
            **body,
            "identity": structured_identity("sml-baseline-thermal-sample-v1", body),
        }
        _write_immutable_json(
            recovery_path / f"{sample_index}.json", document, label="thermal sample"
        )

    def record_recovery_summary(
        self, slot: BaselineSlot, recovery_index: int, summary: dict
    ) -> None:
        recovery_path = self._thermal_wait_path(slot, recovery_index)
        if not isinstance(summary, dict):
            raise ValueError("thermal recovery summary must be an object")
        _require_object_fields(
            summary,
            {"outcome", "duration_seconds", "sample_count"},
            label="thermal recovery summary",
        )
        self._validate_attempt_history()
        self._validate_preflight_history()
        recovery_histories = self._validate_thermal_recovery_history()
        if recovery_index not in recovery_histories.get(slot, ()):
            raise ValueError("thermal recovery index does not exist")
        trigger_path = recovery_path / "trigger.json"
        _require_file(trigger_path, label="thermal recovery trigger")
        self._validate_trigger_source(
            slot,
            read_json_object(trigger_path, label="thermal recovery trigger"),
        )
        samples = self._recovery_sample_paths(recovery_path)
        body = {
            "kind": "sml-baseline-thermal-recovery-summary",
            "version": 1,
            **summary,
        }
        document = {
            **body,
            "identity": structured_identity(
                "sml-baseline-thermal-recovery-summary-v1", body
            ),
        }
        _validate_summary_document(document, sample_count=len(samples))
        _write_immutable_json(
            recovery_path / "summary.json",
            document,
            label="thermal recovery summary",
        )

    def publish_completed(self, document: dict) -> None:
        if not isinstance(document, dict):
            raise ValueError("baseline journal completion must be an object")
        _validate_completion_document(
            document, session_identity=self.session["identity"]
        )
        _write_immutable_json(
            self.completed_path, document, label="baseline journal completion"
        )
