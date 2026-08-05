# V2 Baseline Thermal Recovery and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned v2 baseline capture record exact macOS thermal observations, preserve accepted and rejected work durably, wait for sustained nominal thermals, and resume only identity-compatible missing trials.

**Architecture:** Add a focused journal module for atomic session/trial persistence and a focused recovery module for the pure thermal waiting state machine. Keep benchmark orchestration in `runner.py`, but extract single-trial validation and a dependency-injected capture loop so retry and resume behavior is tested without running the full MLX workload.

**Tech Stack:** Python 3.12.13, standard-library dataclasses/JSON/filesystem primitives, macOS Foundation `ProcessInfo.thermalState`, MLX 0.32+, pytest 9, Ruff 0.15+, `uv run`.

## Global Constraints

- The approved source of truth for this change is `docs/superpowers/specs/2026-08-05-v2-baseline-thermal-resume-design.md`.
- Do not edit `pyproject.toml`, `uv.lock`, or another top-level project file.
- Do not add dependencies.
- Preserve the pinned baseline source commit `3687f8b3214a44c675ae67af52e4997762f6c634`, nine canonical metrics, five trials per metric, 20 warmups, and canonical measured-unit counts.
- Keep `RawTrial.attempt_index == 0` for baseline evidence; journal attempt indices are separate diagnostic sequence numbers.
- Retry only for a non-nominal thermal observation. Never inspect a measured performance value when accepting, rejecting, skipping, or retrying a trial.
- Non-thermal environment, identity, schema, protocol, subprocess, hardware, or software failures stop the invocation while preserving journal state.
- Require five continuous nominal minutes before retry, sample at most every 30 seconds, and stop after the current invocation's two-hour per-slot recovery deadline.
- Write journal and final artifact files with flush, file `fsync`, atomic rename, and parent-directory `fsync`.
- The state directory must resolve outside the harness and measurement source Git checkouts.
- The final manifest and raw JSONL remain absent until exactly 45 accepted slots validate; rejected attempts and wait samples never enter final evidence.
- Run every MLX pytest command outside the sandbox so Metal is available.
- Before the final harness commit, run `uv run ruff check v2`, `uv run ruff format --check v2`, `uv run pytest v2/tests`, and verify `uv.lock` is unchanged.

---

## File Structure

- Create `v2/benchmarks/journal.py`: strict session schema, durable atomic JSON publication, slot paths, accepted/in-flight/rejected state, recovery samples, completion marker, and resume loading.
- Create `v2/benchmarks/recovery.py`: deterministic two-hour/five-minute thermal recovery state machine with injected clock, sleep, collection, and sample persistence.
- Modify `v2/benchmarks/runner.py`: exact raw thermal capture, per-trial baseline validation, resumable capture orchestration, progress output, CLI wiring, and final publication.
- Modify `v2/benchmarks/workload.py`: include both new runtime modules in the harness content identity.
- Modify `v2/benchmarks/README.md`: document `--state-directory`, resume rules, thermal diagnostics, and retry policy.
- Modify `v2/tests/unit/test_benchmark_analysis.py`: add thermal, journal, recovery, orchestration, resume, and final-publication regressions; update valid environment fixtures with raw thermal values.

### Task 1: Record and Validate Exact Thermal Values

**Files:**
- Modify: `v2/benchmarks/runner.py:1391-1411`
- Modify: `v2/benchmarks/runner.py:1549-1631`
- Modify: `v2/benchmarks/runner.py:442-459`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces: `decode_thermal_state(raw_value: int) -> str`
- Produces: `validate_thermal_observation(status: dict[str, object]) -> None`
- Changes: `_thermal_state() -> tuple[str, int]`
- Changes: every collected start/end/merged environment contains matching `thermal_state` and `thermal_state_raw_value`

- [ ] **Step 1: Write failing thermal mapping and merge tests**

Add imports for `decode_thermal_state` and `validate_thermal_observation`, update `_valid_raw_trial` so its top-level environment and nested `start`/`end` observations contain `thermal_state_raw_value=0`, and add:

```python
@pytest.mark.parametrize(
    ("raw_value", "state"),
    [(0, "nominal"), (1, "fair"), (2, "serious"), (3, "critical")],
)
def test_thermal_state_retains_foundation_raw_value(raw_value, state):
    assert decode_thermal_state(raw_value) == state
    validate_thermal_observation(
        {"thermal_state": state, "thermal_state_raw_value": raw_value}
    )


def test_thermal_merge_retains_the_worse_matching_raw_value():
    start = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 60,
        "competing_gpu_workload": False,
    }
    end = {**start, "thermal_state": "serious", "thermal_state_raw_value": 2}

    merged = merge_environment_status(start, end)

    assert merged["thermal_state"] == "serious"
    assert merged["thermal_state_raw_value"] == 2
    validate_thermal_observation(merged)


def test_thermal_observation_rejects_a_mismatched_string_and_raw_value():
    with pytest.raises(ValueError, match="thermal state and raw value disagree"):
        validate_thermal_observation(
            {"thermal_state": "nominal", "thermal_state_raw_value": 1}
        )


def test_thermal_observation_rejects_a_merged_value_that_is_not_the_worst_endpoint():
    nominal = {"thermal_state": "nominal", "thermal_state_raw_value": 0}
    fair = {"thermal_state": "fair", "thermal_state_raw_value": 1}
    with pytest.raises(ValueError, match="merged thermal state"):
        validate_thermal_observation(
            {
                "thermal_state": "nominal",
                "thermal_state_raw_value": 0,
                "start": nominal,
                "end": fair,
            }
        )
```

- [ ] **Step 2: Run the tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'thermal_state_retains or thermal_merge_retains or thermal_observation_rejects' -v
```

Expected: collection fails because `decode_thermal_state` and `validate_thermal_observation` do not exist.

- [ ] **Step 3: Implement strict raw/string thermal capture**

Add:

```python
THERMAL_STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def decode_thermal_state(raw_value: int) -> str:
    if type(raw_value) is not int or raw_value not in THERMAL_STATES:
        raise RuntimeError(f"unsupported ProcessInfo thermal state: {raw_value!r}")
    return THERMAL_STATES[raw_value]


def validate_thermal_observation(status: dict[str, object]) -> None:
    raw_value = status.get("thermal_state_raw_value")
    state = status.get("thermal_state")
    if type(raw_value) is not int or state != decode_thermal_state(raw_value):
        raise ValueError("thermal state and raw value disagree")
    for nested_name in ("start", "end"):
        nested = status.get(nested_name)
        if nested is not None:
            if not isinstance(nested, dict):
                raise ValueError(f"thermal {nested_name} observation must be an object")
            validate_thermal_observation(nested)
    if isinstance(status.get("start"), dict) and isinstance(status.get("end"), dict):
        expected_raw = max(
            status["start"]["thermal_state_raw_value"],
            status["end"]["thermal_state_raw_value"],
        )
        if raw_value != expected_raw:
            raise ValueError("merged thermal state is not the worse endpoint")
```

Parse the Swift output once and return both representations:

```python
def _thermal_state() -> tuple[str, int]:
    raw_text = subprocess.run(
        (
            "swift",
            "-e",
            "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    ).stdout.strip()
    try:
        raw_value = int(raw_text)
    except ValueError as error:
        raise RuntimeError(
            f"unsupported ProcessInfo thermal state: {raw_text!r}"
        ) from error
    return decode_thermal_state(raw_value), raw_value
```

In `collect_environment`, unpack the tuple and store both fields. In `merge_environment_status`, select the maximum raw value and derive the string through `decode_thermal_state`. Call `validate_thermal_observation` from `_validate_acceptance_environment` before comparing required values.

- [ ] **Step 4: Run focused and existing environment tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'thermal or environment or power_status or memory_pressure' -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the thermal observation contract**

```bash
git add v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): record exact macOS thermal state"
```

### Task 2: Add Atomic Session Journal and Compatibility Gate

**Files:**
- Create: `v2/benchmarks/journal.py`
- Modify: `v2/benchmarks/workload.py:19-27`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces: `atomic_write_text(path: Path, text: str, *, create_only: bool = False) -> None`
- Produces: `atomic_write_json(path: Path, value: dict, *, create_only: bool = False) -> None`
- Produces: `read_json_object(path: Path, *, label: str) -> dict`
- Produces: `require_external_state_directory(state_directory: Path, checkouts: Sequence[Path]) -> Path`
- Produces: `build_session_document(*, harness_commit: str, harness_identity: str, source_commit: str, canonical_workload: CanonicalWorkload, canonical_workload_identity: str, protocol: dict[str, JsonValue], hardware: dict[str, JsonValue], software_versions: dict[str, str], paired_representations: dict[str, JsonValue], manifest_path: Path, raw_output_path: Path) -> dict`
- Produces: `BaselineJournal.open(root: Path, expected_session: dict) -> BaselineJournal`

- [ ] **Step 1: Write failing session identity, resume, and location tests**

Append:

```python
def _session_document(
    tmp_path,
    *,
    harness_commit="a" * 40,
    protocol=None,
    hardware=None,
    software_versions=None,
    paired_representations=None,
    manifest_name="baseline.json",
    raw_output_name="baseline.jsonl",
):
    workload = build_canonical_workload()
    return build_session_document(
        harness_commit=harness_commit,
        harness_identity="sha256:" + "b" * 64,
        source_commit="3687f8b3214a44c675ae67af52e4997762f6c634",
        canonical_workload=workload,
        canonical_workload_identity=canonical_workload_identity(workload),
        protocol=protocol
        or {"pairs": 5, "warmup_units": 20, "measured_units": 100},
        hardware=hardware or {"chip": "Apple M5"},
        software_versions=software_versions
        or {"python": "3.12.13", "mlx": "0.32.0"},
        paired_representations=paired_representations
        or {"canonical_row_identity": "sha256:" + "c" * 64},
        manifest_path=tmp_path / manifest_name,
        raw_output_path=tmp_path / raw_output_name,
    )


def test_baseline_journal_resumes_only_an_identical_session(tmp_path):
    state = tmp_path / "state"
    expected = _session_document(tmp_path)

    first = BaselineJournal.open(state, expected)
    resumed = BaselineJournal.open(state, expected)

    assert first.session == expected
    assert resumed.session == expected
    changed = _session_document(tmp_path, harness_commit="d" * 40)
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, changed)


@pytest.mark.parametrize(
    "changed",
    [
        {"protocol": {"pairs": 4, "warmup_units": 20, "measured_units": 100}},
        {"hardware": {"chip": "Apple M4"}},
        {"software_versions": {"python": "3.12.12", "mlx": "0.32.0"}},
        {"manifest_name": "other.json"},
        {"raw_output_name": "other.jsonl"},
    ],
)
def test_baseline_journal_rejects_every_session_compatibility_change(
    tmp_path, changed
):
    state = tmp_path / "state"
    BaselineJournal.open(state, _session_document(tmp_path))
    with pytest.raises(ValueError, match="session does not match"):
        BaselineJournal.open(state, _session_document(tmp_path, **changed))


def test_state_directory_must_be_outside_measured_checkouts(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    with pytest.raises(ValueError, match="outside measured checkouts"):
        require_external_state_directory(harness / "state", (harness,))

    external = tmp_path / "external"
    assert require_external_state_directory(external, (harness,)) == external.resolve()


def test_journal_never_adopts_a_nonempty_directory_without_a_session(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "orphan.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty state directory has no session"):
        BaselineJournal.open(state, _session_document(tmp_path))
```

- [ ] **Step 2: Run the new journal tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'baseline_journal or state_directory or journal_never_adopts' -v
```

Expected: collection fails because `v2.benchmarks.journal` does not exist.

- [ ] **Step 3: Implement crash-safe atomic publication**

Create `journal.py` with directory `fsync` and create-only publication:

```python
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from v2.benchmarks.schema import CanonicalWorkload, JsonValue
from v2.benchmarks.workload import structured_identity


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
```

The same-directory hard link is the create-only publication primitive: it atomically fails with `FileExistsError` when the destination exists and never overwrites it. Keep `os.replace` semantics only for non-create-only writes.

- [ ] **Step 4: Implement strict session creation and resume**

Use the exact body fields below and bind them with a structured identity:

```python
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
```

`BaselineJournal.open` must require the exact field set and exact equality with the recomputed expected document. It creates `session.json` with create-only atomic publication when absent. If `session.json` is absent but the state directory is non-empty, it rejects the directory instead of adopting orphaned content. `require_external_state_directory` resolves both the state path and checkouts and rejects equality or `state.is_relative_to(checkout)`.

- [ ] **Step 5: Add the new module to the harness identity and verify GREEN**

Insert `Path("v2/benchmarks/journal.py")` after `runner.py` in `HARNESS_COMPONENTS`, then run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'baseline_journal or state_directory or journal_never_adopts or harness_identity' -v
```

Expected: all selected tests pass and the identity test proves ordered inclusion of `journal.py`.

- [ ] **Step 6: Commit the session journal**

```bash
git add v2/benchmarks/journal.py v2/benchmarks/workload.py \
  v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): add durable baseline session journal"
```

### Task 3: Persist Accepted, In-Flight, and Rejected Trial Slots

**Files:**
- Modify: `v2/benchmarks/journal.py`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces: `BaselineSlot(metric: MetricName, pair_index: int)`
- Produces: `JournalAttempt(slot: BaselineSlot, journal_attempt_index: int, path: Path)`
- Produces: `BaselineJournal.expected_slots(metrics: Sequence[MetricName], pairs: int) -> tuple[BaselineSlot, ...]`
- Produces: `BaselineJournal.load_accepted(expected_slots) -> dict[BaselineSlot, RawTrial]`
- Produces: `BaselineJournal.next_attempt(slot) -> JournalAttempt`
- Produces: `BaselineJournal.load_inflight(expected_slots: Sequence[BaselineSlot]) -> tuple[JournalAttempt, ...]`
- Produces: `BaselineJournal.accepted_path(slot: BaselineSlot) -> Path`
- Produces: `BaselineJournal.inflight_path(slot: BaselineSlot, journal_attempt_index: int) -> Path`
- Produces: `BaselineJournal.rejected_path(slot: BaselineSlot, journal_attempt_index: int) -> Path`
- Produces: `BaselineJournal.preflight_path(slot: BaselineSlot, preflight_index: int) -> Path`
- Produces: `BaselineJournal.completed_path: Path`
- Produces: `BaselineJournal.accept_inflight(attempt, trial) -> None`
- Produces: `BaselineJournal.reject_inflight(attempt, trial, *, reason: str) -> None`
- Produces: `BaselineJournal.record_thermal_sample(slot: BaselineSlot, recovery_index: int, sample_index: int, sample: dict) -> None`
- Produces: `BaselineJournal.record_preflight(slot: BaselineSlot, preflight_index: int, observation: dict) -> dict`
- Produces: `BaselineJournal.record_recovery_trigger(slot: BaselineSlot, recovery_index: int, trigger: dict) -> None`
- Produces: `BaselineJournal.record_recovery_summary(slot: BaselineSlot, recovery_index: int, summary: dict) -> None`
- Produces: `BaselineJournal.publish_completed(document: dict) -> None`

- [ ] **Step 1: Write failing immutable-slot and diagnostic tests**

Append tests using the existing `_valid_raw_trial` helper:

```python
def test_journal_promotes_an_inflight_trial_and_resumes_the_slot(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    journal.accept_inflight(attempt, trial)

    assert journal.load_accepted((slot,)) == {slot: trial}
    assert not attempt.path.exists()
    with pytest.raises(ValueError, match="accepted slot is immutable"):
        journal.accept_inflight(
            JournalAttempt(slot, 1, journal.inflight_path(slot, 1)),
            replace(trial, value=trial.value + 1.0),
        )


def test_journal_preserves_rejected_trial_and_uses_a_new_attempt_number(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = replace(
        _valid_raw_trial(workload),
        environment_status={
            **_valid_raw_trial(workload).environment_status,
            "thermal_state": "fair",
            "thermal_state_raw_value": 1,
        },
    )
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")

    rejected = read_json_object(
        journal.rejected_path(slot, 0), label="rejected trial"
    )
    assert rejected["journal_attempt_index"] == 0
    assert rejected["reason"] == "non-nominal-thermal"
    assert rejected["trial"]["environment_status"]["thermal_state_raw_value"] == 1
    assert journal.next_attempt(slot).journal_attempt_index == 1


def test_journal_records_every_preflight_observation(tmp_path):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    observation = {
        "observed_at_utc": "2026-08-05T00:00:00+00:00",
        "hardware": trial.hardware,
        "environment_status": trial.environment_status,
        "software_versions": trial.software_versions,
    }

    document = journal.record_preflight(slot, 0, observation)

    assert document["identity"].startswith("sha256:")
    assert read_json_object(
        journal.preflight_path(slot, 0), label="preflight"
    ) == document


def test_journal_rejects_malformed_and_unexpected_accepted_state(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    atomic_write_json(journal.accepted_path(slot), {}, create_only=True)
    with pytest.raises(ValueError, match="raw trial has an invalid field set"):
        journal.load_accepted((slot,))

    journal.accepted_path(slot).unlink()
    unexpected = BaselineSlot("prepared-data", 1)
    atomic_write_json(journal.accepted_path(unexpected), {}, create_only=True)
    with pytest.raises(ValueError, match="unexpected accepted slot"):
        journal.load_accepted((slot,))
```

- [ ] **Step 2: Run the slot tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'journal_promotes or journal_preserves_rejected or journal_records_every or journal_rejects_malformed' -v
```

Expected: failures because slot and attempt APIs do not exist.

- [ ] **Step 3: Implement strict slot paths and attempt discovery**

Add frozen types and validate metric/pair/attempt values at construction:

```python
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
```

Use canonical metric directory names and decimal indices without user-derived path fragments. `next_attempt` scans strict rejected and in-flight filenames, rejects gaps or malformed names, and returns the first unused sequence number. `expected_slots` follows the supplied metric order, then ascending pair index.

- [ ] **Step 4: Implement promotion, rejection, recovery samples, and resume loading**

`load_accepted` must parse every file through `RawTrial.from_dict`, require exact expected slots, and reject unexpected files. `accept_inflight` requires the trial metric/pair to match the attempt and uses atomic rename plus directory `fsync`. If the accepted destination exists, only byte-for-byte equivalent trial content is idempotent; different content raises `ValueError("accepted slot is immutable")`.

Rejected documents have this exact shape and structured identity:

```python
body = {
    "kind": "sml-baseline-rejected-trial",
    "version": 1,
    "journal_attempt_index": attempt.journal_attempt_index,
    "reason": reason,
    "trial": trial.to_dict(),
}
document = {
    **body,
    "identity": structured_identity("sml-baseline-rejected-trial-v1", body),
}
```

Preflight documents are immutable numbered records with wall-clock time, hardware, environment, software, and a structured identity. Each recovery directory first receives an immutable `trigger.json` that records whether recovery began from `preflight` or `rejected-trial` and embeds the exact preflight document or rejected journal attempt identity. Thermal samples are immutable numbered JSON documents with `sample_index`, `observed_at_utc`, `elapsed_seconds`, hardware, environment, and software. A recovery summary is create-only and records `outcome` as `nominal-window` or `timeout`, duration, and sample count. `publish_completed` is create-only and idempotent only for identical content.

- [ ] **Step 5: Verify slot persistence and the complete focused harness suite**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'journal or raw_trial_round_trip' -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit durable trial slots**

```bash
git add v2/benchmarks/journal.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): persist baseline trial slots"
```

### Task 4: Add the Deterministic Thermal Recovery State Machine

**Files:**
- Create: `v2/benchmarks/recovery.py`
- Modify: `v2/benchmarks/workload.py:19-29`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces: `ThermalRecoveryTimeout(RuntimeError)`
- Produces: `ThermalRecoveryResult(duration_seconds: float, sample_count: int)`
- Produces: `wait_for_nominal_thermal_window(*, collect: Callable[[], tuple[dict, dict, dict]], expected_hardware: dict, expected_software_versions: dict[str, str], required_environment: dict[str, object], record_sample: Callable[[int, dict], None], deadline: float, clock: Callable[[], float], sleep: Callable[[float], None], utc_now: Callable[[], str]) -> ThermalRecoveryResult`

- [ ] **Step 1: Write failing continuous-window, reset, and timeout tests**

Append a fake clock and deterministic collector:

```python
class _RecoveryClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _recovery_collect(states):
    remaining = iter(states)
    last = [states[-1]]

    def collect():
        try:
            last[0] = next(remaining)
        except StopIteration:
            pass
        status = {
            "power_connected": True,
            "power_mode": "automatic",
            "low_power_mode": False,
            "thermal_state": last[0],
            "thermal_state_raw_value": {
                "nominal": 0,
                "fair": 1,
                "serious": 2,
                "critical": 3,
            }[last[0]],
            "memory_pressure": "normal",
            "memory_free_percentage": 60,
            "competing_gpu_workload": False,
        }
        return {"chip": "Apple M5"}, status, {"python": "3.12.13"}

    return collect


def test_thermal_recovery_requires_five_continuous_nominal_minutes():
    clock = _RecoveryClock()
    samples = []
    result = wait_for_nominal_thermal_window(
        collect=_recovery_collect(["nominal"] * 4 + ["fair"] + ["nominal"] * 20),
        expected_hardware={"chip": "Apple M5"},
        expected_software_versions={"python": "3.12.13"},
        required_environment=build_canonical_workload().required_environment,
        record_sample=lambda index, sample: samples.append((index, sample)),
        deadline=clock() + 7_200,
        clock=clock,
        sleep=clock.sleep,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
    )

    nominal_times = [
        sample["elapsed_seconds"]
        for _index, sample in samples
        if sample["environment_status"]["thermal_state"] == "nominal"
    ]
    assert result.duration_seconds >= 300
    assert nominal_times[-1] - nominal_times[4] >= 300


def test_thermal_recovery_times_out_without_losing_samples():
    clock = _RecoveryClock()
    samples = []
    with pytest.raises(ThermalRecoveryTimeout, match="two-hour deadline"):
        wait_for_nominal_thermal_window(
            collect=_recovery_collect(["fair"]),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.13"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: samples.append((index, sample)),
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
    assert samples[-1][1]["environment_status"]["thermal_state"] == "fair"
    assert samples[-1][1]["elapsed_seconds"] >= 120
```

- [ ] **Step 2: Run recovery tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'thermal_recovery_requires or thermal_recovery_times_out' -v
```

Expected: collection fails because `v2.benchmarks.recovery` does not exist.

- [ ] **Step 3: Implement the recovery loop**

Create a frozen result and timeout carrying the final result:

```python
@dataclass(frozen=True, slots=True)
class ThermalRecoveryResult:
    duration_seconds: float
    sample_count: int


class ThermalRecoveryTimeout(RuntimeError):
    def __init__(self, result: ThermalRecoveryResult):
        super().__init__("thermal recovery exceeded the two-hour deadline")
        self.result = result
```

Implement `wait_for_nominal_thermal_window` with keyword-only dependencies. Capture `started = clock()`, sample immediately, and schedule on a 27-second cadence so real scheduler overhead remains below the declared 30-second maximum. Cap the next scheduled time at `deadline` so timeout is sampled and raised without overshooting it. For every sample:

1. require exact hardware and software equality;
2. call `validate_thermal_observation`;
3. require AC connected, expected power mode, Low Power Mode false, normal memory pressure, and no competing GPU workload;
4. persist the sample through `record_sample` before a return or exception;
5. set `nominal_since` on the first nominal sample and reset it to `None` on every non-nominal sample;
6. return only when `elapsed - nominal_since >= 300`; and
7. raise `ThermalRecoveryTimeout` when `clock() >= deadline`.

The sample document passed to the callback is:

```python
{
    "schema_version": 1,
    "observed_at_utc": utc_now(),
    "elapsed_seconds": elapsed,
    "hardware": hardware,
    "environment_status": status,
    "software_versions": software_versions,
}
```

- [ ] **Step 4: Add non-thermal failure coverage and verify GREEN**

Add this parameterized non-thermal failure test:

```python
@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("status", "power_connected", False),
        ("status", "low_power_mode", True),
        ("status", "memory_pressure", "warning"),
        ("status", "competing_gpu_workload", True),
        ("hardware", "chip", "Apple M4"),
        ("software", "python", "3.12.12"),
    ],
)
def test_thermal_recovery_stops_on_nonthermal_changes(target, field, value):
    clock = _RecoveryClock()
    hardware = {"chip": "Apple M5"}
    status = _recovery_collect(["fair"])()[1]
    software = {"python": "3.12.13"}
    if target == "hardware":
        hardware[field] = value
    elif target == "software":
        software[field] = value
    else:
        status[field] = value

    with pytest.raises(ValueError, match=field):
        wait_for_nominal_thermal_window(
            collect=lambda: (hardware, status, software),
            expected_hardware={"chip": "Apple M5"},
            expected_software_versions={"python": "3.12.13"},
            required_environment=build_canonical_workload().required_environment,
            record_sample=lambda index, sample: None,
            deadline=120.0,
            clock=clock,
            sleep=clock.sleep,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
        )
```

Then run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k 'thermal_recovery' -v
```

Expected: continuous-window, timeout, and non-thermal failure tests pass.

- [ ] **Step 5: Version and commit the recovery module**

Add `Path("v2/benchmarks/recovery.py")` after `journal.py` in `HARNESS_COMPONENTS`, then run the harness identity test and commit:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'thermal_recovery or harness_identity' -v
git add v2/benchmarks/recovery.py v2/benchmarks/workload.py \
  v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): wait for sustained nominal thermals"
```

### Task 5: Integrate Resumable Baseline Capture and Final Publication

**Files:**
- Modify: `v2/benchmarks/runner.py:297-416`
- Modify: `v2/benchmarks/runner.py:1634-1679`
- Modify: `v2/benchmarks/runner.py:1783-1910`
- Modify: `v2/benchmarks/runner.py:2632-2640`
- Modify: `v2/benchmarks/README.md`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces: `validate_baseline_trial(trial: RawTrial, *, workload: CanonicalWorkload, source_commit: str, harness_commit: str, harness_identity: str, expected_hardware: dict, expected_software_versions: dict[str, str], allow_non_nominal_thermal: bool = False) -> None`
- Produces: `capture_baseline_trials(*, journal: BaselineJournal, slots: Sequence[BaselineSlot], launch_trial: Callable[[BaselineSlot, JournalAttempt], RawTrial], preflight: Callable[[], tuple[dict, dict, dict]], validate_preflight: Callable[[dict, dict, dict], None], recover: Callable[[BaselineSlot, int, float, dict], None], validate_trial: Callable[[RawTrial, bool], None], clock: Callable[[], float] = time.monotonic, utc_now: Callable[[], str] = _utc_now_iso, progress: Callable[[str], None] = print) -> tuple[RawTrial, ...]`
- Produces: `validate_checkout_status(status: str, *, allowed_untracked_paths: frozenset[str]) -> None`
- Produces: `publish_baseline_from_journal(*, journal: BaselineJournal, trials: Sequence[RawTrial], workload: CanonicalWorkload, workload_identity: str, source_commit: str, harness_commit: str, harness_identity: str, command: str, paired_representations: dict, manifest_path: Path, raw_output_path: Path) -> dict`
- Changes: `record-baseline` requires `--state-directory PATH`
- Changes: final output publication is create-only or identical and writes `completed.json` last

- [ ] **Step 1: Write failing parser, retry-only-slot, and resume tests**

Update the existing parser case so `record-baseline` includes `--state-directory state`, assert omission fails, and add a dependency-injected capture test:

```python
def test_capture_retries_only_the_thermal_slot_and_resumes_accepted_slots(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slots = (
        BaselineSlot("prepared-data", 0),
        BaselineSlot("prepared-data", 1),
    )
    accepted_first = _valid_raw_trial(workload, pair_index=0)
    first_attempt = journal.next_attempt(slots[0])
    atomic_write_json(first_attempt.path, accepted_first.to_dict(), create_only=True)
    journal.accept_inflight(first_attempt, accepted_first)
    launches = []

    def launch(slot, attempt):
        launches.append(slot)
        base = _valid_raw_trial(workload, pair_index=slot.pair_index)
        if len(launches) == 1:
            return replace(
                base,
                environment_status={
                    **base.environment_status,
                    "thermal_state": "fair",
                    "thermal_state_raw_value": 1,
                },
            )
        return base

    recovered = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=slots,
        launch_trial=launch,
        preflight=lambda: (
            accepted_first.hardware,
            accepted_first.environment_status,
            accepted_first.software_versions,
        ),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, recovery_index, deadline, trigger: recovered.append(slot),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slots[1], slots[1]]
    assert recovered == [slots[1]]
    assert [(trial.metric, trial.pair_index) for trial in trials] == [
        ("prepared-data", 0),
        ("prepared-data", 1),
    ]
    rejected = read_json_object(
        journal.rejected_path(slots[1], 0), label="rejected trial"
    )
    assert rejected["trial"]["environment_status"]["thermal_state"] == "fair"


def test_capture_records_preflight_thermal_trigger_before_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    nominal_trial = _valid_raw_trial(workload)
    fair = {
        **nominal_trial.environment_status,
        "thermal_state": "fair",
        "thermal_state_raw_value": 1,
    }
    preflights = iter(
        [
            (nominal_trial.hardware, fair, nominal_trial.software_versions),
            (
                nominal_trial.hardware,
                nominal_trial.environment_status,
                nominal_trial.software_versions,
            ),
        ]
    )
    triggers = []
    launches = []

    capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, attempt: (
            launches.append(current_slot) or nominal_trial
        ),
        preflight=lambda: next(preflights),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: (
            triggers.append(trigger)
        ),
        validate_trial=lambda trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert launches == [slot]
    assert triggers[0]["source"] == "preflight"
    assert triggers[0]["preflight"]["environment_status"][
        "thermal_state_raw_value"
    ] == 1
    assert journal.preflight_path(slot, 0).is_file()


def test_capture_classifies_complete_inflight_before_launching(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    trial = _valid_raw_trial(workload)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda current_slot, current_attempt: pytest.fail(
            "resume launched a replacement for a complete in-flight trial"
        ),
        preflight=lambda: pytest.fail("accepted in-flight slot reached preflight"),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda current_slot, recovery_index, deadline, trigger: None,
        validate_trial=lambda current_trial, allow_non_nominal_thermal: None,
        clock=lambda: 0.0,
        utc_now=lambda: "2026-08-05T00:00:00+00:00",
        progress=lambda message: None,
    )

    assert trials == (trial,)
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_capture_timeout_preserves_prior_acceptance_and_rejected_attempt(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    accepted = _valid_raw_trial(workload, pair_index=0)
    attempt = journal.next_attempt(first)
    atomic_write_json(attempt.path, accepted.to_dict(), create_only=True)
    journal.accept_inflight(attempt, accepted)
    fair_trial = replace(
        _valid_raw_trial(workload, pair_index=1),
        environment_status={
            **_valid_raw_trial(workload, pair_index=1).environment_status,
            "thermal_state": "fair",
            "thermal_state_raw_value": 1,
        },
    )

    def timeout_recovery(current_slot, recovery_index, deadline, trigger):
        raise ThermalRecoveryTimeout(ThermalRecoveryResult(7_200.0, 241))

    with pytest.raises(ThermalRecoveryTimeout):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda current_slot, current_attempt: fair_trial,
            preflight=lambda: (
                accepted.hardware,
                accepted.environment_status,
                accepted.software_versions,
            ),
            validate_preflight=lambda hardware, status, software: None,
            recover=timeout_recovery,
            validate_trial=lambda current_trial, allow_non_nominal_thermal: None,
            clock=lambda: 0.0,
            utc_now=lambda: "2026-08-05T00:00:00+00:00",
            progress=lambda message: None,
        )

    assert journal.load_accepted((first, second)) == {first: accepted}
    assert journal.rejected_path(second, 0).is_file()


def test_checkout_status_allows_only_bound_untracked_final_outputs():
    allowed = frozenset(
        {
            "v2/benchmarks/manifests/baseline-3687f8b.json",
            "v2/benchmarks/results/baseline-3687f8b.jsonl",
        }
    )
    validate_checkout_status(
        "?? v2/benchmarks/manifests/baseline-3687f8b.json\n"
        "?? v2/benchmarks/results/baseline-3687f8b.jsonl\n",
        allowed_untracked_paths=allowed,
    )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            "?? v2/benchmarks/manifests/baseline-3687f8b.json\n"
            "?? unexpected.txt\n",
            allowed_untracked_paths=allowed,
        )
    with pytest.raises(ValueError, match="checkout must be clean"):
        validate_checkout_status(
            " M v2/benchmarks/runner.py\n",
            allowed_untracked_paths=allowed,
        )
```

- [ ] **Step 2: Run orchestration tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'capture_ or checkout_status or runner_parser_accepts' -v
```

Expected: the capture API is missing and the parser accepts baseline capture without a state directory.

- [ ] **Step 3: Extract reusable strict single-trial validation**

Move the per-trial checks currently nested in `validate_baseline_manifest` into:

```python
def validate_baseline_trial(
    trial: RawTrial,
    *,
    workload: CanonicalWorkload,
    source_commit: str,
    harness_commit: str,
    harness_identity: str,
    expected_hardware: dict,
    expected_software_versions: dict[str, str],
    allow_non_nominal_thermal: bool = False,
) -> None:
```

Require reference side, `attempt_index == 0`, correct slot indices, commits, identities, canonical input/projection/order, native projection identity, canonical work units, startup verification, synchronization boundaries, `rope_scaling_factor == 1.0`, finite positive value, hardware, software, and all non-thermal environment requirements. Validate raw/string thermal consistency unconditionally. Skip only the equality check against required `thermal_state` when `allow_non_nominal_thermal=True`.

Replace the original nested checks with calls to this function so complete manifest validation and immediate journal validation cannot diverge.

- [ ] **Step 4: Implement the pure resumable capture loop**

Implement the exact orchestration boundary:

```python
def capture_baseline_trials(
    *,
    journal: BaselineJournal,
    slots: Sequence[BaselineSlot],
    launch_trial,
    preflight,
    validate_preflight,
    recover,
    validate_trial,
    clock=time.monotonic,
    utc_now=_utc_now_iso,
    progress=print,
) -> tuple[RawTrial, ...]:
```

At entry, load accepted and classify every complete in-flight attempt before launching work. For each missing slot, keep one `recovery_deadline` initialized to `None`. Before each launch, collect preflight, atomically persist a numbered preflight document with `utc_now()`, and only then call `validate_preflight(hardware, status, software)` to enforce identity, raw/string consistency, and every non-thermal requirement while permitting only a non-nominal thermal value. A preflight thermal violation sets `recovery_deadline = clock() + 7_200` once and calls `recover` with a `source="preflight"` trigger containing the exact persisted preflight document; a non-thermal violation raises only after its diagnostic is durable. Launch into `journal.next_attempt(slot).path`, parse the raw trial, and call `validate_trial(trial, allow_non_nominal_thermal=True)` before inspecting only `trial.environment_status["thermal_state"]`. Accept nominal trials; reject non-nominal trials with reason `non-nominal-thermal`, preserve the same deadline, and call `recover` with a `source="rejected-trial"` trigger containing the rejected document identity before retrying the same slot. Increment the preflight index on every check and the recovery index on every recovery episode. Print deterministic progress messages after every transition.

Return accepted trials in the supplied slot order only after every slot is present. Do not pass measured `value` to any branch condition.

- [ ] **Step 5: Integrate session creation, recovery logs, and clean source worktree**

Implement `validate_checkout_status` by parsing each porcelain line, accepting only `?? <exact-relative-path>` entries present in `allowed_untracked_paths`, and rejecting every tracked change, rename, deletion, ignored mismatch, or additional untracked path. Keep `_require_clean_checkout` behavior unchanged for all other commands.

In `_record_baseline`:

1. require a clean harness and pinned immutable CLI protocol; when a valid existing session binds the exact requested final paths, allow only those two untracked paths so an interrupted final publication can resume, while rejecting every other tracked or untracked change;
2. resolve and validate `args.state_directory` outside the harness checkout;
3. collect initial hardware/software and exact thermal observation, then cache that complete tuple as the first preflight result so it is journaled rather than discarded;
4. create deterministic paired representations;
5. build or strictly resume `session.json`;
6. create the detached source worktree for the invocation and re-run `require_external_state_directory` against both harness and source checkout paths before measurement;
7. call `capture_baseline_trials` with a launch closure that always passes baseline `attempt_index=0` while using the journal attempt path as child output;
8. persist recovery samples and summaries through journal methods;
9. always remove the detached source worktree in `finally`; and
10. leave the external state directory intact on every exit.

Add the required parser option:

```python
baseline.add_argument(
    "--state-directory",
    type=Path,
    required=True,
    help="external durable journal directory for baseline resume and diagnostics",
)
```

- [ ] **Step 6: Add crash-safe final publication tests and implementation**

Add these tests:

```python
def _accepted_complete_journal(tmp_path):
    workload = build_canonical_workload()
    paired = _valid_paired_representations(workload)
    session = _session_document(tmp_path, paired_representations=paired)
    journal = BaselineJournal.open(tmp_path / "state", session)
    trials = _valid_baseline_trials(workload)
    for trial in trials:
        slot = BaselineSlot(trial.metric, trial.pair_index)
        attempt = journal.next_attempt(slot)
        atomic_write_json(attempt.path, trial.to_dict(), create_only=True)
        journal.accept_inflight(attempt, trial)
    return workload, paired, journal, trials


def test_final_publication_uses_exactly_the_45_accepted_trials(tmp_path):
    workload, paired, journal, trials = _accepted_complete_journal(tmp_path)
    manifest_path = tmp_path / "baseline.json"
    raw_path = tmp_path / "baseline.jsonl"

    manifest = publish_baseline_from_journal(
        journal=journal,
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        command="record-baseline --state-directory state",
        paired_representations=paired,
        manifest_path=manifest_path,
        raw_output_path=raw_path,
    )

    raw_trials = tuple(
        RawTrial.from_dict(json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    )
    validate_baseline_manifest(manifest, raw_trials)
    assert len(raw_trials) == 45
    assert [(trial.metric, trial.pair_index) for trial in raw_trials] == [
        (metric, pair_index)
        for metric in METRIC_NAMES
        for pair_index in range(5)
    ]
    completed = read_json_object(journal.completed_path, label="completion")
    assert completed["baseline_identity"] == manifest["identity"]
    assert completed["raw_trial_identities"] == [
        benchmark_runner._raw_trial_identity(trial) for trial in raw_trials
    ]


def test_final_publication_never_overwrites_different_existing_output(tmp_path):
    workload, paired, journal, trials = _accepted_complete_journal(tmp_path)
    raw_path = tmp_path / "baseline.jsonl"
    raw_path.write_text("different existing content\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="final artifact already exists with different content"
    ):
        publish_baseline_from_journal(
            journal=journal,
            trials=trials,
            workload=workload,
            workload_identity=canonical_workload_identity(workload),
            source_commit=trials[0].source_commit,
            harness_commit=trials[0].harness_commit,
            harness_identity=trials[0].harness_identity,
            command="record-baseline --state-directory state",
            paired_representations=paired,
            manifest_path=tmp_path / "baseline.json",
            raw_output_path=raw_path,
        )

    assert raw_path.read_text(encoding="utf-8") == "different existing content\n"
    assert not journal.completed_path.exists()
```

Before publication, require the journal session's paired representation metadata and resolved destination paths to equal the function arguments. Use journal atomic writers for runner output and implement publication as create-only or exact-content-idempotent. Write raw JSONL first, manifest second, and `completed.json` last with this body plus a structured identity:

```python
{
    "kind": "sml-baseline-journal-completion",
    "version": 1,
    "session_identity": journal.session["identity"],
    "baseline_identity": manifest["identity"],
    "manifest_path": str(args.manifest.resolve()),
    "raw_output_path": str(args.raw_output.resolve()),
    "raw_trial_identities": [_raw_trial_identity(trial) for trial in trials],
}
```

- [ ] **Step 7: Document operation and run focused tests**

Update `README.md` with the required external state directory, exact directory layout, 30-second samples, five-minute nominal window, two-hour deadline, resume compatibility checks, rejected diagnostic retention, and the rule that final outputs remain absent until complete validation.

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k 'baseline or journal or thermal or capture or publication' -v
```

Expected: all focused tests pass.

- [ ] **Step 8: Commit integrated resumable capture**

```bash
git add v2/benchmarks/runner.py v2/benchmarks/README.md \
  v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): resume thermally valid baseline trials"
```

### Task 6: Verify and Prepare the New Harness Commit

**Files:**
- Verify: `v2/benchmarks/journal.py`
- Verify: `v2/benchmarks/recovery.py`
- Verify: `v2/benchmarks/runner.py`
- Verify: `v2/benchmarks/workload.py`
- Verify: `v2/benchmarks/README.md`
- Verify: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: complete resumable baseline harness from Tasks 1-5
- Produces: a reviewed, clean harness commit suitable for a separate measurement checkout

- [ ] **Step 1: Run repository-required static verification**

```bash
uv run ruff check v2
uv run ruff format --check v2
git diff --check
git diff --exit-code -- uv.lock
```

Expected: Ruff and diff checks exit zero; `uv.lock` is byte-identical.

- [ ] **Step 2: Run the complete v2 test suite outside the sandbox**

```bash
uv run pytest v2/tests
```

Expected: every v2 test passes with no failures.

- [ ] **Step 3: Request a focused code review**

Use `superpowers:requesting-code-review` against the full change from `db0ce37` through `HEAD`. Require the reviewer to check performance-selection bias, journal crash windows, resume compatibility, thermal deadline semantics, exact final trial cardinality/order, clean-checkout isolation, and missing fail-closed tests.

- [ ] **Step 4: Address findings with red-green tests**

For each accepted correctness finding, add a focused test that fails on the current code, run it to confirm the reported failure, implement the smallest correction, and rerun the focused and full suites. Do not make production-only fixes.

- [ ] **Step 5: Re-run final verification after review fixes**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: all commands pass; status contains only the user's two pre-existing untracked refactor plan documents.

- [ ] **Step 6: Create a clean measurement checkout without starting the baseline**

Resolve the final harness commit, create a detached worktree under `/private/tmp`, and verify both identity and cleanliness:

```bash
git rev-parse HEAD
git worktree add --detach /private/tmp/sml-v2-baseline-resume-harness HEAD
git -C /private/tmp/sml-v2-baseline-resume-harness status --short
uv run python -m v2.benchmarks.runner record-baseline --help
```

Expected: the detached checkout is clean and help shows required `--state-directory`. Do not start the multi-hour baseline until the user explicitly requests it and the live AC/thermal/memory/workload gate passes.
