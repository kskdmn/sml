# V2 Post-Exit Memory Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make immediate parent post-exit memory pressure the authoritative v2 trial gate while retaining identity-bound child-start and child-end diagnostics, strict fail-closed validation, and crash-resumable baseline evidence.

**Architecture:** Add a focused evidence module that builds and validates immutable child-measurement and parent post-exit documents, derives the three-observation environment summary, and deterministically finalizes `RawTrial` schema version 2. The child publishes measurement evidence only; the parent samples memory first after process exit, publishes post-exit evidence, and finalizes the raw trial. Extend the baseline journal with create-only measurement and post-exit stages so resume can reject missing immediate evidence, reconstruct complete trials, or classify finalized trials without selecting by measured performance.

**Tech Stack:** Python 3.12.13, MLX/Metal, immutable dataclasses, canonical JSON with structured SHA-256 identities, `subprocess`, macOS `memory_pressure`, durable atomic files, pytest, Ruff, Git worktrees

## Global Constraints

- Preserve the full 12-layer, hidden-size-768 model and training sequence length 1,024.
- Preserve microbatch size 1, gradient accumulation 8, five warmups, 20 default measured units, all metric-specific measured counts, five baseline/screen pairs, and ten final pairs.
- Parent preflight and child-start memory pressure must be `normal`.
- Child-end memory pressure may be `normal` or `warning`; it must never be `critical` for accepted evidence.
- Immediate parent post-exit memory pressure must be `normal`.
- Sample post-exit memory pressure and free-memory percentage before slower hardware or software probes, without sleeping or cooling first.
- Keep thermal state `nominal` across child start, child end, and parent post-exit for accepted evidence.
- Preserve strict AC-power, automatic-power-mode, Low Power Mode, competing-GPU, hardware, software, source, protocol, checkout-cleanliness, and identity validation.
- A persistent post-exit warning or critical state rejects the current attempt, stops the invocation, preserves accepted slots, and allows only a later manual resume after a new clean preflight.
- A non-nominal thermal state retains the existing five-continuous-minute recovery window and two-hour invocation deadline.
- Do not clear MLX caches, shorten the workload, add an in-child cooldown, or retry persistent memory pressure automatically.
- `RawTrial` version 1, old workload identities, and old journals remain incompatible diagnostic evidence; do not migrate or modify them.
- In particular, do not modify `/private/tmp/sml-v2-baseline-short-state` or copy its five accepted version 1 slots.
- Do not edit top-level project files such as `pyproject.toml` or `uv.lock`.
- Add every new benchmark implementation module to `HARNESS_COMPONENTS` so its bytes affect harness content identity.
- Use `uv run` for Python commands. Run every pytest command outside the sandbox so MLX/Metal can access the Apple GPU.
- Do not start a baseline during implementation, verification, or detached-harness refresh.

---

## File Map

- Create `v2/benchmarks/evidence.py`: own child-measurement and post-exit document schemas, identities, observation validation, three-point environment merging, raw-trial finalization, and embedded-evidence verification.
- Modify `v2/benchmarks/schema.py`: expose strict shared trial-payload validation and require self-contained evidence in `RawTrial` schema version 2.
- Modify `v2/benchmarks/workload.py`: identity-bind the three-stage memory policy and include `evidence.py` in harness content identity.
- Modify `v2/benchmarks/runner.py`: split child measurement from parent finalization, sample post-exit memory first, classify strict environment outcomes, integrate comparisons, and require version 2 evidence everywhere.
- Modify `v2/benchmarks/journal.py`: add immutable measurement and post-exit paths, validate the staged topology, retain stage evidence, and replay interrupted transitions deterministically.
- Modify `v2/tests/unit/test_benchmark_analysis.py`: provide version 2 fixtures and prove schemas, identities, lifecycle ordering, crash recovery, persistent-memory resume, unchanged thermal recovery, and manifest/comparison rejection of old evidence.
- Modify `v2/benchmarks/README.md`: document the authoritative post-exit rule, journal stages, crash states, and manual resume behavior.
- Refresh `/private/tmp/sml-v2-baseline-resume-harness` only after all versioned code is committed and verified; preserve every external state directory.

### Task 1: Define the identity-bound evidence contract

**Files:**
- Create: `v2/benchmarks/evidence.py`
- Modify: `v2/benchmarks/schema.py:1-335`
- Modify: `v2/benchmarks/workload.py:19-30,635-647`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:1-90,617-720,1020-1210`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `structured_identity(domain: str, value: JsonValue) -> str`, `RawTrial`, `JsonValue`, and the existing exact raw-trial scalar validation rules.
- Produces: `TRIAL_PAYLOAD_FIELDS: frozenset[str]`, `validate_trial_payload(raw: Mapping[str, object]) -> dict[str, JsonValue]`, `build_child_trial_measurement(*, session_identity: str, journal_attempt_index: int, trial: Mapping[str, object], start: Mapping[str, object], end: Mapping[str, object]) -> dict[str, JsonValue]`, `validate_child_trial_measurement(document: Mapping[str, object]) -> dict[str, JsonValue]`, `build_post_exit_observation(*, measurement: Mapping[str, object], observed_at_utc: str, hardware: Mapping[str, object], environment_status: Mapping[str, object], software_versions: Mapping[str, object]) -> dict[str, JsonValue]`, `validate_post_exit_observation(document: Mapping[str, object], *, measurement: Mapping[str, object]) -> dict[str, JsonValue]`, `merge_environment_status(start: dict, end: dict, post_exit: dict) -> dict[str, JsonValue]`, `finalize_raw_trial(measurement: Mapping[str, object], post_exit: Mapping[str, object]) -> RawTrial`, and `validate_raw_trial_evidence(trial: RawTrial) -> None`.
- Produces: `RawTrial.schema_version == 2` with exact additional fields `evidence_session_identity`, `journal_attempt_index`, `child_measurement`, and `post_exit_observation`.

- [ ] **Step 1: Write failing canonical-policy and evidence-schema tests**

Add imports for the new evidence functions and add exact tests near the existing environment and raw-trial tests:

```python
from v2.benchmarks.evidence import (
    build_child_trial_measurement,
    build_post_exit_observation,
    finalize_raw_trial,
    validate_child_trial_measurement,
    validate_post_exit_observation,
    validate_raw_trial_evidence,
)


def test_canonical_workload_binds_the_post_exit_memory_policy():
    required = build_canonical_workload().required_environment

    assert required["memory_pressure"] == "normal"
    assert required["measurement_end_memory_pressure_allowed"] == [
        "normal",
        "warning",
    ]
    assert required["post_exit_memory_pressure"] == "normal"
    assert required["post_exit_evidence_required"] is True


def test_child_and_post_exit_documents_are_exactly_identity_bound():
    measurement = _valid_child_measurement(build_canonical_workload())
    post_exit = _valid_post_exit_observation(measurement)

    assert validate_child_trial_measurement(measurement) == measurement
    assert (
        validate_post_exit_observation(post_exit, measurement=measurement)
        == post_exit
    )

    changed = json.loads(json.dumps(post_exit))
    changed["environment_status"]["memory_free_percentage"] -= 1
    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_post_exit_observation(changed, measurement=measurement)


def test_raw_trial_v2_embeds_and_revalidates_both_evidence_documents():
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    post_exit = _valid_post_exit_observation(measurement)
    trial = finalize_raw_trial(measurement, post_exit)

    assert trial.schema_version == 2
    assert trial.child_measurement == measurement
    assert trial.post_exit_observation == post_exit
    validate_raw_trial_evidence(trial)

    version_one = trial.to_dict()
    version_one["schema_version"] = 1
    with pytest.raises(ValueError, match="schema version"):
        RawTrial.from_dict(version_one)
```

Define `_valid_observation`, `_valid_trial_payload`,
`_valid_child_measurement(workload, metric="prepared-data", pair_index=0,
session_identity="sha256:" + "9" * 64, journal_attempt_index=0)`, and
`_valid_post_exit_observation` directly above `_valid_raw_trial`. Use fixed
UTC timestamps `2026-08-08T00:00:00+00:00`,
`2026-08-08T00:00:01+00:00`, and
`2026-08-08T00:00:02+00:00`; use the existing valid hardware, software,
environment, model, identity, timing, and unit-count values. Make
`_valid_raw_trial` call `finalize_raw_trial` so every later test fixture is a
self-contained version 2 trial.

Add one fixture-rebuild helper for every test that needs a different valid raw
payload. It must re-sign both documents rather than using `dataclasses.replace`
on an evidence-bound field:

```python
def _with_trial_payload(trial, **changes):
    measurement = json.loads(json.dumps(trial.child_measurement))
    measurement["trial"].update(changes)
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    post_exit = json.loads(json.dumps(trial.post_exit_observation))
    post_exit["metric"] = measurement["trial"]["metric"]
    post_exit["pair_index"] = measurement["trial"]["pair_index"]
    post_exit["child_measurement_identity"] = measurement["identity"]
    post_exit_body = {
        key: value for key, value in post_exit.items() if key != "identity"
    }
    post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", post_exit_body
    )
    return finalize_raw_trial(measurement, post_exit)
```

Use `_with_trial_payload` for valid comparison sides, pair/order/attempt
changes, source commits, native configurations, measured values, and valid
identity variants. Retain direct `replace` or dictionary mutation only in tests
whose purpose is to prove that cached top-level data cannot disagree with the
embedded evidence.

- [ ] **Step 2: Run the focused tests and verify the missing contract fails**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_binds_the_post_exit_memory_policy \
  v2/tests/unit/test_benchmark_analysis.py::test_child_and_post_exit_documents_are_exactly_identity_bound \
  v2/tests/unit/test_benchmark_analysis.py::test_raw_trial_v2_embeds_and_revalidates_both_evidence_documents \
  -v
```

Expected: FAIL during import because `v2.benchmarks.evidence` does not exist.

- [ ] **Step 3: Implement strict shared payload and document schemas**

In `schema.py`, extract the existing raw measurement fields into this exact
shared set and make both the evidence builder and `RawTrial.from_dict` use one
validator for their types and ranges:

```python
TRIAL_PAYLOAD_FIELDS = frozenset(
    {
        "metric",
        "side",
        "attempt_index",
        "pair_index",
        "process_order",
        "source_commit",
        "source_clean",
        "harness_commit",
        "harness_clean",
        "harness_identity",
        "canonical_workload_identity",
        "native_configuration",
        "native_representation_identity",
        "canonical_row_identity",
        "canonical_input_identity",
        "canonical_projection",
        "execution_order_identity",
        "initial_parameter_identity",
        "comparison_target",
        "warmup_units",
        "measured_units",
        "elapsed_seconds",
        "value",
        "startup_verification_seconds",
        "compilation_seconds",
        "peak_memory_bytes",
        "synchronization_boundaries",
    }
)


```

Move the existing concrete metric, side, integer, clean-checkout, Git SHA,
structured-identity, finite-number, mapping, and boundary-list checks from
`RawTrial.from_dict` into `validate_trial_payload`. The helper must return a
normalized dictionary whose `elapsed_seconds`, `value`, and non-null optional
second fields are floats and whose `synchronization_boundaries` is a JSON list.
Reject booleans for every integer and real-number field exactly as the current
schema does. `RawTrial.from_dict` must call the helper with this exact extraction
instead of keeping a second validation path:

```python
payload = validate_trial_payload(
    {name: raw[name] for name in TRIAL_PAYLOAD_FIELDS}
)
```

Add the four version 2 fields to `RawTrial`, require schema version 2, validate
the session identity and journal attempt index, and require both embedded
documents to be objects:

```python
    evidence_session_identity: str
    journal_attempt_index: int
    child_measurement: dict[str, JsonValue]
    post_exit_observation: dict[str, JsonValue]
```

Create `evidence.py` with exact document bodies and identity domains:

```python
CHILD_KIND = "sml-child-trial-measurement"
CHILD_IDENTITY_DOMAIN = "sml-child-trial-measurement-v1"
POST_EXIT_KIND = "sml-parent-post-exit-observation"
POST_EXIT_IDENTITY_DOMAIN = "sml-parent-post-exit-observation-v1"


def _identity_document(kind: str, domain: str, body: dict[str, JsonValue]) -> dict:
    return {**body, "identity": structured_identity(domain, body)}


def build_child_trial_measurement(
    *,
    session_identity: str,
    journal_attempt_index: int,
    trial: Mapping[str, object],
    start: Mapping[str, object],
    end: Mapping[str, object],
) -> dict[str, JsonValue]:
    body = {
        "kind": CHILD_KIND,
        "version": 1,
        "session_identity": session_identity,
        "journal_attempt_index": journal_attempt_index,
        "trial": validate_trial_payload(trial),
        "start": _validate_observation(start, label="child-start observation"),
        "end": _validate_observation(end, label="child-end observation"),
    }
    return _identity_document(CHILD_KIND, CHILD_IDENTITY_DOMAIN, body)


def build_post_exit_observation(
    *,
    measurement: Mapping[str, object],
    observed_at_utc: str,
    hardware: Mapping[str, object],
    environment_status: Mapping[str, object],
    software_versions: Mapping[str, object],
) -> dict[str, JsonValue]:
    child = validate_child_trial_measurement(measurement)
    trial = child["trial"]
    observation = _validate_observation(
        {
            "observed_at_utc": observed_at_utc,
            "hardware": dict(hardware),
            "environment_status": dict(environment_status),
            "software_versions": dict(software_versions),
        },
        label="parent post-exit observation",
    )
    body = {
        "kind": POST_EXIT_KIND,
        "version": 1,
        "session_identity": child["session_identity"],
        "journal_attempt_index": child["journal_attempt_index"],
        "metric": trial["metric"],
        "pair_index": trial["pair_index"],
        "child_measurement_identity": child["identity"],
        **observation,
    }
    return _identity_document(POST_EXIT_KIND, POST_EXIT_IDENTITY_DOMAIN, body)
```

`validate_child_trial_measurement` must require exactly
`kind`, `version`, `session_identity`, `journal_attempt_index`, `trial`, `start`,
`end`, and `identity`; recompute the identity; validate the trial payload and
both observations; and return the normalized exact document.
`validate_post_exit_observation` must require exactly `kind`, `version`,
`session_identity`, `journal_attempt_index`, `metric`, `pair_index`,
`child_measurement_identity`, `observed_at_utc`, `hardware`,
`environment_status`, `software_versions`, and `identity`; recompute its
identity; then require all session, attempt, metric, pair, and child identity
fields to match the supplied validated child measurement.

The private observation validator must require exactly
`observed_at_utc`, `hardware`, `environment_status`, and `software_versions`.
The nested environment object must have exactly this field set:

```python
ENVIRONMENT_FIELDS = frozenset(
    {
        "power_connected",
        "power_mode",
        "low_power_mode",
        "thermal_state",
        "thermal_state_raw_value",
        "memory_pressure",
        "memory_free_percentage",
        "competing_gpu_workload",
    }
)
```

Require `power_connected`, `low_power_mode`, and `competing_gpu_workload` to be
actual booleans; require non-empty string `power_mode`; require thermal pairs
to be exactly one of `nominal/0`, `fair/1`, `serious/2`, or `critical/3`;
require `memory_pressure` to be one of `normal`, `warning`, or `critical`; and
require `memory_free_percentage` to be a non-boolean integer from 0 through 100.
Require a parseable timezone-aware UTC ISO timestamp whose offset is zero, and
require hardware and software to be non-empty objects with non-empty string
software values. Validate `session_identity`, document identities, and child
measurement identities against `sha256:[0-9a-f]{64}`.

- [ ] **Step 4: Implement deterministic three-observation finalization**

Use the following exact summary rules in `evidence.py`:

```python
def merge_environment_status(start: dict, end: dict, post_exit: dict) -> dict:
    observations = (start, end, post_exit)
    pressure_order = {"normal": 0, "warning": 1, "critical": 2}
    thermal_raw = max(item["thermal_state_raw_value"] for item in observations)
    measurement_pressure = max(
        (start["memory_pressure"], end["memory_pressure"]),
        key=pressure_order.__getitem__,
    )
    power_modes = {item["power_mode"] for item in observations}
    return {
        "power_connected": all(item["power_connected"] for item in observations),
        "power_mode": power_modes.pop() if len(power_modes) == 1 else "changed",
        "low_power_mode": any(item["low_power_mode"] for item in observations),
        "thermal_state": decode_thermal_state(thermal_raw),
        "thermal_state_raw_value": thermal_raw,
        "memory_pressure": post_exit["memory_pressure"],
        "memory_free_percentage": post_exit["memory_free_percentage"],
        "measurement_memory_pressure": measurement_pressure,
        "measurement_min_free_percentage": min(
            start["memory_free_percentage"], end["memory_free_percentage"]
        ),
        "competing_gpu_workload": any(
            item["competing_gpu_workload"] for item in observations
        ),
        "start": dict(start),
        "end": dict(end),
        "post_exit": dict(post_exit),
    }
```

Avoid a runner/evidence import cycle by placing a private raw-thermal decoder in
`evidence.py` with the exact mapping `{0: "nominal", 1: "fair", 2: "serious",
3: "critical"}` and validating that the string and raw value agree.

`finalize_raw_trial` must validate both documents, require matching session,
journal attempt, metric, pair, and measurement identities, require start/end/
post-exit hardware equality, require start/end/post-exit software equality, and
then construct `RawTrial` schema version 2 from the exact child trial payload
plus the embedded documents and derived environment. Implement
`validate_raw_trial_evidence` as an exact deterministic round trip:

```python
def validate_raw_trial_evidence(trial: RawTrial) -> None:
    expected = finalize_raw_trial(
        trial.child_measurement,
        trial.post_exit_observation,
    )
    if expected != trial:
        raise ValueError("raw trial does not match its embedded evidence")
```

- [ ] **Step 5: Bind the memory policy and new module into harness identity**

Add `Path("v2/benchmarks/evidence.py")` immediately after `schema.py` in
`HARNESS_COMPONENTS`. Extend `required_environment` exactly:

```python
            "memory_pressure": "normal",
            "measurement_end_memory_pressure_allowed": ["normal", "warning"],
            "post_exit_memory_pressure": "normal",
            "post_exit_evidence_required": True,
```

- [ ] **Step 6: Run the focused evidence and schema tests**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_binds_the_post_exit_memory_policy \
  v2/tests/unit/test_benchmark_analysis.py::test_child_and_post_exit_documents_are_exactly_identity_bound \
  v2/tests/unit/test_benchmark_analysis.py::test_raw_trial_v2_embeds_and_revalidates_both_evidence_documents \
  -v
```

Expected: PASS. Task 1 intentionally does not run the full benchmark module:
the child producer remains version 1 until Task 3. Update direct version 1
fixture literals in the focused evidence/schema tests to version 2 only when
they represent valid current evidence; retain an explicit version 1 negative
fixture. Task 3 owns the first required complete benchmark-module pass.

- [ ] **Step 7: Commit the evidence contract**

```bash
git add v2/benchmarks/evidence.py v2/benchmarks/schema.py v2/benchmarks/workload.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): define post-exit trial evidence"
```

### Task 2: Enforce the three-observation acceptance policy

**Files:**
- Modify: `v2/benchmarks/runner.py:206-292,498-541,820-970`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:617-720,1270-1435`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `validate_raw_trial_evidence(trial) -> None` and the nested `start`, `end`, and `post_exit` status records produced by Task 1.
- Produces: `TrialEnvironmentDisposition(outcome, reason)`, `classify_trial_environment(workload, trial) -> TrialEnvironmentDisposition`, `_validate_observation_power(status, required, *, label) -> None`, and `_validate_acceptance_environment(workload, trial, *, allow_rejected_environment=False) -> None`.
- Produces exact outcomes `accept`, `thermal-reject`, and `memory-reject`, with reasons `non-normal-start-memory-pressure`, `critical-measurement-memory-pressure`, `persistent-post-exit-memory-pressure`, and `non-nominal-thermal`.

- [ ] **Step 1: Write failing policy-matrix tests**

Add an evidence-preserving helper that rebuilds a trial instead of mutating only
the cached summary:

```python
def _with_environment_observations(trial, *, start=None, end=None, post_exit=None):
    measurement = json.loads(json.dumps(trial.child_measurement))
    parent = json.loads(json.dumps(trial.post_exit_observation))
    if start is not None:
        measurement["start"]["environment_status"] = start
    if end is not None:
        measurement["end"]["environment_status"] = end
    measurement_body = {key: value for key, value in measurement.items() if key != "identity"}
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    parent["child_measurement_identity"] = measurement["identity"]
    if post_exit is not None:
        parent["environment_status"] = post_exit
    parent_body = {key: value for key, value in parent.items() if key != "identity"}
    parent["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", parent_body
    )
    return finalize_raw_trial(measurement, parent)
```

Then add this exact matrix:

```python
@pytest.mark.parametrize(
    ("endpoint", "pressure", "outcome", "reason"),
    (
        ("end", "warning", "accept", None),
        (
            "start",
            "warning",
            "memory-reject",
            "non-normal-start-memory-pressure",
        ),
        (
            "end",
            "critical",
            "memory-reject",
            "critical-measurement-memory-pressure",
        ),
        (
            "post_exit",
            "warning",
            "memory-reject",
            "persistent-post-exit-memory-pressure",
        ),
        (
            "post_exit",
            "critical",
            "memory-reject",
            "persistent-post-exit-memory-pressure",
        ),
    ),
)
def test_trial_memory_disposition_uses_post_exit_as_authority(
    endpoint, pressure, outcome, reason
):
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    statuses = {
        name: dict(trial.environment_status[name])
        for name in ("start", "end", "post_exit")
    }
    statuses[endpoint]["memory_pressure"] = pressure
    changed = _with_environment_observations(trial, **statuses)

    disposition = classify_trial_environment(workload, changed)

    assert (disposition.outcome, disposition.reason) == (outcome, reason)
```

Add separate tests proving non-nominal thermal across each endpoint returns
`thermal-reject`, and that AC disconnection, changed power mode, Low Power Mode,
competing workload, hardware mismatch, software mismatch, and mismatched
embedded identities raise even when `allow_rejected_environment=True`.

- [ ] **Step 2: Run the policy tests and verify the classifier is absent**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_trial_memory_disposition_uses_post_exit_as_authority \
  -v
```

Expected: FAIL during import because `classify_trial_environment` is absent.

- [ ] **Step 3: Split strict validation from retryable disposition**

Add this immutable result and ordered classifier to `runner.py`:

```python
@dataclass(frozen=True, slots=True)
class TrialEnvironmentDisposition:
    outcome: Literal["accept", "thermal-reject", "memory-reject"]
    reason: str | None


def classify_trial_environment(
    workload: CanonicalWorkload, trial: RawTrial
) -> TrialEnvironmentDisposition:
    required = workload.required_environment
    status = trial.environment_status
    if status["start"]["memory_pressure"] != required["memory_pressure"]:
        return TrialEnvironmentDisposition(
            "memory-reject", "non-normal-start-memory-pressure"
        )
    if status["end"]["memory_pressure"] not in required[
        "measurement_end_memory_pressure_allowed"
    ]:
        return TrialEnvironmentDisposition(
            "memory-reject", "critical-measurement-memory-pressure"
        )
    if status["post_exit"]["memory_pressure"] != required[
        "post_exit_memory_pressure"
    ]:
        return TrialEnvironmentDisposition(
            "memory-reject", "persistent-post-exit-memory-pressure"
        )
    if any(
        status[name]["thermal_state"] != required["thermal_state"]
        for name in ("start", "end", "post_exit")
    ):
        return TrialEnvironmentDisposition("thermal-reject", "non-nominal-thermal")
    return TrialEnvironmentDisposition("accept", None)
```

Replace `_validate_nonthermal_environment_status` with two explicit validators:
one for power/mode/competing fields at every observation, and one preflight
validator that additionally requires `memory_pressure == "normal"`. In
`_validate_acceptance_environment`, first call `validate_raw_trial_evidence`,
validate hardware/software equality and requirements, validate thermal
string/raw consistency for all three observations, and validate strict power
fields for all three observations. When `allow_rejected_environment` is false,
require classifier outcome `accept`; when true, allow only the classifier's
three known outcomes.

Rename the `validate_baseline_trial` keyword
`allow_non_nominal_thermal=False` to `allow_rejected_environment=False` and propagate
the keyword through tests and callers. Comparison and predecessor validation
continue using the default `False`, so any memory or thermal rejection stops a
comparison.

- [ ] **Step 4: Run focused policy and external-validator tests**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_trial_memory_disposition_uses_post_exit_as_authority \
  v2/tests/unit/test_benchmark_analysis.py -k 'environment or baseline_manifest or comparison' \
  -v
```

Expected: PASS, including the pre-existing strict power, thermal, manifest, and
comparison tests after converting their mutation helpers to rebuild embedded
evidence.

- [ ] **Step 5: Commit the acceptance policy**

```bash
git add v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): validate post-exit memory authority"
```

### Task 3: Split child measurement from parent finalization

**Files:**
- Modify: `v2/benchmarks/runner.py:1665-1880,1977-2038,2890-2932,3501-3518`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:3640-3810`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: Task 1 evidence builders and finalizer plus create-only `atomic_write_json`.
- Produces: `collect_environment(*, memory_sample: tuple[str, int] | None = None)`, `collect_post_exit_environment() -> tuple[dict, dict, dict]`, child CLI arguments `--evidence-session-identity`, `--journal-attempt-index`, and `--measurement-output`, and `_launch_trial` with required `evidence_session_identity: str`, `journal_attempt_index: int`, `measurement_output: Path`, `post_exit_output: Path`, and `output: Path` keyword arguments returning `RawTrial`.
- Produces: `_comparison_evidence_session_identity(*, harness_commit: str, harness_identity: str, reference_commit: str, candidate_commit: str, comparison_target: str, attempt_index: int, metrics: Sequence[MetricName], pairs: int, warmup: int, measure: int) -> str`, shared by every trial in one exact paired target/attempt.

- [ ] **Step 1: Replace child-output tests with lifecycle tests**

Change `_single_process_arguments` so `output` becomes
`measurement_output`, and add `evidence_session_identity="sha256:" + "9" * 64`
and `journal_attempt_index=0`. Update the child tests to parse a validated child
measurement rather than a raw trial.

Add this argument helper next to `_single_process_arguments`:

```python
def _launch_trial_arguments(tmp_path):
    return {
        "harness_root": tmp_path / "harness",
        "source_root": tmp_path / "source",
        "source_commit": "3687f8b3214a44c675ae67af52e4997762f6c634",
        "harness_commit": "a" * 40,
        "harness_identity": "sha256:" + "b" * 64,
        "adapter": "legacy",
        "metric": "prepared-data",
        "side": "reference",
        "attempt_index": 0,
        "pair_index": 0,
        "order": 0,
        "warmup": 5,
        "measure": 20,
        "comparison_target": "baseline",
    }
```

Add a parent test that stubs subprocess publication, records probe order, and
uses three distinct paths:

```python
def test_parent_samples_memory_first_after_child_exit_and_finalizes_trial(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    measurement_path = tmp_path / "measurement.json"
    post_exit_path = tmp_path / "post-exit.json"
    trial_path = tmp_path / "trial.json"
    events = []

    def run_child(command, *, cwd, check):
        events.append("child-exited")
        atomic_write_json(measurement_path, measurement, create_only=True)

    monkeypatch.setattr(benchmark_runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        benchmark_runner,
        "_memory_pressure",
        lambda: events.append("memory") or ("normal", 69),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda *, memory_sample=None: (
            events.append(("environment", memory_sample))
            or (
                measurement["start"]["hardware"],
                {
                    **measurement["start"]["environment_status"],
                    "memory_pressure": memory_sample[0],
                    "memory_free_percentage": memory_sample[1],
                },
                measurement["start"]["software_versions"],
            )
        ),
    )

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=measurement_path,
        post_exit_output=post_exit_path,
        output=trial_path,
    )

    assert events == ["child-exited", "memory", ("environment", ("normal", 69))]
    assert post_exit_path.is_file()
    assert trial_path.is_file()
    assert trial == RawTrial.from_dict(read_json_object(trial_path, label="trial"))
    assert trial.environment_status["memory_pressure"] == "normal"
```

Also add tests that each of the three create-only paths refuses overwrite and
that a nonzero child exit does not create a post-exit or final raw document.
Add `test_paired_trials_stops_before_the_next_process_on_persistent_memory`:
stub the first `_launch_trial` result with post-exit `warning`, call
`_run_paired_trials` for two pairs, assert strict environment validation raises
`persistent-post-exit-memory-pressure`, and assert the launch stub was called
exactly once.

- [ ] **Step 2: Run lifecycle tests and verify the old child-finalizes behavior fails**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_parent_samples_memory_first_after_child_exit_and_finalizes_trial \
  v2/tests/unit/test_benchmark_analysis.py -k 'child_process or child_output' \
  -v
```

Expected: FAIL because `_launch_trial` has only one output and the child still
writes `RawTrial` directly.

- [ ] **Step 3: Make post-exit memory collection the first parent probe**

Change the collector signature and add a dedicated wrapper:

```python
def collect_environment(
    *, memory_sample: tuple[str, int] | None = None
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    if memory_sample is None:
        memory_sample = _memory_pressure()
    pressure, free_percentage = memory_sample
    hardware_record = _system_profiler("SPHardwareDataType")
    display_record = _system_profiler("SPDisplaysDataType")


def collect_post_exit_environment() -> tuple[dict, dict, dict]:
    memory_sample = _memory_pressure()
    return collect_environment(memory_sample=memory_sample)
```

In the final code, keep ordinary preflight and child collection behavior
unchanged apart from accepting the optional sample. `collect_post_exit_environment`
must call `_memory_pressure` before `system_profiler`, `pmset`, thermal, process,
or package-version work.

- [ ] **Step 4: Make the child publish only its immutable measurement**

Replace the `RawTrial` construction in `_run_single_process` with exact start
and end observation documents and a child measurement:

```python
    start_hardware, start_status, start_versions = collect_environment()
    start_observation = {
        "observed_at_utc": _utc_now_iso(),
        "hardware": start_hardware,
        "environment_status": start_status,
        "software_versions": start_versions,
    }
    measurement = measure_native_process(
        adapter=adapter,
        metric=args.metric,
        native_workload=native,
        warmup_units=0 if args.metric == "compile-cold-start" else args.warmup,
        measured_units=measured_units,
        synchronize=mx.synchronize,
        peak_memory=mx.get_peak_memory,
        reset_peak_memory=mx.reset_peak_memory,
    )
    end_hardware, end_status, end_versions = collect_environment()
    end_observation = {
        "observed_at_utc": _utc_now_iso(),
        "hardware": end_hardware,
        "environment_status": end_status,
        "software_versions": end_versions,
    }
    document = build_child_trial_measurement(
        session_identity=args.evidence_session_identity,
        journal_attempt_index=args.journal_attempt_index,
        trial=trial_payload,
        start=start_observation,
        end=end_observation,
    )
    atomic_write_json(args.measurement_output, document, create_only=True)
```

Build `trial_payload` from the exact current raw measured fields. Do not merge
or discard unequal start/end hardware or software in the child; preserving both
documents lets parent validation diagnose and stop on the mismatch.

- [ ] **Step 5: Make the parent publish post-exit evidence and final raw evidence**

Extend `_launch_trial` with the interface above. Pass the new child CLI fields,
wait for `subprocess.run` to return, read and validate the measurement, call
`collect_post_exit_environment` immediately, build and create-only write the
post-exit document, call `finalize_raw_trial`, create-only write the final raw
trial, and return the exact persisted `RawTrial`.

For comparisons, derive one target/attempt session identity with this content:

```python
def _comparison_evidence_session_identity(
    *,
    harness_commit: str,
    harness_identity: str,
    reference_commit: str,
    candidate_commit: str,
    comparison_target: str,
    attempt_index: int,
    metrics: Sequence[MetricName],
    pairs: int,
    warmup: int,
    measure: int,
) -> str:
    return structured_identity(
        "sml-comparison-evidence-session-v1",
        {
            "harness_commit": harness_commit,
            "harness_identity": harness_identity,
            "reference_commit": reference_commit,
            "candidate_commit": candidate_commit,
            "comparison_target": comparison_target,
            "attempt_index": attempt_index,
            "metrics": list(metrics),
            "pairs": pairs,
            "warmup": warmup,
            "measure": measure,
        },
    )
```

Use filenames ending in `.measurement.json`, `.post-exit.json`, and
`.trial.json` for each temporary comparison process. Pass the baseline journal
session identity for baseline trials.

In `_run_paired_trials`, validate each returned trial before appending it or
launching the next side:

```python
stem = f"{metric}-{comparison_target[-12:]}-{pair_index}-{side}"
measurement_output = output_directory / f"{stem}.measurement.json"
post_exit_output = output_directory / f"{stem}.post-exit.json"
trial_output = output_directory / f"{stem}.trial.json"
trial = _launch_trial(
    harness_root=harness_root,
    source_root=reference_root if is_reference else candidate_root,
    source_commit=reference_commit if is_reference else candidate_commit,
    harness_commit=harness_commit,
    harness_identity=harness_identity,
    adapter=reference_adapter if is_reference else "replacement",
    metric=metric,
    side=side,
    attempt_index=attempt_index,
    pair_index=pair_index,
    order=order_index,
    warmup=warmup,
    measure=measure,
    comparison_target=comparison_target,
    evidence_session_identity=evidence_session_identity,
    journal_attempt_index=attempt_index,
    measurement_output=measurement_output,
    post_exit_output=post_exit_output,
    output=trial_output,
)
_validate_acceptance_environment(workload, trial)
_validate_software_versions(workload, trial.software_versions)
trials.append(trial)
```

Define `workload = build_canonical_workload()` once per `_run_paired_trials`
call. This immediate check ensures a persistent memory state
stops a comparison before another benchmark process starts; the later pair and
report validators remain unchanged defense in depth.

- [ ] **Step 6: Run the child, launcher, and comparison lifecycle tests**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py -k \
  'child_process or child_output or parent_samples_memory or paired_trials or comparison' \
  -v
uv run pytest v2/tests/unit/test_benchmark_analysis.py -q
```

Expected: PASS. The test event sequence must prove the parent memory sample
occurs after child exit and before the rest of parent environment collection.
The complete benchmark unit module must also pass now that every raw-trial
producer and fixture emits version 2 evidence.

- [ ] **Step 7: Commit the process lifecycle**

```bash
git add v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): finalize trials after child exit"
```

### Task 4: Add durable measurement and post-exit journal stages

**Files:**
- Modify: `v2/benchmarks/journal.py:35-50,290-335,510-565,756-780,900-1135,1256-1395`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:3070-3160,3800-4225`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `validate_child_trial_measurement`, `validate_post_exit_observation`, `validate_raw_trial_evidence`, and `finalize_raw_trial` from Task 1.
- Produces: `BaselineJournal.measurement_path(slot, index)`, `BaselineJournal.post_exit_path(slot, index)`, `JournalAttemptEvidence(attempt, measurement, post_exit, trial)`, `BaselineJournal.load_pending_attempts(expected_slots)`, `BaselineJournal.reject_unfinalized(attempt, measurement, *, reason)`, and version 2 rejected-outcome envelopes.
- Preserves: existing hard-linked accepted transition, finalized inflight transition, preflight, thermal-wait, completion, path-safety, symlink, lock, and atomic temporary semantics.

- [ ] **Step 1: Write failing staged-topology and crash-state tests**

Add helpers that persist exact staged evidence to the journal paths, then add:

```python
def test_journal_retains_identity_bound_measurement_and_post_exit_stages(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement = _valid_child_measurement(
        build_canonical_workload(),
        session_identity=journal.session["identity"],
        journal_attempt_index=attempt.journal_attempt_index,
    )
    post_exit = _valid_post_exit_observation(measurement)
    trial = finalize_raw_trial(measurement, post_exit)
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(attempt.path, trial.to_dict(), create_only=True)

    journal.accept_inflight(attempt, trial)

    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert not attempt.path.exists()
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_journal_reports_measurement_only_as_pending_immediate_evidence(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement = _valid_child_measurement(
        build_canonical_workload(),
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)

    pending = journal.load_pending_attempts((slot,))

    assert pending == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=None,
            trial=None,
        ),
    )


def test_journal_rejects_post_exit_without_its_exact_measurement(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    measurement = _valid_child_measurement(
        build_canonical_workload(),
        session_identity=journal.session["identity"],
        journal_attempt_index=0,
    )
    post_exit = _valid_post_exit_observation(measurement)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)

    with pytest.raises(ValueError, match="post-exit evidence has no measurement"):
        journal.load_pending_attempts((slot,))
```

Add cases for measurement plus post-exit without inflight, mismatched
measurement identity, inflight without stages, rejected-memory outcome without
a thermal trigger, accepted and rejected crash splits, attempt-index gaps,
unsupported paths, symlinks, and orphaned atomic temporaries in both new
directories.

- [ ] **Step 2: Run journal stage tests and verify the new paths fail**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_journal_retains_identity_bound_measurement_and_post_exit_stages \
  v2/tests/unit/test_benchmark_analysis.py::test_journal_reports_measurement_only_as_pending_immediate_evidence \
  v2/tests/unit/test_benchmark_analysis.py::test_journal_rejects_post_exit_without_its_exact_measurement \
  -v
```

Expected: FAIL because the journal does not recognize the new directories or
methods.

- [ ] **Step 3: Add exact paths and immutable document validation**

Add `"measurements"` and `"post-exit"` to `STATE_DIRECTORY_NAMES`, accept their
four-component attempt paths in `_is_journal_destination`, and add:

```python
def measurement_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
    self._require_slot(slot)
    _require_non_negative_index(journal_attempt_index, label="journal attempt index")
    return (
        self.root
        / "measurements"
        / slot.metric
        / str(slot.pair_index)
        / f"{journal_attempt_index}.json"
    )


def post_exit_path(self, slot: BaselineSlot, journal_attempt_index: int) -> Path:
    self._require_slot(slot)
    _require_non_negative_index(journal_attempt_index, label="journal attempt index")
    return (
        self.root
        / "post-exit"
        / slot.metric
        / str(slot.pair_index)
        / f"{journal_attempt_index}.json"
    )
```

Generalize `_attempt_records` so each category resolves only its exact canonical
path. Validate measurement session identity against `self.session["identity"]`,
metric/pair against the slot, and journal index against the filename. Validate
post-exit content only after loading its exact measurement.

- [ ] **Step 4: Model and validate every attempt topology**

Add:

```python
@dataclass(frozen=True, slots=True)
class JournalAttemptEvidence:
    attempt: JournalAttempt
    measurement: dict
    post_exit: dict | None
    trial: RawTrial | None
```

Rewrite `_validate_attempt_history` around the union of measurement, post-exit,
inflight, rejected, and accepted attempt keys. Enforce these exact implications:

```text
post-exit -> matching measurement
inflight -> matching post-exit -> matching measurement
accepted -> exact embedded stage documents and exact hard-linked inflight split when present
rejected -> exact recorded stage identities and exact inflight trial when trial is present
every recorded attempt index -> contiguous from zero within its slot
one attempt -> at most one accepted or rejected outcome
```

Return only attempts with staged evidence and no accepted or rejected outcome
from `load_pending_attempts`. Keep measurement and post-exit files after
accept/reject. Keep the current behavior that removes an identical leftover
inflight file after replaying an interrupted accepted or rejected transition.

- [ ] **Step 5: Advance rejected outcomes to version 2**

Use this exact body for all new rejection documents:

```python
body = {
    "kind": "sml-baseline-rejected-trial",
    "version": 2,
    "journal_attempt_index": attempt.journal_attempt_index,
    "reason": reason,
    "child_measurement_identity": measurement["identity"],
    "post_exit_observation_identity": (
        None if post_exit is None else post_exit["identity"]
    ),
    "trial": None if trial is None else trial.to_dict(),
}
document = {
    **body,
    "identity": structured_identity("sml-baseline-rejected-trial-v2", body),
}
```

`reject_unfinalized` accepts only measurement-only evidence and reason
`missing-immediate-post-exit-evidence`. `reject_inflight` requires all three
stages and embeds the exact raw trial. Both transitions are create-only,
idempotent only for identical bytes, and preserve stage files.

Add an `expected_version: int = 1` keyword to `_validate_identity_document`.
Use version 2 and identity domain `sml-baseline-rejected-trial-v2` only for the
new rejected-outcome envelope; keep session, preflight, thermal sample,
thermal trigger, thermal summary, and completion document versions unchanged.

- [ ] **Step 6: Run all journal, lock, and atomicity tests**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py -k \
  'journal or baseline_lock or output_lock or atomic or orphan' \
  -v
```

Expected: PASS, including the existing hard-link identity checks and every new
staged crash topology.

- [ ] **Step 7: Commit durable staged evidence**

```bash
git add v2/benchmarks/journal.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): journal staged trial evidence"
```

### Task 5: Classify, stop, and manually resume memory-rejected baseline trials

**Files:**
- Modify: `v2/benchmarks/runner.py:2038-2325,2640-2715`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:1980-2705`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `BaselineJournal.load_pending_attempts`, stage paths, `finalize_raw_trial`, `classify_trial_environment`, and `validate_baseline_trial` with `allow_rejected_environment=True`.
- Produces: `MemoryPressureTrialRejected(RuntimeError)` with `slot` and `reason`, reason-aware pending replay, deterministic reconstruction after both stage records, memory rejection with immediate stop, and later missing-slot resume.
- Preserves: thermal trigger/recovery semantics and performance-independent acceptance decisions.

- [ ] **Step 1: Write failing capture and manual-resume tests**

Add a helper that persists measurement, post-exit, and raw files for a supplied
`JournalAttempt`. Update existing capture callbacks to use this helper rather
than returning an unpersisted raw trial.

Use these exact test helpers:

```python
def _persist_journal_trial(journal, attempt, trial):
    measurement = json.loads(json.dumps(trial.child_measurement))
    measurement["session_identity"] = journal.session["identity"]
    measurement["journal_attempt_index"] = attempt.journal_attempt_index
    measurement_body = {
        key: value for key, value in measurement.items() if key != "identity"
    }
    measurement["identity"] = structured_identity(
        "sml-child-trial-measurement-v1", measurement_body
    )
    post_exit = json.loads(json.dumps(trial.post_exit_observation))
    post_exit["session_identity"] = journal.session["identity"]
    post_exit["journal_attempt_index"] = attempt.journal_attempt_index
    post_exit["child_measurement_identity"] = measurement["identity"]
    post_exit_body = {
        key: value for key, value in post_exit.items() if key != "identity"
    }
    post_exit["identity"] = structured_identity(
        "sml-parent-post-exit-observation-v1", post_exit_body
    )
    persisted = finalize_raw_trial(measurement, post_exit)
    atomic_write_json(
        journal.measurement_path(attempt.slot, attempt.journal_attempt_index),
        measurement,
        create_only=True,
    )
    atomic_write_json(
        journal.post_exit_path(attempt.slot, attempt.journal_attempt_index),
        post_exit,
        create_only=True,
    )
    atomic_write_json(attempt.path, persisted.to_dict(), create_only=True)
    return persisted


def _preflight_from_trial(trial):
    return trial.hardware, trial.environment_status["start"], trial.software_versions


def _baseline_validator(workload):
    expected = _valid_raw_trial(workload)

    def validate(trial, *, allow_rejected_environment):
        validate_baseline_trial(
            trial,
            workload=workload,
            source_commit=expected.source_commit,
            harness_commit=expected.harness_commit,
            harness_identity=expected.harness_identity,
            expected_hardware=expected.hardware,
            expected_software_versions=expected.software_versions,
            allow_rejected_environment=allow_rejected_environment,
        )

    return validate
```

Add the persistent-memory test as two explicit invocations:

```python
def test_persistent_memory_rejection_stops_then_manual_resume_fills_only_missing_slot(
    tmp_path,
):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    first = BaselineSlot("prepared-data", 0)
    second = BaselineSlot("prepared-data", 1)
    first_attempt = journal.next_attempt(first)
    accepted = _persist_journal_trial(
        journal,
        first_attempt,
        _valid_raw_trial(workload, pair_index=0),
    )
    journal.accept_inflight(first_attempt, accepted)
    normal = _valid_raw_trial(workload, pair_index=1)
    warning = dict(normal.environment_status["post_exit"])
    warning["memory_pressure"] = "warning"
    rejected = _with_environment_observations(normal, post_exit=warning)
    first_run_launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="persistent-post-exit-memory-pressure",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(first, second),
            launch_trial=lambda slot, attempt: (
                first_run_launches.append(slot)
                or _persist_journal_trial(journal, attempt, rejected)
            ),
            preflight=lambda: _preflight_from_trial(normal),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "memory rejection entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert first_run_launches == [second]
    assert journal.load_accepted((first, second)) == {first: accepted}
    assert read_json_object(
        journal.rejected_path(second, 0), label="memory rejection"
    )["reason"] == "persistent-post-exit-memory-pressure"

    second_run_launches = []
    trials = capture_baseline_trials(
        journal=journal,
        slots=(first, second),
        launch_trial=lambda slot, attempt: (
            second_run_launches.append(slot)
            or _persist_journal_trial(journal, attempt, normal)
        ),
        preflight=lambda: _preflight_from_trial(normal),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "memory rejection created a thermal trigger"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert second_run_launches == [second]
    assert trials[0] == accepted
    assert (trials[1].metric, trials[1].pair_index, trials[1].value) == (
        normal.metric,
        normal.pair_index,
        normal.value,
    )
```

Add parameterized variants for child-start warning and child-end critical, plus
a test that child-end warning with post-exit normal is accepted without retry.
Add crash tests for measurement-only rejection and for deterministic
measurement-plus-post-exit reconstruction before any replacement launch.
Add `classify_trial: Callable[[RawTrial], TrialEnvironmentDisposition]` to
`capture_baseline_trials` and supply it in every existing capture test. Use the
real workload-bound classifier for policy tests and a deterministic
`TrialEnvironmentDisposition("accept", None)` stub when a test concerns only
journal ordering; thermal fixtures must return the matching thermal
disposition.

- [ ] **Step 2: Run capture tests and verify the old thermal-only classifier fails**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_persistent_memory_rejection_stops_then_manual_resume_fills_only_missing_slot \
  v2/tests/unit/test_benchmark_analysis.py -k 'capture and memory' \
  -v
```

Expected: FAIL because capture has no memory disposition or staged replay.

- [ ] **Step 3: Replay staged evidence before preflight or launch**

At capture start, load accepted slots and `journal.load_pending_attempts`. For
each pending attempt in canonical slot/attempt order:

```python
if state.post_exit is None:
    journal.reject_unfinalized(
        state.attempt,
        state.measurement,
        reason="missing-immediate-post-exit-evidence",
    )
    continue
if state.trial is None:
    trial = finalize_raw_trial(state.measurement, state.post_exit)
    atomic_write_json(state.attempt.path, trial.to_dict(), create_only=True)
else:
    trial = state.trial
validate_trial(trial, allow_rejected_environment=True)
```

After deterministic finalization, pass the trial through the same disposition
and transition function used for a newly launched process. Never collect a new
post-exit observation for a process from an earlier invocation.

- [ ] **Step 4: Persist memory rejection before stopping**

Add:

```python
class MemoryPressureTrialRejected(RuntimeError):
    def __init__(self, slot: BaselineSlot, reason: str):
        self.slot = slot
        self.reason = reason
        super().__init__(f"{slot.metric} pair {slot.pair_index}: {reason}")
```

Use one classification helper for pending and new trials:

```python
disposition = classify_trial(trial)
if disposition.outcome == "accept":
    journal.accept_inflight(attempt, trial)
elif disposition.outcome == "thermal-reject":
    if recovery_deadline is None:
        recovery_deadline = clock() + 7_200.0
        recovery_deadlines[attempt.slot] = recovery_deadline
    journal.reject_inflight(attempt, trial, reason="non-nominal-thermal")
    rejected = read_json_object(
        journal.rejected_path(
            attempt.slot, attempt.journal_attempt_index
        ),
        label="rejected trial",
    )
    pending_trigger = {
        "source": "rejected-trial",
        "rejected_trial_identity": rejected["identity"],
    }
    progress(
        f"rejected {attempt.slot.metric} pair {attempt.slot.pair_index}: "
        f"thermal={trial.environment_status['thermal_state']} "
        f"raw={trial.environment_status['thermal_state_raw_value']}"
    )
else:
    if disposition.reason is None:
        raise AssertionError("memory rejection is missing its reason")
    journal.reject_inflight(attempt, trial, reason=disposition.reason)
    raise MemoryPressureTrialRejected(attempt.slot, disposition.reason)
```

Keep this as the single concrete thermal transition block; do not duplicate or
alter its five-minute or two-hour logic.

Change `_persisted_pending_thermal_triggers` to inspect `document["reason"]`
before parsing `document["trial"]`. Skip every reason except
`non-nominal-thermal`; require a complete non-nominal trial only for that exact
reason. This prevents memory and missing-evidence rejections from creating a
thermal trigger on manual resume.

- [ ] **Step 5: Wire baseline launch to all three journal paths**

In `_record_baseline_locked`, pass:

```python
evidence_session_identity=journal.session["identity"],
journal_attempt_index=attempt.journal_attempt_index,
measurement_output=journal.measurement_path(
    slot, attempt.journal_attempt_index
),
post_exit_output=journal.post_exit_path(
    slot, attempt.journal_attempt_index
),
output=attempt.path,
```

Pass the workload-bound classifier to `capture_baseline_trials` in the same
function:

```python
classify_trial=lambda trial: classify_trial_environment(workload, trial),
```

Remove the old fallback that accepted a callback return and wrote only
`attempt.path`; all accepted launches must have the exact staged files.

- [ ] **Step 6: Prove thermal behavior is unchanged**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py -k \
  'capture or thermal_recovery or persisted_thermal or recovery_deadline' \
  -v
```

Expected: PASS. Existing tests must still prove five continuous nominal minutes,
reset on regression, one two-hour deadline per slot/invocation, thermal retry of
only the rejected slot, and preservation after timeout.

- [ ] **Step 7: Commit baseline classification and resume**

```bash
git add v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): resume after durable memory rejection"
```

### Task 6: Require version 2 evidence in artifacts and document operations

**Files:**
- Modify: `v2/benchmarks/runner.py:80-90,95-205,293-497,1080-1140,2760-2885,3180-3235`
- Modify: `v2/benchmarks/README.md:7-120`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:360-390,1140-1905,2700-2895,4150-4330`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: self-contained `RawTrial` version 2 and the canonical memory policy.
- Produces: raw identity domain `sml-raw-benchmark-trial-v2`; baseline manifests, JSONL validation, comparison reports, predecessor validation, phase validation, and final validation that reject version 1 or tampered embedded evidence.
- Documents: exact journal layout, authoritative/diagnostic memory distinction, crash behavior, thermal behavior, and manual resume.

- [ ] **Step 1: Write artifact-level incompatibility tests**

Add tests that mutate one otherwise valid version 2 trial inside a baseline raw
JSONL and a comparison report:

```python
def test_baseline_rejects_raw_trial_without_valid_post_exit_evidence():
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload)
    raw = trial.to_dict()
    raw["post_exit_observation"]["child_measurement_identity"] = (
        "sha256:" + "0" * 64
    )

    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_baseline_trial(
            RawTrial.from_dict(raw),
            workload=workload,
            source_commit=trial.source_commit,
            harness_commit=trial.harness_commit,
            harness_identity=trial.harness_identity,
            expected_hardware=trial.hardware,
            expected_software_versions=trial.software_versions,
        )


def test_comparison_rejects_raw_trial_without_valid_post_exit_evidence():
    _workload, baseline, report = _valid_prepared_comparison()
    report["raw_trials"][0]["post_exit_observation"][
        "child_measurement_identity"
    ] = "sha256:" + "0" * 64
    body = {key: value for key, value in report.items() if key != "identity"}
    report["identity"] = structured_identity("sml-performance-comparison-v1", body)

    with pytest.raises(ValueError, match="post-exit observation identity"):
        validate_comparison_report(report, baseline, None)
```

Add an explicit old-journal session test by removing the three new memory-policy
keys from a copied canonical workload, recomputing that old session identity,
and proving `BaselineJournal.open(state, current_session)` raises
`session does not match expected session`. Do not read or mutate any real
`/private/tmp` state in unit tests.

- [ ] **Step 2: Run artifact tests and verify a tampered nested document passes too far**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_baseline_rejects_raw_trial_without_valid_post_exit_evidence \
  v2/tests/unit/test_benchmark_analysis.py::test_comparison_rejects_raw_trial_without_valid_post_exit_evidence \
  v2/tests/unit/test_benchmark_analysis.py -k 'old_journal or raw_trial_identity' \
  -v
```

Expected: FAIL until every artifact entry point invokes embedded-evidence
validation and the identity domain is version 2.

- [ ] **Step 3: Close every artifact and replay entry point**

Change `_raw_trial_identity` to:

```python
def _raw_trial_identity(trial: RawTrial) -> str:
    validate_raw_trial_evidence(trial)
    return structured_identity("sml-raw-benchmark-trial-v2", trial.to_dict())
```

Require `validate_raw_trial_evidence` before using values or identities in
baseline manifest construction, baseline raw JSONL validation, comparison
report construction/validation, predecessor replay, phase validation, and final
validation. Keep outer manifest and comparison report schema versions unchanged;
their canonical identities change naturally because their raw-trial identities
and canonical workload identity change.

Update the harness-content identity test to include `evidence.py` in the exact
ordered file list. Retain one explicit negative version 1 raw dictionary and
remove accidental version 1 literals from valid fixtures.

- [ ] **Step 4: Update the operator README**

Change the journal tree to include:

```text
├── measurements/<metric>/<pair-index>/<attempt-index>.json
├── post-exit/<metric>/<pair-index>/<attempt-index>.json
├── inflight/<metric>/<pair-index>/<attempt-index>.json
```

Document these exact rules:

- the child records start/end observations while MLX is alive and publishes a
  child measurement, not a final trial;
- the parent samples memory first immediately after child exit, then publishes
  identity-bound post-exit evidence and a finalized version 2 raw trial;
- child-end `warning` is diagnostic and may pass only with child-start and
  post-exit `normal` plus every other strict check;
- child `critical` or non-normal post-exit memory is durably rejected and stops
  the invocation without a same-run retry;
- later manual resume revalidates the journal, performs a new strict preflight,
  and retries only the missing slot;
- measurement-only crash evidence is rejected as missing immediate post-exit
  evidence, while matching measurement plus post-exit evidence reconstructs
  deterministically; and
- thermal recovery remains five continuous nominal minutes within the existing
  two-hour invocation deadline.

- [ ] **Step 5: Run the full benchmark unit module**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -q
```

Expected: PASS with no version 1 current-evidence fixtures and with all negative
compatibility cases retained.

- [ ] **Step 6: Commit artifact enforcement and documentation**

```bash
git add v2/benchmarks/runner.py v2/benchmarks/README.md v2/tests/unit/test_benchmark_analysis.py
git commit -m "docs(v2): document post-exit memory evidence"
```

### Task 7: Verify the complete v2 tree and refresh the detached harness

**Files:**
- Verify: `v2/benchmarks/evidence.py`
- Verify: `v2/benchmarks/schema.py`
- Verify: `v2/benchmarks/workload.py`
- Verify: `v2/benchmarks/runner.py`
- Verify: `v2/benchmarks/journal.py`
- Verify: `v2/tests/unit/test_benchmark_analysis.py`
- Verify: `v2/benchmarks/README.md`
- Refresh: `/private/tmp/sml-v2-baseline-resume-harness`
- Preserve: `/private/tmp/sml-v2-baseline-short-state`

**Interfaces:**
- Consumes: all prior tasks and the repository's v2 verification contract.
- Produces: a Ruff-clean, fully tested implementation commit and a clean detached measurement checkout whose commit and content identity are reported to the user.
- Does not produce: a baseline manifest, raw baseline JSONL, new baseline state directory, or modification to any retained journal.

- [ ] **Step 1: Run Ruff check and format verification**

```bash
uv run ruff check v2
uv run ruff format --check v2
```

Expected: both commands exit 0. If formatting is required, run
`uv run ruff format v2`, inspect the exact diff, rerun both checks, and commit
only the mechanical formatting changes with:

```bash
git add v2
git commit -m "style(v2): format post-exit evidence"
```

- [ ] **Step 2: Run the complete v2 test suite outside the sandbox**

Run outside the sandbox:

```bash
uv run pytest v2/tests
```

Expected: all tests pass with zero failures, errors, or skips introduced by this
change. Preserve the complete final pytest summary as completion evidence.

- [ ] **Step 3: Verify repository and retained-journal state before refresh**

```bash
git status --short
git log -1 --oneline
test -d /private/tmp/sml-v2-baseline-short-state
```

Expected: the versioned worktree is clean, HEAD is the verified implementation
commit, and the retained failed shorter-protocol state still exists.

- [ ] **Step 4: Refresh the detached measurement checkout at verified HEAD**

Use Git's worktree commands; do not delete any state directory:

```bash
git worktree remove --force /private/tmp/sml-v2-baseline-resume-harness
git worktree add --detach /private/tmp/sml-v2-baseline-resume-harness HEAD
git -C /private/tmp/sml-v2-baseline-resume-harness status --short
git -C /private/tmp/sml-v2-baseline-resume-harness rev-parse HEAD
```

Expected: detached status is empty and its full commit equals the verified main
HEAD. If the old checkout is already absent, omit only the failing remove
command and run the add command exactly once.

- [ ] **Step 5: Compute and report the new harness content identity**

Run from the repository:

```bash
uv run python -c 'from pathlib import Path; from v2.benchmarks.workload import harness_content_identity; print(harness_content_identity(Path("/private/tmp/sml-v2-baseline-resume-harness")))'
```

Expected: one `sha256:` identity. Report the implementation commit, harness
identity, Ruff results, full pytest count, and the fact that no baseline was
started. The next baseline must use a new empty state directory only after an
explicit user request and a passing live preflight.
