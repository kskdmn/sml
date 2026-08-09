# V2 Post-Exit Memory Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept a v2 benchmark trial only after an immediate post-exit warning is followed by 30 continuous seconds of identity-bound normal memory evidence inside a fixed five-minute recovery window.

**Architecture:** Keep the child measurement and immediate parent observation unchanged, then add immutable recovery-sample and recovery-summary evidence before schema-v3 raw-trial finalization. Put the injected-clock recovery state machine in the existing recovery module, orchestration in the runner, and create-only topology validation in the baseline journal. Every artifact embeds and revalidates the complete recovery chain, while persistent/critical memory stops for manual resume and thermal-only failure retains the existing automatic thermal recovery.

**Tech Stack:** Python 3.12.13, MLX/Metal, immutable dataclasses, canonical JSON with structured SHA-256 identities, monotonic clocks, macOS environment probes, durable atomic files, pytest, Ruff, Git worktrees

## Whole-review fix amendment

The deterministic-termination review advances recovery summaries to version 2
and identity domain `sml-parent-post-exit-recovery-v2`. Every summary requires
identity-bound `completion_source: live | crash-reconstruction`. Live summaries
match one shared first-terminal reducer used by runtime and evidence validation;
that reducer enforces minimum sample-start cadence, non-memory-before-critical
precedence, and no samples after its first terminal event. A deadline timeout
has duration exactly 300 seconds even when its last persisted sample precedes
the deadline.

Missing-summary reconstruction is provenance-based rather than purely
sample-derived. Immediate normal, critical, and non-memory failure with zero
samples reconstruct their matching outcomes. Immediate warning always
reconstructs `interrupted`, even if its final durable sample is the first live
critical, non-memory-failure, stable-completion, or deadline event; a later
sample is invalid. This amendment supersedes the older version-1 domain and
sample-derived `interrupted` snippets below. Raw trials remain version 3;
child, immediate, and recovery-sample documents remain version 1.

## Global Constraints

- Preserve the full 12-layer, hidden-size-768 model and training sequence length 1,024.
- Preserve microbatch size 1, gradient accumulation 8, five warmup units, 20 default measured units, all metric-specific measured counts, five baseline/screen pairs, ten final pairs, and every statistical gate.
- Keep the existing version-1 `ChildTrialMeasurement` and version-1 immediate `PostExitObservation` documents unchanged and immutable.
- Immediate post-exit memory pressure is eligible for acceptance or recovery only when it is `normal` or `warning`; `critical` is always rejected.
- After an immediate `warning`, sample the complete environment every 5 seconds for at most 300 seconds from the immediate memory-sample start.
- Require at least 30 monotonic seconds from the first valid `normal` recovery sample through the final valid `normal` sample; a later `warning` resets that stability window without extending the deadline.
- Sample memory pressure and free-memory percentage before slower environment probes at every immediate and recovery observation.
- Record every recovery sample before classification or collection of the next sample.
- Keep thermal state nominal and preserve strict AC power, automatic power mode, Low Power Mode, competing-GPU, hardware, software, source, protocol, checkout-cleanliness, and identity validation for accepted evidence.
- Never inspect metric values, elapsed time, throughput, compilation time, or peak memory when deciding recovery or trial disposition.
- Do not clear MLX caches, invoke garbage collection, change the child workload, add an in-child cooldown, or launch a second child during memory recovery.
- Persistent or critical memory pressure stops the invocation without a same-run retry. Thermal-only failure retains the existing five-continuous-minute automatic recovery and two-hour slot deadline.
- A missing-summary immediate-warning recovery creates an `interrupted` outcome on resume regardless of its partial samples and requires a new journal attempt; do not continue a stability window across the gap.
- `RawTrial` schema version 2, old workload identities, and old journals remain incompatible diagnostic evidence; do not migrate or modify them.
- Preserve `/private/tmp/sml-v2-baseline-post-exit-state-aa6bb43` with its seven accepted and three rejected schema-v2 attempts, and preserve `/private/tmp/sml-v2-baseline-short-state`.
- Do not edit top-level project files such as `pyproject.toml` or `uv.lock`.
- Keep the current benchmark file layout; `v2/benchmarks/recovery.py` already participates in `HARNESS_COMPONENTS`.
- Use `uv run` for Python commands. Run every pytest command outside the sandbox so MLX/Metal can access the Apple GPU.
- Do not start a replacement baseline during implementation, verification, or detached-harness refresh.

---

## File Map

- Modify `v2/benchmarks/schema.py`: advance `RawTrial` to schema version 3 and embed ordered recovery samples plus their terminal recovery summary.
- Modify `v2/benchmarks/evidence.py`: build and validate recovery documents, derive effective environment summaries across the complete chain, finalize schema-v3 raw trials, and derive exact rejection reasons.
- Modify `v2/benchmarks/workload.py`: identity-bind the fixed recovery policy without changing any workload or statistical count.
- Modify `v2/benchmarks/recovery.py`: own the injected-clock, fixed-cadence post-exit memory recovery state machine while preserving the thermal state machine.
- Modify `v2/benchmarks/runner.py`: collect memory-first recovery observations, persist every stage, integrate baseline/comparison launch paths, validate dispositions, and reconstruct interrupted attempts.
- Modify `v2/benchmarks/journal.py`: add create-only recovery-sample and recovery-summary paths, validate the complete attempt topology, and advance rejection envelopes to version 3.
- Modify `v2/tests/unit/test_benchmark_analysis.py`: update schema-v3 fixtures and prove policy identity, recovery timing, evidence tamper resistance, lifecycle ordering, crash states, classification, compatibility, and artifact rejection.
- Modify `v2/benchmarks/README.md`: document recovery authority, timing, journal paths, crash handling, retry behavior, and schema-v3 compatibility.
- Refresh `/private/tmp/sml-v2-baseline-resume-harness` only after all versioned changes pass final verification; do not touch either retained state directory.

### Task 1: Define the canonical recovery evidence and schema-v3 contract

**Files:**
- Modify: `v2/benchmarks/workload.py:71-72,638-652`
- Modify: `v2/benchmarks/schema.py:338-444`
- Modify: `v2/benchmarks/evidence.py:10-417`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:642-651,1019-1263`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: unchanged `build_child_trial_measurement`, `validate_child_trial_measurement`, `build_post_exit_observation`, `validate_post_exit_observation`, `structured_identity`, and strict observation validation.
- Produces: `RECOVERY_SAMPLE_KIND`, `RECOVERY_SAMPLE_IDENTITY_DOMAIN`, `RECOVERY_KIND`, `RECOVERY_IDENTITY_DOMAIN`, and exact recovery policy fields in `CanonicalWorkload.required_environment`.
- Produces: `post_exit_recovery_policy(workload: CanonicalWorkload) -> dict[str, JsonValue]`, containing the four recovery values, required thermal/power/competing-workload values, and `require_same_hardware_and_software: true`; actual hardware and software references come from the bound child-start observation.
- Produces: `build_post_exit_recovery_sample(*, measurement: Mapping[str, object], post_exit: Mapping[str, object], sample_index: int, previous_sample_identity: str | None, observed_at_utc: str, elapsed_seconds: float, hardware: Mapping[str, object], environment_status: Mapping[str, object], software_versions: Mapping[str, object]) -> dict[str, JsonValue]`.
- Produces: `validate_post_exit_recovery_sample(document: Mapping[str, object], *, measurement: Mapping[str, object], post_exit: Mapping[str, object], previous_sample: Mapping[str, object] | None) -> dict[str, JsonValue]` and `validate_post_exit_recovery_samples(documents: Sequence[Mapping[str, object]], *, measurement: Mapping[str, object], post_exit: Mapping[str, object]) -> tuple[dict[str, JsonValue], ...]`.
- Produces: `build_post_exit_recovery(*, measurement: Mapping[str, object], post_exit: Mapping[str, object], samples: Sequence[Mapping[str, object]], policy: Mapping[str, object], outcome: str, duration_seconds: float, failure_fields: Sequence[str] = (), completion_source: str = "live") -> dict[str, JsonValue]` and `validate_post_exit_recovery(document: Mapping[str, object], *, measurement: Mapping[str, object], post_exit: Mapping[str, object], samples: Sequence[Mapping[str, object]]) -> dict[str, JsonValue]`.
- Produces: `finalize_raw_trial(measurement, post_exit, recovery_samples, recovery) -> RawTrial`, `validate_raw_trial_evidence(trial) -> None`, and schema-v3 fields `post_exit_recovery_samples` and `post_exit_recovery`.

- [ ] **Step 1: Write failing canonical-policy and schema-v3 tests**

Update the canonical policy test and fixture helpers with these exact values and shapes:

```python
def test_canonical_workload_binds_the_post_exit_recovery_policy():
    required = build_canonical_workload().required_environment

    assert required["memory_pressure"] == "normal"
    assert required["measurement_end_memory_pressure_allowed"] == [
        "normal",
        "warning",
    ]
    assert required["post_exit_memory_pressure_allowed"] == ["normal", "warning"]
    assert required["post_exit_memory_pressure"] == "normal"
    assert required["post_exit_recovery_required_for_warning"] is True
    assert required["post_exit_recovery_sample_interval_seconds"] == 5.0
    assert required["post_exit_recovery_timeout_seconds"] == 300.0
    assert required["post_exit_recovery_stability_seconds"] == 30.0
    assert required["post_exit_recovery_evidence_required"] is True


def _valid_recovery_policy(workload):
    return post_exit_recovery_policy(workload)


def _recovery_evidence(
    workload,
    *,
    immediate_pressure="normal",
    sample_pressures=(),
    sample_elapsed=(),
    sample_status_changes=(),
    outcome=None,
    failure_fields=(),
):
    if len(sample_pressures) != len(sample_elapsed):
        raise ValueError("sample pressure and elapsed fixtures must have equal length")
    if sample_status_changes and len(sample_status_changes) != len(sample_pressures):
        raise ValueError("sample status changes must match the sample count")
    measurement = _valid_child_measurement(workload)
    immediate = _valid_observation("2026-08-09T00:00:02+00:00")
    immediate["environment_status"]["memory_pressure"] = immediate_pressure
    post_exit = build_post_exit_observation(measurement=measurement, **immediate)
    samples = []
    previous_identity = None
    changes = sample_status_changes or ({},) * len(sample_pressures)
    for index, (pressure, elapsed, status_changes) in enumerate(
        zip(sample_pressures, sample_elapsed, changes, strict=True)
    ):
        observation = _valid_observation("2026-08-09T00:00:03+00:00")
        observation["environment_status"].update(
            memory_pressure=pressure, **status_changes
        )
        sample = build_post_exit_recovery_sample(
            measurement=measurement,
            post_exit=post_exit,
            sample_index=index,
            previous_sample_identity=previous_identity,
            elapsed_seconds=elapsed,
            **observation,
        )
        samples.append(sample)
        previous_identity = sample["identity"]
    resolved_outcome = outcome or (
        "not-required" if immediate_pressure == "normal" else "recovered"
    )
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=post_exit_recovery_policy(workload),
        outcome=resolved_outcome,
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
        failure_fields=failure_fields,
    )
    trial = finalize_raw_trial(measurement, post_exit, samples, recovery)
    return measurement, post_exit, tuple(samples), recovery, trial


def _valid_post_exit_recovery(measurement, post_exit, samples=()):
    return build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=post_exit_recovery_policy(build_canonical_workload()),
        outcome="not-required" if not samples else "recovered",
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
    )


def _valid_recovered_evidence():
    return _recovery_evidence(
        build_canonical_workload(),
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[:4]


def _valid_recovered_raw_trial(workload):
    return _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",) * 7,
        sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        outcome="recovered",
    )[4]


def _evidence_for_recovery_outcome(outcome, pressure, failure_fields):
    kwargs = {
        "immediate_pressure": pressure,
        "outcome": outcome,
        "failure_fields": failure_fields,
    }
    if outcome == "recovered":
        kwargs.update(
            sample_pressures=("normal",) * 7,
            sample_elapsed=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0),
        )
    elif outcome == "timeout":
        kwargs.update(sample_pressures=("warning",), sample_elapsed=(300.0,))
    elif outcome == "environment-failure":
        kwargs.update(
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
        )
    return _recovery_evidence(build_canonical_workload(), **kwargs)[:3]


def _valid_raw_trial(workload, metric="prepared-data", pair_index=0):
    measurement = _valid_child_measurement(workload, metric, pair_index)
    post_exit = _valid_post_exit_observation(measurement)
    recovery = _valid_post_exit_recovery(measurement, post_exit)
    return finalize_raw_trial(measurement, post_exit, (), recovery)


def test_raw_trial_v3_embeds_and_revalidates_the_recovery_chain():
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    post_exit = _valid_post_exit_observation(measurement)
    recovery = _valid_post_exit_recovery(measurement, post_exit)
    trial = finalize_raw_trial(measurement, post_exit, (), recovery)

    assert trial.schema_version == 3
    assert trial.post_exit_recovery_samples == ()
    assert trial.post_exit_recovery == recovery
    validate_raw_trial_evidence(trial)

    version_two = trial.to_dict()
    version_two["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        RawTrial.from_dict(version_two)
```

- [ ] **Step 2: Run the new contract tests and verify they fail**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_binds_the_post_exit_recovery_policy \
  v2/tests/unit/test_benchmark_analysis.py::test_raw_trial_v3_embeds_and_revalidates_the_recovery_chain \
  -v
```

Expected: FAIL because the canonical recovery fields, recovery builders, and schema-v3 raw-trial fields do not exist.

- [ ] **Step 3: Add exact canonical policy fields and schema-v3 storage**

Replace the immediate-only policy in `build_canonical_workload()` with:

```python
"memory_pressure": "normal",
"measurement_end_memory_pressure_allowed": ["normal", "warning"],
"post_exit_memory_pressure_allowed": ["normal", "warning"],
"post_exit_memory_pressure": "normal",
"post_exit_recovery_required_for_warning": True,
"post_exit_recovery_sample_interval_seconds": 5.0,
"post_exit_recovery_timeout_seconds": 300.0,
"post_exit_recovery_stability_seconds": 30.0,
"post_exit_recovery_evidence_required": True,
```

Add this projection beside the canonical workload helpers so evidence and the
runner use one exact policy shape:

```python
def post_exit_recovery_policy(
    workload: CanonicalWorkload,
) -> dict[str, JsonValue]:
    required = workload.required_environment
    return {
        "sample_interval_seconds": required[
            "post_exit_recovery_sample_interval_seconds"
        ],
        "timeout_seconds": required["post_exit_recovery_timeout_seconds"],
        "stability_seconds": required["post_exit_recovery_stability_seconds"],
        "required_memory_pressure": required["post_exit_memory_pressure"],
        "required_environment": {
            name: required[name]
            for name in (
                "power_connected",
                "power_mode",
                "low_power_mode",
                "thermal_state",
                "competing_gpu_workload",
            )
        },
        "require_same_hardware_and_software": True,
    }
```

Add `post_exit_recovery_samples: tuple[dict[str, JsonValue], ...]` and `post_exit_recovery: dict[str, JsonValue]` after `post_exit_observation` in `RawTrial`. Require schema version 3, require the serialized sample field to be a list of objects, convert it to a tuple, and require the recovery summary to be an object. Keep child and immediate document versions at 1.

- [ ] **Step 4: Write failing identity-chain and disposition-validation tests**

Add a helper that builds a warning immediate observation plus seven normal samples at elapsed seconds 5 through 35. Then add:

```python
def test_recovery_samples_form_an_exact_ordered_identity_chain():
    measurement, post_exit, samples, recovery = _valid_recovered_evidence()

    previous = None
    for index, sample in enumerate(samples):
        assert validate_post_exit_recovery_sample(
            sample,
            measurement=measurement,
            post_exit=post_exit,
            previous_sample=previous,
        ) == sample
        assert sample["sample_index"] == index
        previous = sample
    assert validate_post_exit_recovery(
        recovery,
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
    ) == recovery

    changed = json.loads(json.dumps(samples))
    changed[1]["previous_sample_identity"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="previous sample identity"):
        validate_post_exit_recovery(
            recovery,
            measurement=measurement,
            post_exit=post_exit,
            samples=changed,
        )


@pytest.mark.parametrize(
    ("outcome", "pressure", "failure_fields"),
    (
        ("not-required", "normal", ()),
        ("recovered", "warning", ()),
        ("timeout", "warning", ()),
        ("critical", "critical", ()),
        ("environment-failure", "warning", ("power_connected",)),
        ("interrupted", "warning", ()),
    ),
)
def test_recovery_summary_rejects_incompatible_outcome_evidence(
    outcome, pressure, failure_fields
):
    measurement, post_exit, samples = _evidence_for_recovery_outcome(
        outcome, pressure, failure_fields
    )
    recovery = build_post_exit_recovery(
        measurement=measurement,
        post_exit=post_exit,
        samples=samples,
        policy=_valid_recovery_policy(build_canonical_workload()),
        outcome=outcome,
        duration_seconds=0.0 if not samples else samples[-1]["elapsed_seconds"],
        failure_fields=failure_fields,
    )
    changed = {**recovery, "outcome": "recovered"}

    with pytest.raises(ValueError, match="recovery identity|recovery outcome"):
        validate_post_exit_recovery(
            changed,
            measurement=measurement,
            post_exit=post_exit,
            samples=samples,
        )
```

Use `failure_fields` as a sorted, duplicate-free list. Require exact bindings to session, journal attempt, metric, pair, child identity, and immediate post-exit identity. Reject boolean indices, non-finite/negative elapsed values, duplicate identities, non-contiguous indices, cadence-infeasible elapsed time, samples past 300 seconds, policy drift, and any outcome that conflicts with its immediate/sample evidence. `interrupted` is valid only for an immediate warning with `completion_source: crash-reconstruction`, is unconditional on the terminal meaning of its final partial sample, and is never admissible. Samples after that first decisive event remain invalid.

- [ ] **Step 5: Implement recovery evidence and deterministic schema-v3 finalization**

Use these domains and summary skeletons in `evidence.py`:

```python
RECOVERY_SAMPLE_KIND = "sml-parent-post-exit-recovery-sample"
RECOVERY_SAMPLE_IDENTITY_DOMAIN = "sml-parent-post-exit-recovery-sample-v1"
RECOVERY_KIND = "sml-parent-post-exit-recovery"
RECOVERY_IDENTITY_DOMAIN = "sml-parent-post-exit-recovery-v2"


def _recovery_binding(child, parent):
    trial = child["trial"]
    return {
        "session_identity": child["session_identity"],
        "journal_attempt_index": child["journal_attempt_index"],
        "metric": trial["metric"],
        "pair_index": trial["pair_index"],
        "child_measurement_identity": child["identity"],
        "post_exit_observation_identity": parent["identity"],
    }
```

`build_post_exit_recovery_sample` must add `kind`, `version: 1`, the binding, `sample_index`, `previous_sample_identity`, `elapsed_seconds`, the complete observation, and its identity. `build_post_exit_recovery` must add the binding, exact policy, ordered `sample_identities`, `outcome`, `duration_seconds`, sorted `failure_fields`, terminal sample identity, terminal environment, and identity.

Change `merge_environment_status` to consume `recovery_samples` and `recovery`. Use the immediate observation for top-level memory on `not-required`; otherwise use the last recovery sample when one exists. Merge worst thermal, conservative power, Low Power Mode, and competing-workload values across child start/end, immediate post-exit, and all recovery samples. Retain nested `start`, `end`, and `post_exit`, and add `post_exit_recovery_outcome` and `post_exit_recovery_final`.

Change finalization to:

```python
def finalize_raw_trial(measurement, post_exit, recovery_samples, recovery):
    child = validate_child_trial_measurement(measurement)
    parent = validate_post_exit_observation(post_exit, measurement=child)
    samples = validate_post_exit_recovery_samples(
        recovery_samples, measurement=child, post_exit=parent
    )
    summary = validate_post_exit_recovery(
        recovery,
        measurement=child,
        post_exit=parent,
        samples=samples,
    )
    return RawTrial.from_dict(
        {
            "schema_version": 3,
            **child["trial"],
            "software_versions": child["start"]["software_versions"],
            "hardware": child["start"]["hardware"],
            "environment_status": merge_environment_status(
                child["start"]["environment_status"],
                child["end"]["environment_status"],
                parent["environment_status"],
                samples,
                summary,
            ),
            "evidence_session_identity": child["session_identity"],
            "journal_attempt_index": child["journal_attempt_index"],
            "child_measurement": child,
            "post_exit_observation": parent,
            "post_exit_recovery_samples": list(samples),
            "post_exit_recovery": summary,
        }
    )
```

Do not require hardware/software equality inside finalization: rejected `environment-failure` evidence must remain representable. Accepted-trial validation in Task 5 will enforce equality across every observation.

- [ ] **Step 6: Run the complete evidence/schema slice**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'canonical_workload or child_measurement or post_exit_observation or recovery_sample or recovery_summary or raw_trial' \
  -v
```

Expected: PASS. Retain explicit schema-version-2 negative fixtures while all valid current fixtures use schema version 3.

Update `_with_trial_payload`, `_with_observation_changes`, and
`_with_environment_observations` in the same test cycle so they rebuild the
recovery summary against the changed child/immediate identities before calling
the four-argument finalizer. They must never copy a stale recovery identity into
a changed fixture.

- [ ] **Step 7: Commit the recovery evidence contract**

```bash
git add v2/benchmarks/schema.py v2/benchmarks/evidence.py v2/benchmarks/workload.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): define post-exit recovery evidence"
```

### Task 2: Implement the fixed post-exit recovery state machine

**Files:**
- Modify: `v2/benchmarks/recovery.py:1-109`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:2050-2240`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `(memory_probe_started_at, complete_observation)` collector results, fixed recovery policy, injected `clock`, injected `sleep`, and injected durable `record_sample` callback.
- Produces: `PostExitMemoryRecoveryResult(outcome: Literal["not-required", "recovered", "timeout", "critical", "environment-failure"], duration_seconds: float, samples: tuple[dict, ...], failure_fields: tuple[str, ...])`.
- Produces: `wait_for_post_exit_memory_recovery(*, immediate_observation: dict, immediate_started_at: float, recovery_policy: dict, collect: Callable[[], tuple[float, dict]], classify_nonmemory: Callable[[dict], tuple[str, ...]], record_sample: Callable[[int, float, dict, str | None], dict], clock: Callable[[], float], sleep: Callable[[float], None]) -> PostExitMemoryRecoveryResult`, sharing its reducer with evidence validation.
- Preserves: `wait_for_nominal_thermal_window` behavior and constants unchanged.

- [ ] **Step 1: Write failing immediate and stable-recovery tests with a fake clock**

Add a small deterministic clock and these tests:

```python
class _RecoveryClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_post_exit_recovery_returns_immediately_for_normal_memory():
    clock = _RecoveryClock()
    collected = []
    result = wait_for_post_exit_memory_recovery(
        immediate_observation=_valid_observation("2026-08-09T00:00:00+00:00"),
        immediate_started_at=clock(),
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=lambda: collected.append(True),
        classify_nonmemory=lambda observation: (),
        record_sample=lambda *args: pytest.fail("normal memory must not sample"),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result == PostExitMemoryRecoveryResult("not-required", 0.0, (), ())
    assert collected == []
    assert clock.sleeps == []


def test_post_exit_recovery_requires_thirty_continuous_normal_seconds():
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    observations = [
        _valid_observation(f"2026-08-09T00:00:{second:02d}+00:00")
        for second in (5, 10, 15, 20, 25, 30, 35)
    ]
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=iter(observations).__next__,
        classify_nonmemory=lambda observation: (),
        record_sample=lambda index, elapsed, observation, previous: persisted.append(
            {"sample_index": index, "elapsed_seconds": elapsed, "identity": str(index)}
        ) or persisted[-1],
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.outcome == "recovered"
    assert result.duration_seconds == 35.0
    assert len(result.samples) == 7
    assert clock.sleeps == [5.0] * 7
```

- [ ] **Step 2: Run the immediate/recovered tests and verify the API is absent**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_post_exit_recovery_returns_immediately_for_normal_memory \
  v2/tests/unit/test_benchmark_analysis.py::test_post_exit_recovery_requires_thirty_continuous_normal_seconds \
  -v
```

Expected: FAIL during import because `PostExitMemoryRecoveryResult` and `wait_for_post_exit_memory_recovery` are absent.

- [ ] **Step 3: Implement the minimal fixed-cadence state machine**

Use this control shape in `recovery.py`; every collected observation must be persisted before its outcome is examined:

```python
@dataclass(frozen=True, slots=True)
class PostExitMemoryRecoveryResult:
    outcome: Literal[
        "not-required", "recovered", "timeout", "critical", "environment-failure"
    ]
    duration_seconds: float
    samples: tuple[dict, ...]
    failure_fields: tuple[str, ...]


def wait_for_post_exit_memory_recovery(
    *,
    immediate_observation: dict,
    immediate_started_at: float,
    recovery_policy: dict,
    collect: Callable[[], dict],
    classify_nonmemory: Callable[[dict], tuple[str, ...]],
    record_sample: Callable[[int, float, dict, str | None], dict],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> PostExitMemoryRecoveryResult:
    immediate_failures = tuple(classify_nonmemory(immediate_observation))
    immediate_pressure = immediate_observation["environment_status"][
        "memory_pressure"
    ]
    if immediate_failures:
        return PostExitMemoryRecoveryResult(
            "environment-failure", 0.0, (), immediate_failures
        )
    if immediate_pressure == "critical":
        return PostExitMemoryRecoveryResult("critical", 0.0, (), ())
    if immediate_pressure == "normal":
        return PostExitMemoryRecoveryResult("not-required", 0.0, (), ())
    if immediate_pressure != "warning":
        raise ValueError("immediate post-exit memory pressure is invalid")

    interval = float(recovery_policy["sample_interval_seconds"])
    deadline = immediate_started_at + float(recovery_policy["timeout_seconds"])
    stability = float(recovery_policy["stability_seconds"])
    samples = []
    normal_since = None
    previous_identity = None
    last_sample_started_at = immediate_started_at

    while True:
        scheduled = min(last_sample_started_at + interval, deadline)
        sleep(max(0.0, scheduled - clock()))
        sample_started_at = clock()
        if sample_started_at > deadline:
            return PostExitMemoryRecoveryResult(
                "timeout", deadline - immediate_started_at, tuple(samples), ()
            )
        observation = collect()
        elapsed = sample_started_at - immediate_started_at
        sample = record_sample(
            len(samples), elapsed, observation, previous_identity
        )
        samples.append(sample)
        previous_identity = sample["identity"]
        last_sample_started_at = sample_started_at

        failures = tuple(classify_nonmemory(observation))
        pressure = observation["environment_status"]["memory_pressure"]
        if failures:
            return PostExitMemoryRecoveryResult(
                "environment-failure", elapsed, tuple(samples), failures
            )
        if pressure == "critical":
            return PostExitMemoryRecoveryResult(
                "critical", elapsed, tuple(samples), ()
            )
        if pressure == "warning":
            normal_since = None
        elif pressure == "normal":
            normal_since = elapsed if normal_since is None else normal_since
            if elapsed - normal_since >= stability:
                return PostExitMemoryRecoveryResult(
                    "recovered", elapsed, tuple(samples), ()
                )
        else:
            raise ValueError("recovery memory pressure is invalid")
        if sample_started_at >= deadline:
            return PostExitMemoryRecoveryResult(
                "timeout", elapsed, tuple(samples), ()
            )
```

Validate positive finite interval/timeout/stability values, require `stability_seconds <= timeout_seconds`, and require a finite non-negative `immediate_started_at`.

- [ ] **Step 4: Write the reset, deadline, critical, and environment-failure matrix**

Add exact cases proving a warning at elapsed 20 resets a normal window begun at 5, a sample begun at 300 is recorded before timeout disposition, a critical sample stops immediately, and any reported failure field stops immediately:

```python
def _run_scripted_memory_recovery(*, pressures, failure_fields=()):
    clock = _RecoveryClock()
    immediate = _valid_observation("2026-08-09T00:00:00+00:00")
    immediate["environment_status"]["memory_pressure"] = "warning"
    scripted = []
    for index, pressure in enumerate(pressures, 1):
        observation = _valid_observation(
            f"2026-08-09T00:00:{min(index * 5, 59):02d}+00:00"
        )
        observation["environment_status"]["memory_pressure"] = pressure
        scripted.append(observation)
    persisted = []

    result = wait_for_post_exit_memory_recovery(
        immediate_observation=immediate,
        immediate_started_at=0.0,
        recovery_policy=_valid_recovery_policy(build_canonical_workload()),
        collect=iter(scripted).__next__,
        classify_nonmemory=lambda observation: tuple(failure_fields),
        record_sample=lambda index, elapsed, observation, previous: persisted.append(
            {
                "sample_index": index,
                "elapsed_seconds": elapsed,
                "identity": f"sample-{index}",
            }
        )
        or persisted[-1],
        clock=clock,
        sleep=clock.sleep,
    )
    return result, persisted


@pytest.mark.parametrize(
    ("terminal_pressure", "failure_fields", "expected"),
    (
        ("critical", (), "critical"),
        ("normal", ("power_connected",), "environment-failure"),
        ("normal", ("thermal_state",), "environment-failure"),
        ("normal", ("hardware",), "environment-failure"),
        ("normal", ("software_versions",), "environment-failure"),
    ),
)
def test_post_exit_recovery_stops_on_terminal_failure(
    terminal_pressure, failure_fields, expected
):
    result, persisted = _run_scripted_memory_recovery(
        pressures=(terminal_pressure,), failure_fields=failure_fields
    )
    assert result.outcome == expected
    assert len(persisted) == 1
    assert result.failure_fields == failure_fields
```

The timeout test must assert the original deadline is not extended after any warning reset. The cadence test must simulate a seven-second collector and prove the next collection starts immediately after that collector rather than overlapping or scheduling from an already missed absolute tick.

- [ ] **Step 5: Run all recovery-module tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'post_exit_recovery or thermal_recovery or nominal_window' \
  -v
```

Expected: PASS, including every existing thermal recovery test unchanged.

- [ ] **Step 6: Commit the recovery state machine**

```bash
git add v2/benchmarks/recovery.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): wait for post-exit memory recovery"
```

### Task 3: Integrate memory-first recovery into parent trial finalization

**Files:**
- Modify: `v2/benchmarks/runner.py:27-52,1760-1790,2031-2120,2740-2800,3020-3084`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:4603-5014`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: Task 1 evidence builders/finalizer, Task 2 recovery state machine, and create-only `atomic_write_json`.
- Produces: `collect_post_exit_environment(*, clock: Callable[[], float] = time.monotonic) -> tuple[float, dict]`, where the float is captured immediately before memory pressure and the dictionary is a complete timestamped observation.
- Produces: `_recovery_failure_fields(observation: dict, *, expected_hardware: dict, expected_software_versions: dict[str, str], required_environment: dict) -> tuple[str, ...]`.
- Extends `_launch_trial` with required `recovery_samples_directory: Path` and `recovery_output: Path` keyword paths plus injectable `clock: Callable[[], float] = time.monotonic` and `sleep: Callable[[float], None] = time.sleep`; all terminal outcomes return a complete schema-v3 `RawTrial`.

- [ ] **Step 1: Write failing parent lifecycle and immutable-output tests**

Replace the current parent lifecycle test with an event-ordered warning-to-recovery test:

```python
def _observation_with_memory(pressure, free_percentage):
    observation = _valid_observation("2026-08-09T00:00:03+00:00")
    observation["environment_status"].update(
        memory_pressure=pressure,
        memory_free_percentage=free_percentage,
    )
    return observation


def test_parent_records_immediate_warning_then_every_recovery_sample_before_finalizing(
    tmp_path, monkeypatch
):
    workload = build_canonical_workload()
    measurement = _valid_child_measurement(workload)
    clock = _RecoveryClock()
    events = []
    observations = [
        _observation_with_memory("warning", 36),
        *[_observation_with_memory("normal", 67) for _ in range(7)],
    ]

    monkeypatch.setattr(
        benchmark_runner.subprocess,
        "run",
        lambda command, *, cwd, check: (
            events.append("child-exit"),
            atomic_write_json(
                tmp_path / "measurement.json", measurement, create_only=True
            ),
        )[-1],
    )
    monkeypatch.setattr(
        benchmark_runner,
        "collect_post_exit_environment",
        lambda **kwargs: (
            events.append("memory-first") or 0.0,
            observations.pop(0),
        ),
    )

    trial = benchmark_runner._launch_trial(
        **_launch_trial_arguments(tmp_path),
        evidence_session_identity=measurement["session_identity"],
        journal_attempt_index=0,
        measurement_output=tmp_path / "measurement.json",
        post_exit_output=tmp_path / "post-exit.json",
        recovery_samples_directory=tmp_path / "recovery-samples",
        recovery_output=tmp_path / "recovery.json",
        output=tmp_path / "trial.json",
        clock=clock,
        sleep=clock.sleep,
    )

    assert events[0] == "child-exit"
    assert events[1] == "memory-first"
    assert trial.schema_version == 3
    assert trial.post_exit_observation["environment_status"]["memory_pressure"] == "warning"
    assert trial.post_exit_recovery["outcome"] == "recovered"
    assert len(trial.post_exit_recovery_samples) == 7
```

Inject fake clock/sleep functions so this test does not consume real time. Add create-only collision tests for the first recovery sample, terminal recovery summary, and final raw trial; each must preserve the pre-existing bytes and stop before later publication.

- [ ] **Step 2: Run the parent lifecycle test and verify missing arguments/builders fail**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_parent_records_immediate_warning_then_every_recovery_sample_before_finalizing \
  v2/tests/unit/test_benchmark_analysis.py -k 'post_exit_output_never_overwrites or trial_output_never_overwrites' \
  -v
```

Expected: FAIL because `_launch_trial` has no recovery paths and finalizes immediately after the first post-exit observation.

- [ ] **Step 3: Add complete memory-first observation collection and failure-field derivation**

Refactor the collector to capture the monotonic start immediately before `_memory_pressure()`:

```python
def collect_post_exit_environment(*, clock=time.monotonic) -> tuple[float, dict]:
    started_at = clock()
    memory_sample = _memory_pressure()
    observed_at_utc = _utc_now_iso()
    hardware, environment_status, software_versions = collect_environment(
        memory_sample=memory_sample
    )
    return started_at, {
        "observed_at_utc": observed_at_utc,
        "hardware": hardware,
        "environment_status": environment_status,
        "software_versions": software_versions,
    }
```

`_recovery_failure_fields` must return a sorted tuple containing `hardware` or `software_versions` for identity drift and any mismatched field among `power_connected`, `power_mode`, `low_power_mode`, `thermal_state`, `competing_gpu_workload`. It must call `validate_thermal_observation` first so string/raw mismatches fail as malformed evidence rather than becoming a retryable outcome. It deliberately ignores `memory_pressure` and `memory_free_percentage`.

- [ ] **Step 4: Orchestrate recovery and persist each stage in `_launch_trial`**

After the child exits and its measurement validates:

```python
immediate_started_at, observation = collect_post_exit_environment(clock=clock)
post_exit = build_post_exit_observation(measurement=measurement, **observation)
atomic_write_json(post_exit_output, post_exit, create_only=True)
policy = post_exit_recovery_policy(workload)

def record_sample(index, elapsed, collected, previous_identity):
    sample = build_post_exit_recovery_sample(
        measurement=measurement,
        post_exit=post_exit,
        sample_index=index,
        previous_sample_identity=previous_identity,
        elapsed_seconds=elapsed,
        **collected,
    )
    atomic_write_json(
        recovery_samples_directory / f"{index}.json",
        sample,
        create_only=True,
    )
    return sample

result = wait_for_post_exit_memory_recovery(
    immediate_observation=observation,
    immediate_started_at=immediate_started_at,
    recovery_policy=policy,
    collect=lambda: collect_post_exit_environment(clock=clock)[1],
    classify_nonmemory=lambda item: _recovery_failure_fields(
        item,
        expected_hardware=measurement["start"]["hardware"],
        expected_software_versions=measurement["start"]["software_versions"],
        required_environment=workload.required_environment,
    ),
    record_sample=record_sample,
    clock=clock,
    sleep=sleep,
)
recovery = build_post_exit_recovery(
    measurement=measurement,
    post_exit=post_exit,
    samples=result.samples,
    policy=policy,
    outcome=result.outcome,
    duration_seconds=result.duration_seconds,
    failure_fields=result.failure_fields,
)
atomic_write_json(recovery_output, recovery, create_only=True)
trial = finalize_raw_trial(measurement, post_exit, result.samples, recovery)
atomic_write_json(output, trial.to_dict(), create_only=True)
```

Pass the already-built `workload` into policy derivation. Keep collection, persistence, and classification ordered exactly; do not read metric result fields in this branch.

- [ ] **Step 5: Route baseline and comparison paths through the same lifecycle**

For baseline attempts, pass:

```python
recovery_samples_directory=journal.recovery_samples_path(
    slot, attempt.journal_attempt_index
),
recovery_output=journal.recovery_path(slot, attempt.journal_attempt_index),
```

For paired comparisons, pass unique paths:

```python
recovery_samples_directory=output_directory / f"{stem}.recovery-samples",
recovery_output=output_directory / f"{stem}.recovery.json",
```

Update comparison path tests to require one recovery summary per process and distinct sample directories. A timeout/critical/environment-failure raw trial must reach `_validate_acceptance_environment` and stop before the next paired process.

- [ ] **Step 6: Run all child/parent/comparison lifecycle tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'child_process or parent_ or post_exit or recovery_output or paired_trials or comparison_evidence_session' \
  -v
```

Expected: PASS. Tests must prove memory is sampled before slower probes, the immediate warning is retained, samples precede their summary, and the summary precedes raw-trial publication.

- [ ] **Step 7: Commit parent recovery integration**

```bash
git add v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): finalize trials after memory recovery"
```

### Task 4: Journal recovery samples, summaries, and schema-v3 outcomes

**Files:**
- Modify: `v2/benchmarks/journal.py:19-35,55-66,313-353,843-848,1020-1352,1509-1723`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:5017-6040`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: Task 1 evidence validators/finalizer and existing create-only atomic journal primitives.
- Produces: `BaselineJournal.recovery_samples_path(slot, journal_attempt_index) -> Path`, `recovery_sample_path(slot, journal_attempt_index, sample_index) -> Path`, and `recovery_path(slot, journal_attempt_index) -> Path`.
- Extends: `JournalAttemptEvidence(attempt, measurement, post_exit, recovery_samples, recovery, trial)`.
- Produces: rejected-trial envelope version 3 with exact fields `child_measurement_identity`, `post_exit_observation_identity`, `post_exit_recovery_identity`, and `trial`.

- [ ] **Step 1: Write failing path, topology, and retention tests**

Update `_journal_trial_evidence` and `_persist_journal_evidence` to produce child, immediate, samples, recovery, and schema-v3 trial. Then add:

```python
def test_journal_retains_the_complete_recovery_chain_after_acceptance(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, samples, recovery, trial = _persist_journal_evidence(
        journal, attempt
    )

    journal.accept_inflight(attempt, trial)

    assert journal.measurement_path(slot, 0).is_file()
    assert journal.post_exit_path(slot, 0).is_file()
    assert journal.recovery_path(slot, 0).is_file()
    assert tuple(
        journal.recovery_sample_path(slot, 0, index).is_file()
        for index in range(len(samples))
    ) == (True,) * len(samples)
    assert journal.load_accepted((slot,)) == {slot: trial}


def test_journal_reports_post_exit_without_recovery_as_pending(tmp_path):
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, _trial = _persist_journal_evidence(
        journal, attempt, recovery=False, inflight=False
    )

    assert journal.load_pending_attempts((slot,)) == (
        baseline_journal.JournalAttemptEvidence(
            attempt=attempt,
            measurement=measurement,
            post_exit=post_exit,
            recovery_samples=(),
            recovery=None,
            trial=None,
        ),
    )
```

Add negative cases for sample without post-exit, sample bound to another immediate identity, sample index gap, previous-identity fork, summary without its exact samples, inflight without summary, accepted/rejected evidence missing a summary, unexpected paths, symlinked ancestors, and orphan atomic temporaries under both new categories.

- [ ] **Step 2: Run the new journal tests and verify unrecognized paths fail**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_journal_retains_the_complete_recovery_chain_after_acceptance \
  v2/tests/unit/test_benchmark_analysis.py::test_journal_reports_post_exit_without_recovery_as_pending \
  -v
```

Expected: FAIL because the journal has no recovery paths or recovery fields in pending evidence.

- [ ] **Step 3: Add exact path recognition and immutable stage loading**

Add `recovery-samples` and `recovery` to `STATE_DIRECTORY_NAMES`. Extend `_is_journal_destination` so summary paths use four components and sample paths use five. Add:

```python
def recovery_samples_path(self, slot, journal_attempt_index):
    return (
        self.root
        / "recovery-samples"
        / slot.metric
        / str(slot.pair_index)
        / str(journal_attempt_index)
    )


def recovery_sample_path(self, slot, journal_attempt_index, sample_index):
    _require_non_negative_index(sample_index, label="recovery sample index")
    return self.recovery_samples_path(slot, journal_attempt_index) / f"{sample_index}.json"


def recovery_path(self, slot, journal_attempt_index):
    return (
        self.root
        / "recovery"
        / slot.metric
        / str(slot.pair_index)
        / f"{journal_attempt_index}.json"
    )
```

Generalize attempt-record parsing only for four-component documents. Add a dedicated `_recovery_sample_records()` for the five-component layout. It must validate canonical paths, contiguous indices per attempt, complete identity chains, session/slot/attempt bindings, and each sample only after loading its exact child and immediate documents.

- [ ] **Step 4: Extend every attempt topology through recovery**

Change `JournalAttemptEvidence` to:

```python
@dataclass(frozen=True, slots=True)
class JournalAttemptEvidence:
    attempt: JournalAttempt
    measurement: dict
    post_exit: dict | None
    recovery_samples: tuple[dict, ...]
    recovery: dict | None
    trial: RawTrial | None
```

Enforce these implications in `_validate_attempt_history`:

```text
recovery sample -> exact post-exit -> exact measurement
recovery summary -> exact ordered samples -> exact post-exit -> exact measurement
inflight -> exact recovery summary -> exact ordered samples
accepted -> exact embedded recovery chain
rejected finalized trial -> exact embedded recovery chain
one attempt -> at most one accepted or rejected outcome
all attempt indices -> contiguous from zero per slot
```

Use `finalize_raw_trial(measurement, post_exit, samples, recovery)` for inflight, accepted, and rejected equality checks. Preserve every stage document after accept/reject and preserve the existing hard-linked accepted transition semantics.

- [ ] **Step 5: Advance rejected outcomes to version 3**

Use this exact body for finalized outcomes:

```python
body = {
    "kind": "sml-baseline-rejected-trial",
    "version": 3,
    "journal_attempt_index": attempt.journal_attempt_index,
    "reason": reason,
    "child_measurement_identity": trial.child_measurement["identity"],
    "post_exit_observation_identity": trial.post_exit_observation["identity"],
    "post_exit_recovery_identity": trial.post_exit_recovery["identity"],
    "trial": trial.to_dict(),
}
document = {
    **body,
    "identity": structured_identity("sml-baseline-rejected-trial-v3", body),
}
```

The measurement-only rejection also uses version 3/domain v3 but sets all later identities and `trial` to `None`. Expand exact allowed reasons with `critical-post-exit-memory-pressure`, `post-exit-recovery-environment-violation`, and `interrupted-post-exit-recovery`. Continue recomputing the ordered reason from evidence before a finalized rejection can be written.

- [ ] **Step 6: Run the complete journal/atomicity/lock slice**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'journal or baseline_lock or output_lock or atomic or orphan' \
  -v
```

Expected: PASS, including existing session initialization, hard-link transition, preflight, thermal-recovery, symlink, path-safety, and immutable-write tests.

- [ ] **Step 7: Commit durable recovery topology**

```bash
git add v2/benchmarks/journal.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): journal post-exit recovery evidence"
```

### Task 5: Classify recovery outcomes and reconstruct crash states

**Files:**
- Modify: `v2/benchmarks/evidence.py:325-417`
- Modify: `v2/benchmarks/runner.py:73-83,531-596,2251-2406`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:1266-1425,2580-3525`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: complete schema-v3 raw evidence, `JournalAttemptEvidence`, Task 2 recovery outcomes, and Task 4 durable paths.
- Produces exact dispositions `accept`, `thermal-reject`, `memory-reject`, and `environment-reject`.
- Produces: `EnvironmentTrialRejected(RuntimeError)` for a durable non-memory, non-thermal recovery failure.
- Produces: deterministic pending reconstruction: immediate normal -> `not-required`; immediate critical -> `critical`; immediate invalid non-memory -> `environment-failure`; immediate warning without a summary -> `interrupted` regardless of partial samples.

- [ ] **Step 1: Write the complete schema-v3 disposition matrix**

Replace the immediate-only matrix with:

```python
def _raw_trial_for_recovery_signal(workload, signal):
    if signal == "accepted-not-required":
        return _valid_raw_trial(workload)
    if signal == "accepted-recovered":
        return _valid_recovered_raw_trial(workload)
    if signal in {"start-warning", "end-critical"}:
        trial = _valid_raw_trial(workload)
        start = dict(trial.environment_status["start"])
        end = dict(trial.environment_status["end"])
        if signal == "start-warning":
            start["memory_pressure"] = "warning"
        else:
            end["memory_pressure"] = "critical"
        return _with_environment_observations(trial, start=start, end=end)
    if signal == "recovery-timeout":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("warning",),
            sample_elapsed=(300.0,),
            outcome="timeout",
        )[4]
    if signal == "recovery-critical":
        return _recovery_evidence(
            workload, immediate_pressure="critical", outcome="critical"
        )[4]
    if signal == "recovery-interrupted":
        return _recovery_evidence(
            workload, immediate_pressure="warning", outcome="interrupted"
        )[4]
    if signal == "recovery-thermal":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=(
                {"thermal_state": "fair", "thermal_state_raw_value": 1},
            ),
            outcome="environment-failure",
            failure_fields=("thermal_state",),
        )[4]
    if signal == "recovery-power":
        return _recovery_evidence(
            workload,
            immediate_pressure="warning",
            sample_pressures=("normal",),
            sample_elapsed=(5.0,),
            sample_status_changes=({"power_connected": False},),
            outcome="environment-failure",
            failure_fields=("power_connected",),
        )[4]
    raise AssertionError(f"unknown recovery signal: {signal}")


@pytest.mark.parametrize(
    ("signal", "outcome", "reason"),
    (
        ("accepted-not-required", "accept", None),
        ("accepted-recovered", "accept", None),
        ("start-warning", "memory-reject", "non-normal-start-memory-pressure"),
        ("end-critical", "memory-reject", "critical-measurement-memory-pressure"),
        ("recovery-timeout", "memory-reject", "persistent-post-exit-memory-pressure"),
        ("recovery-critical", "memory-reject", "critical-post-exit-memory-pressure"),
        ("recovery-interrupted", "memory-reject", "interrupted-post-exit-recovery"),
        ("recovery-thermal", "thermal-reject", "non-nominal-thermal"),
        (
            "recovery-power",
            "environment-reject",
            "post-exit-recovery-environment-violation",
        ),
    ),
)
def test_trial_disposition_uses_the_complete_recovery_chain(signal, outcome, reason):
    workload = build_canonical_workload()
    trial = _raw_trial_for_recovery_signal(workload, signal)

    disposition = classify_trial_environment(workload, trial)

    assert (disposition.outcome, disposition.reason) == (outcome, reason)
```

Add a test that changes `trial.value`, `elapsed_seconds`, `compilation_seconds`, and `peak_memory_bytes`, rebuilds the child/recovery identities, and proves the disposition remains identical.

- [ ] **Step 2: Run the disposition matrix and verify old immediate authority fails**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_trial_disposition_uses_the_complete_recovery_chain \
  -v
```

Expected: FAIL because the current reason derives only from immediate post-exit memory and has no recovery/environment outcomes.

- [ ] **Step 3: Derive ordered reasons and validate every observation**

Update `finalized_trial_rejection_reason` with this exact precedence:

```python
def finalized_trial_rejection_reason(trial, required_environment):
    status = trial.environment_status
    start = status["start"]
    end = status["end"]
    recovery = trial.post_exit_recovery
    recovery_statuses = [
        sample["environment_status"]
        for sample in trial.post_exit_recovery_samples
    ]
    if start["memory_pressure"] != required_environment["memory_pressure"]:
        return "non-normal-start-memory-pressure"
    if end["memory_pressure"] not in required_environment[
        "measurement_end_memory_pressure_allowed"
    ]:
        return "critical-measurement-memory-pressure"
    if recovery["outcome"] == "critical":
        return "critical-post-exit-memory-pressure"
    if recovery["outcome"] == "timeout":
        return "persistent-post-exit-memory-pressure"
    if recovery["outcome"] == "interrupted":
        return "interrupted-post-exit-recovery"
    thermal_failed = any(
        observation["thermal_state"] != required_environment["thermal_state"]
        for observation in (
            status["start"],
            status["end"],
            status["post_exit"],
            *recovery_statuses,
        )
    )
    recovery_failures = set(recovery["failure_fields"])
    if recovery["outcome"] == "environment-failure" and (
        recovery_failures - {"thermal_state"}
    ):
        return "post-exit-recovery-environment-violation"
    if thermal_failed:
        return "non-nominal-thermal"
    if recovery["outcome"] == "environment-failure":
        return "post-exit-recovery-environment-violation"
    if recovery["outcome"] not in ("not-required", "recovered"):
        raise ValueError("post-exit recovery outcome is invalid")
    return None
```

The implementation must use this field order. Extend
`TrialEnvironmentDisposition.outcome` and map the final reason to
`environment-reject`. In `_validate_acceptance_environment`, validate thermal
string/raw values and conservative summary derivation across every nested
observation. With `allow_rejected_environment=True`, permit only the exact
observed fields named by a valid `environment-failure` summary; accepted
evidence still requires exact hardware/software and all required
power/environment values at every point.

- [ ] **Step 4: Write failing crash-reconstruction and no-auto-memory-retry tests**

Add these exact accepted and interrupted state transitions:

```python
def test_capture_reconstructs_not_required_summary_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    measurement, post_exit, _samples, _recovery, expected = _journal_trial_evidence(
        journal, attempt, _valid_raw_trial(workload)
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    launches = []

    trials = capture_baseline_trials(
        journal=journal,
        slots=(slot,),
        launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
        preflight=lambda: _preflight_from_trial(expected),
        validate_preflight=lambda hardware, status, software: None,
        recover=lambda slot, index, deadline, trigger: pytest.fail(
            "normal pending evidence entered thermal recovery"
        ),
        validate_trial=_baseline_validator(workload),
        classify_trial=lambda trial: classify_trial_environment(workload, trial),
        progress=lambda message: None,
    )

    assert launches == []
    assert trials == (expected,)
    assert journal.recovery_path(slot, 0).is_file()
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery")[
        "outcome"
    ] == "not-required"


def test_capture_rejects_partial_warning_recovery_before_any_launch(tmp_path):
    workload = build_canonical_workload()
    journal = BaselineJournal.open(tmp_path / "state", _session_document(tmp_path))
    slot = BaselineSlot("prepared-data", 0)
    attempt = journal.next_attempt(slot)
    interrupted = _recovery_evidence(
        workload,
        immediate_pressure="warning",
        sample_pressures=("normal",),
        sample_elapsed=(5.0,),
        outcome="interrupted",
    )[4]
    measurement, post_exit, samples, _recovery, rebound = _journal_trial_evidence(
        journal, attempt, interrupted
    )
    atomic_write_json(journal.measurement_path(slot, 0), measurement, create_only=True)
    atomic_write_json(journal.post_exit_path(slot, 0), post_exit, create_only=True)
    atomic_write_json(
        journal.recovery_sample_path(slot, 0, 0), samples[0], create_only=True
    )
    launches = []

    with pytest.raises(
        benchmark_runner.MemoryPressureTrialRejected,
        match="interrupted-post-exit-recovery",
    ):
        capture_baseline_trials(
            journal=journal,
            slots=(slot,),
            launch_trial=lambda slot, next_attempt: launches.append(next_attempt),
            preflight=lambda: _preflight_from_trial(rebound),
            validate_preflight=lambda hardware, status, software: None,
            recover=lambda slot, index, deadline, trigger: pytest.fail(
                "interrupted memory recovery entered thermal recovery"
            ),
            validate_trial=_baseline_validator(workload),
            classify_trial=lambda trial: classify_trial_environment(workload, trial),
            progress=lambda message: None,
        )

    assert launches == []
    assert read_json_object(journal.recovery_path(slot, 0), label="recovery")[
        "outcome"
    ] == "interrupted"
```

Add two variants using the same explicit journal writes: immediate `critical`
must raise `MemoryPressureTrialRejected` with
`critical-post-exit-memory-pressure`, and an immediate power failure must raise
`EnvironmentTrialRejected` with
`post-exit-recovery-environment-violation`. Add a second invocation after the
interrupted test and assert it launches only journal attempt 1 while preserving
attempt 0's sample, summary, raw trial, and rejection record.

- [ ] **Step 5: Reconstruct terminal summaries and preserve existing retry boundaries**

Add a focused helper used only for pending journal evidence:

```python
def _reconstruct_missing_recovery(state, workload):
    observation = {
        name: state.post_exit[name]
        for name in (
            "observed_at_utc",
            "hardware",
            "environment_status",
            "software_versions",
        )
    }
    failures = _recovery_failure_fields(
        observation,
        expected_hardware=state.measurement["start"]["hardware"],
        expected_software_versions=state.measurement["start"]["software_versions"],
        required_environment=workload.required_environment,
    )
    pressure = observation["environment_status"]["memory_pressure"]
    if failures:
        outcome = "environment-failure"
    elif pressure == "critical":
        outcome = "critical"
    elif pressure == "normal":
        outcome = "not-required"
    else:
        outcome = "interrupted"
    return build_post_exit_recovery(
        measurement=state.measurement,
        post_exit=state.post_exit,
        samples=state.recovery_samples,
        policy=post_exit_recovery_policy(workload),
        outcome=outcome,
        duration_seconds=(
            0.0
            if not state.recovery_samples
            else state.recovery_samples[-1]["elapsed_seconds"]
        ),
        failure_fields=failures,
        completion_source="crash-reconstruction",
    )
```

In pending replay, measurement-only still becomes `missing-immediate-post-exit-evidence`; post-exit without summary calls this helper and writes the summary create-only; summary without inflight finalizes schema v3; inflight classifies directly. Memory rejection calls `journal.reject_inflight` then raises `MemoryPressureTrialRejected`. Environment rejection calls `journal.reject_inflight` then raises `EnvironmentTrialRejected`. Thermal-only rejection retains the current trigger, five-minute nominal recovery, and retry loop.
For immediate warning, the reconstruction outcome is `interrupted` even when
the final partial sample is decisive under live semantics; only a sample after
that first decisive sample is invalid.

- [ ] **Step 6: Run capture, replay, thermal, and disposition tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'trial_disposition or capture_ or persisted_thermal or manual_resume or performance_independent' \
  -v
```

Expected: PASS. Existing accepted-slot reuse, preflight persistence, one two-hour thermal deadline, five-minute thermal window, and rejection-before-new-launch tests must remain green.

- [ ] **Step 7: Commit recovery classification and replay**

```bash
git add v2/benchmarks/evidence.py v2/benchmarks/runner.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "feat(v2): classify recovered post-exit trials"
```

### Task 6: Enforce schema-v3 artifacts and document operations

**Files:**
- Modify: `v2/benchmarks/runner.py:145-150,293-529,640-1030,2519-2630,3087-3410`
- Modify: `v2/benchmarks/README.md:17-140`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:360-430,1500-2050,3979-4043,4928-5014`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: self-contained `RawTrial` schema version 3 and canonical recovery policy.
- Produces: raw identity domain `sml-raw-benchmark-trial-v3`; manifests, raw JSONL, comparison reports, predecessor replay, phase validation, and final validation that reject schema version 2 or any tampered recovery chain.
- Documents: exact immediate/recovered authority, fixed timing, journal layout, crash behavior, thermal behavior, manual resume, and incompatible retained state.

- [ ] **Step 1: Write failing artifact and session-incompatibility tests**

Add exact nested tampering cases at both baseline and comparison entry points:

```python
@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["post_exit_recovery"].update(outcome="recovered"),
        lambda raw: raw["post_exit_recovery_samples"][0].update(
            elapsed_seconds=31.0
        ),
        lambda raw: raw["post_exit_recovery_samples"][0].update(
            previous_sample_identity="sha256:" + "0" * 64
        ),
    ),
)
def test_baseline_rejects_tampered_recovery_evidence(mutate):
    workload = build_canonical_workload()
    raw = _valid_recovered_raw_trial(workload).to_dict()
    mutate(raw)

    with pytest.raises(ValueError, match="recovery|identity"):
        validate_baseline_trial(
            RawTrial.from_dict(raw),
            workload=workload,
            source_commit=raw["source_commit"],
            harness_commit=raw["harness_commit"],
            harness_identity=raw["harness_identity"],
            expected_hardware=raw["hardware"],
            expected_software_versions=raw["software_versions"],
        )
```

Mirror one summary-identity and one sample-chain mutation inside a resigned comparison report. Add an old-session test that removes every new recovery-policy key and proves `BaselineJournal.open` rejects the schema-v2 session before any launch.

- [ ] **Step 2: Run artifact tests and verify version-2 assumptions fail**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -k \
  'tampered_recovery_evidence or old_journal_session or raw_trial_identity or comparison_rejects' \
  -v
```

Expected: FAIL until all artifact entry points require schema-v3 recovery evidence and the raw identity domain changes.

- [ ] **Step 3: Close every artifact and replay entry point**

Change `_raw_trial_identity` to:

```python
def _raw_trial_identity(trial: RawTrial) -> str:
    validate_raw_trial_evidence(trial)
    return structured_identity("sml-raw-benchmark-trial-v3", trial.to_dict())
```

Require `validate_raw_trial_evidence` before using a raw identity or performance value in baseline manifest construction/validation, raw JSONL read/write, comparison construction/validation, predecessor lookup/replay, phase validation, and final validation. Keep outer manifest/comparison schema versions unchanged; their identities change through workload and raw-trial identities. Retain explicit schema-v2 negative fixtures and eliminate accidental version-2 current fixtures.

- [ ] **Step 4: Update the operator README with exact recovery semantics**

Replace the schema-v2 lifecycle text and journal tree with:

```text
├── measurements/<metric>/<pair-index>/<attempt-index>.json
├── post-exit/<metric>/<pair-index>/<attempt-index>.json
├── recovery-samples/<metric>/<pair-index>/<attempt-index>/<sample-index>.json
├── recovery/<metric>/<pair-index>/<attempt-index>.json
├── inflight/<metric>/<pair-index>/<attempt-index>.json
```

Document these exact rules:

- immediate memory remains a version-1 identity-bound observation;
- immediate normal writes `not-required` with zero samples;
- immediate warning samples the complete environment every five seconds for at most five minutes and requires 30 continuous normal seconds;
- every sample is written before classification, warning resets only the stability window, and critical/non-memory failure terminates immediately;
- schema-v3 raw trials embed the ordered sample chain and summary;
- timeout/critical/interrupted memory stops for manual resume, while thermal-only failure retains automatic thermal recovery;
- a crash during warning recovery becomes `interrupted` and never continues its old stability window;
- schema-v2 state at `/private/tmp/sml-v2-baseline-post-exit-state-aa6bb43` is diagnostic only; and
- no baseline starts automatically after implementation.

- [ ] **Step 5: Run the complete benchmark unit module**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -q
```

Expected: PASS with all valid current evidence at schema version 3 and explicit old-schema rejection tests retained.

- [ ] **Step 6: Commit artifact enforcement and documentation**

```bash
git add v2/benchmarks/runner.py v2/benchmarks/README.md v2/tests/unit/test_benchmark_analysis.py
git commit -m "docs(v2): document post-exit memory recovery"
```

### Task 7: Verify the complete v2 tree and refresh the detached harness

**Files:**
- Verify: `v2/benchmarks/schema.py`
- Verify: `v2/benchmarks/evidence.py`
- Verify: `v2/benchmarks/workload.py`
- Verify: `v2/benchmarks/recovery.py`
- Verify: `v2/benchmarks/runner.py`
- Verify: `v2/benchmarks/journal.py`
- Verify: `v2/tests/unit/test_benchmark_analysis.py`
- Verify: `v2/benchmarks/README.md`
- Refresh: `/private/tmp/sml-v2-baseline-resume-harness`
- Preserve: `/private/tmp/sml-v2-baseline-post-exit-state-aa6bb43`
- Preserve: `/private/tmp/sml-v2-baseline-short-state`

**Interfaces:**
- Consumes: all prior tasks and the repository's v2 verification contract.
- Produces: a Ruff-clean, fully tested implementation commit and a clean detached measurement checkout whose full commit and harness content identity are reported.
- Does not produce: a baseline manifest, raw baseline JSONL, new baseline state directory, or modification to retained journals.

- [ ] **Step 1: Run Ruff check and format verification**

```bash
uv run ruff check v2
uv run ruff format --check v2
```

Expected: both commands exit 0. If formatting is required, run `uv run ruff format v2`, inspect the exact diff, rerun both checks, and commit only the mechanical formatting changes:

```bash
git add v2
git commit -m "style(v2): format post-exit recovery"
```

- [ ] **Step 2: Run the complete v2 test suite outside the sandbox**

```bash
uv run pytest v2/tests
```

Expected: all tests pass with zero failures or errors and no skips introduced by this change. Preserve the complete final pytest summary as completion evidence.

- [ ] **Step 3: Verify repository and retained evidence before harness refresh**

```bash
git status --short
git log -1 --oneline
test -d /private/tmp/sml-v2-baseline-post-exit-state-aa6bb43
test -d /private/tmp/sml-v2-baseline-short-state
```

Expected: the versioned worktree is clean, HEAD is the verified implementation commit, and both diagnostic state directories still exist unchanged.

- [ ] **Step 4: Refresh the detached measurement checkout at verified HEAD**

Use Git worktree operations only; do not delete any state directory:

```bash
git worktree remove --force /private/tmp/sml-v2-baseline-resume-harness
git worktree add --detach /private/tmp/sml-v2-baseline-resume-harness HEAD
git -C /private/tmp/sml-v2-baseline-resume-harness status --short
git -C /private/tmp/sml-v2-baseline-resume-harness rev-parse HEAD
```

Expected: the detached status is empty and its full commit equals verified main HEAD. If the checkout is already absent, omit only the failing remove command and add it exactly once.

- [ ] **Step 5: Compute and report the new harness content identity**

```bash
uv run python -c 'from pathlib import Path; from v2.benchmarks.workload import harness_content_identity; print(harness_content_identity(Path("/private/tmp/sml-v2-baseline-resume-harness")))'
```

Expected: one `sha256:` identity. Report the implementation commit, harness identity, Ruff results, full pytest count, both preserved state directories, and the fact that no baseline was started. A replacement baseline requires a new empty state directory, passing live preflight, and explicit user authorization.
