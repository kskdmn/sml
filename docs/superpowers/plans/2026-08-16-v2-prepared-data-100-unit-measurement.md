# V2 Prepared-Data 100-Unit Protocol and Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a versioned benchmark protocol that measures prepared-data with 100 real 1,024-token batches, capture a completely fresh 45-trial baseline, and publish Phase 2 evidence only if every unchanged acceptance gate passes.

**Architecture:** Keep the global measured-unit default at 20 and add a prepared-data-specific canonical count of 100. Baseline validators dispatch between immutable version-1/20 and version-2/100 contracts, while controllers pass each metric's canonical count into a child that verifies it instead of silently replacing it. Commit and independently review the harness before a fresh resumable baseline capture; only then run the non-resumable prepared-data comparison against that new baseline.

**Tech Stack:** Python 3.12.13, `uv`, pytest, Ruff, MLX/Metal, NumPy, the existing `v2.benchmarks` runner/evidence/journal stack, Git, and macOS environment probes.

## Global Constraints

- Work on the user-authorized `main` checkout; do not create a linked worktree unless that authorization changes.
- Do not edit top-level files, including `pyproject.toml` and `uv.lock`.
- Run every `uv run pytest` command outside the sandbox so MLX/Metal can access the Apple GPU.
- Keep `DEFAULT_MEASURED_UNITS = 20`; add `PREPARED_DATA_MEASURED_UNITS = 100`.
- Prepared-data remains microbatch size 1, sequence length 1,024, fixed canonical row order, real native loader consumption, consumer-side MLX transfer/evaluation, and stream closure before the measured call returns.
- Preserve the other canonical counts exactly: pretraining compute/end-to-end/SWAG/checkpoint 20, inference prefill/decode 32, cold compile 1, and peak memory 1.
- Preserve 5 warmup units for non-compile metrics and 0 for cold compile.
- Existing version-1 baseline artifacts remain immutable and valid only for prepared-data 20.
- New version-2 baseline artifacts use identity domain `sml-performance-baseline-v2`, an exact `prepared_data_measured_units` protocol field, and prepared-data 100.
- Capture all 45 version-2 baseline trials from scratch: 5 trials for each of 9 metrics against source commit `3687f8b3214a44c675ae67af52e4997762f6c634`.
- Never copy, migrate, salvage, or edit an old raw trial, journal slot, comparison trial, or evidence report.
- Require AC power, automatic power mode, Low Power Mode off, thermal state nominal/raw `0`, normal memory pressure, and no competing GPU workload.
- Baseline capture may resume only its own compatible external journal after an environment interruption. Phase 2 comparison is non-resumable and permits only the harness-owned noise retry/cooldown.
- Preserve screen mode, 5 pairs, 10,000 bootstrap resamples, ratio floor `0.97`, ratio MAD ceiling `0.02`, report-only lower bound, and predecessors `{"prepared-data":null}`.
- Do not validate, stage, or commit rejected Phase 2 output. Do not relax a threshold. Do not push automatically.
- Each of the harness, baseline evidence, and Phase 2 evidence commits requires an independent review. Finish with one broad review of the complete range.

---

## File Responsibility Map

- `v2/benchmarks/workload.py`: defines the canonical per-metric count map and builds either the legacy prepared-data-20 workload or the new prepared-data-100 workload.
- `v2/benchmarks/runner.py`: owns baseline-version dispatch, exact protocol/identity validation, CLI resolution, controller-to-child count forwarding, comparison construction, and validation.
- `v2/tests/unit/test_benchmark_analysis.py`: pins both protocol versions, CLI behavior, parent/child forwarding, journal identity, artifact validation, and tamper rejection.
- `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json`: new independently validated version-2 baseline manifest.
- `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl`: new 45-trial version-2 raw baseline evidence.
- `v2/benchmarks/results/phase-2-loader.json`: accepted prepared-data comparison against the version-2 baseline.
- `v2/benchmarks/results/phase-2.json`: independently validated Phase 2 result.
- `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/`: ignored reports, review packages, the preserved rejected 20-unit comparison, and execution ledgers only.

---

### Task 1: Implement and Review the Versioned 100-Unit Harness

**Files:**
- Modify: `v2/benchmarks/workload.py:71-72,320-420,411-590`
- Modify: `v2/benchmarks/runner.py:50-65,174-470,736-900,957-995,1250-1450,2022-2110,2209-2290,2700-3160,3260-3645,3770-3915`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:237-300,807-850,2268-2450,2540-3300,3280-3400,4800-5250,5800-5940,6540-6725`
- Report: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-1-report.md`
- Review package: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-1-review-package.md`

**Interfaces:**
- Consumes: `CanonicalWorkload`, `RawTrial`, `build_session_document(...)`, `measure_native_process(...)`, existing evidence/journal/recovery primitives, and immutable version-1 baseline files.
- Produces: `PREPARED_DATA_MEASURED_UNITS: int = 100`; `build_canonical_workload(*, prepared_data_measured_units: int = PREPARED_DATA_MEASURED_UNITS, ...) -> CanonicalWorkload`; private runner helpers `_baseline_workload(version: int) -> CanonicalWorkload`, `_baseline_protocol(version: int, workload: CanonicalWorkload) -> dict`, `_baseline_identity_domain(version: int) -> str`, and `_resolve_prepared_data_measure(args: argparse.Namespace, baseline: dict) -> int`; explicit `--prepared-data-measure` CLI flow; version-aware manifest/comparison validators.

- [ ] **Step 1: Create the ignored SDD workspace and verify the exact base**

```bash
bash /Users/keisukedaimon/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development/scripts/sdd-workspace \
  docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md
git rev-parse HEAD
git status --short
git diff --exit-code -- uv.lock
test -f .superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json
```

Expected: `HEAD` is the committed approved-plan base, status is empty, `uv.lock` is unchanged, and the rejected 20-unit report remains ignored and untouched. Record the full SHA in the task report.

- [ ] **Step 2: Write failing canonical-workload and parser tests**

Update the canonical map assertion and add version-specific construction tests:

```python
def test_canonical_workload_changes_only_prepared_data_to_one_hundred_units():
    legacy = build_canonical_workload(prepared_data_measured_units=20)
    current = build_canonical_workload()
    legacy_raw = legacy.to_dict()
    current_raw = current.to_dict()

    legacy_units = {
        unit["metric"]: unit.pop("measured_units")
        for unit in legacy_raw["work_units"]
    }
    current_units = {
        unit["metric"]: unit.pop("measured_units")
        for unit in current_raw["work_units"]
    }

    assert legacy_raw == current_raw
    assert legacy_units == {
        "prepared-data": 20,
        "pretraining-compute": 20,
        "pretraining-end-to-end": 20,
        "swag-end-to-end": 20,
        "inference-prefill": 32,
        "inference-decode": 32,
        "checkpoint-pause": 20,
        "compile-cold-start": 1,
        "peak-metal-memory": 1,
    }
    assert current_units == {**legacy_units, "prepared-data": 100}


def test_benchmark_parser_exposes_explicit_prepared_data_count():
    baseline = build_parser().parse_args(
        [
            "record-baseline",
            "--source-commit", benchmark_runner.PINNED_BASELINE_SOURCE_COMMIT,
            "--manifest", "manifest.json",
            "--raw-output", "raw.jsonl",
            "--state-directory", "state",
            "--prepared-data-measure", "100",
        ]
    )
    comparison = build_parser().parse_args(
        [
            "compare",
            "--baseline", "manifest.json",
            "--candidate", "HEAD",
            "--metrics", "prepared-data",
            "--predecessors", '{"prepared-data":null}',
            "--output", "report.json",
        ]
    )

    assert (baseline.measure, baseline.prepared_data_measure) == (20, 100)
    assert (comparison.measure, comparison.prepared_data_measure) == (20, None)
```

Also assert parsing `record-baseline` without `--prepared-data-measure` exits through argparse and retain the current parser-default assertions for pairs 5, warmup 5, and global measure 20.

- [ ] **Step 3: Run the canonical/parser tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_changes_only_prepared_data_to_one_hundred_units \
  v2/tests/unit/test_benchmark_analysis.py::test_benchmark_parser_exposes_explicit_prepared_data_count \
  -v
```

Expected: FAIL because `PREPARED_DATA_MEASURED_UNITS`, the workload argument, and the CLI option do not exist.

- [ ] **Step 4: Add the canonical count and CLI surface**

In `workload.py`, implement these signatures and use the argument only for the prepared-data work unit:

```python
WARMUP_UNITS = 5
DEFAULT_MEASURED_UNITS = 20
PREPARED_DATA_MEASURED_UNITS = 100


def _work_units(
    request_count: int,
    *,
    prepared_data_measured_units: int,
) -> tuple[WorkUnitDefinition, ...]:
    measured_units = {
        "prepared-data": prepared_data_measured_units,
        "inference-prefill": request_count,
        "inference-decode": request_count,
        "compile-cold-start": 1,
        "peak-metal-memory": 1,
    }
    return tuple(
        WorkUnitDefinition(
            metric=metric,
            direction=direction,
            numerator=numerator,
            work_unit=work_unit,
            start_boundary=start_boundary,
            end_boundary=end_boundary,
            measured_units=measured_units.get(metric, DEFAULT_MEASURED_UNITS),
        )
        for metric, direction, numerator, work_unit, start_boundary, end_boundary
        in definitions
    )


def build_canonical_workload(
    *,
    model_overrides: dict[str, JsonValue] | None = None,
    optimizer_overrides: dict[str, JsonValue] | None = None,
    loader_overrides: dict[str, JsonValue] | None = None,
    generation_overrides: dict[str, JsonValue] | None = None,
    row_count: int = 968,
    prepared_data_measured_units: int = PREPARED_DATA_MEASURED_UNITS,
) -> CanonicalWorkload:
    if (
        type(prepared_data_measured_units) is not int
        or prepared_data_measured_units <= 0
    ):
        raise ValueError("prepared_data_measured_units must be a positive integer")

    work_units = _work_units(
        int(generation["request_count"]),
        prepared_data_measured_units=prepared_data_measured_units,
    )
```

Insert the validation after the existing override merges and pass `work_units` into the existing `CanonicalWorkload` construction; every other constructor field remains byte-for-byte covered by the normalized equality test. In `runner.py`, import the new constant. Add required `--prepared-data-measure` to `record-baseline`, optional/default-`None` `--prepared-data-measure` to `compare`, and required hidden `--prepared-data-measure` to `_run-process`.

- [ ] **Step 5: Run the canonical/parser tests to verify GREEN**

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_changes_only_prepared_data_to_one_hundred_units \
  v2/tests/unit/test_benchmark_analysis.py::test_benchmark_parser_exposes_explicit_prepared_data_count \
  -v
```

Expected: PASS; the normalized workload differs only at prepared-data measured units.

- [ ] **Step 6: Write failing baseline-version and tamper tests**

Add helpers that build complete manifests for both versions, then pin these cases:

```python
def _baseline_fixture(version):
    prepared_units = 20 if version == 1 else 100
    workload = build_canonical_workload(
        prepared_data_measured_units=prepared_units
    )
    trials = _valid_baseline_trials(workload)
    manifest = build_baseline_manifest(
        trials=trials,
        workload=workload,
        workload_identity=canonical_workload_identity(workload),
        source_commit=trials[0].source_commit,
        harness_commit=trials[0].harness_commit,
        harness_identity=trials[0].harness_identity,
        command="record-baseline",
        pairs=5,
        warmup_units=5,
        measured_units=20,
        paired_representations=_valid_paired_representations(workload),
        baseline_version=version,
    )
    return manifest, trials


@pytest.mark.parametrize(
    ("version", "prepared_units", "domain", "has_explicit_field"),
    [
        (1, 20, "sml-performance-baseline-v1", False),
        (2, 100, "sml-performance-baseline-v2", True),
    ],
)
def test_baseline_versions_bind_their_exact_prepared_data_protocol(
    version, prepared_units, domain, has_explicit_field
):
    manifest, trials = _baseline_fixture(version)

    validate_baseline_manifest(manifest, trials)
    assert manifest["version"] == version
    assert manifest["identity"] == structured_identity(
        domain, {key: value for key, value in manifest.items() if key != "identity"}
    )
    assert (
        "prepared_data_measured_units" in manifest["protocol"]
    ) is has_explicit_field
    assert next(
        unit["measured_units"]
        for unit in manifest["canonical_workload"]["work_units"]
        if unit["metric"] == "prepared-data"
    ) == prepared_units


@pytest.mark.parametrize(
    ("version", "prepared_units"), [(1, 100), (2, 20), (2, 99)]
)
def test_baseline_validator_rejects_cross_version_prepared_data_counts(
    version, prepared_units
):
    manifest, trials = _baseline_fixture(version=version)
    tampered = deepcopy(manifest)
    next(
        unit for unit in tampered["canonical_workload"]["work_units"]
        if unit["metric"] == "prepared-data"
    )["measured_units"] = prepared_units
    _resign_baseline(tampered)

    with pytest.raises(ValueError, match="pinned workload"):
        validate_baseline_manifest(tampered, trials)
```

Update the existing `_resign_baseline` helper to choose `_baseline_identity_domain(manifest["version"])`. Add separate resigned-tamper cases for: removing/adding `prepared_data_measured_units`, setting it to 20/99/`True`, changing a prepared-data raw trial to 20, changing an unchanged metric away from its existing count, swapping version 1/2, using the wrong identity domain, and giving a version-1 manifest the version-2 workload. Explicitly load and validate the committed `baseline-3687f8b.json` plus JSONL to prove legacy compatibility.

- [ ] **Step 7: Run the baseline-version tests to verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "baseline and (version or prepared_data or legacy or tamper)" -v
```

Expected: new tests FAIL because manifest construction and validation support only version 1 and the current default workload.

- [ ] **Step 8: Implement one baseline-version dispatch path**

In `runner.py`, add exact version helpers used by both full and header-only validation:

```python
BASELINE_VERSION_LEGACY = 1
BASELINE_VERSION_PREPARED_DATA_100 = 2


def _baseline_identity_domain(version: int) -> str:
    if version == BASELINE_VERSION_LEGACY:
        return "sml-performance-baseline-v1"
    if version == BASELINE_VERSION_PREPARED_DATA_100:
        return "sml-performance-baseline-v2"
    raise ValueError("unsupported baseline manifest kind or version")


def _baseline_workload(version: int) -> CanonicalWorkload:
    if version == BASELINE_VERSION_LEGACY:
        return build_canonical_workload(prepared_data_measured_units=20)
    if version == BASELINE_VERSION_PREPARED_DATA_100:
        return build_canonical_workload()
    raise ValueError("unsupported baseline manifest kind or version")


def _baseline_protocol(version: int, workload: CanonicalWorkload) -> dict:
    protocol = {
        "pairs": SCREEN_PAIRS,
        "compilation_passes": 1,
        "warmup_units": WARMUP_UNITS,
        "measured_units": DEFAULT_MEASURED_UNITS,
        "bootstrap_seed": 1729,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "synchronization_boundaries": list(workload.synchronization_boundaries),
    }
    if version == BASELINE_VERSION_PREPARED_DATA_100:
        protocol["prepared_data_measured_units"] = (
            PREPARED_DATA_MEASURED_UNITS
        )
    return protocol
```

Add required `baseline_version: int` to `build_baseline_manifest(...)`; construct the protocol only through `_baseline_protocol`; sign with `_baseline_identity_domain`. Refactor common manifest header/workload/protocol checks into one private helper used by both `validate_baseline_manifest(...)` and `_validate_baseline_document(...)`. It must compare exact field sets, exact workload equality, and exact protocol equality before accepting raw evidence.

In the new capture path, always use version 2 and `build_canonical_workload()`; do not add a CLI switch that can create version 1.

- [ ] **Step 9: Run baseline-version tests to verify GREEN**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "baseline and (version or prepared_data or legacy or tamper)" -v
```

Expected: PASS for the committed version-1 baseline and generated version-2 fixtures; every resigned mismatch is rejected.

- [ ] **Step 10: Write failing journal/replay protocol-binding tests**

Add tests proving version-2 sessions and replay commands carry both counts:

```python
def _version_two_session(tmp_path):
    workload = build_canonical_workload()
    return build_session_document(
        harness_commit="a" * 40,
        harness_identity="sha256:" + "b" * 64,
        source_commit=benchmark_runner.PINNED_BASELINE_SOURCE_COMMIT,
        canonical_workload=workload,
        canonical_workload_identity=canonical_workload_identity(workload),
        protocol=benchmark_runner._baseline_protocol(2, workload),
        hardware={"chip": "Apple M5"},
        software_versions={"python": "3.12.13", "mlx": "0.32.0"},
        paired_representations={"canonical_row_identity": "sha256:" + "c" * 64},
        manifest_path=tmp_path / "baseline.json",
        raw_output_path=tmp_path / "baseline.jsonl",
    )


def _resign_session(session):
    body = {key: value for key, value in session.items() if key != "identity"}
    session["identity"] = structured_identity(
        "sml-baseline-journal-session-v1", body
    )


def test_version_two_baseline_session_and_replay_bind_prepared_data_units(tmp_path):
    workload = build_canonical_workload()
    protocol = benchmark_runner._baseline_protocol(2, workload)
    session = _session_document(tmp_path, protocol=protocol)
    journal = BaselineJournal.open(tmp_path / "state", session)

    assert journal.session["protocol"]["measured_units"] == 20
    assert journal.session["protocol"]["prepared_data_measured_units"] == 100
    assert "--measure 20" in benchmark_runner.canonical_baseline_command(journal)
    assert "--prepared-data-measure 100" in (
        benchmark_runner.canonical_baseline_command(journal)
    )


@pytest.mark.parametrize("changed", [20, 99, True])
def test_version_two_journal_rejects_prepared_data_protocol_changes(
    tmp_path, changed
):
    session = _version_two_session(tmp_path)
    BaselineJournal.open(tmp_path / "state", session)
    altered = deepcopy(session)
    altered["protocol"]["prepared_data_measured_units"] = changed
    _resign_session(altered)

    with pytest.raises(ValueError, match="session does not match expected session"):
        BaselineJournal.open(tmp_path / "state", altered)
```

Retain the existing tests for incompatible hardware, software, paths, paired representations, policy, and workload identity.

- [ ] **Step 11: Run the journal/replay tests to verify RED**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "session and prepared_data or replay and prepared_data" -v
```

Expected: FAIL because version-2 protocol creation and canonical replay do not yet include the dedicated option.

- [ ] **Step 12: Bind the version-2 baseline session and publication**

Make `_record_baseline_locked(...)` reject anything except metrics `METRIC_NAMES`, pairs 5, warmup 5, global measure 20, and prepared-data measure 100. Build the new workload and protocol, pass the canonical metric count—not the global count—to each child, and publish with baseline version 2:

```python
workload = build_canonical_workload()
protocol = _baseline_protocol(BASELINE_VERSION_PREPARED_DATA_100, workload)

measured_units_by_metric = {
    unit.metric: unit.measured_units for unit in workload.work_units
}

launch_arguments = {
    "metric": slot.metric,
    "warmup": args.warmup,
    "measure": measured_units_by_metric[slot.metric],
    "prepared_data_measure": args.prepared_data_measure,
}
```

Pass the four entries in `launch_arguments` together with the existing explicit source, harness, side, pair/order, comparison-target, session, evidence-path, recovery-path, and output-path arguments already supplied at the call site. Make `canonical_baseline_command(...)` append `--prepared-data-measure` only when that exact field exists in the session protocol, preserving version-1 replay shape. Add `baseline_version` to `publish_baseline_from_journal(...)` and pass version 2 from the record path. The journal schema/version may remain 1 because its identity already binds the complete protocol object; do not weaken exact session equality.

- [ ] **Step 13: Run journal/replay tests to verify GREEN**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "session or replay or publication or record_baseline" -v
```

Expected: PASS, including all pre-existing resume, atomicity, lock, and recovery tests.

- [ ] **Step 14: Write failing controller/child exact-count tests**

Change the existing child forwarding parametrization to prepared-data 100 and add mismatch-before-adapter tests:

```python
@pytest.mark.parametrize(
    ("metric", "child_measure", "expected"),
    [
        ("prepared-data", 100, 100),
        ("pretraining-compute", 20, 20),
        ("inference-prefill", 32, 32),
        ("compile-cold-start", 1, 1),
        ("peak-metal-memory", 1, 1),
    ],
)
def test_child_process_forwards_each_canonical_metric_count(
    tmp_path, monkeypatch, metric, child_measure, expected
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = child_measure
    args.prepared_data_measure = 100
    captured = {}
    _stub_single_process_measurement(monkeypatch, args, captured)

    benchmark_runner._run_single_process(args)

    assert captured["measured_units"] == expected


@pytest.mark.parametrize(
    ("metric", "supplied"),
    [("prepared-data", 20), ("prepared-data", 99), ("pretraining-compute", 100)],
)
def test_child_rejects_noncanonical_count_before_adapter_import(
    tmp_path, monkeypatch, metric, supplied
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.measure = supplied
    args.prepared_data_measure = 100
    _stub_single_process_measurement(monkeypatch, args)
    resolved = []
    monkeypatch.setattr(
        legacy,
        "resolve_native_workload",
        lambda *args: resolved.append(args),
    )

    with pytest.raises(ValueError, match="child measured units do not match"):
        benchmark_runner._run_single_process(args)

    assert resolved == []
```

Add a parent test that captures `_launch_trial` calls for all metrics and asserts `{metric: measure}` equals the canonical per-metric map. Add a comparison-session identity test showing changing only prepared-data 100 to 20 changes the identity.

- [ ] **Step 15: Run controller/child tests to verify RED**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "child and (canonical or count) or parent and metric_count or evidence_session and prepared_data" -v
```

Expected: prepared-data forwarding still yields 20 or ignores the supplied mismatch, so the new assertions FAIL.

- [ ] **Step 16: Implement exact parent-to-child count flow**

Thread `prepared_data_measure: int` through `_comparison_evidence_session_identity(...)`, `_run_paired_trials(...)`, and `_launch_trial(...)`. Pass the selected metric's `WorkUnitDefinition.measured_units` as `--measure` and the workload-wide prepared count as `--prepared-data-measure`.

At the start of `_run_single_process(...)`, before adapter import or native workload construction:

```python
workload = build_canonical_workload(
    prepared_data_measured_units=args.prepared_data_measure
)
work_unit = next(unit for unit in workload.work_units if unit.metric == args.metric)
if type(args.measure) is not int or args.measure != work_unit.measured_units:
    raise ValueError("child measured units do not match the canonical metric")
measured_units = args.measure
```

Pass this verified `measured_units` unchanged into `measure_native_process(...)` and raw trial publication. Keep adapter import and `resolve_native_workload(...)` after the verification so the ordering test is meaningful. Ensure `_launch_trial(...)` reconstructs/validates returned raw evidence against the same workload instead of the new default by accident.

- [ ] **Step 17: Run controller/child tests to verify GREEN**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "single_process or paired_trials or launch_trial or evidence_session" -v
```

Expected: PASS; prepared-data reaches the adapter as 100, all other metrics retain their exact counts, and mismatches fail before native work starts.

- [ ] **Step 18: Write failing comparison/version-resolution tests**

Add tests for inference, explicit matching, and fail-closed mixing:

```python
@pytest.mark.parametrize(
    ("version", "explicit", "expected"),
    [(1, None, 20), (1, 20, 20), (2, None, 100), (2, 100, 100)],
)
def test_compare_resolves_prepared_data_count_from_baseline(
    version, explicit, expected
):
    baseline, _ = _baseline_fixture(version=version)
    args = SimpleNamespace(prepared_data_measure=explicit)

    assert benchmark_runner._resolve_prepared_data_measure(args, baseline) == expected


@pytest.mark.parametrize(
    ("version", "explicit"), [(1, 100), (2, 20), (2, 99), (2, True)]
)
def test_compare_rejects_prepared_data_count_that_disagrees_with_baseline(
    version, explicit
):
    baseline, _ = _baseline_fixture(version=version)

    with pytest.raises(ValueError, match="prepared-data measured units"):
        benchmark_runner._resolve_prepared_data_measure(
            SimpleNamespace(prepared_data_measure=explicit), baseline
        )
```

Add version-1 and version-2 comparison construction/validation tests. Version 1 must retain the old comparison protocol field set; version 2 must add `prepared_data_measured_units: 100`. Add resigned tamper tests for protocol omission/addition/change, raw trial 20 under v2, raw trial 100 under v1, wrong baseline identity, wrong workload identity, and a v1 report validated against a v2 baseline. Add `validate-phase` and `validate-final` entry-point tests for the same cross-version rejection.

- [ ] **Step 19: Run comparison/version tests to verify RED**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "compare and prepared_data or comparison and version or phase and version or final and version" -v
```

Expected: FAIL because comparison validation hardcodes global measured units and does not derive the dedicated count from the baseline.

- [ ] **Step 20: Implement baseline-driven comparison protocol resolution**

Add this resolver and call it only after reading and validating the baseline:

```python
def _resolve_prepared_data_measure(
    args: argparse.Namespace, baseline: dict
) -> int:
    workload = _baseline_workload(baseline["version"])
    expected = next(
        unit.measured_units
        for unit in workload.work_units
        if unit.metric == "prepared-data"
    )
    supplied = args.prepared_data_measure
    if supplied is not None and (type(supplied) is not int or supplied != expected):
        raise ValueError(
            "prepared-data measured units do not match the baseline protocol"
        )
    return expected
```

Change `_compare(...)` ordering to: read baseline, validate baseline, resolve prepared-data count, resolve comparison mode/protocol, then create worktrees or trials. Change `_validate_comparison_protocol(report, baseline)` to require the exact field set derived from the baseline version. Build version-2 comparison reports with `prepared_data_measured_units: 100`; preserve version-1 report shape. Pass the resolved workload and count through every baseline/predecessor attempt. Make `validate_comparison_report`, `_validate_comparison_document`, `_validate_phase`, and `_validate_final` use the supplied baseline's versioned workload/protocol.

Keep global `--measure` immutable at 20. Supplying `--measure 100` must still fail before launch; only `--prepared-data-measure 100` changes prepared-data.

- [ ] **Step 21: Run comparison/version tests to verify GREEN**

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py \
  -k "compare or comparison or validate_phase or validate_final" -v
```

Expected: PASS for both baseline versions and every cross-version/tamper rejection.

- [ ] **Step 22: Run the complete focused benchmark test module**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -v
```

Expected: PASS with no warnings. Fix only failures caused by this task, preserving all recovery, filesystem, identity, and atomicity behavior.

- [ ] **Step 23: Run all final implementation gates**

Run outside the sandbox where applicable:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: Ruff passes, the complete v2 suite passes without warnings, the diff is clean, `uv.lock` is unchanged, and only the three owned code/test files plus this task's ignored report/package are present.

- [ ] **Step 24: Commit the reviewed implementation candidate**

```bash
git add \
  v2/benchmarks/workload.py \
  v2/benchmarks/runner.py \
  v2/tests/unit/test_benchmark_analysis.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat(v2): version prepared-data 100-unit protocol"
```

Expected: the commit contains exactly those three files. Record the full commit SHA and final test counts in `task-1-report.md`.

- [ ] **Step 25: Obtain an independent review and fix findings test-first**

Give a fresh reviewer the approved design, this task, report, full range from the recorded base through the implementation commit, and these questions:

```text
Verify that version 1 still validates prepared-data 20; version 2 requires prepared-data 100 and its exact protocol field/domain; no other metric count or workload field changed; parent and child use the same canonical metric count; every manifest, journal, replay, comparison, phase, and final validator rejects cross-version/tampered evidence before work starts; recovery and atomicity remain unchanged. Report Critical, Important, and Minor findings with file/line evidence, or explicitly state ready.
```

For every valid finding, use `superpowers:receiving-code-review`, add a focused failing test, run RED, apply the minimal fix, run GREEN, rerun Step 23, and amend the implementation commit. Repeat independent review until there are no Critical or Important findings. The controller must not implement review fixes itself.

---

### Task 2: Capture, Validate, Review, and Commit the Fresh Version-2 Baseline

**Files:**
- Create: `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json`
- Create: `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl`
- Preserve: `v2/benchmarks/manifests/baseline-3687f8b.json`
- Preserve: `v2/benchmarks/results/baseline-3687f8b.jsonl`
- External state: `/private/tmp/sml-v2-baseline-prepared100-state`
- Report: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-2-report.md`
- Review package: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-2-review-package.md`

**Interfaces:**
- Consumes: the reviewed Task 1 harness commit, version-2 `record-baseline`, pinned source commit `3687f8b3214a44c675ae67af52e4997762f6c634`, and the existing journal recovery state machine.
- Produces: one version-2 manifest signed in domain `sml-performance-baseline-v2` and one 45-line raw JSONL whose prepared-data trials record 100 and whose other trials retain their canonical counts.

- [ ] **Step 1: Verify the reviewed harness and immutable legacy evidence**

```bash
git rev-parse HEAD
git status --short
git diff --exit-code -- uv.lock
git diff --exit-code HEAD -- \
  v2/benchmarks/manifests/baseline-3687f8b.json \
  v2/benchmarks/results/baseline-3687f8b.jsonl
test ! -e v2/benchmarks/manifests/baseline-3687f8b-prepared100.json
test ! -e v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
```

Expected: exact reviewed Task 1 SHA, empty status, unchanged lockfile and legacy baseline, and absent version-2 outputs. Stop if the implementation review is not ready.

- [ ] **Step 2: Run the clean-candidate preflight**

Run pytest outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
git diff --exit-code -- uv.lock
```

Expected: every gate passes and the checkout remains clean.

- [ ] **Step 3: Allocate one fresh external journal directory and record it**

```bash
test ! -e /private/tmp/sml-v2-baseline-prepared100-state
mkdir /private/tmp/sml-v2-baseline-prepared100-state
test -d /private/tmp/sml-v2-baseline-prepared100-state
```

Expected: the exact new directory is empty. If it already exists before this task, stop and inspect it; do not adopt or delete unknown state. Record the path in `task-2-report.md` and reuse only that path for every legitimate resume of this capture. Never substitute an old baseline state directory and never delete this state while capture is incomplete or blocked.

- [ ] **Step 4: Require a 30-second nominal launch window**

Run outside the sandbox with `TMPDIR=/private/tmp`:

```bash
env TMPDIR=/private/tmp uv run python - <<'PY'
import json
import time

from v2.benchmarks.runner import collect_environment

started = time.monotonic()
deadline = started + 300.0
stable_since = None
samples = 0
while True:
    _, status, _ = collect_environment()
    now = time.monotonic()
    nominal = (
        status.get("power_connected") is True
        and status.get("power_mode") == "automatic"
        and status.get("low_power_mode") is False
        and status.get("thermal_state") == "nominal"
        and status.get("thermal_state_raw_value") == 0
        and status.get("memory_pressure") == "normal"
        and status.get("competing_gpu_workload") is False
    )
    stable_since = (
        now if nominal and stable_since is None
        else stable_since if nominal
        else None
    )
    stable = 0.0 if stable_since is None else now - stable_since
    print(json.dumps({"sample": samples, "stable_seconds": stable, **status}, sort_keys=True), flush=True)
    samples += 1
    if stable >= 30.0:
        raise SystemExit(0)
    if now >= deadline:
        raise SystemExit(2)
    time.sleep(1.0)
PY
```

Expected: exit 0 after 30 continuous nominal seconds. On exit 2, stop BLOCKED without launching capture; retain the empty state directory for the same task.

- [ ] **Step 5: Launch the exact full 45-trial capture**

Immediately after the gate, run outside the sandbox, substituting only the exact state path from Step 3:

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner record-baseline \
  --source-commit 3687f8b3214a44c675ae67af52e4997762f6c634 \
  --manifest v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --raw-output v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl \
  --state-directory /private/tmp/sml-v2-baseline-prepared100-state \
  --metrics prepared-data,pretraining-compute,pretraining-end-to-end,swag-end-to-end,inference-prefill,inference-decode,checkpoint-pause,compile-cold-start,peak-metal-memory \
  --pairs 5 \
  --warmup 5 \
  --measure 20 \
  --prepared-data-measure 100
```

Expected: the harness captures all 45 fresh legacy/reference trials and atomically publishes both artifacts. Let built-in thermal/environment recovery run. If the process is interrupted or an allowed recovery stops it, keep the same state directory and repeat Steps 1, 2, 4, and this exact command; `BaselineJournal.open` must prove exact compatibility before resuming. Never copy an accepted slot manually.

- [ ] **Step 6: Verify exact artifact shape before validation**

```bash
test "$(wc -l < v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl | tr -d ' ')" = 45
jq -e '
  .kind == "sml-performance-baseline"
  and .version == 2
  and .source.commit == "3687f8b3214a44c675ae67af52e4997762f6c634"
  and .protocol.measured_units == 20
  and .protocol.prepared_data_measured_units == 100
  and (.metrics | keys | length) == 9
' v2/benchmarks/manifests/baseline-3687f8b-prepared100.json
jq -s -e '
  length == 45
  and ([.[] | select(.metric == "prepared-data") | .measured_units] | length == 5 and all(. == 100))
  and ([.[] | select(.metric == "pretraining-compute" or .metric == "pretraining-end-to-end" or .metric == "swag-end-to-end" or .metric == "checkpoint-pause") | .measured_units] | all(. == 20))
  and ([.[] | select(.metric == "inference-prefill" or .metric == "inference-decode") | .measured_units] | all(. == 32))
  and ([.[] | select(.metric == "compile-cold-start" or .metric == "peak-metal-memory") | .measured_units] | all(. == 1))
  and ([.[].environment_status.thermal_state_raw_value] | all(. == 0))
' v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
```

Expected: exactly five records per metric with canonical counts and accepted nominal environment evidence. If any check fails, stop without staging.

- [ ] **Step 7: Independently validate the version-2 baseline**

Run outside the sandbox:

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate \
  --manifest v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --raw-input v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
```

Expected: exit 0. Record manifest identity, workload identity, harness commit/identity, source commit, command, protocol, per-metric trial identities, all environment dispositions, and the journal completion identity in `task-2-report.md`.

- [ ] **Step 8: Run post-capture repository gates**

Run pytest outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: tests and Ruff pass; only the two new baseline artifacts are untracked; `uv.lock` and legacy baseline files are unchanged.

- [ ] **Step 9: Independently review baseline evidence before commit**

Give a fresh reviewer the approved design, Task 2 report, manifest, raw JSONL, journal `session.json`/`completed.json`, and Task 1 harness SHA. Require this verdict:

```text
Verify 45 unique accepted slots; exact version-2 identity domain and protocol field set; prepared-data 100 and unchanged counts for eight other metrics; source/harness/workload/session identities; nominal accepted environment evidence; no old-trial reuse; valid atomic publication; unchanged version-1 artifacts. Report Critical, Important, and Minor findings, or explicitly state ready.
```

Any Critical or Important finding blocks publication. Do not edit measurement evidence to answer a finding; recapture from a new empty state directory only if the finding proves the entire capture invalid.

- [ ] **Step 10: Commit exactly the accepted baseline artifacts**

```bash
git add \
  v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
git diff --cached --check
git diff --cached --stat
git commit -m "bench(v2): record prepared100 baseline"
git show --check --stat --oneline --summary HEAD
git status --short
git diff --exit-code HEAD^ -- uv.lock
```

Expected: exactly the version-2 manifest and 45-trial JSONL are committed, status is empty, and `uv.lock` is unchanged.

---

### Task 3: Run, Validate, Review, and Commit Phase 2 Against Version 2

**Files:**
- Create on success: `v2/benchmarks/results/phase-2-loader.json`
- Create on success: `v2/benchmarks/results/phase-2.json`
- Preserve: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json`
- Report: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-3-report.md`
- Review package: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-3-review-package.md`

**Interfaces:**
- Consumes: reviewed Task 1 harness embedded in the reviewed/committed version-2 baseline, Task 2 manifest, clean candidate `HEAD`, and version-aware compare/validate-phase commands.
- Produces on success: one passing prepared-data comparison report and one independently validated Phase 2 report, both bound to prepared-data 100 and predecessors `{"prepared-data": null}`.

- [ ] **Step 1: Verify the clean accepted base and evidence isolation**

```bash
git rev-parse HEAD
git status --short
git diff --exit-code -- uv.lock
test -f v2/benchmarks/manifests/baseline-3687f8b-prepared100.json
test -f v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
test -f .superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json
test ! -e v2/benchmarks/results/phase-2-loader.json
test ! -e v2/benchmarks/results/phase-2.json
```

Expected: exact reviewed Task 2 commit, empty status, accepted version-2 baseline present, rejected 20-unit report preserved only in ignored diagnostics, and both public Phase 2 paths absent.

- [ ] **Step 2: Run the complete preflight**

Run outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate \
  --manifest v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --raw-input v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
git status --short
git diff --exit-code -- uv.lock
```

Expected: all gates pass and the checkout remains clean.

- [ ] **Step 3: Require a fresh 30-second nominal launch window**

Run the exact transient gate from Task 2 Step 4 outside the sandbox with `TMPDIR=/private/tmp`.

Expected: exit 0 after 30 continuous nominal seconds. If it exits 2 or a probe fails, stop BLOCKED before comparison; do not operator-rerun within the same task turn.

- [ ] **Step 4: Launch the exact non-resumable comparison once**

Immediately after the passing gate, run outside the sandbox:

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner compare \
  --baseline v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --candidate HEAD \
  --metrics prepared-data \
  --mode screen \
  --pairs 5 \
  --warmup 5 \
  --measure 20 \
  --prepared-data-measure 100 \
  --bootstrap-resamples 10000 \
  --minimum-ratio 0.97 \
  --maximum-dispersion 0.02 \
  --lower-bound-report-only \
  --predecessors '{"prepared-data":null}' \
  --output v2/benchmarks/results/phase-2-loader.json
```

Expected: exit 0 and final decision `pass`. Leave it uninterrupted. The harness may perform its single noise retry and cooldown. If the command fails, final decision is `fail`/`too-noisy`, or environment evidence is rejected, preserve the untracked report as diagnostics, stop BLOCKED, and do not validate, stage, commit, or operator-rerun.

- [ ] **Step 5: Inspect exact passing protocol and statistics**

```bash
jq -e '
  .comparison_mode == "screen"
  and .protocol.pairs == 5
  and .protocol.warmup_units == 5
  and .protocol.measured_units == 20
  and .protocol.prepared_data_measured_units == 100
  and .protocol.bootstrap_resamples == 10000
  and .protocol.minimum_ratio == 0.97
  and .protocol.maximum_dispersion == 0.02
  and .protocol.require_lower_bound == false
  and .predecessors == {"prepared-data": null}
  and .metrics["prepared-data"].baseline_comparison.decision == "pass"
  and .metrics["prepared-data"].baseline_comparison.median_ratio >= 0.97
  and .metrics["prepared-data"].baseline_comparison.ratio_mad <= 0.02
  and ([.raw_trials[].measured_units] | length > 0 and all(. == 100))
  and ([.raw_trials[].native_configuration.sequence_length] | all(. == 1024))
  and ([.raw_trials[].native_configuration.microbatch_size] | all(. == 1))
' v2/benchmarks/results/phase-2-loader.json
```

Also inspect every trial's start, end, merged, post-exit, and recovery-final thermal/raw values; power mode; AC; Low Power Mode; memory pressure; competing GPU state; and any retry/cooldown evidence. All accepted observations must satisfy the existing validators.

- [ ] **Step 6: Independently validate Phase 2**

Run outside the sandbox:

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json \
  --output v2/benchmarks/results/phase-2.json
```

Expected: exit 0 and `phase-2.json` exactly validates the passing comparison. Record both identities, candidate commit, baseline identity, harness identity, workload identity, ratio, MAD, confidence bound, attempt count, and cooldown evidence in `task-3-report.md`. On validation failure, stop without staging either file.

- [ ] **Step 7: Run post-measurement gates and inspect the exact diff**

Run pytest outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: only the two public Phase 2 JSON files are untracked; all code gates pass and `uv.lock` is unchanged.

- [ ] **Step 8: Independently review Phase 2 evidence before commit**

Give a fresh reviewer the approved design, Task 3 report, version-2 baseline, both Phase 2 files, and the Task 1/Task 2 commit SHAs. Require this verdict:

```text
Verify exact baseline/harness/workload lineage; prepared-data 100 in protocol and every trial; 5 complete alternating pairs for each recorded attempt; unchanged 0.97 ratio and 0.02 MAD gates; report-only lower bound; exact null predecessor; nominal environment evidence; valid harness-owned retry/cooldown if present; independent validate-phase success. Report Critical, Important, and Minor findings, or explicitly state ready.
```

Any Critical or Important finding blocks acceptance. Never edit a measured report to answer review feedback.

- [ ] **Step 9: Commit exactly the accepted Phase 2 evidence**

```bash
git add \
  v2/benchmarks/results/phase-2-loader.json \
  v2/benchmarks/results/phase-2.json
git diff --cached --check
git diff --cached --stat
git commit -m "bench(v2): accept prepared100 phase 2"
git show --check --stat --oneline --summary HEAD
git status --short
git diff --exit-code HEAD^ -- uv.lock
```

Expected: exactly two accepted reports are committed, status is empty, and `uv.lock` is unchanged. Do not push.

---

### Task 4: Perform the Final Broad Review and Handoff

**Files:**
- Verify: `v2/benchmarks/workload.py`
- Verify: `v2/benchmarks/runner.py`
- Verify: `v2/tests/unit/test_benchmark_analysis.py`
- Verify: `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json`
- Verify: `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl`
- Verify: `v2/benchmarks/results/phase-2-loader.json`
- Verify: `v2/benchmarks/results/phase-2.json`
- Report: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/final-review.md`

**Interfaces:**
- Consumes: the independently reviewed Task 1, Task 2, and Task 3 commits.
- Produces: a read-only final verdict over the complete committed range and a handoff that names exact commits, test evidence, benchmark statistics, and whether a push remains pending.

- [ ] **Step 1: Run the definitive repository verification**

Run pytest outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate \
  --manifest v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --raw-input v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b-prepared100.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json
git status --short
git diff --exit-code -- uv.lock
```

Expected: every command passes, status is empty, and `uv.lock` is unchanged.

- [ ] **Step 2: Dispatch one fresh broad reviewer**

Provide the approved design and full range from the pre-Task-1 base through the Phase 2 evidence commit. Require review of code, tests, baseline artifacts, Phase 2 artifacts, commit boundaries, and all ignored task reports. The reviewer must explicitly answer:

```text
Does the range preserve valid v1/20 evidence, implement v2/100 without changing any other metric, prove exact parent/child counts, publish a fully fresh 45-trial baseline, accept Phase 2 only under the original statistics/environment gates, preserve evidence isolation, and avoid unrelated/top-level changes? Report Critical, Important, and Minor findings with evidence, or state ready.
```

Do not declare completion while any Critical or Important finding remains. Code findings return to Task 1's test-first fix/review loop and require recapturing downstream evidence if they alter any harness content or canonical identity. Evidence findings return to the owning capture task and may require fresh evidence; never patch JSON by hand.

- [ ] **Step 3: Record the final handoff without pushing**

Write `final-review.md` with: full implementation/base/evidence commit SHAs, final test count, Ruff results, v1 and v2 baseline identities, v2 harness/workload identities, Phase 2 identities, median ratio, ratio MAD, confidence bound, attempt/cooldown evidence, environment summary, reviewer verdict, clean status, unchanged `uv.lock`, and `push pending user request`.

Report the same concise outcome to the user. Do not push unless the user explicitly asks.
