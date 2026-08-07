# V2 Shorter Benchmark Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shorten the immutable v2 benchmark protocol to five warmups and 20 default measured units, with one measured peak-memory optimizer step, while preserving the full 1,024-token workload and every existing acceptance gate.

**Architecture:** Define the shared global counts in `v2.benchmarks.workload`, where the canonical identity-bearing workload is built, and import those counts into the runner so CLI defaults, raw-trial validation, sessions, manifests, comparison reports, and replay checks cannot drift. Keep the exact metric-specific measured counts in `CanonicalWorkload.work_units`; the child process continues to obtain its timed count from that identity-bound map rather than trusting the global CLI value.

**Tech Stack:** Python 3.12.13, MLX/Metal, `argparse`, immutable dataclass records, JSON/JSONL benchmark evidence, pytest, Ruff, Git worktrees

## Global Constraints

- Keep the model at 12 layers and hidden size 768, and keep the complete existing model configuration.
- Keep training sequence length 1,024, microbatch size 1, and gradient accumulation 8.
- Keep compilation passes at 1; use five warmup units for every metric except `compile-cold-start`, which uses zero.
- Use 20 measured units for `prepared-data`, `pretraining-compute`, `pretraining-end-to-end`, `swag-end-to-end`, and `checkpoint-pause`.
- Keep `inference-prefill` and `inference-decode` at the complete fixed 32-request set, keep `compile-cold-start` at one invocation, and set `peak-metal-memory` to one complete optimizer step.
- Keep baseline and screen evidence at five process pairs and final evidence at ten process pairs.
- Do not change synchronization boundaries, throughput normalization, bootstrap settings, dispersion limits, acceptance ratios, or any thermal, memory-pressure, AC-power, power-mode, Low Power Mode, competing-workload, clean-checkout, identity, or software validation.
- Do not reuse accepted slots from any retained 20/100 journal; all four retained failure journals remain diagnostic evidence and must not be deleted or modified.
- Do not edit top-level project files such as `pyproject.toml` or `uv.lock`.
- Use `uv run` for Python commands. Run every pytest command outside the sandbox so MLX/Metal can access the Apple GPU.
- Do not start a new baseline automatically. Baseline capture requires a later explicit user request and a new empty external state directory.

---

## File Map

- Modify `v2/benchmarks/workload.py`: own the shared warmup/default-measure constants and embed the exact per-metric counts in the canonical workload.
- Modify `v2/benchmarks/runner.py`: consume the shared constants in parser defaults, immutable protocol validation, raw/comparison checks, session construction, publication, and replay validation.
- Modify `v2/tests/unit/test_benchmark_analysis.py`: pin the full count map, verify child-process forwarding and peak-reset ordering, update valid fixtures, and prove old 20/100 evidence fails closed.
- Modify `v2/benchmarks/README.md`: document the shorter protocol and the one-step peak-memory exception without changing recovery or evidence semantics.
- Refresh `/private/tmp/sml-v2-baseline-resume-harness`: replace the clean detached checkout only after verification, then bind it to the verified implementation commit.
- Preserve without modification: `/private/tmp/sml-v2-baseline-resume-state`, `/private/tmp/sml-v2-baseline-resume-state-retry-1`, `/private/tmp/sml-v2-baseline-resume-state-retry-2`, and `/private/tmp/sml-v2-baseline-resume-state-retry-3`.

### Task 1: Lock the shorter protocol through every harness surface

**Files:**
- Modify: `v2/benchmarks/workload.py:19-67,311-402,559-578`
- Modify: `v2/benchmarks/runner.py:43-67,228-263,391-400,578-596,793-808,1204-1215,2406-2434,2470-2483,2567-2588,2775-2785,3350-3373,3445-3472`
- Modify: `v2/tests/unit/test_benchmark_analysis.py:1-80,213-230,465-539,685-768,913-975,1092-1166,1190-1715,1875-1900,2700-2780,2957-2985,3528-3644`
- Modify: `v2/benchmarks/README.md:7-24`
- Test: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Consumes: `build_canonical_workload() -> CanonicalWorkload`, `CanonicalWorkload.work_units: tuple[WorkUnitDefinition, ...]`, `measure_native_process(...) -> ProcessMeasurement`, and the existing immutable session/manifest/comparison schemas.
- Produces: `v2.benchmarks.workload.WARMUP_UNITS: int = 5`, `v2.benchmarks.workload.DEFAULT_MEASURED_UNITS: int = 20`, and an identity-bound work-unit map with counts `{prepared-data: 20, pretraining-compute: 20, pretraining-end-to-end: 20, swag-end-to-end: 20, inference-prefill: 32, inference-decode: 32, checkpoint-pause: 20, compile-cold-start: 1, peak-metal-memory: 1}`.

- [ ] **Step 1: Strengthen the canonical workload contract test**

Replace the partial measured-unit assertions in `test_canonical_workload_round_trip_pins_complete_benchmark_contract` with exact compilation and work-unit assertions while retaining the existing model, optimizer, precision, loader, SWAG, generation, and metric-order checks:

```python
    assert workload.compilation == {
        "compilation_passes": 1,
        "warmup_units": 5,
        "measured_units": 20,
        "fresh_processes": True,
        "state_reset_policy": "fresh-native-workload-per-process",
    }
    assert workload.generation["request_count"] == 32
    assert workload.generation["decode_chunk_size"] == 8
    assert tuple(unit.metric for unit in workload.work_units) == METRIC_NAMES
    assert {unit.metric: unit.measured_units for unit in workload.work_units} == {
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

    protocol_neutral = workload.to_dict()
    protocol_neutral["compilation"].pop("warmup_units")
    protocol_neutral["compilation"].pop("measured_units")
    for unit in protocol_neutral["work_units"]:
        unit.pop("measured_units")
    assert structured_identity(
        "sml-benchmark-protocol-neutral-workload-v1", protocol_neutral
    ) == "sha256:fd49778a69efe0f3aafa82776b7c043e34fea9ac5b537d630f513772afacfb4a"
```

The protocol-neutral identity is calculated from the current approved pre-change workload after removing only the two global count fields and every per-metric `measured_units` field. It therefore fails if implementation changes any model, optimizer, loader, precision, semantic identity, synchronization boundary, data, SWAG, or generation field.

- [ ] **Step 2: Add a peak-memory sequence test**

Add this test after the general measurement-protocol test. It proves one compilation unit and five steady-state warmups happen before the peak reset, followed by exactly one measured optimizer step:

```python
def test_peak_memory_runs_one_measured_step_after_five_warmups_and_peak_reset():
    workload = build_canonical_workload()
    peak_unit = next(
        unit for unit in workload.work_units if unit.metric == "peak-metal-memory"
    )
    events = []
    adapter = SimpleNamespace(
        run_warmup=lambda _metric, _native, units: events.append(("warmup", units)),
        run_measured=lambda _metric, _native, units: (
            events.append(("measured", units)) or 8_192.0
        ),
    )

    measurement = measure_native_process(
        adapter=adapter,
        metric="peak-metal-memory",
        native_workload="native",
        warmup_units=workload.compilation["warmup_units"],
        measured_units=peak_unit.measured_units,
        synchronize=lambda: events.append(("synchronize",)),
        clock=iter((0.0, 1.0, 2.0, 3.0)).__next__,
        peak_memory=lambda: 7_875_602_848,
        reset_peak_memory=lambda: events.append(("reset-peak-memory",)),
    )

    operations = [event for event in events if event != ("synchronize",)]
    assert operations == [
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("reset-peak-memory",),
        ("measured", 1),
    ]
    assert measurement.value == 7_875_602_848.0
```

- [ ] **Step 3: Add parser-default and child-forwarding tests**

Add an exact parser-default test after `test_record_baseline_parser_requires_state_directory`:

```python
def test_benchmark_parser_defaults_to_the_shorter_protocol():
    baseline = build_parser().parse_args(
        [
            "record-baseline",
            "--source-commit",
            "3687f8b",
            "--manifest",
            "manifest.json",
            "--raw-output",
            "raw.jsonl",
            "--state-directory",
            "state",
        ]
    )
    comparison = build_parser().parse_args(
        [
            "compare",
            "--baseline",
            "manifest.json",
            "--candidate",
            "HEAD",
            "--metrics",
            "prepared-data",
            "--predecessors",
            '{"prepared-data":null}',
            "--output",
            "report.json",
        ]
    )

    assert (baseline.pairs, baseline.warmup, baseline.measure) == (5, 5, 20)
    assert (comparison.pairs, comparison.warmup, comparison.measure) == (5, 5, 20)
```

Change `_single_process_arguments` to use `warmup=5` and `measure=20`. Change `_stub_single_process_measurement` so its fixture trial matches `args.metric` and it can expose the arguments sent to `measure_native_process`:

```python
def _stub_single_process_measurement(monkeypatch, args, captured=None):
    args.harness_root.mkdir()
    args.source_root.mkdir()
    workload = build_canonical_workload()
    trial = _valid_raw_trial(workload, metric=args.metric)
    status = {
        "power_connected": True,
        "power_mode": "automatic",
        "low_power_mode": False,
        "thermal_state": "nominal",
        "thermal_state_raw_value": 0,
        "memory_pressure": "normal",
        "memory_free_percentage": 60,
        "competing_gpu_workload": False,
    }
    native = SimpleNamespace(
        native_configuration=trial.native_configuration,
        native_representation_identity=trial.native_representation_identity,
        canonical_row_identity=trial.canonical_row_identity,
        canonical_input_identity=trial.canonical_input_identity,
        canonical_projection=trial.canonical_projection,
        execution_order_identity=trial.execution_order_identity,
        initial_parameter_identity=trial.initial_parameter_identity,
        startup_verification_seconds=trial.startup_verification_seconds,
    )
    monkeypatch.setattr(benchmark_runner, "_git_root", lambda path: path.resolve())
    monkeypatch.setattr(
        benchmark_runner, "_require_clean_checkout", lambda path, *, label: None
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_git_commit",
        lambda path: (
            args.harness_commit
            if path.resolve() == args.harness_root.resolve()
            else args.source_commit
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "harness_content_identity",
        lambda path: args.harness_identity,
    )
    monkeypatch.setattr(legacy, "resolve_native_workload", lambda *unused: native)
    monkeypatch.setattr(
        benchmark_runner,
        "collect_environment",
        lambda: (trial.hardware, status, trial.software_versions),
    )

    def measure(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return benchmark_runner.ProcessMeasurement(
            elapsed_seconds=trial.elapsed_seconds,
            value=trial.value,
            work_count=100.0,
            compilation_seconds=trial.compilation_seconds,
            peak_memory_bytes=trial.peak_memory_bytes,
        )

    monkeypatch.setattr(benchmark_runner, "measure_native_process", measure)
```

Then add:

```python
@pytest.mark.parametrize(
    ("metric", "expected_warmup", "expected_measured"),
    [
        ("prepared-data", 5, 20),
        ("inference-prefill", 5, 32),
        ("compile-cold-start", 0, 1),
        ("peak-metal-memory", 5, 1),
    ],
)
def test_child_process_forwards_each_canonical_metric_count(
    tmp_path, monkeypatch, metric, expected_warmup, expected_measured
):
    args = _single_process_arguments(tmp_path)
    args.metric = metric
    args.output = tmp_path / "state" / "inflight" / metric / "0" / "0.json"
    captured = {}
    _stub_single_process_measurement(monkeypatch, args, captured)

    benchmark_runner._run_single_process(args)

    assert captured["warmup_units"] == expected_warmup
    assert captured["measured_units"] == expected_measured
    raw = RawTrial.from_dict(read_json_object(args.output, label="child output"))
    assert raw.warmup_units == expected_warmup
    assert raw.measured_units == expected_measured
```

- [ ] **Step 4: Run the new tests and confirm the old protocol fails them**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_round_trip_pins_complete_benchmark_contract \
  v2/tests/unit/test_benchmark_analysis.py::test_peak_memory_runs_one_measured_step_after_five_warmups_and_peak_reset \
  v2/tests/unit/test_benchmark_analysis.py::test_benchmark_parser_defaults_to_the_shorter_protocol \
  v2/tests/unit/test_benchmark_analysis.py::test_child_process_forwards_each_canonical_metric_count \
  -v
```

Expected: FAIL because the canonical document and parser still contain 20 warmups and 100 default measured units, and peak memory still maps to 100 measured steps.

- [ ] **Step 5: Make the canonical workload the source of truth**

Add the shared constants beside the existing workload constants in `v2/benchmarks/workload.py`:

```python
WARMUP_UNITS = 5
DEFAULT_MEASURED_UNITS = 20
```

Change `_work_units` to bind the fixed exceptions and use the shared default:

```python
    measured_units = {
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
        for metric, direction, numerator, work_unit, start_boundary, end_boundary in definitions
    )
```

Change the canonical compilation document to:

```python
        compilation={
            "compilation_passes": 1,
            "warmup_units": WARMUP_UNITS,
            "measured_units": DEFAULT_MEASURED_UNITS,
            "fresh_processes": True,
            "state_reset_policy": "fresh-native-workload-per-process",
        },
```

- [ ] **Step 6: Remove runner-owned protocol literals**

Import the shared constants with the existing workload imports:

```python
from v2.benchmarks.workload import (
    DEFAULT_MEASURED_UNITS,
    LEGACY_PRECISION_POLICY,
    REPLACEMENT_PRECISION_POLICY,
    WARMUP_UNITS,
    build_canonical_workload,
    canonical_execution_order_identity,
    canonical_input_identity,
    canonical_metric_projection,
    canonical_workload_identity,
    fixed_canonical_rows,
    harness_content_identity,
    structured_identity,
    write_paired_pretraining_representations,
)
```

Delete the local `WARMUP_UNITS = 20` and `DEFAULT_MEASURED_UNITS = 100` definitions. Do not change any current validator or record-building use of those names: those paths will now consume the identity module's values. Replace the four parser literals with the shared names:

```python
    baseline.add_argument("--pairs", type=int, default=SCREEN_PAIRS)
    baseline.add_argument("--warmup", type=int, default=WARMUP_UNITS)
    baseline.add_argument("--measure", type=int, default=DEFAULT_MEASURED_UNITS)

    compare.add_argument("--pairs", type=int, default=SCREEN_PAIRS)
    compare.add_argument("--warmup", type=int, default=WARMUP_UNITS)
    compare.add_argument("--measure", type=int, default=DEFAULT_MEASURED_UNITS)
```

Keep `_run_single_process` unchanged where it chooses `0` for `compile-cold-start`, uses `args.warmup` for every other metric, and reads `measured_units` from the canonical metric work unit.

- [ ] **Step 7: Update valid fixtures and make old evidence the explicit invalid case**

In `_valid_raw_trial`, change the non-compile warmup from `20` to `5`. In `_session_document`, use this default protocol:

```python
        protocol=protocol or {"pairs": 5, "warmup_units": 5, "measured_units": 20},
```

Change every valid `build_baseline_manifest(...)` fixture from:

```python
        warmup_units=20,
        measured_units=100,
```

to:

```python
        warmup_units=5,
        measured_units=20,
```

In `test_baseline_validator_rejects_partial_or_weakened_protocols`, make the prior protocol the invalid raw and manifest evidence:

```python
    wrong_warmup = replace(trials[0], warmup_units=20)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_warmup, *trials[1:]))

    wrong_units = replace(trials[0], measured_units=100)
    with pytest.raises(ValueError, match="warmup or measured"):
        validate_baseline_manifest(manifest, (wrong_units, *trials[1:]))

    old_protocol = json.loads(json.dumps(manifest))
    old_protocol["protocol"]["warmup_units"] = 20
    old_protocol["protocol"]["measured_units"] = 100
    _resign_baseline(old_protocol)
    with pytest.raises(ValueError, match="pinned protocol"):
        validate_baseline_manifest(old_protocol, trials)
```

In `test_comparison_validator_rejects_weakened_screen_protocol`, replace the warmup/measured cases with `("warmup_units", 20)` and `("measured_units", 100)`. In `test_baseline_journal_rejects_every_session_compatibility_change`, use both of these protocol mutations so pair-count compatibility and old-protocol compatibility are independently covered:

```python
        {"protocol": {"pairs": 4, "warmup_units": 5, "measured_units": 20}},
        {"protocol": {"pairs": 5, "warmup_units": 20, "measured_units": 100}},
```

Do not change the intentionally arbitrary `warmup_units=2`, `measured_units=1`, or `measured_units=3` values in the low-level measurement math tests.

- [ ] **Step 8: Bind the canonical command to the new session values**

Add `import shlex` beside the existing standard-library test imports. In `test_interrupted_manifest_publication_resumes_byte_identically_across_cli_spellings`, immediately after `first_command` is built, add:

```python
    command_parts = shlex.split(first_command)
    assert command_parts[command_parts.index("--pairs") + 1] == "5"
    assert command_parts[command_parts.index("--warmup") + 1] == "5"
    assert command_parts[command_parts.index("--measure") + 1] == "20"
```

Retain the existing relocation, interruption, resume, byte-identity, and completion assertions. This verifies the shorter values without weakening publication or crash-resume coverage.

- [ ] **Step 9: Audit the count migration before running tests**

Run:

```bash
rg -n 'warmup_units.?[:=].?20|measured_units.?[:=].?100|default=20|default=100' \
  v2/benchmarks v2/tests/unit/test_benchmark_analysis.py
```

Expected: production code has no old defaults. Test matches for 20 warmups or 100 measurements exist only in the explicit old-raw, old-manifest, old-comparison, and old-session rejection cases. There must be no `default=20` false positive for `--warmup`; parser defaults use the named constants.

- [ ] **Step 10: Run the focused protocol and compatibility tests**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/test_benchmark_analysis.py::test_canonical_workload_round_trip_pins_complete_benchmark_contract \
  v2/tests/unit/test_benchmark_analysis.py::test_peak_memory_runs_one_measured_step_after_five_warmups_and_peak_reset \
  v2/tests/unit/test_benchmark_analysis.py::test_benchmark_parser_defaults_to_the_shorter_protocol \
  v2/tests/unit/test_benchmark_analysis.py::test_baseline_validator_rejects_partial_or_weakened_protocols \
  v2/tests/unit/test_benchmark_analysis.py::test_comparison_validator_rejects_weakened_screen_protocol \
  v2/tests/unit/test_benchmark_analysis.py::test_interrupted_manifest_publication_resumes_byte_identically_across_cli_spellings \
  v2/tests/unit/test_benchmark_analysis.py::test_baseline_journal_rejects_every_session_compatibility_change \
  v2/tests/unit/test_benchmark_analysis.py::test_child_process_forwards_each_canonical_metric_count \
  -v
```

Expected: PASS. In particular, the parameterized child test must report `peak-metal-memory` as warmup 5 / measured 1 and `compile-cold-start` as warmup 0 / measured 1.

- [ ] **Step 11: Update the operator documentation**

Replace the README's opening protocol description with:

```markdown
All measurements run in fresh processes from clean source checkouts. Timed MLX
regions synchronize immediately before and after execution, perform one untimed
compilation pass, warm up for 5 units, and measure 20 units by default. Inference
consumes its complete fixed 32-request set; compile cold-start and peak Metal
memory each measure one canonical unit. Trial order alternates by pair. Raw
process records are retained alongside reports; statistical decisions use
direction-normalized paired ratios and a reproducible whole-pair bootstrap.
```

Replace the immutable baseline paragraph's count sentences with:

```markdown
`record-baseline` is intentionally immutable: it accepts only the fully resolved
`3687f8b` source commit, all nine metrics, five fresh processes per metric, five
warmups, and the canonical measured-unit counts. Inference consumes the complete
32-request set exactly once, while compile cold-start and peak Metal memory each
consume one canonical unit. Validation rejects metric subsets, changed protocol
values, duplicate raw records, a different canonical projection or logical work
order, and environment or software mismatches.
```

Leave the journal topology, five-minute nominal recovery window, two-hour slot deadline, 45-slot publication rule, five-pair screen profile, and ten-pair final profile unchanged.

- [ ] **Step 12: Run the complete harness unit module**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -v
```

Expected: PASS with no skipped protocol, recovery, atomic-publication, locking, or resume regression tests caused by this change.

- [ ] **Step 13: Commit the implementation**

```bash
git add \
  v2/benchmarks/workload.py \
  v2/benchmarks/runner.py \
  v2/tests/unit/test_benchmark_analysis.py \
  v2/benchmarks/README.md
git commit -m "feat(v2): shorten benchmark measurement protocol"
```

Expected: the commit contains only the four listed v2 files; the approved design and this plan remain in their separate planning commit.

### Task 2: Verify the implementation and refresh the detached harness

**Files:**
- Verify: `v2/`
- Replace only after verification: `/private/tmp/sml-v2-baseline-resume-harness`
- Preserve: `/private/tmp/sml-v2-baseline-resume-state*`

**Interfaces:**
- Consumes: the committed `WARMUP_UNITS = 5`, `DEFAULT_MEASURED_UNITS = 20`, exact canonical metric map, and the repository `HEAD` produced by Task 1.
- Produces: a clean, detached `/private/tmp/sml-v2-baseline-resume-harness` whose `HEAD` equals the verified repository commit; it produces no baseline journal, raw JSONL, or manifest.

- [ ] **Step 1: Confirm the implementation worktree contains no uncommitted changes**

```bash
git status --short --branch
git show --stat --oneline --decorate HEAD
```

Expected: `git status` has no short-status file entries, and `HEAD` is `feat(v2): shorten benchmark measurement protocol`.

- [ ] **Step 2: Run Ruff lint verification**

```bash
uv run ruff check v2
```

Expected: `All checks passed!`

- [ ] **Step 3: Run Ruff format verification**

```bash
uv run ruff format --check v2
```

Expected: every v2 file is already formatted. If Ruff reports a changed file, run `uv run ruff format v2`, inspect the diff, rerun Steps 2-3, rerun the complete pytest command in Step 4, and commit only the formatting correction.

- [ ] **Step 4: Run the complete v2 test suite with Metal access**

Run outside the sandbox:

```bash
uv run pytest v2/tests
```

Expected: PASS. A failure in protocol identity, environment validation, recovery, locking, or publication blocks the detached-checkout refresh.

- [ ] **Step 5: Reconfirm the verified commit and clean status**

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: no short-status file entries. Record the full 40-character commit printed here for the detached-checkout comparison in Step 9.

- [ ] **Step 6: Prove the retained journals still exist and remain diagnostic only**

```bash
ls -ld \
  /private/tmp/sml-v2-baseline-resume-state \
  /private/tmp/sml-v2-baseline-resume-state-retry-1 \
  /private/tmp/sml-v2-baseline-resume-state-retry-2 \
  /private/tmp/sml-v2-baseline-resume-state-retry-3
```

Expected: all four directories exist. Do not write to them, copy accepted slots from them, or pass any of them to the shorter-protocol harness.

- [ ] **Step 7: Confirm no baseline process is active**

```bash
pgrep -af 'v2\.benchmarks\.runner.*record-baseline'
```

Expected: exit status 1 and no process output. If a matching process appears, stop and report it instead of changing either checkout.

- [ ] **Step 8: Verify the old detached checkout is clean, then replace that exact checkout**

```bash
git -C /private/tmp/sml-v2-baseline-resume-harness status --short
git worktree remove /private/tmp/sml-v2-baseline-resume-harness
git worktree add --detach /private/tmp/sml-v2-baseline-resume-harness HEAD
```

Expected: the first command prints nothing. The removal and addition target only `/private/tmp/sml-v2-baseline-resume-harness`; none of the four retained state directories is touched.

- [ ] **Step 9: Verify the refreshed harness identity and cleanliness**

```bash
git -C /private/tmp/sml-v2-baseline-resume-harness status --short --branch
git -C /private/tmp/sml-v2-baseline-resume-harness rev-parse HEAD
uv run python -c 'from pathlib import Path; from v2.benchmarks.workload import harness_content_identity; print(harness_content_identity(Path("/private/tmp/sml-v2-baseline-resume-harness")))'
```

Expected: detached `HEAD`, no file-status entries, the same full commit recorded in Step 5, and a `sha256:` harness content identity. Record that identity for the eventual fresh baseline session.

- [ ] **Step 10: Stop before measurement and report the handoff**

Do not invoke `record-baseline`. Report the exact full implementation commit from Step 5, the detached harness path, the exact `sha256:` harness identity from Step 9, and the protocol counts `warmup 5 / default measure 20 / inference 32 / compile 1 / peak memory 1`. State explicitly that the retained journals are unchanged and incompatible, that no baseline was started, and that a new empty external state directory is required for any later capture.
