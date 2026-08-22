# V2 Phase 2 Prepared-Data Benchmark Bridge Implementation Plan

**Status update (2026-08-22):** Task 1 is complete. Task 2 is superseded as a
required task: the screen is optional diagnostic work and no Phase 2 benchmark
result or acceptance commit is needed to continue the refactor.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the frozen replacement adapter to exercise and, when useful, measure the real immutable prepared-data loader. The corrected Phase 2 screen is optional.

**Architecture:** `sml.data.pretraining` exposes a lazy two-argument factory that delegates to a private prepared-data benchmark module. The private module materializes one deterministic, immutable int32 NPY bundle from the harness's canonical rows, reopens it with FULL verification, and executes every timed unit through `PretrainingBatchStream` plus consumer-side MLX transfer. The existing harness and pinned baseline remain byte-identical.

**Tech Stack:** Python 3.12.13, NumPy, MLX, existing SML artifact/publication APIs, `uv`, pytest, Ruff, Git.

## Global Constraints

- Work on `main`; the user explicitly authorized continued main-branch work.
- Do not modify `v2/benchmarks/**`: those files define the pinned harness content identity.
- Do not modify top-level project files, `pyproject.toml`, or `uv.lock`.
- The production factory signature is exactly `build_benchmark_workload(metric: str, canonical_workload: object) -> object` and accepts only `prepared-data`.
- Ordinary imports of `sml.data.pretraining` must not import MLX or `v2.benchmarks`; both remain lazy inside the benchmark-only call path.
- The runtime must use the real `PreparedDataBundle`, immutable publication, FULL verification, `PretrainingBatchStream`, `BatchEnvelope`, and consumer-thread `mx.array`/`mx.eval` path.
- Use one NPY shard so shard permutation preserves the harness's fixed canonical row order; dtype is little-endian int32, C order, and row width is `sequence_length + 1`.
- Compute the product `sml-row-content-v1` identity for the manifest and the harness `sml-pretraining-rows-v1` identity for `canonical_row_identity` from the same canonical int32 bytes; do not compare the two domain-separated digests for equality.
- Startup artifact creation and FULL verification remain outside steady-state timing.
- The factory creates and FULL-verifies the first stream before any timing. Every `run()` closes that stream in the same call; `reset_after_warmup()` constructs the next initial-cursor stream before the measured timer starts. Runtime cleanup is idempotent and registered for normal process exit without relying on object finalization.
- If the optional screen is run, its protocol remains exactly five pairs, five warmups, 20 measured units, 10,000 bootstrap resamples, minimum ratio `0.97`, maximum dispersion `0.02`, report-only lower bound, and predecessor mapping `{"prepared-data": null}`.
- Run every `uv run pytest` command outside the sandbox so MLX/Metal can access the Apple GPU.
- Before finishing any v2 change, run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests`.

## File Structure

- Create `v2/src/sml/data/_pretraining_benchmark.py`: deterministic benchmark bundle construction, FULL verification, runtime execution, identity proofs, and cleanup.
- Modify `v2/src/sml/data/pretraining.py`: add only the lazy public owner factory and export it.
- Modify `v2/tests/unit/data/test_pretraining.py`: direct validation, artifact, execution-order, and cleanup behavior for the owner factory/runtime.
- Modify `v2/tests/unit/test_benchmark_analysis.py`: change the real-owner transition regression from unavailable to available while retaining the synthetic future-owner ABI test for later metrics.
- Modify `v2/tests/integration/test_pretraining_data_workflow.py`: prove the replacement adapter resolves and runs the real prepared-data owner through MLX.
- Optional: create `v2/benchmarks/results/phase-2-loader.json` as a diagnostic comparison report.
- Optional: create `v2/benchmarks/results/phase-2.json` as its independently validated diagnostic report.

---

### Task 1: Enable the Real Prepared-Data Benchmark Owner

**Files:**
- Create: `v2/src/sml/data/_pretraining_benchmark.py`
- Modify: `v2/src/sml/data/pretraining.py`
- Modify: `v2/tests/unit/data/test_pretraining.py`
- Modify: `v2/tests/unit/test_benchmark_analysis.py`
- Modify: `v2/tests/integration/test_pretraining_data_workflow.py`

**Interfaces:**
- Consumes: `PreparedDataBundle`, `PretrainingBatchStream`, `PretrainingCursor.initial()`, `publish_immutable_bundle`, `read_manifest`, `row_content_identity`, `canonical_json_bytes`, and the frozen functions/constants in `v2.benchmarks.workload`.
- Produces: `sml.data.pretraining.build_benchmark_workload(metric: str, canonical_workload: object) -> object`.
- Produces runtime attributes: `verification_level`, `native_configuration`, `native_representation_identity`, `canonical_row_identity`, `canonical_input_identity`, `canonical_projection`, `execution_order_identity`, `initial_parameter_identity`.
- Produces runtime methods: `run(units: int) -> float`, `reset_after_warmup() -> None`, `reset_measured_order() -> None`, and idempotent `close() -> None`.

- [ ] **Step 1: Change the owner-availability regression and add direct validation tests**

Replace the current unavailable assertion in
`v2/tests/unit/test_benchmark_analysis.py` with a small canonical workload and
an available-owner assertion:

```python
def test_replacement_adapter_enables_real_prepared_data_owner():
    workload = build_canonical_workload(
        model_overrides={"vocab_size": 32},
        loader_overrides={"sequence_length": 8},
        row_count=32,
    )

    native = resolve_native_workload("prepared-data", workload, Path.cwd())

    assert isinstance(native, ReplacementNativeWorkload)
    assert native.owner_import == "sml.data.pretraining"
    assert native.canonical_row_identity == workload.semantic_identities[
        "canonical_training_rows"
    ]
    assert native.canonical_projection == canonical_metric_projection(
        "prepared-data", workload
    )
    native.runtime.close()
```

Add direct tests in `v2/tests/unit/data/test_pretraining.py` that call the public
factory with the same 32-row workload and assert:

```python
def test_benchmark_factory_rejects_non_prepared_metric_before_runtime_start():
    workload = _small_benchmark_workload()
    threads_before = _prefetch_threads()

    with pytest.raises(ValueError, match="prepared-data"):
        build_benchmark_workload("pretraining-compute", workload)

    assert _prefetch_threads() == threads_before


def test_benchmark_runtime_proves_both_row_identity_domains():
    workload = _small_benchmark_workload()
    runtime = build_benchmark_workload("prepared-data", workload)
    try:
        canonical = fixed_canonical_rows(
            row_count=32,
            row_width=9,
            vocab_size=32,
        )
        assert runtime.canonical_row_identity == semantic_row_content_identity(
            canonical
        )
        assert runtime.bundle.manifest.row_content_identity == row_content_identity(
            canonical,
            32,
            9,
        )
        assert (
            runtime.bundle.manifest.row_content_identity
            != runtime.canonical_row_identity
        )
        assert runtime.verification_level == "full"
        assert runtime.bundle.verification is VerificationLevel.FULL
    finally:
        runtime.close()
```

The production runtime may expose its immutable `bundle` as a read-only
property because the adapter and tests need an auditable native-representation
proof; it must not expose mutable stream state.

- [ ] **Step 2: Add real execution, reset, epoch-rollover, and cleanup tests**

Add behavior tests that use real MLX and the runtime's real bundle:

```python
def test_benchmark_runtime_runs_real_stream_and_resets_canonical_order():
    workload = _small_benchmark_workload(row_count=4)
    runtime = build_benchmark_workload("prepared-data", workload)
    try:
        recorder = _RecordingMX(runtime._mx)
        runtime._mx = recorder
        canonical = fixed_canonical_rows(row_count=4, row_width=9, vocab_size=32)
        assert runtime.run(3) == 3.0
        assert np.concatenate(recorder.rows).tolist() == canonical[:3].tolist()
        runtime.reset_after_warmup()
        recorder.rows.clear()
        assert runtime.run(2) == 2.0
        assert np.concatenate(recorder.rows).tolist() == canonical[:2].tolist()
        runtime.reset_after_warmup()
        runtime.reset_measured_order()
        recorder.rows.clear()
        assert runtime.run(6) == 6.0
        expected = np.concatenate((canonical, canonical[:2]))
        assert np.concatenate(recorder.rows).tolist() == expected.tolist()
    finally:
        runtime.close()


def test_benchmark_runtime_closes_stream_and_temporary_tree_idempotently():
    threads_before = _prefetch_threads()
    runtime = build_benchmark_workload(
        "prepared-data", _small_benchmark_workload()
    )
    temporary_root = runtime.temporary_root

    assert runtime.run(1) == 1.0
    assert _prefetch_threads() == threads_before
    runtime.close()
    runtime.close()

    assert not temporary_root.exists()
    assert _prefetch_threads() == threads_before
```

`_RecordingMX.array()` copies the incoming NumPy rows and delegates to the real
`mlx.core.array`; `_RecordingMX.eval()` delegates to real `mlx.core.eval`.
Therefore the order assertion observes the actual consumer boundary without
replacing MLX semantics. Add a separate injected consumer-transfer failure by
replacing only `array()` with a proxy that raises
`RuntimeError("transfer failed")`. Assert `run(1)` raises that error, no
`sml-pretraining-prefetch` thread remains, and `close()` removes the temporary
root.

In `v2/tests/integration/test_pretraining_data_workflow.py`, resolve the owner
through `v2.benchmarks.adapters.replacement`, run one warmup unit and three
measured units, assert both return paths preserve the exact work count, and
close the runtime in `finally`.

- [ ] **Step 3: Run the focused tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/data/test_pretraining.py \
  v2/tests/unit/test_benchmark_analysis.py \
  v2/tests/integration/test_pretraining_data_workflow.py \
  -k "benchmark or real_prepared_data_owner" -v
```

Expected: FAIL because the real owner still returns
`UnavailableNativeWorkload` and `sml.data.pretraining` has no
`build_benchmark_workload`.

- [ ] **Step 4: Add the lazy owner wrapper**

Append this wrapper to `v2/src/sml/data/pretraining.py` and include its name in
`__all__`:

```python
def build_benchmark_workload(metric: str, canonical_workload: object) -> object:
    """Build the production prepared-data path for the pinned benchmark ABI."""
    from sml.data._pretraining_benchmark import (
        build_prepared_data_benchmark_workload,
    )

    return build_prepared_data_benchmark_workload(metric, canonical_workload)
```

Do not add module-level MLX or `v2.benchmarks` imports to `pretraining.py`.

- [ ] **Step 5: Implement deterministic immutable benchmark input**

Create `v2/src/sml/data/_pretraining_benchmark.py`. Define these constants and
helpers with exact behavior:

```python
_BENCHMARK_METRIC = "prepared-data"
_BENCHMARK_SEED = 0
_PREFETCH_DEPTH = 2
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_MODEL_BYTES = b"sml-benchmark-tokenizer-model-v1\n"
_VOCAB_BYTES = b"sml-benchmark-tokenizer-vocab-v1\n"


def _require_plain_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value
```

`_materialize_bundle(canonical_workload, output)` must:

1. require a `CanonicalWorkload` and exact prepared-data projection fields;
2. derive `row_count`, `sequence_length`, `row_width`, `vocab_size`, and special
   IDs as plain integers;
3. build
   `fixed_canonical_rows(row_count=row_count, row_width=row_width, vocab_size=vocab_size)`
   as contiguous little-endian int32;
4. recompute `semantic_row_content_identity(rows)` and compare it to
   `canonical_training_rows`;
5. publish one `shards/train-000000.npy` through
   `publish_immutable_bundle`;
6. write deterministic nested tokenizer model/vocab bytes and a canonical
   `TokenizerManifest` with the workload's vocabulary and special IDs;
7. construct the outer `PretrainingDataManifest` with one shard, product
   `row_content_identity(rows, row_count, row_width)`, preparation seed `0`,
   row-order policy
   `{"algorithm": "benchmark-fixed-canonical-order-v1", "output_shard_rows": row_count}`,
   and source summary
   `{"kind": "pinned-canonical-benchmark-rows", "row_count": row_count}`;
8. reopen the published bundle with
   `read_manifest(output, PretrainingDataManifest, VerificationLevel.FULL)`;
9. return `PreparedDataBundle` only if the reopened manifest exactly matches the
   published manifest and the product row identity recomputes from the same
   matrix.

Use existing `PayloadRef`, `canonical_json_bytes`, `_payload_ref`, and
`_write_shard` behavior rather than duplicating file hashing or NPY writing.
All files are created inside the publisher's private directory before its
atomic rename.

- [ ] **Step 6: Implement the runtime and adapter proofs**

Implement `_PreparedDataBenchmarkRuntime` with read-only properties for
`bundle` and `temporary_root`. Its constructor owns the `TemporaryDirectory`,
registers `close` with `atexit`, stores the lazily imported `mlx.core` module,
and creates one initial-cursor `PretrainingBatchStream` only after the bundle's
FULL verification succeeds. This construction is part of factory startup and
therefore precedes every harness timing boundary.

Each `run(units)` must validate `units`, atomically take the currently prepared
stream, and execute this ownership pattern:

```python
stream = self._take_prepared_stream()
try:
    for _ in range(units):
        envelope = next(stream)
        try:
            device_rows = self._mx.array(envelope.rows)
        finally:
            envelope.release()
        input_ids = device_rows[:, :-1]
        labels = device_rows[:, 1:]
        self._mx.eval(input_ids, labels)
    return float(units)
finally:
    stream.close()
```

`reset_after_warmup()` closes any remaining stream and creates a new stream from
`PretrainingCursor.initial()`. The replacement adapter calls it at the end of
every compilation/warmup call, including the final warmup before the measured
timer starts. `reset_measured_order()` must not repeat FULL validation inside
the measured region; it only verifies that an active fresh initial-cursor stream
is ready. The one-shard bundle preserves canonical order across deterministic
epoch rollover. The taken stream must close before `run()` returns or raises.

`close()` unregisters its `atexit` callback, cleans the `TemporaryDirectory`,
marks the runtime closed, and is idempotent. A closed runtime rejects `run()`.

`build_prepared_data_benchmark_workload` must build the runtime and populate:

```python
runtime.native_configuration = {
    "metric": "prepared-data",
    "native_input_format": "prepared-data-bundle-npy-int32",
    "parameter_dtype": "bfloat16",
    "moment_dtype": "float32",
    "master_parameters": True,
    "rope_scaling_factor": float(canonical_workload.model["rope_scaling_factor"]),
    "sequence_length": sequence_length,
    "microbatch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "canonical_row_identity": canonical_row_identity,
    "canonical_input_identity": canonical_input_identity(
        "prepared-data", canonical_workload
    ),
    "canonical_projection_identity": structured_identity(
        "sml-benchmark-metric-projection-v1",
        canonical_metric_projection("prepared-data", canonical_workload),
    ),
    "canonical_execution_order_identity": canonical_execution_order_identity(
        "prepared-data", canonical_workload
    ),
    "parameter_precision_policy": REPLACEMENT_PRECISION_POLICY,
}
```

Set `native_representation_identity` to the verified bundle manifest identity,
`canonical_row_identity` to the harness semantic row identity,
`canonical_input_identity` and `canonical_projection` through the frozen helper
functions, `execution_order_identity` through the frozen helper,
`initial_parameter_identity` to the workload's `initial_bf16_parameters`, and
`verification_level` to literal `"full"`. If any construction step fails,
close the temporary directory before re-raising.

- [ ] **Step 7: Run focused GREEN and the complete loader suites**

Run outside the sandbox:

```bash
uv run pytest \
  v2/tests/unit/data/test_pretraining.py \
  v2/tests/unit/test_benchmark_analysis.py \
  v2/tests/integration/test_pretraining_data_workflow.py \
  -k "benchmark or real_prepared_data_owner" -v

uv run pytest \
  v2/tests/unit/data/test_pretraining.py \
  v2/tests/unit/test_benchmark_analysis.py \
  v2/tests/integration/test_pretraining_data_workflow.py -v
```

Expected: all selected and complete loader/benchmark tests pass with no warnings
or leaked prefetch threads.

- [ ] **Step 8: Run full verification and commit the bridge**

Run:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: Ruff and all v2 tests pass, `uv.lock` is byte-identical, and only the
five Task 1 files are changed. Then commit:

```bash
git add \
  v2/src/sml/data/_pretraining_benchmark.py \
  v2/src/sml/data/pretraining.py \
  v2/tests/unit/data/test_pretraining.py \
  v2/tests/unit/test_benchmark_analysis.py \
  v2/tests/integration/test_pretraining_data_workflow.py
git commit -m "perf(v2): enable prepared-data benchmark bridge"
```

Generate a task review package from the Task 1 base through this commit. The
review must explicitly verify the frozen harness is unchanged, startup FULL
verification is outside timing, actual loader/MLX ownership is used, canonical
order resets, identity domains remain separate, and all resources close.

---

### Task 2: Optional Phase 2 Screen (Superseded as Required Work)

Do not execute this task merely to unblock the refactor. The remaining steps
are retained only as a reproducible diagnostic procedure. Failure, noise,
thermal rejection, or absent output does not block Phase 3, and the result
files do not need to exist or be committed.

**Files:**
- Optional create: `v2/benchmarks/results/phase-2-loader.json`
- Optional create: `v2/benchmarks/results/phase-2.json`

**Interfaces:**
- Consumes: a clean committed Task 1 bridge, pinned baseline manifest `v2/benchmarks/manifests/baseline-3687f8b.json`, and predecessor mapping `{"prepared-data": null}`.
- Produces only when requested: a valid diagnostic screen report and its independently validated Phase 2 report.

- [ ] **Step 1: Verify the exact candidate and environment before measurement**

Run outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
git diff --exit-code -- uv.lock
```

Expected: all checks pass and tracked status is empty. Confirm the Mac is on AC
power, automatic power mode, low-power mode off, nominal thermal state, normal
memory pressure, and has no competing GPU workload. Do not create placeholder
result files and do not launch when preflight is non-nominal.

- [ ] **Step 2: Run the committed prepared-data comparison**

Run exactly:

```bash
uv run python -m v2.benchmarks.runner compare \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --candidate HEAD \
  --metrics prepared-data \
  --mode screen \
  --pairs 5 \
  --warmup 5 \
  --measure 20 \
  --bootstrap-resamples 10000 \
  --minimum-ratio 0.97 \
  --maximum-dispersion 0.02 \
  --lower-bound-report-only \
  --predecessors '{"prepared-data":null}' \
  --output v2/benchmarks/results/phase-2-loader.json
```

Expected: ten accepted fresh-process trials are recorded for the five
alternating pairs. A thermal, power, memory, software, or competing-workload
violation aborts without acceptance; comparison mode is non-resumable. Excess
dispersion triggers only the harness-owned full retry after its 15-minute
cooldown. Persistent noise or an environment rejection blocks Task 2.

- [ ] **Step 3: Validate the phase report independently**

Run exactly:

```bash
uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json \
  --output v2/benchmarks/results/phase-2.json
```

Expected: exit zero; the prepared-data median ratio is at least `0.97`,
dispersion is at most `0.02`, all evidence and canonical proofs validate, and
the Phase 2 predecessor set is empty.

- [ ] **Step 4: Inspect, stage, and commit only accepted evidence**

Run:

```bash
git diff --check
git status --short
uv run python -c 'import json, pathlib; paths = [pathlib.Path("v2/benchmarks/results/phase-2-loader.json"), pathlib.Path("v2/benchmarks/results/phase-2.json")]; reports = [json.loads(path.read_text()) for path in paths]; print([(path.name, report["identity"], report["metrics"]["prepared-data"]["baseline_comparison"]["decision"]) for path, report in zip(paths, reports, strict=True)])'
```

Expected: exactly the two result files are untracked, both parse, both carry a
`pass` prepared-data baseline decision, and their identities are printed. Then:

```bash
git add \
  v2/benchmarks/results/phase-2-loader.json \
  v2/benchmarks/results/phase-2.json
git diff --cached --check
git commit -m "bench(v2): accept artifacts and prepared-data phase"
```

Do not commit either report if validation fails, the environment was rejected,
or the second noise attempt remains above the dispersion limit.

- [ ] **Step 5: Verify the accepted commit**

Run:

```bash
git show --check --stat --oneline --summary HEAD
git status --short
git diff --exit-code HEAD^ -- uv.lock
```

Expected: the commit contains exactly the two validated result files, the
tracked worktree is clean, and `uv.lock` remains unchanged. Record the bridge
and Phase 2 commit hashes, exact test count, report identities, median ratio,
dispersion, trial thermal states, and whether a cooldown retry occurred in the
plan ledger.
