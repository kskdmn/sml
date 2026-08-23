# V2 Performance-First Refactor Part 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Execution status (2026-08-23):** Tasks 0.1 and 1.1-3.6 are implemented,
reviewed at their task boundaries, and have passed the repeated functional
gate. The recorded-source quality validator and deeply immutable
`VerifiedCheckpointContents` closed at `4c190f3` with scoped review approved.
The unchecked boxes below preserve the original TDD procedure; they are not
live task status. Part 1 is complete; Part 2 Task 4.1 is complete at `7cc45ed`.
Part 2 Task 4.2 is the next live task.

**Goal:** Replace the v2 model/data/artifact foundation with the `sml` package and deliver exact, resumable BF16-compute pretraining with authoritative FP32 master parameters and FP32 optimizer state. The existing benchmark harness may be used for diagnostics, but baseline comparison is not required.

**Architecture:** This first part creates the package and temporary migration bridge, captures legacy equivalence fixtures, implements the model and immutable artifact contracts, then builds prepared-data and pretraining runtimes on those contracts. It ends with a portable pretraining run/checkpoint that Part 2 consumes for inference, evaluation, SWAG fine-tuning, and final cutover.

**Tech Stack:** Python 3.12.13, MLX 0.32+, NumPy 2.4+, SentencePiece 0.2+, safetensors through `mlx.core.save_safetensors`, APFS/macOS file-descriptor APIs, `uv run`, pytest 9, Ruff 0.15+

## Global Constraints

**Superseding acceptance policy (2026-08-22):** Phase progression is gated by
Ruff, the full v2 test suite, the controlled correctness/quality checks, and
relevant integration or CLI smoke workflows. Every baseline capture,
before/after comparison, performance threshold, thermal launch gate, and
benchmark evidence commit in this plan is optional and non-blocking.

- The approved source of truth is `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`; do not revive the superseded checkpoint/SWAG-only plan.
- This is a clean break: provide no readers, aliases, conversions, warnings, or fallback interpretations for existing v2 imports, CLI flags, tokenizer inputs, prepared datasets, manifests, checkpoints, metadata, or resume state.
- Temporary coexistence is allowed only through Phase 5. The temporary package bridge is deleted in Phase 6 of Part 2.
- Preserve Transformer, YaRN/RoPE, GQA, RMSNorm, SwiGLU, causal-loss, KV-cache, generation, and SentencePiece BPE mathematics.
- The only model/training corrections are canonical summed tied-embedding gradients, mean continuation-token SWAG scores, and authoritative FP32 base-model master parameters with BF16 working parameters, FP32 Adam moments/update arithmetic, and explicit BF16 derivation after every update.
- Base training checkpoints FP32 master parameters, their exact BF16 working casts, and FP32 Adam state; it has no dynamic loss scaler. SWAG uses the selected frozen BF16 working base and FP32 LoRA state.
- Base pretraining always uses and authoritatively records `rope_scaling_factor=1.0`. Fresh-run validation rejects any other value, resume restores exactly `1.0`, and every pretraining benchmark records `1.0` in the canonical and native configurations.
- Training with `rope_scaling_factor > 1.0` belongs to a separate future long-context fine-tuning workflow. That workflow will consume a completed base-pretraining artifact and publish a distinct run/model identity; it is not implemented, simulated, or silently enabled anywhere in this refactor.
- MLX is the only model, training, inference, and evaluation backend, and Apple Silicon is the target.
- Throughput wins over peak memory only while the default workload still fits the Apple M5 10-core CPU, 10-core GPU, 24 GB target without critical memory pressure.
- Optional performance investigations use source commit `3687f8b` and the same independently versioned harness/canonical workload identity on both sides.
- The fixed pretraining workload is vocabulary size 28,672, hidden size 768, 12 layers, 12 query heads, 3 KV heads, intermediate size 2,176, sequence length 1,024, microbatch size 1, gradient accumulation 8, BF16 compute, and `rope_scaling_factor=1.0`.
- The historical phase-screen and final-acceptance protocols remain available for optional diagnostics; their ratios, dispersion, confidence bounds, power state, and thermal state never block implementation work.
- Correctness-sensitive workflows fully rehash payloads before GPU initialization or destructive action. Read-only inference/evaluation may report `manifest-trusted`; they must never report `full` without rehashing.
- Artifact publication and writable run operations require local APFS. Never weaken descriptor-relative no-follow path traversal, lock, fsync, rename, recovery, or retention guarantees for test convenience.
- Use Python 3.12.13 through `uv run`. Run every MLX pytest command and every benchmark outside the sandbox so Metal is available.
- Do not add dependencies. The design authorizes only the top-level `pyproject.toml` source-package mapping needed by `uv run python -m sml`; keep `uv.lock` byte-identical.
- Before each phase commit, run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests` outside the sandbox.
- Keep performance-sensitive loops direct. Do not add a generic pipeline, callback, registry, plugin system, per-row dictionaries, Python token lists in hot loaders, or host synchronization inside kernels.
- An `mx.compile` boundary accepts and returns only MLX-supported built-in array trees (`dict`, `list`, or `tuple` with array leaves) and scalar constants. Frozen host dataclasses may wrap those trees outside the boundary, but `AdamState`, `TrainerState`, `KVCache`, or another custom object never crosses it.

---

## Master Phase Index

The approved design calls for six ordered, independently reviewable implementation plans. To honor the user's two-document limit, each phase below is a self-contained plan section with its own correctness/workflow/commit gate and optional performance diagnostics.

1. [Foundation and model package](#phase-1-foundation-and-model-package)
2. [Tokenizer, artifacts, and prepared data](#phase-2-tokenizer-artifacts-and-prepared-data)
3. [Pretraining runtime](#phase-3-pretraining-runtime)
4. [Inference and evaluation](2026-08-01-v2-performance-first-refactor-part-2.md#phase-4-inference-and-evaluation)
5. [LoRA and SWAG](2026-08-01-v2-performance-first-refactor-part-2.md#phase-5-lora-and-swag)
6. [Unified CLI and final cutover](2026-08-01-v2-performance-first-refactor-part-2.md#phase-6-unified-cli-and-final-cutover)

Part 2 must not begin until Part 1's Phase 3 gate passes, a valid tiny
pretraining run can be opened through the new artifact API, the committed
quality evidence validates against its recorded source boundary, and checkpoint
reader contents are deeply immutable for the Task 4.1 consumer.

## File Structure for Part 1

Create these production modules:

- `v2/src/sml/__init__.py` — temporary legacy bridge, later the narrow public package exports
- `v2/src/sml/__main__.py` — package entrypoint; dispatch remains minimal until Phase 6
- `v2/src/sml/cli.py` — typed command dispatch seam, completed in Part 2
- `v2/src/sml/errors.py` — configuration, artifact, data, and runtime exceptions
- `v2/src/sml/model/config.py` — frozen initializer, model, and generation configuration
- `v2/src/sml/model/rope.py` — YaRN/RoPE functions and caches
- `v2/src/sml/model/layers.py` — keyed dropout, RMSNorm, attention, MLP, block
- `v2/src/sml/model/cache.py` — explicit fixed-capacity KV state
- `v2/src/sml/model/generation.py` — repetition/no-repeat processors and token selection
- `v2/src/sml/model/language_model.py` — canonical tied/untied model, causal loss, parameter count
- `v2/src/sml/data/corpus.py` — compressed JSONL discovery, filtering, and deterministic file order
- `v2/src/sml/data/tokenizer.py` — tokenizer bundle training/loading
- `v2/src/sml/data/pretraining.py` — NPY preparation, mmap batching, cursors, prefetch
- `v2/src/sml/artifacts/manifest.py` — canonical identities, strict typed schemas, safe artifact-root traversal
- `v2/src/sml/artifacts/checkpoint.py` — locks, atomic publication, checkpoint resolution, recovery, retention
- `v2/src/sml/training/common.py` — schedules, FP32 gradient math, mixed-precision Adam, progress types
- `v2/src/sml/training/pretrain.py` — compiled pretraining step and direct training/resume loop

Create mirrored tests under `v2/tests/unit`, `v2/tests/equivalence`, and `v2/tests/integration`, plus a versioned harness under `v2/benchmarks`. Keep legacy flat source and its tests until Phase 6 except where a new package test replaces ownership without breaking unmigrated workflows.

Test snippets omit routine imports only. Put MLX availability, `assert_close`, `assert_tree_close`, immutable legacy fixture loading, tiny model/tokenizer/corpus builders, and temporary-APFS fixtures in `v2/tests/conftest.py`; put a helper used by only one module above its first test in that module. Artifact fault injectors wrap the explicit filesystem dependency passed to artifact APIs and live in `v2/tests/integration/test_artifact_integrity.py`; do not add test switches to production code.

## Existing Optional Benchmark Tooling

Task 0.1 is already complete. Its original prerequisite wording below is
historical: the baseline remains useful for optional diagnostics but no new
baseline or performance validation is required for phase progression.

### Task 0.1: Version and Record the `3687f8b` Baseline

**Files:**
- Create: `v2/benchmarks/README.md`
- Create: `v2/benchmarks/schema.py`
- Create: `v2/benchmarks/workload.py`
- Create: `v2/benchmarks/runner.py`
- Create: `v2/benchmarks/analysis.py`
- Create: `v2/benchmarks/adapters/legacy.py`
- Create: `v2/benchmarks/adapters/replacement.py`
- Create: `v2/benchmarks/manifests/baseline-3687f8b.json`
- Create: `v2/benchmarks/results/baseline-3687f8b.jsonl`
- Create: `v2/tests/unit/test_benchmark_analysis.py`

**Interfaces:**
- Produces `CanonicalWorkload`, `RawTrial`, `TrialPair`, `MetricReport`, `analyze_pairs`, and CLI operations `record-baseline` / `compare`.
- `analyze_pairs(reference, candidate, *, direction, bootstrap_seed, resamples, minimum_ratio, maximum_dispersion, require_lower_bound) -> MetricReport` always normalizes throughput as candidate/reference and latency/memory as reference/candidate only when a gate explicitly uses that direction.
- The harness content identity is `sha256:` plus SHA-256 over the ordered bytes of `schema.py`, `workload.py`, `runner.py`, `analysis.py`, and both adapter modules.

- [ ] **Step 1: Write deterministic analysis tests**

```python
from v2.benchmarks.analysis import analyze_pairs


def test_throughput_ratio_and_bound_are_direction_normalized():
    report = analyze_pairs(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[103.0, 103.0, 103.0, 103.0, 103.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=1.03,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert report.paired_ratios == (1.03, 1.03, 1.03, 1.03, 1.03)
    assert report.median_ratio == 1.03
    assert report.lower_confidence_bound == 1.03
    assert report.decision == "pass"


def test_noise_and_confidence_fail_closed():
    noisy = analyze_pairs(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[95.0, 100.0, 103.0, 106.0, 110.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=0.97,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert noisy.decision == "too-noisy"


def test_bootstrap_is_reproducible_and_point_only_pass_is_inconclusive():
    arguments = dict(
        reference=[100.0, 100.0, 100.0, 100.0, 100.0],
        candidate=[96.0, 98.0, 100.0, 103.0, 108.0],
        direction="higher-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=0.99,
        maximum_dispersion=0.10,
        require_lower_bound=True,
    )
    first = analyze_pairs(**arguments)
    second = analyze_pairs(**arguments)
    assert first.lower_confidence_bound == second.lower_confidence_bound
    assert first.median_ratio >= 0.99
    assert first.lower_confidence_bound < 0.99
    assert first.decision == "inconclusive"


def test_latency_ratios_reverse_direction():
    report = analyze_pairs(
        reference=[10.0] * 5,
        candidate=[8.0] * 5,
        direction="lower-is-better",
        bootstrap_seed=1729,
        resamples=10_000,
        minimum_ratio=1.20,
        maximum_dispersion=0.02,
        require_lower_bound=True,
    )
    assert report.paired_ratios == (1.25,) * 5
    assert report.decision == "pass"
```

- [ ] **Step 2: Run the analysis test and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -v
```

Expected: FAIL because `v2.benchmarks.analysis` does not exist.

- [ ] **Step 3: Implement the pinned schema, analysis, adapters, and runner**

Use these exact report fields and decision order:

```python
@dataclass(frozen=True, slots=True)
class MetricReport:
    reference_median: float
    reference_mad: float
    candidate_median: float
    candidate_mad: float
    paired_ratios: tuple[float, ...]
    median_ratio: float
    ratio_mad: float
    lower_confidence_bound: float
    decision: Literal["pass", "fail", "inconclusive", "too-noisy"]


def decide(
    report: MetricReport,
    minimum_ratio: float,
    maximum_dispersion: float,
    require_lower_bound: bool,
) -> str:
    dispersions = (
        report.reference_mad / report.reference_median,
        report.candidate_mad / report.candidate_median,
        report.ratio_mad / report.median_ratio,
    )
    if any(value > maximum_dispersion for value in dispersions):
        return "too-noisy"
    if report.median_ratio < minimum_ratio:
        return "fail"
    if require_lower_bound and report.lower_confidence_bound < minimum_ratio:
        return "inconclusive"
    return "pass"
```

The workload schema must carry model/optimizer/precision/loader/generation settings, canonical row/example/request identities, native representation identities, work-unit definitions, synchronization boundaries, power/thermal/memory status, software versions, and exact clean-checkout commits. The runner must alternate reference/candidate process order by pair, call `mx.synchronize()` at both timed boundaries, run one compilation pass, 20 warmups, then 100 measured units, and emit one JSON object per raw process. A fixed request set smaller than 100 is the only allowed exception and its exact size must be recorded. Compute the lower bound with a one-sided 5th-percentile bootstrap over whole pairs. Phase screens set `require_lower_bound=False` but still report it; final acceptance sets it true. Measure mandatory full input verification as separate startup work, and never disable it in end-to-end CLI smoke measurements.

- [ ] **Step 4: Verify and commit the harness before measuring**

Run the test outside the sandbox, verify that the lock file is unchanged, and commit only the harness, adapters, schema, and analysis test. Do not create or stage the baseline manifest/results yet: the committed harness identity must exist before the measurement that records it.

```bash
uv run pytest v2/tests/unit/test_benchmark_analysis.py -v
git diff --exit-code -- uv.lock
git add v2/benchmarks/README.md v2/benchmarks/schema.py v2/benchmarks/workload.py v2/benchmarks/runner.py v2/benchmarks/analysis.py v2/benchmarks/adapters/legacy.py v2/benchmarks/adapters/replacement.py v2/tests/unit/test_benchmark_analysis.py
git commit -m "bench(v2): version performance harness"
```

Expected: the analysis tests pass, `uv.lock` is unchanged, and the committed tree contains the complete harness but no baseline evidence generated by that harness.

- [ ] **Step 5: Capture and validate the baseline from clean checkouts**

Create a dedicated clean checkout at the harness commit from Step 4 and a separate clean source checkout at `3687f8b`. Run the command from the clean harness checkout. Before creating either output file, the runner must record and verify the harness checkout's exact commit and clean status; it must create or validate the separate source checkout at exactly `3687f8b` and record its clean status.

```bash
uv run python -m v2.benchmarks.runner record-baseline --source-commit 3687f8b --manifest v2/benchmarks/manifests/baseline-3687f8b.json --raw-output v2/benchmarks/results/baseline-3687f8b.jsonl
uv run python -m v2.benchmarks.runner validate --manifest v2/benchmarks/manifests/baseline-3687f8b.json --raw-input v2/benchmarks/results/baseline-3687f8b.jsonl
```

Expected for an optional recapture: validation passes; the manifest records the already-committed clean harness commit/content identity, the clean `3687f8b` source proof, M5 hardware identity, paired legacy NPZ/`uint16` input identity, canonical ordered `int32` row identity, all raw metric values, and no invalid thermal/power/memory condition. This diagnostic no longer controls whether Phase 1 or any later phase may start.

- [ ] **Step 6: Commit the baseline evidence**

```bash
git add v2/benchmarks/manifests/baseline-3687f8b.json v2/benchmarks/results/baseline-3687f8b.jsonl
git commit -m "bench(v2): record pinned performance baseline"
```

## Phase 1: Foundation and Model Package

### Task 1.1: Create the Installable Package and Temporary Bridge

**Files:**
- Modify: `pyproject.toml`
- Create: `v2/src/sml/__init__.py`
- Create: `v2/src/sml/__main__.py`
- Create: `v2/src/sml/cli.py`
- Create: `v2/src/sml/errors.py`
- Create: `v2/src/sml/model/__init__.py`
- Create: `v2/src/sml/data/__init__.py`
- Create: `v2/src/sml/artifacts/__init__.py`
- Create: `v2/src/sml/training/__init__.py`
- Create: `v2/tests/conftest.py`
- Create: `v2/tests/unit/test_package.py`
- Modify: `v2/tests/test_module_layout.py`
- Modify: `v2/tests/test_sml.py`
- Modify: `v2/tests/test_train.py`
- Modify: `v2/tests/test_infer.py`
- Modify: `v2/tests/test_evaluate.py`
- Modify: `v2/tests/test_lora.py`
- Modify: `v2/tests/test_ft_swag.py`

**Interfaces:**
- Produces `sml.errors.{SMLConfigurationError,SMLArtifactError,SMLDataError,SMLRuntimeError}`.
- `python -m sml` delegates only to `sml.cli.main(argv: Sequence[str] | None = None) -> int`.
- Through Phase 5, the bridge exposes the exact legacy public model surface still imported by flat production modules and their tests: `ParameterInitializerRangeConfig`, `SMLConfig`, `GenerationConfig`, `SMLForwardOutput`, `yarn_find_correction_dim`, `yarn_find_correction_range`, `yarn_get_mscale`, `resolve_yarn_attention_factor`, `yarn_linear_ramp_mask`, `rotate_half`, `apply_rotary_pos_emb`, `apply_repetition_penalty`, `apply_no_repeat_ngram`, `select_next_token`, `RMSNorm`, `RotaryEmbedding`, `KVCache`, `GroupedQueryAttention`, `SwiGLUFeedForward`, `TransformerBlock`, `SMLLanguageModel`, `compute_causal_lm_loss`, `count_parameters`, `create_model`, and `estimate_model_size`. This exact tuple is named `LEGACY_BRIDGE_EXPORTS`; no imported libraries or private helpers are forwarded.
- New package code imports replacements from their owner modules, never from the bridged top-level names. The complete bridge remains stable until Task 6.3 removes every flat consumer and replaces `sml.__init__` atomically; landing an owner-module replacement earlier does not remove or shadow a bridge export.

- [ ] **Step 1: Write package/bridge tests**

```python
import importlib
import subprocess
import sys

import pytest


EXPECTED_LEGACY_BRIDGE_EXPORTS = {
    "ParameterInitializerRangeConfig", "SMLConfig", "GenerationConfig", "SMLForwardOutput",
    "yarn_find_correction_dim", "yarn_find_correction_range", "yarn_get_mscale",
    "resolve_yarn_attention_factor", "yarn_linear_ramp_mask", "rotate_half",
    "apply_rotary_pos_emb", "apply_repetition_penalty", "apply_no_repeat_ngram",
    "select_next_token", "RMSNorm", "RotaryEmbedding", "KVCache",
    "GroupedQueryAttention", "SwiGLUFeedForward", "TransformerBlock",
    "SMLLanguageModel", "compute_causal_lm_loss", "count_parameters", "create_model",
    "estimate_model_size",
}


def test_package_wins_over_legacy_module():
    import sml

    assert sml.__file__.endswith("sml/__init__.py")
    assert sml.SMLLanguageModel.__name__ == "SMLLanguageModel"


def test_bridge_covers_every_unmigrated_flat_import():
    import sml

    assert set(sml.LEGACY_BRIDGE_EXPORTS) == EXPECTED_LEGACY_BRIDGE_EXPORTS
    legacy = sys.modules["sml._legacy"]
    for name in EXPECTED_LEGACY_BRIDGE_EXPORTS:
        assert getattr(sml, name) is getattr(legacy, name)


@pytest.mark.parametrize("module_name", ["train_sml", "infer_sml", "evaluate_sml", "lora", "ft_swag"])
def test_every_unmigrated_flat_module_imports_through_bridge(module_name):
    importlib.import_module(module_name)


def test_module_entrypoint_is_available():
    completed = subprocess.run(
        [sys.executable, "-m", "sml", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "SML workflows" in completed.stdout
```

- [ ] **Step 2: Run package and all legacy import smoke tests; verify RED**

```bash
uv run pytest v2/tests/unit/test_package.py v2/tests/test_module_layout.py -v
```

Expected: FAIL because `v2/src/sml/` and the source mapping do not exist.

- [ ] **Step 3: Add package mapping, focused errors, entrypoint, and bridge**

Add only these packaging tables to `pyproject.toml` and leave `uv.lock` unchanged:

```toml
[build-system]
requires = ["uv_build>=0.11.0,<0.12.0"]
build-backend = "uv_build"

[tool.uv.build-backend]
module-name = "sml"
module-root = "v2/src"
```

Implement `sml.__main__` as:

```python
from sml.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Load the legacy file with `importlib.util.spec_from_file_location("sml._legacy", legacy_path)` so package precedence cannot recurse through `import sml`; insert the module into `sys.modules` before executing it, then re-export exactly `LEGACY_BRIDGE_EXPORTS`. Resolve the legacy sibling-import path without changing global import precedence for the replacement package. `cli.main` must return help successfully and raise `SMLConfigurationError("a command is required")` for an empty non-help dispatch. Keep optional dependencies out of imports.

- [ ] **Step 4: Verify package precedence and every unmigrated consumer**

```bash
uv run pytest v2/tests/unit/test_package.py v2/tests/test_module_layout.py v2/tests/test_sml.py v2/tests/test_train.py v2/tests/test_infer.py v2/tests/test_evaluate.py v2/tests/test_lora.py v2/tests/test_ft_swag.py -v
uv run python -m sml --help
git diff --exit-code -- uv.lock
```

Expected: all tests pass, help prints without importing `datasets` or `lm_eval`, and `uv.lock` is unchanged. If the installed `uv` cannot honor the approved mapping without a lock/dependency change, stop this task and report the exact resolver output rather than editing `uv.lock`.

- [ ] **Step 5: Commit the package foundation**

```bash
git add pyproject.toml v2/src/sml v2/tests/conftest.py v2/tests/unit/test_package.py v2/tests/test_module_layout.py v2/tests/test_sml.py v2/tests/test_train.py v2/tests/test_infer.py v2/tests/test_evaluate.py v2/tests/test_lora.py v2/tests/test_ft_swag.py
git commit -m "refactor(v2): establish sml package foundation"
```

### Task 1.2: Capture Equivalence Fixtures Before Replacing the Model

**Files:**
- Create: `v2/tests/equivalence/capture_legacy.py`
- Create: `v2/tests/equivalence/fixtures/legacy-control.json`
- Create: `v2/tests/equivalence/fixtures/legacy-arrays.safetensors`
- Create: `v2/tests/equivalence/test_legacy_fixture_integrity.py`
- Modify: `v2/tests/conftest.py`

**Interfaces:**
- `capture_legacy.py` is a committed driver/worker script. The driver runs from its own clean capture-tool commit, creates a separate detached clean worktree at `3687f8b`, then launches the same absolute script in `--worker` mode with the source worktree's `v2/src` first on `PYTHONPATH`; the new script is never copied into or expected to exist in `3687f8b`.
- Before starting MLX, the driver records and verifies its own commit/clean status and the worker verifies the source worktree's exact `3687f8b` commit/clean status. The fixture records both the capture-tool commit/content identity and source commit.
- The worker writes deterministic inputs, outputs, MLX dtypes, seeds, margins, source commit, and the exact evaluated parameter arrays used to produce every parameter-dependent reference.
- `legacy-arrays.safetensors` stores one canonical tied vocabulary weight plus every other base-model leaf under replacement-loadable `model_state.*` names. It also stores the exact base and adapter leaves for LoRA cases under `lora_base_state.*` and `lora_state.*`. `legacy-control.json` records the ordered legacy-to-replacement name mapping, shape, dtype, and payload identity. The duplicated legacy `lm_head.weight` is asserted byte-identical to `embed_tokens.weight` and is represented only by the canonical embedding leaf in replacement state.
- `load_legacy_model_state(model, legacy_arrays, legacy_control)` and `load_legacy_lora_state(model, legacy_arrays, legacy_control)` live in `v2/tests/conftest.py`, validate the complete expected destination name/shape/dtype set, assign the captured arrays, and evaluate the loaded state before a reference comparison. A legacy-equivalence test may never rely on a seed alone to recreate parameter values.
- Fixtures cover model logits/loss, YaRN ranges/rotations, GQA output, sequential/chunked cache, greedy/seeded generation, serial/unequal batch generation, serial/padded log-likelihood, LoRA forward/merge, compiled consecutive steps, and the legacy leaves used by the two correction oracles. Pretraining cases use `rope_scaling_factor=1.0`; separate RoPE cases cover factors above `1.0` only to preserve model mathematics for the future long-context fine-tuning workflow.

- [ ] **Step 1: Write fixture schema/integrity tests**

```python
def test_legacy_fixture_records_source_and_required_cases(legacy_control):
    assert legacy_control["source_commit"] == "3687f8b"
    assert legacy_control["capture_tool_commit"] == CAPTURE_TOOL_COMMIT
    assert legacy_control["capture_tool_identity"].startswith("sha256:")
    assert legacy_control["pretraining_model_config"]["rope_scaling_factor"] == 1.0
    assert set(legacy_control["cases"]) == {
        "model",
        "parameter_state",
        "rope",
        "gqa",
        "cache",
        "generation",
        "loglikelihood",
        "tied_gradient_leaves",
        "lora",
        "compiled_state",
        "swag_legacy_sum",
    }
    assert legacy_control["dropout"] == 0.0
    assert legacy_control["parameter_state"]["tied_source_names"] == [
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    assert legacy_control["parameter_state"]["canonical_destination"] == "embed_tokens.weight"
    assert legacy_control["parameter_state"]["payload_identity"].startswith("sha256:")
```

- [ ] **Step 2: Run the integrity test and verify RED**

```bash
uv run pytest v2/tests/equivalence/test_legacy_fixture_integrity.py -v
```

Expected: FAIL because the captured fixture is absent.

- [ ] **Step 3: Implement deterministic capture with explicit synchronization**

Implement two explicit modes in the same committed script:

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        assert_clean_checkout(args.source_root, expected_commit="3687f8b")
        capture_from_loaded_legacy_source(args)
        return 0
    assert_clean_checkout(PROJECT_ROOT, expected_commit=args.capture_tool_commit)
    with detached_worktree(PROJECT_ROOT, commit="3687f8b") as source_root:
        run_worker_from_external_script(args, source_root=source_root)
    return 0
```

The driver hashes the ordered bytes of `capture_legacy.py` and the fixture-schema test, then passes its clean commit and content identity to the worker. The worker imports only `source_root / "v2/src"`, uses a fixed pretraining config whose `rope_scaling_factor` is exactly `1.0` plus a small exact fixture config, sets CPU and MLX seeds explicitly, applies the same BF16 base-parameter conversion used by the pinned semantic workload, and calls `mx.eval` on the complete legacy model/base/adapter state before running or serializing any reference. Flatten parameters in stable sorted-name order, reject aliases with different bytes, write the canonical replacement-loadable arrays and ordered mapping described above, and then compute all parameter-dependent outputs from that exact evaluated state. Save arrays with stable names and record generated-token winning margins. For corrections, store both tied legacy gradient leaves separately at the captured parameter state and store legacy SWAG sums only as proof that the changed path is exercised.

- [ ] **Step 4: Commit the capture tool before producing evidence**

```bash
uv run pytest v2/tests/equivalence/test_legacy_fixture_integrity.py -k "capture_driver" -v
git add v2/tests/conftest.py v2/tests/equivalence/capture_legacy.py v2/tests/equivalence/test_legacy_fixture_integrity.py
git commit -m "test(v2): version legacy fixture capture tool"
```

Expected: driver-only tests pass, fixture-presence tests remain unselected, and `git status --short` is empty after the commit. Record this exact commit as `CAPTURE_TOOL_COMMIT`; do not amend it after fixture production.

- [ ] **Step 5: Capture through separate clean driver/source checkouts**

```bash
capture_a="$(mktemp -d)"
capture_b="$(mktemp -d)"
uv run python v2/tests/equivalence/capture_legacy.py --capture-tool-commit HEAD --source-commit 3687f8b --output-control "$capture_a/legacy-control.json" --output-arrays "$capture_a/legacy-arrays.safetensors"
uv run python v2/tests/equivalence/capture_legacy.py --capture-tool-commit HEAD --source-commit 3687f8b --output-control "$capture_b/legacy-control.json" --output-arrays "$capture_b/legacy-arrays.safetensors"
cmp "$capture_a/legacy-control.json" "$capture_b/legacy-control.json"
cmp "$capture_a/legacy-arrays.safetensors" "$capture_b/legacy-arrays.safetensors"
mkdir -p v2/tests/equivalence/fixtures
install -m 0644 "$capture_a/legacy-control.json" v2/tests/equivalence/fixtures/legacy-control.json
install -m 0644 "$capture_a/legacy-arrays.safetensors" v2/tests/equivalence/fixtures/legacy-arrays.safetensors
uv run pytest v2/tests/equivalence/test_legacy_fixture_integrity.py -v
```

Expected: both driver invocations start from the still-clean capture-tool commit because their outputs remain outside the repository until both finish; each separate source worktree is clean at `3687f8b`; `cmp` proves byte identity before one copy is installed; and the fixture test passes. The recorded mapping accounts for every captured legacy base/adapter leaf, proves the tied aliases are byte-identical, and contains no duplicate destination. Task 1.4 performs the first strict load into the replacement model once that model exists.

- [ ] **Step 6: Commit immutable reference fixtures**

```bash
git add v2/tests/equivalence/fixtures/legacy-control.json v2/tests/equivalence/fixtures/legacy-arrays.safetensors
git commit -m "test(v2): capture legacy model equivalence fixtures"
```

### Task 1.3: Extract Frozen Configuration and YaRN/RoPE

**Files:**
- Create: `v2/src/sml/model/config.py`
- Create: `v2/src/sml/model/rope.py`
- Create: `v2/tests/unit/model/test_config.py`
- Create: `v2/tests/unit/model/test_rope.py`
- Create: `v2/tests/equivalence/test_model_math.py`

**Interfaces:**
- Produces frozen `InitializerConfig`, `ModelConfig`, and `GenerationConfig` dataclasses with these exact fields/defaults; the training-only `PrecisionConfig` is defined once in Task 3.1:

```python
@dataclass(frozen=True, slots=True)
class InitializerConfig:
    embed_tokens: float = 0.02
    lm_head: float = 0.02
    q_proj: float = 0.02
    k_proj: float = 0.02
    v_proj: float = 0.02
    o_proj: float = 0.02 / math.sqrt(24)
    gate_proj: float = 0.02
    up_proj: float = 0.02
    down_proj: float = 0.02 / math.sqrt(24)
    other: float = 0.02


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 28_672
    hidden_size: int = 768
    num_layers: int = 12
    num_q_heads: int = 12
    num_kv_heads: int = 3
    intermediate_size: int = 2_176
    original_context_length: int = 1_024
    rope_theta: float = 10_000.0
    rope_scaling_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attention_factor: float | None = None
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_truncate: bool = True
    rms_norm_epsilon: float = 1e-6
    hidden_dropout: float = 0.01
    initializer_range: float = 0.02
    initializers: InitializerConfig | Mapping[str, float] | None = None
    pad_token_id: int = 3
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 0
    tie_word_embeddings: bool = True
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: int | None = None
```

- When `initializers is None`, `ModelConfig.__post_init__` installs `InitializerConfig.depth_scaled(initializer_range, num_layers)` with `object.__setattr__`; therefore changing `num_layers` still derives the residual `o_proj`/`down_proj` standard deviations from the changed depth. An explicit `initializers` value is preserved exactly. `rope_scaling_factor` defaults to `1.0` so the default base-pretraining CLI is valid, while model math continues to accept validated values above `1.0` for the future distinct long-context workflow.
- `ModelConfig.head_dim` and `effective_context_length` are read-only properties.
- Produces `find_correction_dimension`, `find_correction_range`, `resolve_attention_factor`, `RotaryEmbedding`, `rotate_half`, and `apply_rotary`.
- RoPE caches are FP32; rotated Q/K are returned in the input BF16 dtype.

- [ ] **Step 1: Write config and RoPE equivalence tests**

```python
def test_model_config_is_frozen_and_derives_context():
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=8,
        rope_scaling_factor=2.0,
    )
    assert config.head_dim == 4
    assert config.effective_context_length == 16
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.hidden_size = 32


def test_rope_matches_captured_reference(legacy_arrays, tiny_model_config):
    rope = RotaryEmbedding(tiny_model_config)
    actual_q, actual_k = rope(
        legacy_arrays["rope.q"],
        legacy_arrays["rope.k"],
        legacy_arrays["rope.positions"],
    )
    mx.eval(actual_q, actual_k)
    assert actual_q.dtype == mx.bfloat16
    assert_close(actual_q, legacy_arrays["rope.output_q"], atol=2e-2, rtol=2e-2)
    assert_close(actual_k, legacy_arrays["rope.output_k"], atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
uv run pytest v2/tests/unit/model/test_config.py v2/tests/unit/model/test_rope.py v2/tests/equivalence/test_model_math.py -k "config or rope" -v
```

Expected: FAIL because package configuration and RoPE modules are missing.

- [ ] **Step 3: Port validation and math without constructor mutation**

Move the legacy formulas, rename `original_max_position_embeddings` to `original_context_length`, expose the exact derived properties above, and precompute FP32 inverse frequencies/cosine/sine. Reject nonfinite values, invalid head divisibility, odd head dimension, duplicate/out-of-range special-token IDs, invalid generation controls, and incomplete YaRN scale pairs in `__post_init__`; generation seeds are `None` or integers in `[0, 2**32 - 1]`. Use `object.__setattr__` only to derive/normalize `initializers` from `None` or a nested mapping into the frozen dataclass.

- [ ] **Step 4: Verify unit/equivalence tests and bridge imports**

```bash
uv run pytest v2/tests/unit/model/test_config.py v2/tests/unit/model/test_rope.py v2/tests/equivalence/test_model_math.py v2/tests/unit/test_package.py -v
```

Expected: all tests pass; integer/control outputs are exact and floating outputs meet the pinned BF16/FP32 tolerances.

- [ ] **Step 5: Commit config and RoPE**

```bash
git add v2/src/sml/model/config.py v2/src/sml/model/rope.py v2/tests/unit/model v2/tests/equivalence/test_model_math.py
git commit -m "refactor(v2): extract model configuration and rope"
```

### Task 1.4: Build Explicit Cache, Layers, and Canonically Tied Model

**Files:**
- Create: `v2/src/sml/model/cache.py`
- Create: `v2/src/sml/model/layers.py`
- Create: `v2/src/sml/model/language_model.py`
- Modify: `v2/src/sml/model/__init__.py`
- Modify: `v2/src/sml/__init__.py`
- Create: `v2/tests/unit/model/test_cache.py`
- Create: `v2/tests/unit/model/test_layers.py`
- Create: `v2/tests/unit/model/test_language_model.py`
- Modify: `v2/tests/equivalence/test_model_math.py`

**Interfaces:**
- `KVArrayState` is exactly `tuple[tuple[mx.array, ...], tuple[mx.array, ...], mx.array]`: per-layer key arrays, per-layer value arrays, and an `int32` logical-length array. It is a built-in MLX array tree and is the only cache representation accepted by compiled functions.
- `allocate_kv_state(config, batch_size, capacity, dtype) -> KVArrayState` and `append_kv_state(state, layer_index, keys, values, positions, valid_mask) -> tuple[KVArrayState, KVView]` are pure array-tree functions.
- `KVCache.allocate(config, batch_size, capacity, dtype) -> KVCache`, `KVCache.state -> KVArrayState`, `KVCache.replace_state(state) -> None`, and `reset() -> None` form the eager/session owner wrapper. `KVCache` itself never crosses `mx.compile`.
- `keyed_dropout(x, probability, key) -> tuple[mx.array, mx.array]` consumes no key when probability is zero.
- `SMLLanguageModel(config, *, key)`, `initialize_state(key)`, and `__call__(input_ids, *, attention_mask=None, positions=None, cache=None, training=False, key=None) -> ForwardOutput`.
- `SMLLanguageModel.forward_arrays(parameters, input_ids, *, attention_mask, positions, cache_state, training, key) -> tuple[mx.array, KVArrayState | None, mx.array | None]` evaluates the immutable model structure from the explicit built-in BF16 parameter tree without mutating the module and returns arrays only. Compiled training/inference cores call this method; eager `__call__` unwraps an optional `KVCache`, passes the module's current BF16 parameter tree to the same pure graph, and wraps the returned cache state in `ForwardOutput`.
- `ForwardOutput` contains `logits: mx.array`, `cache: KVCache | None`, and `next_key: mx.array | None`.
- `causal_lm_loss(logits, labels, valid_mask=None) -> mx.array` returns FP32.

- [ ] **Step 1: Write layer/cache/model dtype, equivalence, and correction tests**

```python
def test_tied_model_registers_one_vocabulary_parameter_and_sums_gradients(
    tiny_model_config,
    legacy_arrays,
    legacy_control,
):
    model = SMLLanguageModel(replace(tiny_model_config, tie_word_embeddings=True), key=mx.random.key(7))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    names = [name for name, _value in tree_flatten(model.trainable_parameters())]
    assert names.count("embed_tokens.weight") == 1
    assert "lm_head.weight" not in names
    loss_and_grad = nn.value_and_grad(model, causal_batch_loss)
    _loss, grads = loss_and_grad(model, legacy_arrays["model.input_ids"], legacy_arrays["model.labels"])
    expected = legacy_arrays["tied.embedding_grad"] + legacy_arrays["tied.head_grad"]
    assert_close(grads["embed_tokens"]["weight"], expected, atol=2e-2, rtol=2e-2)


def test_base_precision_boundaries(tiny_model_config):
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(9))
    output = model(mx.array([[1, 2, 3]], dtype=mx.int32), training=False)
    mx.eval(output.logits)
    assert all(value.dtype == mx.bfloat16 for _name, value in tree_flatten(model.parameters()))
    assert output.logits.dtype == mx.bfloat16
    assert model.layers[0].self_attn.rope.cos_cache.dtype == mx.float32
```

Add exact cache sequential/chunk equivalence, GQA output, RMSNorm FP32-reduction/BF16-output, keyed-dropout replay, untied-head, no-forward-validation-sync, and logits/loss fixture tests. Before every comparison to a parameter-dependent legacy output or gradient, call `load_legacy_model_state`; initialization keys test explicit randomness separately and are never used as a substitute for captured weights.

Add a compiled-cache contract test that passes `KVCache.state` to a core returning a new `KVArrayState`, evaluates two consecutive calls, and proves the second call sees the first call's returned lengths and payload. Add a negative source test proving no function decorated by `mx.compile` accepts or returns `KVCache`, `KVView`, `ForwardOutput`, or another custom dataclass.

- [ ] **Step 2: Run model tests and verify RED**

```bash
uv run pytest v2/tests/unit/model/test_cache.py v2/tests/unit/model/test_layers.py v2/tests/unit/model/test_language_model.py v2/tests/equivalence/test_model_math.py -v
```

Expected: FAIL because the new cache/layers/model do not exist.

- [ ] **Step 3: Implement the direct model graph and canonical tie**

Register only `embed_tokens.weight` when tied and compute output logits with:

```python
def project_vocabulary(self, hidden: mx.array) -> mx.array:
    if self.config.tie_word_embeddings:
        return hidden @ self.embed_tokens.weight.T
    return self.lm_head(hidden)
```

Implement RMSNorm and attention reductions in FP32, cast outputs back to BF16, keep RoPE caches FP32, use fused scaled-dot-product attention with Boolean masks, and allocate cache arrays once. Split one explicit key per active dropout site in canonical layer order. `forward_arrays` indexes the explicit BF16 working-parameter tree and evaluates the model graph without calling `model.update(...)`, mutating a captured module, or installing traced arrays into a host owner. It returns only built-in array trees; the captured model/config shell never appears in a compiled signature or result and contains only immutable structure/configuration. `append_kv_state` returns replacement arrays instead of mutating a Python cache object; eager/session wrappers install returned state only after successful evaluation. Forward code must contain no `.item()`, `.tolist()`, or `mx.eval` validation.

Export the replacement types from `sml.model`, but do not replace, remove, or shadow any `LEGACY_BRIDGE_EXPORTS` entry in top-level `sml.__init__`; flat workflows still require the legacy call signatures through Phase 5. New package tests and code import `sml.model.language_model` or `sml.model` directly.

- [ ] **Step 4: Verify all model and legacy consumer tests**

```bash
uv run pytest v2/tests/unit/model v2/tests/equivalence/test_model_math.py v2/tests/test_sml.py v2/tests/test_train.py v2/tests/test_ft_swag.py -v
```

Expected: new tests pass through owner-module imports, every unmigrated workflow still receives the legacy object through the unchanged bridge, and the bridge coverage test still proves all flat imports resolve.

- [ ] **Step 5: Commit model core**

```bash
git add v2/src/sml/model v2/src/sml/__init__.py v2/tests/unit/model v2/tests/equivalence/test_model_math.py
git commit -m "refactor(v2): build canonical mlx language model"
```

### Task 1.5: Extract Token Processors and Explicit Sampling

**Files:**
- Create: `v2/src/sml/model/generation.py`
- Create: `v2/tests/unit/model/test_generation.py`
- Modify: `v2/tests/equivalence/test_model_math.py`

**Interfaces:**
- `apply_repetition_penalty(logits, tokens, logical_lengths, penalty) -> mx.array`.
- `apply_no_repeat_ngram(logits, tokens, logical_lengths, ngram_size) -> mx.array`.
- `select_next_token_arrays(logits_row, key, *, temperature, top_p) -> tuple[mx.array, mx.array]` is the array-only single-row primitive used by later compiled `vmap`; `logits_row` is rank one and `key` has shape `(2,)`.
- `select_next_token(logits, config, key) -> TokenSelection(token_ids, next_key)`; greedy mode consumes no key.
- All processors stay on device and accept batched token storage without Python conversion.

- [ ] **Step 1: Write processor and sampling equivalence tests**

```python
def test_greedy_selection_does_not_advance_key():
    key = mx.random.key(11)
    selected = select_next_token(
        mx.array([[0.0, 3.0, 1.0]], dtype=mx.float32),
        GenerationConfig(temperature=0.0),
        key,
    )
    mx.eval(selected.token_ids, selected.next_key)
    assert selected.token_ids.tolist() == [1]
    assert mx.array_equal(selected.next_key, key).item()


def test_seeded_sampling_matches_legacy_fixture(legacy_arrays):
    result = select_next_token(
        legacy_arrays["generation.logits"],
        GenerationConfig(temperature=0.8, top_p=0.9),
        mx.random.key(1234),
    )
    mx.eval(result.token_ids)
    assert mx.array_equal(result.token_ids, legacy_arrays["generation.sampled_token"]).item()
```

- [ ] **Step 2: Run generation tests and verify RED**

```bash
uv run pytest v2/tests/unit/model/test_generation.py v2/tests/equivalence/test_model_math.py -k "generation or processor or sampling" -v
```

Expected: FAIL because the package generation primitives are absent.

- [ ] **Step 3: Port processors using MLX indexing and split-return keys**

Implement repetition sign handling, n-gram blocking with fixed-capacity device comparisons, stable top-p sorting/masking, direct `mx.random.categorical(key=...)`, and one `mx.random.split` per sampled request. The public host wrapper vectorizes only for eager compatibility and wraps the returned arrays; compiled callers close over validated scalar policy constants and call/vmap `select_next_token_arrays` directly. Validate configuration at request construction, not in the device processor.

- [ ] **Step 4: Verify generation unit/equivalence tests**

```bash
uv run pytest v2/tests/unit/model/test_generation.py v2/tests/equivalence/test_model_math.py -v
```

Expected: greedy and seeded outputs match fixtures, processor masks match exactly, and no processor source contains `.item(` or `.tolist(`.

- [ ] **Step 5: Commit generation primitives**

```bash
git add v2/src/sml/model/generation.py v2/tests/unit/model/test_generation.py v2/tests/equivalence/test_model_math.py
git commit -m "refactor(v2): add explicit mlx generation primitives"
```

### Phase 1 Functional Gate

- [ ] Run full correctness checks:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
```

Expected: Ruff and all v2 tests pass and the tracked worktree is clean. An
optional Phase 1 performance comparison may be recorded separately, but its
absence or result does not block Phase 2.

## Phase 2: Tokenizer, Artifacts, and Prepared Data

### Task 2.1: Implement Canonical Identities and Strict Typed Manifests

**Files:**
- Create: `v2/src/sml/artifacts/manifest.py`
- Create: `v2/tests/unit/artifacts/test_manifest.py`
- Create: `v2/tests/integration/test_artifact_integrity.py`

**Interfaces:**
- `canonical_json_bytes(value: object) -> bytes` implements only `sml-json-v1`.
- `file_identity(file: BinaryIO) -> str`, `structured_identity(domain_tag: str, value: object) -> str`, and `row_content_identity(rows: Iterable[np.ndarray], row_count: int, row_width: int) -> str`.
- Strict frozen manifests: `TokenizerManifest`, `PretrainingDataManifest`,
  distinct `PretrainingCheckpointManifest` and `LoRACheckpointManifest`,
  distinct `PretrainingRunManifest` and `LoRARunManifest`, `LatestIndex`,
  `BaseSnapshotManifest`, `SwagDataManifest`, and `ExportManifest`.
- `read_manifest(root: Path, manifest_type: type[M], verification: VerificationLevel) -> Verified[M]` rejects unknown/missing fields and always recomputes the structured identity.
- `VerificationLevel` values are exactly `manifest-trusted` and `full`.

Use one shared payload vocabulary: `PayloadRef(logical_path, identity, byte_size)`, `ArraySpec(name, shape, dtype)`, and `ArrayPayloadRef(payload, arrays)`. `TokenizerManifest` fields are exactly `kind`, `version`, `identity`, `algorithm`, `training`, `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `unk_token_id`, `model: PayloadRef`, `vocab: PayloadRef`, and `diagnostic_source_locator: str | None`; only the last field is excluded from identity. `PretrainingDataManifest` contains sequence/row width/dtype, shard row counts and ordered refs, preparation seed/order policy, tokenizer identity/copied tokenizer refs, source summary/diagnostic locator, and row-content identity. The distinct pretraining and LoRA run manifests each contain only their kind's immutable model/precision/optimizer/loader/checkpoint configuration and tokenizer/base/data identities. A pretraining run's model must record `rope_scaling_factor=1.0`. The distinct checkpoint manifests each bind the owning-run identity, checkpoint step, scalar-state ref, and that kind's exact fixed array payload groups; strict readers reject fields or groups from the other kind. `RunManifest` and `CheckpointManifest` may be union aliases only, never generic optional-field schemas. `LatestIndex` contains owning-run identity, step, and checkpoint identity. `BaseSnapshotManifest`, `SwagDataManifest`, and `ExportManifest` contain the complete fields stated by their owner tasks in Part 2; define their schema versions here so later tasks cannot invent alternate encodings.

- [ ] **Step 1: Write canonical-vector and strict-schema tests**

```python
def test_sml_json_v1_vectors_are_stable():
    left = {"z": -0.0, "a": ("雪", 1, 1.0)}
    right = {"a": ["雪", 1, 1.0], "z": 0.0}
    expected = b'{"a":["\xe9\x9b\xaa",1,1.0],"z":0}'
    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected


def test_manifest_identity_ignores_diagnostic_locator_but_not_payload():
    first = tokenizer_manifest_fixture(
        diagnostic_source_locator="/old/path",
        model=PayloadRef(
            logical_path="tokenizer.model",
            identity="sha256:" + "a" * 64,
            byte_size=10,
        ),
    )
    moved = replace(first, diagnostic_source_locator="/new/path")
    changed = replace(
        first,
        model=replace(first.model, identity="sha256:" + "b" * 64),
    )
    assert first.recompute_identity() == moved.recompute_identity()
    assert first.recompute_identity() != changed.recompute_identity()


def test_strict_parser_rejects_unknown_field(tmp_path):
    write_tokenizer_manifest(tmp_path, extra={"legacy_format": 1})
    with pytest.raises(SMLArtifactError, match="unknown field.*legacy_format"):
        read_manifest(tmp_path, TokenizerManifest, VerificationLevel.MANIFEST_TRUSTED)
```

- [ ] **Step 2: Run manifest tests and verify RED**

```bash
uv run pytest v2/tests/unit/artifacts/test_manifest.py v2/tests/integration/test_artifact_integrity.py -k "identity or manifest" -v
```

Expected: FAIL because canonical encoding and schemas do not exist.

- [ ] **Step 3: Implement the sole canonical encoder and schema registry**

Normalize dataclasses, enums, `Path`, tuples, schema-typed numeric values, negative zero, UTF-8, and insertion order; reject nonfinite values and surrogate Unicode. Use `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`. Identity projections must exclude only `identity`, creation times, temporary names, absolute diagnostic locators, and explicitly typed diagnostic fields. Hash structured values as `domain_tag.encode() + b"\0" + canonical_json_bytes(projection)`.

Every schema parser must compare `set(raw)` with its exact field set, validate kind/version before constructing, and validate identity strings against `sha256:[0-9a-f]{64}`.

- [ ] **Step 4: Verify vectors across fresh processes and semantic mutations**

```bash
uv run pytest v2/tests/unit/artifacts/test_manifest.py v2/tests/integration/test_artifact_integrity.py -k "identity or manifest" -v
uv run python -c 'from sml.artifacts.manifest import canonical_json_bytes; print(canonical_json_bytes({"b":-0.0,"a":"雪"}).hex())'
uv run python -c 'from sml.artifacts.manifest import canonical_json_bytes; print(canonical_json_bytes({"a":"雪","b":0.0}).hex())'
```

Expected: tests pass and the two printed hex values are identical.

- [ ] **Step 5: Commit manifest contracts**

```bash
git add v2/src/sml/artifacts/manifest.py v2/tests/unit/artifacts/test_manifest.py v2/tests/integration/test_artifact_integrity.py
git commit -m "feat(v2): define strict artifact identities"
```

### Task 2.2: Add Descriptor-Relative Artifact Traversal

**Files:**
- Modify: `v2/src/sml/artifacts/manifest.py`
- Create: `v2/tests/unit/artifacts/test_artifact_root.py`
- Modify: `v2/tests/integration/test_artifact_integrity.py`

**Interfaces:**
- `parse_logical_path(value: str) -> tuple[str, ...]` applies `unicodedata.normalize("NFKC", component)`, then the exact portable-ASCII grammar `[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?`, alternate-separator rejection, and normalized `casefold()` collision keys without touching the filesystem.
- `ArtifactRoot.open(path: Path, *, writable: bool) -> ArtifactRoot` opens the root once with `O_DIRECTORY | O_NOFOLLOW`, validates it with `fstat`, records whether local APFS writer guarantees are available, and owns the descriptor until `close()` / context exit.
- `ArtifactRoot.open_payload(logical_path: str) -> BinaryIO` walks every component relative to owned descriptors, opens the final regular file with `O_NOFOLLOW`, verifies `st_nlink == 1`, and rejects repeated `(st_dev, st_ino)` payload aliases.
- `ArtifactRoot.verify_payloads(refs: Sequence[PayloadRef], *, full: bool) -> None` checks exact paths/metadata and hashes bytes only when `full=True`.

- [ ] **Step 1: Write lexical-path and descriptor traversal tests**

```python
@pytest.mark.parametrize(
    "logical_path",
    ["/absolute", "../escape", "a//b", "a/./b", "a\\b", "trail."],
)
def test_logical_paths_are_rejected_before_open(tmp_path, logical_path):
    with ArtifactRoot.open(tmp_path, writable=False) as root:
        with pytest.raises(SMLArtifactError, match="logical path"):
            root.open_payload(logical_path)


def test_open_payload_rejects_symlink_swap_and_hard_link_alias(
    artifact_with_payload,
    swap_component_for_symlink,
    external_hard_link,
):
    with ArtifactRoot.open(artifact_with_payload, writable=False) as root:
        swap_component_for_symlink()
        with pytest.raises(SMLArtifactError, match="symlink|no-follow"):
            root.open_payload("nested/model.safetensors")
    with pytest.raises(SMLArtifactError, match="link count"):
        verify_artifact(external_hard_link.bundle, full=True)
```

Name the remaining cases exactly `test_exact_duplicate_logical_paths_collide`,
`test_unicode_normalized_paths_collide`, `test_casefolded_paths_collide`,
`test_two_logical_paths_cannot_share_inode`,
`test_non_apfs_writer_is_rejected`, and
`test_non_apfs_read_only_open_reports_no_writer_guarantee`. Cover exact
duplicate prepared-shard and checkpoint-group paths separately from normalized
spelling collisions. The Unicode case uses an NFKC-equivalent full-width/ASCII
component pair; each test creates the named filesystem condition before calling
the public API and asserts the focused `SMLArtifactError` category.

- [ ] **Step 2: Run traversal tests and verify RED**

```bash
uv run pytest v2/tests/unit/artifacts/test_artifact_root.py v2/tests/integration/test_artifact_integrity.py -k "path or symlink or link or inode or apfs" -v
```

Expected: FAIL because descriptor-owned traversal does not exist.

- [ ] **Step 3: Implement lexical validation and openat-style ownership**

Implement `parse_logical_path` as a pure function first. Then open roots with `os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)`, walk intermediate components with `os.open(component, flags, dir_fd=current_fd)`, and validate every opened descriptor with `os.fstat`. Duplicate each descriptor whose lifetime escapes a local block, close all intermediate descriptors in `finally`, and create Python binary streams only from an already validated final descriptor. Never call `Path.resolve()`, `Path.open()`, or a check-then-open symlink helper for a manifest payload.

- [ ] **Step 4: Verify descriptor closure and alias rejection**

```bash
uv run pytest v2/tests/unit/artifacts/test_artifact_root.py v2/tests/integration/test_artifact_integrity.py -k "path or descriptor or symlink or link or inode or apfs" -v
```

Expected: every opened descriptor is closed on success and failure, all alias/symlink cases fail, and read-only non-APFS access reports reduced guarantees without granting write authority.

- [ ] **Step 5: Commit descriptor-safe artifact reads**

```bash
git add v2/src/sml/artifacts/manifest.py v2/tests/unit/artifacts/test_artifact_root.py v2/tests/integration/test_artifact_integrity.py
git commit -m "feat(v2): traverse artifacts by directory descriptor"
```

### Task 2.3: Add Publication Locks and Immutable Bundle Commit

**Files:**
- Create: `v2/src/sml/artifacts/checkpoint.py`
- Create: `v2/tests/unit/artifacts/test_publication.py`
- Modify: `v2/tests/integration/test_artifact_integrity.py`

**Interfaces:**
- `FilesystemOps` is the explicit production I/O dependency used by artifact writers. Its methods are exactly `open`, `mkdir`, `write_all`, `fsync_file`, `fsync_directory`, `rename`, `replace`, `unlink`, `rmdir`, `stat`, `listdir`, and `flock`; the default implementation delegates to descriptor-relative `os`/`fcntl` operations.
- Test-owned `RecordingFilesystemOps` wraps that interface and raises after a named completed operation. Production configuration has no fault flag or injector.
- `publication_lock(target)`, `run_writer_lock(run)`, and `run_access_lock(run, *, exclusive)` are nonblocking context managers with owner diagnostics and kernel-released `flock` ownership.
- `publish_immutable_bundle(target, build: Callable[[Path], M], *, fs: FilesystemOps = OS_FILESYSTEM) -> Published[M]` is idempotent only after full verification of an identical target.
- Immutable publication stages are exactly `payloads-written`, `manifest-written`, `temporary-directory-fsynced`, `directory-renamed`, and `parent-directory-fsynced`; tests inject after each completed stage through `RecordingFilesystemOps`.

- [ ] **Step 1: Write lock, collision, interruption, and cleanup tests**

```python
def test_concurrent_identical_publication_is_idempotent(bundle_builder, target):
    first, second = run_concurrently(lambda: publish_immutable_bundle(target, bundle_builder))
    assert first.manifest.identity == second.manifest.identity
    assert verify_artifact(target, full=True).verification is VerificationLevel.FULL


@pytest.mark.parametrize("stage", IMMUTABLE_PUBLICATION_STAGES)
def test_interrupted_bundle_never_exposes_partial_target(stage, target, bundle_builder):
    fs = RecordingFilesystemOps.raise_after(stage)
    with pytest.raises(InjectedFailure):
        publish_immutable_bundle(target, bundle_builder, fs=fs)
    if target.exists():
        assert verify_artifact(target, full=True).verification is VerificationLevel.FULL
```

Name the other tests `test_different_existing_target_is_collision`, `test_identical_existing_target_requires_full_verification`, `test_conflicting_writer_reports_owner`, `test_lock_is_released_on_process_exit`, `test_cleanup_accepts_only_exact_target_digest_marker`, and `test_cleanup_revalidates_candidate_before_descriptor_relative_delete`.

- [ ] **Step 2: Run publication tests and verify RED**

```bash
uv run pytest v2/tests/unit/artifacts/test_publication.py v2/tests/integration/test_artifact_integrity.py -k "publication or lock or collision or cleanup" -v
```

Expected: FAIL because locks, publication stages, and cleanup do not exist.

- [ ] **Step 3: Implement lock ownership and manifest-last publication**

Acquire the target-name-derived publication sidecar before inspecting the target. Create a uniquely named direct sibling whose name contains the exact target-name SHA-256 digest and `.sml-tmp-` marker, pass that private path to the builder, validate/hash payloads, write the manifest last, fsync every file and directory, recheck target absence, rename once, and fsync the parent. Cleanup may descend only through validated directory descriptors and must revalidate marker, digest, non-symlink directory type, and parent membership immediately before deletion.

- [ ] **Step 4: Verify all publication stages and process conflicts**

```bash
uv run pytest v2/tests/unit/artifacts/test_publication.py v2/tests/integration/test_artifact_integrity.py -k "publication or lock or collision or cleanup" -v
```

Expected: interruption before rename leaves no target, interruption after rename leaves a fully valid target, identical reuse rehashes before success, and colliding/concurrent writes never overwrite.

- [ ] **Step 5: Commit atomic immutable publication**

```bash
git add v2/src/sml/artifacts/checkpoint.py v2/tests/unit/artifacts/test_publication.py v2/tests/integration/test_artifact_integrity.py
git commit -m "feat(v2): publish immutable artifact bundles"
```

### Task 2.4: Publish, Resolve, Recover, and Retain Checkpoints

**Files:**
- Modify: `v2/src/sml/artifacts/checkpoint.py`
- Create: `v2/tests/unit/artifacts/test_checkpoint.py`
- Modify: `v2/tests/integration/test_artifact_integrity.py`

**Interfaces:**
- `publish_checkpoint`, `resolve_latest_step`, `resolve_exact_step`,
  `recover_latest_index`, and parameterless `prune_to_latest` implement the
  design's step-commit, recovery, and mandatory latest-only algorithms using
  `FilesystemOps`. There is no public configurable-retention operation.
- Checkpoint fault stages are exactly `arrays-written`, `scalar-state-written`, `checkpoint-manifest-written`, `step-directory-fsynced`, `step-directory-renamed`, `step-parent-fsynced`, `latest-temporary-fsynced`, `latest-replaced`, and `latest-parent-fsynced`.
- Retention obtains the exclusive access lock, resolves latest, proves the retained step with current full verification, then deletes only eligible non-latest steps by descriptor.
- `open_checkpoint_reader(...) -> CheckpointReader` owns the shared access lock
  and verified run/checkpoint descriptors until the consumer has eagerly loaded
  and evaluated the exact scalar and array bytes it will use. Its
  `VerifiedCheckpointContents` result contains an immutable scalar mapping and
  immutable outer/inner array-group mappings.
- Resolution returns:

```python
@dataclass(frozen=True, slots=True)
class ResolvedStep:
    run: RunManifest
    checkpoint: CheckpointManifest
    step_directory: Path
    run_step_identity: str
    verification: VerificationLevel
    latest_recovered: bool
    latest_repair_persisted: bool
```

`ResolvedStep.step` is a read-only property that returns `checkpoint.step`; callers never parse the directory name to recover the logical step.

- [ ] **Step 1: Write exact-step, recovery, publication, and retention tests**

```python
def test_exact_step_ignores_latest_and_malformed_newer_step(valid_run):
    (valid_run / "latest.json").write_text("not-json", encoding="utf-8")
    (valid_run / "checkpoints/step-000000009").mkdir()
    resolved = resolve_exact_step(valid_run, step=1, verification=VerificationLevel.FULL)
    assert resolved.step == 1
    assert resolved.latest_recovered is False


def test_step_rename_before_latest_replace_recovers_new_step(valid_run):
    fs = RecordingFilesystemOps.raise_after("step-directory-renamed")
    with pytest.raises(InjectedFailure):
        publish_checkpoint(valid_run, checkpoint_payload(step=2), fs=fs)
    resolved = resolve_latest_step(valid_run, writable=True, verification=VerificationLevel.FULL)
    assert resolved.step == 2
    assert resolved.latest_recovered is True
    assert resolved.latest_repair_persisted is True
```

Parameterize `test_checkpoint_interruption_recovery` over every checkpoint fault stage. Add named tests `test_read_only_recovery_never_persists_latest`, `test_malformed_required_scan_candidate_fails`, `test_older_idempotent_publication_cannot_move_latest_backward`, `test_retention_waits_for_active_reader`, `test_retention_never_deletes_latest`, and `test_retention_requires_current_full_proof_before_first_delete`.

- [ ] **Step 2: Run checkpoint tests and verify RED**

```bash
uv run pytest v2/tests/unit/artifacts/test_checkpoint.py v2/tests/integration/test_artifact_integrity.py -k "checkpoint or latest or exact or recovery or retention" -v
```

Expected: FAIL because checkpoint state machines are absent.

- [ ] **Step 3: Implement commit-point recovery and retention**

Write arrays and scalar state, write `checkpoint.json` last, fsync, rename the
step directory as the commit point, then publish `latest.json` through fsync
plus atomic replace. Latest resolution validates the pointed step and required
newer scan range; exact resolution opens only `run.json` plus the requested
step. Writable recovery persists a repaired index, read-only recovery records
it in memory, and neither path silently skips malformed candidates it is
required to examine. The owned checkpoint reader consumes descriptor-bound
verified bytes without a pathname reopen. Latest-only pruning begins only after
successful latest recovery and a current retained-step proof.

- [ ] **Step 4: Verify checkpoint safety independently and with all artifact tests**

```bash
uv run pytest v2/tests/unit/artifacts v2/tests/integration/test_artifact_integrity.py -v
```

Expected: all interruption, recovery, exact-step, reader/retention, corruption, and full/manifest-trusted reporting tests pass without a production test switch.

- [ ] **Step 5: Commit checkpoint recovery and retention**

```bash
git add v2/src/sml/artifacts/checkpoint.py v2/tests/unit/artifacts/test_checkpoint.py v2/tests/integration/test_artifact_integrity.py
git commit -m "feat(v2): recover and retain atomic checkpoints"
```

### Task 2.5: Train and Load Immutable Tokenizer Bundles

**Files:**
- Create: `v2/src/sml/data/corpus.py`
- Create: `v2/src/sml/data/tokenizer.py`
- Create: `v2/tests/unit/data/test_corpus.py`
- Create: `v2/tests/unit/data/test_tokenizer.py`
- Create: `v2/tests/integration/test_tokenizer_workflow.py`

**Interfaces:**
- `CorpusConfig`, `discover_corpus_files`, and `iter_filtered_texts` preserve current compression/filter/normalization behavior with isolated seeded file ordering.
- `TokenizerTrainingConfig` owns all SentencePiece BPE settings.
- `train_tokenizer_bundle(config, output: Path) -> TokenizerBundle` atomically produces `manifest.json`, `tokenizer.model`, and `tokenizer.vocab`.
- `load_tokenizer_bundle(path, verification) -> LoadedTokenizer` validates manifest, bytes, vocab, and special IDs before returning the processor.

- [ ] **Step 1: Write corpus/tokenizer bundle tests**

```python
def test_tokenizer_bundle_is_self_describing_and_idempotent(tiny_corpus, tmp_path):
    output = tmp_path / "tokenizer"
    first = train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)
    second = train_tokenizer_bundle(tiny_tokenizer_config(tiny_corpus), output)
    assert first.manifest.identity == second.manifest.identity
    assert {path.name for path in output.iterdir()} == {"manifest.json", "tokenizer.model", "tokenizer.vocab"}
    loaded = load_tokenizer_bundle(output, VerificationLevel.FULL)
    assert loaded.processor.get_piece_size() == first.manifest.vocab_size
    assert loaded.manifest.algorithm == "bpe"


def test_downstream_loader_rejects_loose_model(tmp_path):
    loose = tmp_path / "tokenizer.model"
    loose.write_bytes(b"not-a-bundle")
    with pytest.raises(SMLArtifactError, match="tokenizer bundle directory"):
        load_tokenizer_bundle(loose, VerificationLevel.FULL)
```

- [ ] **Step 2: Run data tests and verify RED**

```bash
uv run pytest v2/tests/unit/data/test_corpus.py v2/tests/unit/data/test_tokenizer.py v2/tests/integration/test_tokenizer_workflow.py -v
```

Expected: FAIL because package data workflows do not exist.

- [ ] **Step 3: Port corpus behavior and wrap SentencePiece output in atomic publication**

Keep BPE, special token IDs 0/1/2/3, byte fallback, normalization, compressed JSONL discovery, and filtering semantics. Feed SentencePiece a lazy nonempty iterator, write into the publisher's temporary sibling, hash output bytes during production, construct the strict `TokenizerManifest`, then publish. Lazy-import `sentencepiece` inside tokenizer workflow functions.

- [ ] **Step 4: Verify unit/integration and package import tests**

```bash
uv run pytest v2/tests/unit/data/test_corpus.py v2/tests/unit/data/test_tokenizer.py v2/tests/integration/test_tokenizer_workflow.py v2/tests/unit/test_package.py -v
```

Expected: tests pass, an identical target is accepted only after full verification, a changed config collides, and importing `sml` does not import SentencePiece.

- [ ] **Step 5: Commit tokenizer bundles**

```bash
git add v2/src/sml/data/corpus.py v2/src/sml/data/tokenizer.py v2/tests/unit/data v2/tests/integration/test_tokenizer_workflow.py
git commit -m "feat(v2): publish immutable tokenizer bundles"
```

### Task 2.6: Prepare Deterministic Int32 NPY Shards

**Files:**
- Create: `v2/src/sml/data/pretraining.py`
- Create: `v2/tests/unit/data/test_pretraining.py`
- Create: `v2/tests/integration/test_pretraining_data_workflow.py`

**Interfaces:**
- `PretrainingPreparationConfig` includes input corpus, tokenizer bundle, sequence length, shuffle-window rows, shuffle algorithm version, output-shard rows, seed, and source limits.
- `prepare_pretraining_bundle(config, output) -> PreparedDataBundle` writes copied tokenizer and `shards/train-NNNNNN.npy` arrays with `int32` shape `(rows, sequence_length + 1)`.
- Row packing uses stride `sequence_length`; `windowed-row-shuffle-v1` permutes complete rows within fixed windows before shard division.
- Bundle identity changes with shard boundaries; `row_content_identity` does not.

- [ ] **Step 1: Write packing, shuffle, identity, and empty-output tests**

```python
def test_packing_overlaps_one_boundary_token():
    rows = list(pack_token_ranges([[1, 2, 3, 4, 5, 6, 7]], sequence_length=3))
    assert [row.tolist() for row in rows] == [[1, 2, 3, 4], [4, 5, 6, 7]]


def test_resharding_preserves_semantic_rows_but_changes_bundle_identity(tiny_tokenizer_bundle, tiny_corpus, tmp_path):
    first = prepare_pretraining_bundle(preparation_config(tiny_corpus, tiny_tokenizer_bundle, shard_rows=2), tmp_path / "two")
    second = prepare_pretraining_bundle(preparation_config(tiny_corpus, tiny_tokenizer_bundle, shard_rows=5), tmp_path / "five")
    assert first.manifest.row_content_identity == second.manifest.row_content_identity
    assert first.manifest.identity != second.manifest.identity


def test_preparation_refuses_zero_complete_rows(tiny_tokenizer_bundle, empty_corpus, tmp_path):
    with pytest.raises(SMLDataError, match="no complete pretraining rows"):
        prepare_pretraining_bundle(preparation_config(empty_corpus, tiny_tokenizer_bundle), tmp_path / "data")
```

- [ ] **Step 2: Run preparation tests and verify RED**

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py -k "pack or prepare or shard or identity" -v
```

Expected: FAIL because the prepared-data writer is absent.

- [ ] **Step 3: Implement buffered packing, window shuffle, and shard publication**

Use separate preallocated NumPy row-window and shard-output buffers with write cursors. Validate token ranges by vectorized min/max before each buffer commit. Convert identity rows to little-endian `int32` C order and hash domain tag, unsigned 64-bit little-endian row/column counts, then bytes. Write contiguous uncompressed `.npy` files and copy the tokenizer bundle through safe artifact APIs.

- [ ] **Step 4: Verify representation and identity invariants**

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py -v
```

Expected: all tests pass; changing seed/window/version changes row identity, changing only shard rows preserves row identity, every published shard is C-contiguous `int32`, and no empty bundle is visible.

- [ ] **Step 5: Commit prepared-data production**

```bash
git add v2/src/sml/data/pretraining.py v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py
git commit -m "feat(v2): prepare mmap pretraining bundles"
```

### Task 2.7: Add Mmap Epoch Streams, Canonical Cursors, and Bounded Prefetch

**Files:**
- Modify: `v2/src/sml/data/pretraining.py`
- Modify: `v2/tests/unit/data/test_pretraining.py`
- Modify: `v2/tests/integration/test_pretraining_data_workflow.py`

**Interfaces:**
- `PretrainingCursor(epoch: int, shard_order_position: int, row_offset: int)` names the next row eligible for a batch.
- `BatchEnvelope(rows: np.ndarray, cursor_after: PretrainingCursor)` owns a read-only row view and implements idempotent `release()` plus context-manager exit. A contiguous single-shard envelope may hold a read-only mmap slice; every queued cross-shard envelope owns a distinct contiguous staging array that is not returned to the pool until the consumer has created the MLX row array and releases that envelope.
- `PretrainingBatchStream(bundle, *, batch_size, seed, prefetch_depth, cursor)` is an iterator plus context manager with idempotent `close()`. It memory-maps shards, treats ordered shards as one epoch stream, and drops at most one incomplete epoch tail; `close()` stops/joins the producer, releases queued envelopes, and closes mappings/descriptors.
- Only the main consumer calls `mx.array`; the producer touches/copies NumPy arrays only.

- [ ] **Step 1: Write boundary, tail, resume, and full-queue tests**

```python
def test_epoch_stream_crosses_shards_and_drops_one_tail(prepared_bundle):
    stream = PretrainingBatchStream(prepared_bundle, batch_size=3, seed=5, prefetch_depth=2, cursor=PretrainingCursor.initial())
    row_count = 0
    shapes = []
    for envelope in stream.iter_epoch(0):
        try:
            row_count += envelope.rows.shape[0]
            shapes.append(envelope.rows.shape)
        finally:
            envelope.release()
    assert row_count == 6
    assert all(shape == (3, prepared_bundle.manifest.row_width) for shape in shapes)


def test_resume_starts_at_normalized_cursor_without_replay(prepared_bundle):
    cursor = PretrainingCursor(epoch=0, shard_order_position=1, row_offset=2)
    resumed = next(iter(PretrainingBatchStream(prepared_bundle, batch_size=2, seed=5, prefetch_depth=4, cursor=cursor)))
    try:
        assert resumed.rows.tolist() == expected_rows_from_cursor(prepared_bundle, cursor, count=2)
    finally:
        resumed.release()


def test_prefetch_producer_position_is_not_committed(prepared_bundle):
    stream = PretrainingBatchStream(prepared_bundle, batch_size=1, seed=5, prefetch_depth=8, cursor=PretrainingCursor.initial())
    first = next(iter(stream))
    try:
        assert stream.committed_cursor == PretrainingCursor.initial()
        stream.commit(first.cursor_after)
        assert stream.committed_cursor == first.cursor_after
    finally:
        first.release()


def test_full_queue_cross_shard_envelopes_do_not_share_mutable_staging(prepared_bundle):
    stream = stream_with_every_batch_crossing_a_shard(prepared_bundle, prefetch_depth=4)
    envelopes = take(stream, 4)
    try:
        snapshots = [envelope.rows.copy() for envelope in envelopes]
        assert all(not envelope.rows.flags.writeable for envelope in envelopes)
        assert len({envelope.rows.__array_interface__["data"][0] for envelope in envelopes}) == 4
        assert [envelope.rows.tolist() for envelope in envelopes] == [row.tolist() for row in snapshots]
    finally:
        for envelope in envelopes:
            envelope.release()


def test_mlx_transfer_does_not_alias_released_staging(prepared_bundle):
    stream = stream_with_every_batch_crossing_a_shard(prepared_bundle, prefetch_depth=1)
    iterator = iter(stream)
    first = next(iterator)
    snapshot = first.rows.copy()
    device_rows = mx.array(first.rows)
    first.release()
    reused = next(iterator)
    try:
        mx.eval(device_rows)
        assert device_rows.tolist() == snapshot.tolist()
    finally:
        reused.release()
        stream.close()
```

- [ ] **Step 2: Run loader tests and verify RED**

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py -k "stream or cursor or prefetch or tail" -v
```

Expected: FAIL because loader/cursor APIs do not exist.

- [ ] **Step 3: Implement deterministic shard order and bounded producer**

Derive a local `np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))`, where orchestration passes `PretrainingConfig.loader.epoch_seed` as `seed`; `PretrainingConfig.seed` is reserved for model/dropout keys. Permute shard indices only, normalize cursors immediately at shard ends, and attach the normalized post-batch cursor to each envelope. Map validated NPY payload descriptors directly and keep their descriptors/mappings alive for the stream lifetime. For a cross-shard batch, lease one contiguous staging array from a pool bounded by `prefetch_depth`; do not return it to the pool until the corresponding envelope is released by the consumer. Mark every exposed row array read-only. Do not advance `committed_cursor` on dequeue; expose an explicit `commit(cursor_after)` called only after evaluated optimizer state. Encountering an incomplete tail yields no envelope and retains the committed cursor.

The consumer constructs `device_rows = mx.array(envelope.rows)` and then calls `envelope.release()` in `finally`; the released NumPy storage is never read again by that MLX array. Stream shutdown releases queued envelopes and closes mappings/descriptors. A leaked or double-released envelope cannot silently reassign a live buffer: double release is harmless, and pool return uses an envelope-owned lease generation checked under the pool lock.

- [ ] **Step 4: Verify cursor behavior**

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py -v
```

Expected: cursor, staging ownership, and prepared-data correctness pass before the implementation is committed.

- [ ] **Step 5: Commit loader and cursor**

```bash
git add v2/src/sml/data/pretraining.py v2/tests/unit/data/test_pretraining.py v2/tests/integration/test_pretraining_data_workflow.py
git commit -m "perf(v2): stream mmap pretraining batches"
```

### Phase 2 Functional Gate

- [ ] Run complete Phase 2 correctness and workflow validation:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
```

Expected: Ruff and all v2 tests pass, including the tokenizer, artifact,
prepared-data, stream, cursor, resume, and benchmark-owner integration paths.
The tracked worktree is clean. No Phase 2 performance artifact is required.

## Phase 3: Pretraining Runtime

### Task 3.1: Add Training Policies and Project-Owned Mixed-Precision Adam

**Files:**
- Create: `v2/src/sml/training/common.py`
- Create: `v2/tests/unit/training/test_common.py`
- Modify: `v2/tests/equivalence/test_model_math.py`

**Interfaces:**
- Training configs compose rather than inherit and have these exact fields/defaults:

```python
@dataclass(frozen=True, slots=True)
class WeightDecayPolicy:
    embed_tokens: float = 0.0
    lm_head: float = 0.0
    rms_norm: float = 0.0
    q_proj: float = 0.1
    k_proj: float = 0.1
    v_proj: float = 0.1
    o_proj: float = 0.1
    gate_proj: float = 0.1
    up_proj: float = 0.1
    down_proj: float = 0.1
    lora_a: float = 0.0
    lora_b: float = 0.0
    other: float = 0.1


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    bias_correction: bool = False
    schedule_steps: int | None = 268_000
    warmup_steps: int | None = None
    minimum_learning_rate_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    weight_decay: WeightDecayPolicy = field(default_factory=WeightDecayPolicy)


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 8
    prefetch_depth: int = 2
    epoch_seed: int = 42


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    interval: int = 1_000


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    master_parameter_dtype: Literal["float32"] = "float32"
    working_parameter_dtype: Literal["bfloat16"] = "bfloat16"
    gradient_accumulator_dtype: Literal["float32"] = "float32"
    optimizer_state_dtype: Literal["float32"] = "float32"
    update_dtype: Literal["float32"] = "float32"
    master_weights: Literal[True] = True
    dynamic_loss_scaling: Literal[False] = False


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    data: Path
    output_run: Path
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    checkpoint: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    maximum_steps: int | None = None
    maximum_epochs: int | None = 1
    log_interval: int = 10
    seed: int = 42
    compile: bool = True
```

- `PretrainingConfig.__post_init__` rejects
  `model.rope_scaling_factor != 1.0`; the value is saved unchanged in
  `PretrainingRunManifest` and every `PretrainingCheckpointManifest`. It also
  rejects the absence of both termination limits, invalid finite numeric
  values, nonpositive sizes/intervals, and any step/epoch value that cannot be
  represented by the int32 device counters. When both limits are present,
  training stops at the first one reached. The learning-rate schedule clamps
  after `schedule_steps`, so a larger finite `maximum_steps` is valid.
- If `OptimizerConfig.warmup_steps is None`, `resolved_warmup_steps(config)` is exactly `int(0.01 * (10_000 if schedule_steps is None else schedule_steps))`. If `schedule_steps is None` and `maximum_steps` is finite, fresh orchestration replaces it with `maximum_steps` before identity calculation; resume never re-derives saved configuration.
- `learning_rate_at(step: mx.array, config: OptimizerConfig) -> mx.array` implements warmup `(step + 1) / max(1, warmup_steps)`, then either holds the base learning rate when `schedule_steps is None` or applies cosine decay over `schedule_steps - warmup_steps` clamped to `minimum_learning_rate_ratio`; it performs no `int(step)` or host synchronization.
- `AdamState.step` is an int32 scalar counting already-completed optimizer updates. An update uses `learning_rate_at(state.step, config)` and returns `step + 1`; checkpoint step and Adam step must match at every canonical optimizer boundary. Validate `0 <= warmup_steps < schedule_steps` when a finite schedule is present.
- `initialize_base_parameter_state(working_parameters: dict) -> BaseParameterState` requires a complete BF16 working tree, creates its authoritative FP32 master tree by explicit cast, and proves that every working leaf is the exact BF16 cast of the corresponding master leaf.
- `initialize_adam_state(master_parameters: dict) -> AdamState` creates zero FP32 first/second moments matching the complete master tree and an int32 zero step.
- `build_weight_decay_tree(named_parameters, policy)`, `accumulate_fp32`, `normalize_and_clip`, and `adamw_mixed_precision_update(master_parameters, gradients, state, config, weight_decay_tree) -> tuple[dict, dict, AdamState]` use the exact saved policy. The update returns the updated FP32 master tree, its explicitly derived BF16 working tree, and FP32/int32 Adam state. Tied embeddings classify only as `embed_tokens`; `lm_head` applies only to an untied head; Part 2 classifies adapter leaves as `lora_a`/`lora_b` rather than `other`.
- `BaseParameterState(master_parameters: dict, working_parameters: dict)`,
  `AdamState(step: mx.array, first_moments: dict, second_moments: dict)`, and
  `TrainerState(accumulators, accumulation_count, next_key, loss_numerator)`
  are frozen host wrappers. The trainer's fourth field is an FP32 scalar
  additive loss numerator so compiled accumulation remains accurate without
  per-microstep synchronization. Their `to_tree()` methods return built-in
  dict/tuple array trees and `from_tree()` validates exact keys/shapes plus
  FP32-master, BF16-working, FP32-moment/accumulator/loss, int32-counter, and
  PRNG-key dtypes. Only those trees cross `mx.compile`.

- [ ] **Step 1: Write formula, dtype, overflow, and two-step state tests**

```python
def test_pretraining_config_pins_standard_rope_and_composed_defaults(tmp_path):
    config = PretrainingConfig(
        data=tmp_path / "data",
        output_run=tmp_path / "run",
        model=replace(tiny_model_config(), rope_scaling_factor=1.0),
    )
    assert config.model.rope_scaling_factor == 1.0
    assert config.loader.microbatch_size == 1
    assert config.loader.gradient_accumulation_steps == 8
    assert config.optimizer.bias_correction is False
    assert config.optimizer.warmup_steps is None
    assert resolved_warmup_steps(config.optimizer) == 2_680
    with pytest.raises(SMLConfigurationError, match="pretraining.*rope_scaling_factor.*1.0"):
        replace(config, model=replace(config.model, rope_scaling_factor=2.0))


def test_learning_rate_schedule_matches_scalar_oracle_without_host_step_conversion():
    config = OptimizerConfig(schedule_steps=100, warmup_steps=10, minimum_learning_rate_ratio=0.1)
    steps = mx.array([0, 9, 10, 55, 100], dtype=mx.int32)
    actual = mx.vmap(lambda step: learning_rate_at(step, config))(steps)
    expected = numpy_schedule_oracle(steps.tolist(), config)
    mx.eval(actual)
    assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    assert source_has_none_of(common_module, ["int(step)", ".item(", "mx.eval("])


def test_adam_keeps_fp32_masters_bf16_working_parameters_and_fp32_moments():
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0, -2.0], dtype=mx.bfloat16)},
    )
    gradients = {"weight": mx.array([0.25, -0.5], dtype=mx.bfloat16)}
    state = initialize_adam_state(parameter_state.master_parameters)
    masters, working, state = adamw_mixed_precision_update(
        parameter_state.master_parameters,
        gradients,
        state,
        optimizer_config(weight_decay=0.1),
        {"weight": True},
    )
    mx.eval(masters, working, state.to_tree())
    assert masters["weight"].dtype == mx.float32
    assert working["weight"].dtype == mx.bfloat16
    assert mx.array_equal(working["weight"], masters["weight"].astype(mx.bfloat16)).item()
    assert state.first_moments["weight"].dtype == mx.float32
    assert state.second_moments["weight"].dtype == mx.float32


def test_sub_bf16_ulp_update_survives_in_master_state():
    parameter_state = initialize_base_parameter_state(
        {"weight": mx.array([1.0], dtype=mx.bfloat16)},
    )
    masters, working, _state = adamw_mixed_precision_update(
        parameter_state.master_parameters,
        {"weight": mx.array([1.0], dtype=mx.bfloat16)},
        initialize_adam_state(parameter_state.master_parameters),
        optimizer_config(learning_rate=1e-4, beta1=0.0, beta2=0.0, epsilon=1e-8),
        {"weight": False},
    )
    mx.eval(masters, working)
    assert not mx.array_equal(masters["weight"], parameter_state.master_parameters["weight"]).item()
    assert mx.array_equal(working["weight"], parameter_state.working_parameters["weight"]).item()


def test_compiled_second_update_observes_first_state():
    eager = run_two_updates(compiled=False)
    compiled = run_two_updates(compiled=True)
    assert_tree_close(compiled.parameter_state.master_parameters, eager.parameter_state.master_parameters, atol=1e-6, rtol=1e-6)
    assert_tree_close(compiled.parameter_state.working_parameters, eager.parameter_state.working_parameters, atol=0.0, rtol=0.0)
    assert_tree_close(compiled.optimizer.first_moments, eager.optimizer.first_moments, atol=1e-6, rtol=1e-6)
    assert int(compiled.optimizer.step.item()) == 2
```

- [ ] **Step 2: Run training-common tests and verify RED**

```bash
uv run pytest v2/tests/unit/training/test_common.py v2/tests/equivalence/test_model_math.py -k "adam or gradient or master or compiled" -v
```

Expected: FAIL because mixed-precision training primitives are absent.

- [ ] **Step 3: Implement explicit FP32 optimizer arithmetic**

Validate every frozen configuration field before allocation. Initialize the authoritative FP32 master tree exactly once from the evaluated BF16 working tree; never reconstruct it from BF16 after step zero or resume. Immediately cast raw BF16 gradient leaves to FP32, accumulate additive numerators, divide once by the actual normalization count, compute FP32 global norm/clipping, and perform epsilon, schedule, optional saved bias correction, decoupled weight decay, and the parameter update explicitly against the FP32 masters. The default `bias_correction=False` matches the pinned MLX AdamW baseline. Preserve the updated masters in FP32 and derive every next working leaf with one explicit BF16 cast. Range-check int32 device counters before overflow. Build immutable weight-decay classification once; tied embedding uses the embedding policy.

- [ ] **Step 4: Verify formulas, dtypes, and eager/compiled state transitions**

```bash
uv run pytest v2/tests/unit/training/test_common.py v2/tests/equivalence/test_model_math.py -v
```

Expected: exact integer/control checks and formula-oracle tolerances pass; every master/moment leaf is FP32, every working leaf is its exact BF16 cast, sub-BF16-ULP updates survive in masters, and no authoritative master leaf exists outside the returned/checkpointed state.

- [ ] **Step 5: Commit shared training primitives**

```bash
git add v2/src/sml/training/common.py v2/tests/unit/training/test_common.py v2/tests/equivalence/test_model_math.py
git commit -m "feat(v2): add fp32-master bf16-working adam runtime"
```

### Task 3.2: Compile the Pretraining Microstep and Optimizer Step

**Files:**
- Create: `v2/src/sml/training/pretrain.py`
- Create: `v2/tests/unit/training/test_pretrain.py`
- Modify: `v2/tests/equivalence/test_model_math.py`

**Interfaces:**
- `build_pretraining_kernels(model, config, weight_decay_tree) -> PretrainingKernels` builds private eager cores, compiles those cores, and returns host wrapper callables plus eager references.
- `microstep_core(working_parameters: dict, trainer_tree: dict, input_ids: mx.array, labels: mx.array) -> tuple[dict, dict]` accepts/returns only built-in array trees. The first result is the unchanged BF16 working tree required as an explicit dependency; the second is a trainer tree with FP32 accumulators/loss numerator, int32 count, and next PRNG key. FP32 masters do not enter forward/backward because this core neither reads nor mutates them.
- `optimizer_step_core(master_parameters: dict, working_parameters: dict, adam_tree: dict, trainer_tree: dict) -> tuple[dict, dict, dict, dict, dict]` returns updated FP32 masters, their exact BF16 working casts, FP32/int32 Adam state, reset trainer state, and a built-in metrics dict. No `BaseParameterState`, `AdamState`, `TrainerState`, `MicrostepState`, `OptimizerStepState`, `KVCache`, or other custom object crosses the compiled core.
- Public `PretrainingKernels.microstep(parameters: BaseParameterState, trainer: TrainerState, rows) -> MicrostepState` and `.optimizer_step(parameters: BaseParameterState, optimizer: AdamState, trainer: TrainerState) -> OptimizerStepState` only unwrap before the compiled call and validate/wrap returned trees afterward; wrapping arrays performs no evaluation or Python scalar conversion.
- Input and label views derive from one `(batch, sequence_length + 1)` MLX transfer.

- [ ] **Step 1: Write stable-shape, no-microstep-sync, partial-window, and PRNG tests**

```python
def test_microstep_transfers_rows_once_and_keeps_state_on_device(tiny_runtime):
    rows = np.arange(10, dtype=np.int32).reshape(2, 5)
    state = tiny_runtime.microstep(tiny_runtime.initial_state, mx.array(rows))
    assert state.accumulation_count.dtype == mx.int32
    assert state.accumulators["embed_tokens"]["weight"].dtype == mx.float32
    assert source_has_none_of(pretrain_module, ["loss.item(", ".tolist(", "mx.eval(loss"])


def test_partial_epoch_window_divides_by_actual_microbatch_count(tiny_runtime):
    eager = tiny_runtime.run_eager(microbatches=3, configured_accumulation=4, flush_epoch=True)
    compiled = tiny_runtime.run_compiled(microbatches=3, configured_accumulation=4, flush_epoch=True)
    assert eager.completed_steps == compiled.completed_steps == 1
    assert_tree_close(compiled.parameters.master_parameters, eager.parameters.master_parameters, atol=1e-6, rtol=1e-6)
    assert_tree_close(compiled.parameters.working_parameters, eager.parameters.working_parameters, atol=0.0, rtol=0.0)
    assert int(compiled.trainer.accumulation_count.item()) == 0


def test_compiled_cores_use_only_builtin_array_trees(tiny_runtime):
    core = tiny_runtime.kernels.compiled_microstep_core
    working_parameters, trainer_tree = core(
        tiny_runtime.parameters.working_parameters,
        tiny_runtime.trainer.to_tree(),
        tiny_runtime.input_ids,
        tiny_runtime.labels,
    )
    assert isinstance(working_parameters, dict)
    assert isinstance(trainer_tree, dict)
    assert all_builtin_array_tree_leaves(working_parameters, trainer_tree)
    assert source_has_none_of(
        pretrain_module,
        ["mx.compile(BaseParameterState", "mx.compile(AdamState", "mx.compile(TrainerState", "mx.compile(KVCache", "model.update("],
    )
```

- [ ] **Step 2: Run kernel tests and verify RED**

```bash
uv run pytest v2/tests/unit/training/test_pretrain.py v2/tests/equivalence/test_model_math.py -k "microstep or optimizer_step or partial_epoch or prng" -v
```

Expected: FAIL because compiled pretraining kernels are absent.

- [ ] **Step 3: Implement explicit compiled state flow**

Pass BF16 working parameters, FP32 master parameters, optimizer moments/step, accumulation buffers/count, and PRNG key as the explicit dict/tuple array-tree inputs/outputs of the cores that use them. Capture only immutable model structure/configuration and immutable classifications. The loss transform calls the pure `model.forward_arrays(working_parameters, ...)`, so every working weight remains an explicit compiled input and no core calls `model.update(...)` or mutates a captured module shell. The optimizer core updates the explicit FP32 masters, derives and returns the complete BF16 working tree, and returns mutually consistent Adam/trainer/metric trees in the same dependency barrier. Construct host wrapper objects strictly outside decorated functions. Evaluate all updated master, working, Adam, trainer, and metric leaves together at the optimizer barrier. Disabled dropout must return the original key. Reset accumulators only in the successful optimizer-step result. Keep logging, cursor commit, checkpoint I/O, and iteration outside compiled functions.

- [ ] **Step 4: Verify eager/compiled consecutive steps and source synchronization rules**

```bash
uv run pytest v2/tests/unit/training/test_pretrain.py v2/tests/equivalence/test_model_math.py -v
```

Expected: all kernel tests pass; every working leaf is the exact BF16 cast of its returned FP32 master, and the second compiled call observes the first call's returned masters, working parameters, optimizer state, counters, and key.

- [ ] **Step 5: Commit compiled kernels**

```bash
git add v2/src/sml/training/pretrain.py v2/tests/unit/training/test_pretrain.py v2/tests/equivalence/test_model_math.py
git commit -m "perf(v2): compile explicit pretraining state"
```

### Task 3.3: Create, Checkpoint, Resume, and Retain Pretraining Runs

**Files:**
- Modify: `v2/src/sml/training/pretrain.py`
- Modify: `v2/src/sml/artifacts/checkpoint.py`
- Modify: `v2/tests/unit/training/test_pretrain.py`
- Create: `v2/tests/integration/test_pretraining_workflow.py`

**Interfaces:**
- `train(config: PretrainingConfig) -> TrainingResult` creates a new run only.
- `resume(run: Path, *, data: Path | None, overrides: ResumeOverrides) -> TrainingResult` resolves/rebuilds latest under the writer lock.
- Resume/result types are exact:

```python
@dataclass(frozen=True, slots=True)
class ResumeOverrides:
    maximum_steps: int | None = None
    maximum_epochs: int | None = None
    log_interval: int | None = None
    checkpoint_interval: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    run: Path
    step: int
    epoch: int
    rows: int
```

- `None` means “keep the saved invocation value”; resume cannot change
  model/optimizer/precision/loader semantics. Checkpoint retention is not an
  override: every successful checkpoint operation prunes to the latest
  checkpoint only.
- `PretrainingConfig` contains no runtime object, fault injector, callback, or test-only dependency. Tests replace the owner-level `build_pretraining_kernels` symbol with a test-owned wrapper when they need a controlled failure.
- Fresh training requires `config.model.rope_scaling_factor == 1.0`; `run.json`, step zero, every resumed step, and model-only inference from the base run all expose that same authoritative value. Resume rejects any checkpoint/run metadata that claims another value before model allocation.
- Every checkpoint has `model.safetensors`, `master.safetensors`, `optimizer.safetensors`, `trainer.safetensors`, `state.json`, and `checkpoint.json`; `model.safetensors` is the BF16 working tree, `master.safetensors` is the authoritative FP32 tree, and full verification proves exact leaf/key/shape correspondence plus `model_leaf == cast_bf16(master_leaf)`. Step zero is published atomically with `run.json`, tokenizer copy, and `latest.json`.

- [ ] **Step 1: Write tiny run, exact resume, crash replay, relocation, and retention tests**

```python
def fault_after_one_successful_microstep(real_builder):
    def build(*args, **kwargs):
        kernels = real_builder(*args, **kwargs)
        real_microstep = kernels.microstep
        calls = 0

        def microstep(*microstep_args, **microstep_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InjectedFailure("after one uncommitted microstep")
            return real_microstep(*microstep_args, **microstep_kwargs)

        return replace(kernels, microstep=microstep)

    return build


def test_fresh_run_atomically_publishes_step_zero(tiny_training_config):
    result = train(tiny_training_config)
    resolved = resolve_latest_step(result.run, writable=False, verification=VerificationLevel.FULL)
    assert resolved.step == 0
    assert read_scalar_state(resolved).cursor == PretrainingCursor.initial()
    assert_checkpoint_has_exact_arrays(resolved, kind="pretraining")
    assert_checkpoint_array_dtypes(
        resolved,
        model=mx.bfloat16,
        master=mx.float32,
        optimizer_moments=mx.float32,
        trainer_accumulators=mx.float32,
    )
    assert_model_is_exact_bf16_cast_of_master(resolved)


def test_interrupted_accumulation_replays_window(tiny_training_config, monkeypatch):
    uninterrupted = train(replace(tiny_training_config, output_run=tiny_training_config.output_run.with_name("full")))
    real_builder = pretrain.build_pretraining_kernels
    monkeypatch.setattr(
        pretrain,
        "build_pretraining_kernels",
        fault_after_one_successful_microstep(real_builder),
    )
    with pytest.raises(InjectedFailure):
        train(replace(tiny_training_config, output_run=tiny_training_config.output_run.with_name("crashed")))
    monkeypatch.setattr(pretrain, "build_pretraining_kernels", real_builder)
    resumed = resume(tiny_training_config.output_run.with_name("crashed"), data=tiny_training_config.data, overrides=ResumeOverrides(maximum_steps=uninterrupted.step))
    assert_run_states_equal(resumed, uninterrupted, fields=("master", "model", "optimizer", "cursor", "step", "accumulation", "prng"))


def test_resume_accepts_relocated_identical_bundle_and_rejects_resharded_bundle(tiny_completed_run, relocated_data, resharded_same_rows):
    assert resume(tiny_completed_run, data=relocated_data, overrides=ResumeOverrides(maximum_steps=2)).step == 2
    with pytest.raises(SMLArtifactError, match="prepared-data identity"):
        resume(tiny_completed_run, data=resharded_same_rows, overrides=ResumeOverrides(maximum_steps=2))
```

Also cover full queue cursor safety, shard/cross-shard/tail/epoch cursor
normalization, completed-limit early return before iterators/compilation,
existing run rejection, writer conflict, checkpoint dtype/key corruption,
missing/additional master or working leaves, wrong master/working dtypes, a BF16
working leaf that is not the exact cast of its FP32 master, latest recovery,
moved run model-only portability, latest-only pruning waiting for a reader,
checkpoint intervals counting completed optimizer steps, no progress-only
duplicate step at dropped tail, and rejection of a non-`1.0` pretraining model
config before allocation. Use a fully identity-consistent but semantically
invalid prepared bundle to prove complete NPY metadata, token ranges, canonical
cursor, and full-batch availability are checked before checkpoint restoration,
pruning, or a completed-limit return. Mutate one payload byte without changing
its manifest in fresh train, resume, writable recovery, and pruning cases; each
must fail before model/optimizer allocation or deletion.

- [ ] **Step 2: Run workflow tests and verify RED**

```bash
uv run pytest v2/tests/unit/training/test_pretrain.py v2/tests/integration/test_pretraining_workflow.py -v
```

Expected: FAIL because run orchestration and checkpoint serialization are incomplete.

- [ ] **Step 3: Implement pre-GPU validation and direct orchestration**

Hold the nonblocking writer lock for the workflow lifetime. Before
model/optimizer allocation or any pruning, synchronously FULL-verify prepared
data, parse every NPY header, validate token ranges and full-batch availability,
and validate every restored scalar/cursor/array through the descriptor-owned
checkpoint reader. Enforce saved semantic config, require the authoritative
base-pretraining `ModelConfig.rope_scaling_factor` to equal `1.0`, and prove
exact FP32-master/BF16-working key, shape, dtype, and cast relationships.
Construct the training model from that unchanged configuration; do not
substitute a larger inference-only value. For a fresh run, initialize and
evaluate the BF16 working tree once, derive the authoritative FP32 masters from
that exact tree, initialize FP32 moments, and publish both trees in step zero.
Persist model, precision, optimizer, loader, checkpoint, tokenizer, and
authoritative prepared-data identities in immutable `run.json`; keep the
original data location diagnostic. For each accumulation window, retain the
last consumed `cursor_after`, call the compiled update, evaluate
master/working/Adam/trainer state together, then commit the cursor. Checkpoint
only canonical empty accumulation state at optimizer boundaries and always
write both `master.safetensors` and its exact BF16 `model.safetensors`
derivation. Prune to latest only after latest recovery and a full retained-step
proof.

- [ ] **Step 4: Verify uninterrupted/resumed equivalence and portable run behavior**

```bash
uv run pytest v2/tests/unit/training/test_pretrain.py v2/tests/integration/test_pretraining_workflow.py -v
```

Expected: all tests pass with exact step/cursor/key equality, FP32-master equality, dtype-appropriate optimizer equality, and exact BF16 working derivation after uninterrupted and resumed updates.

- [ ] **Step 5: Commit pretraining workflows**

```bash
git add v2/src/sml/training/pretrain.py v2/src/sml/artifacts/checkpoint.py v2/tests/unit/training/test_pretrain.py v2/tests/integration/test_pretraining_workflow.py
git commit -m "feat(v2): train and resume portable pretraining runs"
```

### Task 3.4: Prove Complete Part 1 Integration

**Files:**
- Create: `v2/tests/integration/test_part1_workflow.py`

**Interfaces:**
- The integration test trains a tiny tokenizer, prepares int32 shards, trains
  two optimizer steps, resumes to three, FULL-reopens exact latest step three,
  proves step one is unavailable and only the latest checkpoint remains, and
  fully verifies the run including its FP32-master/BF16-working relationship.

- [ ] **Step 1: Write the complete Part 1 integration test**

```python
def test_part1_tokenizer_to_resumed_pretraining(tiny_raw_corpus, tmp_path):
    tokenizer = train_tokenizer_bundle(tiny_tokenizer_config(tiny_raw_corpus), tmp_path / "tokenizer")
    data = prepare_pretraining_bundle(tiny_preparation_config(tiny_raw_corpus, tokenizer), tmp_path / "data")
    first = train(tiny_pretraining_config(data, tmp_path / "run", maximum_steps=2))
    resumed = resume(first.run, data=None, overrides=ResumeOverrides(maximum_steps=3))
    exact = resolve_exact_step(first.run, step=3, verification=VerificationLevel.FULL)
    assert resumed.step == 3
    assert exact.step == 3
    assert exact.run.model.rope_scaling_factor == 1.0
    assert_model_is_exact_bf16_cast_of_master(exact)
    with pytest.raises(SMLArtifactError, match="does not exist"):
        resolve_exact_step(first.run, step=1, verification=VerificationLevel.FULL)
    assert sorted(path.name for path in (first.run / "checkpoints").iterdir()) == [
        "step-000000003",
    ]
    assert verify_artifact(first.run, full=True).verification is VerificationLevel.FULL
```

- [ ] **Step 2: Run integration and verify RED or missing coverage**

```bash
uv run pytest v2/tests/integration/test_part1_workflow.py -v
```

Expected: initial failure identifies any missing package-to-package wiring; do not weaken the test's full verification or resume assertions.

- [ ] **Step 3: Add, verify, and commit only the wiring needed by the complete flow**

Keep domain entrypoints typed and direct. Ensure tokenizer/prepared/run identities are threaded through manifests and result objects, and expose `verify_artifact(path, full: bool) -> VerificationResult` from artifacts without importing CLI code.

```bash
uv run pytest v2/tests/integration/test_part1_workflow.py -v
git add v2/src/sml v2/tests/integration/test_part1_workflow.py
git commit -m "test(v2): prove tokenizer to pretraining workflow"
```

Expected: the complete flow passes from the new committed tree; the exact
latest step proves that its BF16 model is derived from the checkpointed FP32
masters, and no historical checkpoint remains selectable.

### Task 3.5: Version and Run the Controlled Pretraining-Quality Gate

**Execution update (2026-08-23):** The harness and canonical 1,000-step
candidate/oracle evidence are committed, and the numerical gate passes. The
remaining work is to make standalone validation consume the manifest's recorded
source boundary instead of rebuilding an expected workload from current-tree
bytes. The approved policy does not require rerecording unless that correction
changes the evidence identity.

**Files:**
- Create: `v2/benchmarks/quality.py`
- Create: `v2/tests/unit/test_pretraining_quality.py`
- Create: `v2/benchmarks/fixtures/pretraining-quality-train-v1.npy`
- Create: `v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy`
- Create: `v2/benchmarks/manifests/pretraining-quality-v1.json`
- Create: `v2/benchmarks/results/pretraining-quality-v1.jsonl`
- Create: `v2/benchmarks/results/pretraining-quality-v1.json`

**Interfaces:**
- `PretrainingQualityWorkload` records the fixed source-disjoint training/validation row identities, initial BF16 working-parameter identity, model/optimizer/loader configuration, ordered batches, checkpoint steps `(0, 10, 100, 1_000)`, evaluation request identity, seeds, and harness content identity.
- `PretrainingQualityCheckpoint` records candidate/oracle train loss, held-out FP32 validation NLL, finite-state status, per-leaf update-to-BF16-ULP statistics, fraction of changed BF16 working values, RMSNorm-master movement, and whether a sub-BF16-ULP update survived in the master tree.
- `PretrainingQualityReport(candidate_validation_nll: float, oracle_validation_nll: float, candidate_finite: bool, oracle_finite: bool, rms_norm_master_moved: bool, sub_bf16_update_survived: bool, matching_work_identity: bool)` is the strict acceptance summary reconstructed from the raw checkpoint records.
- `decide_pretraining_quality(report: PretrainingQualityReport) -> Literal["pass", "fail"]` passes only when both runs remain finite, the candidate RMSNorm master moves, at least one individually sub-BF16-ULP update survives in FP32 master state, the candidate/FP32-oracle real-work identities match exactly, and candidate step-1,000 validation NLL is at most `1.01 * oracle_validation_nll`.
- `python -m v2.benchmarks.quality record` runs the candidate FP32-master/BF16-compute runtime and an FP32-master/FP32-compute oracle from the same initial BF16 tree for exactly 1,000 optimizer steps. `validate` treats the committed manifest workload and source commit as authoritative, reconstructs the recorded harness, production-dependency, and fixture bytes from that commit, and recomputes identities and the acceptance decision from raw evidence without first deriving an expected workload from current-tree bytes.

- [ ] **Step 1: Write deterministic quality-decision tests**

```python
def test_quality_gate_requires_fp32_master_evidence_and_oracle_bound():
    passing = PretrainingQualityReport(
        candidate_validation_nll=2.01,
        oracle_validation_nll=2.00,
        candidate_finite=True,
        oracle_finite=True,
        rms_norm_master_moved=True,
        sub_bf16_update_survived=True,
        matching_work_identity=True,
    )
    assert decide_pretraining_quality(passing) == "pass"
    assert decide_pretraining_quality(
        replace(passing, sub_bf16_update_survived=False),
    ) == "fail"
    assert decide_pretraining_quality(
        replace(passing, candidate_validation_nll=2.021),
    ) == "fail"
```

- [ ] **Step 2: Run quality-analysis tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_pretraining_quality.py -v
```

Expected: FAIL because the controlled quality harness and decision function do not exist.

- [ ] **Step 3: Implement deterministic candidate/oracle execution and strict validation**

Build the source-disjoint row fixtures from fixed checked-in token rows,
evaluate the shared initial BF16 parameter tree before hashing or casting, and
feed both runs the same ordered batches, corrected tied-embedding graph, AdamW
formula, schedule, clipping, decay tree, and PRNG sequence. The candidate keeps
authoritative FP32 masters, derives BF16 working parameters after every update,
and performs model compute in BF16; the oracle keeps authoritative FP32
parameters and performs model compute in FP32. Synchronize only at the pinned
reporting steps. Write one raw JSON object per run/checkpoint, then validate
exact workload/row/batch/request identities, finite values, expected record
cardinality, FP32-master evidence, and the 1-percent validation-NLL rule. The
harness content identity hashes the ordered bytes of `quality.py` and
`test_pretraining_quality.py`; separate manifest fields bind the recorded source
commit and exact production dependency closure. Standalone validation verifies
those recorded commit bytes. It must remain valid after unrelated source edits,
the Task 4.1 checkpoint-reader edit, and Phase 6 migration-bridge deletion.

- [ ] **Step 4: Verify and commit the quality harness before producing evidence**

```bash
uv run pytest v2/tests/unit/test_pretraining_quality.py -v
git add v2/benchmarks/quality.py v2/tests/unit/test_pretraining_quality.py v2/benchmarks/fixtures/pretraining-quality-train-v1.npy v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy
git commit -m "test(v2): version pretraining quality harness"
```

Expected: deterministic decision tests pass and the committed harness exists before its manifest or results are created.

- [ ] **Step 5: Record and validate the 1,000-step quality evidence from a clean checkout**

```bash
uv run python -m v2.benchmarks.quality record --steps 1000 --manifest v2/benchmarks/manifests/pretraining-quality-v1.json --raw-output v2/benchmarks/results/pretraining-quality-v1.jsonl --output v2/benchmarks/results/pretraining-quality-v1.json
uv run python -m v2.benchmarks.quality validate --manifest v2/benchmarks/manifests/pretraining-quality-v1.json --raw-input v2/benchmarks/results/pretraining-quality-v1.jsonl --report v2/benchmarks/results/pretraining-quality-v1.json
```

Expected: PASS with complete records for steps 0, 10, 100, and 1,000; no nonfinite state; nonzero RMSNorm-master movement; at least one surviving sub-BF16-ULP master update; and candidate validation NLL no more than 1 percent above the FP32-compute oracle.

The committed evidence at `0538b3a` already satisfies the numerical conditions.
After the recorded-source validator correction, rerun this 1,000-step command
only if the evidence identity changes; otherwise revalidate the existing
manifest/raw/report set and preserve it byte-for-byte.

- [ ] **Step 6: Commit controlled quality evidence**

```bash
git add v2/benchmarks/manifests/pretraining-quality-v1.json v2/benchmarks/results/pretraining-quality-v1.jsonl v2/benchmarks/results/pretraining-quality-v1.json
git commit -m "test(v2): record pretraining quality evidence"
```

### Task 3.6: Run the Phase 3 Correctness and Workflow Gate

**Interfaces:**
- Proves the complete Part 1 package, artifact, data, pretraining, checkpoint,
  exact-resume, and controlled-quality paths run correctly together.

- [ ] **Step 1: Run all Part 1 verification**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
```

Expected: correctness and the committed quality report pass; a tiny fresh run,
checkpoint, exact resume, retained-step recovery, and model-only artifact load
complete through the production APIs; mandatory full input verification occurs
before GPU allocation where specified; and the tracked worktree is clean.

- [ ] **Step 2: Optionally collect performance diagnostics**

The historical Phase 3 comparison may be run when speed or memory information
is useful. It is report-only: no baseline, ratio, dispersion, confidence bound,
or thermal outcome is required to complete Part 1 or begin Part 2.

## Part 1 Completion Gate

Part 1 is complete only when:

- package imports, model equivalence/correction tests, artifact integrity tests, and tokenizer/preparation/pretraining workflows all pass;
- a tiny pretraining run is portable for model-only operations, resumes exactly with a matching data bundle, checkpoints every authoritative FP32 master, proves every BF16 working leaf is its exact cast, and rejects corrupt/mismatched state before GPU allocation;
- the committed 1,000-step quality report proves finite state, RMSNorm-master movement, survival of sub-BF16-ULP master updates, and candidate validation NLL no more than 1 percent above the FP32-compute oracle;
- base pretraining and every base checkpoint authoritatively record `rope_scaling_factor=1.0`; training a factor above `1.0` belongs to a later distinct long-context fine-tuning workflow, not this pretraining plan;
- the Phase 3 functional gate and all controlled quality checks pass; performance reports are optional diagnostics;
- standalone controlled-quality validation verifies the evidence's recorded
  source commit and does not invalidate it because of unrelated later-tree
  edits or migration-bridge deletion;
- `VerifiedCheckpointContents` exposes immutable scalar and nested array-group
  mappings before the Part 2 model resolver consumes that reader boundary;
- `uv.lock` remains byte-identical;
- Part 2 starts from a committed, functionally verified Phase 3 tree, not from an uncommitted worktree.
