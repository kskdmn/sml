# V2 Prepared-Data 100-Unit Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce independently validated Phase 2 prepared-data evidence using 100 real 1,024-token batches per measured trial.

**Architecture:** Keep production code and the benchmark harness unchanged. Isolate the rejected 20-unit report, then invoke the existing comparison and validation interfaces with exactly one protocol change: `--measure 100`. Publish evidence only if the unchanged ratio, dispersion, canonical-identity, and environment gates all pass.

**Tech Stack:** Python 3.12.13, `uv`, MLX/Metal, NumPy, the existing `v2.benchmarks.runner` CLI, Git, macOS environment probes.

## Global Constraints

- Every measured unit remains one real prepared-data batch with microbatch size 1 and sequence length 1,024.
- Change only the comparison's measured units from 20 to 100; do not modify production code, benchmark code, the pinned baseline, or `uv.lock`.
- Preserve screen mode, 5 pairs, 5 warmup units, 10,000 bootstrap resamples, ratio floor `0.97`, ratio MAD ceiling `0.02`, report-only lower confidence bound, and predecessors `{"prepared-data": null}`.
- Use pinned baseline `v2/benchmarks/manifests/baseline-3687f8b.json` and canonical `TMPDIR=/private/tmp`.
- Run every pytest, environment-gate, comparison, and validation command outside the sandbox so MLX/Metal can access the Apple GPU.
- Require 30 continuous nominal seconds immediately before launch: AC connected, automatic power mode, low-power mode off, thermal nominal/raw `0`, normal memory pressure, and no competing GPU workload.
- The comparison is non-resumable. Never salvage trials or perform an operator retry; permit only the harness-owned statistical-noise retry and cooldown.
- Do not validate, stage, or commit a rejected comparison. Commit only two independently passing Phase 2 reports. Do not push.

---

### Task 1: Run and Accept the 100-Unit Phase 2 Protocol

**Files:**
- Preserve: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json`
- Create on success: `v2/benchmarks/results/phase-2-loader.json`
- Create on success: `v2/benchmarks/results/phase-2.json`
- Report: `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-1-report.md`

**Interfaces:**
- Consumes: reviewed candidate `HEAD`, `collect_environment() -> tuple[dict[str, object], dict[str, object], dict[str, str]]`, the pinned baseline manifest, and the existing `compare`/`validate-phase` CLI contracts.
- Produces on success: a fresh comparison report whose raw trials all record `measured_units=100`, plus an independently validated Phase 2 acceptance report and an evidence-only commit.

- [ ] **Step 1: Record the exact execution base and isolate rejected evidence**

Create the ignored SDD workspace for this plan. Verify the current public loader
report before moving it:

```bash
bash /Users/keisukedaimon/.codex/plugins/cache/openai-curated-remote/superpowers/6.2.0/skills/subagent-driven-development/scripts/sdd-workspace \
  docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md
jq -e '
  .identity == "sha256:1b19e60760cd84144395eb70cf001bd683a2afc552dfa68e344c3902bc09ea58"
  and .metrics["prepared-data"].result_identity == "sha256:bd73c94cf4e6b5445a3353eab07997b518bc08dbfc937883a8a7995b6ab4c3b7"
  and .metrics["prepared-data"].baseline_comparison.decision == "too-noisy"
  and .protocol.measured_units == 20
' v2/benchmarks/results/phase-2-loader.json
mv \
  v2/benchmarks/results/phase-2-loader.json \
  .superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json
test -f .superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json
test ! -e v2/benchmarks/results/phase-2-loader.json
test ! -e v2/benchmarks/results/phase-2.json
```

Move that exact rejected report to the preservation path above. Confirm the
public loader and Phase 2 acceptance paths are both absent. The preserved file
is diagnostic evidence only: never modify, validate, stage, or delete it.

Record `git rev-parse HEAD` as the execution base. Confirm `git status --short`
is empty and `git diff --exit-code -- uv.lock` exits zero. Stop if any tracked,
staged, or unrelated untracked change exists.

- [ ] **Step 2: Run the exact clean-candidate preflight outside the sandbox**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
git diff --exit-code -- uv.lock
```

Expected: both Ruff gates pass; the full v2 suite passes without warnings;
status is empty; `uv.lock` is unchanged. Reconfirm the execution-base SHA and
both public result paths' absence after preflight.

- [ ] **Step 3: Require a fresh 30-second nominal launch window**

Run this transient gate outside the sandbox; it creates no repository file:

```bash
env TMPDIR=/private/tmp uv run python - <<'PY'
import json
import time

from v2.benchmarks.runner import collect_environment

started = time.monotonic()
deadline = started + 300.0
stable_since = None
sample_count = 0

while True:
    _, status, _ = collect_environment()
    observed = time.monotonic()
    nominal = (
        status.get("power_connected") is True
        and status.get("power_mode") == "automatic"
        and status.get("low_power_mode") is False
        and status.get("thermal_state") == "nominal"
        and status.get("thermal_state_raw_value") == 0
        and status.get("memory_pressure") == "normal"
        and status.get("competing_gpu_workload") is False
    )
    if nominal:
        if stable_since is None:
            stable_since = observed
        stable_seconds = observed - stable_since
    else:
        stable_since = None
        stable_seconds = 0.0
    print(
        json.dumps(
            {
                "elapsed_seconds": observed - started,
                "sample": sample_count,
                "stable_seconds": stable_seconds,
                **status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    sample_count += 1
    if stable_seconds >= 30.0:
        print(json.dumps({"gate": "pass", "samples": sample_count}))
        raise SystemExit(0)
    if observed >= deadline:
        print(json.dumps({"gate": "timeout", "samples": sample_count}))
        raise SystemExit(2)
    time.sleep(1.0)
PY
```

Expected: exit zero after at least 30 continuous nominal seconds. If it exits
2 or any probe fails, stop BLOCKED before comparison; do not retry in this task
turn.

- [ ] **Step 4: Run the exact fresh 100-unit comparison outside the sandbox**

Launch immediately after the passing gate:

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner compare \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --candidate HEAD \
  --metrics prepared-data \
  --mode screen \
  --pairs 5 \
  --warmup 5 \
  --measure 100 \
  --bootstrap-resamples 10000 \
  --minimum-ratio 0.97 \
  --maximum-dispersion 0.02 \
  --lower-bound-report-only \
  --predecessors '{"prepared-data":null}' \
  --output v2/benchmarks/results/phase-2-loader.json
```

Leave the command uninterrupted. Do not inspect or reuse partial files. The
harness may perform its one statistical-noise retry and cooldown. After exit,
require all of the following before continuing:

```bash
jq -e '
  .protocol.measured_units == 100
  and .metrics["prepared-data"].baseline_comparison.decision == "pass"
  and .metrics["prepared-data"].baseline_comparison.median_ratio >= 0.97
  and .metrics["prepared-data"].baseline_comparison.ratio_mad <= 0.02
  and ([.raw_trials[].measured_units] | length > 0 and all(. == 100))
  and ([.raw_trials[].native_configuration.sequence_length] | all(. == 1024))
  and ([.raw_trials[].native_configuration.microbatch_size] | all(. == 1))
  and ([.raw_trials[].environment_status |
    .thermal_state,
    .start.thermal_state,
    .end.thermal_state,
    .post_exit.thermal_state,
    .post_exit_recovery_final.thermal_state
  ] | all(. == "nominal"))
  and ([.raw_trials[].environment_status |
    .thermal_state_raw_value,
    .start.thermal_state_raw_value,
    .end.thermal_state_raw_value,
    .post_exit.thermal_state_raw_value,
    .post_exit_recovery_final.thermal_state_raw_value
  ] | all(. == 0))
' v2/benchmarks/results/phase-2-loader.json
```

If the command, decision, units, statistics, trial identity, or environment
evidence fails, stop BLOCKED. Preserve the fresh report as an untracked
diagnostic; do not validate, stage, commit, or operator-rerun.

- [ ] **Step 5: Independently validate Phase 2 outside the sandbox**

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json \
  --output v2/benchmarks/results/phase-2.json
```

Expected: exit zero. Inspect both reports and record their identities, candidate
commit, protocol units, comparison and Phase 2 decisions, median ratio, ratio
MAD, lower confidence bound, every trial's start/end/merged/post-exit thermal
state and raw value, memory/power/GPU fields, and retry/cooldown evidence.
Confirm the predecessor mapping is exactly `{"prepared-data": null}`.

If validation fails or the acceptance report does not prove a passing
prepared-data baseline decision, stop BLOCKED without staging either file.

- [ ] **Step 6: Commit only accepted evidence**

Confirm `git status --short` lists exactly the two untracked public result
files. Stage only them:

```bash
git add \
  v2/benchmarks/results/phase-2-loader.json \
  v2/benchmarks/results/phase-2.json
git diff --cached --check
git diff --cached --stat
git commit -m "bench(v2): accept artifacts and prepared-data phase"
```

Then verify:

```bash
git show --check --stat --oneline --summary HEAD
git status --short
git diff --exit-code HEAD^ -- uv.lock
```

Expected: exactly the two accepted reports are committed; status is empty;
`uv.lock` is unchanged; no production, harness, plan, or rejected-evidence file
entered the evidence commit. Do not push.

- [ ] **Step 7: Write the operator report and prepare independent review**

Write the complete command/exit log, execution base, gate samples, comparison
attempts, cooldown/retry status, raw trial environment/thermal evidence,
identities, statistics, validation decision, commit hash or blocking reason,
and final repository state to the report path above. On success, generate a
task review package from the execution base through the evidence commit. The
independent review must verify protocol immutability (`100`, `0.97`, `0.02`),
evidence isolation, canonical identities, all environment observations,
validation, and the exact two-file commit scope. On a blocked outcome, do not
manufacture a diff package; return the complete read-only evidence report to
the controller and stop.
