# V2 Performance Refactor Phase 2 Handoff

**Recorded:** 2026-08-22 (Asia/Shanghai)

**Purpose:** Allow a fresh session to take over the v2 performance-first
refactor without relying on conversation history.

## Current Snapshot

- Branch: `main`, tracking `origin/main`.
- Current local and remote-visible `HEAD`:
  `8c3b1beda277f2c085a10cfd82ffe1cd787b9124`
  (`feat(v2): version prepared-data 100-unit protocol`).
- Tracked/staged/untracked repository status was empty when this handoff was
  written.
- `uv.lock` has not changed in the active prepared-data-100 work.
- Last completed verification on the exact Task 1 implementation commit:
  `uv run pytest v2/tests` reported 955 passing tests; the benchmark module
  reported 412 passing tests; Ruff check and format-check passed. These are
  historical results from 2026-08-17, not a substitute for the fresh
  preflight required before measurement.
- Phase 2 has **not** been accepted. No prepared-data-100 baseline or Phase 2
  acceptance artifact exists.

## Primary Specification and Plan Files

| File | Role | Done | Remaining |
| --- | --- | --- | --- |
| `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md` | Approved umbrella design for the complete v2 replacement | Governs the implemented foundation/model/artifact/prepared-data work | Phase 2 gate, all Phase 3 work, Part 1 completion, and all Part 2 phases remain |
| `docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-1.md` | Umbrella Part 1 execution plan | Benchmark prerequisite, Phase 1 code tasks 1.1–1.5, and Phase 2 code tasks 2.1–2.7 are represented by commits on `main` | Phase 2 performance gate is not accepted; Tasks 3.1–3.6 and the Part 1 completion gate have not started |
| `docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md` | Umbrella Part 2 execution plan | Planning document only | All Phase 4 inference/evaluation tasks, Phase 5 LoRA/SWAG tasks, Phase 6 CLI/cutover tasks, and Part 2 completion gate remain |
| `docs/superpowers/specs/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge-design.md` | Defines the real prepared-data benchmark owner/adapter bridge | Production bridge goal implemented at `ffb29f97b7770d06a95df41c2b612c07a9fa6c1b` | Its original 20-unit Phase 2 acceptance path did not produce accepted evidence and is superseded by the prepared-data-100 protocol |
| `docs/superpowers/plans/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge.md` | Bridge implementation plus original Phase 2 screen | Task 1 complete at `ffb29f9` | Task 2 was attempted but rejected; do not resume its old 20-unit comparison command |
| `docs/superpowers/specs/2026-08-16-v2-stop-interruptible-pretraining-stream-design.md` | Root-cause design for the fixed ~50 ms stream-close delay | Shutdown behavior implemented and independently reviewed at `8e2291f1c87244fa8acd33d375c9e49961bb70fa` | Its Task 2 comparison was too noisy under 20 measured units and is superseded by the prepared-data-100 protocol |
| `docs/superpowers/plans/2026-08-16-v2-stop-interruptible-pretraining-stream.md` | Shutdown fix plus rerun plan | Task 1 complete at `8e2291f` | Task 2 did not produce accepted evidence; do not rerun its old protocol |
| `docs/superpowers/specs/2026-08-16-v2-prepared-data-100-unit-measurement-design.md` | **Active approved design** for versioned prepared-data=100 evidence | Design approved and harness behavior implemented | Fresh version-2 baseline, Phase 2 evidence, and final review remain |
| `docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md` | **Active takeover plan** | Task 1 complete and review-clean | Task 2 must restart from zero; Tasks 3 and 4 remain |

## Supporting Benchmark Specifications and Plans

These documents explain the current harness, recovery, and evidence contracts.
Their implementation is already on `main`; they are prerequisites, not active
work to redesign:

| Spec | Plan | Current use |
| --- | --- | --- |
| `docs/superpowers/specs/2026-08-05-v2-baseline-thermal-resume-design.md` | `docs/superpowers/plans/2026-08-05-v2-baseline-thermal-resume.md` | Durable baseline journal, accepted/rejected slots, thermal recovery |
| `docs/superpowers/specs/2026-08-08-v2-shorter-benchmark-protocol-design.md` | `docs/superpowers/plans/2026-08-08-v2-shorter-benchmark-protocol.md` | Global default 20, warmup 5, fixed inference/compile/peak counts, committed v1 baseline |
| `docs/superpowers/specs/2026-08-08-v2-post-exit-memory-evidence-design.md` | `docs/superpowers/plans/2026-08-08-v2-post-exit-memory-evidence.md` | Child/parent/post-exit evidence split and validation |
| `docs/superpowers/specs/2026-08-09-v2-post-exit-memory-recovery-design.md` | `docs/superpowers/plans/2026-08-09-v2-post-exit-memory-recovery.md` | Post-exit memory recovery, rejection dispositions, cooldown evidence |

## Umbrella Refactor Task Status

### Benchmark Prerequisite and Phase 1

| Task | Status | Evidence / note |
| --- | --- | --- |
| Task 0.1: pinned `3687f8b` baseline | Done | `f4e927f` and immutable v1 artifacts listed below |
| Tasks 1.1–1.5: package foundation, fixtures, config/RoPE, model/cache, token processing | Code complete | Commits from `cff6090` through `0a39c11` on `main` |
| Phase 1 performance gate | No committed Phase 1 result is present | Not the immediate takeover target; do not invent a predecessor for prepared-data. The approved Phase 2 mapping is `{"prepared-data": null}` |

### Phase 2 Code Tasks

| Task | Status | Main implementation commit(s) |
| --- | --- | --- |
| 2.1 Canonical identities and strict manifests | Done | `42d1293` plus hardening commits |
| 2.2 Descriptor-relative traversal | Done | `8a489e8` plus `19735ac` |
| 2.3 Publication locks and immutable bundles | Done | `540d4d0`, `51cb470`, `78ae725` |
| 2.4 Checkpoint publication/recovery/retention | Done | `3310de7`, `ab8d99e`, `b88f07a` |
| 2.5 Immutable tokenizer bundles | Done | `3261366` |
| 2.6 Deterministic int32 NPY shards | Done | `6abe045` |
| 2.7 mmap streams, cursors, bounded prefetch | Done | `a2e7ac4` |
| Real prepared-data benchmark bridge | Done | `ffb29f9` |
| Stop-interruptible full-queue shutdown | Done and independently reviewed | `8e2291f` |
| Phase 2 performance gate | **Not done** | Blocked on fresh prepared-data-100 baseline and accepted comparison |

### Later Umbrella Work

- Phase 3 Tasks 3.1–3.6: not started.
- Part 1 completion gate: not started.
- Part 2 Phases 4–6: not started.
- Do not start Phase 3 or Part 2 until the active Phase 2 gate is accepted and
  its evidence has passed review.

## Active Prepared-Data-100 Plan Status

### Task 1: Versioned 100-Unit Harness — Complete

**Commit:** `8c3b1beda277f2c085a10cfd82ffe1cd787b9124`

**Files completed:**

| File | Completed behavior | Remaining in this file |
| --- | --- | --- |
| `v2/benchmarks/workload.py` | Keeps `DEFAULT_MEASURED_UNITS=20`, adds prepared-data-specific 100, preserves all other metric counts | None unless a later review finds a defect |
| `v2/benchmarks/runner.py` | Version-1/20 and version-2/100 baseline dispatch; exact protocol/identity validation; dedicated CLI option; exact parent-to-child metric counts; type-strict tamper rejection; version-aware comparison/phase/final validation | None unless evidence execution exposes a defect |
| `v2/tests/unit/test_benchmark_analysis.py` | Canonical count, parser, v1/v2, session/replay, child/parent, tamper, comparison, phase/final cross-version coverage | None for Task 1 |

Task 1 went through two independent review fix rounds. The final scoped
re-review found all Important findings addressed and no new breakage.

### Task 2: Fresh Version-2 Baseline — Must Restart From Zero

The 2026-08-17 attempt reached 44/45 accepted journal slots, but the last
`peak-metal-memory` trial was thermally rejected and the resume was delayed by
execution quota. During the five-day pause, macOS `/private/tmp` cleanup
removed **every file** in the journal, including `session.json` and all accepted
records, while leaving empty directories. No compatible copy exists.

Therefore:

- The old path `/private/tmp/sml-v2-baseline-prepared100-state` is **not
  resumable**. It is a nonempty directory skeleton with zero files.
- The old 44 trials are diagnostic history only. Never reconstruct, copy, or
  salvage them.
- Use a new empty state directory, for example
  `/private/tmp/sml-v2-baseline-prepared100-state-retry-1`, and capture all 45
  trials again.
- After capture, run the exact shape checks and independent `validate` command
  from Task 2 of the active plan.
- Then obtain independent evidence review and commit exactly:
  - `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json`
  - `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl`

The previous attempt report remains useful only as diagnostics:
`.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/task-2-report.md`.
Its statement that the journal is preserved is superseded by the 2026-08-22
filesystem audit recorded here.

### Task 3: Prepared-Data Phase 2 Comparison — Not Started Under Version 2

After Task 2 is committed and reviewed:

- Run the active plan's exact screen comparison against
  `baseline-3687f8b-prepared100.json`.
- Protocol remains: prepared-data only, 5 pairs, warmup 5, global measure 20,
  `--prepared-data-measure 100`, 10,000 bootstraps, ratio floor 0.97, ratio MAD
  ceiling 0.02, lower bound report-only, predecessor null.
- Comparison is non-resumable. Only the harness may perform its one noise retry
  and cooldown.
- On pass, run independent `validate-phase`, review the evidence, and commit
  exactly:
  - `v2/benchmarks/results/phase-2-loader.json`
  - `v2/benchmarks/results/phase-2.json`
- On fail or `too-noisy`, do not validate, stage, commit, edit, or manually
  combine the report.

### Task 4: Final Broad Review — Not Started

After Task 3 is accepted:

- Rerun Ruff, full v2 pytest, baseline validation, and Phase 2 validation.
- Perform the broad independent review over the harness, baseline evidence,
  and Phase 2 evidence commits.
- Resolve any load-bearing finding using the plan's review loop.
- Record the final handoff. Do not push unless the user explicitly requests it.

## Current Artifact Status

| File | Current state | Action |
| --- | --- | --- |
| `v2/benchmarks/manifests/baseline-3687f8b.json` | Present, immutable v1 baseline; SHA-256 `9281b4b5d70c90931474d334580ce08caf176f1f073cae64dceac658efebd2f3` | Preserve; validate as legacy compatibility evidence |
| `v2/benchmarks/results/baseline-3687f8b.jsonl` | Present, immutable v1 raw evidence; SHA-256 `f141e159164142fd582345a821e5261f74a9ecb0b24dc50bbb0753bb8b3337bc` | Preserve; never reuse its trials in v2 baseline |
| `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json` | Absent | Create only through a successful fresh Task 2 capture |
| `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl` | Absent | Create only through the same successful fresh Task 2 capture |
| `v2/benchmarks/results/phase-2-loader.json` | Absent | Create only through Task 3 comparison |
| `v2/benchmarks/results/phase-2.json` | Absent | Create only after Task 3 independent validation |
| `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/failed-phase-2-loader-too-noisy-1b19e607.json` | Present, ignored diagnostic evidence from the rejected 20-unit run | Preserve as diagnostic only; never stage or validate as new evidence |

## Takeover Procedure

1. Read, in order:
   1. this handoff;
   2. `docs/superpowers/specs/2026-08-16-v2-prepared-data-100-unit-measurement-design.md`;
   3. `docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md`;
   4. `.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/progress.md`;
   5. the Task 1 and Task 2 reports in that workspace.
2. Verify exact `HEAD` is `8c3b1be`, status is empty, `uv.lock` is unchanged,
   the v1 artifacts retain the hashes above, and all four new public output
   paths are absent.
3. Run fresh Ruff and full v2 pytest preflight outside the sandbox with
   MLX/Metal access.
4. Create a brand-new empty external journal path. Do not reuse the stale
   directory skeleton.
5. Require the active plan's 30 continuous nominal seconds: AC connected,
   automatic power mode, Low Power Mode off, thermal nominal/raw 0, normal
   memory pressure, and no competing GPU workload.
6. Immediately run Task 2's exact `record-baseline` command, changing only
   `--state-directory` to the new empty path. Keep `TMPDIR=/private/tmp`,
   `--measure 20`, and `--prepared-data-measure 100`.
7. Complete active plan Tasks 2, 3, and 4 without pausing for redesign unless
   a real blocker or specification conflict appears.

## Safety and Protocol Invariants

- Do not use `--measure 100`; global measure remains 20. Prepared-data uses
  the dedicated `--prepared-data-measure 100` option.
- Do not compare version-2 candidate evidence against the old version-1
  baseline.
- Do not relax ratio, dispersion, environment, identity, or validation gates.
- Do not manually edit benchmark JSON/JSONL or recover old slots by copying.
- All pytest and MLX/Metal benchmark commands run outside the sandbox.
- Work may continue on the user-authorized `main` checkout.
- No automatic push.

## Ignored Execution Records

The plan-specific workspace is:
`.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/`.

Important files there:

- `progress.md`: execution ledger; its final 2026-08-17 Task 2 line is
  superseded only with respect to journal resumability by this handoff.
- `task-1-report.md`: final harness implementation and verification report.
- `task-2-report.md`: failed/lost capture diagnostic report.
- `failed-phase-2-loader-too-noisy-1b19e607.json`: rejected 20-unit diagnostic.
- Task briefs and review/re-review diff packages: historical review evidence.

Because `.superpowers/` is git-ignored, this tracked handoff is the authoritative
cross-session status record.
