# V2 Refactor Handoff After Phase 2

**Recorded:** 2026-08-22 (Asia/Shanghai)

**Last updated:** 2026-08-23 (Asia/Shanghai)

**Purpose:** Give a fresh session enough repository-backed context to continue
the v2 refactor without reading the prior conversation.

## Superseding Acceptance Policy

The user removed before/after performance comparison as a refactor gate on
2026-08-22.

Required before completing each future phase:

- uv run ruff check v2
- uv run ruff format --check v2
- uv run pytest v2/tests outside the sandbox so MLX/Metal is available
- the phase's relevant integration, resume, artifact, quality, and CLI smoke
  workflows
- git diff --exit-code -- uv.lock

Optional and non-blocking:

- capturing or validating a performance baseline
- comparing the implementation with commit 3687f8b
- producing phase-N or final-acceptance benchmark JSON/JSONL
- satisfying a throughput ratio, confidence bound, dispersion threshold,
  thermal launch window, or cooldown requirement

The benchmark harness remains supported. If someone chooses to publish a
performance claim, its identity, environment, protocol, and evidence validators
still apply to that claim. A missing, noisy, thermally rejected, inconclusive,
or slower result never blocks implementation work.

Controlled mathematical and training-quality checks are still required because
they verify correctness rather than speed.

## Current Repository Snapshot

- Branch: main, tracking origin/main.
- Refactor implementation HEAD before the documentation handoff:
  5eb977a (fix(v2): preserve compiled pretraining loss state).
- The first tracked handoff was committed as 24b3ba4.
- The functional acceptance-policy update was committed as 8970a16.
- Task 3.1 was implemented at 243d168 and hardened at 5856e9d and 589a2f1.
- Task 3.2 was implemented at 882314b and its review findings were fixed at
  5eb977a.
- Fresh controller verification on 5eb977a passed on 2026-08-23:
  `uv run ruff check v2`, `uv run ruff format --check v2`, 37 focused training
  and model-math tests, and all 984 v2 tests. `uv.lock` was unchanged and
  `git diff --check` passed.
- Independent reviews for Tasks 3.1 and 3.2 are clean. One non-blocking Task
  3.1 Minor is deferred: an
  extremely large Python integer passed to scalar validation can surface
  `OverflowError` instead of the project configuration-error type.
- Task 3.2 ruling: the later compiled-runtime contract supersedes Task 3.1's
  earlier three-field `TrainerState` declaration. Its canonical built-in tree
  now also carries an FP32 scalar additive loss numerator, enabling accurate
  on-device aggregation without per-microstep synchronization.

## Related Specifications and Plans

| File | Role | Done | Remaining under current policy |
| --- | --- | --- | --- |
| docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md | Umbrella design | Foundation, model, artifacts, tokenizer, prepared data, and benchmark-owner architecture are implemented through Phase 2 | Phases 3-6; performance section is optional diagnostic guidance |
| docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-1.md | Phases 1-3 plan | Phase 1 tasks 1.1-1.5, Phase 2 tasks 2.1-2.7, the Phase 2 functional preflight, and Tasks 3.1-3.2 are complete | Tasks 3.3-3.6 |
| docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md | Phases 4-6 plan | Planning only | All Tasks 4.1-4.4, 5.1-5.5, and 6.1-6.4 |
| docs/superpowers/specs/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge-design.md | Real prepared-data benchmark owner design | Production goal implemented at ffb29f97b7770d06a95df41c2b612c07a9fa6c1b | Nothing required; screen is optional |
| docs/superpowers/plans/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge.md | Bridge implementation plan | Task 1 complete at ffb29f9 | Task 2 retained only as an optional diagnostic |
| docs/superpowers/specs/2026-08-16-v2-stop-interruptible-pretraining-stream-design.md | Full-queue shutdown correctness design | Implemented and reviewed at 8e2291f1c87244fa8acd33d375c9e49961bb70fa | Nothing required; timing rerun is optional |
| docs/superpowers/plans/2026-08-16-v2-stop-interruptible-pretraining-stream.md | Shutdown fix plan | Task 1 complete at 8e2291f | Task 2 retained only as an optional diagnostic |
| docs/superpowers/specs/2026-08-16-v2-prepared-data-100-unit-measurement-design.md | Versioned 20/100 benchmark protocol | Harness design implemented | Baseline capture/comparison only if explicitly desired |
| docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md | Versioned protocol implementation plan | Task 1 complete and reviewed at 8c3b1be | Tasks 2 and 3 cancelled as required work; Task 4 is functional handoff maintenance |
| docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md | Authoritative cross-session status | Updated to the functional acceptance policy | Keep current as later tasks complete |

## Supporting Benchmark Documents

These files describe existing optional harness behavior. They are implemented
technical references, not active progression gates, so their internal protocol
details remain unchanged:

| Spec | Plan | Implemented capability |
| --- | --- | --- |
| docs/superpowers/specs/2026-08-05-v2-baseline-thermal-resume-design.md | docs/superpowers/plans/2026-08-05-v2-baseline-thermal-resume.md | Durable baseline journal and thermal recovery |
| docs/superpowers/specs/2026-08-08-v2-shorter-benchmark-protocol-design.md | docs/superpowers/plans/2026-08-08-v2-shorter-benchmark-protocol.md | Shorter canonical measurement protocol |
| docs/superpowers/specs/2026-08-08-v2-post-exit-memory-evidence-design.md | docs/superpowers/plans/2026-08-08-v2-post-exit-memory-evidence.md | Child/parent/post-exit evidence |
| docs/superpowers/specs/2026-08-09-v2-post-exit-memory-recovery-design.md | docs/superpowers/plans/2026-08-09-v2-post-exit-memory-recovery.md | Post-exit recovery and cooldown evidence |

## Completed Refactor Tasks

### Benchmark and Phase 1

| Task | Status | Evidence |
| --- | --- | --- |
| 0.1: versioned benchmark harness and pinned v1 baseline | Complete | f4e927f and existing v1 artifacts |
| 1.1: package foundation and temporary bridge | Complete | Commits beginning at cff6090 |
| 1.2: legacy equivalence fixtures | Complete | On main |
| 1.3: frozen config and YaRN/RoPE | Complete | On main |
| 1.4: explicit cache/layers/tied model | Complete | On main |
| 1.5: token processors and explicit sampling | Complete | Through 0a39c11 |
| Phase 1 performance report | Not present | No longer required |

### Phase 2

| Task | Status | Implementation commit(s) |
| --- | --- | --- |
| 2.1: canonical identities and strict manifests | Complete | 42d1293 plus hardening |
| 2.2: descriptor-relative traversal | Complete | 8a489e8 and 19735ac |
| 2.3: publication locks and immutable bundles | Complete | 540d4d0, 51cb470, 78ae725 |
| 2.4: checkpoint publication/recovery/retention | Complete | 3310de7, ab8d99e, b88f07a |
| 2.5: immutable tokenizer bundles | Complete | 3261366 |
| 2.6: deterministic int32 NPY shards | Complete | 6abe045 |
| 2.7: mmap streams, cursors, bounded prefetch | Complete | a2e7ac4 |
| Real prepared-data benchmark owner | Complete | ffb29f9 |
| Stop-interruptible full-queue shutdown | Complete and reviewed | 8e2291f |
| Versioned prepared-data 100 harness | Complete and reviewed | 8c3b1be |
| Phase 2 before/after report | Not present | No longer required |

Phase 2 implementation and its fresh functional preflight are complete. No
benchmark recapture is required.

### Phase 3

| Task | Status | Implementation commit(s) |
| --- | --- | --- |
| 3.1: training policies and project-owned mixed-precision Adam | Complete and independently reviewed | 243d168, 5856e9d, 589a2f1 |
| 3.2: compiled pretraining microstep and optimizer step | Complete and independently reviewed | 882314b, 5eb977a |

## Cancelled Required Benchmark Work

The old prepared-data-100 Task 2 capture reached 44 of 45 accepted temporary
journal slots before a thermal rejection. macOS later cleaned the files from
the temporary journal, so that attempt is not resumable.

Under the current policy:

- do not restart the 45-trial baseline merely to continue the refactor;
- do not run the prepared-data comparison merely to continue the refactor;
- do not create placeholder baseline or Phase 2 result artifacts; and
- preserve existing benchmark code and committed v1 evidence.

If the user later explicitly requests comparable performance evidence, use a
new empty external journal and follow the optional prepared-data-100 protocol.

## Remaining Part 1 Tasks

| Task | Status | Required completion evidence |
| --- | --- | --- |
| Phase 2 functional preflight | Complete | Ruff clean; prepared-data workflow tests passed; all 955 then-current v2 tests passed; unchanged uv.lock |
| 3.1: training policies and project-owned mixed-precision Adam | Complete and reviewed | 27 focused tests and all 974 v2 tests passed on 589a2f1; Ruff clean; unchanged uv.lock |
| 3.2: compiled pretraining microstep and optimizer step | Complete and reviewed | 37 focused tests and all 984 v2 tests passed on 5eb977a; Ruff clean; unchanged uv.lock |
| 3.3: pretraining create/checkpoint/resume/retention | Next | Fresh/resume/recovery workflow tests |
| 3.4: complete Part 1 integration | Not started | End-to-end tiny production workflow |
| 3.5: controlled pretraining-quality gate | Not started | Required correctness/quality evidence |
| 3.6: Phase 3 correctness and workflow gate | Not started | Ruff, full v2 pytest, quality validation, integration smoke |
| Part 1 completion | Not started | Functionally verified committed Phase 3 tree |

## Remaining Part 2 Tasks

| Phase | Tasks | Status | Required gate |
| --- | --- | --- | --- |
| 4: inference and evaluation | 4.1-4.4 | Not started | Correctness, inference/evaluation integration, applicable CLI smoke |
| 5: LoRA and SWAG | 5.1-5.5 | Not started | Correctness, controlled SWAG quality, resume/export/inference workflow |
| 6: unified CLI and cutover | 6.1-6.4 | Not started | Full Ruff/pytest, all CLI workflows, clean cutover, unchanged uv.lock |

No Phase 4, 5, 6, or final performance result file is required.

## Benchmark Artifact Status

| File | State | Required action |
| --- | --- | --- |
| v2/benchmarks/manifests/baseline-3687f8b.json | Present, immutable v1 evidence | Preserve |
| v2/benchmarks/results/baseline-3687f8b.jsonl | Present, immutable v1 evidence | Preserve |
| v2/benchmarks/manifests/baseline-3687f8b-prepared100.json | Absent | None |
| v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl | Absent | None |
| v2/benchmarks/results/phase-2-loader.json | Absent | None |
| v2/benchmarks/results/phase-2.json | Absent | None |

The absent files are not blockers and should not be fabricated.

## Takeover Procedure

1. Read this handoff, the umbrella design, and Part 1 plan.
2. Confirm the latest main commit and inspect git status. Preserve unrelated
   user changes if any exist.
3. Begin Part 1 Task 3.3, pretraining run creation, checkpoint, exact resume,
   and retention, using the reviewed kernels in
   `v2/src/sml/training/pretrain.py` and the canonical four-item trainer tree.
4. Use test-driven implementation and verify fresh/resume/recovery workflow
   contracts. Run MLX pytest commands outside the sandbox.
5. Follow the updated functional gate at the end of each phase. Do not pause
   for baseline capture or performance comparison.
6. Keep this handoff current with completed commits, fresh test evidence, and
   the next functional task.

## Invariants That Remain Strict

- Run MLX pytest commands outside the sandbox.
- Preserve mathematical, dtype, explicit-randomness, artifact-integrity,
  path-safety, lock, atomic publication, checkpoint, cursor, and exact-resume
  contracts.
- Keep controlled pretraining/SWAG quality validation required.
- Do not add test-only production paths or weaken production verification.
- Do not edit top-level files or uv.lock without explicit approval.
- Work may continue on the user-authorized main checkout.
- Do not push unless the user explicitly requests it.

## Ignored Historical Records

The ignored workspace
.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/
contains Task 1 reports, rejected benchmark diagnostics, review packages, and
the old execution ledger. Those files are useful history but are not required
takeover state. This tracked handoff is authoritative.
