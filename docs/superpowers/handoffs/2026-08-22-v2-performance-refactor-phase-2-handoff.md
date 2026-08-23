# V2 Refactor Handoff: Part 1 Remediation and Part 2 Pending

**Recorded:** 2026-08-22 (Asia/Shanghai)

**Last updated:** 2026-08-23 (Asia/Shanghai)

**Purpose:** Give a fresh session enough repository-backed context to finish
Part 1 remediation before beginning Part 2, without reading the prior
conversation.

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
- Documentation-audit base before this consistency update:
  ddad9a8 (docs(v2): record residual Part 1 provenance blocker).
- Latest Part 1 repair/evidence HEAD before this documentation handoff:
  0538b3a (bench(v2): regenerate portable quality evidence).
- The first tracked handoff was committed as 24b3ba4.
- The functional acceptance-policy update was committed as 8970a16.
- Task 3.1 was implemented at 243d168 and hardened at 5856e9d and 589a2f1.
- Task 3.2 was implemented at 882314b and its review findings were fixed at
  5eb977a.
- Task 3.3 was implemented at 0d8ad69 and its review findings were fixed at
  4a47221.
- Task 3.4 was implemented and reviewed at 2b4e9d2.
- Task 3.5's controlled-quality harness was implemented at 8ccf4b6, hardened
  at 4acc76d, and given complete local source-tree provenance at 10338f3.
  Canonical evidence was committed at dba8e1d.
- Task 3.6's functional gate passed on dba8e1d on 2026-08-23:
  `uv run ruff check v2`, `uv run ruff format --check v2`, and all 1,058 v2
  tests passed in 60.21 seconds. The standalone controlled-quality validator
  exited 0 with `pass`; `uv.lock` was unchanged, `git diff --check` passed,
  and the tracked worktree was clean.
- The subsequent broad Part 1 architecture review over `aa6bb43..79a46d4`
  found 0 Critical, 5 Important, and 1 Minor issue. Part 2 Task 4.1 is paused
  pending implementation and review of the approved provenance policy below.
- The consolidated repair was committed at 54f1749 and regenerated evidence
  at 0538b3a. Fresh Ruff passed, 92 files were already formatted, all 1,062
  v2 tests passed, standalone/current-relocated quality validation passed, and
  `uv.lock` remained unchanged. Scoped re-review closed four Important issues
  and the original Minor, but kept one Important quality-provenance issue open
  and found one new non-blocking shallow-immutability Minor.
- Independent reviews for Tasks 3.1 through 3.5 are clean. One non-blocking
  Task 3.1 Minor is deferred: an
  extremely large Python integer passed to scalar validation can surface
  `OverflowError` instead of the project configuration-error type.
- Task 3.2 ruling: the later compiled-runtime contract supersedes Task 3.1's
  earlier three-field `TrainerState` declaration. Its canonical built-in tree
  now also carries an FP32 scalar additive loss numerator, enabling accurate
  on-device aggregation without per-microstep synchronization.
- Task 3.3 ruling: checkpoint retention is mandatory latest-only. There is no
  public retention/history control, and `ResumeOverrides` has exactly four
  fields: `maximum_steps`, `maximum_epochs`, `log_interval`, and
  `checkpoint_interval`.
- Task 3.4 ruling: its stale example of reopening step 1 after reaching step 3
  conflicts with the authoritative latest-only design. The integration instead
  FULL-reopens exact latest step 3, proves step 1 is unavailable, and proves
  only the canonical latest checkpoint remains.
- Quality-provenance ruling (approved 2026-08-23): committed controlled-quality
  evidence is immutable evidence for its recorded source commit. Validation
  must reconstruct and verify that recorded source boundary instead of
  rebuilding an expected workload from unrelated current-worktree bytes.
  Evidence is rerecorded only after a change to the controlled numerical
  execution contract, harness/workload, fixtures, telemetry, or decision rule.
- Documentation consistency audit (2026-08-23): all 25 tracked spec/plan files
  have explicit current status and appear in this handoff; active relative plan
  links resolve and Markdown code fences are balanced. Fresh verification on
  the resulting documentation tree passed Ruff check, Ruff format check (92
  files already formatted), and all 1,062 v2 tests in 65.04 seconds.

## Related Specifications and Plans

| File | Role | Done | Remaining under current policy |
| --- | --- | --- | --- |
| docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md | Umbrella design | Part 1 foundation, model, artifacts, tokenizer, prepared data, pretraining, integration, and numerical quality run are functionally implemented | Implement the approved source-bound quality-evidence validator and checkpoint-content immutability fix, then Phases 4-6; performance remains optional |
| docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-1.md | Phases 1-3 plan | All task bodies, the controlled run, and the repeated functional gate ran successfully | Correct quality validation to use the recorded source boundary, pass scoped review, and freeze checkpoint-content mappings |
| docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md | Phases 4-6 plan | Planning is complete; implementation has not started | All Tasks 4.1-4.4, 5.1-5.5, and 6.1-6.4 after Part 1 remediation |
| docs/superpowers/specs/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge-design.md | Real prepared-data benchmark owner design | Production goal implemented at ffb29f97b7770d06a95df41c2b612c07a9fa6c1b | Nothing required; screen is optional |
| docs/superpowers/plans/2026-08-16-v2-phase-2-prepared-data-benchmark-bridge.md | Bridge implementation plan | Task 1 complete at ffb29f9 | Task 2 retained only as an optional diagnostic |
| docs/superpowers/specs/2026-08-16-v2-stop-interruptible-pretraining-stream-design.md | Full-queue shutdown correctness design | Implemented and reviewed at 8e2291f1c87244fa8acd33d375c9e49961bb70fa | Nothing required; timing rerun is optional |
| docs/superpowers/plans/2026-08-16-v2-stop-interruptible-pretraining-stream.md | Shutdown fix plan | Task 1 complete at 8e2291f | Task 2 retained only as an optional diagnostic |
| docs/superpowers/specs/2026-08-16-v2-prepared-data-100-unit-measurement-design.md | Versioned 20/100 benchmark protocol | Harness design implemented | Baseline capture/comparison only if explicitly desired |
| docs/superpowers/plans/2026-08-16-v2-prepared-data-100-unit-measurement.md | Versioned protocol implementation plan | Required Tasks 1 and 4 complete; Task 1 reviewed at 8c3b1be | Tasks 2 and 3 remain optional diagnostic procedures only |
| docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md | Authoritative cross-session status | Updated through the approved quality-provenance ruling | Keep current as remediation and later tasks complete |

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

## Historical Precursor Documents

These implemented documents describe v1 work or the legacy flat v2 system.
They are historical records, not remaining work for the package refactor:

| Spec | Plan(s) | Status |
| --- | --- | --- |
| docs/superpowers/specs/2026-07-01-model-training-input-file-shuffle-design.md | docs/superpowers/plans/2026-07-01-model-training-input-file-shuffle.md | Implemented in v1; no remaining refactor task |
| docs/superpowers/specs/2026-07-05-v2-mlx-native-training-design.md | docs/superpowers/plans/2026-07-05-v2-mlx-model.md; docs/superpowers/plans/2026-07-05-v2-mlx-native-training.md; docs/superpowers/plans/2026-07-11-v2-mlx-only.md | Implemented in the legacy flat v2 tree and superseded by the umbrella package refactor |
| docs/superpowers/specs/2026-07-15-train-on-pretraining-data-design.md | docs/superpowers/plans/2026-07-15-train-on-pretraining-data.md | Implemented legacy NPZ prepared-data flow; superseded by the strict NPY bundle design |

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
| 3.3: pretraining create/checkpoint/exact resume/latest-only retention | Complete and independently reviewed | 0d8ad69, 4a47221 |
| 3.4: complete tokenizer-to-resumed-pretraining integration | Complete and independently reviewed | 2b4e9d2 |
| 3.5: controlled pretraining-quality gate | Numerical gate passed; final portable-provenance review open | 8ccf4b6, 4acc76d, 10338f3, dba8e1d, 54f1749, 0538b3a |
| 3.6: Phase 3 correctness and workflow gate | Functional gate repeated successfully | Latest gate passed on 0538b3a; no separate implementation commit |

Task 3.3 now supplies fresh training and exact resume around the reviewed
compiled kernels. A committed checkpoint is a closed six-file bundle:
`checkpoint.json`, `state.json`, `model.safetensors`, `master.safetensors`,
`optimizer.safetensors`, and `trainer.safetensors`. Resume validates the full
run/checkpoint identity, tensor keys/shapes/dtypes, scalar state, canonical
prepared-data cursor, and empty accumulation boundary before latest-only
pruning. Interrupted uncommitted work replays from the last committed cursor.
The portable model payload was also FULL-resolved after relocating the run and
removing the original prepared-data path, then exercised through a real model
forward pass; the future public model-only consumer remains owned by Task 4.

Task 3.4 exercises the real raw-corpus, SentencePiece tokenizer, deterministic
int32 preparation, two-step training, resume-to-three, exact latest resolution,
FP32-master/BF16-working cast, and latest-only retention path. It also exposes
`verify_artifact(path, full)` and recursive `VerificationResult` from
`sml.artifacts`. Prepared-data verification binds its copied tokenizer's
canonical paths, identities, and byte sizes; run verification holds the shared
access lock while FULL-verifying the copied tokenizer and latest checkpoint.

Task 3.5 adds an immutable candidate-versus-FP32-oracle quality harness with
two checked-in row-source-disjoint int32 fixtures. Canonical evidence contains
exactly eight records: candidate then oracle at optimizer steps 0, 10, 100,
and 1,000. The public recorder accepts exactly 1,000 optimizer steps, preserves
the default 12-layer/768-hidden/1,024-token workload, accumulation of 8, and
the actual 268,000-step optimizer schedule while terminating this controlled
run after 1,000 optimizer updates. Harness identity remains a separate
two-file identity. The regenerated harness uses an AST-derived local import
closure and repository-relative destinations, but the residual review below
explains why that closure is not yet stable across the written Part 2 edits.

The canonical run passed. Candidate final validation NLL is
`10.743444323539734`; the FP32 oracle is `10.729785919189453`; their ratio is
`1.0012729428576812`, below the required `1.01`. All recorded state is finite,
RMSNorm FP32 masters moved, and sub-BF16-ULP master updates survived into later
working state. The regenerated evidence at 0538b3a measured
`5683.967791709001` seconds, peak Metal allocation `19804952442` bytes, and
1,114,535 bytes for fixtures plus evidence. These resource values are evidence
metadata, not performance gates. The recorder's post-publication validation,
standalone validator, unrelated-module probe, and relocated-checkout validator
all returned `pass`; the residual concern is guaranteed future-source
stability, not the numerical experiment or current/relocated byte integrity.

## Part 1 Final Architecture Review Status

The whole-plan review found cross-task issues not caught by the task-scoped
reviews. The consolidated repair and scoped re-review closed these findings:

1. Replace generic run/checkpoint schemas with the design's distinct strict
   pretraining and LoRA kinds, and make recursive FULL verification perform
   owner-kind semantic validation including real array metadata and the
   FP32-master-to-BF16-working cast proof.
2. Run a complete synchronous prepared-data semantic preflight (NPY metadata,
   token ranges, and batch availability) before checkpoint array restoration,
   retention, or completed-limit return.
3. Consume checkpoint scalar/array payloads through owned descriptor-bound
   verified reads so a namespace swap cannot separate FULL proof from the
   bytes actually loaded.
4. Reject exact duplicate logical payload paths as well as Unicode/case-fold
   collisions, including duplicate prepared shards and checkpoint groups.
The original Minor was also closed: arbitrary public `keep_last` retention is
gone, with only parameterless latest-only pruning exposed.

One Important finding remains open. The regenerated evidence is repository-
relative and validates from a relocated checkout, but its source closure still
hashes `artifacts/checkpoint.py`, which Task 4.1 must edit, and requires the
temporary flat `v2/src/sml.py` bridge, which Phase 6 must delete. Rebuilding the
workload from current bytes therefore guarantees that the final Part 2 gate
will reject this evidence. The closure also omits executed legacy
`v2/src/config.py`. The numerical result is accepted as internally coherent,
but the artifact is rejected as the promised portable Part 2/final acceptance
proof.

The user approved the source-bound provenance policy on 2026-08-23. Treat the
quality evidence as immutable historical evidence bound to its recorded source
commit. Validation must reconstruct and verify that recorded source boundary
rather than compare with unrelated current-worktree bytes. Do not regenerate
the approximately 95-minute controlled run unless implementing the policy
changes the recorded evidence identity or a later change alters the controlled
numerical execution contract.

The scoped re-review's new Minor is that `VerifiedCheckpointContents` is only
shallowly frozen because it exposes mutable outer/inner dictionaries. Current
pretraining copies them and is safe; freeze these mappings before Task 4.1
retains or exposes that reader result. The eight older deferred Minors remain
non-blocking under the deadlines recorded in the final review workspace.

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

## Part 1 Completion Status

| Task | Status | Required completion evidence |
| --- | --- | --- |
| Phase 2 functional preflight | Complete | Ruff clean; prepared-data workflow tests passed; all 955 then-current v2 tests passed; unchanged uv.lock |
| 3.1: training policies and project-owned mixed-precision Adam | Complete and reviewed | 27 focused tests and all 974 v2 tests passed on 589a2f1; Ruff clean; unchanged uv.lock |
| 3.2: compiled pretraining microstep and optimizer step | Complete and reviewed | 37 focused tests and all 984 v2 tests passed on 5eb977a; Ruff clean; unchanged uv.lock |
| 3.3: pretraining create/checkpoint/exact resume/latest-only retention | Complete and reviewed | 194 focused tests and all 1,022 v2 tests passed on 4a47221; Ruff clean; unchanged uv.lock |
| 3.4: complete Part 1 integration | Complete and reviewed | 133 focused tests and all 1,024 v2 tests passed on 2b4e9d2; Ruff clean; unchanged uv.lock |
| 3.5: controlled pretraining-quality gate | Numerically passed; validator remediation remains | Regenerated evidence at 0538b3a passes current/relocated validation; implement the approved recorded-source validation rule so it remains valid through Tasks 4.1 and 6.3 |
| 3.6: Phase 3 correctness and workflow gate | Repeated after repair | Ruff clean; all 1,062 v2 tests passed; current standalone quality validation passed; unchanged uv.lock |
| Part 1 completion | Blocked by one residual Important | Implement the approved stable quality-provenance contract, regenerate only if that changes evidence identity, pass scoped review, and freeze checkpoint-content mappings before Task 4.1 exposes them |

## Remaining Part 2 Tasks

| Phase | Tasks | Status | Required gate |
| --- | --- | --- | --- |
| 4: inference and evaluation | 4.1-4.4 | Blocked until Part 1 remediation closes | Correctness, inference/evaluation integration, applicable CLI smoke |
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
| v2/benchmarks/manifests/pretraining-quality-v1.json | Present, immutable Task 3.5 evidence | Preserve |
| v2/benchmarks/results/pretraining-quality-v1.jsonl | Present, immutable Task 3.5 raw evidence | Preserve |
| v2/benchmarks/results/pretraining-quality-v1.json | Present, immutable Task 3.5 report | Preserve |

The absent files are not blockers and should not be fabricated.

## Takeover Procedure

1. Read this handoff, the umbrella design, the Part 1 plan, and the Part 2
   Task 4.1 consumer contract.
2. Confirm the latest main commit and inspect git status. Preserve unrelated
   user changes if any exist.
3. Implement the approved recorded-source quality-provenance policy described
   above. Do not start Task 4.1 until the reviewed validator survives its
   required checkpoint-file edit and the Phase 6 bridge deletion.
4. Use a strict test-first repair for the selected provenance contract and the
   shallow checkpoint-content mapping Minor. Run MLX pytest outside the
   sandbox; rerun canonical quality only if the approved policy changes the
   recorded evidence identity.
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

## Ignored Review and Historical Records

The active Part 1 final-review evidence is preserved in
.superpowers/sdd/2026-08-01-v2-performance-first-refactor-part-1/, especially
`part1-final-review.md`, `part1-final-fix-report.md`,
`part1-final-re-review.md`, and `progress.md`. Keep that workspace until Part 1
remediation closes; it is supporting evidence, while this tracked handoff is
the authoritative status.

The historical benchmark workspace
.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/
contains Task 1 reports, rejected benchmark diagnostics, review packages, and
the old execution ledger. Those files are useful history but are not required
takeover state.
