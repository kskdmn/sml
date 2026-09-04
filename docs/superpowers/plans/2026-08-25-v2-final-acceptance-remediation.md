# V2 Final-Acceptance Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every conformance gap found by the whole-scope v2 review, regenerate the numerically affected SWAG evidence, and finish Task 6.4 with fresh final-commit verification.

**Architecture:** Five focused plans implement the approved remediation in dependency order. Evaluation, LoRA, and inference first establish their local contracts; descriptor-bound artifact ownership then supplies the common safe-loading foundation and recursive semantic verifier; the final plan regenerates evidence and repeats all repository gates and reviews.

**Tech Stack:** Python 3.12.13, MLX, NumPy, SentencePiece, lm-evaluation-harness 0.4.12, Hugging Face datasets, pytest, Ruff, canonical SML JSON/artifact schemas, Git.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

**Final status (2026-09-05): Test-only follow-up required.** Task 6.4 and the
remediation implementation are complete, but final acceptance is not closed.
Final production source/tests are at
`d6a28498a33624ccb6e58b17b380c15c9f072211`; final SWAG evidence
retirement/source-harness is `0f767cb73715eb77bd54e5fd02d6b9bc13b9c0e6`;
and verified pre-documentation evidence HEAD is
`da3dc9365503059bd0e2c1f60c3b2b3c257c3443`. The original Task 6
documentation history remains: completion
`34c7ba4f775ead472aa780a231e7475be1bd3831`, exact-SHA metadata
`fc9d00a4f2b97159f80eebd99599a557247cc399`, and review-scope clarification
`6dbc059f7c36494be717c6622a2960ca26dea84c`.

Task 5's plan-required architecture acceptance at `6647282` historically
found Critical 0, Important 0, Minor 0 after two fix rounds. The additional SDD
final whole-branch review at `6dbc059` found Critical 0, Important 1, Minor 0:
a reduced-guarantee checkpoint-reader issue. The source/test wave at `d6a2849`
fixed it, SWAG evidence was retired at `0f767cb`, and final evidence was
refreshed at `da3dc93`. The sole scoped re-review at `099509f` confirmed the
original Important was addressed but returned Critical 0, Important 1, Minor 1
and did not approve final closure. The Important is missing lasting regression
coverage for authority failure after sidecar-protocol mutation. The Minor was
the ambiguous word “reviewed” for `da3dc93`; this update replaces it with
“verified.” The test gap does not invalidate the production correction or
recorded SWAG evidence, but acceptance remains open until an authorized
test-only correction and follow-up review pass.

Final gates at `da3dc93`: full V2 `1611 passed in 105.66s`;
integration `252 passed in 23.10s`; CLI workflows `31 passed in 6.25s`;
CLI config `13 passed in 1.38s`; source/package `9 passed in 0.63s`; Ruff clean
with `104 files already formatted`; pretraining and SWAG validators both
`pass`. The SWAG manifest binds source/harness
`0f767cb73715eb77bd54e5fd02d6b9bc13b9c0e6`, `harness_clean=true`; its final
manifest/raw/report hashes are
`5af2abb5200eba0a0faa3d7774bca16e129b7e5dcf9473c50c9b700a9ed8976f`,
`837308e38e364a65a4cb1e021ec085846559ab6eb710868a83a7b2cbb8643f8d`, and
`0bf3d935a0165178b2746a08f853bd39620ccd8ad63f66ae82adb725513d5999`.

All seven protected hashes are unchanged: pretraining manifest
`17a346df8e0ded255cb50e40a568517b3a1c72c0ccbc1828a044c3f3dac12763`, raw
`e80197de96c2733a5f6790bb85a3f6f142c2a8475436772dee521656b2248beb`, report
`b64f13920d1ffc78754070c6a5107635b2507d50ba54348f7c1d6462b6b30bd2`, train
fixture `6de8260bb5c060c2391ab69df4baae500d1b549e812233321f46843a156e33aa`,
validation fixture
`b5fd31a7de28084a916290c2c89ce87b320b6aca4519d0cf6f125d20c3f14d49`, SWAG
train fixture
`ff5fb55512e02fd3a4a5b6eb9e72aa2bc747e4bbdb64107d183230dc94750d60`, and
SWAG validation fixture
`a82fe60cc118ffc68119f4b99e8cf04d859fa39997d7df5b797e5b676323101a`.
`uv.lock` is unchanged from `4225c54`; there are no flat `v2/src/*.py` or
forbidden legacy bridge strings; the CLI/cutover scans pass; help lists all
eight commands (`tokenize`, `prepare`, `train`, `infer`, `evaluate`,
`finetune`, `export`, `verify`); and the accepted checkout was clean.
Performance measurement remains optional and is not an acceptance gate.

Completed behavior includes strict version dispatch that preserves the frozen
evaluation-result v1 field set and `sml-evaluation-result-v1` identity domain,
reports legacy-v1 recovery state as unavailable, and makes the current writer
emit strict v2 with Boolean `latest_recovered`/`pruning_pending` under
`sml-evaluation-result-v2`. Read-only resolution reports recovered latest and
pending pruning without mutation; merged export performs FULL source-run
semantic validation; and evaluation provenance uses regular-file,
load-adjacent, byte-stable YAML snapshots rechecked before publication.
Checkpoint reads retain shared-lock protection on local APFS and enter an
explicit reduced-guarantee read-only mode only for proven local non-APFS
storage or narrow sidecar permission/read-only failures; that mode does not
promise concurrent-writer or pruning exclusion, and other failures remain
fail-closed.

## Global Constraints

- Continue in the current checkout and current `main` branch; the user explicitly rejected creating a worktree for this effort.
- Run Python through `uv run`.
- Run every `uv run pytest` command outside the sandbox so MLX/Metal can access the Apple GPU.
- Run MLX quality recording and validation outside the sandbox.
- Do not edit top-level project files, including `pyproject.toml` and `uv.lock`.
- Preserve the clean-break package: no flat modules, migration bridge, compatibility aliases, legacy readers, or historical checkpoint selection.
- Preserve canonical pretraining quality evidence byte-for-byte and only revalidate it.
- Regenerate canonical SWAG quality evidence because restored LoRA dropout changes its controlled numerical trajectory.
- Performance comparisons are optional diagnostics and are not acceptance gates.
- Use TDD for every behavior change: focused RED, minimal GREEN, focused verification, then a commit.
- Keep the tracked checkout clean before SWAG evidence recording; the evidence must bind the exact committed repair source.
- Do not update umbrella completion/status prose until all final gates and the whole-scope review pass.

---

## Plan Set and Dependency Order

1. [`2026-08-25-v2-remediation-evaluation.md`](2026-08-25-v2-remediation-evaluation.md)
   freezes the strict evaluation artifact, instruments lm-eval provenance, and
   publishes complete provider results.
2. [`2026-08-25-v2-remediation-lora.md`](2026-08-25-v2-remediation-lora.md)
   removes BF16 scale state and threads explicit-key LoRA dropout through live,
   pure-array, eager, compiled, and resume paths.
3. [`2026-08-25-v2-remediation-inference.md`](2026-08-25-v2-remediation-inference.md)
   separates prefill and cache-capacity buckets and gathers target logits before
   log-sum-exp subtraction.
4. [`2026-08-25-v2-remediation-artifacts.md`](2026-08-25-v2-remediation-artifacts.md)
   introduces descriptor-bound payload ownership, migrates SWAG/safetensors
   consumers, and makes FULL verification recursively semantic.
5. [`2026-08-25-v2-remediation-final-acceptance.md`](2026-08-25-v2-remediation-final-acceptance.md)
   runs integration gates, recaptures SWAG quality evidence, verifies repository
   invariants, obtains whole-scope review, and closes documentation.

The first three plans are locally independent but execute in the listed order
to keep review history deterministic. The artifact plan runs after them because
its integration tests exercise the final evaluation, LoRA, and inference
consumers. The final-acceptance plan runs last and never starts from uncommitted
source changes.

## Execution Protocol

- [x] **Step 1: Confirm the approved planning base**

Run:

```bash
git status --short --branch
git log -2 --oneline
```

Expected: the tracked checkout is clean and `9a8b170` is the committed
remediation design at the tip or an ancestor of the current tip.

- [x] **Step 2: Execute and review the evaluation plan**

Read the remediation spec and the complete evaluation plan, execute every
checkbox in order, and stop at each commit for requirements and code-quality
review.

Expected: strict evaluation artifacts round-trip complete provider metrics and
task provenance; focused evaluation and package/CLI tests pass.

- [x] **Step 3: Execute and review the LoRA plan**

Read the complete LoRA plan, execute its RED/GREEN cycles, and review the
numerical formula, key order, compiled state, and exact resume evidence before
continuing.

Expected: scale is static FP32 configuration, adapter leaves alone are
trainable FP32 arrays, configured adapter dropout runs in pure-array training,
and the old SWAG quality evidence is now expected to require replacement.

- [x] **Step 4: Execute and review the inference plan**

Read the complete inference plan, execute each task, and review both stable
shape domains and the scoring allocation contract.

Expected: short-prompt/long-generation requests use prompt-sized prefill and
capacity-sized storage; scoring produces only token-shaped log probabilities.

- [x] **Step 5: Execute and review the artifact plan**

Read the complete artifact plan, execute descriptor primitives before consumer
migrations, then build recursive verification only on the migrated owner
loaders.

Expected: verification and consumption share retained descriptors, SWAG owns
its mappings deterministically, and re-signed semantic corruptions fail FULL
CLI verification.

- [x] **Step 6: Execute the final-acceptance plan**

Run its focused integration checkpoint, commit all repair source, retire the old
SWAG evidence in a dedicated commit, record new evidence from the resulting
clean commit, and perform every final gate.

Expected: Ruff, all 1,104-or-more tests, quality validators, CLI workflows,
integration workflows, repository invariants, and whole-scope review all pass
at the final committed tree.

- [x] **Step 7: Record completion only after evidence and review**

Update the umbrella spec, Part 2 plan status, and handoff with exact final
commit/evidence/test counts. Commit those documentation changes and rerun the
lightweight repository checks named in the final-acceptance plan.

Expected: Task 6.4 and the v2 performance-first refactor are marked complete
without stale “next task” prose, and the tracked checkout is clean.

## Commit Sequence

Use these semantic checkpoints; a task may add a narrower intermediate commit
when its focused tests form an independently reviewable unit:

```text
feat(v2): persist strict evaluation artifacts
fix(v2): restore lora precision and dropout
fix(v2): separate inference shape domains
fix(v2): bind artifact proof to consumption
fix(v2): verify artifact semantics recursively
test(v2): retire stale swag quality evidence
test(v2): record repaired swag quality evidence
docs(v2): complete final refactor acceptance
```

Never combine generated quality evidence with source changes. Never amend a
source commit after evidence has recorded it.
