# V2 Final-Acceptance Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every conformance gap found by the whole-scope v2 review, regenerate the numerically affected SWAG evidence, and finish Task 6.4 with fresh final-commit verification.

**Architecture:** Five focused plans implement the approved remediation in dependency order. Evaluation, LoRA, and inference first establish their local contracts; descriptor-bound artifact ownership then supplies the common safe-loading foundation and recursive semantic verifier; the final plan regenerates evidence and repeats all repository gates and reviews.

**Tech Stack:** Python 3.12.13, MLX, NumPy, SentencePiece, lm-evaluation-harness 0.4.12, Hugging Face datasets, pytest, Ruff, canonical SML JSON/artifact schemas, Git.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

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

- [ ] **Step 1: Confirm the approved planning base**

Run:

```bash
git status --short --branch
git log -2 --oneline
```

Expected: the tracked checkout is clean and `9a8b170` is the committed
remediation design at the tip or an ancestor of the current tip.

- [ ] **Step 2: Execute and review the evaluation plan**

Read the remediation spec and the complete evaluation plan, execute every
checkbox in order, and stop at each commit for requirements and code-quality
review.

Expected: strict evaluation artifacts round-trip complete provider metrics and
task provenance; focused evaluation and package/CLI tests pass.

- [ ] **Step 3: Execute and review the LoRA plan**

Read the complete LoRA plan, execute its RED/GREEN cycles, and review the
numerical formula, key order, compiled state, and exact resume evidence before
continuing.

Expected: scale is static FP32 configuration, adapter leaves alone are
trainable FP32 arrays, configured adapter dropout runs in pure-array training,
and the old SWAG quality evidence is now expected to require replacement.

- [ ] **Step 4: Execute and review the inference plan**

Read the complete inference plan, execute each task, and review both stable
shape domains and the scoring allocation contract.

Expected: short-prompt/long-generation requests use prompt-sized prefill and
capacity-sized storage; scoring produces only token-shaped log probabilities.

- [ ] **Step 5: Execute and review the artifact plan**

Read the complete artifact plan, execute descriptor primitives before consumer
migrations, then build recursive verification only on the migrated owner
loaders.

Expected: verification and consumption share retained descriptors, SWAG owns
its mappings deterministically, and re-signed semantic corruptions fail FULL
CLI verification.

- [ ] **Step 6: Execute the final-acceptance plan**

Run its focused integration checkpoint, commit all repair source, retire the old
SWAG evidence in a dedicated commit, record new evidence from the resulting
clean commit, and perform every final gate.

Expected: Ruff, all 1,104-or-more tests, quality validators, CLI workflows,
integration workflows, repository invariants, and whole-scope review all pass
at the final committed tree.

- [ ] **Step 7: Record completion only after evidence and review**

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
