# V2 Remediation Final Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Use superpowers:requesting-code-review for the whole-scope review and superpowers:verification-before-completion before any completion claim. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the complete repaired v2 tree at a committed source revision, replace only the numerically stale SWAG quality evidence, resolve whole-scope review findings, and record final Task 6.4 completion in tracked documentation.

**Architecture:** Final acceptance begins only after all four source plans are committed and the checkout is clean. A focused integration gate freezes the repair source. Historical SWAG evidence is removed in its own commit so the create-only recorder can bind new evidence to a clean exact source commit; pretraining evidence remains unchanged. The evidence commit is subjected to full static, functional, CLI, quality, repository, and architecture review gates before status documentation changes.

**Tech Stack:** Python 3.12.13, MLX/Metal, pytest, Ruff, controlled pretraining and SWAG quality harnesses, unified `sml` CLI, Git.

**Specs:**
- `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`
- `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

## Global Constraints

- Continue in the current checkout; do not create a worktree and do not push.
- Use `uv run`; run every pytest and MLX quality command outside the sandbox.
- Do not edit top-level project files or `uv.lock`.
- Preserve the canonical pretraining manifest/raw/report and both quality fixture sets byte-for-byte.
- Replace exactly the canonical SWAG manifest/raw/report; do not fabricate optional benchmark artifacts.
- Record SWAG evidence only when the tracked checkout is clean and all repair source is committed.
- Never combine production-source changes with generated evidence in one commit.
- If any production source changes after SWAG recording, retire and regenerate all three SWAG evidence files again from the new clean source commit.
- Performance measurement remains optional and is not an acceptance gate.
- Do not mark documentation complete until every final gate and the whole-scope review have no unresolved finding of any severity.

---

## File Structure

**Evidence replaced:**
- `v2/benchmarks/manifests/swag-quality-v1.json`
- `v2/benchmarks/results/swag-quality-v1.jsonl`
- `v2/benchmarks/results/swag-quality-v1.json`

**Evidence preserved:**
- `v2/benchmarks/manifests/pretraining-quality-v1.json`
- `v2/benchmarks/results/pretraining-quality-v1.jsonl`
- `v2/benchmarks/results/pretraining-quality-v1.json`
- `v2/benchmarks/fixtures/pretraining-quality-train-v1.npy`
- `v2/benchmarks/fixtures/pretraining-quality-validation-v1.npy`
- `v2/benchmarks/fixtures/swag-quality-train-v1.npz`
- `v2/benchmarks/fixtures/swag-quality-validation-v1.npz`

**Documentation updated after acceptance:**
- `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`
- `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`
- `docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md`
- `docs/superpowers/plans/2026-08-25-v2-final-acceptance-remediation.md`
- `docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md`

Production/test files may change in this plan only to resolve a newly proven
acceptance or review defect. Such a fix uses a fresh focused RED/GREEN cycle,
its own source commit, and triggers SWAG re-recording.

## Task 1: Freeze the Committed Repair Source with Focused Integration Gates

**Files:**
- Verify: all files changed by the evaluation, LoRA, inference, and artifact plans.
- Do not modify evidence or completion documentation in this task.

- [ ] **Step 1: Confirm all component plans ended at clean commits**

```bash
git status --short --branch
git log --oneline -20
git diff --check
```

Expected: no tracked or untracked source/test changes, all planned component
commits are present, and whitespace validation passes. If a component left
changes, return to that component's tests/review and commit them before
continuing.

- [ ] **Step 2: Run focused unit/equivalence gates**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation.py v2/tests/unit/test_inference.py v2/tests/unit/training/test_lora.py v2/tests/unit/training/test_swag.py v2/tests/unit/data/test_swag.py v2/tests/unit/artifacts -q
uv run pytest v2/tests/equivalence/test_evaluation_equivalence.py v2/tests/equivalence/test_inference_equivalence.py v2/tests/equivalence/test_lora_equivalence.py v2/tests/equivalence/test_swag_equivalence.py -q
```

Expected: every remediation-specific behavior and numerical equivalence test
passes on the committed tree.

- [ ] **Step 3: Run all integration and CLI workflow tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/integration -q
uv run pytest v2/tests/integration/test_cli_workflows.py v2/tests/integration/test_cli_config.py -q
uv run python -m sml --help
```

Expected: all integration workflows pass, CLI subprocess workflows pass, and
unified help lists the supported package commands.

- [ ] **Step 4: Run static gates and confirm a clean source commit**

```bash
uv run ruff check v2
uv run ruff format --check v2
git status --short
git rev-parse HEAD
```

Record the resulting commit as `REPAIRED_SOURCE_HEAD` in the execution notes.
Expected: both Ruff gates pass and the checkout is clean.

## Task 2: Retire Only the Stale Canonical SWAG Evidence

**Files:**
- Delete: `v2/benchmarks/manifests/swag-quality-v1.json`
- Delete: `v2/benchmarks/results/swag-quality-v1.jsonl`
- Delete: `v2/benchmarks/results/swag-quality-v1.json`

- [ ] **Step 1: Prove the current files are the tracked canonical evidence**

```bash
git ls-files --error-unmatch v2/benchmarks/manifests/swag-quality-v1.json v2/benchmarks/results/swag-quality-v1.jsonl v2/benchmarks/results/swag-quality-v1.json
git status --short
```

Expected: exactly the three files are tracked and the checkout is clean.

- [ ] **Step 2: Remove and commit the stale evidence**

```bash
git rm v2/benchmarks/manifests/swag-quality-v1.json v2/benchmarks/results/swag-quality-v1.jsonl v2/benchmarks/results/swag-quality-v1.json
git diff --cached --stat
git commit -m "test(v2): retire stale swag quality evidence"
git status --short
```

Expected: the removal commit changes exactly those three files and the checkout
is clean. This commit is the clean source revision the create-only recorder
must bind; it contains all repair source but no old SWAG evidence.

- [ ] **Step 3: Prove protected evidence was not touched**

```bash
git diff --exit-code HEAD^ -- v2/benchmarks/manifests/pretraining-quality-v1.json v2/benchmarks/results/pretraining-quality-v1.jsonl v2/benchmarks/results/pretraining-quality-v1.json v2/benchmarks/fixtures
```

Expected: no pretraining evidence or fixture difference.

## Task 3: Record and Commit Repaired SWAG Quality Evidence

**Files:**
- Create: `v2/benchmarks/manifests/swag-quality-v1.json`
- Create: `v2/benchmarks/results/swag-quality-v1.jsonl`
- Create: `v2/benchmarks/results/swag-quality-v1.json`

- [ ] **Step 1: Record from the exact clean repair revision**

Run outside the sandbox:

```bash
uv run python -m v2.benchmarks.swag_quality record --steps 256 --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-output v2/benchmarks/results/swag-quality-v1.jsonl --output v2/benchmarks/results/swag-quality-v1.json
```

Expected: the controlled run completes, creates exactly the three absent files,
and reports a passing quality decision. Do not modify source while it runs.

- [ ] **Step 2: Independently validate the new files**

Run outside the sandbox:

```bash
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
```

Expected: `pass`, including source-commit/import-closure, raw journal, controlled
trajectory, and report identity validation.

- [ ] **Step 3: Inspect the evidence binding and scope**

```bash
git status --short
uv run python -c 'import json; from pathlib import Path; print(json.loads(Path("v2/benchmarks/manifests/swag-quality-v1.json").read_text())["source_commit"])'
git rev-parse HEAD
```

Expected: only the three SWAG evidence files are untracked/changed; their
recorded source commit is the clean retirement commit from Task 2, and no
production or protected evidence file changed.

- [ ] **Step 4: Commit generated evidence separately**

```bash
git add v2/benchmarks/manifests/swag-quality-v1.json v2/benchmarks/results/swag-quality-v1.jsonl v2/benchmarks/results/swag-quality-v1.json
git commit -m "test(v2): record repaired swag quality evidence"
git status --short
```

Expected: the evidence commit contains only those three files and the checkout
is clean.

## Task 4: Run Fresh Final-Commit Verification

**Files:**
- Verify: complete tracked v2 tree and repository invariants.

- [ ] **Step 1: Run full Ruff and pytest gates**

Run pytest outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
```

Record the exact test count, duration, and commit in the execution notes.
Expected: both Ruff commands and the complete suite pass with no skip or failure
that weakens an acceptance requirement.

- [ ] **Step 2: Run both independent controlled-quality validators**

Run outside the sandbox:

```bash
uv run python -m v2.benchmarks.quality validate --manifest v2/benchmarks/manifests/pretraining-quality-v1.json --raw-input v2/benchmarks/results/pretraining-quality-v1.jsonl --report v2/benchmarks/results/pretraining-quality-v1.json
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
```

Expected: both print `pass`. Pretraining validates its unchanged recorded
source; SWAG validates the repaired source revision recorded in Task 3.

- [ ] **Step 3: Repeat CLI and integration gates at the evidence commit**

Run outside the sandbox:

```bash
uv run pytest v2/tests/integration -q
uv run pytest v2/tests/integration/test_cli_workflows.py -q
uv run python -m sml --help
```

Expected: every workflow remains green at the evidence commit.

- [ ] **Step 4: Verify clean cutover and protected repository state**

```bash
uv run pytest v2/tests/unit/test_source_contract.py v2/tests/unit/test_package.py -q
git diff --exit-code 4225c54 -- uv.lock
git diff --check
git status --short
find v2/src -maxdepth 1 -type f -name '*.py' -print
rg -n 'spec_from_file_location|sml\._legacy|LEGACY_BRIDGE_EXPORTS|train_sml|infer_sml|evaluate_sml|ft_swag|from config import' v2/src/sml
```

Expected: source/package contract passes; `uv.lock` is unchanged from the audit
base; no flat Python module or bridge string is found; checkout is clean.

## Task 5: Obtain and Resolve a Whole-Scope Architecture Review

**Files:**
- Review: umbrella spec, remediation spec, all five remediation plans, complete source/test/evidence diff since `9a8b170`.
- Modify only if a concrete finding is reproduced.

- [ ] **Step 1: Request a fresh whole-scope review**

Use `superpowers:requesting-code-review`. Give the reviewer both specs, this
plan set, the commit range from `9a8b170` through current HEAD, and the fresh
gate/evidence results. Ask explicitly for Critical, Important, and Minor
findings in:

- evaluation schema closure and lm-eval provenance;
- LoRA FP32 policy, dropout keys, compile, resume, and merge;
- inference shape domains and target-logit allocation;
- descriptor ownership, mutation windows, cleanup, and path fallbacks;
- recursive semantic verification and deterministic result trees;
- evidence binding, clean cutover, and documentation truthfulness.

- [ ] **Step 2: Verify each finding before changing code**

Use `superpowers:receiving-code-review`. Reproduce each reported issue against
the committed tree and classify it with file/line evidence. Do not implement a
suggestion that does not satisfy the specs or that duplicates an existing
guarantee.

- [ ] **Step 3: Resolve confirmed findings with TDD**

For every confirmed finding, write the smallest failing lasting regression,
run it RED outside the sandbox, implement the scoped fix, run focused plus
affected integration tests, Ruff the touched files, and commit source/tests
without evidence.

If any `v2/src` file changes, repeat Tasks 2 through 4 so SWAG evidence binds
the new final source commit. If only tests or documentation change, rerun Task 4
and both validators; regenerate only when the quality harness says its bound
production closure changed.

- [ ] **Step 4: Re-review until no findings remain**

Request review of the updated commit range and provide verification evidence.
Expected: no unresolved Critical, Important, or Minor finding. Record the
review commit and disposition in the handoff update.

## Task 6: Close Status Documentation and Verify the Handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`
- Modify: `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`
- Modify: `docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md`
- Modify: `docs/superpowers/plans/2026-08-25-v2-final-acceptance-remediation.md`
- Modify: `docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md`

- [ ] **Step 1: Update exact completion facts**

Mark Task 6.4, the remediation, Part 2, and the umbrella refactor complete.
Record exact source/evidence/review/documentation commit IDs, full pytest count,
Ruff results, integration/CLI results, both quality decisions, unchanged
`uv.lock`, clean-cutover checks, and the no-findings review. Remove stale “next
task” prose while preserving historical records as history.

- [ ] **Step 2: Cross-check documentation consistency**

```bash
rg -n 'Task 6\.4 is next|next: Task 6\.4|Task 6\.4 remains incomplete|Remaining Part 2 Tasks' docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md docs/superpowers/plans/2026-08-25-v2-final-acceptance-remediation.md docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md
git diff --check
git diff -- docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md docs/superpowers/plans/2026-08-25-v2-final-acceptance-remediation.md docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md
```

Expected: no active stale status remains; historical descriptions are clearly
past-tense; whitespace validation passes.

- [ ] **Step 3: Commit completion documentation**

```bash
git add docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-2.md docs/superpowers/plans/2026-08-25-v2-final-acceptance-remediation.md docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md
git commit -m "docs(v2): complete final refactor acceptance"
```

- [ ] **Step 4: Run verification-before-completion at documentation HEAD**

Use `superpowers:verification-before-completion`, then run:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run python -m v2.benchmarks.quality validate --manifest v2/benchmarks/manifests/pretraining-quality-v1.json --raw-input v2/benchmarks/results/pretraining-quality-v1.jsonl --report v2/benchmarks/results/pretraining-quality-v1.json
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
uv run python -m sml --help
git diff --exit-code 4225c54 -- uv.lock
git diff --check
git status --short --branch
```

Expected: all lightweight gates and both validators pass, `uv.lock` is
unchanged, and the tracked checkout is clean at the final documentation commit.
