# V2 Stop-Interruptible Pretraining Stream Implementation Plan

**Status update (2026-08-22):** Task 1 is complete and functionally accepted.
Task 2 is superseded as required work; its commands are retained only as an
optional reproducible performance diagnostic. The unchecked Task 1 boxes
preserve its historical TDD procedure.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the fixed 50 ms full-queue shutdown delay from the real pretraining stream and prove the lifecycle behavior through deterministic tests. A Phase 2 timing rerun is optional.

**Architecture:** `PretrainingBatchStream.close()` will release queued ownership before joining its producer, creating capacity that wakes an in-flight full-queue put immediately, and will drain again after the join to own the wake-up race. The public API and benchmark lifecycle remain unchanged: measured `run()` still transfers real batches and closes its taken production stream before returning.

**Tech Stack:** Python 3.12.13, `queue.Queue`, Python threads/events, NumPy, MLX, pytest, Ruff, existing v2 benchmark harness, Git.

## Global Constraints

- Work on `main`; the user explicitly authorized continued main-branch work.
- Do not modify `v2/benchmarks/**` code, manifests, thresholds, schemas, adapters, or harness behavior. Only accepted generated result JSON files may be added there.
- Do not modify top-level project files, `pyproject.toml`, or `uv.lock`.
- Do not push any commit; pushing requires a separate user instruction after final review.
- Do not move stream teardown outside `runtime.run()` and do not add a benchmark-only fast path, test switch, lower queue timeout, or busy polling.
- `PretrainingBatchStream.close()` remains idempotent and must not return with a live producer, queued/pending/owned envelope, staging lease, mmap, file descriptor, or artifact root.
- Producer failures, cursor commits, abandoned envelopes, pending epoch envelopes, consumer wake-up, and concurrent close semantics remain unchanged.
- The failed loader report with identity `sha256:7626e22f39fbe49140ced49a6af5a5fa0c9669802f59149dbf9071b437b4448b` is diagnostic evidence only. It must never be staged or committed as accepted Phase 2 evidence.
- If the optional Phase 2 diagnostic is run, its protocol remains exactly five pairs, five warmups, 20 measured units, 10,000 bootstrap resamples, minimum ratio `0.97`, maximum dispersion `0.02`, report-only lower bound, and predecessor mapping `{"prepared-data": null}`.
- Run every `uv run pytest` and MLX diagnostic command outside the sandbox.
- Before finishing any v2 change, run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests`.

## File Structure

- Modify `v2/src/sml/data/pretraining.py`: add one private queue-drain helper and reorder stream shutdown around producer join.
- Modify `v2/tests/integration/test_pretraining_data_workflow.py`: add a deterministic full-queue producer/close ordering regression using real stream ownership.
- Preserve the failed comparison only in this plan's ignored `.superpowers/sdd` workspace before a fresh run.
- Optional: create `v2/benchmarks/results/phase-2-loader.json` as fresh diagnostic comparison evidence.
- Optional: create `v2/benchmarks/results/phase-2.json` as its independently validated diagnostic report.

---

### Task 1: Make Full-Queue Stream Shutdown Stop-Interruptible

**Files:**
- Modify: `v2/src/sml/data/pretraining.py:817-875`
- Modify: `v2/tests/integration/test_pretraining_data_workflow.py:743-778`

**Interfaces:**
- Consumes: `PretrainingBatchStream.close() -> None`, `BatchEnvelope.release() -> None`, the existing `_consumer_lock`, `_queue`, `_pending_envelope`, `_owned_envelopes`, `_stop`, `_pool`, `_producer`, and `_close_open_resources()` ownership model.
- Produces: unchanged public close behavior with private `_drain_consumer_items() -> None`, called before and after producer join.

- [ ] **Step 1: Add the deterministic full-queue RED regression**

In `v2/tests/integration/test_pretraining_data_workflow.py`, add `import queue`
and `import sml.data.pretraining as pretraining_module`, then add this test-local
queue immediately above the close regression:

```python
class _CloseCoordinatedQueue(queue.Queue):
    def __init__(self, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.full_put_entered = threading.Event()
        self.capacity_drained = threading.Event()

    def put(self, item, block=True, timeout=None):
        if self.full():
            self.full_put_entered.set()
            self.capacity_drained.wait(2)
        return super().put(item, block=block, timeout=timeout)

    def get_nowait(self):
        item = super().get_nowait()
        self.capacity_drained.set()
        return item
```

Add a real-stream regression that monkeypatches only the queue constructor:

```python
def test_stream_close_drains_full_queue_before_join(prepared_bundle, monkeypatch):
    monkeypatch.setattr(pretraining_module.queue, "Queue", _CloseCoordinatedQueue)
    stream = PretrainingBatchStream(
        prepared_bundle,
        batch_size=3,
        seed=5,
        prefetch_depth=1,
        cursor=PretrainingCursor.initial(),
    )
    queue_instance = stream._queue
    assert isinstance(queue_instance, _CloseCoordinatedQueue)
    assert queue_instance.full_put_entered.wait(2)

    closed = threading.Event()

    def close_stream():
        stream.close()
        closed.set()

    closer = threading.Thread(target=close_stream, daemon=True)
    closer.start()
    closed_before_force_release = closed.wait(0.25)
    if not closed_before_force_release:
        queue_instance.capacity_drained.set()
    closer.join(timeout=2)

    assert closed_before_force_release
    assert not closer.is_alive()
    assert stream._producer is None or not stream._producer.is_alive()
    assert stream._queue.empty()
    assert stream._owned_envelopes == {}
    stream.close()
```

The `capacity_drained` emergency release exists only to let the current RED implementation terminate cleanly after recording the ordering failure. It is not a production switch or a wall-clock performance assertion.

- [ ] **Step 2: Run the exact RED test outside the sandbox**

```bash
uv run pytest \
  v2/tests/integration/test_pretraining_data_workflow.py::test_stream_close_drains_full_queue_before_join \
  -v
```

Expected: FAIL at `assert closed_before_force_release` because current `close()` joins the producer before any `get_nowait()` drain can signal `capacity_drained`.

- [ ] **Step 3: Add one private ownership drain and reorder close**

In `PretrainingBatchStream`, add:

```python
def _drain_consumer_items(self) -> None:
    pending = self._pending_envelope
    self._pending_envelope = None
    if pending is not None:
        pending.release()
    while True:
        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(item, BatchEnvelope):
            item.release()
```

Keep the existing state-condition owner election, stop event, and pool stop. Replace join-before-drain ordering with:

```python
with self._consumer_lock:
    self._drain_consumer_items()

producer = self._producer
if producer is not None and producer is not threading.current_thread():
    producer.join()

with self._consumer_lock:
    self._drain_consumer_items()
```

Then retain the existing owned-envelope release, `_close_open_resources()`, closed-state publication, and waiter notification. Do not change `_put()` or its 50 ms ordinary-operation timeout.

- [ ] **Step 4: Run focused GREEN and shutdown regressions outside the sandbox**

```bash
uv run pytest \
  v2/tests/integration/test_pretraining_data_workflow.py::test_stream_close_drains_full_queue_before_join \
  v2/tests/integration/test_pretraining_data_workflow.py::test_stream_close_wakes_full_queue_and_abandoned_envelope \
  v2/tests/integration/test_pretraining_data_workflow.py::test_stream_close_wakes_consumer_waiting_on_empty_queue \
  v2/tests/integration/test_pretraining_data_workflow.py::test_stream_producer_exception_propagates_as_focused_data_error \
  -v
```

Expected: all four pass without warnings, deadlock, or surviving producer.

- [ ] **Step 5: Run the complete loader suites outside the sandbox**

```bash
uv run pytest \
  v2/tests/unit/data/test_pretraining.py \
  v2/tests/integration/test_pretraining_data_workflow.py -v
```

Expected: all tests pass with no leaked `sml-pretraining-prefetch` thread.

- [ ] **Step 6: Re-run the focused work/close diagnostic outside the sandbox**

Run five fresh real runtimes, consume and synchronize 20 batches, and time `stream.close()` separately using the same diagnostic recorded in the design. Expected: no close sample clusters near the 50 ms queue timeout; every producer is dead and every runtime is closed after its sample. This is diagnostic evidence, not a permanent performance assertion in pytest.

- [ ] **Step 7: Run final v2 verification and commit**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git diff --check
git diff --exit-code -- uv.lock
git status --short
```

Expected: Ruff and all v2 tests pass; the failed Phase 2 loader report remains untracked; only the production file and integration test are staged for this commit.

```bash
git add \
  v2/src/sml/data/pretraining.py \
  v2/tests/integration/test_pretraining_data_workflow.py
git diff --cached --check
git commit -m "perf(v2): unblock pretraining stream shutdown"
```

Generate a task review package from the recorded execution base immediately
after this plan commit through the implementation commit. Review must cover
consumer/producer races, exact-once lease release, pending and queued
ownership, concurrent/idempotent close, unchanged cursor/failure semantics,
and absence of benchmark-specific behavior.

---

### Task 2: Optional Phase 2 Rerun (Superseded as Required Work)

Do not execute this task to unblock the refactor. The steps below are retained
only to reproduce the historical diagnostic protocol. Any missing, failed,
noisy, or thermally rejected result does not block Phase 3 and need not be
committed.

**Files:**
- Preserve: `.superpowers/sdd/2026-08-16-v2-stop-interruptible-pretraining-stream/failed-phase-2-loader-7626e22f.json`
- Optional create: `v2/benchmarks/results/phase-2-loader.json`
- Optional create: `v2/benchmarks/results/phase-2.json`

**Interfaces:**
- Consumes: the reviewed Task 1 commit, pinned baseline `v2/benchmarks/manifests/baseline-3687f8b.json`, canonical `TMPDIR=/private/tmp`, and predecessor mapping `{"prepared-data": null}`.
- Produces only when requested: a fresh diagnostic comparison report with ten valid trials and an independently validated Phase 2 report.

- [ ] **Step 1: Preserve rejected evidence and verify a clean candidate**

Confirm the current untracked loader report has identity `sha256:7626e22f39fbe49140ced49a6af5a5fa0c9669802f59149dbf9071b437b4448b` and decision `fail`, then move it to the ignored preservation path above. Do not stage it. Confirm `v2/benchmarks/results/phase-2-loader.json` and `v2/benchmarks/results/phase-2.json` are absent.

Run outside the sandbox:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
git diff --exit-code -- uv.lock
```

Expected: all verification passes and tracked/staged status is empty.

- [ ] **Step 2: Require a stable nominal launch environment**

Using the benchmark harness's `collect_environment()` under `env TMPDIR=/private/tmp`, require 30 continuous seconds of AC connected, automatic power mode, low-power mode off, nominal thermal/raw `0`, normal memory pressure, and no competing GPU workload. Do not launch if the gate is not achieved within five minutes.

- [ ] **Step 3: Run the exact fresh comparison outside the sandbox**

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner compare \
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

The run is non-resumable. Do not salvage trials or manually retry. Permit only the harness-owned statistical-noise retry and cooldown. Any environment rejection or final `fail` decision blocks acceptance.

- [ ] **Step 4: Validate Phase 2 independently**

```bash
env TMPDIR=/private/tmp uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json \
  --output v2/benchmarks/results/phase-2.json
```

Expected: exit zero; median ratio is at least `0.97`, dispersion is at most `0.02`, every evidence/canonical proof validates, and predecessor set is empty.

- [ ] **Step 5: Inspect and commit only accepted evidence**

Parse both JSON documents and print their identities, prepared-data baseline decisions, median ratio, dispersion, all trial thermal states/raw values, and retry/cooldown status. Confirm exactly the two fresh result files are untracked and both decisions are `pass`.

```bash
git add \
  v2/benchmarks/results/phase-2-loader.json \
  v2/benchmarks/results/phase-2.json
git diff --cached --check
git commit -m "bench(v2): accept artifacts and prepared-data phase"
```

Do not stage or commit either file if comparison or validation fails.

- [ ] **Step 6: Verify the accepted commit**

```bash
git show --check --stat --oneline --summary HEAD
git status --short
git diff --exit-code HEAD^ -- uv.lock
```

Expected: exactly two accepted reports are committed, the tracked worktree is clean, and `uv.lock` is unchanged. Record both task commits, test count, report identities, median ratio, dispersion, all thermal states, and retry/cooldown status in this plan's ledger before task review and whole-branch review.
