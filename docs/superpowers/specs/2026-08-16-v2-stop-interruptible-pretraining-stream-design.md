# V2 Stop-Interruptible Pretraining Stream Design

## Context

The first valid Phase 2 prepared-data comparison produced ten accepted trials
under nominal power, thermal, memory, and GPU conditions, but validation failed
with a median candidate/reference throughput ratio of `0.0260086273` against the
required `0.97` floor. Dispersion was low (`0.0003729627`), so the result is a
stable implementation effect rather than statistical noise.

Focused timing separated the candidate's measured call into two parts:

- transferring and synchronizing 20 real batches: approximately `0.64 ms` after
  the first diagnostic sample;
- closing the real `PretrainingBatchStream`: approximately `50-55 ms`.

`PretrainingBatchStream.close()` currently sets its stop flag and joins the
producer before draining the bounded queue. When that queue is full, the
producer can already be blocked in `queue.put(..., timeout=0.05)`, so the join
waits for almost the full polling timeout. The benchmark runtime is required to
close its taken stream before `run()` returns, placing this fixed shutdown delay
inside the measured interval.

## Goal

Make `PretrainingBatchStream.close()` stop-interruptible when its bounded queue
is full, without changing public APIs, cursor semantics, envelope ownership,
the benchmark timing boundary, or the requirement that every benchmark
`run()` closes its taken stream before returning.

## Chosen Design

Shutdown will keep the existing one-owner lifecycle but change its ordering:

1. Under the state condition, the first closer marks the stream as closing;
   concurrent closers continue to wait for the same terminal closed state.
2. The closer sets the stop event and stops the staging pool.
3. Before joining the producer, the closer takes the consumer lock and drains
   queued envelopes, releasing each lease. This creates capacity immediately
   if the producer is blocked in a full-queue `put()`.
4. The producer completes that in-flight `put()`, observes the stop event on its
   next loop boundary, and exits without waiting for the 50 ms timeout.
5. After joining the producer, the closer drains the queue again. This second
   drain owns any envelope or terminal marker published during the wake-up
   race.
6. The closer releases pending and still-owned envelopes, closes mmap/file/root
   resources, publishes the closed state, and wakes concurrent closers exactly
   as today.

The drain logic will be a private helper used before and after `join()` so the
release rules have one implementation. It will treat only `BatchEnvelope`
items as leased resources; producer-failure and stop markers require no
release.

## Concurrency and Error Semantics

- No producer thread may survive `close()`.
- Every staging-pool lease must be released exactly once, including an envelope
  queued during the pre-join drain/join race.
- A consumer already inside `__next__()` is serialized by the existing
  consumer lock. Once shutdown begins, no new consumer delivery may outlive
  final resource closure.
- Repeated and concurrent `close()` calls remain idempotent.
- Existing producer exceptions, abandoned envelopes, pending epoch envelopes,
  cursor commit rules, and bundle resource ownership remain unchanged.
- The producer's 50 ms queue polling interval remains available for ordinary
  operation; the fix removes dependence on that interval during shutdown
  rather than lowering it or adding a busy loop.

## Rejected Alternatives

### Move stream close outside benchmark timing

This would make the comparison faster but would revise the approved lifecycle
contract and make the reference and candidate teardown boundaries less
auditable. It also leaves production shutdown latency unfixed.

### Reduce the queue timeout

A shorter timeout only reduces the symptom. It retains polling-dependent
shutdown, introduces unnecessary wakeups, and makes latency depend on an
arbitrary constant.

### Add a benchmark-only fast path

This would stop measuring the real production stream and is prohibited by the
Phase 2 design.

## Testing

The production change will be test-driven.

The RED regression will construct a real stream around a deterministically
coordinated bounded queue. The queue will expose when the producer attempts a
put while full and will release that blocked put only after the consumer-side
drain creates capacity. The test will call `close()` and assert that it drains
before joining, terminates the producer, releases all envelopes/leases, and
leaves the stream idempotently closed. With the current join-before-drain
ordering, the test must fail deterministically rather than relying on a loose
wall-clock threshold.

Existing shutdown, cursor, abandoned-envelope, producer-failure, and workflow
tests remain unchanged and must pass. Final verification is:

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py \
  v2/tests/integration/test_pretraining_data_workflow.py -v
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
```

Every pytest invocation runs outside the sandbox so MLX/Metal remains
available.

After tests and an independent code review pass, a focused diagnostic will
again separate 20-batch work from shutdown. Phase 2 may be rerun only if stream
shutdown no longer waits for the 50 ms queue timeout. The failed comparison is
diagnostic evidence only and must never be staged or accepted.

## Success Criteria

- The deterministic RED test fails on join-before-drain and passes on the new
  stop-interruptible ordering.
- Full-queue close has no dependency on the 50 ms `queue.put()` timeout.
- No stream thread, envelope, staging lease, mmap, payload file, or artifact
  root survives closure.
- All existing v2 tests and Ruff gates pass without warnings.
- Independent review finds no lifecycle, race, or scope defect.
- The benchmark runtime continues to close every taken stream before `run()`
  returns and continues to use the real production loader and MLX consumer
  path.
- Only after those gates pass is the Phase 2 comparison rerun from zero under
  canonical `TMPDIR=/private/tmp` and the existing acceptance thresholds.
