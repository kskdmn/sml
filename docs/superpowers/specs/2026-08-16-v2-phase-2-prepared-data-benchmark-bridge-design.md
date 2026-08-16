# V2 Phase 2 Prepared-Data Benchmark Bridge Design

## Context

Phase 2 cannot run at commit `a2e7ac4` because the replacement benchmark
adapter resolves `prepared-data` through
`sml.data.pretraining.build_benchmark_workload`, but that owner-side factory is
absent. This is a plan integration gap rather than a loader failure: the mmap
stream exists and passes its correctness suite, while the harness intentionally
reports the metric as unavailable until its production owner enables the
factory.

The original Phase 2 command is also stale. The accepted short protocol is five
warmups and 20 measured units, the runner now requires an explicit predecessor
mapping, and the approved Phase 1 amendment produced no prepared-data
predecessor. Phase 2 therefore uses `{"prepared-data": null}`.

## Goals

- Enable the existing replacement adapter ABI for only the `prepared-data`
  metric.
- Measure the committed `PreparedDataBundle` and `PretrainingBatchStream`
  consumer path, including NumPy-to-MLX transfer and evaluation.
- Build and fully verify all benchmark input artifacts before the timed region.
- Preserve the canonical rows, semantic identities, logical execution order,
  precision annotation, and work count required by the pinned harness.
- Close the stream, mappings, descriptors, producer thread, and temporary input
  tree deterministically after every process, including failures.

## Non-Goals

- No alternate loader, path-based `np.load(..., mmap_mode=...)` hot path, or
  benchmark-only timing shortcut.
- No support for metrics owned by later phases.
- No change to the pinned harness, baseline evidence, canonical workload, gate
  thresholds, or top-level dependency files.
- No timing of artifact creation or FULL startup verification.

## Owner-Side Interface

`sml.data.pretraining` adds:

```python
def build_benchmark_workload(metric: str, canonical_workload: object) -> object:
    ...
```

The function accepts only `prepared-data`; other metric names fail closed. It
returns a runtime implementing the adapter's frozen attributes and methods:

- `verification_level == "full"`
- canonical/native configuration and identity fields required by
  `v2.benchmarks.adapters.replacement`
- `run(units: int) -> float`
- `reset_after_warmup()` and `reset_measured_order()` where needed to restore
  the canonical first-row order
- idempotent cleanup for the stream and temporary artifact tree

The benchmark hook may lazily import benchmark workload helpers inside the
factory. Normal training and data-preparation imports remain independent of the
benchmark package.

## Artifact Construction and Verification

The factory derives the canonical fixed int32 rows from the supplied workload,
then materializes a deterministic immutable pretraining bundle in a private
temporary directory. The bundle uses the real manifest, payload-reference,
canonical JSON, immutable publication, nested tokenizer binding, NPY shard, and
FULL verification APIs. The benchmark tokenizer payload is deterministic and
provides the canonical vocabulary bounds needed by the loader; tokenization is
not part of the prepared-data work unit.

The prepared-data manifest binds:

- the workload sequence length and row width;
- all canonical rows in fixed order;
- uncompressed little-endian int32 C-order NPY shards;
- the product `sml-row-content-v1` identity for those rows;
- deterministic shard layout and epoch seed metadata; and
- copied nested tokenizer manifest/model/vocab references.

Factory return is permitted only after reopening the published bundle at FULL
verification. The bridge computes both identity domains from the same canonical
int32 matrix: the manifest must equal the product `sml-row-content-v1` identity,
while the adapter-facing `canonical_row_identity` must equal the harness's
`sml-pretraining-rows-v1` `canonical_training_rows` identity. These identities
are intentionally not equal because their domains differ. Artifact construction
and verification time is reported as startup verification by the replacement
adapter and remains outside steady-state timing.

## Timed Data Flow

The runtime owns one `PretrainingBatchStream` configured with the canonical
microbatch size, a fixed benchmark epoch seed, bounded prefetch, and the initial
cursor. For every requested unit, the consumer thread:

1. receives one real `BatchEnvelope`;
2. creates an MLX array from `envelope.rows`;
3. releases the envelope in `finally` after the transfer has been initiated;
4. evaluates the MLX array before requesting the next batch; and
5. records the delivered cursor/order without committing training progress.

If an epoch ends, the runtime continues from the next deterministic epoch. A
reset closes the current stream and creates a fresh stream from the initial
cursor, so warmup cannot shift the measured logical order. `run(units)` returns
exactly `float(units)` after all units have synchronized.

MLX is imported lazily inside benchmark runtime construction. The producer
thread remains NumPy-only.

## Failure and Cleanup Semantics

Invalid metric names, workload types/fields, semantic identities, dtype, shape,
row order, token bounds, manifest data, or adapter proofs fail before timed
measurement. Runtime exhaustion before the requested unit count is an error,
not a partial result.

Cleanup is idempotent and closes the stream before deleting the temporary tree.
Failure during construction closes every already-created resource. The runtime
must not rely on garbage collection to stop a producer thread or release an
mmap.

## Tests

Behavior-first tests will prove:

- RED: the availability regression is first changed to require the real owner,
  and fails because the factory is absent;
- the real owner resolves to `ReplacementNativeWorkload` with FULL verification
  and exact canonical/native identity proofs;
- canonical int32 rows are delivered through the real stream and transferred to
  MLX in the exact expected order;
- warmup and measured-order resets restart at the canonical first row;
- multiple units and epoch rollover return the exact work count;
- malformed workloads and unsupported metrics fail closed before producer
  startup; and
- success and injected failure both close threads, mappings, descriptors, and
  temporary storage.

Focused tests run RED before production edits and GREEN afterward. Final
verification runs the complete v2 pytest suite plus Ruff check and format check.

## Phase 2 Gate

After the bridge is reviewed, committed, and the tracked checkout is clean, the
screen uses the accepted protocol:

```bash
uv run python -m v2.benchmarks.runner compare \
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

uv run python -m v2.benchmarks.runner validate-phase \
  --phase 2 \
  --baseline v2/benchmarks/manifests/baseline-3687f8b.json \
  --predecessors '{"prepared-data":null}' \
  --results v2/benchmarks/results/phase-2-loader.json \
  --output v2/benchmarks/results/phase-2.json
```

The environment must match the baseline: AC power, automatic power mode, low
power mode off, nominal thermal state, normal memory pressure, and no competing
GPU workload. A thermal or power rejection aborts this non-resumable comparison.
Only statistical noise triggers the harness's one full retry after its recorded
15-minute cooldown.
