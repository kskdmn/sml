# V2 Prepared-Data 100-Unit Protocol and Baseline Design

**Status:** Harness implemented; evidence capture optional as of 2026-08-22

**Date:** 2026-08-16

## Context and Correction

The stop-interruptible stream fix made prepared-data throughput roughly twice
the current pinned baseline, but two 20-unit comparison attempts failed the
unchanged dispersion ceiling. Their measured regions were only about `0.7 ms`
for the candidate and `1.4 ms` for the reference, so fixed scheduling and timer
variation remained material.

The first follow-up design treated `--measure 100` as an ordinary comparison
override. A clean execution proved that assumption false before any trial ran:

```text
ValueError: comparison protocol has invalid measured_units
```

Root-cause tracing established three binding facts:

1. comparison validation requires `DEFAULT_MEASURED_UNITS == 20`;
2. child measurement ignores the supplied global CLI count and derives its
   actual count from the canonical metric work unit; and
3. the pinned baseline's canonical workload, harness identity, session, raw
   trials, and manifest all encode the current per-metric count map.

Prepared-data at 100 is therefore a new versioned benchmark protocol. It needs
harness changes and a fresh baseline; old trials cannot be reused.

## Goal

Provide an optional protocol that can measure prepared-data with 100 real
microbatches per trial while keeping the full 1,024-token workload and every
other metric's current count. The protocol no longer gates refactor progress;
functional tests and runnable workflows are the required acceptance evidence.

## Protocol Model

The protocol keeps the global default and adds one explicit prepared-data
count:

```text
DEFAULT_MEASURED_UNITS = 20
PREPARED_DATA_MEASURED_UNITS = 100
```

The canonical per-metric map becomes:

| Metric | Measured units |
| --- | ---: |
| `prepared-data` | 100 |
| `pretraining-compute` | 20 |
| `pretraining-end-to-end` | 20 |
| `swag-end-to-end` | 20 |
| `inference-prefill` | 32 fixed requests |
| `inference-decode` | 32 fixed requests |
| `checkpoint-pause` | 20 |
| `compile-cold-start` | 1 |
| `peak-metal-memory` | 1 optimizer step |

Warmup remains 5 for every non-compile metric and 0 for cold compile.
Prepared-data still means microbatch size 1, sequence length 1,024, canonical
row order, real native loader consumption, consumer-side MLX transfer and
evaluation, and stream closure before the measured call returns.

The canonical workload, protocol records, journal session, baseline manifest,
raw trials, comparison reports, and replay commands all bind both the global
default `20` and prepared-data count `100`.

## Versioning and Compatibility

The existing baseline is version 1 and remains bound to prepared-data 20. The
new baseline is version 2 and is bound to prepared-data 100. Version 2 uses the
identity domain `sml-performance-baseline-v2` and adds
`prepared_data_measured_units` to its exact protocol field set.

Current validators dispatch by baseline version:

- version 1 reconstructs the legacy canonical workload with prepared-data 20,
  requires the old protocol field set, and preserves validation of the
  existing committed baseline;
- version 2 reconstructs the new canonical workload with prepared-data 100 and
  requires the new explicit protocol field.

Comparison and phase validation derive the expected prepared-data count and
protocol shape from the supplied baseline version. Old reports are never
silently upgraded, and a report or raw trial cannot mix version-1 and
version-2 workload, harness, protocol, or evidence identities.

## CLI and Runtime Data Flow

`record-baseline` and `compare` receive an explicit option:

```text
--prepared-data-measure 100
```

The existing `--measure 20` remains the immutable global default for metrics
without a dedicated count. New baseline capture requires the explicit value
100. Comparison resolves the expected prepared-data count from the supplied
baseline: omitting the new option infers that exact value for compatibility,
while supplying it requires an exact match. Canonical replay commands always
write it explicitly. Supplying any other global or prepared-data value fails
before process launch.

The controller constructs one canonical workload with the exact per-metric
map. For every trial it resolves the selected metric's canonical count and
passes that count to the child process. The child reconstructs the same
canonical workload, verifies that its received count equals the selected work
unit, and then passes that exact value to `measure_native_process()` and the
adapter. It must no longer silently ignore a mismatched child count.

Raw-trial validation continues to compare `trial.measured_units` to the
canonical metric work unit. Prepared-data trials with 20 or any value other
than 100 fail; unchanged metrics fail if they depart from their existing
counts.

## Optional Baseline Versioning and Capture

The existing baseline remains immutable and continues to support evidence
created under the 20-unit prepared-data protocol:

- `v2/benchmarks/manifests/baseline-3687f8b.json`
- `v2/benchmarks/results/baseline-3687f8b.jsonl`

If a new comparison is desired, the protocol publishes separate artifacts:

- `v2/benchmarks/manifests/baseline-3687f8b-prepared100.json`
- `v2/benchmarks/results/baseline-3687f8b-prepared100.jsonl`

Changing the harness and canonical workload changes their identities. Such a
baseline must capture all 45 raw trials—five reference-side trials
for each of nine metrics—using the pinned legacy source commit
`3687f8b3214a44c675ae67af52e4997762f6c634` and the newly committed harness.
No raw trial or journal slot from the old baseline is copied.

Capture uses a new external state directory under `/private/tmp`, durable
journaling, atomic trial publication, and the existing retry/cooldown rules.
It may resume only its own compatible journal after interruption or an
environment rejection. AC, automatic power mode, Low Power Mode off, nominal
thermal/raw `0`, normal memory pressure, no competing GPU workload, clean
checkout, software, and identity gates remain fail-closed.

The harness implementation commit must exist and pass independent review
before baseline capture starts. The new baseline manifest and JSONL are
independently validated before their separate evidence commit.

## Optional Phase 2 Comparison

If diagnostics are requested and the new baseline validates, the comparison starts from zero against
`baseline-3687f8b-prepared100.json`. The exact screen profile remains:

- prepared-data only;
- 5 pairs;
- 5 warmup units;
- prepared-data measured units 100;
- 10,000 bootstrap resamples;
- median candidate/reference ratio at least `0.97`;
- ratio MAD no greater than `0.02`;
- lower confidence bound report-only; and
- predecessors `{"prepared-data": null}`.

The comparison is non-resumable. Only the harness may perform its one
statistical-noise retry and cooldown. Independent Phase 2 validation runs only
after a passing comparison. Exactly `phase-2-loader.json` and `phase-2.json`
may enter an optional diagnostic evidence commit.

## Evidence Isolation

The rejected 20-unit comparison is preserved at:

```text
.superpowers/sdd/2026-08-16-v2-prepared-data-100-unit-measurement/
failed-phase-2-loader-too-noisy-1b19e607.json
```

It remains diagnostic-only and unstaged. The failed no-trial 100-unit CLI
attempt produced no public report. Neither is merged into the new baseline or
Phase 2 evidence.

## Failure Handling for Optional Evidence

- Harness or test failure blocks only the optional baseline capture.
- A non-nominal live gate blocks launch.
- Baseline environment rejection preserves its compatible journal and resumes
  only under the baseline-capture protocol's recovery rules.
- Baseline validation failure blocks publication and commit.
- Comparison rejection or `too-noisy` after the harness retry blocks only
  validation, staging, and commit of that diagnostic result.
- Phase 2 validation failure rejects only the diagnostic evidence.
- No threshold is relaxed and no result is edited, salvaged, or manually
  combined.
- No evidence outcome blocks Phase 3 or later refactor work.
- Nothing is pushed automatically.

## Rejected Alternatives

### Set the global measured default to 100

This would also lengthen expensive training and checkpoint metrics fivefold,
recreating the sustained thermal problem that motivated the 20-unit default.

### Add a prepared-data-only baseline schema

This avoids full recapture but creates a second lineage/validation model and
weakens the single canonical baseline contract. The existing full baseline
format is retained instead.

### Bypass protocol validation or edit the report

The child would still run 20 units, and the result would falsely claim 100.
This is invalid evidence.

### Reuse unchanged old baseline trials

Those trials name the old harness and workload identities. Mixing them with
new evidence would break the manifest's single-protocol guarantee.

## Tests and Review

Implementation is test-driven. Tests must prove:

- only prepared-data changes from 20 to 100 in the canonical per-metric map;
- all model, loader, sequence, optimizer, precision, data, and other metric
  fields remain identical;
- workload and protocol identities change and bind the prepared-data count;
- version-1 baseline validation reconstructs prepared-data 20 while version-2
  validation reconstructs prepared-data 100;
- new baseline and comparison replay commands require global 20 and explicitly
  record prepared-data 100;
- parent process commands pass each metric's canonical count to children;
- children reject a mismatched supplied count before measurement;
- prepared-data adapters receive 100 while every other adapter receives its
  unchanged count;
- raw trials, journals, manifests, comparisons, phase validation, and final
  validation reject old or tampered count combinations;
- old baseline artifacts still validate under their recorded version without
  being accepted as the new protocol;
- thermal recovery, post-exit evidence, locking, atomicity, and resume behavior
  remain unchanged; and
- Ruff and the full v2 pytest suite pass without warnings.

The harness commit receives independent review. If optional baseline or Phase 2
evidence is later produced, each evidence commit receives its own review and a
final evidence review checks the lineage from protocol constants through that
diagnostic report.

## Success Criteria

- The new harness identity-binds prepared-data at 100 and all other metric
  counts remain unchanged.
- Version-1 and version-2 protocol validation, parent/child count propagation,
  tamper rejection, and replay behavior pass the automated test suite.
- If optional evidence is produced, every new prepared-data baseline and
  comparison raw trial records and runs exactly 100 units with the full
  1,024-token workload, and it is committed only after its own validators pass.
- Old baseline and rejected evidence remain preserved and unmodified.
- `uv.lock` and production model/training code remain unchanged.
