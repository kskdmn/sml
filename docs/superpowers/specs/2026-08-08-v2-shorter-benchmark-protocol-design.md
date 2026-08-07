# V2 Shorter Benchmark Protocol Design

**Status:** Approved design; implementation planned

**Date:** 2026-08-08

## Purpose

The current v2 protocol runs 20 warmup units and 100 measured units for most
metrics. On the 24 GiB Apple M5 baseline machine, `pretraining-compute` takes
about four minutes per process. Repeated attempts reached a steady-state Metal
peak of 7,875,602,848 bytes, ended with thermal state `fair`, and frequently
ended with macOS memory pressure `warning`. The strict environment validator
correctly refused those trials, but the sustained duration makes a valid
45-trial baseline impractical on this machine.

This change shortens repeated execution while preserving the workload shape.
It keeps the full 1,024-token training sequence and all model, optimizer, data,
precision, synchronization, environment, pairing, and statistical semantics.

## Goals

- Reduce sustained training-process duration enough to make nominal thermal
  completion realistic on the baseline machine.
- Preserve the full model and 1,024-token workload that the v2 refactor is
  intended to optimize.
- Use one immutable protocol for the pinned baseline and every later candidate
  comparison.
- Keep five baseline/screen process pairs, ten final-comparison pairs, and all
  existing fail-closed acceptance gates.
- Make protocol counts explicit in canonical workload identity, raw trials,
  sessions, manifests, and canonical commands.

## Non-Goals

- Do not change model dimensions, sequence lengths, microbatch sizes, gradient
  accumulation, data rows, request sets, precision policies, or optimizer
  semantics.
- Do not relax thermal, memory-pressure, AC-power, power-mode, Low Power Mode,
  competing-workload, clean-checkout, identity, or software validation.
- Do not change the five-pair baseline/screen or ten-pair final statistical
  profiles, bootstrap settings, dispersion limits, or acceptance ratios.
- Do not add adaptive unit counts, early stopping, or performance-dependent
  retries.
- Do not migrate evidence from an older protocol into the new protocol.

## Protocol

The shared defaults become:

- compilation passes: 1, unchanged;
- warmup units: 5 instead of 20 for every metric except
  `compile-cold-start`, which remains 0;
- default measured units: 20 instead of 100;
- baseline and screen process pairs: 5, unchanged; and
- final-comparison process pairs: 10, unchanged.

Canonical per-metric measured units are:

| Metric | Measured units |
| --- | ---: |
| `prepared-data` | 20 |
| `pretraining-compute` | 20 |
| `pretraining-end-to-end` | 20 |
| `swag-end-to-end` | 20 |
| `inference-prefill` | 32 fixed requests |
| `inference-decode` | 32 fixed requests |
| `checkpoint-pause` | 20 |
| `compile-cold-start` | 1 |
| `peak-metal-memory` | 1 optimizer step |

The peak-memory metric performs the same compilation pass and five steady-state
warmup units as other non-compile metrics. It then resets the Metal peak counter
and measures exactly one complete optimizer step. This matches its existing
work-unit definition and avoids sustaining another 20-step training loop merely
to observe a maximum.

`record-baseline`, screen comparisons, and final comparisons all use these
counts. CLI defaults become `--warmup 5` and `--measure 20`. Commands that
provide different global values fail the same immutable-protocol checks used
today. Per-metric fixed counts remain canonical-workload data rather than CLI
overrides.

## Preserved Workload Shape

The following values remain unchanged:

- 12 transformer layers, hidden size 768, and the complete existing model
  configuration;
- training sequence length 1,024;
- microbatch size 1 and gradient accumulation 8;
- BF16 legacy parameter, moment, and compute policies and the replacement
  precision policy;
- canonical row count, order, identities, and prepared representations;
- SWAG sequence length 256 and fixed 128-example set;
- inference prompt length, decode chunk, and fixed 32-request set;
- synchronization immediately before and after every timed region; and
- fresh native workload and fresh process boundaries.

Reducing unit counts changes sampling duration, not the operation being timed or
the amount of work represented by one unit. Throughput remains normalized by
the actual work count returned by the adapter.

## Runtime and Data Flow

Each child process continues to:

1. construct the canonical workload and native runtime;
2. collect the start environment;
3. execute one untimed compilation unit;
4. execute five synchronized warmup units, except for cold compile;
5. reset the Metal peak counter;
6. execute the metric's canonical measured-unit count;
7. synchronize and collect elapsed time, work count, and peak memory;
8. collect the end environment; and
9. atomically publish the complete raw trial.

No cache clearing, artificial pause, or environment sampling is inserted inside
the timed region. Thermal recovery and crash-resumable journaling remain outside
the child process and retain their existing behavior.

## Identity and Compatibility

The canonical workload document records warmup 5, default measurement 20, and
the complete per-metric measured-unit map. Those changes alter the canonical
workload identity and the harness content identity.

All prior baseline journals, including the retained failed attempts, remain
diagnostic evidence only. They cannot resume under the new harness because
their session, harness, workload, and protocol identities differ. The first
shorter-protocol baseline must use a new empty external state directory. No
accepted prepared-data slots are copied from older journals.

The final manifest and canonical replay command record the new global values.
Candidate comparisons must load a baseline produced by the same protocol and
must reject old raw trials or manifests with 20 warmups or 100 default measured
units.

## Validation and Failure Handling

Raw-trial validation requires warmup 5 for every non-compile metric, warmup 0
for cold compile, and the exact per-metric measured count in the table above.
Session and manifest validation require the new immutable global defaults and
canonical workload identity.

All existing environment behavior is retained. A non-nominal thermal trial is
rejected and retried only after five continuous nominal minutes. A non-thermal
environment violation still stops the invocation and preserves the journal.
The shorter protocol is intended to reduce exposure to those states; it does
not make invalid evidence acceptable.

If the shorter protocol still consistently produces memory-pressure warnings,
that is evidence for a separate recovery or memory-lifecycle design. This
change must not silently weaken the validator or shorten the 1,024-token
workload further.

## Statistical Impact

Five independent paired processes remain the baseline and screen evidence; the
final profile retains ten pairs. Twenty measured optimizer steps are expected
to retain a measurement region of tens of seconds, while reducing the current
four-minute sustained load by about fivefold. The existing within-pair
normalization, whole-pair bootstrap, and dispersion gates detect inadequate
stability.

The implementation does not loosen a dispersion gate to compensate for shorter
trials. If real comparisons become persistently noisy, the protocol must be
reconsidered explicitly rather than selecting longer or shorter trials based on
their measured performance.

## Tests

Tests must prove:

- the canonical workload preserves every model, sequence, optimizer, data, and
  precision field while changing only protocol counts;
- the exact per-metric measured-unit table is present and identity-bound;
- CLI defaults and immutable-protocol validation require warmup 5 and global
  measure 20;
- the measurement layer passes five warmups and each metric's exact measured
  count to its adapter;
- peak-memory runs one measured optimizer step after warmup and peak reset;
- raw trials with old or wrong warmup/measured counts fail validation;
- sessions, manifests, and canonical commands bind the new counts;
- incompatible old journal sessions fail before process launch;
- thermal recovery, atomic publication, output locking, and resume behavior are
  otherwise unchanged; and
- all existing v2 tests plus Ruff check and format verification pass.

## Operational Handoff

After implementation and verification, create or refresh a clean detached
measurement checkout at the new harness commit. Do not delete any retained
failure journal. Do not start another baseline automatically: a new baseline
requires an explicit user request and a new external state directory after the
live AC, thermal, memory, and competing-workload preflight passes.

## Success Criteria

- The harness and every comparison surface use five warmups, 20 default
  measured units, one peak-memory step, and one cold-compile invocation exactly.
- A raw trial cannot mix counts from the old and new protocols.
- The full 1,024-token workload shape and existing five-pair baseline/screen and
  ten-pair final evidence profiles remain unchanged.
- Full v2 Ruff and pytest verification passes.
- The detached measurement checkout is clean and bound to the verified harness
  commit.
