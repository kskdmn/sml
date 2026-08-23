# V2 Post-Exit Memory Recovery Design

**Status (2026-08-23):** Implemented and reviewed, including the deterministic-
termination amendment. This is an optional benchmark-evidence protocol, not
remaining refactor work.

**Date:** 2026-08-09

## Purpose

The schema-v2 baseline capture completed seven of 45 required trials, then
rejected `pretraining-compute` pair 2 three times because the immediate parent
post-exit observation still reported macOS memory pressure at `warning`.
Every attempt began with normal memory pressure and 65% to 74% free memory,
ended with nominal thermal state, and used the same 7,875,602,848-byte Metal
peak and approximately 44.5-second measured region. Two earlier
`pretraining-compute` pairs with the same workload and peak were accepted after
their immediate post-exit observations returned to `normal`.

A controlled five-minute nominal pre-launch window did not change the rejected
result. The failure is therefore not explained by a hotter machine, a larger
allocation, a longer outlier, or insufficient pre-launch settling. The
remaining variable is how quickly macOS reports recovered memory pressure
after the child process and its Metal allocations exit.

The current protocol samples that transition once and requires it to be
`normal` immediately. This design retains that immediate observation as
immutable evidence, then adds a bounded, fully recorded post-exit recovery
window when the first observation is `warning`. A trial is admissible only
after memory pressure remains `normal` continuously for a fixed stability
period while every other volatile environment condition remains valid.

## Goals

- Preserve the full 1,024-token workload, five warmup units, 20 measured units,
  model configuration, precision policy, data, process pairs, and statistical
  gates.
- Keep the immediate post-exit memory observation as durable evidence rather
  than hiding or delaying it.
- Distinguish transient OS memory-pressure recovery from persistent or critical
  pressure using a fixed protocol that never examines performance values.
- Require positive, continuous evidence that memory pressure returned to
  `normal` before accepting a trial.
- Record every recovery sample and outcome as immutable, identity-bound
  evidence.
- Make interruption and resume behavior deterministic at every recovery and
  finalization boundary.
- Apply exactly the same recovery contract to baseline and candidate trials.

## Non-Goals

- Do not accept `warning` as the final recovered state.
- Do not accept `critical` memory pressure at any observation.
- Do not discard, replace, or rewrite the immediate post-exit observation.
- Do not clear MLX caches, run garbage collection, or mutate the child workload
  to manufacture a passing state.
- Do not insert recovery work inside a timed measurement region.
- Do not change measured-unit counts, warmup counts, sequence lengths, batch
  sizes, gradient accumulation, model dimensions, or statistical thresholds.
- Do not make cadence, timeout, stability duration, or retry behavior adaptive
  to measured performance.
- Do not migrate existing schema-v2 trials or journal slots into the new
  protocol.
- Do not automatically rerun an attempt rejected for persistent or critical
  memory pressure. The existing automatic thermal-recovery behavior remains
  unchanged.

## Considered Approaches

### Bounded post-exit recovery — selected

The parent records the current immediate observation. If memory pressure is
`warning`, it samples the volatile environment every five seconds for up to
five minutes and requires 30 continuous seconds of valid `normal` samples.
This preserves measurement semantics and turns recovery into explicit
evidence. Its cost is additional wall-clock time only after a warning.

### Reduce measured units again — rejected

Reducing training metrics below 20 measured units would shorten sustained load,
but the observed Metal peak is already identical across accepted and rejected
trials. A shorter region may therefore leave the post-exit transition unchanged
while reducing statistical stability and invalidating the current protocol for
an uncertain benefit.

### One process per measured unit — rejected

Releasing process memory after every optimizer step would create a clean
lifecycle boundary, but it would redefine the paired process measurement,
multiply startup and compilation work, and require cross-process aggregation.
That is a substantially different benchmark rather than a focused recovery
policy.

## Canonical Recovery Policy

The canonical workload records the following immutable values:

- immediate post-exit memory pressure eligible for acceptance or recovery:
  `normal` or `warning`;
- recovered post-exit memory pressure required: `normal`;
- recovery required after an immediate `warning`: true;
- recovery sample interval: 5 seconds;
- recovery timeout: 300 seconds;
- continuous-normal stability window: 30 seconds; and
- recovery evidence required: true.

These values participate in canonical workload and harness identities. They are
not CLI overrides. Baseline capture, screen comparisons, and final comparisons
must use the same values.

The timeout uses a monotonic clock. It starts at the immediate post-exit memory
sample, so evidence validation and durable writes do not silently extend the
five-minute bound. UTC timestamps remain in evidence for auditability.

## Acceptance State Machine

The parent always samples immediate post-exit memory first, before slower
environment probes, exactly as in the current protocol.

1. If any non-memory field in the immediate observation is invalid, the parent
   records `environment-failure` without starting memory recovery.
2. If the immediate pressure is `critical`, the parent records `critical`
   without starting memory recovery.
3. If the immediate pressure is `normal`, the parent records a recovery summary
   with outcome `not-required` and no recovery samples.
4. If the immediate pressure is `warning`, the parent enters recovery without
   launching another workload.
5. Recovery collects complete environment samples at five-second intervals.
6. The first fully valid `normal` sample starts the stability window.
7. Each subsequent fully valid `normal` sample extends the window. Acceptance
   requires at least 30 monotonic seconds between its first and final samples.
8. A later `warning` resets the stability window but does not discard earlier
   samples or extend the original timeout.
9. A `critical` sample rejects the attempt immediately.
10. Any non-memory environment violation rejects the attempt immediately.
11. Remaining at `warning`, or failing to complete a 30-second stable window,
   when the 300-second deadline is reached rejects the attempt.

Each recovery sample uses the existing complete environment collector. Memory
pressure and free-memory percentage are sampled first and timestamped before
the slower hardware, software, thermal, power, and competing-workload probes.
That captured memory-probe start is the authoritative elapsed-time input to a
shared first-terminal reducer used by both live collection and evidence
validation. The first recovery probe starts at least five seconds after the
immediate probe, every later probe starts at least five seconds after its
predecessor, and non-memory failure precedes critical pressure when both occur
in one sample. The five-second cadence is measured between memory-sample start
times; a slow collector cannot cause overlapping probes or extend the original
deadline.
Recovery collection must not run unrelated work between samples.

The recovery branch and every disposition depend only on environment evidence,
time, and canonical policy. Metric values, elapsed time, throughput, and peak
memory are not inputs to the state machine.

## Evidence Model

### Existing stages

The version-1 `ChildTrialMeasurement` and version-1 immediate
`PostExitObservation` retain their current content and authority. The latter is
renamed only conceptually as the immediate observation; its stored document is
never rewritten after recovery.

### Recovery samples

Each recovery sample is a create-only versioned document containing:

- session, harness, workload, source, metric, pair, side, process-attempt, and
  journal-attempt identities;
- the exact child-measurement and immediate post-exit identities;
- a zero-based sample index and the prior sample identity, when one exists;
- UTC observation time and monotonic elapsed seconds from the immediate memory
  sample;
- the complete hardware, software, and environment observation; and
- a structured SHA-256 identity over its canonical content.

The prior-sample binding makes the ordered sequence an immutable chain. Sample
indices must be contiguous, identities unique, elapsed times satisfy the
canonical minimum cadence, and elapsed time must not exceed the canonical
timeout. No identity-valid chain may contain a sample after the shared
reducer's first terminal event.

### Recovery summary

Every attempt receives a create-only version-2 `PostExitRecovery` summary using
the `sml-parent-post-exit-recovery-v2` identity domain. It
binds the child measurement, immediate post-exit observation, canonical
recovery policy, ordered recovery-sample identities, and required
`completion_source` (`live` or `crash-reconstruction`). Its outcome is exactly
one of:

- `not-required`: the immediate observation was `normal` and the sample list is
  empty;
- `recovered`: an immediate `warning` was followed by a complete continuous
  normal window;
- `timeout`: the deadline expired without a complete stable window;
- `critical`: the immediate observation or a recovery sample reached
  `critical`;
- `environment-failure`: a non-memory environment condition became invalid; or
- `interrupted`: the process stopped after an immediate warning but before a
  terminal recovery summary existed.

For `completion_source: live`, the outcome, duration, and failure fields match
the shared reducer's first terminal result. For
`completion_source: crash-reconstruction`, immediate normal, critical, and
non-memory failure with zero samples reconstruct `not-required`, `critical`,
and `environment-failure` respectively. An immediate warning with no summary
always reconstructs `interrupted`, with or without partial samples. Its final
sample may itself be the first decisive critical, non-memory-failure,
stable-completion, or deadline event because the crash may occur between the
durable sample write and summary publication; no sample may follow that event.

The summary records its terminal sample identity when samples exist, terminal
environment observation, elapsed duration, and structured content identity.
Validation recomputes live completion from the immutable immediate observation,
sample chain, and canonical policy, and validates crash completion against its
explicit provenance rather than trusting cached fields. Version-1 recovery
summaries are incompatible.

### Final raw trial

`RawTrial` advances to schema version 3. It binds the child measurement,
immediate post-exit observation, and recovery-summary identities. Every
terminal recovery outcome deterministically produces a complete raw trial
before classification. Only `not-required` and `recovered` outcomes can be
admitted. Rejected raw trials remain durable journal evidence but cannot enter
a manifest, published raw JSONL file, or comparison.

For an immediate-normal trial, top-level memory pressure and free-memory
percentage derive from the immediate observation. For a recovered trial, they
derive from the terminal recovery sample. Measurement memory pressure remains
the worse of child start and child end. Thermal, power, Low Power Mode, and
competing-workload summaries conservatively include child start, child end,
immediate post-exit, and every recovery sample. Validation recomputes these
derivations from their nested evidence.

## Runtime Data Flow

For every missing baseline or comparison slot:

1. The parent collects and validates the existing preflight observation.
2. A fresh child constructs and runs the canonical native workload.
3. The child atomically publishes `ChildTrialMeasurement` and exits.
4. The parent identity-validates that document and immediately samples
   post-exit memory before the remaining environment fields.
5. The parent atomically publishes the immediate `PostExitObservation`.
6. The parent creates either a `not-required` recovery summary or begins the
   bounded recovery state machine.
7. During recovery, each sample is atomically published before the next sample
   is collected.
8. The parent atomically publishes the terminal recovery summary.
9. For every terminal summary, the parent deterministically builds and
   atomically publishes `RawTrial` version 3.
10. The caller validates and classifies the finalized trial without inspecting
    its performance value.

No second child is launched during recovery. Persistent or critical memory
pressure stops the invocation under the existing manual-resume policy; a later
explicit resume uses a new journal attempt for the still-missing slot. A
thermal rejection continues to use the existing bounded automatic thermal
recovery before a new attempt. Other environment failures stop fail-closed as
they do today.

## Durable Journal and Crash Recovery

The journal adds create-only recovery paths:

```text
<state-directory>/
├── measurements/<metric>/<pair>/<attempt>.json
├── post-exit/<metric>/<pair>/<attempt>.json
├── recovery-samples/<metric>/<pair>/<attempt>/<sample>.json
├── recovery/<metric>/<pair>/<attempt>.json
├── inflight/<metric>/<pair>/<attempt>.json
├── accepted/<metric>/<pair>.json
└── rejected/<metric>/<pair>/<attempt>.json
```

Resume validates the full topology before preflight or process launch:

- A child measurement without immediate post-exit evidence remains
  unrecoverable and is rejected as `missing-immediate-post-exit-evidence`.
- An immediate-normal observation without a recovery summary deterministically
  reconstructs the `not-required` summary.
- An immediate-warning observation with no terminal summary, with or without
  partial samples, cannot resume a continuous window after an unknown gap. It
  creates an `interrupted` summary and rejects the attempt as
  `interrupted-post-exit-recovery`.
- That immediate-warning reconstruction remains `interrupted` when its last
  partial sample is the first decisive live terminal event (critical,
  non-memory failure, stable completion, or deadline); later samples are
  corruption.
- A complete admissible recovery summary without a final raw trial
  deterministically reconstructs schema-v3 finalization.
- A complete non-admissible recovery summary without a final raw trial
  deterministically reconstructs schema-v3 finalization, then the rejected
  outcome.
- A finalized raw trial without classification completes the same immutable
  accepted or rejected transition used by the current journal.
- Samples without their exact immediate observation, gaps or forks in the
  identity chain, conflicting terminal summaries, or evidence bound to another
  attempt are corruption and stop without rewriting evidence.

The recovery rejection reason is `persistent-post-exit-memory-pressure` for a
timeout, `critical-post-exit-memory-pressure` for critical pressure,
`post-exit-recovery-environment-violation` for another invalid condition, and
`interrupted-post-exit-recovery` for an incomplete recovery lifecycle.

## Compatibility and Baseline Transition

The recovery policy changes the canonical workload identity, harness content
identity, evidence session, and raw-trial schema. The retained schema-v2 state
at `/private/tmp/sml-v2-baseline-post-exit-state-aa6bb43` remains diagnostic
evidence only. Its seven accepted trials and three rejected attempts must not be
copied, upgraded, or resumed under the new protocol.
Recovery-summary-v1 documents likewise cannot enter the current evidence
session; summary provenance begins at version 2.

After implementation and verification, a new baseline begins in a new empty
external state directory using a clean detached checkout at the verified
harness commit. Existing diagnostic state is preserved unless the user later
explicitly approves its deletion.

## Error Handling

- Immediate or recovered `critical` pressure rejects immediately.
- A timeout rejects without extending its deadline or adapting the cadence.
- Any thermal violation is retained as trial evidence and follows the existing
  automatic five-minute nominal-window recovery policy before another attempt.
- Power, Low Power Mode, hardware, software, source, identity, or competing
  workload drift fails closed.
- A failed atomic write stops without overwriting an existing evidence file.
- Malformed, incomplete, duplicated, reordered, or identity-mismatched recovery
  evidence stops before another workload launch.
- Recovery failure never deletes an accepted slot or another attempt's staged
  evidence.

## Tests

Tests use a fake monotonic clock and scripted observations to prove:

- immediate `normal` creates a zero-sample `not-required` summary without
  sleeping;
- immediate `warning` followed by 30 continuous seconds of valid normal samples
  creates `recovered`;
- a warning after partial stability resets the window without extending the
  deadline;
- timeout, critical pressure, thermal drift, power drift, Low Power Mode, and a
  competing workload reject with their exact dispositions;
- every sample includes free-memory percentage, thermal, and power evidence;
- sampling cadence and the 300-second monotonic deadline are fixed;
- performance values cannot alter recovery or disposition;
- sample-chain, summary, raw-trial, and derived-environment tampering fail
  closed;
- every documented crash boundary resumes or rejects deterministically before
  process launch;
- schema-v2 raw trials and sessions fail compatibility checks;
- baseline and comparison capture use the same recovery protocol; and
- all existing v2 tests plus Ruff check and format verification pass.

After the automated suite, refresh the detached measurement harness and prove
its commit and content identity. Do not launch a replacement 45-trial baseline
until live AC, thermal, memory, and competing-workload preflight passes and the
user explicitly authorizes the run.

## Success Criteria

- The full 1,024-token, five-warmup, 20-measured-unit workload is unchanged.
- Immediate post-exit warning remains visible in immutable evidence.
- No warning trial is accepted without 30 continuous seconds of recorded
  normal pressure inside the fixed five-minute deadline.
- Critical pressure or any unrelated environment violation cannot be accepted.
- Crash recovery cannot invent, continue across a gap, reorder, or reuse a
  recovery observation.
- Old schema-v2 evidence cannot enter the new baseline or comparisons.
- Full v2 Ruff and pytest verification passes before a new baseline begins.
