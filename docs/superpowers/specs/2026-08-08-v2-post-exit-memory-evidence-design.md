# V2 Post-Exit Memory Evidence Design

**Status:** Approved design; written specification pending user review

**Date:** 2026-08-08

## Purpose

The shorter v2 benchmark protocol completed five `prepared-data` trials, then
rejected the first `pretraining-compute` trial because the child process ended
with macOS memory pressure at `warning`. The child had already completed all 20
measured units, but it sampled the environment while the native MLX workload
and its Metal allocations were still alive. Immediately after that child
exited, the parent observed memory pressure `normal` with substantially more
free memory.

The current raw-trial boundary therefore conflates two different facts:

- memory pressure while the benchmark process still owns its MLX allocations;
  and
- system memory pressure after that process and its allocations have been
  released.

This change records both facts. The child-end observation remains mandatory
diagnostic evidence, while an immediate parent observation after child exit
becomes the authoritative memory-pressure gate. A child-end `warning` is
acceptable only when the trial started at `normal`, the child never reached
`critical`, the immediate post-exit observation returns to `normal`, and every
other strict environment check passes.

## Goals

- Preserve the full model and 1,024-token benchmark workload.
- Preserve strict fail-closed environment validation without rejecting a trial
  solely because released MLX allocations were still alive at child sampling.
- Record start, child-end, and parent post-exit observations as durable,
  identity-bound evidence.
- Make crash recovery deterministic at every boundary between measurement,
  post-exit collection, trial finalization, and journal classification.
- Stop on persistent memory pressure while preserving accepted slots so a
  later manual invocation can resume the same missing slot.
- Apply one raw-trial contract to baseline capture and candidate comparisons.

## Non-Goals

- Do not clear MLX caches inside the child to manufacture a passing result.
- Do not sleep or cool down between child exit and the post-exit observation.
- Do not accept `critical` memory pressure at any observation.
- Do not accept post-exit memory pressure other than `normal`.
- Do not weaken AC-power, power-mode, Low Power Mode, thermal, competing-GPU,
  hardware, software, source, protocol, or identity checks.
- Do not change workload dimensions, sequence lengths, measured units, warmup
  units, process-pair counts, or statistical gates.
- Do not automatically retry persistent memory-pressure failures in the same
  invocation.
- Do not migrate trials from an older raw schema or workload identity.

## Acceptance Policy

The canonical workload records the complete memory policy in
`required_environment`:

- `memory_pressure: "normal"` applies to parent preflight and child start;
- `measurement_end_memory_pressure_allowed: ["normal", "warning"]` applies to
  the child-end observation;
- `post_exit_memory_pressure: "normal"` applies to the immediate parent
  observation; and
- `post_exit_evidence_required: true` prevents a consumer from treating the
  child measurement alone as a complete trial.

These fields are canonical workload data and participate in workload identity.
Their values are immutable protocol requirements, not CLI overrides.

A trial is accepted only if all of the following are true:

1. Parent preflight passed before process launch.
2. Child-start memory pressure is `normal`.
3. Child-end memory pressure is either `normal` or `warning`.
4. Parent post-exit memory pressure is `normal`.
5. Thermal state is `nominal` at child start, child end, and parent post-exit.
6. The existing strict power, hardware, software, source, protocol, identity,
   and competing-workload checks pass at every observation where they apply.
7. All staged evidence is complete, internally consistent, and bound to the
   same session, slot, process attempt, and measured result.

This policy does not claim that all child-end warnings are caused by MLX. It
accepts only the directly observed lifecycle pattern: pressure was normal at
start, never critical while the child was alive, and normal again immediately
after the benchmark process exited.

## Architecture

Trial production becomes a three-stage protocol with separate responsibilities:

1. **Child measurement:** the fresh child builds the native workload, records
   its start environment, runs the benchmark, records its end environment, and
   atomically publishes an immutable child-measurement document.
2. **Parent post-exit observation:** after the child exits, the parent samples
   memory pressure first, then collects the remaining environment fields. It
   atomically publishes an immutable post-exit document bound to the exact
   child-measurement identity.
3. **Parent finalization:** the parent validates the two documents, derives the
   complete environment status, and atomically publishes a finalized raw trial.
   Only a finalized raw trial can be accepted or compared.

The parent starts post-exit collection directly after `subprocess.run` returns.
It performs no wait, cooling cycle, new benchmark launch, or unrelated work
first. Reading the child document and computing its content identity may happen
before collection so the observation can be bound to exact bytes. Within the
collector, memory pressure and free-memory percentage are sampled before slower
hardware or software probes. The evidence records the observation timestamp.

This boundary makes the authority clear: the child owns measured performance
and in-process diagnostics; the parent owns the post-process memory gate and
final trial publication.

## Evidence Documents

### Child measurement

The child writes a versioned `ChildTrialMeasurement` document. It contains:

- the session, harness, workload, source, metric, pair, and process identities;
- the journal attempt index used for durable state binding;
- metric result, elapsed time, work count, compilation time, and peak Metal
  memory;
- exact model, data, precision, protocol, and software fields required by the
  existing raw-trial validator;
- complete child-start and child-end hardware and environment observations;
- process timestamps; and
- a structured SHA-256 identity over its canonical content.

The child measurement is not a `RawTrial` and cannot enter a manifest, raw
JSONL file, accepted slot, or statistical comparison.

### Post-exit observation

The parent writes a versioned `PostExitObservation` document containing:

- session, metric, pair, and journal attempt identity;
- the exact child-measurement content identity;
- the observation timestamp;
- complete hardware, environment, and software observations; and
- a structured SHA-256 identity over its canonical content.

The validator requires the measurement identity to match the exact staged
child document. A post-exit document cannot be reused for another attempt even
when its slot and environment values happen to be identical.

### Final raw trial

`RawTrial` advances to schema version 2. Its measured fields come exactly from
the child measurement. Its `environment_status` contains required nested
`start`, `end`, and `post_exit` observations plus deterministic top-level
derivations. The raw trial also binds the child-measurement and post-exit
content identities.

Schema version 2 is complete only with valid post-exit evidence. Version 1 raw
trials are incompatible and cannot be silently upgraded because no immediate
post-exit observation exists for them.

## Derived Environment Semantics

The final `environment_status` preserves each observation unchanged and derives
summary fields as follows:

- top-level `memory_pressure` and `memory_free_percentage` come from
  `post_exit`; these are the authoritative acceptance values;
- `measurement_memory_pressure` is the worse of child `start` and `end` and is
  retained as diagnostic evidence;
- `measurement_min_free_percentage` is the minimum child free-memory value;
- top-level thermal state and raw value are the worse of `start`, `end`, and
  `post_exit`;
- AC-power presence uses the existing conservative all-observations merge;
- Low Power Mode uses the existing conservative any-observation merge; and
- power mode, hardware, software, and competing-workload summaries extend the
  existing strict merge rules across all applicable observations.

Validation recomputes every summary from the nested observations and rejects a
raw trial whose cached summary differs. Consumers therefore cannot change an
authoritative post-exit value while retaining contradictory nested evidence.

## Runtime Data Flow

For every missing baseline or comparison slot:

1. The parent records and validates the existing preflight observation.
2. The parent launches a fresh child with a create-only measurement path.
3. The child records its start environment, runs the canonical workload, and
   records its end environment while the native MLX workload is still alive.
4. The child atomically publishes `ChildTrialMeasurement` and exits.
5. The parent receives the exit status, loads and identity-validates the child
   document, then immediately samples post-exit memory and the rest of the
   environment.
6. The parent atomically publishes `PostExitObservation`.
7. The parent deterministically builds and atomically publishes `RawTrial` v2.
8. The caller classifies the finalized trial without inspecting its performance
   value for acceptance or retry decisions.

If the child exits unsuccessfully without a complete measurement document, the
invocation stops and preserves the journal. If it publishes a complete document
before a nonzero exit, the parent retains the document for diagnosis but does
not invent a successful process outcome.

## Durable Baseline Journal

The external baseline state adds immutable staging directories:

```text
<state-directory>/
├── session.json
├── measurements/
│   └── <metric>/<pair-index>/<journal-attempt-index>.json
├── post-exit/
│   └── <metric>/<pair-index>/<journal-attempt-index>.json
├── inflight/
│   └── <metric>/<pair-index>/<journal-attempt-index>.json
├── accepted/
│   └── <metric>/<pair-index>.json
├── rejected/
│   └── <metric>/<pair-index>/<journal-attempt-index>.json
├── preflight/
├── thermal-waits/
└── completed.json
```

`measurements`, `post-exit`, and `inflight` hold the three protocol stages.
Every file is create-only and uses the journal's atomic-write and directory
sync guarantees. Accepted slots remain immutable finalized `RawTrial` v2
documents. A rejected outcome is an immutable envelope containing its reason
and identities for every stage that existed; when a final raw trial exists, the
envelope contains that exact trial.

Measurement and post-exit documents remain as audit evidence after
classification. The finalized `inflight` raw trial retains its current pending
transition semantics: promotion or rejection links or copies its exact content
to the immutable outcome before removing the inflight entry. Resume recognizes
and completes an interrupted identical transition. A completed attempt's
topology must form one internally consistent chain. An accepted raw trial and a
rejected outcome for the same journal attempt are mutually exclusive.

The journal attempt index remains distinct from baseline
`RawTrial.attempt_index`, which stays at its canonical baseline value. All stage
documents carry the journal attempt index so two process launches for one slot
cannot exchange evidence.

## Crash Recovery

Resume validates the complete journal topology before launching a process:

- **No child measurement:** no process result exists; the missing slot may run
  normally after preflight.
- **Child measurement only:** an immediate post-exit observation can no longer
  be collected. The attempt is recorded as rejected with reason
  `missing-immediate-post-exit-evidence`; a later attempt may fill the slot.
- **Post-exit without its exact child measurement:** journal corruption; stop
  without deleting or rewriting evidence.
- **Child measurement and matching post-exit, but no final raw trial:** rebuild
  the version 2 raw trial deterministically from the immutable documents and
  continue classification.
- **Final inflight raw trial:** recompute identities and derived fields, then
  classify it through the same rules used during uninterrupted execution.
- **Accepted or rejected outcome with contradictory stage topology:** journal
  corruption; stop without launching or deleting anything.

Recovery never collects a delayed post-exit observation for a process that
exited during an earlier invocation. It never replaces an observation merely
because the replacement would pass.

## Validation and Failure Classification

Classification is deterministic and ordered so an attempt receives one primary
outcome:

1. Schema, content identity, session, source, workload, protocol, hardware, or
   software mismatch stops as corrupt or incompatible evidence.
2. AC power, power mode, Low Power Mode, or competing-workload violation stops
   as a strict non-memory environment failure.
3. Child-start memory other than `normal` is rejected as
   `non-normal-start-memory-pressure`, and the invocation stops.
4. Child-end memory `critical` is rejected as
   `critical-measurement-memory-pressure`, and the invocation stops.
5. Parent post-exit memory other than `normal` is rejected as
   `persistent-post-exit-memory-pressure`, and the invocation stops.
6. A non-nominal thermal observation is rejected through the existing thermal
   path, followed by five continuous nominal minutes before retrying the same
   slot within the existing two-hour recovery window.
7. A trial passing every check is promoted into its accepted slot.

Unknown memory states or missing required fields fail schema or environment
validation; they are never treated as `warning`. A child-end `warning` does not
cause a retry when the authoritative post-exit state is `normal` and all other
checks pass.

The validator does not inspect throughput, latency, compilation time, or peak
memory when choosing acceptance, rejection, or retry behavior.

## Stop and Manual Resume Behavior

A persistent post-exit warning or critical state rejects only the current
attempt, stops the invocation immediately, and preserves every previously
accepted slot. It does not permanently invalidate the compatible session and
does not start an automatic same-run memory retry.

A later manual invocation may resume the same state directory. It first
revalidates the session and complete journal, then requires a fresh strict
preflight. If preflight is clean, it launches a new journal attempt for the same
missing slot. Rejected memory attempts do not create thermal recovery triggers
and are never candidates for accepted evidence.

This separates transient operator-controlled recovery, such as closing other
applications, from automatic thermal recovery. It also prevents a hot retry
loop from increasing pressure.

## Comparison Integration

Baseline capture and candidate comparison share the same child measurement,
post-exit collection, raw finalization, and validation functions. Comparison
processes may stage their intermediate documents in their existing temporary
workspace, but every persisted comparison raw trial is schema version 2 and
contains the identity-bound post-exit evidence.

A comparison stops on persistent memory pressure under the same policy as a
baseline invocation. Existing comparison retry rules cannot override the
environment decision. Baseline manifests and comparison sessions require the
same canonical memory policy and raw schema; mixing version 1 baseline evidence
with version 2 candidate evidence fails before measurement.

## Identity and Compatibility

The new canonical `required_environment` fields change workload identity. The
new document schemas and implementation change harness content identity.
`RawTrial` schema version 2 makes the evidence boundary explicit.

All existing baseline journals and raw files are incompatible, including the
shorter-protocol journal that currently contains five accepted `prepared-data`
slots and one failed `pretraining-compute` attempt. That directory remains
untouched as diagnostic evidence. Its accepted version 1 slots cannot be copied
or promoted because they lack an immediate parent post-exit observation.

The first capture under this design requires a fresh external state directory
and a clean detached measurement checkout at the verified implementation
commit. No baseline run starts automatically as part of implementation or
verification.

## Tests

Tests use deterministic environment sequences, fake clocks where applicable,
real schema validation, and crash-boundary fixtures. They must prove:

- child-end `warning` plus post-exit `normal` is accepted when every other
  observation passes;
- child-end `critical` is rejected and stops the invocation;
- post-exit `warning` and `critical` are rejected and stop the invocation;
- child-start memory other than `normal` cannot be accepted;
- thermal, AC power, power mode, Low Power Mode, hardware, software, identity,
  and competing-workload checks remain strict across all observations;
- post-exit evidence is bound to the exact child measurement and journal
  attempt;
- a child measurement without immediate post-exit evidence cannot be accepted;
- a crash after both immutable stage documents reconstructs the same finalized
  raw trial deterministically;
- derived top-level fields cannot disagree with their nested observations;
- persistent memory rejection preserves prior accepted slots, stops without an
  automatic retry, and retries only the missing slot on a later clean resume;
- the existing thermal rejection, five-minute recovery, and two-hour deadline
  behavior is unchanged;
- baseline manifests, raw JSONL files, and comparisons require version 2
  post-exit evidence;
- older workload identities, raw trials, and journals fail closed; and
- full v2 Ruff check, Ruff format verification, and pytest pass.

## Operational Handoff

After implementation and verification, refresh the clean detached measurement
checkout at the new harness commit. Keep every retained older journal unchanged.
Report the new harness commit and content identity, but do not launch a baseline
until the user explicitly requests it, AC power is connected, and the complete
live preflight passes. Use a new empty external state directory for that run.

## Success Criteria

- Every accepted raw trial contains immutable child-start, child-end, and
  immediate parent post-exit observations.
- A child-end memory warning is accepted only when start and post-exit memory
  are normal and all strict non-memory checks pass.
- Critical or persistent post-exit memory pressure never enters accepted
  evidence.
- A crash cannot turn missing, delayed, mismatched, or partially published
  evidence into an accepted trial.
- Baseline resume preserves completed slots without selecting attempts by
  measured performance.
- Baseline and comparison paths enforce the same version 2 evidence contract.
- Full v2 static and test verification passes before any new baseline capture.
