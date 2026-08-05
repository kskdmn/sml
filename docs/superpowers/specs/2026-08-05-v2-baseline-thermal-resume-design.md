# V2 Baseline Thermal Recovery and Resume Design

**Status:** Approved

**Date:** 2026-08-05

## Purpose

The pinned v2 baseline runs 45 fresh processes and may take several hours. The
current runner keeps every trial in a temporary directory, validates the entire
set only after the last process, and deletes the temporary directory on any
failure. A single non-nominal macOS thermal observation can therefore discard
all otherwise valid work and the observation that caused rejection.

This change makes baseline capture resumable and records every thermal result.
It retains the existing fail-closed acceptance rules: only trials produced under
the required environment enter the baseline, and performance values never
influence whether a trial is retained or retried.

## Scope

This design changes only the versioned v2 benchmark harness and its tests. It
does not relax the pinned source commit, canonical workload, clean-checkout
requirements, trial count, warmup or measurement counts, or final baseline
validation. It does not change comparison noise retries.

Changing the harness produces a new harness content identity and requires a new
harness commit before baseline capture resumes. No result from the failed
`f5aa48f` capture can be reused because that invocation deleted its temporary
trial records.

## Command and State Location

`record-baseline` requires an explicit `--state-directory`. The resolved state
directory must be outside the harness checkout and every measurement source
checkout so journal writes cannot make a measured checkout dirty.

The requested baseline manifest and raw JSONL remain final artifacts. They are
not created or modified until all 45 required trial slots have been accepted
and the complete baseline validates.

The state directory is durable, non-authoritative resume and diagnostic state:

```text
<state-directory>/
├── session.json
├── accepted/
│   └── <metric>/<pair-index>.json
├── rejected/
│   └── <metric>/<pair-index>/<attempt-index>.json
├── inflight/
│   └── <metric>/<pair-index>/<attempt-index>.json
├── thermal-waits/
│   └── <metric>/<pair-index>/<recovery-index>/
│       ├── <sample-index>.json
│       └── summary.json
└── completed.json
```

`completed.json` is absent until final artifacts have been atomically published.
The state directory remains after success for auditability.

## Session Identity and Resume

`session.json` is atomically written before the first trial and binds the run to:

- schema version;
- fully resolved harness commit and harness content identity;
- fully resolved pinned source commit;
- canonical workload and workload identity;
- complete immutable benchmark protocol;
- hardware identity;
- exact required software versions; and
- final manifest and raw-output destinations.

The session body has a structured SHA-256 identity. Resuming recomputes and
validates every field. A mismatch fails before measurement; the runner never
silently replaces or combines incompatible state.

Each accepted slot is an immutable, atomically written `RawTrial`. Resume loads
all accepted slots, rejects duplicates or unexpected slots, and reruns full raw
trial validation. A valid slot is skipped. An accepted slot cannot be replaced
because of its performance value. Missing slots continue in canonical
metric/pair order.

Journal attempt indices are diagnostic sequence numbers and persist across
invocations. They are separate from `RawTrial.attempt_index`, which remains `0`
for every baseline measurement so the accepted evidence retains the immutable
baseline schema and protocol.

Rejected attempts and thermal-wait logs are diagnostic evidence only. They
cannot satisfy an accepted slot and are excluded from the final baseline
manifest and JSONL.

## Environment Observations

Every environment observation records both representations of the macOS
Foundation result:

- `thermal_state`: `nominal`, `fair`, `serious`, or `critical`; and
- `thermal_state_raw_value`: `0`, `1`, `2`, or `3`.

Start and end observations remain embedded in each raw trial. The merged trial
environment retains the worse thermal state and its matching raw value. Thermal
wait samples record both values, wall-clock time, and elapsed recovery time.

Hardware, software, AC power, power mode, Low Power Mode, memory pressure, and
competing GPU workload remain part of the environment record. A non-thermal
environment violation stops the invocation while preserving journal state.

## Trial Acceptance and Thermal Recovery

For every missing slot, the parent runner follows this state machine:

1. Collect and record the pre-trial environment.
2. If the thermal state is non-nominal, enter thermal recovery without launching
   the trial. Any non-thermal violation stops the invocation.
3. Launch one fresh child process with the pinned harness and source checkouts.
4. The child atomically writes the complete raw trial into its unique `inflight`
   path before the parent decides acceptance.
5. Validate its schema, identities, configuration, protocol, hardware, software,
   and environment.
6. If every field is valid, atomically promote it into its immutable accepted
   slot.
7. If its merged thermal state is non-nominal, retain it as a rejected attempt,
   enter thermal recovery, and retry only that slot.
8. Any other validation or subprocess failure stops the invocation and preserves
   all prior journal state. It is not retried automatically.

Promotion decisions cannot inspect throughput, latency, memory, or another
measured value. This prevents selective retry from biasing the baseline.

Thermal recovery samples the environment every 30 seconds. A retry is allowed
only after thermal state has remained continuously `nominal` for at least five
minutes. Any non-nominal sample resets that continuous window. Power,
low-power-mode, memory-pressure, or competing-workload violations stop rather
than wait.

A two-hour recovery deadline starts at the first non-nominal observation for
the current slot during an invocation. It does not reset after a failed retry.
If the deadline expires, the command exits nonzero with accepted trials,
rejected attempts, and all thermal samples preserved. A later invocation may
resume the same compatible session and receives a new two-hour recovery window.

## Atomicity and Failure Handling

State files use temporary-file write, flush, file `fsync`, atomic rename, and
parent-directory `fsync`. Accepted slot files are create-once. Existing
different content, malformed JSON, partial state, duplicate slots, or an invalid
session identity fails closed.

On resume, a complete in-flight raw trial is classified using the same
identity-and-environment rules before any new process is launched. A valid
in-flight trial is promoted; a thermally invalid one is recorded as rejected;
and a malformed or non-thermal-invalid one stops the invocation. This closes the
crash window between child completion and parent promotion without selectively
discarding a measured value.

The parent prints progress after every state transition, including the current
metric and pair, accepted count out of 45, rejected thermal state and raw value,
elapsed cooling time, and retry start. Progress output is informational and is
not evidence.

A child-process crash preserves previously accepted state and any diagnostic
output that was successfully published, then exits nonzero. It never promotes a
partial raw trial.

## Final Publication

When all 45 slots exist, the runner:

1. reloads and revalidates the complete session and accepted slot set;
2. orders trials by canonical metric order and pair index;
3. requires exactly five accepted trials for each of the nine metrics;
4. builds the baseline manifest;
5. runs the existing complete baseline validation;
6. atomically writes the final raw JSONL and manifest; and
7. atomically writes `completed.json` binding their identities and paths.

Rejected attempts and wait samples are never copied into final performance
evidence. They remain available in the state directory to explain every retry
or timeout.

## Tests

Tests use fake clocks, deterministic environment sequences, and real schema
validation to prove:

- string and raw thermal values are mapped and retained together;
- a pre-trial non-nominal state waits without launching a child;
- a post-trial non-nominal state rejects the attempt and retries only its slot;
- five continuous nominal minutes are required and reset by regression;
- the two-hour deadline exits while preserving accepted and diagnostic state;
- resume skips accepted slots and fills only missing slots;
- session, identity, hardware, software, protocol, and destination mismatches
  fail closed;
- accepted slots are immutable and cannot be replaced based on measured values;
- rejected attempts never enter final evidence;
- interrupted or malformed state cannot be treated as accepted; and
- final publication contains exactly 45 valid trials in canonical order.

Before the new harness commit, run Ruff check and format verification plus the
complete v2 test suite outside the sandbox. Baseline measurement begins only
from a separate clean checkout at that new harness commit.
