# V2 performance harness

This directory contains the independently versioned benchmark harness for the
performance-first v2 refactor. It compares a canonical semantic workload across
the legacy and replacement native representations.

All measurements run in fresh processes from clean source checkouts. Timed MLX
regions synchronize immediately before and after execution, perform one untimed
compilation pass, warm up for 5 units, and measure 20 units by default. Inference
consumes its complete fixed 32-request set; compile cold-start and peak Metal
memory each measure one canonical unit. Trial order alternates by pair. Raw
process records are retained alongside reports; statistical decisions use
direction-normalized paired ratios and a reproducible whole-pair bootstrap.

The harness identity hashes the ordered bytes of the schema, workload, runner,
analysis, both adapters, and the fixed analysis-vector test module. Any change
to those files invalidates dependent baseline and phase evidence.

`record-baseline` is intentionally immutable: it accepts only the fully resolved
`3687f8b` source commit, all nine metrics, five fresh processes per metric, five
warmups, and the canonical measured-unit counts. Inference consumes the complete
32-request set exactly once, while compile cold-start and peak Metal memory each
consume one canonical unit. Validation rejects metric subsets, changed protocol
values, duplicate raw records, a different canonical projection or logical work
order, and environment or software mismatches.

Baseline capture also requires `--state-directory PATH`. The path is resolved
before use and must be outside both the harness checkout and the detached pinned
source checkout. The final manifest and raw JSONL paths must be distinct from
each other and both must be outside the state directory; none of the three
resolved locations may contain another. The state directory is a durable
journal that remains after success or failure:

```text
<state-directory>/
├── .baseline-session.lock
├── session.json
├── accepted/<metric>/<pair-index>.json
├── rejected/<metric>/<pair-index>/<attempt-index>.json
├── inflight/<metric>/<pair-index>/<attempt-index>.json
├── preflight/<metric>/<pair-index>/<preflight-index>.json
├── thermal-waits/<metric>/<pair-index>/<recovery-index>/
│   ├── trigger.json
│   ├── <sample-index>.json
│   └── summary.json
└── completed.json
```

One crash-released advisory lock serializes initialization, resume, capture, and
final publication for the state directory. Advisory locks for each normalized,
resolved final-output destination are stored in a private per-user directory
under `/tmp`; they serialize cleanup and publication across different state
directories that target any shared output. Lock inodes remain in place. Atomic
writes use the exact
`.DESTINATION.sml-atomic-<32-lowercase-hex>.tmp` namespace. At locked resume,
only regular orphan files in that namespace whose destination is valid for its
exact journal location are removed; the initialization marker is recovered in
the same locked pass. Exact manifest/raw-output temporaries are also removed
before clean-checkout validation. Other hidden files, malformed names,
unsupported locations, directories, and symlinks are retained and continue to
fail the strict topology or checkout checks.

Every preflight is persisted before validation. A non-thermal hardware,
software, power, memory-pressure, competing-workload, protocol, identity,
subprocess, or schema failure stops the invocation and leaves the journal for
diagnosis. Only a consistent non-nominal thermal observation enters recovery.
Recovery samples at intervals no longer than 30 seconds and permits another
trial only after five continuous minutes of nominal thermals. The first thermal
violation for a missing slot starts one two-hour deadline for that slot during
the invocation; rejected retries do not reset it. Rejected trials and recovery
samples remain diagnostic evidence and never enter the baseline.

On resume, every persisted preflight and every persisted thermal-recovery
sample is replayed through the same complete session-bound hardware, software,
raw/string thermal, and non-thermal environment validator used for live
observations. This replay happens before recovery classification, a new
preflight, or a process launch, including when a later recovery summary records
a nominal window.

Resume requires the exact same harness commit and content identity, pinned
source commit, canonical workload, immutable protocol, hardware, software,
paired representations, and resolved final output paths. Compatible accepted
slots are validated and reused; a complete in-flight trial is classified before
any replacement is launched. Persisted non-nominal preflights, thermally
rejected trials, unfinished recovery episodes, and timed-out recovery episodes
resume with a fresh five-continuous-minute nominal window before a new preflight
or trial can run. Only the same thermally rejected slot is retried. The final raw
JSONL and manifest remain absent until all 45 canonical slots validate.
Publication creates the raw JSONL first, the manifest second, and
`completed.json` last; existing identical bytes are accepted for crash resume,
while different final content is never overwritten. The external journal is
retained for auditability. The manifest command is a canonical session-derived
command template with `SESSION_STATE_DIRECTORY` as its only operator-supplied
placeholder; it does not depend on the invoking Python executable, working
directory, argument ordering, or relative path spelling.

Comparisons have two strict profiles. Screen mode uses five pairs, a 0.97 median
gate, 2% maximum dispersion, and a report-only confidence bound. Final mode uses
ten pairs, 1.5% maximum dispersion, required lower bounds, and a 1.03 baseline
gate for end-to-end pretraining. Each metric receives an explicit predecessor
mapping entry whose value is a report path or identity, or `null` for its first
replacement measurement; one phase-wide predecessor is not accepted.
`validate-phase` therefore takes `--predecessors` with the same complete
metric-to-report mapping as `compare`. `validate-final` independently reloads
the predecessor reports by their recorded identities, requires the complete
eight-metric final profile, verifies that `--raw-input` exactly equals the
report's ordered and complete raw-trial set, and rejects failed or persistently
noisy throughput gates.

When any comparison is too noisy, all temporary checkouts have already been
removed before a 15-minute cooldown begins. The last five minutes must remain on
the recorded power mode with nominal thermal state, normal memory pressure, and
no competing GPU work. The complete alternating-order comparison then runs once
more. Both attempts and the cooldown samples remain in the report, and noise in
the second attempt blocks acceptance.
