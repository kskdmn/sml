# V2 performance harness

This directory contains the independently versioned benchmark harness for the
performance-first v2 refactor. It compares a canonical semantic workload across
the legacy and replacement native representations.

All measurements run in fresh processes from clean source checkouts. Timed MLX
regions synchronize immediately before and after execution, perform one untimed
compilation pass, and, except for compile cold-start, warm up for 5 units and
measure 20 units by default. Compile cold-start uses zero warmups and measures
its first compiled invocation. Inference consumes its complete fixed 32-request
set; compile cold-start and peak Metal memory each measure one canonical unit.
Trial order alternates by pair. Raw process records are retained alongside
reports; statistical decisions use direction-normalized paired ratios and a
reproducible whole-pair bootstrap.

The harness identity hashes the ordered bytes of the schema, evidence,
workload, runner, journal, recovery, and analysis modules, both adapters, and
the fixed analysis-vector test module. Any change to those files invalidates
dependent baseline and phase evidence.

`record-baseline` is intentionally immutable: it accepts only the fully resolved
`3687f8b` source commit, all nine metrics, five fresh processes per metric, five
warmups for every non-compile metric, and the canonical measured-unit counts.
Compile cold-start uses zero warmups and measures its first compiled invocation.
Inference consumes the complete 32-request set exactly once, while compile
cold-start and peak Metal memory each consume one canonical unit. Validation
rejects metric subsets, changed protocol values, duplicate raw records, a
different canonical projection or logical work order, and environment or software
mismatches.

Each process publishes an identity-bound child measurement containing separate
start and end observations recorded while MLX is alive; the child never
publishes a final trial. Immediately after the child exits, the parent samples
memory pressure and free-memory percentage before any slower hardware or
software probe. This immediate memory observation remains a version-1,
identity-bound observation. Immediate normal memory writes the `not-required`
recovery summary with zero samples. Immediate warning memory samples the
complete environment every five seconds for at most five minutes and requires
30 continuous normal seconds before recovery; every sample is written before
classification, a warning resets only the stability window, and a critical or
non-memory failure terminates immediately. Schema-v3 raw trials embed the
ordered recovery-sample identity chain and its recovery summary. Raw-trial
identities use the `sml-raw-benchmark-trial-v3` domain. Baseline manifests, raw
JSONL, comparison reports, predecessor replay, phase validation, and final
validation all revalidate the complete embedded recovery evidence before
accepting a raw identity or measured value. Schema-v2 (and older) raw trials,
and evidence whose nested identities were changed, are incompatible diagnostic
records, not current benchmark evidence.

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
├── measurements/<metric>/<pair-index>/<attempt-index>.json
├── post-exit/<metric>/<pair-index>/<attempt-index>.json
├── recovery-samples/<metric>/<pair-index>/<attempt-index>/<sample-index>.json
├── recovery/<metric>/<pair-index>/<attempt-index>.json
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

Every preflight is persisted before validation. Child-start memory pressure
must be `normal`. Child-end `warning` is diagnostic and may pass only when
child-start and immediate parent post-exit pressure are both `normal` and every
other strict check passes. Any non-normal child-start pressure, child-end
`critical`, or any non-normal post-exit memory pressure is classified as a
memory rejection: the version 3 rejected-trial record is written durably and
the invocation stops without a same-run retry. The accepted slots remain in
place, and the rejected attempt index remains part of the journal's retry
accounting.

Other strict hardware, software, power, competing-workload, protocol,
identity, subprocess, or schema failures are not classified trial rejections
and do not create a rejected-trial record. They stop the invocation immediately
and preserve the current persisted journal topology—such as a preflight,
measurement, post-exit observation, or in-flight trial—for diagnosis and exact
revalidation on resume.

Only a consistent non-nominal thermal observation enters recovery. Recovery
samples at intervals no longer than 30 seconds and permits another trial only
after five continuous minutes of nominal thermals. The first thermal violation
for a missing slot starts one two-hour deadline for that slot during the
invocation; rejected retries do not reset it. Rejected trials and recovery
samples remain diagnostic evidence and never enter the baseline.

On resume, every persisted preflight and every persisted thermal-recovery
sample is replayed through the same complete session-bound hardware, software,
raw/string thermal, and non-thermal environment validator used for live
observations. This replay happens before recovery classification, a new
preflight, or a process launch, including when a later recovery summary records
a nominal window.

Resume requires the exact same harness commit and content identity, pinned
source commit, canonical workload, immutable protocol, hardware, software,
paired representations, and resolved final output paths. A session created
before the schema-v3 post-exit memory policy is incompatible and is never
migrated. In particular, retained schema-v2 state at
`/private/tmp/sml-v2-baseline-post-exit-state-aa6bb43` is diagnostic only.
Compatible accepted slots are validated and reused. Measurement-only crash
evidence is durably rejected as `missing-immediate-post-exit-evidence`; a later
invocation never fabricates a new post-exit observation for an old process. A
matching child measurement, immediate observation, ordered recovery samples,
and recovery summary reconstruct the final schema-v3 trial deterministically
before classification. A complete in-flight trial is likewise classified before
any replacement is launched. A crash during warning recovery is `interrupted`
and never continues its old stability window.

Timeout, critical, and interrupted post-exit memory outcomes stop for manual
resume; only a thermal-only failure retains automatic thermal recovery. After a
memory rejection, the operator resumes by running the same canonical
`record-baseline` command against the same state directory. The later manual
resume revalidates the complete journal, performs a new strict preflight, and
launches only the still-missing slot with the next journal attempt index; it
does not rerun accepted slots. Persisted non-nominal preflights, thermally
rejected trials, unfinished thermal recovery episodes, and timed-out thermal
recovery episodes resume with a fresh five-continuous-minute nominal window
before a new preflight or trial can run. Only the same thermally rejected slot
is retried. No baseline starts automatically after implementation. The final raw
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
