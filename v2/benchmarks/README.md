# V2 performance harness

This directory contains the independently versioned benchmark harness for the
performance-first v2 refactor. It compares a canonical semantic workload across
the legacy and replacement native representations.

All measurements run in fresh processes from clean source checkouts. Timed MLX
regions synchronize immediately before and after execution, perform one untimed
compilation pass, warm up for 20 units, and measure 100 units unless a fixed
request set is smaller. Trial order alternates by pair. Raw process records are
retained alongside reports; statistical decisions use direction-normalized
paired ratios and a reproducible whole-pair bootstrap.

The harness identity hashes the ordered bytes of the schema, workload, runner,
analysis, both adapters, and the fixed analysis-vector test module. Any change
to those files invalidates dependent baseline and phase evidence.

`record-baseline` is intentionally immutable: it accepts only the fully resolved
`3687f8b` source commit, all nine metrics, five fresh processes per metric, 20
warmups, and the canonical measured-unit counts. Inference consumes the complete
32-request set exactly once and compile cold-start consumes one unit. Validation
rejects metric subsets, changed protocol values, duplicate raw records, a
different canonical projection or logical work order, and environment or
software mismatches.

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
