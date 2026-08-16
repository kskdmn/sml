# V2 Prepared-Data 100-Unit Measurement Design

## Context

The reviewed stop-interruptible shutdown fix at
`8e2291f1c87244fa8acd33d375c9e49961bb70fa` removed the candidate loader's
fixed shutdown delay. A fresh Phase 2 comparison using 20 measured batches per
trial then showed strong throughput but failed the unchanged dispersion gate:

- attempt 0 median candidate/reference ratio: `2.0102596413753253`;
- attempt 0 ratio MAD: `0.041183156349385186`;
- harness-owned retry median ratio: `2.05762452965475`;
- harness-owned retry ratio MAD: `0.04980990612059033`;
- required maximum ratio MAD: `0.02`.

All 20 raw trials, the 30-second launch gate, and the harness-owned 15-minute
cooldown were nominal: AC connected, automatic power mode, low-power mode off,
thermal state nominal/raw `0`, normal memory pressure, and no competing GPU
workload. The final comparison decision was therefore `too-noisy`, not a
thermal or power rejection.

At the observed medians, 20 measured batches cover only about `0.7 ms` on the
candidate and `1.4 ms` on the reference. Fixed scheduling and timer variation
therefore represent a meaningful fraction of each measured interval.

## Goal

Increase the measured work inside each prepared-data trial from 20 to 100
batches so each timing interval contains five times more real loader work,
while preserving the full 1,024-token workload and every existing acceptance
threshold and evidence rule.

## Chosen Protocol

The next Phase 2 comparison changes exactly one comparison parameter:

```text
--measure 20  ->  --measure 100
```

Every measured unit remains one real prepared-data batch with microbatch size
1 and sequence length 1,024. The replacement continues to exercise the real
`PretrainingBatchStream`, consumer-side MLX transfer/evaluation, and stream
shutdown before the measured call returns. The reference and candidate receive
the same 100-unit work request.

The following protocol fields remain unchanged:

- screen mode;
- 5 paired comparisons per attempt;
- 5 warmup units;
- 10,000 bootstrap resamples;
- median ratio floor `0.97`;
- ratio MAD ceiling `0.02`;
- lower confidence bound report-only;
- predecessor mapping `{"prepared-data": null}`;
- pinned baseline manifest `baseline-3687f8b.json`;
- canonical `TMPDIR=/private/tmp`;
- harness-owned statistical-noise retry and cooldown only.

This is a duration increase, not a workload-shape change or threshold
relaxation. It is expected to lengthen each measured interval by roughly five
times, but it does not guarantee that dispersion will pass.

## Evidence Isolation and Execution

The rejected 20-unit report with comparison identity
`sha256:1b19e60760cd84144395eb70cf001bd683a2afc552dfa68e344c3902bc09ea58`
must be moved from the public result path to an ignored preservation path
before the new run. It must never be staged, validated, merged with new trials,
or used as resumable evidence.

The 100-unit attempt starts from zero after:

1. a clean candidate/status check;
2. Ruff and full v2 pytest verification;
3. confirmation that `uv.lock` is unchanged;
4. a fresh 30-continuous-second launch gate requiring AC connected, automatic
   power mode, low-power mode off, thermal nominal/raw `0`, normal memory
   pressure, and no competing GPU workload.

The comparison is non-resumable. The operator must not manually retry or
salvage trials. Only the harness may perform its configured statistical-noise
retry and cooldown.

## Validation and Acceptance

Independent Phase 2 validation runs only if the fresh comparison's final
prepared-data decision is `pass`. Acceptance still requires:

- median candidate/reference ratio at least `0.97`;
- ratio MAD no greater than `0.02`;
- valid canonical workload, baseline, harness, candidate, trial, and
  environment identities;
- every retained trial passing its power, thermal, memory, and competing-work
  checks;
- an empty predecessor set for prepared-data.

Only a passing comparison and passing independent validation may publish and
commit `phase-2-loader.json` and `phase-2.json`. Any gate timeout, comparison
rejection, environment rejection, or validation failure blocks staging and
commit. Nothing is pushed as part of this protocol task.

## Rejected Alternatives

### Aggregate several 20-unit submeasurements

This would require benchmark-harness code and a new aggregation/statistical
contract. It is unnecessary when the existing measured-unit interface can
increase real work directly.

### Relax the dispersion ceiling

Changing the `0.02` threshold after observing a rejection would weaken the
prospective acceptance contract. The threshold remains unchanged.

### Reuse or merge rejected trials

The comparison is explicitly non-resumable. Reusing the 20-unit trials would
mix different protocols and invalidate the evidence identity.

## Verification and Success Criteria

- The written plan and exact comparison command use `--measure 100` once and
  contain no active `--measure 20` command for this attempt.
- The rejected 20-unit report remains preserved outside the public result
  paths and unstaged.
- Preflight and the 30-second nominal launch gate pass before comparison.
- Every fresh trial records `measured_units=100` and retains the 1,024-token
  prepared-data workload identity.
- Comparison and validation both decide `pass` under the unchanged `0.97` and
  `0.02` gates before evidence is staged.
- Exactly the two accepted Phase 2 reports are committed; `uv.lock` and
  production code remain unchanged.
