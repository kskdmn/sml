# V2 benchmarks

The harness compares clean Git source commits through the same current native
runtime. Any commit that implements the current production API can be a baseline
or candidate. Source checkouts must support that API; there is no compatibility
adapter. Run commands from a clean checkout with Apple Metal access.

The canonical workload uses the current `ModelConfig` names, including
`hidden_size`, `num_layers`, `num_q_heads`, `num_kv_heads`, and `intermediate_size`.
Both sides receive identical deterministic parameters, inputs, and logical work
order. Training uses FP32 authoritative parameters and Adam moments with BF16
working parameters. The harness verifies parameter values, canonical projections,
input identities, native representations, and execution order before accepting
measurements. Its content identity covers the ordered source files listed in
`workload.HARNESS_COMPONENTS`; changing them requires a new baseline.

All nine metrics execute real production operations:

| Metric | Timed work | Measured units per process |
| --- | --- | --- |
| `prepared-data` | Production prepared-data stream and batch delivery | 100 |
| `pretraining-compute` | Optimizer steps with inputs transferred before timing | 20 |
| `pretraining-end-to-end` | Loader, model, gradients, and optimizer updates | 20 |
| `swag-end-to-end` | Encoded batch loading, LoRA ranking, and adapter updates | 20 |
| `inference-prefill` | Cache creation and cached model prefill | 32 fixed requests |
| `inference-decode` | Fixed decode transitions for cached requests | 32 fixed requests |
| `checkpoint-pause` | Serialization, durable publication, and obsolete-step pruning | 20 |
| `compile-cold-start` | First compiled invocation in a fresh process | 1 |
| `peak-metal-memory` | Peak allocation during an end-to-end optimizer step | 1 |

Each process synchronizes MLX at timing boundaries. Except for cold compilation,
it performs one untimed compilation pass and five warmup units. Cold compilation
uses zero warmups. Inference measures encoded model primitives; tokenization and
public-session scheduling are outside its timing boundaries. Decode uses a fixed
transition count without early EOS termination. Each checkpoint publication
follows a real optimizer update, which is outside the timed pause. Checkpoints
carry the fixed benchmark tokenizer specification; this metric does not load
SentencePiece. Prepared-data setup lives in `adapters/prepared_data.py` and calls
the production stream.

Baseline manifests use version 3. Capture requires all nine metrics, five fresh
processes per metric, and the counts above. Replace `FULL_GIT_COMMIT` with the
source commit to measure; omitting `--metrics` selects the complete metric set:

```sh
uv run python -m v2.benchmarks.runner record-baseline \
  --source-commit FULL_GIT_COMMIT \
  --manifest /private/tmp/sml-benchmark-baseline.json \
  --raw-output /private/tmp/sml-benchmark-baseline.jsonl \
  --state-directory /private/tmp/sml-benchmark-state \
  --pairs 5 --warmup 5 --measure 20 --prepared-data-measure 100

uv run python -m v2.benchmarks.runner validate \
  --manifest /private/tmp/sml-benchmark-baseline.json \
  --raw-input /private/tmp/sml-benchmark-baseline.jsonl
```

The state directory must be outside both the harness and source checkouts.
Manifest, raw-output, and state locations must be distinct and cannot contain
one another. The durable journal retains session metadata, preflights, child
measurements, immediate post-exit observations, recovery samples, accepted and
rejected attempts, and publication completion. Advisory locks serialize sessions
and shared output destinations. Atomic publication creates raw JSONL, then the
manifest, then the completion marker, after all 45 slots validate. Existing
identical output is reusable after a crash; conflicting bytes are never replaced.
Only recognized regular atomic temporary files are cleaned up on locked resume.

Repeat the same baseline command and state directory to resume. The harness
revalidates the complete journal against the same source and harness commits,
workload, protocol, hardware, software, and output paths, then reuses accepted
slots. Child measurements require the original immediate post-exit evidence; a
later invocation cannot recreate that observation. Complete evidence reconstructs
a trial deterministically. An unfinished warning-recovery sequence becomes an
interrupted attempt and starts no inherited stability window.

Strict environment checks cover hardware, software, connected power, power mode,
thermals, memory pressure, and competing GPU work. Child-start memory must be
normal. Child-end warning is admissible only with acceptable immediate parent
memory and all other checks passing; child-end critical is rejected. The parent
samples memory immediately after child exit, before slower environment probes.
Immediate warning triggers samples at least five seconds apart for at most five
minutes and requires 30 continuous normal seconds. The first terminal event ends
the sequence. Timeout, critical pressure, or interrupted recovery stops for manual
resume. Thermal-only failures use automatic recovery with five continuous nominal
minutes under a two-hour deadline for the missing slot. All samples are persisted
before classification and replayed during resume. Rejected attempts never enter
the baseline.

Comparisons rerun reference and candidate in fresh processes, alternating their
order by pair. Statistics use direction-normalized ratios and reproducible
whole-pair bootstrap samples. Screen mode requires five pairs, a 0.97 median
ratio gate, at most 2% dispersion, and a report-only confidence bound. Every
selected metric needs an explicit predecessor mapping: a report path or identity,
or `null` for its first measurement. For example:

```sh
uv run python -m v2.benchmarks.runner compare \
  --baseline /private/tmp/sml-benchmark-baseline.json \
  --candidate CANDIDATE_FULL_GIT_COMMIT \
  --metrics pretraining-compute,inference-prefill,inference-decode \
  --mode screen --pairs 5 --warmup 5 --measure 20 \
  --prepared-data-measure 100 --lower-bound-report-only \
  --predecessors '{"pretraining-compute":null,"inference-prefill":null,"inference-decode":null}' \
  --raw-output /private/tmp/sml-screen.jsonl \
  --output /private/tmp/sml-screen.json

uv run python -m v2.benchmarks.runner validate-phase \
  --phase 1 --baseline /private/tmp/sml-benchmark-baseline.json \
  --predecessors '{"pretraining-compute":null,"inference-prefill":null,"inference-decode":null}' \
  --results /private/tmp/sml-screen.json
```

`validate-phase` enforces the selected regression metric group and its required
predecessors. Groups are: 1, compute plus both inference metrics; 2, prepared data;
3, prepared data plus end-to-end pretraining, checkpoint pause, and peak memory;
4, both inference metrics; 5, SWAG. Group 3 requires a prepared-data predecessor;
group 4 requires both inference predecessors. The other groups require null
predecessors.

Final comparisons require all metrics except `pretraining-compute`, in canonical
order, with `--mode final --pairs 10 --maximum-dispersion 0.015
--pretraining-minimum-ratio 1.03` and required confidence lower bounds (omit
`--lower-bound-report-only`). Predecessors are required for prepared data,
end-to-end pretraining, SWAG, prefill, and decode; the other metrics use null.
Final validation reloads predecessor reports and requires the raw input to equal
the report's complete ordered trial set:

```sh
uv run python -m v2.benchmarks.runner validate-final \
  --baseline /private/tmp/sml-benchmark-baseline.json \
  --raw-input /private/tmp/sml-final.jsonl \
  --report /private/tmp/sml-final.json
```

A noisy comparison removes temporary checkouts, cools down for 15 minutes, and
repeats the complete alternating experiment once. Its last five cooldown minutes
must satisfy the recorded power mode, nominal thermals, normal memory, and no
competing GPU work. Reports retain both attempts and cooldown evidence; persistent
noise in any selected metric blocks acceptance. Checkpoint pause, cold compilation,
and peak-memory ratio gates are report-only. Final comparisons publish their valid
report and requested raw output before enforcing acceptance, so rejected results
remain available for inspection. Baseline, comparison, phase, final, and predecessor
validation all verify raw-trial version 3 identities and their complete embedded
post-exit recovery evidence before using measurements.

Quality checks use pretraining workload version 3 and SWAG workload version 2.
Both follow the production seed/microstep schedule for dropout. SWAG validation
uses forward passes with dropout disabled. Capture and validation accept only
the current paths below; canonical training and validation fixtures are checked
in. Pretraining publication recovery uses
`results/.pretraining-quality-v3.recording`.

```sh
uv run python -m v2.benchmarks.quality record --steps 1000 \
  --manifest v2/benchmarks/manifests/pretraining-quality-v3.json \
  --raw-output v2/benchmarks/results/pretraining-quality-v3.jsonl \
  --output v2/benchmarks/results/pretraining-quality-v3.json

uv run python -m v2.benchmarks.quality validate \
  --manifest v2/benchmarks/manifests/pretraining-quality-v3.json \
  --raw-input v2/benchmarks/results/pretraining-quality-v3.jsonl \
  --report v2/benchmarks/results/pretraining-quality-v3.json

uv run python -m v2.benchmarks.swag_quality record --steps 256 \
  --manifest v2/benchmarks/manifests/swag-quality-v2.json \
  --raw-output v2/benchmarks/results/swag-quality-v2.jsonl \
  --output v2/benchmarks/results/swag-quality-v2.json

uv run python -m v2.benchmarks.swag_quality validate \
  --manifest v2/benchmarks/manifests/swag-quality-v2.json \
  --raw-input v2/benchmarks/results/swag-quality-v2.jsonl \
  --report v2/benchmarks/results/swag-quality-v2.json
```
