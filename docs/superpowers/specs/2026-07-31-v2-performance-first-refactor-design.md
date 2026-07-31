# V2 Performance-First Refactor Design

## Status

Approved umbrella design for a clean replacement of the entire `v2` tree.
This specification supersedes the former checkpoint/SWAG-only design and plan.

## Goal

Refactor all v2 capabilities into a cohesive MLX package optimized first for
Apple-Silicon training throughput, while preserving the current model
mathematics and SentencePiece BPE algorithm.

The completed system retains these capabilities:

- tokenizer training
- pretraining-data preparation
- base-model training and resume
- base-model inference and evaluation
- SWAG LoRA fine-tuning and resume
- fine-tuned inference and evaluation

## Compatibility and Scope

This is an intentional clean break. The completed v2 provides no compatibility
for existing:

- Python import paths or flat script entrypoints
- CLI flags or defaults
- standalone tokenizer inputs
- prepared NPZ datasets or manifests
- pretraining or SWAG checkpoints
- run metadata or resume state

There are no legacy readers, aliases, conversion paths, warning-based
downgrades, or fallback schema interpretations. Temporary coexistence during
implementation is permitted only to keep each commit runnable; all replaced
code is removed at final cutover.

The following remain unchanged by design:

- the Transformer language-model equations
- YaRN/RoPE equations and position handling
- grouped-query attention
- RMSNorm and SwiGLU
- tied input/output embeddings as an intended mathematical constraint
- causal language-model loss semantics
- KV-cache semantics
- greedy and configured sampling semantics
- SentencePiece BPE as the tokenizer algorithm
- MLX-only model, training, and inference execution on Apple Silicon

The user explicitly authorized the top-level `pyproject.toml` packaging/source
mapping needed for `uv run python -m sml`. No unrelated top-level changes or
dependency changes are part of this refactor, and `uv.lock` is not changed
unless a separately approved dependency change becomes necessary.

## Primary Optimization Target

Highest end-to-end training throughput is the primary objective. When
throughput and peak memory conflict, throughput wins provided the current
default model and batch configuration still fit the target Apple-Silicon
hardware.

The acceptance hardware is an Apple M5 with 10 CPU cores (4 performance and 6
efficiency cores), 10 GPU cores, and 24 GB of unified memory. Performance
comparisons use the pre-refactor implementation at commit `3687f8b` as the
source baseline. The benchmark protocol below pins the remaining workload and
environment identity; results from other Apple-Silicon systems are informative
but do not satisfy the acceptance gate.

Correctness is a hard constraint, not a tradeoff. Performance changes must
retain mathematical equivalence where specified and must be measured against a
recorded baseline.

## Package Architecture

Replace the flat `v2/src/*.py` layout with a real package:

```text
v2/src/sml/
├── __init__.py
├── __main__.py          unified command entrypoint
├── cli.py               subcommand parsing and dispatch
├── errors.py            domain error taxonomy
├── model/
│   ├── __init__.py
│   ├── config.py        model and generation configuration
│   ├── rope.py          YaRN/RoPE mathematics
│   ├── layers.py        normalization, attention, MLP, blocks
│   ├── cache.py         KV-cache state and storage
│   ├── generation.py    token processors and selection
│   └── language_model.py
├── data/
│   ├── __init__.py
│   ├── corpus.py        compressed JSONL discovery/filtering
│   ├── tokenizer.py     SentencePiece training and loading
│   ├── pretraining.py   shard production, mmap loading, prefetch
│   └── swag.py          SWAG encoding cache and bucketed batches
├── artifacts/
│   ├── __init__.py
│   ├── manifest.py      typed schemas and content identities
│   └── checkpoint.py    atomic checkpoint/run/export I/O
├── training/
│   ├── __init__.py
│   ├── common.py        precision, schedules, gradients, progress
│   ├── pretrain.py      specialized base-training runtime
│   ├── lora.py          adapter layers, state, and merged export
│   └── swag.py          specialized ranking runtime
├── inference.py         persistent inference sessions
└── evaluation.py        batched lm-eval adapter and result writing
```

Configuration dataclasses live with their owners. Shared training primitives
are extracted only when both training workflows require the same operation.
The pretraining and SWAG loops remain specialized; the design rejects a generic
pipeline, callback, registry, or plugin framework in performance-critical code.

Dependency direction is one-way:

```text
cli
 ├── data workflows
 ├── training workflows
 ├── inference
 └── evaluation

training/inference/evaluation
 ├── model
 ├── data
 └── artifacts
```

One workflow never imports another workflow or a CLI module. Optional heavy
dependencies are imported lazily by the command that needs them.

## Artifact Model

Tokenizer, prepared-data, and export bundles are immutable, self-describing,
and directory based. A run is a mutable, self-contained container with narrowly
defined mutable indexes: checkpoint directories may be appended, `latest.json`
may be atomically replaced, and retention may delete non-latest checkpoint
directories after a newer latest step has been published. Retention never
deletes the checkpoint currently named by `latest.json` and always performs
latest-index recovery before selecting deletion candidates. `run.json`, copied
tokenizer and base snapshots, and every published checkpoint directory are
immutable once written.

A step directory may reference immutable files at its owning run root and is
therefore not independently portable. A run directory and an export directory
are independently self-contained and portable. Run identity is the identity of
immutable `run.json` plus the selected checkpoint manifest, not a digest of the
mutable directory as a whole. Relative references and content identities are
canonical; absolute source paths are diagnostic only.

### Tokenizer bundle

```text
tokenizer/
├── manifest.json
├── tokenizer.model
└── tokenizer.vocab
```

The typed manifest records the schema kind/version, SentencePiece BPE settings,
vocabulary size, BOS/EOS/PAD IDs, and content identities for both tokenizer
files. Downstream commands accept the bundle directory, never a loose `.model`
path.

### Prepared pretraining bundle

```text
pretraining-data/
├── manifest.json
├── tokenizer/
│   └── ... copied tokenizer bundle ...
└── shards/
    ├── train-000000.npy
    └── ...
```

The manifest records sequence length, row width, dtype, shard row counts,
preparation seed and ordering policy, tokenizer identity, source summary, a
canonical digest over the ordered `int32` rows independent of shard boundaries,
and file identities produced during writing.

### Pretraining run

```text
run/
├── run.json
├── tokenizer/
├── latest.json
└── checkpoints/
    └── step-000010000/
        ├── checkpoint.json
        ├── model.safetensors
        ├── optimizer.safetensors
        ├── trainer.safetensors
        └── state.json
```

`run.json` contains the immutable run configuration and dataset identity.
`state.json` contains scalar progress and the data cursor. MLX array state,
including the explicit PRNG key, uses safetensors. Checkpoints are committed at
optimizer-step boundaries, so the saved accumulation state is canonical and
empty; the microstep counter still proves exact boundary alignment.

`latest.json` is a derived mutable index, not authoritative checkpoint state.
It records the selected step number and checkpoint-manifest identity so readers
can detect a stale, cross-run, or manually edited pointer.

### LoRA run and export

A LoRA run has this self-contained layout:

```text
lora-run/
├── run.json
├── tokenizer/
│   └── ... copied tokenizer bundle ...
├── base/
│   ├── manifest.json
│   └── model.safetensors
├── latest.json
└── checkpoints/
    └── step-000001000/
        ├── checkpoint.json
        ├── adapters.safetensors
        ├── optimizer.safetensors
        ├── trainer.safetensors
        └── state.json
```

At run creation, `base/model.safetensors` is copied exactly once from the
selected pretraining step. `base/manifest.json` contains the complete model and
precision configuration, source run/step identity for diagnostics, and the
copied file's content identity. The tokenizer bundle is copied from the owning
pretraining run. No copied base or tokenizer file may be a symlink or external
reference, and the LoRA run remains usable after the source pretraining run is
removed. Periodic step directories contain adapters, optimizer state, trainer
state, and cursor, but do not rewrite frozen base weights. A selected step can
be exported as:

```text
export/
├── manifest.json
├── tokenizer/
└── model.safetensors
```

The exported model contains directly merged base-plus-LoRA weights under plain
inference parameter names. Export does not mutate or deep-copy the live model.

### Atomicity and validation

Checkpoint writers:

1. create a temporary sibling step directory
2. write and evaluate all array files and scalar state
3. write the checkpoint manifest last
4. `fsync` the files and temporary directory, or use the platform durability
   equivalent
5. atomically rename the step directory and `fsync` its parent directory
6. write and `fsync` a temporary latest index, atomically replace `latest.json`,
   and `fsync` its parent directory

Readers ignore incomplete temporary directories. A completed checkpoint is
never modified or repaired in place.

The step-directory rename is the checkpoint commit point; `latest.json` is only
a recoverable index. When opening a run with a valid pointer, the reader scans
published `step-*` directories whose step number is greater than the pointed
step. When the pointer is missing, malformed, or cross-run, the reader scans all
published steps. In either case the highest valid completed step belonging to
the run is selected, and a stale or invalid `latest.json` is atomically rebuilt.
Temporary directories are ignored, while malformed published step directories
raise an artifact error rather than being silently skipped.

Recovery does not depend on write access. Resume and other writable workflows
persist the repaired index before continuing; a read-only inference or
evaluation workflow uses the recovered step in memory and reports that the
stale index could not be persisted.

If a writer finds an existing target step, it compares the completed manifest
identity. An identical step is an idempotent success and only the latest index
is refreshed; a different identity is a collision and fails without overwriting
either checkpoint. Tests inject interruption after every publication stage,
including the step rename immediately before the latest-index replacement.

Typed manifest parsing rejects unknown fields, unknown schema kinds or
versions, missing files, incorrect array keys/shapes/dtypes, inconsistent model
or tokenizer configuration, and cross-run state before GPU-heavy work begins.
Large-file digests are computed while artifacts are produced. Normal local
startup trusts immutable manifests; an explicit verification operation may
reread and hash every file.

## Data Flow and Loading

The primary flows are:

```text
raw corpus
  -> tokenizer bundle
  -> prepared pretraining bundle
  -> pretraining run/checkpoints
  -> base inference/evaluation

HF SWAG + base checkpoint
  -> cached encoded SWAG bundle
  -> LoRA run/checkpoints
  -> merged export
  -> fine-tuned inference/evaluation
```

### Pretraining preparation

Prepared shards are contiguous, uncompressed NPY arrays with dtype `int32` and
shape `(rows, sequence_length + 1)`. Spending additional disk space avoids NPZ
decompression and a recurring uint16-to-int32 conversion in every epoch.

Preparation uses a preallocated NumPy shard buffer and write cursor. Token
ranges are checked in vectorized batches rather than one Python integer at a
time. The one-token causal overlap between adjacent rows is preserved.
Contiguous batch groups are shuffled during preparation; training changes shard
order deterministically per epoch instead of performing millions of random
row reads.

### Pretraining loading

Training memory-maps each shard, slices a complete
`(batch, sequence_length + 1)` array, transfers it to MLX once, and derives
adjacent input and label views. It creates no per-row dictionaries, Python token
lists, or recollation step.

A bounded CPU prefetcher touches/copies upcoming NumPy batches while the main
thread owns all MLX array creation. The throughput-first default drops an
incomplete tail batch to keep compiled shapes stable. Resume stores epoch,
deterministic shard-order position, and row offset, allowing constant-time
continuation without replaying prior batches.

The prefetch producer position is never checkpoint state. Every queued batch is
an immutable envelope containing the NumPy slice and its cursor-after value.
The training loop tracks a separate committed cursor. For an accumulation
window, it retains the cursor-after value of the last consumed microbatch and
publishes that value only after the optimizer update and its MLX state have
successfully evaluated. A failure before that point replays the entire
uncommitted accumulation window against unchanged weights. Checkpoints serialize
only the committed cursor, so an arbitrarily full prefetch queue cannot skip
data after resume.

### SWAG preparation and loading

SWAG rows are encoded once. Cache identity includes the provider and dataset
namespace/name, dataset configuration, immutable revision and provider
fingerprint, split, preprocessing-schema version, context/ending join policy,
truncation policy, BOS/EOS policy, tokenizer bundle identity, and maximum
sequence length. Any change to one of those fields is a cache miss; caches are
never reinterpreted under a newer preprocessing schema. Source loading and
tokenization are batched. Cache construction validates four candidates, label
range, EOS inclusion, and at least one scored continuation token.

Examples are grouped by their longest candidate into a finite set of length
buckets. Batches pad only to their bucket shape, keep all four candidates
aligned, and use deterministic epoch ordering. The source dataset is not
reloaded or retokenized inside the training loop.

## Model Mathematics and Corrections

The refactor preserves deterministic forward results for the approved model
mathematics within dtype-appropriate tolerances.

### Canonical tied embeddings

The current implementation registers the tied vocabulary matrix twice in the
parameter tree. Autodiff therefore produces separate embedding and output-head
gradient leaves, after which retie logic keeps one update and discards the
other. It also creates duplicate optimizer state.

The replacement registers exactly one embedding parameter. The output
projection applies that same matrix as a linear projection. Its gradient must
equal the sum of the two current leaves. This corrects the implementation of
the intended tied-weight mathematics rather than changing the architecture.

### Boundary validation

Token range validation moves from every model forward to tokenizer, prepared
data, SWAG cache, and inference request boundaries. Forward, loss, attention,
and generation kernels contain no `.item()`, `.tolist()`, `mx.eval`, or other
host synchronization for validation.

### Explicit randomness

Dropout and sampling consume explicit MLX PRNG keys carried by trainer or
inference state. Keys are split deterministically and stored in checkpoints.
No workflow relies on an unrecorded process-global random state for resumable
behavior.

## Training Runtime

### Precision policy

The default base-training policy uses:

- BF16 model weights and activations
- FP32 causal-loss and metric reductions
- FP32 gradient-accumulation buffers
- FP32 Adam moments

The default SWAG policy uses a frozen BF16 base and FP32 LoRA parameters,
gradients, reductions, and Adam moments. Precision is an explicit validated
configuration, and checkpoint state must match it exactly.

### Compiled execution

Stable-shape prefill, decode, pretraining microstep, optimizer-step, and SWAG
ranking/update functions are compiled with MLX. Immutable parameter
classifications and weight-decay structures are built once.

Microbatch loss and gradients remain on-device. Gradient accumulation is
scheduled asynchronously, and the host synchronizes only at an optimizer-step
dependency, requested logging event, checkpoint, or final result. The runtime
does not call `loss.item()` for every microbatch.

The loops remain direct and task-specific. Logging, artifact I/O, dataset
iteration, and CLI concerns stay outside compiled kernels.

### SWAG scoring

For each labeled token, scoring computes:

```text
target_logit - logsumexp(all_logits)
```

It does not materialize a full log-probability tensor. Candidate scores are the
FP32 mean of valid continuation-token log probabilities, including EOS. This
removes systematic preference for shorter endings. One compiled kernel is
cached per finite length-bucket shape.

### Checkpoint timing

Training checkpoint intervals count completed optimizer steps. A resume whose
restored step already satisfies its limit returns before data iterators,
compiled gradients, or model training mode are constructed.

## Inference and Evaluation Runtime

`InferenceSession.from_checkpoint()` loads and validates a pretraining run, a
step within its owning run, a LoRA run, or a self-contained merged export once.
It owns the tokenizer, model, compiled prefill/decode functions, token storage,
and KV cache.

The public runtime supports `generate()` and `generate_batch()`. Generation:

- preallocates token and KV-cache capacity
- avoids concatenating the complete prefix for each token
- tracks finished/EOS state on-device
- decodes in compiled chunks and synchronizes once per chunk
- implements repetition penalty and no-repeat-ngram processing without Python
  conversion of device arrays
- preserves greedy and seeded sampling behavior through explicit PRNG keys

`generate_batch()` accepts unequal encoded prompt lengths. Internal length
buckets may pad for stable compiled shapes, but every request carries its own
valid-prefix length, attention mask, logical position, and KV-cache length.
Prefill gathers the next-token logits from each request's last real token.
Decode excludes padding cache slots and writes/rotates each token at that
request's logical position; padding must not change any generated token or
score. Results are restored to caller order after bucketing.

Sampling assigns one explicit key to each request before bucketing and carries
that key with the request, so bucket membership or internal reordering cannot
change its random stream. Equivalence tests compare heterogeneous batched
generation with the corresponding serial requests for greedy decoding, seeded
sampling, EOS termination, repetition penalty, and no-repeat n-gram processing.

Evaluation uses the same artifact resolution and model session. Log-likelihood
requests are length-bucketed and dynamically padded, target logits are gathered
on-device, and each batch synchronizes once. Generation requests use
`generate_batch()` rather than a request-serial loop.

Both inference and evaluation can consume:

- a pretraining run directory, resolving `latest.json`
- an exact pretraining step directory within its owning run
- a LoRA run directory, resolving base plus latest adapters
- a merged export directory

An optional step selector chooses an exact step inside a run.

## Unified CLI and Configuration

The top-level package mapping and `sml.__main__` support:

```sh
uv run python -m sml tokenize
uv run python -m sml prepare
uv run python -m sml train

uv run python -m sml infer \
  --checkpoint RUN_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint RUN_DIR \
  --task hellaswag

uv run python -m sml finetune \
  --checkpoint RUN_DIR

uv run python -m sml export \
  --checkpoint FINETUNE_RUN_DIR

uv run python -m sml infer \
  --checkpoint EXPORT_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint EXPORT_DIR \
  --task hellaswag
```

`infer` and `evaluate` require `--checkpoint`; `--step` optionally selects an
exact run step. Evaluation accepts repeated `--task` arguments.

Each command accepts an optional TOML configuration file plus explicit CLI
overrides. Precedence is defaults, then TOML, then CLI. Frozen validated
dataclasses own configuration; derived values are read-only properties rather
than constructor mutations.

Training configurations compose shared optimizer, precision, loader, and
checkpoint policies. They do not inherit from one another. Throughput-critical
controls are available without editing source, including batch size, sequence
length, accumulation, precision, prefetch depth, compilation, maximum steps,
logging, checkpoint interval, and seed.

CLI modules only parse, construct typed configurations, import workflow code
lazily, dispatch, print typed results, and map expected domain failures to exit
codes. `argparse.Namespace` values never enter domain APIs.

### Resume configuration

`train --resume RUN_DIR` and `finetune --resume RUN_DIR` recover and resolve the
latest atomically published checkpoint. Saved model, optimizer, dataset
identity, precision, and immutable run configuration are authoritative. Resume
may override only termination and observability fields: maximum steps/epochs,
logging interval, checkpoint interval, and retention.

## Error Handling

Configuration, artifact, data, and runtime failures use focused domain
exceptions. Library APIs raise them; the CLI renders concise actionable
messages and nonzero exit codes. Unexpected exceptions retain full tracebacks.

Validation happens before GPU-heavy initialization whenever possible. Errors
identify the exact field, file, array, shape, dtype, content identity, dataset
revision, or task that violated the contract. Missing uncached network
resources identify the required provider and cache key.

The system never silently coerces incompatible artifact state, substitutes a
different tokenizer, skips an unknown configuration field, or guesses a
checkpoint kind.

## Testing Structure

Tests mirror the package:

```text
v2/tests/
├── conftest.py
├── unit/
│   ├── model/
│   ├── data/
│   ├── artifacts/
│   ├── training/
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_cli.py
├── equivalence/
└── integration/
```

Shared fixtures replace repeated path manipulation, MLX probes, tiny configs,
and fake tokenizers. Tests use package imports rather than editing `sys.path`.

### Equivalence coverage

Before replacing the old implementation, deterministic fixtures capture:

- model logits and causal loss
- YaRN correction ranges, positions, and rotated outputs
- GQA attention output
- sequential and chunked KV-cache output
- greedy and seeded sampled generation
- serial and unequal-length batched generation
- tied embedding gradient sum
- LoRA forward and direct merged-weight output
- SWAG continuation scores

Dropout is disabled except in explicit PRNG/resume tests. Comparisons use exact
matches for integer/control results and dtype-appropriate numerical tolerances
for floating-point results.

### Integration coverage

Tiny local artifacts exercise tokenizer training, preparation, base training,
resume, base inference, base evaluation, SWAG cache/fine-tuning, LoRA resume,
export, fine-tuned inference, and fine-tuned evaluation.

Resume tests compare uninterrupted and interrupted runs at optimizer-step
boundaries, including weights, optimizer, cursor, step, canonical accumulation
state, and explicit PRNG state. Artifact tests simulate interrupted writes and
cross-run state. The normal suite injects or caches external dataset/evaluation
providers and never requires network access.

The integration suite additionally proves:

- a crash after step-directory publication but before `latest.json` replacement
  resumes from the newly published step
- a full prefetch queue cannot advance the checkpointed committed cursor
- conflicting reuse of an existing step number is rejected without overwrite
- deleting the source pretraining run does not affect LoRA resume, inference, or
  export from the copied base snapshot
- every SWAG cache-identity field produces a cache miss when changed
- unequal-length batched generation matches serial generation and remains
  invariant to internal bucket ordering

## Performance Measurement and Acceptance

Performance claims require explicit Metal synchronization around timed regions
and compiled-kernel warmup before measurement. Before phase 1 begins, a
machine-readable baseline manifest and its raw measurements are committed. The
manifest records:

- source commit, dirty-worktree state, benchmark command, and benchmark schema
  version
- Apple chip/core count and unified memory, macOS build, power mode, and whether
  the machine was connected to power
- Python, MLX, NumPy, SentencePiece, and relevant provider versions
- complete model, optimizer, precision, loader, compilation, and generation
  configurations
- tokenizer, prepared-data representation, canonical training-row content, and
  SWAG dataset/cache identities
- warmup steps, measured steps or requests, synchronization points, trial count,
  and raw per-trial values

The acceptance baseline uses commit `3687f8b` on the Apple M5 10-core CPU,
10-core GPU, 24 GB target. The fixed pretraining workload uses that commit's
default model configuration with vocabulary size 28,672, hidden size 768, 12
layers, 12 query heads, 3 KV heads, intermediate size 2,176, sequence length
1,024, microbatch size 1, gradient accumulation 8, and BF16 compute.

The prerequisite creates one canonical ordered `int32` row matrix from a fixed
tokenizer and source-corpus sample, then serializes that matrix into paired
legacy NPZ/`uint16` and replacement NPY/`int32` benchmark bundles without
changing row order. The baseline manifest records the canonical row digest and
each representation's own file identities; the replacement prepared-data
manifest records the same canonical row digest. Benchmarks reject different
canonical row identities while allowing the expected representation identities
to differ. Other benchmark workloads likewise use identical serialized
configurations and semantic input identities on both sides.

Steady-state measurements use five fresh-process trials. Each trial performs one
untimed compilation pass, 20 synchronized warmup work units, and 100
synchronized measured work units. Each benchmark defines its work unit as a
data batch, optimizer step, SWAG batch, prefill request batch, or decode chunk.
Benchmarks that cannot supply 100 work units use the complete fixed request set
and record its size. Reports include every raw result, the median, and median
absolute deviation. Compile cold-start time is measured separately in a fresh
process without warmup.

Benchmarks cover:

- prepared-data batches per second
- end-to-end pretraining tokens per second
- SWAG examples per second
- inference prefill tokens per second
- inference decode tokens per second
- checkpoint pause duration
- peak Metal memory

Performance benchmarks remain outside ordinary pytest. Each relevant phase
records before/after results using the pinned manifest. A result is invalid if
the target enters critical memory pressure, thermal throttling is detected, the
power mode changes, or the serialized configurations or content identities
differ outside the explicitly paired legacy/new prepared-data representations.

A benchmark is relevant to a phase when the phase changes code executed in its
timed region. Every relevant training-throughput median must remain within 3
percent of both the pinned baseline and the previous accepted phase; the final
end-to-end pretraining gate below still requires an improvement over baseline.
Acceptance measurements run from clean worktrees at the recorded commits.

Acceptance gates:

- every relevant phase satisfies the per-phase throughput rule above
- the completed refactor must improve median end-to-end pretraining throughput
  by at least 3 percent over commit `3687f8b`
- the fixed default workload must complete on the Apple M5 24 GB target without
  out-of-memory failure or critical memory pressure
- mathematical-equivalence and correctness tests must pass regardless of speed

Compile cold-start time and peak memory are reported even when steady-state
throughput is the deciding metric.

## Verification Commands

Python commands use Python 3.12.13 through `uv run`. MLX pytest and benchmarks
run outside the sandbox so Metal is available.

Every phase runs:

```sh
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
```

The final phase additionally runs every performance comparison and all unified
CLI workflow smoke tests.

## Delivery Sequence

Creating and committing the pinned baseline manifest and raw baseline results is
a prerequisite, not an implementation phase. No performance-sensitive source
change begins before that record exists.

One master plan index links six ordered implementation plans:

1. Foundation and model package
2. Tokenizer, artifacts, and prepared data
3. Pretraining runtime
4. Inference and evaluation
5. LoRA and SWAG
6. Unified CLI and final cutover

Each phase is independently reviewable, testable, benchmarked where relevant,
and committed before the next begins. Migrated and unmigrated workflows may
coexist through phase 5. Phase 6 deletes all replaced flat source modules,
legacy tests, old entrypoints, and compatibility scaffolding, then updates v2
documentation to describe only the new system.

## Out of Scope

- changing the model architecture or mathematical algorithms
- replacing SentencePiece BPE
- supporting non-MLX training/inference backends
- compatibility with any existing v2 artifact or API
- adding a generic training framework or plugin system
- unrelated top-level repository or dependency changes
