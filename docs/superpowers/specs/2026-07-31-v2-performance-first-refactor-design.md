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
- tied/untied embedding configuration, with tied embeddings as the intended
  default mathematical constraint
- causal language-model loss semantics
- KV-cache semantics
- greedy and configured sampling semantics
- SentencePiece BPE as the tokenizer algorithm
- MLX-only model, training, and inference execution on Apple Silicon

Two mathematical corrections are explicitly authorized within this refactor:

- canonical tied-embedding gradients sum the input-embedding and output-
  projection contributions instead of discarding one contribution
- SWAG candidate scores use mean continuation-token log likelihood instead of
  the current length-sensitive sum

These corrections are part of the completed behavior and are tested against
their stated formulas rather than required to reproduce the corresponding
legacy training result. No other model-architecture, loss-objective, scoring, or
generation-algorithm changes are authorized.

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

During the phased migration, creating `v2/src/sml/` causes the package to take
precedence over the legacy `v2/src/sml.py` module. Phase 1 therefore installs a
temporary internal bridge in `sml.__init__` that exposes the model symbols still
needed by unmigrated flat workflows. Every legacy consumer has a package-import
smoke test until it is migrated. Phase 6 removes the bridge together with the
flat modules; it is never part of the completed public API.

## Artifact Model

Tokenizer, prepared-data, encoded-SWAG, and export bundles are immutable,
self-describing, and directory based. A run is a mutable model-state container
that is self-contained for model-consuming operations, with narrowly defined
mutable indexes: checkpoint directories may be appended, `latest.json` may be
atomically replaced, and retention may delete non-latest checkpoint directories
after a newer latest step has been published. Retention never deletes the
checkpoint currently named by `latest.json` and always performs latest-index
recovery before selecting deletion candidates. `run.json`, copied tokenizer and
base snapshots, and every published checkpoint directory are immutable once
written.

A step directory may reference immutable files at its owning run root and is
therefore not independently portable. A run directory and an export directory
are independently portable for inference, evaluation, and export operations
that use only model state. Training data is deliberately not copied into a run:
pretraining resume requires a prepared-data bundle, and LoRA resume requires an
encoded-SWAG bundle, whose authoritative bundle identity must match `run.json`.
A moved run may be given a new data-bundle location without changing its
semantic configuration. Run-step identity combines immutable `run.json` with
the selected checkpoint manifest as defined below; it is not a digest of the
mutable directory as a whole. Relative references and content identities are
canonical; absolute source paths are diagnostic locators only.

### Content identities and path safety

All SML-defined identities use SHA-256 and are rendered as `sha256:` followed by
64 lowercase hexadecimal digits. A file identity is the SHA-256 digest of its
exact bytes. A structured identity is the SHA-256 digest of a domain tag
followed by a null byte and canonical JSON bytes. Domain tags contain the
artifact kind, schema version, and identity-encoding version, so identical JSON
under different contracts cannot share an identity.

One artifacts helper is the only implementation of the project-specific
`sml-json-v1` canonical encoding. It normalizes dataclasses, enums, paths,
tuples, and schema-typed numeric configuration values; rejects non-finite
numbers and invalid Unicode; normalizes negative floating-point zero to zero;
and serializes with the pinned Python 3.12 JSON implementation using sorted
keys, compact separators, and UTF-8 without ASCII escaping. Arrays are
represented by typed metadata and content identities, never embedded JSON
values. Fixed project test vectors pin the canonical bytes and identities
across formatting, dictionary insertion order, and process restarts. This is a
versioned SML format, not a cross-language canonical-JSON promise.

Every manifest and `run.json` contains its own structured `identity` field.
Identity calculation excludes that self-referential field, creation timestamps,
temporary names, absolute source paths, and fields explicitly typed as
diagnostic locators. It includes the schema kind/version, all semantic
configuration, ordered relative logical paths, array metadata, and the
identities of referenced payloads. Changing whitespace or a diagnostic locator
cannot change an identity; changing a semantic field, relative logical path,
payload byte, dtype, shape, or order must change it.

Readers always recompute a manifest or `run.json` structured identity from its
parsed identity projection and compare it with the stored value. Normal startup
may trust the file identities recorded by a valid immutable manifest and avoid
rehashing large payloads; full verification recomputes every file identity.

Every logical payload path in a manifest uses `/`-separated relative form.
Readers reject absolute paths, empty components, `.` or `..` components,
platform-specific alternate separators, duplicate normalized paths, and any
path whose resolved location escapes the artifact root. SML-produced bundles
contain regular files and directories only. Readers reject symlinked payloads
or symlinked path components before opening a payload. Absolute source paths may
appear only in fields typed as diagnostic locators and are never followed as
artifact payload references.

Prepared data additionally has a semantic row-content identity independent of
its storage representation. It hashes a domain tag, row and column counts as
unsigned 64-bit little-endian integers, and the ordered row matrix converted to
little-endian `int32` C-order bytes. Its authoritative bundle identity also
includes shard boundaries, ordered relative shard paths, and shard file
identities. Resume therefore accepts a relocated byte-identical bundle but
rejects a differently sharded representation; performance comparisons may use
different representations only when their semantic row-content identities
match. Other bundles use their manifest identity as their authoritative bundle
identity. A run-step identity is a domain-separated structured identity over
the owning `run.json` identity and selected checkpoint-manifest identity.

### Tokenizer bundle

```text
tokenizer/
├── manifest.json
├── tokenizer.model
└── tokenizer.vocab
```

The typed manifest records the schema kind/version, SentencePiece BPE settings,
vocabulary size, BOS/EOS/PAD/UNK IDs, and content identities for both tokenizer
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
preparation seed and row-ordering policy, tokenizer identity, source summary,
the semantic row-content identity over ordered `int32` rows, and file identities
produced during writing. Prepared row order and semantic row-content identity
never depend on shard boundaries or a future runtime batch size.

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

Every fresh run publishes a canonical step-zero checkpoint containing the
initialized model or adapters, initialized optimizer state, initial cursor, and
next unused PRNG key before training begins. Run creation builds `run.json`,
copied immutable inputs, step zero, and `latest.json` in a temporary run
directory and atomically publishes that directory under the run writer lock.
A crash before the first optimizer step is therefore resumable; a crash before
run-directory publication leaves no visible run target.

`checkpoint.json` binds the owning `run.json` identity, checkpoint kind, step
number, `state.json` file identity, and every array file's identity, array keys,
shapes, and dtypes. `state.json` repeats the owning run identity and step so a
scalar-state file cannot be transplanted between checkpoints. Array readers
reject missing and additional keys as well as mismatched metadata.

`latest.json` is a derived mutable index, not authoritative checkpoint state.
It records the owning run identity, selected step number, and checkpoint-
manifest identity so readers can detect a stale, cross-run, or manually edited
pointer.

### LoRA run and export

A LoRA run has this self-contained model-state layout:

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
pretraining run. `run.json` records the copied base identity and authoritative
encoded-SWAG bundle identity. No copied base or tokenizer file may be a symlink
or external reference, and the LoRA run remains usable after the source
pretraining run is removed, provided the matching encoded-SWAG bundle remains
available when resuming training. Periodic step directories contain adapters,
optimizer state, trainer state, and cursor, but do not rewrite frozen base
weights. A selected step can be exported as:

```text
export/
├── manifest.json
├── tokenizer/
└── model.safetensors
```

The export manifest records its schema kind/version, complete model and
precision configuration, tokenizer identity, source run/step identity for
diagnostics, and content identities for the tokenizer files and merged model.
The exported model contains directly merged base-plus-LoRA weights under plain
inference parameter names. Export does not mutate or deep-copy the live model.

### Encoded SWAG bundle

An encoded SWAG cache is an immutable input artifact rather than hidden provider
state:

```text
swag-data/
├── manifest.json
└── buckets/
    ├── length-0064/
    │   ├── input_ids.npy
    │   ├── valid_token_mask.npy
    │   ├── score_mask.npy
    │   └── labels.npy
    └── ...
```

Each bucket stores uncompressed, memory-mappable arrays. `input_ids` is `int32`,
`valid_token_mask` and `score_mask` are Boolean, and all three have shape
`(examples, 4, bucket_length)`. `labels` is `int32` with shape `(examples,)` and
selects one of the four candidates. Token position zero is never scored.
Scoring uses `logits[..., :-1, :]`, targets `input_ids[..., 1:]`, and
`score_mask[..., 1:]`; the score mask is true for continuation tokens including
EOS and false for context and padding. The valid-token mask controls attention
and is true for every non-padding token.

The manifest records the complete preprocessing identity, bucket-boundary
policy, per-bucket counts and shapes, array dtypes, dropped-row counts, and file
identities. Bucket policy is part of the authoritative bundle identity, so a
cache is never silently rebucketed or reinterpreted.

### Atomicity and validation

All writable artifact operations require a local APFS filesystem. Each immutable
target has an exclusive advisory publication-lock sidecar in its parent
directory. A writer acquires that lock before inspecting the target, uses a
uniquely named temporary sibling directory, materializes and validates payload
files, writes the manifest last, `fsync`s files and directories, rechecks that
the target is absent, and publishes with one atomic directory rename. These
concurrency guarantees cover cooperating SML processes; external mutation of
the artifact namespace is treated as corruption, not as a supported concurrent
writer.

Readers ignore temporary directories and never accept a bundle without a
complete valid manifest. An identical existing target is an idempotent success;
a different target identity is a collision and is never overwritten. This
protocol applies to tokenizer, prepared-data, encoded-SWAG, and export bundles.
Creating a fresh mutable run is stricter: any existing target fails, and only an
explicit resume operation may open an existing run.

Checkpoint writers:

1. create a temporary sibling step directory
2. materialize, write, and validate all array files and scalar state
3. write the checkpoint manifest last
4. `fsync` the files and temporary directory
5. atomically rename the step directory and `fsync` its parent directory
6. write and `fsync` a temporary latest index, atomically replace `latest.json`,
   and `fsync` its parent directory

Readers ignore incomplete temporary directories. A completed checkpoint is
never modified or repaired in place.

The step-directory rename is the checkpoint commit point; `latest.json` is only
a recoverable index. When opening a run with a valid pointer, the reader fully
validates the pointed step and scans published `step-*` directories whose step
number is greater than the pointed step. When the pointer is missing, malformed,
or cross-run, the reader scans all published steps. In either case the highest
valid completed step belonging to the run is selected, and a stale or invalid
`latest.json` is atomically rebuilt. Temporary directories are ignored. A
malformed published directory that the normal selection algorithm must examine
raises an artifact error rather than being silently skipped; older steps below a
valid pointer are checked only by an explicit full-artifact verification.

Recovery does not depend on write access. Resume and other writable workflows
persist the repaired index before continuing; a read-only inference or
evaluation workflow uses the recovered step in memory and reports that the
stale index could not be persisted.

If a writer finds an existing target step, it compares the completed manifest
identity. An identical step is an idempotent success; a different identity is a
collision and fails without overwriting either checkpoint. Before publishing an
index, the writer performs recovery and advances `latest.json` only when the
target is not older than the recovered latest step. The index is never moved
backward. Tests inject interruption after every publication stage, including the
step rename immediately before the latest-index replacement.

Typed manifest parsing rejects unknown fields, unknown schema kinds or
versions, missing files, incorrect array keys/shapes/dtypes, inconsistent model
or tokenizer configuration, and cross-run state before GPU-heavy work begins.
Large-file digests are computed while artifacts are produced. Normal local
APFS startup trusts immutable manifests after schema, path, and array-metadata
validation. `sml verify --full` rereads and hashes every file. Read-only use on
another local filesystem has no concurrent writer or retention guarantee;
writable recovery, publication, and retention reject it.

### Concurrency and lifecycle

A run permits one mutable workflow at a time. Run creation, training, resume,
writable latest-index recovery, checkpoint publication, and retention hold an
exclusive advisory writer lock for their complete lifetime. The lock is a
sidecar in the run's parent directory, derived from the resolved run-directory
name, and is not part of the portable run or any identity. Lock acquisition is
non-blocking and a conflicting writer fails before GPU initialization with the
run path and available owner diagnostics. The macOS implementation uses a
kernel-managed file lock whose ownership is released automatically when the
process exits; lock-file contents are diagnostic and are never used for unsafe
PID-based stale-lock deletion.

Readers do not take the writer lock and may consume completed immutable steps
while training continues. They do take a shared sidecar access lock from step
resolution until all required state has been validated and fully evaluated into
owned arrays. Retention takes the corresponding exclusive access lock only while
deleting eligible steps, so it cannot remove an exact step between resolution
and loading. Latest-index replacement and checkpoint publication do not require
the exclusive access lock because readers validate immutable manifests and
recover stale indexes. Immutable-bundle writers use the target publication lock
and identity comparison, so concurrent production is either an idempotent
success or a collision rather than a partial overwrite.

Publication-lock files are durable sidecars and are not identities or payloads.
After acquiring a target's publication lock, a later writer may delete abandoned
temporary siblings only when each candidate is a direct, non-symlink directory
whose generated name contains the exact target-name digest and temporary marker.
The cleaner revalidates those properties immediately before deletion and never
follows entries outside the candidate directory. No per-temporary-directory
lifecycle locks or general destructive maintenance command are part of v2.

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

Preparation uses separate preallocated NumPy shuffle-window and output-shard
buffers with write cursors. Token ranges are checked in vectorized batches
rather than one Python integer at a time. Packing uses a stride of
`sequence_length`, so logically consecutive rows share the one boundary token
required by causal next-token training before any ordering step.

A versioned deterministic windowed-row shuffle permutes complete rows within
fixed preparation windows. The shuffle-window size and algorithm version are
semantic preparation configuration; output-shard size and future runtime batch
size are not. The shuffled logical row stream is then divided into contiguous
output shards. Changing only shard boundaries therefore preserves row order and
the semantic row-content identity. Training changes shard order
deterministically per epoch instead of performing millions of random row reads.
Preparation fails rather than publishing a bundle with no complete rows.

### Pretraining loading

Training memory-maps each shard. Most batches are complete contiguous
`(batch, sequence_length + 1)` slices that transfer to MLX once before adjacent
input and label views are derived. At a shard boundary, the bounded prefetcher
may combine the remaining rows with rows from the next shard in one contiguous
NumPy staging buffer; this is at most one copied batch per boundary. The loader
creates no per-row dictionaries, Python token lists, or general recollation
step.

A bounded CPU prefetcher touches/copies upcoming NumPy batches while the main
thread owns all MLX array creation. The throughput-first default treats the
epoch's deterministically ordered shards as one logical row stream and drops at
most one incomplete batch at the end of that stream, never one tail per shard.
Resume stores epoch, deterministic shard-order position, and row offset,
allowing constant-time continuation without replaying prior batches.

Before model or optimizer initialization, training validates that the bundle
contains at least one full runtime batch under the configured batch policy.

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

SWAG rows are encoded once. The authoritative bundle identity includes the
provider and dataset namespace/name, dataset configuration, resolved immutable
revision, provider fingerprint, provider-library version, split,
preprocessing-schema version, context/ending join policy, overlength policy,
BOS/EOS policy, tokenizer bundle identity, maximum sequence length, and bucket-
boundary policy. Any change to one of those fields is a cache miss; caches are
never reinterpreted under a newer preprocessing schema. Source loading and
tokenization are batched.

To preserve the current preprocessing contract, context and each normalized
ending are encoded separately and their token IDs are concatenated. EOS is
appended to every candidate when configured. If any of a row's four candidates
exceeds the maximum sequence length, the complete row is dropped; candidates
are never truncated independently. Cache construction validates four
candidates, label range, token range, EOS inclusion, and at least one scored
continuation token per candidate.

Cache production fails rather than publishing a bundle with no usable examples.

Examples are placed in the smallest configured finite length bucket that fits
their longest candidate. Batches pad only to their bucket shape, keep all four
candidates aligned, and use deterministic seed-and-epoch-derived bucket and row
permutations. The source dataset is not reloaded or retokenized inside the
training loop.

The configured SWAG batch dimension is fixed for every bucket kernel. A final
partial batch in a bucket is padded to that full dimension with finite synthetic
examples that contain at least one valid scored token, and carries a separate
on-device example mask. Per-example losses are finite before the example mask is
applied. Padded example slots contribute nothing to loss, accuracy, gradient
normalization, progress, or the committed cursor. Every real example is
therefore consumed exactly once per epoch without compiling a tail shape or
dropping one tail per bucket.

SWAG loss kernels return an additive loss sum and valid-example count. Gradient
accumulation sums gradients of those loss sums and divides once by the total
valid-example count at the optimizer step, so a small bucket tail cannot receive
the weight of a full microbatch. Resume state records epoch, deterministic
bucket-order position, and real-example row offset, never padded slots.

## Model Mathematics and Corrections

The refactor preserves deterministic forward results for the approved model
mathematics within dtype-appropriate tolerances.

### Canonical tied embeddings

The current implementation registers the tied vocabulary matrix twice in the
parameter tree. Autodiff therefore produces separate embedding and output-head
gradient leaves, after which retie logic keeps one update and discards the
other. It also creates duplicate optimizer state.

When `tie_word_embeddings` is true, the replacement registers exactly one
embedding parameter. The output projection applies that same matrix directly as
a linear projection, and the tied parameter uses the embedding weight-decay
policy. Its gradient must equal the sum of the two current leaves. This corrects
the implementation of the intended tied-weight mathematics rather than changing
the architecture. Untied configurations remain supported with an independent
output-head parameter and weight-decay policy.

### Boundary validation

Token range validation moves from every model forward to tokenizer, prepared
data, SWAG cache, and inference request boundaries. Forward, loss, attention,
and generation kernels contain no `.item()`, `.tolist()`, `mx.eval`, or other
host synchronization for validation.

### Explicit randomness

Model and LoRA initialization, dropout, and sampling consume explicit MLX PRNG
keys carried by run creation, trainer, or inference state. Because MLX
`nn.Dropout` does not accept an explicit key, model and LoRA layers use a small
keyed-dropout primitive implemented from `mx.random.bernoulli`, not
`nn.Dropout`. A training forward accepts a key, splits one subkey per active
dropout site in canonical layer order, and returns the next unused key as
explicit compiled state. Disabled dropout consumes no subkeys. Sampling follows
the same split-and-return rule.

Dataset permutations use isolated deterministic CPU RNG instances derived from
the configured seed and epoch. Checkpoints store the next unused trainer key,
and no resumable workflow relies on an unrecorded process-global random state.

## Training Runtime

### Precision policy

The default base-training policy uses:

- BF16 model weights and activations
- FP32 causal-loss and metric reductions
- FP32 gradient-accumulation buffers
- FP32 Adam moments

The default SWAG policy uses a frozen BF16 base and FP32 LoRA parameters,
gradients, reductions, and Adam moments. Precision is an explicit validated
configuration, and checkpoint state must match it exactly. LoRA casts its input
to FP32 before the adapter matmuls rather than relying on implicit mixed-dtype
promotion. Each adapter delta is then explicitly cast to the wrapped base
projection's output dtype before residual addition. This keeps base attention
and downstream activations BF16 while gradients and optimizer state for the FP32
adapter parameters remain FP32. Precision tests assert these dtypes at every
targeted projection.

### Compiled execution

Stable-shape prefill, decode, pretraining microstep, optimizer-step, and SWAG
ranking/update functions are compiled with MLX. Immutable parameter
classifications and weight-decay structures are built once.

Every compiled function is explicit about the mutable array state it uses.
Training functions declare model parameters, optimizer state, gradient-
accumulation buffers and counters, and PRNG keys; generation functions declare
token storage, KV-cache state, finished state, and per-request keys. State is
passed and returned directly or declared as MLX compile inputs and outputs.
Mutable arrays captured only through a Python closure are forbidden because MLX
may treat them as compile-time constants. Static validated configuration and
immutable parameter classifications may be captured. A synchronization barrier
evaluates all mutually consistent updated state together. Multi-step tests
compare eager and compiled execution and prove that a second compiled step
observes the state returned by the first.

Microbatch loss and gradients remain on-device. Gradient accumulation is
scheduled asynchronously, and the host synchronizes only at an optimizer-step
dependency, requested logging event, checkpoint, or final result. The runtime
does not call `loss.item()` for every microbatch.

Accumulation stores FP32 additive numerators and an explicit normalization
count. Pretraining uses the count of equal-shaped full microbatches; SWAG uses
the number of valid examples. Global-norm clipping and the optimizer update
occur only after this normalization. If an epoch ends with a nonempty but
incomplete accumulation window, the runtime divides by its actual count and
performs one optimizer update, preserving current end-of-epoch behavior without
overweighting a partial SWAG batch. It never carries an accumulation window
across an epoch boundary, and any checkpoint written afterward still contains
the canonical empty accumulation state.

The loops remain direct and task-specific. Logging, artifact I/O, dataset
iteration, and CLI concerns stay outside compiled kernels.

### SWAG scoring

For each labeled token, scoring computes:

```text
target_logit - logsumexp(all_logits)
```

It does not materialize a full log-probability tensor. Candidate scores are the
FP32 mean of valid continuation-token log probabilities, including EOS. This is
an explicitly authorized correction to the current summed score and removes the
sum's systematic score-magnitude dependence on continuation length. It is not a
legacy-equivalence requirement. One compiled kernel is cached per finite
length-bucket shape and configured batch size.

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

`generate_batch()` accepts unequal encoded prompt lengths and a generation
configuration for each request. Internal length buckets may pad for stable
compiled shapes, but every request carries its own valid-prefix length,
attention mask, logical position, KV-cache length, maximum-new-token limit, and
optional seed. Prefill gathers the next-token logits from each request's last
real token. Decode excludes padding cache slots and writes/rotates each token at
that request's logical position. Prompt overflow remains an error. Results are
restored to caller order after bucketing.

For seeded sampling, each request key is created directly from that request's
seed before bucketing and carried with the request. Calling `generate()` with
the same request configuration therefore uses exactly the same initial key as
its `generate_batch()` counterpart. Requests with equal seeds intentionally
start equal random streams. For an omitted seed, the request boundary allocates
a concrete seed before any reordering and returns it in result metadata so the
request can be reproduced. Bucket membership, batch neighbors, and internal
reordering cannot change a request's random stream. Equivalence tests compare
heterogeneous batched generation with corresponding serial requests for greedy
decoding, seeded sampling, EOS termination, repetition penalty, and no-repeat
n-gram processing.

Padding and bucketing are mathematically inert, but the design does not require
bitwise-identical BF16 logits from different kernel shapes. Serial and batched
scores use dtype-appropriate tolerances. Generated-token equality is required
for deterministic fixtures whose winning-token margins exceed those tolerances;
fixtures near a numerical selection boundary compare the logits and processor
masks instead.

Evaluation uses the same artifact resolution and model session. Log-likelihood
requests are length-bucketed and dynamically padded, target logits are gathered
on-device, and each batch synchronizes once. Each request carries its valid-token
mask and logical positions; padding is excluded from attention and scoring and
cannot change any real-token result beyond the same dtype-appropriate tolerance.
The adapter returns both continuation log likelihood and whether the continuation
matches greedy token selection. Generation requests use `generate_batch()`
rather than a request-serial loop. Equivalence tests compare heterogeneous
batched log-likelihood results with the corresponding serial requests for both
left- and right-padding layouts used by the adapter.

The completed v2 evaluation CLI supports the current HellaSwag and WinoGrande
scope. Its lm-eval adapter implements `loglikelihood` and `generate_until`;
unsupported request methods fail explicitly rather than returning partial or
invented results.

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
uv run python -m sml prepare pretraining
uv run python -m sml prepare swag \
  --checkpoint BASE_RUN \
  --output SWAG_BUNDLE
uv run python -m sml train \
  --data PRETRAINING_BUNDLE

uv run python -m sml infer \
  --checkpoint RUN_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint RUN_DIR \
  --task hellaswag

uv run python -m sml finetune \
  --checkpoint RUN_DIR \
  --data SWAG_BUNDLE

uv run python -m sml export \
  --checkpoint FINETUNE_RUN_DIR

uv run python -m sml infer \
  --checkpoint EXPORT_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint EXPORT_DIR \
  --task hellaswag

uv run python -m sml verify \
  --full \
  ARTIFACT
```

Initial `train` and `finetune` commands require explicit prepared-data and
encoded-SWAG bundles respectively. `infer` and `evaluate` require
`--checkpoint`; `--step` optionally selects an exact run step. A step directory
is accepted only in the context of its owning run and is never treated as a
standalone portable artifact. Evaluation accepts repeated `--task` arguments
from the supported HellaSwag and WinoGrande set. `verify --full` performs the
explicit payload-rehash operation described above.

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
logging interval, checkpoint interval, and retention. A resume command may also
supply a prepared-data or encoded-SWAG bundle location with `--data`; when it is
omitted, the original diagnostic locator is tried. A supplied location is not a
semantic override and is accepted only after its authoritative bundle identity
matches the dataset identity stored in `run.json`. Starting without `--resume`
never reuses, truncates, or overwrites an existing run target.

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

Before replacing the old implementation, deterministic legacy reference values
are captured wherever equivalence is required. The completed equivalence and
correction suite covers:

- model logits and causal loss
- YaRN correction ranges, positions, and rotated outputs
- GQA attention output
- sequential and chunked KV-cache output
- greedy and seeded sampled generation
- serial and unequal-length batched generation
- serial and dynamically padded batched log-likelihood
- tied embedding gradient sum
- LoRA forward and direct merged-weight output
- FP32 LoRA-state and BF16 base-activation dtype boundaries
- compiled consecutive-step state transitions against eager execution
- SWAG per-token log likelihood and authorized mean-normalized candidate scores

Dropout is disabled except in explicit PRNG/resume tests. Comparisons use exact
matches for integer/control results and dtype-appropriate numerical tolerances
for floating-point results. Serial/batched generation fixtures require token
equality only when the winning-token margin exceeds the applicable tolerance;
boundary fixtures compare logits and token-processor masks. Most fixtures
require legacy equivalence. The two authorized mathematical corrections instead
use direct formula oracles: tied embedding gradients must equal the sum of both
legacy leaves, and SWAG scores must equal the FP32 continuation-token mean.
Legacy corrected-path values are captured only to prove that the intended
correction is exercised, not as the expected result.

### Integration coverage

Tiny local artifacts exercise tokenizer training, preparation, base training,
resume, base inference, base evaluation, SWAG cache/fine-tuning, LoRA resume,
export, fine-tuned inference, and fine-tuned evaluation.

Resume tests compare uninterrupted and interrupted runs at optimizer-step
boundaries and after failure inside an uncommitted accumulation window,
including weights, optimizer, cursor, step, canonical accumulation state, and
explicit PRNG state. Artifact tests simulate interrupted writes, concurrent
publication, and cross-run state. The normal suite injects or caches external
dataset/evaluation providers and never requires network access.

The integration suite additionally proves:

- `sml-json-v1` identity test vectors are stable across JSON formatting and
  insertion order, ignore diagnostic locators, and change for every semantic or
  payload mutation
- manifest paths reject absolute, escaping, duplicate-normalized, and symlinked
  payload references before opening them
- normal validation detects schema and array-metadata corruption, while full
  verification detects a payload byte changed without a manifest update
- interruption at every immutable-bundle publication stage never exposes a
  partial bundle at its final path
- concurrent production of the same immutable target is either an idempotent
  success or a collision, never a partial overwrite
- fresh run creation atomically publishes step zero, rejects any existing
  target, and explicit resume before the first optimizer step restores the
  initialized state from that checkpoint
- a crash after step-directory publication but before `latest.json` replacement
  resumes from the newly published step
- a full prefetch queue cannot advance the checkpointed committed cursor
- a crash inside an accumulation window replays the complete uncommitted window
  with its original PRNG sequence
- shard-boundary batching drops at most one tail for the complete epoch stream
- conflicting reuse of an existing step number is rejected without overwrite
- idempotent publication of an older step cannot move `latest.json` backward
- deleting the source pretraining run does not affect LoRA resume, inference, or
  export from the copied base snapshot when the encoded-SWAG bundle is supplied
- a moved run performs model-only operations without its training data, resumes
  with an identity-matching relocated bundle, and rejects a mismatched bundle
- a second mutable workflow for the same run is rejected, concurrent readers
  can load published steps, and retention waits for an active step loader
- every SWAG cache-identity field produces a cache miss when changed
- SWAG cache fixtures pin separate context/ending encoding, mask alignment, EOS
  scoring, overlength-row drops, bucket placement, and stored array contracts
- padded SWAG bucket tails consume every real example exactly once, ignore
  finite synthetic slots, and produce the same example-weighted update as an
  unpadded eager reference
- unequal-length batched generation matches serial generation under the margin-
  aware equivalence rule and remains invariant to internal bucket ordering
- dynamically padded batched log-likelihood matches serial scoring within the
  configured tolerance and remains invariant to padding layout

The benchmark-analysis tests use fixed synthetic raw measurements to pin paired
ratio direction, bootstrap reproducibility, confidence-bound calculation, noise
rejection, and pass/fail/inconclusive decisions without running a performance
benchmark inside pytest.

## Performance Measurement and Acceptance

Performance claims require explicit Metal synchronization around timed regions
and compiled-kernel warmup before measurement. Before phase 1 begins, a
machine-readable baseline manifest and its raw measurements are committed. The
manifest records:

- source commit and clean-worktree proof, harness commit and clean-worktree
  proof, benchmark command, benchmark schema version, and content identity of
  the benchmark harness
- Apple chip/core count and unified memory, macOS build, power mode, and whether
  the machine was connected to power
- Python, MLX, NumPy, SentencePiece, and relevant provider versions
- complete model, optimizer, precision, loader, compilation, and generation
  configurations
- tokenizer, prepared-data representation, canonical training-row content, and
  SWAG dataset/cache identities
- warmup steps, measured steps or requests, synchronization points, trial count,
  and raw per-trial values

The benchmark harness runs from a dedicated clean checkout separate from the
clean source-under-test checkout. It is versioned independently and the
identical harness identity is used on both sides of any comparison. If the
harness or analysis changes, every affected baseline and retained phase result
is rerun; reports never compare measurements produced by different harness
identities. Its schema defines the exact start and end synchronization points,
included data movement and host work, state-reset policy, numerator, and work
unit for every metric.

In particular, each end-to-end pretraining work unit starts before requesting
the first microbatch in a measured accumulation window and ends after that
window's optimizer update is evaluated. It includes loader or prefetch stalls,
forward, backward, gradient accumulation, clipping, and the update, while
compilation, warmup, checkpointing, and unrelated logging are measured
separately.

The harness owns an implementation-independent benchmark-workload schema.
Version-specific adapters map that semantic workload to the legacy and
replacement configuration APIs outside timed regions. The manifest records the
canonical workload identity plus each side's resolved native configuration for
diagnostics; native serialization syntax need not match across the clean break.
A comparison is valid only when both adapters round-trip to the same canonical
semantic workload. Version-native representations may differ wherever the
metric contract explicitly permits them; prepared-data storage is one such
difference, not a special global exception. Each metric defines its equivalence
boundary:

- compute-only metrics receive tensors with the same semantic values, shapes,
  valid-work masks, and work counts
- end-to-end loading and training metrics intentionally use each version's
  native prepared-data or encoded-dataset representation and include the work
  named by the timed-region contract
- every native representation must resolve to the same canonical token rows,
  examples, requests, and real-work counts for that metric
- when work order can affect the executed kernels or state evolution, both
  adapters emit the same logical row, example, or request order

The acceptance baseline uses commit `3687f8b` on the Apple M5 10-core CPU,
10-core GPU, 24 GB target. The fixed pretraining workload uses that commit's
default model configuration with vocabulary size 28,672, hidden size 768, 12
layers, 12 query heads, 3 KV heads, intermediate size 2,176, sequence length
1,024, microbatch size 1, gradient accumulation 8, and BF16 compute.

The prerequisite creates one canonical ordered `int32` row matrix from a fixed
tokenizer and source-corpus sample, then serializes that matrix into paired
legacy NPZ/`uint16` and replacement NPY/`int32` benchmark bundles without
changing row order. The baseline manifest records the semantic row-content
identity and each representation's own file identities; the replacement
prepared-data manifest records the same semantic row-content identity.
Benchmarks reject different row-content identities while allowing the expected
representation identities to differ. SWAG and inference workloads similarly
record version-native representations while proving identical canonical
examples or requests. Other benchmark workloads likewise use identical
canonical workload configurations and semantic input identities on both sides.

Steady-state phase screens use five fresh-process comparison pairs; final
acceptance measurements use ten. Each pair contains one reference and one
candidate process, with reference-first and candidate-first order alternating
between pairs. Each process performs one untimed compilation pass, 20
synchronized warmup work units, and 100 synchronized measured work units. Each
benchmark defines its work unit as a data batch, optimizer step, SWAG batch,
prefill request batch, or decode chunk. Benchmarks that cannot supply 100 work
units use the complete fixed request set and record its size.

For throughput, every pair produces the direction-normalized ratio
`candidate / reference`. Reports include every raw result, per-side medians and
median absolute deviations, every paired ratio, the median paired ratio, and a
one-sided 95-percent lower confidence bound for that median. The bound uses a
10,000-resample percentile bootstrap over whole pairs with a fixed seed recorded
in the benchmark manifest. Phase screens report the bound but do not gate on it;
final acceptance does. The analysis implementation and fixed statistical test
vectors are part of the versioned harness identity. Compile cold-start time is
measured separately in a fresh process without warmup and is report-only.

A phase screen is too noisy when `MAD / median` exceeds 2 percent for either
side or for the paired ratios. Final acceptance uses a 1.5-percent threshold.
After cooldown the complete comparison is repeated once; persistent excess
dispersion blocks the phase or final acceptance. A final point estimate that
satisfies a gate while its required lower confidence bound does not is reported
as inconclusive and blocks acceptance rather than being rounded to a pass or
resolved by selecting favorable trials.

Benchmarks cover:

- prepared-data batches per second
- end-to-end pretraining tokens per second
- SWAG examples per second
- inference prefill tokens per second
- inference decode tokens per second
- checkpoint pause duration
- peak Metal memory

Performance benchmarks remain outside ordinary pytest. Each relevant phase
records before/after results using the pinned manifest and identical harness. A
result is invalid if the target enters critical memory pressure, thermal
throttling is detected, the power mode changes, or the canonical workload
configurations or semantic content identities differ. Native configuration and
representation identities may differ only within the metric-specific boundaries
described above.

A benchmark is relevant to a phase when the phase changes code executed in its
timed region. For every relevant steady-state throughput metric, the five-pair
phase-screen median must be at least `0.97` in direct comparisons against both
the pinned baseline and the previous accepted phase; improvements have no upper
bound. Checkpoint pause and peak memory are report-only except for the explicit
fit and memory-pressure gate. All measurements run from the clean checkouts at
the recorded commits.

Acceptance gates:

- every relevant phase satisfies the per-phase throughput rule above
- in the completed refactor, every steady-state throughput metric's ten-pair
  median and one-sided 95-percent lower confidence bound are both at least
  `0.97` against the pinned baseline
- the completed refactor's median paired end-to-end pretraining ratio and its
  one-sided 95-percent lower confidence bound must both be at least `1.03`
  against commit `3687f8b`
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

Creating and committing the versioned benchmark harness, pinned baseline
manifest, and raw baseline results is a prerequisite, not an implementation
phase. No performance-sensitive source change begins before that record exists.

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

- changing the model architecture or mathematical algorithms beyond the
  canonical tied-gradient and mean-normalized SWAG-score corrections authorized
  above
- replacing SentencePiece BPE
- supporting non-MLX training/inference backends
- compatibility with any existing v2 artifact or API
- adding a generic training framework or plugin system
- unrelated top-level repository or dependency changes
