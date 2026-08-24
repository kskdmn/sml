# V2 Performance-First Refactor Design

## Status

Approved umbrella design for a clean replacement of the entire `v2` tree.
This specification supersedes the former checkpoint/SWAG-only design and plan.

**Execution update (2026-08-24):** Phases 1-3 are complete at `4c190f3`.
Phase 4 is complete at `0d95441`. Task 5.1 is complete at `232918c`.
Task 5.2 is complete at `4a8e469`. Task 5.3 is complete at `c06beae`.
Task 5.4 is complete at `4ebb4b7`. Task 5.5 is complete at `e89ce2b`.
Task 6.1 is complete at `9aa0f18`. Task 6.2 is next.

**Acceptance-policy update (2026-08-22):** Functional correctness is the
required refactor gate. Ruff, the full v2 test suite, controlled mathematical
and training-quality checks, and relevant end-to-end/CLI smoke workflows must
pass. Before/after performance comparisons, baseline capture, statistical
thresholds, thermal launch windows, and benchmark evidence commits are
optional diagnostics. Missing, noisy, thermally rejected, or slower benchmark
results do not block this refactor or progression between phases.

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

One additional precision-policy correction is explicitly authorized: base
training uses FP32 master parameters as the authoritative optimizer state,
derives BF16 working parameters for forward/backward compute, keeps Adam moments
in FP32, and performs accumulation, clipping, weight decay, and Adam arithmetic
in FP32. After every optimizer update it deterministically derives a new BF16
working tree from the updated FP32 masters. This changes the numerical training
trajectory and is not a legacy-equivalence requirement.

These corrections are part of the completed behavior and are tested against
their stated formulas rather than required to reproduce the corresponding
legacy training result. No other model-architecture, loss-objective, scoring,
generation-algorithm, or optimizer-algorithm changes are authorized.

The user explicitly authorized the top-level `pyproject.toml` packaging/source
mapping needed for `uv run python -m sml`. No unrelated top-level changes or
dependency changes are part of this refactor, and `uv.lock` is not changed
unless a separately approved dependency change becomes necessary.

## Primary Optimization Target

Highest end-to-end training throughput is the primary objective. When
throughput and peak memory conflict, throughput wins provided the current
default model and batch configuration still fit the target Apple-Silicon
hardware.

The reference hardware for optional diagnostics is an Apple M5 with 10 CPU
cores (4 performance and 6 efficiency cores), 10 GPU cores, and 24 GB of
unified memory. A performance investigation may compare with the pre-refactor
implementation at commit `3687f8b` using the protocol below, but neither that
comparison nor any particular ratio is required for acceptance.

Correctness and controlled training quality are hard constraints, not
tradeoffs. Performance-sensitive changes must retain mathematical equivalence
where specified and pass the quality gates below. Measurement against a
recorded baseline is optional.

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

Every implementation-plan interface is complete at the task that first owns
it. A named dataclass or wrapper must list all fields and types; a named function
must define its complete signature, verification behavior, and ownership
semantics. This applies in particular to verified/published artifact wrappers,
tokenizer and prepared-data bundle results, KV and token-selection results,
compiled-kernel host results, resolved provider records, export results, and
recursive verification results. A later task may extend a discriminated schema
only by introducing a new schema kind/version; it may not invent optional fields
inside an already frozen kind.

Artifact schemas are discriminated rather than one optional-field container.
`PretrainingRunManifest` and `LoRARunManifest` are distinct strict run kinds;
`PretrainingCheckpointManifest` and `LoRACheckpointManifest` are distinct strict
checkpoint kinds. Tokenizer, prepared-data, base-snapshot, SWAG-data, export,
latest-index, and evaluation-result schemas likewise have exact field sets,
payload references, identity projections, and versions before their first
writer is implemented. Strict readers reject fields from another kind.

During the phased migration, creating `v2/src/sml/` causes the package to take
precedence over the legacy `v2/src/sml.py` module. Phase 1 therefore installs a
temporary internal bridge in `sml.__init__` that exposes the model symbols still
needed by unmigrated flat workflows. Every legacy consumer has a package-import
smoke test until it is migrated. Phase 6 removes the bridge together with the
flat modules; it is never part of the completed public API.

## Artifact Model

Tokenizer, prepared-data, encoded-SWAG, and export bundles are immutable,
self-describing, and directory based. A run is a mutable model-state container
that is self-contained for model-consuming operations. Each pretraining or
fine-tuning run retains exactly one latest committed checkpoint bundle in steady
state. Checkpoint publication may temporarily retain the previous committed
bundle until the replacement is durable and `latest.json` names it; the writer
then waits for active readers under the exclusive access lock and deletes the
previous bundle before the checkpoint operation completes. A crash may leave
the latest bundle plus one obsolete predecessor, which the next writable open
must verify and prune. Checkpoint history, configurable retention, historical
step selection, and standalone step artifacts are not part of v2.

`run.json`, copied tokenizer and base snapshots, and a committed checkpoint
directory are immutable once written. A checkpoint directory may reference
immutable files at its owning run root and is therefore not independently
portable. A run directory and an export directory are independently portable
for inference, evaluation, and export operations that use only model state.
Training data is deliberately not copied into a run: pretraining resume requires
a prepared-data bundle, and LoRA resume requires an encoded-SWAG bundle, whose
authoritative bundle identity must match `run.json`. A moved run may be given a
new data-bundle location without changing its semantic configuration. The
current run-state identity combines immutable `run.json` with the latest
checkpoint manifest; it is not a digest of the mutable directory as a whole.
Relative references and content identities are canonical; absolute source paths
are diagnostic locators only. A future fine-tuning method creates its own run
kind and owns one latest checkpoint under the same bounded-publication rule.

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
parsed identity projection and compare it with the stored value. Correctness-
sensitive workflows that will mutate or derive model state -- fresh training,
resume, fine-tuning, export, writable recovery, and obsolete-checkpoint pruning
-- fully rehash every referenced input payload before GPU initialization or
destructive action.
An immutable target that already exists must also pass full payload verification
before a writer may accept it as an idempotent success. Payloads produced and
hashed by the same live writer need not be reread immediately.

Read-only inference and evaluation default to manifest-trusted validation and
avoid rehashing large payloads; their APIs and CLI accept `full_verify=True` /
`--full` to request payload rehashing. Every inference result and evaluation
result records whether the selected model state was `full` verified or
`manifest-trusted`. `sml verify --full` remains the standalone recursive
payload-verification command.

Every logical payload path in a manifest uses `/`-separated relative form. Each
component must match the portable-ASCII grammar
`[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?`, which also forbids a trailing
dot. Readers normalize before validating that grammar and reject absolute paths,
empty components, `.` or `..` components, platform-specific alternate
separators, duplicate paths after Unicode normalization and case folding, and
any path whose resolved location escapes the artifact root. After opening all
payloads, readers also reject distinct logical paths that resolve to the same
`(device, inode)` pair. Every regular payload file must also have link count one,
so a hard link outside the bundle cannot provide a second mutation path.

SML-produced bundles contain regular files and directories only. The artifact
root itself is opened with `O_DIRECTORY | O_NOFOLLOW` and validated with `fstat`.
Readers then walk payload paths with `openat`-style operations relative to that
descriptor. Intermediate components are opened with
`O_DIRECTORY | O_NOFOLLOW`; final payloads are opened with `O_NOFOLLOW`, and
every opened object is validated with `fstat` before use. Readers never perform
a symlink check followed by a separate ordinary path open. Writers, renames, and
temporary-directory cleanup use the same descriptor-relative discipline.
Absolute source paths may appear only in fields typed as diagnostic locators and
are never followed as artifact payload references.

Prepared data additionally has a semantic row-content identity independent of
its storage representation. It hashes a domain tag, row and column counts as
unsigned 64-bit little-endian integers, and the ordered row matrix converted to
little-endian `int32` C-order bytes. Its authoritative bundle identity also
includes shard boundaries, ordered relative shard paths, and shard file
identities. Resume therefore accepts a relocated byte-identical bundle but
rejects a differently sharded representation; performance comparisons may use
different representations only when their semantic row-content identities
match. Other bundles use their manifest identity as their authoritative bundle
identity. A current run-state identity is a domain-separated structured identity
over the owning `run.json` identity and latest checkpoint-manifest identity.

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
    └── step-000010000/        latest committed checkpoint only
        ├── checkpoint.json
        ├── model.safetensors
        ├── master.safetensors
        ├── optimizer.safetensors
        ├── trainer.safetensors
        └── state.json
```

`run.json` contains the immutable run configuration and dataset identity.
`model.safetensors` contains the BF16 working parameters used by inference and
the next forward/backward call. `master.safetensors` contains the authoritative
FP32 trainable parameters. `state.json` contains scalar progress and the data
cursor. Remaining MLX array state, including the explicit PRNG key, uses
safetensors. Checkpoints are committed at optimizer-step boundaries, so the
saved accumulation state is canonical and empty; the microstep counter still
proves exact boundary alignment. Only the latest committed checkpoint is kept;
the directory's step number records progress but does not provide historical
selection.

Every fresh base run publishes a canonical step-zero checkpoint containing an
FP32 master tree initialized from the exact BF16 working tree, that BF16 working
tree, initialized optimizer state, initial cursor, and next unused PRNG key
before training begins. LoRA step zero analogously contains its FP32 adapters
and optimizer state. Run creation builds `run.json`, copied immutable inputs,
step zero, and `latest.json` in a temporary run directory and atomically
publishes that directory under the run writer lock. A crash before the first
optimizer step is therefore resumable; a crash before run-directory publication
leaves no visible run target.

`checkpoint.json` binds the owning `run.json` identity, checkpoint kind, step
number, `state.json` file identity, and every array file's identity, array keys,
shapes, and dtypes. `state.json` repeats the owning run identity and step so a
scalar-state file cannot be transplanted between checkpoints. Array readers
reject missing and additional keys as well as mismatched metadata. A base
checkpoint additionally proves that every BF16 working leaf equals the explicit
BF16 cast of its corresponding FP32 master leaf; a mismatch is corruption.

`latest.json` is a derived mutable index, not authoritative checkpoint state.
It records the owning run identity, selected step number, and checkpoint-
manifest identity so readers can detect a stale, cross-run, or manually edited
pointer. A healthy quiescent run contains the pointed checkpoint and no other
published `step-*` directory.

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
    └── step-000001000/        latest committed checkpoint only
        ├── checkpoint.json
        ├── adapters.safetensors
        ├── optimizer.safetensors
        ├── trainer.safetensors
        └── state.json
```

At run creation, `base/model.safetensors` is copied exactly once from the
latest pretraining checkpoint. `base/manifest.json` contains the complete model
and precision configuration, source current-run-state identity for diagnostics,
and the copied file's content identity. The tokenizer bundle is copied from the
owning pretraining run. The source run's shared access lock is held from latest
resolution through full verification and completion of the exact BF16 file-byte
copy; the target run writer lock does not substitute for that source lock.
`run.json` records the copied base identity and authoritative
encoded-SWAG bundle identity. No copied base or tokenizer file may be a symlink
or external reference, and the LoRA run remains usable after the source
pretraining run is removed, provided the matching encoded-SWAG bundle remains
available when resuming training. Each replacement checkpoint contains
adapters, optimizer state, trainer state, and cursor, but does not rewrite
frozen base weights. The latest checkpoint can be exported as:

```text
export/
├── manifest.json
├── tokenizer/
└── model.safetensors
```

The export manifest records its schema kind/version, complete model and
precision configuration, tokenizer identity, source current-run-state identity
for diagnostics, and content identities for the tokenizer files and merged
model.
The exported model contains directly merged base-plus-LoRA weights under plain
inference parameter names. For each adapter, export deterministically computes

```text
delta_weight_fp32 = scale_fp32 * (B_fp32 @ A_fp32)
merged_weight_bf16 =
    cast_bf16(cast_fp32(base_weight_bf16) + delta_weight_fp32)
```

and stores the merged weight as BF16. Live adapters deterministically compute

```text
adapter_fp32 = scale_fp32 * ((input_fp32 @ A_fp32.T) @ B_fp32.T)
output_bf16 = base_output_bf16 + cast_bf16(adapter_fp32)
```

The merged-weight formula is an exact array oracle; merged-model output is
compared with live-adapter output using pinned BF16 tolerances because the two
rounding locations are intentionally different. Export does not mutate or
deep-copy the live model.

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

Before creating or resuming a LoRA run, fine-tuning requires exact equality
between the selected base snapshot and encoded-SWAG bundle for tokenizer bundle
identity, vocabulary size, BOS/EOS/PAD/UNK IDs, and the preprocessing fields
derived from those values. The cache maximum sequence length must not exceed the
base model's effective context length. A mismatch fails before base weights,
adapters, or optimizer state are allocated.

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
7. acquire the exclusive access lock, revalidate the new latest checkpoint,
   delete every older published step by descriptor, and `fsync` the checkpoint
   parent before returning

Readers ignore incomplete temporary directories. A completed checkpoint is
never modified or repaired in place.

The step-directory rename is the checkpoint commit point; `latest.json` is only
a recoverable index. Latest resolution validates every published `step-*`
directory needed to establish the highest valid completed step belonging to the
run. A missing, malformed, stale, or cross-run pointer is recovered from those
immutable manifests. A malformed published candidate raises an artifact error
rather than being silently skipped. Temporary directories are ignored.

Recovery does not depend on write access. Resume and other writable workflows
persist the repaired index, take the exclusive access lock, and prune every
obsolete published checkpoint before allocating training state. A read-only
inference or evaluation workflow never writes or prunes; it uses the recovered
latest checkpoint in memory and reports whether repair or pruning remains
pending. Thus normal operation retains one checkpoint, while an interrupted
publication can leave at most the previous and replacement checkpoints until
the next writable recovery.

If a writer finds an existing target step, it compares the completed manifest
identity and fully verifies the existing step's payloads. An identical, fully
verified step is an idempotent success; a different identity or payload mismatch
is a collision or corruption error and fails without overwriting either
checkpoint. Before publishing an index, the writer performs recovery and
rejects a target older than the recovered latest step. The index is never moved
backward. Tests inject interruption after every publication and pruning stage,
including the step rename immediately before the latest-index replacement and
the index replacement immediately before obsolete-checkpoint deletion.

Typed manifest parsing rejects unknown fields, unknown schema kinds or
versions, missing files, incorrect array keys/shapes/dtypes, inconsistent model
or tokenizer configuration, and cross-run state before GPU-heavy work begins.
Large-file digests are computed while artifacts are produced. Correctness-
sensitive workflows perform the full input verification defined above; normal
read-only local-APFS startup may trust immutable manifests after schema, path,
and array-metadata validation. Read-only use on another local filesystem has no
concurrent writer or pruning guarantee; writable recovery, publication, and
obsolete-checkpoint pruning reject it.

### Concurrency and lifecycle

A run permits one mutable workflow at a time. Run creation, training, resume,
writable latest-index recovery, checkpoint publication, and obsolete-checkpoint
pruning hold an exclusive advisory writer lock for their complete lifetime. The
lock is a sidecar in the run's parent directory, derived from the resolved
run-directory name, and is not part of the portable run or any identity. Lock
acquisition is non-blocking and a conflicting writer fails before GPU
initialization with the run path and available owner diagnostics. The macOS
implementation uses a kernel-managed file lock whose ownership is released
automatically when the process exits; lock-file contents are diagnostic and are
never used for unsafe PID-based stale-lock deletion.

Readers do not take the writer lock and may consume a completed immutable latest
checkpoint while training continues. They take a shared sidecar access lock
from checkpoint resolution until all required state has been validated and
fully evaluated into owned arrays. The writer takes the corresponding exclusive
access lock before deleting the previous checkpoint, so it cannot remove a
checkpoint that an active reader is still loading. Latest-index replacement and
new-checkpoint publication do not require the exclusive access lock; pruning
does. Immutable-bundle writers use the target publication lock and identity
comparison, so concurrent production is either an idempotent success or a
collision rather than a partial overwrite.

Before pruning an older checkpoint, the writer proves that the new latest
checkpoint is payload-valid. A checkpoint produced and hashed by the same live
writer already carries that proof. After resume or writable recovery, the
workflow fully rehashes the preexisting latest checkpoint before deleting an
obsolete predecessor. Manifest-trusted read-only resolution never grants
deletion authority. Lock ordering is always run-writer lock, source-run shared
access lock when copying a base, then target-run exclusive access lock for
pruning; no workflow acquires those locks in reverse order.

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
  -> pretraining run/latest checkpoint
  -> base inference/evaluation

HF SWAG + latest base checkpoint
  -> cached encoded SWAG bundle
  -> LoRA run/latest checkpoint
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

Before model or optimizer initialization, fresh training and resume fully hash
the prepared-data manifest's tokenizer and shard payloads, recompute the
manifest identities, and validate array metadata and bundle-wide token-range
invariants. The later hot loop relies on that verified immutable byte set and
does not repeat per-forward token-range synchronization.

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

A pretraining cursor is the canonical location of the next row that could enter
a runtime batch: `(epoch, shard_order_position, row_offset)`. `row_offset` is
strictly less than the current shard's row count; reaching a shard end
normalizes immediately to offset zero in the next ordered shard. A batch that
crosses a shard boundary records the normalized position after its final row.
The intentionally dropped incomplete epoch tail does not advance the committed
cursor because it produces no optimizer update. Resume deterministically
encounters and drops that same tail again before moving in memory to the next
epoch. The first optimizer update in the next epoch commits a cursor in that
epoch. This avoids both data skips and a progress-only checkpoint with a
different manifest at an already-published optimizer step.

### SWAG preparation and loading

SWAG rows are encoded once. The authoritative bundle identity includes the
provider and dataset namespace/name, dataset configuration, resolved immutable
revision, provider fingerprint, provider-library version, split,
preprocessing-schema version, context/ending join policy, overlength policy,
BOS/EOS policy, tokenizer bundle identity, maximum sequence length, and bucket-
boundary policy. Any change to one of those fields is a cache miss; caches are
never reinterpreted under a newer preprocessing schema. Source loading and
tokenization are batched.

Before base weights or adapters are allocated, fresh fine-tuning and resume
fully hash the selected base snapshot, copied tokenizer, encoded-SWAG arrays,
and any restored adapter/optimizer/trainer state, then apply the encoded-SWAG
compatibility checks above. The hot loop performs no redundant payload hashing.

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

A SWAG cursor likewise names the next real example. Reaching the end of a bucket
normalizes to offset zero in the next deterministic bucket; the padded synthetic
slots are never cursor positions. Consuming the final real example of the final
bucket normalizes to `(next_epoch, first_bucket, 0)`. Because the padded tail
still performs an optimizer update, that normalized epoch transition is
committed atomically with updated model, optimizer, and PRNG state.

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

The default base-training and inference dtype contract is:

| State or computation | Required dtype |
| --- | --- |
| authoritative trainable base-model master parameters | FP32 |
| base-model working parameters used by forward/backward and inference | BF16 |
| embeddings, linear outputs, residuals, MLP activations, Q/K/V, and model logits | BF16 |
| KV-cache keys and values | BF16 |
| YaRN/RoPE frequencies and cosine/sine caches | FP32, with rotated Q/K remaining BF16 |
| RMSNorm reduction | FP32 internally, BF16 output |
| attention softmax | FP32 internally, BF16 output |
| causal loss, SWAG log-sum-exp/scores, and metric reductions | FP32 |
| raw base-parameter gradient leaves | BF16 |
| gradient-accumulation buffers, normalized gradients, clipping, and global norm | FP32 |
| Adam first and second moments | FP32 |
| learning-rate, weight-decay, and Adam update arithmetic | FP32 |
| updated authoritative base master parameter | FP32 |
| post-update base working parameter | explicit BF16 cast of the updated master |
| token IDs, labels, positions, and device lengths | int32 |
| device optimizer-step and accumulation counters | int32, range-checked before overflow |
| checkpoint steps, epochs, data cursors, and host counts | nonnegative signed 64-bit integers |
| attention, score, valid-token, example, and finished masks | Boolean |
| explicit MLX PRNG keys | native uint32 key representation |

Autodiff may return BF16 gradient leaves for BF16 working parameters even when
the scalar loss is FP32; each leaf is cast to FP32 immediately before
accumulation. A project-owned mixed-precision Adam update, rather than an
unmodified stock `mlx.optimizers.Adam`/`AdamW` parameter update, computes for
each parameter:

```text
g_fp32 = normalized_and_clipped_accumulator_fp32
m_fp32 = beta1 * m_fp32 + (1 - beta1) * g_fp32
v_fp32 = beta2 * v_fp32 + (1 - beta2) * square(g_fp32)
updated_fp32 = adam_and_decoupled_weight_decay(
    master_parameter_fp32, m_fp32, v_fp32, step, config
)
master_parameter_fp32 = updated_fp32
working_parameter_bf16 = cast_bf16(master_parameter_fp32)
```

The exact bias-correction, epsilon, schedule, and per-parameter weight-decay
semantics remain those of the saved optimizer configuration. The optimizer must
not rely on implicit mixed-dtype promotion, and every compiled update returns
an FP32 master tree, its derived BF16 working tree, and an FP32 moment tree.
BF16 training uses no dynamic loss scaler. The master tree is checkpointed; it
is never reconstructed from BF16 working weights during resume.

The default SWAG policy uses a frozen BF16 base and FP32 LoRA parameters, raw
gradients, accumulation, reductions, and Adam moments. LoRA casts its input to
FP32 before adapter matmuls and applies the live-adapter formula specified in the
export section. This keeps base attention, KV state, and downstream activations
BF16 while adapter state remains FP32. Precision is explicit validated semantic
configuration, and checkpoint state must match it exactly.

Precision tests assert all listed dtypes after initialization, two consecutive
eager and compiled updates, checkpoint save/load, and interrupted resume. They
prove that sub-BF16-ULP updates accumulate in the FP32 masters, that sensitive
leaves such as RMSNorm scales eventually move, that every BF16 working leaf is
the exact cast of its master, and that no uncheckpointed master state exists.

### Compiled execution

Stable-shape prefill, decode, pretraining microstep, optimizer-step, and SWAG
ranking/update functions are compiled with MLX. Immutable parameter
classifications and weight-decay structures are built once.

Every compiled function is explicit about the mutable array state it uses.
Training functions declare FP32 master parameters, BF16 working parameters,
optimizer state, gradient-accumulation buffers and counters, and PRNG keys;
generation functions declare token storage, KV-cache state, finished state, and
per-request keys. State is passed and returned directly as built-in array trees,
or declared through MLX's documented compile `inputs`/`outputs` state capture.
Compiled functions are pure with respect to uncaptured Python state: they never
call `model.update(...)`, mutate a captured module shell, or install arrays into
a host wrapper while MLX traces. Mutable arrays captured only through a Python
closure are forbidden because MLX may treat them as compile-time constants.
Static validated configuration and immutable parameter classifications may be
captured. A synchronization barrier evaluates all mutually consistent updated
state together. Multi-step tests compare eager and compiled execution, prove
that a second compiled step observes the state returned by the first, and prove
that tracing leaves no placeholder or stale array in an owning model/session.

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

Training checkpoint intervals count completed optimizer steps. Each successful
checkpoint operation replaces and prunes the previous checkpoint before
returning; checkpoint history and retention controls do not exist. A resume
whose restored step already satisfies its limit returns before data iterators,
compiled gradients, or model training mode are constructed.

## Inference and Evaluation Runtime

`InferenceSession.from_checkpoint()` loads and validates the latest checkpoint
of a pretraining run, the latest checkpoint of a LoRA run, or a self-contained
merged export once.
It owns the tokenizer, immutable loaded model state, compiled prefill/decode
functions, and a reusable buffer pool. Token storage, logical lengths, finished
state, per-request PRNG keys, and KV-cache contents belong to one generation
call and are reset before use. A failed call discards or clears its leased state
in `finally` before the session can be reused.

An `InferenceSession` is explicitly non-reentrant. A non-blocking in-process
call guard rejects overlapping `generate()` or `generate_batch()` calls with a
focused runtime error; callers that need concurrency create independent
sessions. Model arrays and compiled-function caches persist safely across
sequential calls, but no request-specific array state does.

The public runtime supports `generate()` and `generate_batch()`. Generation:

- preallocates token and KV-cache capacity
- avoids concatenating the complete prefix for each token
- tracks finished/EOS state on-device
- decodes in compiled chunks and synchronizes once per chunk
- implements repetition penalty and no-repeat-ngram processing without Python
  conversion of device arrays
- preserves greedy and seeded sampling behavior through explicit PRNG keys

`generate_batch()` accepts unequal encoded prompt lengths and a generation
configuration for each request. It selects two independent stable shapes. The
prefill-length bucket is the smallest configured prompt bucket that fits the
real encoded prompt and bounds only prefill token/mask/position work. The
cache-capacity bucket is the smallest configured capacity that fits
`prompt_token_count + max_new_tokens` and bounds token storage, KV storage, and
decode state. A short prompt with a large generation allowance therefore does
not execute prefill at the full cache-capacity length. Every request carries its
own valid-prefix length, attention mask, logical position, KV-cache length,
maximum-new-token limit, and optional seed. Prefill gathers the next-token
logits from each request's last real token. Decode excludes padding cache slots
and writes/rotates each token at that request's logical position. Prompt
overflow remains an error. Results are restored to caller order after
bucketing.

The request boundary applies configured BOS insertion before validation and
rejects any prompt that still encodes to zero tokens. This includes empty or
tokenizer-normalized-empty text when the selected tokenizer/model has no usable
BOS token. Empty lm-eval contexts use the model's configured prefix token and
fail before bucketing when none exists.

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

Artifact resolution completes once before the first evaluation request. The
persisted evaluation result records artifact kind, run identity when applicable,
resolved step, checkpoint-manifest identity, current run-state identity, tokenizer
identity, and payload-verification level. For each ordered task invocation it
also records a structured task identity over the resolved task YAML and complete
include/template closure, task metadata version, prompt/few-shot/generation and
metric-normalization configuration, seeds, limit, ordered request identity,
lm-eval package version and source commit when available, resolved dataset
revision/fingerprint, and relevant provider versions. Evaluation fails rather
than publishing a result when an authoritative task file, dataset revision, or
request identity cannot be resolved. A later `latest.json`, task-package, or
dataset-cache update cannot change or obscure which model and evaluation
definition produced the result. Inference result metadata records the same
resolved model identity and verification level in addition to any allocated
sampling seed.

The result contract is explicit rather than a provider-owned untyped mapping.
`EvaluationTaskRecord` contains task name, resolved task-YAML identity, ordered
include/template identities, task metadata version, prompt/few-shot/generation
configuration, metric-normalization configuration, seeds, limit, ordered
request identity, lm-eval package version/source commit, dataset
revision/fingerprint, sorted provider versions, and the task's complete metric
payload. `EvaluationResult` contains schema kind/version/identity, resolved
model identity, the ordered tuple of task records, and the complete top-level
provider result payload. Evaluation writes one immutable JSON result file by
temporary-file write, file `fsync`, no-replace publication, and parent `fsync`.
An identical fully validated target is idempotent; a different existing target
is a collision and is never overwritten.

The completed v2 evaluation CLI supports the current HellaSwag and WinoGrande
scope. Its lm-eval adapter implements `loglikelihood` and `generate_until`;
unsupported request methods fail explicitly rather than returning partial or
invented results.

Both inference and evaluation can consume:

- a pretraining run directory, resolving `latest.json`
- a LoRA run directory, resolving base plus latest adapters
- a merged export directory

Run-directory resolution always uses latest-checkpoint recovery. Historical
step paths and step selectors are rejected because runs intentionally retain no
checkpoint history.

## Unified CLI and Configuration

The top-level package mapping and `sml.__main__` support:

```sh
uv run python -m sml tokenize \
  --corpus RAW_CORPUS \
  --output TOKENIZER_BUNDLE
uv run python -m sml prepare pretraining \
  --corpus RAW_CORPUS \
  --tokenizer TOKENIZER_BUNDLE \
  --output PRETRAINING_BUNDLE
uv run python -m sml prepare swag \
  --checkpoint BASE_RUN \
  --revision IMMUTABLE_PROVIDER_COMMIT \
  --output SWAG_BUNDLE
uv run python -m sml train \
  --data PRETRAINING_BUNDLE \
  --output PRETRAINING_RUN

uv run python -m sml infer \
  --checkpoint RUN_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint RUN_DIR \
  --task hellaswag \
  --output BASE_EVALUATION.json

uv run python -m sml finetune \
  --checkpoint RUN_DIR \
  --data SWAG_BUNDLE \
  --output FINETUNE_RUN

uv run python -m sml export \
  --checkpoint FINETUNE_RUN_DIR \
  --output MERGED_EXPORT

uv run python -m sml infer \
  --checkpoint EXPORT_DIR \
  "Once upon a time"

uv run python -m sml evaluate \
  --checkpoint EXPORT_DIR \
  --task hellaswag \
  --output EXPORT_EVALUATION.json

uv run python -m sml verify \
  --full \
  ARTIFACT
```

Artifact-producing commands require explicit input and output paths; v2 never
writes to an implicit project or home-directory location. Initial `train` and
`finetune` commands require explicit prepared-data and encoded-SWAG bundles
respectively. `infer` and `evaluate` require `--checkpoint`; historical step
selectors and standalone step paths are invalid. Evaluation requires an
explicit output file and accepts repeated `--task` arguments from the supported
HellaSwag and WinoGrande set. `verify --full` performs the explicit
payload-rehash operation described above. Read-only `infer` and `evaluate` use
manifest-trusted validation by default; their optional `--full` flag rehashes
every payload needed by the selected model before GPU initialization.

Each command accepts an optional TOML configuration file plus explicit CLI
overrides. Precedence is defaults, then TOML, then CLI. Frozen validated
dataclasses own configuration; derived values are read-only properties rather
than constructor mutations. Before a run identity is calculated, fresh-run
orchestration materializes every semantic default and derivation into a fully
resolved frozen configuration, including initializer mappings, warmup steps,
finite schedule horizons, and nested precision/loader/checkpoint policies.
`run.json` stores those resolved values rather than `None` sentinels that resume
would need to reinterpret.

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
logging interval and checkpoint interval. Overrides apply only to that resume
invocation; they do not modify `run.json` or become implicit defaults for a
later resume. A resume command may also
supply a prepared-data or encoded-SWAG bundle location with `--data`; when it is
omitted, the original diagnostic locator is tried. A supplied location is not a
semantic override and is accepted only after its authoritative bundle identity
matches the dataset identity stored in `run.json`. Starting without `--resume`
never reuses, truncates, or overwrites an existing run target.

The shared frozen `ResumeOverrides` type lives in `training/common.py` and has
exact fields `maximum_steps`, `maximum_epochs`, `log_interval`, and
`checkpoint_interval`, each `int | None`. Pretraining and fine-tuning workflows
import it from `training.common`; they never import one another. There is no
retention field because latest-only checkpoint pruning is mandatory rather than
configurable.

Before allocating model or optimizer arrays, resume fully verifies the resolved
checkpoint payloads and matching data bundle, then validates every checkpoint
array against the saved precision contract. A payload mismatch, BF16/FP32 dtype
mismatch, missing or additional working/master/optimizer leaf, or failed exact
master-to-working cast relationship is corruption and cannot be coerced or
regenerated.

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
- exact FP32 LoRA merge formula and BF16 export-weight dtype
- FP32 LoRA-state and BF16 base-activation dtype boundaries
- FP32 master parameters with BF16 working parameters/raw gradients, FP32
  accumulation/moments/update arithmetic, exact BF16 derivation after each
  update, and survival of sub-BF16-ULP updates in the masters
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
- every correctness-sensitive workflow rejects a payload byte changed without a
  manifest update before GPU initialization or obsolete-checkpoint deletion; fast
  read-only use reports `manifest-trusted` rather than claiming full verification
- interruption at every immutable-bundle publication stage never exposes a
  partial bundle at its final path
- concurrent production of the same immutable target is either an idempotent
  success or a collision, never a partial overwrite
- fresh run creation atomically publishes step zero, rejects any existing
  target, and explicit resume before the first optimizer step restores the
  initialized state from that checkpoint
- a crash after step-directory publication but before `latest.json` replacement
  resumes from the newly published step and prunes the predecessor
- a full prefetch queue cannot advance the checkpointed committed cursor
- a crash inside an accumulation window replays the complete uncommitted window
  with its original PRNG sequence
- cursors normalize identically at shard, cross-shard-batch, bucket, padded-tail,
  and epoch boundaries; a dropped pretraining tail cannot create a second state
  for an already-published optimizer step
- shard-boundary batching drops at most one tail for the complete epoch stream
- conflicting reuse of an existing step number is rejected without overwrite
- publication of a non-increasing step cannot move `latest.json` backward
- deleting the source pretraining run does not affect LoRA resume, inference, or
  export from the copied base snapshot when the encoded-SWAG bundle is supplied
- a moved run performs model-only operations without its training data, resumes
  with an identity-matching relocated bundle, and rejects a mismatched bundle
- a second mutable workflow for the same run is rejected, concurrent readers
  can load the published checkpoint, and pruning waits for an active loader
- LoRA run creation holds the source run's shared access lock through full
  verification and exact BF16 base-byte copying, so source pruning cannot race
  the copy
- every SWAG cache-identity field produces a cache miss when changed
- fine-tuning rejects tokenizer identity, vocabulary, special-token, preprocessing,
  and maximum-length mismatches between encoded SWAG and the selected base
- SWAG cache fixtures pin separate context/ending encoding, mask alignment, EOS
  scoring, overlength-row drops, bucket placement, and stored array contracts
- padded SWAG bucket tails consume every real example exactly once, ignore
  finite synthetic slots, and produce the same example-weighted update as an
  unpadded eager reference
- unequal-length batched generation matches serial generation under the margin-
  aware equivalence rule and remains invariant to internal bucket ordering
- dynamically padded batched log-likelihood matches serial scoring within the
  configured tolerance and remains invariant to padding layout
- sequential session calls start from empty token/KV state, failed calls cannot
  contaminate the next call, and overlapping calls fail before mutation
- interrupted checkpoint replacement recovers the highest valid checkpoint and
  the next writable open prunes every obsolete predecessor
- a successful pretraining or fine-tuning checkpoint operation leaves exactly
  one published `step-*` directory after active readers release their locks
- evaluation and inference results pin the resolved model identity and
  verification level even if `latest.json` advances afterward
- evaluation results pin task YAML/include/template identities, prompt and
  metric policy, provider code, resolved dataset fingerprints, seeds, limits,
  and ordered request identities so provider/cache changes cannot masquerade as
  the same evaluation
- evaluation result publication is immutable and atomic: identical reuse is
  idempotent and a different existing output is a collision
- CLI and library entrypoints reject historical `--step` selection and direct
  `step-*` paths
- fresh run manifests contain fully resolved semantic configuration, and
  invocation-scoped resume overrides do not alter later resume defaults
- strict schema tests prove each run/checkpoint kind rejects fields belonging to
  another kind, and every named public wrapper has a complete field contract
- prefill compilation depends on prompt-length buckets independently of KV/token
  cache-capacity buckets, and short-prompt/long-generation requests never run a
  capacity-length prefill
- empty encoded prompts without a usable prefix token fail before bucketing
- case-folded, Unicode-normalized, internal/external hard-link-alias, and
  symlink-swap payload paths are rejected by descriptor-relative path traversal

The benchmark-analysis tests use fixed synthetic raw measurements to pin paired
ratio direction, bootstrap reproducibility, confidence-bound calculation, noise
rejection, and pass/fail/inconclusive decisions without running a performance
benchmark inside pytest.

## Training-Quality Acceptance

Formula and resume tests do not by themselves prove that the revised numerical
trajectory can train. A separately versioned deterministic quality harness owns
a fixed source-disjoint training/validation row set, initial BF16 parameter
identity, ordered batches, optimizer configuration, checkpoint steps, and
evaluation request identities. Quality work is never included in a throughput
timed region.

Quality inputs are checked-in immutable fixtures rather than arrays hidden in
Python source. Their logical paths, shapes, dtypes, byte identities, semantic
row/example identities, and exact ordered reuse schedule are part of each
quality-workload manifest and harness identity. The pretraining fixture contains
fixed `int32` training and source-disjoint validation rows; the SWAG fixture
contains fixed encoded train and source-disjoint validation candidate arrays
with masks and labels. All quality fixtures and committed raw/report evidence
together must remain at or below 64 MiB; model/optimizer states are temporary
run outputs and are not committed as quality fixtures.

The base-training quality workload uses the default model architecture and
precision-independent semantic configuration for 1,000 optimizer steps. From
the same initial BF16 working tree it runs:

- the candidate FP32-master/BF16-compute runtime; and
- a correctness oracle that keeps authoritative parameters and model compute in
  FP32 while using the same corrected tied-embedding graph, optimizer formula,
  batches, schedule, and seeds.

The report records training loss, held-out FP32 validation negative log
likelihood, finite-state checks, per-leaf update-to-BF16-ULP statistics, the
fraction of changed BF16 working values, and RMSNorm-master movement at steps 0,
10, 100, and 1,000. Acceptance requires no nonfinite state, nonzero RMSNorm
master movement, survival of at least one update that was individually below a
BF16 ULP, and candidate step-1,000 validation NLL no more than 1 percent above
the FP32-compute oracle. The candidate and oracle reports, exact workload
identity, and raw checkpoint metrics are committed before Part 1 is accepted.

Committed controlled-quality evidence is immutable evidence for the exact
source boundary recorded in its manifest; it is not a rolling certificate over
every later checkout. Validation treats the committed manifest workload and
source commit as authoritative, reconstructs the recorded harness, production
dependency, and fixture bytes from that commit, and then recomputes identities,
record cardinality, telemetry, report values, and the acceptance decision from
the committed raw evidence. It must not first rebuild an expected workload from
unrelated current-worktree bytes.

Later changes that do not alter the controlled numerical execution contract do
not invalidate accepted evidence. In particular, adding inference/evaluation
consumers, changing checkpoint-reader ownership without changing the training
calculation, or deleting the temporary flat migration bridge does not require a
rerun. A new recording is required when a change alters model or optimizer
mathematics, precision behavior, ordered training/evaluation work, fixtures,
quality telemetry or decision semantics, or another source component that
actually changes the controlled execution result. The same source-boundary and
rerun rules apply to the later SWAG quality evidence.

The SWAG quality workload fine-tunes the same frozen BF16 base and FP32 adapters
for 256 optimizer steps on fixed source-train rows, then evaluates a disjoint
fixed validation split with the authorized mean continuation-token score. Its
compiled result must have absolute relative validation-loss difference
`abs(candidate - oracle) / max(abs(oracle), 1e-12) <= 0.01` and absolute
validation-accuracy difference `abs(candidate - oracle) <= 0.01`, with
identical real-example counts and no nonfinite state. Final HellaSwag and
WinoGrande results are recorded with the complete evaluation provenance defined
above; they are diagnostic rather than substitutes for these controlled quality
gates.

On the acceptance M5, the pretraining quality command has a 12-hour wall-time
budget and the SWAG quality command has a 4-hour wall-time budget. Each manifest
records measured wall time, peak Metal memory, temporary disk usage, fixture
cardinality, validation cardinality, and ordered work count. Exceeding a budget
fails the quality-production task and requires an explicit workload/design
revision; it never silently reduces validation work or steps. Interrupted
quality evidence is discarded and rerun from the committed harness and fixtures
so a partial record can never be accepted.

## Implementation-Quality Acceptance

Completion guarantees implementation discipline, not a promise about downstream
model capability. The release must pass formula oracles, legacy-equivalence
tests where mathematics is preserved, dtype and explicit-PRNG contracts,
multi-step eager/compiled state tests, uninterrupted/resumed equality, artifact
schema and adversarial path tests, crash-stage checkpoint replacement tests,
source-lock concurrency tests, offline end-to-end CLI workflows, controlled
training-quality gates, and relevant runnable workflow checks. Production code may
contain no hidden global randomness, unverified state coercion, legacy fallback,
test-only switch, mutable compiled closure state, or per-token host conversion.
HellaSwag/WinoGrande results remain diagnostic; acceptance never substitutes a
benchmark score for these implementation checks.

## Optional Performance Measurement

This section defines how to produce a defensible performance claim when a
developer chooses to investigate speed or memory. It is not a refactor
acceptance gate. No baseline, phase report, final report, ratio, confidence
bound, dispersion result, power state, or thermal state is required to start or
finish an implementation phase. A failed or inconclusive run invalidates only
the performance claim from that run.

Performance claims require explicit Metal synchronization around timed regions
and compiled-kernel warmup before measurement. If a before/after comparison is
published, a machine-readable baseline manifest and its raw measurements are
committed first. The
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
unit for every metric. The harness content identity covers the ordered bytes of
the schema, workload, runner, analysis, legacy adapter, replacement adapter, and
fixed benchmark-analysis test-vector module. Changing any one invalidates every
dependent result.

Both adapters implement one frozen phase-independent ABI:
`resolve_native_workload(metric, canonical_workload, source_root)`,
`run_warmup(metric, native_workload, units)`, and
`run_measured(metric, native_workload, units)`. The replacement adapter is
committed before baseline capture with lazy string-based imports for every
planned owner module. A metric whose owner module is not yet present reports
`unavailable` and is not measured in that phase; the adapter itself is never
edited to enable a later phase. Unit tests pin every metric name, owner import,
canonical round trip, synchronization boundary, and unavailable/available
transition without importing absent modules.

In particular, each end-to-end pretraining work unit starts before requesting
the first microbatch in a measured accumulation window and ends after that
window's optimizer update is evaluated. It includes loader or prefetch stalls,
forward, backward, gradient accumulation, clipping, and the update, while
compilation, warmup, checkpointing, and unrelated logging are measured
separately.

Mandatory pre-GPU full input verification is startup work outside the steady-
state training region and is measured and reported separately for each native
artifact representation. A benchmark may not disable that verification in the
candidate's end-to-end CLI smoke measurement, even though it is excluded from
the steady-state optimizer-step ratio.

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
- end-to-end pretraining compares each version's actual default training-state
  precision: legacy BF16 persistent parameters and Adam moments without masters
  against replacement FP32 master parameters/moments with BF16 working
  parameters. This explicitly authorized precision-policy difference is
  recorded in both native configurations and means the benchmark compares
  product-default throughput, not identical numerical trajectories
- every native representation must resolve to the same canonical token rows,
  examples, requests, and real-work counts for that metric
- when work order can affect the executed kernels or state evolution, both
  adapters emit the same logical row, example, or request order

All other optimizer hyperparameters, BF16 working-parameter and activation
dtypes, initial BF16 parameter values, input rows, batch shapes, accumulation
count, clipping, weight-decay classification, schedules, and synchronization
boundaries match. Reports state both the master-parameter and moment-precision
differences next to every end-to-end pretraining ratio and never describe that
metric as legacy trajectory equivalence.

The optional reference baseline uses commit `3687f8b` on the Apple M5 10-core CPU,
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

Steady-state diagnostic screens use five fresh-process comparison pairs; the
historical full comparison uses ten. Each pair contains one reference and one
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
in the benchmark manifest. Phase screens report the bound but do not use it for
their diagnostic decision; the historical full comparison does. The analysis implementation and fixed statistical test
vectors are part of the versioned harness identity. Compile cold-start time is
measured separately in a fresh process without warmup and is report-only.

A diagnostic screen is too noisy when `MAD / median` exceeds 2 percent for either
side or for the paired ratios. The historical full comparison uses a 1.5-percent threshold.
After a noisy comparison, terminate both benchmark checkouts/processes, keep the
machine connected to power in the recorded power mode, and cool down for at
least 15 minutes; the last 5 minutes must report nominal thermal state, normal
memory pressure, and no competing GPU workload. The complete alternating-order
comparison is then repeated exactly once and both attempts are retained in the
raw report. Persistent excess dispersion makes that diagnostic inconclusive.
A final point estimate that satisfies a chosen threshold while its required
lower confidence bound does not is also reported as inconclusive rather than
being rounded to a pass or resolved by selecting favorable trials. Neither
outcome blocks feature work.

Benchmarks cover:

- prepared-data batches per second
- end-to-end pretraining tokens per second
- SWAG examples per second
- inference prefill tokens per second
- inference decode tokens per second
- checkpoint pause duration
- peak Metal memory

Performance benchmarks remain outside ordinary pytest. Any phase may
optionally record before/after results using the pinned manifest and identical harness. A
result is invalid if the target enters critical memory pressure, thermal
throttling is detected, the power mode changes, or the canonical workload
configurations or semantic content identities differ. Native configuration and
representation identities may differ only within the metric-specific boundaries
described above.

The harness maintains an explicit per-metric lineage rather than assuming that
the immediately preceding phase measured every metric. Each accepted metric
record names its pinned-baseline result identity and either the identity of the
most recent accepted measurement of that same metric or `null` when this is the
replacement metric's first appearance. A validator rejects a missing, wrong-
metric, incompatible-workload, or non-latest predecessor. The lineage begins:

- prepared data: baseline -> Phase 2 -> Phase 3 -> final acceptance
- end-to-end pretraining: baseline -> Phase 3 -> final acceptance
- inference prefill/decode: baseline -> Phase 1 -> Phase 4 -> final acceptance
- SWAG end to end: baseline -> Phase 5 -> final acceptance

Runner and validator commands pass predecessors as an explicit mapping from
metric name to result identity/path (or `null`), never as one phase-wide
`--previous` value. A phase report that contains several metrics therefore may
name different predecessors for each metric.

A benchmark is relevant to a phase when the phase changes code executed in its
timed region. For an optional diagnostic, the original targets remain useful
context: a five-pair phase-screen median of `0.97` against the pinned baseline
and prior measurement, and a final pretraining target of `1.03`. Missing those
targets records a performance observation; it does not fail the phase.
Checkpoint pause, compile cold-start, peak memory, and all ratios are
report-only for refactor acceptance.

Required acceptance instead consists of:

- mathematical-equivalence and correctness tests passing;
- the base and SWAG controlled quality reports passing their committed rules;
- relevant artifact, resume, integration, and CLI workflows completing on the
  target Apple-Silicon environment; and
- the supported default/tiny verification workloads completing without an
  out-of-memory or critical runtime failure.

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

The final phase additionally runs all unified CLI workflow smoke tests.
Performance comparisons may be run separately when useful, but are not part of
the required verification set.

## Delivery Sequence

The existing versioned benchmark harness and baseline artifacts remain
available for optional diagnostics. New baseline or phase evidence is not a
prerequisite for implementation work.

One master plan index links six ordered implementation plans:

1. Foundation and model package
2. Tokenizer, artifacts, and prepared data
3. Pretraining runtime
4. Inference and evaluation
5. LoRA and SWAG
6. Unified CLI and final cutover

Each phase is independently reviewable, testable, exercised through its
relevant functional workflows, and committed before the next begins. Migrated and unmigrated workflows may
coexist through phase 5. Phase 6 deletes all replaced flat source modules,
legacy tests, old entrypoints, and compatibility scaffolding, then updates v2
documentation to describe only the new system.

## Out of Scope

- changing the model architecture, mathematical algorithms, or optimizer
  algorithm beyond the canonical tied-gradient, mean-normalized SWAG-score, and
  FP32-master/BF16-working-parameter precision corrections authorized above
- replacing SentencePiece BPE
- supporting non-MLX training/inference backends
- compatibility with any existing v2 artifact or API
- checkpoint history, historical step selection, or configurable retention;
  every run owns one latest checkpoint in steady state
- adding a generic training framework or plugin system
- unrelated top-level repository or dependency changes
