# SML v2

SML v2 is an MLX-only language-model workflow for Apple Silicon. Run every
command from the repository root with Python 3.12 through `uv run`.

## Unified command line

The package exposes one entrypoint:

```sh
uv run python -m sml --help
```

Each command accepts `--help` and an optional command-specific TOML file through
`--config`. Values are resolved in this order: domain defaults, the exact TOML
command table, then explicit command-line options. A config file must contain
only the selected table, such as `[train]`, `[prepare.pretraining]`, or
`[prepare.swag]`; unknown and duplicate semantic fields are rejected.

### Tokenizer

Train a self-describing SentencePiece bundle from compressed JSONL rows whose
selected text field contains the corpus text:

```sh
mkdir -p v2/output
uv run python -m sml tokenize --input data/corpus --output v2/output/tokenizer
```

Tokenizer input defaults to a reproducible sample of at most **100,000 unique
normalized documents** and **128 MiB of UTF-8 text**. Exact duplicates are removed
after whitespace normalization. Selection uses seeded document priorities across
all scanned files, so filling the sample does not exclude later scanned rows or
files. Whole documents are retained; the byte limit can leave some unused space.

Control the tokenizer input and the work spent reading candidate documents:

```sh
uv run python -m sml tokenize --input data/corpus --output v2/output/tokenizer --max-documents 50000 --max-corpus-bytes 67108864 --max-files 40 --max-rows-per-file 8192 --sampling-seed 42
```

The equivalent TOML configuration is:

```toml
[tokenize]
input = "data/corpus"
output = "v2/output/tokenizer"

[tokenize.corpus]
max_files = 40
max_rows_per_file = 8192

[tokenize.sampling]
max_documents = 50000
max_bytes = 67108864
seed = 42
```

Discovery includes all nonhidden `*.jsonl.zst` direct children and selects up to
100 files by default, using `corpus.file_order_seed`. Each selected file contributes
at most its first 8,192 physical lines to the candidate scan. Compressed files
must be read from the beginning: this is sampling across bounded scanned prefixes,
not uniform sampling across entire files. Increase the scan limits to consider
more source text. The default scan is at most 819,200 lines; document and byte
caps bound the text passed to SentencePiece, not total process memory or elapsed
time. SentencePiece's optional `input_sentence_size` can further reduce its input.
Normalized documents outside the existing 100–16,384-byte range are still filtered.
Sampling settings are recorded in the tokenizer manifest; older bundles remain
readable without rewriting their metadata.

### Pretraining data

Encode and deterministically shuffle fixed-width pretraining rows into an
immutable, memory-mapped directory bundle:

```sh
uv run python -m sml prepare pretraining --input data/corpus --tokenizer v2/output/tokenizer --output v2/output/pretraining-data
```

Pretraining rows are packed without padding. Full verification and training
preflight reject rows containing the tokenizer's padding token, including custom
prepared bundles. This keeps every microbatch's target count equal during gradient
accumulation.

Pretraining uses all eligible text within its own corpus scan limits. It does not
inherit the tokenizer's document/byte sample caps or deduplication. Configure a
larger scan independently through `[prepare.pretraining.corpus]`, for example
`max_files = 300` and `max_rows_per_file = 32768`.

### Base training

Start a new pretraining run from a prepared-data bundle:

```sh
uv run python -m sml train --data v2/output/pretraining-data --output v2/output/base-run
```

For both `train` and `finetune`, `gradient_accumulation_steps` counts microbatches
per optimizer update. An incomplete accumulation window at the end of an epoch
is still applied. Progress is printed to stderr every `log_interval` optimizer
updates, including loss, learning rate, and processed rows or examples;
fine-tuning also reports accuracy.

For a fresh base run, an omitted `optimizer.schedule_steps` resolves to the planned
number of optimizer updates, using the prepared row count, microbatch size,
accumulation, and whichever step or epoch limit is reached first. An incomplete
microbatch is dropped at the end of an epoch; an incomplete accumulation window
is applied. Warmup defaults to 1% of the resolved schedule, followed by cosine
decay. An explicit `[train.optimizer] schedule_steps` overrides this automatic
choice. The resolved schedule is saved with the run and remains unchanged on
resume, including when extending the stopping limits. SWAG keeps its existing
fine-tuning schedule.

The final checkpoint is the selected model after the requested budget completes.
Intermediate checkpoints support recovery; training does not select an earlier
model through evaluation or stop based on validation scores.

Training stops on nonfinite loss, gradients, or updated parameters before committing
the failed update. The preceding published checkpoint remains available for recovery.

Resume an existing run with only the allowed operational overrides:

```sh
uv run python -m sml train --resume v2/output/base-run --data v2/output/pretraining-data --maximum-epochs 2 --checkpoint-interval 100
```

Resume accepts `maximum_steps`, `maximum_epochs`, `log_interval`, and
`checkpoint_interval`. A relocated prepared-data directory may be supplied with
`--data`, but its recorded identity must match the run. Model, optimizer,
precision, loader, seed, compile, and output settings are immutable run
semantics and are rejected on resume.

Step and epoch limits are absolute totals, and training stops when either limit
is reached. The default base run has a one-epoch limit and no step limit, so the
example adds a second epoch. Set any active limits beyond the saved progress when
continuing a completed run; increasing only the step limit leaves a reached epoch
limit in effect.

For a 100,000-update run, set `maximum_steps = 100000` and an epoch limit large
enough to supply those updates. Setting the step limit alone still leaves the
default one-epoch limit active.

### Inference

Generate from a pretraining run, LoRA run, or merged export:

```sh
uv run python -m sml infer --checkpoint v2/output/base-run --max-new-tokens 128 --seed 42 "Once upon a time"
```

Add `--include-prompt` to include the prompt in rendered text and `--full` to
rehash every consumed payload.

Inference requires a KV cache. Model configurations with `use_cache=false` are
rejected instead of silently enabling caching. Persistent sessions retain a
bounded set of recently used compiled functions and share prefill compilation
across decoding settings.

### Evaluation

Evaluate one or more supported lm-eval tasks and atomically write the result:

```sh
uv run python -m sml evaluate --checkpoint v2/output/base-run --task hellaswag --task winogrande --output v2/output/evaluation.json
```

Use `--limit` for a smoke run and `--full` for full payload verification. The
immutable JSON artifact preserves complete provider metrics plus resolved
task, model, dataset, and ordered-request provenance; its destination path is
intentionally excluded from the artifact and its identity.

Results use schema version 3 and bind each model to its complete artifact
identity, so different exported weights remain distinguishable even when the
tokenizer and source step match.

Repeating an evaluation at the same output path reuses the saved result when
only the provider's execution date changes. The original timestamp, artifact
identity, and file contents are preserved; other result differences are rejected.

### SWAG data

Resolve an immutable Hugging Face revision, encode SWAG candidates with the
selected model's copied tokenizer, and publish an offline-reusable bundle:

```sh
uv run python -m sml prepare swag --checkpoint v2/output/base-run --revision 0123456789abcdef --output v2/output/swag-data
```

Preparation fully verifies the selected base run before publication.
Repeating the command with the same configuration and output reuses a matching,
fully verified bundle without contacting the dataset provider. Changed configuration
or damaged cached payloads are rejected.

### LoRA fine-tuning

Start a self-contained SWAG LoRA run:

```sh
uv run python -m sml finetune --checkpoint v2/output/base-run --data v2/output/swag-data --output v2/output/swag-run
```

Resume uses the same override rules as base training. A moved SWAG bundle can be
supplied through `--data` only when its identity matches the run:

```sh
uv run python -m sml finetune --resume v2/output/swag-run --data /new/location/swag-data --maximum-steps 16000 --maximum-epochs 6
```

### Merged export

Fully verify a LoRA run's recovered latest checkpoint, merge its FP32 adapter
delta into the copied BF16 base, and publish a portable inference artifact:

```sh
uv run python -m sml export --checkpoint v2/output/swag-run --output v2/output/swag-export
```

### Artifact verification

Validate an artifact's canonical manifest and required structure:

```sh
uv run python -m sml verify v2/output/base-run
uv run python -m sml verify --full v2/output/swag-export
```

The default `manifest-trusted` level validates canonical manifests, identities,
paths, structure, and the payloads a read-only workflow consumes. `--full`
rehashes every declared payload. Training, resume, SWAG preparation, merged
export, recovery, and retention always use full correctness-sensitive checks;
inference and evaluation default to `manifest-trusted` and opt into full checks
with `--full`.

## Artifact layouts

All supported artifacts are directories. Manifests use canonical JSON and bind
the identities, sizes, dtypes, shapes, and relative paths of their payloads.

Tokenizer bundle:

```text
tokenizer/
├── manifest.json
├── tokenizer.model
└── tokenizer.vocab
```

Prepared pretraining data stores little-endian `int32` rows in uncompressed,
memory-mapped NPY shards and copies the tokenizer it used:

```text
pretraining-data/
├── manifest.json
├── tokenizer/
│   ├── manifest.json
│   ├── tokenizer.model
│   └── tokenizer.vocab
└── shards/
    ├── train-000000.npy
    └── train-000001.npy
```

Prepared SWAG data stores one directory per fixed sequence-length bucket. Each
array is directly memory-mappable:

```text
swag-data/
├── manifest.json
└── buckets/
    └── length-0256/
        ├── input_ids.npy
        ├── labels.npy
        ├── score_mask.npy
        └── valid_token_mask.npy
```

A pretraining run owns its tokenizer and one latest checkpoint in steady state:

```text
base-run/
├── run.json
├── latest.json
├── tokenizer/
│   ├── manifest.json
│   ├── tokenizer.model
│   └── tokenizer.vocab
└── checkpoints/
    └── step-000000123/
        ├── checkpoint.json
        ├── master.safetensors
        ├── model.safetensors
        ├── optimizer.safetensors
        ├── state.json
        └── trainer.safetensors
```

A LoRA run additionally owns a frozen BF16 base snapshot; its checkpoint stores
only adapter and training state:

```text
swag-run/
├── run.json
├── latest.json
├── tokenizer/
├── base/
│   ├── manifest.json
│   └── model.safetensors
└── checkpoints/
    └── step-000000123/
        ├── checkpoint.json
        ├── adapters.safetensors
        ├── optimizer.safetensors
        ├── state.json
        └── trainer.safetensors
```

New checkpoints use format version 2. They save trainer counters and RNG state,
and reconstruct empty gradient accumulators on resume instead of storing a full
tree of zeros. Version 1 checkpoints remain readable and resumable; the next saved
checkpoint uses version 2.

A merged export is independently portable and contains no optimizer, master, or
adapter training state:

```text
swag-export/
├── manifest.json
├── model.safetensors
└── tokenizer/
    ├── manifest.json
    ├── tokenizer.model
    └── tokenizer.vocab
```

## Latest-only model selection

Model-consuming commands accept a complete pretraining run, LoRA run, or merged
export directory. A run always resolves through its recovered `latest.json`.
Artifact manifests determine the accepted directory kind. Writable resume and
export operations recover and prune crash leftovers only after full verification,
while read-only
inference and evaluation can recover a stale latest index in memory without
mutating the artifact.
