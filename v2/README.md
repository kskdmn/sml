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

### Inference

Generate from a pretraining run, LoRA run, or merged export:

```sh
uv run python -m sml infer --checkpoint v2/output/base-run --max-new-tokens 128 --seed 42 "Once upon a time"
```

Add `--include-prompt` to include the prompt in rendered text and `--full` to
rehash every consumed payload.

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
