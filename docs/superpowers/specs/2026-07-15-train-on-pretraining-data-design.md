# Train on Prepared Pretraining Data Design

**Status (2026-08-23):** The legacy NPZ prepared-data flow was implemented.
The active refactor's strict immutable int32 NPY bundle design supersedes this
document; no task remains here.

## Goal

Make `v2/src/train_sml.py` consume offline-prepared pretraining shards
(`v2/output/pretraining_data/manifest.json` + `train-*.npz`) instead of streaming
and tokenizing JSONL online. Preparation remains a required step before training.

## Decisions

- Replace the JSONL training path entirely (no dual-mode fallback).
- Exact resume via `shard_index` + `block_index` (and `epoch`).
- Manifest `sequence_length` is authoritative; mismatch with
  `TrainingConfig.sequence_length` fails fast.
- Remove JSONL-only `TrainingConfig` fields:
  `input_file_name_regex`, `max_rows_per_file`, `shuffle_input_files`.
- Default `TrainingConfig.input_dir` to `v2/output/pretraining_data`.
- Walk shards in manifest order (no train-time shard shuffle).
- Extract shared on-disk format helpers into `v2/src/pretraining_format.py`
  used by prepare, train, and `peek_npz`.

## Architecture

```text
prepare_pretraining_data.py  ──writes──▶  pretraining_data/
                                              ├── manifest.json
                                              └── train-*.npz
                                                    ▲
v2/src/pretraining_format.py  ◀── shared constants + load/validate
                                                    │
train_sml.py  ──reads blocks──┘    peek_npz.py ──inspect──┘
```

### `pretraining_format.py`

Owns:

- Format constants: `FORMAT_NAME`, `TOKENS_ARRAY_NAME`, `TOKEN_DTYPE_NAME`,
  `MANIFEST_NAME`, shard name prefix/suffix helpers
- Default prepared-data directory (`OUTPUT_DIR / "pretraining_data"`)
- `load_manifest(path) -> dict`
- Validation of format name, shard list shape, and expected tokens-per-block
- Resolving relative shard paths against the manifest directory

`prepare_pretraining_data.py` and `common/scripts/peek_npz.py` import these
instead of duplicating local constants.

## Data Flow

1. Resolve `TrainingConfig.input_dir` and load `manifest.json`.
2. Require `format == "sml-pretokenized-blocks-v1"`.
3. Fail if `TrainingConfig.sequence_length != manifest["sequence_length"]`.
4. Resolve and require every listed shard file exists.
5. Load tokenizer from `TrainingConfig.tokenizer_model_path` for model
   `vocab_size`. Fail if tokenizer piece count disagrees with
   `manifest["tokenizer_vocab_size"]`.
6. For each epoch, iterate shards in manifest order. Load one `.npz` at a
   time; read the `tokens` uint16 array of shape
   `[N, sequence_length + 1]`.
7. Convert each block to
   `input_ids = block[:-1]`, `labels = block[1:]`, then reuse
   `iter_mlx_batches`.

Remove online tokenization helpers from the train path
(`iter_texts`, `iter_mlx_token_blocks`, and related JSONL progress wiring).

## Resume and Checkpointing

`TrainingDataState` becomes:

- `epoch: int`
- `shard_index: int`
- `block_index: int`

Drop `input_file_index`, `line_number`, and `token_buffer`.

On `--resume`, continue from the saved shard/block. Do not keep a legacy
“skip N batches from global step” path for new prepared-data checkpoints.

Checkpoint metadata stores the resolved shard path list (and data state).
Training logs report `shard=` / `block=` instead of JSONL `input=` / `line=`.

## Error Handling

Fail fast with clear errors when:

- `manifest.json` is missing or has an empty shard list
- Format, dtype/array name, or block width is invalid
- Config `sequence_length` disagrees with the manifest
- A shard listed in the manifest is missing
- Tokenizer vocab size disagrees with the manifest
- Resume `shard_index` / `block_index` is out of range

## Testing

- Unit tests for `pretraining_format` load/validate and shard path resolution
- Training loader tests with tiny fixture shards covering label slicing and
  resume skip-to-block behavior
- Update train/config/module tests that still assume JSONL fields or online
  tokenization
- Keep prepare and `peek_npz` tests green after the shared-module import
- Update `v2/README.md` so the train step documents prepare-then-train and
  `pretraining_data` as the training input

## Out of Scope

- Train-time shard shuffling
- Dual JSONL + prepared-data training modes
- Memory-mapping optimizations
- Changes to the on-disk `.npz` / manifest schema
