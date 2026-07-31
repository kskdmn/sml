# V2 Checkpoint and SWAG Corrections Design

## Goal

Fix the seven reviewed correctness, efficiency, and numerical issues in the v2
LoRA/SWAG/inference path by introducing a shared checkpoint subsystem and by
making SWAG batching, scoring, precision, and merged-weight export explicit.

## Compatibility Boundary

- Existing `train_tokenizer.py` output remains usable as the same SentencePiece
  `.model` and `.vocab` files. No tokenizer sidecar becomes mandatory.
- Existing `train_sml.py` checkpoints remain loadable and resumable through a
  dedicated legacy metadata reader.
- New pretraining and SWAG checkpoints use the versioned shared checkpoint
  format described below.
- Existing SWAG fine-tuning checkpoints are unsupported. Attempting to resume
  one fails before model or optimizer state is changed.
- Artifact paths are informational. New-format compatibility is based on file
  contents, model configuration, and training semantics, so artifacts may move
  without becoming invalid.

## Architecture

Create `v2/src/checkpoint_io.py` as the shared checkpoint boundary used by
pretraining, SWAG fine-tuning, and inference.

`checkpoint_io.py` owns:

- checkpoint format constants and metadata parsing
- legacy pretraining metadata normalization
- file SHA-256 fingerprints
- tokenizer identities and compatibility validation
- strict named-array key, shape, and dtype validation
- model-weight and optimizer-state serialization helpers

Domain-specific transformations stay outside the I/O module:

- `lora.py` builds inference-compatible merged model weights from an adapted
  model.
- `train_sml.py` owns pretraining progress and configuration.
- `ft_swag.py` owns SWAG progress, LoRA configuration, and ranking behavior.
- `infer_sml.py` owns prompt encoding, generation, and output decoding.

The dependency direction is:

```text
train_tokenizer.py ──writes──▶ tokenizer.model
                                  │
                                  ▼
                    checkpoint_io.py ◀──── lora.py merged weights
                       ▲      ▲      ▲
                       │      │      │
                 train_sml  ft_swag  infer_sml
```

## Versioned Checkpoint Metadata

New checkpoints keep the existing filenames:

- `model.safetensors`
- `optimizer.npz`
- `metadata.json`
- `lora.npz` for resumable SWAG checkpoints

New `metadata.json` objects contain these common fields:

- `format`: `"sml-checkpoint"`
- `format_version`: `1`
- `checkpoint_kind`: `"pretraining"` or `"swag_lora"`
- `step`
- `model_config`
- `training_config`
- `data_state`
- `tokenizer`: resolved path for diagnostics, SHA-256, byte size, vocabulary
  size, and BOS/EOS/PAD IDs

Pretraining metadata continues to include `input_files` and the existing
stochastic-resume note.

SWAG metadata additionally contains:

- `lora_config`: the complete serialized `LoRAConfig`
- `base_checkpoint`: resolved path for diagnostics, SHA-256 and byte size of
  the base `model.safetensors`, and the base model configuration
- the existing stochastic-resume note

The model checkpoint itself is not re-hashed after every save. The tokenizer
and base-model fingerprints are computed once when a training run starts and
reused for each checkpoint written by that run.

## Legacy Pretraining Compatibility

Metadata without `format` and `format_version` is accepted only as a legacy
pretraining checkpoint when it has the current `train_sml.py` structure:

- dictionary `model_config`
- dictionary `training_config`
- integer `step`
- optional pretraining `data_state`
- optional `input_files`

Legacy model weights and optimizer state retain their current filenames and
load strictly. On the first subsequent save, the checkpoint is upgraded to the
new format.

For inference with a legacy checkpoint:

- tokenizer vocabulary size and BOS/EOS/PAD IDs must match the model
  configuration
- if the tokenizer path recorded in legacy `training_config` exists, its
  fingerprint must match the requested tokenizer
- if that historical path no longer exists, structural checks are the maximum
  possible verification and the tokenizer remains accepted for compatibility

Legacy metadata is never interpreted as a SWAG checkpoint.

## Strict Resume Contract

SWAG resume performs validation before loading adapter or optimizer arrays:

1. Require versioned `swag_lora` metadata.
2. Compare the current tokenizer identity with the saved tokenizer identity.
3. Compare the current base-model weights fingerprint and configuration with
   `base_checkpoint`.
4. Compare the full current `LoRAConfig` with `lora_config`, including rank,
   alpha, scaling mode, dropout, and target modules.
5. Compare the effective model configuration with `model_config`.
6. Require the adapter checkpoint to contain exactly the expected A/B keys,
   shapes, and dtypes; missing and unexpected keys both fail.
7. Require optimizer checkpoint keys, shapes, and dtypes to match the freshly
   initialized optimizer state before replacing it.

Once resume state is loaded, `fine_tune_swag` checks `global_step` against
`max_steps` before constructing the dataset iterator or computing gradients.
A checkpoint already at the limit returns without another optimizer update.

## Tokenizer Compatibility

`checkpoint_io.py` represents tokenizer identity using:

- SHA-256 and byte size of the `.model` file
- vocabulary size
- BOS, EOS, and PAD IDs resolved from SentencePiece with model-config fallbacks

`train_sml.py` and `ft_swag.py` record this identity when consuming a tokenizer.
`infer_sml.py` validates it before prompt encoding. New-format checkpoints
require an exact fingerprint and structural match. Legacy pretraining
checkpoints use the compatibility rules above.

`train_tokenizer.py` continues to produce its existing files and return its
existing `TrainingResult`; it does not need a new manifest.

## Dynamic SWAG Batching

`SwagExampleDataset` produces unpadded candidate inputs and labels. Each
candidate is constructed as a token sequence containing optional BOS, context,
ending, and optional EOS. It then produces the causal pair:

- `input_ids = tokens[:-1]`
- `labels = tokens[1:]`, with context labels masked to `pad_token_id`

`sequence_length` is the maximum shifted input length. A candidate fits when
`len(tokens) - 1 <= sequence_length`.

`collate_swag_batch` finds the longest candidate input across every example and
candidate in the batch, then pads inputs and labels to only that length. This
removes the fixed 256-token computation for short rows while preserving the
existing rectangular `(batch, candidates, sequence)` MLX interface.

As today, a SWAG row is skipped when any candidate exceeds the configured
maximum, so candidate labels remain aligned and every row always has four
scores.

## LoRA and Ranking Precision

Base-model parameters and activations retain the configured `bfloat16` default.
After applying the base-model dtype, `ft_swag.py` explicitly casts LoRA A/B
matrices back to FP32. Optimizer initialization therefore creates FP32 Adam
moments for the trainable adapters. Resume loads adapters into those FP32
destinations and validates FP32 optimizer state.

`score_swag_candidates` casts logits to FP32 before log-softmax, token gather,
and reduction. Candidate scores are mean continuation log-likelihoods:

```text
sum(valid continuation token log probabilities) / valid continuation token count
```

EOS remains a scored continuation token. Every candidate must contain at least
one valid scored token; otherwise scoring raises a clear error. Mean scoring
removes the systematic preference for shorter endings.

## Merged LoRA Weight Export

Replace `copy.deepcopy(model)` with a flat merged-weight exporter in `lora.py`.
The exporter:

1. Flattens the adapted model parameter tree.
2. Omits `lora_A` and `lora_B` arrays.
3. Remaps wrapped linear parameter names such as `q_proj.linear.weight` to the
   base inference name `q_proj.weight`.
4. Computes `base_weight + scaling * (lora_B @ lora_A)` only for adapted
   projection weights, casting the delta to the base-weight dtype before the
   addition.
5. Remaps a wrapped bias without modifying it.
6. Reuses unchanged base arrays directly in the output dictionary.

`checkpoint_io.py` saves the resulting dictionary with
`mx.save_safetensors`. The training model is never mutated, and peak checkpoint
memory is limited to the merged projection arrays rather than a second full
model.

## Error Handling

Fail before training or prompt encoding with errors that identify the failing
contract:

- unsupported checkpoint format or kind
- malformed or missing metadata fields
- missing checkpoint files
- model configuration mismatch
- tokenizer vocabulary, special-token, or fingerprint mismatch
- base-model fingerprint mismatch
- LoRA configuration mismatch
- missing, unexpected, wrongly shaped, or wrongly typed adapter arrays
- incompatible optimizer state
- candidate with no scored continuation tokens

No compatibility warning silently downgrades a new-format checkpoint to legacy
validation.

## Testing Strategy

Every behavior change follows an independent red-green TDD cycle.

### Shared checkpoint tests

Create `v2/tests/test_checkpoint_io.py` covering:

- versioned pretraining and SWAG metadata parsing
- current legacy pretraining metadata normalization
- rejection of legacy SWAG resume
- stable fingerprints and moved-file identity
- tokenizer identity validation
- strict named-array keys, shapes, and dtypes
- malformed format, kind, and required fields

### LoRA tests

Extend `v2/tests/test_lora.py` covering:

- rejection of unexpected adapter keys
- explicit FP32 adapter casting
- merged flat weights using inference names
- direct exported weights producing the same inference output as `merge_lora`
- no mutation of the adapted source model

### SWAG tests

Extend `v2/tests/test_ft_swag.py` covering:

- dynamic padding to the longest candidate in a batch
- exact `sequence_length` boundary behavior
- FP32 mean candidate scores and removal of length bias
- strict base, tokenizer, model, and LoRA resume validation
- strict optimizer state validation
- immediate return when a resumed checkpoint is already at `max_steps`
- new-format checkpoint round trips using direct merged-weight serialization

### Pretraining and inference tests

Extend `v2/tests/test_train.py` and `v2/tests/test_infer.py` covering:

- new pretraining metadata contains tokenizer identity
- the current legacy pretraining checkpoint format remains loadable/resumable
- legacy inference accepts the existing tokenizer output
- new inference rejects vocabulary, special-token, and fingerprint mismatches
- moved new-format tokenizer artifacts remain valid by content identity

Final verification runs:

```text
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
```

The pytest command runs outside the sandbox so MLX/Metal is available.

## Out of Scope

- Compatibility with existing SWAG fine-tuning checkpoints
- Changing SentencePiece training or tokenizer file formats
- Changing the prepared pretraining-data shard format
- Changing the base model architecture or generation algorithm
- Adding external dependencies or editing top-level project files
