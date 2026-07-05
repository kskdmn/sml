# V2 MLX-Native Training Design

## Goal

Add a v2 training script that keeps the model-training hot loop fully MLX-native:
MLX model, MLX arrays, MLX gradients, MLX optimizer updates, and MLX checkpoint
weights. The existing PyTorch `train_sml.py` remains unchanged.

## Approach

Create `v2/src/train_sml_mlx.py` as a sibling to `train_sml.py`. The script will
reuse cold-path helpers and configuration from `train_sml.py` where doing so does
not introduce Torch tensors into the hot loop:

- `TrainingConfig`
- `TrainingDataState`, `ReadingProgress`, and resume-position helpers
- input file discovery and deterministic shuffling
- tokenizer loading and text filtering
- `model_config_for_training`, `lr_lambda`, and training log formatting

The new script will not reuse `build_dataloader`, because it constructs a
PyTorch `DataLoader` and Torch tensors. Instead it will add an MLX batch iterator
that consumes the same token-block stream and emits `{"input_ids": mx.array,
"labels": mx.array}` batches directly.

## Training Loop

The MLX trainer will construct `sml_mlx.SMLLanguageModel` with the same v2
training config rules as PyTorch training: tokenizer vocab is folded into the
checkpoint model config, training disables YaRN with `rope_scaling_factor=1.0`,
and the saved metadata preserves the inference-capable checkpoint config.

The hot loop will use:

- `nn.value_and_grad(model, loss_fn)` for loss and gradients
- `mlx.optimizers.AdamW` for optimizer state and parameter updates
- an MLX schedule callable equivalent to `lr_lambda`
- MLX tree operations for gradient accumulation and global-norm clipping
- `mx.eval(model.parameters(), optimizer.state)` after each optimizer step

No Torch tensors or Torch optimizer calls should appear inside the MLX training
step.

## Checkpointing

Use an MLX checkpoint directory named by `TrainingConfig.checkpoint_name`; when a
name ends in `.pt`, replace that suffix with `_mlx` to avoid confusing it with a
Torch checkpoint file.

Each checkpoint directory will contain:

- `model.safetensors`: MLX model weights via `model.save_weights`
- `optimizer.npz`: flattened MLX optimizer state arrays
- `metadata.json`: step, model config, training config, input files, and data state

Resume will load the model weights, optimizer state, step, input files, and data
state from that directory. Missing explicit resume checkpoints will raise
`FileNotFoundError`, matching the current trainer's behavior.

## Error Handling

The script should fail early when no input files are discovered or the tokenizer
path is missing, by reusing the existing helpers. Invalid checkpoints should
raise clear errors for missing metadata, model weights, or optimizer state.

## Testing

Add `v2/tests/test_train_mlx.py`. Tests should skip cleanly when MLX/Metal is not
available. Coverage should include:

- argument parsing and `main` pass-through behavior
- MLX batch iterator emits `mx.array` batches with current label alignment
- gradient clipping helper returns a finite pre-clip norm and scales gradients
- one tiny MLX training run writes checkpoint files
- missing explicit resume checkpoint raises `FileNotFoundError`
- full v2 test suite remains green
