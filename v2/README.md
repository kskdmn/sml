# SML v2

## Updates

- Target Apple Silicon through MLX only.
- Remove the legacy accelerator-specific training path.
- Remove attention dropout to use MLX fast attention.

## Running scripts

Run commands from the repository root with `uv run`. The scripts use the shared
defaults in `v2/src/config.py`:

- Training data: `~/Documents/data-common_pile/`
- Input shard pattern: `.*-00[0-9][0-9]\.jsonl\.zst\Z`
- Default tokenizer: `v2/output/bpe_tokenizer.model`
- Default model checkpoint: `v2/output/sml`
- Default output directory: `v2/output`

Each script also supports `--help`, for example:

```sh
uv run python v2/src/train_sml.py --help
```

### Train the tokenizer

Builds a SentencePiece tokenizer from the configured JSONL Zstandard shards.
Input rows must contain a `text` field.

```sh
uv run python v2/src/train_tokenizer.py
```

Use `--tokenizer-model` to write the tokenizer somewhere other than
`v2/output/bpe_tokenizer.model`:

```sh
uv run python v2/src/train_tokenizer.py --tokenizer-model v2/output/custom.model
```

### Train the base model

Trains the MLX SML language model from the configured corpus. Run tokenizer
training first, or pass an existing tokenizer with `--tokenizer-model`.

```sh
uv run python v2/src/train_sml.py
```

Useful variants:

```sh
uv run python v2/src/train_sml.py --resume
uv run python v2/src/train_sml.py --model v2/output/sml --tokenizer-model v2/output/bpe_tokenizer.model
```

The command prints the checkpoint path when training finishes or saves.

### Generate text

Loads a checkpoint and tokenizer, then continues the prompt.

```sh
uv run python v2/src/infer_sml.py "Once upon a time"
```

Useful decoding options:

```sh
uv run python v2/src/infer_sml.py "Once upon a time" --max-new-tokens 128
uv run python v2/src/infer_sml.py "Once upon a time" --temperature 0.8 --top-p 0.95 --seed 42
uv run python v2/src/infer_sml.py "Once upon a time" --include-prompt
```

Use `--model` and `--tokenizer-model` to load non-default artifacts.

### Evaluate a checkpoint

Runs the local checkpoint through `lm-eval`. Supported benchmarks are
`hellaswag` and `winogrande`.

```sh
uv run python v2/src/evaluate_sml.py --benchmark hellaswag
uv run python v2/src/evaluate_sml.py --benchmark winogrande
```

Useful variants:

```sh
uv run python v2/src/evaluate_sml.py --benchmark hellaswag --limit 100
uv run python v2/src/evaluate_sml.py --benchmark winogrande --output v2/output/winogrande-smoke.json
```

Results default to `v2/output/<benchmark>.json`.

### Fine-tune on SWAG with LoRA

Starts from a pretrained SML checkpoint, downloads or loads `allenai/swag`
(`regular`, `train`) through Hugging Face datasets, and trains LoRA adapters.
The Hugging Face cache defaults to `.hf-cache` beside the repository.

```sh
uv run python v2/src/ft_swag.py
```

Useful variants:

```sh
uv run python v2/src/ft_swag.py --resume
uv run python v2/src/ft_swag.py --model v2/output/sml --tokenizer-model v2/output/bpe_tokenizer.model
```

The default LoRA checkpoint is written under `v2/output/sml-swag`.
