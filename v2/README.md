# SML v2

## Updates

- Target Apple Silicon through MLX only.
- Remove the legacy accelerator-specific training path.
- Remove attention dropout to use MLX fast attention.

## Running scripts

Run commands from the repository root with `uv run`. The scripts use the shared
defaults in `v2/src/config.py`:

- JSONL training data (tokenizer + prepare): `~/Documents/training_data-common_pile/`
- Prepared pretraining shards (`train_sml`): `v2/output/pretraining_data/`
- Input shard pattern (JSONL): `.*-00[0-9][0-9]\.jsonl\.zst\Z`
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

### Prepare pretraining data

Tokenizes the configured JSONL Zstandard shards offline, packs fixed
`sequence_length + 1` token blocks, shuffles blocks deterministically within
each output shard, and writes compressed NumPy `.npz` files under
`v2/output/pretraining_data/`. The shard array is named `tokens` and stored as
`uint16`; preparation fails if the tokenizer vocabulary is larger than 65,536
pieces.

```sh
uv run python v2/src/prepare_pretraining_data.py
```

Useful variants:

```sh
uv run python v2/src/prepare_pretraining_data.py --sequence-length 1024 --blocks-per-shard 8192
uv run python v2/src/prepare_pretraining_data.py --max-rows-per-file none --tokenizer-model v2/output/bpe_tokenizer.model
```

The command writes `manifest.json` beside `train-000000.npz`,
`train-000001.npz`, and so on.

### Train the base model

Trains the MLX SML language model from prepared pretraining shards under
`v2/output/pretraining_data/` (`manifest.json` plus `train-*.npz`). Run data
preparation first, and train the tokenizer unless you pass an existing model
with `--tokenizer-model`.

```sh
uv run python v2/src/prepare_pretraining_data.py
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
