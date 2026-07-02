# SML

## Environment

- MacBook Air M5 24GB

## Training

Resume from the saved checkpoint.

```bash
uv run python v1/src/train_sml.py --resume
```

## Inference

Run the saved `v1/output/sml.pt` checkpoint with:

```bash
uv run python v1/src/infer_sml.py "Hello." --max-new-tokens 500
```

Useful options:

```bash
uv run python v1/src/infer_sml.py "Hello." --include-prompt

uv run python v1/src/infer_sml.py "Hello." --max-new-tokens 500 \
  --repetition-penalty 1.15 --no-repeat-ngram-size 4
```

## Other

Read the saved checkpoint.

```bash
uv run python -c "import torch; x=torch.load('v1/output/sml.pt', map_location='cpu', weights_only=False); print(x.keys())"
```
