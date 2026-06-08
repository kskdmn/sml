# SML

## Environment

- MacBook Air M5 24GB

## Inference

Run the saved `v1/output/sml.pt` checkpoint with:

```bash
uv run python v1/src/infer_sml.py "Hello" --max-new-tokens 50
```

Useful options:

```bash
uv run python v1/src/infer_sml.py "Hello" --include-prompt
```
