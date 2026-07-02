# SML v1

First SML model, tokenizer, training scripts, and inference entrypoint. Run
commands from the repository root.

## Training

```bash
uv run python v1/src/train_sml.py
uv run python v1/src/train_sml.py --resume
```

Configure training in `TrainingConfig` (`sml_config.py`). See `train_sml.py`
for `--resume` behavior.

## Checkpoint

Default path: `v1/output/sml.pt`.

```bash
uv run python -c "import torch; x=torch.load('v1/output/sml.pt', map_location='cpu', weights_only=False); print(x.keys())"
```

See `train_sml.load_training_checkpoint` and `infer_sml.load_checkpoint` for
safe loading details.

## Inference

Requires `v1/output/sml.pt` and `v1/output/bpe_tokenizer.model`.

### One-shot generation

```bash
uv run python v1/src/infer_sml.py "Hello." --max-new-tokens 50

uv run python v1/src/infer_sml.py "Hello." --include-prompt

uv run python v1/src/infer_sml.py "Hello." --max-new-tokens 500 \
  --repetition-penalty 1.15 --no-repeat-ngram-size 4

uv run python v1/src/infer_sml.py "Hello." --max-new-tokens 500 \
  --temperature 0.8 --top-p 0.9 --seed 42
```

CLI flags: `infer_sml.py --help`. Decoding details: `GenerationConfig` in
`sml_config.py`.

### OpenAI-compatible API

```bash
uv run python v1/src/infer_sml.py --serve --host 127.0.0.1 --port 8000 --model sml
```

Endpoints: `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`.
See `infer_sml.py` for request/response details.

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sml",
    "prompt": "Hello",
    "max_tokens": 50
  }'

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sml",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Explain SML in one sentence."}
    ],
    "max_tokens": 50
  }'
```

## HumanEval

```bash
uv run python v1/src/eval_humaneval.py

uv run python v1/src/eval_humaneval.py --limit 1 --device cpu
```

Results default to `v1/output/humaneval.json`. See `eval_humaneval.py` for
flags and safety notes.
