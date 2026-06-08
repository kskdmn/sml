from __future__ import annotations

from pathlib import Path


SUCCESS_RETURN_CODE = 0
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"
