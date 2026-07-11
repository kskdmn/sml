from __future__ import annotations

from pathlib import Path


SUCCESS_RETURN_CODE = 0
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "sml"
DEFAULT_TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"


def resolve_path(path: Path) -> Path:
    """
    Only expand `~`; callers decide separately whether the resulting path must exist.
    """
    return path.expanduser()
