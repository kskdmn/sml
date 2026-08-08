from __future__ import annotations

from pathlib import Path

SUCCESS_RETURN_CODE = 0
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "sml"
DEFAULT_TOKENIZER_MODEL_PATH = OUTPUT_DIR / "bpe_tokenizer.model"
PRETRAINING_INPUT_DIR = Path("~/Documents/training_data-common_pile/")
PRETRAINING_INPUT_FILE_NAME_REGEX = r".*-00[0-9][0-9]\.jsonl\.zst\Z"
MODEL_WEIGHTS_NAME = "model.safetensors"
OPTIMIZER_STATE_NAME = "optimizer.npz"
METADATA_NAME = "metadata.json"
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3


def resolve_path(path: Path) -> Path:
    """
    Only expand `~`; callers decide separately whether the resulting path must exist.
    """
    return path.expanduser()
