import importlib
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestModuleLayout:
    def test_tokenizer_module_owns_corpus_helpers(self):
        tokenizer = importlib.import_module("tokenizer")

        assert hasattr(tokenizer, "FilteredTextIterable")
        assert hasattr(tokenizer, "discover_input_files")
        assert hasattr(tokenizer, "filter_text")

    def test_model_training_and_inference_configs_live_with_owners(self):
        try:
            sml = importlib.import_module("sml")
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"mlx is not available: {exc}")

        train_sml = importlib.import_module("train_sml")
        infer_sml = importlib.import_module("infer_sml")
        lora = importlib.import_module("lora")
        ft_swag = importlib.import_module("ft_swag")

        assert hasattr(sml, "SMLConfig")
        assert hasattr(sml, "SMLLanguageModel")
        assert hasattr(train_sml, "TrainingConfig")
        assert hasattr(infer_sml, "GenerationConfig")
        assert hasattr(lora, "LoRAConfig")
        assert hasattr(ft_swag, "SwagFineTuneConfig")

    def test_removed_modules_are_absent(self):
        assert not (SRC_DIR / "sml_config.py").exists()
        assert not (SRC_DIR / "eval_humaneval.py").exists()
        assert not (PROJECT_DIR / "tests" / "test_eval_humaneval.py").exists()
