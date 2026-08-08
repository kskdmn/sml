import importlib
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestModuleLayout:
    def test_utils_module_owns_shared_corpus_helpers(self):
        utils = importlib.import_module("utils")

        assert hasattr(utils, "discover_input_files")
        assert hasattr(utils, "filter_text")
        assert hasattr(utils, "iter_jsonl_records")
        assert hasattr(utils, "load_tokenizer")
        assert hasattr(utils, "lr_lambda")
        assert hasattr(utils, "build_lr_schedule")

    def test_tokenizer_module_owns_sentencepiece_options(self):
        tokenizer = importlib.import_module("tokenizer")
        utils = importlib.import_module("utils")

        assert hasattr(tokenizer, "FilteredTextIterable")
        assert 28_672 == tokenizer.VOCAB_SIZE
        assert "bpe" == tokenizer.MODEL_TYPE
        assert not hasattr(tokenizer, "discover_input_files")
        assert tokenizer.filter_text is utils.filter_text

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

    def test_shared_input_constants_live_in_config(self):
        config = importlib.import_module("config")
        train_tokenizer = importlib.import_module("train_tokenizer")
        prepare_pretraining_data = importlib.import_module("prepare_pretraining_data")

        assert (
            r".*-00[0-9][0-9]\.jsonl\.zst\Z" == config.PRETRAINING_INPUT_FILE_NAME_REGEX
        )
        assert (
            train_tokenizer.PRETRAINING_INPUT_FILE_NAME_REGEX
            is config.PRETRAINING_INPUT_FILE_NAME_REGEX
        )
        assert (
            prepare_pretraining_data.PRETRAINING_INPUT_FILE_NAME_REGEX
            is config.PRETRAINING_INPUT_FILE_NAME_REGEX
        )
        assert train_tokenizer.PRETRAINING_INPUT_DIR is config.PRETRAINING_INPUT_DIR
        assert (
            prepare_pretraining_data.PRETRAINING_INPUT_DIR
            is config.PRETRAINING_INPUT_DIR
        )
        assert 8_192 == train_tokenizer.MAX_ROWS_PER_FILE

    def test_removed_modules_are_absent(self):
        assert not (SRC_DIR / "sml_config.py").exists()
        assert not (SRC_DIR / "sml_mlx.py").exists()
        assert not (SRC_DIR / "train_sml_mlx.py").exists()
        assert not (SRC_DIR / "eval_humaneval.py").exists()
        assert not (SRC_DIR / "eval_utils.py").exists()
        assert not (SRC_DIR / "eval_hellaswag.py").exists()
        assert not (SRC_DIR / "eval_winogrande.py").exists()
        assert not (PROJECT_DIR / "tests" / "test_eval_humaneval.py").exists()
        assert not (PROJECT_DIR / "tests" / "test_eval_utils.py").exists()
        assert not (PROJECT_DIR / "tests" / "test_eval_hellaswag.py").exists()
        assert not (PROJECT_DIR / "tests" / "test_eval_winogrande.py").exists()
