import importlib.util
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "src" / "config.py"


def load_config():
    spec = importlib.util.spec_from_file_location("config", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestConfig:
    def test_common_constants_are_defined_from_current_version(self):
        config = load_config()

        assert 0 == config.SUCCESS_RETURN_CODE
        assert PROJECT_DIR == config.PROJECT_DIR
        assert PROJECT_DIR / "output" == config.OUTPUT_DIR
        assert config.OUTPUT_DIR / "sml" == config.DEFAULT_MODEL_PATH
        assert (
            config.OUTPUT_DIR / "bpe_tokenizer.model"
            == config.DEFAULT_TOKENIZER_MODEL_PATH
        )
        assert (
            Path("~/Documents/training_data-common_pile/")
            == config.PRETRAINING_INPUT_DIR
        )
        assert (
            r".*-00[0-9][0-9]\.jsonl\.zst\Z" == config.PRETRAINING_INPUT_FILE_NAME_REGEX
        )
        assert "model.safetensors" == config.MODEL_WEIGHTS_NAME
        assert "optimizer.npz" == config.OPTIMIZER_STATE_NAME
        assert "metadata.json" == config.METADATA_NAME
        assert 0 == config.UNK_TOKEN_ID
        assert 1 == config.BOS_TOKEN_ID
        assert 2 == config.EOS_TOKEN_ID
        assert 3 == config.PAD_TOKEN_ID
        assert Path("/tmp/example") == config.resolve_path(Path("/tmp/example"))
        assert Path.home() / "example" == config.resolve_path(Path("~/example"))
