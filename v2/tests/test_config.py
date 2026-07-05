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
        assert PROJECT_DIR / 'output' == config.OUTPUT_DIR
        assert config.OUTPUT_DIR / 'bpe_tokenizer.model' == config.TOKENIZER_MODEL_PATH
