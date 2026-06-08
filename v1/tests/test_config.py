import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "src" / "config.py"


def load_config():
    spec = importlib.util.spec_from_file_location("config", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConfigTest(unittest.TestCase):
    def test_common_constants_are_defined_from_current_version(self):
        config = load_config()

        self.assertEqual(0, config.SUCCESS_RETURN_CODE)
        self.assertEqual(PROJECT_DIR, config.PROJECT_DIR)
        self.assertEqual(PROJECT_DIR / "output", config.OUTPUT_DIR)
        self.assertEqual(
            config.OUTPUT_DIR / "bpe_tokenizer.model",
            config.TOKENIZER_MODEL_PATH,
        )


if __name__ == "__main__":
    unittest.main()
