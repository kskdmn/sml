import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import zstandard as zstd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "src" / "train_tokenizer.py"
MAX_ROWS_IN_TEST = 3
EXPECTED_NORMALIZED_TEXT_COUNT = 1
REPEATED_WORD_COUNT = 60


def load_script():
    spec = importlib.util.spec_from_file_location("train_tokenizer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"builtin type SwigPy.* has no __module__ attribute",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"builtin type swigvarlink has no __module__ attribute",
            category=DeprecationWarning,
        )
        spec.loader.exec_module(module)
    return module


class TrainTokenizerTest(unittest.TestCase):
    def test_output_dir_defaults_to_current_numbered_project(self):
        tokenizer = load_script()

        self.assertEqual(PROJECT_DIR / "output", tokenizer.OUTPUT_DIR)

    def test_discover_input_files_sorts_supported_files_and_skips_hidden_files(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "z.jsonl").write_text("", encoding="utf-8")
            (root / "a.jsonl.zst").write_text("", encoding="utf-8")
            (root / ".DS_Store").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")

            files = tokenizer.discover_input_files(root)

        self.assertEqual(["a.jsonl.zst", "z.jsonl"], [path.name for path in files])

    def test_iter_filtered_texts_reads_only_first_rows_and_filters_text_length(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.jsonl"
            rows = [
                {"text": "a" * tokenizer.MIN_TEXT_LENGTH},
                {"text": "too short"},
                {"text": "b" * tokenizer.MAX_TEXT_LENGTH},
                {"text": "c" * tokenizer.MIN_TEXT_LENGTH},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            texts = list(
                tokenizer.iter_filtered_texts(
                    [path],
                    max_rows_per_file=MAX_ROWS_IN_TEST,
                )
            )

        self.assertEqual(
            ["a" * tokenizer.MIN_TEXT_LENGTH, "b" * tokenizer.MAX_TEXT_LENGTH],
            texts,
        )

    def test_iter_filtered_texts_normalizes_multiline_text_to_one_line(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.jsonl"
            text = "word\n" * REPEATED_WORD_COUNT
            path.write_text(json.dumps({"text": text}), encoding="utf-8")

            texts = list(tokenizer.iter_filtered_texts([path]))

        self.assertEqual(EXPECTED_NORMALIZED_TEXT_COUNT, len(texts))
        self.assertNotIn("\n", texts[0])
        self.assertGreaterEqual(len(texts[0]), tokenizer.MIN_TEXT_LENGTH)

    def test_iter_jsonl_lines_reads_zst_files_without_subprocess(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.jsonl.zst"
            text = "\n".join(json.dumps({"text": value}) for value in ("first", "second"))
            compressed = zstd.ZstdCompressor().compress(text.encode("utf-8"))
            path.write_bytes(compressed)

            with mock.patch(
                "subprocess.Popen",
                side_effect=AssertionError("zstd files must not shell out"),
            ):
                lines = list(tokenizer.iter_jsonl_lines(path, max_rows_per_file=1))

        self.assertEqual([(1, '{"text": "first"}\n')], lines)


if __name__ == "__main__":
    unittest.main()
