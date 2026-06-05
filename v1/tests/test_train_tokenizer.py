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


def write_zst_text(path: Path, text: str) -> None:
    compressed = zstd.ZstdCompressor().compress(text.encode("utf-8"))
    path.write_bytes(compressed)


def write_zst_rows(path: Path, rows: list[dict[str, object]]) -> None:
    write_zst_text(path, "\n".join(json.dumps(row) for row in rows))


def collect_filtered_texts(tokenizer, input_files, max_rows_per_file=None):
    if max_rows_per_file is None:
        return list(tokenizer.FilteredTextIterable(input_files))

    return list(tokenizer.FilteredTextIterable(input_files, max_rows_per_file))


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

    def test_module_does_not_export_test_only_filtered_text_wrapper(self):
        tokenizer = load_script()

        self.assertFalse(hasattr(tokenizer, "iter_filtered_texts"))

    def test_module_does_not_export_jsonl_lines_wrapper(self):
        tokenizer = load_script()

        self.assertFalse(hasattr(tokenizer, "iter_jsonl_lines"))

    def test_discover_input_files_sorts_matching_zst_shards_and_skips_others(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "dataset-0009.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0000.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0010.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0000.jsonl").write_text("", encoding="utf-8")
            (root / "dataset.jsonl.zst").write_text("", encoding="utf-8")
            (root / ".dataset-0001.jsonl.zst").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")

            files = tokenizer.discover_input_files(root)

        self.assertEqual(
            ["dataset-0000.jsonl.zst", "dataset-0009.jsonl.zst"],
            [path.name for path in files],
        )

    def test_filtered_text_iterable_reads_only_first_rows_and_filters_text_length(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            rows = [
                {"text": "a" * tokenizer.MIN_TEXT_LENGTH},
                {"text": "too short"},
                {"text": "b" * tokenizer.MAX_TEXT_LENGTH},
                {"text": "c" * tokenizer.MIN_TEXT_LENGTH},
            ]
            write_zst_rows(path, rows)

            texts = collect_filtered_texts(
                tokenizer,
                [path],
                max_rows_per_file=MAX_ROWS_IN_TEST,
            )

        self.assertEqual(
            ["a" * tokenizer.MIN_TEXT_LENGTH, "b" * tokenizer.MAX_TEXT_LENGTH],
            texts,
        )

    def test_filtered_text_iterable_filters_text_longer_than_max_utf8_bytes(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            byte_long_text = "\u3042" * (tokenizer.MAX_TEXT_LENGTH // 2 + 1)
            valid_text = "a" * tokenizer.MIN_TEXT_LENGTH
            write_zst_rows(path, [{"text": byte_long_text}, {"text": valid_text}])

            texts = collect_filtered_texts(tokenizer, [path])

        self.assertEqual([valid_text], texts)

    def test_filtered_text_iterable_normalizes_multiline_text_to_one_line(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = "word\n" * REPEATED_WORD_COUNT
            write_zst_rows(path, [{"text": text}])

            texts = collect_filtered_texts(tokenizer, [path])

        self.assertEqual(EXPECTED_NORMALIZED_TEXT_COUNT, len(texts))
        self.assertNotIn("\n", texts[0])
        self.assertGreaterEqual(len(texts[0]), tokenizer.MIN_TEXT_LENGTH)

    def test_filtered_text_iterable_removes_null_characters(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = ("a" * 50) + "\x00" + ("b" * 50)
            write_zst_rows(path, [{"text": text}])

            texts = collect_filtered_texts(tokenizer, [path])

        self.assertEqual(EXPECTED_NORMALIZED_TEXT_COUNT, len(texts))
        self.assertNotIn("\x00", texts[0])

    def test_filtered_text_iterable_does_not_recheck_discovered_file_names(self):
        tokenizer = load_script()

        class MatchCountingPattern:
            def __init__(self):
                self.names = []

            def fullmatch(self, name):
                self.names.append(name)
                return object()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            expected_text = "a" * tokenizer.MIN_TEXT_LENGTH
            write_zst_rows(input_path, [{"text": expected_text}])
            pattern = MatchCountingPattern()

            with mock.patch.object(tokenizer, "INPUT_FILE_NAME_PATTERN", pattern):
                input_files = tokenizer.discover_input_files(root)
                texts = collect_filtered_texts(tokenizer, input_files)

        self.assertEqual([expected_text], texts)
        self.assertEqual(["sample-0000.jsonl.zst"], pattern.names)

    def test_iter_zstd_jsonl_lines_reads_zst_files_without_subprocess(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = "\n".join(json.dumps({"text": value}) for value in ("first", "second"))
            write_zst_text(path, text)

            with mock.patch(
                "subprocess.Popen",
                side_effect=AssertionError("zstd files must not shell out"),
            ):
                lines = list(tokenizer.iter_zstd_jsonl_lines(path, max_rows_per_file=1))

        self.assertEqual([(1, '{"text": "first"}\n')], lines)

    def test_train_tokenizer_enables_byte_fallback(self):
        tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            write_zst_rows(input_path, [{"text": "a" * tokenizer.MIN_TEXT_LENGTH}])

            with (
                mock.patch.object(tokenizer, "INPUT_DIR", root),
                mock.patch.object(tokenizer, "OUTPUT_DIR", root / "output"),
                mock.patch.object(tokenizer.spm.SentencePieceTrainer, "train") as train,
            ):
                tokenizer.train_tokenizer()

        self.assertIs(True, train.call_args.kwargs.get("byte_fallback"))


if __name__ == "__main__":
    unittest.main()
