import json
import inspect
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import zstandard as zstd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import torch
except ImportError:  # pragma: no cover - exercised only before torch is installed
    torch = None


def write_zst_rows(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(row) for row in rows)
    compressed = zstd.ZstdCompressor().compress(text.encode("utf-8"))
    path.write_bytes(compressed)


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]


class TrainDataTest(unittest.TestCase):
    def test_train_module_does_not_export_cli_parsing_helpers(self):
        import train_sml

        self.assertFalse(hasattr(train_sml, "parse_args"))
        self.assertFalse(hasattr(train_sml, "parse_optional_int"))

    def test_train_model_accepts_config_objects(self):
        import train_sml

        parameters = inspect.signature(train_sml.train_model).parameters

        self.assertEqual(["training_config", "model_config"], list(parameters))

    def test_discover_input_files_uses_supplied_regex_and_sorts_matches(self):
        import train_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pile-0002.jsonl.zst").write_text("", encoding="utf-8")
            (root / "pile-0000.jsonl.zst").write_text("", encoding="utf-8")
            (root / "pile-0010.jsonl.zst").write_text("", encoding="utf-8")
            (root / "pile-0001.jsonl").write_text("", encoding="utf-8")
            (root / ".pile-0001.jsonl.zst").write_text("", encoding="utf-8")

            files = train_sml.discover_input_files(root, r".*-000[0-9]\.jsonl\.zst\Z")

        self.assertEqual(
            ["pile-0000.jsonl.zst", "pile-0002.jsonl.zst"],
            [path.name for path in files],
        )

    def test_iter_texts_streams_zst_jsonl_rows_without_loading_all_files(self):
        import train_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            first = Path(tmp_dir) / "pile-0000.jsonl.zst"
            second = Path(tmp_dir) / "pile-0001.jsonl.zst"
            write_zst_rows(
                first,
                [{"text": "a" * 100}, {"text": "too short"}, {"other": "missing"}],
            )
            write_zst_rows(second, [{"text": "b" * 100}])

            iterator = train_sml.iter_texts([first, second], max_rows_per_file=2)

            self.assertEqual("a" * 100, next(iterator))
            self.assertEqual("b" * 100, next(iterator))
            with self.assertRaises(StopIteration):
                next(iterator)

    def test_iter_texts_reads_all_rows_when_max_rows_per_file_is_none(self):
        import train_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pile-0000.jsonl.zst"
            write_zst_rows(
                path,
                [
                    {"text": "a" * 100},
                    {"text": "b" * 100},
                    {"text": "c" * 100},
                ],
            )

            texts = list(train_sml.iter_texts([path], max_rows_per_file=None))

        self.assertEqual(["a" * 100, "b" * 100, "c" * 100], texts)

    def test_step_limit_is_never_reached_when_max_steps_is_none(self):
        import train_sml

        self.assertFalse(train_sml.is_step_limit_reached(global_step=10_000, max_steps=None))

    def test_format_training_log_includes_timestamp(self):
        import train_sml

        log_line = train_sml.format_training_log(
            epoch=2,
            global_step=3,
            lr=0.0003,
            avg_loss=1.23456,
            grad_norm=5.859,
            timestamp=datetime(2026, 6, 5, 12, 34, 56),
        )

        self.assertEqual(
            "time=2026-06-05 12:34:56 epoch=2 step=3 "
            "lr=3.000e-04 loss=1.2346 grad_norm=5.859 (before clipping)",
            log_line,
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_token_block_dataset_yields_fixed_length_input_label_pairs(self):
        import train_sml

        dataset = train_sml.TokenBlockDataset(
            texts=iter(["4 5 6 7", "8 9 10"]),
            tokenizer=FakeTokenizer(),
            sequence_length=3,
        )

        first = next(iter(dataset))

        self.assertTrue(torch.equal(torch.tensor([1, 4, 5]), first["input_ids"]))
        self.assertTrue(torch.equal(torch.tensor([4, 5, 6]), first["labels"]))


if __name__ == "__main__":
    unittest.main()
