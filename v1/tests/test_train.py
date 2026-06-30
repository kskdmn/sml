import json
import inspect
import sys
import tempfile
import unittest
from unittest import mock
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

    def test_training_config_disables_input_file_shuffle_by_default(self):
        from sml_config import TrainingConfig

        self.assertFalse(TrainingConfig().shuffle_input_file_order)

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

    def test_shuffle_input_files_uses_seeded_deterministic_order(self):
        import train_sml

        files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )

        first_shuffle = train_sml.shuffle_input_files(files, seed=42)
        second_shuffle = train_sml.shuffle_input_files(files, seed=42)

        self.assertEqual(
            [
                "pile-0002.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0003.jsonl.zst",
                "pile-0000.jsonl.zst",
            ],
            [path.name for path in first_shuffle],
        )
        self.assertEqual(first_shuffle, second_shuffle)

    def test_shuffle_input_files_uses_seed_to_change_order(self):
        import train_sml

        files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )

        self.assertEqual(
            [
                "pile-0002.jsonl.zst",
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0003.jsonl.zst",
            ],
            [path.name for path in train_sml.shuffle_input_files(files, seed=99)],
        )

    def test_shuffle_input_files_returns_tuple_without_mutating_input(self):
        import train_sml

        files = [
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
            Path("pile-0002.jsonl.zst"),
            Path("pile-0003.jsonl.zst"),
        ]
        original_names = [path.name for path in files]

        shuffled = train_sml.shuffle_input_files(files, seed=42)

        self.assertIsInstance(shuffled, tuple)
        self.assertEqual(original_names, [path.name for path in files])

    def test_training_config_shuffles_input_files_by_default(self):
        from sml_config import TrainingConfig

        self.assertIs(True, TrainingConfig().shuffle_input_files)

    def test_train_model_shuffles_discovered_input_files_before_loading_tokenizer(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
        )
        shuffled = tuple(reversed(discovered))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
                shuffle_input_files=True,
                seed=123,
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "shuffle_input_files",
                    return_value=shuffled,
                ) as shuffle,
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    side_effect=RuntimeError("stop after shuffle"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after shuffle"):
                    train_sml.train_model(training_config)

        shuffle.assert_called_once_with(discovered, seed=123)

    def test_train_model_can_keep_discovered_input_file_order(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
                shuffle_input_files=False,
                seed=123,
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "shuffle_input_files",
                    side_effect=AssertionError("shuffle should be skipped"),
                ),
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    side_effect=RuntimeError("stop after input order"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after input order"):
                    train_sml.train_model(training_config)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_model_keeps_discovered_input_file_order_by_default(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered_files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir) / "input",
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
            )
            tokenizer = mock.Mock()
            tokenizer.get_piece_size.return_value = 8
            model = mock.Mock()
            model.to.return_value = model
            optimizer = mock.Mock()
            scheduler = mock.Mock()

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered_files,
                ),
                mock.patch.object(train_sml, "shuffle_input_files") as shuffle_files,
                mock.patch.object(train_sml, "load_tokenizer", return_value=tokenizer),
                mock.patch.object(train_sml, "SMLLanguageModel", return_value=model),
                mock.patch.object(
                    train_sml,
                    "resolve_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    train_sml,
                    "resolve_autocast_dtype",
                    return_value=None,
                ),
                mock.patch.object(train_sml, "count_parameters", return_value=(0, 0)),
                mock.patch.object(train_sml.torch.optim, "AdamW", return_value=optimizer),
                mock.patch.object(
                    train_sml.torch.optim.lr_scheduler,
                    "LambdaLR",
                    return_value=scheduler,
                ),
                mock.patch.object(train_sml, "save_checkpoint"),
                mock.patch.object(train_sml, "build_dataloader", return_value=[]) as build,
            ):
                train_sml.train_model(training_config)

        shuffle_files.assert_not_called()
        self.assertEqual(discovered_files, build.call_args.kwargs["input_files"])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_model_shuffles_discovered_input_file_order_when_enabled(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered_files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )
        shuffled_files = (
            discovered_files[2],
            discovered_files[1],
            discovered_files[3],
            discovered_files[0],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir) / "input",
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
                shuffle_input_file_order=True,
                seed=42,
            )
            tokenizer = mock.Mock()
            tokenizer.get_piece_size.return_value = 8
            model = mock.Mock()
            model.to.return_value = model
            optimizer = mock.Mock()
            scheduler = mock.Mock()

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered_files,
                ),
                mock.patch.object(
                    train_sml,
                    "shuffle_input_files",
                    return_value=shuffled_files,
                ) as shuffle_files,
                mock.patch.object(train_sml, "load_tokenizer", return_value=tokenizer),
                mock.patch.object(train_sml, "SMLLanguageModel", return_value=model),
                mock.patch.object(
                    train_sml,
                    "resolve_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    train_sml,
                    "resolve_autocast_dtype",
                    return_value=None,
                ),
                mock.patch.object(train_sml, "count_parameters", return_value=(0, 0)),
                mock.patch.object(train_sml.torch.optim, "AdamW", return_value=optimizer),
                mock.patch.object(
                    train_sml.torch.optim.lr_scheduler,
                    "LambdaLR",
                    return_value=scheduler,
                ),
                mock.patch.object(train_sml, "save_checkpoint"),
                mock.patch.object(train_sml, "build_dataloader", return_value=[]) as build,
            ):
                train_sml.train_model(training_config)

        shuffle_files.assert_called_once_with(discovered_files, seed=42)
        self.assertEqual(shuffled_files, build.call_args.kwargs["input_files"])

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

    def test_resolve_lr_total_steps_prefers_lr_total_steps(self):
        import train_sml
        from sml_config import TrainingConfig

        training_config = TrainingConfig(
            lr_total_steps=5_000,
            max_steps=1_000,
        )

        self.assertEqual(5_000, train_sml.resolve_lr_total_steps(training_config))

    def test_resolve_lr_total_steps_falls_back_to_max_steps(self):
        import train_sml
        from sml_config import TrainingConfig

        training_config = TrainingConfig(lr_total_steps=None, max_steps=1_000)

        self.assertEqual(1_000, train_sml.resolve_lr_total_steps(training_config))

    def test_resolve_lr_total_steps_is_none_without_lr_or_max_steps(self):
        import train_sml
        from sml_config import TrainingConfig

        training_config = TrainingConfig(lr_total_steps=None, max_steps=None)

        self.assertIsNone(train_sml.resolve_lr_total_steps(training_config))

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

    def test_format_training_log_includes_reading_progress(self):
        import train_sml

        log_line = train_sml.format_training_log(
            epoch=1,
            global_step=10,
            lr=0.0003,
            avg_loss=9.8457,
            grad_norm=6.069,
            timestamp=datetime(2026, 6, 30, 7, 50, 0),
            progress=train_sml.ReadingProgress(
                input_file="pile-0000.jsonl.zst",
                line_number=42,
            ),
        )

        self.assertEqual(
            "time=2026-06-30 07:50:00 epoch=1 step=10 "
            "input=pile-0000.jsonl.zst line=42 "
            "lr=3.000e-04 loss=9.8457 grad_norm=6.069 (before clipping)",
            log_line,
        )

    def test_iter_texts_updates_reading_progress(self):
        import train_sml

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pile-0000.jsonl.zst"
            write_zst_rows(
                path,
                [{"text": "a" * 100}, {"text": "too short"}, {"other": "missing"}],
            )
            progress = train_sml.ReadingProgress()
            texts = list(
                train_sml.iter_texts([path], max_rows_per_file=None, progress=progress)
            )

        self.assertEqual(["a" * 100], texts)
        self.assertEqual("pile-0000.jsonl.zst", progress.input_file)
        self.assertEqual(3, progress.line_number)

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
