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

    def get_piece_size(self):
        return 16


class TrainDataTest(unittest.TestCase):
    def tiny_config(self):
        from sml_config import SMLConfig

        return SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=16,
            rope_scaling_factor=2.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_checkpointing=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )

    def test_parse_args_defaults_to_fresh_training(self):
        import train_sml

        args = train_sml.parse_args([])

        self.assertFalse(args.resume)

    def test_parse_args_enables_resume(self):
        import train_sml

        args = train_sml.parse_args(["--resume"])

        self.assertTrue(args.resume)

    def test_main_passes_resume_flag_to_training_config(self):
        import train_sml

        with mock.patch.object(
            train_sml,
            "train_model",
            return_value=Path("/tmp/sml.pt"),
        ) as train_model:
            return_code = train_sml.main(["--resume"])

        self.assertEqual(train_sml.SUCCESS_RETURN_CODE, return_code)
        self.assertTrue(train_model.call_args.kwargs["resume_from_checkpoint"])

    def test_train_model_accepts_config_objects_and_resume_flag(self):
        import train_sml

        parameters = inspect.signature(train_sml.train_model).parameters

        self.assertEqual(
            ["training_config", "model_config", "resume_from_checkpoint"],
            list(parameters),
        )

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

    def test_training_config_does_not_store_resume_cli_state(self):
        from sml_config import TrainingConfig

        self.assertFalse(hasattr(TrainingConfig(), "resume_from_checkpoint"))

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
    def test_train_model_starts_fresh_without_resume_even_when_checkpoint_exists(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (Path("pile-0000.jsonl.zst"),)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            training_config = TrainingConfig(
                input_dir=root,
                output_dir=root / "output",
                tokenizer_model_path=root / "tokenizer.model",
                checkpoint_name="sml.pt",
                device="cpu",
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                mock.patch.object(
                    train_sml,
                    "load_training_checkpoint",
                    side_effect=AssertionError("checkpoint should not be loaded"),
                ),
                mock.patch.object(
                    train_sml,
                    "build_dataloader",
                    side_effect=RuntimeError("stop after dataloader"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after dataloader"):
                    train_sml.train_model(
                        training_config,
                        model_config=self.tiny_config(),
                    )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_model_restarts_from_checkpoint_name_when_resume_is_enabled(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (Path("pile-0000.jsonl.zst"),)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            training_config = TrainingConfig(
                input_dir=root,
                output_dir=root / "output",
                tokenizer_model_path=root / "tokenizer.model",
                checkpoint_name="sml.pt",
                device="cpu",
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                mock.patch.object(
                    train_sml,
                    "load_training_checkpoint",
                    return_value=train_sml.TrainingResumeState(),
                ) as load_training_checkpoint,
                mock.patch.object(
                    train_sml,
                    "build_dataloader",
                    side_effect=RuntimeError("stop after dataloader"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after dataloader"):
                    train_sml.train_model(
                        training_config,
                        model_config=self.tiny_config(),
                        resume_from_checkpoint=True,
                    )

        load_training_checkpoint.assert_called_once()
        args = load_training_checkpoint.call_args.args
        self.assertEqual(training_config.output_dir / "sml.pt", args[0])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_train_model_uses_checkpoint_input_file_order_when_resume_is_enabled(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
        )
        checkpoint_order = (
            Path("pile-0001.jsonl.zst"),
            Path("pile-0000.jsonl.zst"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            training_config = TrainingConfig(
                input_dir=root,
                output_dir=root / "output",
                tokenizer_model_path=root / "tokenizer.model",
                checkpoint_name="sml.pt",
                device="cpu",
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                mock.patch.object(
                    train_sml,
                    "load_training_checkpoint",
                    return_value=train_sml.TrainingResumeState(
                        step=0,
                        input_files=checkpoint_order,
                        data_state=train_sml.TrainingDataState(),
                    ),
                ),
                mock.patch.object(
                    train_sml,
                    "build_dataloader",
                    side_effect=RuntimeError("stop after dataloader"),
                ) as build_dataloader,
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after dataloader"):
                    train_sml.train_model(
                        training_config,
                        model_config=self.tiny_config(),
                        resume_from_checkpoint=True,
                    )

        self.assertEqual(checkpoint_order, build_dataloader.call_args.kwargs["input_files"])

    def test_count_resume_batches_uses_completed_optimizer_steps(self):
        import train_sml
        from sml_config import TrainingConfig

        training_config = TrainingConfig(gradient_accumulation_steps=8)

        self.assertEqual(
            56,
            train_sml.count_resume_batches(
                global_step=7,
                training_config=training_config,
            ),
        )

    def test_iter_unseen_batches_skips_consumed_batches_across_dataloaders(self):
        import train_sml

        progress = train_sml.ResumeProgress(batches_to_skip=3)

        first_epoch = list(train_sml.iter_unseen_batches(["a", "b"], progress))
        second_epoch = list(train_sml.iter_unseen_batches(["c", "d", "e"], progress))

        self.assertEqual([], first_epoch)
        self.assertEqual(["d", "e"], second_epoch)
        self.assertEqual(0, progress.batches_to_skip)

    def test_iter_texts_resumes_after_training_data_state_line_number(self):
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
            data_state = train_sml.TrainingDataState(
                input_file_index=0,
                line_number=2,
            )

            texts = list(
                train_sml.iter_texts(
                    [path],
                    max_rows_per_file=None,
                    data_state=data_state,
                )
            )

        self.assertEqual(["c" * 100], texts)
        self.assertEqual(3, data_state.line_number)

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

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_load_training_checkpoint_raises_when_checkpoint_is_absent(self):
        import train_sml
        from sml import SMLLanguageModel

        model = SMLLanguageModel(self.tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml.pt"
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint does not exist"):
                train_sml.load_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    torch.device("cpu"),
                )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_save_and_load_training_checkpoint_restores_training_and_data_state(self):
        import train_sml
        from sml import SMLLanguageModel

        config = self.tiny_config()
        source_model = SMLLanguageModel(config)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.1)
        source_scheduler = torch.optim.lr_scheduler.LambdaLR(
            source_optimizer,
            lambda step: 1.0,
        )
        input_ids = torch.tensor([[1, 4, 5]])
        labels = torch.tensor([[4, 5, 6]])
        loss = source_model(input_ids, labels=labels).loss
        self.assertIsNotNone(loss)
        loss.backward()
        source_optimizer.step()
        source_scheduler.step()

        target_model = SMLLanguageModel(config)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.1)
        target_scheduler = torch.optim.lr_scheduler.LambdaLR(
            target_optimizer,
            lambda step: 1.0,
        )
        data_state = train_sml.TrainingDataState(
            epoch=2,
            input_file_index=1,
            line_number=42,
            token_buffer=[4, 5, 6],
        )
        input_files = (
            Path("/data/pile-0001.jsonl.zst"),
            Path("/data/pile-0000.jsonl.zst"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "sml.pt"
            train_sml.save_checkpoint(
                checkpoint_path,
                source_model,
                source_optimizer,
                source_scheduler,
                config,
                train_sml.TrainingConfig(checkpoint_name="sml.pt"),
                step=7,
                input_files=input_files,
                data_state=data_state,
            )

            resume_state = train_sml.load_training_checkpoint(
                checkpoint_path,
                target_model,
                target_optimizer,
                target_scheduler,
                torch.device("cpu"),
            )

        self.assertEqual(7, resume_state.step)
        self.assertEqual(input_files, resume_state.input_files)
        self.assertEqual(data_state, resume_state.data_state)
        self.assertEqual(
            source_scheduler.state_dict()["last_epoch"],
            target_scheduler.state_dict()["last_epoch"],
        )
        self.assertEqual(
            len(source_optimizer.state_dict()["state"]),
            len(target_optimizer.state_dict()["state"]),
        )
        for source_param, target_param in zip(
            source_model.parameters(),
            target_model.parameters(),
            strict=True,
        ):
            self.assertTrue(torch.equal(source_param, target_param))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_save_and_load_training_checkpoint_restores_rng_state(self):
        import random
        import train_sml
        from sml import SMLLanguageModel

        original_python_rng_state = random.getstate()
        original_torch_rng_state = torch.get_rng_state()
        try:
            config = self.tiny_config()
            source_model = SMLLanguageModel(config)
            source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.1)
            source_scheduler = torch.optim.lr_scheduler.LambdaLR(
                source_optimizer,
                lambda step: 1.0,
            )
            target_model = SMLLanguageModel(config)
            target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.1)
            target_scheduler = torch.optim.lr_scheduler.LambdaLR(
                target_optimizer,
                lambda step: 1.0,
            )

            random.seed(123)
            torch.manual_seed(456)
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint_path = Path(tmp_dir) / "sml.pt"
                train_sml.save_checkpoint(
                    checkpoint_path,
                    source_model,
                    source_optimizer,
                    source_scheduler,
                    config,
                    train_sml.TrainingConfig(checkpoint_name="sml.pt"),
                    step=7,
                )

                expected_python_value = random.random()
                expected_torch_values = torch.rand(3)
                random.seed(999)
                torch.manual_seed(999)

                train_sml.load_training_checkpoint(
                    checkpoint_path,
                    target_model,
                    target_optimizer,
                    target_scheduler,
                    torch.device("cpu"),
                )

            self.assertEqual(expected_python_value, random.random())
            self.assertTrue(torch.equal(expected_torch_values, torch.rand(3)))
        finally:
            random.setstate(original_python_rng_state)
            torch.set_rng_state(original_torch_rng_state)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_restore_rng_state_moves_torch_cpu_rng_state_to_cpu(self):
        import random
        import train_sml

        mapped_rng_state = mock.Mock()
        cpu_rng_state = mock.Mock()
        mapped_rng_state.cpu.return_value = cpu_rng_state
        checkpoint = {
            "python_rng_state": random.getstate(),
            "torch_rng_state": mapped_rng_state,
            "cuda_rng_state_all": None,
            "mps_rng_state": None,
        }

        with (
            mock.patch.object(train_sml.random, "setstate"),
            mock.patch.object(train_sml.torch, "set_rng_state") as set_rng_state,
        ):
            train_sml.restore_rng_state(checkpoint)

        mapped_rng_state.cpu.assert_called_once_with()
        set_rng_state.assert_called_once_with(cpu_rng_state)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_restore_rng_state_moves_cuda_rng_states_to_cpu(self):
        import random
        import train_sml

        mapped_cuda_rng_state = mock.Mock()
        cpu_cuda_rng_state = mock.Mock()
        mapped_cuda_rng_state.cpu.return_value = cpu_cuda_rng_state
        checkpoint = {
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": [mapped_cuda_rng_state],
            "mps_rng_state": None,
        }

        with (
            mock.patch.object(train_sml.random, "setstate"),
            mock.patch.object(train_sml.torch, "set_rng_state"),
            mock.patch.object(train_sml.torch.cuda, "is_available", return_value=True),
            mock.patch.object(train_sml.torch.cuda, "set_rng_state_all") as set_rng_state_all,
        ):
            train_sml.restore_rng_state(checkpoint)

        mapped_cuda_rng_state.cpu.assert_called_once_with()
        set_rng_state_all.assert_called_once_with([cpu_cuda_rng_state])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_restore_rng_state_moves_mps_rng_state_to_cpu(self):
        import random
        import train_sml

        mapped_mps_rng_state = mock.Mock()
        cpu_mps_rng_state = mock.Mock()
        mapped_mps_rng_state.cpu.return_value = cpu_mps_rng_state
        checkpoint = {
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": None,
            "mps_rng_state": mapped_mps_rng_state,
        }

        with (
            mock.patch.object(train_sml.random, "setstate"),
            mock.patch.object(train_sml.torch, "set_rng_state"),
            mock.patch.object(train_sml, "is_mps_rng_available", return_value=True),
            mock.patch.object(train_sml.torch.mps, "set_rng_state") as set_rng_state,
        ):
            train_sml.restore_rng_state(checkpoint)

        mapped_mps_rng_state.cpu.assert_called_once_with()
        set_rng_state.assert_called_once_with(cpu_mps_rng_state)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_token_block_dataset_updates_training_data_state_after_yield(self):
        import train_sml

        data_state = train_sml.TrainingDataState()
        dataset = train_sml.TokenBlockDataset(
            texts=iter(["4 5 6 7 8"]),
            tokenizer=FakeTokenizer(),
            sequence_length=3,
            data_state=data_state,
        )

        first = next(iter(dataset))

        self.assertTrue(torch.equal(torch.tensor([1, 4, 5]), first["input_ids"]))
        self.assertEqual([6, 7, 8, 2], data_state.token_buffer)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_token_block_dataset_resumes_from_training_data_state_token_buffer(self):
        import train_sml

        data_state = train_sml.TrainingDataState(token_buffer=[6, 7, 8, 2])
        dataset = train_sml.TokenBlockDataset(
            texts=iter([]),
            tokenizer=FakeTokenizer(),
            sequence_length=3,
            data_state=data_state,
        )

        first = next(iter(dataset))

        self.assertTrue(torch.equal(torch.tensor([6, 7, 8]), first["input_ids"]))
        self.assertTrue(torch.equal(torch.tensor([7, 8, 2]), first["labels"]))
        self.assertEqual([2], data_state.token_buffer)

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

    def test_format_training_log_includes_example_index(self):
        import train_sml

        log_line = train_sml.format_training_log(
            epoch=1,
            global_step=10,
            lr=0.0003,
            avg_loss=9.8457,
            grad_norm=6.069,
            timestamp=datetime(2026, 6, 30, 7, 50, 0),
            progress=train_sml.ReadingProgress(
                input_file="swag-train",
                line_number=42,
                example_index=17_203,
            ),
        )

        self.assertEqual(
            "time=2026-06-30 07:50:00 epoch=1 step=10 "
            "input=swag-train line=42 example=17203 "
            "lr=3.000e-04 loss=9.8457 grad_norm=6.069 (before clipping)",
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
