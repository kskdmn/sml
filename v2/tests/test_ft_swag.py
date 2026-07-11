import inspect
import sys
from pathlib import Path

from helpers import Spy
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class FakeSwagDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]


class TestFtSwag:
    def test_format_swag_parts_returns_parts_without_separator(self):
        import ft_swag

        row = {
            "startphrase": "The girl ",
            "ending0": " stops clutching her diary.",
            "ending1": "runs away.",
            "ending2": "looks around.",
            "ending3": "opens the door.",
            "label": 0,
        }

        assert ("The girl", "stops clutching her diary.") == ft_swag.format_swag_parts(
            row
        )

    def test_format_swag_example_concatenates_gold_ending(self):
        import ft_swag

        row = {
            "startphrase": "A man is sitting on a roof. he",
            "ending0": " starts typing on a laptop.",
            "ending1": " is using wrap to blend a container.",
            "ending2": " is ripping level tiles off.",
            "ending3": " is holding a rubik's cube.",
            "label": 0,
        }

        assert 'A man is sitting on a roof. he starts typing on a laptop.' == ft_swag.format_swag_example(row)

    def test_format_swag_example_inserts_space_before_gold_ending(self):
        import ft_swag

        row = {
            "startphrase": "The girl",
            "ending0": "stops clutching her diary.",
            "ending1": "runs away.",
            "ending2": "looks around.",
            "ending3": "opens the door.",
            "label": 0,
        }

        assert "The girl stops clutching her diary." == ft_swag.format_swag_example(row)

    def test_resolve_swag_label_accepts_string_labels(self):
        import ft_swag

        assert 2 == ft_swag.resolve_swag_label('2')

    def test_iter_swag_texts_resumes_after_saved_position(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig
        from train_sml import TrainingDataState

        rows = [
            {
                "startphrase": "first",
                "ending0": " one",
                "ending1": " two",
                "ending2": " three",
                "ending3": " four",
                "label": 0,
            },
            {
                "startphrase": "second",
                "ending0": " alpha",
                "ending1": " beta",
                "ending2": " gamma",
                "ending3": " delta",
                "label": 1,
            },
            {
                "startphrase": "third",
                "ending0": " x",
                "ending1": " y",
                "ending2": " z",
                "ending3": " w",
                "label": 2,
            },
        ]
        dataset = FakeSwagDataset(rows)
        data_state = TrainingDataState(line_number=0)

        monkeypatch.setattr(ft_swag, "load_swag_dataset", Spy(return_value=dataset))
        texts = list(
            ft_swag.iter_swag_texts(
                SwagFineTuneConfig(shuffle_examples=False, seed=42),
                epoch=0,
                data_state=data_state,
            )
        )

        assert ['second beta', 'third z'] == texts
        assert 2 == data_state.line_number

    def test_iter_swag_texts_resumes_same_shuffled_order(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig
        from train_sml import TrainingDataState

        rows = [
            {
                "startphrase": f"row-{index}",
                "ending0": " end",
                "ending1": " wrong",
                "ending2": " wrong",
                "ending3": " wrong",
                "label": 0,
            }
            for index in range(8)
        ]
        dataset = FakeSwagDataset(rows)
        config = SwagFineTuneConfig(shuffle_examples=True, seed=7)

        monkeypatch.setattr(ft_swag, "load_swag_dataset", Spy(return_value=dataset))
        full_epoch = list(ft_swag.iter_swag_texts(config, epoch=2))

        data_state = TrainingDataState(line_number=2)
        resumed_epoch = list(
            ft_swag.iter_swag_texts(config, epoch=2, data_state=data_state)
        )

        assert full_epoch[3:] == resumed_epoch
        assert 7 == data_state.line_number

    def test_iter_swag_texts_updates_reading_progress_example_index(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig
        from train_sml import ReadingProgress

        rows = [
            {
                "startphrase": "first",
                "ending0": " one",
                "ending1": " two",
                "ending2": " three",
                "ending3": " four",
                "label": 0,
            },
            {
                "startphrase": "second",
                "ending0": " alpha",
                "ending1": " beta",
                "ending2": " gamma",
                "ending3": " delta",
                "label": 1,
            },
        ]
        dataset = FakeSwagDataset(rows)
        progress = ReadingProgress()

        monkeypatch.setattr(ft_swag, "load_swag_dataset", Spy(return_value=dataset))
        texts = list(
            ft_swag.iter_swag_texts(
                SwagFineTuneConfig(shuffle_examples=False, seed=42),
                epoch=0,
                progress=progress,
            )
        )

        assert ['first one', 'second beta'] == texts
        assert 1 == progress.line_number
        assert 1 == progress.example_index

    def test_build_swag_dataloader_masks_context_and_scores_gold_ending_and_eos(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        class TextTokenizer:
            bos_id = 1
            eos_id = 2

            def encode(self, text, out_type=int):
                del out_type
                return {
                    "ctx": [10, 11],
                    " end": [20, 21],
                    "ctx end": [10, 11, 20, 21],
                }[text]

        rows = [
            {
                "startphrase": "ctx",
                "ending0": " end",
                "ending1": " wrong",
                "ending2": " wrong",
                "ending3": " wrong",
                "label": 0,
            },
        ]
        dataset = FakeSwagDataset(rows)
        config = SwagFineTuneConfig(
            sequence_length=6,
            batch_size=1,
            shuffle_examples=False,
        )

        monkeypatch.setattr(ft_swag, "load_swag_dataset", Spy(return_value=dataset))
        batches = list(
            ft_swag.build_swag_dataloader(
                fine_tune_config=config,
                tokenizer=TextTokenizer(),
                epoch=0,
            )
        )

        assert [[1, 10, 11, 20, 21, 2]] == batches[0]["input_ids"].tolist()
        assert [[3, 3, 20, 21, 2, 3]] == batches[0]["labels"].tolist()

    def test_build_swag_dataloader_pads_short_examples_without_packing(self, monkeypatch):
        import torch
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=2)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4"), ("", "5")])),
        )
        batches = list(
            ft_swag.build_swag_dataloader(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 2 == len(batches)
        assert torch.equal(torch.tensor([[1, 4, 2]]), batches[0]["input_ids"])
        assert torch.equal(torch.tensor([[4, 2, 3]]), batches[0]["labels"])
        assert torch.equal(torch.tensor([[1, 5, 2]]), batches[1]["input_ids"])
        assert torch.equal(torch.tensor([[5, 2, 3]]), batches[1]["labels"])

    def test_build_swag_dataloader_skips_examples_longer_than_sequence(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4 5 6 7")])),
        )
        batches = list(
            ft_swag.build_swag_dataloader(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert [] == batches

    def test_build_swag_dataloader_keeps_examples_equal_to_sequence_length_with_eos(self, monkeypatch):
        import torch
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=5, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4 5 6")])),
        )
        batches = list(
            ft_swag.build_swag_dataloader(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 1 == len(batches)
        assert torch.equal(torch.tensor([[1, 4, 5, 6, 2]]), batches[0]["input_ids"])
        assert torch.equal(torch.tensor([[4, 5, 6, 2, 3]]), batches[0]["labels"])

    def test_parse_args_defaults_to_fresh_fine_tuning(self):
        import ft_swag

        args = ft_swag.parse_args([])

        assert not args.resume

    def test_parse_args_enables_resume(self):
        import ft_swag

        args = ft_swag.parse_args(["--resume"])

        assert args.resume

    def test_main_passes_resume_flag_to_fine_tune_swag(self, monkeypatch):
        import ft_swag

        fine_tune_swag = Spy(return_value=Path("/tmp/sml-swag.pt"))
        monkeypatch.setattr(ft_swag, "fine_tune_swag", fine_tune_swag)

        return_code = ft_swag.main(["--resume"])

        assert ft_swag.SUCCESS_RETURN_CODE == return_code
        assert fine_tune_swag.call_args.kwargs['resume_from_checkpoint']

    def test_fine_tune_swag_accepts_config_objects_and_resume_flag(self):
        import ft_swag

        parameters = inspect.signature(ft_swag.fine_tune_swag).parameters

        assert ['fine_tune_config', 'resume_from_checkpoint'] == list(parameters)

    def test_fine_tune_swag_forces_yarn_rope_scaling_for_lora_model(self, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig
        from sml import SMLConfig

        class StopAfterModelConstruction(Exception):
            pass

        class FakeTokenizer:
            def get_piece_size(self):
                return 32

        captured_rope_scaling_factor = None

        def capture_model_config(config):
            nonlocal captured_rope_scaling_factor
            captured_rope_scaling_factor = config.rope_scaling_factor
            raise StopAfterModelConstruction

        base_model_config = SMLConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=32,
            original_max_position_embeddings=16,
            rope_scaling_factor=1.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )

        monkeypatch.setattr(ft_swag, "load_tokenizer", Spy(return_value=FakeTokenizer()))
        monkeypatch.setattr(ft_swag, "resolve_device", Spy(return_value="cpu"))
        monkeypatch.setattr(
            ft_swag,
            "load_pretrained_model_config",
            Spy(return_value=base_model_config),
        )
        monkeypatch.setattr(ft_swag, "SMLLanguageModel", capture_model_config)

        with pytest.raises(StopAfterModelConstruction):
            ft_swag.fine_tune_swag(
                SwagFineTuneConfig(
                    pretrained_checkpoint_path=Path(__file__),
                    tokenizer_model_path=Path(__file__),
                )
            )

        assert SMLConfig().rope_scaling_factor == captured_rope_scaling_factor
