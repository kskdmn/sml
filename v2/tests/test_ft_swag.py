import inspect
import sys
from pathlib import Path

from helpers import Spy
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def require_mlx_runtime():
    try:
        import mlx.core as mx
        import mlx.nn  # noqa: F401

        mx.eval(mx.array([0]))
        return mx
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"mlx is not available: {exc}")


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

    def test_build_swag_batches_masks_context_and_scores_gold_ending_and_eos(self, monkeypatch):
        mx = require_mlx_runtime()
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
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=TextTokenizer(),
                epoch=0,
            )
        )

        assert [[1, 10, 11, 20, 21, 2]] == batches[0]["input_ids"].tolist()
        assert [[3, 3, 20, 21, 2, 3]] == batches[0]["labels"].tolist()
        assert mx.int32 == batches[0]["input_ids"].dtype
        assert mx.int32 == batches[0]["labels"].dtype

    def test_build_swag_batches_pads_short_examples_without_packing(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=2)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4"), ("", "5")])),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 1 == len(batches)
        assert [[1, 4, 2], [1, 5, 2]] == batches[0]["input_ids"].tolist()
        assert [[4, 2, 3], [5, 2, 3]] == batches[0]["labels"].tolist()

    def test_build_swag_batches_skips_examples_longer_than_sequence(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4 5 6 7")])),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert [] == batches

    def test_build_swag_batches_keeps_examples_equal_to_sequence_length_with_eos(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=5, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_parts",
            Spy(return_value=iter([("", "4 5 6")])),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 1 == len(batches)
        assert [[1, 4, 5, 6, 2]] == batches[0]["input_ids"].tolist()
        assert [[4, 5, 6, 2, 3]] == batches[0]["labels"].tolist()

    def test_parse_args_defaults_to_fresh_fine_tuning(self):
        import ft_swag

        args = ft_swag.parse_args([])

        assert not args.resume

    def test_parse_args_enables_resume(self):
        import ft_swag

        args = ft_swag.parse_args(["--resume"])

        assert args.resume

    def test_resume_help_documents_stochastic_continuity(self):
        import ft_swag

        parser = ft_swag.build_parser()

        assert "Stochastic continuity is not guaranteed." in parser.format_help()

    def test_main_passes_resume_flag_to_fine_tune_swag(self, monkeypatch):
        import ft_swag

        fine_tune_swag = Spy(return_value=Path("/tmp/sml-swag"))
        monkeypatch.setattr(ft_swag, "fine_tune_swag", fine_tune_swag)

        return_code = ft_swag.main(["--resume"])

        assert ft_swag.SUCCESS_RETURN_CODE == return_code
        assert fine_tune_swag.call_args.kwargs['resume_from_checkpoint']

    def test_fine_tune_swag_accepts_config_objects_and_resume_flag(self):
        import ft_swag

        parameters = inspect.signature(ft_swag.fine_tune_swag).parameters

        assert ['fine_tune_config', 'resume_from_checkpoint'] == list(parameters)

    def test_fine_tune_swag_forces_yarn_rope_scaling_for_lora_model(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        import sml
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
        monkeypatch.setattr(
            ft_swag,
            "load_pretrained_model_config",
            Spy(return_value=base_model_config),
        )
        monkeypatch.setattr(sml, "SMLLanguageModel", capture_model_config)

        with pytest.raises(StopAfterModelConstruction):
            ft_swag.fine_tune_swag(
                SwagFineTuneConfig(
                    pretrained_checkpoint_path=Path(__file__),
                    tokenizer_model_path=Path(__file__),
                )
            )

        assert SMLConfig().rope_scaling_factor == captured_rope_scaling_factor

    def test_lora_checkpoint_round_trip_writes_mlx_directory_files(self, tmp_path):
        mx = require_mlx_runtime()
        import json
        import mlx.optimizers as optim
        import ft_swag
        from ft_swag import SwagFineTuneConfig
        from lora import apply_lora
        from sml import SMLConfig, SMLLanguageModel
        from train_sml import TrainingDataState

        model_config = SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=16,
            rope_scaling_factor=1.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            gradient_checkpointing=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
        fine_tune_config = SwagFineTuneConfig(
            pretrained_checkpoint_path=tmp_path / "pretrained",
            output_dir=tmp_path,
            checkpoint_name="sml-swag",
            lora=ft_swag.LoRAConfig(
                rank=2,
                alpha=4.0,
                dropout=0.0,
                target_modules=("q_proj",),
            ),
        )
        checkpoint_path = tmp_path / fine_tune_config.checkpoint_name
        source = SMLLanguageModel(model_config)
        target = SMLLanguageModel(model_config)
        apply_lora(source, fine_tune_config.lora)
        apply_lora(target, fine_tune_config.lora)
        source.layers[0].self_attn.q_proj.lora_A = mx.full(
            source.layers[0].self_attn.q_proj.lora_A.shape,
            0.25,
        )
        source.layers[0].self_attn.q_proj.lora_B = mx.full(
            source.layers[0].self_attn.q_proj.lora_B.shape,
            0.5,
        )
        optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(source.trainable_parameters())
        target_optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
        target_optimizer.init(target.trainable_parameters())

        ft_swag.save_lora_checkpoint(
            checkpoint_path,
            source,
            optimizer,
            model_config,
            fine_tune_config,
            step=3,
            data_state=TrainingDataState(epoch=1, line_number=2, token_buffer=[7]),
        )

        assert (checkpoint_path / ft_swag.MODEL_WEIGHTS_NAME).exists()
        assert (checkpoint_path / ft_swag.LORA_STATE_NAME).exists()
        assert (checkpoint_path / ft_swag.OPTIMIZER_STATE_NAME).exists()
        assert (checkpoint_path / ft_swag.METADATA_NAME).exists()
        metadata = json.loads(
            (checkpoint_path / ft_swag.METADATA_NAME).read_text(encoding="utf-8")
        )
        assert 3 == metadata["step"]
        assert "sml-swag" == metadata["training_config"]["checkpoint_name"]
        assert "not_guaranteed" == metadata["stochastic_resume"]
        assert ft_swag.STOCHASTIC_RESUME_NOTE == metadata["resume_note"]

        resume_state = ft_swag.load_lora_checkpoint(
            checkpoint_path,
            target,
            target_optimizer,
        )

        assert 3 == resume_state.step
        assert 1 == resume_state.data_state.epoch
        assert 2 == resume_state.data_state.line_number
        assert [7] == resume_state.data_state.token_buffer
        assert bool(
            mx.array_equal(
                source.layers[0].self_attn.q_proj.lora_A,
                target.layers[0].self_attn.q_proj.lora_A,
            ).item()
        )
        assert bool(
            mx.array_equal(
                source.layers[0].self_attn.q_proj.lora_B,
                target.layers[0].self_attn.q_proj.lora_B,
            ).item()
        )

    def test_load_pretrained_model_config_reads_mlx_metadata(self, tmp_path):
        require_mlx_runtime()
        import ft_swag

        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        (checkpoint_path / ft_swag.METADATA_NAME).write_text(
            (
                '{"model_config": {"vocab_size": 32, "hidden_size": 16, '
                '"num_layers": 1, "num_q_heads": 4, "num_kv_heads": 2, '
                '"intermediate_size": 32, "max_position_embeddings": 16, '
                '"attention_dropout": 0.0, "hidden_dropout": 0.0}}'
            ),
            encoding="utf-8",
        )

        model_config = ft_swag.load_pretrained_model_config(checkpoint_path)

        assert 16 == model_config.original_max_position_embeddings

    def test_fine_tune_swag_validates_pretrained_checkpoint_path(self, tmp_path, monkeypatch):
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        class FakeTokenizer:
            def get_piece_size(self):
                return 32

        missing_checkpoint = tmp_path / "missing"
        monkeypatch.setattr(ft_swag, "load_tokenizer", Spy(return_value=FakeTokenizer()))

        with pytest.raises(FileNotFoundError, match="Pretrained checkpoint does not exist"):
            ft_swag.fine_tune_swag(
                SwagFineTuneConfig(
                    pretrained_checkpoint_path=missing_checkpoint,
                    tokenizer_model_path=Path(__file__),
                    output_dir=tmp_path,
                )
            )
