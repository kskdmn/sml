import inspect
from pathlib import Path

import pytest
from helpers import Spy


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

    def test_join_swag_parts_inserts_space_before_gold_ending(self):
        import ft_swag

        assert "The girl stops clutching her diary." == ft_swag.join_swag_parts(
            "The girl", "stops clutching her diary."
        )

    def test_resolve_swag_label_accepts_string_labels(self):
        import ft_swag

        assert 2 == ft_swag.resolve_swag_label("2")

    def test_swag_parameter_weight_decay_config_extends_base_defaults(self):
        import ft_swag
        import train_sml

        config = ft_swag.SwagParameterWeightDecayConfig()

        assert isinstance(config, train_sml.ParameterWeightDecayConfig)
        assert config.q_proj == pytest.approx(0.1)
        assert config.down_proj == pytest.approx(0.1)
        assert config.lora_a == pytest.approx(0.0)
        assert config.lora_b == pytest.approx(0.0)

    def test_swag_parameter_initializer_range_config_extends_base_defaults(self):
        import ft_swag
        from sml import ParameterInitializerRangeConfig

        config = ft_swag.SwagParameterInitializerRangeConfig()

        assert isinstance(config, ParameterInitializerRangeConfig)
        assert config.q_proj == pytest.approx(0.02)
        assert config.down_proj == pytest.approx(0.02)
        assert config.lora_a == pytest.approx(0.01)
        assert config.lora_b == pytest.approx(0.0)

    def test_swag_fine_tune_config_uses_depth_scaled_initializer_defaults(self):
        import ft_swag
        from sml import SMLConfig

        base_initializer_range = SMLConfig().parameter_initializer_range
        config = ft_swag.SwagFineTuneConfig()

        assert config.parameter_initializer_range.o_proj == pytest.approx(
            base_initializer_range.o_proj
        )
        assert config.parameter_initializer_range.down_proj == pytest.approx(
            base_initializer_range.down_proj
        )
        assert config.parameter_initializer_range.lora_a == pytest.approx(0.01)
        assert config.parameter_initializer_range.lora_b == pytest.approx(0.0)

    def test_swag_parameter_weight_decay_config_classifies_lora_parameters(self):
        require_mlx_runtime()
        import ft_swag
        import train_sml
        from lora import LoRAConfig, apply_lora
        from mlx.utils import tree_flatten
        from sml import SMLLanguageModel
        from test_train import tiny_config

        model = SMLLanguageModel(tiny_config())
        apply_lora(
            model,
            LoRAConfig(
                rank=2,
                alpha=4.0,
                dropout=0.0,
                target_modules=("q_proj", "down_proj"),
            ),
        )

        decays = dict(
            tree_flatten(
                train_sml.build_parameter_weight_decay_tree(
                    model.trainable_parameters(),
                    parameter_weight_decay=ft_swag.SwagParameterWeightDecayConfig(
                        lora_a=0.001,
                        lora_b=0.002,
                    ),
                )
            )
        )

        assert decays["layers.0.self_attn.q_proj.lora_A"] == pytest.approx(0.001)
        assert decays["layers.0.self_attn.q_proj.lora_B"] == pytest.approx(0.002)
        assert decays["layers.0.mlp.down_proj.lora_A"] == pytest.approx(0.001)
        assert decays["layers.0.mlp.down_proj.lora_B"] == pytest.approx(0.002)

    def test_iter_swag_parts_resumes_after_saved_position(self, monkeypatch):
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
        parts = list(
            ft_swag.iter_swag_parts(
                SwagFineTuneConfig(shuffle_examples=False, seed=42),
                epoch=0,
                data_state=data_state,
            )
        )

        assert [("second", "beta"), ("third", "z")] == parts
        assert 2 == data_state.line_number

    def test_iter_swag_parts_resumes_same_shuffled_order(self, monkeypatch):
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
        full_epoch = list(ft_swag.iter_swag_parts(config, epoch=2))

        data_state = TrainingDataState(line_number=2)
        resumed_epoch = list(
            ft_swag.iter_swag_parts(config, epoch=2, data_state=data_state)
        )

        assert full_epoch[3:] == resumed_epoch
        assert 7 == data_state.line_number

    def test_iter_swag_parts_updates_reading_progress_example_index(self, monkeypatch):
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
        parts = list(
            ft_swag.iter_swag_parts(
                SwagFineTuneConfig(shuffle_examples=False, seed=42),
                epoch=0,
                progress=progress,
            )
        )

        assert [("first", "one"), ("second", "beta")] == parts
        assert 1 == progress.line_number
        assert 1 == progress.example_index

    def test_build_swag_batches_scores_all_candidate_endings_and_eos(self, monkeypatch):
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
                    " wrong": [30],
                    " maybe": [31, 32],
                    " nope": [33],
                }[text]

        rows = [
            {
                "startphrase": "ctx",
                "ending0": " end",
                "ending1": " wrong",
                "ending2": " maybe",
                "ending3": " nope",
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

        assert [
            [
                [1, 10, 11, 20, 21, 2],
                [1, 10, 11, 30, 2, 3],
                [1, 10, 11, 31, 32, 2],
                [1, 10, 11, 33, 2, 3],
            ]
        ] == batches[0]["input_ids"].tolist()
        assert [
            [
                [3, 3, 20, 21, 2, 3],
                [3, 3, 30, 2, 3, 3],
                [3, 3, 31, 32, 2, 3],
                [3, 3, 33, 2, 3, 3],
            ]
        ] == batches[0]["labels"].tolist()
        assert [0] == batches[0]["candidate_labels"].tolist()
        assert mx.int32 == batches[0]["input_ids"].dtype
        assert mx.int32 == batches[0]["labels"].dtype
        assert mx.int32 == batches[0]["candidate_labels"].dtype

    def test_build_swag_batches_pads_short_examples_without_packing(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=2)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_examples",
            Spy(
                return_value=iter(
                    [
                        ("", ("4", "5", "6", "7"), 0),
                        ("", ("8", "9", "10", "11"), 2),
                    ]
                )
            ),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 1 == len(batches)
        assert [
            [[1, 4, 2], [1, 5, 2], [1, 6, 2], [1, 7, 2]],
            [[1, 8, 2], [1, 9, 2], [1, 10, 2], [1, 11, 2]],
        ] == batches[0]["input_ids"].tolist()
        assert [
            [[4, 2, 3], [5, 2, 3], [6, 2, 3], [7, 2, 3]],
            [[8, 2, 3], [9, 2, 3], [10, 2, 3], [11, 2, 3]],
        ] == batches[0]["labels"].tolist()
        assert [0, 2] == batches[0]["candidate_labels"].tolist()

    def test_build_swag_batches_skips_examples_longer_than_sequence(self, monkeypatch):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=3, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_examples",
            Spy(return_value=iter([("", ("4", "5", "6", "7 8 9 10"), 0)])),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert [] == batches

    def test_build_swag_batches_keeps_examples_equal_to_sequence_length_with_eos(
        self, monkeypatch
    ):
        require_mlx_runtime()
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        config = SwagFineTuneConfig(sequence_length=5, batch_size=1)

        monkeypatch.setattr(
            ft_swag,
            "iter_swag_examples",
            Spy(return_value=iter([("", ("4 5 6", "7", "8", "9"), 0)])),
        )
        batches = list(
            ft_swag.build_swag_batches(
                fine_tune_config=config,
                tokenizer=FakeTokenizer(),
                epoch=0,
            )
        )

        assert 1 == len(batches)
        assert [
            [[1, 4, 5, 6, 2], [1, 7, 2, 3, 3], [1, 8, 2, 3, 3], [1, 9, 2, 3, 3]]
        ] == batches[0]["input_ids"].tolist()
        assert [
            [[4, 5, 6, 2, 3], [7, 2, 3, 3, 3], [8, 2, 3, 3, 3], [9, 2, 3, 3, 3]]
        ] == batches[0]["labels"].tolist()

    def test_swag_ranking_scores_sum_candidate_continuation_loglikelihoods(self):
        mx = require_mlx_runtime()
        import ft_swag

        class CandidateModel:
            def __call__(self, input_ids):
                next_token_ids = mx.concatenate(
                    [
                        input_ids[:, 1:],
                        mx.zeros((input_ids.shape[0], 1), dtype=mx.int32),
                    ],
                    axis=1,
                )
                logits = mx.full((*input_ids.shape, 64), -10.0)
                logits = mx.put_along_axis(
                    logits,
                    mx.expand_dims(next_token_ids, axis=-1),
                    mx.full((*input_ids.shape, 1), 10.0),
                    axis=-1,
                )
                return type("Output", (), {"logits": logits})()

        input_ids = mx.array([[[1, 20, 21], [1, 30, 31]]], dtype=mx.int32)
        labels = mx.array([[[20, 21, 3], [32, 3, 3]]], dtype=mx.int32)

        scores = ft_swag.score_swag_candidates(
            CandidateModel(),
            input_ids,
            labels,
            pad_token_id=3,
        )
        mx.eval(scores)

        assert scores.shape == (1, 2)
        assert float(scores[0, 0].item()) > float(scores[0, 1].item())

    def test_parse_args_defaults_to_fresh_fine_tuning(self):
        import ft_swag

        args = ft_swag.parse_args([])

        assert not args.resume
        assert ft_swag.DEFAULT_MODEL_PATH == args.model
        assert ft_swag.DEFAULT_TOKENIZER_MODEL_PATH == args.tokenizer_model

    def test_parse_args_enables_resume(self):
        import ft_swag

        args = ft_swag.parse_args(["--resume"])

        assert args.resume

    def test_parse_args_accepts_model_path(self):
        import ft_swag

        args = ft_swag.parse_args(["--model", "/tmp/custom-sml"])

        assert Path("/tmp/custom-sml") == args.model

    def test_parse_args_accepts_tokenizer_model_path(self):
        import ft_swag

        args = ft_swag.parse_args(["--tokenizer-model", "/tmp/custom-tokenizer.model"])

        assert Path("/tmp/custom-tokenizer.model") == args.tokenizer_model

    def test_resume_help_documents_stochastic_continuity(self):
        import ft_swag

        parser = ft_swag.build_parser()

        assert "Stochastic continuity" in parser.format_help()
        assert "guaranteed." in parser.format_help()

    def test_main_passes_resume_flag_to_fine_tune_swag(self, monkeypatch):
        import ft_swag

        fine_tune_swag = Spy(return_value=Path("/tmp/sml-swag"))
        monkeypatch.setattr(ft_swag, "fine_tune_swag", fine_tune_swag)

        return_code = ft_swag.main(["--resume"])

        assert ft_swag.SUCCESS_RETURN_CODE == return_code
        assert fine_tune_swag.call_args.kwargs["resume_from_checkpoint"]

    def test_main_passes_model_path_to_fine_tune_config(self, monkeypatch):
        import ft_swag

        fine_tune_swag = Spy(return_value=Path("/tmp/sml-swag"))
        monkeypatch.setattr(ft_swag, "fine_tune_swag", fine_tune_swag)

        return_code = ft_swag.main(["--model", "/tmp/custom-sml"])

        assert ft_swag.SUCCESS_RETURN_CODE == return_code
        fine_tune_config = fine_tune_swag.call_args.kwargs["fine_tune_config"]
        assert Path("/tmp/custom-sml") == fine_tune_config.pretrained_checkpoint_path

    def test_main_passes_tokenizer_model_path_to_fine_tune_config(self, monkeypatch):
        import ft_swag

        fine_tune_swag = Spy(return_value=Path("/tmp/sml-swag"))
        monkeypatch.setattr(ft_swag, "fine_tune_swag", fine_tune_swag)

        return_code = ft_swag.main(["--tokenizer-model", "/tmp/custom-tokenizer.model"])

        assert ft_swag.SUCCESS_RETURN_CODE == return_code
        fine_tune_config = fine_tune_swag.call_args.kwargs["fine_tune_config"]
        assert (
            Path("/tmp/custom-tokenizer.model") == fine_tune_config.tokenizer_model_path
        )

    def test_fine_tune_swag_accepts_config_objects_and_resume_flag(self):
        import ft_swag

        parameters = inspect.signature(ft_swag.fine_tune_swag).parameters

        assert ["fine_tune_config", "resume_from_checkpoint"] == list(parameters)

    def test_fine_tune_swag_forces_yarn_rope_scaling_for_lora_model(self, monkeypatch):
        require_mlx_runtime()
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
            hidden_dropout=0.0,
        )

        monkeypatch.setattr(
            ft_swag, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )
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

    def test_lora_checkpoint_round_trip_writes_mlx_directory_files(self, tmp_path):
        mx = require_mlx_runtime()
        import json

        import ft_swag
        import mlx.optimizers as optim
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
            hidden_dropout=0.0,
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
                '"intermediate_size": 32, "original_max_position_embeddings": 16, '
                '"hidden_dropout": 0.0}}'
            ),
            encoding="utf-8",
        )

        model_config = ft_swag.load_pretrained_model_config(checkpoint_path)

        assert 16 == model_config.original_max_position_embeddings

    @pytest.mark.parametrize("resume_from_checkpoint", [False, True])
    def test_fine_tune_swag_validates_pretrained_checkpoint_path(
        self, tmp_path, monkeypatch, resume_from_checkpoint
    ):
        import ft_swag
        from ft_swag import SwagFineTuneConfig

        class FakeTokenizer:
            def get_piece_size(self):
                return 32

        missing_checkpoint = tmp_path / "missing"
        monkeypatch.setattr(
            ft_swag, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )

        with pytest.raises(
            FileNotFoundError, match="Pretrained checkpoint does not exist"
        ):
            ft_swag.fine_tune_swag(
                SwagFineTuneConfig(
                    pretrained_checkpoint_path=missing_checkpoint,
                    tokenizer_model_path=Path(__file__),
                    output_dir=tmp_path,
                ),
                resume_from_checkpoint=resume_from_checkpoint,
            )
