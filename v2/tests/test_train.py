import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from helpers import Spy
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def require_mlx():
    try:
        import mlx.core as mx

        mx.eval(mx.array([0]))
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - depends on host
        pytest.skip(f"mlx is not available: {exc}")
    return mx


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]

    def get_piece_size(self):
        return 16


def write_token_shard(path: Path, blocks: list[list[int]]) -> None:
    import numpy as np

    tokens = np.asarray(blocks, dtype=np.uint16)
    np.savez_compressed(path, tokens=tokens)


def write_pretraining_fixture(
    root: Path,
    *,
    sequence_length: int,
    vocab_size: int,
    shards: list[list[list[int]]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard_entries = []
    for index, blocks in enumerate(shards):
        relative = f"train-{index:06d}.npz"
        write_token_shard(root / relative, blocks)
        shard_entries.append({"path": relative, "blocks": len(blocks)})
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "array_name": "tokens",
        "dtype": "uint16",
        "sequence_length": sequence_length,
        "tokens_per_block": sequence_length + 1,
        "tokenizer_vocab_size": vocab_size,
        "shards": shard_entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@dataclass(frozen=True, slots=True)
class PureModelConfig:
    vocab_size: int = 16
    hidden_size: int = 8
    num_layers: int = 1
    num_q_heads: int = 2
    num_kv_heads: int = 1
    intermediate_size: int = 16
    original_max_position_embeddings: int = 16
    rope_scaling_factor: float = 2.0
    hidden_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3


def tiny_config():
    try:
        from sml import SMLConfig

        return SMLConfig(
            vocab_size=16,
            hidden_size=8,
            num_layers=1,
            num_q_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
            original_max_position_embeddings=16,
            rope_scaling_factor=2.0,
            hidden_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
        )
    except RuntimeError as exc:  # pragma: no cover - depends on host
        pytest.skip(f"mlx is not available: {exc}")


class TestTrainData:
    def test_parse_args_defaults_to_fresh_training(self):
        import train_sml

        args = train_sml.parse_args([])

        assert not args.resume
        assert train_sml.DEFAULT_MODEL_PATH == args.model
        assert train_sml.DEFAULT_TOKENIZER_MODEL_PATH == args.tokenizer_model

    def test_parse_args_enables_resume(self):
        import train_sml

        args = train_sml.parse_args(["--resume"])

        assert args.resume

    def test_parse_args_accepts_model_path(self):
        import train_sml

        args = train_sml.parse_args(["--model", "/tmp/custom-sml"])

        assert Path("/tmp/custom-sml") == args.model

    def test_parse_args_accepts_tokenizer_model_path(self):
        import train_sml

        args = train_sml.parse_args(
            ["--tokenizer-model", "/tmp/custom-tokenizer.model"]
        )

        assert Path("/tmp/custom-tokenizer.model") == args.tokenizer_model

    def test_resume_help_documents_stochastic_continuity(self):
        import train_sml

        parser = train_sml.build_parser()

        assert "Stochastic continuity" in parser.format_help()
        assert "guaranteed." in parser.format_help()

    def test_main_passes_resume_flag_to_training_config(self, monkeypatch):
        import train_sml

        train_model = Spy(return_value=Path("/tmp/sml"))
        monkeypatch.setattr(train_sml, "train_model", train_model)

        return_code = train_sml.main(["--resume"])

        assert train_sml.SUCCESS_RETURN_CODE == return_code
        assert train_model.call_args.kwargs["resume_from_checkpoint"]

    def test_main_passes_model_path_to_training_config(self, monkeypatch):
        import train_sml

        train_model = Spy(return_value=Path("/tmp/custom-sml"))
        monkeypatch.setattr(train_sml, "train_model", train_model)

        return_code = train_sml.main(["--model", "/tmp/custom-sml"])

        assert train_sml.SUCCESS_RETURN_CODE == return_code
        training_config = train_model.call_args.kwargs["training_config"]
        assert Path("/tmp/custom-sml") == training_config.model_path

    def test_main_passes_tokenizer_model_path_to_training_config(self, monkeypatch):
        import train_sml

        train_model = Spy(return_value=Path("/tmp/sml"))
        monkeypatch.setattr(train_sml, "train_model", train_model)

        return_code = train_sml.main(
            ["--tokenizer-model", "/tmp/custom-tokenizer.model"]
        )

        assert train_sml.SUCCESS_RETURN_CODE == return_code
        training_config = train_model.call_args.kwargs["training_config"]
        assert (
            Path("/tmp/custom-tokenizer.model") == training_config.tokenizer_model_path
        )

    def test_train_model_accepts_config_objects_and_resume_flag(self):
        import train_sml

        parameters = inspect.signature(train_sml.train_model).parameters

        assert ["training_config", "model_config", "resume_from_checkpoint"] == list(
            parameters
        )

    def test_training_config_defaults_to_mlx_checkpoint_name(self):
        import train_sml
        from train_sml import TrainingConfig

        training_config = TrainingConfig()

        assert "sml" == training_config.checkpoint_name
        assert training_config.model_path is None
        assert (
            train_sml.DEFAULT_TOKENIZER_MODEL_PATH
            == training_config.tokenizer_model_path
        )
        assert not hasattr(training_config, "resume_from_checkpoint")

    def test_training_config_defaults_to_bfloat16(self):
        from train_sml import TrainingConfig

        training_config = TrainingConfig()

        assert "bfloat16" == training_config.autocast_dtype

    def test_parameter_weight_decay_config_has_recommended_base_defaults(self):
        import train_sml

        config = train_sml.ParameterWeightDecayConfig()

        assert config.embed_tokens == pytest.approx(0.0)
        assert config.lm_head == pytest.approx(0.0)
        assert config.rms_norm == pytest.approx(0.0)
        assert config.q_proj == pytest.approx(0.1)
        assert config.k_proj == pytest.approx(0.1)
        assert config.v_proj == pytest.approx(0.1)
        assert config.o_proj == pytest.approx(0.1)
        assert config.gate_proj == pytest.approx(0.1)
        assert config.up_proj == pytest.approx(0.1)
        assert config.down_proj == pytest.approx(0.1)
        assert config.other == pytest.approx(0.1)
        assert not hasattr(config, "lora_a")
        assert not hasattr(config, "lora_b")

    def test_parameter_weight_decay_config_rejects_unset_values(self):
        import train_sml

        with pytest.raises(ValueError, match="rms_norm"):
            train_sml.ParameterWeightDecayConfig(rms_norm=None)

    def test_parameter_weight_decay_config_validates_weight_decay_with_member_method(
        self,
    ):
        import train_sml

        config = train_sml.ParameterWeightDecayConfig()

        config.validate_weight_decay(0.0, "rms_norm")
        with pytest.raises(ValueError, match="rms_norm"):
            config.validate_weight_decay(float("inf"), "rms_norm")

    def test_parameter_weight_decay_config_classifies_base_model_parameters(self):
        require_mlx()
        from mlx.utils import tree_flatten

        import train_sml
        from sml import SMLLanguageModel

        model = SMLLanguageModel(tiny_config())
        config = train_sml.ParameterWeightDecayConfig(
            embed_tokens=0.01,
            lm_head=0.02,
            rms_norm=0.03,
            q_proj=0.04,
            k_proj=0.05,
            v_proj=0.06,
            o_proj=0.07,
            gate_proj=0.08,
            up_proj=0.09,
            down_proj=0.10,
        )

        decays = dict(
            tree_flatten(
                train_sml.build_parameter_weight_decay_tree(
                    model.trainable_parameters(),
                    parameter_weight_decay=config,
                )
            )
        )

        assert decays["embed_tokens.weight"] == pytest.approx(0.01)
        assert decays["lm_head.weight"] == pytest.approx(0.02)
        assert decays["layers.0.input_norm.weight"] == pytest.approx(0.03)
        assert decays["layers.0.post_attn_norm.weight"] == pytest.approx(0.03)
        assert decays["norm.weight"] == pytest.approx(0.03)
        assert decays["layers.0.self_attn.q_proj.weight"] == pytest.approx(0.04)
        assert decays["layers.0.self_attn.k_proj.weight"] == pytest.approx(0.05)
        assert decays["layers.0.self_attn.v_proj.weight"] == pytest.approx(0.06)
        assert decays["layers.0.self_attn.o_proj.weight"] == pytest.approx(0.07)
        assert decays["layers.0.mlp.gate_proj.weight"] == pytest.approx(0.08)
        assert decays["layers.0.mlp.up_proj.weight"] == pytest.approx(0.09)
        assert decays["layers.0.mlp.down_proj.weight"] == pytest.approx(0.10)

    def test_decoupled_weight_decay_scales_trainable_parameters_by_decay_tree(self):
        mx = require_mlx()
        from mlx.utils import tree_flatten, tree_map

        import train_sml
        from sml import SMLLanguageModel

        model = SMLLanguageModel(tiny_config())
        model.update(tree_map(lambda value: mx.ones_like(value), model.parameters()))
        trainable = model.trainable_parameters()
        decay_tree = tree_map(lambda _: 0.0, trainable)
        decay_tree["layers"][0]["self_attn"]["q_proj"]["weight"] = 0.5

        train_sml.apply_decoupled_weight_decay(
            model,
            weight_decay_tree=decay_tree,
            learning_rate=mx.array(0.1),
        )
        parameters = dict(tree_flatten(model.parameters()))
        mx.eval(parameters["layers.0.self_attn.q_proj.weight"])
        mx.eval(parameters["layers.0.self_attn.k_proj.weight"])

        assert float(parameters["layers.0.self_attn.q_proj.weight"][0, 0].item()) == (
            pytest.approx(0.95)
        )
        assert float(parameters["layers.0.self_attn.k_proj.weight"][0, 0].item()) == (
            pytest.approx(1.0)
        )

    def test_training_config_derives_warmup_steps_from_lr_total_steps(self):
        from train_sml import TrainingConfig

        assert 500 == TrainingConfig(lr_total_steps=50_000).warmup_steps
        assert 100 == TrainingConfig(lr_total_steps=None).warmup_steps
        assert 7 == TrainingConfig(warmup_steps=7).warmup_steps

    def test_resolve_compute_dtype_maps_supported_names(self):
        import mlx.core as mx
        import train_sml

        assert train_sml.resolve_compute_dtype("none") is None
        assert mx.bfloat16 == train_sml.resolve_compute_dtype("bfloat16")
        assert mx.float16 == train_sml.resolve_compute_dtype("float16")

    def test_resolve_compute_dtype_rejects_unknown_name(self):
        import train_sml

        with pytest.raises(ValueError, match="Unsupported compute dtype"):
            train_sml.resolve_compute_dtype("float64")

    def test_apply_model_dtype_casts_parameters(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        import train_sml
        from sml import create_model

        model = create_model()
        train_sml.apply_model_dtype(model, "bfloat16")

        for _, parameter in tree_flatten(model.parameters()):
            assert mx.bfloat16 == parameter.dtype

    def test_apply_model_dtype_keeps_float32_when_disabled(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        import train_sml
        from sml import create_model

        model = create_model()
        train_sml.apply_model_dtype(model, "none")

        for _, parameter in tree_flatten(model.parameters()):
            assert mx.float32 == parameter.dtype

    def test_get_special_token_id_uses_fallback_for_disabled_or_missing_token(self):
        import utils

        class Tokenizer:
            def bos_id(self):
                return -1

        assert 1 == utils.get_special_token_id(Tokenizer(), "bos_id", 1)
        assert 2 == utils.get_special_token_id(object(), "eos_id", 2)

    def test_get_special_token_id_uses_tokenizer_value_when_enabled(self):
        import utils

        assert 1 == utils.get_special_token_id(FakeTokenizer(), "bos_id", 9)

    def test_load_tokenizer_rejects_missing_model(self, tmp_path):
        import train_sml

        with pytest.raises(FileNotFoundError, match="Tokenizer model does not exist"):
            train_sml.load_tokenizer(tmp_path / "missing.model")

    def test_parse_checkpoint_data_state_restores_fields(self):
        import train_sml

        data_state = train_sml.parse_checkpoint_data_state(
            {
                "epoch": 2,
                "input_file_index": 1,
                "line_number": 42,
                "token_buffer": ["4", 5, 6],
            }
        )

        assert data_state == train_sml.TrainingDataState(
            epoch=2,
            input_file_index=1,
            line_number=42,
            token_buffer=[4, 5, 6],
        )

    def test_parse_checkpoint_data_state_rejects_invalid_shapes(self):
        import train_sml

        with pytest.raises(ValueError, match="Checkpoint data_state must be"):
            train_sml.parse_checkpoint_data_state([])
        with pytest.raises(ValueError, match="token_buffer must be a list"):
            train_sml.parse_checkpoint_data_state({"token_buffer": "4"})
        with pytest.raises(ValueError, match="line_number must be an integer"):
            train_sml.parse_checkpoint_data_state({"line_number": "4"})

    def test_reset_training_data_state_starts_new_epoch(self):
        import train_sml

        data_state = train_sml.TrainingDataState(
            epoch=1,
            input_file_index=2,
            line_number=99,
            token_buffer=[4, 5],
        )

        train_sml.reset_training_data_state(data_state, epoch=3)

        assert data_state == train_sml.TrainingDataState(epoch=3)

    def test_count_resume_batches_uses_completed_optimizer_steps(self):
        import train_sml
        from train_sml import TrainingConfig

        training_config = TrainingConfig(gradient_accumulation_steps=8)

        assert 56 == train_sml.count_resume_batches(
            global_step=7,
            training_config=training_config,
        )

    def test_iter_unseen_batches_skips_consumed_batches_across_dataloaders(self):
        import train_sml

        progress = train_sml.ResumeProgress(batches_to_skip=3)

        first_epoch = list(train_sml.iter_unseen_batches(["a", "b"], progress))
        second_epoch = list(train_sml.iter_unseen_batches(["c", "d", "e"], progress))

        assert [] == first_epoch
        assert ["d", "e"] == second_epoch
        assert 0 == progress.batches_to_skip

    def test_step_limit_is_never_reached_when_max_steps_is_none(self):
        import train_sml

        assert not train_sml.is_step_limit_reached(global_step=10000, max_steps=None)

    def test_resolve_lr_total_steps_prefers_lr_total_steps(self):
        import train_sml
        from train_sml import TrainingConfig

        training_config = TrainingConfig(lr_total_steps=5_000, max_steps=1_000)

        assert 5000 == train_sml.resolve_lr_total_steps(training_config)

    def test_resolve_lr_total_steps_falls_back_to_max_steps(self):
        import train_sml
        from train_sml import TrainingConfig

        training_config = TrainingConfig(lr_total_steps=None, max_steps=1_000)

        assert 1000 == train_sml.resolve_lr_total_steps(training_config)

    def test_resolve_lr_total_steps_is_none_without_lr_or_max_steps(self):
        import train_sml
        from train_sml import TrainingConfig

        training_config = TrainingConfig(lr_total_steps=None, max_steps=None)

        assert train_sml.resolve_lr_total_steps(training_config) is None

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

        assert (
            "time=2026-06-05 12:34:56 epoch=2 step=3 "
            "lr=3.000e-04 loss=1.2346 grad_norm=5.859 (before clipping)"
        ) == log_line

    def test_format_training_log_omits_example_index(self):
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

        assert (
            "time=2026-06-30 07:50:00 epoch=1 step=10 input=swag-train "
            "line=42 lr=3.000e-04 loss=9.8457 grad_norm=6.069 (before clipping)"
        ) == log_line

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

        assert (
            "time=2026-06-30 07:50:00 epoch=1 step=10 input=pile-0000.jsonl.zst "
            "line=42 lr=3.000e-04 loss=9.8457 grad_norm=6.069 (before clipping)"
        ) == log_line

    def test_model_config_for_training_disables_rope_scaling_without_mutating_input(
        self,
    ):
        import train_sml

        config = PureModelConfig(rope_scaling_factor=2.0)

        training_config = train_sml.model_config_for_training(config)

        assert 1.0 == training_config.rope_scaling_factor
        assert 2.0 == config.rope_scaling_factor

    def test_lr_lambda_warms_up_then_cosine_decays_to_floor(self):
        import utils

        assert 0.5 == pytest.approx(
            utils.lr_lambda(
                step=0,
                total_steps=10,
                warmup_steps=2,
                min_lr_ratio=0.1,
            )
        )
        assert 1.0 == pytest.approx(
            utils.lr_lambda(
                step=2,
                total_steps=10,
                warmup_steps=2,
                min_lr_ratio=0.1,
            )
        )
        assert 0.1 == pytest.approx(
            utils.lr_lambda(
                step=10,
                total_steps=10,
                warmup_steps=2,
                min_lr_ratio=0.1,
            )
        )

    def test_lr_lambda_keeps_constant_schedule_without_total_steps(self):
        import utils

        assert 1.0 == utils.lr_lambda(
            step=10,
            total_steps=None,
            warmup_steps=2,
            min_lr_ratio=0.1,
        )


class TestPreparedPretrainingBlocks:
    def test_iter_prepared_token_blocks_splits_input_ids_and_labels(self, tmp_path):
        import train_sml

        root = write_pretraining_fixture(
            tmp_path,
            sequence_length=4,
            vocab_size=128,
            shards=[
                [
                    [1, 10, 11, 12, 2],
                    [2, 1, 20, 21, 2],
                ]
            ],
        )
        data_state = train_sml.PretrainingDataState()
        progress = train_sml.ReadingProgress()

        blocks = list(
            train_sml.iter_prepared_token_blocks(
                shard_paths=(root / "train-000000.npz",),
                sequence_length=4,
                data_state=data_state,
                progress=progress,
            )
        )

        assert [
            {"input_ids": [1, 10, 11, 12], "labels": [10, 11, 12, 2]},
            {"input_ids": [2, 1, 20, 21], "labels": [1, 20, 21, 2]},
        ] == blocks
        assert data_state.shard_index == 0
        assert data_state.block_index == 2
        assert progress.input_file == "train-000000.npz"
        assert progress.line_number == 1

    def test_iter_prepared_token_blocks_resumes_from_shard_and_block(self, tmp_path):
        import train_sml

        root = write_pretraining_fixture(
            tmp_path,
            sequence_length=4,
            vocab_size=128,
            shards=[
                [[1, 10, 11, 12, 2], [2, 1, 20, 21, 2]],
                [[3, 30, 31, 32, 2]],
            ],
        )
        data_state = train_sml.PretrainingDataState(shard_index=1, block_index=0)

        blocks = list(
            train_sml.iter_prepared_token_blocks(
                shard_paths=(
                    root / "train-000000.npz",
                    root / "train-000001.npz",
                ),
                sequence_length=4,
                data_state=data_state,
            )
        )

        assert [{"input_ids": [3, 30, 31, 32], "labels": [30, 31, 32, 2]}] == blocks

    def test_iter_prepared_token_blocks_rejects_wrong_block_width(self, tmp_path):
        import train_sml
        import numpy as np

        path = tmp_path / "train-000000.npz"
        np.savez_compressed(
            path,
            tokens=np.asarray([[1, 2, 3]], dtype=np.uint16),
        )

        with pytest.raises(ValueError, match="tokens_per_block"):
            list(
                train_sml.iter_prepared_token_blocks(
                    shard_paths=(path,),
                    sequence_length=4,
                )
            )

    def test_parse_checkpoint_pretraining_data_state_restores_fields(self):
        import train_sml

        data_state = train_sml.parse_checkpoint_pretraining_data_state(
            {"epoch": 2, "shard_index": 1, "block_index": 7}
        )

        assert data_state == train_sml.PretrainingDataState(
            epoch=2,
            shard_index=1,
            block_index=7,
        )

    def test_reset_pretraining_data_state_starts_new_epoch(self):
        import train_sml

        data_state = train_sml.PretrainingDataState(
            epoch=1,
            shard_index=2,
            block_index=9,
        )
        train_sml.reset_pretraining_data_state(data_state, epoch=3)

        assert data_state == train_sml.PretrainingDataState(epoch=3)


class TestCanonicalMlxTraining:
    def test_mlx_batch_iterator_emits_mx_arrays_with_next_token_labels(self):
        mx = require_mlx()
        import train_sml

        examples = (
            {"input_ids": [1, 4, 5], "labels": [4, 5, 6]},
            {"input_ids": [6, 7, 8], "labels": [7, 8, 2]},
        )

        batches = list(train_sml.iter_mlx_batches(examples, batch_size=2))

        assert len(batches) == 1
        assert isinstance(batches[0]["input_ids"], mx.array)
        assert isinstance(batches[0]["labels"], mx.array)
        assert batches[0]["input_ids"].shape == (2, 3)
        assert batches[0]["labels"].shape == (2, 3)
        assert batches[0]["input_ids"].tolist() == [[1, 4, 5], [6, 7, 8]]
        assert batches[0]["labels"].tolist() == [[4, 5, 6], [7, 8, 2]]

    def test_mlx_lr_schedule_matches_training_helper(self):
        mx = require_mlx()
        import utils

        schedule = utils.build_lr_schedule(
            learning_rate=0.01,
            total_steps=10,
            warmup_steps=2,
            min_lr_ratio=0.1,
        )

        for step in (0, 1, 5, 10):
            expected = 0.01 * utils.lr_lambda(
                step=step,
                total_steps=10,
                warmup_steps=2,
                min_lr_ratio=0.1,
            )
            assert float(schedule(mx.array(step)).item()) == pytest.approx(expected)

    def test_tree_add_and_tree_scale_apply_to_nested_arrays(self):
        mx = require_mlx()
        import train_sml

        left = {"a": mx.array([1.0, 2.0]), "b": {"c": mx.array([3.0])}}
        right = {"a": mx.array([4.0, 5.0]), "b": {"c": mx.array([6.0])}}

        total = train_sml.tree_add(left, right)
        scaled = train_sml.tree_scale(total, 0.5)
        mx.eval(scaled)

        assert scaled["a"].tolist() == [2.5, 3.5]
        assert scaled["b"]["c"].tolist() == [4.5]

    def test_clip_gradients_by_global_norm_scales_large_grads(self):
        mx = require_mlx()
        import train_sml

        grads = {"w": mx.array([3.0, 4.0])}

        clipped, grad_norm = train_sml.clip_gradients_by_global_norm(
            grads,
            max_norm=1.0,
        )
        clipped_norm = mx.sqrt(mx.sum(clipped["w"] * clipped["w"]))
        mx.eval(clipped_norm, grad_norm)

        assert float(train_sml.global_grad_norm(grads).item()) == pytest.approx(5.0)
        assert float(grad_norm.item()) == pytest.approx(5.0)
        assert float(clipped_norm.item()) == pytest.approx(1.0)

    def test_consume_full_accumulation_window_does_not_rescale(self):
        mx = require_mlx()
        import train_sml

        window = train_sml.GradientAccumulationWindow()
        for _ in range(4):
            train_sml.accumulate_gradients(
                window,
                {"w": mx.array([1.0])},
                2.0,
                gradient_accumulation_steps=4,
            )

        grads, avg_loss, micro_batches = train_sml.consume_accumulated_grads(window, 4)
        mx.eval(grads["w"])

        assert micro_batches == 4
        assert avg_loss == pytest.approx(2.0)
        assert float(grads["w"].item()) == pytest.approx(1.0)
        assert window.accumulated_grads is None

    def test_consume_partial_accumulation_window_rescales_gradients(self):
        mx = require_mlx()
        import train_sml

        window = train_sml.GradientAccumulationWindow()
        for _ in range(3):
            train_sml.accumulate_gradients(
                window,
                {"w": mx.array([2.0])},
                4.0,
                gradient_accumulation_steps=8,
            )

        grads, avg_loss, micro_batches = train_sml.consume_accumulated_grads(window, 8)
        mx.eval(grads["w"])

        assert micro_batches == 3
        assert avg_loss == pytest.approx(4.0)
        assert float(grads["w"].item()) == pytest.approx(2.0)

    def test_is_accumulation_window_ready_only_after_full_window(self):
        import train_sml

        window = train_sml.GradientAccumulationWindow()
        assert not train_sml.is_accumulation_window_ready(window, 4)

        window.micro_step = 3
        assert not train_sml.is_accumulation_window_ready(window, 4)

        window.micro_step = 4
        assert train_sml.is_accumulation_window_ready(window, 4)

    def test_resolve_mlx_checkpoint_path_uses_configured_name(self, tmp_path):
        import train_sml
        from train_sml import TrainingConfig

        checkpoint_path = train_sml.resolve_mlx_checkpoint_path(
            TrainingConfig(output_dir=tmp_path, checkpoint_name="custom")
        )

        assert checkpoint_path == tmp_path / "custom"

    def test_resolve_mlx_checkpoint_path_uses_model_path(self, tmp_path):
        import train_sml
        from train_sml import TrainingConfig

        checkpoint_path = train_sml.resolve_mlx_checkpoint_path(
            TrainingConfig(
                output_dir=tmp_path,
                checkpoint_name="ignored",
                model_path=tmp_path / "model-dir",
            )
        )

        assert checkpoint_path == tmp_path / "model-dir"

    def test_missing_explicit_resume_checkpoint_is_rejected(self, tmp_path):
        import train_sml

        with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
            train_sml.load_training_checkpoint(tmp_path / "missing", object(), object())

    def test_train_model_starts_fresh_without_resume_even_when_checkpoint_exists(
        self,
        monkeypatch,
        tmp_path,
    ):
        require_mlx()
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=16,
            shards=[[[1, 4, 5, 6, 2]]],
        )
        training_config = TrainingConfig(
            input_dir=data_dir,
            output_dir=tmp_path / "output",
            tokenizer_model_path=tmp_path / "tokenizer.model",
            sequence_length=4,
            batch_size=1,
            max_steps=1,
            lr_total_steps=1,
            epochs=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
        )
        checkpoint_path = tmp_path / "output" / "sml"
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / train_sml.METADATA_NAME).write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )
        monkeypatch.setattr(
            train_sml,
            "load_training_checkpoint",
            Spy(side_effect=AssertionError("checkpoint should not be loaded")),
        )
        monkeypatch.setattr(
            train_sml,
            "iter_mlx_batches",
            Spy(side_effect=RuntimeError("stop after batches")),
        )

        with pytest.raises(RuntimeError, match="stop after batches"):
            train_sml.train_model(training_config, model_config=tiny_config())

    def test_train_model_restarts_from_checkpoint_name_when_resume_is_enabled(
        self,
        monkeypatch,
        tmp_path,
    ):
        require_mlx()
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=16,
            shards=[[[1, 4, 5, 6, 2]]],
        )
        training_config = TrainingConfig(
            input_dir=data_dir,
            output_dir=tmp_path / "output",
            tokenizer_model_path=tmp_path / "tokenizer.model",
            sequence_length=4,
            batch_size=1,
            max_steps=1,
            lr_total_steps=1,
            epochs=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
            checkpoint_name="sml",
            model_path=tmp_path / "output" / "sml",
        )
        load_training_checkpoint = Spy(return_value=train_sml.TrainingResumeState())

        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )
        monkeypatch.setattr(
            train_sml,
            "load_training_checkpoint",
            load_training_checkpoint,
        )
        monkeypatch.setattr(
            train_sml,
            "iter_mlx_batches",
            Spy(side_effect=RuntimeError("stop after batches")),
        )

        with pytest.raises(RuntimeError, match="stop after batches"):
            train_sml.train_model(
                training_config,
                model_config=tiny_config(),
                resume_from_checkpoint=True,
            )

        load_training_checkpoint.assert_called_once()
        assert (
            training_config.output_dir / "sml"
            == load_training_checkpoint.call_args.args[0]
        )

    def test_train_model_uses_checkpoint_input_file_order_when_resume_is_enabled(
        self,
        monkeypatch,
        tmp_path,
    ):
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=16,
            shards=[
                [[1, 4, 5, 6, 2]],
                [[2, 7, 8, 9, 2]],
            ],
        )
        checkpoint_order = (
            data_dir / "train-000001.npz",
            data_dir / "train-000000.npz",
        )
        training_config = TrainingConfig(
            input_dir=data_dir,
            output_dir=tmp_path / "output",
            tokenizer_model_path=tmp_path / "tokenizer.model",
            sequence_length=4,
            batch_size=1,
            max_steps=1,
            lr_total_steps=1,
            epochs=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
        )
        iter_prepared_token_blocks = Spy(
            side_effect=RuntimeError("stop after shard order")
        )

        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )
        monkeypatch.setattr(
            train_sml,
            "load_training_checkpoint",
            Spy(
                return_value=train_sml.TrainingResumeState(
                    step=0,
                    input_files=checkpoint_order,
                    data_state=train_sml.PretrainingDataState(),
                )
            ),
        )
        monkeypatch.setattr(
            train_sml, "iter_prepared_token_blocks", iter_prepared_token_blocks
        )

        with pytest.raises(RuntimeError, match="stop after shard order"):
            train_sml.train_model(
                training_config,
                model_config=tiny_config(),
                resume_from_checkpoint=True,
            )

        assert (
            checkpoint_order
            == iter_prepared_token_blocks.call_args.kwargs["shard_paths"]
        )

    def test_tiny_mlx_training_run_writes_checkpoint(self, tmp_path, monkeypatch):
        require_mlx()
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=16,
            shards=[[[1, 4, 5, 6, 2], [2, 7, 8, 9, 2]]],
        )
        tokenizer_path = tmp_path / "tokenizer.model"
        training_config = TrainingConfig(
            input_dir=data_dir,
            output_dir=tmp_path / "out",
            tokenizer_model_path=tokenizer_path,
            sequence_length=4,
            batch_size=1,
            max_steps=1,
            lr_total_steps=1,
            epochs=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
        )

        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )

        checkpoint_path = train_sml.train_model(
            training_config=training_config,
            model_config=tiny_config(),
        )

        assert checkpoint_path == tmp_path / "out" / "sml"
        assert (checkpoint_path / train_sml.MODEL_WEIGHTS_NAME).exists()
        assert (checkpoint_path / train_sml.OPTIMIZER_STATE_NAME).exists()
        assert (checkpoint_path / train_sml.METADATA_NAME).exists()
        metadata = json.loads(
            (checkpoint_path / train_sml.METADATA_NAME).read_text(encoding="utf-8")
        )
        assert "not_guaranteed" == metadata["stochastic_resume"]
        assert train_sml.STOCHASTIC_RESUME_NOTE == metadata["resume_note"]
        assert metadata["data_state"] == {
            "epoch": 0,
            "shard_index": 0,
            "block_index": 1,
        }

    def test_tiny_mlx_training_run_can_resume_checkpoint(self, tmp_path, monkeypatch):
        require_mlx()
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=16,
            shards=[
                [[1, 4, 5, 6, 2], [2, 7, 8, 9, 2], [1, 10, 11, 12, 2]],
            ],
        )
        tokenizer_path = tmp_path / "tokenizer.model"

        def make_config(max_steps: int):
            return TrainingConfig(
                input_dir=data_dir,
                output_dir=tmp_path / "out",
                tokenizer_model_path=tokenizer_path,
                sequence_length=4,
                batch_size=1,
                max_steps=max_steps,
                lr_total_steps=max_steps,
                epochs=1,
                learning_rate=1e-4,
                gradient_accumulation_steps=1,
                log_every=1,
                save_every=1,
            )

        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )

        checkpoint_path = train_sml.train_model(
            training_config=make_config(max_steps=1),
            model_config=tiny_config(),
        )
        resumed_path = train_sml.train_model(
            training_config=make_config(max_steps=2),
            model_config=tiny_config(),
            resume_from_checkpoint=True,
        )
        metadata = json.loads(
            (resumed_path / train_sml.METADATA_NAME).read_text(encoding="utf-8")
        )

        assert resumed_path == checkpoint_path
        assert metadata["step"] == 2
        assert metadata["data_state"] == {
            "epoch": 0,
            "shard_index": 0,
            "block_index": 2,
        }
