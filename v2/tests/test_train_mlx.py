import json
import sys
from pathlib import Path

import pytest

from helpers import Spy


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import mlx.core as mx

    mx.eval(mx.array([0]))
except (ImportError, RuntimeError) as exc:  # pragma: no cover - depends on host Metal access
    pytestmark = pytest.mark.skip(reason=f"mlx is not available: {exc}")


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, out_type=int):
        del out_type
        return [int(part) for part in text.split()]

    def get_piece_size(self):
        return 16


def tiny_config():
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
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_checkpointing=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )


def test_mlx_batch_iterator_emits_mx_arrays_with_next_token_labels():
    from train_sml_mlx import iter_mlx_batches

    examples = (
        {"input_ids": [1, 4, 5], "labels": [4, 5, 6]},
        {"input_ids": [6, 7, 8], "labels": [7, 8, 2]},
    )

    batches = list(iter_mlx_batches(examples, batch_size=2))

    assert len(batches) == 1
    assert isinstance(batches[0]["input_ids"], mx.array)
    assert isinstance(batches[0]["labels"], mx.array)
    assert batches[0]["input_ids"].shape == (2, 3)
    assert batches[0]["labels"].shape == (2, 3)
    assert batches[0]["input_ids"].tolist() == [[1, 4, 5], [6, 7, 8]]
    assert batches[0]["labels"].tolist() == [[4, 5, 6], [7, 8, 2]]


def test_mlx_lr_schedule_matches_training_helper():
    from train_sml import lr_lambda
    from train_sml_mlx import build_lr_schedule

    schedule = build_lr_schedule(
        learning_rate=0.01,
        total_steps=10,
        warmup_steps=2,
        min_lr_ratio=0.1,
    )

    for step in (0, 1, 5, 10):
        expected = 0.01 * lr_lambda(
            step=step,
            total_steps=10,
            warmup_steps=2,
            min_lr_ratio=0.1,
        )
        assert float(schedule(mx.array(step)).item()) == pytest.approx(expected)


def test_clip_gradients_by_global_norm_scales_large_grads():
    from train_sml_mlx import clip_gradients_by_global_norm

    grads = {"w": mx.array([3.0, 4.0])}

    clipped, grad_norm = clip_gradients_by_global_norm(grads, max_norm=1.0)
    clipped_norm = mx.sqrt(mx.sum(clipped["w"] * clipped["w"]))
    mx.eval(clipped_norm, grad_norm)

    assert float(grad_norm.item()) == pytest.approx(5.0)
    assert float(clipped_norm.item()) == pytest.approx(1.0)


def test_resolve_mlx_checkpoint_path_avoids_pt_suffix(tmp_path):
    from train_sml import TrainingConfig
    from train_sml_mlx import resolve_mlx_checkpoint_path

    checkpoint_path = resolve_mlx_checkpoint_path(
        TrainingConfig(output_dir=tmp_path, checkpoint_name="sml.pt")
    )

    assert checkpoint_path == tmp_path / "sml_mlx"


def test_missing_explicit_resume_checkpoint_is_rejected(tmp_path):
    from train_sml_mlx import load_training_checkpoint

    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        load_training_checkpoint(tmp_path / "missing", object(), object())


def test_main_passes_resume_flag_to_train_model(monkeypatch):
    import train_sml_mlx

    train_model = Spy(return_value=Path("/tmp/sml_mlx"))
    monkeypatch.setattr(train_sml_mlx, "train_model", train_model)

    return_code = train_sml_mlx.main(["--resume"])

    assert train_sml_mlx.SUCCESS_RETURN_CODE == return_code
    assert train_model.call_args.kwargs["resume_from_checkpoint"]


def test_tiny_mlx_training_run_writes_checkpoint(tmp_path, monkeypatch):
    import train_sml_mlx
    from train_sml import TrainingConfig

    data_file = tmp_path / "data.jsonl.zst"
    tokenizer_path = tmp_path / "tokenizer.model"
    training_config = TrainingConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        tokenizer_model_path=tokenizer_path,
        sequence_length=4,
        batch_size=1,
        max_steps=1,
        lr_total_steps=1,
        epochs=1,
        max_rows_per_file=None,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        log_every=1,
        save_every=1,
    )

    monkeypatch.setattr(
        train_sml_mlx,
        "discover_input_files",
        Spy(return_value=(data_file,)),
    )
    monkeypatch.setattr(
        train_sml_mlx,
        "load_tokenizer",
        Spy(return_value=FakeTokenizer()),
    )
    monkeypatch.setattr(
        train_sml_mlx,
        "iter_texts",
        lambda *args, **kwargs: iter(["4 5 6 7 8 9 10 11"]),
    )

    checkpoint_path = train_sml_mlx.train_model(
        training_config=training_config,
        model_config=tiny_config(),
    )

    assert checkpoint_path == tmp_path / "out" / "sml_mlx"
    assert (checkpoint_path / "model.safetensors").exists()
    assert (checkpoint_path / "optimizer.npz").exists()
    assert (checkpoint_path / "metadata.json").exists()


def test_tiny_mlx_training_run_can_resume_checkpoint(tmp_path, monkeypatch):
    import train_sml_mlx
    from train_sml import TrainingConfig

    data_file = tmp_path / "data.jsonl.zst"
    tokenizer_path = tmp_path / "tokenizer.model"

    def make_config(max_steps: int):
        return TrainingConfig(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            tokenizer_model_path=tokenizer_path,
            sequence_length=4,
            batch_size=1,
            max_steps=max_steps,
            lr_total_steps=max_steps,
            epochs=1,
            max_rows_per_file=None,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
        )

    monkeypatch.setattr(
        train_sml_mlx,
        "discover_input_files",
        Spy(return_value=(data_file,)),
    )
    monkeypatch.setattr(
        train_sml_mlx,
        "load_tokenizer",
        Spy(return_value=FakeTokenizer()),
    )
    monkeypatch.setattr(
        train_sml_mlx,
        "iter_texts",
        lambda *args, **kwargs: iter(["4 5 6 7 8 9 10 11 12 13 14"]),
    )

    checkpoint_path = train_sml_mlx.train_model(
        training_config=make_config(max_steps=1),
        model_config=tiny_config(),
    )
    resumed_path = train_sml_mlx.train_model(
        training_config=make_config(max_steps=2),
        model_config=tiny_config(),
        resume_from_checkpoint=True,
    )
    metadata = json.loads((resumed_path / "metadata.json").read_text(encoding="utf-8"))

    assert resumed_path == checkpoint_path
    assert metadata["step"] == 2
