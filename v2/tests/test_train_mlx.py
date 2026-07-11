import os
import subprocess
import sys
from pathlib import Path

from helpers import Spy


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def test_train_sml_mlx_re_exports_canonical_training_entrypoints(monkeypatch):
    import train_sml
    import train_sml_mlx

    assert train_sml.METADATA_NAME == train_sml_mlx.METADATA_NAME
    assert train_sml.MODEL_WEIGHTS_NAME == train_sml_mlx.MODEL_WEIGHTS_NAME
    assert train_sml.OPTIMIZER_STATE_NAME == train_sml_mlx.OPTIMIZER_STATE_NAME
    assert train_sml.SUCCESS_RETURN_CODE == train_sml_mlx.SUCCESS_RETURN_CODE
    assert train_sml.build_parser is train_sml_mlx.build_parser
    assert train_sml.build_lr_schedule is train_sml_mlx.build_lr_schedule
    assert train_sml.clip_gradients_by_global_norm is train_sml_mlx.clip_gradients_by_global_norm
    assert train_sml.global_grad_norm is train_sml_mlx.global_grad_norm
    assert train_sml.iter_mlx_batches is train_sml_mlx.iter_mlx_batches
    assert train_sml.iter_mlx_token_blocks is train_sml_mlx.iter_mlx_token_blocks
    assert train_sml.load_training_checkpoint is train_sml_mlx.load_training_checkpoint
    assert train_sml.parse_args is train_sml_mlx.parse_args
    assert train_sml.resolve_mlx_checkpoint_path is train_sml_mlx.resolve_mlx_checkpoint_path
    assert train_sml.save_checkpoint is train_sml_mlx.save_checkpoint
    assert train_sml.set_seed is train_sml_mlx.set_seed
    assert train_sml.train_model is train_sml_mlx.train_model
    assert train_sml.tree_add is train_sml_mlx.tree_add
    assert train_sml.tree_scale is train_sml_mlx.tree_scale

    train_model = Spy(return_value=Path("/tmp/sml"))
    monkeypatch.setattr(train_sml, "train_model", train_model)

    return_code = train_sml_mlx.main(["--resume"])

    assert train_sml_mlx.SUCCESS_RETURN_CODE == return_code
    assert train_model.call_args.kwargs["resume_from_checkpoint"]


def test_train_sml_mlx_script_help_uses_compatibility_entrypoint():
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = ".uv-cache"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SRC_DIR / "train_sml_mlx.py"),
            "--help",
        ],
        cwd=PROJECT_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "Train the MLX SML language model." in result.stdout
