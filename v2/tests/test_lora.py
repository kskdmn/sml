import sys
from pathlib import Path

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


class TestLoRA:
    def tiny_config(self):
        from lora import LoRAConfig
        from sml import SMLConfig

        return (
            SMLConfig(
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
            ),
            LoRAConfig(rank=4, alpha=8.0, dropout=0.0, target_modules=("q_proj",)),
        )

    def test_lora_config_keeps_expected_defaults(self):
        from lora import LoRAConfig

        config = LoRAConfig()

        assert 16 == config.rank
        assert 32.0 == config.alpha
        assert "q_proj" in config.target_modules

    def test_lora_linear_validates_rank_and_alpha(self):
        require_mlx_runtime()
        import mlx.nn as nn
        from lora import LoRALinear

        linear = nn.Linear(8, 4)

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRALinear(linear, rank=0, alpha=8.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRALinear(linear, rank=4, alpha=0.0)

    def test_apply_lora_freezes_base_weights(self):
        mx = require_mlx_runtime()
        from lora import apply_lora, lora_parameters
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)
        original_q_weight = model.layers[0].self_attn.q_proj.weight

        apply_lora(model, lora_config)
        trainable = lora_parameters(model)

        assert 2 == len(trainable)
        assert "weight" not in model.layers[0].self_attn.q_proj.linear.trainable_parameters()
        assert bool(
            mx.array_equal(
                original_q_weight,
                model.layers[0].self_attn.q_proj.linear.weight,
            ).item()
        )

    def test_apply_lora_leaves_only_adapter_arrays_trainable(self):
        require_mlx_runtime()
        from lora import apply_lora, lora_parameters
        from mlx.utils import tree_flatten
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)

        apply_lora(model, lora_config)

        adapter_array_ids = {id(parameter) for parameter in lora_parameters(model)}
        trainable_array_ids = {
            id(parameter) for _, parameter in tree_flatten(model.trainable_parameters())
        }
        assert adapter_array_ids == trainable_array_ids

    def test_lora_forward_changes_output_after_nonzero_b(self):
        mx = require_mlx_runtime()
        import mlx.nn as nn
        from lora import LoRALinear

        linear = nn.Linear(8, 4, bias=True)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = mx.random.normal(shape=(2, 8))
        lora.lora_B = mx.full(lora.lora_B.shape, 0.1)

        base_output = linear(x)
        lora_output = lora(x)
        mx.eval(base_output, lora_output)

        assert not bool(mx.allclose(base_output, lora_output).item())

    def test_lora_forward_matches_base_output_dtype(self):
        mx = require_mlx_runtime()
        import mlx.nn as nn
        from lora import LoRALinear

        linear = nn.Linear(8, 4, bias=False)
        linear.weight = linear.weight.astype(mx.float16)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        lora.lora_B = mx.full(lora.lora_B.shape, 0.1)
        x = mx.random.normal(shape=(2, 8)).astype(mx.float16)

        output = lora(x)
        mx.eval(output)

        assert mx.float16 == output.dtype

    def test_merge_lora_preserves_output_and_restores_base_linear(self):
        mx = require_mlx_runtime()
        import mlx.nn as nn
        from lora import apply_lora, merge_lora
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)
        apply_lora(model, lora_config)
        model.layers[0].self_attn.q_proj.lora_B = mx.full(
            model.layers[0].self_attn.q_proj.lora_B.shape,
            0.05,
        )

        x = mx.random.normal(shape=(1, 1, config.hidden_size))
        before = model.layers[0].self_attn(x, kv_cache=None)
        merge_lora(model)
        after = model.layers[0].self_attn(x, kv_cache=None)
        mx.eval(before, after)

        assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
        assert bool(mx.allclose(before, after, atol=1e-5, rtol=1e-5).item())

    def test_lora_state_dict_round_trip(self):
        mx = require_mlx_runtime()
        from lora import apply_lora, load_lora_state_dict, lora_state_dict
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        source = SMLLanguageModel(config)
        target = SMLLanguageModel(config)
        apply_lora(source, lora_config)
        apply_lora(target, lora_config)
        source.layers[0].self_attn.q_proj.lora_A = mx.full(
            source.layers[0].self_attn.q_proj.lora_A.shape,
            0.25,
        )
        source.layers[0].self_attn.q_proj.lora_B = mx.full(
            source.layers[0].self_attn.q_proj.lora_B.shape,
            0.5,
        )

        load_lora_state_dict(target, lora_state_dict(source))

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
