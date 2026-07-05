import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only before torch is installed
    torch = None
    nn = None


@pytest.mark.skipif(torch is None, reason="torch is not installed")
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

    def test_apply_lora_freezes_base_weights(self):
        from lora import apply_lora, lora_parameters
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)
        original_q_weight = model.layers[0].self_attn.q_proj.weight.detach().clone()

        apply_lora(model, lora_config)
        trainable = lora_parameters(model)

        assert 2 == len(trainable)
        assert not model.layers[0].self_attn.q_proj.linear.weight.requires_grad
        assert torch.equal(original_q_weight, model.layers[0].self_attn.q_proj.linear.weight.detach())

    def test_apply_lora_leaves_only_adapter_parameters_trainable(self):
        from lora import apply_lora, lora_parameters
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)

        apply_lora(model, lora_config)

        adapter_parameter_ids = {id(parameter) for parameter in lora_parameters(model)}
        trainable_parameter_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        assert adapter_parameter_ids == trainable_parameter_ids

    def test_apply_lora_places_adapters_on_base_device(self):
        from lora import apply_lora
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config).to("cpu")
        apply_lora(model, lora_config)

        wrapped = model.layers[0].self_attn.q_proj
        assert wrapped.lora_A.device == wrapped.linear.weight.device
        assert wrapped.lora_B.device == wrapped.linear.weight.device

    def test_lora_forward_changes_output(self):
        from lora import LoRALinear

        linear = nn.Linear(8, 4, bias=True)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 8)

        with torch.no_grad():
            lora.lora_B.fill_(0.1)

        base_output = linear(x)
        lora_output = lora(x)

        assert not torch.allclose(base_output, lora_output)

    def test_lora_forward_matches_activation_dtype(self):
        from lora import LoRALinear

        linear = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 8)

        with torch.no_grad():
            lora.lora_B.fill_(0.1)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = lora(x)

        assert output.dtype == torch.bfloat16

    def test_merge_lora_updates_base_linear(self):
        from lora import LoRALinear, merge_lora
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        model = SMLLanguageModel(config)
        apply_lora = __import__("lora").apply_lora
        apply_lora(model, lora_config)

        with torch.no_grad():
            model.layers[0].self_attn.q_proj.lora_B.fill_(0.05)

        x = torch.randn(1, 1, config.hidden_size)
        wrapped = model.layers[0].self_attn.q_proj
        expected_delta = (
            wrapped.scaling
            * (x @ wrapped.lora_A.T @ wrapped.lora_B.T)
        ).detach()
        before = model.layers[0].self_attn(x, kv_cache=None).detach()

        merge_lora(model)
        after = model.layers[0].self_attn(x, kv_cache=None).detach()

        assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
        assert torch.allclose(before, after, atol=1e-05, rtol=1e-05)
        assert expected_delta.abs().sum().item() > 0.0

    def test_lora_state_dict_round_trip(self):
        from lora import apply_lora, load_lora_state_dict, lora_state_dict
        from sml import SMLLanguageModel

        config, lora_config = self.tiny_config()
        source = SMLLanguageModel(config)
        target = SMLLanguageModel(config)
        apply_lora(source, lora_config)
        apply_lora(target, lora_config)

        with torch.no_grad():
            source.layers[0].self_attn.q_proj.lora_A.fill_(0.25)
            source.layers[0].self_attn.q_proj.lora_B.fill_(0.5)

        load_lora_state_dict(target, lora_state_dict(source))

        assert torch.equal(source.layers[0].self_attn.q_proj.lora_A, target.layers[0].self_attn.q_proj.lora_A)
        assert torch.equal(source.layers[0].self_attn.q_proj.lora_B, target.layers[0].self_attn.q_proj.lora_B)
