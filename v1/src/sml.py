from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from sml_config import SMLConfig


@dataclass(slots=True)
class SMLForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        end = position_offset + seq_len
        return (
            self.cos_cached[position_offset:end],
            self.sin_cached[position_offset:end],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class KVCache:
    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_len(self, layer_idx: int) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=2)
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value),
                dim=2,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_q_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_q_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            config.num_q_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2)
        x = x.expand(batch_size, kv_heads, self.num_groups, seq_len, head_dim)
        return x.reshape(batch_size, self.num_q_heads, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        position_offset = 0 if kv_cache is None else kv_cache.get_seq_len(self.layer_idx)
        cos, sin = self.rope(seq_len, position_offset)
        q, k = apply_rotary_pos_emb(
            q,
            k,
            cos.to(device=x.device, dtype=q.dtype),
            sin.to(device=x.device, dtype=q.dtype),
        )

        past_seq_len = position_offset
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        dropout_p = self.attention_dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=past_seq_len == 0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: SMLConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_norm(x), kv_cache)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class SMLLanguageModel(nn.Module):
    def __init__(self, config: SMLConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> SMLForwardOutput:
        if input_ids.shape[-1] > self.config.max_position_embeddings:
            raise ValueError(
                "input sequence length exceeds max_position_embeddings: "
                f"{input_ids.shape[-1]} > {self.config.max_position_embeddings}"
            )

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self._should_checkpoint_layers(kv_cache):
                x = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, kv_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = compute_causal_lm_loss(
                logits,
                labels,
                pad_token_id=self.config.pad_token_id,
            )
        return SMLForwardOutput(logits=logits, loss=loss)

    def _should_checkpoint_layers(self, kv_cache: KVCache | None) -> bool:
        return (
            self.config.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and kv_cache is None
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        generated = input_ids
        kv_cache = KVCache() if self.config.use_cache else None

        for _ in range(max_new_tokens):
            if kv_cache is not None and len(kv_cache.key_cache) > 0:
                model_input = generated[:, -1:]
            else:
                model_input = generated

            output = self(model_input, kv_cache=kv_cache)
            next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        return generated


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "labels must have the same batch and sequence shape as logits"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_token_id,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def create_model(config: SMLConfig | None = None) -> SMLLanguageModel:
    return SMLLanguageModel(config or SMLConfig())


def estimate_model_size(config: SMLConfig) -> int:
    attention_params = config.num_layers * (
        config.hidden_size * config.num_q_heads * config.head_dim
        + 2 * config.hidden_size * config.num_kv_heads * config.head_dim
        + config.hidden_size * config.hidden_size
    )
    mlp_params = config.num_layers * (
        2 * config.hidden_size * config.intermediate_size
        + config.intermediate_size * config.hidden_size
    )
    embedding_params = config.vocab_size * config.hidden_size
    norm_params = (config.num_layers * 2 + 1) * config.hidden_size
    untied_head_params = 0 if config.tie_word_embeddings else embedding_params
    return int(
        attention_params
        + mlp_params
        + embedding_params
        + norm_params
        + untied_head_params
    )


def lr_lambda(
    step: int,
    total_steps: int | None,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps is None:
        return 1.0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return max(min_lr_ratio, cosine)
