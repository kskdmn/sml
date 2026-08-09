from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from sml import model

LEGACY_BRIDGE_EXPORTS = (
    "ParameterInitializerRangeConfig",
    "SMLConfig",
    "GenerationConfig",
    "SMLForwardOutput",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_get_mscale",
    "resolve_yarn_attention_factor",
    "yarn_linear_ramp_mask",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_repetition_penalty",
    "apply_no_repeat_ngram",
    "select_next_token",
    "RMSNorm",
    "RotaryEmbedding",
    "KVCache",
    "GroupedQueryAttention",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "SMLLanguageModel",
    "compute_causal_lm_loss",
    "count_parameters",
    "create_model",
    "estimate_model_size",
)


def _load_legacy_module():
    module_name = "sml._legacy"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    legacy_path = Path(__file__).resolve().parents[1] / "sml.py"
    spec = importlib.util.spec_from_file_location(module_name, legacy_path)
    if spec is None or spec.loader is None:  # pragma: no cover - filesystem invariant
        raise ImportError(f"cannot load legacy module from {legacy_path}")

    legacy_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = legacy_module
    legacy_root = str(legacy_path.parent)
    added_legacy_root = legacy_root not in sys.path
    if added_legacy_root:
        sys.path.append(legacy_root)
    try:
        spec.loader.exec_module(legacy_module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if added_legacy_root:
            sys.path.remove(legacy_root)
    return legacy_module


_legacy = _load_legacy_module()
for _name in LEGACY_BRIDGE_EXPORTS:
    globals()[_name] = getattr(_legacy, _name)

__all__ = (
    "LEGACY_BRIDGE_EXPORTS",
    "GenerationConfig",
    "GroupedQueryAttention",
    "KVCache",
    "ParameterInitializerRangeConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "SMLConfig",
    "SMLForwardOutput",
    "SMLLanguageModel",
    "SwiGLUFeedForward",
    "TransformerBlock",
    "apply_no_repeat_ngram",
    "apply_repetition_penalty",
    "apply_rotary_pos_emb",
    "compute_causal_lm_loss",
    "count_parameters",
    "create_model",
    "estimate_model_size",
    "model",
    "resolve_yarn_attention_factor",
    "rotate_half",
    "select_next_token",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_get_mscale",
    "yarn_linear_ramp_mask",
)

del _name
