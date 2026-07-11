import ast
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SML_PATH = PROJECT_DIR / "src" / "sml.py"


def get_all_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            raise AssertionError("__all__ must be a list literal")
        return [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant)
        ]
    raise AssertionError("__all__ is not defined")


def test_sml_owns_merged_public_api() -> None:
    assert sorted(get_all_names(SML_PATH)) == [
        "GenerationConfig",
        "GroupedQueryAttention",
        "KVCache",
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
        "lr_lambda",
        "resolve_yarn_attention_factor",
        "rotate_half",
        "select_next_token",
        "yarn_find_correction_dim",
        "yarn_find_correction_range",
        "yarn_get_mscale",
        "yarn_linear_ramp_mask",
    ]


def test_grouped_query_attention_always_uses_fused_mlx_attention() -> None:
    source = SML_PATH.read_text(encoding="utf-8")
    attention_source = source[
        source.index("class GroupedQueryAttention") : source.index(
            "class SwiGLUFeedForward"
        )
    ]

    assert "mx.fast.scaled_dot_product_attention" in attention_source
    assert "self.attention_dropout" not in attention_source
    assert "nn.Dropout" not in attention_source
    assert "_attention_with_dropout" not in attention_source
    assert "does not expose attention-dropout" in attention_source
