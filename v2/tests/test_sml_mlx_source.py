from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SML_PATH = PROJECT_DIR / "src" / "sml.py"


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
