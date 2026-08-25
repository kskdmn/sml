import ast
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT.parent
SCAN_ROOTS = (PROJECT / "src", PROJECT / "tests", PROJECT / "README.md")
LEGACY_TENSOR_LIB = "tor" + "ch"
LEGACY_FRAMEWORK = "py" + LEGACY_TENSOR_LIB
LEGACY_APPLE_ACCELERATOR = "m" + "ps"
LEGACY_OTHER_ACCELERATOR = "cu" + "da"
DEVICE_FLAG = "--" + "device"
RESOLVE_DEVICE = "resolve_" + "device"
DEVICE_NAME = "device_" + "name"
FORBIDDEN_MLX_ONLY_PATTERNS = (
    re.compile(rf"\b{LEGACY_TENSOR_LIB}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_TENSOR_LIB}\s*\.", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_FRAMEWORK}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_APPLE_ACCELERATOR}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_OTHER_ACCELERATOR}\b", re.IGNORECASE),
    re.compile(DEVICE_FLAG, re.IGNORECASE),
    re.compile(rf"\b{RESOLVE_DEVICE}\b", re.IGNORECASE),
    re.compile(rf"\b{DEVICE_NAME}\b", re.IGNORECASE),
)


def _iter_mlx_only_scan_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*.py"):
            if path == Path(__file__):
                continue
            yield path


def _find_legacy_imports(path: Path) -> list[str]:
    if path.suffix != ".py":
        return []

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", maxsplit=1)[0].lower() == LEGACY_TENSOR_LIB:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module_root = (node.module or "").split(".", maxsplit=1)[0].lower()
            if module_root == LEGACY_TENSOR_LIB:
                offenders.append(f"line {node.lineno}: from {node.module}")
    return offenders


def test_final_v2_source_has_only_package_modules():
    source_root = PROJECT / "src"

    assert (
        sorted(path.name for path in source_root.iterdir() if path.suffix == ".py")
        == []
    )
    assert (source_root / "sml" / "__main__.py").is_file()


def test_final_source_has_no_bridge_or_legacy_imports():
    package_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT / "src" / "sml").rglob("*.py")
    )

    for forbidden in (
        "spec_from_file_location",
        "sml._legacy",
        "LEGACY_BRIDGE_EXPORTS",
        "train_sml",
        "infer_sml",
        "evaluate_sml",
        "ft_swag",
        "from config import",
    ):
        assert forbidden not in package_source


def test_v2_source_tests_and_readme_are_mlx_only():
    offenders: list[str] = []
    for path in _iter_mlx_only_scan_files():
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPO_ROOT)
        for detail in _find_legacy_imports(path):
            offenders.append(f"{relative_path}: {detail}")
        for pattern in FORBIDDEN_MLX_ONLY_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{relative_path}: {pattern.pattern}")

    assert offenders == []


def test_readme_documents_only_unified_commands_and_directory_artifacts():
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")

    for command in (
        "tokenize",
        "prepare pretraining",
        "prepare swag",
        "train",
        "infer",
        "evaluate",
        "finetune",
        "export",
        "verify",
    ):
        assert f"uv run python -m sml {command}" in readme
    for forbidden in (
        ".npz",
        "v2/src/config.py",
        "v2/src/evaluate_sml.py",
        "v2/src/ft_swag.py",
        "v2/src/infer_sml.py",
        "v2/src/prepare_pretraining_data.py",
        "v2/src/train_sml.py",
        "v2/src/train_tokenizer.py",
    ):
        assert forbidden not in readme
