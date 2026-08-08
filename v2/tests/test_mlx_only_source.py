import ast
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
SCAN_ROOTS = (PROJECT_DIR / "src", PROJECT_DIR / "tests", PROJECT_DIR / "README.md")
LEGACY_TENSOR_LIB = "tor" + "ch"
LEGACY_FRAMEWORK = "py" + LEGACY_TENSOR_LIB
LEGACY_APPLE_ACCELERATOR = "m" + "ps"
LEGACY_OTHER_ACCELERATOR = "cu" + "da"
DEVICE_FLAG = "--" + "device"
RESOLVE_DEVICE = "resolve_" + "device"
DEVICE_NAME = "device_" + "name"
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(rf"\b{LEGACY_TENSOR_LIB}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_TENSOR_LIB}\s*\.", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_FRAMEWORK}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_APPLE_ACCELERATOR}\b", re.IGNORECASE),
    re.compile(rf"\b{LEGACY_OTHER_ACCELERATOR}\b", re.IGNORECASE),
    re.compile(DEVICE_FLAG, re.IGNORECASE),
    re.compile(rf"\b{RESOLVE_DEVICE}\b", re.IGNORECASE),
    re.compile(rf"\b{DEVICE_NAME}\b", re.IGNORECASE),
)


def iter_scan_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*.py"):
            if path.name == "test_mlx_only_source.py":
                continue
            yield path


def find_legacy_imports(path: Path) -> list[str]:
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


def test_v2_source_and_tests_are_mlx_only():
    offenders: list[str] = []
    for path in iter_scan_files():
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(REPO_ROOT)
        for detail in find_legacy_imports(path):
            offenders.append(f"{relative_path}: {detail}")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{relative_path}: {pattern.pattern}")

    assert offenders == []
