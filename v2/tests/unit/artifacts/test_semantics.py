from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "modules",
    (
        ("sml.inference", "sml.artifacts.verify", "sml.artifacts.semantics"),
        ("sml.artifacts.verify", "sml.artifacts.semantics", "sml.inference"),
    ),
)
def test_shared_semantics_module_has_safe_import_order(
    modules: tuple[str, ...],
) -> None:
    source_root = Path(__file__).parents[3] / "src"
    imports = ";".join(f"import {module}" for module in modules)
    code = f"import sys;sys.path.insert(0,{str(source_root)!r});{imports}"

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
