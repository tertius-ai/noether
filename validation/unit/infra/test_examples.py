# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Every shipped example is pure Noether and remains executable."""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))


def test_release_has_exactly_forty_examples() -> None:
    assert len(EXAMPLES) == 40


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.stem)
def test_example_is_pure_noether_and_executes(
    path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imports <= sys.stdlib_module_names | {"noether"}

    runpy.run_path(str(path), run_name="__main__")
    assert capsys.readouterr().out.strip(), f"{path.name} produced no mathematical result"
