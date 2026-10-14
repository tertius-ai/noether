# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Execute the curated mathematical corpus through the public Python surface."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import noether

ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = ROOT / "validation" / "corpus"
MANIFEST_PATH = CORPUS_ROOT / "cases.json"
ACCEPTED_DOMAINS = {
    "combin",
    "commalg",
    "geometry",
    "groups",
    "homalg",
    "linalg",
    "numthy",
    "topology",
}
CORPUS_RUNNER = """\
import runpy
import sys

sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2], run_name="__main__")
"""


def _manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


MANIFEST = _manifest()
CASES = tuple(MANIFEST["cases"])


def _case_path(case: dict[str, Any]) -> Path:
    owner, slug = case["id"].split(".", maxsplit=1)
    return CORPUS_ROOT / owner / f"{slug}.py"


def _imports(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None
            names.append(node.module)
            if node.module == "noether":
                assert all(alias.name != "_engine" for alias in node.names)
    return tuple(names)


def test_corpus_manifest_and_programs_are_a_closed_public_contract() -> None:
    assert set(MANIFEST) == {
        "cases",
        "schema",
        "source_noether_version",
        "source_repository",
        "source_revision",
    }
    assert MANIFEST["schema"] == "noether.corpus-acceptance.v1"
    assert MANIFEST["source_repository"] == "tertius-ai/cas-corpus"
    assert len(MANIFEST["source_revision"]) == 40

    ids = [case["id"] for case in CASES]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))

    expected_paths: set[Path] = set()
    participating_domains: set[str] = set()
    capabilities = noether.capabilities()
    for case in CASES:
        assert set(case) == {
            "domains",
            "id",
            "operations",
            "seed",
            "source",
            "timeout_seconds",
            "title",
        }
        owner, _ = case["id"].split(".", maxsplit=1)
        domains = case["domains"]
        assert owner in domains
        assert len(domains) == len(set(domains))
        assert set(domains) <= ACCEPTED_DOMAINS
        participating_domains.update(domains)
        assert case["title"].strip()

        operations = case["operations"]
        assert operations and len(operations) == len(set(operations))
        assert all(capabilities.supports(operation) for operation in operations)
        assert 1 <= case["timeout_seconds"] <= 300
        assert case["seed"] is None or isinstance(case["seed"], int)

        source = case["source"]
        assert set(source) == {"citation", "kind", "license", "redistribution", "url"}
        assert source["citation"].strip()
        assert source["license"].strip().lower() not in {"", "tbd", "unknown", "unreviewed"}
        assert source["redistribution"] in {"citation-only", "licensed-copy", "original"}

        path = _case_path(case)
        expected_paths.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imports(tree)
        assert all(
            name.partition(".")[0] == "noether" or name.partition(".")[0] in sys.stdlib_module_names
            for name in imports
        )
        assert all(
            name != "noether._engine" and not name.startswith("noether._engine.")
            for name in imports
        )
        assert any(isinstance(node, ast.Assert) for node in ast.walk(tree))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "main" for node in ast.walk(tree)
        )
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        )

    assert participating_domains == ACCEPTED_DOMAINS
    actual_paths = set(CORPUS_ROOT.glob("*/*.py"))
    assert actual_paths == expected_paths


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_curated_corpus_program(case: dict[str, Any]) -> None:
    path = _case_path(case)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    result = subprocess.run(
        [sys.executable, "-I", "-c", CORPUS_RUNNER, str(ROOT / "python"), str(path)],
        cwd=path.parent,
        env=environment,
        text=True,
        capture_output=True,
        timeout=case["timeout_seconds"],
        check=False,
    )
    assert result.returncode == 0, (
        f"{case['id']} failed ({result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert result.stdout.strip(), f"{case['id']} produced no mathematical result"
