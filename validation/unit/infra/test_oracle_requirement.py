# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The release oracle gate must distinguish a real pass from a skip."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_required_oracles(
    *test_files: Path,
    required: tuple[str, ...] = ("sage",),
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "NOETHER_VALIDATE_GAP",
        "NOETHER_VALIDATE_GP",
        "NOETHER_VALIDATE_MAGMA",
        "NOETHER_VALIDATE_SAGE",
    ):
        env.pop(name, None)
    for name in required:
        env[f"NOETHER_{name.upper()}_BIN"] = sys.executable
    required_args = [part for name in required for part in ("--require-oracle", name)]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "validation.conftest",
            *required_args,
            *extra_args,
            *(str(test_file) for test_file in test_files),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_required_oracle_accepts_an_executed_test(tmp_path: Path) -> None:
    test_file = tmp_path / "test_oracle_pass.py"
    test_file.write_text(
        "import pytest\n@pytest.mark.sage_oracle\ndef test_oracle():\n    assert True\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_required_oracle_turns_a_skip_into_a_failure(tmp_path: Path) -> None:
    test_file = tmp_path / "test_oracle_skip.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.sage_oracle\n"
        "def test_oracle():\n"
        "    pytest.skip('oracle declined the case')\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file)
    assert completed.returncode == 1
    assert "required sage oracle test skipped" in completed.stdout


def test_required_oracle_rejects_an_empty_selection(tmp_path: Path) -> None:
    test_file = tmp_path / "test_not_an_oracle.py"
    test_file.write_text("def test_unit():\n    assert True\n", encoding="utf-8")
    completed = _run_required_oracles(test_file)
    assert completed.returncode == 4
    assert "no selected test is marked sage_oracle" in completed.stderr


def test_required_oracle_rejects_a_deselected_oracle_test(tmp_path: Path) -> None:
    test_file = tmp_path / "test_deselected_oracle.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.sage_oracle\n"
        "def test_oracle():\n"
        "    assert True\n"
        "def test_plain():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file, extra_args=("-k", "test_plain"))
    assert completed.returncode == 4
    assert "no selected test is marked sage_oracle" in completed.stderr


@pytest.mark.parametrize("disposition", ["xfail", "xpass"])
def test_required_oracle_rejects_expected_failure_markers(tmp_path: Path, disposition: str) -> None:
    assertion = "False" if disposition == "xfail" else "True"
    test_file = tmp_path / f"test_oracle_{disposition}.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.sage_oracle\n"
        "@pytest.mark.xfail(reason='oracle result is not accepted')\n"
        "def test_oracle():\n"
        f"    assert {assertion}\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file)
    assert completed.returncode == 1
    assert f"required sage oracle test {disposition}" in completed.stdout


def test_required_oracle_rejects_a_setup_skip(tmp_path: Path) -> None:
    test_file = tmp_path / "test_oracle_setup_skip.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def unavailable():\n"
        "    pytest.skip('fixture unavailable')\n"
        "@pytest.mark.sage_oracle\n"
        "def test_oracle(unavailable):\n"
        "    assert True\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file)
    assert completed.returncode == 1
    assert "required sage oracle test skipped" in completed.stdout


def test_required_oracle_rejects_a_collection_skip(tmp_path: Path) -> None:
    skipped_file = tmp_path / "test_oracle_collection_skip.py"
    skipped_file.write_text(
        "import pytest\n"
        "pytest.skip('module unavailable', allow_module_level=True)\n"
        "@pytest.mark.sage_oracle\n"
        "def test_uncollected_oracle():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    passing_file = tmp_path / "test_oracle_pass.py"
    passing_file.write_text(
        "import pytest\n@pytest.mark.sage_oracle\ndef test_oracle():\n    assert True\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(skipped_file, passing_file)
    assert completed.returncode == 1
    assert "collection skipped while oracle parity was required" in completed.stdout


def test_each_required_oracle_needs_a_selected_marked_test(tmp_path: Path) -> None:
    test_file = tmp_path / "test_only_sage.py"
    test_file.write_text(
        "import pytest\n@pytest.mark.sage_oracle\ndef test_sage():\n    assert True\n",
        encoding="utf-8",
    )
    completed = _run_required_oracles(test_file, required=("sage", "gap"))
    assert completed.returncode == 4
    assert "no selected test is marked gap_oracle" in completed.stderr
