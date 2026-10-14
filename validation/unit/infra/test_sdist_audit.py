# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The source-distribution audit is total on realistically large manifests."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECK = ROOT / "scripts" / "check-sdist.sh"


def _archive(path: Path, *, cargo_lock: bool = True) -> None:
    root = "tertius_noether-0.3.0"
    entries = {
        "Cargo.toml": b"[workspace]\nmembers = []\n",
        "pyproject.toml": b"[project]\nname = 'tertius-noether'\nversion = '0.3.0'\n",
        "LICENSE": b"MIT\n",
        "README.md": b"# Noether\n",
    }
    if cargo_lock:
        entries["Cargo.lock"] = b"version = 4\n"
    # Exceed a typical pipe buffer so an early-exiting grep would SIGPIPE its
    # producer under `set -o pipefail`, the original false-negative boundary.
    entries.update(
        {f"python/noether/generated_{index:04d}.py": b"VALUE = 1\n" for index in range(1_500)}
    )
    with tarfile.open(path, "w:gz") as archive:
        for relative, data in entries.items():
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))


def test_large_clean_sdist_manifest_passes(tmp_path: Path) -> None:
    path = tmp_path / "tertius_noether-0.3.0.tar.gz"
    _archive(path)

    completed = subprocess.run([CHECK, path], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "SDIST GATE PASSED" in completed.stdout


def test_missing_lockfile_still_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "tertius_noether-0.3.0.tar.gz"
    _archive(path, cargo_lock=False)

    completed = subprocess.run([CHECK, path], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "source distribution is missing Cargo.lock" in completed.stderr
