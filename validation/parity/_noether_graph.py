# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Noether's answer for the graph-core parity harness (`noether` backend).

The system under test for the graph-core lane. This adapter drives the crate's
`parity_oracle` example binary as a subprocess.

`parity_oracle` (`crates/graph-core/examples/parity_oracle.rs`) reads a
JSON graph spec on stdin and writes the harness's comparable
invariants (component count + labels, BFS/DFS orders + depths,
shortest-path distances) on stdout. Keeping the oracle a plain
`examples/` binary keeps graph-core a zero-dependency pure-Rust kernel.

The binary is built on first use (`cargo build --release -p graph-core
--example parity_oracle`); set `GRAPH_CORE_ORACLE_BIN` to point at a
prebuilt binary to skip the build.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

# crates/graph-core lives at <repo>/crates/graph-core; this file is at
# <repo>/validation/parity/_noether_graph.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CRATE_DIR = _REPO_ROOT / "crates" / "graph-core"
_ORACLE_REL = Path("target") / "release" / "examples" / "parity_oracle"


@lru_cache(maxsize=1)
def oracle_binary() -> str | None:
    """Locate (building if needed) the `parity_oracle` binary.

    Returns the path, or `None` if it cannot be built — in which case
    the `noether` backend is simply absent from the scorecard and the
    harness reports it honestly rather than faking a result.
    """
    override = os.environ.get("GRAPH_CORE_ORACLE_BIN")
    if override and Path(override).is_file():
        return override

    built = _REPO_ROOT / _ORACLE_REL
    if built.is_file():
        return str(built)

    # Build it once.
    try:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "graph-core",
                "--example",
                "parity_oracle",
            ],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return str(built) if built.is_file() else None


def query(spec: dict) -> dict:
    """Run the `parity_oracle` on one graph spec, return its JSON dict.

    `spec` is the oracle's request schema (see `parity_oracle.rs`).
    Raises `RuntimeError` if the oracle is unavailable or fails — the
    harness catches it and records `ERROR`.
    """
    binary = oracle_binary()
    if binary is None:
        raise RuntimeError("graph-core parity_oracle could not be built")
    proc = subprocess.run(
        [binary],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"parity_oracle failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)
