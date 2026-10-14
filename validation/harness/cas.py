# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Locating CAS binaries for the comparison harness.

A CAS executable is resolved by, in order:

  1. a ``NOETHER_<CAS>_BIN`` environment variable (an explicit path),
  2. the ``PATH``,
  3. common conda / miniforge env install locations.

Step 3 is what lets the harness find a Sage installed into a conda
env that was never added to ``PATH`` — the usual case for a
miniforge SageMath. Magma, which is licensed and typically lives on a
separate machine, is found via step 1 or 2 there and is simply absent
(backend skipped) on hosts without it.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

# CAS id → the executable's name on PATH.
_BINARY_NAME = {
    "sage": "sage",
    "magma": "magma",
    "gap": "gap",
    "gp": "gp",
    "pari": "gp",
    "macaulay2": "M2",
}


def _conda_fallback(name: str) -> str | None:
    """Search common conda / miniforge env layouts for `name`."""
    home = Path.home()
    patterns = [
        home / "miniforge3" / "envs" / "*" / "bin" / name,
        home / "miniconda3" / "envs" / "*" / "bin" / name,
        home / "anaconda3" / "envs" / "*" / "bin" / name,
        home / "mambaforge" / "envs" / "*" / "bin" / name,
        home / "miniforge3" / "bin" / name,
        home / "miniconda3" / "bin" / name,
    ]
    for pattern in patterns:
        for hit in sorted(glob.glob(str(pattern))):
            if os.access(hit, os.X_OK):
                return hit
    return None


def resolve_binary(cas: str) -> str | None:
    """Return the path to the `cas` executable, or None if not found.

    `cas` is a harness backend id: ``sage``, ``magma``, ``gap``, ``gp``,
    ``pari``, ``macaulay2``. The ``gp`` and ``pari`` ids find the same
    executable but honor ``NOETHER_GP_BIN`` and ``NOETHER_PARI_BIN``,
    respectively.
    """
    name = _BINARY_NAME.get(cas, cas)
    env = os.environ.get(f"NOETHER_{cas.upper()}_BIN", "")
    if env and os.access(env, os.X_OK):
        return env
    found = shutil.which(name)
    if found:
        return found
    return _conda_fallback(name)
