# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for the accepted forward RSK surface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from noether.combin import Tableau


def _sage_binary():
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    for candidate in ("sage", os.path.expanduser("~/miniforge3/envs/sage/bin/sage")):
        if shutil.which(candidate) or os.path.exists(candidate):
            return candidate
    return None


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a sage binary"),
]


def _run_sage(script: str):
    result = subprocess.run(
        [SAGE, "-python", "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sage failed: {result.stderr[-700:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "word",
    [[3, 1, 4, 1, 5], [2, 1, 1, 3, 2], [1, 2, 3, 4], [4, 3, 2, 1]],
)
def test_rsk_vs_sage(word):
    expected = _run_sage(
        f"""
from sage.all import RSK
import json
P, Q = RSK({word})
print(json.dumps({{"P": [[int(x) for x in row] for row in P],
                   "Q": [[int(x) for x in row] for row in Q]}}))
"""
    )
    insertion, recording = Tableau.rsk(word)

    assert [list(row) for row in insertion.rows] == expected["P"]
    assert [list(row) for row in recording.rows] == expected["Q"]
