# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for the accepted public finite-matroid operations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from noether.combin import Matroid


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


def _normalized(values):
    return sorted(sorted(int(element) for element in value) for value in values)


_CORPUS = [
    (3, [[0, 1], [0, 2], [1, 2]]),
    (4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
    (4, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    (4, [[0, 2], [0, 3], [1, 2], [1, 3]]),
]


def _sage_matroid(order, bases):
    return _run_sage(
        f"""
from sage.all import Matroid
import json
M = Matroid(groundset=list(range({order})), bases={bases})
T = M.tutte_polynomial(); x, y = T.parent().gens(); r = M.rank()
out = {{
  "rank_full": int(r),
  "circuits": sorted(sorted(int(e) for e in c) for c in M.circuits()),
  "tutte": [[int(T.coefficient({{x: i, y: j}})) for j in range({order} - r + 1)] for i in range(r + 1)],
  "delete_last": sorted(sorted(int(e) for e in b) for b in M.delete([{order} - 1]).bases()),
  "contract_first": sorted(sorted(int(e) for e in b) for b in M.contract([0]).bases()),
}}
print(json.dumps(out))
"""
    )


@pytest.mark.parametrize("order,bases", _CORPUS)
def test_matroid_operations_vs_sage(order, bases):
    reference = _sage_matroid(order, bases)
    value = Matroid.from_bases(list(range(order)), bases)

    assert value.rank(list(range(order))) == reference["rank_full"]
    assert all(value.rank(basis) == len(basis) for basis in bases)
    assert _normalized(value.circuits) == reference["circuits"]
    tutte = value.tutte_polynomial()
    assert [
        [tutte.coefficient([i, j]) for j in range(order - value.rank(value.ground_set) + 1)]
        for i in range(value.rank(value.ground_set) + 1)
    ] == reference["tutte"]
    assert _normalized(value.delete(order - 1).bases) == reference["delete_last"]
    assert _normalized(value.contract(0).bases) == reference["contract_first"]
