# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for Noether's finite-poset relation and symmetries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

ne = pytest.importorskip("noether.combin")


def _sage_binary():
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    for cand in ("sage", os.path.expanduser("~/miniforge3/envs/sage/bin/sage")):
        if shutil.which(cand) or os.path.exists(cand):
            return cand
    return None


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a sage binary"),
]


def _run_sage(script: str):
    r = subprocess.run([SAGE, "-python", "-c", script], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"sage failed: {r.stderr[-700:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# (n, covers) — lattices and a non-lattice.
_CORPUS = [
    (4, [[0, 1], [0, 2], [1, 3], [2, 3]]),  # B₂
    (3, [[0, 1], [1, 2]]),  # 3-chain
    (6, [[0, 1], [0, 2], [1, 3], [1, 4], [2, 4], [3, 5], [4, 5]]),  # divisor lattice of 12
    (5, [[0, 1], [1, 2], [2, 4], [0, 3], [3, 4]]),  # N₅ pentagon (non-modular)
    (3, [[0, 1], [0, 2]]),  # not a lattice (1,2 have no join)
]


def _sage_poset(n, covers):
    script = f"""
from sage.all import Poset
import json
n = {n}; P = Poset((list(range(n)), {covers}), cover_relations=True)
leq = [[1 if P.le(a, b) else 0 for b in range(n)] for a in range(n)]
out = {{
  "leq": leq,
  "covers": sorted([[int(a), int(b)] for a, b in P.cover_relations_iterator()]),
  "automorphism_order": int(P.hasse_diagram().automorphism_group().cardinality()),
}}
print(json.dumps(out))
"""
    return _run_sage(script)


@pytest.mark.parametrize("n,covers", _CORPUS, ids=lambda x: str(x))
def test_poset_invariants_vs_sage(n, covers):
    if isinstance(n, list):
        return
    P = ne.Poset(n, covers)
    s = _sage_poset(n, covers)
    assert [[1 if P.leq(a, b) else 0 for b in range(n)] for a in range(n)] == s["leq"]
    assert [edge[:2] for edge in P.covers.weighted_edges] == [tuple(edge) for edge in s["covers"]]
    assert P.automorphism_group().order == s["automorphism_order"]
