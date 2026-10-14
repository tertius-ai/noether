# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact short-vector parity between Noether and SageMath.

The corpus covers root lattices, diagonal forms, a dense form, and E8. Sage's
``IntegralLattice.short_vectors`` is the independent bounded-enumeration
oracle. Noether and Sage may choose opposite representatives of a ``±`` pair,
so both outputs are canonicalized before exact set comparison.

CVP has an exhaustive exact brute-force differential corpus in the Rust
kernel; Sage does not expose a finite all-closest-vectors result with a stated
ordering/completeness contract, so this file does not turn its iterator into a
stronger oracle than it is.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from validation.backends.sage import sage_binary

from noether import Z
from noether.linalg import Lattice, Matrix

SAGE = sage_binary() if os.environ.get("NOETHER_VALIDATE_SAGE") == "1" else None
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a Sage binary"),
]


CASES = [
    {"name": "a2", "gram": [[2, -1], [-1, 2]], "bound": 6},
    {
        "name": "a3",
        "gram": [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
        "bound": 4,
    },
    {
        "name": "diagonal_1_2_3_5",
        "gram": [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 5]],
        "bound": 6,
    },
    {
        "name": "dense_rank_3",
        "gram": [[4, 1, 1], [1, 5, 2], [1, 2, 6]],
        "bound": 12,
    },
    {
        "name": "e8",
        "gram": [
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 2, -1, 0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0, 0, 0, 0],
            [0, 0, -1, 2, -1, 0, 0, 0],
            [0, 0, 0, -1, 2, -1, 0, -1],
            [0, 0, 0, 0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, 0],
            [0, 0, 0, 0, -1, 0, 0, 2],
        ],
        "bound": 4,
    },
]


_SAGE_SCRIPT = r"""
import json, sys
from sage.all import IntegralLattice, Matrix, ZZ

def canonical(v):
    v = tuple(int(x) for x in v)
    for x in v:
        if x:
            return tuple(-y for y in v) if x < 0 else v
    return v

out = {}
for case in json.loads(sys.argv[1]):
    lattice = IntegralLattice(Matrix(ZZ, case["gram"]))
    shells = lattice.short_vectors(case["bound"] + 1, up_to_sign_flag=True)
    vectors = sorted({canonical(v) for shell in shells[1:] for v in shell})
    out[case["name"]] = [list(v) for v in vectors]
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def sage_short_vectors():
    result = subprocess.run(
        [SAGE, "-python", "-c", _SAGE_SCRIPT, json.dumps(CASES)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Sage lattice oracle failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_bounded_short_vectors_match_sage(case, sage_short_vectors):
    lattice = Lattice(Matrix.from_rows(Z, case["gram"]))
    actual = lattice.shortest_vectors(bound=case["bound"])
    expected = sage_short_vectors[case["name"]]

    assert {tuple(vector) for vector in actual.vectors} == {tuple(vector) for vector in expected}
    assert actual.pair_count == len(expected)
    assert actual.signed_count == 2 * len(expected)
    actual.verify(lattice)
