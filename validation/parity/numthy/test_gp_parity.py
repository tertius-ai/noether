# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""PARI/GP parity for the accepted elementary number-theory surface."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from validation.harness.cas import resolve_binary

from noether.numthy import cyclotomic_polynomial, factor, is_prime

GP_REQUESTED = os.environ.get("NOETHER_VALIDATE_GP", "") not in (
    "",
    "0",
    "false",
    "False",
)
GP = resolve_binary("gp") if GP_REQUESTED else None
if GP_REQUESTED and GP is None:
    raise RuntimeError("NOETHER_VALIDATE_GP=1 requires `gp` or NOETHER_GP_BIN")

pytestmark = [
    pytest.mark.gp_oracle,
    pytest.mark.skipif(not GP_REQUESTED, reason="needs NOETHER_VALIDATE_GP=1 and a gp binary"),
]


def _run_gp(script: str) -> list[str]:
    result = subprocess.run(
        [GP, "-fq"],
        input=f"{script}\nquit\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gp failed: {result.stderr[-700:]}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "value",
    [
        -360,
        -1,
        1,
        2,
        561,
        65_537,
        3_215_031_751,
        2**32 * 3**7 * 101,
    ],
)
def test_factorization_matches_gp(value: int) -> None:
    lines = _run_gp(
        f'f=factor({value}); print(matsize(f)[1]); for(i=1,matsize(f)[1],print(f[i,1],":",f[i,2]))'
    )
    row_count = int(lines[0])
    gp_rows = [tuple(map(int, line.split(":"))) for line in lines[1:]]
    assert len(gp_rows) == row_count

    expected_unit = -1 if gp_rows and gp_rows[0][0] == -1 else 1
    expected_factors = gp_rows[1:] if expected_unit == -1 else gp_rows
    result = factor(value)
    assert result.unit == expected_unit
    assert result.factors == expected_factors
    assert result.verify(value)


@pytest.mark.parametrize(
    "value",
    [
        -7,
        0,
        1,
        2,
        3,
        4,
        561,
        65_537,
        2**61 - 1,
        (2**31 - 1) * (2**19 - 1),
    ],
)
def test_primality_matches_gp(value: int) -> None:
    reference = bool(int(_run_gp(f"print(isprime({value}))")[0]))
    assert is_prime(value) is reference


@pytest.mark.parametrize("index", [1, 2, 3, 4, 6, 8, 10, 12, 15, 30, 105])
def test_cyclotomic_polynomials_match_gp(index: int) -> None:
    reference = json.loads(_run_gp(f"print(Vec(polcyclo({index})))")[0])
    polynomial = cyclotomic_polynomial(index)
    degree = max(exponents[0] for exponents, _ in polynomial.terms())
    coefficients = [polynomial.coefficient([power]) for power in range(degree, -1, -1)]
    assert coefficients == reference
