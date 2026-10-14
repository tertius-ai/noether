# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Magma parity for the full arXiv:1906.02261 surface computation."""

from __future__ import annotations

import os
import subprocess

import pytest

from noether.commalg import CyclicHypersurfaceCover, ProjectiveSpace
from noether.numthy import FiniteField
from validation.bench._magma import DEFAULT_MAGMA_SEED, magma_command, magma_environment
from validation.harness.cas import resolve_binary

PAPER_SEXTIC_SUPPORT = [
    [6, 0, 0],
    [5, 0, 1],
    [4, 2, 0],
    [4, 0, 2],
    [3, 1, 2],
    [2, 2, 2],
    [1, 5, 0],
    [1, 4, 1],
    [1, 0, 5],
    [0, 6, 0],
    [0, 3, 3],
    [0, 2, 4],
    [0, 1, 5],
    [0, 0, 6],
]

MAGMA_REQUESTED = os.environ.get("NOETHER_VALIDATE_MAGMA", "") not in (
    "",
    "0",
    "false",
    "False",
)
MAGMA = resolve_binary("magma") if MAGMA_REQUESTED else None
if MAGMA_REQUESTED and MAGMA is None:
    raise RuntimeError("NOETHER_VALIDATE_MAGMA=1 requires Magma on PATH or NOETHER_MAGMA_BIN")

pytestmark = [
    pytest.mark.magma_oracle,
    pytest.mark.skipif(
        not MAGMA_REQUESTED,
        reason="needs NOETHER_VALIDATE_MAGMA=1 and a Magma executable",
    ),
    pytest.mark.very_slow,
]

COEFFICIENTS_BEGIN = "NOETHER_COEFFICIENTS_BEGIN"
COEFFICIENT_PREFIX = "NOETHER_COEFFICIENT|"
COEFFICIENTS_END = "NOETHER_COEFFICIENTS_END"


def noether_coefficients() -> tuple[int, ...]:
    field = FiniteField.prime(31)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    branch = plane.coordinate_ring.from_terms(
        (exponent, field(-1)) for exponent in PAPER_SEXTIC_SUPPORT
    )
    cover = CyclicHypersurfaceCover(plane, 2, branch)
    counts = cover.point_counts(list(range(1, 12)))
    frobenius = cover.surface_cohomology().frobenius_polynomial(counts)
    terms = {
        exponent[0]: int(coefficient) for exponent, coefficient in frobenius.polynomial.terms()
    }
    return tuple(terms.get(exponent, 0) for exponent in range(frobenius.degree + 1))


def _magma_script() -> str:
    return r"""
R<x,y,z> := PolynomialRing(Integers(),3);
f := -x^6 - x^5*z - x^4*y^2 - x^4*z^2 - x^3*y*z^2 - x^2*y^2*z^2
     - x*y^5 - x*y^4*z - x*z^5 - y^6 - y^3*z^3 - y^2*z^4
     - y*z^5 - z^6;
wp := WeilPolynomialOfDegree2K3Surface(f,31);
print "NOETHER_COEFFICIENTS_BEGIN";
for i in [0..Degree(wp)] do
    printf "NOETHER_COEFFICIENT|%o|%o\n", i, Integers()!Coefficient(wp,i);
end for;
print "NOETHER_COEFFICIENTS_END";
quit;
"""


def _parse_magma_coefficients(output: str) -> tuple[int, ...]:
    active = False
    ended = False
    coefficients: list[int] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == COEFFICIENTS_BEGIN:
            if active or ended:
                raise ValueError("Magma coefficient protocol began more than once")
            active = True
            continue
        if line == COEFFICIENTS_END:
            if not active:
                raise ValueError("Magma coefficient protocol ended before it began")
            active = False
            ended = True
            continue
        if not active:
            continue
        if not line.startswith(COEFFICIENT_PREFIX):
            raise ValueError(f"unexpected output inside Magma coefficient protocol: {line[:500]}")
        _, index_text, coefficient_text = line.split("|", 2)
        index = int(index_text)
        if index != len(coefficients):
            raise ValueError(
                f"Magma emitted coefficient {index}; expected coefficient {len(coefficients)}"
            )
        coefficients.append(int(coefficient_text))
    if active or not ended:
        raise ValueError("Magma coefficient protocol sentinels are incomplete")
    if not coefficients:
        raise ValueError("Magma coefficient protocol emitted no coefficients")
    return tuple(coefficients)


def magma_coefficients() -> tuple[int, ...]:
    assert MAGMA is not None
    completed = subprocess.run(
        magma_command(threads=1, seed=DEFAULT_MAGMA_SEED, binary=MAGMA),
        env=magma_environment(),
        input=_magma_script(),
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Magma failed:\n{completed.stdout[-1000:]}\n{completed.stderr[-1000:]}")
    try:
        return _parse_magma_coefficients(completed.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"Magma coefficient protocol failed: {error}\n{completed.stdout[-1000:]}"
        ) from error


def test_paper_surface_frobenius_polynomial_matches_magma() -> None:
    assert noether_coefficients() == magma_coefficients()
