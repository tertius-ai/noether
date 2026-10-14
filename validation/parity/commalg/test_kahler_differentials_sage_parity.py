# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Independent Sage parity for derivatives, Jacobians, and cotangent dimensions."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from noether import Q
from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def _sage_binary() -> str | None:
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    candidates = ("sage", str(Path.home() / "miniforge3/envs/sage/bin/sage"))
    return next((candidate for candidate in candidates if shutil.which(candidate)), None)


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a sage binary"),
]


@pytest.fixture(scope="module")
def sage_reference() -> dict[str, Any]:
    script = r"""
import json
from sage.all import GF, QQ, PolynomialRing, matrix

def q_terms(value):
    return sorted([
        [list(exponents), int(coefficient.numerator()), int(coefficient.denominator())]
        for exponents, coefficient in value.dict().items()
    ])

def fp_terms(value):
    return sorted([
        [list(exponents), int(coefficient)]
        for exponents, coefficient in value.dict().items()
    ])

def case(field, encode):
    ring = PolynomialRing(field, names=("x", "y"), order="degrevlex")
    x, y = ring.gens()
    relation = x**3 - y**2
    value = x**2 * y + 3 * x
    derivatives = [value.derivative(x), value.derivative(y)]
    points = [(0, 0), (1, 1)]
    jacobians = []
    dimensions = []
    for point in points:
        row = [relation.derivative(variable)(*point) for variable in (x, y)]
        jacobians.append([int(entry) if field.is_finite() else [int(entry), 1] for entry in row])
        dimensions.append(int(2 - matrix(field, [row]).rank()))
    return {
        "derivatives": [encode(derivative) for derivative in derivatives],
        "jacobians": jacobians,
        "cotangent_dimensions": dimensions,
    }

print(json.dumps({"Q": case(QQ, q_terms), "F5": case(GF(5), fp_terms)}, sort_keys=True))
"""
    result = subprocess.run(
        [SAGE, "-python", "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Sage Kähler parity failed: {result.stderr[-1200:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _q_terms(value: Any) -> list[list[Any]]:
    return sorted(
        [
            [exponents, coefficient.numerator, coefficient.denominator]
            for exponents, coefficient in value.lift.terms()
        ]
    )


def _fp_terms(value: Any) -> list[list[Any]]:
    return sorted([[exponents, int(coefficient)] for exponents, coefficient in value.lift.terms()])


def test_rational_derivatives_jacobians_and_cotangent_dimensions_match_sage(
    sage_reference: dict[str, Any],
) -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    algebra = ring.quotient(ring.ideal([x**3 - y**2]))
    differential = algebra.kahler_differentials().differentiate_polynomial(x**2 * y + 3 * x)
    coordinates = differential.coordinates.coordinates.to_values()
    fibers = [algebra.cotangent_space_at(point) for point in ([0, 0], [1, 1])]

    assert [_q_terms(value) for value in coordinates] == sage_reference["Q"]["derivatives"]
    assert [
        [[entry.numerator, entry.denominator] for entry in fiber.evaluated_jacobian.to_rows()[0]]
        for fiber in fibers
    ] == sage_reference["Q"]["jacobians"]
    assert [fiber.embedding_dimension for fiber in fibers] == sage_reference["Q"][
        "cotangent_dimensions"
    ]


def test_prime_field_derivatives_jacobians_and_cotangent_dimensions_match_sage(
    sage_reference: dict[str, Any],
) -> None:
    field = FiniteField.prime(5)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    algebra = ring.quotient(ring.ideal([x**3 - y**2]))
    differential = algebra.kahler_differentials().differentiate_polynomial(x**2 * y + 3 * x)
    coordinates = differential.coordinates.coordinates.to_values()
    zero, one = field.zero(), field.one()
    fibers = [algebra.cotangent_space_at(point) for point in ([zero, zero], [one, one])]

    assert [_fp_terms(value) for value in coordinates] == sage_reference["F5"]["derivatives"]
    assert [
        [int(entry) for entry in fiber.evaluated_jacobian.to_rows()[0]] for fiber in fibers
    ] == sage_reference["F5"]["jacobians"]
    assert [fiber.embedding_dimension for fiber in fibers] == sage_reference["F5"][
        "cotangent_dimensions"
    ]
