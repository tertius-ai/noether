# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for exact complete-intersection characteristic numbers.

Sage owns the oracle arithmetic here: its multivariate quotient rings reduce
the ambient Chow classes, its formal power-series ring supplies the Todd
series, and its generalized binomial function evaluates the Koszul Euler
characteristic. The corpus covers projective space and multiprojective
ambients, negative twists, and codimension greater than one.
"""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import pytest

from noether.commalg import ChowRing, ProjectiveCompleteIntersection
from validation.backends.sage._run import sage_binary

SAGE = sage_binary() if os.environ.get("NOETHER_VALIDATE_SAGE") == "1" else None
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a Sage binary"),
]

CASES = [
    {"dimensions": [2], "degrees": [[3]], "twists": [[-2], [0], [1], [5]]},
    {
        "dimensions": [3],
        "degrees": [[2], [2]],
        "twists": [[-3], [0], [2], [6]],
    },
    {
        "dimensions": [1, 2],
        "degrees": [[1, 1], [1, 2]],
        "twists": [[0, 0], [2, -1], [-2, 3]],
    },
    {
        "dimensions": [2, 2],
        "degrees": [[2, 3]],
        "twists": [[0, 0], [1, 2], [-2, 1]],
    },
]


def _terms(value) -> list[list[object]]:
    return [[exponent, str(Fraction(coefficient))] for exponent, coefficient in value.terms()]


def _noether_payload(case: dict[str, object]) -> dict[str, object]:
    ring = ChowRing(case["dimensions"])
    variety = ProjectiveCompleteIntersection.assuming_smooth(ring, case["degrees"])
    return {
        "fundamental": _terms(variety.fundamental_class()),
        "normal_chern": _terms(variety.normal_total_chern_class()),
        "tangent_chern": _terms(variety.tangent_total_chern_class()),
        "todd": _terms(variety.todd_class()),
        "euler": [
            variety.line_bundle_euler_characteristic(twist).value for twist in case["twists"]
        ],
    }


def _sage_payloads() -> list[dict[str, object]]:
    assert SAGE is not None
    cases = json.dumps(CASES)
    script = f"""
from sage.all import *
import builtins
import json

cases = json.loads({cases!r})

def payload(case):
    dimensions = case["dimensions"]
    degrees = case["degrees"]
    names = ["h%d" % i for i in range(len(dimensions))]
    # Supplying the variable count keeps the one-factor case multivariate too,
    # so exponent keys have one uniform tuple representation.
    polynomial_ring = PolynomialRing(QQ, len(names), names=names)
    polynomial_generators = polynomial_ring.gens()
    chow_ring = polynomial_ring.quotient([
        polynomial_generators[i] ** (dimensions[i] + 1)
        for i in range(len(dimensions))
    ])
    generators = chow_ring.gens()
    ambient_dimension = builtins.sum(dimensions)
    variety_dimension = ambient_dimension - len(degrees)

    def truncate(value, maximum):
        lifted = value.lift()
        return chow_ring(builtins.sum(
            coefficient * prod(
                polynomial_generators[i] ** int(exponent[i])
                for i in range(len(dimensions))
            )
            for exponent, coefficient in lifted.dict().items()
            if builtins.sum(exponent) <= maximum
        ))

    def normalized_terms(value):
        return sorted([
            ([int(entry) for entry in exponent], str(coefficient))
            for exponent, coefficient in value.lift().dict().items()
            if coefficient != 0
        ])

    divisors = [
        builtins.sum(ZZ(degree[i]) * generators[i] for i in range(len(dimensions)))
        for degree in degrees
    ]
    fundamental = prod(divisors, chow_ring.one())
    normal_chern = truncate(prod([1 + divisor for divisor in divisors], chow_ring.one()), variety_dimension)

    tangent_chern = chow_ring.one()
    for i, dimension in enumerate(dimensions):
        tangent_chern = truncate(tangent_chern * (1 + generators[i]) ** (dimension + 1), variety_dimension)
    for divisor in divisors:
        tangent_chern = truncate(
            tangent_chern * builtins.sum([(-divisor) ** power for power in range(variety_dimension + 1)]),
            variety_dimension,
        )

    series_ring = PowerSeriesRing(QQ, "t", default_prec=variety_dimension + 2)
    t = series_ring.gen()
    todd_series = t / (1 - (-t).exp())
    todd = chow_ring.one()
    for i, dimension in enumerate(dimensions):
        factor = builtins.sum(todd_series[power] * generators[i] ** power for power in range(variety_dimension + 1))
        todd = truncate(todd * factor ** (dimension + 1), variety_dimension)
    for divisor in divisors:
        factor = builtins.sum(todd_series[power] * divisor ** power for power in range(variety_dimension + 1))
        positive = truncate(factor - 1, variety_dimension)
        inverse = builtins.sum([(-positive) ** power for power in range(variety_dimension + 1)], chow_ring.zero())
        todd = truncate(todd * inverse, variety_dimension)

    euler = []
    for twist in case["twists"]:
        answer = ZZ.zero()
        for mask in range(1 << len(degrees)):
            shifted = list(twist)
            sign = 1
            for equation, degree in enumerate(degrees):
                if mask & (1 << equation):
                    sign = -sign
                    shifted = [shifted[i] - degree[i] for i in range(len(dimensions))]
            answer += sign * prod([
                binomial(shifted[i] + dimensions[i], dimensions[i])
                for i in range(len(dimensions))
            ])
        euler.append(int(answer))

    return {{
        "fundamental": normalized_terms(fundamental),
        "normal_chern": normalized_terms(normal_chern),
        "tangent_chern": normalized_terms(tangent_chern),
        "todd": normalized_terms(todd),
        "euler": euler,
    }}

print(json.dumps([payload(case) for case in cases], sort_keys=True))
"""
    completed = subprocess.run(
        [SAGE, "-python", "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Sage failed:\n{completed.stdout[-1000:]}\n{completed.stderr[-2000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_complete_intersection_characteristic_data_matches_sage() -> None:
    assert [_noether_payload(case) for case in CASES] == _sage_payloads()
