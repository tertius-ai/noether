# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Independent Sage parity for exterior signs, contraction, and induced maps."""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

import pytest
from validation.harness.cas import resolve_binary

from noether import Q
from noether.commalg import PolynomialRing
from noether.homalg import FreeModule, ModuleElement, ModuleHomomorphism
from noether.linalg import Matrix, Vector


def _sage_binary() -> str | None:
    if os.environ.get("NOETHER_VALIDATE_SAGE", "") in ("", "0", "false", "False"):
        return None
    return resolve_binary("sage")


SAGE = _sage_binary()
pytestmark = [
    pytest.mark.sage_oracle,
    pytest.mark.skipif(SAGE is None, reason="needs NOETHER_VALIDATE_SAGE=1 and a Sage binary"),
]


def _run_sage_oracle() -> dict[str, object]:
    script = r"""
from sage.all import ExteriorAlgebra, QQ, matrix
import json

E = ExteriorAlgebra(QQ, names=('e0', 'e1', 'e2'))
e0, e1, e2 = E.gens()

def encode(value):
    answer = []
    for basis, coefficient in value.monomial_coefficients().items():
        answer.append({
            'basis': list(basis),
            'coefficient': [int(coefficient.numerator()), int(coefficient.denominator())],
        })
    return sorted(answer, key=lambda term: term['basis'])

linear_map = matrix(QQ, [[1, 1, 0], [0, 1, 0], [0, 0, 2]])
induced = E.lift_morphism(linear_map, ['f0', 'f1', 'f2'])
print(json.dumps({
    'reverse': encode(e1 * e0),
    'square': encode(e0 * e0),
    'contract_e0': encode((e0 * e1).interior_product(e0)),
    'contract_e1': encode((e0 * e1).interior_product(e1)),
    'induced': [encode(induced(value)) for value in (e0 * e1, e0 * e2, e1 * e2)],
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(SAGE), "-python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _encode_noether(value) -> list[dict[str, object]]:
    answer = []
    for basis, coefficient in value.terms:
        constant = Fraction(coefficient.constant_coefficient())
        answer.append(
            {
                "basis": basis,
                "coefficient": [constant.numerator, constant.denominator],
            }
        )
    return answer


def test_exterior_algebra_agrees_with_sage() -> None:
    sage = _run_sage_oracle()
    ring = PolynomialRing(Q, ["t"])
    module = FreeModule(ring, 3)
    exterior = module.exterior_algebra()
    e0, e1, e2 = [exterior.generator(position) for position in range(3)]
    dual_e0 = ModuleElement(module.dual(), Vector.from_values(ring, [1, 0, 0]))
    dual_e1 = ModuleElement(module.dual(), Vector.from_values(ring, [0, 1, 0]))
    linear_map = ModuleHomomorphism(
        module,
        module,
        Matrix.from_rows(ring, [[1, 1, 0], [0, 1, 0], [0, 0, 2]]),
    ).exterior_power(2)

    assert _encode_noether(e1 * e0) == sage["reverse"]
    assert _encode_noether(e0 * e0) == sage["square"]
    assert _encode_noether((e0 * e1).contract(dual_e0)) == sage["contract_e0"]
    assert _encode_noether((e0 * e1).contract(dual_e1)) == sage["contract_e1"]
    assert [
        _encode_noether(linear_map.apply_exterior(value)) for value in (e0 * e1, e0 * e2, e1 * e2)
    ] == sage["induced"]
