# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage parity for the canonical symmetric-function owner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction

import pytest

from noether.combin import SymmetricFunction


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


def _terms(value: SymmetricFunction):
    return {tuple(partition.parts): Fraction(coefficient) for partition, coefficient in value.terms}


def _sage_terms(source: str, target: str, partition: list[int]):
    return _run_sage(
        f"""
from sage.all import QQ, SymmetricFunctions
import json
Sym = SymmetricFunctions(QQ)
bases = {{
    "monomial": Sym.monomial(),
    "power_sum": Sym.powersum(),
    "elementary": Sym.elementary(),
    "complete_homogeneous": Sym.homogeneous(),
    "schur": Sym.schur(),
}}
value = bases[{source!r}][{partition!r}]
converted = bases[{target!r}](value)
terms = sorted(
    ([int(x) for x in part], int(coeff.numerator()), int(coeff.denominator()))
    for part, coeff in converted.monomial_coefficients().items()
)
print(json.dumps(terms))
"""
    )


@pytest.mark.parametrize(
    ("source", "target", "partition"),
    [
        ("schur", "monomial", [2, 1]),
        ("schur", "power_sum", [3, 1]),
        ("monomial", "schur", [2, 1, 1]),
        ("elementary", "complete_homogeneous", [2, 1]),
        ("complete_homogeneous", "elementary", [3, 1]),
        ("power_sum", "monomial", [2, 2]),
    ],
)
def test_basis_conversion_vs_sage(source, target, partition):
    expected = {
        tuple(parts): Fraction(numerator, denominator)
        for parts, numerator, denominator in _sage_terms(source, target, partition)
    }
    value = SymmetricFunction(source, [(partition, 1)]).convert_basis(target)

    assert _terms(value) == expected


def test_schur_product_vs_sage():
    expected = _run_sage(
        """
from sage.all import QQ, SymmetricFunctions
import json
s = SymmetricFunctions(QQ).schur()
value = s([2, 1]) * s([2, 1])
terms = sorted(
    ([int(x) for x in part], int(coeff))
    for part, coeff in value.monomial_coefficients().items()
)
print(json.dumps(terms))
"""
    )
    value = SymmetricFunction("schur", [([2, 1], 1)])
    product = value * value

    assert _terms(product) == {
        tuple(parts): Fraction(coefficient) for parts, coefficient in expected
    }
