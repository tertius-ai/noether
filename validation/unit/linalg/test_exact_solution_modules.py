# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact affine solution modules and replayable inconsistency evidence."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from noether import Z
from noether.commalg import ResidueRing
from noether.linalg import (
    CongruenceInconsistencyCertificate,
    CongruenceSolutionModule,
    CongruenceSystemResult,
    IntegerInconsistencyCertificate,
    IntegerLinearSystemResult,
    IntegerSolutionModule,
    Matrix,
    Vector,
)


def test_integer_solution_lattice_and_obstruction_are_replayable() -> None:
    operator = Matrix.from_rows(Z, [[2, 4, 0], [0, 0, 3]])
    rhs = Vector.from_values(Z, [6, 9])
    result = operator.solve_exact(rhs)

    assert isinstance(result, IntegerLinearSystemResult)
    assert result.is_consistent
    assert isinstance(result.solutions, IntegerSolutionModule)
    assert result.obstruction is None
    solution = result.solutions
    assert solution is not None
    assert operator @ solution.particular_solution == rhs
    assert operator @ solution.homogeneous_generators == Matrix.zeros(Z, (2, 1))
    assert solution.homogeneous_rank == 1
    result.verify()
    IntegerLinearSystemResult.from_json(result.to_json()).verify()

    payload = json.loads(result.to_json())
    payload["outcome"]["solution"]["particular_solution"] = json.loads(Vector.zeros(Z, 3).to_json())
    with pytest.raises(ValueError):
        IntegerLinearSystemResult.from_json(json.dumps(payload))

    impossible = Matrix.from_rows(Z, [[6]]).solve_exact(Vector.from_values(Z, [9]))
    assert not impossible.is_consistent
    assert isinstance(impossible.obstruction, IntegerInconsistencyCertificate)
    assert impossible.solutions is None
    certificate = impossible.obstruction
    assert certificate is not None
    assert certificate.left_gcd == 6
    assert abs(certificate.combined_rhs) == 9
    certificate.verify()


@given(
    modulus=st.integers(min_value=2, max_value=15),
    coefficient=st.integers(min_value=-20, max_value=20),
    rhs_value=st.integers(min_value=-20, max_value=20),
)
def test_rank_one_composite_congruences_match_complete_enumeration(
    modulus: int, coefficient: int, rhs_value: int
) -> None:
    ring = ResidueRing(modulus)
    operator = Matrix.from_rows(ring, [[coefficient]])
    rhs = Vector.from_values(ring, [rhs_value])
    result = operator.solve_exact(rhs)
    expected = {
        value for value in range(modulus) if coefficient * value % modulus == rhs_value % modulus
    }

    assert isinstance(result, CongruenceSystemResult)
    assert result.is_consistent == bool(expected)
    result.verify()
    if not expected:
        assert isinstance(result.obstruction, CongruenceInconsistencyCertificate)
        assert result.solutions is None
        assert result.obstruction is not None
        result.obstruction.verify()
        return

    assert isinstance(result.solutions, CongruenceSolutionModule)
    assert result.obstruction is None
    solution = result.solutions
    assert solution is not None
    particular = int(solution.particular_solution[0])
    generators = [int(row) for row in solution.homogeneous_generators.to_rows()[0]]
    generated = {particular}
    for generator in generators:
        generated = {
            (value + multiple * generator) % modulus
            for value in generated
            for multiple in range(modulus)
        }
    assert generated == expected
    restored = CongruenceSystemResult.from_json(result.to_json())
    restored.verify()
