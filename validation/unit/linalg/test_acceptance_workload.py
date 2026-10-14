# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The bounded exact linear-algebra release calculation."""

import pytest

from noether import Assumption, Q, UnsupportedOperationError, Z
from noether.commalg import PolynomialRing, ResidueRing
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import (
    FiniteField,
    NumberField,
    PadicField,
    PadicRing,
    QadicField,
    QadicRing,
)


def test_one_matrix_surface_covers_the_exact_release_contexts() -> None:
    for field in [
        FiniteField.prime(2),
        FiniteField.extension(3, 2),
        FiniteField.extension(2, 4),
        FiniteField.extension(5, 2),
    ]:
        a = field.generator()
        matrix = Matrix.from_rows(field, [[a, 1], [1, 0]])
        right_hand_side = Vector.from_values(field, [1, a])
        solution = matrix.solve(right_hand_side)

        assert matrix.det() == -field.one()
        assert matrix.rank() == 2
        assert matrix @ solution.particular == right_hand_side
        matrix.pluq().verify()
        solution.verify()

    rational_polynomials = PolynomialRing(Q, ["x"])
    x = rational_polynomials.generator(0)
    for field in [
        NumberField(x**2 + 1, name="i"),
        NumberField(x**3 - x - 1, name="a"),
    ]:
        a = field.generator
        matrix = Matrix.from_rows(field, [[a, 1], [1, a + 1]])
        right_hand_side = Vector.from_values(field, [1, 0])
        solution = matrix.solve(right_hand_side)
        assert matrix @ solution.particular == right_hand_side
        matrix.pluq().verify()
        solution.verify()

    residue_ring = ResidueRing(25)
    residue_matrix = Matrix.from_rows(residue_ring, [[5, 10], [0, 5]])
    assert residue_matrix.howell_normal_form().to_rows() == [
        [residue_ring(5), residue_ring(0)],
        [residue_ring(0), residue_ring(5)],
    ]

    residue_field = FiniteField.extension(3, 2)
    for field, unit in [
        (PadicField(3, absolute_precision=8), None),
        (QadicField(residue_field, absolute_precision=8), "generator"),
    ]:
        unit = field.one() if unit is None else field.generator() + field.one()
        uniformizer = field(3)
        matrix = Matrix.from_rows(
            field,
            [[uniformizer, field.zero()], [field.zero(), unit]],
        )
        right_hand_side = Vector.from_values(field, [uniformizer, unit])
        assert matrix @ Vector.from_values(field, [1, 1]) == right_hand_side
        assert matrix.det() == uniformizer * unit
        with pytest.raises(UnsupportedOperationError):
            matrix.rank()
        with pytest.raises(UnsupportedOperationError):
            matrix.solve(right_hand_side)
        with pytest.raises(UnsupportedOperationError):
            matrix.pluq()
        characteristic_polynomial = matrix.charpoly()
        assert [characteristic_polynomial.coefficient([degree]) for degree in range(3)] == [
            uniformizer * unit,
            -(uniformizer + unit),
            field.one(),
        ]

    for ring, unit in [
        (PadicRing(3, absolute_precision=6), None),
        (QadicRing(residue_field, absolute_precision=6), "generator"),
    ]:
        unit = ring.one() if unit is None else ring.generator() + ring.one()
        uniformizer = ring(3)
        matrix = Matrix.from_rows(
            ring,
            [[uniformizer, ring.one()], [ring.zero(), unit]],
        )
        vector = Vector.from_values(ring, [1, 2])
        assert matrix @ vector == Vector.from_values(ring, [uniformizer + ring(2), unit * ring(2)])
        characteristic_polynomial = matrix.charpoly()
        assert [characteristic_polynomial.coefficient([degree]) for degree in range(3)] == [
            uniformizer * unit,
            -(uniformizer + unit),
            ring.one(),
        ]


def test_sparse_and_dense_operators_share_exact_context_identity() -> None:
    sparse = SparseMatrix.from_triplets(
        Q,
        (3, 3),
        [(0, 0, 2), (0, 2, -1), (1, 1, 3), (2, 0, 4)],
    )
    vector = Vector.from_values(Q, [1, 2, 3])

    assert sparse.apply(vector).to_values() == [-1, 6, 4]
    assert sparse.to_dense() @ vector == sparse.apply(vector)
    assert sparse.transpose().transpose() == sparse
    assert SparseMatrix.from_json(sparse.to_json()) == sparse


def test_smith_default_transforms_and_smooth_evidence_are_distinct() -> None:
    matrix = Matrix.from_rows(Z, [[2, 4, 4], [6, 6, 12]])

    default = matrix.smith_normal_form()
    assert default.invariant_factors == [2, 6]
    assert default.validity.is_unconditional
    assert not hasattr(default, "left")
    default.verify()

    transformed = matrix.smith_normal_form_with_transforms()
    assert transformed.left @ matrix @ transformed.right == transformed.form
    assert transformed.smith == default
    transformed.verify()

    conditional = Matrix.from_rows(Z, [[2, 0, 0], [0, 6, 0]]).smith_normal_form(
        algorithm="smooth", smooth_bound=3
    )
    assert conditional.invariant_factors == [2, 6]
    assert conditional.validity.assumptions == [Assumption.torsion_order_is_smooth(3)]
    assert conditional.verification.checked_primes == [2, 3]
    conditional.verify()

    unconditional = Matrix.identity(Z, 2).smith_normal_form(algorithm="smooth", smooth_bound=2)
    assert unconditional.validity.is_unconditional
    unconditional.verify()
