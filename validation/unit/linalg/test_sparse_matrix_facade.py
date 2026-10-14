# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from fractions import Fraction

import pytest

from noether import ComplexBallField, ContextMismatchError, Q, RealBallField, Z
from noether.commalg import ResidueRing
from noether.linalg import LinearOperator, Matrix, SparseMatrix, Vector
from noether.numthy import FiniteField, PadicField, PadicRing, QadicField, QadicRing


def test_triplets_have_one_canonical_csr_identity() -> None:
    matrix = SparseMatrix.from_triplets(
        Z,
        (3, 4),
        [
            (2, 3, 7),
            (0, 1, 5),
            (0, 1, -2),
            (1, 2, 4),
            (1, 2, -4),
            (0, 0, 0),
        ],
    )

    assert matrix.type_id == "noether/linalg/SparseMatrix@1"
    assert matrix.shape == (3, 4)
    assert matrix.nnz == 2
    assert matrix.density == pytest.approx(1 / 6)
    assert matrix.csr == ([0, 1, 1, 2], [1, 3], [3, 7])
    assert matrix.row_offsets == [0, 1, 1, 2]
    assert matrix.column_indices == [1, 3]
    assert matrix[0, 1] == 3
    assert matrix[1, 1] == 0
    assert matrix[-1, -1] == 7
    assert isinstance(matrix, LinearOperator)

    dense = Matrix.from_rows(Z, [[0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 7]])
    assert matrix.to_dense() == dense
    assert SparseMatrix.from_dense(dense) == matrix
    restored = SparseMatrix.from_json(matrix.to_json())
    assert restored == matrix
    assert restored.csr == matrix.csr


def test_exact_sparse_arithmetic_and_operator_application() -> None:
    left = SparseMatrix.from_triplets(
        Z,
        (2, 3),
        [(0, 0, 1), (0, 2, 2), (1, 1, 3)],
    )
    right = SparseMatrix.from_csr(
        Z,
        (3, 2),
        [0, 1, 2, 3],
        [1, 0, 1],
        [4, 5, 6],
    )

    product = left @ right
    assert product.to_dense() == Matrix.from_rows(Z, [[0, 16], [15, 0]])
    vector = Vector.from_values(Z, [2, 3, 5])
    assert left.apply(vector) == Vector.from_values(Z, [12, 9])
    assert left.apply_block(right.to_dense()) == product.to_dense()
    assert left.transpose().transpose() == left
    assert left + left == 2 * left
    assert (left - left).nnz == 0


def test_supported_exact_contexts_preserve_coefficient_types() -> None:
    residue_ring = ResidueRing(25)
    finite_field = FiniteField.extension(3, 2)
    contexts_and_values = [
        (Q, Fraction(2, 3)),
        (residue_ring, residue_ring(7)),
        (finite_field, finite_field.generator()),
    ]

    for ring, value in contexts_and_values:
        matrix = SparseMatrix.from_triplets(ring, (1, 1), [(0, 0, value)])
        assert matrix.ring == ring
        assert matrix[0, 0] == value
        assert matrix.values == [value]
        assert SparseMatrix.from_json(matrix.to_json()) == matrix


def test_sparse_local_contexts_share_the_generic_csr_operations() -> None:
    residue_field = FiniteField.extension(3, 2)
    padic_ring = PadicRing(3, 6)
    padic_field = PadicField(3, 6)
    qadic_ring = QadicRing(residue_field, 6)
    qadic_field = QadicField(residue_field, 6)
    contexts_and_units = [
        (padic_ring, padic_ring(2)),
        (padic_field, padic_field.inverse(padic_field(2))),
        (qadic_ring, qadic_ring.generator()),
        (qadic_field, qadic_field.inverse(qadic_field.generator())),
    ]

    for ring, unit in contexts_and_units:
        sparse = SparseMatrix.from_triplets(
            ring,
            (2, 2),
            [(0, 0, unit), (0, 1, 1), (1, 1, unit)],
        )
        dense = sparse.to_dense()
        vector = Vector.from_values(ring, [1, unit])

        assert sparse.ring == ring
        assert sparse.values == [unit, ring.one(), unit]
        assert sparse.apply(vector) == dense @ vector
        assert sparse.apply_block(dense) == dense @ dense
        assert (sparse @ sparse).to_dense() == dense @ dense
        assert SparseMatrix.from_dense(dense) == sparse
        assert (
            SparseMatrix.from_csr(
                ring,
                sparse.shape,
                sparse.row_offsets,
                sparse.column_indices,
                sparse.values,
            )
            == sparse
        )
        assert SparseMatrix.from_json(sparse.to_json()) == sparse
        assert sparse.transpose().transpose() == sparse
        assert sparse + sparse == ring(2) * sparse
        assert SparseMatrix.zeros(ring, (2, 3)).shape == (2, 3)


def test_ball_matrices_project_sparb_through_the_python_facade() -> None:
    real_field = RealBallField(128)
    real = SparseMatrix.from_triplets(
        real_field,
        (2, 2),
        [(0, 0, 1), (0, 1, 2), (1, 1, 3)],
    )
    assert real.ring == real_field
    assert (real @ real).to_dense() == Matrix.from_rows(real_field, [[1, 8], [0, 9]])
    real_vector = Vector.from_values(real_field, [1, 2])
    assert real.apply(real_vector) == Vector.from_values(real_field, [5, 6])
    assert real_vector.dot(real_vector) == real_field(5)
    assert SparseMatrix.from_json(real.to_json()) == real

    complex_field = ComplexBallField(128)
    imaginary_unit = complex_field(0, 1)
    complex_matrix = SparseMatrix.from_triplets(
        complex_field,
        (2, 2),
        [(0, 0, 1), (0, 1, imaginary_unit), (1, 1, 1)],
    )
    expected = Matrix.from_rows(
        complex_field,
        [[1, imaginary_unit + imaginary_unit], [0, 1]],
    )
    assert (complex_matrix @ complex_matrix).to_dense() == expected
    assert Matrix.from_rows(complex_field, [[1, imaginary_unit]]).ring == complex_field


def test_malformed_structure_and_context_mismatches_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseMatrix.from_csr(Z, (1, 3), [0, 2], [2, 1], [1, 1])
    with pytest.raises(ValueError, match="outside shape"):
        SparseMatrix.from_triplets(Z, (1, 1), [(1, 0, 1)])

    integer = SparseMatrix.from_triplets(Z, (1, 1), [(0, 0, 1)])
    rational = SparseMatrix.from_triplets(Q, (1, 1), [(0, 0, 1)])
    with pytest.raises(ContextMismatchError, match="context"):
        _ = integer + rational
