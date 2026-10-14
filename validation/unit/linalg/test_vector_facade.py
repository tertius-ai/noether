# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Vectors and operators are thin projections of ``noether::linalg``."""

from fractions import Fraction

import pytest

from noether import ContextMismatchError, Q, Z
from noether.commalg import ResidueRing
from noether.linalg import LinearOperator, Matrix, Vector


def test_integer_vector_owns_the_canonical_protocol() -> None:
    vector = Vector.from_values(Z, [1, 2, 3])

    assert vector.ring == Z
    assert vector.dimension == 3
    assert vector[0] == 1
    assert vector[-1] == 3
    assert vector.to_values() == [1, 2, 3]
    assert list(vector) == [1, 2, 3]
    assert vector.to_column_matrix().to_rows() == [[1], [2], [3]]
    assert vector.to_row_matrix().to_rows() == [[1, 2, 3]]
    assert (vector + vector).to_values() == [2, 4, 6]
    assert (vector - vector).to_values() == [0, 0, 0]
    assert (-vector).to_values() == [-1, -2, -3]
    assert (vector * 3).to_values() == [3, 6, 9]
    assert (3 * vector).to_values() == [3, 6, 9]
    assert vector.dot(vector) == 14
    assert Vector.type_id == "noether/linalg/Vector@1"
    assert Vector.from_json(vector.to_json()) == vector


def test_rational_vectors_are_exact() -> None:
    vector = Vector.from_values(Q, [Fraction(1, 2), Fraction(2, 3)])

    assert vector[0] == Fraction(1, 2)
    assert vector.dot(vector) == Fraction(25, 36)
    assert (vector * Fraction(3, 2)).to_values() == [Fraction(3, 4), 1]
    assert Vector.from_json(vector.to_json()) == vector


def test_matrix_is_a_linear_operator_on_vectors_and_blocks() -> None:
    matrix = Matrix.from_rows(Z, [[1, 2, 3], [4, 5, 6]])
    vector = Vector.from_values(Z, [7, 8, 9])
    identity = Matrix.identity(Z, 3)

    assert isinstance(matrix, LinearOperator)
    assert matrix.input_dimension == 3
    assert matrix.output_dimension == 2
    assert matrix.apply(vector).to_values() == [50, 122]
    assert (matrix @ vector).to_values() == [50, 122]
    assert matrix.apply_block(identity) == matrix


def test_zero_dimensions_and_mismatches_fail_explicitly() -> None:
    empty = Vector.zeros(Z, 0)

    assert len(empty) == 0
    assert empty.to_values() == []
    assert empty.to_column_matrix().shape == (0, 1)
    assert empty.to_row_matrix().shape == (1, 0)
    assert empty.dot(empty) == 0

    for modulus in (4, 8):
        ring = ResidueRing(modulus)
        packed_empty = Vector.zeros(ring, 0)
        assert packed_empty.dot(packed_empty) == ring(0)

    with pytest.raises(ValueError, match="dimension mismatch"):
        _ = Vector.zeros(Z, 2) + Vector.zeros(Z, 3)
    with pytest.raises(ContextMismatchError, match="expected Z.*received Q"):
        _ = Vector.zeros(Z, 2) + Vector.zeros(Q, 2)
    with pytest.raises(ValueError, match="linear-operator application"):
        _ = Matrix.zeros(Z, (2, 3)) @ Vector.zeros(Z, 2)
    with pytest.raises(IndexError, match="outside dimension 0"):
        _ = empty[0]


def test_arbitrary_precision_and_repr_remain_bounded() -> None:
    huge = 1 << 20_000
    vector = Vector.from_values(Z, [huge] * 12)
    rendered = repr(vector)
    summary = vector._tertius_summary_()

    assert vector[0] == huge
    assert type(vector).__module__ == "noether.linalg"
    assert "dimension=12" in rendered
    assert len(rendered) < 700
    assert summary["type"] == "noether/linalg/Vector@1"
    assert summary["dimension"] == 12
