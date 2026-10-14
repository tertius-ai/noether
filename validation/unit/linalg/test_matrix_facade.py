# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The public Python matrix is a thin projection of ``noether::linalg``."""

import json
from fractions import Fraction

import pytest

from noether import ContextMismatchError, Field, IntegerRing, Q, RationalField, Ring, Z
from noether._engine_gateway import _import_engine
from noether.commalg import ResidueRing
from noether.linalg import Matrix, Vector
from noether.numthy import FiniteField, PadicRing, QadicRing


def test_contexts_are_typed_and_use_builtin_scalar_values() -> None:
    assert isinstance(Z, IntegerRing)
    assert isinstance(Q, RationalField)
    assert isinstance(Z, Ring)
    assert isinstance(Q, Field)
    assert Z.zero() == 0
    assert Z.one() == 1
    assert Q(Fraction(2, 4)) == Fraction(1, 2)
    assert Q.inverse(Fraction(-2, 3)) == Fraction(-3, 2)


def test_integer_matrix_owns_the_canonical_protocol() -> None:
    matrix = Matrix.from_rows(Z, [[1, 2], [3, 4]])

    assert matrix.shape == (2, 2)
    assert matrix.ring == Z
    assert matrix[0, 1] == 2
    assert matrix[-1, -1] == 4
    assert matrix.to_rows() == [[1, 2], [3, 4]]
    assert list(matrix) == [[1, 2], [3, 4]]
    assert matrix.transpose().to_rows() == [[1, 3], [2, 4]]
    assert (matrix + matrix).to_rows() == [[2, 4], [6, 8]]
    assert (matrix - matrix).to_rows() == [[0, 0], [0, 0]]
    assert (-matrix).to_rows() == [[-1, -2], [-3, -4]]
    assert (matrix * 3).to_rows() == [[3, 6], [9, 12]]
    assert (3 * matrix).to_rows() == [[3, 6], [9, 12]]
    assert (matrix @ Matrix.identity(Z, 2)) == matrix
    assert matrix.matmul(Matrix.identity(Z, 2)) == matrix
    assert Matrix.type_id == "noether/linalg/Matrix@1"
    assert Matrix.from_json(matrix.to_json()) == matrix
    assert json.loads(matrix.to_json()) == {
        "type": Matrix.type_id,
        "ring": json.loads(Z.to_json()),
        "shape": [2, 2],
        "entries": ["1", "2", "3", "4"],
    }


def test_rational_matrix_returns_fractions_and_is_exact() -> None:
    matrix = Matrix.from_rows(Q, [[Fraction(1, 2), 1], [2, Fraction(3, 4)]])
    square = matrix @ matrix

    assert matrix.ring == Q
    assert matrix[0, 0] == Fraction(1, 2)
    assert square.to_rows() == [
        [Fraction(9, 4), Fraction(5, 4)],
        [Fraction(5, 2), Fraction(41, 16)],
    ]
    assert (matrix * Fraction(2, 3))[1, 1] == Fraction(1, 2)


def test_typed_json_dispatches_over_integral_scalar_contexts() -> None:
    finite = FiniteField.extension(3, 2)
    contexts = [ResidueRing(12), PadicRing(3, 4), QadicRing(finite, 4)]

    for ring in contexts:
        matrix = Matrix.from_rows(ring, [[ring(2), 1]])
        vector = Vector.from_values(ring, [ring(2), 1])
        assert Matrix.from_json(matrix.to_json()) == matrix
        assert Vector.from_json(vector.to_json()) == vector


def test_zero_dimensional_shapes_are_preserved() -> None:
    zero_by_three = Matrix.zeros(Z, (0, 3))
    three_by_zero = Matrix.zeros(Z, (3, 0))
    rational_zero = Matrix.zeros(Q, (3, 0))

    assert zero_by_three.shape == (0, 3)
    assert zero_by_three.to_rows() == []
    assert three_by_zero.shape == (3, 0)
    assert three_by_zero.to_rows() == [[], [], []]
    assert rational_zero.ring == Q
    assert (zero_by_three @ three_by_zero).shape == (0, 0)
    assert Matrix.from_json(zero_by_three.to_json()).shape == (0, 3)

    for modulus in (2, 4, 5, 8, 1 << 80):
        ring = ResidueRing(modulus)
        for shape in ((2, 0), (0, 2), (0, 0)):
            zero = Matrix.zeros(ring, shape)
            assert zero + zero == zero
            assert zero - zero == zero
            assert -zero == zero
            assert zero * ring(3) == zero
            assert zero.transpose().shape == shape[::-1]
        assert (Matrix.zeros(ring, (2, 0)) @ Matrix.zeros(ring, (0, 3))).shape == (2, 3)

    malformed = json.loads(Matrix.from_rows(Z, [[1, 2]]).to_json())
    malformed["shape"] = [2, 2]
    with pytest.raises(ValueError, match="shape 2x2 requires 4"):
        Matrix.from_json(json.dumps(malformed))


def test_arbitrary_precision_integers_do_not_cross_a_bounded_word() -> None:
    huge = -(1 << 20_000) + 17
    matrix = Matrix.from_rows(Z, [[huge]])

    assert matrix[0, 0] == huge
    assert (matrix @ matrix)[0, 0] == huge * huge
    assert len(repr(matrix)) < 180


def test_shapes_indices_and_contexts_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="row 1 has 1 columns; expected 2"):
        Matrix.from_rows(Z, [[1, 2], [3]])
    with pytest.raises(IndexError, match="outside shape 1x1"):
        _ = Matrix.from_rows(Z, [[1]])[1, 0]
    with pytest.raises(ContextMismatchError, match="expected Z.*received Q"):
        _ = Matrix.identity(Z, 1) + Matrix.identity(Q, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = Matrix.zeros(Z, (2, 3)) @ Matrix.zeros(Z, (2, 1))
    with pytest.raises(TypeError, match="Python int"):
        _ = Matrix.identity(Z, 1) * Fraction(1, 2)


def test_repr_and_summary_are_bounded_and_publicly_named() -> None:
    matrix = Matrix.from_rows(Z, [[row * 10 + column for column in range(8)] for row in range(8)])
    rendered = repr(matrix)
    summary = matrix._tertius_summary_()

    assert type(matrix).__module__ == "noether.linalg"
    assert "_engine" not in rendered
    assert "shape=(8, 8)" in rendered
    assert len(rendered) < 240
    assert summary["type"] == "noether/linalg/Matrix@1"
    assert summary["shape"] == (8, 8)


def test_superseded_integer_matrix_record_is_not_exported() -> None:
    assert not hasattr(_import_engine(), "IntegerMatrix")
