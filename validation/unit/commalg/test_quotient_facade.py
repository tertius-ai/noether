# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Polynomial quotients project the canonical Rust ring and element contexts."""

import json

import pytest

from noether import Q, RationalField
from noether.commalg import (
    FiniteQuotientAlgebra,
    PolynomialRing,
    QuotientElement,
    QuotientRing,
    QuotientRingMap,
    WeightedGrading,
)
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import FiniteField


def test_rational_quotient_has_canonical_arithmetic_and_identity() -> None:
    ambient = PolynomialRing(Q, ["x", "y"])
    x, y = ambient.generators()
    ideal = ambient.ideal([x**2, x * y, y**2])
    quotient = ambient.quotient(ideal)

    assert isinstance(quotient, QuotientRing)
    assert quotient.base_ring == ambient
    assert quotient.defining_ideal == ideal
    assert quotient.is_graded
    assert quotient.grading == ambient.grading
    assert quotient(x + x**2) == quotient(x)
    assert quotient(x + y).lift == x + y
    assert quotient(x) * quotient(y) == quotient.zero()
    assert quotient(x + y) ** 2 == quotient.zero()
    assert quotient.one() + quotient(x) - quotient.one() == quotient(x)
    assert quotient(x).ring == quotient
    degree = quotient(x).multihomogeneous_degree()
    assert degree is not None and degree.coordinates == [1]
    assert QuotientRing.type_id == "noether/commalg/QuotientRing@1"
    assert QuotientElement.type_id == "noether/commalg/QuotientElement@1"


def test_quotient_units_json_and_context_mismatch_fail_closed() -> None:
    ambient = PolynomialRing(Q, ["x"])
    x = ambient.generator(0)
    quotient = ambient.quotient(ambient.ideal([x**3]))
    unit = quotient(ambient.one() + x)

    assert unit.is_unit()
    assert unit * unit.inverse() == quotient.one()
    assert quotient.one() / unit == unit.inverse()
    assert not quotient(x).is_unit()
    with pytest.raises(ZeroDivisionError, match="not a unit"):
        quotient(x).inverse()

    restored = QuotientRing.from_json(quotient.to_json())
    assert restored == quotient
    assert hash(restored) == hash(quotient)
    restored_unit = QuotientElement.from_json(unit.to_json())
    assert restored_unit == unit
    assert hash(restored_unit) == hash(unit)

    payload = json.loads(unit.to_json())
    payload["lift"]["terms"].append(
        {"exponents": [3], "coefficient": {"numerator": "1", "denominator": "1"}}
    )
    with pytest.raises(ValueError):
        QuotientElement.from_json(json.dumps(payload))

    other = PolynomialRing(Q, ["x"], order="lex")
    with pytest.raises(ValueError, match="context mismatch"):
        quotient(other.generator(0))


def test_checked_quotient_maps_require_relations_to_vanish() -> None:
    source_ambient = PolynomialRing(Q, ["x", "y"])
    x, y = source_ambient.generators()
    source = source_ambient.quotient(source_ambient.ideal([x**2 - y, y**2]))
    target_ambient = PolynomialRing(Q, ["t"])
    t = target_ambient.generator(0)
    target = target_ambient.quotient(target_ambient.ideal([t**4]))
    base_map = source_ambient.homomorphism(target_ambient, [t, t**2])

    descended = source.induced_map(target, base_map)
    assert isinstance(descended, QuotientRingMap)
    assert descended.source == source
    assert descended.target == target
    assert descended.base_map == base_map
    assert descended(source(x + y)) == target(t + t**2)
    descended.verify()
    restored = QuotientRingMap.from_json(descended.to_json())
    assert restored == descended
    assert hash(restored) == hash(descended)
    assert QuotientRingMap.type_id == "noether/commalg/QuotientRingMap@1"

    invalid = source_ambient.homomorphism(target_ambient, [t, target_ambient.zero()])
    with pytest.raises(ValueError, match="relation 0 has nonzero image"):
        source.induced_map(target, invalid)


def test_checked_quotient_maps_use_finite_field_polynomial_lanes() -> None:
    field = FiniteField.prime(101)
    source_ambient = PolynomialRing(field, ["x"])
    x = source_ambient.generator(0)
    source = source_ambient.quotient(source_ambient.ideal([x**3]))
    target_ambient = PolynomialRing(field, ["t"])
    t = target_ambient.generator(0)
    target = target_ambient.quotient(target_ambient.ideal([t**2]))
    base_map = source_ambient.homomorphism(target_ambient, [t])

    descended = source.induced_map(target, base_map)
    assert descended(source(x + x**2)) == target(t)
    assert QuotientRingMap.from_json(descended.to_json()) == descended


def test_quotient_hilbert_and_finite_algebra_reuse_defining_ideal() -> None:
    field = FiniteField.prime(5)
    ambient = PolynomialRing(field, ["x", "y"])
    x, y = ambient.generators()
    ideal = ambient.ideal([x**2, y**3])
    quotient = ambient.quotient(ideal)

    assert quotient.hilbert_series() == ideal.hilbert_series()
    assert quotient.hilbert_numerator() == ideal.hilbert_numerator()
    assert quotient.dimension() == 0
    assert quotient.degree() == 6
    assert quotient.hilbert_polynomial() == ideal.hilbert_polynomial()
    algebra = quotient.finite_algebra()
    assert isinstance(algebra, FiniteQuotientAlgebra)
    assert algebra.source == ideal
    assert algebra.dimension == 6


def test_multigraded_and_ungraded_quotients_report_honestly() -> None:
    ambient = PolynomialRing(Q, ["x", "y"], grading=WeightedGrading([1, 2]))
    x, y = ambient.generators()
    quotient = ambient.quotient(ambient.ideal([x**2 - y]))
    assert quotient.is_graded
    degree = quotient(x**2).multihomogeneous_degree()
    assert degree is not None and degree.coordinates == [2]

    ungraded = ambient.quotient(ambient.ideal([x + y]))
    assert not ungraded.is_graded
    assert ungraded.grading is None
    with pytest.raises(ValueError, match="grading"):
        ungraded(x).multihomogeneous_degree()


def test_dense_matrices_use_quotient_coefficients_without_a_second_surface() -> None:
    ambient = PolynomialRing(Q, ["x", "y"])
    x, y = ambient.generators()
    quotient = ambient.quotient(ambient.ideal([x**2, y**2]))
    matrix = Matrix.from_rows(quotient, [[x, quotient(y)], [quotient.one(), x + y]])

    assert matrix.ring == quotient
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == quotient(x)
    assert matrix.to_rows()[0][1] == quotient(y)
    assert matrix.det() == quotient(x * (x + y) - y)
    assert matrix.transpose().transpose() == matrix
    assert matrix @ Matrix.identity(quotient, 2) == matrix
    assert matrix + Matrix.zeros(quotient, (2, 2)) == matrix
    assert matrix * quotient(x) == Matrix.from_rows(
        quotient,
        [[quotient.zero(), quotient(x * y)], [quotient(x), quotient(x * y)]],
    )
    assert Matrix.from_json(matrix.to_json()) == matrix

    vector = Vector.from_values(quotient, [quotient.one(), y])
    assert vector.ring == quotient
    assert vector.to_values() == [quotient.one(), quotient(y)]
    assert matrix @ vector == Vector.from_values(quotient, [x, quotient.one() + quotient(x * y)])
    assert vector.dot(vector) == quotient.one()
    assert vector.to_column_matrix().to_rows() == [[quotient.one()], [quotient(y)]]
    assert Vector.from_json(vector.to_json()) == vector

    sparse = SparseMatrix.from_triplets(
        quotient,
        (2, 2),
        [(0, 0, x), (0, 1, quotient(y)), (1, 0, quotient.one())],
    )
    assert sparse.ring == quotient
    assert sparse.nnz == 3
    assert sparse.apply(vector) == Vector.from_values(quotient, [x, quotient.one()])
    assert sparse.to_dense() == Matrix.from_rows(
        quotient,
        [[x, y], [quotient.one(), quotient.zero()]],
    )
    assert SparseMatrix.from_dense(sparse.to_dense()) == sparse
    restored_sparse: SparseMatrix[QuotientRing[RationalField]] = SparseMatrix.from_json(
        sparse.to_json()
    )
    assert restored_sparse == sparse
