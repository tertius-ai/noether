# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Ramified local fields project the native Rust context and arithmetic."""

import json

import pytest

from noether.commalg import MonomialMap, PolynomialRing, PowerSeriesRing, SeriesTruncation
from noether.linalg import Matrix, SparseMatrix, Vector
from noether.numthy import (
    FiniteField,
    PadicLocalField,
    PadicLocalFieldCoefficientFrobenius,
    PadicLocalFieldElement,
    PadicLocalFieldOrder,
    PadicLocalFieldRestriction,
    PadicLocalFractionalIdeal,
    PadicLocalPrecision,
    QadicRing,
)


def eisenstein_field(*, twisted: bool = False) -> PadicLocalField:
    residue = FiniteField.extension(3, 2)
    coefficients = QadicRing(residue, 4)
    polynomials = PolynomialRing(coefficients, ["x"])
    x = polynomials.generator(0)
    linear = coefficients(3) * coefficients.generator() if twisted else coefficients.zero()
    eisenstein = x**2 + polynomials(linear) * x + 3
    return PadicLocalField(
        coefficients,
        eisenstein,
        PadicLocalPrecision(7),
        uniformizer_name="pi",
    )


def test_context_identity_precision_and_json_are_explicit() -> None:
    field = eisenstein_field()

    assert PadicLocalField.type_id == "noether/numthy/PadicLocalField@1"
    assert field.prime == 3
    assert field.ramification_index == 2
    assert field.residue_degree == 2
    assert field.degree == 4
    assert field.coordinate_precisions == [4, 3]
    assert field.uniformizer_name == "pi"
    assert field.coefficient_field.residue_field == field.residue_field
    assert field.local_precision == PadicLocalPrecision(7)
    assert field.defining_polynomial.ring.coefficient_ring == field.coefficient_ring
    assert PadicLocalField.from_json(field.to_json()) == field
    assert json.loads(field.to_json())["type"] == PadicLocalField.type_id

    with pytest.raises(ValueError, match="at least one"):
        PadicLocalPrecision(0)


def test_arithmetic_residue_and_tower_invariants_stay_native() -> None:
    field = eisenstein_field()
    pi = field.uniformizer()
    unit = field.one() + pi
    inverse = unit.inverse()

    assert isinstance(pi, PadicLocalFieldElement)
    assert pi.valuation().value == 1
    assert pi.known_absolute_precision == 7
    assert unit * inverse == field.one()
    assert field.residue(field.one()) == field.residue_field.one()
    assert field.residue(pi) == field.residue_field.zero()
    assert field.teichmuller(field.residue_field.generator()).valuation().value == 0
    trace = pi.trace()
    trace_field = trace.field
    assert trace_field is not None
    assert trace == trace_field.zero()
    assert pi.norm().valuation().value == 2
    assert pi.characteristic_polynomial().degree() == field.degree
    theta = field.element_from_qadic(field.coefficient_field.generator())
    primitive = theta + pi
    assert primitive.minimal_polynomial() == primitive.characteristic_polynomial()
    with pytest.raises(ValueError, match="does not certify"):
        pi.minimal_polynomial()
    assert PadicLocalFieldElement.from_json(inverse.to_json()) == inverse


def test_restriction_is_a_checked_cross_context_map() -> None:
    field = eisenstein_field()
    restriction = field.precision_restriction(PadicLocalPrecision(5))
    value = field.one() + field.uniformizer()
    restricted = restriction(value)

    assert isinstance(restriction, PadicLocalFieldRestriction)
    assert restriction.source == field
    assert restriction.target.local_precision == PadicLocalPrecision(5)
    assert restricted.field == restriction.target
    assert restricted.known_absolute_precision == 5
    restriction.verify()
    assert PadicLocalFieldRestriction.from_json(restriction.to_json()) == restriction

    with pytest.raises(ValueError, match="cannot restrict"):
        field.precision_restriction(PadicLocalPrecision(8))


def test_coefficient_frobenius_targets_the_conjugate_presentation() -> None:
    field = eisenstein_field(twisted=True)
    sigma = field.coefficient_frobenius_conjugate(1)
    sigma_again = sigma.target.coefficient_frobenius_conjugate(1)
    value = field.one() + field.uniformizer()
    return_map = sigma_again.compose(sigma)

    assert isinstance(sigma, PadicLocalFieldCoefficientFrobenius)
    assert sigma.source == field
    assert sigma.target != field
    assert not sigma.is_endomorphism()
    assert sigma(value).field == sigma.target
    assert return_map.target == field
    assert return_map(value) == value
    sigma.verify()
    assert PadicLocalFieldCoefficientFrobenius.from_json(sigma.to_json()) == sigma


def test_generic_coefficient_containers_and_checked_maps() -> None:
    field = eisenstein_field(twisted=True)
    pi = field.uniformizer()
    restriction = field.precision_restriction(PadicLocalPrecision(5))

    polynomials = PolynomialRing(field, ["T"])
    polynomial = polynomials.generator(0) ** 2 + polynomials(pi)
    restricted_polynomial = restriction.apply_polynomial(polynomial)
    assert restricted_polynomial.ring.coefficient_ring == restriction.target
    assert restricted_polynomial.constant_coefficient() == restriction(pi)

    series_ring = PowerSeriesRing(field, ["z"], SeriesTruncation.univariate(4))
    series = series_ring(pi) + series_ring.generator(0)
    restricted_series = restriction.apply_power_series(series)
    assert restricted_series.ring.coefficient_ring == restriction.target
    assert restricted_series.constant_coefficient() == restriction(pi)

    vector = Vector.from_values(field, [field.one(), pi])
    matrix = Matrix.from_rows(field, [[field.one(), pi], [field.zero(), field.one()]])
    sparse = SparseMatrix.from_triplets(
        field,
        (2, 2),
        [(0, 0, field.one()), (0, 1, pi), (1, 1, field.one())],
    )
    assert matrix @ vector == sparse.apply(vector)
    assert restriction.apply_vector(vector).ring == restriction.target
    assert restriction.apply_matrix(matrix).ring == restriction.target
    assert restriction.apply_sparse_matrix(sparse).ring == restriction.target
    assert Matrix.from_json(matrix.to_json()) == matrix
    sparse_round_trip: SparseMatrix[PadicLocalField] = SparseMatrix.from_json(sparse.to_json())
    assert sparse_round_trip == sparse

    series_matrix = Matrix.from_rows(series_ring, [[series]])
    assert series_matrix.det() == series

    sigma = field.coefficient_frobenius_conjugate(1)
    assert sigma.apply_polynomial(polynomial).ring.coefficient_ring == sigma.target
    assert sigma.apply_power_series(series).ring.coefficient_ring == sigma.target
    assert sigma.apply_vector(vector).ring == sigma.target
    assert sigma.apply_matrix(matrix).ring == sigma.target
    assert sigma.apply_sparse_matrix(sparse).ring == sigma.target
    assert polynomial.semilinear_substitution(sigma, MonomialMap.identity(1)) == (
        sigma.apply_polynomial(polynomial)
    )
    assert series.semilinear_substitution(sigma, MonomialMap.identity(1)) == (
        sigma.apply_power_series(series)
    )


def test_maximal_and_nonmaximal_local_orders_are_certified_zp_lattices() -> None:
    field = eisenstein_field()
    maximal = field.maximal_order()

    assert isinstance(maximal, PadicLocalFieldOrder)
    assert maximal == field.equation_order()
    assert maximal.is_maximal()
    assert maximal.index_in_maximal_order_valuation == 0
    assert maximal.certified_precision == 3
    assert maximal.different_valuation() == 1
    assert maximal.discriminant_valuation() == 2
    assert field.uniformizer() in maximal
    assert PadicLocalFieldOrder.from_json(maximal.to_json()) == maximal

    p = field.integral_element([field.coefficient_ring(3)])
    basis = maximal.basis
    nonmaximal = field.order([basis[0], *(p * element for element in basis[1:])])
    assert not nonmaximal.is_maximal()
    assert nonmaximal.index_in_maximal_order_valuation == field.degree - 1
    assert nonmaximal.index_valuation(maximal) == field.degree - 1
    assert nonmaximal.maximal_overorder() == maximal
    assert field.uniformizer() not in nonmaximal
    with pytest.raises(ValueError, match="maximal local order"):
        nonmaximal.different_valuation()
    with pytest.raises(ValueError, match="does not belong"):
        nonmaximal(field.uniformizer())

    unit = field.one() + p * field.uniformizer()
    assert nonmaximal.inverse_unit(unit) in nonmaximal

    vector = Vector.from_values(nonmaximal, [field.one(), p * field.uniformizer()])
    matrix = Matrix.from_rows(
        nonmaximal,
        [[field.one(), p * field.uniformizer()], [field.zero(), field.one()]],
    )
    sparse = SparseMatrix.from_dense(matrix)
    assert matrix.ring == nonmaximal
    assert vector.ring == nonmaximal
    assert sparse.ring == nonmaximal
    assert matrix @ vector == sparse.apply(vector)
    assert Matrix.from_json(matrix.to_json()) == matrix
    sparse_round_trip: SparseMatrix[PadicLocalFieldOrder] = SparseMatrix.from_json(sparse.to_json())
    assert sparse_round_trip == sparse
    with pytest.raises(ValueError, match="does not belong"):
        Matrix.from_rows(nonmaximal, [[field.uniformizer()]])


def test_local_fractional_ideals_keep_nonmaximal_arithmetic_honest() -> None:
    field = eisenstein_field()
    order = field.maximal_order()
    unit = order.fractional_ideal([field.one()])
    maximal = order.maximal_ideal()

    assert isinstance(maximal, PadicLocalFractionalIdeal)
    assert maximal.is_integral()
    assert maximal.norm_valuation() == field.residue_degree
    assert maximal.maximal_ideal_exponent() == 1
    assert field.uniformizer() in maximal
    assert field.one() not in maximal
    assert (maximal + unit) == unit
    assert (maximal * maximal).maximal_ideal_exponent() == 2

    inverse = maximal.inverse()
    assert inverse.maximal_ideal_exponent() == -1
    product = maximal * inverse
    assert product.maximal_ideal_exponent() == 0
    assert field.one() in product
    assert product.certified_precision < unit.certified_precision
    assert PadicLocalFractionalIdeal.from_json(inverse.to_json()) == inverse

    p = field.integral_element([field.coefficient_ring(3)])
    basis = order.basis
    nonmaximal = field.order([basis[0], *(p * element for element in basis[1:])])
    principal = nonmaximal.fractional_ideal([p])
    assert principal.is_integral()
    assert principal.norm_valuation() == field.degree
    with pytest.raises(ValueError, match="maximal local order"):
        principal.inverse()
    with pytest.raises(ValueError, match="maximal local order"):
        nonmaximal.maximal_ideal()
