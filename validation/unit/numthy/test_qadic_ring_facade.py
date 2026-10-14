# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Fixed-cap Witt rings expose FLINT q-adics through the numthy facade."""

import json

import pytest

from noether.commalg import PolynomialRing, PowerSeriesRing, SeriesTruncation
from noether.linalg import Matrix, Vector
from noether.numthy import (
    FiniteField,
    PadicElement,
    PadicRing,
    QadicElement,
    QadicRing,
    QadicRingFrobenius,
    QadicRingRestriction,
)


def test_witt_ring_context_and_arithmetic() -> None:
    residue_field = FiniteField.extension(3, 2)
    ring = QadicRing(residue_field, 4)
    generator = ring.generator()
    unit = generator + ring.one()

    assert ring.residue_field == residue_field
    assert ring.prime == 3
    assert ring.residue_degree == 2
    assert ring.coefficient_ring == PadicRing(3, 4)
    assert ring.absolute_precision == 4
    assert ring.cardinality() == 3**8
    assert isinstance(generator, QadicElement)
    assert all(isinstance(coefficient, PadicElement) for coefficient in generator.coefficients)
    assert all(coefficient.ring == ring.coefficient_ring for coefficient in generator.coefficients)
    assert unit * unit.inverse() == ring.one()
    assert ring.one().trace() == ring.coefficient_ring(2)
    assert ring(3).norm() == ring.coefficient_ring(9)
    assert hash(ring) == hash(QadicRing(residue_field, 4))


def test_teichmuller_frobenius_and_restriction() -> None:
    residue_field = FiniteField.extension(5, 2)
    ring = QadicRing(residue_field, 4)
    residue = residue_field.generator() + residue_field.one()
    lift = ring.teichmuller(residue)
    restricted = ring.restrict(lift + ring(125), 2)

    assert ring.residue(lift) == residue
    assert lift.residue() == residue
    assert lift**residue_field.order == lift
    assert ring.residue(lift.frobenius(1)) == residue.frobenius(1)
    assert restricted.ring == QadicRing(residue_field, 2)
    assert restricted.known_absolute_precision == 2
    assert restricted.ring.residue(restricted) == residue


def test_integral_power_basis_coordinates_construct_and_validate() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 4)
    coordinates = [ring.coefficient_ring(7), ring.coefficient_ring(11)]

    assert ring.element_from_coefficients(coordinates).coefficients == coordinates
    assert ring.element_from_coefficients([]) == ring.zero()

    with pytest.raises(ValueError, match="degree-2"):
        ring.element_from_coefficients([*coordinates, ring.coefficient_ring.zero()])
    with pytest.raises(ValueError, match="coordinate 0.*context mismatch"):
        ring.element_from_coefficients([PadicRing(5, 4).one()])


def test_ring_restriction_map_and_uniformizer_division() -> None:
    residue_field = FiniteField.extension(3, 2)
    ring = QadicRing(residue_field, 5)
    restriction = ring.precision_restriction(3)
    value = ring(3**2) * (ring.generator() + ring.one())

    assert isinstance(restriction, QadicRingRestriction)
    assert restriction.source == ring
    assert restriction.target == QadicRing(residue_field, 3)
    assert QadicRingRestriction.from_json(restriction.to_json()) == restriction
    assert restriction(ring.generator()).ring == restriction.target

    quotient = ring.divide_uniformizer_exact(value, 2)
    assert quotient.ring == QadicRing(residue_field, 3)
    assert quotient == quotient.ring.generator() + quotient.ring.one()

    with pytest.raises(ValueError, match="cannot certify integral division"):
        ring.divide_uniformizer_exact(ring.generator(), 1)


def test_qadic_restriction_maps_composite_values() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 4)
    restriction = ring.precision_restriction(2)
    polynomials = PolynomialRing(ring, ["x"])
    polynomial = polynomials.generator(0) + polynomials(ring.generator())
    series_ring = PowerSeriesRing(ring, ["z"], SeriesTruncation.univariate(4))
    series = series_ring.generator(0) + series_ring.constant(ring.generator())
    vector = Vector.from_values(ring, [ring.one(), ring.generator()])
    matrix = Matrix.from_rows(ring, [[ring.generator(), ring.one()]])

    assert restriction.apply_polynomial(polynomial).ring.coefficient_ring == restriction.target
    assert restriction.apply_power_series(series).ring.coefficient_ring == restriction.target
    assert restriction.apply_vector(vector).ring == restriction.target
    assert restriction.apply_matrix(matrix).ring == restriction.target


def test_qadic_coefficient_frobenius_has_residue_degree_order() -> None:
    ring = QadicRing(FiniteField.extension(3, 2), 5)
    sigma = ring.coefficient_frobenius(1)
    generator = ring.generator()

    assert isinstance(sigma, QadicRingFrobenius)
    assert sigma.context == ring
    assert sigma.exponent == 1
    assert sigma(generator) != generator
    assert sigma.compose(sigma)(generator) == generator
    assert QadicRingFrobenius.from_json(sigma.to_json()) == sigma

    with pytest.raises(ValueError, match="context mismatch"):
        sigma(QadicRing(FiniteField.extension(5, 2), 5).generator())


def test_nonunits_and_context_mismatches_fail_closed() -> None:
    first = QadicRing(FiniteField.extension(3, 2), 3)
    second = QadicRing(FiniteField.extension(5, 2), 3)

    with pytest.raises(ZeroDivisionError, match="not a unit"):
        first(first.prime).inverse()
    with pytest.raises(ValueError, match="context mismatch"):
        _ = first.one() + second.one()
    with pytest.raises(ValueError, match="at least one"):
        QadicRing(first.residue_field, 0)


def test_witt_contexts_and_integral_coordinates_have_typed_json() -> None:
    residue_field = FiniteField.extension(3, 2)
    ring = QadicRing(residue_field, 4)
    value = ring.generator() + ring(17)

    assert QadicRing.type_id == "noether/numthy/QadicRing@1"
    assert QadicElement.type_id == "noether/numthy/QadicElement@1"
    assert QadicRing.from_json(ring.to_json()) == ring
    assert QadicElement.from_json(value.to_json()) == value
    context = json.loads(ring.to_json())
    assert context == {
        "type": QadicRing.type_id,
        "residue_field": json.loads(residue_field.to_json()),
        "absolute_precision": 4,
    }

    noncanonical = json.loads(value.to_json())
    noncanonical["value"]["coefficients"][0] = "81"
    with pytest.raises(ValueError, match="canonical FLINT form|must lie in"):
        QadicElement.from_json(json.dumps(noncanonical))

    reducible = json.loads(ring.to_json())
    reducible["residue_field"]["modulus"] = [2, 0, 1]
    with pytest.raises(ValueError, match="reducible"):
        QadicRing.from_json(json.dumps(reducible))
