# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether import Q
from noether.commalg import (
    LocalCompletionMap,
    LocalIdeal,
    LocalMonomialOrder,
    LocalPolynomial,
    LocalPolynomialRing,
    LocalReduction,
    LocalStandardBasis,
    PolynomialRing,
)
from noether.numthy import FiniteField


def test_local_context_translates_a_rational_center_and_round_trips() -> None:
    ambient = PolynomialRing(Q, ["x"])
    (x,) = ambient.generators()
    order = LocalMonomialOrder.LOCAL_DEGREVLEX
    local = LocalPolynomialRing(ambient, [1], order=order)
    (u,) = local.coordinate_ring.generators()

    translated = local.from_polynomial(x - 1)

    assert translated.numerator == u
    assert translated.denominator == local.coordinate_ring.one()
    assert local.point == [1]
    assert local.order == order
    assert LocalMonomialOrder.from_json(order.to_json()) == order
    assert LocalPolynomialRing.from_json(local.to_json()) == local
    assert LocalPolynomial.from_json(translated.to_json()) == translated


def test_local_fractions_reject_the_maximal_ideal_and_invert_units() -> None:
    ambient = PolynomialRing(Q, ["x", "y"])
    local = LocalPolynomialRing.at_origin(ambient)
    u, v = local.coordinate_ring.generators()
    unit = local.fraction(1 + u, local.coordinate_ring.one())

    assert unit.is_unit
    assert unit * unit.inverse() == local.one()
    with pytest.raises(ValueError, match="denominator must be nonzero"):
        local.fraction(v, u)


def test_mora_membership_and_tangent_cone_differ_from_global_division() -> None:
    ambient = PolynomialRing(Q, ["x"])
    (x,) = ambient.generators()
    local = LocalPolynomialRing.at_origin(ambient)
    ideal = local.ideal([local.from_polynomial(x + x**2)])
    basis = ideal.standard_basis()
    reduction = ideal.reduce(local.from_polynomial(x))

    assert reduction.is_member
    assert reduction.multiplier.is_unit
    assert len(reduction.quotients) == len(basis.basis)
    reduction.verify()
    assert LocalReduction.from_json(reduction.to_json()) == reduction
    assert ideal.contains(local.from_polynomial(x))
    assert basis.dimension == 0
    assert basis.multiplicity == 1
    assert basis.associated_graded().dimension() == 0
    basis.verify()
    assert LocalStandardBasis.from_json(basis.to_json()) == basis
    ideal.verify()
    assert LocalIdeal.from_json(ideal.to_json()) == ideal


def test_cusp_hilbert_samuel_data_is_coefficient_generic() -> None:
    field = FiniteField.prime(5)
    ambient = PolynomialRing(field, ["x", "y"])
    x, y = ambient.generators()
    local = LocalPolynomialRing.at_origin(ambient, order=LocalMonomialOrder.LOCAL_DEGLEX)

    basis = local.standard_basis([local.from_polynomial(y**2 - x**3)])

    assert basis.dimension == 1
    assert basis.multiplicity == 2
    assert basis.tangent_cone_ideal().dimension() == 1
    basis.verify()


def test_bounded_completion_maps_restrict_compatibly() -> None:
    ambient = PolynomialRing(Q, ["x"])
    local = LocalPolynomialRing.at_origin(ambient)
    (u,) = local.coordinate_ring.generators()
    value = local.fraction(local.coordinate_ring.one(), 1 - u)
    fine = local.completion(6)
    coarse = fine.restrict(3)
    jet = fine.apply(value)

    assert [jet.coefficient([exponent]) for exponent in range(6)] == [1] * 6
    assert fine.restrict_value(jet, 3) == coarse.apply(value)
    fine.verify_restriction(coarse, value)
    assert LocalCompletionMap.from_json(fine.to_json()) == fine
