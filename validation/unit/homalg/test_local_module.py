# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from noether import Q
from noether.commalg import LocalPolynomialRing, PolynomialRing
from noether.homalg import (
    LocalModuleOrder,
    LocalModuleReduction,
    LocalModuleStandardBasis,
)
from noether.numthy import FiniteField


def test_local_module_weak_normal_form_records_membership_evidence() -> None:
    ambient = PolynomialRing(Q, ["x"])
    (x,) = ambient.generators()
    local = LocalPolynomialRing.at_origin(ambient)
    local_x = local.from_polynomial(x)
    generator = local.from_polynomial(x + x**2)
    zero = local.zero()

    basis = LocalModuleStandardBasis.from_generators(
        local,
        2,
        [[generator, zero]],
        order=LocalModuleOrder.TERM_OVER_POSITION,
    )
    reduction = basis.reduce([local_x, zero])

    assert basis.rank == 2
    assert basis.ring == local
    assert basis.order == LocalModuleOrder.TERM_OVER_POSITION
    assert reduction.is_member
    assert reduction.multiplier.is_unit
    assert reduction.remainder == [zero, zero]
    assert len(reduction.quotients) == len(basis.basis)
    assert basis.contains([local_x, zero])
    reduction.verify()
    basis.verify()
    assert LocalModuleReduction.from_json(reduction.to_json()) == reduction
    assert LocalModuleStandardBasis.from_json(basis.to_json()) == basis


def test_position_over_term_is_available_over_finite_fields() -> None:
    field = FiniteField.prime(5)
    ambient = PolynomialRing(field, ["x", "y"])
    x, y = ambient.generators()
    local = LocalPolynomialRing.at_origin(ambient)
    local_x = local.from_polynomial(x)
    local_y = local.from_polynomial(y)
    zero = local.zero()

    basis = LocalModuleStandardBasis.from_generators(
        local,
        2,
        [[local_x, zero], [zero, local_y]],
        order=LocalModuleOrder.POSITION_OVER_TERM,
    )

    assert basis.contains([local_x, local_y])
    assert not basis.contains([local.one(), zero])
    basis.verify()
    assert LocalModuleOrder.from_json(basis.order.to_json()) == basis.order
