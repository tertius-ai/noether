# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

from noether import Z
from noether.commalg import MonomialOrder, PolynomialRing, StrongGroebnerBasis


def test_integral_strong_basis_is_typed_and_verifiable() -> None:
    ring = PolynomialRing(Z, ["x"], order=MonomialOrder.LEX)
    (x,) = ring.generators()
    basis = ring.strong_groebner_basis([2 * x, 3 * x])

    assert isinstance(basis, StrongGroebnerBasis)
    assert basis.basis == [x]
    assert 2 * x in basis
    assert ring.one() not in basis
    assert basis.normal_form(5 * x) == ring.zero()
    basis.verify()
    assert StrongGroebnerBasis.from_json(basis.to_json()) == basis
