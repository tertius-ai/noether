# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether import Q
from noether.commalg import (
    PolynomialPrimeLocalElement,
    PolynomialPrimeLocalIdeal,
    PolynomialPrimeLocalization,
    PolynomialPrimeLocalMembership,
    PolynomialRing,
)


def test_coordinate_prime_localization_inverts_exactly_the_complement() -> None:
    ambient = PolynomialRing(Q, ["x", "y"])
    x, y = ambient.generators()
    prime = ambient.ideal([x]).minimal_primes().prime_certificates[0]
    local = PolynomialPrimeLocalization(prime)
    local_x, local_y = local.generators()

    inverse_y = local.fraction(ambient.one(), y)

    assert not local_x.is_unit
    assert local_y.is_unit
    assert inverse_y * local_y == local.one()
    with pytest.raises(ValueError, match="outside the prime ideal"):
        local.fraction(ambient.one(), x)
    assert PolynomialPrimeLocalization.from_json(local.to_json()) == local
    assert PolynomialPrimeLocalElement.from_json(inverse_y.to_json()) == inverse_y


def test_colon_witness_certifies_membership_at_a_nonmaximal_prime() -> None:
    ambient = PolynomialRing(Q, ["x", "y"])
    x, y = ambient.generators()
    prime = ambient.ideal([x]).minimal_primes().prime_certificates[0]
    local = PolynomialPrimeLocalization(prime)
    local_x = local.from_polynomial(x)
    ideal = local.ideal([local.from_polynomial(x * y)])

    member = ideal.certify_membership(local_x)
    nonmember = ideal.certify_membership(local.one())

    assert member.is_member
    assert member.multiplier is not None
    assert member.multiplier.is_unit
    assert len(member.quotients) == len(ideal.generators)
    member.verify()
    assert ideal.contains(local_x)
    assert not nonmember.is_member
    assert nonmember.multiplier is None
    assert nonmember.quotients == []
    nonmember.verify()
    assert not ideal.contains(local.one())
    ideal.verify()
    assert PolynomialPrimeLocalIdeal.from_json(ideal.to_json()) == ideal
    assert PolynomialPrimeLocalMembership.from_json(member.to_json()) == member
