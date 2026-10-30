# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Projection of shared univariate Euclidean operations."""

import pytest

from noether import ContextMismatchError, Q
from noether.commalg import PolynomialRing, ResidueRing
from noether.numthy import FiniteField


def test_bezout_and_inverse() -> None:
    ring = PolynomialRing(Q, ["x"])
    (x,) = ring.generators()
    f, modulus = x**3 - 1, x**2 + 1
    gcd, a, b = f.extended_gcd(modulus)
    assert gcd == a * f + b * modulus == ring.one()
    inverse = f.inverse_mod(modulus)
    assert (f * inverse).div_rem(modulus)[1] == ring.one()
    with pytest.raises(ValueError):
        f.inverse_mod(f)
    other = PolynomialRing(FiniteField.prime(3), ["x"])
    with pytest.raises(ContextMismatchError):
        f.extended_gcd(other.one())
    other = PolynomialRing(Q, ["z"])
    with pytest.raises(ContextMismatchError):
        f.inverse_mod(other.one())


def test_monic_division_over_composite_residues() -> None:
    ring = PolynomialRing(ResidueRing(9), ["x"])
    (x,) = ring.generators()
    f, divisor = x**7 + 3 * x + 1, x**2 + 1
    quotient, remainder = f.monic_div_rem(divisor)
    assert quotient * divisor + remainder == f
    assert remainder.degree() < divisor.degree()
    with pytest.raises(ValueError):
        f.monic_div_rem(3 * x + 1)
