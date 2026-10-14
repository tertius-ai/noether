# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the primes in (7,14] as polynomial support over Z/143Z."""

from math import comb, factorial, gcd

from noether.commalg import PolynomialRing, ResidueRing


def main() -> None:
    n = 7
    central = comb(2 * n, n)
    modulus = central // gcd(central, factorial(n) ** 2)
    assert (central, modulus) == (3432, 143)
    coefficients = ResidueRing(modulus)
    ring = PolynomialRing(coefficients, ["X"])
    x = ring.generator("X")
    frobenius_remainder = (1 + x) ** modulus - 1
    support = [
        (exponent[0], coefficient.representative)
        for exponent, coefficient in frobenius_remainder.terms()
        if exponent[0] <= 2 * n
    ]
    assert support == [(13, 11), (11, 13)]
    assert {degree: modulus // gcd(modulus, coefficient) for degree, coefficient in support} == {
        11: 11,
        13: 13,
    }
    print("prime support =", sorted(degree for degree, _ in support))


if __name__ == "__main__":
    main()
