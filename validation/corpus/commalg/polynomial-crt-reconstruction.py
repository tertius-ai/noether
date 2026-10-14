# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reconstruct x/1234 - 1/5 from its two modular images."""

from fractions import Fraction
from math import isqrt

from noether.commalg import ResidueRing


def rational_reconstruct(residue: int, modulus: int) -> Fraction:
    old, current = modulus, residue
    old_denominator, denominator = 0, 1
    bound = isqrt(modulus // 2)
    while abs(current) > bound:
        quotient = old // current
        old, current = current, old - quotient * current
        old_denominator, denominator = denominator, old_denominator - quotient * denominator
    return Fraction(current, denominator)


def main() -> None:
    modulus = 12347 * 23459
    decomposition = ResidueRing(modulus).crt()
    decomposition.verify()
    factors = decomposition.factors
    x_coefficient = decomposition.reconstruct([factors[0](-5293), factors[1](-1806)])
    constant = decomposition.reconstruct([factors[0](-4939), factors[1](-4692)])
    assert x_coefficient.representative == 79571122
    assert constant.representative == 115859309
    assert rational_reconstruct(x_coefficient.representative, modulus) == Fraction(1, 1234)
    assert rational_reconstruct(constant.representative, modulus) == Fraction(-1, 5)
    print("reconstructed (1/1234)*x - 1/5")


if __name__ == "__main__":
    main()
