# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a familiar identity with real and complex balls."""

from fractions import Fraction

from noether import ComplexBallField, RealBallField


def main() -> None:
    reals = RealBallField(192)
    one_third = reals(Fraction(1, 3))
    assert (3 * one_third).overlaps(reals.one())

    complexes = ComplexBallField(192)
    imaginary_unit = complexes(0, 1)
    assert (imaginary_unit**2).overlaps(complexes(-1))
    assert (1 + imaginary_unit).overlaps(complexes(1, 1))
    print(f"1/3 = {one_third}; i^2 = {imaginary_unit**2}")


if __name__ == "__main__":
    main()
