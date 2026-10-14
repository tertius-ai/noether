# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the lexicographic elimination basis of two plane quadrics."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["y", "x"], order="lex")
    y, x = ring.generators()
    ideal = ring.ideal(
        [
            2 * x**2 - x * y + 2 * y**2 - 2,
            2 * x**2 - 3 * x * y + 3 * y**2 - 2,
        ]
    )

    basis = ideal.groebner_basis()
    expected = [
        y + Fraction(8, 3) * x**3 - Fraction(8, 3) * x,
        x**4 - Fraction(5, 4) * x**2 + Fraction(1, 4),
    ]
    assert basis == expected
    assert all(polynomial in ideal for polynomial in expected)
    assert ideal == ring.ideal(expected)
    print("lexicographic Groebner basis degrees:", [item.degree() for item in basis])


if __name__ == "__main__":
    main()
