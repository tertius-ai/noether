# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the published GF(11) degree-seven permutation witnesses."""

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    coefficient_tuples = [
        (0, 0, 0, 5, 0),
        (0, 0, 0, 8, 0),
        (0, 1, 0, 0, 4),
        (0, 1, 0, 8, 5),
        (0, 1, 0, 9, 5),
        (1, 0, 5, 0, 2),
        (1, 0, 5, 0, 7),
        (1, 0, 5, 8, 6),
        (1, 1, 5, 2, 5),
        (1, 2, 5, 9, 8),
        (1, 4, 5, 8, 0),
        (1, 8, 5, 8, 0),
        (1, 5, 5, 1, 5),
        (2, 0, 9, 0, 8),
        (2, 0, 9, 0, 9),
        (2, 0, 9, 4, 4),
        (2, 1, 9, 5, 3),
        (2, 2, 9, 5, 8),
        (2, 4, 9, 8, 3),
        (2, 8, 9, 5, 2),
        (2, 8, 9, 7, 8),
        (2, 8, 9, 8, 1),
        (2, 5, 9, 5, 2),
        (2, 5, 9, 6, 1),
        (2, 5, 9, 8, 3),
    ]
    field = FiniteField.prime(11)
    ring = PolynomialRing(field, ["x"])
    x = ring.generator("x")
    elements = [field(value) for value in range(11)]

    for a5, a4, a3, a2, a1 in coefficient_tuples:
        polynomial = x**7 + a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x
        assert len({polynomial.evaluate([value]) for value in elements}) == 11

    assert len(set(coefficient_tuples)) == 25
    print("verified published GF(11) degree-seven witnesses:", len(coefficient_tuples))


if __name__ == "__main__":
    main()
