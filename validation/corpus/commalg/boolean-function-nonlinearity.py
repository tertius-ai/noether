# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a Boolean function's minimum distance from affine functions."""

from itertools import product

from noether.commalg import Polynomial, PolynomialRing
from noether.numthy import FiniteField, FiniteFieldElement


def truth_table(polynomial: Polynomial[FiniteField]) -> list[FiniteFieldElement]:
    field = polynomial.ring.coefficient_ring
    return [polynomial.evaluate([field(x), field(y)]) for x, y in product(range(2), repeat=2)]


def main() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x1", "x2"])
    x1, x2 = ring.generators()
    function = x1 * x2 + 1
    target = truth_table(function)
    distances = [
        sum(
            left != right
            for left, right in zip(target, truth_table(a * x1 + b * x2 + c), strict=True)
        )
        for a, b, c in product(range(2), repeat=3)
    ]
    assert min(distances) == 1
    assert sorted(distances) == [1, 1, 1, 1, 3, 3, 3, 3]
    print("the nonlinearity of x1*x2+1 is one")


if __name__ == "__main__":
    main()
