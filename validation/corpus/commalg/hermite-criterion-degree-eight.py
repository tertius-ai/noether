# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Expand one degree-eight Hermite-criterion polynomial over GF(2)."""

from itertools import product

from noether.commalg import PolynomialRing
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["a1", "a2", "a3", "a5", "a7"])
    a1, a2, a3, a5, a7 = ring.generators()
    coefficients = [0, a1, a2, a3, 1, a5, 0, a7]

    criterion = ring.zero()
    # Expand HC(5, 27, a7, 0, a5, 1, a3, a2, a1) exactly as in the source.
    binary_weights = [1, 2, 8, 16]
    for allocation in product(range(8), repeat=4):
        counts = [0] * 8
        for index, weight in zip(allocation, binary_weights, strict=True):
            counts[index] += weight
        weighted_degree = sum(index * counts[index] for index in range(1, 8)) + 8 * counts[0]
        if weighted_degree % 31 != 0 or weighted_degree > 8 * 31:
            continue
        term = ring.one()
        for coefficient, exponent in zip(coefficients, counts, strict=True):
            term *= coefficient**exponent
        criterion += term

    assert criterion.nonzero_term_count == 42
    assert criterion.degree() == 27
    assert criterion.coefficient([24, 2, 1, 0, 0]) == field.one()
    assert criterion.coefficient([0, 0, 0, 0, 26]) == field.one()
    assert criterion.coefficient([0, 0, 0, 17, 10]) == field.one()
    print("HC(5, 27, ...) nonzero terms after GF(2) cancellation:", criterion.nonzero_term_count)


if __name__ == "__main__":
    main()
