# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the primary-operation input to a published secondary example."""

from collections import Counter

from noether.numthy import FiniteField
from noether.topology import SimplicialComplex


def main() -> None:
    complex_ = SimplicialComplex(
        [
            [1, 3, 7],
            [3, 4, 7],
            [1, 4, 7],
            [1, 2, 8],
            [2, 3, 8],
            [1, 3, 8],
            [4, 5, 9],
            [4, 6, 9],
            [5, 6, 9],
            [3, 4, 10],
            [3, 6, 10],
            [4, 6, 10],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 11],
            [1, 2, 3, 4, 6, 11],
            [1, 2, 3, 5, 6, 11],
            [1, 2, 4, 5, 6, 11],
            [1, 3, 4, 5, 6, 11],
            [2, 3, 4, 5, 6, 11],
        ]
    )
    complex_.verify()
    field = FiniteField.prime(2)

    betti_numbers = [complex_.betti_number(degree, ring=field) for degree in range(6)]
    assert betti_numbers == [1, 0, 4, 0, 0, 1]

    cohomology = complex_.cohomology_ring(field)
    cohomology.verify()
    assert Counter(item.degree for item in cohomology.basis) == {0: 1, 2: 4, 5: 1}
    degree_two = [index for index, item in enumerate(cohomology.basis) if item.degree == 2]
    assert len(degree_two) == 4
    assert all(cohomology.steenrod_square(2, source) == [] for source in degree_two)

    print("mod-two Betti numbers:", betti_numbers, "; ker Sq^2 on H^2 has dimension 4")


if __name__ == "__main__":
    main()
