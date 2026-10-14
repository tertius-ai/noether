# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct two small lattices from the HDR notebook."""

from itertools import permutations

from noether.combin import Poset


def inversions(permutation: tuple[int, ...]) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
        if permutation[i] > permutation[j]
    }


def main() -> None:
    symmetric_three = list(permutations(range(3)))
    weak_relations = [
        (i, j)
        for i, left in enumerate(symmetric_three)
        for j, right in enumerate(symmetric_three)
        if i != j and inversions(left) < inversions(right)
    ]
    weak_order = Poset(6, weak_relations)
    tamari = Poset(5, [(0, 1), (1, 3), (3, 4), (0, 2), (2, 4)])

    assert weak_order.is_lattice() and tamari.is_lattice()
    assert (weak_order.height(), weak_order.width()) == (4, 2)
    assert (tamari.height(), tamari.width()) == (4, 2)
    assert weak_order.covers.edge_count == 6
    assert tamari.covers.edge_count == 5
    print("weak order S3: 6 elements; Tamari T3: 5 elements; both are lattices")


if __name__ == "__main__":
    main()
