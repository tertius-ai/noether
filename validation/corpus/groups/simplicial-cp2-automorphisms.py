# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the automorphism group of a minimal CP2 triangulation."""

from noether.groups import Permutation, PermutationGroup
from noether.topology import SimplicialComplex

FACETS = [
    [0, 1, 2, 6, 7],
    [0, 1, 2, 6, 8],
    [0, 1, 2, 7, 8],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 8],
    [0, 1, 3, 5, 6],
    [0, 1, 3, 6, 8],
    [0, 1, 4, 5, 7],
    [0, 1, 4, 7, 8],
    [0, 1, 5, 6, 7],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
    [0, 2, 3, 5, 7],
    [0, 2, 3, 6, 7],
    [0, 2, 4, 5, 8],
    [0, 2, 4, 6, 8],
    [0, 2, 5, 7, 8],
    [0, 3, 4, 6, 8],
    [0, 3, 5, 6, 7],
    [0, 4, 5, 7, 8],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 7],
    [1, 2, 3, 5, 8],
    [1, 2, 3, 7, 8],
    [1, 2, 4, 5, 6],
    [1, 2, 4, 6, 7],
    [1, 2, 5, 6, 8],
    [1, 3, 4, 7, 8],
    [1, 3, 5, 6, 8],
    [1, 4, 5, 6, 7],
    [2, 3, 4, 6, 7],
    [2, 3, 5, 7, 8],
    [2, 4, 5, 6, 8],
    [3, 4, 6, 7, 8],
    [3, 5, 6, 7, 8],
    [4, 5, 6, 7, 8],
]


def main() -> None:
    projective_plane = SimplicialComplex(FACETS)
    projective_plane.verify()
    generators = [
        Permutation.from_images([0, 1, 2, 5, 3, 4, 7, 8, 6]),
        Permutation.from_images([0, 2, 1, 3, 5, 4, 6, 8, 7]),
        Permutation.from_images([1, 0, 2, 5, 4, 3, 6, 8, 7]),
        Permutation.from_images([3, 4, 5, 6, 7, 8, 0, 1, 2]),
    ]
    displayed_group = PermutationGroup.from_generators(9, generators)
    computed_group = projective_plane.automorphism_group()

    assert displayed_group.order == 54
    assert computed_group.order == 54
    assert all(generator in computed_group for generator in generators)
    print("CP2 triangulation automorphism group order:", computed_group.order)


if __name__ == "__main__":
    main()
