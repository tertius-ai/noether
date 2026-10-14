# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the integral homology of a nine-vertex CP2."""

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

    assert [projective_plane.count(degree) for degree in range(5)] == [
        9,
        36,
        84,
        90,
        36,
    ]
    assert projective_plane.euler_characteristic() == 3
    homology = [projective_plane.homology_invariants(degree) for degree in range(5)]
    for group in homology:
        group.verify()
    assert [group.free_rank for group in homology] == [1, 0, 1, 0, 1]
    assert all(group.torsion_coefficients == [] for group in homology)
    print("face counts: [9, 36, 84, 90, 36]; homology ranks: [1, 0, 1, 0, 1]")


if __name__ == "__main__":
    main()
