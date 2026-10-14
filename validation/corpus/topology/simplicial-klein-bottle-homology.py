# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the integral homology of an eight-vertex Klein bottle."""

from noether.topology import SimplicialComplex


def main() -> None:
    facets = [
        [0, 1, 4],
        [0, 1, 6],
        [0, 2, 4],
        [0, 2, 5],
        [0, 5, 6],
        [1, 2, 3],
        [1, 2, 6],
        [1, 3, 5],
        [1, 4, 7],
        [1, 5, 7],
        [2, 3, 7],
        [2, 4, 6],
        [2, 5, 7],
        [3, 4, 6],
        [3, 4, 7],
        [3, 5, 6],
    ]
    klein_bottle = SimplicialComplex(facets)
    klein_bottle.verify()

    assert [klein_bottle.count(degree) for degree in range(3)] == [8, 24, 16]
    assert klein_bottle.euler_characteristic() == 0
    homology = [klein_bottle.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()
    assert [(group.free_rank, group.torsion_coefficients) for group in homology] == [
        (1, []),
        (1, [2]),
        (0, []),
    ]
    print("face counts: [8, 24, 16]; H_0 = Z, H_1 = Z + Z/2, H_2 = 0")


if __name__ == "__main__":
    main()
