# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the integral homology of a seven-vertex torus."""

from noether.topology import SimplicialComplex


def main() -> None:
    facets = [
        [0, 1, 2],
        [0, 1, 5],
        [0, 2, 6],
        [0, 3, 4],
        [0, 3, 5],
        [0, 4, 6],
        [1, 2, 4],
        [1, 3, 4],
        [1, 3, 6],
        [1, 5, 6],
        [2, 3, 5],
        [2, 3, 6],
        [2, 4, 5],
        [4, 5, 6],
    ]
    torus = SimplicialComplex(facets)
    torus.verify()

    assert [torus.count(degree) for degree in range(3)] == [7, 21, 14]
    assert torus.euler_characteristic() == 0
    homology = [torus.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()
    assert [(group.free_rank, group.torsion_coefficients) for group in homology] == [
        (1, []),
        (2, []),
        (1, []),
    ]
    print("face counts: [7, 21, 14]; H_0 = Z, H_1 = Z^2, H_2 = Z")


if __name__ == "__main__":
    main()
