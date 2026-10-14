# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute and verify the integral homology of a triangulated RP^2."""

from noether import Z
from noether.topology import SimplicialComplex


def main() -> None:
    facets = [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [1, 5, 4],
        [5, 2, 1],
        [5, 6, 2],
        [4, 6, 2],
        [1, 6, 4],
        [3, 6, 5],
        [1, 6, 3],
    ]
    projective_plane = SimplicialComplex(facets)
    projective_plane.verify()

    assert [projective_plane.count(degree) for degree in range(3)] == [6, 15, 10]
    assert projective_plane.euler_characteristic() == 1

    chains = projective_plane.chain_complex(Z)
    chains.verify()
    assert chains.degrees == [0, 1, 2]
    assert [chains.module_dimension(degree) for degree in chains.degrees] == [6, 15, 10]

    homology = [projective_plane.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()

    assert homology[0].free_rank == 1
    assert homology[0].torsion_coefficients == []
    assert homology[1].free_rank == 0
    assert homology[1].torsion_coefficients == [2]
    assert homology[2].free_rank == 0
    assert homology[2].torsion_coefficients == []

    print("H_0 = Z")
    print("H_1 = Z/2")
    print("H_2 = 0")


if __name__ == "__main__":
    main()
