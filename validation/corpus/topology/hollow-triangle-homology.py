# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the integral homology of a hollow triangle."""

from noether.topology import SimplicialComplex


def main() -> None:
    circle = SimplicialComplex([[1, 2], [2, 3], [1, 3]])
    circle.verify()

    assert circle.dimension == 1
    assert [circle.count(degree) for degree in range(2)] == [3, 3]
    assert circle.euler_characteristic() == 0

    homology = [circle.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()

    assert homology[0].free_rank == 1
    assert homology[0].torsion_coefficients == []
    assert homology[1].free_rank == 1
    assert homology[1].torsion_coefficients == []
    assert homology[2].free_rank == 0
    assert homology[2].torsion_coefficients == []

    print("simplex counts:", [3, 3])
    print("Euler characteristic:", 0)
    print("H_0 = Z, H_1 = Z, H_2 = 0")


if __name__ == "__main__":
    main()
