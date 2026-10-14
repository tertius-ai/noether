# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the homology of a five-element matroid independence complex."""

from noether.topology import SimplicialComplex


def main() -> None:
    bases = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 4, 5],
        [2, 3, 5],
        [2, 4, 5],
        [3, 4, 5],
    ]
    independence_complex = SimplicialComplex(bases)
    independence_complex.verify()

    assert [independence_complex.count(degree) for degree in range(3)] == [5, 10, 8]
    assert independence_complex.euler_characteristic() == 3
    homology = [independence_complex.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()
    assert [group.free_rank for group in homology] == [1, 0, 2]
    assert all(group.torsion_coefficients == [] for group in homology)
    print("face counts: [5, 10, 8]; H_0 = Z, H_1 = 0, H_2 = Z^2")


if __name__ == "__main__":
    main()
