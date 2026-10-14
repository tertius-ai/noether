# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the integral homology of a triangulated Mobius strip."""

from noether.topology import SimplicialComplex


def main() -> None:
    mobius_strip = SimplicialComplex([[0, 1, 3], [1, 2, 3], [2, 3, 4], [0, 2, 4], [0, 1, 4]])
    mobius_strip.verify()

    assert [mobius_strip.count(degree) for degree in range(3)] == [5, 10, 5]
    assert mobius_strip.euler_characteristic() == 0
    homology = [mobius_strip.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()
    assert [(group.free_rank, group.torsion_coefficients) for group in homology] == [
        (1, []),
        (1, []),
        (0, []),
    ]
    print("face counts: [5, 10, 5]; H_0 = Z, H_1 = Z, H_2 = 0")


if __name__ == "__main__":
    main()
