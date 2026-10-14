# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the face counts and integral homology of a bowtie complex."""

from noether.topology import SimplicialComplex


def main() -> None:
    bowtie = SimplicialComplex([[0, 1, 2], [2, 3, 4]])
    bowtie.verify()

    assert bowtie.dimension == 2
    face_counts = [bowtie.count(degree) for degree in range(3)]
    assert face_counts == [5, 6, 2]
    assert bowtie.euler_characteristic() == 1

    homology = [bowtie.homology_invariants(degree) for degree in range(3)]
    for group in homology:
        group.verify()
    assert homology[0].free_rank == 1 and homology[0].torsion_coefficients == []
    assert homology[1].free_rank == 0 and homology[1].torsion_coefficients == []
    assert homology[2].free_rank == 0 and homology[2].torsion_coefficients == []

    print("face counts:", face_counts)
    print("Euler characteristic:", bowtie.euler_characteristic())
    print("homology:", homology)


if __name__ == "__main__":
    main()
