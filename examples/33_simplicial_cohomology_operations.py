# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the mod-two cohomology ring of RP2 and its Steenrod square."""

from noether.numthy import FiniteField
from noether.topology import AdmissibleCohomologyOperation, SimplicialComplex


def main() -> None:
    projective_plane = SimplicialComplex(
        [
            [0, 1, 2],
            [0, 1, 4],
            [0, 2, 3],
            [0, 3, 5],
            [0, 4, 5],
            [1, 2, 5],
            [1, 3, 4],
            [1, 3, 5],
            [2, 3, 4],
            [2, 4, 5],
        ]
    )
    cohomology = projective_plane.cohomology_ring(FiniteField.prime(2))
    cohomology.verify()
    assert [basis.degree for basis in cohomology.basis] == [0, 1, 2]
    assert cohomology.product(1, 1)[0].basis == 2

    square = AdmissibleCohomologyOperation.steenrod_square(1)
    image = cohomology.apply_operation(square, 1)
    assert image[0].basis == 2 and int(image[0].coefficient) == 1
    print(f"H*(RP2; F2) degrees={[basis.degree for basis in cohomology.basis]}; Sq^1(h)=h^2")


if __name__ == "__main__":
    main()
