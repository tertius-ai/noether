# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Cohomology and Steenrod squares of ℝP²

Build a six-vertex triangulation of the real projective plane and
compute its cohomology ring over 𝔽₂. Apply Sq¹ to the degree-1 class.

## Result

The ring is 𝔽₂[h]/(h³), with h of degree 1. Its basis has degrees
0, 1, and 2; the product h·h is the degree-2 class, and Sq¹(h) = h².
"""

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
