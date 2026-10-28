# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Persistence of a triangular loop

Add the vertices and edges of a triangle at the listed filtration
values, then fill its 2-simplex at value 3. Compute persistent homology
over 𝔽₂.

## Result

The one-dimensional class is born at 2, when the last edge completes
the loop, and dies at 3, when the triangle is filled. At 2.5 the Betti
numbers are `[1, 1]`: one connected component and one loop.
"""

from noether.numthy import FiniteField
from noether.topology import FilteredComplex


def main() -> None:
    filtration = FilteredComplex(
        [
            ([0], 0.0),
            ([1], 0.0),
            ([2], 0.0),
            ([0, 1], 1.0),
            ([0, 2], 1.0),
            ([1, 2], 2.0),
            ([0, 1, 2], 3.0),
        ]
    )
    filtration.verify()
    barcode = filtration.persistent_homology(ring=FiniteField.prime(2))
    assert (1, 2.0, 3.0) in barcode
    assert barcode.betti_at(2.5) == [1, 1]
    print(f"barcode: {barcode.bars}")


if __name__ == "__main__":
    main()
