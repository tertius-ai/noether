# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""See a one-cycle appear and die in a filtered triangle."""

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
