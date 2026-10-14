# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Enumerate and verify the six shortest vectors of the A2 lattice."""

from noether import Z
from noether.linalg import Lattice, Matrix


def main() -> None:
    lattice = Lattice(Matrix.from_rows(Z, [[2, -1], [-1, 2]]))
    shortest = lattice.shortest_vectors()
    shortest.verify(lattice)
    assert shortest.signed_count == 6
    assert shortest.max_norm == 2
    print(f"A2 has {shortest.signed_count} shortest vectors of norm {shortest.max_norm}")


if __name__ == "__main__":
    main()
