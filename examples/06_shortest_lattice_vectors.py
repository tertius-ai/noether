# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Shortest vectors in the A₂ lattice

Describe the A₂ lattice by its Gram matrix `[[2, -1], [-1, 2]]` and
enumerate all its shortest nonzero vectors.

## Result

There are six signed shortest vectors, each of squared norm 2.
`verify(lattice)` checks the enumeration against the specified lattice.
Opposite vectors count separately.
"""

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
