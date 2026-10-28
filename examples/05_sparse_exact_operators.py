# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Applying a sparse matrix

Construct a 3-by-3 integer matrix from four nonzero triplets, apply it
to `(5, 7, 11)`, and multiply it by itself.

## Result

The image is `[-1, 21, 20]`. The square has four nonzero entries, reported
by `nnz`. Triplets use zero-based row and column indices.
"""

from noether import Z
from noether.linalg import SparseMatrix, Vector


def main() -> None:
    operator = SparseMatrix.from_triplets(
        Z,
        (3, 3),
        [(0, 0, 2), (0, 2, -1), (1, 1, 3), (2, 0, 4)],
    )
    vector = Vector.from_values(Z, [5, 7, 11])
    image = operator.apply(vector)
    assert image.to_values() == [-1, 21, 20]
    assert (operator @ operator).nnz == 4
    print(f"A v = {image.to_values()}; nnz(A^2) = {(operator @ operator).nnz}")


if __name__ == "__main__":
    main()
