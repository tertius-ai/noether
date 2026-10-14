# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Apply and multiply canonical sparse exact operators."""

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
