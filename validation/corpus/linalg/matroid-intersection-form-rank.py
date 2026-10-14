# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the rank and determinant of a matroid intersection form."""

from noether import Q
from noether.linalg import Matrix


def main() -> None:
    matrix = Matrix.from_rows(
        Q,
        [
            [-1, 0, 0, 0, 0, 0, 0, 1],
            [0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 1],
            [0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 1],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0],
            [1, 0, 1, 0, 1, 0, 0, -2],
        ],
    )
    assert matrix.rank() == 8
    assert matrix.det() == -1
    print("rank = 8; determinant = -1")


if __name__ == "__main__":
    main()
