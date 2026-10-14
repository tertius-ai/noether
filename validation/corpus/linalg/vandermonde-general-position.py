# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify every maximal minor of the paper's Vandermonde matrix."""

from itertools import combinations

from noether import Q
from noether.linalg import Matrix


def main() -> None:
    parameters = list(range(1, 9))
    checked = 0

    for columns in combinations(parameters, 4):
        minor = Matrix.from_rows(Q, [[value**power for value in columns] for power in range(4)])
        determinant = minor.det()
        vandermonde = 1
        for left, right in combinations(columns, 2):
            vandermonde *= right - left
        assert determinant == vandermonde
        assert determinant != 0
        checked += 1

    assert checked == 70
    print("certified 70 nonzero maximal minors")


if __name__ == "__main__":
    main()
