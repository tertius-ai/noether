# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a restricted Grassmannian secant dimension from exact cofactors."""

from itertools import combinations

from noether import Q
from noether.linalg import Matrix


def cofactor(matrix: list[list[int]], row: int, column: int, subset: tuple[int, ...]) -> int:
    if column not in subset:
        return 0
    position = subset.index(column)
    rows = [index for index in range(3) if index != row]
    columns = [index for index in subset if index != column]
    determinant = (
        matrix[rows[0]][columns[0]] * matrix[rows[1]][columns[1]]
        - matrix[rows[0]][columns[1]] * matrix[rows[1]][columns[0]]
    )
    return (-1) ** (row + position) * determinant


def jacobian(first: list[list[int]], second: list[list[int]]) -> Matrix:
    rows = []
    for subset in combinations(range(7), 3):
        shared = [
            cofactor(first, 0, column, subset) + cofactor(second, 0, column, subset)
            for column in range(7)
        ]
        first_only = [cofactor(first, row, column, subset) for row in (1, 2) for column in range(7)]
        second_only = [
            cofactor(second, row, column, subset) for row in (1, 2) for column in range(7)
        ]
        rows.append(shared + first_only + second_only)
    return Matrix.from_rows(Q, rows)


def main() -> None:
    ranks = []
    for shift in (0, 1, 2):
        parameters = list(range(1 + shift, 8 + shift))
        first = [[1] * 7, parameters, [value**2 for value in parameters]]
        second = [
            [1] * 7,
            [value**3 + 1 for value in parameters],
            [value**4 + value for value in parameters],
        ]
        matrix = jacobian(first, second)
        assert matrix.shape == (35, 35)
        ranks.append(matrix.rank())

    assert ranks == [20, 20, 20]
    print("deterministic restricted-secant Jacobian ranks:", ranks)


if __name__ == "__main__":
    main()
