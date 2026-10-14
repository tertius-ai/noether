# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the exact kernel of fourfold Hodge-diagonal constraints."""

from fractions import Fraction

from noether import Q
from noether.linalg import Matrix

DIMENSION = 4
REPRESENTATIVES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
]


def orbit_representative(p: int, q: int) -> tuple[int, int]:
    """Choose a representative modulo Hodge symmetry and Serre duality."""
    return min(
        (p, q),
        (q, p),
        (DIMENSION - p, DIMENSION - q),
        (DIMENSION - q, DIMENSION - p),
    )


def coefficient_row(positions: list[tuple[int, int]]) -> list[int]:
    row = [0] * len(REPRESENTATIVES)
    for position in positions:
        row[REPRESENTATIVES.index(orbit_representative(*position))] += 1
    return row


def main() -> None:
    boundary_rows = [coefficient_row([(0, q)]) for q in range(DIMENSION + 1)]
    diagonal_rows = [
        coefficient_row(
            [
                (p, q)
                for p in range(DIMENSION + 1)
                for q in range(DIMENSION + 1)
                if p - q == diagonal
            ]
        )
        for diagonal in range(-DIMENSION, DIMENSION + 1)
    ]
    constraints = Matrix.from_rows(Q, boundary_rows + diagonal_rows)

    assert constraints.shape == (14, 9)
    assert constraints.rank() == 8

    kernel = constraints.kernel()
    kernel.verify()
    assert kernel.dimension == 1
    assert kernel.ambient_dimension == 9

    generator = kernel.basis[0].to_values()
    assert generator == [
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(-1, 2),
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]

    print("constraint shape:", constraints.shape)
    print("rank, nullity:", constraints.rank(), kernel.dimension)
    print("integral kernel direction: d[1,1] = -1, d[2,2] = 2")


if __name__ == "__main__":
    main()
