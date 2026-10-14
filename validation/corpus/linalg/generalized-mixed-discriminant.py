# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the generic 3-by-3 mixed-discriminant coefficient identity."""

from itertools import permutations

from noether import Q, RationalField
from noether.commalg import Polynomial, PolynomialRing
from noether.linalg import Matrix


def permutation_sign(values: tuple[int, ...]) -> int:
    inversions = sum(
        values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def main() -> None:
    entry_names = [f"{letter}{i}{j}" for letter in "abc" for i in range(3) for j in range(3)]
    ring = PolynomialRing(Q, [*entry_names, "x", "y", "z"])
    generators = ring.generators()
    matrices = [
        [generators[offset + 3 * i : offset + 3 * i + 3] for i in range(3)] for offset in (0, 9, 18)
    ]
    x, y, z = generators[-3:]
    pencil_rows: list[list[Polynomial[RationalField] | int]] = []
    for i in range(3):
        pencil_rows.append(
            [
                (1 if i == j else 0)
                + x * matrices[0][i][j]
                + y * matrices[1][i][j]
                + z * matrices[2][i][j]
                for j in range(3)
            ]
        )
    determinant = Matrix.from_rows(ring, pencil_rows).det()
    xyz_part = ring.from_terms(
        (exponents, coefficient)
        for exponents, coefficient in determinant.terms()
        if tuple(exponents[-3:]) == (1, 1, 1)
    )

    mixed = ring.zero()
    for columns in permutations(range(3)):
        sign = permutation_sign(columns)
        for colors in permutations(range(3)):
            term = ring.one()
            for row in range(3):
                term *= matrices[colors[row]][row][columns[row]]
            mixed += sign * term

    assert xyz_part == x * y * z * mixed
    assert len(mixed.terms()) == 36
    print("the xyz determinant coefficient equals the 36-term mixed discriminant")


if __name__ == "__main__":
    main()
