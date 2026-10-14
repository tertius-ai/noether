# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover the six quadratic pieces of a Sylvester denumerant."""

from fractions import Fraction

from noether import Q
from noether.linalg import Matrix, Vector


def denumerant(total: int) -> int:
    return sum(
        1
        for x in range(total // 6 + 1)
        for y in range(total // 2 + 1)
        for z in range(total // 3 + 1)
        if 6 * x + 2 * y + 3 * z == total
    )


def main() -> None:
    vandermonde = Matrix.from_rows(Q, [[1, k, k * k] for k in range(3)])
    pieces: list[list[Fraction]] = []
    for residue in range(6):
        values = [denumerant(residue + 6 * k) for k in range(3)]
        solution = vandermonde.solve(Vector.from_values(Q, values))
        solution.verify()
        coefficients = solution.particular.to_values()
        pieces.append(coefficients)
        for k in range(11):
            assert sum(coefficients[i] * k**i for i in range(3)) == denumerant(residue + 6 * k)
    assert pieces[1] == [Fraction(0), Fraction(1, 2), Fraction(1, 2)]
    assert all(piece == [Fraction(1), Fraction(3, 2), Fraction(1, 2)] for piece in pieces[2:])
    print("six exact quadratic residue-class pieces certified modulo 6")


if __name__ == "__main__":
    main()
