# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify a diagonal symmetric representation of a reducible cubic."""

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    line = x + 3 * y - 7 * z
    doubled_line = x + 2 * y + 5 * z
    cubic = line * doubled_line**2
    pencil = Matrix.from_rows(
        ring,
        [[line, 0, 0], [0, doubled_line, 0], [0, 0, doubled_line]],
    )

    assert pencil.det() == cubic
    factorization = cubic.factor()
    factorization.verify()
    assert factorization.coefficient == Q(1)
    assert {(factor, exponent) for factor, exponent in factorization.factors} == {
        (line, 1),
        (doubled_line, 2),
    }
    print("the symmetric pencil has determinant (x+3y-7z)(x+2y+5z)^2")


if __name__ == "__main__":
    main()
