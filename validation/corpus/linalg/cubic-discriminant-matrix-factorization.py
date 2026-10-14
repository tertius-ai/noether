# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the paper's cubic-discriminant matrix factorization."""

from fractions import Fraction

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, ["s1", "s2", "s3"])
    s1, s2, s3 = ring.generators()
    a = Matrix.from_rows(
        ring,
        [
            [-2 * s2, -2 * s1 * s3, -6 * s3],
            [4 * s1, s1 * s2 + 3 * s3, 4 * s2],
            [-6, -2 * s2, -2 * s1],
        ],
    )
    b = Matrix.from_rows(
        ring,
        [
            [
                Fraction(-1, 2) * s1**2 * s2 + 2 * s2**2 - Fraction(3, 2) * s1 * s3,
                -(s1**2) * s3 + 3 * s2 * s3,
                Fraction(-1, 2) * s1 * s2 * s3 + Fraction(9, 2) * s3**2,
            ],
            [
                2 * s1**2 - 6 * s2,
                s1 * s2 - 9 * s3,
                2 * s2**2 - 6 * s1 * s3,
            ],
            [
                Fraction(-1, 2) * s1 * s2 + Fraction(9, 2) * s3,
                -(s2**2) + 3 * s1 * s3,
                Fraction(-1, 2) * s1 * s2**2 + 2 * s1**2 * s3 - Fraction(3, 2) * s2 * s3,
            ],
        ],
    )
    discriminant = s1**2 * s2**2 - 4 * s1**3 * s3 - 4 * s2**3 + 18 * s1 * s2 * s3 - 27 * s3**2
    expected = Matrix.from_rows(
        ring,
        [
            [discriminant, 0, 0],
            [0, discriminant, 0],
            [0, 0, discriminant],
        ],
    )
    assert a @ b == expected
    assert b @ a == expected
    print("verified both products for discriminant:", discriminant)


if __name__ == "__main__":
    main()
