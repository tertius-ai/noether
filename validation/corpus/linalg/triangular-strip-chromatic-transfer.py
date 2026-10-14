# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a triangular-strip chromatic polynomial by polynomial transfer matrix."""

from noether import Z
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Z, ["q"])
    q = ring.generator("q")
    transition = Matrix.from_rows(
        ring,
        [[-14 + 17 * q - 7 * q**2 + q**3, 9 - 6 * q + q**2], [2 - q, -3 + q]],
    )
    left = Matrix.from_rows(ring, [[q * (q - 1) ** 2, q * (q - 1)]])
    right = Matrix.from_rows(ring, [[1], [0]])
    chromatic = (left @ transition @ transition @ right)[0, 0]

    assert chromatic.terms() == [
        ([9], 1),
        ([8], -16),
        ([7], 112),
        ([6], -448),
        ([5], 1120),
        ([4], -1791),
        ([3], 1786),
        ([2], -1012),
        ([1], 248),
    ]
    assert [chromatic.evaluate([colors]) for colors in (2, 3, 4)] == [0, 6, 1056]
    print("3-by-3 triangular-strip chromatic polynomial terms:", chromatic.nonzero_term_count)


if __name__ == "__main__":
    main()
