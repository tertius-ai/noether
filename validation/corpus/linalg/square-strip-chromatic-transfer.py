# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute a square-strip chromatic polynomial by polynomial transfer matrix."""

from noether import Z
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Z, ["q"])
    q = ring.generator("q")
    transition = Matrix.from_rows(
        ring,
        [[-8 + 10 * q - 5 * q**2 + q**3, 5 - 4 * q + q**2], [1, -2 + q]],
    )
    left = Matrix.from_rows(ring, [[q * (q - 1) ** 2, q * (q - 1)]])
    right = Matrix.from_rows(ring, [[1], [0]])
    chromatic = (left @ transition @ transition @ right)[0, 0]

    assert chromatic.terms() == [
        ([9], 1),
        ([8], -12),
        ([7], 66),
        ([6], -216),
        ([5], 459),
        ([4], -648),
        ([3], 594),
        ([2], -323),
        ([1], 79),
    ]
    assert [chromatic.evaluate([colors]) for colors in (2, 3, 4)] == [2, 246, 9612]
    print("3-by-3 square-strip chromatic polynomial terms:", chromatic.nonzero_term_count)


if __name__ == "__main__":
    main()
