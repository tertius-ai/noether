# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the Tutte polynomial of the K4 graphic matroid."""

from noether.combin import Matroid


def main() -> None:
    bases = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 3],
        [0, 2, 4],
        [0, 2, 5],
        [0, 3, 5],
        [0, 4, 5],
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 4, 5],
        [2, 3, 4],
        [2, 3, 5],
        [3, 4, 5],
    ]
    matroid = Matroid.from_bases(range(6), bases)
    assert matroid.rank(matroid.ground_set) == 3
    assert len(matroid.bases) == 16

    tutte = matroid.tutte_polynomial()
    x, y = tutte.ring.generators()
    assert tutte == x**3 + 3 * x**2 + 4 * x * y + 2 * x + y**3 + 3 * y**2 + 2 * y
    print("T(K4) = x^3 + 3x^2 + 4xy + 2x + y^3 + 3y^2 + 2y")


if __name__ == "__main__":
    main()
