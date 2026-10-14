# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Show that every 6-by-6 minor of the semi-Clifford matrix vanishes."""

from itertools import combinations

from noether import Q
from noether.commalg import PolynomialRing
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, ["q11", "q12", "q21", "q22", "q31", "q32", "q41", "q42"])
    q11, q12, q21, q22, q31, q32, q41, q42 = ring.generators()
    rows = [
        [q21, q22, -q11, -q12, 0, 0, 0, 0],
        [q31, q32, 0, 0, -q11, -q12, 0, 0],
        [q41, q42, 0, 0, 0, 0, -q11, -q12],
        [0, 0, q31, q32, -q21, -q22, 0, 0],
        [0, 0, q41, q42, 0, 0, -q21, -q22],
        [0, 0, 0, 0, q41, q42, -q31, -q32],
    ]
    maximal_minors = [
        Matrix.from_rows(ring, [[row[column] for column in columns] for row in rows]).det()
        for columns in combinations(range(8), 6)
    ]
    assert len(maximal_minors) == 28
    assert all(minor.is_zero() for minor in maximal_minors)
    print("all 28 maximal minors vanish, so the symbolic matrix has rank at most five")


if __name__ == "__main__":
    main()
