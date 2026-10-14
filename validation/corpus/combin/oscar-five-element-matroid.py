# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover circuits of an OSCAR five-element matroid."""

from noether.combin import Matroid


def main() -> None:
    bases = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 4],
        [1, 4, 5],
        [2, 3, 5],
        [2, 4, 5],
        [3, 4, 5],
    ]
    matroid = Matroid.from_bases([1, 2, 3, 4, 5], bases)

    assert matroid.rank(matroid.ground_set) == 3
    assert matroid.bases == tuple(frozenset(basis) for basis in bases)
    assert set(matroid.circuits) == {
        frozenset({1, 3, 5}),
        frozenset({2, 3, 4}),
        frozenset({1, 2, 4, 5}),
    }
    print("circuits:", [sorted(circuit) for circuit in matroid.circuits])


if __name__ == "__main__":
    main()
