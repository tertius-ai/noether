# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Construct a four-element matroid and recover its circuits."""

from noether.combin import Matroid


def main() -> None:
    bases = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
    matroid = Matroid.from_bases([1, 2, 3, 4], bases)

    assert matroid.bases == tuple(frozenset(basis) for basis in bases)
    assert matroid.rank([1, 2, 3, 4]) == 2
    assert matroid.rank([3, 4]) == 1
    assert set(matroid.circuits) == {
        frozenset({3, 4}),
        frozenset({1, 2, 3}),
        frozenset({1, 2, 4}),
    }

    tutte = matroid.tutte_polynomial()
    expected_terms = {
        ((2, 0), 1),
        ((1, 1), 1),
        ((1, 0), 1),
        ((0, 2), 1),
        ((0, 1), 1),
    }
    assert {
        (tuple(exponent), coefficient) for exponent, coefficient in tutte.terms()
    } == expected_terms
    print("circuits:", [sorted(circuit) for circuit in matroid.circuits])


if __name__ == "__main__":
    main()
