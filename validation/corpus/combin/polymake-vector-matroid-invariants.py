# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the exact invariants of polymake's first vector matroid."""

from noether.combin import Matroid


def main() -> None:
    bases = [[0, 1, 2], [0, 2, 3], [1, 2, 3]]
    matroid = Matroid.from_bases([0, 1, 2, 3], bases)

    assert matroid.bases == tuple(frozenset(basis) for basis in bases)
    assert matroid.circuits == (frozenset({0, 1, 3}),)
    assert {
        (tuple(exponents), int(coefficient))
        for exponents, coefficient in matroid.tutte_polynomial().terms()
    } == {((3, 0), 1), ((2, 0), 1), ((1, 1), 1)}
    assert matroid.delete(0).bases == (frozenset({1, 2, 3}),)
    assert matroid.contract(0).bases == (
        frozenset({1, 2}),
        frozenset({2, 3}),
    )
    print("T(M) = x^3 + x^2 + xy, with deletion and contraction certified")


if __name__ == "__main__":
    main()
