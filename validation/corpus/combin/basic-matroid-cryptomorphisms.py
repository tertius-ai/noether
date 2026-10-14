# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Recover elementary matroid data and minors from a basis presentation."""

from noether.combin import Matroid


def main() -> None:
    matroid = Matroid.from_bases([0, 1, 2, 3], [[0, 1], [0, 2]])

    assert matroid.ground_set == [0, 1, 2, 3]
    assert matroid.bases == (frozenset({0, 1}), frozenset({0, 2}))
    assert matroid.rank(matroid.ground_set) == 2
    assert matroid.rank([0, 3]) == 1
    assert set(matroid.circuits) == {frozenset({1, 2}), frozenset({3})}

    deletion = matroid.delete(3)
    assert deletion.ground_set == [0, 1, 2]
    assert deletion.bases == matroid.bases
    assert deletion.circuits == (frozenset({1, 2}),)

    contraction = matroid.contract(1)
    assert contraction.ground_set == [0, 2, 3]
    assert contraction.bases == (frozenset({0}),)
    assert contraction.rank(contraction.ground_set) == 1
    assert set(contraction.circuits) == {frozenset({2}), frozenset({3})}

    print("rank:", matroid.rank(matroid.ground_set))
    print("rank({0,3}):", matroid.rank([0, 3]))
    print("circuits:", matroid.circuits)
    print("delete 3 bases:", deletion.bases)
    print("contract 1 bases:", contraction.bases)


if __name__ == "__main__":
    main()
