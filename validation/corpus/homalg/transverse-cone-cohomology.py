# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute cohomology of a reduced transverse-cone cochain complex."""

from noether import Q, RationalField
from noether.homalg import ChainComplex
from noether.linalg import Matrix


def reduced_differential(
    source_dimension: int,
    target_dimension: int,
    pivots: list[tuple[int, int]],
) -> Matrix[RationalField]:
    rows = [[0] * source_dimension for _ in range(target_dimension)]
    for target, source in pivots:
        rows[target][source] = 1
    return Matrix.from_rows(Q, rows)


def main() -> None:
    coboundaries = {
        -1: reduced_differential(1, 6, []),
        0: reduced_differential(6, 15, [(0, 0)]),
        1: reduced_differential(15, 20, [(0, 1), (1, 2), (2, 3)]),
        2: reduced_differential(20, 15, [(0, 3), (1, 4), (2, 5)]),
        3: reduced_differential(15, 6, [(0, 3)]),
        4: reduced_differential(6, 1, []),
    }
    complex_ = ChainComplex.from_coboundaries(Q, coboundaries)
    complex_.verify()

    assert complex_.degrees == [-1, 0, 1, 2, 3, 4, 5]
    assert [complex_.module_dimension(degree) for degree in complex_.degrees] == [
        1,
        6,
        15,
        20,
        15,
        6,
        1,
    ]

    cohomology = complex_.graded_cohomology()
    cohomology.verify()
    for group in cohomology.groups:
        group.verify()

    dimensions = {group.degree: group.dimension for group in cohomology.groups}
    assert dimensions == {-1: 1, 0: 5, 1: 11, 2: 14, 3: 11, 4: 5, 5: 1}

    print("cochain dimensions:", [1, 6, 15, 20, 15, 6, 1])
    print("differential ranks:", [0, 1, 3, 3, 1, 0])
    print("cohomology dimensions:", [dimensions[d] for d in complex_.degrees])


if __name__ == "__main__":
    main()
