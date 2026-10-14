# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the paper's weighted four-hyperplane RLCT example."""

from fractions import Fraction
from itertools import combinations

from noether import Q
from noether.linalg import Matrix


def row_space_key(rows: list[list[int]]) -> tuple[tuple[Fraction, ...], ...]:
    reduced = Matrix.from_rows(Q, rows).rref().to_rows()
    return tuple(tuple(row) for row in reduced if any(row))


def contained(
    left: tuple[tuple[Fraction, ...], ...], right: tuple[tuple[Fraction, ...], ...]
) -> bool:
    """Return whether rowspace(left) is contained in rowspace(right)."""
    if not left:
        return True
    stacked = [list(row) for row in right + left]
    return Matrix.from_rows(Q, stacked).rank() == len(right)


def main() -> None:
    normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    weights = [1, 2, 2, 1]
    row_spaces = {
        row_space_key([normals[i] for i in subset])
        for size in range(1, len(normals) + 1)
        for subset in combinations(range(len(normals)), size)
    }

    ratios: dict[tuple[tuple[Fraction, ...], ...], Fraction] = {}
    for space in row_spaces:
        total_weight = sum(
            weight
            for normal, weight in zip(normals, weights, strict=True)
            if contained(row_space_key([normal]), space)
        )
        ratios[space] = Fraction(len(space), total_weight)

    threshold = min(ratios.values())
    minimizing = [space for space, ratio in ratios.items() if ratio == threshold]
    lengths: dict[tuple[tuple[Fraction, ...], ...], int] = {}
    for space in sorted(minimizing, key=len, reverse=True):
        lengths[space] = 1 + max(
            (
                lengths[smaller_flat]
                for smaller_flat in lengths
                if len(smaller_flat) > len(space) and contained(space, smaller_flat)
            ),
            default=0,
        )
    multiplicity = max(lengths.values())

    assert len(row_spaces) == 11
    assert threshold == Fraction(1, 2)
    assert multiplicity == 3
    print("real log canonical threshold and multiplicity:", (threshold, multiplicity))


if __name__ == "__main__":
    main()
