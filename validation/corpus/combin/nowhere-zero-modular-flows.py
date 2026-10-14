# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Enumerate nowhere-zero flows by exact residue-ring incidence equations."""

from itertools import product

from noether.commalg import ResidueRing


def count_flows(
    vertex_count: int,
    edges: list[tuple[int, int]],
    modulus: int,
    orientation_mask: int = 0,
) -> int:
    ring = ResidueRing(modulus)
    count = 0
    for values in product(range(1, modulus), repeat=len(edges)):
        divergence = [ring.zero() for _ in range(vertex_count)]
        for edge_index, ((left, right), value) in enumerate(zip(edges, values, strict=True)):
            if (orientation_mask >> edge_index) & 1:
                left, right = right, left
            divergence[left] = divergence[left] - ring(value)
            divergence[right] = divergence[right] + ring(value)
        if all(value.is_zero() for value in divergence):
            count += 1
    return count


def main() -> None:
    complete_four = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    assert count_flows(4, complete_four, 3) == 0
    assert count_flows(4, complete_four, 4) == 6
    assert {count_flows(4, complete_four, 4, mask) for mask in range(1 << 6)} == {6}

    theta_three = [(0, 1), (0, 1), (0, 1)]
    assert count_flows(2, theta_three, 3) == 2
    assert count_flows(2, theta_three, 4) == 6
    print("K4 has six nowhere-zero Z/4 flows for all 64 orientations")


if __name__ == "__main__":
    main()
