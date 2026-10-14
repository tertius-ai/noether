# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the fivefold rotation of a FlexRiLoG framework graph."""

from noether.combin import Graph
from noether.groups import Permutation, PermutationGroup


def main() -> None:
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 14),
        (1, 2),
        (1, 11),
        (1, 12),
        (1, 15),
        (1, 18),
        (1, 23),
        (1, 24),
        (2, 3),
        (2, 7),
        (3, 4),
        (3, 5),
        (3, 7),
        (4, 5),
        (4, 14),
        (4, 15),
        (4, 16),
        (4, 17),
        (4, 20),
        (5, 6),
        (5, 10),
        (6, 7),
        (6, 8),
        (6, 10),
        (7, 8),
        (7, 17),
        (7, 18),
        (7, 19),
        (7, 22),
        (8, 9),
        (8, 13),
        (9, 10),
        (9, 11),
        (9, 13),
        (10, 11),
        (10, 19),
        (10, 20),
        (10, 21),
        (10, 24),
        (11, 12),
        (12, 13),
        (12, 14),
        (13, 14),
        (13, 16),
        (13, 21),
        (13, 22),
        (13, 23),
        (15, 16),
        (15, 18),
        (16, 23),
        (17, 18),
        (17, 20),
        (19, 20),
        (19, 22),
        (21, 22),
        (21, 24),
        (23, 24),
    ]
    graph = Graph.from_unweighted(25, edges, directed=False)
    rotation = Permutation.from_images(
        [
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            0,
            1,
            2,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            15,
            16,
        ]
    )
    cyclic = PermutationGroup.from_generators(25, [rotation])
    automorphisms = graph.automorphism_group()

    assert graph.edge_count == 60
    assert cyclic.order == 5
    assert cyclic.orbit(0) == [0, 3, 6, 9, 12]
    assert rotation in automorphisms
    assert automorphisms.order == 20
    print(
        "displayed symmetry order:",
        cyclic.order,
        "; full automorphism order:",
        automorphisms.order,
    )


if __name__ == "__main__":
    main()
