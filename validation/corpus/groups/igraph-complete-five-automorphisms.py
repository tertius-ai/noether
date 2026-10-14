# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the full automorphism group of K5."""

from noether.combin import Graph


def main() -> None:
    complete = Graph.from_unweighted(
        5,
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
        directed=False,
    )
    group = complete.automorphism_group()
    evidence = group.orbit_stabilizer(0)

    assert group.order == 120
    assert group.is_transitive
    assert len(evidence.orbit) == 5
    assert evidence.stabilizer.order == 24
    assert group.order == len(evidence.orbit) * evidence.stabilizer.order
    print("|Aut(K5)| =", group.order)


if __name__ == "__main__":
    main()
