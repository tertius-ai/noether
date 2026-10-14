# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonically label a cycle and compute its automorphism group."""

from noether.combin import Graph


def main() -> None:
    cycle = Graph.from_unweighted(
        5,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        directed=False,
    )
    canonical = cycle.canonical_label()
    assert canonical.verify(cycle)
    assert canonical.automorphism_group.order == 10
    assert cycle.laplacian()[0] == [2, -1, 0, 0, -1]
    print(f"Aut(C5) has order {canonical.automorphism_group.order}")


if __name__ == "__main__":
    main()
