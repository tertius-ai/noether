# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Automorphisms of a cycle graph

Construct the undirected 5-cycle, compute a canonical labeling, and
inspect its automorphism group and graph Laplacian.

## Result

The automorphism group has order 10: five rotations and five reflections.
The first Laplacian row has diagonal entry 2 and entries −1 at the
two neighbors of vertex 0.
"""

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
