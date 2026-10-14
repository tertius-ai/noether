# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Check the ancillary G721 symmetric rank witness."""

from noether import Q
from noether.combin import Graph
from noether.linalg import Matrix


def main() -> None:
    rows = [
        [-1, 1, 0, 0, 1, 0, 1],
        [1, -1, 1, 0, 0, 1, 0],
        [0, 1, -1, 1, 0, -1, 0],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, -1, 1, 0, -1, 0],
        [1, 0, 0, 1, 0, 0, 0],
    ]
    witness = Matrix.from_rows(Q, rows)
    edges = [(i, j) for i in range(7) for j in range(i + 1, 7) if rows[i][j] != 0]
    graph = Graph.from_unweighted(7, edges, directed=False)

    assert witness == witness.transpose()
    assert witness.rank() == 3
    assert graph.edge_count == 10
    assert graph.degree_sequence() == [4, 3, 3, 3, 3, 2, 2]
    print("G721 witness rank and graph edges:", witness.rank(), graph.edge_count)


if __name__ == "__main__":
    main()
