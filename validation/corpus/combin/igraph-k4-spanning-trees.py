# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Count the spanning trees of K4 by the Matrix-Tree theorem."""

from itertools import combinations

from noether import Z
from noether.combin import Graph
from noether.linalg import Matrix


def main() -> None:
    graph = Graph(4, list(combinations(range(4), 2)), name="K4")
    assert graph.edge_count == 6
    reduced = Matrix.from_rows(Z, graph.reduced_laplacian())
    assert reduced.det() == 16
    print("det(reduced Laplacian of K4) = 16")


if __name__ == "__main__":
    main()
