# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Count the spanning trees of a two-by-three grid."""

from noether import Z
from noether.combin import Graph
from noether.linalg import Matrix


def main() -> None:
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    grid = Graph(6, edges, name="2-by-3 grid")
    assert grid.edge_count == 7
    reduced = Matrix.from_rows(Z, grid.reduced_laplacian())
    assert reduced.det() == 15
    print("det(reduced Laplacian of the 2-by-3 grid) = 15")


if __name__ == "__main__":
    main()
