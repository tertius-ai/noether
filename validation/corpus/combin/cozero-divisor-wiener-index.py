# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the paper's Wiener index for the cozero-divisor graph of Z/100Z."""

from math import gcd

from noether.combin import Graph


def main() -> None:
    modulus = 100
    values = [a for a in range(1, modulus) if gcd(a, modulus) != 1]
    ideals = [gcd(a, modulus) for a in values]
    edges = []
    for i in range(len(values)):
        for j in range(i):
            if ideals[i] % ideals[j] and ideals[j] % ideals[i]:
                edges.append((i, j))
    graph = Graph(len(values), edges, "cozero(Z/100Z)")
    assert graph.connected_components().count == 1
    wiener = sum(graph.shortest_path(i, j).distance for i in range(len(values)) for j in range(i))
    assert wiener == 2954
    print("W(cozero(Z/100Z)) =", wiener)


if __name__ == "__main__":
    main()
