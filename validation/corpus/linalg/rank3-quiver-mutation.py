# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Replay an exact alternating mutation sequence for a rank-three quiver."""

from noether import Z
from noether.linalg import Matrix


def quiver_matrix(p: int, q: int, r: int):
    return Matrix.from_rows(Z, [[0, p, -r], [-p, 0, q], [r, -q, 0]])


def main() -> None:
    p, q, r = 3, 2, 1
    markov = p * p + q * q + r * r - p * q * r
    weights = [(q, r)]
    for _ in range(8):
        next_r = p * q - r
        next_q = p * next_r - q
        q, r = next_q, next_r
        weights.append((q, r))
        matrix = quiver_matrix(p, q, r)
        assert matrix.transpose().to_rows() == [[-x for x in row] for row in matrix.to_rows()]
        assert p * p + q * q + r * r - p * q * r == markov
    assert weights[:4] == [(2, 1), (13, 5), (89, 34), (610, 233)]
    assert all(weights[i][0] > weights[i][1] > weights[i - 1][0] for i in range(1, len(weights)))
    print("final weights =", (p, q, r), "; Markov constant =", markov)


if __name__ == "__main__":
    main()
