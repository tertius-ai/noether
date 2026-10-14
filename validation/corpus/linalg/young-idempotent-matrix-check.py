# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Check the exact Young-symmetrizer matrix identities printed in the paper."""

from noether import Q
from noether.linalg import Matrix


def main() -> None:
    ys = Matrix.from_rows(Q, [[0, 0, 0, 0], [0, 0, 0, 0], [-30, 0, 30, 0], [0, 0, 0, 0]])
    adjoint = Matrix.from_rows(Q, [[6, 6, -24, 6], [0, 0, 0, 0], [-6, -6, 24, -6], [0, 0, 0, 0]])
    f = Matrix.from_rows(
        Q, [[0, 0, 0, 0], [0, 0, 0, 0], [-Q(1) / 4, -Q(1) / 4, 1, -Q(1) / 4], [0, 0, 0, 0]]
    )
    assert (ys @ ys).to_rows() == [[30 * x for x in row] for row in ys.to_rows()]
    assert (adjoint @ adjoint).to_rows() == [[30 * x for x in row] for row in adjoint.to_rows()]
    assert (ys @ adjoint).to_rows() == [[1440 * x for x in row] for row in f.to_rows()]
    assert (f @ f).to_rows() == f.to_rows()
    assert (f @ ys).to_rows() == ys.to_rows()
    assert (ys @ f).to_rows() == [[30 * x for x in row] for row in f.to_rows()]
    print("verified the displayed rank-one idempotent")


if __name__ == "__main__":
    main()
