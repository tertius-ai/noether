# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the edge ideal of a four-cycle."""

from noether import Q
from noether.commalg import Degree, MonomialOrder, PolynomialRing
from noether.homalg import (
    FreeModule,
    ModuleHomomorphism,
    ModuleMonomialOrder,
    PresentedModule,
)
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, ["w", "x", "y", "z"])
    w, x, y, z = ring.generators()
    relations = [w * x, x * y, y * z, w * z]
    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(ring, 4, grading_rank=1, degree_shifts=[Degree([-2])] * 4)
    quotient = PresentedModule(
        ModuleHomomorphism(source, target, Matrix.from_rows(ring, [relations]))
    )
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    resolution = quotient.minimal_free_resolution(order)
    resolution.verify()
    assert resolution.is_minimal()
    betti = resolution.graded_betti_table()
    betti.verify()
    assert betti.entries == [
        (0, Degree([0]), 1),
        (1, Degree([2]), 4),
        (2, Degree([3]), 4),
        (3, Degree([4]), 1),
    ]
    totals = [betti.total(index) for index in range(4)]
    assert totals == [1, 4, 4, 1]
    print("four-cycle Betti totals:", totals)


if __name__ == "__main__":
    main()
