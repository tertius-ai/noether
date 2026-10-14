# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the S4-stable ideal of squarefree quadrics."""

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
    ring = PolynomialRing(Q, ["x1", "x2", "x3", "x4"])
    variables = ring.generators()
    relations = [variables[i] * variables[j] for i in range(4) for j in range(i + 1, 4)]
    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(ring, 6, grading_rank=1, degree_shifts=[Degree([-2])] * 6)
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
        (1, Degree([2]), 6),
        (2, Degree([3]), 8),
        (3, Degree([4]), 3),
    ]
    totals = [betti.total(index) for index in range(4)]
    assert totals == [1, 6, 8, 3]
    print("S4-stable Betti totals:", totals)


if __name__ == "__main__":
    main()
