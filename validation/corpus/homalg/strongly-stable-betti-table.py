# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the strongly stable ideal in a weighted-Borel example."""

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
    ring = PolynomialRing(Q, ["x1", "x2", "x3"])
    x1, x2, x3 = ring.generators()
    relations = [
        x1 * x2 * x3**2,
        x1**2 * x3**2,
        x1 * x2**2 * x3,
        x1**2 * x2 * x3,
        x1**3 * x3,
        x1 * x2**3,
        x1**2 * x2**2,
        x1**3 * x2,
        x1**4,
    ]
    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(ring, 9, grading_rank=1, degree_shifts=[Degree([-4])] * 9)
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
        (1, Degree([4]), 9),
        (2, Degree([5]), 13),
        (3, Degree([6]), 5),
    ]
    totals = [betti.total(index) for index in range(4)]
    assert totals == [1, 9, 13, 5]
    print("strongly stable Betti totals:", totals)


if __name__ == "__main__":
    main()
