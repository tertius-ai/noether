# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the quotient underlying a codepth-three H(4,3) example."""

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
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    relations = [z**2, y**3, x * y**2, x**2 * y, x**3]
    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(
        ring,
        len(relations),
        grading_rank=1,
        degree_shifts=[Degree([-relation.homogeneous_degree()]) for relation in relations],
    )
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
        (1, Degree([2]), 1),
        (1, Degree([3]), 4),
        (2, Degree([4]), 3),
        (2, Degree([5]), 4),
        (3, Degree([6]), 3),
    ]
    totals = [betti.total(index) for index in range(4)]
    assert totals == [1, 5, 7, 3]
    print("graded Betti entries:", betti.entries)
    print("Betti totals:", totals)


if __name__ == "__main__":
    main()
