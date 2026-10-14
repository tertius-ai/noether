# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the quotient used in a codepth-three classification example."""

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
    relations = [x * y**2, x * y * z, y * z**2, x**4 - y**3 * z, x * z**3 - y**4]

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
    assert resolution.is_minimal() and resolution.length == 3

    betti = resolution.graded_betti_table()
    betti.verify()
    assert betti.entries == [
        (0, Degree([0]), 1),
        (1, Degree([3]), 3),
        (1, Degree([4]), 2),
        (2, Degree([4]), 2),
        (2, Degree([6]), 3),
        (2, Degree([7]), 1),
        (3, Degree([8]), 2),
    ]
    totals = [betti.total(degree) for degree in range(4)]
    assert totals == [1, 5, 6, 2]
    print("codepth-three Betti totals:", totals)


if __name__ == "__main__":
    main()
