# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Resolve the homogeneous coordinate ring of a rational quartic."""

from noether.commalg import Degree, MonomialOrder, PolynomialRing
from noether.homalg import (
    FreeModule,
    ModuleHomomorphism,
    ModuleMonomialOrder,
    PresentedModule,
)
from noether.linalg import Matrix
from noether.numthy import FiniteField


def main() -> None:
    field = FiniteField.prime(32003)
    ring = PolynomialRing(field, ["w", "x", "y", "z"])
    w, x, y, z = ring.generators()
    relations = [x * y - w * z, y**3 - x * z**2, w * y**2 - x**2 * z, x**3 - w**2 * y]
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
        (1, Degree([3]), 3),
        (2, Degree([4]), 4),
        (3, Degree([5]), 1),
    ]
    totals = [betti.total(index) for index in range(4)]
    assert totals == [1, 4, 4, 1]
    print("rational-quartic Betti totals:", totals)


if __name__ == "__main__":
    main()
