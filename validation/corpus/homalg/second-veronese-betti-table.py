# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the minimal Betti table of the second Veronese surface."""

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
    ring = PolynomialRing(Q, ["a", "b", "c", "d", "e", "f"])
    a, b, c, d, e, f = ring.generators()
    relations = [
        a * d - b**2,
        a * e - b * c,
        a * f - c**2,
        b * e - c * d,
        b * f - c * e,
        d * f - e**2,
    ]

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
    print("second Veronese Betti totals: [1, 6, 8, 3]")


if __name__ == "__main__":
    main()
