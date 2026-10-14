# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute and certify a graded minimal free resolution."""

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
    relations = [
        w**2 - x * z,
        w * x - y * z,
        x**2 - w * y,
        x * y - z**2,
        y**2 - w * z,
    ]

    generators = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    relation_module = FreeModule(ring, 5, grading_rank=1, degree_shifts=[Degree([-2])] * 5)
    presentation = ModuleHomomorphism(
        relation_module,
        generators,
        Matrix.from_rows(ring, [relations]),
    )
    algebra_as_module = PresentedModule(presentation)
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    resolution = algebra_as_module.minimal_free_resolution(order)
    resolution.verify()
    assert resolution.is_minimal()
    assert resolution.length == 3

    betti = resolution.graded_betti_table()
    betti.verify()
    expected_entries = [
        (0, Degree([0]), 1),
        (1, Degree([2]), 5),
        (2, Degree([3]), 5),
        (3, Degree([5]), 1),
    ]
    assert betti.entries == expected_entries
    totals = [betti.total(degree) for degree in range(resolution.length + 1)]
    assert totals == [1, 5, 5, 1]
    print("minimal graded Betti totals:", totals)


if __name__ == "__main__":
    main()
