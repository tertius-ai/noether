# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Build the Hibi ideal of the divisor poset of 12 and resolve it."""

from noether import Q
from noether.combin import Poset
from noether.commalg import Degree, MonomialOrder, PolynomialRing
from noether.homalg import (
    FreeModule,
    ModuleHomomorphism,
    ModuleMonomialOrder,
    PresentedModule,
)
from noether.linalg import Matrix


def main() -> None:
    divisibility = Poset(
        6,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)],
    )
    assert divisibility.height() == 4 and divisibility.width() == 2

    order_ideals: list[set[int]] = []
    for mask in range(1 << divisibility.order):
        subset = {index for index in range(divisibility.order) if mask & (1 << index)}
        if all(
            lower in subset
            for upper in subset
            for lower in range(divisibility.order)
            if divisibility.leq(lower, upper)
        ):
            order_ideals.append(subset)
    assert len(order_ideals) == 10

    ring = PolynomialRing(
        Q,
        ["x1", "x2", "x3", "x4", "x6", "x12", "y1", "y2", "y3", "y4", "y6", "y12"],
    )
    variables = ring.generators()
    x, y = variables[:6], variables[6:]
    hibi_generators = []
    for ideal in order_ideals:
        monomial = ring.one()
        for index in range(6):
            monomial *= x[index] if index in ideal else y[index]
        hibi_generators.append(monomial)

    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(ring, 10, grading_rank=1, degree_shifts=[Degree([-6])] * 10)
    quotient = PresentedModule(
        ModuleHomomorphism(source, target, Matrix.from_rows(ring, [hibi_generators]))
    )
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    resolution = quotient.minimal_free_resolution(order)
    resolution.verify()
    assert resolution.is_minimal()

    betti = resolution.graded_betti_table()
    betti.verify()
    assert betti.entries == [
        (0, Degree([0]), 1),
        (1, Degree([6]), 10),
        (2, Degree([7]), 12),
        (3, Degree([8]), 3),
    ]
    print("Hibi quotient Betti totals: [1, 10, 12, 3]")


if __name__ == "__main__":
    main()
