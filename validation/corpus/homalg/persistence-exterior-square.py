# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute the underlying Q[t]-module of a persistence exterior square."""

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
    ring = PolynomialRing(Q, ["t"])
    t = ring.generator("t")
    generators = FreeModule(ring, 4, grading_rank=1, degree_shifts=[Degree([0])] * 4)
    relations = FreeModule(
        ring,
        3,
        grading_rank=1,
        degree_shifts=[Degree([-3]), Degree([-2]), Degree([-1])],
    )
    module = PresentedModule(
        ModuleHomomorphism(
            relations,
            generators,
            Matrix.from_rows(ring, [[0, 0, 0], [t**3, 0, 0], [0, t**2, 0], [0, 0, t]]),
        )
    )

    exterior_square = module.exterior_power(2)
    exterior_square.verify()
    result = exterior_square.module
    assert result.generator_count == 6

    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    assert result.annihilator(order).groebner_basis() == [t**3]
    fitting_exponents = [10, 7, 5, 3, 2, 1, 0]
    for index, exponent in enumerate(fitting_exponents):
        assert result.fitting_ideal(index).groebner_basis() == [t**exponent]

    print("Lambda^2 M has invariant factors t^3, t^2, t^2, t, t, t")


if __name__ == "__main__":
    main()
