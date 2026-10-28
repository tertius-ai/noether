# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Module reduction and a free resolution

Reduce an explicitly generated module element by a module Gröbner
basis. Then compute the minimal graded free resolution of the
twisted-cubic coordinate ring.

## Result

The reduction certifies membership. The resolution has one generator
in degree 0, three relations in degree 2, and two syzygies in degree 3:
0 → S(−3)² → S(−2)³ → S → S/I → 0.
"""

from noether import Q
from noether.commalg import Degree, MonomialOrder, PolynomialRing
from noether.homalg import (
    FreeModule,
    ModuleElement,
    ModuleHomomorphism,
    ModuleMonomialOrder,
    PresentedModule,
    Submodule,
)
from noether.linalg import Matrix, Vector


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    ambient = FreeModule(ring, 2)
    first = ModuleElement(ambient, Vector.from_values(ring, [x, y]))
    second = ModuleElement(ambient, Vector.from_values(ring, [y, 1]))
    submodule = Submodule.from_generators(ambient, [first, second])
    basis = submodule.groebner_basis(order)
    reduction = basis.reduce(first + second * x)
    basis.verify()
    reduction.verify()
    assert reduction.is_member()

    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    source = FreeModule(ring, 3, grading_rank=1, degree_shifts=[Degree([-2])] * 3)
    twisted_cubic = PresentedModule(
        ModuleHomomorphism(
            source,
            target,
            Matrix.from_rows(ring, [[x * z - y**2, x * w - y * z, y * w - z**2]]),
        )
    )
    resolution = twisted_cubic.minimal_free_resolution(order)
    resolution.verify()
    betti = resolution.graded_betti_table()
    betti.verify()
    assert betti.entries == [
        (0, Degree([0]), 1),
        (1, Degree([2]), 3),
        (2, Degree([3]), 2),
    ]
    print(f"module member certified; minimal Betti entries={betti.entries}")


if __name__ == "__main__":
    main()
