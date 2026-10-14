# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Reproduce the SingularInterface example's basis and syzygy module."""

from noether import Q
from noether.commalg import MonomialOrder, PolynomialRing
from noether.homalg import (
    FreeModule,
    ModuleElement,
    ModuleHomomorphism,
    ModuleMonomialOrder,
    Submodule,
)
from noether.linalg import Matrix, Vector


def main() -> None:
    ring = PolynomialRing(Q, ["x0", "x1", "x2", "x3"], order="grevlex")
    x0, x1, x2, x3 = ring.generators()
    ideal = ring.ideal([x0**2 - x1 * x3, x0 * x1 - x2 * x3])
    displayed_basis = [
        x0 * x1 - x2 * x3,
        x0**2 - x1 * x3,
        x1**2 * x3 - x0 * x2 * x3,
    ]
    computed = ideal.groebner_basis()
    assert all(any(value == candidate for candidate in computed) for value in displayed_basis)
    assert all(ideal.normal_form(value).is_zero() for value in displayed_basis)

    source = FreeModule(ring, 3)
    target = FreeModule(ring, 1)
    presentation = ModuleHomomorphism(source, target, Matrix.from_rows(ring, [displayed_basis]))
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    syzygies = presentation.kernel(order)
    displayed_syzygies = [
        ModuleElement(source, Vector.from_values(ring, [x0, -x1, -1])),
        ModuleElement(source, Vector.from_values(ring, [x1 * x3, -x2 * x3, -x0])),
    ]
    assert all(
        all(coordinate.is_zero() for coordinate in presentation(value).coordinates)
        for value in displayed_syzygies
    )
    assert syzygies.is_equal(Submodule.from_generators(source, displayed_syzygies), order)
    print("the three-element basis has the two displayed syzygy generators")


if __name__ == "__main__":
    main()
