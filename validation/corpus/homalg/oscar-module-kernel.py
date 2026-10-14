# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute and certify the kernel of a polynomial-module homomorphism."""

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
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    source = FreeModule(ring, 3)
    target = FreeModule(ring, 2)
    homomorphism = ModuleHomomorphism(
        source,
        target,
        Matrix.from_rows(ring, [[y, x + y, 0], [0, y, z]]),
    )
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)

    kernel = homomorphism.kernel(order)
    displayed = ModuleElement(
        source,
        Vector.from_values(ring, [z * (x + y), -y * z, y**2]),
    )
    assert all(value.is_zero() for value in homomorphism(displayed).coordinates)
    expected = Submodule.from_generators(source, [displayed])
    assert kernel.is_equal(expected, order)
    basis = kernel.groebner_basis(order)
    basis.verify()
    assert basis.is_complete()
    assert kernel.generator_count == 1
    print("kernel certified with one quadratic generator")


if __name__ == "__main__":
    main()
