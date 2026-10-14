# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute graded Jacobian syzygies of a six-line arrangement."""

from noether import Q
from noether.commalg import Degree, MonomialOrder, PolynomialRing
from noether.homalg import FreeModule, ModuleHomomorphism, ModuleMonomialOrder
from noether.linalg import Matrix


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    polynomial = (x**2 - y**2) * (x**2 - z**2) * (y**2 - z**2)
    jacobian = [polynomial.derivative(variable) for variable in ("x", "y", "z")]

    source = FreeModule(ring, 3, grading_rank=1, degree_shifts=[Degree([-5])] * 3)
    target = FreeModule(ring, 1, grading_rank=1, degree_shifts=[Degree([0])])
    homomorphism = ModuleHomomorphism(source, target, Matrix.from_rows(ring, [jacobian]))
    order = ModuleMonomialOrder.term_over_position(MonomialOrder.GREVLEX)
    kernel = homomorphism.kernel(order)
    minimal = kernel.minimal_homogeneous_generators()

    assert kernel.generator_count == 16
    assert minimal.generator_count == 2
    generator_columns = zip(*minimal.generator_map.matrix.to_rows(), strict=True)
    coordinate_degrees = sorted(
        max(coordinate.degree() or 0 for coordinate in generator) for generator in generator_columns
    )
    assert coordinate_degrees == [2, 3]
    # The degree-three piece consists of three linear multiples of the quadratic
    # generator and the independent cubic generator.
    assert 3 + 1 == 4
    print("minimal Jacobian-syzygy degrees:", coordinate_degrees)
    print("dimension of the degree-three syzygy space: 4")


if __name__ == "__main__":
    main()
