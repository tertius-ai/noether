# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Differentiate exactly and distinguish a cusp's smooth and singular points."""

from noether import Q
from noether.commalg import KahlerDifferentials, PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    relation = y**2 - x**3
    cusp = ring.quotient(ring.ideal([relation]))
    omega = cusp.kahler_differentials()

    assert isinstance(omega, KahlerDifferentials)
    dx = omega.differentiate(cusp(x))
    dy = omega.differentiate(cusp(y))
    assert omega.differentiate(cusp(x * y)) == dx * cusp(y) + dy * cusp(x)
    assert omega.differentiate(cusp(relation)).is_zero()

    singular = cusp.cotangent_space_at([0, 0])
    smooth = cusp.cotangent_space_at([1, 1])
    assert singular.embedding_dimension == 2
    assert smooth.embedding_dimension == 1
    assert singular.presentation_matrix == singular.evaluated_jacobian.transpose()

    complex = cusp.cotangent_complex()
    assert complex.codimension == 1
    assert (complex.amplitude.lowest_degree, complex.amplitude.highest_degree) == (-1, 0)
    assert complex.hypothesis.equation_count == 1
    complex.verify()

    print(
        "cusp cotangent dimensions: "
        f"singular={singular.embedding_dimension}, smooth={smooth.embedding_dimension}; "
        "complete-intersection amplitude=[-1,0]"
    )


if __name__ == "__main__":
    main()
