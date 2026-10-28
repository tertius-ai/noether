# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Differentials and tangent dimensions of a cusp

For A = ℚ[x,y]/(y² − x³), construct the module of Kähler differentials,
check the product rule, and compare cotangent spaces at (0,0) and (1,1).

## Result

The differential relation is 2y dy − 3x² dx = 0. At the origin it
gives no linear relation, so the cotangent dimension is 2; at (1,1)
it gives one relation, leaving dimension 1. The hypersurface cotangent
complex has amplitude [−1,0].
"""

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
