# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Characteristic numbers of a complete-intersection curve

In ℙ¹ × ℙ², assume a smooth complete intersection of hypersurfaces
of bidegrees (1,1) and (1,2). Compute the Euler characteristic of
the restriction of O(2,−1). Smoothness is a supplied hypothesis.

## Result

The intersection is a curve with degrees 2 and 3 against the two
hyperplane classes. The Euler characteristic is 1. Hirzebruch–Riemann–Roch
and the Koszul formula give the same value under the stated hypothesis.
"""

from fractions import Fraction

from noether.commalg import ChowRing, ProjectiveCompleteIntersection


def main() -> None:
    # The Chow ring of P^1 x P^2 is Z[h1,h2]/(h1^2,h2^3).
    chow = ChowRing([1, 2])
    h1, h2 = chow.generators()
    assert chow.integrate(h1 * h2**2) == 1

    # A smooth (1,1),(1,2) complete intersection is a curve. Smoothness is an
    # explicit hypothesis here.
    variety = ProjectiveCompleteIntersection.assuming_smooth(chow, [[1, 1], [1, 2]])
    assert variety.dimension == 1
    assert variety.integrate_restriction(h1) == 2
    assert variety.integrate_restriction(h2) == 3

    euler = variety.line_bundle_euler_characteristic([2, -1])
    euler.verify()
    assert euler.hrr_integral == Fraction(euler.koszul_value)
    assert euler.value == 1
    print(
        f"smooth (1,1),(1,2) curve in P^1 x P^2: chi(O(2,-1))={euler.value}; HRR and Koszul agree"
    )


if __name__ == "__main__":
    main()
