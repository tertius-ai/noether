# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Decompose a polynomial by a nonstandard positive grading."""

from noether import Z
from noether.commalg import PolynomialRing, WeightedGrading


def main() -> None:
    ring = PolynomialRing(Z, ["x", "y", "z"], grading=WeightedGrading([1, 2, 3]))
    x, y, z = ring.generators()
    sparse = ring.from_terms([([0, 0, 0], 1), ([2, 0, 0], 1), ([0, 1, 0], 1)])
    polynomial = sparse + z
    assert sparse * (x + y) == (1 + x**2 + y) * (x + y)
    components = polynomial.homogeneous_components()
    assert [degree for degree, _ in components] == [0, 2, 3]
    assert x**2 + y == components[1][1]
    print(f"weighted homogeneous degrees: {[degree for degree, _ in components]}")


if __name__ == "__main__":
    main()
