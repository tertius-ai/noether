# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Weighted homogeneous components

Give x, y, and z weights 1, 2, and 3, and decompose
`1 + x**2 + y + z` into weighted homogeneous components.

## Result

The nonzero components have degrees 0, 2, and 3. The degree-2 component
is x² + y. A grading assigns degrees; it does not truncate polynomials.
"""

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
