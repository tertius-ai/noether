# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Hensel lifting a square root

Lift the root 1 of x² − 6 modulo 5 to a root modulo 5⁶. The derivative
at that residue is 2, which is invertible modulo 5.

## Result

The returned representative squares to 6 modulo 15625. This is a
fixed-precision 5-adic calculation; the result carries the precision
chosen in `PadicRing`.
"""

from noether.commalg import PolynomialRing
from noether.numthy import PadicRing


def main() -> None:
    integers = PadicRing(5, absolute_precision=6)
    polynomials = PolynomialRing(integers, ["x"])
    x = polynomials.generator("x")
    equation = x**2 - 6
    root = integers.hensel_lift(equation, integers.residue_field(1))
    assert equation.evaluate([root]) == integers.zero()
    print(f"sqrt(6) in Z_5 mod 5^6 is represented by {int(root)}")


if __name__ == "__main__":
    main()
