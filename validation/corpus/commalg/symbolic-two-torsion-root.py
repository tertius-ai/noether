# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the published symbolic root of a Jacobian cubic."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["S"])
    s = ring.generator("S")
    h = 30
    t = s + h

    # Binary quartic a*u^4 + c*u^2 + e from the paper.
    a = 100 * s * t
    c = 200 * s**3 * t
    e = 80 * t**6 + 20 * t * s**5
    invariant_i = 12 * a * e + c**2
    invariant_j = 72 * a * c * e - 2 * c**3
    a4 = -27 * invariant_i
    a6 = -27 * invariant_j

    root = -1200 * s**3 * t
    assert root**3 + a4 * root + a6 == ring.zero()
    print("verified x = -1200*S^3*(S+30)")


if __name__ == "__main__":
    main()
