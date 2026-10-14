# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Certify the two radical-membership decisions in the Hough example."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["a", "e"])
    a, e = ring.generators()
    diagonal = a - e

    regular_double = ring.ideal([a**2 - e**2, a**3 - e**3])
    regular_radical = regular_double.radical()
    regular_radical.verify()
    assert diagonal in regular_radical.ideal

    nonregular_double = ring.ideal([a**2 - e**2, a**4 - e**4])
    nonregular_radical = nonregular_double.radical()
    nonregular_radical.verify()
    assert diagonal not in nonregular_radical.ideal

    print("a-e is radical-member for exponents (2,3), but not for (2,4)")


if __name__ == "__main__":
    main()
