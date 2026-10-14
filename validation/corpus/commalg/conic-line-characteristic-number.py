# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Evaluate the logarithmic Chern-number ratio of the Klein arrangement."""

from fractions import Fraction

from noether import Q


def main() -> None:
    lines = conics = 21
    double, triple, quadruple = 42, 252, 189
    numerator = 9 - 8 * lines - 5 * conics + 2 * double + 5 * triple + 8 * quadruple
    denominator = 3 - 2 * lines - 2 * conics + double + 2 * triple + 3 * quadruple
    ratio = Q(numerator) / Q(denominator)
    assert ratio == Fraction(108, 43)
    assert 2.511 < float(ratio) < 2.513
    print("gamma(CL) =", ratio)


if __name__ == "__main__":
    main()
