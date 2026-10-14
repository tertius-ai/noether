# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Verify the Mobius-generator identity in Q[t]/(t^22)."""

from noether import Q, RationalField
from noether.commalg import Polynomial, PolynomialRing


def mobius_generator(
    ring: PolynomialRing[RationalField], parameter: int
) -> Polynomial[RationalField]:
    t = ring.generator(0)
    return sum(
        ((parameter**degree) * t ** (degree + 1) for degree in range(21)),
        ring.zero(),
    )


def main() -> None:
    ring = PolynomialRing(Q, ["t"])
    t = ring.generator(0)
    truncation = ring.ideal([t**22])
    for left, right in [(1, 2), (-1, 3), (0, 4)]:
        u_left = mobius_generator(ring, left)
        u_right = mobius_generator(ring, right)
        relation = u_right - u_left - (right - left) * u_left * u_right
        assert truncation.normal_form(relation).is_zero()
    print("u_b-u_a=(b-a)u_a*u_b holds in Q[t]/(t^22)")


if __name__ == "__main__":
    main()
