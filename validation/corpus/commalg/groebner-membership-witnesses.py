# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Obtain and independently check ideal-membership multipliers."""

from noether import Q
from noether.commalg import PolynomialRing


def main() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order="lex")
    x, y = ring.generators()
    ideal = ring.ideal([x + y**2, y**2])
    target = x + 2 * y**2
    witness = ideal.reduce(target)
    witness.verify()
    assert witness.is_member and witness.remainder.is_zero()
    assert (
        sum(
            (
                multiplier * generator
                for multiplier, generator in zip(witness.multipliers, ideal.generators, strict=True)
            ),
            ring.zero(),
        )
        == target
    )

    nonmember = ideal.reduce(x + y)
    nonmember.verify()
    assert not nonmember.is_member and not nonmember.remainder.is_zero()

    first = ring.ideal([x + y, x * y - 1])
    second = ring.ideal([x + y, y**2 + 1])
    for generator in [*first.generators, *second.generators]:
        containing = second if generator in first.generators else first
        reduction = containing.reduce(generator)
        reduction.verify()
        assert reduction.is_member
    print("membership, nonmembership, and bidirectional ideal witnesses verified")


if __name__ == "__main__":
    main()
