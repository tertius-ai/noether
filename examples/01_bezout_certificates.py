# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Bézout identity

Find integers u and v such that `987654321*u + 123456789*v` is the greatest
common divisor of the two inputs. `Z.extended_gcd()` returns the divisor
and both coefficients.

## Result

The gcd is 9. The displayed identity expresses 9 as an integer linear
combination of the inputs. `verify()` checks both the Bézout equation and
the gcd claim against the supplied integers.
"""

from noether import Z


def main() -> None:
    identity = Z.extended_gcd(987_654_321, 123_456_789)
    identity.verify(Z, 987_654_321, 123_456_789)
    assert (
        identity.coefficient_a * 987_654_321 + identity.coefficient_b * 123_456_789 == identity.gcd
    )
    print(identity)


if __name__ == "__main__":
    main()
