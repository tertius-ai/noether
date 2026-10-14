# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute and verify a Bézout certificate over the integers."""

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
