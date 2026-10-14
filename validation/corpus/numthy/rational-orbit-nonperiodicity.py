# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Iterate the four-step rational recurrence exactly through 1000 terms."""

from noether import Q


def main() -> None:
    orbit = [Q(1)] * 4
    for _ in range(996):
        orbit.append((Q(1) + orbit[-1] + orbit[-2] + orbit[-3]) * (Q(1) / orbit[-4]))
    assert len(orbit) == 1000
    assert all(value != Q(1) for value in orbit[4:])
    print("the initial value does not recur in the first 1000 exact iterates")


if __name__ == "__main__":
    main()
