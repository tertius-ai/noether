# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Translate the paper's exact advance-bid recurrence."""

from functools import cache

from noether import Q


@cache
def winning_probability(bid: tuple[int, ...], suits: tuple[int, ...], safe: int = 0):
    total = safe + sum(suits)
    if not bid:
        return Q(1)
    if bid[0] == 0:
        return winning_probability(bid[1:], suits[1:], safe + suits[0])

    next_bid = (bid[0] - 1,) + bid[1:]
    probability = Q(0)
    for index in range(1, len(suits)):
        if suits[index] > 0:
            next_suits = suits[:index] + (suits[index] - 1,) + suits[index + 1 :]
            probability += (
                Q(suits[index]) / Q(total) * winning_probability(next_bid, next_suits, safe)
            )
    if safe > 0:
        probability += Q(safe) / Q(total) * winning_probability(next_bid, suits, safe - 1)
    return probability


def main() -> None:
    checks = {
        ((1, 1, 1), (0, 1, 2)): Q(1) / Q(3),
        ((1, 1, 1), (1, 1, 1)): Q(1) / Q(3),
        ((2, 1, 1), (0, 2, 2)): Q(1) / Q(3),
        ((3, 1, 1), (0, 2, 3)): Q(3) / Q(10),
        ((2, 2, 1), (0, 1, 4)): Q(1) / Q(5),
        ((2, 2, 1), (0, 2, 3)): Q(1) / Q(5),
        ((2, 2, 1), (1, 1, 3)): Q(1) / Q(5),
    }
    for (suits, bid), expected in checks.items():
        assert winning_probability(bid, suits) == expected
    print("exact optimal-bid probabilities checked:", len(checks))


if __name__ == "__main__":
    main()
