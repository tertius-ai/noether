# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute exact tipsy cop-and-robber survival and capture data."""

from fractions import Fraction

from noether import Q
from noether.linalg import Matrix, Vector


def main() -> None:
    robber = Fraction(3, 10)
    tipsy = Fraction(2, 5)
    cop = Fraction(3, 10)
    states = 10
    rounds = 7
    rows = [[Fraction(0) for _ in range(states)] for _ in range(states)]
    for state in range(states - 1):
        rows[state][state + 1] = robber + 2 * tipsy / 3
    for state in range(1, states):
        rows[state][state - 1] = cop + tipsy / 3
    rows[-1][-1] = robber + 2 * tipsy / 3

    transition = Matrix.from_rows(Q, rows)
    power = Matrix.identity(Q, states)
    for _ in range(rounds):
        power = power @ transition
    ones = Vector.from_values(Q, [1] * states)
    survival = power.apply(ones)

    identity = Matrix.identity(Q, states)
    fundamental_operator = Matrix.from_rows(
        Q,
        [
            [identity[row, column] - transition[row, column] for column in range(states)]
            for row in range(states)
        ],
    )
    expected = fundamental_operator.solve(ones)
    expected.verify()

    assert survival[0] == Fraction(1_644_277_927, 4_374_000_000)
    assert survival[9] == 1
    assert expected.particular[0] == Fraction(14_086_015_564_500, 137_858_491_849)
    assert expected.particular[9] == Fraction(49_526_179_260_450, 137_858_491_849)
    print("seven-round survival from distance one:", survival[0])
    print("expected duration from distance one:", expected.particular[0])


if __name__ == "__main__":
    main()
