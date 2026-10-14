# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Compute structural invariants of a six-element integer poset."""

from noether.combin import Poset


def main() -> None:
    relations = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (2, 4),
        (3, 4),
        (5, 4),
        (5, 3),
        (5, 2),
        (5, 1),
        (5, 0),
        (3, 2),
        (3, 1),
    ]
    poset = Poset(6, relations)

    assert poset.height() == 5
    assert poset.width() == 2
    assert poset.minimal_elements() == [5]
    assert poset.maximal_elements() == [1, 4]
    assert poset.comparable_pair_count() == 13
    assert poset.strict_lower_counts() == [1, 3, 3, 2, 4, 0]
    assert poset.strict_upper_counts() == [4, 0, 1, 3, 0, 5]
    assert not poset.is_lattice()
    assert not poset.is_self_dual()
    assert poset.automorphism_group().order == 1
    print("height 5; width 2; 13 comparable pairs; automorphism order 1")


if __name__ == "__main__":
    main()
