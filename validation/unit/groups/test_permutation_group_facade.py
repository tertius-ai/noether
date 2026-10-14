# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam

import pytest

from noether.groups import FiniteGroup, Permutation, PermutationGroup


def symmetric_group_3() -> PermutationGroup:
    return PermutationGroup.from_generators(
        3,
        [
            Permutation.from_images([1, 0, 2]),
            Permutation.from_images([1, 2, 0]),
        ],
    )


def test_canonical_group_is_the_live_bsgs_object() -> None:
    group = symmetric_group_3()

    assert type(group).__name__ == "PermutationGroup"
    assert isinstance(group, FiniteGroup)
    assert group.type_id == "noether/groups/PermutationGroup@1"
    assert group.order == 6
    assert PermutationGroup.group_order(group.degree, group.generators) == 6
    assert group.orbit(0) == [0, 1, 2]
    assert group.is_transitive
    assert group.is_solvable
    assert not group.is_nilpotent
    assert not group.is_perfect
    assert group.stabilizer(0).order == 2


def test_permutation_operators_follow_the_documented_right_action() -> None:
    group = symmetric_group_3()
    transposition, three_cycle = group.generators
    product = transposition * three_cycle

    assert product(0) == three_cycle(transposition(0))
    assert product == transposition.compose(three_cycle)
    assert product in group
    assert group.multiply(product, group.inverse(product)) == group.identity()


def test_permutation_order_is_a_python_big_integer() -> None:
    cycle_lengths = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    degree = sum(cycle_lengths)
    images = list(range(degree))
    start = 0
    expected = 1
    for length in cycle_lengths:
        for offset in range(length):
            images[start + offset] = start + (offset + 1) % length
        start += length
        expected *= length

    permutation = Permutation.from_images(images)
    assert permutation.order == expected
    assert permutation.order > 2**64


def test_seeded_uniform_elements_are_reproducible() -> None:
    group = symmetric_group_3()
    first = group.uniform_random_element(seed=271828)
    second = group.uniform_random_element(seed=271828)

    assert first.seed == 271828
    assert first.element == second.element
    assert first.element in group


def test_checked_operations_fail_closed() -> None:
    cyclic = PermutationGroup.from_generators(3, [Permutation.from_images([1, 2, 0])])
    nonmember = Permutation.from_images([1, 0, 2])

    assert nonmember not in cyclic
    with pytest.raises(ValueError, match="not an element"):
        cyclic.inverse(nonmember)
    with pytest.raises(ValueError, match="degree mismatch"):
        cyclic.contains(Permutation.identity(4))


def test_sylow_subgroups_and_literal_intersections_are_projected() -> None:
    symmetric = symmetric_group_3()
    assert symmetric.sylow_subgroup(2).order == 2
    assert symmetric.sylow_subgroup(3).order == 3
    with pytest.raises(ValueError, match="not prime"):
        symmetric.sylow_subgroup(4)

    alternating = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 2, 0, 3]),
            Permutation.from_images([0, 2, 3, 1]),
        ],
    )
    dihedral = PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 2, 3, 0]),
            Permutation.from_images([2, 1, 0, 3]),
        ],
    )
    intersection = alternating.intersection(dihedral)
    assert intersection.order == 4
    assert all(generator in alternating for generator in intersection.generators)
    assert all(generator in dihedral for generator in intersection.generators)
