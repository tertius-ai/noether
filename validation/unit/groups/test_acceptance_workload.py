# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The bounded native permutation-group release calculation."""

import pytest

from noether.combin import Poset
from noether.groups import (
    ConjugacyClass,
    DoubleCoset,
    FiniteGroup,
    OrbitStabilizer,
    Permutation,
    PermutationGroup,
    RandomElement,
    SchreierGenerator,
    SchreierTree,
    StabilizerChain,
    StabilizerChainStep,
    SubsetOrbit,
    Transversal,
)


def symmetric_group_four() -> PermutationGroup:
    return PermutationGroup.from_generators(
        4,
        [
            Permutation.from_images([1, 0, 2, 3]),
            Permutation.from_images([1, 2, 3, 0]),
        ],
    )


def test_permutation_group_operations_return_live_checked_groups() -> None:
    group = symmetric_group_four()

    assert isinstance(group, FiniteGroup)
    assert group.order == 24
    assert group.orbit(0) == [0, 1, 2, 3]
    assert group.stabilizer(0).order == 6
    assert group.pointwise_stabilizer([0, 1]).order == 2
    orbit_stabilizer = group.orbit_stabilizer(0)
    assert isinstance(orbit_stabilizer, OrbitStabilizer)
    assert orbit_stabilizer.stabilizer.order * len(orbit_stabilizer.orbit) == group.order

    transversal = group.transversal(0)
    assert isinstance(transversal, Transversal)
    assert transversal.base == 0
    assert transversal.orbit == [0, 1, 2, 3]
    assert len(transversal.reps) == len(transversal.orbit)

    tree = group.schreier_tree(0)
    assert isinstance(tree, SchreierTree)
    assert tree.root == 0
    assert tree.contains(3)
    assert tree.pred[0] is None
    assert tree.generators_used == group.generators

    schreier_generators = group.schreier_generators(0)
    assert schreier_generators
    assert all(isinstance(value, SchreierGenerator) for value in schreier_generators)
    assert all(value.coset_index < 4 for value in schreier_generators)
    assert all(value.generator_index < len(group.generators) for value in schreier_generators)
    assert all(value.perm in group for value in schreier_generators)

    chain = group.stabilizer_chain()
    assert isinstance(chain, StabilizerChain)
    assert len(chain) == len(chain.steps) == 3
    assert all(isinstance(step, StabilizerChainStep) for step in chain.steps)
    assert all(len(step.transversal) == len(step.orbit) for step in chain.steps)
    assert all(generator in group for generator in group.strong_generating_set)
    normalizer = group.normalizer_in_symmetric()
    assert normalizer.order == group.order
    assert all(generator in normalizer for generator in group.generators)
    assert all(generator in group for generator in normalizer.generators)

    classes = group.conjugacy_classes()
    assert all(isinstance(value, ConjugacyClass) for value in classes)
    assert sorted(value.size for value in classes) == [1, 3, 6, 6, 8]
    assert sum(value.size for value in classes) == group.order
    for value in classes:
        assert value.centralizer().order * value.size == group.order
        value.verify()

    seeded = group.uniform_random_element(seed=1729)
    repeated = group.uniform_random_element(seed=1729)
    assert isinstance(seeded, RandomElement)
    assert seeded.seed == repeated.seed == 1729
    assert seeded.element.images == repeated.element.images
    assert seeded.element in group

    elements = list(group.iterate_elements())
    assert len(elements) == group.order
    assert len({tuple(element.images) for element in elements}) == group.order

    randomized = PermutationGroup.from_generators_randomized(
        4, group.generators, seed=1729, stabilize=16
    )
    assert randomized.order == group.order
    residue, is_member = randomized.sift(group.generators[0])
    assert is_member
    assert residue.is_identity

    conjugating = group.conjugating_element(group)
    embedding = group.embeds_up_to_conjugacy(group)
    assert conjugating is not None
    assert embedding is not None


def test_combinatorial_automorphisms_use_the_same_group_identity() -> None:
    diamond = Poset(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    automorphisms = diamond.automorphism_group()

    assert isinstance(automorphisms, PermutationGroup)
    assert automorphisms.degree == 4
    assert automorphisms.order == 2
    assert automorphisms.set_stabilizer([1, 2]).order == 2


def test_subset_and_pair_orbits_are_exact_finite_actions() -> None:
    cyclic = PermutationGroup.from_generators(4, [Permutation.from_images([1, 2, 3, 0])])

    pair_orbits = cyclic.orbits_on_pairs()
    assert [orbit.representative for orbit in pair_orbits] == [[0, 1], [0, 2]]
    assert [orbit.size for orbit in pair_orbits] == [4, 2]
    assert sum(orbit.size for orbit in pair_orbits) == 6
    for orbit in pair_orbits:
        assert isinstance(orbit, SubsetOrbit)
        assert orbit.type_id == "noether/groups/SubsetOrbit@1"
        assert orbit.subset_size == 2
        orbit.verify(cyclic)

    edge = cyclic.orbit_of_subset([0, 1])
    assert edge.elements == pair_orbits[0].elements


def test_double_cosets_are_exact_and_batchable() -> None:
    cyclic = PermutationGroup.from_generators(4, [Permutation.from_images([1, 2, 3, 0])])
    trivial = PermutationGroup.from_generators(4, [])
    whole = cyclic.double_cosets(trivial, trivial)

    assert len(whole) == 4
    assert sum(double_coset.size for double_coset in whole) == cyclic.order
    assert all(isinstance(double_coset, DoubleCoset) for double_coset in whole)
    assert all(double_coset.right_coset_count == 1 for double_coset in whole)
    for double_coset in whole:
        assert double_coset.type_id == "noether/groups/DoubleCoset@1"
        double_coset.verify(cyclic, trivial, trivial)

    triangle = cyclic.double_coset_triangle([trivial, cyclic])
    assert [len(row) for row in triangle] == [1, 2]
    assert len(triangle[0][0]) == 4
    assert len(triangle[1][0]) == 1
    assert len(triangle[1][1]) == 1


def test_bounded_conjugacy_fails_before_returning_partial_evidence() -> None:
    symmetric_ten = PermutationGroup.from_generators(
        10,
        [
            Permutation.from_images([1, 0, *range(2, 10)]),
            Permutation.from_images([*range(1, 10), 0]),
        ],
    )
    with pytest.raises(OverflowError, match="more than 1000000 group elements"):
        symmetric_ten.conjugacy_classes()
