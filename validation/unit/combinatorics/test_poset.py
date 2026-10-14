# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The public finite-poset facade has one Rust-owned mathematical value."""

from __future__ import annotations

import pytest

from noether.combin import BeatPoint, CoreReduction, Graph, Poset, PosetCanonicalLabel
from noether.groups import PermutationGroup
from noether.topology import SimplicialComplex


def boolean_lattice() -> Poset:
    return Poset(4, [(0, 1), (0, 2), (1, 3), (2, 3)])


def test_relation_and_hasse_graph_are_one_validated_value():
    # Lists and tuples are both natural Python relation-pair spellings.
    poset = Poset(3, [[0, 2], [0, 1], [1, 2], [0, 2]])

    assert poset.type_id == "noether/combin/Poset@1"
    assert poset.order == 3
    assert isinstance(poset.covers, Graph)
    assert poset.covers.is_directed
    assert poset.covers.weighted_edges == [(0, 1, 1), (1, 2, 1)]
    assert poset.leq(0, 0)
    assert poset.leq(0, 2)
    assert not poset.leq(2, 0)


def test_invalid_posets_fail_at_construction():
    with pytest.raises(ValueError, match="out of range"):
        Poset(2, [(0, 2)])
    with pytest.raises(ValueError, match="strict"):
        Poset(2, [(1, 1)])
    with pytest.raises(ValueError, match="cycle"):
        Poset(3, [(0, 1), (1, 2), (2, 0)])


def test_canonical_json_revalidates_and_ignores_no_derived_state():
    poset = Poset(3, [(0, 1), (1, 2), (0, 2)])
    assert poset.to_json() == '{"order":3,"covers":[[0,1],[1,2]]}'
    assert Poset.from_json(poset.to_json()) == poset
    with pytest.raises(ValueError, match="strict"):
        Poset.from_json('{"order":2,"covers":[[0,0]]}')


def test_symmetry_uses_the_public_permutation_group():
    poset = boolean_lattice()
    canonical = poset.canonical_label()
    group = poset.automorphism_group()

    assert isinstance(canonical, PosetCanonicalLabel)
    assert canonical.verify(poset)
    assert canonical.poset == poset
    assert canonical.incidence_sequence == [1, 3, 5, 15]
    assert isinstance(group, PermutationGroup)
    assert int(group.order) == 2


def test_structural_operations_use_the_validated_order_relation():
    square = Poset(
        8,
        [
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 6),
            (2, 5),
            (2, 7),
            (3, 6),
            (3, 7),
        ],
    )

    assert square.strict_lower_counts() == [0, 0, 0, 0, 2, 2, 2, 2]
    assert square.strict_upper_counts() == [2, 2, 2, 2, 0, 0, 0, 0]
    assert square.minimal_elements() == [0, 1, 2, 3]
    assert square.maximal_elements() == [4, 5, 6, 7]
    assert square.comparable_pair_count() == 8
    assert square.connected_components() == [list(range(8))]
    assert square.height() == 2
    assert square.width() == 4
    assert not square.is_lattice()
    assert square.is_self_dual()
    assert square.opposite().opposite() == square


def test_core_reduction_carries_replayable_beat_point_evidence():
    chain = Poset(3, [(0, 1), (1, 2)])
    beat_points = chain.beat_points()

    assert all(isinstance(point, BeatPoint) for point in beat_points)
    assert [(point.element, point.kind, point.witness) for point in beat_points] == [
        (0, "up", 1),
        (1, "down", 0),
        (1, "up", 2),
        (2, "down", 1),
    ]
    reduction = chain.core()
    assert isinstance(reduction, CoreReduction)
    assert reduction.verify(chain)
    assert reduction.core.order == 1
    assert reduction.core.is_core()
    assert reduction.reduced.canonical_label().labeling == reduction.canonical_labeling
    assert [group.free_rank for group in reduction.graded_homology_invariants()] == [1]
    assert [group.free_rank for group in chain.graded_homology_invariants()] == [1]


def test_order_complex_is_a_linkable_full_simplicial_complex():
    empty = Poset(0, []).order_complex()
    assert empty.is_empty()
    assert empty.dimension == -1

    square = Poset(
        8,
        [
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 6),
            (2, 5),
            (2, 7),
            (3, 6),
            (3, 7),
        ],
    )
    complex_ = square.order_complex()

    assert isinstance(complex_, SimplicialComplex)
    assert complex_.facets == [
        [0, 4],
        [0, 5],
        [1, 4],
        [1, 6],
        [2, 5],
        [2, 7],
        [3, 6],
        [3, 7],
    ]
    assert [complex_.count(degree) for degree in range(2)] == [8, 8]
    assert complex_.euler_characteristic() == 0
    assert complex_.homology_invariants(1).free_rank == 1


def test_queries_check_element_ids():
    with pytest.raises(IndexError, match="out of range"):
        boolean_lattice().leq(0, 9)
