# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import pytest

from noether import Q, Z
from noether.homalg import (
    ChainComplex,
    ChainReduction,
    FinitelyGeneratedAbelianGroup,
    HomologyGroup,
)
from noether.linalg import Vector
from noether.numthy import FiniteField
from noether.topology import (
    AdmissibleCohomologyOperation,
    SimplicialSet,
    SimplicialSetAnodyneMove,
    SimplicialSetAnodyneZigzag,
    SimplicialSetCohomologyRing,
    SimplicialSetElementaryHornCollapse,
    SimplicialSetHornCollapseSequence,
)


def test_empty_and_zero_dimensional_sets_have_distinct_dimensions() -> None:
    empty = SimplicialSet([])
    point = SimplicialSet([[[]]])

    assert empty.is_empty()
    assert empty.dimension == -1
    assert empty.non_degenerate_counts == []
    assert not point.is_empty()
    assert point.dimension == 0


def test_canonical_form_automorphisms_and_json_round_trip() -> None:
    interval = SimplicialSet([[[], []], [[1, 0]]])
    canonical = interval.canonical_form()
    assert canonical.canonical_form() == canonical
    # A simplicial-set map preserves d_0 and d_1, so it cannot reverse Delta[1].
    assert interval.automorphism_group().order == 1
    assert SimplicialSet.from_json(interval.to_json()) == interval


def test_minimal_circle_uses_canonical_checked_results() -> None:
    circle = SimplicialSet.circle()

    assert SimplicialSet.__module__ == "noether.topology"
    assert circle.non_degenerate_counts == [1, 1]
    assert circle.faces == [[[]], [[0, 0]]]
    assert circle.euler_characteristic() == 0
    assert type(circle.chain_complex(Z)) is ChainComplex
    assert circle.betti_number(0) == 1
    assert circle.betti_number(1) == 1
    assert circle.betti_number(2) == 0
    invariants = circle.homology_invariants(1)
    assert type(invariants) is FinitelyGeneratedAbelianGroup
    assert invariants.free_rank == 1
    assert invariants.torsion_coefficients == []

    integral = circle.graded_homology()
    assert [group.free_rank for group in integral.groups] == [1, 1]
    assert all(type(group) is HomologyGroup for group in integral.groups)
    assert circle.homology(1, ring=Q).dimension == 1
    assert circle.homology(1, ring=FiniteField.prime(3)).dimension == 1
    circle.verify()


def test_explicit_interval_is_contractible() -> None:
    interval = SimplicialSet(
        [
            [[], []],
            [[1, 0]],
        ]
    )

    assert interval.faces_of(1, 0) == [1, 0]
    assert interval.homology(0).free_rank == 1
    assert interval.homology(1).free_rank == 0
    assert interval.homology_invariants(0).free_rank == 1
    assert interval.homology_invariants(1).free_rank == 0
    with pytest.raises(IndexError, match="no non-degenerate 1-simplex 1"):
        interval.faces_of(1, 1)


def test_elementary_horn_collapse_is_replayable_anodyne_evidence() -> None:
    interval = SimplicialSet([[[], []], [[1, 0]]])
    available = interval.elementary_horn_collapses()

    assert len(available) == 2
    assert all(type(step) is SimplicialSetElementaryHornCollapse for step in available)
    point = interval.elementary_horn_collapse(available[0])
    assert point.non_degenerate_counts == [1]
    sequence = interval.greedy_horn_collapse_reduction()
    assert type(sequence) is SimplicialSetHornCollapseSequence
    assert sequence.target == point
    assert sequence.is_terminal()
    sequence.verify()
    assert SimplicialSetHornCollapseSequence.from_json(sequence.to_json()) == sequence

    with pytest.raises(ValueError, match="free horn pair"):
        SimplicialSet.circle().elementary_horn_collapse(
            SimplicialSetElementaryHornCollapse(1, 0, 0, 0)
        )


def test_anodyne_zigzag_replays_horn_expansions() -> None:
    interval = SimplicialSet([[[], []], [[1, 0]]])
    step = interval.elementary_horn_collapses()[0]
    zigzag = interval.anodyne_zigzag(
        [
            SimplicialSetAnodyneMove.collapse(step),
            SimplicialSetAnodyneMove.expansion(interval, step),
        ]
    )

    assert isinstance(zigzag, SimplicialSetAnodyneZigzag)
    assert zigzag.target == interval
    assert [move.direction for move in zigzag.moves] == ["collapse", "expansion"]
    zigzag.verify()
    assert SimplicialSetAnodyneZigzag.from_json(zigzag.to_json()).target == interval


def test_normalized_chain_reduction_is_replayable() -> None:
    interval = SimplicialSet([[[], []], [[1, 0]]])
    reduction = interval.normalized_chain_reduction(Z)

    assert type(reduction) is ChainReduction
    assert reduction.source == interval.chain_complex(Z)
    assert reduction.removed_pairs == 1
    assert reduction.reduced.module_dimension(0) == 1
    assert reduction.reduced.module_dimension(1) == 0
    assert (
        reduction.source.reduction_from_cancellations(reduction.cancellations).reduced
        == reduction.reduced
    )
    reduction.verify()


def test_normalized_cochains_cup_i_and_steenrod_square() -> None:
    sphere = SimplicialSet.sphere(2)
    field = FiniteField.prime(2)
    sphere.cochain_complex(field).verify()
    top = Vector.from_values(field, [1])

    assert sphere.cup_i(2, 2, top, 2, top) == top
    assert sphere.steenrod_square(0, 2, top) == top
    assert sphere.steenrod_square(1, 2, top).to_values() == []


def test_complete_normalized_cohomology_ring_and_cup_product() -> None:
    circle = SimplicialSet.circle()
    field = FiniteField.prime(3)
    unit = Vector.from_values(field, [1])
    loop = Vector.from_values(field, [1])

    assert circle.cup_product(0, unit, 1, loop) == loop
    ring = circle.cohomology_ring(field)
    assert type(ring) is SimplicialSetCohomologyRing
    assert [element.degree for element in ring.basis] == [0, 1]
    assert ring.product(1, 1) == []
    assert ring.presentation is not None
    ring.verify()
    assert SimplicialSetCohomologyRing.from_json(ring.to_json()) == ring


def test_odd_primary_may_steenrod_operations_detect_a_bockstein() -> None:
    degenerate_edge = {
        "simplex_dimension": 0,
        "simplex": 0,
        "surjection": [0, 0],
    }
    moore = SimplicialSet(
        [
            [[]],
            [[0, 0], [0, 0]],
            [[0, 1, 0], [0, degenerate_edge, 1]],
        ]
    )
    field = FiniteField.prime(3)
    cocycle = Vector.from_values(field, [1, 2])

    assert moore.steenrod_reduced_power(0, 1, cocycle) == cocycle
    assert moore.bockstein_reduced_power(0, 1, cocycle).to_values() != [0, 0]
    ring = moore.cohomology_ring(field)
    assert [element.degree for element in ring.basis] == [0, 1, 2]
    assert ring.steenrod_reduced_power(0, 1)[0].basis == 1
    assert ring.bockstein_reduced_power(0, 1)[0].basis == 2
    bockstein = AdmissibleCohomologyOperation.bockstein_reduced_power(3, 0)
    assert ring.apply_operation(bockstein, 1)[0].basis == 2


def test_degenerate_faces_round_trip_without_losing_the_surjection() -> None:
    degenerate_edge = {
        "simplex_dimension": 0,
        "simplex": 0,
        "surjection": [0, 0],
    }
    sphere = SimplicialSet(
        [
            [[]],
            [],
            [[degenerate_edge, degenerate_edge, degenerate_edge]],
        ]
    )

    assert sphere == SimplicialSet.sphere(2)
    assert sphere.faces[2][0] == [degenerate_edge] * 3
    assert sphere.non_degenerate_counts == [1, 0, 1]
    assert [group.free_rank for group in sphere.graded_homology().groups] == [1, 1]
    sphere.verify()


def test_invalid_simplicial_data_fails_closed() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        SimplicialSet([[[]], [[0]]])

    with pytest.raises(ValueError, match="not a monotone surjection"):
        SimplicialSet(
            [
                [[]],
                [],
                [
                    [
                        {
                            "simplex_dimension": 0,
                            "simplex": 0,
                            "surjection": [0, 1],
                        },
                        {
                            "simplex_dimension": 0,
                            "simplex": 0,
                            "surjection": [0, 0],
                        },
                        {
                            "simplex_dimension": 0,
                            "simplex": 0,
                            "surjection": [0, 0],
                        },
                    ]
                ],
            ]
        )

    with pytest.raises(ValueError, match="violate"):
        SimplicialSet(
            [
                [[], []],
                [[0, 0], [0, 0], [0, 1]],
                [[0, 1, 2]],
            ]
        )


def test_zero_sphere_has_two_components() -> None:
    sphere = SimplicialSet.sphere(0)

    assert sphere.non_degenerate_counts == [2]
    assert sphere.homology(0).free_rank == 2
    assert sphere.euler_characteristic() == 2
    sphere.verify()
