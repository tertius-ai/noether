# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import pytest

from noether import Q, Z
from noether.homalg import (
    ChainComplex,
    FinitelyGeneratedAbelianGroup,
    GradedHomology,
    HomologyGroup,
)
from noether.linalg import Vector
from noether.numthy import FiniteField
from noether.topology import (
    AdmissibleCohomologyOperation,
    DominatedVertex,
    ElementaryCollapse,
    SimplicialCohomologyRing,
    SimplicialCollapseSequence,
    SimplicialComplex,
    SimplicialMorseReduction,
    SimplicialSimpleHomotopyMove,
    SimplicialSimpleHomotopyZigzag,
    StrongCoreReduction,
)

RP2 = [
    [0, 1, 2],
    [0, 1, 4],
    [0, 2, 3],
    [0, 3, 5],
    [0, 4, 5],
    [1, 2, 5],
    [1, 3, 4],
    [1, 3, 5],
    [2, 3, 4],
    [2, 4, 5],
]


def test_empty_and_zero_dimensional_complexes_have_distinct_dimensions() -> None:
    empty = SimplicialComplex([])
    point = SimplicialComplex([[0]])

    assert empty.is_empty()
    assert empty.dimension == -1
    assert not point.is_empty()
    assert point.dimension == 0


def test_canonical_module_and_downward_closure() -> None:
    complex_ = SimplicialComplex([[2, 1, 0], [3]])

    assert SimplicialComplex.__module__ == "noether.topology"
    assert complex_.dimension == 2
    assert [complex_.count(k) for k in range(3)] == [4, 3, 1]
    assert complex_.facets == [[0, 1, 2], [3]]
    assert complex_.simplices(1) == [[0, 1], [0, 2], [1, 2]]
    assert len(complex_) == 8
    assert complex_.euler_characteristic() == 2
    complex_.verify()


def test_canonical_form_and_incidence_automorphisms_share_exact_semantics() -> None:
    complex_ = SimplicialComplex([[0, 1, 2], [0, 3], [1, 3]])
    relabeled = SimplicialComplex([[4, 1, 3], [4, 2], [1, 2]])

    canonical = complex_.canonical_form()
    assert canonical == relabeled.canonical_form()
    assert canonical == canonical.canonical_form()

    simplex = SimplicialComplex([list(range(10))])
    assert simplex.automorphism_group().order == 3_628_800


def test_strong_core_reduction_is_replayable_and_serializable() -> None:
    simplex = SimplicialComplex([[4, 2, 7, 1]])
    reduction = simplex.strong_core_reduction()

    assert type(reduction) is StrongCoreReduction
    assert reduction.source == simplex
    assert len(reduction.steps) == 3
    assert reduction.core == SimplicialComplex([[0]])
    assert reduction.reduced.strong_core() == reduction.core
    reduction.verify()
    assert StrongCoreReduction.from_json(reduction.to_json()) == reduction
    assert SimplicialComplex.from_json(simplex.to_json()) == simplex

    with pytest.raises(ValueError, match="not dominated"):
        simplex.strong_collapse(DominatedVertex(4, 99))


def test_elementary_collapse_sequence_is_replayable_and_serializable() -> None:
    disk = SimplicialComplex([[0, 1, 2]])
    available = disk.elementary_collapses()
    assert available
    assert type(available[0]) is ElementaryCollapse

    first_target = disk.elementary_collapse(available[0])
    sequence = disk.greedy_collapse_reduction()
    assert type(sequence) is SimplicialCollapseSequence
    assert sequence.source == disk
    assert sequence.target == SimplicialComplex([[2]])
    assert sequence.is_terminal()
    assert first_target == disk.collapse_sequence([available[0]]).target
    sequence.verify()
    assert SimplicialCollapseSequence.from_json(sequence.to_json()) == sequence

    circle = SimplicialComplex([[0, 1], [0, 2], [1, 2]])
    with pytest.raises(ValueError, match="not a free codimension-one face"):
        circle.elementary_collapse(ElementaryCollapse([0], [0, 1]))


def test_simple_homotopy_zigzag_replays_expansions() -> None:
    disk = SimplicialComplex([[0, 1, 2]])
    step = disk.elementary_collapses()[0]
    zigzag = disk.simple_homotopy_zigzag(
        [
            SimplicialSimpleHomotopyMove.collapse(step),
            SimplicialSimpleHomotopyMove.expansion(disk, step),
        ]
    )

    assert isinstance(zigzag, SimplicialSimpleHomotopyZigzag)
    assert zigzag.target == disk
    assert [move.direction for move in zigzag.moves] == ["collapse", "expansion"]
    zigzag.verify()
    assert SimplicialSimpleHomotopyZigzag.from_json(zigzag.to_json()).target == disk


def test_chain_complex_is_the_canonical_homalg_type() -> None:
    circle = SimplicialComplex([[0, 1], [1, 2], [0, 2]])
    chains = circle.chain_complex(Z)

    assert type(chains) is ChainComplex
    assert chains.degrees == [0, 1]
    assert chains.module_dimension(0) == 3
    assert chains.module_dimension(1) == 3
    chains.verify()


def test_alexander_whitney_cochains_and_cup_products() -> None:
    interval = SimplicialComplex([[0, 1]])
    cochains = interval.cochain_complex(Z)
    cochains.verify()
    assert cochains.cohomology(0).free_rank == 1

    left = Vector.from_values(Z, [2, 3])
    right = Vector.from_values(Z, [5])
    assert interval.cup_product(0, left, 1, right).to_values() == [10]
    assert interval.cup_product(1, right, 0, left).to_values() == [15]
    assert interval.alexander_whitney_diagonal(1, 0) == [
        ([0], [0, 1]),
        ([0, 1], [1]),
    ]
    with pytest.raises(ValueError, match="left degree-0 cochain"):
        interval.cup_product(0, right, 0, right)


def test_characteristic_two_cup_i_and_steenrod_square() -> None:
    triangle = SimplicialComplex([[0, 1, 2]])
    field = FiniteField.prime(2)
    left = Vector.from_values(field, [1, 0, 0])
    right = Vector.from_values(field, [0, 0, 1])
    assert triangle.cup_i(0, 1, left, 1, right) == triangle.cup_product(1, left, 1, right)

    top = Vector.from_values(field, [1])
    assert triangle.cup_i(2, 2, top, 2, top) == top
    assert triangle.steenrod_square(0, 2, top) == top

    with pytest.raises(ValueError, match="characteristic 2"):
        odd = FiniteField.prime(3)
        triangle.cup_i(
            2,
            2,
            Vector.from_values(odd, [1]),
            2,
            Vector.from_values(odd, [1]),
        )


def test_complete_finite_field_cohomology_ring_has_cocycles_and_products() -> None:
    projective_plane = SimplicialComplex(RP2)
    ring = projective_plane.cohomology_ring(FiniteField.prime(2))

    assert type(ring) is SimplicialCohomologyRing
    assert [basis.degree for basis in ring.basis] == [0, 1, 2]
    assert [basis.id for basis in ring.basis] == ["h0_0", "h1_0", "h2_0"]
    assert ring.product(1, 1)[0].basis == 2
    assert ring.steenrod_square(0, 1)[0].basis == 1
    assert ring.steenrod_square(1, 1)[0].basis == 2
    assert ring.steenrod_square(2, 1) == []
    square = AdmissibleCohomologyOperation.steenrod_square(1)
    assert square.display_name == "Sq^1"
    assert ring.apply_operation(square, 1)[0].basis == 2
    assert ring.presentation is not None
    assert ring.presentation.hilbert_function(3) == [1, 1, 1, 0]
    ring.verify()
    assert SimplicialCohomologyRing.from_json(ring.to_json()) == ring


def test_simplicial_morse_reduction_builds_a_verified_chain_model() -> None:
    simplex = SimplicialComplex([list(range(9))])
    reduction = simplex.morse_reduction(Z)

    assert type(reduction) is SimplicialMorseReduction
    assert reduction.source == simplex
    assert reduction.original_cells == 511
    assert sum(reduction.critical_cells) == 1
    assert reduction.matched_pairs == 255
    assert reduction.reduced == simplex.morse_chain_complex(Z)
    reduction.verify()


def test_isolated_vertices_are_not_lost() -> None:
    points = SimplicialComplex([[0], [1], [2]])
    homology = points.homology(0, ring=Q)

    assert type(homology) is HomologyGroup
    assert homology.dimension == 3
    assert homology.free_rank == 3
    assert homology.torsion_coefficients == []
    homology.verify()


def test_projective_plane_distinguishes_q_f2_and_z() -> None:
    projective_plane = SimplicialComplex(RP2)
    rational = projective_plane.homology(1, ring=Q)
    mod_two = projective_plane.homology(1, ring=FiniteField.prime(2))
    integral = projective_plane.homology(1)
    invariants = projective_plane.homology_invariants(1)

    assert projective_plane.betti_number(1) == 0
    assert projective_plane.betti_number(1, ring=FiniteField.prime(2)) == 1
    with pytest.raises(ValueError, match="field coefficients"):
        projective_plane.betti_number(1, ring=Z)
    assert rational.dimension == 0
    assert mod_two.dimension == 1
    assert integral.free_rank == 0
    assert integral.torsion_coefficients == [2]
    assert type(invariants) is FinitelyGeneratedAbelianGroup
    assert invariants.free_rank == integral.free_rank
    assert invariants.torsion_coefficients == integral.torsion_coefficients
    invariants.verify()
    with pytest.raises(AttributeError, match="no vector-space dimension"):
        _ = integral.dimension
    with pytest.raises(AttributeError, match="transform-producing"):
        _ = integral.cycles
    integral.verify()


def test_graded_integral_homology_uses_one_result_identity() -> None:
    complex_ = SimplicialComplex(RP2)
    graded = complex_.graded_homology()
    lean = complex_.graded_homology_invariants()

    assert type(graded) is GradedHomology
    assert [group.degree for group in graded.groups] == [0, 1, 2]
    assert [group.free_rank for group in graded.groups] == [1, 0, 0]
    assert [group.torsion_coefficients for group in graded.groups] == [[], [2], []]
    assert type(graded.get(1)) is HomologyGroup
    graded.verify()
    assert all(type(group) is FinitelyGeneratedAbelianGroup for group in lean)
    assert [group.free_rank for group in lean] == [1, 0, 0]
    assert [group.torsion_coefficients for group in lean] == [[], [2], []]
