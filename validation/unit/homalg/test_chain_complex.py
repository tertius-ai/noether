# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

from types import MappingProxyType

import pytest

from noether import Q
from noether.homalg import (
    ChainCancellation,
    ChainComplex,
    ChainReduction,
    FinitelyGeneratedAbelianGroup,
    GradedHomology,
    HomologyGroup,
)
from noether.linalg import Matrix, SolutionSpace, SparseMatrix, Vector
from noether.numthy import FiniteField


def test_circle_homology_is_checked_and_canonical() -> None:
    boundary = Matrix.from_rows(Q, [[0]])
    circle = ChainComplex.from_boundaries(Q, {1: boundary})

    assert circle.degrees == [0, 1]
    assert circle.differential(1) == boundary
    assert circle.differential(0) is None
    assert circle.module_dimension(0) == 1
    assert circle.module_dimension(1) == 1
    assert circle.module_dimension(2) == 0
    assert circle.betti_number(0) == 1
    assert circle.betti_number(1) == 1
    assert circle.betti_number(2) == 0
    h0 = circle.homology(0)
    h1 = circle.homology(1)
    assert isinstance(h0, HomologyGroup)
    assert h0.dimension == h1.dimension == 1
    assert not h0.is_cohomology
    assert isinstance(h0.cycles, SolutionSpace)
    assert isinstance(h0.representatives[0], Vector)
    assert h0.validity.is_unconditional
    h0.verify()
    h1.verify()
    circle.verify()
    assert type(circle).__module__ == "noether.homalg"
    assert type(h0).__module__ == "noether.homalg"
    assert ChainComplex.type_id == "noether/homalg/ChainComplex@1"
    assert HomologyGroup.type_id == "noether/homalg/HomologyGroup@1"


def test_nonzero_boundaries_are_quotiented_with_representatives() -> None:
    d1 = Matrix.from_rows(Q, [[0, 0]])
    d2 = Matrix.from_rows(Q, [[1], [0]])
    complex_ = ChainComplex.from_boundaries(Q, {1: d1, 2: d2})

    h1 = complex_.homology(1)
    assert h1.cycles.dimension == 2
    assert h1.boundaries.dimension == 1
    assert h1.dimension == 1
    assert h1.representatives[0].to_values() == [0, 1]
    h1.verify()

    graded = complex_.graded_homology()
    assert isinstance(graded, GradedHomology)
    assert graded.coefficient_context == Q
    assert [group.degree for group in graded.groups] == [0, 1, 2]
    assert graded.get(1) == h1
    assert graded.get(9) is None
    graded.verify()


def test_universal_unit_reduction_is_replayable() -> None:
    complex_ = ChainComplex.from_boundaries(Q, {1: Matrix.from_rows(Q, [[1, 2], [3, 8]])})
    reduction = complex_.algebraic_morse_reduction()

    assert type(reduction) is ChainReduction
    assert reduction.source == complex_
    assert reduction.removed_pairs == 1
    assert reduction.reduced.module_dimension(0) == 1
    assert reduction.reduced.module_dimension(1) == 1
    assert len(reduction.cancellations) == 1
    cancellation = reduction.cancellations[0]
    assert type(cancellation) is ChainCancellation
    assert (cancellation.degree, cancellation.row, cancellation.column) == (1, 0, 0)
    assert cancellation.unit == 1
    reconstructed = ChainCancellation(
        cancellation.degree, cancellation.row, cancellation.column, cancellation.unit
    )
    assert reconstructed == cancellation
    assert complex_.reduction_from_cancellations([reconstructed]).reduced == reduction.reduced
    reduction.verify()

    with pytest.raises(ValueError, match="unit must be"):
        ChainCancellation(1, 0, 0, 2)  # type: ignore[arg-type]


def test_coboundary_constructor_reindexes_without_a_second_type() -> None:
    delta0 = Matrix.from_rows(Q, [[-1, 1]])
    cochains = ChainComplex.from_coboundaries(Q, MappingProxyType({0: delta0}))

    assert cochains.degrees == [0, 1]
    assert cochains.module_dimension(0) == 2
    assert cochains.module_dimension(1) == 1
    assert cochains.betti_number(0) == 1
    assert cochains.betti_number(1) == 0
    assert cochains.cohomology(0).dimension == 1
    assert cochains.cohomology(0).is_cohomology
    assert cochains.cohomology(1).dimension == 0
    with pytest.raises(ValueError, match="from_boundaries"):
        cochains.homology(0)


def test_extension_field_uses_the_same_python_surface() -> None:
    field = FiniteField.extension(3, 2)
    boundary = Matrix.from_rows(field, [[0]])
    complex_ = ChainComplex.from_boundaries(field, {1: boundary})

    h1 = complex_.homology(1)
    assert h1.dimension == 1
    assert h1.coefficient_context == field
    h1.verify()


def test_rank_one_noncomplex_and_mixed_contexts_fail_closed() -> None:
    one = Matrix.from_rows(Q, [[1]])
    with pytest.raises(ValueError, match="compose to zero|nonzero"):
        ChainComplex.from_boundaries(Q, {1: one, 2: one})

    field = FiniteField.prime(3)
    wrong = Matrix.from_rows(field, [[0]])
    with pytest.raises(ValueError, match="coefficient context Q"):
        ChainComplex.from_boundaries(Q, {1: wrong})


def test_betti_number_requires_field_coefficients() -> None:
    from noether import Z

    chains = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[0]])})
    with pytest.raises(ValueError, match="field coefficients"):
        chains.betti_number(1)


def test_integral_homology_invariants_are_a_lean_canonical_group() -> None:
    from noether import Z, _engine

    chains = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[6]])})
    group = chains.homology_invariants(0)

    assert type(group) is FinitelyGeneratedAbelianGroup
    assert group.type_id == "noether/homalg/FinitelyGeneratedAbelianGroup@1"
    assert group.free_rank == 0
    assert group.torsion_coefficients == [6]
    assert "torsion_summands=1" in repr(group)
    group.verify()

    type_id, payload = _engine._dump(group)
    restored = _engine._hydrate(type_id, payload)
    assert type(restored) is FinitelyGeneratedAbelianGroup
    assert restored == group

    graded = chains.graded_homology_invariants()
    assert list(graded) == [0, 1]
    assert graded[0] == group
    assert graded[1].free_rank == 0
    assert graded[1].torsion_coefficients == []

    cochains = ChainComplex.from_coboundaries(Z, {0: Matrix.from_rows(Z, [[6]])})
    graded_cohomology = cochains.graded_cohomology_invariants()
    assert list(graded_cohomology) == [0, 1]
    assert graded_cohomology[0].free_rank == 0
    assert graded_cohomology[1].torsion_coefficients == [6]

    with pytest.raises(ValueError, match="Z coefficients"):
        ChainComplex.from_boundaries(Q, {1: Matrix.from_rows(Q, [[0]])}).homology_invariants(0)
    with pytest.raises(ValueError, match="Z coefficients"):
        ChainComplex.from_boundaries(
            Q, {1: Matrix.from_rows(Q, [[0]])}
        ).graded_homology_invariants()


def test_sparse_integral_complex_uses_the_same_homology_surface() -> None:
    from noether import Z

    boundary = SparseMatrix.from_triplets(Z, (1, 1), [(0, 0, 6)])
    chains = ChainComplex.from_boundaries(Z, {1: boundary})

    assert chains.differential_layout(1) == "sparse"
    assert chains.homology_invariants(0).torsion_coefficients == [6]
    reduction = chains.algebraic_morse_reduction()
    assert reduction.reduced.differential_layout(1) == "sparse"
    reduction.verify()
