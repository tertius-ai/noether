# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import pytest

from noether import Q, Z
from noether.homalg import ChainComplex, GradedHomology, HomologyGroup
from noether.linalg import Matrix, SolutionSpace, Vector
from noether.numthy import FiniteField


def test_additive_cohomology_acceptance_workload() -> None:
    field = FiniteField.prime(3)
    delta_zero = Matrix.from_rows(field, [[0]])
    cochains = ChainComplex.from_coboundaries(field, {0: delta_zero})

    graded = cochains.graded_cohomology()
    assert isinstance(graded, GradedHomology)
    assert [group.degree for group in graded.groups] == [0, 1]
    assert [group.dimension for group in graded.groups] == [1, 1]
    for group in graded.groups:
        assert isinstance(group, HomologyGroup)
        assert group.is_cohomology
        assert isinstance(group.cycles, SolutionSpace)
        assert isinstance(group.representatives[0], Vector)
        group.verify()
    graded.verify()
    cochains.verify()


def test_integral_cohomology_keeps_torsion_evidence() -> None:
    cochains = ChainComplex.from_coboundaries(Z, {0: Matrix.from_rows(Z, [[2]])})
    invariants = cochains.cohomology_invariants(1)
    graded = cochains.graded_cohomology()

    degree_zero = graded.get(0)
    degree_one = graded.get(1)
    assert degree_zero is not None
    assert degree_one is not None
    assert degree_zero.free_rank == 0
    assert degree_zero.torsion_coefficients == []
    assert degree_one.free_rank == 0
    assert degree_one.torsion_coefficients == [2]
    assert invariants.free_rank == degree_one.free_rank
    assert invariants.torsion_coefficients == degree_one.torsion_coefficients
    graded.verify()


def test_boundary_and_coboundary_presentations_share_one_type() -> None:
    chains = ChainComplex.from_boundaries(Q, {1: Matrix.from_rows(Q, [[0]])})
    cochains = ChainComplex.from_coboundaries(Q, {0: Matrix.from_rows(Q, [[0]])})

    assert type(chains) is type(cochains) is ChainComplex
    assert chains.type_id == cochains.type_id == "noether/homalg/ChainComplex@1"
    assert [group.dimension for group in chains.graded_homology().groups] == [1, 1]
    assert [group.dimension for group in cochains.graded_cohomology().groups] == [1, 1]
    with pytest.raises(ValueError, match="from_coboundaries"):
        chains.graded_cohomology()
    with pytest.raises(ValueError, match="from_boundaries"):
        cochains.graded_homology()

    with pytest.raises(ValueError, match="do not form a complex"):
        ChainComplex.from_boundaries(
            Q,
            {
                2: Matrix.from_rows(Q, [[1]]),
                1: Matrix.from_rows(Q, [[1]]),
            },
        )


def test_unverified_ring_presentations_are_not_public() -> None:
    import noether.groups as groups
    import noether.homalg as homalg

    assert not hasattr(homalg, "CohomologyRing")
    assert not hasattr(homalg, "group_cohomology")
    assert not hasattr(groups, "group_cohomology")
