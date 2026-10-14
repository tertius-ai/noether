# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
import json

import pytest

from noether import Q
from noether.commalg import (
    CyclicHypersurfaceCover,
    FrobeniusPolynomial,
    GeometricDivisorRank,
    PointCountMethod,
    PointCountSequence,
    PolynomialRing,
    ProjectiveSpace,
    SplitCyclicPullback,
    WeightedProjectiveSpace,
    ZetaFunction,
)
from noether.numthy import FiniteField


def test_weighted_projective_context_normalizes_without_losing_quotient_data() -> None:
    space = WeightedProjectiveSpace(Q, [6, 2, 2, 2], names=["w", "x", "y", "z"])

    assert space.weights == [3, 1, 1, 1]
    assert space.is_well_formed()
    assert space.chart(0).stabilizer_order == 3
    assert not space.chart(0).is_affine_space()
    assert space.chart(1).is_affine_space()
    assert WeightedProjectiveSpace.from_json(space.to_json()) == space


def test_surface_point_counts_frobenius_and_independent_rank_bounds() -> None:
    field = FiniteField.prime(5)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    x, y, z = plane.coordinate_ring.generators()
    cover = CyclicHypersurfaceCover(plane, 2, x * z - y**2)

    smoothness = cover.smoothness_evidence()
    assert smoothness.branch_is_smooth is True
    assert smoothness.is_smooth()
    assert cover.weighted_realization.ambient.weights == [1, 1, 1, 1]

    counts = cover.point_counts([1])
    assert counts.count(1) == 36
    assert counts.counts[0].evidence.method == PointCountMethod.ExhaustiveCyclicFibers
    assert counts.counts[0].evidence.independently_verified()
    assert PointCountSequence.from_json(counts.to_json()) == counts

    forged = json.loads(counts.to_json())
    forged["counts"][0]["count"] = "35"
    with pytest.raises(ValueError, match="recomputation changed"):
        PointCountSequence.from_json(json.dumps(forged))

    forged = json.loads(counts.to_json())
    forged["counts"][0]["evidence"]["independently_verified"] = False
    with pytest.raises(ValueError, match="recomputation changed"):
        PointCountSequence.from_json(json.dumps(forged))

    cohomology = cover.surface_cohomology()
    assert cohomology.second_betti_number == 2
    assert cohomology.required_point_count_degree == 1
    frobenius = cohomology.frobenius_polynomial(counts)
    assert FrobeniusPolynomial.from_json(frobenius.to_json()) == frobenius
    zeta = frobenius.zeta_function()
    assert ZetaFunction.from_json(zeta.to_json()) == zeta

    parameter_ring = PolynomialRing(field, ["t"])
    t = parameter_ring.generator(0)
    split = SplitCyclicPullback(cover, [parameter_ring.one(), t, parameter_ring.zero()])
    lower = split.divisor_rank_lower_bound()
    upper = frobenius.geometric_divisor_rank_upper_bound()
    assert lower.lower_bound == 2
    assert upper.upper_bound == 2
    rank = upper.certify_exact(lower)
    assert rank.rank == 2
    assert GeometricDivisorRank.from_json(rank.to_json()) == rank


def test_padic_trace_dispatch_is_visible_without_exposing_a_backend_api() -> None:
    field = FiniteField.prime(11)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    x, y, z = plane.coordinate_ring.generators()
    cover = CyclicHypersurfaceCover(plane, 2, x * z - y**2)

    count = cover.point_counts([4]).counts[0]
    assert count.evidence.method == PointCountMethod.PadicDoubleCoverTrace
    assert count.evidence.precision is not None
    assert count.evidence.modulus is not None
    assert count.evidence.largest_matrix_dimension is not None
