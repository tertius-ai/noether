# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact hand checks for complete-intersection Chow and HRR arithmetic."""

from fractions import Fraction

import pytest

from noether import Assumption
from noether.commalg import (
    ChowClass,
    ChowRing,
    CompleteIntersectionHypotheses,
    HirzebruchRiemannRochResult,
    ProjectiveCompleteIntersection,
    RationalChowClass,
)


def test_chow_ring_of_p1_times_p2_is_exact_and_canonical() -> None:
    ring = ChowRing([1, 2])
    h1, h2 = ring.generators()

    assert ring.projective_dimensions == [1, 2]
    assert ring.factor_count == 2
    assert ring.dimension == 3
    assert h1**2 == ring.zero()
    assert h2**3 == ring.zero()
    assert ring.integrate(h1 * h2**2) == 1
    assert ring.integrate(ring.one() + (h1 * h2**2).scalar_mul(7)) == 7
    assert (h1 + h2).codimension == 1
    assert (ring.one() + h1).codimension is None

    value = ring.from_terms([([0, 0], 2**100), ([1, 2], -3)])
    value.verify()
    assert ChowClass.from_json(value.to_json()) == value
    rational = value.over_rationals().scalar_mul(Fraction(2, 3))
    rational.verify()
    assert RationalChowClass.from_json(rational.to_json()) == rational


def test_plane_cubic_records_smoothness_as_an_assumption() -> None:
    ring = ChowRing([2])
    (h,) = ring.generators()
    cubic = ProjectiveCompleteIntersection.assuming_smooth(ring, [[3]])

    assert cubic.hypotheses == CompleteIntersectionHypotheses.Smooth
    assert cubic.validity.assumptions == [Assumption.smooth_complete_intersection()]
    assert cubic.dimension == 1
    assert cubic.codimension == 1
    assert cubic.fundamental_class() == h.scalar_mul(3)
    assert cubic.integrate_restriction(h) == 3
    assert cubic.tangent_total_chern_class().terms() == [([0], 1)]

    structure_sheaf = cubic.line_bundle_euler_characteristic([0])
    hyperplane = cubic.line_bundle_euler_characteristic([1])
    assert structure_sheaf.value == structure_sheaf.koszul_value == 0
    assert structure_sheaf.hrr_integral == Fraction(0)
    assert hyperplane.value == hyperplane.koszul_value == 3
    structure_sheaf.verify()
    assert HirzebruchRiemannRochResult.from_json(structure_sheaf.to_json()) == structure_sheaf
    assert ProjectiveCompleteIntersection.from_json(cubic.to_json()) == cubic


def test_bidegree_two_two_curve_has_genus_one() -> None:
    ring = ChowRing([1, 1])
    h1, h2 = ring.generators()
    curve = ProjectiveCompleteIntersection.assuming_smooth(ring, [[2, 2]])

    assert curve.integrate_restriction(h1) == 2
    assert curve.integrate_restriction(h2) == 2
    assert curve.line_bundle_euler_characteristic([0, 0]).value == 0
    assert curve.line_bundle_euler_characteristic([1, 0]).value == 2


def test_regular_sequence_lane_refuses_smooth_only_formulas() -> None:
    ring = ChowRing([3])
    complete_intersection = ProjectiveCompleteIntersection.assuming_regular_sequence(
        ring, [[2], [2]]
    )

    assert complete_intersection.hypotheses == CompleteIntersectionHypotheses.RegularSequence
    assert complete_intersection.validity.assumptions == [Assumption.regular_sequence()]
    complete_intersection.fundamental_class().verify()
    complete_intersection.normal_total_chern_class().verify()
    assert complete_intersection.line_bundle_euler_characteristic_koszul([0]) == 0
    with pytest.raises(ValueError, match="smoothness hypothesis"):
        complete_intersection.tangent_total_chern_class()
    with pytest.raises(ValueError, match="smoothness hypothesis"):
        complete_intersection.line_bundle_euler_characteristic([0])


def test_context_and_shape_errors_fail_closed() -> None:
    first = ChowRing([1])
    second = ChowRing([2])
    with pytest.raises(TypeError, match="different rings"):
        _ = first.generator(0) + second.generator(0)
    with pytest.raises(ValueError, match="expected a multidegree"):
        ProjectiveCompleteIntersection.assuming_smooth(first, [[1, 2]])
    with pytest.raises(ValueError, match="cannot form a complete intersection"):
        ProjectiveCompleteIntersection.assuming_smooth(first, [[1], [1]])
    with pytest.raises(ValueError, match="nonempty proper complete intersection"):
        ProjectiveCompleteIntersection.assuming_smooth(first, [[0]])


def test_typed_json_rejects_wrong_semantic_ids() -> None:
    ring = ChowRing([2])
    encoded = ring.to_json().replace(ChowRing.type_id, ChowClass.type_id)
    with pytest.raises(ValueError, match="expected type"):
        ChowRing.from_json(encoded)
