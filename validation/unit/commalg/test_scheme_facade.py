# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
import json

import pytest

from noether import Q, UnsupportedOperationError
from noether.commalg import (
    AffineSpace,
    BlowUp,
    BlowUpChart,
    FiniteSchemeClosedPoint,
    FiniteSchemeClosedPointSet,
    PolynomialRing,
    ProjectiveSpace,
    RationalPointEnumeration,
    Scheme,
    SchemeCompletion,
    SchemePoint,
    SeriesTruncation,
)
from noether.numthy import FiniteField, PointCountCertificate


def test_affine_points_and_partitioned_count_agree() -> None:
    field = FiniteField.prime(3)
    ambient = AffineSpace(field, 2, names=["x", "y"])
    ring = ambient.coordinate_ring
    x, y = ring.generators()
    scheme = ambient.closed_subscheme(ring.ideal([x * y - 1]))

    points = list(scheme.rational_points())
    count, certificate = scheme.point_count(shards=4)

    assert len(points) == count == certificate.point_count == 2
    assert scheme.dimension() == 1
    assert scheme.degree() == 2
    assert scheme.hilbert_polynomial().constant_coefficient() == 2
    assert scheme.is_smooth()
    assert scheme.singular_subscheme().dimension() == -1
    assert certificate.candidate_count == 9
    assert certificate.model_checksum > 0
    assert certificate.shard_candidate_counts == [3, 2, 2, 2]
    assert sum(certificate.shard_point_counts) == 2
    certificate.verify(scheme)
    assert PointCountCertificate.from_json(certificate.to_json()) == certificate

    altered = json.loads(certificate.to_json())
    altered["shard_checksums"][0] ^= 1
    altered_certificate = PointCountCertificate.from_json(json.dumps(altered))
    with pytest.raises(ValueError, match="does not match"):
        altered_certificate.verify(scheme)

    line = AffineSpace(field, 1)
    other_scheme = line.closed_subscheme(line.coordinate_ring.ideal([]))
    with pytest.raises(ValueError, match="does not match"):
        certificate.verify(other_scheme)


def test_large_field_enumeration_streams_and_replays() -> None:
    for field in [FiniteField.prime(4099), FiniteField.extension(3, 8)]:
        ambient = AffineSpace(field, 1, names=["x"])
        (x,) = ambient.coordinate_ring.generators()
        scheme = ambient.closed_subscheme(ambient.coordinate_ring.ideal([x**2 - x]))
        count, certificate = scheme.point_count(shards=3)
        assert count == 2
        certificate.verify(scheme)
        points = scheme.rational_points()
        first = next(points)
        resumed = RationalPointEnumeration.from_json(points.to_json())
        assert [first, *resumed] == list(scheme.rational_points())


def test_projective_points_have_unique_normalized_coordinates() -> None:
    field = FiniteField.prime(3)
    ambient = ProjectiveSpace(field, 1, names=["u", "v"])
    whole_space = ambient.closed_subscheme(ambient.coordinate_ring.ideal([]))

    points = list(whole_space.rational_points())
    assert len(points) == 4
    assert len({tuple(point.coordinates) for point in points}) == 4
    assert whole_space.dimension() == 1
    assert whole_space.degree() == 1
    hilbert_ring = PolynomialRing(Q, ["m"])
    m = hilbert_ring.generator(0)
    assert whole_space.hilbert_polynomial() == m + 1

    point = whole_space.point([field(2), field(1)])
    assert point.coordinates == [field.one(), field(2)]
    point.verify()
    assert SchemePoint.from_json(point.to_json()) == point

    u, _ = ambient.coordinate_ring.generators()
    with pytest.raises(ValueError, match="homogeneous"):
        ambient.closed_subscheme(ambient.coordinate_ring.ideal([u + 1]))


def test_extension_fields_use_the_same_enumerator() -> None:
    field = FiniteField.extension(3, 2)
    ambient = AffineSpace(field, 1)
    scheme = ambient.closed_subscheme(ambient.coordinate_ring.ideal([]))

    assert len(list(scheme.rational_points())) == 9
    assert scheme.point_count(shards=3)[0] == 9


def test_finite_scheme_solver_returns_verified_closed_points_and_orbits() -> None:
    field = FiniteField.prime(3)
    line = AffineSpace(field, 1, names=["x"])
    x = line.coordinate_ring.generator(0)
    scheme = line.closed_subscheme(line.coordinate_ring.ideal([x**2 + 1]))

    solution = scheme.closed_points()
    assert isinstance(solution, FiniteSchemeClosedPointSet)
    assert solution.scheme == scheme
    assert solution.geometric_point_count == 2
    assert len(solution.points) == 1
    point = solution.points[0]
    assert isinstance(point, FiniteSchemeClosedPoint)
    assert point.degree == 2
    assert point.residue_field.degree == 2
    assert len(point.frobenius_orbit) == 2
    point.verify()
    solution.verify()
    assert FiniteSchemeClosedPoint.from_json(point.to_json()) == point
    assert FiniteSchemeClosedPointSet.from_json(solution.to_json()) == solution


def test_closed_point_solver_does_not_enumerate_a_huge_prime_field() -> None:
    field = FiniteField.prime((1 << 127) - 1)
    line = AffineSpace(field, 1, names=["x"])
    x = line.coordinate_ring.generator(0)
    solution = line.closed_subscheme(line.coordinate_ring.ideal([x - 42])).closed_points()

    assert solution.geometric_point_count == 1
    assert solution.points[0].degree == 1
    assert solution.points[0].coordinates == [field(42)]


def test_same_projective_model_matches_slow_and_compiled_counts_over_f3_and_f9() -> None:
    for field, expected in [
        (FiniteField.prime(3), 4),
        (FiniteField.extension(3, 2), 10),
    ]:
        plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
        ring = plane.coordinate_ring
        x, y, z = ring.generators()
        conic = plane.closed_subscheme(ring.ideal([x * z - y**2]))

        slow_count = len(list(conic.rational_points()))
        first = conic.point_count(shards=4)
        second = conic.point_count(shards=4)

        assert slow_count == first[0] == expected
        assert first == second
        assert sum(first[1].shard_candidate_counts) == first[1].candidate_count
        first[1].verify(conic)


def test_enumeration_resume_and_shards_are_deterministic() -> None:
    field = FiniteField.prime(3)
    ambient = AffineSpace(field, 2)
    scheme = ambient.closed_subscheme(ambient.coordinate_ring.ideal([]))
    expected = list(scheme.rational_points())

    live = scheme.rational_points()
    assert live.next_candidate_index == 0
    first = next(live)
    assert live.next_candidate_index == 1
    resumed = RationalPointEnumeration.from_json(live.to_json())
    assert [first, *list(resumed)] == expected

    completed = scheme.rational_points(shard_index=2, shard_count=4)
    list(completed)
    assert list(RationalPointEnumeration.from_json(completed.to_json())) == []

    sharded = []
    for index in range(4):
        sharded.extend(scheme.rational_points(shard_index=index, shard_count=4))
    assert {point.to_json() for point in sharded} == {point.to_json() for point in expected}


def test_scheme_json_revalidates_context_and_rational_scope() -> None:
    ambient = AffineSpace(Q, 2, names=["x", "y"])
    ring = ambient.coordinate_ring
    scheme = ambient.closed_subscheme(ring.ideal([ring.generator("x")]))

    assert scheme.is_affine
    assert not scheme.is_projective
    assert Scheme.from_json(scheme.to_json()) == scheme
    with pytest.raises(UnsupportedOperationError) as raised:
        scheme.point_count()
    assert raised.value.operation == "noether/commalg/Scheme/point_count@1"
    assert raised.value.supported_contexts == (FiniteField.type_id,)

    malformed = json.loads(scheme.to_json())
    malformed["ambient_kind"] = "unknown"
    with pytest.raises(ValueError):
        Scheme.from_json(json.dumps(malformed))

    other = AffineSpace(Q, 2, names=["u", "v"])
    with pytest.raises((TypeError, ValueError), match="context mismatch"):
        ambient.closed_subscheme(other.coordinate_ring.ideal([]))


def test_jacobian_singular_locus_is_exact_and_fails_closed() -> None:
    ambient = AffineSpace(Q, 2, names=["x", "y"])
    ring = ambient.coordinate_ring
    x, y = ring.generators()
    cusp = ambient.closed_subscheme(ring.ideal([y**2 - x**3]))

    singular = cusp.singular_subscheme()
    assert singular.defining_ideal == ring.ideal([x**2, y])
    assert singular.dimension() == 0
    assert cusp.jacobian_subscheme() == singular
    assert not cusp.is_smooth()

    cone_ambient = AffineSpace(Q, 4, names=["x", "y", "z", "w"])
    ring = cone_ambient.coordinate_ring
    x, y, z, w = ring.generators()
    cone = cone_ambient.closed_subscheme(ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2]))
    assert cone.jacobian_subscheme().dimension() == 0
    with pytest.raises(UnsupportedOperationError) as raised:
        cone.singular_subscheme()
    assert "complete-intersection" in raised.value.detail
    with pytest.raises(UnsupportedOperationError) as raised:
        cone.is_smooth()
    assert "complete-intersection" in raised.value.detail


def test_projective_cusp_blow_up_tangent_cone_and_completion() -> None:
    field = FiniteField.prime(5)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    ring = plane.coordinate_ring
    x, y, z = ring.generators()
    cusp = plane.closed_subscheme(ring.ideal([y**2 * z - x**3]))
    center = cusp.point([0, 0, 1])

    blow_up = cusp.blow_up_at(center)
    assert blow_up.source == cusp
    assert blow_up.local_source.is_affine
    assert blow_up.center == center
    assert blow_up.source_affine_chart == 2
    assert len(blow_up.charts) == 2
    blow_up.verify()

    chart = blow_up.charts[0]
    chart.verify()
    chart_ring = chart.strict_transform.coordinate_ring
    exceptional, ratio = chart_ring.generators()
    expected_strict = chart_ring.ideal([exceptional - ratio**2])
    assert chart.weak_transform.defining_ideal == expected_strict
    assert chart.strict_transform.defining_ideal == expected_strict
    assert chart.strict_transform.is_smooth()
    assert chart.exceptional_divisor.defining_ideal == chart_ring.ideal([exceptional, ratio**2])
    assert chart.exceptional_divisor.dimension() == 0
    assert chart.exceptional_divisor.degree() == 2
    assert chart.source_coordinate_map == [
        exceptional,
        exceptional * ratio,
        chart_ring.one(),
    ]
    assert chart.total_transform.defining_ideal.generators == [
        exceptional**2 * chart.weak_transform.defining_ideal.generators[0]
    ]

    tangent = cusp.tangent_cone(center)
    _, tangent_y = tangent.coordinate_ring.generators()
    assert tangent.defining_ideal == tangent.coordinate_ring.ideal([tangent_y**2])
    assert tangent.dimension() == 1
    assert tangent.degree() == 2

    residual = chart.strict_transform.point([0, 0])
    completion = chart.strict_transform.completion(residual, SeriesTruncation.total_degree(4))
    completion.verify()
    series_exceptional, series_ratio = completion.ambient_ring.generators()
    assert completion.equations == [series_ratio**2 - series_exceptional]
    assert completion.truncation.bound == 4

    assert BlowUp.from_json(blow_up.to_json()) == blow_up
    assert BlowUpChart.from_json(chart.to_json()) == chart
    assert SchemeCompletion.from_json(completion.to_json()) == completion
    malformed = json.loads(blow_up.to_json())
    malformed["charts"][0]["chart_index"] = 1
    with pytest.raises(ValueError, match="does not match"):
        BlowUp.from_json(json.dumps(malformed))
