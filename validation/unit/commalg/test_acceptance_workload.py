# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The four bounded commutative-algebra and geometry release calculations."""

import json

import pytest

from noether import Q, UnsupportedOperationError
from noether.commalg import (
    AffineSpace,
    IdealDecomposition,
    PolynomialRing,
    PowerSeries,
    PowerSeriesRing,
    ProjectiveSpace,
    RationalFunctionField,
    SchemeCompletion,
    SeriesTruncation,
)
from noether.numthy import FiniteField, PadicRing, PointCountCertificate, QadicRing


def test_context_tower_drives_a_weighted_witt_power_series() -> None:
    residue_field = FiniteField.extension(3, 2)
    integers = PadicRing(3, absolute_precision=6)
    witt_ring = QadicRing(residue_field, absolute_precision=6)
    truncation = SeriesTruncation.weighted_degree([1, 2, 3], 6)
    series_ring = PowerSeriesRing(witt_ring, ["x", "y", "z"], truncation)
    x, y, z = series_ring.generators()

    residue = integers.residue_field(2)
    lift = integers.teichmuller(residue)
    assert lift**3 == lift
    assert lift.residue() == residue

    a = residue_field.generator() + residue_field.one()
    witt_lift = witt_ring.teichmuller(a)
    assert witt_lift.frobenius(2) == witt_lift
    assert witt_lift.residue() == a
    assert witt_ring.one().trace() == integers(2)
    assert witt_ring(3).norm() == integers(9)

    value = 1 + x + y + x * y**2 + z
    expected_exponents = {
        (i, j, k) for i in range(6) for j in range(3) for k in range(2) if i + 2 * j + 3 * k < 6
    }
    assert len(expected_exponents) == 16
    assert series_ring.coefficient_capacity_bound() == 36
    assert value.nonzero_term_count == 5
    assert (
        value.restrict_truncation(SeriesTruncation.weighted_degree([1, 2, 3], 4)).nonzero_term_count
        == 4
    )
    assert PowerSeries.from_json(value.to_json()) == value


def test_generic_components_split_only_after_the_checked_base_change() -> None:
    base = FiniteField.prime(3)
    function_field = RationalFunctionField(base, "c")
    ring = PolynomialRing(function_field, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x**2 + y**2])

    embedded = function_field.from_polynomial(function_field.polynomial_ring.generator(0) + 1)
    assert embedded == function_field.generator() + 1

    with pytest.raises(UnsupportedOperationError) as raised:
        source.minimal_primes()
    assert raised.value.operation == "noether/commalg/Ideal/minimal_primes@1"
    assert raised.value.context == RationalFunctionField.type_id
    assert "split completely" in raised.value.detail

    changed = source.base_change_constant_field(FiniteField.extension(3, 2))
    decomposition = changed.minimal_primes()

    assert decomposition.frobenius_permutation == [1, 0]
    assert [component.dimension() for component in decomposition.components] == [1, 1]
    assert [component.degree() for component in decomposition.components] == [1, 1]
    hilbert_ring = PolynomialRing(Q, ["m"])
    assert decomposition.hilbert_polynomials() == [
        hilbert_ring.one(),
        hilbert_ring.one(),
    ]
    decomposition.verify()
    decomposition.verify_frobenius()
    assert IdealDecomposition.from_json(decomposition.to_json()) == decomposition


def test_cusp_blow_up_keeps_chart_tangent_and_completion_evidence() -> None:
    field = FiniteField.prime(5)
    plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
    ring = plane.coordinate_ring
    x, y, z = ring.generators()
    cusp = plane.closed_subscheme(ring.ideal([y**2 * z - x**3]))
    center = cusp.point([0, 0, 1])

    blow_up = cusp.blow_up_at(center)
    blow_up.verify()
    assert len(blow_up.charts) == 2
    for candidate_chart in blow_up.charts:
        candidate_chart.verify()
        assert candidate_chart.source == blow_up.source
        assert candidate_chart.center == blow_up.center
    chart = blow_up.charts[0]
    exceptional, ratio = chart.strict_transform.coordinate_ring.generators()
    assert chart.strict_transform.defining_ideal == chart.strict_transform.coordinate_ring.ideal(
        [exceptional - ratio**2]
    )
    assert chart.strict_transform.is_smooth()

    tangent = cusp.tangent_cone(center)
    _, tangent_y = tangent.coordinate_ring.generators()
    assert tangent.defining_ideal == tangent.coordinate_ring.ideal([tangent_y**2])
    assert (tangent.dimension(), tangent.degree()) == (1, 2)

    residual = chart.strict_transform.point([0, 0])
    completion = chart.strict_transform.completion(residual, SeriesTruncation.total_degree(4))
    completion.verify()
    series_exceptional, series_ratio = completion.ambient_ring.generators()
    assert completion.equations == [series_ratio**2 - series_exceptional]
    assert SchemeCompletion.from_json(completion.to_json()) == completion


def test_compiled_projective_counts_match_the_slow_enumerator() -> None:
    for field, expected in [
        (FiniteField.prime(3), 4),
        (FiniteField.extension(3, 2), 10),
    ]:
        plane = ProjectiveSpace(field, 2, names=["x", "y", "z"])
        ring = plane.coordinate_ring
        x, y, z = ring.generators()
        conic = plane.closed_subscheme(ring.ideal([x * z - y**2]))

        count, certificate = conic.point_count(shards=4)
        assert count == len(list(conic.rational_points())) == expected
        assert sum(certificate.shard_candidate_counts) == certificate.candidate_count
        certificate.verify(conic)
        assert PointCountCertificate.from_json(certificate.to_json()) == certificate

        tampered = json.loads(certificate.to_json())
        tampered["shard_checksums"][0] ^= 1
        tampered_certificate = PointCountCertificate.from_json(json.dumps(tampered))
        with pytest.raises(ValueError, match="does not match"):
            tampered_certificate.verify(conic)

    affine = AffineSpace(FiniteField.prime(3), 1)
    line = affine.closed_subscheme(affine.coordinate_ring.ideal([]))
    assert line.point_count(shards=2)[0] == 3
