# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Projection, geometry, and replay tests for Rust hyperelliptic Frobenius."""

import json

import pytest

from noether import NoetherError, UnsupportedOperationError
from noether.commalg import (
    FrobeniusPolynomial,
    HyperellipticCurve,
    HyperellipticFrobeniusEvidence,
    PolynomialRing,
    ZetaFunction,
)
from noether.numthy import FiniteField


def test_genus_four_characteristic_three_and_large_extension() -> None:
    ring = PolynomialRing(FiniteField.prime(3), ["x"])
    (x,) = ring.generators()
    curve = HyperellipticCurve(x**9 + x)
    assert curve.genus == 4
    assert curve.infinity_residue_degrees == [1]
    frobenius = curve.frobenius_polynomial()
    (t,) = frobenius.polynomial.ring.generators()
    assert frobenius.polynomial == t**8 + 18 * t**4 + 81
    assert frobenius.cohomology is None
    assert frobenius.point_counts is None
    assert frobenius.hyperelliptic_evidence.curve == curve
    assert frobenius.cohomological_degree == frobenius.cohomological_weight == 1
    assert len(repr(frobenius)) < 160
    counts = curve.cover.point_counts([1, 2, 3, 4])
    assert [frobenius.point_count(i) for i in range(1, 5)] == [r.count for r in counts.counts]
    assert frobenius.point_count(20) > 0
    assert curve.point_count() == frobenius.point_count()
    curve.verify()


def test_full_frobenius_from_direct_counts_has_independent_replay() -> None:
    ring = PolynomialRing(FiniteField.prime(3), ["x"])
    (x,) = ring.generators()
    curve = HyperellipticCurve(x**9 + x)
    counts = curve.cover.point_counts([1, 2, 3, 4])
    direct = FrobeniusPolynomial.from_hyperelliptic_counts(counts)
    assert direct.polynomial == curve.frobenius_polynomial().polynomial
    assert direct.degree == 8
    assert direct.point_counts == counts
    assert direct.hyperelliptic_evidence is None
    direct.verify()
    assert FrobeniusPolynomial.from_json(direct.to_json()) == direct
    forged = json.loads(direct.to_json())
    forged["field_order"] = "9"
    with pytest.raises(ValueError):
        FrobeniusPolynomial.from_json(json.dumps(forged))


def test_even_degree_nonsplit_infinity_and_general_equation() -> None:
    field = FiniteField.extension(3, 2)
    ring = PolynomialRing(field, ["x"])
    (x,) = ring.generators()
    curve = HyperellipticCurve.from_equation(x**5 + field.generator() * x + 1, x + 1)
    assert 2 * curve.completion_shift() == curve.h
    completed = HyperellipticCurve(curve.branch_equation)
    assert curve.frobenius_polynomial().polynomial == completed.frobenius_polynomial().polynomial
    prime_ring = PolynomialRing(FiniteField.prime(3), ["x"])
    (x,) = prime_ring.generators()
    twist = HyperellipticCurve(2 * x**6 + 2 * x + 2)
    assert twist.infinity_residue_degrees == [2]
    assert twist.smoothness_evidence().is_smooth()
    assert twist.point_count() == twist.cover.point_counts([1]).count(1)
    evidence = twist.frobenius_evidence()
    assert evidence.quadratic_twist_sign == -1
    assert evidence.normalized_branch == x**6 + x + 1
    assert evidence.frobenius_polynomial().trace(1) != 0


def test_wire_evidence_and_precision() -> None:
    ring = PolynomialRing(FiniteField.prime(3), ["x"])
    (x,) = ring.generators()
    curve = HyperellipticCurve(x**5 + x)
    assert HyperellipticCurve.from_json(curve.to_json()) == curve
    evidence = curve.frobenius_evidence()
    evidence.verify()
    assert evidence.working_precision == evidence.precision + evidence.denominator_loss
    assert HyperellipticFrobeniusEvidence.from_json(evidence.to_json()) == evidence
    frobenius = evidence.frobenius_polynomial()
    restored = FrobeniusPolynomial.from_json(frobenius.to_json())
    assert restored.polynomial == frobenius.polynomial
    zeta = curve.zeta_function()
    restored_zeta = ZetaFunction.from_json(zeta.to_json())
    assert len(repr(zeta)) < 160
    assert restored_zeta.numerator == zeta.numerator
    forged = json.loads(evidence.to_json())
    forged["binomial_terms"] = 0
    with pytest.raises(NoetherError):
        HyperellipticFrobeniusEvidence.from_json(json.dumps(forged))
    with pytest.raises(ValueError):
        curve.frobenius_evidence(1)


def test_malformed_and_characteristic_two() -> None:
    ring = PolynomialRing(FiniteField.prime(3), ["x"])
    (x,) = ring.generators()
    with pytest.raises(ValueError):
        HyperellipticCurve(x**3)
    with pytest.raises(ValueError):
        HyperellipticCurve(x)
    two = PolynomialRing(FiniteField.prime(2), ["x"])
    (t,) = two.generators()
    with pytest.raises(UnsupportedOperationError):
        HyperellipticCurve(t**3 + t + 1)
