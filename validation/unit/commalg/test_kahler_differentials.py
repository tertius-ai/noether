# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact Kähler differentials and cotangent evidence across the Python boundary."""

import json
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from noether import ContextMismatchError, Q, RationalField
from noether.commalg import (
    ConormalSequenceEvidence,
    CotangentComplexAmplitude,
    CotangentComplexHypothesis,
    CotangentComplexPresentation,
    CotangentSpace,
    KahlerDifferential,
    KahlerDifferentials,
    Polynomial,
    PolynomialRing,
)
from noether.numthy import FiniteField


def test_polynomial_line_has_free_rank_one_differentials() -> None:
    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    omega = ring.kahler_differentials()
    dx = omega.differentiate_polynomial(x)

    assert isinstance(omega, KahlerDifferentials)
    assert isinstance(dx, KahlerDifferential)
    assert omega.ambient_generator_count == 1
    assert omega.module.generator_count == 1
    assert omega.module.relation_count == 0
    assert not dx.is_zero()
    assert omega.differentiate_polynomial(x**3) == dx * omega.algebra(ring(3) * x**2)
    omega.verify()
    dx.verify()

    fiber = ring.cotangent_space_at([Fraction(7, 3)])
    assert isinstance(fiber, CotangentSpace)
    assert fiber.point == [Fraction(7, 3)]
    assert fiber.jacobian_rank == 0
    assert fiber.ambient_coordinate_count == 1
    assert fiber.embedding_dimension == 1
    assert fiber.presentation_matrix.shape == (1, 0)
    fiber.verify()


def test_dual_numbers_retain_conormal_and_relation_evidence() -> None:
    ring = PolynomialRing(Q, ["x"])
    x = ring.generator(0)
    algebra = ring.quotient(ring.ideal([x**2]))
    omega = algebra.kahler_differentials()

    assert omega.differentiate(algebra(x**2)).is_zero()
    assert omega.module.generator_count == 1
    assert omega.module.relation_count == 1
    evidence = omega.conormal_evidence
    assert isinstance(evidence, ConormalSequenceEvidence)
    assert evidence.chosen_generators == [x**2]
    assert evidence.jacobian.to_rows() == [[algebra(2 * x)]]
    assert evidence.conormal_module.generator_count == 1
    evidence.verify()

    fiber = algebra.cotangent_space_at([0])
    assert fiber.evaluated_jacobian.to_rows() == [[Fraction(0)]]
    assert fiber.presentation_matrix.to_rows() == [[Fraction(0)]]
    assert fiber.embedding_dimension == 1
    assert fiber.maximal_ideal_generators() == [x]


def test_cusp_has_different_embedding_dimensions_at_singular_and_smooth_points() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    relation = x**3 - y**2
    algebra = ring.quotient(ring.ideal([relation]))
    omega = algebra.kahler_differentials()

    singular = omega.cotangent_space_at([0, 0])
    smooth = omega.cotangent_space_at([1, 1])
    assert singular.evaluated_jacobian.to_rows() == [[Fraction(0), Fraction(0)]]
    assert smooth.evaluated_jacobian.to_rows() == [[Fraction(3), Fraction(-2)]]
    assert singular.embedding_dimension == 2
    assert smooth.embedding_dimension == 1
    assert singular.has_embedding_dimension(2)
    assert not singular.has_embedding_dimension(1)

    complex = omega.cotangent_complex()
    assert isinstance(complex, CotangentComplexPresentation)
    assert isinstance(complex.amplitude, CotangentComplexAmplitude)
    assert isinstance(complex.hypothesis, CotangentComplexHypothesis)
    assert (complex.amplitude.lowest_degree, complex.amplitude.highest_degree) == (-1, 0)
    assert complex.codimension == 1
    assert complex.hypothesis.ambient_dimension == 2
    assert complex.hypothesis.quotient_dimension == 1
    assert complex.hypothesis.codimension == 1
    assert complex.hypothesis.equation_count == 1
    complex.verify()


def test_characteristic_sensitive_derivative_and_redundant_equations() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x"])
    x = ring.generator(0)
    algebra = ring.quotient(ring.ideal([x**2]))
    omega = algebra.kahler_differentials()

    # In characteristic two, d(x²)=0 and the Jacobian relation vanishes.
    assert omega.differentiate(algebra(x**2)).is_zero()
    assert omega.conormal_evidence.jacobian.to_rows() == [[algebra.zero()]]
    assert algebra.cotangent_space_at([field.zero()]).embedding_dimension == 1

    redundant = algebra.conormal_sequence_from_generators([x**2, x**3])
    assert redundant.chosen_generators == [x**2, x**3]
    assert redundant.conormal_module.generator_count == 2
    redundant.verify()
    with pytest.raises(ValueError, match="equations but codimension"):
        algebra.cotangent_complex_from_generators([x**2, x**3])


def test_context_points_and_forged_json_fail_closed() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    algebra = ring.quotient(ring.ideal([y**2 - x**3]))
    omega = algebra.kahler_differentials()

    with pytest.raises(TypeError, match="2 coordinates"):
        algebra.cotangent_space_at([0])
    with pytest.raises(ValueError, match="does not annihilate"):
        algebra.cotangent_space_at([1, 0])
    other = PolynomialRing(Q, ["u"])
    with pytest.raises(ContextMismatchError, match="context mismatch"):
        omega.differentiate_polynomial(other.generator(0))

    restored = KahlerDifferentials.from_json(omega.to_json())
    assert restored == omega
    restored.verify()
    differential = omega.differentiate(algebra(x + y))
    assert KahlerDifferential.from_json(differential.to_json()) == differential
    fiber = algebra.cotangent_space_at([0, 0])
    assert CotangentSpace.from_json(fiber.to_json()) == fiber
    complex = algebra.cotangent_complex()
    assert CotangentComplexPresentation.from_json(complex.to_json()) == complex

    forged = json.loads(complex.to_json())
    forged["hypothesis"] = {
        "codimension": 99,
        "equation_count": 99,
    }
    with pytest.raises(ValueError):
        CotangentComplexPresentation.from_json(json.dumps(forged))
    with pytest.raises(TypeError):
        CotangentComplexAmplitude()
    with pytest.raises(TypeError):
        CotangentComplexHypothesis()


COEFFICIENTS = st.lists(st.integers(min_value=-4, max_value=4), min_size=6, max_size=6)


@settings(max_examples=36, deadline=None)
@given(left=COEFFICIENTS, right=COEFFICIENTS)
def test_universal_derivation_is_linear_and_leibniz(left: list[int], right: list[int]) -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    monomials = ([0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2])

    def polynomial(coefficients: list[int]) -> Polynomial[RationalField]:
        return ring.from_terms(zip(monomials, coefficients, strict=True))

    f = polynomial(left)
    g = polynomial(right)
    omega = ring.kahler_differentials()
    algebra = omega.algebra
    df = omega.differentiate_polynomial(f)
    dg = omega.differentiate_polynomial(g)

    assert omega.differentiate_polynomial(f + g) == df + dg
    assert omega.differentiate_polynomial(f * g) == df * algebra(g) + dg * algebra(f)
