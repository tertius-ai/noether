# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Polynomial ideals have one canonical Rust/Python identity."""

import json
from fractions import Fraction

import pytest

from noether import ContextMismatchError, Q, RationalField, UnsupportedOperationError, Z
from noether.commalg import (
    ArithmeticFiberSupport,
    ArithmeticNullstellensatzCertificate,
    AssociatedPrimeExtStage,
    AssociatedPrimeSet,
    EquidimensionalHull,
    EquidimensionalRadicalComponent,
    EquidimensionalRadicalDecomposition,
    FiniteQuotientAlgebra,
    Ideal,
    IdealDecomposition,
    IdealElimination,
    IdealReduction,
    MinimalPrimeDecomposition,
    Polynomial,
    PolynomialPrimaryCertificate,
    PolynomialPrimeCertificate,
    PolynomialRing,
    PrimaryDecomposition,
    RadicalIdeal,
    RationalFunctionField,
    RationalUnivariateRepresentation,
    RealSolutionBox,
    WeightedGrading,
)
from noether.homalg import FreeResolution, PresentedModule
from noether.numthy import FiniteField


def test_rational_ideal_owns_its_reduced_groebner_basis() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order="lex")
    x, y = ring.generators()

    ideal = ring.ideal([x + y, x - y])

    assert isinstance(ideal, Ideal)
    assert ideal.ring == ring
    assert ideal.generators == [x, y]
    assert ideal.groebner_basis() == [x, y]
    assert x in ideal
    assert y in ideal
    assert ideal.normal_form(x + 1) == ring.one()
    assert ideal.normal_forms([x + 1, y]) == [ring.one(), ring.zero()]
    ideal.verify()


def test_zero_unit_and_extension_field_ideals_share_the_surface() -> None:
    field = FiniteField.extension(3, 2)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    zero = ring.ideal([])
    unit = ring.ideal([ring.constant(field(2))])
    first = ring.ideal([x * y - 1, y**2 - x])
    reordered = ring.ideal([y**2 - x, field(2) * (x * y - 1)])

    assert zero.generators == []
    assert x not in zero
    assert unit.generators == [ring.one()]
    assert ring.one() in unit
    assert first == reordered
    assert x * y - 1 in first
    first.verify()


def test_change_order_projects_generic_fglm_and_the_direct_floor() -> None:
    field = FiniteField.extension(3, 2)
    ring = PolynomialRing(field, ["x", "y"], order="grevlex")
    x, y = ring.generators()
    zero_dimensional = ring.ideal([x**2 - y, y**2 - x])

    lex = zero_dimensional.change_order("lex")
    assert lex.ring.term_order.name == "lex"
    assert lex.change_order("grevlex") == zero_dimensional
    lex.verify()

    positive_dimensional = ring.ideal([x * y]).change_order("lex")
    assert positive_dimensional.ring.term_order.name == "lex"
    assert positive_dimensional.dimension() == 1
    positive_dimensional.verify()


def test_rational_univariate_representation_is_exact_and_typed() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    ideal = ring.ideal([x**2 - y, y**2 - x])

    representation = ideal.rational_univariate_representation()
    assert isinstance(representation, RationalUnivariateRepresentation)
    assert representation.source == ideal
    assert representation.parameter_ring.variables == ["t"]
    assert representation.quotient_dimension == 4
    assert len(representation.coordinate_numerators) == 2
    assert representation.denominator == representation.parameter_ring.one()
    assert len(representation.linear_form) == 2
    assert representation.is_reduced
    representation.verify()
    assert RationalUnivariateRepresentation.from_json(representation.to_json()) == representation

    finite = FiniteField.prime(5)
    finite_ring = PolynomialRing(finite, ["u"])
    u = finite_ring.generator(0)
    nonreduced = finite_ring.ideal([u**3]).rational_univariate_representation()
    assert nonreduced.quotient_dimension == 3
    assert not nonreduced.is_reduced
    nonreduced.verify()


def test_finite_quotient_algebra_represents_nonmonogenic_algebras() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    ideal = ring.ideal([x**2, x * y, y**2])

    algebra = ideal.finite_quotient_algebra()
    assert isinstance(algebra, FiniteQuotientAlgebra)
    assert algebra.source == ideal
    assert algebra.dimension == 3
    assert len(algebra.basis) == 3
    assert not algebra.is_zero
    assert len(algebra.multiplication_operator_nonzeros) == 2
    assert len(algebra.multiplication_operator(0)) == algebra.multiplication_operator_nonzeros[0]

    one = algebra.one_coordinates()
    x_coordinates = algebra.normal_form_coordinates(x)
    y_coordinates = algebra.normal_form_coordinates(y)
    zero = [Fraction(0), Fraction(0), Fraction(0)]
    assert algebra.multiply(one, x_coordinates) == x_coordinates
    assert algebra.multiply(y_coordinates, one) == y_coordinates
    assert algebra.multiply(x_coordinates, x_coordinates) == zero
    assert algebra.multiply(x_coordinates, y_coordinates) == zero
    assert algebra.multiply_by_variable(0, one) == x_coordinates
    with pytest.raises(ValueError):
        algebra.multiply(one[:2], one)
    with pytest.raises(ValueError):
        algebra.multiplication_operator(2)

    algebra.verify()
    encoded = algebra.to_json()
    assert FiniteQuotientAlgebra.from_json(encoded) == algebra
    assert hash(FiniteQuotientAlgebra.from_json(encoded)) == hash(algebra)
    with pytest.raises(ValueError):
        ideal.rational_univariate_representation()

    zero_algebra = ring.ideal([ring.one()]).finite_quotient_algebra()
    assert zero_algebra.is_zero
    assert zero_algebra.dimension == 0


def test_rational_rur_projects_certified_real_solution_boxes() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    representation = ring.ideal([x**2 - 1, y - x]).rational_univariate_representation()

    solutions = representation.real_solutions(128)
    assert len(solutions) == 2
    assert all(isinstance(solution, RealSolutionBox) for solution in solutions)
    for solution in solutions:
        assert solution.representation == representation
        assert solution.precision == 128
        assert solution.multiplicity == 1
        assert solution.parameter_ball.field.precision == 128
        assert len(solution.coordinates) == 2
        solution.verify()
        assert RealSolutionBox.from_json(solution.to_json()) == solution


def test_polynomial_ideal_context_and_json_fail_closed() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order="lex")
    x, y = ring.generators()
    ideal = ring.ideal([x + y, x - y])

    assert Ideal.type_id == "noether/commalg/Ideal@1"
    assert Ideal.from_json(ideal.to_json()) == ideal
    assert hash(Ideal.from_json(ideal.to_json())) == hash(ideal)

    payload = json.loads(ideal.to_json())
    payload["generators"].reverse()
    with pytest.raises(ValueError, match="canonical reduced Groebner basis"):
        Ideal.from_json(json.dumps(payload))

    other = PolynomialRing(Q, ["x"])
    with pytest.raises((ContextMismatchError, ValueError), match="context mismatch"):
        ring.ideal([other.generator(0)])
    with pytest.raises((ContextMismatchError, ValueError), match="context mismatch"):
        ideal.normal_form(other.generator(0))


def test_twisted_cubic_has_exact_hilbert_invariants() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    generators = [x * z - y**2, x * w - y * z, y * w - z**2]
    ideal = ring.ideal(generators)

    assert ideal.hilbert_numerator() == [1, 0, -3, 2]
    target = ideal.hilbert_series()
    assert target.standard_numerator() == ideal.hilbert_numerator()
    assert ring.ideal_with_hilbert_series(generators, target) == ideal
    assert ideal.dimension() == 2
    assert ideal.degree() == 3

    hilbert_ring = PolynomialRing(Q, ["m"])
    m = hilbert_ring.generator(0)
    assert ideal.hilbert_polynomial() == 3 * m + 1


def test_affine_and_boundary_hilbert_invariants_are_hand_checked() -> None:
    field = FiniteField.prime(3)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    complete_intersection = ring.ideal([x**2, y**3])

    assert complete_intersection.dimension() == 0
    assert complete_intersection.degree() == 6
    assert complete_intersection.hilbert_polynomial().is_zero()
    assert ring.ideal([]).dimension() == 2
    assert ring.ideal([]).degree() == 1
    assert ring.ideal([ring.one()]).dimension() == -1
    assert ring.ideal([ring.one()]).degree() == 0


def test_standard_hilbert_invariants_fail_closed_on_unsupported_contexts() -> None:
    weighted = PolynomialRing(
        Q,
        ["x", "y"],
        grading=WeightedGrading([1, 2]),
    )
    ideal = weighted.ideal([weighted.generator("x")])
    assert ideal.dimension() == 1
    with pytest.raises(ValueError, match="standard grading"):
        ideal.degree()

    lex = PolynomialRing(Q, ["x"], order="lex")
    x = lex.generator(0)
    nonhomogeneous = lex.ideal([x**2 - 1])
    assert nonhomogeneous.dimension() == 0
    with pytest.raises(ValueError, match="degree-compatible"):
        nonhomogeneous.hilbert_polynomial()


def test_elimination_accepts_names_and_indices_and_preserves_the_ring() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    ideal = ring.ideal([x - y**2, z - y])

    eliminated = ideal.eliminate(["y", 1])
    assert eliminated == ring.ideal([x - z**2])
    assert eliminated.ring == ring
    assert ideal.eliminate([]) == ideal

    with pytest.raises(ValueError, match="outside arity"):
        ideal.eliminate([3])
    with pytest.raises(TypeError, match="not one string"):
        ideal.eliminate("y")


def test_field_membership_and_elimination_return_replayable_evidence() -> None:
    ring = PolynomialRing(Q, ["x", "y"], order="lex")
    x, y = ring.generators()
    ideal = ring.ideal([x - y**2, y - 1])

    reduction = ideal.reduce(x - 1)
    assert isinstance(reduction, IdealReduction)
    assert reduction.ideal == ideal
    assert reduction.value == x - 1
    assert reduction.is_member
    assert reduction.remainder.is_zero()
    reduction.verify()
    assert IdealReduction.from_json(reduction.to_json()) == reduction

    elimination = ideal.eliminate_with_evidence(["y"])
    assert isinstance(elimination, IdealElimination)
    assert elimination.source == ideal
    assert elimination.variables == [1]
    assert elimination.result == ring.ideal([x - 1])
    elimination.verify()
    assert IdealElimination.from_json(elimination.to_json()) == elimination


def test_integral_ideals_expose_contraction_certificates_and_support_strength() -> None:
    ring = PolynomialRing(Z, ["x"], order="lex")
    x = ring.generator(0)
    raw_generators = [2 * x, x + 1]
    fast_certificate = ring.nullstellensatz_certificate(raw_generators)
    assert fast_certificate.ring == ring
    assert fast_certificate.generators == raw_generators
    assert fast_certificate.integer == 2
    fast_certificate.verify()

    ideal = ring.ideal(raw_generators)

    assert ideal.contract_to_coefficients().principal_generator() == 2
    assert ideal.base_change(Z.rational_embedding()).generators == [
        ring.base_change(Z.rational_embedding()).one()
    ]
    reduction = ideal.reduce(ring.constant(2))
    assert reduction.is_member
    reduction.verify()

    certificate = ideal.nullstellensatz_certificate()
    assert isinstance(certificate, ArithmeticNullstellensatzCertificate)
    assert certificate.integer == 2
    assert certificate.principal_ideal == Z.ideal([2])
    assert certificate.is_exact_contraction()
    certificate.verify()
    assert ArithmeticNullstellensatzCertificate.from_json(certificate.to_json()) == certificate

    support = ideal.arithmetic_fiber_support(2, 64)
    assert isinstance(support, ArithmeticFiberSupport)
    assert support.claim == "exact_contraction"
    assert support.is_exact
    assert support.proved_primes == [(2, 1)]
    assert support.prime_support_is_complete
    support.verify()
    assert ArithmeticFiberSupport.from_json(support.to_json()) == support

    inflated = ArithmeticNullstellensatzCertificate.from_identity(
        ideal,
        6,
        [3 * multiplier for multiplier in certificate.multipliers],
    )
    containing = inflated.arithmetic_fiber_support(2, 64)
    assert containing.claim == "certified_containment"
    assert not containing.is_exact
    assert containing.proved_primes == [(2, 1), (3, 1)]
    assert Z.ideal([6]).sum(Z.ideal([10])).principal_generator() == 2


def test_intersection_colon_and_saturation_are_exact_and_context_checked() -> None:
    field = FiniteField.prime(3)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    x_ideal = ring.ideal([x])
    y_ideal = ring.ideal([y])

    assert x_ideal.intersection(y_ideal) == ring.ideal([x * y])
    numerator = ring.ideal([x**2, x * y])
    assert numerator.colon(x_ideal) == ring.ideal([x, y])
    assert numerator.colon(ring.ideal([x, y])) == x_ideal
    assert numerator.saturate(y_ideal) == x_ideal
    assert numerator.colon(ring.ideal([])) == ring.ideal([ring.one()])
    assert numerator.saturate(ring.ideal([ring.one()])) == numerator

    other = PolynomialRing(field, ["u", "v"]).ideal([])
    with pytest.raises((ContextMismatchError, ValueError), match="context"):
        numerator.intersection(other)


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(5)])
def test_polynomial_ideal_arithmetic_is_canonical_and_exact(coefficient_ring) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    left = ring.ideal([x])
    right = ring.ideal([y])

    assert left.sum(right) == ring.ideal([x, y])
    assert left.product(right) == ring.ideal([x * y])
    assert left + right == ring.ideal([x, y])
    assert left * right == ring.ideal([x * y])
    assert left.pow(0) == ring.ideal([ring.one()])
    assert left.pow(1) == left
    assert left.pow(3) == ring.ideal([x**3])
    assert left**3 == ring.ideal([x**3])
    assert left.product(right) == right.product(left)
    assert left.product(left.sum(right)) == left.pow(2).sum(left.product(right))

    other = PolynomialRing(coefficient_ring, ["u", "v"]).ideal([])
    with pytest.raises((ContextMismatchError, ValueError), match="context"):
        left.sum(other)
    with pytest.raises((ContextMismatchError, ValueError), match="context"):
        left.product(other)
    with pytest.raises(OverflowError):
        left.pow(-1)


def test_integral_polynomial_ideal_arithmetic_uses_strong_bases() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator(0)
    two_x = ring.ideal([2 * x])
    three_x = ring.ideal([3 * x])

    assert two_x.sum(three_x) == ring.ideal([x])
    assert two_x.product(three_x) == ring.ideal([6 * x**2])
    assert two_x.pow(0) == ring.ideal([ring.one()])
    assert two_x.pow(2) == ring.ideal([4 * x**2])
    two_x.pow(2).verify()


def test_finite_rational_function_ideals_share_the_exact_field_surface() -> None:
    base = FiniteField.prime(3)
    functions = RationalFunctionField(base, "c")
    c = functions.generator()
    ring = PolynomialRing(functions, ["x", "y"], order="lex")
    x, y = ring.generators()
    ideal = ring.ideal([x - c * y, y**2 - 1])

    assert c + x == x + c
    assert c - x == -(x - c)
    assert ideal.dimension() == 0
    assert ideal.eliminate(["y"]) == ring.ideal([x**2 - c**2])
    assert ideal.normal_form(x**2) == ring.constant(c**2)
    assert ideal.ring.coefficient_ring == functions
    assert Ideal.from_json(ideal.to_json()) == ideal
    assert PolynomialRing.from_json(ring.to_json()) == ring
    assert ring.constant(c).constant_coefficient() == c
    ideal.verify()

    algebra = ideal.finite_quotient_algebra()
    assert algebra.dimension == 2
    one = algebra.one_coordinates()
    assert algebra.multiply_by_variable(0, one) == algebra.normal_form_coordinates(x)
    algebra.verify()
    assert FiniteQuotientAlgebra.from_json(algebra.to_json()) == algebra


def test_qt_polynomials_ideals_and_rurs_share_the_exact_field_surface() -> None:
    functions = RationalFunctionField(Q, "t")
    t = functions.generator()
    ring = PolynomialRing(functions, ["x", "y"])
    x, y = ring.generators()
    coefficient = t**2 / (t + 1)
    polynomial = (t + 1) * x + coefficient * y + functions(Fraction(1, 2))

    assert polynomial.coefficient([1, 0]) == t + 1
    assert polynomial.coefficient([0, 1]) == coefficient
    assert polynomial.evaluate([t, 1]) == (t + 1) * t + coefficient + Fraction(1, 2)
    assert PolynomialRing.from_json(ring.to_json()) == ring
    assert Polynomial.from_json(polynomial.to_json()) == polynomial

    ideal = ring.ideal([x - t, y - x])
    assert ideal.dimension() == 0
    assert ideal.degree() == 1
    assert ideal.normal_form(y - t).is_zero()
    assert Ideal.from_json(ideal.to_json()) == ideal
    ideal.verify()

    algebra = ideal.finite_quotient_algebra()
    assert algebra.dimension == 1
    assert algebra.multiply(algebra.one_coordinates(), algebra.one_coordinates()) == [functions(1)]
    algebra.verify()
    assert FiniteQuotientAlgebra.from_json(algebra.to_json()) == algebra

    representation = ideal.rational_univariate_representation()
    assert representation.source == ideal
    assert representation.parameter_ring.coefficient_ring == functions
    assert representation.quotient_dimension == 1
    assert representation.is_reduced
    representation.verify()
    assert RationalUnivariateRepresentation.from_json(representation.to_json()) == representation


def test_generic_lines_split_with_a_verified_frobenius_permutation() -> None:
    base = FiniteField.prime(3)
    functions = RationalFunctionField(base, "c")
    ring = PolynomialRing(functions, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x**2 + y**2])

    with pytest.raises(UnsupportedOperationError) as raised:
        source.minimal_primes()
    assert "split completely" in raised.value.detail

    changed = source.base_change_constant_field(FiniteField.extension(3, 2))
    decomposition = changed.minimal_primes()

    assert isinstance(decomposition, IdealDecomposition)
    assert decomposition.source == changed
    assert len(decomposition.components) == 2
    assert decomposition.frobenius_permutation == [1, 0]
    hilbert_ring = PolynomialRing(Q, ["m"])
    assert decomposition.hilbert_polynomials() == [
        hilbert_ring.one(),
        hilbert_ring.one(),
    ]
    decomposition.verify()
    decomposition.verify_frobenius()
    assert IdealDecomposition.from_json(decomposition.to_json()) == decomposition

    payload = json.loads(decomposition.to_json())
    payload["frobenius_permutation"] = [0, 1]
    with pytest.raises(ValueError, match="Frobenius image"):
        IdealDecomposition.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(2), FiniteField.extension(3, 2)])
def test_zero_dimensional_radical_retains_coordinate_certificates(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x**2, y**3])

    radical = source.radical()
    assert isinstance(radical, RadicalIdeal)
    assert radical.source == source
    assert radical.ideal == ring.ideal([x, y])
    assert len(radical.characteristic_polynomials) == 2
    assert len(radical.coordinate_certificates) == 2
    assert radical.theorem == "Seidenberg coordinate-polynomial lemma over a perfect field"
    radical.verify()
    assert RadicalIdeal.from_json(radical.to_json()) == radical

    payload = json.loads(radical.to_json())
    payload["proof"]["coordinate_certificates"] = []
    with pytest.raises(ValueError, match="recomputation changed"):
        RadicalIdeal.from_json(json.dumps(payload))

    larger = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = larger.generators()
    reduced_complete_intersection = larger.ideal([x**2 - y, x * z]).radical()
    assert reduced_complete_intersection.proof_kind == "generator_factor_split"
    assert reduced_complete_intersection.ideal == reduced_complete_intersection.source

    unsupported_ring = PolynomialRing(coefficient_ring, ["x", "y", "z", "w"])
    x, y, z, w = unsupported_ring.generators()
    twisted_cubic = unsupported_ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2])
    with pytest.raises(UnsupportedOperationError, match="certified split"):
        twisted_cubic.radical()


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_positive_dimensional_principal_and_monomial_radicals_are_verified(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    first, second = x + y, x - z
    principal_source = ring.ideal([first**3 * second**2])

    principal = principal_source.radical()
    assert principal.proof_kind == "principal_factorization"
    assert sorted(multiplicity for _, multiplicity in principal.principal_factors or []) == [2, 3]
    assert principal.monomial_proof is None
    assert principal.ideal == ring.ideal([first * second])
    principal.verify()
    assert RadicalIdeal.from_json(principal.to_json()) == principal

    monomial_source = ring.ideal([x**3 * y, y**2 * z**4])
    monomial = monomial_source.radical()
    assert monomial.proof_kind == "monomial_support"
    assert monomial.principal_factors is None
    source_proof, radical_proof = monomial.monomial_proof or (None, None)
    assert source_proof is not None and radical_proof is not None
    assert radical_proof == source_proof.radical()
    assert monomial.ideal == radical_proof.to_ideal()
    monomial.verify()


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_ehv_associated_primes_retain_ext_annihilator_evidence(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    linear = x + y
    source = ring.ideal([linear**2, linear * z])

    result = source.associated_primes()

    assert isinstance(result, AssociatedPrimeSet)
    assert result.source == source
    assert isinstance(result.resolution, FreeResolution)
    assert [stage.codimension for stage in result.stages] == [1, 2, 3]
    assert all(isinstance(stage, AssociatedPrimeExtStage) for stage in result.stages)
    assert all(isinstance(stage.ext_module, PresentedModule) for stage in result.stages)
    assert result.stages[0].annihilator == ring.ideal([linear])
    assert result.stages[1].annihilator == ring.ideal([linear, z])
    assert result.stages[2].annihilator == ring.ideal([ring.one()])
    assert [len(stage.associated_primes) for stage in result.stages] == [1, 1, 0]
    assert all(stage.minimal_primes is not None for stage in result.stages[:2])
    assert result.stages[2].minimal_primes is None
    assert len(result.prime_certificates) == 2
    assert ring.ideal([linear]) in result.primes
    assert ring.ideal([linear, z]) in result.primes
    assert "ann Ext^i" in result.theorem
    result.verify()
    assert AssociatedPrimeSet.from_json(result.to_json()) == result
    assert AssociatedPrimeExtStage.from_json(result.stages[0].to_json()) == result.stages[0]

    payload = json.loads(result.to_json())
    payload["stages"][0]["annihilator"] = payload["source"]
    with pytest.raises(ValueError, match="annihilator"):
        AssociatedPrimeSet.from_json(json.dumps(payload))

    decomposition = source.primary_decomposition()
    minimal_component = ring.ideal([linear])
    embedded_component = ring.ideal([linear**2, linear * z, z**2])
    assert len(decomposition.components) == 2
    assert minimal_component in decomposition.components
    assert embedded_component in decomposition.components
    assert any(
        certificate.ideal == minimal_component
        and certificate.proof_kind == "ehv_single_associated_prime"
        and certificate.ehv_associated_primes is not None
        and certificate.ehv_associated_primes.source == certificate.ideal
        for certificate in decomposition.certificates
    )
    assert any(
        certificate.ideal == embedded_component
        and certificate.proof_kind == "ehv_equidimensional_hull"
        and certificate.ehv_equidimensional_hull is not None
        and certificate.ehv_equidimensional_hull.ideal == certificate.ideal
        for certificate in decomposition.certificates
    )
    assert minimal_component.intersection(embedded_component) == source
    decomposition.verify()
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition

    singleton = next(
        certificate
        for certificate in decomposition.certificates
        if certificate.ehv_associated_primes is not None
    )
    certificate_payload = json.loads(singleton.to_json())
    certificate_payload["proof"]["associated_primes"]["source"] = json.loads(source.to_json())
    with pytest.raises(ValueError):
        PolynomialPrimaryCertificate.from_json(json.dumps(certificate_payload))

    hull_certificate = next(
        certificate
        for certificate in decomposition.certificates
        if certificate.ehv_equidimensional_hull is not None
    )
    certificate_payload = json.loads(hull_certificate.to_json())
    certificate_payload["proof"]["hull"]["ideal"] = json.loads(source.to_json())
    with pytest.raises(ValueError):
        PolynomialPrimaryCertificate.from_json(json.dumps(certificate_payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_equidimensional_hull_retains_nonradical_top_component(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["a", "b", "c"])
    a, b, c = ring.generators()
    top = ring.ideal([a**2, b**2])
    embedded = ring.ideal([a**3, b**3, c**3])
    source = top.intersection(embedded)

    hull = source.equidimensional_hull()

    assert isinstance(hull, EquidimensionalHull)
    assert hull.source == source
    assert hull.codimension == 2
    assert hull.ideal == top
    assert hull.associated_primes.source == source
    assert hull.minimal_primes is not None
    assert hull.minimal_primes.source == top
    assert len(hull.prime_certificates) == 1
    assert hull.prime_certificates[0].ideal == ring.ideal([a, b])
    assert "ann Ext^c" in hull.theorem
    hull.verify()
    assert EquidimensionalHull.from_json(hull.to_json()) == hull

    payload = json.loads(hull.to_json())
    payload["ideal"] = payload["source"]
    with pytest.raises(ValueError):
        EquidimensionalHull.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_associated_prime_localization_separates_incomparable_embedded_components(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    linear = x + y
    minimal = ring.ideal([linear])
    first = ring.ideal([linear**2, linear * y, y**2])
    second = ring.ideal([linear**2, linear * z, z**2])
    source = minimal.intersection(first).intersection(second)

    decomposition = source.primary_decomposition()

    assert len(decomposition.components) == 3
    assert minimal in decomposition.components
    assert first in decomposition.components
    assert second in decomposition.components
    assert (
        sum(
            certificate.proof_kind == "ehv_equidimensional_hull"
            for certificate in decomposition.certificates
        )
        == 2
    )
    assert {certificate.ideal for certificate in decomposition.associated_primes} == {
        ring.ideal([linear]),
        ring.ideal([linear, y]),
        ring.ideal([linear, z]),
    }
    decomposition.verify()
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition


def test_common_factor_radical_and_minimal_primes_are_projected() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    common = z - 1
    source = ring.ideal([common * x, common * y])

    radical = source.radical()

    assert radical.proof_kind == "common_factor_split"
    assert radical.ideal == source
    factors = radical.common_factors or []
    residual_source, residual_radical = radical.common_factor_residual or (None, None)
    assert factors == [(common.monic(), 1)]
    assert residual_source == ring.ideal([x, y])
    assert residual_radical == residual_source
    radical.verify()
    assert RadicalIdeal.from_json(radical.to_json()) == radical

    decomposition = source.minimal_primes()
    decomposition.verify()
    assert len(decomposition.components) == 2
    assert sorted(component.dimension() for component in decomposition.components) == [1, 2]
    assert decomposition.radical == radical

    primary = source.primary_decomposition()
    primary.verify()
    assert len(primary.components) == len(decomposition.components)
    assert all(component in decomposition.components for component in primary.components)
    assert all(
        certificate.proof_kind == "prime"
        and certificate.ideal == certificate.associated_prime.ideal
        for certificate in primary.certificates
    )


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_common_factor_power_primary_decomposition_preserves_embedded_residual_primes(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    common = w - 1
    common_power = common**2
    source = ring.ideal([common_power * x**3 * y, common_power * x * z**4])

    decomposition = source.primary_decomposition()

    decomposition.verify()
    assert len(decomposition.components) == 4
    assert [certificate.proof_kind for certificate in decomposition.certificates].count(
        "principal_power"
    ) == 1
    assert [certificate.proof_kind for certificate in decomposition.certificates].count(
        "monomial_primary"
    ) == 3
    assert ring.ideal([common_power]) in decomposition.components
    coordinate_radicals = [
        prime.coordinate_variables
        for prime in decomposition.associated_primes
        if prime.coordinate_variables is not None
    ]
    assert [0] in coordinate_radicals
    assert [0, 2] in coordinate_radicals
    assert [1, 2] in coordinate_radicals
    assert sorted(component.dimension() for component in decomposition.components) == [2, 2, 3, 3]
    reconstructed = decomposition.components[0]
    for component in decomposition.components[1:]:
        reconstructed = reconstructed.intersection(component)
    assert reconstructed == source
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition


def test_generator_factor_split_and_affine_prime_evidence_are_projected() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    left = x + 1
    right = y + 1
    splitting_generator = left**2 * right**3
    source = ring.ideal([z, splitting_generator])

    radical = source.radical()

    assert radical.proof_kind == "generator_factor_split"
    assert radical.factor_split_generator == splitting_generator
    factors = radical.factor_split_factors or []
    assert len(factors) == 2
    assert (left, 2) in factors
    assert (right, 3) in factors
    branches = radical.factor_split_branches or []
    assert len(branches) == 2
    assert all(branch == branch_radical for branch, branch_radical in branches)
    assert radical.ideal == ring.ideal([z, left * right])
    radical.verify()

    decomposition = source.minimal_primes()
    decomposition.verify()
    assert len(decomposition.components) == 2
    assert all(
        certificate.proof_kind == "affine_linear"
        and len(certificate.affine_linear_pivots or []) == 2
        for certificate in decomposition.prime_certificates
    )
    assert sorted(component.dimension() for component in decomposition.components) == [1, 1]

    primary = source.primary_decomposition()
    primary.verify()
    assert len(primary.components) == 2
    assert all(
        certificate.proof_kind == "polynomial_graph_hypersurface_power"
        and certificate.polynomial_graph_base == ring.ideal([z])
        and certificate.polynomial_graph_pivots == [2]
        and certificate.polynomial_graph_factor in [left, right]
        and certificate.polynomial_graph_factor_exponent in [2, 3]
        for certificate in primary.certificates
    )
    assert sorted(
        certificate.polynomial_graph_factor_exponent
        for certificate in primary.certificates
        if certificate.polynomial_graph_factor_exponent is not None
    ) == [2, 3]
    assert PrimaryDecomposition.from_json(primary.to_json()) == primary

    payload = json.loads(primary.certificates[0].to_json())
    payload["proof"]["factors"][0][1] = 1
    with pytest.raises(ValueError, match="regular-sequence power"):
        PolynomialPrimaryCertificate.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_polynomial_graph_complete_intersection_primary_evidence_is_projected(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["z", "x", "y", "u", "v", "t"], order="lex")
    z, x, y, u, v, t = ring.generators()
    graph = z - t**2
    first_left, first_right = x + 1, y + 1
    second_left, second_right = u + 1, v + 1
    source = ring.ideal(
        [
            graph,
            first_left**2 * first_right**3,
            second_left**4 * second_right**5,
        ]
    )

    decomposition = source.primary_decomposition()

    decomposition.verify()
    assert len(decomposition.components) == 4
    exponent_pairs: list[tuple[int, int]] = []
    for certificate in decomposition.certificates:
        assert certificate.proof_kind == "polynomial_graph_complete_intersection_powers"
        assert certificate.polynomial_graph_base == ring.ideal([graph])
        assert certificate.polynomial_graph_pivots == [0]
        assert certificate.polynomial_graph_factor is None
        assert certificate.polynomial_graph_factor_exponent is None
        assert certificate.associated_prime.proof_kind == "polynomial_graph"
        factors = certificate.polynomial_graph_factors or []
        assert len(factors) == 2
        first_exponent = next(
            exponent for factor, exponent in factors if factor in [first_left, first_right]
        )
        second_exponent = next(
            exponent for factor, exponent in factors if factor in [second_left, second_right]
        )
        exponent_pairs.append((first_exponent, second_exponent))
    assert sorted(exponent_pairs) == [(2, 4), (2, 5), (3, 4), (3, 5)]
    reconstructed = decomposition.components[0]
    for component in decomposition.components[1:]:
        reconstructed = reconstructed.intersection(component)
    assert reconstructed == source
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition

    payload = json.loads(decomposition.certificates[0].to_json())
    payload["proof"]["factors"][0][1] = 0
    with pytest.raises(ValueError, match="invalid factor data"):
        PolynomialPrimaryCertificate.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_polynomial_graph_lifts_monomial_embedded_primary_components(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["y", "z", "x", "u", "v"], order="lex")
    y, z, x, u, v = ring.generators()
    first_graph = y - x**2
    second_graph = z - u**2 - v
    base = ring.ideal([first_graph, second_graph])
    source = ring.ideal([first_graph, second_graph, x**3 * u, x * v**4])

    decomposition = source.primary_decomposition()

    decomposition.verify()
    assert len(decomposition.components) == 3
    assert all(
        certificate.proof_kind == "polynomial_graph_monomial_primary"
        and certificate.polynomial_graph_base == base
        and certificate.polynomial_graph_pivots == [0, 1]
        and certificate.polynomial_graph_monomial_source is not None
        and certificate.polynomial_graph_factors is None
        and certificate.associated_prime.proof_kind == "polynomial_graph"
        for certificate in decomposition.certificates
    )
    pivot_sets = [
        certificate.associated_prime.polynomial_graph_pivots
        for certificate in decomposition.certificates
    ]
    assert [0, 1, 2] in pivot_sets
    assert [0, 1, 2, 4] in pivot_sets
    assert [0, 1, 3, 4] in pivot_sets
    assert sorted(component.dimension() for component in decomposition.components) == [1, 1, 2]
    reconstructed = decomposition.components[0]
    for component in decomposition.components[1:]:
        reconstructed = reconstructed.intersection(component)
    assert reconstructed == source
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition

    payload = json.loads(decomposition.certificates[0].to_json())
    payload["proof"]["pivot_variables"][0] = 2
    with pytest.raises(ValueError, match="invalid base or residual data"):
        PolynomialPrimaryCertificate.from_json(json.dumps(payload))


def test_saturation_split_evidence_is_projected_and_replayed() -> None:
    field = FiniteField.prime(101)
    ring = PolynomialRing(field, ["x", "y", "z", "w"])

    def polynomial(terms: dict[tuple[int, int, int, int], int]) -> Polynomial[FiniteField]:
        return ring.from_terms(
            (exponent, field(coefficient)) for exponent, coefficient in terms.items()
        )

    source = ring.ideal(
        [
            polynomial(
                {
                    (4, 0, 0, 0): 1,
                    (0, 0, 4, 0): -19,
                    (0, 1, 2, 1): 13,
                    (2, 0, 0, 2): -25,
                    (0, 0, 2, 2): -26,
                    (0, 1, 0, 3): 16,
                    (0, 0, 0, 4): 40,
                }
            ),
            polynomial(
                {
                    (2, 0, 2, 0): 1,
                    (0, 0, 4, 0): 25,
                    (2, 0, 1, 1): -32,
                    (0, 1, 2, 1): 26,
                    (0, 0, 3, 1): 8,
                    (2, 0, 0, 2): 4,
                    (0, 1, 1, 2): -24,
                    (0, 0, 2, 2): -50,
                    (0, 1, 0, 3): 3,
                    (0, 0, 1, 3): -48,
                    (0, 0, 0, 4): 6,
                }
            ),
            polynomial(
                {
                    (2, 1, 0, 0): 1,
                    (2, 0, 1, 0): 1,
                    (0, 1, 2, 0): 25,
                    (0, 0, 3, 0): 25,
                    (2, 0, 0, 1): -32,
                    (0, 1, 1, 1): 26,
                    (0, 0, 2, 1): -49,
                    (0, 1, 0, 2): 46,
                    (0, 0, 1, 2): -49,
                    (0, 0, 0, 3): 14,
                }
            ),
            polynomial(
                {
                    (0, 2, 0, 0): 1,
                    (0, 0, 2, 0): -25,
                    (0, 1, 0, 1): -24,
                    (0, 0, 0, 2): -49,
                }
            ),
        ]
    )

    radical = source.radical()

    assert radical.proof_kind == "saturation_split"
    assert radical.saturation_split_witness is not None
    branches = radical.saturation_split_branches
    assert branches is not None
    (closed_source, closed_radical), (open_source, open_radical) = branches
    assert closed_source != source
    assert open_source != source
    assert closed_radical.intersection(open_radical) == source
    assert radical.ideal == source
    radical.verify()
    assert RadicalIdeal.from_json(radical.to_json()) == radical

    decomposition = source.minimal_primes()
    decomposition.verify()
    assert len(decomposition.components) == 2
    assert sorted(component.dimension() for component in decomposition.components) == [1, 2]
    assert sorted(certificate.proof_kind for certificate in decomposition.prime_certificates) == [
        "homogeneous_affine_patch",
        "smooth_projective_complete_intersection",
    ]
    assert decomposition.components[0].intersection(decomposition.components[1]) == source

    equidimensional = source.equidimensional_radical_decomposition()
    equidimensional.verify()
    assert len(equidimensional.components) == 2
    assert sorted(component.dimension for component in equidimensional.components) == [1, 2]

    primary = source.primary_decomposition()
    primary.verify()
    assert len(primary.components) == len(decomposition.components)
    assert all(component in decomposition.components for component in primary.components)
    assert all(certificate.proof_kind == "prime" for certificate in primary.certificates)

    payload = json.loads(radical.to_json())
    payload["proof"]["closed_radical"] = json.loads(ring.ideal([ring.one()]).to_json())
    with pytest.raises(ValueError, match="recomputation changed"):
        RadicalIdeal.from_json(json.dumps(payload))


def test_polynomial_graph_prime_evidence_is_projected() -> None:
    ring = PolynomialRing(Q, ["y", "z", "x"], order="lex")
    y, z, x = ring.generators()
    source = ring.ideal([y - x**2, z - x**3])

    radical = source.radical()
    assert radical.proof_kind == "polynomial_graph"
    assert radical.polynomial_graph_pivots == [0, 1]
    assert radical.ideal == source

    decomposition = source.minimal_primes()
    assert decomposition.components == [source]
    certificate = decomposition.prime_certificates[0]
    assert certificate.proof_kind == "polynomial_graph"
    assert certificate.polynomial_graph_pivots == [0, 1]
    certificate.verify()


def test_smooth_projective_complete_intersection_prime_evidence_is_projected() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z", "w"])
    x, y, z, w = ring.generators()
    first = x**2 + y**2 + z**2 + w**2
    second = x**2 + 2 * y**2 + 3 * z**2 + 4 * w**2
    source = ring.ideal([first, second])

    decomposition = source.minimal_primes()

    assert decomposition.components == [source]
    certificate = decomposition.prime_certificates[0]
    assert certificate.proof_kind == "smooth_projective_complete_intersection"
    assert certificate.complete_intersection_codimension == 2
    assert len(certificate.jacobian_minors or []) == 6
    assert certificate.singular_locus_ideal is not None
    assert certificate.projective_singular_saturation == ring.ideal([ring.one()])
    certificate.verify()
    assert PolynomialPrimeCertificate.from_json(certificate.to_json()) == certificate


def test_homogeneous_affine_patch_prime_evidence_is_projected() -> None:
    ring = PolynomialRing(Q, ["x", "y", "z"])
    x, y, z = ring.generators()
    source = ring.ideal([y, x**2 + z**2])

    decomposition = source.minimal_primes()

    assert decomposition.components == [source]
    certificate = decomposition.prime_certificates[0]
    assert certificate.proof_kind == "homogeneous_affine_patch"
    assert certificate.homogeneous_patch_variable == 0
    assert certificate.homogeneous_patch_saturation == source
    patch_prime = certificate.homogeneous_patch_prime
    assert patch_prime is not None
    assert patch_prime.ideal == ring.ideal([y, x**2 + z**2, x - 1])
    assert patch_prime.proof_kind == "zero_dimensional_field"
    patch_prime.verify()
    certificate.verify()
    assert PolynomialPrimeCertificate.from_json(certificate.to_json()) == certificate

    primary = source.primary_decomposition()
    assert primary.components == [source]
    assert primary.certificates[0].proof_kind == "prime"
    assert primary.certificates[0].associated_prime == certificate
    primary.verify()

    payload = json.loads(certificate.to_json())
    payload["proof"]["variable"] = 3
    with pytest.raises(ValueError, match="nonhomogeneous or invalid"):
        PolynomialPrimeCertificate.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_complete_intersection_jacobian_radical_evidence_is_projected(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    source = ring.ideal([x**2 + y**2 - 1, z**2 - x])

    radical = source.radical()

    assert radical.proof_kind == "complete_intersection_jacobian"
    assert radical.complete_intersection_codimension == 2
    assert len(radical.jacobian_minors or []) == 3
    assert radical.singular_locus_ideal == ring.ideal([ring.one()])
    assert radical.singular_locus_dimension == -1
    assert radical.ideal == source
    radical.verify()
    assert RadicalIdeal.from_json(radical.to_json()) == radical

    with pytest.raises(UnsupportedOperationError, match="primality proof"):
        source.minimal_primes()


def test_radical_rejects_unsupported_coefficient_contexts() -> None:
    ring = PolynomialRing(Z, ["x"])
    x = ring.generator(0)

    with pytest.raises(UnsupportedOperationError):
        ring.ideal([x**2]).radical()


def test_zero_dimensional_minimal_primes_reconstruct_the_radical() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([(x - 1) ** 2 * (x + 2) ** 3, y**2])

    decomposition = source.minimal_primes()
    assert isinstance(decomposition, MinimalPrimeDecomposition)
    assert decomposition.source == source
    assert decomposition.radical == source.radical()
    assert len(decomposition.components) == 2
    assert len(decomposition.prime_certificates) == 2
    representations = [
        certificate.rational_univariate_representation
        for certificate in decomposition.prime_certificates
    ]
    assert all(
        representation is not None and representation.is_reduced
        for representation in representations
    )
    assert (
        decomposition.theorem == "the minimal primes intersect to the radical of the source ideal"
    )
    decomposition.verify()
    assert MinimalPrimeDecomposition.from_json(decomposition.to_json()) == decomposition

    payload = json.loads(decomposition.to_json())
    payload["components"] = []
    payload["prime_certificates"] = []
    with pytest.raises(ValueError, match="intersection does not reconstruct"):
        MinimalPrimeDecomposition.from_json(json.dumps(payload))


def test_nonmonogenic_finite_algebra_still_has_all_minimal_primes() -> None:
    field = FiniteField.prime(2)
    ring = PolynomialRing(field, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([x**2 + x, y**2 + y])

    with pytest.raises(ValueError):
        source.rational_univariate_representation()
    decomposition = source.minimal_primes()
    decomposition.verify()
    assert len(decomposition.components) == 4
    representations = [
        certificate.rational_univariate_representation
        for certificate in decomposition.prime_certificates
    ]
    assert all(
        representation is not None and representation.quotient_dimension == 1
        for representation in representations
    )


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_positive_dimensional_minimal_primes_have_typed_exact_certificates(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    first, second = x + y, x - z

    principal = ring.ideal([first**3 * second**2]).minimal_primes()
    principal.verify()
    assert len(principal.components) == 2
    assert all(
        certificate.proof_kind == "irreducible_hypersurface"
        and certificate.irreducible_factor is not None
        and certificate.coordinate_variables is None
        and certificate.rational_univariate_representation is None
        for certificate in principal.prime_certificates
    )

    monomial = ring.ideal([x**3 * y, x * z**4]).minimal_primes()
    monomial.verify()
    coordinate_sets = [
        certificate.coordinate_variables for certificate in monomial.prime_certificates
    ]
    assert all(variables is not None for variables in coordinate_sets)
    assert sorted(variables for variables in coordinate_sets if variables is not None) == [
        [0],
        [1, 2],
    ]
    for certificate in monomial.prime_certificates:
        assert isinstance(certificate, PolynomialPrimeCertificate)
        assert certificate.proof_kind == "coordinate_prime"
        certificate.verify()
        assert PolynomialPrimeCertificate.from_json(certificate.to_json()) == certificate


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_equidimensional_radical_decomposition_groups_all_minimal_primes(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()
    source = ring.ideal([x**3 * y, x * z**4])

    decomposition = source.equidimensional_radical_decomposition()
    assert isinstance(decomposition, EquidimensionalRadicalDecomposition)
    assert decomposition.source == source
    assert decomposition.radical == source.radical()
    assert [component.dimension for component in decomposition.components] == [2, 1]
    assert all(
        isinstance(component, EquidimensionalRadicalComponent)
        for component in decomposition.components
    )
    for component in decomposition.components:
        component.verify()
        assert all(
            certificate.ideal.dimension() == component.dimension
            for certificate in component.prime_certificates
        )
        assert EquidimensionalRadicalComponent.from_json(component.to_json()) == component
    decomposition.verify()
    assert EquidimensionalRadicalDecomposition.from_json(decomposition.to_json()) == decomposition


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(65_521)])
def test_primary_decomposition_retains_multiplicity_and_embedded_primes(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y", "z"])
    x, y, z = ring.generators()

    principal = ring.ideal([(x + y) ** 3 * (x - z) ** 2]).primary_decomposition()
    assert isinstance(principal, PrimaryDecomposition)
    assert len(principal.components) == 2
    assert all(
        isinstance(certificate, PolynomialPrimaryCertificate)
        and certificate.proof_kind == "principal_power"
        and certificate.nilpotence_bound is None
        for certificate in principal.certificates
    )
    principal.verify()

    source = ring.ideal([x**3 * y, x * z**4])
    decomposition = source.primary_decomposition()
    assert decomposition.source == source
    assert len(decomposition.components) == 3
    assert decomposition.components[0].ring == ring
    coordinate_sets = sorted(
        prime.coordinate_variables for prime in decomposition.associated_primes
    )
    assert coordinate_sets == [[0], [0, 2], [1, 2]]
    assert len(source.minimal_primes().components) == 2
    assert all(
        certificate.proof_kind == "monomial_primary" for certificate in decomposition.certificates
    )
    decomposition.verify()
    assert PrimaryDecomposition.from_json(decomposition.to_json()) == decomposition


def test_zero_dimensional_primary_components_have_local_nilpotence_evidence() -> None:
    ring = PolynomialRing(Q, ["x", "y"])
    x, y = ring.generators()
    source = ring.ideal([(x - 1) ** 3 * (x + 2) ** 2, y**4])

    decomposition = source.primary_decomposition()
    assert len(decomposition.components) == 2
    assert all(
        certificate.proof_kind == "zero_dimensional_local"
        and certificate.nilpotence_bound is not None
        and certificate.associated_prime.ideal.dimension() == 0
        for certificate in decomposition.certificates
    )
    for certificate in decomposition.certificates:
        certificate.verify()
        assert PolynomialPrimaryCertificate.from_json(certificate.to_json()) == certificate

    payload = json.loads(decomposition.to_json())
    payload["components"] = []
    payload["certificates"] = []
    with pytest.raises(ValueError, match="do not reconstruct"):
        PrimaryDecomposition.from_json(json.dumps(payload))


@pytest.mark.parametrize("coefficient_ring", [Q, FiniteField.prime(101)])
def test_independently_proposed_primary_components_are_exactly_admitted(
    coefficient_ring: RationalField | FiniteField,
) -> None:
    ring = PolynomialRing(coefficient_ring, ["x", "y"])
    x, y = ring.generators()
    first = ring.ideal([x**2])
    second = ring.ideal([y**3])
    source = first.intersection(second)

    certificate = first.certify_primary()
    certificate.verify()
    assert certificate.ideal == first

    admitted = source.admit_primary_decomposition([second, first])
    admitted.verify()
    assert len(admitted.components) == 2
    assert first in admitted.components
    assert second in admitted.components
    assert PrimaryDecomposition.from_json(admitted.to_json()) == admitted

    with pytest.raises(ValueError, match="not primary"):
        source.certify_primary()
    with pytest.raises(ValueError, match="reconstruct"):
        source.admit_primary_decomposition([first])
    with pytest.raises(ValueError, match="redundant"):
        source.admit_primary_decomposition([first, second, ring.ideal([x, y])])
