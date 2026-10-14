// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    FiniteQuotientAlgebra, Ideal, IdealElimination, IdealReduction, MonomialOrder, PolynomialRing,
    RationalFunctionField, RationalUnivariateRepresentation, WeightedGrading,
};
use noether::numthy::FiniteField;
use noether::{Integer, Rational, RationalField, Ring, Q};

#[test]
fn rational_linear_ideal_has_a_canonical_reduced_basis() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let generators = ring.generators().unwrap();
    let x = &generators[0];
    let y = &generators[1];
    let sum = (x + y).unwrap();
    let difference = (x - y).unwrap();

    let ideal = ring.ideal([sum, difference]).unwrap();

    assert_eq!(ideal.groebner_basis(), [x.clone(), y.clone()]);
    assert!(ideal.contains(x).unwrap());
    assert!(ideal.contains(y).unwrap());
    assert_eq!(
        ideal.normal_form(&(x + &ring.one()).unwrap()).unwrap(),
        ring.one()
    );
    let value = (&(&x.clone() + y).unwrap() + &ring.one()).unwrap();
    let reduction = ideal.reduce(&value).unwrap();
    assert_eq!(reduction.remainder(), &ring.one());
    assert_eq!(
        ideal.normal_form(&value).unwrap(),
        reduction.remainder().clone()
    );
    assert_eq!(
        ideal.normal_forms(&[value.clone(), x.clone()]).unwrap(),
        vec![ring.one(), ring.zero()]
    );
    assert!(!reduction.is_member());
    reduction.verify().unwrap();
    assert_eq!(
        IdealReduction::<RationalField>::from_json(&reduction.to_json().unwrap()).unwrap(),
        reduction
    );
    ideal.verify().unwrap();
}

#[test]
fn normal_forms_retain_the_exact_large_exponent_floor() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let ideal = ring.ideal([x]).unwrap();
    let exponent = Integer::from(u64::from(u32::MAX) + 1);
    let value = ring
        .from_terms([
            (vec![exponent], Rational::from(1)),
            (vec![Integer::zero()], Rational::from(1)),
        ])
        .unwrap();

    assert_eq!(ideal.normal_form(&value).unwrap(), ring.one());
    assert_eq!(
        ideal.normal_forms(std::slice::from_ref(&value)).unwrap(),
        vec![ring.one()]
    );
}

#[test]
fn finite_field_normal_forms_match_witnessed_reduction_and_retain_large_exponents() {
    for field in [
        FiniteField::prime(65_521_u64).unwrap(),
        FiniteField::extension(2_u64, 4).unwrap(),
    ] {
        let ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let ideal = ring
            .ideal([(&x.pow(2).unwrap() - &y).unwrap(), y.pow(2).unwrap()])
            .unwrap();
        let value = (&(&x.pow(3).unwrap() + &y).unwrap() + &ring.one()).unwrap();
        let witnessed = ideal.reduce(&value).unwrap();

        assert_eq!(
            ideal.normal_form(&value).unwrap(),
            witnessed.remainder().clone()
        );
        assert_eq!(
            ideal
                .normal_forms(&[value.clone(), y.pow(2).unwrap()])
                .unwrap(),
            vec![witnessed.remainder().clone(), ring.zero()]
        );
        witnessed.verify().unwrap();
    }

    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let ideal = ring.ideal([x]).unwrap();
    let value = ring
        .from_terms([
            (vec![Integer::from(u64::from(u32::MAX) + 1)], field.one()),
            (vec![Integer::zero()], field.one()),
        ])
        .unwrap();
    assert_eq!(ideal.normal_form(&value).unwrap(), ring.one());
}

#[test]
fn block_order_uses_the_verified_generic_groebner_floor() {
    let order = MonomialOrder::elimination(3, [0]).unwrap();
    let ring = PolynomialRing::new(Q, ["x", "y", "z"], order).unwrap();
    let [x, y, z] = ring.generators().unwrap().try_into().unwrap();
    let first = (&x - &y).unwrap();
    let second = (&y - &z).unwrap();
    let ideal = ring.ideal([first, second]).unwrap();
    ideal.verify().unwrap();
    assert!(ideal.contains(&(&x - &z).unwrap()).unwrap());
    assert_eq!(ideal.dimension().unwrap(), 1);
}

#[test]
fn modular_rational_lane_returns_a_canonical_exact_ideal() {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let exponent = |x: u64, y: u64| vec![Integer::from(x), Integer::from(y)];
    let first = ring
        .from_terms([
            (exponent(4, 0), Rational::from(1)),
            (exponent(3, 1), Rational::from(1)),
            (exponent(2, 2), Rational::from(1)),
            (exponent(1, 3), Rational::from(1)),
            (exponent(0, 4), Rational::from(1)),
            (exponent(0, 0), Rational::from(1)),
        ])
        .unwrap();
    let second = ring
        .from_terms([
            (exponent(3, 0), Rational::from(1)),
            (exponent(2, 1), Rational::from(2)),
            (exponent(1, 2), Rational::from(3)),
            (exponent(0, 3), Rational::from(4)),
            (exponent(1, 0), Rational::from(1)),
            (exponent(0, 1), Rational::from(1)),
        ])
        .unwrap();
    let ideal = ring.ideal([first.clone(), second.clone()]).unwrap();

    assert!(ideal.contains(&first).unwrap());
    assert!(ideal.contains(&second).unwrap());
    ideal.verify().unwrap();
}

#[test]
fn zero_and_unit_ideals_have_exact_canonical_conventions() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    let zero = ring.ideal(Vec::new()).unwrap();
    let unit = ring
        .ideal([ring.constant(Rational::from(2)).unwrap()])
        .unwrap();

    assert!(zero.groebner_basis().is_empty());
    assert!(!zero.contains(&x).unwrap());
    assert_eq!(zero.normal_form(&x).unwrap(), x);
    assert_eq!(unit.groebner_basis(), [ring.one()]);
    assert!(unit.contains(&ring.one()).unwrap());
    zero.verify().unwrap();
    unit.verify().unwrap();
}

#[test]
fn finite_quotient_algebra_covers_a_nonmonogenic_local_algebra() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let ideal = ring
        .ideal([x.pow(2).unwrap(), (&x * &y).unwrap(), y.pow(2).unwrap()])
        .unwrap();
    let algebra = ideal.finite_quotient_algebra().unwrap();

    assert_eq!(algebra.dimension(), 3);
    assert_eq!(algebra.basis().len(), 3);
    assert!(!algebra.is_zero());
    assert_eq!(algebra.multiplication_operator_nonzeros().len(), 2);
    let one = algebra.one_coordinates();
    let x_coordinates = algebra.normal_form_coordinates(&x).unwrap();
    let y_coordinates = algebra.normal_form_coordinates(&y).unwrap();
    assert_eq!(
        algebra.multiply(&one, &x_coordinates).unwrap(),
        x_coordinates
    );
    assert_eq!(
        algebra.multiply(&y_coordinates, &one).unwrap(),
        y_coordinates
    );
    assert_eq!(
        algebra.multiply(&x_coordinates, &x_coordinates).unwrap(),
        vec![Rational::from(0); 3]
    );
    assert_eq!(
        algebra.multiply(&x_coordinates, &y_coordinates).unwrap(),
        vec![Rational::from(0); 3]
    );
    assert_eq!(
        algebra.multiply_by_variable(0, &one).unwrap(),
        algebra.normal_form_coordinates(&x).unwrap()
    );
    assert!(algebra.multiplication_operator_entries(2).is_err());
    assert!(algebra.multiply(&one[..2], &one).is_err());
    algebra.verify().unwrap();
    assert_eq!(
        FiniteQuotientAlgebra::<RationalField>::from_json(&algebra.to_json().unwrap()).unwrap(),
        algebra
    );

    // No primitive element generates this three-dimensional local algebra,
    // but its finite quotient description remains complete.
    assert!(ideal.rational_univariate_representation().is_err());

    let zero_algebra = ring
        .ideal([ring.one()])
        .unwrap()
        .finite_quotient_algebra()
        .unwrap();
    assert!(zero_algebra.is_zero());
    assert_eq!(zero_algebra.dimension(), 0);
    assert!(ring
        .ideal(Vec::new())
        .unwrap()
        .finite_quotient_algebra()
        .is_err());
}

#[test]
fn extension_fields_use_the_same_verified_ideal_floor() {
    let field = FiniteField::extension(3_u64, 2).unwrap();
    let ring = PolynomialRing::new(
        field.clone(),
        ["x", "y"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let generators = ring.generators().unwrap();
    let x = &generators[0];
    let y = &generators[1];
    let xy_minus_one = (&(x * y).unwrap() - &ring.one()).unwrap();
    let y_squared_minus_x = (&y.pow(2).unwrap() - x).unwrap();
    let scaled = xy_minus_one.scalar_mul(&field.element(2_u8)).unwrap();

    let first = ring
        .ideal([xy_minus_one.clone(), y_squared_minus_x.clone()])
        .unwrap();
    let reordered = ring.ideal([y_squared_minus_x, scaled]).unwrap();

    assert_eq!(first, reordered);
    assert!(first.contains(&xy_minus_one).unwrap());
    first.verify().unwrap();
}

#[test]
fn change_order_uses_verified_fglm_over_arbitrary_fields() {
    for field in [
        FiniteField::prime(101_u64).unwrap(),
        FiniteField::extension(3_u64, 2).unwrap(),
    ] {
        let source_ring = PolynomialRing::new(
            field.clone(),
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let source_variables = source_ring.generators().unwrap();
        let source = source_ring
            .ideal([
                (&source_variables[0].pow(2).unwrap() - &source_variables[1]).unwrap(),
                (&source_variables[1].pow(2).unwrap() - &source_variables[0]).unwrap(),
            ])
            .unwrap();

        let converted = source.change_order(MonomialOrder::Lexicographic).unwrap();
        let target_ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let target_variables = target_ring.generators().unwrap();
        let expected = target_ring
            .ideal([
                (&target_variables[0].pow(2).unwrap() - &target_variables[1]).unwrap(),
                (&target_variables[1].pow(2).unwrap() - &target_variables[0]).unwrap(),
            ])
            .unwrap();

        assert_eq!(converted, expected);
        assert_eq!(converted.ring().term_order(), MonomialOrder::Lexicographic);
        assert_eq!(
            converted
                .change_order(MonomialOrder::GradedReverseLexicographic)
                .unwrap(),
            source
        );
        converted.verify().unwrap();
    }
}

#[test]
fn change_order_retains_the_positive_dimensional_exact_floor() {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let variables = ring.generators().unwrap();
    let source = ring
        .ideal([(&variables[0] * &variables[1]).unwrap()])
        .unwrap();
    let changed = source.change_order(MonomialOrder::Lexicographic).unwrap();

    assert_eq!(changed.dimension().unwrap(), 1);
    assert_eq!(changed.ring().term_order(), MonomialOrder::Lexicographic);
    changed.verify().unwrap();
}

#[test]
fn rational_univariate_representation_is_generic_exact_and_serializable() {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let variables = ring.generators().unwrap();
    let ideal = ring
        .ideal([
            (&variables[0].pow(2).unwrap() - &variables[1]).unwrap(),
            (&variables[1].pow(2).unwrap() - &variables[0]).unwrap(),
        ])
        .unwrap();
    let representation = ideal.rational_univariate_representation().unwrap();

    assert_eq!(representation.source(), &ideal);
    assert_eq!(representation.quotient_dimension(), 4);
    assert_eq!(representation.coordinate_numerators().len(), 2);
    assert_eq!(
        representation.denominator(),
        &representation.parameter_ring().one()
    );
    assert_eq!(representation.linear_form().len(), 2);
    assert!(representation.is_reduced());
    representation.verify().unwrap();
    assert_eq!(
        RationalUnivariateRepresentation::<RationalField>::from_json(
            &representation.to_json().unwrap()
        )
        .unwrap(),
        representation
    );
}

#[test]
fn rational_univariate_representation_records_nilpotents() {
    let field = FiniteField::prime(5_u64).unwrap();
    let ring = PolynomialRing::new(field, ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    let ideal = ring.ideal([x.pow(3).unwrap()]).unwrap();
    let representation = ideal.rational_univariate_representation().unwrap();

    assert_eq!(representation.quotient_dimension(), 3);
    assert!(!representation.is_reduced());
    representation.verify().unwrap();
}

#[test]
fn binary_field_kernel_lanes_preserve_verified_facade_semantics() {
    for field in [
        FiniteField::prime(2_u64).unwrap(),
        FiniteField::extension(2_u64, 2).unwrap(),
    ] {
        let ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let generators = ring.generators().unwrap();
        let sum = (&generators[0] + &generators[1]).unwrap();
        let quadratic = (&generators[1].pow(2).unwrap() + &generators[1]).unwrap();
        let ideal = ring.ideal([sum.clone(), quadratic.clone()]).unwrap();

        assert!(ideal.contains(&sum).unwrap());
        assert!(ideal.contains(&quadratic).unwrap());
        ideal.verify().unwrap();
    }
}

#[test]
fn polynomial_ideal_json_recomputes_canonical_identity() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let generators = ring.generators().unwrap();
    let ideal = ring
        .ideal([
            (&generators[0] + &generators[1]).unwrap(),
            (&generators[0] - &generators[1]).unwrap(),
        ])
        .unwrap();

    let encoded = ideal.to_json().unwrap();
    assert_eq!(
        Ideal::<PolynomialRing<RationalField>>::from_json(&encoded).unwrap(),
        ideal
    );

    let mut reordered: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    reordered["generators"].as_array_mut().unwrap().reverse();
    assert!(Ideal::<PolynomialRing<RationalField>>::from_json(&reordered.to_string()).is_err());
}

#[test]
fn twisted_cubic_has_exact_hilbert_invariants() {
    let ring = PolynomialRing::new(Q, ["x", "y", "z", "w"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let (x, y, z, w) = (&variables[0], &variables[1], &variables[2], &variables[3]);
    let ideal = ring
        .ideal([
            (&(x * z).unwrap() - &y.pow(2).unwrap()).unwrap(),
            (&(x * w).unwrap() - &(y * z).unwrap()).unwrap(),
            (&(y * w).unwrap() - &z.pow(2).unwrap()).unwrap(),
        ])
        .unwrap();

    assert_eq!(ideal.dimension().unwrap(), 2);
    assert_eq!(ideal.degree().unwrap().to_string(), "3");
    assert_eq!(
        ideal
            .hilbert_numerator()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["1", "0", "-3", "2"]
    );

    let polynomial = ideal.hilbert_polynomial().unwrap();
    let hilbert_ring = polynomial.ring();
    let m = hilbert_ring.generator(0).unwrap();
    assert_eq!(
        polynomial,
        (&m.scalar_mul(&Rational::from(3)).unwrap() + &hilbert_ring.one()).unwrap()
    );
}

#[test]
fn affine_complete_intersections_and_boundary_ideals_are_hand_checked() {
    let field = FiniteField::prime(3_u64).unwrap();
    let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let ideal = ring
        .ideal([variables[0].pow(2).unwrap(), variables[1].pow(3).unwrap()])
        .unwrap();
    assert_eq!(ideal.dimension().unwrap(), 0);
    assert_eq!(ideal.degree().unwrap().to_string(), "6");
    assert!(ideal.hilbert_polynomial().unwrap().is_zero());

    let zero = ring.ideal(Vec::new()).unwrap();
    assert_eq!(zero.dimension().unwrap(), 2);
    assert_eq!(zero.degree().unwrap().to_string(), "1");
    let unit = ring.ideal([ring.one()]).unwrap();
    assert_eq!(unit.dimension().unwrap(), -1);
    assert_eq!(unit.degree().unwrap().to_string(), "0");
}

#[test]
fn standard_hilbert_polynomial_rejects_a_weighted_context() {
    let ring = PolynomialRing::with_grading(
        Q,
        ["x", "y"],
        MonomialOrder::default(),
        WeightedGrading::new(vec![1, 2]).unwrap(),
    )
    .unwrap();
    let ideal = ring.ideal([ring.generator(0).unwrap()]).unwrap();

    assert_eq!(ideal.dimension().unwrap(), 1);
    assert!(ideal.degree().is_err());
    assert!(ideal.hilbert_polynomial().is_err());
}

#[test]
fn elimination_reorders_privately_and_returns_to_the_original_context() {
    let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let (x, y, z) = (&variables[0], &variables[1], &variables[2]);
    let ideal = ring
        .ideal([(x - &y.pow(2).unwrap()).unwrap(), (z - y).unwrap()])
        .unwrap();

    let eliminated = ideal.eliminate(&[1, 1]).unwrap();
    let expected = ring.ideal([(x - &z.pow(2).unwrap()).unwrap()]).unwrap();
    assert_eq!(eliminated, expected);
    assert_eq!(eliminated.ring(), &ring);
    let evidence = ideal.eliminate_with_evidence(&[1, 1]).unwrap();
    assert_eq!(evidence.variables(), &[1]);
    assert_eq!(evidence.result(), &expected);
    evidence.verify().unwrap();
    assert_eq!(
        IdealElimination::<RationalField>::from_json(&evidence.to_json().unwrap()).unwrap(),
        evidence
    );
    assert_eq!(ideal.eliminate(&[]).unwrap(), ideal);
    assert!(ideal.eliminate(&[3]).is_err());
}

#[test]
fn elimination_handles_all_variables_and_finite_fields() {
    let field = FiniteField::prime(3_u64).unwrap();
    let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let ideal = ring
        .ideal([(&(&variables[0] * &variables[1]).unwrap() - &ring.one()).unwrap()])
        .unwrap();

    assert_eq!(ideal.eliminate(&[0, 1]).unwrap(), ring.ideal([]).unwrap());
    let unit = ring.ideal([ring.one()]).unwrap();
    assert_eq!(unit.eliminate(&[0, 1]).unwrap(), unit);
}

#[test]
fn intersection_and_colon_have_hand_checked_monomial_results() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let (x, y) = (&variables[0], &variables[1]);
    let x_ideal = ring.ideal([x.clone()]).unwrap();
    let y_ideal = ring.ideal([y.clone()]).unwrap();

    assert_eq!(
        x_ideal.intersection(&y_ideal).unwrap(),
        ring.ideal([(x * y).unwrap()]).unwrap()
    );

    let numerator = ring.ideal([x.pow(2).unwrap(), (x * y).unwrap()]).unwrap();
    assert_eq!(
        numerator.colon(&x_ideal).unwrap(),
        ring.ideal([x.clone(), y.clone()]).unwrap()
    );
    let denominator = ring.ideal([x.clone(), y.clone()]).unwrap();
    assert_eq!(numerator.colon(&denominator).unwrap(), x_ideal.clone());

    let difference = (x - y).unwrap();
    let sum = (x + y).unwrap();
    let factored = ring
        .ideal([(&x.pow(2).unwrap() - &y.pow(2).unwrap()).unwrap()])
        .unwrap();
    assert_eq!(
        factored.colon(&ring.ideal([difference]).unwrap()).unwrap(),
        ring.ideal([sum]).unwrap()
    );
}

#[test]
fn saturation_stabilizes_and_ideal_calculus_checks_contexts() {
    let field = FiniteField::prime(3_u64).unwrap();
    let ring = PolynomialRing::new(field.clone(), ["x", "y"], MonomialOrder::default()).unwrap();
    let variables = ring.generators().unwrap();
    let (x, y) = (&variables[0], &variables[1]);
    let numerator = ring.ideal([x.pow(2).unwrap(), (x * y).unwrap()]).unwrap();
    let y_ideal = ring.ideal([y.clone()]).unwrap();

    assert_eq!(
        numerator.saturate(&y_ideal).unwrap(),
        ring.ideal([x.clone()]).unwrap()
    );
    assert_eq!(
        numerator.colon(&ring.ideal([]).unwrap()).unwrap(),
        ring.ideal([ring.one()]).unwrap()
    );
    assert_eq!(
        numerator
            .saturate(&ring.ideal([ring.one()]).unwrap())
            .unwrap(),
        numerator
    );

    let other = PolynomialRing::new(field, ["u", "v"], MonomialOrder::default())
        .unwrap()
        .ideal([])
        .unwrap();
    assert!(numerator.intersection(&other).is_err());
    assert!(numerator.colon(&other).is_err());
    assert!(numerator.saturate(&other).is_err());
}

#[test]
fn rational_function_coefficients_use_the_same_exact_field_ideal_floor() {
    let base = FiniteField::prime(3_u64).unwrap();
    let functions = RationalFunctionField::new(base, "c").unwrap();
    let ring =
        PolynomialRing::new(functions.clone(), ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let variables = ring.generators().unwrap();
    let (x, y) = (&variables[0], &variables[1]);
    let c = functions.generator();
    let ideal = ring
        .ideal([
            (x - &y.scalar_mul(&c).unwrap()).unwrap(),
            (&y.pow(2).unwrap() - &ring.one()).unwrap(),
        ])
        .unwrap();

    ideal.verify().unwrap();
    assert_eq!(ideal.dimension().unwrap(), 0);
    let c_squared = (&c * &c).unwrap();
    let expected = ring
        .ideal([(&x.pow(2).unwrap() - &ring.constant(c_squared).unwrap()).unwrap()])
        .unwrap();
    assert_eq!(ideal.eliminate(&[1]).unwrap(), expected);

    let degree_ring = PolynomialRing::new(
        functions,
        ["x", "y"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let degree_variables = degree_ring.generators().unwrap();
    let degree_ideal = degree_ring
        .ideal([
            (&degree_variables[0] - &degree_variables[1].scalar_mul(&c).unwrap()).unwrap(),
            (&degree_variables[1].pow(2).unwrap() - &degree_ring.one()).unwrap(),
        ])
        .unwrap();
    assert_eq!(degree_ideal.degree().unwrap().to_string(), "2");

    let encoded = ideal.to_json().unwrap();
    assert_eq!(
        Ideal::<PolynomialRing<RationalFunctionField<FiniteField>>>::from_json(&encoded).unwrap(),
        ideal
    );
}
