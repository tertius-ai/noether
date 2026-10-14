// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    ArithmeticFiberSupport, ArithmeticFiberSupportClaim, ArithmeticIdealError,
    ArithmeticNullstellensatzCertificate, Ideal, IdealReduction, MonomialOrder, PolynomialRing,
    StrongGroebnerBasis,
};
use noether::numthy::FiniteField;
use noether::{Integer, Ring, Z};

#[test]
fn integral_strong_basis_detects_coefficient_gcds_and_membership() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let two_x = ring
        .mul(&ring.constant(Integer::from(2)).unwrap(), &x)
        .unwrap();
    let three_x = ring
        .mul(&ring.constant(Integer::from(3)).unwrap(), &x)
        .unwrap();

    let basis = ring
        .strong_groebner_basis([two_x.clone(), three_x.clone()])
        .unwrap();
    assert_eq!(basis.basis(), std::slice::from_ref(&x));
    assert!(basis.contains(&two_x).unwrap());
    assert!(basis.contains(&three_x).unwrap());
    assert!(!basis.contains(&ring.one()).unwrap());
    basis.verify().unwrap();

    let decoded = StrongGroebnerBasis::from_json(&basis.to_json().unwrap()).unwrap();
    assert_eq!(decoded, basis);
}

#[test]
fn integral_strong_basis_is_deterministic_under_generator_order_and_sign() {
    let ring =
        PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let first = ring.add(&ring.mul(&x, &x).unwrap(), &y).unwrap();
    let second = ring.add(&ring.mul(&x, &y).unwrap(), &ring.one()).unwrap();
    let negative_second = ring.neg(&second).unwrap();

    let left = ring.strong_groebner_basis([first.clone(), second]).unwrap();
    let right = ring
        .strong_groebner_basis([negative_second, first])
        .unwrap();
    assert_eq!(left, right);
}

#[test]
fn integral_polynomial_ideal_has_canonical_equality_and_multiplier_evidence() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let two_x = x.scalar_mul(&Integer::from(2)).unwrap();
    let three_x = x.scalar_mul(&Integer::from(3)).unwrap();
    let ideal = ring.ideal([two_x, three_x]).unwrap();
    assert_eq!(ideal.generators(), std::slice::from_ref(&x));
    assert_eq!(ideal, ring.ideal([x.clone()]).unwrap());
    assert!(ideal.contains(&x).unwrap());

    let value = x.scalar_mul(&Integer::from(7)).unwrap();
    let reduction = ideal.reduce(&value).unwrap();
    assert!(reduction.is_member());
    assert_eq!(reduction.remainder(), &ring.zero());
    assert_eq!(
        ideal.normal_form(&value).unwrap(),
        reduction.remainder().clone()
    );
    assert_eq!(
        ideal
            .normal_forms(&[value.clone(), (&x + &ring.one()).unwrap()])
            .unwrap(),
        vec![ring.zero(), ring.one()]
    );
    assert_eq!(reduction.multipliers(), &[ring.constant(7.into()).unwrap()]);
    reduction.verify().unwrap();

    let ideal_json = ideal.to_json().unwrap();
    assert_eq!(
        Ideal::<PolynomialRing<noether::IntegerRing>>::from_json(&ideal_json).unwrap(),
        ideal
    );
    let reduction_json = reduction.to_json().unwrap();
    assert_eq!(
        IdealReduction::<noether::IntegerRing>::from_json(&reduction_json).unwrap(),
        reduction
    );
}

#[test]
fn integral_ideal_elimination_contraction_and_base_change_are_exact() {
    let ring =
        PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let first = (&x.scalar_mul(&2.into()).unwrap() + &y).unwrap();
    let second = x.scalar_mul(&3.into()).unwrap();
    let ideal = ring.ideal([first, second]).unwrap();

    let elimination = ideal.eliminate(&[0]).unwrap();
    let three_y = y.scalar_mul(&3.into()).unwrap();
    assert!(elimination.contains(&three_y).unwrap());
    assert!(elimination
        .generators()
        .iter()
        .all(|generator| generator.terms().all(|(exponent, _)| exponent[0].is_zero())));

    let contraction = ideal.contract(&[0]).unwrap();
    assert_eq!(contraction.ring().variables(), &["y"]);
    assert_eq!(
        contraction.generators(),
        &[contraction
            .ring()
            .generator(0)
            .unwrap()
            .scalar_mul(&3.into())
            .unwrap()]
    );

    let rational = ideal.base_change(&Z.rational_embedding()).unwrap();
    assert_eq!(rational.generators().len(), 2);
    assert!(rational
        .contains(&rational.ring().generator(0).unwrap())
        .unwrap());
    assert!(rational
        .contains(&rational.ring().generator(1).unwrap())
        .unwrap());

    let field = FiniteField::prime(3_u8).unwrap();
    let reduction = ideal.base_change(&Z.finite_field_map(&field)).unwrap();
    let [x_mod_three, y_mod_three] = reduction.ring().generators().unwrap().try_into().unwrap();
    assert_eq!(
        reduction.generators(),
        &[(&x_mod_three + &y_mod_three.scalar_mul(&field.element(2)).unwrap()).unwrap()]
    );

    let one_variable = PolynomialRing::new(Z, ["u"], MonomialOrder::Lexicographic).unwrap();
    let u = one_variable.generator(0).unwrap();
    let bad_at_two = one_variable
        .ideal([
            u.scalar_mul(&2.into()).unwrap(),
            (&u + &one_variable.one()).unwrap(),
        ])
        .unwrap();
    let coefficient_contraction = bad_at_two.contract_to_coefficients().unwrap();
    assert_eq!(
        coefficient_contraction.principal_generator(),
        Integer::from(2)
    );
    assert!(coefficient_contraction.contains(&Integer::from(10)));
    assert!(!coefficient_contraction.contains(&Integer::from(3)));
    assert_eq!(
        Ideal::<noether::IntegerRing>::from_json(&coefficient_contraction.to_json().unwrap())
            .unwrap(),
        coefficient_contraction
    );
    assert_eq!(
        coefficient_contraction
            .sum(&Z.ideal([Integer::from(6)]))
            .principal_generator(),
        Integer::from(2)
    );
}

#[test]
fn arithmetic_nullstellensatz_certificate_and_fiber_support_separate_claims() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let x_plus_one = (&x + &ring.one()).unwrap();
    let at_two_ideal = ring
        .ideal([x.scalar_mul(&2.into()).unwrap(), x_plus_one.clone()])
        .unwrap();
    let at_two = at_two_ideal.nullstellensatz_certificate().unwrap();
    assert_eq!(at_two.integer(), &Integer::from(2));
    assert_eq!(
        at_two.principal_ideal().principal_generator(),
        Integer::from(2)
    );
    at_two.verify().unwrap();
    assert!(at_two.is_exact_contraction().unwrap());
    assert_eq!(
        ArithmeticNullstellensatzCertificate::from_json(&at_two.to_json().unwrap()).unwrap(),
        at_two
    );

    let at_three = ring
        .ideal([x.scalar_mul(&3.into()).unwrap(), x_plus_one])
        .unwrap()
        .nullstellensatz_certificate()
        .unwrap();
    assert_eq!(at_three.integer(), &Integer::from(3));
    assert_eq!(
        at_two
            .principal_ideal()
            .sum(at_three.principal_ideal())
            .principal_generator(),
        Integer::one()
    );

    let positive_dimensional = ring.ideal([x]).unwrap();
    assert_eq!(
        positive_dimensional.nullstellensatz_certificate(),
        Err(ArithmeticIdealError::GenericFiberNonempty)
    );

    let exact = at_two_ideal.arithmetic_fiber_support(2, 64).unwrap();
    assert_eq!(exact.claim(), ArithmeticFiberSupportClaim::ExactContraction);
    assert!(exact.is_exact());
    assert!(exact.certificate().is_none());
    assert_eq!(exact.proved_primes(), vec![(Integer::from(2), 1)]);
    assert!(exact.prime_support_is_complete());
    assert!(exact.unfactored_cofactor().is_none());
    exact.verify().unwrap();
    assert_eq!(
        ArithmeticFiberSupport::from_json(&exact.to_json().unwrap()).unwrap(),
        exact
    );

    let inflated = ArithmeticNullstellensatzCertificate::from_generator_identity(
        at_two.ring().clone(),
        at_two.generators().to_vec(),
        Integer::from(6),
        at_two
            .multipliers()
            .iter()
            .map(|multiplier| multiplier.scalar_mul(&Integer::from(3)).unwrap())
            .collect(),
    )
    .unwrap();
    assert!(!inflated.is_exact_contraction().unwrap());
    let containing = inflated.arithmetic_fiber_support(2, 64).unwrap();
    assert_eq!(
        containing.claim(),
        ArithmeticFiberSupportClaim::CertifiedContainment
    );
    assert!(!containing.is_exact());
    assert!(containing.certificate().is_some());
    assert_eq!(
        containing.proved_primes(),
        vec![(Integer::from(2), 1), (Integer::from(3), 1)]
    );
    containing.verify().unwrap();
}

#[test]
fn rational_nullstellensatz_path_preserves_the_supplied_generator_family() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let generators = vec![
        x.scalar_mul(&Integer::from(2)).unwrap(),
        (&x + &ring.one()).unwrap(),
    ];

    let certificate = ring
        .nullstellensatz_certificate(generators.clone())
        .unwrap();
    assert_eq!(certificate.ring(), &ring);
    assert_eq!(certificate.generators(), generators);
    assert_eq!(certificate.integer(), &Integer::from(2));
    assert_eq!(certificate.multipliers().len(), 2);
    certificate.verify().unwrap();
    assert_eq!(
        ArithmeticNullstellensatzCertificate::from_json(&certificate.to_json().unwrap()).unwrap(),
        certificate
    );

    assert_eq!(
        ring.nullstellensatz_certificate([x]),
        Err(ArithmeticIdealError::GenericFiberNonempty)
    );
}

#[test]
fn arithmetic_fiber_support_reports_an_unfactored_cofactor() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let witness = Integer::from(2) * Integer::from(101) * Integer::from(103);
    let ideal = ring
        .ideal([ring.constant(witness.clone()).unwrap()])
        .unwrap();
    let support = ideal.arithmetic_fiber_support(2, 8).unwrap();
    assert!(support.is_exact());
    assert!(!support.prime_support_is_complete());
    assert_eq!(support.proved_primes(), vec![(Integer::from(2), 1)]);
    assert_eq!(
        support.unfactored_cofactor(),
        Some(&(Integer::from(101) * Integer::from(103)))
    );
    support.verify().unwrap();
}
