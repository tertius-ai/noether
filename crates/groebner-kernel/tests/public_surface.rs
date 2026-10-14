// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use groebner_kernel::{
    groebner_basis_prime, module_groebner_basis_prime, module_groebner_basis_with_transform_prime,
    module_membership, module_schreyer_syzygies_prime, strong_groebner_basis, GroebnerEvidence,
    GroebnerOptions, Integer, IntegerCoefficientRing, Monomial, MonomialOrder, PrimeField32,
    SparseModuleElement, SparseModuleOrder, SparsePolynomial, StrongGroebnerOptions,
};

fn monomial(exponents: &[u32]) -> Monomial {
    Monomial::new(exponents.to_vec().into_boxed_slice())
}

#[test]
fn public_prime_field_lane_returns_a_verified_reduced_basis() {
    let field = PrimeField32::new(101).unwrap();
    let generators = [
        SparsePolynomial::new(
            2,
            MonomialOrder::GradedReverseLexicographic,
            [(monomial(&[2, 0]), 1), (monomial(&[0, 1]), 1)],
            &field,
        ),
        SparsePolynomial::new(
            2,
            MonomialOrder::GradedReverseLexicographic,
            [(monomial(&[1, 1]), 1), (monomial(&[0, 0]), 1)],
            &field,
        ),
    ];
    let output = groebner_basis_prime(&field, &generators, &GroebnerOptions::default()).unwrap();

    assert!(!output.basis.is_empty());
    assert!(output.basis.iter().all(|polynomial| {
        polynomial
            .leading_term()
            .is_some_and(|(_, coefficient)| *coefficient == 1)
    }));
}

#[test]
fn public_integer_lane_computes_coefficient_gcds() {
    let generators = [
        SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            [(monomial(&[1]), Integer::from(2))],
            &IntegerCoefficientRing,
        ),
        SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            [(monomial(&[1]), Integer::from(3))],
            &IntegerCoefficientRing,
        ),
    ];
    let output = strong_groebner_basis(&generators, &StrongGroebnerOptions::default()).unwrap();

    assert!(output
        .basis
        .iter()
        .any(|polynomial| { polynomial.terms() == [(monomial(&[1]), Integer::one())].as_slice() }));
}

#[test]
fn public_module_lane_returns_exact_generator_coordinates() {
    let field = PrimeField32::new(101).unwrap();
    let polynomial =
        |terms| SparsePolynomial::new(2, MonomialOrder::GradedReverseLexicographic, terms, &field);
    let generators = [
        SparseModuleElement::from_coordinates(
            vec![
                polynomial([(monomial(&[2, 0]), 1)]),
                polynomial([(monomial(&[0, 1]), 1)]),
            ],
            &field,
        ),
        SparseModuleElement::from_coordinates(
            vec![
                polynomial([(monomial(&[1, 1]), 1)]),
                polynomial([(monomial(&[1, 0]), 1)]),
            ],
            &field,
        ),
    ];
    let output = module_groebner_basis_with_transform_prime(
        &field,
        &generators,
        &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
        &GroebnerOptions::default(),
    )
    .unwrap();
    let basis_only = module_groebner_basis_prime(
        &field,
        &generators,
        &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
        &GroebnerOptions::default(),
    )
    .unwrap();

    assert_eq!(
        output.evidence,
        GroebnerEvidence::IndependentBuchbergerSweep
    );
    assert_eq!(basis_only.basis, output.basis);
    assert_eq!(basis_only.evidence, output.evidence);
    assert!(basis_only.metrics.peak_matrix_columns < output.metrics.peak_matrix_columns);
    assert_eq!(output.basis.len(), output.transformations.len());
    assert!(output
        .transformations
        .iter()
        .all(|row| row.rank() == generators.len()));

    let syzygies = module_schreyer_syzygies_prime(
        &field,
        &output.basis,
        &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
        &GroebnerOptions::default(),
    )
    .unwrap();
    assert!(syzygies
        .generators
        .iter()
        .all(|relation| relation.rank() == output.basis.len()));
    assert!(module_membership(
        &field,
        &output.basis,
        &generators,
        &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
        &GroebnerOptions::default(),
    )
    .unwrap()
    .into_iter()
    .all(|member| member));
}
