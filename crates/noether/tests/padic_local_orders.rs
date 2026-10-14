// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::linalg::{LinearOperator, Matrix, SparseMatrix, Vector};
use noether::numthy::{
    FiniteField, PadicLocalField, PadicLocalFieldOrder, PadicLocalOrderError, PadicLocalPrecision,
    QadicRing,
};
use noether::{Ring, SemanticType, UnitInversion};

fn quadratic_field(local_precision: u64) -> PadicLocalField {
    let residue = FiniteField::prime(3_u64).unwrap();
    let coefficients = QadicRing::new(residue, local_precision.div_ceil(2)).unwrap();
    let polynomials =
        PolynomialRing::new(coefficients.clone(), ["T"], MonomialOrder::Lexicographic).unwrap();
    let eisenstein = polynomials
        .from_u64_terms([
            (vec![0], coefficients.element(3)),
            (vec![2], coefficients.one()),
        ])
        .unwrap();
    PadicLocalField::from_eisenstein(
        coefficients,
        eisenstein,
        PadicLocalPrecision::new(local_precision).unwrap(),
        Some("pi".to_owned()),
    )
    .unwrap()
}

fn ramified_unramified_field() -> PadicLocalField {
    let residue = FiniteField::extension(3_u64, 2).unwrap();
    let coefficients = QadicRing::new(residue, 5).unwrap();
    let polynomials =
        PolynomialRing::new(coefficients.clone(), ["T"], MonomialOrder::Lexicographic).unwrap();
    let eisenstein = polynomials
        .from_u64_terms([
            (vec![0], coefficients.element(3)),
            (vec![2], coefficients.one()),
        ])
        .unwrap();
    PadicLocalField::from_eisenstein(
        coefficients,
        eisenstein,
        PadicLocalPrecision::new(9).unwrap(),
        Some("pi".to_owned()),
    )
    .unwrap()
}

#[test]
fn eisenstein_equation_order_is_the_certified_maximal_order() {
    let field = quadratic_field(10);
    let order = field.maximal_order().unwrap();

    assert_eq!(
        PadicLocalFieldOrder::TYPE_ID,
        "noether/numthy/PadicLocalFieldOrder@1"
    );
    assert_eq!(order, field.equation_order().unwrap());
    assert_eq!(order.basis().len(), field.degree());
    assert_eq!(order.certified_precision(), 5);
    assert!(order.is_maximal());
    assert_eq!(order.index_in_maximal_order_valuation(), 0);
    assert_eq!(order.different_valuation().unwrap(), 1);
    assert_eq!(order.discriminant_valuation().unwrap(), 1);
    assert!(order.contains(&field.uniformizer()));
    assert_eq!(
        PadicLocalFieldOrder::from_json(&order.to_json().unwrap()).unwrap(),
        order
    );

    let unit = field.add(&field.one(), &field.uniformizer()).unwrap();
    assert!(order.contains(&order.inverse_unit(&unit).unwrap()));
}

#[test]
fn nonmaximal_zp_plus_p_ok_has_canonical_lattice_and_index() {
    let field = quadratic_field(10);
    let pi = field.uniformizer();
    let p = field
        .integral_element(&[field.coefficient_ring().element(3)])
        .unwrap();
    let p_pi = field.mul(&p, &pi).unwrap();
    let order = field.order(&[field.one(), p_pi.clone()]).unwrap();

    assert!(!order.is_maximal());
    assert_eq!(order.index_in_maximal_order_valuation(), 1);
    assert_eq!(order.discriminant_valuation().unwrap(), 3);
    assert_eq!(
        order.different_valuation(),
        Err(PadicLocalOrderError::DifferentRequiresMaximalOrder)
    );
    assert!(order.contains(&p_pi));
    assert!(!order.contains(&pi));

    let reordered = field.order(&[p_pi, field.one()]).unwrap();
    assert_eq!(reordered, order);
    let maximal = order.maximal_overorder().unwrap();
    assert_eq!(order.index_valuation(&maximal).unwrap(), 1);
    assert_eq!(maximal, field.maximal_order().unwrap());
}

#[test]
fn order_construction_fails_closed_on_missing_or_unresolved_lattice_data() {
    let field = quadratic_field(6);
    let pi = field.uniformizer();
    let p = field
        .integral_element(&[field.coefficient_ring().element(3)])
        .unwrap();

    assert_eq!(
        field.order(&[p.clone(), pi.clone()]),
        Err(PadicLocalOrderError::MissingOne)
    );

    let unresolved = field.mul(&p.pow(3), &pi).unwrap();
    assert!(matches!(
        field.order(&[field.one(), unresolved]),
        Err(PadicLocalOrderError::InsufficientPrecision { .. })
    ));

    let inverse_pi = pi.inverse().unwrap();
    assert!(matches!(
        field.order(&[field.one(), inverse_pi]),
        Err(PadicLocalOrderError::NonIntegralBasis { .. })
    ));
}

#[test]
fn zp_plus_p_ok_uses_a_full_zp_basis_when_the_residue_degree_is_nontrivial() {
    let field = ramified_unramified_field();
    let maximal = field.maximal_order().unwrap();
    let p = field
        .integral_element(&[field.coefficient_ring().element(3)])
        .unwrap();
    let mut basis = maximal.basis().to_vec();
    for element in &mut basis[1..] {
        *element = field.mul(&p, element).unwrap();
    }
    let order = field.order(&basis).unwrap();

    assert_eq!(field.degree(), 4);
    assert_eq!(order.basis().len(), 4);
    assert_eq!(order.index_in_maximal_order_valuation(), 3);
    assert_eq!(order.index_valuation(&maximal).unwrap(), 3);
    assert!(order.contains(&field.one()));
    let theta = field
        .element_from_qadic(&field.coefficient_field().generator())
        .unwrap();
    assert!(!order.contains(&theta));
}

#[test]
fn local_orders_are_generic_dense_and_sparse_coefficient_contexts() {
    let field = quadratic_field(10);
    let pi = field.uniformizer();
    let p = field
        .integral_element(&[field.coefficient_ring().element(3)])
        .unwrap();
    let p_pi = field.mul(&p, &pi).unwrap();
    let order = field.order(&[field.one(), p_pi.clone()]).unwrap();
    let vector = Vector::from_values(order.clone(), [field.one(), p_pi.clone()]).unwrap();
    let matrix = Matrix::from_rows(
        order.clone(),
        [[field.one(), p_pi.clone()], [field.zero(), field.one()]],
    )
    .unwrap();
    let sparse = SparseMatrix::from_dense(&matrix).unwrap();

    assert_eq!(matrix.ring(), &order);
    assert_eq!(sparse.ring(), &order);
    assert_eq!(
        matrix.apply(&vector).unwrap(),
        sparse.apply(&vector).unwrap()
    );
    assert_eq!(sparse.to_dense(), matrix);
}
