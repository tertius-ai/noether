// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{Degree, MonomialOrder, PolynomialRing};
use noether::homalg::{
    FreeModule, ModuleElement, ModuleGroebnerBasis, ModuleHomomorphism, ModuleMonomialOrder,
    ModuleReduction, PresentedModule, StrongModuleGroebnerBasis, StrongModuleReduction, Submodule,
};
use noether::linalg::{Matrix, SparseMatrix, Vector};
use noether::numthy::FiniteField;
use noether::{Integer, RationalField, Ring, SemanticType, Q, Z};

#[test]
fn module_values_work_generically_over_a_polynomial_ring() {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let y = ring.generator(1).unwrap();
    let source = FreeModule::new(ring.clone(), 2)
        .with_degree_shifts(
            1,
            vec![
                Degree::new(vec![Integer::zero()]),
                Degree::new(vec![Integer::zero()]),
            ],
        )
        .unwrap();
    let target = FreeModule::new(ring.clone(), 1)
        .with_degree_shifts(1, vec![Degree::new(vec![Integer::one()])])
        .unwrap();
    let map = ModuleHomomorphism::from_matrix(
        source.clone(),
        target.clone(),
        Matrix::from_rows(ring.clone(), [[x.clone(), y.clone()]]).unwrap(),
    )
    .unwrap();
    let relation = ModuleElement::from_coordinates(
        source,
        Vector::from_values(ring.clone(), [y, ring.neg(&x).unwrap()]).unwrap(),
    )
    .unwrap();

    assert_eq!(
        map.apply(&relation)
            .unwrap()
            .coordinates()
            .values()
            .collect::<Vec<_>>(),
        vec![ring.zero()]
    );
    let presented = PresentedModule::cokernel(map);
    assert_eq!(presented.generator_count(), 1);
    assert_eq!(presented.relation_count(), 2);
    assert_eq!(
        PresentedModule::<PolynomialRing<noether::RationalField>>::from_json(
            &presented.to_json().unwrap()
        )
        .unwrap(),
        presented
    );
}

#[test]
fn sparse_generator_maps_preserve_zero_row_and_column_shapes() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let zero = FreeModule::new(ring.clone(), 0);
    let ambient = FreeModule::new(ring.clone(), 3);
    let empty_map = ModuleHomomorphism::from_sparse_matrix(
        zero,
        ambient.clone(),
        SparseMatrix::zeros(ring, 3, 0),
    )
    .unwrap();
    let submodule = Submodule::from_generator_map(ambient, empty_map).unwrap();

    assert_eq!(submodule.generator_count(), 0);
    assert_eq!(submodule.generator_map().matrix().shape(), (3, 0));
    assert_eq!(
        Submodule::<PolynomialRing<noether::RationalField>>::from_json(
            &submodule.to_json().unwrap()
        )
        .unwrap()
        .canonical_bytes()
        .unwrap(),
        submodule.canonical_bytes().unwrap()
    );
}

#[test]
fn sparse_generator_lists_retain_sparse_module_maps() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let ambient = FreeModule::new(ring, 4);
    let submodule = Submodule::from_generators(
        ambient.clone(),
        vec![
            ambient.basis_element(0).unwrap(),
            ambient.basis_element(3).unwrap(),
        ],
    )
    .unwrap();

    assert!(submodule.generator_map().is_sparse());
    assert_eq!(submodule.generator_map().sparse_matrix().unwrap().nnz(), 2);
}

#[test]
fn module_groebner_results_certify_lifts_and_kernels() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let y = ring.generator(1).unwrap();
    let source = FreeModule::new(ring.clone(), 3);
    let target = FreeModule::new(ring.clone(), 1);
    let map = ModuleHomomorphism::from_matrix(
        source.clone(),
        target,
        Matrix::from_rows(ring.clone(), [[x.clone(), y.clone(), ring.zero()]]).unwrap(),
    )
    .unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    let image = Submodule::from_generator_map(map.target().clone(), map.clone()).unwrap();
    let basis = image.groebner_basis(order.clone()).unwrap();
    basis.verify().unwrap();
    assert_eq!(
        ModuleGroebnerBasis::<PolynomialRing<RationalField>>::from_json(&basis.to_json().unwrap())
            .unwrap(),
        basis
    );

    let value = map.apply(&source.basis_element(0).unwrap()).unwrap();
    let reduction = basis.reduce(&value).unwrap();
    assert!(reduction.is_member());
    assert_eq!(
        ModuleReduction::<PolynomialRing<RationalField>>::from_json(&reduction.to_json().unwrap())
            .unwrap(),
        reduction
    );
    assert_eq!(map.apply(reduction.lift().unwrap()).unwrap(), value);

    let kernel = map.kernel(order.clone()).unwrap();
    let expected = ModuleElement::from_coordinates(
        source.clone(),
        Vector::from_values(ring.clone(), [y, ring.neg(&x).unwrap(), ring.zero()]).unwrap(),
    )
    .unwrap();
    assert!(kernel
        .groebner_basis(order)
        .unwrap()
        .contains(&expected)
        .unwrap());
    assert!(basis.schreyer_order().schreyer_data().is_some());
}

#[test]
fn strong_integral_module_results_preserve_coefficient_divisibility() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let monomial = vec![Integer::one()];
    let two_x = ring
        .from_terms([(monomial.clone(), Integer::from(2))])
        .unwrap();
    let three_x = ring
        .from_terms([(monomial.clone(), Integer::from(3))])
        .unwrap();
    let five_x = ring.from_terms([(monomial, Integer::from(5))]).unwrap();
    let ambient = FreeModule::new(ring.clone(), 2);
    let generators = vec![
        ModuleElement::from_coordinates(
            ambient.clone(),
            Vector::from_values(ring.clone(), [two_x.clone(), ring.zero()]).unwrap(),
        )
        .unwrap(),
        ModuleElement::from_coordinates(
            ambient.clone(),
            Vector::from_values(ring.clone(), [three_x, ring.zero()]).unwrap(),
        )
        .unwrap(),
        ModuleElement::from_coordinates(
            ambient.clone(),
            Vector::from_values(ring.clone(), [ring.zero(), five_x]).unwrap(),
        )
        .unwrap(),
    ];
    let input = Submodule::from_generators(ambient, generators.clone()).unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

    let basis = input.strong_groebner_basis(order.clone()).unwrap();
    basis.verify().unwrap();
    assert_eq!(basis.leading_terms()[0].coefficient(), &Integer::one());
    assert_eq!(basis.leading_terms()[1].coefficient(), &Integer::from(5));
    assert_eq!(
        StrongModuleGroebnerBasis::from_json(&basis.to_json().unwrap()).unwrap(),
        basis
    );

    let reduction = basis.reduce(&generators[0]).unwrap();
    reduction.verify().unwrap();
    assert!(reduction.is_member());
    assert_eq!(
        StrongModuleReduction::from_json(&reduction.to_json().unwrap()).unwrap(),
        reduction
    );
    assert_eq!(
        input
            .generator_map()
            .apply(reduction.lift().unwrap())
            .unwrap(),
        generators[0]
    );

    let relations = input.generator_map().kernel(order).unwrap();
    for column in 0..relations.generator_count() {
        let relation = relations
            .generator_map()
            .apply(
                &relations
                    .generator_map()
                    .source()
                    .basis_element(column)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.generator_map().apply(&relation).unwrap(),
            ModuleElement::zero(input.ambient().clone())
        );
    }
}

#[test]
fn strong_integral_modules_retain_an_arbitrary_exponent_floor() {
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let exponent = Integer::from(u64::from(u32::MAX) + 1);
    let ambient = FreeModule::new(ring.clone(), 1);
    let generators = [2_i64, 3]
        .into_iter()
        .map(|coefficient| {
            let polynomial = ring
                .from_terms([(vec![exponent.clone()], Integer::from(coefficient))])
                .unwrap();
            ModuleElement::from_coordinates(
                ambient.clone(),
                Vector::from_values(ring.clone(), [polynomial]).unwrap(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let input = Submodule::from_generators(ambient, generators.clone()).unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

    let basis = input.strong_groebner_basis(order.clone()).unwrap();
    basis.verify().unwrap();
    assert_eq!(basis.leading_terms()[0].monomial().exponent(), [exponent]);
    assert_eq!(basis.leading_terms()[0].coefficient(), &Integer::one());
    assert_eq!(
        StrongModuleGroebnerBasis::from_json(&basis.to_json().unwrap()).unwrap(),
        basis
    );
    assert!(basis.reduce(&generators[0]).unwrap().is_member());

    let relations = input.generator_map().kernel(order).unwrap();
    assert!(relations.generator_count() > 0);
    for column in 0..relations.generator_count() {
        let relation = relations
            .generator_map()
            .apply(
                &relations
                    .generator_map()
                    .source()
                    .basis_element(column)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input.generator_map().apply(&relation).unwrap(),
            ModuleElement::zero(input.ambient().clone())
        );
    }
}

#[test]
fn minimal_homogeneous_generators_use_the_generic_extension_field_floor() {
    let field = FiniteField::extension(3_u64, 2).unwrap();
    let ring =
        PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let y = ring.generator(1).unwrap();
    let ambient = FreeModule::new(ring.clone(), 1)
        .with_degree_shifts(1, vec![Degree::zero(1)])
        .unwrap();
    let source = FreeModule::new(ring.clone(), 4);
    let map = ModuleHomomorphism::from_matrix(
        source,
        ambient.clone(),
        Matrix::from_rows(
            ring,
            [[x.clone(), y.clone(), (&x + &y).unwrap(), x.pow(2).unwrap()]],
        )
        .unwrap(),
    )
    .unwrap();
    let input = Submodule::from_generator_map(ambient, map).unwrap();

    let minimal = input.minimal_homogeneous_generators().unwrap();

    assert_eq!(minimal.generator_count(), 2);
    assert_eq!(
        minimal.generator_map().matrix().rows().next().unwrap(),
        vec![x, y]
    );
}
