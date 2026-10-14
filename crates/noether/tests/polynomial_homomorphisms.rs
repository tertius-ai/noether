// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    MonomialOrder, PolynomialConstantEmbedding, PolynomialHomomorphism, PolynomialRing,
};
use noether::numthy::FiniteField;
use noether::{
    CompositeCoefficientMap, IdentityCoefficientMap, Integer, IntegerFiniteFieldMap,
    IntegerRationalEmbedding, Ring, Q, Z,
};

type IntegerFinitePolynomialHomomorphism = PolynomialHomomorphism<
    noether::IntegerRing,
    PolynomialRing<FiniteField>,
    CompositeCoefficientMap<IntegerFiniteFieldMap, PolynomialConstantEmbedding<FiniteField>>,
>;

#[test]
fn arbitrary_substitution_and_base_change_are_one_checked_map() {
    let source = PolynomialRing::new(Z, ["x", "y", "z"], MonomialOrder::Lexicographic).unwrap();
    let field = FiniteField::prime(5_u8).unwrap();
    let target = PolynomialRing::new(
        field.clone(),
        ["a", "b", "t"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let [a, b, t] = target.generators().unwrap().try_into().unwrap();
    let line = target.add(&a, &target.mul(&b, &t).unwrap()).unwrap();
    let homomorphism = source
        .homomorphism(
            &target,
            Z.finite_field_map(&field),
            vec![target.one(), t.clone(), line.clone()],
        )
        .unwrap();

    let value = source
        .from_u64_terms([
            (vec![1, 1, 0], Integer::from(2)),
            (vec![0, 0, 2], Integer::from(3)),
        ])
        .unwrap();
    let expected = target
        .add(
            &t.scalar_mul(&field.element(2)).unwrap(),
            &target
                .mul(&line, &line)
                .unwrap()
                .scalar_mul(&field.element(3))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(homomorphism.apply(&value).unwrap(), expected);

    let json = homomorphism.to_json().unwrap();
    let restored = IntegerFinitePolynomialHomomorphism::from_json(&json).unwrap();
    assert_eq!(restored, homomorphism);
    restored.verify().unwrap();

    let wrong_target = PolynomialRing::new(
        FiniteField::prime(7_u8).unwrap(),
        ["a", "b", "t"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    assert!(source
        .homomorphism(
            &target,
            Z.finite_field_map(&field),
            vec![wrong_target.one(), wrong_target.one(), wrong_target.one()],
        )
        .is_err());
}

#[test]
fn coefficient_targets_and_composition_preserve_ring_map_laws() {
    let source = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
    let value = source
        .from_u64_terms([(vec![2], Integer::from(3)), (vec![0], Integer::from(-1))])
        .unwrap();
    let evaluation = PolynomialHomomorphism::new(
        source.clone(),
        Q,
        IntegerRationalEmbedding::new(),
        vec![Q.element(2)],
    )
    .unwrap();
    assert_eq!(evaluation.apply(&value).unwrap(), Q.element(11));

    let middle = PolynomialRing::new(Q, ["u"], MonomialOrder::Lexicographic).unwrap();
    let u = middle.generator(0).unwrap();
    let inner = source
        .homomorphism(&middle, IntegerRationalEmbedding::new(), vec![u])
        .unwrap();
    let target = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic).unwrap();
    let t_plus_one = target
        .add(&target.generator(0).unwrap(), &target.one())
        .unwrap();
    let outer = middle
        .homomorphism(&target, IdentityCoefficientMap::new(Q), vec![t_plus_one])
        .unwrap();
    let composite = outer.compose(&inner).unwrap();
    assert_eq!(
        composite.apply(&value).unwrap(),
        outer.apply(&inner.apply(&value).unwrap()).unwrap()
    );
}
