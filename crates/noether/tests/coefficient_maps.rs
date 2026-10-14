// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{RationalFunctionField, RationalFunctionFieldMap, ResidueRing};
use noether::numthy::FiniteField;
use noether::{
    CoefficientMap, Integer, IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction,
    PrimeSubfieldEmbedding, PrimeSubfieldEmbeddingError, Q, Z,
};

#[test]
fn standard_scalar_maps_are_checked_public_values() {
    let rational = Z.rational_embedding();
    assert_eq!(rational.apply(&Integer::from(-9)).unwrap(), Q.element(-9));
    assert_eq!(
        IntegerRationalEmbedding::from_json(&rational.to_json().unwrap()).unwrap(),
        rational
    );

    let residues = ResidueRing::new(12_u8).unwrap();
    let reduction = Z.reduction(&residues);
    assert_eq!(
        reduction.apply(&Integer::from(-9)).unwrap(),
        residues.element(3)
    );
    assert_eq!(
        IntegerReduction::from_json(&reduction.to_json().unwrap()).unwrap(),
        reduction
    );

    let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let finite = Z.finite_field_map(&extension);
    assert_eq!(
        finite.apply(&Integer::from(12)).unwrap(),
        extension.element(2)
    );
    assert_eq!(
        IntegerFiniteFieldMap::from_json(&finite.to_json().unwrap()).unwrap(),
        finite
    );

    let embedding = extension.prime_subfield_embedding();
    assert_eq!(
        embedding.apply(&embedding.source().element(4)).unwrap(),
        extension.element(4)
    );
    assert_eq!(
        PrimeSubfieldEmbedding::from_json(&embedding.to_json().unwrap()).unwrap(),
        embedding
    );

    let wrong = FiniteField::prime(7_u8).unwrap().element(4);
    assert!(matches!(
        embedding.apply(&wrong),
        Err(PrimeSubfieldEmbeddingError::ContextMismatch(_))
    ));
}

#[test]
fn prime_subfield_maps_induce_checked_rational_function_field_maps() {
    let target_coefficients = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let coefficient_map = target_coefficients.prime_subfield_embedding();
    let source = RationalFunctionField::new(coefficient_map.source().clone(), "t").unwrap();
    let target = RationalFunctionField::new(target_coefficients, "u").unwrap();
    let map =
        RationalFunctionFieldMap::new(source.clone(), target.clone(), coefficient_map.clone())
            .unwrap();
    let t = source.generator();
    let t_plus_two = (&t
        + &source
            .constant(source.coefficient_ring().element(2))
            .unwrap())
        .unwrap();
    let t_plus_one = (&t
        + &source
            .constant(source.coefficient_ring().element(1))
            .unwrap())
        .unwrap();
    let value = (&t_plus_two / &t_plus_one).unwrap();
    let u = target.generator();
    let u_plus_two = (&u
        + &target
            .constant(target.coefficient_ring().element(2))
            .unwrap())
        .unwrap();
    let u_plus_one = (&u
        + &target
            .constant(target.coefficient_ring().element(1))
            .unwrap())
        .unwrap();
    let expected = (&u_plus_two / &u_plus_one).unwrap();

    assert_eq!(map.apply(&value).unwrap(), expected);
    map.verify().unwrap();
    assert_eq!(
        RationalFunctionFieldMap::from_json(&map.to_json().unwrap()).unwrap(),
        map
    );

    let wrong_source = RationalFunctionField::new(coefficient_map.source().clone(), "v").unwrap();
    assert!(map.apply(&wrong_source.generator()).is_err());
}
