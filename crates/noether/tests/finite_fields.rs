// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::{LinearOperator, Matrix, Vector};
use noether::numthy::{FiniteField, FiniteFieldElement, FiniteFieldIsomorphism};
use noether::{Field, FiniteContext, Integer, Ring, SemanticType};

fn assert_field<F: Field>(_field: &F) {}

#[test]
fn prime_fields_verify_the_field_laws_they_claim() {
    let field = FiniteField::prime(7_u8).unwrap();
    let three = field.element(3_u8);
    let five = field.element(5_u8);

    assert_field(&field);
    assert_eq!(field.characteristic(), &Integer::from(7));
    assert_eq!(field.degree(), 1);
    assert_eq!(field.order(), Integer::from(7));
    assert_eq!(
        field.add(&three, &five).unwrap().representative(),
        &Integer::from(1)
    );
    assert_eq!(
        field.mul(&three, &five).unwrap().representative(),
        &Integer::from(1)
    );
    assert_eq!(three.inverse().unwrap().representative(), &Integer::from(5));
    assert_eq!(three.pow(6), field.one());
    assert_eq!(three.frobenius(100), three);
    assert_eq!(three.trace(), three);
    assert_eq!(three.norm(), three);
    assert!(field.zero().inverse().is_err());
}

#[test]
fn composite_characteristics_are_rejected_deterministically() {
    for composite in [0_u64, 1, 4, 15, 561, 1_105, 1_729] {
        assert!(FiniteField::prime(composite).is_err());
    }
    assert!(FiniteField::prime(-7_i8).is_err());
}

#[test]
fn arbitrary_size_prime_characteristics_are_supported() {
    let mersenne: Integer = "170141183460469231731687303715884105727".parse().unwrap();
    let field = FiniteField::prime(mersenne.clone()).unwrap();
    let minus_one = field.element(&mersenne - &Integer::one());

    assert_eq!(field.cardinality(), mersenne.clone());
    assert_eq!(field.mul(&minus_one, &minus_one).unwrap(), field.one());
    assert_eq!(minus_one.inverse().unwrap(), minus_one);
    let identity = field.isomorphism_to(&field).unwrap();
    assert_eq!(identity.apply(&minus_one).unwrap(), minus_one);
}

#[test]
fn matrices_and_vectors_exist_from_the_ring_contract() {
    let field = FiniteField::prime(7_u8).unwrap();
    let matrix = Matrix::from_rows(
        field.clone(),
        [
            [field.element(1_u8), field.element(2_u8)],
            [field.element(3_u8), field.element(4_u8)],
        ],
    )
    .unwrap();
    let vector =
        Vector::from_values(field.clone(), [field.element(5_u8), field.element(6_u8)]).unwrap();

    assert_eq!(
        matrix.matmul(&matrix).unwrap().get(0, 0).unwrap(),
        field.element(0_u8)
    );
    assert_eq!(
        matrix.apply(&vector).unwrap().get(0).unwrap(),
        field.element(3_u8)
    );
}

#[test]
fn contexts_iteration_and_semantic_identity_are_explicit() {
    let seven = FiniteField::prime(7_u8).unwrap();
    let eleven = FiniteField::prime(11_u8).unwrap();

    assert!((&seven.element(1_u8) + &eleven.element(1_u8)).is_err());
    assert!(matches!(
        seven.inverse(&eleven.element(1_u8)),
        Err(noether::InversionError::ContextMismatch(_))
    ));
    assert_eq!(seven.elements().count(), 7);
    assert_eq!(FiniteField::TYPE_ID, "noether/numthy/FiniteField@1");
    assert_eq!(
        FiniteFieldElement::TYPE_ID,
        "noether/numthy/FiniteFieldElement@1"
    );
    assert_eq!(
        FiniteFieldElement::describe().python_path,
        Some("noether.numthy.FiniteFieldElement")
    );
}

#[test]
fn conway_extensions_have_explicit_stable_presentations() {
    let field = FiniteField::extension(5, 2).unwrap();
    let generator = field.generator();

    assert_eq!(field.characteristic(), &Integer::from(5));
    assert_eq!(field.degree(), 2);
    assert_eq!(field.order(), Integer::from(25));
    assert_eq!(field.defining_modulus(), Some(&[2, 4, 1][..]));
    assert_eq!(
        generator.coefficients(),
        vec![Integer::zero(), Integer::one()]
    );

    // The Conway generator satisfies a^2 + 4a + 2 = 0.
    let relation = &(&generator.pow(2) + &(&generator * &field.element(4_u8)).unwrap()).unwrap()
        + &field.element(2_u8);
    assert_eq!(relation.unwrap(), field.zero());
    assert_eq!(generator.frobenius(2), generator);
    assert_ne!(generator.frobenius(1), generator);
    assert_eq!(generator.trace(), field.element(1_u8));
    assert_eq!(generator.norm(), field.element(2_u8));
    assert_eq!(
        field
            .mul(&generator, &generator.inverse().unwrap())
            .unwrap(),
        field.one()
    );
}

#[test]
fn explicit_moduli_are_verified_and_are_part_of_context_identity() {
    let explicit = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let conway = FiniteField::extension(5, 2).unwrap();
    let generator = explicit.generator();

    assert_ne!(explicit, conway);
    assert_eq!(explicit.defining_modulus(), Some(&[2, 0, 1][..]));
    assert_eq!(generator.pow(2), explicit.element(3_u8));
    assert!(FiniteField::with_modulus(5, vec![4, 0, 1]).is_err());
    assert!(FiniteField::with_modulus(5, vec![2, 0, 2]).is_err());
    assert!(FiniteField::with_modulus(5, vec![5, 0, 1]).is_err());
    assert!(FiniteField::extension(9, 2).is_err());
    assert!(FiniteField::extension(5, 0).is_err());
}

#[test]
fn distinct_presentations_have_a_checked_bidirectional_isomorphism() {
    let source = FiniteField::extension(5, 2).unwrap();
    let target = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let isomorphism = source.isomorphism_to(&target).unwrap();

    assert_ne!(source, target);
    assert_eq!(isomorphism.source(), &source);
    assert_eq!(isomorphism.target(), &target);
    isomorphism.verify().unwrap();

    for value in source.elements() {
        let image = isomorphism.apply(&value).unwrap();
        assert_eq!(isomorphism.apply_inverse(&image).unwrap(), value);
    }
    let a = source.element_from_coefficients(&[3, 2]).unwrap();
    let b = source.element_from_coefficients(&[4, 1]).unwrap();
    assert_eq!(
        isomorphism.apply(&source.add(&a, &b).unwrap()).unwrap(),
        target
            .add(
                &isomorphism.apply(&a).unwrap(),
                &isomorphism.apply(&b).unwrap(),
            )
            .unwrap()
    );
    assert_eq!(
        isomorphism.apply(&source.mul(&a, &b).unwrap()).unwrap(),
        target
            .mul(
                &isomorphism.apply(&a).unwrap(),
                &isomorphism.apply(&b).unwrap(),
            )
            .unwrap()
    );

    let encoded = isomorphism.to_json().unwrap();
    assert_eq!(
        FiniteFieldIsomorphism::from_json(&encoded).unwrap(),
        isomorphism
    );
    let mut tampered = serde_json::from_str::<serde_json::Value>(&encoded).unwrap();
    tampered["generator_image"]["coefficients"] = serde_json::json!(["0", "0"]);
    assert!(FiniteFieldIsomorphism::from_json(&tampered.to_string()).is_err());

    assert!(source
        .isomorphism_to(&FiniteField::extension(5, 3).unwrap())
        .is_err());
    assert!(isomorphism
        .apply(&FiniteField::prime(5_u8).unwrap().one())
        .is_err());
}

#[test]
fn extension_elements_iterate_by_power_basis_and_work_in_matrices() {
    let field = FiniteField::extension(2, 2).unwrap();
    let coefficients = field
        .elements()
        .map(|value| value.coefficients())
        .collect::<Vec<_>>();
    assert_eq!(
        coefficients,
        vec![
            vec![Integer::zero(), Integer::zero()],
            vec![Integer::one(), Integer::zero()],
            vec![Integer::zero(), Integer::one()],
            vec![Integer::one(), Integer::one()],
        ]
    );

    let generator = field.generator();
    let matrix = Matrix::from_rows(
        field.clone(),
        [
            [generator.clone(), field.one()],
            [field.one(), generator.clone()],
        ],
    )
    .unwrap();
    let vector = Vector::from_values(field.clone(), [generator, field.one()]).unwrap();
    assert_eq!(matrix.apply(&vector).unwrap().dimension(), 2);
    assert_eq!(matrix.matmul(&matrix).unwrap().shape(), (2, 2));
}

#[test]
fn typed_json_round_trips_and_revalidates_presentations() {
    let prime = FiniteField::prime(7_u8).unwrap();
    let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let element = extension.element_from_coefficients(&[4, 3]).unwrap();

    assert_eq!(
        FiniteField::from_json(&prime.to_json().unwrap()).unwrap(),
        prime
    );
    assert_eq!(
        FiniteField::from_json(&extension.to_json().unwrap()).unwrap(),
        extension
    );
    assert_eq!(
        FiniteFieldElement::from_json(&element.to_json().unwrap()).unwrap(),
        element
    );

    let reducible = extension.to_json().unwrap().replace("[2,0,1]", "[4,0,1]");
    assert!(FiniteField::from_json(&reducible).is_err());
    let wrong_type = element
        .to_json()
        .unwrap()
        .replace(FiniteFieldElement::TYPE_ID, "noether/numthy/Other@1");
    assert!(FiniteFieldElement::from_json(&wrong_type).is_err());
}
