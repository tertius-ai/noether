// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::Matrix;
use noether::{
    ComplexBall, ComplexBallField, Integer, Rational, RealBall, RealBallField, Ring, SemanticType,
};

#[test]
fn real_balls_keep_their_precision_context() {
    let field = RealBallField::new(128).unwrap();
    let one_third = field.element(Rational::new(1.into(), 3.into()).unwrap());
    let sum = field.add(&one_third, &one_third).unwrap();

    assert_eq!(field.precision_bits(), 128);
    assert_eq!(sum.field(), field);
    assert!(!sum.is_exact());
    assert!(sum
        .overlaps(&field.element(Rational::new(2.into(), 3.into()).unwrap()))
        .unwrap());
    assert!(RealBallField::new(1).is_err());

    let other = RealBallField::new(256).unwrap().element(1);
    assert!(field.add(&one_third, &other).is_err());
}

#[test]
fn complex_balls_expose_certified_components() {
    let field = ComplexBallField::new(192).unwrap();
    let imaginary_unit = field.element_parts(0, 1);
    let minus_one = field.element(-1);
    let square = imaginary_unit.pow(2);

    assert!(square.overlaps(&minus_one).unwrap());
    assert!(imaginary_unit.real().is_zero());
    assert!(imaginary_unit.imaginary().is_one());
    assert_eq!(
        imaginary_unit.real().field(),
        RealBallField::new(192).unwrap()
    );
}

#[test]
fn exact_dump_json_round_trips_and_revalidates() {
    let real = RealBallField::new(160)
        .unwrap()
        .element(Rational::new(22.into(), 7.into()).unwrap());
    assert_eq!(RealBall::from_json(&real.to_json().unwrap()).unwrap(), real);
    assert_eq!(RealBall::TYPE_ID, "noether/root/RealBall@1");

    let complex = ComplexBallField::new(160).unwrap().element_parts(
        Rational::new(22.into(), 7.into()).unwrap(),
        Rational::new(Integer::from(-5), Integer::from(11)).unwrap(),
    );
    assert_eq!(
        ComplexBall::from_json(&complex.to_json().unwrap()).unwrap(),
        complex
    );
    assert_eq!(ComplexBall::TYPE_ID, "noether/root/ComplexBall@1");

    assert!(RealBall::from_json(
        r#"{"type":"noether/root/RealBall@1","precision":1,"value":"0 0"}"#
    )
    .is_err());
    assert!(
        ComplexBall::from_json(
            r#"{"type":"noether/root/ComplexBall@1","precision":128,"real":"not-arb","imaginary":"0 0"}"#
        )
        .is_err()
    );
    assert!(RealBall::from_json(
        r#"{"type":"noether/root/ComplexBall@1","precision":128,"value":"0 0"}"#
    )
    .is_err());
}

#[test]
fn matrices_exist_over_ball_contexts_without_new_matrix_types() {
    let real = RealBallField::new(96).unwrap();
    let matrix = Matrix::from_rows(real, [vec![real.element(1), real.element(2)]]).unwrap();
    assert_eq!(matrix.shape(), (1, 2));
    assert_eq!(
        Matrix::<RealBallField>::from_json(&matrix.to_json().unwrap()).unwrap(),
        matrix
    );

    let complex = ComplexBallField::new(96).unwrap();
    let matrix = Matrix::identity(complex, 2);
    assert_eq!(matrix.get(0, 0).unwrap(), complex.one());
    assert_eq!(
        Matrix::<ComplexBallField>::from_json(&matrix.to_json().unwrap()).unwrap(),
        matrix
    );
}
