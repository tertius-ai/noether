// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::linalg::{LinalgError, LinearOperator, Matrix, SolutionSpace, Vector};
use noether::numthy::{FiniteField, NumberField};
use noether::{Rational, Ring, SemanticType, Q};

fn rational(value: i64) -> Rational {
    Rational::from(value)
}

#[test]
fn rational_elimination_is_exact_and_hand_checkable() {
    let matrix = Matrix::from_rows(
        Q,
        [
            [rational(1), rational(2), rational(3)],
            [rational(2), rational(4), rational(6)],
        ],
    )
    .unwrap();

    assert_eq!(matrix.rank().unwrap(), 1);
    assert_eq!(
        matrix.rref().unwrap().rows().collect::<Vec<_>>(),
        vec![
            vec![rational(1), rational(2), rational(3)],
            vec![rational(0), rational(0), rational(0)],
        ]
    );

    let rhs = Vector::from_values(Q, [rational(1), rational(2)]).unwrap();
    let solutions = matrix.solve(&rhs).unwrap();
    assert_eq!(
        solutions.particular().values().collect::<Vec<_>>(),
        vec![rational(1), rational(0), rational(0),]
    );
    assert_eq!(solutions.dimension(), 2);
    assert_eq!(solutions.ambient_dimension(), 3);
    assert_eq!(
        solutions.basis()[0].values().collect::<Vec<_>>(),
        vec![rational(-2), rational(1), rational(0),]
    );
    assert_eq!(
        solutions.basis()[1].values().collect::<Vec<_>>(),
        vec![rational(-3), rational(0), rational(1),]
    );
    solutions.verify().unwrap();

    let square =
        Matrix::from_rows(Q, [[rational(1), rational(2)], [rational(3), rational(4)]]).unwrap();
    assert_eq!(square.det().unwrap(), rational(-2));
    assert!(matches!(
        matrix.det(),
        Err(LinalgError::NotSquare {
            operation: "determinant",
            shape: (2, 3),
        })
    ));
}

#[test]
fn kernel_and_image_are_complete_checked_subspaces() {
    let matrix = Matrix::from_rows(
        Q,
        [
            [rational(1), rational(2), rational(3)],
            [rational(2), rational(4), rational(6)],
        ],
    )
    .unwrap();

    let kernel = matrix.kernel().unwrap();
    let image = matrix.image().unwrap();
    assert_eq!(kernel.dimension(), 2);
    assert_eq!(kernel.ambient_dimension(), 3);
    assert_eq!(image.dimension(), 1);
    assert_eq!(image.ambient_dimension(), 2);
    assert_eq!(
        image.basis()[0].values().collect::<Vec<_>>(),
        vec![rational(1), rational(2)]
    );
    for direction in kernel.basis() {
        assert_eq!(
            matrix.apply(direction).unwrap(),
            Vector::zeros(Q, matrix.shape().0)
        );
    }
    kernel.verify().unwrap();
    image.verify().unwrap();
    assert_eq!(
        SolutionSpace::<noether::RationalField>::TYPE_ID,
        "noether/linalg/SolutionSpace@1"
    );
}

#[test]
fn inconsistent_and_incompatible_systems_fail_closed() {
    let matrix =
        Matrix::from_rows(Q, [[rational(1), rational(1)], [rational(1), rational(1)]]).unwrap();
    let inconsistent = Vector::from_values(Q, [rational(0), rational(1)]).unwrap();
    assert!(matches!(
        matrix.solve(&inconsistent),
        Err(LinalgError::InconsistentSystem)
    ));

    let seven = FiniteField::prime(7_u8).unwrap();
    let eleven = FiniteField::prime(11_u8).unwrap();
    let over_seven = Matrix::identity(seven.clone(), 1);
    let over_eleven = Vector::from_values(eleven.clone(), [eleven.one()]).unwrap();
    assert!(matches!(
        over_seven.solve(&over_eleven),
        Err(LinalgError::ContextMismatch(_))
    ));
}

#[test]
fn the_same_surface_works_over_f9() {
    let field = FiniteField::extension(3_u64, 2).unwrap();
    let generator = field.generator();
    let matrix = Matrix::from_rows(
        field.clone(),
        [
            [generator.clone(), field.one()],
            [field.zero(), generator.clone()],
        ],
    )
    .unwrap();
    let rhs = Vector::from_values(field.clone(), [field.one(), field.one()]).unwrap();

    assert_eq!(matrix.rank().unwrap(), 2);
    assert_eq!(matrix.det().unwrap(), generator.pow(2));
    let solution = matrix.solve(&rhs).unwrap();
    assert_eq!(solution.dimension(), 0);
    assert_eq!(matrix.apply(solution.particular()).unwrap(), rhs);
    solution.verify().unwrap();
}

#[test]
fn packed_binary_fields_solve_and_find_kernels_without_changing_the_surface() {
    for field in [
        FiniteField::prime(2_u8).unwrap(),
        FiniteField::with_modulus(2_u64, vec![1, 1, 1]).unwrap(),
    ] {
        let one = field.one();
        let generator = field.generator();
        let matrix = Matrix::from_rows(
            field.clone(),
            [
                [one.clone(), generator.clone(), field.zero()],
                [field.zero(), one.clone(), generator.clone()],
            ],
        )
        .unwrap();
        let rhs = Vector::from_values(field.clone(), [one, generator]).unwrap();

        let solutions = matrix.solve(&rhs).unwrap();
        assert_eq!(solutions.dimension(), 1);
        assert_eq!(matrix.apply(solutions.particular()).unwrap(), rhs);
        solutions.verify().unwrap();

        let kernel = matrix.kernel().unwrap();
        assert_eq!(kernel.dimension(), 1);
        assert_eq!(
            matrix.apply(&kernel.basis()[0]).unwrap(),
            Vector::zeros(field, 2)
        );
        kernel.verify().unwrap();

        let factors = matrix.pluq().unwrap();
        assert_eq!(factors.rank(), 2);
        factors.verify().unwrap();
    }
}

#[test]
fn the_same_surface_works_over_a_number_field() {
    let polynomial_ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let defining_polynomial = polynomial_ring
        .from_u64_terms([(vec![0], rational(1)), (vec![2], rational(1))])
        .unwrap();
    let field = NumberField::new(defining_polynomial, Some("i".to_owned())).unwrap();
    let i = field.generator();
    let matrix = Matrix::from_rows(
        field.clone(),
        [[field.one(), i.clone()], [field.zero(), field.one()]],
    )
    .unwrap();
    let rhs = Vector::from_values(field.clone(), [field.zero(), i]).unwrap();

    assert_eq!(matrix.rank().unwrap(), 2);
    assert_eq!(matrix.det().unwrap(), field.one());
    let solution = matrix.solve(&rhs).unwrap();
    assert_eq!(matrix.apply(solution.particular()).unwrap(), rhs);
    solution.verify().unwrap();
}
