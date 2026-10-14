// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::Matrix;
use noether::{Integer, Rational, Q, Z};

fn main() {
    let matrix = Matrix::from_rows(
        Z,
        [
            [Integer::from(2), Integer::from(4)],
            [Integer::from(6), Integer::from(8)],
        ],
    )
    .expect("the rows are rectangular");
    let result = matrix.smith_normal_form();
    assert_eq!(
        result.invariant_factors(),
        &[Integer::from(2), Integer::from(4)]
    );
    result.verify().expect("the Smith form verifies");

    let determinant = Matrix::from_rows(
        Q,
        [
            [Rational::from(1), Rational::from(2)],
            [Rational::from(3), Rational::from(5)],
        ],
    )
    .expect("the rows are rectangular")
    .det()
    .expect("the matrix is square");
    assert_eq!(determinant, Rational::from(-1));
}
