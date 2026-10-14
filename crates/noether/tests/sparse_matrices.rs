// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::{LinalgError, LinearOperator, Matrix, SparseMatrix, Vector};
use noether::numthy::FiniteField;
use noether::{Integer, Ring, SemanticType, Z};
use spar_core::SparseRingStorage;

fn integer(value: i64) -> Integer {
    Integer::from(value)
}

#[test]
fn triplets_canonicalize_duplicates_and_zeros() {
    let matrix = SparseMatrix::from_triplets(
        Z,
        3,
        4,
        [
            (2, 3, integer(7)),
            (0, 1, integer(5)),
            (0, 1, integer(-2)),
            (1, 2, integer(4)),
            (1, 2, integer(-4)),
            (0, 0, integer(0)),
        ],
    )
    .unwrap();

    assert_eq!(matrix.shape(), (3, 4));
    assert_eq!(matrix.nnz(), 2);
    assert_eq!(matrix.row_offsets(), [0, 1, 1, 2]);
    assert_eq!(matrix.column_indices(), [1, 3]);
    assert_eq!(matrix.values(), [integer(3), integer(7)]);
    assert_eq!(matrix.get(1, 1).unwrap(), integer(0));
    assert_eq!(SparseRingStorage::nnz(&matrix), 2);
    assert_eq!(matrix.density(), 2.0 / 12.0);
}

#[test]
fn sparse_arithmetic_and_linear_operator_match_dense_results() {
    let left = SparseMatrix::from_rows(
        Z,
        [
            [integer(1), integer(0), integer(2)],
            [integer(0), integer(3), integer(0)],
        ],
    )
    .unwrap();
    let right = SparseMatrix::from_rows(
        Z,
        [
            [integer(0), integer(4)],
            [integer(5), integer(0)],
            [integer(0), integer(6)],
        ],
    )
    .unwrap();
    let product = left.matmul(&right).unwrap();
    assert_eq!(
        product.to_dense(),
        Matrix::from_rows(Z, [[integer(0), integer(16)], [integer(15), integer(0)]]).unwrap()
    );

    let vector = Vector::from_values(Z, [integer(2), integer(3), integer(5)]).unwrap();
    assert_eq!(
        left.apply(&vector).unwrap().values().collect::<Vec<_>>(),
        [integer(12), integer(9)]
    );
    assert_eq!(
        left.apply_block(&right.to_dense()).unwrap(),
        product.to_dense()
    );
    assert_eq!(left.transpose().transpose(), left);
    assert_eq!(
        (&left + &left).unwrap(),
        left.scalar_mul(&integer(2)).unwrap()
    );
    assert_eq!((&left - &left).unwrap().nnz(), 0);
}

#[test]
fn csr_and_json_are_revalidated() {
    let matrix = SparseMatrix::from_csr(
        Z,
        2,
        3,
        vec![0, 1, 2],
        vec![0, 2],
        vec![integer(2), integer(5)],
    )
    .unwrap();
    let json = matrix.to_json().unwrap();
    let restored = SparseMatrix::from_json(&json).unwrap();
    assert_eq!(restored, matrix);
    assert_eq!(
        restored.canonical_bytes().unwrap(),
        matrix.canonical_bytes().unwrap()
    );

    assert!(matches!(
        SparseMatrix::from_csr(
            Z,
            1,
            3,
            vec![0, 2],
            vec![2, 1],
            vec![integer(1), integer(1)]
        ),
        Err(LinalgError::InvalidSparseStructure { .. })
    ));
    assert!(matches!(
        SparseMatrix::from_triplets(Z, 1, 1, [(1, 0, integer(1))]),
        Err(LinalgError::SparseEntryOutsideShape { .. })
    ));
}

#[test]
fn sparse_binary_matmul_matches_dense_finite_field_arithmetic() {
    let field = FiniteField::prime(2_u8).unwrap();
    let left = SparseMatrix::from_triplets(
        field.clone(),
        4,
        5,
        [(0, 0), (0, 3), (1, 1), (2, 2), (2, 4), (3, 0)]
            .map(|(row, column)| (row, column, field.one())),
    )
    .unwrap();
    let right = SparseMatrix::from_triplets(
        field.clone(),
        5,
        3,
        [(0, 1), (1, 0), (2, 2), (3, 1), (4, 0), (4, 2)]
            .map(|(row, column)| (row, column, field.one())),
    )
    .unwrap();

    assert_eq!(
        left.matmul(&right).unwrap().to_dense(),
        left.to_dense().matmul(&right.to_dense()).unwrap()
    );
}
