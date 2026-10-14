// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Correctness of basic arithmetic on Mat<T: Z2n>.

use pwvla::mat::SparseCsrMat;
use pwvla::{scalar, Mat};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn random_u16(rows: usize, cols: usize, seed: u64) -> Mat<u16> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u16>::from_fn(rows, cols, |_, _| rng.random::<u16>())
}

fn random_u32(rows: usize, cols: usize, seed: u64) -> Mat<u32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<u32>::from_fn(rows, cols, |_, _| rng.random::<u32>())
}

#[test]
fn add_matches_scalar_reference() {
    let a = random_u16(17, 23, 1);
    let b = random_u16(17, 23, 2);
    let fast = a.add(&b);
    // Scalar reference: element-wise wrapping_add.
    let expect: Vec<u16> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    assert_eq!(fast.data, expect);
}

#[test]
fn sub_is_add_neg() {
    let a = random_u16(40, 50, 3);
    let b = random_u16(40, 50, 4);
    // a - b == a + (-b)
    let direct = a.sub(&b);
    let indirect = a.add(&b.neg());
    assert_eq!(direct, indirect);
}

#[test]
fn neg_is_self_inverse() {
    let a = random_u16(10, 10, 5);
    assert_eq!(a.neg().neg(), a);
}

#[test]
fn add_neg_is_zero() {
    let a = random_u16(30, 30, 6);
    let z = a.add(&a.neg());
    assert!(z.data.iter().all(|&x| x == 0));
}

#[test]
fn scalar_mul_by_zero_is_zero() {
    let a = random_u32(15, 25, 7);
    let z = a.scalar_mul(0u32);
    assert!(z.data.iter().all(|&x| x == 0));
}

#[test]
fn scalar_mul_by_one_is_identity() {
    let a = random_u32(15, 25, 8);
    assert_eq!(a.scalar_mul(1u32), a);
}

#[test]
fn scalar_mul_distributes_over_add() {
    let a = random_u16(20, 20, 9);
    let b = random_u16(20, 20, 10);
    let s: u16 = 12345;
    let lhs = a.add(&b).scalar_mul(s);
    let rhs = a.scalar_mul(s).add(&b.scalar_mul(s));
    assert_eq!(lhs, rhs);
}

#[test]
fn transpose_involution() {
    for (r, c) in [(1, 1), (4, 4), (17, 23), (64, 33), (128, 200)] {
        let a = random_u16(r, c, (r * 100 + c) as u64);
        let t = a.transpose();
        assert_eq!((t.rows, t.cols), (c, r));
        assert_eq!(t.transpose(), a);
    }
}

#[test]
fn transpose_matches_manual() {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let a = Mat::<u16> {
        rows: 2,
        cols: 3,
        data: vec![1, 2, 3, 4, 5, 6],
    };
    let t = a.transpose();
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.data, vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn transpose_of_matmul_equals_reversed_matmul_of_transposes() {
    // (AB)^T = B^T · A^T
    let a = random_u16(16, 24, 20);
    let b = random_u16(24, 32, 21);
    let lhs = scalar::matmul(&a, &b).transpose();
    let rhs = scalar::matmul(&b.transpose(), &a.transpose());
    assert_eq!(lhs, rhs);
}

#[test]
fn zero_extent_matrices_are_closed_under_basic_arithmetic() {
    for (rows, cols) in [(0, 0), (3, 0), (0, 4)] {
        let matrix = Mat::<u16>::zeros(rows, cols);
        assert_eq!(matrix.add(&matrix), matrix);
        assert_eq!(matrix.sub(&matrix), matrix);
        assert_eq!(matrix.neg(), matrix);
        assert_eq!(matrix.scalar_mul(7), matrix);
        assert_eq!(matrix.reduce_mod_2(), matrix);

        let transposed = matrix.transpose();
        assert_eq!((transposed.rows, transposed.cols), (cols, rows));
        assert_eq!(transposed.transpose(), matrix);
    }
}

#[test]
fn zero_width_products_keep_their_mathematical_shape() {
    let left = Mat::<u16>::zeros(3, 0);
    let right = Mat::<u16>::zeros(0, 4);
    assert_eq!(scalar::matmul(&left, &right), Mat::<u16>::zeros(3, 4));

    let left = Mat::<u16>::zeros(2, 3);
    let right = Mat::<u16>::zeros(3, 0);
    assert_eq!(scalar::matmul(&left, &right), Mat::<u16>::zeros(2, 0));

    let sparse = SparseCsrMat::from_dense(&left);
    assert_eq!(sparse.matmul_dense(&right), Mat::<u16>::zeros(2, 0));

    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        pwvla::strassen::matmul_u16_with_crossover(
            &Mat::<u16>::zeros(3, 0),
            &Mat::<u16>::zeros(0, 4),
            0,
        ),
        Mat::<u16>::zeros(3, 4)
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[should_panic(expected = "Strassen crossover must be positive")]
fn strassen_rejects_zero_crossover_for_nonempty_inputs() {
    let matrix = Mat::<u16>::zeros(1, 1);
    let _ = pwvla::strassen::matmul_u16_with_crossover(&matrix, &matrix, 0);
}
