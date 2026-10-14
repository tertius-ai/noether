// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Smoke tests for construction, accessors, and the per-cell
//! `set` / `get` / `flip` round-trip. Matmul correctness lives in
//! `tests/matmul.rs`; algebraic invariants in `tests/algebraic.rs`.

use mzs::Mzs;

#[test]
fn new_round_trips_shape() {
    let m = Mzs::new(4, 7);
    assert_eq!(m.rows(), 4);
    assert_eq!(m.cols(), 7);
    assert_eq!(m.nnz(), 0);
    assert_eq!(m.density(), 0.0);
}

#[test]
fn empty_shapes_are_legal() {
    for &(r, c) in &[(0usize, 0usize), (0, 5), (5, 0)] {
        let m = Mzs::new(r, c);
        assert_eq!(m.rows(), r);
        assert_eq!(m.cols(), c);
        assert_eq!(m.nnz(), 0);
    }
}

#[test]
fn set_get_round_trip() {
    let mut m = Mzs::new(3, 5);
    m.set(0, 0, 1);
    m.set(1, 4, 1);
    m.set(2, 2, 1);
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(0, 1), 0);
    assert_eq!(m.get(1, 4), 1);
    assert_eq!(m.get(1, 3), 0);
    assert_eq!(m.get(2, 2), 1);
    assert_eq!(m.nnz(), 3);
}

#[test]
fn set_zero_removes_entry() {
    let mut m = Mzs::new(2, 2);
    m.set(0, 0, 1);
    m.set(1, 1, 1);
    assert_eq!(m.nnz(), 2);
    m.set(0, 0, 0);
    assert_eq!(m.get(0, 0), 0);
    assert_eq!(m.nnz(), 1);
}

#[test]
fn set_masks_high_bits_of_value() {
    let mut m = Mzs::new(1, 1);
    m.set(0, 0, 0xFE); // even — should be treated as 0
    assert_eq!(m.get(0, 0), 0);
    m.set(0, 0, 0xFF); // odd — treated as 1
    assert_eq!(m.get(0, 0), 1);
}

#[test]
fn flip_is_involution() {
    let mut m = Mzs::new(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let before = m.get(i, j);
            m.flip(i, j);
            m.flip(i, j);
            assert_eq!(m.get(i, j), before);
        }
    }
}

#[test]
fn from_triples_basic() {
    let m = Mzs::from_triples(2, 3, &[(0, 0), (0, 2), (1, 1)]);
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(0, 1), 0);
    assert_eq!(m.get(0, 2), 1);
    assert_eq!(m.get(1, 1), 1);
    assert_eq!(m.nnz(), 3);
}

#[test]
fn from_triples_duplicates_xor() {
    let m = Mzs::from_triples(1, 4, &[(0, 0), (0, 0), (0, 1), (0, 2), (0, 2), (0, 2)]);
    // (0,0): even count → 0. (0,1): once → 1. (0,2): three → 1.
    assert_eq!(m.get(0, 0), 0);
    assert_eq!(m.get(0, 1), 1);
    assert_eq!(m.get(0, 2), 1);
    assert_eq!(m.nnz(), 2);
}

#[test]
fn row_indices_are_sorted() {
    let mut m = Mzs::new(1, 100);
    for j in [99, 1, 50, 27, 0, 73] {
        m.set(0, j, 1);
    }
    let row = m.row_indices(0);
    let mut expected = vec![0u32, 1, 27, 50, 73, 99];
    expected.sort();
    assert_eq!(row, expected.as_slice());
}

#[test]
fn nnz_and_density() {
    let mut m = Mzs::new(4, 4);
    for i in 0..4 {
        m.set(i, i, 1);
    }
    assert_eq!(m.nnz(), 4);
    assert_eq!(m.density(), 0.25);
}

#[test]
fn clone_preserves_content() {
    let a = Mzs::from_triples(3, 3, &[(0, 1), (1, 2), (2, 0)]);
    let b = a.clone();
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(a.get(i, j), b.get(i, j));
        }
    }
    assert_eq!(a, b);
}

#[test]
fn add_is_symmetric_difference_and_scalar_mul_stays_sparse() {
    let a = Mzs::from_triples(2, 4, &[(0, 0), (0, 2), (1, 3)]);
    let b = Mzs::from_triples(2, 4, &[(0, 1), (0, 2), (1, 0)]);
    let sum = a.add(&b);
    assert_eq!(
        sum,
        Mzs::from_triples(2, 4, &[(0, 0), (0, 1), (1, 0), (1, 3)])
    );
    assert_eq!(a.scalar_mul(1), a);
    assert_eq!(a.scalar_mul(0), Mzs::new(2, 4));
}
