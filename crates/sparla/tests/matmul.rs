// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrFp::matmul`] and [`CsrZ::matmul`].
//!
//! The unit tests in `src/matmul.rs` cover hand-checked cases and
//! invariants. This suite is the **dense-reference parity sweep**:
//! generate random sparse inputs across sizes and densities,
//! multiply them with both sparla's sparse kernel and a textbook
//! dense triple-loop oracle, and assert bit-for-bit agreement on
//! every cell.
//!
//! Same LCG-driven generators as the bench examples — every test
//! is reproducible.

use sparla::storage::{CsrFp, CsrZ, SparseRingStorage};
use sparla::Integer;

// =====================================================================
// LCG + matrix generators
// =====================================================================

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
}

fn gen_sparse_fp(rows: usize, cols: usize, density: f64, prime: u64, seed: u64) -> CsrFp {
    let mut rng = Lcg::new(seed);
    let target = ((rows * cols) as f64 * density) as usize;
    let p = prime as i64;
    let mut triples: Vec<(usize, usize, i64)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % rows;
        let c = (rng.next() as usize) % cols;
        let v = (rng.next() as i64 % (p - 1)) + 1;
        triples.push((r, c, v));
    }
    CsrFp::from_coo(rows, cols, &triples, prime)
}

fn gen_sparse_z(rows: usize, cols: usize, density: f64, seed: u64) -> CsrZ {
    let mut rng = Lcg::new(seed);
    let target = ((rows * cols) as f64 * density) as usize;
    let mut triples: Vec<(usize, usize, Integer)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % rows;
        let c = (rng.next() as usize) % cols;
        // Mixed-sign entries in [-9, 9] \ {0}.
        let mut v = (rng.next() as i64 % 19) - 9;
        if v == 0 {
            v = 1;
        }
        triples.push((r, c, Integer::from(v)));
    }
    CsrZ::from_coo(rows, cols, &triples)
}

// =====================================================================
// Dense reference matmuls
// =====================================================================

fn dense_matmul_fp(a: &[Vec<u32>], b: &[Vec<u32>], prime: u64) -> Vec<Vec<u32>> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let n = if b.is_empty() { 0 } else { b[0].len() };
    assert_eq!(b.len(), k);
    let mut c: Vec<Vec<u64>> = vec![vec![0u64; n]; m];
    for (i, c_row) in c.iter_mut().enumerate() {
        for kk in 0..k {
            let av = a[i][kk] as u64;
            for (j, cell) in c_row.iter_mut().enumerate() {
                *cell = (*cell + av * b[kk][j] as u64) % prime;
            }
        }
    }
    c.into_iter()
        .map(|row| row.into_iter().map(|v| v as u32).collect())
        .collect()
}

fn dense_matmul_z(a: &[Vec<Integer>], b: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let n = if b.is_empty() { 0 } else { b[0].len() };
    assert_eq!(b.len(), k);
    let mut c: Vec<Vec<Integer>> = (0..m).map(|_| vec![Integer::new(); n]).collect();
    for (i, c_row) in c.iter_mut().enumerate() {
        for kk in 0..k {
            for (j, cell) in c_row.iter_mut().enumerate() {
                let prod = &a[i][kk] * &b[kk][j];
                *cell += prod;
            }
        }
    }
    c
}

// =====================================================================
// F_p parity
// =====================================================================

const PRIME_SMALL: u64 = 7;
const PRIME_LARGE: u64 = (1u64 << 31) - 1; // M_31

fn assert_fp_parity(a: &CsrFp, b: &CsrFp) {
    let c_sparse = a.matmul(b);
    let c_dense = dense_matmul_fp(&a.to_dense(), &b.to_dense(), a.modulus());
    assert_eq!(c_sparse.to_dense(), c_dense);
    // Output ring & shape sanity.
    assert_eq!(c_sparse.modulus(), a.modulus());
    assert_eq!(c_sparse.rows(), a.rows());
    assert_eq!(c_sparse.cols(), b.cols());
}

#[test]
fn parity_fp_small_grid_p7() {
    // Many small configs to shake out edge cases.
    for &(m, k, n, density) in &[
        (5, 5, 5, 0.2),
        (10, 10, 10, 0.1),
        (10, 10, 10, 0.5),
        (5, 7, 3, 0.4),
        (20, 20, 20, 0.05),
        (20, 20, 20, 0.25),
        (50, 50, 50, 0.05),
    ] {
        let a = gen_sparse_fp(m, k, density, PRIME_SMALL, 0xAA);
        let b = gen_sparse_fp(k, n, density, PRIME_SMALL, 0xBB);
        assert_fp_parity(&a, &b);
    }
}

#[test]
fn parity_fp_rectangular_shapes_p7() {
    // Skinny and fat shapes.
    for &(m, k, n) in &[(3, 20, 3), (20, 3, 20), (1, 50, 1), (50, 1, 50)] {
        let a = gen_sparse_fp(m, k, 0.2, PRIME_SMALL, 0xCC);
        let b = gen_sparse_fp(k, n, 0.2, PRIME_SMALL, 0xDD);
        assert_fp_parity(&a, &b);
    }
}

#[test]
fn parity_fp_31_bit_prime() {
    // Largest 31-bit prime — exercises the upper edge of v1.0 ring scope.
    let a = gen_sparse_fp(30, 30, 0.1, PRIME_LARGE, 0xEE);
    let b = gen_sparse_fp(30, 30, 0.1, PRIME_LARGE, 0xFF);
    assert_fp_parity(&a, &b);
}

#[test]
fn parity_fp_zero_input_row_in_a() {
    // A's middle row is all zero — output's middle row should be zero too.
    let mut a = gen_sparse_fp(5, 5, 0.4, PRIME_SMALL, 1);
    // Re-create A but force row 2 to be empty.
    let mut triples: Vec<(usize, usize, i64)> = Vec::new();
    for r in 0..a.rows() {
        if r == 2 {
            continue;
        }
        for (c, v) in a.iter_row(r) {
            triples.push((r, c, v as i64));
        }
    }
    a = CsrFp::from_coo(5, 5, &triples, PRIME_SMALL);
    let b = gen_sparse_fp(5, 5, 0.4, PRIME_SMALL, 2);
    let c = a.matmul(&b);
    let row2: Vec<_> = c.iter_row(2).collect();
    assert!(row2.is_empty(), "row 2 of A · B should be empty");
    assert_fp_parity(&a, &b);
}

// =====================================================================
// Z parity
// =====================================================================

fn assert_z_parity(a: &CsrZ, b: &CsrZ) {
    let c_sparse = a.matmul(b);
    let c_dense = dense_matmul_z(&a.to_dense(), &b.to_dense());
    assert_eq!(c_sparse.to_dense(), c_dense);
    assert_eq!(c_sparse.rows(), a.rows());
    assert_eq!(c_sparse.cols(), b.cols());
}

#[test]
fn parity_z_small_grid() {
    for &(m, k, n, density) in &[
        (5, 5, 5, 0.2),
        (10, 10, 10, 0.1),
        (10, 10, 10, 0.5),
        (5, 7, 3, 0.4),
        (20, 20, 20, 0.05),
        (20, 20, 20, 0.25),
    ] {
        let a = gen_sparse_z(m, k, density, 0xA0);
        let b = gen_sparse_z(k, n, density, 0xB0);
        assert_z_parity(&a, &b);
    }
}

#[test]
fn parity_z_rectangular_shapes() {
    for &(m, k, n) in &[(3, 20, 3), (20, 3, 20), (1, 30, 1), (30, 1, 30)] {
        let a = gen_sparse_z(m, k, 0.3, 0xC0);
        let b = gen_sparse_z(k, n, 0.3, 0xD0);
        assert_z_parity(&a, &b);
    }
}

#[test]
fn parity_z_signed_entries() {
    // gen_sparse_z already yields mixed-sign entries; verify the
    // sparse and dense oracles agree on negation patterns.
    let a = gen_sparse_z(15, 15, 0.4, 0xE0);
    let b = gen_sparse_z(15, 15, 0.4, 0xF0);
    assert_z_parity(&a, &b);
}

// =====================================================================
// Cross-checks with other operations
// =====================================================================

#[test]
fn fp_matmul_kernel_gives_zero() {
    // For any A over F_p, A · kernel(A) = 0 (the kernel is in the
    // null space). This cross-checks matmul against kernel.
    let a = gen_sparse_fp(8, 10, 0.3, PRIME_SMALL, 0x42);
    let k = a.kernel();
    let prod = a.matmul(&k);
    assert_eq!(prod.nnz(), 0, "A · kernel(A) should be all zero");
}

#[test]
fn fp_matmul_associativity_random() {
    let a = gen_sparse_fp(7, 5, 0.4, PRIME_SMALL, 1);
    let b = gen_sparse_fp(5, 6, 0.4, PRIME_SMALL, 2);
    let c = gen_sparse_fp(6, 4, 0.4, PRIME_SMALL, 3);
    let lhs = a.matmul(&b).matmul(&c);
    let rhs = a.matmul(&b.matmul(&c));
    assert_eq!(lhs.to_dense(), rhs.to_dense(), "(A·B)·C ≠ A·(B·C)");
}

#[test]
fn z_matmul_associativity_random() {
    let a = gen_sparse_z(5, 4, 0.4, 1);
    let b = gen_sparse_z(4, 6, 0.4, 2);
    let c = gen_sparse_z(6, 3, 0.4, 3);
    let lhs = a.matmul(&b).matmul(&c);
    let rhs = a.matmul(&b.matmul(&c));
    assert_eq!(lhs.to_dense(), rhs.to_dense());
}
