// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse · sparse → sparse matrix multiplication.
//!
//! Gustavson's algorithm with a per-row dense accumulator. For
//! each output row `i`, walks the rows of `B` indexed by the
//! nonzeros of `A`'s row `i`, accumulating into an `n`-wide scratch
//! buffer. A `touched` index list lets the buffer reset in
//! `O(nnz_per_output_row)` time instead of `O(n)`.
//!
//! `examples/matmul_bench_flint.rs` compares this path with FLINT's dense
//! `nmod_mat_mul`; sparse storage avoids dense materialization at low density.

use std::cmp::Ordering;
use std::mem;

use crate::storage::{CsrFp, CsrZ, SparseRingStorage};
use crate::Integer;

/// `A · B` over `F_p`. Both matrices must share the same modulus
/// and dimensions must match (`A.cols() == B.rows()`).
///
/// # Panics
///
/// Panics on dimension mismatch or modulus mismatch.
pub fn matmul_fp(a: &CsrFp, b: &CsrFp) -> CsrFp {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_fp: inner dimensions {} ≠ {}",
        a.cols(),
        b.rows()
    );
    assert_eq!(
        a.modulus(),
        b.modulus(),
        "matmul_fp: moduli differ ({} vs {})",
        a.modulus(),
        b.modulus()
    );
    let p = a.modulus();
    let prime_u32 = p as u32;
    let m = a.rows();
    let n = b.cols();

    let mut out_rows: Vec<Vec<(usize, u32)>> = Vec::with_capacity(m);
    let mut acc: Vec<u64> = vec![0; n];
    let mut touched: Vec<usize> = Vec::new();

    for i in 0..m {
        for (k, av) in a.iter_row(i) {
            let av = av as u64;
            for (j, bv) in b.iter_row(k) {
                if acc[j] == 0 {
                    touched.push(j);
                }
                acc[j] = (acc[j] + av * bv as u64) % p;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        let mut row: Vec<(usize, u32)> = Vec::with_capacity(touched.len());
        for &j in &touched {
            let v = acc[j];
            if v != 0 {
                row.push((j, v as u32));
            }
            acc[j] = 0;
        }
        touched.clear();
        out_rows.push(row);
    }

    CsrFp::from_sorted_rows(m, n, prime_u32, out_rows)
}

/// `A · B` over `Z`. Dimensions must match (`A.cols() == B.rows()`).
///
/// Same Gustavson shape as [`matmul_fp`], but the accumulator holds
/// FLINT-backed [`Integer`] values — products accumulate exactly with no
/// modular reduction. `mem::take` moves nonzero entries out of the
/// accumulator when emitting each row, leaving `Integer::new()`
/// (zero) behind, so the buffer is naturally reset.
///
/// # Panics
///
/// Panics on dimension mismatch.
pub fn matmul_z(a: &CsrZ, b: &CsrZ) -> CsrZ {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_z: inner dimensions {} ≠ {}",
        a.cols(),
        b.rows()
    );
    let m = a.rows();
    let n = b.cols();

    let mut out_rows: Vec<Vec<(usize, Integer)>> = Vec::with_capacity(m);
    let mut acc: Vec<Integer> = (0..n).map(|_| Integer::new()).collect();
    let mut touched: Vec<usize> = Vec::new();

    for i in 0..m {
        for (k, av) in a.iter_row(i) {
            for (j, bv) in b.iter_row(k) {
                if acc[j].cmp0() == Ordering::Equal {
                    touched.push(j);
                }
                let prod = av * bv;
                acc[j] += prod;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        let mut row: Vec<(usize, Integer)> = Vec::with_capacity(touched.len());
        for &j in &touched {
            let v = mem::take(&mut acc[j]);
            if v.cmp0() != Ordering::Equal {
                row.push((j, v));
            }
        }
        touched.clear();
        out_rows.push(row);
    }

    CsrZ::from_sorted_rows(m, n, out_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    fn csr_fp_from_dense(dense: &[Vec<i64>], prime: u64) -> CsrFp {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut triples = Vec::new();
        for (r, row) in dense.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v.rem_euclid(prime as i64) != 0 {
                    triples.push((r, c, v));
                }
            }
        }
        CsrFp::from_coo(rows, cols, &triples, prime)
    }

    fn csr_z_from_dense(dense: &[Vec<i64>]) -> CsrZ {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
        for (r, row) in dense.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0 {
                    triples.push((r, c, z(v)));
                }
            }
        }
        CsrZ::from_coo(rows, cols, &triples)
    }

    // -- F_p ---------------------------------------------------------

    const P: u64 = 7;

    #[test]
    fn matmul_fp_identity_right() {
        let a = csr_fp_from_dense(&[vec![1, 2, 3], vec![4, 5, 6]], P);
        let id = csr_fp_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], P);
        let c = matmul_fp(&a, &id);
        assert_eq!(c.to_dense(), vec![vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn matmul_fp_identity_left() {
        let id = csr_fp_from_dense(&[vec![1, 0], vec![0, 1]], P);
        let a = csr_fp_from_dense(&[vec![3, 1, 4], vec![1, 5, 9]], P);
        let c = matmul_fp(&id, &a);
        assert_eq!(c.to_dense(), a.to_dense());
    }

    #[test]
    fn matmul_fp_zero_times_anything_is_zero() {
        let z_mat = CsrFp::zeros(2, 3, P);
        let a = csr_fp_from_dense(&[vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]], P);
        let c = matmul_fp(&z_mat, &a);
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 3);
    }

    #[test]
    fn matmul_fp_2x2_known() {
        // [[1, 2], [3, 4]] · [[5, 6], [7, 0]] mod 7
        //  = [[1·5 + 2·7, 1·6 + 2·0], [3·5 + 4·7, 3·6 + 4·0]]
        //  = [[19, 6], [43, 18]] mod 7
        //  = [[5, 6], [1, 4]]
        let a = csr_fp_from_dense(&[vec![1, 2], vec![3, 4]], P);
        let b = csr_fp_from_dense(&[vec![5, 6], vec![7, 0]], P);
        let c = matmul_fp(&a, &b);
        assert_eq!(c.to_dense(), vec![vec![5, 6], vec![1, 4]]);
    }

    #[test]
    fn matmul_fp_rectangular() {
        // 2×3 · 3×2 = 2×2.
        let a = csr_fp_from_dense(&[vec![1, 0, 2], vec![0, 3, 0]], P);
        let b = csr_fp_from_dense(&[vec![1, 1], vec![0, 2], vec![3, 0]], P);
        // C[0][0] = 1·1 + 0·0 + 2·3 = 7 ≡ 0
        // C[0][1] = 1·1 + 0·2 + 2·0 = 1
        // C[1][0] = 0·1 + 3·0 + 0·3 = 0
        // C[1][1] = 0·1 + 3·2 + 0·0 = 6
        let c = matmul_fp(&a, &b);
        assert_eq!(c.to_dense(), vec![vec![0, 1], vec![0, 6]]);
    }

    #[test]
    #[should_panic(expected = "inner dimensions")]
    fn matmul_fp_dim_mismatch_panics() {
        let a = csr_fp_from_dense(&[vec![1, 2, 3]], P);
        let b = csr_fp_from_dense(&[vec![1], vec![2]], P);
        let _ = matmul_fp(&a, &b);
    }

    #[test]
    #[should_panic(expected = "moduli differ")]
    fn matmul_fp_modulus_mismatch_panics() {
        let a = csr_fp_from_dense(&[vec![1]], 7);
        let b = csr_fp_from_dense(&[vec![1]], 11);
        let _ = matmul_fp(&a, &b);
    }

    // -- Z -----------------------------------------------------------

    #[test]
    fn matmul_z_identity_right() {
        let a = csr_z_from_dense(&[vec![1, 2, 3], vec![4, 5, 6]]);
        let id = csr_z_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        let c = matmul_z(&a, &id);
        assert_eq!(c.to_dense(), a.to_dense());
    }

    #[test]
    fn matmul_z_zero_times_anything_is_zero() {
        let z_mat = CsrZ::zeros(2, 3);
        let a = csr_z_from_dense(&[vec![1, 2], vec![3, 4], vec![5, 6]]);
        let c = matmul_z(&z_mat, &a);
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
    }

    #[test]
    fn matmul_z_2x2_known() {
        // [[1, 2], [3, 4]] · [[5, 6], [7, 8]]
        //   = [[1·5 + 2·7, 1·6 + 2·8], [3·5 + 4·7, 3·6 + 4·8]]
        //   = [[19, 22], [43, 50]]
        let a = csr_z_from_dense(&[vec![1, 2], vec![3, 4]]);
        let b = csr_z_from_dense(&[vec![5, 6], vec![7, 8]]);
        let c = matmul_z(&a, &b);
        assert_eq!(c.to_dense(), vec![vec![z(19), z(22)], vec![z(43), z(50)]]);
    }

    #[test]
    fn matmul_z_handles_negatives() {
        let a = csr_z_from_dense(&[vec![-1, 2], vec![3, -4]]);
        let b = csr_z_from_dense(&[vec![5, -6], vec![-7, 8]]);
        // C[0][0] = -5 - 14 = -19
        // C[0][1] = 6 + 16 = 22
        // C[1][0] = 15 + 28 = 43
        // C[1][1] = -18 - 32 = -50
        let c = matmul_z(&a, &b);
        assert_eq!(c.to_dense(), vec![vec![z(-19), z(22)], vec![z(43), z(-50)]]);
    }

    #[test]
    fn matmul_z_accumulator_zero_cancellation_drops_entry() {
        // Choose entries so a · b zeroes a cell exactly.
        // [[1, 1]] · [[1], [-1]] = [[0]] — sparsity invariant: no
        // explicit zero stored.
        let a = csr_z_from_dense(&[vec![1, 1]]);
        let b = csr_z_from_dense(&[vec![1], vec![-1]]);
        let c = matmul_z(&a, &b);
        assert_eq!(c.rows(), 1);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn matmul_z_with_large_integer_products() {
        // (10^9, 1) · (10^9; 10^9) = 10^18 + 10^9 → multi-limb GMP.
        let a = csr_z_from_dense(&[vec![1_000_000_000, 1]]);
        let b = csr_z_from_dense(&[vec![1_000_000_000], vec![1_000_000_000]]);
        let c = matmul_z(&a, &b);
        let expected: Integer = Integer::from(1_000_000_000u64) * Integer::from(1_000_000_000u64)
            + Integer::from(1_000_000_000u64);
        assert_eq!(c.to_dense(), vec![vec![expected]]);
    }

    #[test]
    fn matmul_z_outer_product_inner_product() {
        // (1, 2, 3) (column) · (4, 5, 6) (row) = 3×3 outer product.
        let col = csr_z_from_dense(&[vec![1], vec![2], vec![3]]);
        let row = csr_z_from_dense(&[vec![4, 5, 6]]);
        let outer = matmul_z(&col, &row);
        assert_eq!(
            outer.to_dense(),
            vec![
                vec![z(4), z(5), z(6)],
                vec![z(8), z(10), z(12)],
                vec![z(12), z(15), z(18)],
            ]
        );
        // (4, 5, 6) · (1; 2; 3) = 4 + 10 + 18 = 32.
        let inner = matmul_z(&row, &col);
        assert_eq!(inner.to_dense(), vec![vec![z(32)]]);
    }

    #[test]
    #[should_panic(expected = "inner dimensions")]
    fn matmul_z_dim_mismatch_panics() {
        let a = csr_z_from_dense(&[vec![1, 2, 3]]);
        let b = csr_z_from_dense(&[vec![1], vec![2]]);
        let _ = matmul_z(&a, &b);
    }
}
