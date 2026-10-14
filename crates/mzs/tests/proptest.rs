// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests over random `Mzs` inputs.
//!
//! Each property runs proptest's default of 256 randomized cases.
//! Shapes are bounded so the schoolbook reference (used as the
//! oracle) stays tractable: `m, k, n ∈ 0..16`.

use mzs::Mzs;
use proptest::prelude::*;

// ─── strategies ─────────────────────────────────────────────────────

/// `Mzs` of given shape with `Bernoulli(density)` entries.
fn mzs_strategy(rows: usize, cols: usize, density: f64) -> impl Strategy<Value = Mzs> {
    let cells = rows * cols;
    proptest::collection::vec(any::<u8>(), cells).prop_map(move |bytes| {
        let mut m = Mzs::new(rows, cols);
        let cutoff = (density * 255.0) as u8;
        for i in 0..rows {
            for j in 0..cols {
                if bytes[i * cols + j] <= cutoff {
                    m.set(i, j, 1);
                }
            }
        }
        m
    })
}

/// Dense schoolbook matmul over F_2 — the parity oracle.
fn schoolbook(a: &Mzs, b: &Mzs) -> Mzs {
    assert_eq!(a.cols(), b.rows());
    let mut c = Mzs::new(a.rows(), b.cols());
    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut acc = 0u8;
            for k in 0..a.cols() {
                acc ^= a.get(i, k) & b.get(k, j);
            }
            if acc != 0 {
                c.set(i, j, 1);
            }
        }
    }
    c
}

/// Schoolbook rank/rref over F_2 — the parity oracle for the
/// Gaussian path. Returns `(rref_as_mzs, rank)`.
fn schoolbook_rref(a: &Mzs) -> (Mzs, usize) {
    let m = a.rows();
    let n = a.cols();
    let mut dense: Vec<Vec<u8>> = (0..m)
        .map(|i| (0..n).map(|j| a.get(i, j)).collect())
        .collect();
    let mut rank = 0usize;
    let mut col = 0usize;
    while rank < m && col < n {
        let mut pi = None;
        for r in rank..m {
            if dense[r][col] == 1 {
                pi = Some(r);
                break;
            }
        }
        let p = match pi {
            Some(r) => r,
            None => {
                col += 1;
                continue;
            }
        };
        dense.swap(p, rank);
        for i in 0..m {
            if i == rank {
                continue;
            }
            if dense[i][col] == 1 {
                for j in col..n {
                    dense[i][j] ^= dense[rank][j];
                }
            }
        }
        rank += 1;
        col += 1;
    }
    let mut out = Mzs::new(m, n);
    for i in 0..m {
        for j in 0..n {
            if dense[i][j] != 0 {
                out.set(i, j, 1);
            }
        }
    }
    (out, rank)
}

// ─── matmul ─────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn matmul_matches_schoolbook(
        a in mzs_strategy(5, 7, 0.3),
        b in mzs_strategy(7, 4, 0.3),
    ) {
        let c_fast = a.matmul(&b);
        let c_ref = schoolbook(&a, &b);
        prop_assert_eq!(c_fast, c_ref);
    }

    #[test]
    fn matmul_associativity(
        a in mzs_strategy(4, 5, 0.3),
        b in mzs_strategy(5, 6, 0.3),
        c in mzs_strategy(6, 4, 0.3),
    ) {
        let lhs = a.matmul(&b).matmul(&c);
        let rhs = a.matmul(&b.matmul(&c));
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn matmul_transpose_identity(
        a in mzs_strategy(5, 6, 0.3),
        b in mzs_strategy(6, 4, 0.3),
    ) {
        let lhs = a.matmul(&b).transpose();
        let rhs = b.transpose().matmul(&a.transpose());
        prop_assert_eq!(lhs, rhs);
    }
}

// ─── matmul_dense / matmul_dense_packed ─────────────────────────────

/// Random row-major byte buffer.
fn random_bytes(rows: usize, cols: usize, density: f64) -> impl Strategy<Value = Vec<u8>> {
    let cells = rows * cols;
    proptest::collection::vec(any::<u8>(), cells).prop_map(move |bytes| {
        let cutoff = (density * 255.0) as u8;
        bytes
            .into_iter()
            .map(|b| if b <= cutoff { 1u8 } else { 0u8 })
            .collect()
    })
}

fn pack_lsb(rows: usize, cols: usize, b: &[u8]) -> Vec<u64> {
    let stride = cols.div_ceil(64);
    let mut out = vec![0u64; rows * stride];
    if stride == 0 {
        return out;
    }
    for r in 0..rows {
        for c in 0..cols {
            if b[r * cols + c] & 1 == 1 {
                out[r * stride + c / 64] |= 1u64 << (c % 64);
            }
        }
    }
    out
}

fn unpack_lsb(rows: usize, cols: usize, b: &[u64]) -> Vec<u8> {
    let stride = cols.div_ceil(64);
    let mut out = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = ((b[r * stride + c / 64] >> (c % 64)) & 1) as u8;
        }
    }
    out
}

proptest! {
    /// `matmul_dense` agrees with sparse matmul routed through
    /// `from_byte_buffer`. This is the byte-buffer side of the
    /// "densify-via-m4rst" oracle, without the m4rst path-dep.
    #[test]
    fn matmul_dense_matches_sparse_via_byte_buffer(
        a in mzs_strategy(4, 6, 0.3),
        b in random_bytes(6, 5, 0.3),
    ) {
        let dense_out = a.matmul_dense(6, 5, &b);
        let b_mzs = Mzs::from_byte_buffer(6, 5, &b);
        let sparse_out = a.matmul(&b_mzs).to_byte_buffer();
        prop_assert_eq!(dense_out, sparse_out);
    }

    /// `matmul_dense_packed` agrees with `matmul_dense` through
    /// pack/unpack. Covers stride > 1 and ragged-tail cases.
    #[test]
    fn matmul_dense_packed_matches_byte_variant(
        a in mzs_strategy(4, 6, 0.3),
        b in random_bytes(6, 70, 0.3),
    ) {
        let bytes_out = a.matmul_dense(6, 70, &b);
        let packed_out = a.matmul_dense_packed(6, 70, &pack_lsb(6, 70, &b));
        prop_assert_eq!(unpack_lsb(4, 70, &packed_out), bytes_out);
    }
}

// ─── transpose ──────────────────────────────────────────────────────

proptest! {
    #[test]
    fn transpose_is_involution(a in mzs_strategy(6, 8, 0.3)) {
        prop_assert_eq!(a.clone(), a.transpose().transpose());
    }

    #[test]
    fn transpose_swaps_shape(a in mzs_strategy(5, 7, 0.3)) {
        let t = a.transpose();
        prop_assert_eq!(t.rows(), a.cols());
        prop_assert_eq!(t.cols(), a.rows());
    }
}

// ─── byte-buffer round-trip ─────────────────────────────────────────

proptest! {
    #[test]
    fn byte_buffer_round_trip(a in mzs_strategy(7, 9, 0.4)) {
        let bytes = a.to_byte_buffer();
        let b = Mzs::from_byte_buffer(a.rows(), a.cols(), &bytes);
        prop_assert_eq!(a, b);
    }

    #[test]
    fn byte_buffer_length(a in mzs_strategy(5, 8, 0.3)) {
        prop_assert_eq!(a.to_byte_buffer().len(), a.rows() * a.cols());
    }
}

// ─── rref / rank ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn rref_matches_schoolbook(a in mzs_strategy(5, 7, 0.3)) {
        let (rref_fast, rank_fast) = a.rref();
        let (rref_ref, rank_ref) = schoolbook_rref(&a);
        prop_assert_eq!(rank_fast, rank_ref);
        prop_assert_eq!(rref_fast, rref_ref);
    }

    #[test]
    fn rref_is_idempotent(a in mzs_strategy(6, 6, 0.3)) {
        let (r1, _) = a.rref();
        let (r2, _) = r1.rref();
        prop_assert_eq!(r1, r2);
    }

    #[test]
    fn rank_unchanged_after_rref(a in mzs_strategy(5, 7, 0.3)) {
        let r1 = a.rank();
        let (rref, _) = a.rref();
        let r2 = rref.rank();
        prop_assert_eq!(r1, r2);
    }
}

// ─── nullspace / rank-nullity ───────────────────────────────────────

proptest! {
    #[test]
    fn nullspace_times_a_is_zero(a in mzs_strategy(4, 6, 0.3)) {
        let k = a.nullspace();
        let ak = a.matmul(&k);
        prop_assert_eq!(ak.nnz(), 0);
    }

    #[test]
    fn rank_nullity(a in mzs_strategy(5, 7, 0.3)) {
        let r = a.rank();
        let nullity = a.nullspace().cols();
        prop_assert_eq!(r + nullity, a.cols());
    }
}

// ─── solve_left ─────────────────────────────────────────────────────

proptest! {
    #[test]
    fn solve_left_round_trip_when_solvable(
        a in mzs_strategy(5, 5, 0.4),
        x in mzs_strategy(5, 3, 0.3),
    ) {
        // Construct B = A·X — always consistent by construction.
        let b = a.matmul(&x);
        let x_solved = Mzs::solve_left(&a, &b);
        // Even with a sparsified random A, the system A·X=B is solvable
        // because we built B from a real X. The solver may return a
        // different X' but A·X' must equal B.
        if let Some(x_prime) = x_solved {
            let b_check = a.matmul(&x_prime);
            prop_assert_eq!(b_check, b);
        } else {
            // Shouldn't happen by construction — but if it does the
            // assertion below makes it visible.
            prop_assert!(false, "solve_left returned None on a consistent system");
        }
    }
}
