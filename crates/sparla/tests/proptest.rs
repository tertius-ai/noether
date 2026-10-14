// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests via `proptest`.
//!
//! Complements the hand-rolled deterministic-seed tests by walking
//! a much wider random shape/content space. Every property here is
//! a mathematical invariant that must hold on any valid input —
//! when proptest finds a counterexample, the failure shrinks to a
//! minimal case and is replayable via the regression file
//! `proptest-regressions/proptest.txt`.

use std::cmp::Ordering;

use proptest::collection::vec;
use proptest::prelude::*;
use sparla::storage::{CsrFp, CsrZ, SparseRingStorage};
use sparla::Integer;

const SMALL_PRIME: u64 = 7;

// =====================================================================
// Generators
// =====================================================================

/// Sparse `F_p` matrix with shape in `[1, max_dim]` and density
/// driven by the requested target nnz. Values are in `[1, prime)`
/// (nonzero) so the actual stored nnz roughly tracks the target.
fn arb_csr_fp(max_dim: usize, prime: u64) -> impl Strategy<Value = CsrFp> {
    (1..=max_dim, 1..=max_dim).prop_flat_map(move |(r, c)| {
        let cells = r * c;
        let max_nnz = cells.min(8 * max_dim); // cap so generation stays cheap
        vec((0..r, 0..c, 1i64..(prime as i64)), 0..=max_nnz)
            .prop_map(move |triples| CsrFp::from_coo(r, c, &triples, prime))
    })
}

/// Shape-compatible pair: `a.cols() == b.rows()`.
fn arb_fp_matmul_pair(max_dim: usize, prime: u64) -> impl Strategy<Value = (CsrFp, CsrFp)> {
    (1..=max_dim, 1..=max_dim, 1..=max_dim).prop_flat_map(move |(m, k, n)| {
        let cells_a = m * k;
        let cells_b = k * n;
        let triples_a = vec((0..m, 0..k, 1i64..(prime as i64)), 0..=cells_a.min(40));
        let triples_b = vec((0..k, 0..n, 1i64..(prime as i64)), 0..=cells_b.min(40));
        (Just((m, k, n)), triples_a, triples_b).prop_map(move |((m, k, n), ta, tb)| {
            (
                CsrFp::from_coo(m, k, &ta, prime),
                CsrFp::from_coo(k, n, &tb, prime),
            )
        })
    })
}

/// Sparse `Z` matrix with entries in `[-bound, bound]` (nonzero).
fn arb_csr_z(max_dim: usize, bound: i64) -> impl Strategy<Value = CsrZ> {
    (1..=max_dim, 1..=max_dim).prop_flat_map(move |(r, c)| {
        let cells = r * c;
        let max_nnz = cells.min(8 * max_dim);
        vec((0..r, 0..c, -bound..=bound), 0..=max_nnz).prop_map(move |triples| {
            let triples_int: Vec<(usize, usize, Integer)> = triples
                .into_iter()
                .filter(|&(_, _, v)| v != 0)
                .map(|(rr, cc, v)| (rr, cc, Integer::from(v)))
                .collect();
            CsrZ::from_coo(r, c, &triples_int)
        })
    })
}

/// Shape-compatible matmul pair over Z.
fn arb_z_matmul_pair(max_dim: usize, bound: i64) -> impl Strategy<Value = (CsrZ, CsrZ)> {
    (1..=max_dim, 1..=max_dim, 1..=max_dim).prop_flat_map(move |(m, k, n)| {
        let cells_a = m * k;
        let cells_b = k * n;
        let triples_a = vec((0..m, 0..k, -bound..=bound), 0..=cells_a.min(40));
        let triples_b = vec((0..k, 0..n, -bound..=bound), 0..=cells_b.min(40));
        (Just((m, k, n)), triples_a, triples_b).prop_map(move |((m, k, n), ta, tb)| {
            let ta_int: Vec<(usize, usize, Integer)> = ta
                .into_iter()
                .filter(|&(_, _, v)| v != 0)
                .map(|(r, c, v)| (r, c, Integer::from(v)))
                .collect();
            let tb_int: Vec<(usize, usize, Integer)> = tb
                .into_iter()
                .filter(|&(_, _, v)| v != 0)
                .map(|(r, c, v)| (r, c, Integer::from(v)))
                .collect();
            (CsrZ::from_coo(m, k, &ta_int), CsrZ::from_coo(k, n, &tb_int))
        })
    })
}

/// Square sparse `Z` matrix.
fn arb_csr_z_square(max_dim: usize, bound: i64) -> impl Strategy<Value = CsrZ> {
    (1..=max_dim).prop_flat_map(move |n| {
        let cells = n * n;
        let max_nnz = cells.min(8 * max_dim);
        vec((0..n, 0..n, -bound..=bound), 0..=max_nnz).prop_map(move |triples| {
            let triples_int: Vec<(usize, usize, Integer)> = triples
                .into_iter()
                .filter(|&(_, _, v)| v != 0)
                .map(|(r, c, v)| (r, c, Integer::from(v)))
                .collect();
            CsrZ::from_coo(n, n, &triples_int)
        })
    })
}

// =====================================================================
// F_p properties
// =====================================================================

proptest! {
    /// `rank(A) + cols(kernel(A)) = cols(A)` — rank-nullity over F_p.
    #[test]
    fn fp_rank_nullity(a in arb_csr_fp(8, SMALL_PRIME)) {
        let r = a.rank();
        let k = a.kernel();
        prop_assert_eq!(r + k.cols(), a.cols());
    }

    /// `A · kernel(A) = 0` — kernel basis lies in the null space.
    #[test]
    fn fp_a_dot_kernel_is_zero(a in arb_csr_fp(8, SMALL_PRIME)) {
        let k = a.kernel();
        let prod = a.matmul(&k);
        prop_assert_eq!(prod.nnz(), 0);
    }

    /// `image(A).cols() = rank(A)` — image basis has exactly rank
    /// columns.
    #[test]
    fn fp_image_dim_equals_rank(a in arb_csr_fp(8, SMALL_PRIME)) {
        prop_assert_eq!(a.image().cols(), a.rank());
    }

    /// Matmul associativity: `(A·B)·C == A·(B·C)`.
    #[test]
    fn fp_matmul_associativity(
        ab in arb_fp_matmul_pair(5, SMALL_PRIME),
        c_triples in vec((0usize..5, 0usize..5, 1i64..(SMALL_PRIME as i64)), 0..30),
    ) {
        let (a, b) = ab;
        // Build C with rows = b.cols and cols = something small.
        let c_rows = b.cols();
        let c_cols = 5;
        let c_filtered: Vec<(usize, usize, i64)> = c_triples
            .into_iter()
            .filter(|&(r, c, _)| r < c_rows && c < c_cols)
            .collect();
        let c = CsrFp::from_coo(c_rows, c_cols, &c_filtered, SMALL_PRIME);
        let lhs = a.matmul(&b).matmul(&c);
        let rhs = a.matmul(&b.matmul(&c));
        prop_assert_eq!(lhs.to_dense(), rhs.to_dense());
    }
}

// =====================================================================
// Z properties
// =====================================================================

proptest! {
    /// `U · A = H` — the hnf_with_transforms invariant.
    #[test]
    fn z_u_times_a_equals_h(a in arb_csr_z(5, 5)) {
        let (h, u) = a.hnf_with_transforms();
        let ua = u.matmul(&a);
        prop_assert_eq!(ua.to_dense(), h.to_dense());
    }

    /// `det(U) ∈ {±1}` — the transform is unimodular.
    #[test]
    fn z_transform_is_unimodular(a in arb_csr_z(5, 5)) {
        let (_h, u) = a.hnf_with_transforms();
        let det = u.det();
        prop_assert!(det == 1 || det == -1);
    }

    /// `A · kernel(A) = 0` over Z.
    #[test]
    fn z_a_dot_kernel_is_zero(a in arb_csr_z(5, 5)) {
        let k = a.kernel();
        let prod = a.matmul(&k);
        prop_assert_eq!(prod.nnz(), 0);
    }

    /// `rank(A) + cols(kernel(A)) = cols(A)` over Z.
    #[test]
    fn z_rank_nullity(a in arb_csr_z(5, 5)) {
        let r = a.rank();
        let k = a.kernel();
        prop_assert_eq!(r + k.cols(), a.cols());
    }

    /// `∏ snf(A) = |det(A)|` for nonsingular square A.
    #[test]
    fn z_snf_product_equals_abs_det(a in arb_csr_z_square(5, 4)) {
        let det = a.det();
        let abs_det: Integer = if det.cmp0() == Ordering::Less { -det } else { det };
        let s = a.snf();
        let product: Integer = s.iter().fold(Integer::from(1), |acc, d| acc * d);
        prop_assert_eq!(product, abs_det);
    }

    /// Matmul associativity over Z: `(A·B)·C == A·(B·C)`.
    #[test]
    fn z_matmul_associativity(
        ab in arb_z_matmul_pair(4, 4),
        c_triples in vec((0usize..4, 0usize..4, -4i64..=4), 0..20),
    ) {
        let (a, b) = ab;
        let c_rows = b.cols();
        let c_cols = 4;
        let c_int: Vec<(usize, usize, Integer)> = c_triples
            .into_iter()
            .filter(|&(r, c, v)| r < c_rows && c < c_cols && v != 0)
            .map(|(r, c, v)| (r, c, Integer::from(v)))
            .collect();
        let c = CsrZ::from_coo(c_rows, c_cols, &c_int);
        let lhs = a.matmul(&b).matmul(&c);
        let rhs = a.matmul(&b.matmul(&c));
        prop_assert_eq!(lhs.to_dense(), rhs.to_dense());
    }

    /// HNF rank matches the column-reduction rank.
    #[test]
    fn z_hnf_nonzero_rows_equals_rank(a in arb_csr_z(5, 5)) {
        let r = a.rank();
        let h = a.hnf();
        let nonzero: usize = (0..h.rows())
            .filter(|&row| h.iter_row(row).next().is_some())
            .count();
        prop_assert_eq!(r, nonzero);
    }

    /// Transpose round-trip: `(Aᵀ)ᵀ = A`.
    #[test]
    fn z_transpose_round_trip(a in arb_csr_z(6, 6)) {
        let att = a.transpose().transpose();
        prop_assert_eq!(att.to_dense(), a.to_dense());
    }
}
