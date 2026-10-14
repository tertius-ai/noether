// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests for the packed Z/p^n kernel (`MatZpn`).
//! These cover mathematical invariants over a wide random space:
//!
//!   * `MatZpn::matmul` matches the scalar u128 reference (the packed/Barrett
//!     kernel's correctness oracle);
//!   * addition commutes;
//!   * transpose is an involution `(Aᵀ)ᵀ = A`, and `(A·B)ᵀ = Bᵀ·Aᵀ`;
//!   * PLUQ reconstructs: `P·L·U·Q = A`;
//!   * `rank()` agrees with the `rref` pivot count.
//!
//! Run over Z/7² (= Z/49). `MatZpn` has a manual `PartialEq`, so `==` is used.
//! Pure Rust — `cargo test -p pwvla --release --test proptest`.

use proptest::prelude::*;
use pwvla::zpn::MatZpn;
use pwvla::Mat;

const P: u32 = 7;
const N: u32 = 2;
const MOD: u64 = 49; // 7^2

fn property_config(default_cases: u32) -> ProptestConfig {
    let configured = ProptestConfig::default();
    let cases = if std::env::var_os("PROPTEST_CASES").is_some() {
        configured.cases
    } else {
        default_cases
    };
    ProptestConfig {
        cases,
        max_shrink_iters: 64,
        ..configured
    }
}

/// Scalar reference matmul via u128 — independent of the packed Barrett path.
fn reference_matmul(a: &MatZpn, b: &MatZpn) -> MatZpn {
    assert_eq!(a.cols, b.rows);
    let (m, k, n) = (a.rows, a.cols, b.cols);
    MatZpn::from_fn(P, N, m, n, |i, j| {
        let mut s: u128 = 0;
        for kk in 0..k {
            s = (s + (a.get(i, kk) as u128) * (b.get(kk, j) as u128)) % (MOD as u128);
        }
        s as u64
    })
}

fn buf(len: usize) -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(0u64..MOD, len..=len)
}

fn mk(r: usize, c: usize, b: &[u64]) -> MatZpn {
    MatZpn::from_fn(P, N, r, c, |i, j| b[i * c + j])
}

fn arb_mat(max: usize) -> impl Strategy<Value = MatZpn> {
    (1usize..=max, 1usize..=max).prop_flat_map(|(r, c)| buf(r * c).prop_map(move |b| mk(r, c, &b)))
}

/// Two matrices of identical shape (for addition).
fn arb_shape_pair(max: usize) -> impl Strategy<Value = (MatZpn, MatZpn)> {
    (1usize..=max, 1usize..=max).prop_flat_map(|(r, c)| {
        (buf(r * c), buf(r * c)).prop_map(move |(ba, bb)| (mk(r, c, &ba), mk(r, c, &bb)))
    })
}

/// A shape-compatible pair `(m×k, k×n)` for matmul.
fn arb_matmul_pair(max: usize) -> impl Strategy<Value = (MatZpn, MatZpn)> {
    (1usize..=max, 1usize..=max, 1usize..=max).prop_flat_map(|(m, k, n)| {
        (buf(m * k), buf(k * n)).prop_map(move |(ba, bb)| (mk(m, k, &ba), mk(k, n, &bb)))
    })
}

/// `A = L·U` with unit-(lower/upper)-triangular factors over Z/p^n. This
/// guarantees a unit-pivot PLUQ decomposition exists, so `P·L·U·Q = A` holds —
/// a *general* matrix over a composite modulus need NOT be unit-pivot
/// decomposable (pwvla's PLUQ stops when a pivot lands in the maximal ideal),
/// so the reconstruction property is only meaningful under this precondition.
fn arb_lu_product(max: usize) -> impl Strategy<Value = MatZpn> {
    use std::cmp::Ordering::{Equal, Greater, Less};
    (1usize..=max).prop_flat_map(|n| {
        (buf(n * n), buf(n * n)).prop_map(move |(lb, ub)| {
            let l = MatZpn::from_fn(P, N, n, n, |i, j| match i.cmp(&j) {
                Greater => lb[i * n + j], // strictly lower: free
                Equal => 1,               // unit diagonal
                Less => 0,
            });
            let u = MatZpn::from_fn(P, N, n, n, |i, j| match i.cmp(&j) {
                Less => ub[i * n + j], // strictly upper: free
                Equal => 1,            // unit diagonal
                Greater => 0,
            });
            l.matmul(&u)
        })
    })
}

proptest! {
    #![proptest_config(property_config(32))]

    /// `MatZpn::matmul` matches the scalar u128 reference.
    #[test]
    fn matmul_matches_reference((a, b) in arb_matmul_pair(10)) {
        prop_assert_eq!(a.matmul(&b), reference_matmul(&a, &b));
    }

    /// Addition commutes.
    #[test]
    fn add_commutes((a, b) in arb_shape_pair(12)) {
        prop_assert_eq!(a.add(&b), b.add(&a));
    }

    /// Transpose is an involution.
    #[test]
    fn transpose_involution(a in arb_mat(14)) {
        prop_assert_eq!(a.transposed().transposed(), a);
    }

    /// `(A·B)ᵀ = Bᵀ·Aᵀ`.
    #[test]
    fn transpose_reverses_product((a, b) in arb_matmul_pair(10)) {
        let lhs = a.matmul(&b).transposed();
        let rhs = b.transposed().matmul(&a.transposed());
        prop_assert_eq!(lhs, rhs);
    }

    /// Zero-width generic matrices remain valid under every row-parallel
    /// basic operation, regardless of the number of rows.
    #[test]
    fn generic_zero_width_arithmetic_is_total(rows in 0usize..32, scalar in any::<u16>()) {
        let matrix = Mat::<u16>::zeros(rows, 0);
        prop_assert_eq!(matrix.add(&matrix), matrix.clone());
        prop_assert_eq!(matrix.sub(&matrix), matrix.clone());
        prop_assert_eq!(matrix.neg(), matrix.clone());
        prop_assert_eq!(matrix.scalar_mul(scalar), matrix.clone());
        prop_assert_eq!(matrix.reduce_mod_2(), matrix.clone());
        prop_assert_eq!(matrix.transpose().transpose(), matrix);
    }

    /// PLUQ reconstructs a unit-pivot-decomposable matrix: `P·L·U·Q = A`.
    #[test]
    fn pluq_reconstructs(a in arb_lu_product(7)) {
        let (pp, l, u, qq, _rank) = a.pluq();
        let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
        prop_assert_eq!(prod, a);
    }

    /// `rank()` agrees with the `rref` pivot count.
    #[test]
    fn rank_matches_rref(a in arb_mat(8)) {
        let (_, rref_rank) = a.rref();
        prop_assert_eq!(a.rank(), rref_rank);
    }
}
