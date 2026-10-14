// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests for the GF(2^e) dense kernel. These
//! mathematical invariants must hold for every input:
//!
//!   * `Mzed::matmul` matches the schoolbook GF(2^e) product (the bitsliced
//!     kernel's correctness oracle);
//!   * addition commutes;
//!   * the multiplicative identity `A·I = I·A = A`;
//!   * matmul associates;
//!   * `rref` is idempotent and `rank()` agrees with the RREF pivot count.
//!
//! Run over GF(2^4). `Mzed` derives only `Clone`, so equality is compared via
//! the row-major `to_u16_buffer()`. Pure Rust — `cargo test -p m4rste`.

use m4rste::{Gf2e, Mzed};
use proptest::prelude::*;

const MINPOLY: u64 = 0x13; // GF(2^4): x^4 + x + 1
const ORDER: u16 = 16; // 2^4 field elements live in 0..16

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

fn ctx() -> Gf2e {
    Gf2e::new(MINPOLY)
}

fn eq(a: &Mzed, b: &Mzed) -> bool {
    a.rows() == b.rows() && a.cols() == b.cols() && a.to_u16_buffer() == b.to_u16_buffer()
}

/// Schoolbook GF(2^e) matmul oracle: `C[i,j] = ⊕_k a[i,k]·b[k,j]`.
fn schoolbook(a: &Mzed, b: &Mzed) -> Vec<u16> {
    let g = ctx();
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    let mut out = vec![0u16; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0u16;
            for kk in 0..k {
                acc = g.add(acc, g.mul(a.get(i, kk), b.get(kk, j)));
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn buf(len: usize) -> impl Strategy<Value = Vec<u16>> {
    prop::collection::vec(0u16..ORDER, len..=len)
}

/// Two matrices of identical shape (for addition).
fn arb_shape_pair(max: usize) -> impl Strategy<Value = (Mzed, Mzed)> {
    (1usize..=max, 1usize..=max).prop_flat_map(|(r, c)| {
        (buf(r * c), buf(r * c)).prop_map(move |(ba, bb)| {
            (
                Mzed::from_u16_buffer(ctx(), r, c, &ba),
                Mzed::from_u16_buffer(ctx(), r, c, &bb),
            )
        })
    })
}

/// A shape-compatible pair `(m×k, k×n)` for matmul.
fn arb_matmul_pair(max: usize) -> impl Strategy<Value = (Mzed, Mzed)> {
    (1usize..=max, 1usize..=max, 1usize..=max).prop_flat_map(|(m, k, n)| {
        (buf(m * k), buf(k * n)).prop_map(move |(ba, bb)| {
            (
                Mzed::from_u16_buffer(ctx(), m, k, &ba),
                Mzed::from_u16_buffer(ctx(), k, n, &bb),
            )
        })
    })
}

/// A chainable triple `(m×k, k×n, n×p)` for associativity.
fn arb_matmul_triple(max: usize) -> impl Strategy<Value = (Mzed, Mzed, Mzed)> {
    (1usize..=max, 1usize..=max, 1usize..=max, 1usize..=max).prop_flat_map(|(m, k, n, p)| {
        (buf(m * k), buf(k * n), buf(n * p)).prop_map(move |(ba, bb, bc)| {
            (
                Mzed::from_u16_buffer(ctx(), m, k, &ba),
                Mzed::from_u16_buffer(ctx(), k, n, &bb),
                Mzed::from_u16_buffer(ctx(), n, p, &bc),
            )
        })
    })
}

fn arb_mzed(max: usize) -> impl Strategy<Value = Mzed> {
    (1usize..=max, 1usize..=max).prop_flat_map(|(r, c)| {
        buf(r * c).prop_map(move |b| Mzed::from_u16_buffer(ctx(), r, c, &b))
    })
}

fn arb_square(max: usize) -> impl Strategy<Value = Mzed> {
    (1usize..=max)
        .prop_flat_map(|n| buf(n * n).prop_map(move |b| Mzed::from_u16_buffer(ctx(), n, n, &b)))
}

proptest! {
    #![proptest_config(property_config(32))]

    /// `Mzed::matmul` matches the schoolbook GF(2^e) oracle.
    #[test]
    fn matmul_matches_schoolbook((a, b) in arb_matmul_pair(12)) {
        prop_assert_eq!(Mzed::matmul(&a, &b).to_u16_buffer(), schoolbook(&a, &b));
    }

    /// `matmul_into` agrees with `matmul`.
    #[test]
    fn matmul_into_matches_matmul((a, b) in arb_matmul_pair(12)) {
        let want = Mzed::matmul(&a, &b);
        let mut got = Mzed::zeros(ctx(), a.rows(), b.cols());
        Mzed::matmul_into(&mut got, &a, &b);
        prop_assert!(eq(&got, &want));
    }

    /// Addition commutes.
    #[test]
    fn add_commutes((a, b) in arb_shape_pair(16)) {
        prop_assert!(eq(&Mzed::add(&a, &b), &Mzed::add(&b, &a)));
    }

    /// Multiplicative identity: `A·I = A` and `I·A = A`.
    #[test]
    fn matmul_identity(a in arb_mzed(14)) {
        let i_right = Mzed::identity(ctx(), a.cols());
        let i_left = Mzed::identity(ctx(), a.rows());
        prop_assert!(eq(&Mzed::matmul(&a, &i_right), &a), "A·I = A");
        prop_assert!(eq(&Mzed::matmul(&i_left, &a), &a), "I·A = A");
    }

    /// Matmul associates.
    #[test]
    fn matmul_associates((a, b, c) in arb_matmul_triple(8)) {
        let lhs = Mzed::matmul(&Mzed::matmul(&a, &b), &c);
        let rhs = Mzed::matmul(&a, &Mzed::matmul(&b, &c));
        prop_assert!(eq(&lhs, &rhs));
    }

    /// `rref` is idempotent, and `rank()` agrees with the RREF pivot count.
    #[test]
    fn rref_idempotent_and_rank(a in arb_square(12)) {
        let (r1, rank1) = a.rref();
        let (r2, _) = r1.rref();
        prop_assert!(eq(&r1, &r2), "rref is idempotent");
        prop_assert_eq!(a.rank(), rank1, "rank() agrees with rref pivot count");
    }

    /// `pluq`: `P · L · U · Q == A` and the returned rank agrees with `rank()`.
    #[test]
    fn pluq_reconstructs_and_rank_matches(a in arb_mzed(12)) {
        let (p, l, u, q, r) = a.pluq();
        let lu = Mzed::matmul(&l, &u);
        let plu = Mzed::matmul(&p, &lu);
        let pluq = Mzed::matmul(&plu, &q);
        prop_assert!(eq(&pluq, &a), "P · L · U · Q ≠ A");
        prop_assert_eq!(r, a.rank(), "pluq rank disagrees with rank()");
    }

    /// Rank-nullity theorem: `rank(A) + nullity(A) == cols(A)`.
    #[test]
    fn rank_nullity(a in arb_mzed(12)) {
        let r = a.rank();
        let n = Mzed::nullspace(&a).cols();
        prop_assert_eq!(r + n, a.cols());
    }

    /// Every column of `nullspace(A)` is in the kernel of A: `A · k == 0`.
    #[test]
    fn nullspace_columns_in_kernel(a in arb_mzed(12)) {
        let k = Mzed::nullspace(&a);
        if k.cols() == 0 {
            return Ok(());
        }
        let ak = Mzed::matmul(&a, &k);
        for v in ak.to_u16_buffer() {
            prop_assert_eq!(v, 0);
        }
    }

    /// `solve_left(A, A·y)` returns SOME x with `A · x == A · y`.
    /// (Not necessarily `x == y` when A is rank-deficient.)
    #[test]
    fn solve_round_trip((a, y) in arb_matmul_pair(10)) {
        let b = Mzed::matmul(&a, &y);
        let x = Mzed::solve_left(&a, &b).expect("Ax = A·y is always consistent");
        let ax = Mzed::matmul(&a, &x);
        prop_assert!(eq(&ax, &b));
    }

    /// When `A` is square and full-rank, `inverse(A) · A == I` and `A · inverse(A) == I`.
    #[test]
    fn inverse_round_trip(a in arb_square(10)) {
        let n = a.rows();
        if a.rank() != n {
            return Ok(()); // not invertible; the property doesn't apply
        }
        let inv = Mzed::inverse(&a).expect("rank=n ⇒ invertible");
        let i = Mzed::identity(ctx(), n);
        let ai = Mzed::matmul(&a, &inv);
        let ia = Mzed::matmul(&inv, &a);
        prop_assert!(eq(&ai, &i));
        prop_assert!(eq(&ia, &i));
    }

    /// `transpose` is an involution: `(Aᵀ)ᵀ == A`.
    #[test]
    fn transpose_involution(a in arb_mzed(12)) {
        prop_assert!(eq(&a.transpose().transpose(), &a));
    }
}
