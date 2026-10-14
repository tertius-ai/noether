// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests using `proptest`.
//!
//! These complement the hand-rolled deterministic-seed tests in
//! `tests/{construction,kernels,echelon,pluq,solve,views}.rs` by
//! exploring a much wider random shape and content space. Each
//! property here is a mathematical invariant that must hold for any
//! valid input.

use m4rst::Mzd;
use proptest::prelude::*;

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

/// Schoolbook F_2 matmul as the oracle: C[i,j] = XOR_k (A[i,k] AND B[k,j]).
/// Used by the matmul property tests below.
fn schoolbook_matmul(a: &Mzd, b: &Mzd) -> Mzd {
    assert_eq!(a.cols(), b.rows());
    let mut c = Mzd::new(a.rows(), b.cols());
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

/// Random Mzd with controlled shape, seeded by the proptest u64.
fn arb_mzd(min_dim: usize, max_dim: usize) -> impl Strategy<Value = Mzd> {
    (min_dim..=max_dim, min_dim..=max_dim, any::<u64>())
        .prop_map(|(r, c, seed)| Mzd::random(r, c, seed))
}

/// Random shape-compatible pair of Mzds for matmul (`a.cols == b.rows`).
fn arb_matmul_pair(min_dim: usize, max_dim: usize) -> impl Strategy<Value = (Mzd, Mzd)> {
    (
        min_dim..=max_dim,
        min_dim..=max_dim,
        min_dim..=max_dim,
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(|(r, k, c, seed_a, seed_b)| {
            (Mzd::random(r, k, seed_a), Mzd::random(k, c, seed_b))
        })
}

proptest! {
    // Keep the ordinary count modest: each case is a real matmul. Stress runs
    // override it through PROPTEST_CASES.
    #![proptest_config(property_config(32))]

    /// `Mzd::matmul` matches the schoolbook oracle for small random shapes.
    /// Covers every dispatcher branch indirectly via varied (m, k, n) shapes.
    #[test]
    fn matmul_matches_schoolbook((a, b) in arb_matmul_pair(1, 33)) {
        let got = Mzd::matmul(&a, &b);
        let oracle = schoolbook_matmul(&a, &b);
        prop_assert_eq!(got.as_words(), oracle.as_words());
    }

    /// `matmul_into` agrees with `matmul` for small random shapes.
    #[test]
    fn matmul_into_matches_matmul((a, b) in arb_matmul_pair(1, 33)) {
        let want = Mzd::matmul(&a, &b);
        let mut got = Mzd::new(a.rows(), b.cols());
        Mzd::matmul_into(&mut got, &a, &b);
        prop_assert_eq!(got.as_words(), want.as_words());
    }

    /// `transpose` is an involution: `(A^T)^T == A`.
    #[test]
    fn transpose_is_involutory(a in arb_mzd(1, 80)) {
        let aa = a.transpose().transpose();
        prop_assert_eq!(a.as_words(), aa.as_words());
    }

    /// `(A · B)^T == B^T · A^T` for shape-compatible A, B.
    #[test]
    fn transpose_distributes_over_matmul((a, b) in arb_matmul_pair(1, 33)) {
        let ab_t = Mzd::matmul(&a, &b).transpose();
        let bt_at = Mzd::matmul(&b.transpose(), &a.transpose());
        prop_assert_eq!(ab_t.as_words(), bt_at.as_words());
    }

    /// PLUQ structure: P, Q permutation matrices; L unit lower
    /// triangular in first `rank` cols; U upper triangular in first
    /// `rank` rows; `P · L · U · Q == A`.
    #[test]
    fn pluq_structure(a in arb_mzd(1, 40)) {
        let (p, l, u, q, rank) = a.pluq();

        // P is a permutation matrix.
        prop_assert!(is_permutation(&p));
        prop_assert!(is_permutation(&q));

        // L: unit lower triangular in first `rank` cols, zero in the rest.
        for i in 0..l.rows() {
            for j in 0..l.cols() {
                if j >= rank {
                    prop_assert_eq!(l.get(i, j), 0, "L[{},{}] != 0 (j >= rank)", i, j);
                } else if j > i {
                    prop_assert_eq!(l.get(i, j), 0, "L[{},{}] != 0 (above diag)", i, j);
                } else if j == i && i < rank {
                    prop_assert_eq!(l.get(i, j), 1, "L[{},{}] != 1 (unit diag)", i, j);
                }
            }
        }

        // U: upper triangular in first `rank` rows, zero below.
        for i in 0..u.rows() {
            for j in 0..u.cols() {
                if i >= rank {
                    prop_assert_eq!(u.get(i, j), 0, "U[{},{}] != 0 (i >= rank)", i, j);
                } else if j < i {
                    prop_assert_eq!(u.get(i, j), 0, "U[{},{}] != 0 (below diag)", i, j);
                }
            }
        }

        // P · L · U · Q == A.
        let lu = Mzd::matmul(&l, &u);
        let plu = Mzd::matmul(&p, &lu);
        let pluq = Mzd::matmul(&plu, &q);
        prop_assert_eq!(pluq.as_words(), a.as_words());
    }

    /// Rank-nullity theorem: `rank(A) + nullity(A) == cols(A)`.
    #[test]
    fn rank_nullity(a in arb_mzd(1, 40)) {
        let r = a.rank();
        let n = Mzd::nullspace(&a).cols();
        prop_assert_eq!(r + n, a.cols());
    }

    /// Every column of `nullspace(A)` is in the kernel of A: `A · k == 0`.
    #[test]
    fn nullspace_columns_in_kernel(a in arb_mzd(1, 40)) {
        let k = Mzd::nullspace(&a);
        if k.cols() == 0 {
            return Ok(());
        }
        let ak = Mzd::matmul(&a, &k);
        for &w in ak.as_words() {
            prop_assert_eq!(w, 0);
        }
    }

    /// `solve_left(A, A·y)` returns SOME x with `A · x == A · y`.
    /// (Not necessarily `x == y` when A is rank-deficient.)
    #[test]
    fn solve_round_trip((a, y) in arb_matmul_pair(1, 33)) {
        let b = Mzd::matmul(&a, &y);
        let x = Mzd::solve_left(&a, &b).expect("Ax = A·y is always consistent");
        let ax = Mzd::matmul(&a, &x);
        prop_assert_eq!(ax.as_words(), b.as_words());
    }

    /// When `A` is square and full-rank, `inverse(A) · A == I` and `A · inverse(A) == I`.
    #[test]
    fn inverse_round_trip(n in 1usize..=32, seed in any::<u64>()) {
        let a = Mzd::random(n, n, seed);
        if a.rank() != n {
            // Not invertible; the property doesn't apply.
            return Ok(());
        }
        let inv = Mzd::inverse(&a).expect("rank=n ⇒ invertible");
        let i = Mzd::identity(n);
        let ai = Mzd::matmul(&a, &inv);
        let ia = Mzd::matmul(&inv, &a);
        prop_assert_eq!(ai.as_words(), i.as_words());
        prop_assert_eq!(ia.as_words(), i.as_words());
    }

    /// `from_byte_buffer ∘ to_byte_buffer == identity` on the buffer.
    #[test]
    fn byte_buffer_round_trip(a in arb_mzd(1, 80)) {
        let bytes = a.to_byte_buffer();
        let a2 = Mzd::from_byte_buffer(a.rows(), a.cols(), &bytes);
        prop_assert_eq!(a.as_words(), a2.as_words());
    }
}

fn is_permutation(m: &Mzd) -> bool {
    if m.rows() != m.cols() {
        return false;
    }
    let n = m.rows();
    for i in 0..n {
        let mut sum = 0;
        for j in 0..n {
            if m.get(i, j) == 1 {
                sum += 1;
            }
        }
        if sum != 1 {
            return false;
        }
    }
    for j in 0..n {
        let mut sum = 0;
        for i in 0..n {
            if m.get(i, j) == 1 {
                sum += 1;
            }
        }
        if sum != 1 {
            return false;
        }
    }
    true
}
