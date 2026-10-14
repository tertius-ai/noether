// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Correctness tests for caller-owned-output `Mzd::matmul_into`.
//!
//! Sizes are chosen to cross the dispatcher's tier boundaries
//! (`matmul/mod.rs`'s `M4RM_MIN_ROWS_A`/`M4RM_MIN_COLS_A`/
//! `NTABLES_MIN_INNER`) so `matmul_into` is exercised against the
//! naive, M4RM, and NTABLES kernels — not just whichever one a small
//! matrix happens to fall into. The Strassen tiers (`N ≥ 16384`
//! serial, `N ≥ 32768`/`49152` parallel) write straight into
//! `dst` via `MzdViewMut` (`matmul/mod.rs`'s `matmul_into` doc); the
//! "Strassen-windowing" test block below at the end of this file
//! covers that path — both directly at low thresholds and, once, at
//! the real `N = 16384` production threshold through `Mzd::matmul_into`
//! itself.

use m4rst::{MatrixWrite, Mzd};

fn assert_matmul_into_matches(a: &Mzd, b: &Mzd) {
    let want = Mzd::matmul(a, b);
    let mut got = Mzd::new(a.rows(), b.cols());
    Mzd::matmul_into(&mut got, a, b);
    assert_eq!(
        got.as_words(),
        want.as_words(),
        "matmul_into ≠ matmul for {}×{} · {}×{}",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
}

#[test]
fn matmul_into_matches_matmul_naive_tier() {
    // Below M4RM_MIN_ROWS_A (32) — naive kernel.
    for &(m, k, n, seed) in &[
        (1usize, 1usize, 1usize, 1u64),
        (5, 12, 8, 2),
        (16, 16, 16, 3),
        (31, 31, 31, 4),
    ] {
        let a = Mzd::random(m, k, seed);
        let b = Mzd::random(k, n, seed ^ 0xABCD);
        assert_matmul_into_matches(&a, &b);
    }
}

#[test]
fn matmul_into_matches_matmul_m4rm_tier() {
    // rows_a >= 32, cols_a in [8, 64) — plain M4RM (below NTABLES_MIN_INNER).
    for &(m, k, n, seed) in &[
        (32usize, 8usize, 8usize, 5u64),
        (40, 20, 30, 6),
        (63, 40, 50, 7),
    ] {
        let a = Mzd::random(m, k, seed);
        let b = Mzd::random(k, n, seed ^ 0xBEEF);
        assert_matmul_into_matches(&a, &b);
    }
}

#[test]
fn matmul_into_matches_matmul_ntables_tier() {
    // rows_a >= 32, cols_a >= 64 — NTABLES-fused M4RM.
    for &(m, k, n, seed) in &[
        (32usize, 64usize, 40usize, 8u64),
        (100, 100, 100, 9),
        (65, 130, 33, 10),
    ] {
        let a = Mzd::random(m, k, seed);
        let b = Mzd::random(k, n, seed ^ 0xCAFE);
        assert_matmul_into_matches(&a, &b);
    }
}

#[test]
fn matmul_into_identity() {
    let n = 40;
    let id = Mzd::identity(n);
    let a = Mzd::random(n, n, 42);
    let mut c = Mzd::new(n, n);
    Mzd::matmul_into(&mut c, &id, &a);
    assert_eq!(c.as_words(), a.as_words());
}

#[test]
fn matmul_into_on_reused_buffer_does_not_accumulate_stale_data() {
    // matmul_into must fully overwrite dst (via the upfront fill_zero),
    // not leave any residue from a prior unrelated call reusing the
    // same allocation.
    let a1 = Mzd::random(40, 40, 0xA1);
    let b1 = Mzd::random(40, 40, 0xA2);
    let a2 = Mzd::random(40, 40, 0xB1);
    let b2 = Mzd::random(40, 40, 0xB2);

    let mut dst = Mzd::new(40, 40);
    Mzd::matmul_into(&mut dst, &a1, &b1); // first call: populates dst
    Mzd::matmul_into(&mut dst, &a2, &b2); // reuse: must fully overwrite

    let want = Mzd::matmul(&a2, &b2);
    assert_eq!(dst.as_words(), want.as_words());
}

#[test]
fn matmul_into_rectangular_shapes() {
    for &(m, k, n, seed) in &[
        (5usize, 12usize, 20usize, 11u64),
        (20, 5, 12, 12),
        (8, 65, 8, 13),
    ] {
        let a = Mzd::random(m, k, seed);
        let b = Mzd::random(k, n, seed ^ 0xDEAD);
        assert_matmul_into_matches(&a, &b);
    }
}

// ---- Strassen windowing ------------------------------------------------
//
// `matmul_into`'s Strassen tiers write through `MzdViewMut`
// straight into `dst`'s own backing store (`strassen::matmul_strassen_view`
// / `_parallel_leaf` / `_peeled_..._into`). These tests call those functions
// directly against an externally supplied `dst`: a destination
// that starts holding an unrelated prior result must come out exactly
// right, not accumulate onto the stale data.

fn schoolbook_matmul(a: &Mzd, b: &Mzd) -> Mzd {
    let (r, k, c) = (a.rows(), a.cols(), b.cols());
    let mut out = Mzd::new(r, c);
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0u8;
            for kk in 0..k {
                acc ^= a.get(i, kk) & b.get(kk, j);
            }
            if acc != 0 {
                out.set(i, j, 1);
            }
        }
    }
    out
}

#[test]
fn strassen_view_into_reused_dst_matches_schoolbook() {
    use m4rst::matmul::strassen::matmul_strassen_view;
    // threshold=128, N=512 gives two levels of recursion through the same
    // view-writing function used by `matmul_into`'s serial Strassen branch.
    let n = 512;
    let a1 = Mzd::random(n, n, 0xA1);
    let b1 = Mzd::random(n, n, 0xA2);
    let a2 = Mzd::random(n, n, 0xB1);
    let b2 = Mzd::random(n, n, 0xB2);

    let mut dst = Mzd::new(n, n);
    {
        let mut v = dst.full_view_mut();
        matmul_strassen_view(&mut v, &a1.full_view(), &b1.full_view(), 128);
    }
    // Reuse: dst currently holds a1*b1. matmul_into's contract is to
    // fill_zero() dst before handing the view to matmul_strassen_view
    // (which itself assumes a pre-zeroed dst) — mirror that here.
    dst.fill_zero();
    {
        let mut v = dst.full_view_mut();
        matmul_strassen_view(&mut v, &a2.full_view(), &b2.full_view(), 128);
    }
    let want = schoolbook_matmul(&a2, &b2);
    assert_eq!(dst.as_words(), want.as_words());
}

#[cfg_attr(coverage, ignore)]
#[test]
fn matmul_into_strassen_serial_tier_reused_dst() {
    // Force the real STRASSEN_SERIAL_MIN=16384 production branch of
    // Mzd::matmul_into (not a low-threshold stand-in) and confirm a
    // dst reused from a prior, unrelated call comes out exactly like
    // a freshly-allocated Mzd::matmul — the invariant this whole
    // family of tests exists to protect.
    let n = 16384;
    let a1 = Mzd::random(n, n, 0xC1);
    let b1 = Mzd::random(n, n, 0xC2);
    let a2 = Mzd::random(n, n, 0xD1);
    let b2 = Mzd::random(n, n, 0xD2);

    let mut dst = Mzd::new(n, n);
    Mzd::matmul_into(&mut dst, &a1, &b1);
    Mzd::matmul_into(&mut dst, &a2, &b2);

    let want = Mzd::matmul(&a2, &b2);
    assert_eq!(dst.as_words(), want.as_words());
}

#[cfg(feature = "parallel")]
#[test]
fn strassen_view_parallel_leaf_into_reused_dst_matches_schoolbook() {
    use m4rst::matmul::strassen::matmul_strassen_view_parallel_leaf;
    let n = 512;
    let a1 = Mzd::random(n, n, 0xE1);
    let b1 = Mzd::random(n, n, 0xE2);
    let a2 = Mzd::random(n, n, 0xF1);
    let b2 = Mzd::random(n, n, 0xF2);

    let mut dst = Mzd::new(n, n);
    {
        let mut v = dst.full_view_mut();
        matmul_strassen_view_parallel_leaf(&mut v, &a1.full_view(), &b1.full_view(), 128);
    }
    dst.fill_zero();
    {
        let mut v = dst.full_view_mut();
        matmul_strassen_view_parallel_leaf(&mut v, &a2.full_view(), &b2.full_view(), 128);
    }
    let want = schoolbook_matmul(&a2, &b2);
    assert_eq!(dst.as_words(), want.as_words());
}

#[cfg(feature = "parallel")]
#[test]
fn strassen_peeled_parallel_leaf_into_reused_dst_matches_schoolbook() {
    use m4rst::matmul::strassen::matmul_strassen_peeled_parallel_leaf_into;
    // Non-128-aligned shape so the peeling path (not just the inner
    // Strassen core) is exercised.
    let (r, k, c) = (300usize, 300usize, 300usize);
    let a1 = Mzd::random(r, k, 0x11);
    let b1 = Mzd::random(k, c, 0x12);
    let a2 = Mzd::random(r, k, 0x21);
    let b2 = Mzd::random(k, c, 0x22);

    let mut dst = Mzd::new(r, c);
    matmul_strassen_peeled_parallel_leaf_into(&mut dst, &a1, &b1); // populates dst
    matmul_strassen_peeled_parallel_leaf_into(&mut dst, &a2, &b2); // reuse: must fully overwrite

    let want = schoolbook_matmul(&a2, &b2);
    assert_eq!(dst.as_words(), want.as_words());
}
