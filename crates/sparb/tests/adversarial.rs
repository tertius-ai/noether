// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial inputs — the edge cases a kernel must survive.
//!
//! Random fuzz finds random bugs; structural fuzz finds structural
//! bugs. The cases below pin sparb's behaviour on the inputs that
//! would crash, NaN-poison, or silently corrupt a less careful
//! kernel: empty matrices, extreme aspect ratios, the FLINT minimum
//! `prec=2`, balls at the boundaries of FLINT's representable
//! magnitudes, NaN / Inf inputs, and the all-zero-but-not-exact-
//! zero radii regime that exposes whether radius accumulation
//! actually happens.
//!
//! Each test is small (5–15 lines) and pins one corner. New entries
//! land whenever a bug shows up in the wild — that's the convention
//! `crates/AGENTS.md` lays out for adversarial-sweep coverage.

use sparb::{Arb, CsrArb, SparseRingStorage};

// ── empty / degenerate shapes ─────────────────────────────────────────

#[test]
fn empty_matrix_zero_rows() {
    let a = CsrArb::zeros(0, 5, 53);
    assert_eq!((a.rows(), a.cols(), a.nnz()), (0, 5, 0));
    let x: Vec<Arb> = (0..5).map(|_| Arb::zero()).collect();
    let y = a.spmv(&x);
    assert_eq!(y.len(), 0);
}

#[test]
fn empty_matrix_zero_cols() {
    let a = CsrArb::zeros(5, 0, 53);
    assert_eq!((a.rows(), a.cols(), a.nnz()), (5, 0, 0));
    // x must have length 0 since cols == 0.
    let y = a.spmv(&[]);
    assert_eq!(y.len(), 5);
    for b in &y {
        assert!(b.is_exact_zero());
    }
}

#[test]
fn empty_matrix_zero_zero() {
    let a = CsrArb::zeros(0, 0, 53);
    assert_eq!((a.rows(), a.cols(), a.nnz()), (0, 0, 0));
    let y = a.spmv(&[]);
    assert_eq!(y.len(), 0);
}

#[test]
fn singleton_1x1_matrix() {
    let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 7.0, 0.0)]);
    let y = a.spmv(&[Arb::from_f64(3.0)]);
    assert_eq!(y[0].midpoint(), 21.0);
    assert_eq!(y[0].radius(53), 0.0);
}

// ── extreme aspect ratios ─────────────────────────────────────────────

#[test]
fn very_wide_1_by_1000() {
    // 1×1000 — single row, dense. Stresses the per-row inner loop
    // without exercising the row-loop machinery.
    let triples: Vec<(usize, usize, f64, f64)> = (0..1000).map(|c| (0, c, 1.0, 0.0)).collect();
    let a = CsrArb::from_coo_f64(1, 1000, 53, &triples);
    let x: Vec<Arb> = (0..1000).map(|_| Arb::from_f64(1.0)).collect();
    let y = a.spmv(&x);
    assert_eq!(y.len(), 1);
    assert_eq!(y[0].midpoint(), 1000.0);
}

#[test]
fn very_tall_1000_by_1() {
    // 1000×1 — every row has at most one entry. Stresses row-loop
    // overhead with trivial per-row work.
    let triples: Vec<(usize, usize, f64, f64)> =
        (0..1000).map(|r| (r, 0, (r as f64) + 1.0, 0.0)).collect();
    let a = CsrArb::from_coo_f64(1000, 1, 53, &triples);
    let y = a.spmv(&[Arb::from_f64(1.0)]);
    assert_eq!(y.len(), 1000);
    for (r, b) in y.iter().enumerate() {
        assert_eq!(b.midpoint(), (r as f64) + 1.0);
    }
}

#[test]
fn prime_shape_17_by_23() {
    // Prime dimensions catch bugs that hide behind power-of-two
    // strides and even-row optimisations.
    let mut triples = Vec::new();
    for r in 0..17 {
        for c in 0..23 {
            if (r * c) % 5 == 0 {
                triples.push((r, c, (r + c) as f64, 0.0));
            }
        }
    }
    let a = CsrArb::from_coo_f64(17, 23, 64, &triples);
    let x: Vec<Arb> = (0..23).map(|c| Arb::from_f64((c as f64) * 0.1)).collect();
    let y = a.spmv(&x);
    assert_eq!(y.len(), 17);
}

// ── precision boundaries ──────────────────────────────────────────────

#[test]
fn prec_2_flint_minimum() {
    // `prec=2` is FLINT's lower bound — 2 bits of mantissa, almost
    // no precision. SpMV should still produce a *rigorous* ball
    // that contains the true value; the radius will just be wide.
    let a = CsrArb::from_coo_f64(
        1,
        3,
        2,
        &[(0, 0, 2.0, 0.0), (0, 1, 3.0, 0.0), (0, 2, 5.0, 0.0)],
    );
    let x: Vec<Arb> = vec![Arb::from_f64(1.0), Arb::from_f64(1.0), Arb::from_f64(1.0)];
    let y = a.spmv(&x);
    // True answer: 10. Output ball must contain it.
    let (m, r) = y[0].to_pair(64);
    assert!(
        m - r <= 10.0 && 10.0 <= m + r,
        "prec=2 ball [{m} ± {r}] doesn't contain true value 10"
    );
}

#[test]
#[should_panic(expected = "prec must be >= 2")]
fn prec_1_panics() {
    let _ = CsrArb::zeros(2, 2, 1);
}

#[test]
fn prec_4096_high_precision_is_tight() {
    // High precision: a chain of exact mults should give exact (or
    // near-exact) results. Use 1/3 in arb — irrational binary — at
    // very high precision to check the radius stays tight.
    let third = 1.0_f64 / 3.0; // f64 approx of 1/3
    let a = CsrArb::from_coo_f64(1, 1, 4096, &[(0, 0, third, 0.0)]);
    let y = a.spmv(&[Arb::from_f64(3.0)]);
    // f64 third * 3 == 1.0 exactly in f64. Output ball:
    let (_m, r) = y[0].to_pair(64);
    assert!(
        r <= 1e-15,
        "prec=4096 result has surprisingly wide radius {r}"
    );
}

// ── ball regime boundaries ────────────────────────────────────────────

#[test]
fn huge_radius_entry() {
    // Single entry with an astronomical radius. The ball spans
    // ±1e30, SpMV scales by an exact f64 → output radius is
    // 1e30 * |x[0]|.
    let a = CsrArb::from_coo_f64(1, 1, 64, &[(0, 0, 0.0, 1e30)]);
    let y = a.spmv(&[Arb::from_f64(2.0)]);
    let (m, r) = y[0].to_pair(64);
    assert_eq!(m, 0.0);
    assert!(r >= 2e30, "huge-radius SpMV: expected r ≥ 2e30, got {r}");
}

#[test]
fn tiny_radius_entry_doesnt_collapse_to_exact() {
    // A ball with rad = 1e-300 (sub-subnormal-adjacent) should
    // still be representable. The entry must be stored — it's not
    // the exact zero ball.
    let a = CsrArb::from_coo_f64(1, 1, 256, &[(0, 0, 0.0, 1e-300)]);
    assert_eq!(a.nnz(), 1);
    let y = a.spmv(&[Arb::from_f64(1.0)]);
    let (m, r) = y[0].to_pair(256);
    assert_eq!(m, 0.0);
    assert!(r > 0.0, "tiny radius should survive SpMV, got 0");
}

#[test]
fn ball_straddling_zero_is_stored() {
    // mid=0 ± rad=0.1 — the ball [-0.1, 0.1] contains zero but is
    // NOT the exact zero ball. Storing it as zero would lose the
    // rigorous enclosure.
    let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 0.0, 0.1)]);
    assert_eq!(a.nnz(), 1);
    assert!(!a.get(0, 0).unwrap().is_exact_zero());
}

#[test]
fn all_balls_contain_zero_radius_accumulates() {
    // 5 entries, all mid=0 ± 0.2. SpMV against x=[1,1,1,1,1]:
    // every term is the zero ball with rad 0.2. Sum: midpoint 0,
    // radius ≥ 5 * 0.2 = 1.0 (cross terms with exact x just
    // multiply radii by |x|=1).
    let triples: Vec<(usize, usize, f64, f64)> = (0..5).map(|c| (0, c, 0.0, 0.2)).collect();
    let a = CsrArb::from_coo_f64(1, 5, 64, &triples);
    let x: Vec<Arb> = (0..5).map(|_| Arb::from_f64(1.0)).collect();
    let y = a.spmv(&x);
    let (m, r) = y[0].to_pair(64);
    assert_eq!(m, 0.0);
    assert!(r >= 1.0, "expected accumulated radius ≥ 1.0, got {r}");
}

// ── exact zero ball is dropped, only exact zero ───────────────────────

#[test]
fn exact_zero_triples_are_dropped() {
    let a = CsrArb::from_coo_f64(3, 3, 53, &[(0, 0, 0.0, 0.0), (1, 1, 0.0, 0.0)]);
    assert!(a.is_zero());
    assert_eq!(a.nnz(), 0);
}

#[test]
fn duplicate_triples_cancelling_to_exact_zero_drop_with_kept_radius_nearby() {
    // (0, 0, 5, 0.1) and (0, 0, -5, 0.1) sum to a ball with mid=0
    // and rad ≥ 0.2 — NOT the exact zero ball. Must be kept.
    let a = CsrArb::from_coo_f64(1, 1, 64, &[(0, 0, 5.0, 0.1), (0, 0, -5.0, 0.1)]);
    assert_eq!(a.nnz(), 1);
    let b = a.get(0, 0).unwrap();
    assert_eq!(b.midpoint(), 0.0);
    assert!(b.radius(64) >= 0.2);
}

// ── NaN / Inf propagation (documented behaviour: poison the ball) ─────

#[test]
fn nan_midpoint_propagates() {
    // FLINT's arb_set_d(_, NaN) sets the arb to a NaN-arb. SpMV
    // propagates that into the output. The library doesn't try to
    // sanitise — NaN-in → NaN-out, matching IEEE 754 conventions
    // and Arb's own behaviour. Callers responsible for guarding
    // inputs.
    let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, f64::NAN, 0.0)]);
    let y = a.spmv(&[Arb::from_f64(1.0)]);
    let m = y[0].midpoint();
    assert!(m.is_nan(), "NaN midpoint expected, got {m}");
}

#[test]
fn inf_midpoint_propagates() {
    let a = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, f64::INFINITY, 0.0)]);
    let y = a.spmv(&[Arb::from_f64(2.0)]);
    let m = y[0].midpoint();
    assert!(m.is_infinite() && m > 0.0, "+Inf expected, got {m}");
}

#[test]
#[should_panic(expected = "radius must be finite and >= 0")]
fn negative_radius_panics_at_ingest() {
    // Defensive: a negative radius is meaningless. Construction
    // should panic loudly rather than silently propagate.
    let _ = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 1.0, -0.5)]);
}

#[test]
#[should_panic(expected = "radius must be finite and >= 0")]
fn nan_radius_panics_at_ingest() {
    let _ = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 1.0, f64::NAN)]);
}

#[test]
#[should_panic(expected = "radius must be finite and >= 0")]
fn inf_radius_panics_at_ingest() {
    let _ = CsrArb::from_coo_f64(1, 1, 53, &[(0, 0, 1.0, f64::INFINITY)]);
}

// ── SpMV dimension mismatch panics rather than misbehaves ─────────────

#[test]
#[should_panic(expected = "x has length")]
fn spmv_x_too_short_panics() {
    let a = CsrArb::zeros(3, 5, 53);
    let x: Vec<Arb> = (0..4).map(|_| Arb::from_f64(1.0)).collect();
    let _ = a.spmv(&x);
}

#[test]
#[should_panic(expected = "x has length")]
fn spmv_x_too_long_panics() {
    let a = CsrArb::zeros(3, 5, 53);
    let x: Vec<Arb> = (0..6).map(|_| Arb::from_f64(1.0)).collect();
    let _ = a.spmv(&x);
}
