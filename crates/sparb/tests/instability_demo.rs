// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Numerical instability — what sparb buys over plain `f64`.
//!
//! This is the load-bearing demonstration of *why* a ball-arithmetic
//! sparse matrix-vector kernel exists. The single-row matrix
//!
//! ```text
//! A = [[1e16, 1, -1e16]],   x = [1, 1, 1]
//! ```
//!
//! has exact answer `A · x = 1e16 + 1 + (-1e16) = 1`. Plain `f64`
//! SpMV accumulates left-to-right and gets exactly `0.0` — the `+1`
//! is silently absorbed when added into `1e16` (because `1e16 + 1`
//! is not representable in 53-bit mantissa and rounds back to `1e16`),
//! then the `-1e16` wipes the accumulator. **No warning, no error,
//! catastrophically wrong answer.** This is the classic
//! catastrophic-cancellation pattern; the example is small enough
//! to trace by hand.
//!
//! sparb's `CsrArb::spmv` runs the same arithmetic but tracks the
//! lost bit in the ball's `rad` field. At the f64-equivalent
//! precision (`prec = 53`) the result is a ball whose midpoint is
//! `0` but whose radius is large enough to **rigorously contain
//! the true answer 1**. At slightly higher precision (`prec >= 64`)
//! the intermediate `1e16 + 1` is exactly representable and the
//! result is the exact midpoint `1` with zero radius. The kernel
//! never lies: it either gives the right answer or honestly admits
//! the uncertainty.
//!
//! The tests below pin three properties:
//!   1. **f64 catastrophically fails.** SpMV returns `0.0` exactly.
//!   2. **Every sparb result rigorously contains the true value 1**
//!      across `prec ∈ {24, 53, 64, 80, 128, 256}`.
//!   3. **Radius is monotonically non-increasing in `prec`.** Higher
//!      working precision can only ever tighten the enclosure, never
//!      widen it.
//!
//! The `print_instability_table` test (marked `#[ignore]`) prints a
//! human-readable markdown table — run with
//! `cargo test --release --test instability_demo -- --ignored --nocapture`.
//! The README's "Numerical instability" section quotes the table verbatim.

#[path = "common/mod.rs"]
mod common;

use common::csr_f64::CsrF64;
use sparb::{Arb, CsrArb};

/// The catastrophic-cancellation row: `[1e16, 1, -1e16]`.
const ROW: &[(usize, usize, f64)] = &[(0, 0, 1e16), (0, 1, 1.0), (0, 2, -1e16)];

/// `x = [1, 1, 1]` so the SpMV reduces to summing the row.
const X: [f64; 3] = [1.0, 1.0, 1.0];

/// True mathematical answer: `1e16 + 1 + (-1e16) = 1`.
const TRUE_ANSWER: f64 = 1.0;

/// The precisions swept across all tests. `24` is below f64; `53`
/// is f64-equivalent; `64+` is enough to make the intermediate sum
/// exact.
const PREC_SWEEP: &[i64] = &[24, 53, 64, 80, 128, 256];

fn triples_with_rad_zero() -> Vec<(usize, usize, f64, f64)> {
    ROW.iter().map(|&(r, c, v)| (r, c, v, 0.0)).collect()
}

fn x_as_arb() -> Vec<Arb> {
    X.iter().map(|&v| Arb::from_f64(v)).collect()
}

// ── property #1: f64 catastrophically fails ───────────────────────────

#[test]
fn f64_catastrophically_returns_zero() {
    let a = CsrF64::from_triples(1, 3, ROW.to_vec());
    let y = a.spmv(&X);
    assert_eq!(
        y[0], 0.0,
        "f64 SpMV expected to catastrophically return 0.0; got {}",
        y[0]
    );
    assert_ne!(
        y[0], TRUE_ANSWER,
        "f64 result happens to be correct — pick a more aggressive fixture"
    );
}

// ── property #2: every sparb enclosure contains the true value ────────

#[test]
fn sparb_ball_contains_true_value_at_every_prec() {
    let triples = triples_with_rad_zero();
    let xs = x_as_arb();
    for &prec in PREC_SWEEP {
        let a = CsrArb::from_coo_f64(1, 3, prec, &triples);
        let y = a.spmv(&xs);
        let (mid, rad) = y[0].to_pair(prec);
        let lo = mid - rad;
        let hi = mid + rad;
        assert!(
            lo <= TRUE_ANSWER && TRUE_ANSWER <= hi,
            "prec={prec}: ball [{lo}, {hi}] (mid={mid}, rad={rad}) does \
             NOT contain the true answer {TRUE_ANSWER}"
        );
    }
}

// ── property #3: radius is monotonically non-increasing in prec ───────

#[test]
fn radius_monotonically_non_increasing_in_prec() {
    let triples = triples_with_rad_zero();
    let xs = x_as_arb();
    // Read radii at a fixed high readback precision so the reported
    // numbers reflect the *internal* radius without f64-ulp floor
    // noise dominating the small-prec → smaller-prec comparison. We
    // care about the *computation*'s radius monotonicity, not the
    // materialisation's.
    let readback_prec = 128;
    let mut prev = f64::INFINITY;
    for &prec in PREC_SWEEP {
        let a = CsrArb::from_coo_f64(1, 3, prec, &triples);
        let y = a.spmv(&xs);
        let rad = y[0].radius(readback_prec);
        assert!(
            rad <= prev,
            "prec={prec}: radius {rad} > previous radius {prev} — \
             monotonicity broken"
        );
        prev = rad;
    }
}

// ── property #4: sparb at modest extra precision is exact ─────────────

#[test]
fn sparb_at_prec_64_is_exact() {
    // 1e16 + 1 fits in 54 bits, so 64-bit mantissa is enough to
    // compute the SpMV exactly. The output ball collapses to the
    // exact answer with zero radius.
    let triples = triples_with_rad_zero();
    let xs = x_as_arb();
    let a = CsrArb::from_coo_f64(1, 3, 64, &triples);
    let y = a.spmv(&xs);
    assert_eq!(y[0].midpoint(), TRUE_ANSWER);
    assert_eq!(y[0].radius(128), 0.0);
}

// ── headline demo: print the table for the README ─────────────────────

/// Prints a markdown table summarising the instability demo at every
/// precision in the sweep, plus the plain-f64 catastrophe. Ignored
/// by default so `cargo test` stays quiet; run with
/// `cargo test --release --test instability_demo -- --ignored --nocapture`.
#[test]
#[ignore]
fn print_instability_table() {
    let f64_a = CsrF64::from_triples(1, 3, ROW.to_vec());
    let f64_y = f64_a.spmv(&X);

    println!();
    println!("## Numerical instability — sparb vs f64");
    println!();
    println!("Single-row SpMV on `A = [[1e16, 1, -1e16]]`, `x = [1, 1, 1]`.");
    println!("True answer: `A · x = 1`.");
    println!();
    println!(
        "- plain f64 SpMV returns **{}** (catastrophic — no warning).",
        f64_y[0]
    );
    println!();
    println!("| prec | sparb midpoint | sparb radius | sparb encloses true? |");
    println!("|-----:|---------------:|-------------:|:--------------------:|");

    let triples = triples_with_rad_zero();
    let xs = x_as_arb();
    for &prec in PREC_SWEEP {
        let a = CsrArb::from_coo_f64(1, 3, prec, &triples);
        let y = a.spmv(&xs);
        let (mid, rad) = y[0].to_pair(128);
        let encloses = (mid - rad) <= TRUE_ANSWER && TRUE_ANSWER <= (mid + rad);
        println!(
            "| {prec:>4} | {mid:>14.6e} | {rad:>12.6e} | {} |",
            if encloses { "✓" } else { "✗" }
        );
    }
    println!();
}
