// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The rigorous-enclosure contract — why sparb exists.
//!
//! A `CsrArb` SpMV does not *approximate* `A · x`; it returns a ball
//! that **provably contains** the true `A · x`, accounting for both
//! the input uncertainty and every rounding step. These tests pin
//! that guarantee on worked examples.

use sparb::{Arb, CsrArb};

/// A ball `[mid ± rad]` contains a real value `v` iff
/// `mid - rad <= v <= mid + rad`.
fn ball_contains(ball: &Arb, v: f64, prec: i64) -> bool {
    let (mid, rad) = ball.to_pair(prec);
    (mid - rad) <= v && v <= (mid + rad)
}

#[test]
fn spmv_encloses_true_value_under_input_uncertainty() {
    // A = [[2, 0], [1, 3]] but each entry carries ±0.01 uncertainty.
    // x = [1 ± 0.1, 2 ± 0.1].  True (midpoint) value: A·x = [2, 7].
    let prec = 80;
    let a = CsrArb::from_coo_f64(
        2,
        2,
        prec,
        &[(0, 0, 2.0, 0.01), (1, 0, 1.0, 0.01), (1, 1, 3.0, 0.01)],
    );
    let x = vec![Arb::from_mid_rad(1.0, 0.1), Arb::from_mid_rad(2.0, 0.1)];
    let y = a.spmv(&x);

    // The exact-midpoint products lie inside the output enclosures.
    assert!(ball_contains(&y[0], 2.0, prec));
    assert!(ball_contains(&y[1], 7.0, prec));

    // And so does every value reachable within the input balls — e.g.
    // the worst-case corners of row 0: 2*x0 with x0 ∈ [0.9, 1.1] and
    // the entry in [1.99, 2.01] ⇒ product ∈ [1.791, 2.211].
    assert!(ball_contains(&y[0], 1.791, prec));
    assert!(ball_contains(&y[0], 2.211, prec));

    // The enclosure is honest, not trivially wide: row 0's radius is
    // on the order of the input uncertainty, not unbounded.
    assert!(y[0].radius(prec) < 0.5);
}

#[test]
fn exact_inputs_give_exact_outputs() {
    // Small integers are exactly representable: a CsrArb of integers
    // times an integer vector has radius exactly 0 — sparb does not
    // manufacture uncertainty that isn't there.
    let a = CsrArb::from_coo_f64(
        2,
        2,
        53,
        &[(0, 0, 3.0, 0.0), (0, 1, 4.0, 0.0), (1, 1, 5.0, 0.0)],
    );
    let x = vec![Arb::from_f64(2.0), Arb::from_f64(6.0)];
    let y = a.spmv(&x);
    assert_eq!(y[0].midpoint(), 30.0); // 3*2 + 4*6
    assert_eq!(y[1].midpoint(), 30.0); // 5*6
    assert_eq!(y[0].radius(53), 0.0);
    assert_eq!(y[1].radius(53), 0.0);
}

#[test]
fn higher_precision_tightens_inexact_enclosures() {
    // 1/7 is not binary-representable. The same SpMV at higher
    // working precision yields a tighter (never wider) enclosure of
    // the true value — the rigorous form of "more bits, less error".
    let seventh = 1.0_f64 / 7.0;
    let true_value = 7.0 * seventh; // ≈ 1.0, but the f64 midpoint is inexact

    let radius_at = |prec: i64| {
        let a = CsrArb::from_coo_f64(1, 1, prec, &[(0, 0, seventh, 1e-3)]);
        let y = a.spmv(&[Arb::from_f64(7.0)]);
        assert!(ball_contains(&y[0], true_value, prec));
        y[0].radius(prec)
    };

    // Coarser precision is never tighter than finer precision.
    assert!(radius_at(24) >= radius_at(128));
}
