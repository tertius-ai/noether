// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Rigorous-enclosure contract for `spectrum::*` exercised through
//! sparb's *public* surface (no `crate::` access). The in-module
//! tests pin individual helpers; this file pins the externally-
//! observable behaviour.
//!
//! Three layers:
//!
//! - **Containment** — the returned ball provably contains the true
//!   value, on examples where the true value is hand-known or
//!   tightly-bounded by analytic argument (identity, diagonal, small
//!   symmetric, the Hilbert matrix).
//! - **Tightness** — radius decays under higher precision / more
//!   iterations. Honest power iteration: pay more, get tighter.
//! - **Cross-consistency** — `lambda_min_enclosure ≤
//!   operator_2_norm_enclosure` on the same matrix.

use sparb::spectrum::{lambda_min_enclosure, operator_2_norm_enclosure, SpectrumOptions};
use sparb::CsrArb;

/// `(row, col, mid, rad)` quad with radius 0.
fn e(r: usize, c: usize, v: f64) -> (usize, usize, f64, f64) {
    (r, c, v, 0.0)
}

/// Symmetrise a list of (r, c, v) triples — emits both `(r, c, v)`
/// and `(c, r, v)` when `r != c`. For building symmetric test
/// matrices from upper-triangle data.
fn symm(triples: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64, f64)> {
    let mut out = Vec::with_capacity(triples.len() * 2);
    for &(r, c, v) in triples {
        out.push((r, c, v, 0.0));
        if r != c {
            out.push((c, r, v, 0.0));
        }
    }
    out
}

// ── Containment ─────────────────────────────────────────────────────

#[test]
fn op_norm_encloses_hilbert_5() {
    // Hilbert(5): H[i,j] = 1/(i+j+1). Symmetric, SPD, ill-conditioned.
    // λ_max ≈ 1.5670506910982306 (well-known, from Wolfram Alpha).
    let n = 5;
    let mut triples = Vec::new();
    for i in 0..n {
        for j in 0..n {
            triples.push((i, j, 1.0 / ((i + j + 1) as f64), 0.0));
        }
    }
    let h = CsrArb::from_coo_f64(n, n, 128, &triples);
    let opts = SpectrumOptions {
        max_iters: 500,
        rel_tol: 1e-10,
        seed: 12345,
    };
    let n2 = operator_2_norm_enclosure(&h, &opts);
    let true_lam_max = 1.567_050_691_098_230_6_f64;
    assert!(
        n2.lower_endpoint(128) <= true_lam_max,
        "lb={} should be <= true λ_max={}",
        n2.lower_endpoint(128),
        true_lam_max
    );
    assert!(
        n2.upper_endpoint(128) >= true_lam_max,
        "ub={} should be >= true λ_max={}",
        n2.upper_endpoint(128),
        true_lam_max
    );
}

#[test]
fn lambda_min_encloses_hilbert_5() {
    // λ_min(H_5) ≈ 3.287928772171585e-06. The most demanding part of
    // the containment contract — the lower bound is what FEM
    // a-posteriori consumes, and it must be ≤ the true value.
    let n = 5;
    let mut triples = Vec::new();
    for i in 0..n {
        for j in 0..n {
            triples.push((i, j, 1.0 / ((i + j + 1) as f64), 0.0));
        }
    }
    let h = CsrArb::from_coo_f64(n, n, 128, &triples);
    let opts = SpectrumOptions {
        max_iters: 500,
        rel_tol: 1e-10,
        seed: 12345,
    };
    let lm = lambda_min_enclosure(&h, &opts);
    let true_lam_min = 3.287_928_772_171_585e-6_f64;
    assert!(
        lm.lower_endpoint(128) <= true_lam_min,
        "lb={} should be <= true λ_min={}",
        lm.lower_endpoint(128),
        true_lam_min
    );
    assert!(
        lm.upper_endpoint(128) >= true_lam_min,
        "ub={} should be >= true λ_min={}",
        lm.upper_endpoint(128),
        true_lam_min
    );
}

#[test]
fn op_norm_encloses_tridiagonal_laplacian() {
    // 1D Laplacian, n=10: tridiagonal with 2 on diag, -1 off-diag.
    // Eigenvalues are 2 - 2 cos(k π / (n+1)) for k = 1..n.
    // λ_max = 2 - 2 cos(n π / (n+1)) ≈ 3.918985...
    let n = 10;
    let mut tri = Vec::new();
    for i in 0..n {
        tri.push((i, i, 2.0));
        if i + 1 < n {
            tri.push((i, i + 1, -1.0));
        }
    }
    let lap = CsrArb::from_coo_f64(n, n, 128, &symm(&tri));
    let opts = SpectrumOptions {
        max_iters: 1000,
        rel_tol: 1e-8,
        seed: 1234,
    };
    let n2 = operator_2_norm_enclosure(&lap, &opts);
    let pi = std::f64::consts::PI;
    let true_lam_max = 2.0 - 2.0 * (n as f64 * pi / ((n + 1) as f64)).cos();
    assert!(n2.lower_endpoint(128) <= true_lam_max);
    assert!(n2.upper_endpoint(128) >= true_lam_max);
}

#[test]
fn lambda_min_tridiagonal_laplacian() {
    // Same matrix as above; λ_min = 2 - 2 cos(π/(n+1)) ≈ 0.08101...
    let n = 10;
    let mut tri = Vec::new();
    for i in 0..n {
        tri.push((i, i, 2.0));
        if i + 1 < n {
            tri.push((i, i + 1, -1.0));
        }
    }
    let lap = CsrArb::from_coo_f64(n, n, 128, &symm(&tri));
    let opts = SpectrumOptions {
        max_iters: 1000,
        rel_tol: 1e-8,
        seed: 1234,
    };
    let lm = lambda_min_enclosure(&lap, &opts);
    let pi = std::f64::consts::PI;
    let true_lam_min = 2.0 - 2.0 * (pi / ((n + 1) as f64)).cos();
    assert!(
        lm.lower_endpoint(128) <= true_lam_min,
        "lb={} should be <= true λ_min={}",
        lm.lower_endpoint(128),
        true_lam_min
    );
    assert!(
        lm.upper_endpoint(128) >= true_lam_min,
        "ub={} should be >= true λ_min={}",
        lm.upper_endpoint(128),
        true_lam_min
    );
}

#[test]
fn op_norm_balls_propagate_input_uncertainty() {
    // Diagonal with one [5 ± 0.1] entry: ‖A‖₂ = |true value of that
    // entry|, which is in [4.9, 5.1]. The output ball must enclose
    // every value in [4.9, 5.1].
    let a = CsrArb::from_coo_f64(
        3,
        3,
        128,
        &[(0, 0, 5.0, 0.1), (1, 1, 1.0, 0.0), (2, 2, 2.0, 0.0)],
    );
    let n2 = operator_2_norm_enclosure(&a, &SpectrumOptions::default());
    assert!(n2.lower_endpoint(128) <= 4.9);
    assert!(n2.upper_endpoint(128) >= 5.1);
}

// ── Tightness ────────────────────────────────────────────────────────

#[test]
fn op_norm_radius_decays_with_iterations() {
    // [[2, 1], [1, 2]]: ‖·‖₂ = 3. More iters → tighter sandwich.
    let a = CsrArb::from_coo_f64(
        2,
        2,
        128,
        &[e(0, 0, 2.0), e(0, 1, 1.0), e(1, 0, 1.0), e(1, 1, 2.0)],
    );
    let opts_few = SpectrumOptions {
        max_iters: 1,
        rel_tol: 0.0,
        seed: 1,
    };
    let opts_many = SpectrumOptions {
        max_iters: 100,
        rel_tol: 0.0,
        seed: 1,
    };
    let n2_few = operator_2_norm_enclosure(&a, &opts_few);
    let n2_many = operator_2_norm_enclosure(&a, &opts_many);
    // More iters never widen the sandwich (running maximum on lower
    // side, fixed upper). Allow tiny FLINT-rounding slack.
    assert!(
        n2_many.upper_endpoint(128) - n2_many.lower_endpoint(128)
            <= n2_few.upper_endpoint(128) - n2_few.lower_endpoint(128) + 1e-12
    );
}

#[test]
fn op_norm_converges_to_eigenvalue() {
    // [[2, 1], [1, 2]]: ‖·‖₂ = 3. With enough iterations and tight
    // tolerance, the lower endpoint of the enclosure should be
    // within 1% of 3 (the Gershgorin upper is exactly 3 for this
    // particular matrix because the dominant Gershgorin disk is
    // [2-1, 2+1] = [1, 3], so the upper of ‖A‖_∞ equals 3 — the
    // sandwich is genuinely tight).
    let a = CsrArb::from_coo_f64(
        2,
        2,
        128,
        &[e(0, 0, 2.0), e(0, 1, 1.0), e(1, 0, 1.0), e(1, 1, 2.0)],
    );
    let opts = SpectrumOptions {
        max_iters: 200,
        rel_tol: 1e-12,
        seed: 7,
    };
    let n2 = operator_2_norm_enclosure(&a, &opts);
    assert!(
        n2.lower_endpoint(128) >= 3.0 * 0.99,
        "after convergence lb={} should be near 3",
        n2.lower_endpoint(128)
    );
    assert!(n2.upper_endpoint(128) >= 3.0);
}

// ── Cross-consistency ────────────────────────────────────────────────

#[test]
fn lambda_min_is_le_op_norm_for_spd_matrices() {
    // Random-ish SPD matrix: I + outer product of small ones.
    // λ_min should be ≥ 1 (positive shift), λ_max larger.
    let n = 4;
    let mut tri = vec![];
    for i in 0..n {
        tri.push((i, i, 1.5));
    }
    tri.push((0, 1, 0.3));
    tri.push((0, 2, 0.2));
    tri.push((1, 2, 0.1));
    tri.push((2, 3, 0.4));
    let a = CsrArb::from_coo_f64(n, n, 128, &symm(&tri));

    let opts = SpectrumOptions {
        max_iters: 200,
        rel_tol: 1e-8,
        seed: 99,
    };
    let lm = lambda_min_enclosure(&a, &opts);
    let n2 = operator_2_norm_enclosure(&a, &opts);
    assert!(
        lm.upper_endpoint(128) <= n2.upper_endpoint(128) + 1e-12,
        "λ_min upper {} should be ≤ ‖A‖₂ upper {}",
        lm.upper_endpoint(128),
        n2.upper_endpoint(128)
    );
}

// ── FEM-shaped sanity ────────────────────────────────────────────────

#[test]
fn lambda_min_recovers_for_2d_5point_laplacian_3x3() {
    // 2D 5-point Laplacian on a 3×3 interior grid (9 DOF total).
    // Diagonal 4, off-diagonal -1 for nearest neighbours.
    // λ_min = 4 - 2 cos(π/4) - 2 cos(π/4) = 4 - 2√2 ≈ 1.17157287525...
    let n = 3;
    let dof = n * n;
    let mut tri = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            tri.push((k, k, 4.0));
            if i + 1 < n {
                let k2 = (i + 1) * n + j;
                tri.push((k, k2, -1.0));
            }
            if j + 1 < n {
                let k2 = i * n + (j + 1);
                tri.push((k, k2, -1.0));
            }
        }
    }
    let lap = CsrArb::from_coo_f64(dof, dof, 128, &symm(&tri));
    let opts = SpectrumOptions {
        max_iters: 1000,
        rel_tol: 1e-8,
        seed: 42,
    };
    let lm = lambda_min_enclosure(&lap, &opts);
    let true_lam_min = 4.0 - 2.0 * 2.0_f64.sqrt(); // ≈ 1.17157
    assert!(
        lm.lower_endpoint(128) <= true_lam_min,
        "lb={} should be ≤ true λ_min={}",
        lm.lower_endpoint(128),
        true_lam_min
    );
    assert!(
        lm.upper_endpoint(128) >= true_lam_min,
        "ub={} should be ≥ true λ_min={}",
        lm.upper_endpoint(128),
        true_lam_min
    );
}
