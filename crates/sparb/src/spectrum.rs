// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Rigorous operator-norm and `λ_min` enclosures on [`CsrArb`].
//!
//! This module provides:
//!
//! - [`operator_2_norm_enclosure`] — a rigorous `Arb` enclosure of
//!   `‖A‖₂ = λ_max(A)` for any symmetric `A`. Uses power iteration in
//!   ball arithmetic, sandwiched against the Gershgorin upper bound
//!   `‖A‖_∞` from [`CsrArb::linf_norm`].
//! - [`lambda_min_enclosure`] — a rigorous `Arb` enclosure of
//!   `λ_min(A)` for symmetric `A`, computed by reducing to the
//!   spectrum of the shifted matrix `B := cI − A`.
//!
//! The implementation uses:
//!
//! - **Power iteration**, not Lanczos.
//! - **Sandwich bounds**: variational lower bound on `λ_max` from the
//!   Rayleigh quotient (`R_A(v) ≤ λ_max` for symmetric `A`);
//!   Gershgorin upper bound (`λ_max ≤ ‖A‖_∞`).
//! - **Stopping criterion**: rigorous — halt when the *sandwich's*
//!   relative radius drops below `rel_tol`, or after `max_iters`.
//!   Returns the *current* sandwich, never a tighter claim than the
//!   iteration proves.

use crate::ball::Arb;
use crate::storage::{CsrArb, SparseRingStorage, Triple};

/// Knobs for the spectrum iteration.
///
/// `max_iters` and `rel_tol` together set the cost / tightness
/// tradeoff: the loop halts as soon as *either* limit is reached, and
/// the returned enclosure honestly reports the radius achieved.
#[derive(Clone, Debug)]
pub struct SpectrumOptions {
    /// Maximum power-iteration steps. Each step is one SpMV plus a
    /// few ball-vector operations. The default `64` is enough for
    /// FEM-stiffness-shaped problems to converge to `rel_tol = 1e-6`
    /// at IEEE-double-precision matrices; longer-tailed spectra may
    /// need more.
    pub max_iters: usize,
    /// Stop when `sandwich_radius / max(|sandwich_midpoint|, 1) <
    /// rel_tol`. Default `1e-6` — loose but useful for the FEM
    /// a-posteriori consumer. Pass smaller `rel_tol` for tighter
    /// enclosures; pass `0.0` to disable early stopping and run all
    /// `max_iters` iterations.
    pub rel_tol: f64,
    /// PRNG seed for the initial vector. Deterministic by design —
    /// the same `seed` with the same matrix yields the same iterate
    /// trace, so test regressions are reproducible. Default `0x5eed_5eed`.
    pub seed: u64,
}

impl Default for SpectrumOptions {
    fn default() -> Self {
        Self {
            max_iters: 64,
            rel_tol: 1e-6,
            seed: 0x5eed_5eed,
        }
    }
}

/// Rigorous `Arb` enclosure of the operator 2-norm `‖A‖₂` of a
/// symmetric matrix.
///
/// # Contract
///
/// Returns a ball `B` such that for the true `A` (and every matrix in
/// `A`'s ball-entry enclosure), `‖A‖₂ ∈ B`. For SPD `A`, `‖A‖₂ =
/// λ_max(A)`; for general symmetric `A`, `‖A‖₂ = max_i |λ_i|`. Both
/// cases are handled — power iteration on `A` converges to the
/// eigenpair of largest *magnitude* eigenvalue, and we take the
/// absolute value of the Rayleigh quotient before using it as a
/// lower-bound side.
///
/// # Algorithm
///
/// Power iteration: `v_{k+1} = unit(A v_k)`, with the Rayleigh
/// quotient `μ_k := ⟨v_k, A v_k⟩ / ⟨v_k, v_k⟩` accumulated in ball
/// arithmetic. The variational characterisation gives
/// `|R_A(v_k)| ≤ ‖A‖₂` (any unit `v_k`, symmetric `A`); the
/// Gershgorin / row-sum bound gives `‖A‖₂ ≤ ‖A‖_∞`. The output ball
/// is the [`Arb::union`] of (a) the converged `|μ_k|` ball and (b)
/// the `‖A‖_∞` ball from [`CsrArb::linf_norm`] — guaranteed to
/// contain `‖A‖₂`.
///
/// # Stopping criterion
///
/// Halts after `opts.max_iters` iterations OR when the sandwich's
/// relative radius drops below `opts.rel_tol`. Never claims a
/// tighter enclosure than the iteration proves — the returned ball's
/// radius is the achieved tightness.
///
/// # Cost
///
/// Each iteration: one [`CsrArb::spmv`] (the dominant cost), plus
/// `O(n)` ball arithmetic for two dot products, a divide, and a
/// renormalisation. Memory is `O(n)` ball storage (two length-`n`
/// vectors).
///
/// # Panics
///
/// - `A.is_symmetric()` must hold (checked via ball-identity, not enclosure
///   overlap). This routine does not compute singular values for a
///   non-symmetric matrix.
/// - `A.rows() == A.cols()` — non-square is rejected upstream by
///   `is_symmetric`.
///
/// # Examples
///
/// ```
/// use sparb::{spectrum::{operator_2_norm_enclosure, SpectrumOptions}, CsrArb};
/// // diag(3, 5, 1): symmetric with λ_max = 5.
/// let a = CsrArb::from_coo_f64(3, 3, 64,
///     &[(0, 0, 3.0, 0.0), (1, 1, 5.0, 0.0), (2, 2, 1.0, 0.0)]);
/// let n2 = operator_2_norm_enclosure(&a, &SpectrumOptions::default());
/// // The ball must enclose 5.
/// assert!(n2.lower_endpoint(64) <= 5.0);
/// assert!(n2.upper_endpoint(64) >= 5.0);
/// ```
pub fn operator_2_norm_enclosure(a: &CsrArb, opts: &SpectrumOptions) -> Arb {
    assert!(
        a.is_symmetric(),
        "operator_2_norm_enclosure: matrix must be symmetric"
    );
    let n = a.rows();
    let prec = a.prec();

    // Empty matrix: no eigenvalues, norm is conventionally 0.
    if n == 0 {
        return Arb::zero();
    }

    // Rigorous upper bound on max|λ_i| from Gershgorin / row sums.
    let ub_ball = a.linf_norm();

    // Initial vector: deterministic from opts.seed, normalised in
    // ball arithmetic. The (degenerate) case where the seeded vector
    // is orthogonal to every dominant eigenvector recovers naturally
    // — power iteration drifts onto the dominant subspace via the
    // tiny rounding-error components.
    let mut v = prng_unit_vec(n, opts.seed, prec);
    v = unit_normalize(v, prec);

    // Track the ball whose lower endpoint pushes ‖A‖₂ up most. Each
    // |R_A(v_k)| is a real number ≤ ‖A‖₂; mu_abs_k is a ball
    // containing it; the best lower bound across iterations is what
    // we keep.
    let mut best_lb_ball: Option<Arb> = None;
    let mut best_lo: f64 = f64::NEG_INFINITY;

    // Pre-compute the upper-endpoint f64 of ub_ball once.
    let ub_hi = ub_ball.upper_endpoint(prec);

    for _ in 0..opts.max_iters {
        let w = a.spmv(&v);
        let vw = dot(&v, &w, prec);
        let vv = dot(&v, &v, prec);
        // Degenerate denominator: v collapsed to the zero ball.
        // This can happen if A is rank-deficient and v_k landed on
        // the null space. Stop with the current sandwich.
        if vv.is_exact_zero() {
            break;
        }
        let mu = vw.div(&vv, prec);
        let mu_abs = mu.abs();

        // Update the best lower-side ball — by rigorous lower endpoint.
        let lo = mu_abs.lower_endpoint(prec);
        if lo > best_lo {
            best_lo = lo;
            best_lb_ball = Some(mu_abs);
        }

        // Renormalise for next iter. If w is all-zero (eigenvalue 0
        // along v's direction), bail — no further progress.
        if w.iter().all(|wi| wi.is_exact_zero()) {
            break;
        }
        v = unit_normalize(w, prec);

        // Stopping: relative radius of the sandwich.
        let lb_ball_ref = best_lb_ball.as_ref().unwrap();
        let lb_lo = lb_ball_ref.lower_endpoint(prec);
        let mid = 0.5 * (lb_lo + ub_hi);
        let rad = 0.5 * (ub_hi - lb_lo).max(0.0);
        let denom = mid.abs().max(1.0);
        if rad / denom < opts.rel_tol {
            break;
        }
    }

    // No iterations produced a finite lower-bound estimate (n == 0
    // was already handled, so this is the all-zero-spmv case): fall
    // back to "‖A‖₂ ∈ [0, ub_ball]".
    let lb_ball = best_lb_ball.unwrap_or_else(Arb::zero);
    lb_ball.union(&ub_ball, prec)
}

/// Rigorous `Arb` enclosure of the smallest eigenvalue `λ_min(A)` of
/// a symmetric matrix.
///
/// # Contract
///
/// Returns a ball `B` such that `λ_min(A) ∈ B`. For SPD `A` (the FEM
/// consumer's case), the *lower* endpoint of `B` is what the
/// a-posteriori bound consumes: an underestimate of `λ_min(A)`
/// gives a conservative (loose-but-rigorous) upper bound on the
/// energy-norm error.
///
/// # Algorithm
///
/// Shifted spectrum: pick `c := upper_endpoint(‖A‖_∞)`, a rigorous
/// upper bound on `λ_max(A)`. Let `B := cI − A`. Then `B` is
/// symmetric with `λ_max(B) = c − λ_min(A)`. Apply
/// [`operator_2_norm_enclosure`] to `B`, then subtract from `c`
/// (in ball arithmetic) to recover `λ_min(A)`.
///
/// # Cost
///
/// One [`operator_2_norm_enclosure`] call on the shifted matrix `B`,
/// plus the (`O(n + nnz)`) construction of `B`. The shifted matrix
/// has the same sparsity pattern as `A` plus a diagonal — the same
/// asymptotic SpMV cost per iteration.
///
/// # Panics
///
/// Same as [`operator_2_norm_enclosure`].
///
/// # Examples
///
/// ```
/// use sparb::{spectrum::{lambda_min_enclosure, SpectrumOptions}, CsrArb};
/// // diag(3, 5, 1): symmetric SPD with λ_min = 1.
/// let a = CsrArb::from_coo_f64(3, 3, 64,
///     &[(0, 0, 3.0, 0.0), (1, 1, 5.0, 0.0), (2, 2, 1.0, 0.0)]);
/// let lm = lambda_min_enclosure(&a, &SpectrumOptions::default());
/// assert!(lm.lower_endpoint(64) <= 1.0);
/// assert!(lm.upper_endpoint(64) >= 1.0);
/// ```
pub fn lambda_min_enclosure(a: &CsrArb, opts: &SpectrumOptions) -> Arb {
    assert!(
        a.is_symmetric(),
        "lambda_min_enclosure: matrix must be symmetric"
    );
    let n = a.rows();
    let prec = a.prec();

    if n == 0 {
        return Arb::zero();
    }

    // Rigorous upper bound on λ_max(A), as an exact Arb at the f64
    // upper endpoint of ‖A‖_∞. The exact-Arb posture is deliberate:
    // we want the SHIFT itself to be a precisely known real number,
    // not a ball, so that the subtraction `c − λ_max(B)` doesn't
    // double-count uncertainty in the shift.
    let c = a.linf_norm().upper_endpoint(prec);
    let c_ball = Arb::from_f64(c);

    // Build B := cI − A as a CsrArb.
    let b = shifted_matrix(a, &c_ball);

    let lam_max_b = operator_2_norm_enclosure(&b, opts);

    // λ_min(A) = c − λ_max(B). Ball subtraction: radii add — this is
    // the standard ball-arithmetic propagation.
    c_ball.sub(&lam_max_b, prec)
}

// ── helpers (private) ────────────────────────────────────────────────

/// `⟨u, v⟩` over ball vectors, accumulating with `arb_addmul` at the
/// given working precision.
fn dot(u: &[Arb], v: &[Arb], prec: i64) -> Arb {
    debug_assert_eq!(u.len(), v.len(), "dot: length mismatch");
    let mut acc = Arb::zero();
    for (ui, vi) in u.iter().zip(v.iter()) {
        acc.addmul_assign(ui, vi, prec);
    }
    acc
}

/// `‖v‖₂` over a ball vector.
fn l2_norm(v: &[Arb], prec: i64) -> Arb {
    dot(v, v, prec).sqrt(prec)
}

/// Divide every component of `v` by `‖v‖₂` in ball arithmetic. If the
/// norm enclosure is the *exact* zero ball, returns `v` unchanged —
/// the caller should treat this as a degenerate state and stop.
fn unit_normalize(v: Vec<Arb>, prec: i64) -> Vec<Arb> {
    let nrm = l2_norm(&v, prec);
    if nrm.is_exact_zero() {
        return v;
    }
    v.into_iter().map(|vi| vi.div(&nrm, prec)).collect()
}

/// Deterministic length-`n` vector with f64 components uniformly in
/// `(-1, 1)`, seeded by `seed`. SplitMix64 PRNG — no rand-crate
/// dependency. The components are exact `Arb`s.
fn prng_unit_vec(n: usize, seed: u64, _prec: i64) -> Vec<Arb> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // SplitMix64 step.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to f64 in [-1, 1) with 53 bits of mantissa.
        let bits = z >> 11;
        let unit = (bits as f64) * (1.0 / (1u64 << 53) as f64); // [0, 1)
        let signed = unit * 2.0 - 1.0; // [-1, 1)
        out.push(Arb::from_f64(signed));
    }
    // Pathological case: `n == 1` and the sample happened to be 0
    // (vanishingly unlikely from SplitMix; deterministic fix-up).
    if n == 1 && out[0].is_exact_zero() {
        out[0] = Arb::from_f64(1.0);
    }
    out
}

/// Build `cI − A` as a CSR ball matrix at `A`'s precision. The
/// diagonal of `A` gets `c − A[i,i]`; off-diagonal entries are
/// negated.
fn shifted_matrix(a: &CsrArb, c: &Arb) -> CsrArb {
    let n = a.rows();
    let prec = a.prec();
    // Build `cI` as a length-n diagonal then subtract.
    let diag_triples: Vec<Triple<Arb>> = (0..n).map(|i| (i, i, c.clone())).collect();
    let c_i = CsrArb::from_coo_balls(n, n, prec, diag_triples);
    c_i.sub(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(row, col, mid, rad)` quad with radius 0.
    fn e(r: usize, c: usize, v: f64) -> (usize, usize, f64, f64) {
        (r, c, v, 0.0)
    }

    #[test]
    fn dot_of_unit_vectors() {
        let u = vec![Arb::from_f64(3.0), Arb::from_f64(4.0)];
        let v = vec![Arb::from_f64(1.0), Arb::from_f64(2.0)];
        let d = dot(&u, &v, 64);
        // 3·1 + 4·2 = 11.
        assert_eq!(d.midpoint(), 11.0);
    }

    #[test]
    fn l2_norm_of_3_4() {
        let v = vec![Arb::from_f64(3.0), Arb::from_f64(4.0)];
        let n = l2_norm(&v, 64);
        assert_eq!(n.midpoint(), 5.0);
    }

    #[test]
    fn unit_normalize_yields_unit_norm() {
        let v = vec![Arb::from_f64(3.0), Arb::from_f64(4.0)];
        let u = unit_normalize(v, 64);
        let n = l2_norm(&u, 64);
        // Norm should be ≈ 1 (small radius from the division).
        let (m, r) = n.to_pair(64);
        assert!(m - r <= 1.0 && 1.0 <= m + r);
    }

    #[test]
    fn unit_normalize_zero_vector_is_unchanged() {
        let v = vec![Arb::zero(), Arb::zero()];
        let u = unit_normalize(v, 64);
        assert!(u[0].is_exact_zero());
        assert!(u[1].is_exact_zero());
    }

    #[test]
    fn prng_is_deterministic() {
        let a = prng_unit_vec(5, 42, 64);
        let b = prng_unit_vec(5, 42, 64);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(x.same_ball(y));
        }
    }

    #[test]
    fn shifted_matrix_diagonal() {
        // A = diag(1, 2, 3); c = 5 → cI − A = diag(4, 3, 2).
        let a = CsrArb::from_coo_f64(3, 3, 64, &[e(0, 0, 1.0), e(1, 1, 2.0), e(2, 2, 3.0)]);
        let b = shifted_matrix(&a, &Arb::from_f64(5.0));
        assert_eq!(b.get(0, 0).unwrap().midpoint(), 4.0);
        assert_eq!(b.get(1, 1).unwrap().midpoint(), 3.0);
        assert_eq!(b.get(2, 2).unwrap().midpoint(), 2.0);
    }

    // ── operator_2_norm_enclosure: containment ──────────────────────

    #[test]
    fn op_norm_identity_is_one() {
        let i = CsrArb::from_coo_f64(
            4,
            4,
            64,
            &[e(0, 0, 1.0), e(1, 1, 1.0), e(2, 2, 1.0), e(3, 3, 1.0)],
        );
        let n2 = operator_2_norm_enclosure(&i, &SpectrumOptions::default());
        assert!(n2.lower_endpoint(64) <= 1.0);
        assert!(n2.upper_endpoint(64) >= 1.0);
    }

    #[test]
    fn op_norm_diagonal_picks_largest() {
        // diag(3, 7, 2, 5): ‖·‖₂ = 7.
        let a = CsrArb::from_coo_f64(
            4,
            4,
            64,
            &[e(0, 0, 3.0), e(1, 1, 7.0), e(2, 2, 2.0), e(3, 3, 5.0)],
        );
        let n2 = operator_2_norm_enclosure(&a, &SpectrumOptions::default());
        assert!(n2.lower_endpoint(64) <= 7.0);
        assert!(n2.upper_endpoint(64) >= 7.0);
    }

    #[test]
    fn op_norm_diagonal_with_negative_picks_largest_magnitude() {
        // diag(-10, 1, 2): ‖·‖₂ = max|λ| = 10.
        let a = CsrArb::from_coo_f64(3, 3, 64, &[e(0, 0, -10.0), e(1, 1, 1.0), e(2, 2, 2.0)]);
        let opts = SpectrumOptions {
            max_iters: 200,
            rel_tol: 1e-8,
            seed: 7,
        };
        let n2 = operator_2_norm_enclosure(&a, &opts);
        assert!(n2.lower_endpoint(64) <= 10.0);
        assert!(n2.upper_endpoint(64) >= 10.0);
    }

    #[test]
    fn op_norm_2x2_symmetric_hand_computed() {
        // [[2, 1], [1, 2]]: eigenvalues 1, 3. ‖·‖₂ = 3.
        let a = CsrArb::from_coo_f64(
            2,
            2,
            64,
            &[e(0, 0, 2.0), e(0, 1, 1.0), e(1, 0, 1.0), e(1, 1, 2.0)],
        );
        let n2 = operator_2_norm_enclosure(&a, &SpectrumOptions::default());
        assert!(n2.lower_endpoint(64) <= 3.0);
        assert!(n2.upper_endpoint(64) >= 3.0);
    }

    #[test]
    fn op_norm_zero_matrix() {
        let z = CsrArb::zeros(5, 5, 64);
        let n2 = operator_2_norm_enclosure(&z, &SpectrumOptions::default());
        // ‖0‖₂ = 0; ball must contain 0.
        assert!(n2.lower_endpoint(64) <= 0.0);
        assert!(n2.upper_endpoint(64) >= 0.0);
    }

    #[test]
    fn op_norm_empty_matrix() {
        let z = CsrArb::zeros(0, 0, 64);
        let n2 = operator_2_norm_enclosure(&z, &SpectrumOptions::default());
        assert!(n2.is_exact_zero());
    }

    #[test]
    #[should_panic(expected = "must be symmetric")]
    fn op_norm_rejects_non_symmetric() {
        let a = CsrArb::from_coo_f64(2, 2, 64, &[e(0, 1, 1.0)]);
        let _ = operator_2_norm_enclosure(&a, &SpectrumOptions::default());
    }

    // ── lambda_min_enclosure ───────────────────────────────────────

    #[test]
    fn lambda_min_diag_picks_smallest() {
        // diag(3, 7, 1, 5): λ_min = 1.
        let a = CsrArb::from_coo_f64(
            4,
            4,
            64,
            &[e(0, 0, 3.0), e(1, 1, 7.0), e(2, 2, 1.0), e(3, 3, 5.0)],
        );
        let lm = lambda_min_enclosure(&a, &SpectrumOptions::default());
        assert!(lm.lower_endpoint(64) <= 1.0);
        assert!(lm.upper_endpoint(64) >= 1.0);
    }

    #[test]
    fn lambda_min_2x2_symmetric_hand_computed() {
        // [[2, 1], [1, 2]]: eigenvalues 1, 3. λ_min = 1.
        let a = CsrArb::from_coo_f64(
            2,
            2,
            64,
            &[e(0, 0, 2.0), e(0, 1, 1.0), e(1, 0, 1.0), e(1, 1, 2.0)],
        );
        let lm = lambda_min_enclosure(&a, &SpectrumOptions::default());
        assert!(lm.lower_endpoint(64) <= 1.0);
        assert!(lm.upper_endpoint(64) >= 1.0);
    }

    #[test]
    fn lambda_min_le_lambda_max_sandwich() {
        // For any symmetric A, lower_endpoint(λ_min) ≤ upper_endpoint(‖A‖₂).
        let a = CsrArb::from_coo_f64(
            3,
            3,
            64,
            &[
                e(0, 0, 4.0),
                e(0, 1, 1.0),
                e(1, 0, 1.0),
                e(1, 1, 3.0),
                e(1, 2, 2.0),
                e(2, 1, 2.0),
                e(2, 2, 5.0),
            ],
        );
        let opts = SpectrumOptions::default();
        let lm = lambda_min_enclosure(&a, &opts);
        let n2 = operator_2_norm_enclosure(&a, &opts);
        assert!(lm.lower_endpoint(64) <= n2.upper_endpoint(64));
    }

    #[test]
    fn lambda_min_empty_matrix() {
        let z = CsrArb::zeros(0, 0, 64);
        let lm = lambda_min_enclosure(&z, &SpectrumOptions::default());
        assert!(lm.is_exact_zero());
    }
}
