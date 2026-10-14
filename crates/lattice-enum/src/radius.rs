// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Gaussian-heuristic sizing utilities — [`gaussian_heuristic_bound`] and
//! [`enumeration_radius_set`] — plus the shared lattice-point estimator the
//! SVP sanity cap uses.

use crate::error::{EnumError, EnumResult};
use crate::gso::{validate_gram, Cholesky};
use flint_core::Integer;

/// `ln Γ(n/2 + 1)` for integer `n`, exact on both half-integer ladders (no
/// external special-function dependency in the kernel).
pub(crate) fn ln_gamma_half_plus_one(n: usize) -> f64 {
    // Γ(m+1) = m!  ;  Γ(m + 1/2) = (2m)!/(4^m m!) · √π
    if n.is_multiple_of(2) {
        let m = n / 2; // Γ(m+1) = m!
        (1..=m).map(|k| (k as f64).ln()).sum()
    } else {
        let m = n.div_ceil(2); // Γ((m+1) + 1/2) → want Γ(m + 1/2)
        let ln_fact_2m: f64 = (1..=(2 * m)).map(|k| (k as f64).ln()).sum();
        let ln_fact_m: f64 = (1..=m).map(|k| (k as f64).ln()).sum();
        ln_fact_2m - (m as f64) * 4.0f64.ln() - ln_fact_m + 0.5 * std::f64::consts::PI.ln()
    }
}

/// Estimate the number of integer points in `{v : vᵀGv ≤ B}` via its
/// continuous volume `B^{n/2} · π^{n/2} / (Γ(n/2+1)·√det G)` (Gauss's
/// lattice-point heuristic). Counts every lattice point (both `v` and `−v`).
pub(crate) fn ellipsoid_point_estimate(det_g: f64, bound: f64, n: usize) -> f64 {
    if bound <= 0.0 || det_g <= 0.0 {
        return 0.0;
    }
    let half = n as f64 / 2.0;
    let ln_vol = half * bound.ln() + half * std::f64::consts::PI.ln()
        - ln_gamma_half_plus_one(n)
        - 0.5 * det_g.ln();
    ln_vol.exp()
}

/// The Gaussian-heuristic prediction of the shortest nonzero **squared**
/// `G`-length: `(Γ(n/2 + 1) · vol(L))^{2/n} / π`, where `vol(L) = √det G`.
///
/// # Errors
///
/// [`EnumError::InvalidShape`] / [`EnumError::Asymmetric`] (bad Gram),
/// [`EnumError::NotPositiveDefinite`].
pub fn gaussian_heuristic_bound(gram: &[Vec<Integer>]) -> EnumResult<f64> {
    let n = validate_gram(gram)?;
    if n == 0 {
        return Ok(0.0);
    }
    let ch = Cholesky::factor(gram)?;
    let vol = ch.det().ok_or(EnumError::GramOverflow)?.sqrt(); // √det G
    let gamma = ln_gamma_half_plus_one(n).exp();
    Ok((gamma * vol).powf(2.0 / n as f64) / std::f64::consts::PI)
}

/// Choose a norm bound `B` under which the Gaussian heuristic predicts
/// approximately `target_count` lattice vectors (signed count, i.e. counting
/// both `v` and `−v`).
///
/// Inverts the internal ellipsoid point estimate:
/// `B = (target_count · Γ(n/2+1) · √det G / π^{n/2})^{2/n}`. Returns the
/// ceiling as an arbitrary-precision integer bound ready for [`crate::svp_enumerate`].
///
/// # Errors
///
/// [`EnumError::InvalidShape`] / [`EnumError::Asymmetric`] /
/// [`EnumError::NotPositiveDefinite`] (bad Gram),
/// [`EnumError::GramOverflow`] when the floating estimate is not finite.
pub fn enumeration_radius_set(gram: &[Vec<Integer>], target_count: u64) -> EnumResult<Integer> {
    let n = validate_gram(gram)?;
    if n == 0 {
        return Ok(Integer::zero());
    }
    let ch = Cholesky::factor(gram)?;
    if target_count == 0 {
        return Ok(Integer::zero());
    }
    let vol = ch.det().ok_or(EnumError::GramOverflow)?.sqrt();
    let half = n as f64 / 2.0;
    // ln B = (2/n)·(ln target + lnΓ(n/2+1) + ln vol − half·ln π)
    let ln_b = (2.0 / n as f64)
        * ((target_count as f64).ln() + ln_gamma_half_plus_one(n) + vol.ln()
            - half * std::f64::consts::PI.ln());
    let b = ln_b.exp().ceil();
    if !b.is_finite() {
        return Err(EnumError::GramOverflow);
    }
    format!("{b:.0}")
        .parse()
        .map_err(|_| EnumError::GramOverflow)
}
