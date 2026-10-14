// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Closest-vector enumeration — [`cvp_enumerate`].
//!
//! The target may be **rational**: it arrives as an integer numerator vector
//! `p` and a common denominator `q > 0`, so the target coordinate `i` is
//! `p[i]/q`. The exact squared `G`-distance of an integer `v` is then
//! `(q·v − p)ᵀ G (q·v − p) / q²`, an exact rational compared numerator-wise
//! (all candidates share the denominator `q²`).

use crate::error::{EnumError, EnumResult};
use crate::gso::{exact_quadratic, validate_gram, Cholesky};
use crate::schnorr_euchner::{enumerate, enumerate_diagonal, Flow};
use crate::ClosestVectors;
use flint_core::{Integer, Rational};

/// Integer vector `q·v − p`, checked (never a silent `i64` wrap).
fn scaled_offset(v: &[i64], p: &[i64], q: i64) -> EnumResult<Vec<i64>> {
    let mut w = Vec::with_capacity(v.len());
    for (vi, pi) in v.iter().zip(p.iter()) {
        let qv = q.checked_mul(*vi).ok_or(EnumError::GramOverflow)?;
        w.push(qv.checked_sub(*pi).ok_or(EnumError::GramOverflow)?);
    }
    Ok(w)
}

/// Enumerate the integer vector(s) closest to `target = p/q` under the
/// `G`-inner product.
///
/// * `gram` — symmetric PD integer Gram matrix (`n×n`).
/// * `target_num` (`p`) — integer numerator vector, length `n`.
/// * `target_den` (`q`) — common denominator, `q ≥ 1`.
/// * `max_count` — cap on the number of equidistant closest vectors
///   returned (`None` = all ties).
///
/// # Errors
///
/// [`EnumError::InvalidShape`] (target length ≠ `n`, or `q < 1`),
/// [`EnumError::Asymmetric`] / [`EnumError::NotPositiveDefinite`] (bad Gram),
/// or [`EnumError::GramOverflow`] when a returned coordinate cannot fit `i64`.
pub fn cvp_enumerate(
    gram: &[Vec<Integer>],
    target_num: &[i64],
    target_den: i64,
    max_count: Option<usize>,
) -> EnumResult<ClosestVectors> {
    let n = validate_gram(gram)?;
    if target_num.len() != n {
        return Err(EnumError::InvalidShape {
            detail: format!("target length {} != Gram dim {n}", target_num.len()),
        });
    }
    if target_den < 1 {
        return Err(EnumError::InvalidShape {
            detail: format!("target denominator {target_den} must be ≥ 1"),
        });
    }
    if n == 0 {
        return Ok(ClosestVectors {
            distance_sq_num: Integer::zero(),
            distance_sq_den: Integer::one(),
            vectors: vec![vec![]],
        });
    }
    let ch = Cholesky::factor(gram)?;
    let q = target_den;
    let q2 = Integer::from(q).square();

    let center: Vec<Rational> = target_num
        .iter()
        .map(|&p| {
            Rational::new(Integer::from(p), Integer::from(q))
                .expect("the target denominator is positive")
        })
        .collect();
    // Babai nearest-plane seed → a tight initial radius (exact distance of a
    // concrete lattice vector, so the true closest is inside this ellipsoid).
    // If the floating projection is unavailable, coordinate-wise exact
    // rounding remains a valid (if looser) concrete seed.
    let mut v0: Vec<i64> = center
        .iter()
        .map(nearest_rational_i64)
        .collect::<EnumResult<_>>()?;
    if ch.diagonal().is_some() {
        let center_float: Vec<f64> = center
            .iter()
            .map(Rational::to_f64)
            .collect::<Option<Vec<_>>>()
            .filter(|values| values.iter().all(|value| value.is_finite()))
            .unwrap_or_default();
        if center_float.len() == n {
            for k in (0..n).rev() {
                let mut tail = 0.0;
                for j in (k + 1)..n {
                    tail += ch.get(k, j).expect("complete projection")
                        * (v0[j] as f64 - center_float[j]);
                }
                let candidate =
                    center_float[k] - tail / ch.get(k, k).expect("positive floating pivot");
                if candidate.is_finite()
                    && candidate >= i64::MIN as f64
                    && candidate <= i64::MAX as f64
                {
                    v0[k] = candidate.round() as i64;
                }
            }
        }
    }
    let w0 = scaled_offset(&v0, target_num, q)?;
    let n0 = exact_quadratic(gram, &w0)?; // = q² · dist²(v0)
    let radius_sq =
        Rational::new(n0.clone(), q2.clone()).expect("a squared positive denominator is nonzero");

    let level_bounds = vec![radius_sq; n];

    let mut best: Option<Integer> = None;
    let mut winners: Vec<Vec<i64>> = Vec::new();
    let mut err: Option<EnumError> = None;
    let mut visit = |v: &[i64]| {
        let w = match scaled_offset(v, target_num, q) {
            Ok(w) => w,
            Err(e) => {
                err = Some(e);
                return Flow::Stop;
            }
        };
        let dist = match exact_quadratic(gram, &w) {
            Ok(d) => d,
            Err(e) => {
                err = Some(e);
                return Flow::Stop;
            }
        };
        match &best {
            None => {
                best = Some(dist);
                winners.push(v.to_vec());
            }
            Some(b) if &dist < b => {
                best = Some(dist);
                winners.clear();
                winners.push(v.to_vec());
            }
            Some(b) if &dist == b => winners.push(v.to_vec()),
            _ => {}
        }
        Flow::Continue
    };
    let diagonal = (0..n)
        .all(|row| (0..n).all(|column| row == column || gram[row][column].is_zero()))
        .then(|| (0..n).map(|i| gram[i][i].clone()).collect::<Vec<_>>());
    if let Some(diagonal) = diagonal {
        enumerate_diagonal(&diagonal, target_num, q, &n0, &mut visit)?;
    } else {
        enumerate(&ch, &center, &level_bounds, &mut visit)?;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // The Babai seed v0 is guaranteed to be inside the search ellipsoid, so
    // `winners` is non-empty; guard defensively anyway.
    let best = best.unwrap_or(n0);
    if winners.is_empty() {
        winners.push(v0);
    }

    winners.sort();
    winners.dedup();
    if let Some(cap) = max_count {
        winners.truncate(cap);
    }

    Ok(ClosestVectors {
        distance_sq_num: best,
        distance_sq_den: q2,
        vectors: winners,
    })
}

fn nearest_rational_i64(value: &Rational) -> EnumResult<i64> {
    let numerator = value.numerator();
    let denominator = value.denominator();
    let floor = numerator.div_floor(&denominator);
    let remainder = &numerator - &floor * &denominator;
    let nearest = if Integer::from(2_u8) * remainder >= denominator {
        floor + Integer::one()
    } else {
        floor
    };
    nearest.to_i64().ok_or(EnumError::GramOverflow)
}
