// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Short-vector enumeration — [`svp_enumerate`].

use crate::error::{EnumError, EnumResult};
use crate::gso::{exact_quadratic, validate_gram, Cholesky};
use crate::radius::ellipsoid_point_estimate;
use crate::reduction::{lll_reduce_gram, original_coordinates};
use crate::schnorr_euchner::{enumerate, enumerate_diagonal, Flow};
use crate::ShortVectors;
use flint_core::{Integer, Rational};

/// The reference-impl sanity cap: refuse to enumerate when the bounding
/// ellipsoid is estimated to hold more than this many integer points and no
/// `max_count` was given.
pub(crate) const SANITY_CAP: f64 = 2.0e7;

/// Enumerate the short vectors of a positive-definite integer Gram matrix.
///
/// * `bound = None` returns the minimum-norm vectors only (`Min(G)` mod ±).
/// * `bound = Some(B)` returns all `v ∈ Zⁿ` with `vᵀGv ≤ B`.
/// * `max_count` caps the number of ± pair representatives returned
///   (`None` = no cap); when `None`, the ellipsoid sanity cap
///   (the internal sanity cap) guards against a runaway enumeration.
///
/// Returns one representative per ± pair, canonical sign (first nonzero
/// coordinate positive).
///
/// # Errors
///
/// [`EnumError::InvalidShape`] / [`EnumError::Asymmetric`] (bad Gram),
/// [`EnumError::NotPositiveDefinite`], [`EnumError::InvalidBound`]
/// (`bound < 0`), [`EnumError::NoNonzeroVectors`] (an unbounded rank-zero
/// query), [`EnumError::EnumerationOverflow`] (estimate over the cap with no
/// `max_count`), or [`EnumError::GramOverflow`] when the optional floating work
/// estimate is unavailable or a coordinate cannot fit `i64`.
pub fn svp_enumerate(
    gram: &[Vec<Integer>],
    bound: Option<Integer>,
    max_count: Option<usize>,
) -> EnumResult<ShortVectors> {
    let n = validate_gram(gram)?;
    if let Some(b) = &bound {
        if b.is_negative() {
            return Err(EnumError::InvalidBound { bound: b.clone() });
        }
    }
    if n == 0 {
        if bound.is_none() {
            return Err(EnumError::NoNonzeroVectors);
        }
        return Ok(ShortVectors {
            count: Integer::zero(),
            max_norm: Integer::zero(),
            bound,
            complete: true,
            vectors: Vec::new(),
        });
    }
    if max_count == Some(0) {
        let complete = bound.as_ref().is_some_and(|value| !value.is_positive());
        return Ok(ShortVectors {
            count: Integer::zero(),
            max_norm: Integer::zero(),
            bound,
            complete,
            vectors: Vec::new(),
        });
    }
    let diagonal_gram =
        (0..n).all(|row| (0..n).all(|column| row == column || gram[row][column].is_zero()));
    // An orthogonal diagonal form needs no size-reduction mixing for exact
    // enumeration. Avoid the general LLL pass and the identity coordinate
    // transform at every enumerated leaf; both are pure overhead in this
    // common lattice case. The factorization below still checks every
    // diagonal pivot exactly.
    let (reduced_gram, transformation) = if diagonal_gram {
        (gram.to_vec(), None)
    } else {
        // Validate positive-definiteness before the reduction helper assumes it.
        let _ = Cholesky::factor(gram)?;
        let (reduced, transformation) = lll_reduce_gram(gram);
        (reduced, Some(transformation))
    };
    let ch = Cholesky::factor(&reduced_gram)?;

    let min_diag = (0..n)
        .map(|i| reduced_gram[i][i].clone())
        .min()
        .unwrap_or_else(Integer::zero);
    let bound_int = bound.clone().unwrap_or(min_diag);
    if !bound_int.is_positive() {
        // No nonzero vector can have norm ≤ 0.
        return Ok(ShortVectors {
            count: Integer::zero(),
            max_norm: Integer::zero(),
            bound,
            complete: true,
            vectors: Vec::new(),
        });
    }

    // Sanity cap (only when uncapped): estimate the ellipsoid's integer-point
    // count and refuse an astronomical enumeration rather than OOM.
    if max_count.is_none() {
        let bound_float = Rational::from(bound_int.clone())
            .to_f64()
            .ok_or(EnumError::GramOverflow)?;
        let det = ch.det().ok_or(EnumError::GramOverflow)?;
        let est = ellipsoid_point_estimate(det, bound_float, n);
        if est > SANITY_CAP {
            return Err(EnumError::EnumerationOverflow { estimate: est });
        }
    }

    let level_bounds = vec![Rational::from(bound_int.clone()); n];
    let center = vec![Rational::zero(); n];

    let mut reps: Vec<(Integer, Vec<i64>)> = Vec::new();
    let mut err: Option<EnumError> = None;
    let mut cap_exceeded = false;
    let mut visit = |reduced_vector: &[i64]| {
        let v = if let Some(transformation) = &transformation {
            let Some(vector) = original_coordinates(transformation, reduced_vector) else {
                err = Some(EnumError::GramOverflow);
                return Flow::Stop;
            };
            vector
        } else {
            reduced_vector.to_vec()
        };
        // one rep per ± pair: keep the canonical (first nonzero > 0) sign
        let mut first_nz = 0i64;
        for &x in &v {
            if x != 0 {
                first_nz = x;
                break;
            }
        }
        if first_nz <= 0 {
            return Flow::Continue; // zero vector or the negated rep
        }
        match exact_quadratic(gram, &v) {
            Ok(norm) => {
                if norm.is_positive() && norm <= bound_int {
                    reps.push((norm, v));
                    // A bounded query may return a deterministic partial set.
                    // A minimum-shell query must finish the enclosing search
                    // before it knows which visited norm is truly minimal.
                    if let (Some(cap), Some(_)) = (max_count, &bound) {
                        if reps.len() > cap {
                            cap_exceeded = true;
                            return Flow::Stop;
                        }
                    }
                }
                Flow::Continue
            }
            Err(e) => {
                err = Some(e);
                Flow::Stop
            }
        }
    };
    let diagonal = (0..n)
        .all(|row| (0..n).all(|column| row == column || reduced_gram[row][column].is_zero()))
        .then(|| {
            (0..n)
                .map(|i| reduced_gram[i][i].clone())
                .collect::<Vec<_>>()
        });
    if let Some(diagonal) = diagonal {
        enumerate_diagonal(&diagonal, &vec![0_i64; n], 1, &bound_int, &mut visit)?;
    } else {
        enumerate(&ch, &center, &level_bounds, &mut visit)?;
    }
    if let Some(e) = err {
        return Err(e);
    }

    // No-bound query: keep only the true minimum-norm shell.
    if bound.is_none() && !reps.is_empty() {
        let true_min = reps.iter().map(|(no, _)| no).min().unwrap().clone();
        reps.retain(|(no, _)| no == &true_min);
    }

    // Deterministic order: by (norm, lexicographic vector).
    reps.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let complete = max_count.is_none_or(|cap| !cap_exceeded && reps.len() <= cap);
    if let Some(cap) = max_count {
        reps.truncate(cap);
    }

    let max_norm = reps
        .iter()
        .map(|(norm, _)| norm)
        .max()
        .cloned()
        .unwrap_or_else(Integer::zero);
    let vectors: Vec<Vec<i64>> = reps.into_iter().map(|(_, v)| v).collect();
    Ok(ShortVectors {
        count: Integer::from(2_u8) * Integer::from(vectors.len()),
        max_norm,
        bound,
        complete,
        vectors,
    })
}
