// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Pruned enumeration via [`pruned_enumeration`].
//!
//! Cylinder pruning replaces the constant enumeration radius `B` at every
//! tree depth with a **non-decreasing schedule** `p_d · B` (`0 < p_d ≤ 1`,
//! `p_n = 1`): shallow levels get a tighter radius, cutting the sub-tree at
//! the cost of possibly missing some short vectors (the whole point — you
//! trade completeness for a large speed-up, then re-randomise the basis).
//!
//! * **Linear** pruning (Gama–Nguyen–Regev 2010): `p_d = d/n`.
//! * **Extreme** pruning (Aono–Wang 2018): a steeper curve, here the
//!   documented heuristic `p_d = (d/n)²`, far more aggressive.
//!
//! The pruning *vector* is an explicit parameter, and
//! [`PruningMode::pruning_vector`] generates the two standard curves.
//! Passing an all-ones vector recovers full [`crate::svp_enumerate`].

use crate::error::{EnumError, EnumResult};
use crate::gso::{exact_quadratic, validate_gram, Cholesky};
use crate::schnorr_euchner::{enumerate, Flow};
use crate::ShortVectors;
use flint_core::{Integer, Rational};

/// Which standard pruning curve [`PruningMode::pruning_vector`] generates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMode {
    /// Gama–Nguyen–Regev linear pruning, `p_d = d/n`.
    Linear,
    /// Aono–Wang extreme pruning, the steeper `p_d = (d/n)²`.
    Extreme,
}

impl PruningMode {
    /// The pruning vector `[p_1, …, p_n]` (`0 < p_d ≤ 1`, non-decreasing,
    /// `p_n = 1`) for dimension `n`.
    pub fn pruning_vector(&self, n: usize) -> Vec<f64> {
        (1..=n)
            .map(|d| {
                let x = d as f64 / n as f64;
                match self {
                    PruningMode::Linear => x,
                    PruningMode::Extreme => x * x,
                }
            })
            .collect()
    }
}

/// Enumerate short vectors under a pruning schedule.
///
/// `pruning[d-1] = p_d ∈ (0, 1]` must be non-decreasing with `p_n = 1`.
/// Returns the (exact-verified, ± deduped) vectors the pruned search finds
/// with `vᵀGv ≤ bound`. With `pruning` all-ones this equals
/// [`crate::svp_enumerate`]`(gram, Some(bound), None)`; with a genuine
/// pruning curve it may return a **subset** (missed short vectors are the
/// accepted cost of the speed-up).
///
/// # Errors
///
/// [`EnumError::InvalidBound`] (`bound < 0`), [`EnumError::PruningShape`]
/// (wrong length / out-of-range / non-monotone / `p_n ≠ 1`), plus the Gram
/// validation / PD / overflow errors.
pub fn pruned_enumeration(
    gram: &[Vec<Integer>],
    bound: Integer,
    pruning: &[f64],
) -> EnumResult<ShortVectors> {
    let n = validate_gram(gram)?;
    if bound.is_negative() {
        return Err(EnumError::InvalidBound { bound });
    }
    if pruning.len() != n {
        return Err(EnumError::PruningShape {
            detail: format!("pruning length {} != Gram dim {n}", pruning.len()),
        });
    }
    let mut prev = 0.0f64;
    for (d, &p) in pruning.iter().enumerate() {
        if !(p > 0.0 && p <= 1.0) {
            return Err(EnumError::PruningShape {
                detail: format!("p_{} = {p} not in (0, 1]", d + 1),
            });
        }
        if p < prev - 1e-12 {
            return Err(EnumError::PruningShape {
                detail: "pruning schedule must be non-decreasing".into(),
            });
        }
        prev = p;
    }
    if n == 0 {
        return Ok(ShortVectors {
            count: Integer::zero(),
            max_norm: Integer::zero(),
            bound: Some(bound),
            complete: true,
            vectors: Vec::new(),
        });
    }
    if let Some(&last) = pruning.last() {
        if (last - 1.0).abs() > 1e-9 {
            return Err(EnumError::PruningShape {
                detail: format!("p_n must be 1, got {last}"),
            });
        }
    }
    if bound.is_zero() {
        return Ok(ShortVectors {
            count: Integer::zero(),
            max_norm: Integer::zero(),
            bound: Some(bound),
            complete: true,
            vectors: Vec::new(),
        });
    }

    let ch = Cholesky::factor(gram)?;
    let center = vec![Rational::zero(); n];
    let exact_bound = Rational::from(bound.clone());
    let level_bounds: Vec<Rational> = pruning
        .iter()
        .map(|&p| {
            Rational::from_f64_exact(p).expect("validated finite pruning coefficient")
                * &exact_bound
        })
        .collect();

    let mut reps: Vec<(Integer, Vec<i64>)> = Vec::new();
    let mut err: Option<EnumError> = None;
    enumerate(&ch, &center, &level_bounds, &mut |v| {
        let mut first_nz = 0i64;
        for &x in v {
            if x != 0 {
                first_nz = x;
                break;
            }
        }
        if first_nz <= 0 {
            return Flow::Continue;
        }
        match exact_quadratic(gram, v) {
            Ok(norm) => {
                if norm.is_positive() && norm <= bound {
                    reps.push((norm, v.to_vec()));
                }
                Flow::Continue
            }
            Err(e) => {
                err = Some(e);
                Flow::Stop
            }
        }
    })?;
    if let Some(e) = err {
        return Err(e);
    }
    reps.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    reps.dedup();
    let max_norm = reps
        .iter()
        .map(|(norm, _)| norm)
        .max()
        .cloned()
        .unwrap_or_else(Integer::zero);
    let vectors: Vec<Vec<i64>> = reps.into_iter().map(|(_, v)| v).collect();
    let complete = pruning.iter().all(|value| (*value - 1.0).abs() <= 1e-12);
    Ok(ShortVectors {
        count: Integer::from(2_u8) * Integer::from(vectors.len()),
        max_norm,
        bound: Some(bound),
        complete,
        vectors,
    })
}
