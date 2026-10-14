// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Gram-matrix validation, the exact Gram–Schmidt substrate the enumeration
//! recursion walks, and exact integer bilinear/quadratic forms.
//!
//! # Exact square-root-free GSO
//!
//! The FLINT-backed rational decomposition is `G = μᵀ D μ`, with `μ` unit
//! upper-triangular and `D = diag(dᵢ)`. The exact signs of the pivots decide
//! positive-definiteness, and the exact `μ` and `D` drive every enumeration
//! interval. A best-effort `f64` Cholesky projection is retained only for the
//! Babai seed and Gaussian heuristics; its absence never compromises exact
//! bounded enumeration.

use crate::error::{EnumError, EnumResult};
use flint_core::{Integer, Rational};

/// Validate a Gram matrix: nonempty, square, symmetric. Returns the
/// dimension `n`. (Positive-definiteness is decided later, by [`Cholesky`].)
pub fn validate_gram(gram: &[Vec<Integer>]) -> EnumResult<usize> {
    let n = gram.len();
    if n == 0 {
        return Ok(0);
    }
    for (i, row) in gram.iter().enumerate() {
        if row.len() != n {
            return Err(EnumError::InvalidShape {
                detail: format!("row {i} has length {}, expected {n}", row.len()),
            });
        }
    }
    for (i, row) in gram.iter().enumerate() {
        for (j, gij) in row.iter().enumerate().skip(i + 1) {
            if gij != &gram[j][i] {
                return Err(EnumError::Asymmetric { at: (i, j) });
            }
        }
    }
    Ok(n)
}

/// Convert an exact `Rational` to the nearest finite `f64` when possible.
fn q_to_f64(q: &Rational) -> Option<f64> {
    let x = q.to_f64()?;
    if x.is_finite() {
        Some(x)
    } else {
        None
    }
}

/// Exact square-root-free Gram–Schmidt decomposition, with an optional
/// floating Cholesky projection for heuristics.
///
/// Construction fails with [`EnumError::NotPositiveDefinite`] the moment an
/// exact pivot is non-positive. A form may be perfectly enumerable even when
/// its optional floating projection is unavailable.
#[derive(Debug, Clone)]
pub struct Cholesky {
    /// Dimension.
    pub n: usize,
    /// Row-major floating `R`, when every entry is finite.
    r: Option<Vec<f64>>,
    /// Floating `det(G)`, when finite. Used only by heuristics.
    det: Option<f64>,
    /// Exact square-root-free pivots in `G = μᵀDμ`.
    d: Vec<Rational>,
    /// Exact unit-upper `μ`, row-major.
    mu: Vec<Rational>,
}

impl Cholesky {
    /// Factor a validated (square, symmetric) integer Gram matrix.
    ///
    /// Computes the exact rational `LDLᵀ` GSO `G = μᵀ D μ` (`μ` unit upper-
    /// triangular, `D = diag(dᵢ)`), and fails with
    /// [`EnumError::NotPositiveDefinite`] on the first non-positive exact
    /// pivot. It also records a floating projection when representable.
    pub fn factor(gram: &[Vec<Integer>]) -> EnumResult<Cholesky> {
        let n = gram.len();
        // Exact LDLᵀ. `d[i]` = the i-th pivot (rational); `mu[i][j]` for j > i
        // = the exact unit-upper-triangular GSO coefficient.
        //   d_i      = G[i][i] - Σ_{k<i} d_k · μ[k][i]²
        //   μ[i][j]  = (G[i][j] - Σ_{k<i} d_k · μ[k][i]·μ[k][j]) / d_i
        let mut d: Vec<Rational> = Vec::with_capacity(n);
        let mut mu = vec![vec![Rational::zero(); n]; n];
        let mut det = Rational::from(Integer::one());
        for i in 0..n {
            let mut di = Rational::from(gram[i][i].clone());
            for k in 0..i {
                di -= &d[k] * &mu[k][i] * &mu[k][i];
            }
            if di <= Rational::zero() {
                return Err(EnumError::NotPositiveDefinite);
            }
            for j in (i + 1)..n {
                let mut s = Rational::from(gram[i][j].clone());
                for k in 0..i {
                    s -= &d[k] * &mu[k][i] * &mu[k][j];
                }
                mu[i][j] = s / &di;
            }
            det *= &di;
            d.push(di);
        }

        // Project √dₖ·μ[k][j] for Babai and sizing heuristics. Exact search
        // uses `d` and `mu` below, so projection failure is not a factor error.
        let r = (|| {
            let mut r = vec![0.0f64; n * n];
            for i in 0..n {
                let sqrt_di = q_to_f64(&d[i])?.sqrt();
                if !sqrt_di.is_finite() || sqrt_di == 0.0 {
                    return None;
                }
                r[i * n + i] = sqrt_di;
                for j in (i + 1)..n {
                    let value = sqrt_di * q_to_f64(&mu[i][j])?;
                    if !value.is_finite() {
                        return None;
                    }
                    r[i * n + j] = value;
                }
            }
            Some(r)
        })();
        let det = q_to_f64(&det);
        let mu = mu.into_iter().flatten().collect();
        Ok(Cholesky { n, r, det, d, mu })
    }

    /// Floating `R[i][j]` (only meaningful for `j ≥ i`), when the complete
    /// projection is finite.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        self.r.as_ref().map(|r| r[i * self.n + j])
    }

    /// Floating diagonal pivots `R[i][i] = √dᵢ`, when representable.
    pub fn diagonal(&self) -> Option<Vec<f64>> {
        self.r
            .as_ref()
            .map(|r| (0..self.n).map(|i| r[i * self.n + i]).collect())
    }

    /// `det(G)` projected to finite `f64`, when representable.
    pub fn det(&self) -> Option<f64> {
        self.det
    }

    /// Exact square-root-free pivot `D[i]`.
    pub(crate) fn pivot_exact(&self, i: usize) -> &Rational {
        &self.d[i]
    }

    /// Exact unit-upper coefficient `μ[i,j]` (`i < j`).
    pub(crate) fn mu_exact(&self, i: usize, j: usize) -> &Rational {
        &self.mu[i * self.n + j]
    }
}

/// Exact `vᵀ G v` in arbitrary-precision FLINT integer arithmetic.
pub fn exact_quadratic(gram: &[Vec<Integer>], v: &[i64]) -> EnumResult<Integer> {
    exact_bilinear(gram, v, v)
}

/// Exact `uᵀ G w` in arbitrary-precision FLINT integer arithmetic.
pub fn exact_bilinear(gram: &[Vec<Integer>], u: &[i64], w: &[i64]) -> EnumResult<Integer> {
    let n = gram.len();
    let mut total = Integer::zero();
    for i in 0..n {
        if u[i] == 0 {
            continue;
        }
        // gw_i = Σ_j G[i][j] w[j]
        let mut gw = Integer::zero();
        let row = &gram[i];
        for j in 0..n {
            if w[j] == 0 {
                continue;
            }
            gw += &row[j] * Integer::from(w[j]);
        }
        total += Integer::from(u[i]) * gw;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a2_cholesky_reconstructs_gram() {
        let g = integer_gram(&[&[2, -1], &[-1, 2]]);
        let ch = Cholesky::factor(&g).unwrap();
        // R^T R == G (float)
        for (i, row) in g.iter().enumerate() {
            for (j, gij) in row.iter().enumerate() {
                let s: f64 = (0..2)
                    .map(|k| ch.get(k, i).unwrap() * ch.get(k, j).unwrap())
                    .sum();
                assert!((s - Rational::from(gij.clone()).to_f64().unwrap()).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn non_pd_rejected() {
        // det = 1*1 - 2*2 = -3 < 0 → indefinite
        let g = integer_gram(&[&[1, 2], &[2, 1]]);
        assert!(matches!(
            Cholesky::factor(&g),
            Err(EnumError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn exact_quadratic_matches_hand() {
        let g = integer_gram(&[&[2, -1], &[-1, 2]]);
        // v=(1,1): 2 -1 -1 + 2 = 2
        assert_eq!(exact_quadratic(&g, &[1, 1]).unwrap(), Integer::from(2));
        // v=(3,-2): 9*2 + 2*(-1)*(3)(-2) + 4*2 = 18 + 12 + 8 = 38
        assert_eq!(exact_quadratic(&g, &[3, -2]).unwrap(), Integer::from(38));
    }

    #[test]
    fn asymmetric_detected() {
        let g = integer_gram(&[&[2, -1], &[0, 2]]);
        assert_eq!(validate_gram(&g), Err(EnumError::Asymmetric { at: (0, 1) }));
    }

    fn integer_gram(rows: &[&[i64]]) -> Vec<Vec<Integer>> {
        rows.iter()
            .map(|row| row.iter().copied().map(Integer::from).collect())
            .collect()
    }
}
