// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared bounded Jacobian-minor construction for exact commutative algebra.

use crate::{ContextMismatchError, Field, Ring};

use super::{Ideal, Polynomial, PolynomialError, PolynomialRing};

pub(super) const MAX_JACOBIAN_MINOR_SIZE: usize = 8;
pub(super) const MAX_JACOBIAN_MINORS: usize = 10_000;

#[derive(Debug)]
pub(super) enum JacobianError {
    Polynomial(PolynomialError),
    ContextMismatch(ContextMismatchError),
    ResourceLimit {
        resource: &'static str,
        limit: usize,
    },
    Invalid(&'static str),
}

impl From<PolynomialError> for JacobianError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<ContextMismatchError> for JacobianError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

/// All expected-codimension minors of the Jacobian of a canonical ideal basis.
pub(super) fn jacobian_minors<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
    codimension: usize,
) -> Result<Vec<Polynomial<R>>, JacobianError> {
    let ring = source.ring();
    let defining = source.generators();
    let row_count = defining.len();
    let column_count = ring.variables().len();
    if codimension == 0 {
        return Ok(vec![ring.one()]);
    }
    if codimension > row_count || codimension > column_count {
        return Ok(Vec::new());
    }
    if codimension > MAX_JACOBIAN_MINOR_SIZE {
        return Err(JacobianError::ResourceLimit {
            resource: "Jacobian minor size",
            limit: MAX_JACOBIAN_MINOR_SIZE,
        });
    }
    let minor_count = bounded_binomial(row_count, codimension, MAX_JACOBIAN_MINORS)
        .and_then(|rows| {
            bounded_binomial(column_count, codimension, MAX_JACOBIAN_MINORS)
                .and_then(|columns| rows.checked_mul(columns))
        })
        .filter(|count| *count <= MAX_JACOBIAN_MINORS)
        .ok_or(JacobianError::ResourceLimit {
            resource: "Jacobian minor count",
            limit: MAX_JACOBIAN_MINORS,
        })?;

    let jacobian = defining
        .iter()
        .map(|generator| {
            (0..column_count)
                .map(|variable| generator.derivative(variable))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_subsets = combinations(row_count, codimension);
    let column_subsets = combinations(column_count, codimension);
    let mut minors = Vec::with_capacity(minor_count);
    for rows in &row_subsets {
        for columns in &column_subsets {
            let minor = rows
                .iter()
                .map(|row| {
                    columns
                        .iter()
                        .map(|column| jacobian[*row][*column].clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            minors.push(determinant(ring, &minor)?);
        }
    }
    Ok(minors)
}

fn bounded_binomial(n: usize, k: usize, limit: usize) -> Option<usize> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut value = 1_usize;
    for index in 0..k {
        value = value.checked_mul(n - index)?.checked_div(index + 1)?;
        if value > limit {
            return None;
        }
    }
    Some(value)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn extend(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            output.push(current.clone());
            return;
        }
        let remaining = k - current.len();
        for value in start..=n - remaining {
            current.push(value);
            extend(n, k, value + 1, current, output);
            current.pop();
        }
    }

    if k > n {
        return Vec::new();
    }
    let mut output = Vec::new();
    extend(n, k, 0, &mut Vec::with_capacity(k), &mut output);
    output
}

fn determinant<R: Field>(
    ring: &PolynomialRing<R>,
    matrix: &[Vec<Polynomial<R>>],
) -> Result<Polynomial<R>, JacobianError> {
    let size = matrix.len();
    if matrix.iter().any(|row| row.len() != size) {
        return Err(JacobianError::Invalid("a Jacobian minor was not square"));
    }
    let state_count = 1_usize
        .checked_shl(
            u32::try_from(size).map_err(|_| JacobianError::ResourceLimit {
                resource: "Jacobian minor size",
                limit: MAX_JACOBIAN_MINOR_SIZE,
            })?,
        )
        .ok_or(JacobianError::ResourceLimit {
            resource: "Jacobian determinant state count",
            limit: 1 << MAX_JACOBIAN_MINOR_SIZE,
        })?;
    let mut states = vec![ring.zero(); state_count];
    states[0] = ring.one();
    for mask in 0..state_count {
        let row = mask.count_ones() as usize;
        if row == size {
            continue;
        }
        let current = states[mask].clone();
        for (column, value) in matrix[row].iter().enumerate() {
            let bit = 1_usize << column;
            if mask & bit != 0 {
                continue;
            }
            let term = (&current * value)?;
            let inversions = (mask >> (column + 1)).count_ones();
            let next = mask | bit;
            states[next] = if inversions % 2 == 0 {
                (&states[next] + &term)?
            } else {
                (&states[next] - &term)?
            };
        }
    }
    Ok(states[state_count - 1].clone())
}
