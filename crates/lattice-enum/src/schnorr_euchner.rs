// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified Schnorr–Euchner / Fincke–Pohst enumeration.
//!
//! The recursion uses the exact square-root-free decomposition
//! `G = μᵀDμ`. At level `k`, with higher coordinates fixed, it derives the
//! complete integer interval from the exact rational inequality
//!
//! `D[k] (v[k] - center[k] + Σ μ[k,j](v[j]-center[j]))² ≤ remaining`.
//!
//! Integer square roots in FLINT turn that inequality into exact `i64`
//! endpoints. Candidates are visited center-out in genuine Schnorr–Euchner
//! order. No floating search box can exclude a valid vector; the exact leaf
//! check in each caller is an independent backstop, not the source of
//! completeness.

use crate::error::{EnumError, EnumResult};
use crate::gso::Cholesky;
use flint_core::{Integer, Rational};

/// Whether the visitor wants the walk to keep going or stop early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Keep enumerating.
    Continue,
    /// Stop the whole enumeration now.
    Stop,
}

fn ceil_div(numerator: &Integer, denominator: &Integer) -> Integer {
    -(-numerator).div_floor(denominator)
}

/// Complete integer interval satisfying `(z - center)² ≤ radius_sq`.
fn integer_range(center: &Rational, radius_sq: &Rational) -> EnumResult<Option<(i64, i64)>> {
    debug_assert!(!radius_sq.is_negative());
    let a = center.numerator();
    let b = center.denominator();
    let p = radius_sq.numerator();
    let q = radius_sq.denominator();

    // q(bz-a)^2 <= p b^2. Since `bz-a` is integral, its absolute value is
    // at most floor(sqrt(floor(p*b^2/q))).
    let scaled = p * b.square();
    let radius = scaled.div_floor(&q).sqrt_floor();
    let minimum = ceil_div(&(&a - &radius), &b)
        .to_i64()
        .ok_or(EnumError::GramOverflow)?;
    let maximum = (&a + radius)
        .div_floor(&b)
        .to_i64()
        .ok_or(EnumError::GramOverflow)?;
    Ok((minimum <= maximum).then_some((minimum, maximum)))
}

fn nearest_integer(value: &Rational) -> EnumResult<i64> {
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

/// Walk every integer vector inside the exact level schedule.
///
/// `center` and `level_bounds` are exact rational values of length `r.n`.
/// Bounds are indexed by depth from the top of the tree and must be
/// non-decreasing. A coordinate outside `i64` returns
/// [`EnumError::GramOverflow`] instead of saturating.
pub fn enumerate<F>(
    r: &Cholesky,
    center: &[Rational],
    level_bounds: &[Rational],
    visitor: &mut F,
) -> EnumResult<()>
where
    F: FnMut(&[i64]) -> Flow,
{
    let n = r.n;
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(center.len(), n);
    debug_assert_eq!(level_bounds.len(), n);
    let mut vector = vec![0_i64; n];
    recurse(
        r,
        center,
        level_bounds,
        n,
        n - 1,
        &Rational::zero(),
        &mut vector,
        visitor,
    )?;
    Ok(())
}

/// Exact integer recursion for a diagonal Gram and rational center `p/q`.
///
/// The full squared-distance numerator is
/// `Σ diagonal[i] * (q*v[i] - p[i])²`; `bound_num` is in that same scale.
pub(crate) fn enumerate_diagonal<F>(
    diagonal: &[Integer],
    target_num: &[i64],
    target_den: i64,
    bound_num: &Integer,
    visitor: &mut F,
) -> EnumResult<()>
where
    F: FnMut(&[i64]) -> Flow,
{
    if diagonal.is_empty() {
        return Ok(());
    }
    let mut vector = vec![0_i64; diagonal.len()];
    if let (Some(bound), Some(word_diagonal)) = (
        bound_num.to_i64().filter(|bound| *bound >= 0),
        diagonal
            .iter()
            .map(Integer::to_i64)
            .collect::<Option<Vec<_>>>()
            .filter(|entries| entries.iter().all(|entry| *entry > 0)),
    ) {
        recurse_diagonal_word(
            &word_diagonal,
            target_num,
            target_den,
            diagonal.len() - 1,
            i128::from(bound),
            &mut vector,
            visitor,
        )?;
        return Ok(());
    }
    recurse_diagonal(
        diagonal,
        target_num,
        &Integer::from(target_den),
        diagonal.len() - 1,
        bound_num,
        &mut vector,
        visitor,
    )?;
    Ok(())
}

fn recurse_diagonal_word<F>(
    diagonal: &[i64],
    target_num: &[i64],
    target_den: i64,
    k: usize,
    remaining: i128,
    vector: &mut [i64],
    visitor: &mut F,
) -> EnumResult<Flow>
where
    F: FnMut(&[i64]) -> Flow,
{
    let radius = (remaining / i128::from(diagonal[k])).isqrt();
    let target = i128::from(target_num[k]);
    let denominator = i128::from(target_den);
    let minimum = (-(-(target - radius)).div_euclid(denominator))
        .try_into()
        .map_err(|_| EnumError::GramOverflow)?;
    let maximum = ((target + radius).div_euclid(denominator))
        .try_into()
        .map_err(|_| EnumError::GramOverflow)?;
    if minimum > maximum {
        return Ok(Flow::Continue);
    }
    let floor = target.div_euclid(denominator);
    let remainder = target.rem_euclid(denominator);
    let nearest: i64 = (if 2 * remainder >= denominator {
        floor + 1
    } else {
        floor
    })
    .try_into()
    .map_err(|_| EnumError::GramOverflow)?;
    let nearest = nearest.clamp(minimum, maximum);

    let mut step = 0_i128;
    loop {
        let left = nearest as i128 - step;
        let right = nearest as i128 + step;
        if left < minimum as i128 && right > maximum as i128 {
            break;
        }
        let mut visit = |candidate: i128| -> EnumResult<Flow> {
            if candidate < minimum as i128 || candidate > maximum as i128 {
                return Ok(Flow::Continue);
            }
            let offset = denominator
                .checked_mul(candidate)
                .and_then(|value| value.checked_sub(target))
                .ok_or(EnumError::GramOverflow)?;
            let contribution = i128::from(diagonal[k])
                .checked_mul(offset.checked_mul(offset).ok_or(EnumError::GramOverflow)?)
                .ok_or(EnumError::GramOverflow)?;
            if contribution > remaining {
                return Ok(Flow::Continue);
            }
            vector[k] = candidate as i64;
            if k == 0 {
                Ok(visitor(vector))
            } else {
                recurse_diagonal_word(
                    diagonal,
                    target_num,
                    target_den,
                    k - 1,
                    remaining - contribution,
                    vector,
                    visitor,
                )
            }
        };
        if visit(left)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        if step != 0 && visit(right)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        step += 1;
    }
    vector[k] = 0;
    Ok(Flow::Continue)
}

fn recurse_diagonal<F>(
    diagonal: &[Integer],
    target_num: &[i64],
    target_den: &Integer,
    k: usize,
    remaining: &Integer,
    vector: &mut [i64],
    visitor: &mut F,
) -> EnumResult<Flow>
where
    F: FnMut(&[i64]) -> Flow,
{
    let radius = remaining.div_floor(&diagonal[k]).sqrt_floor();
    let target = Integer::from(target_num[k]);
    let minimum = ceil_div(&(&target - &radius), target_den)
        .to_i64()
        .ok_or(EnumError::GramOverflow)?;
    let maximum = (&target + radius)
        .div_floor(target_den)
        .to_i64()
        .ok_or(EnumError::GramOverflow)?;
    if minimum > maximum {
        return Ok(Flow::Continue);
    }
    let nearest = nearest_integer(
        &Rational::new(target, target_den.clone()).expect("target denominator is positive"),
    )?
    .clamp(minimum, maximum);

    let mut step = 0_i128;
    loop {
        let left = nearest as i128 - step;
        let right = nearest as i128 + step;
        if left < minimum as i128 && right > maximum as i128 {
            break;
        }
        let mut visit = |candidate: i128| -> EnumResult<Flow> {
            if candidate < minimum as i128 || candidate > maximum as i128 {
                return Ok(Flow::Continue);
            }
            let candidate = candidate as i64;
            vector[k] = candidate;
            let offset = target_den * Integer::from(candidate) - Integer::from(target_num[k]);
            let contribution = &diagonal[k] * offset.square();
            if contribution > *remaining {
                return Ok(Flow::Continue);
            }
            let next = remaining - contribution;
            if k == 0 {
                Ok(visitor(vector))
            } else {
                recurse_diagonal(
                    diagonal,
                    target_num,
                    target_den,
                    k - 1,
                    &next,
                    vector,
                    visitor,
                )
            }
        };
        if visit(left)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        if step != 0 && visit(right)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        step += 1;
    }
    vector[k] = 0;
    Ok(Flow::Continue)
}

#[allow(clippy::too_many_arguments)]
fn recurse<F>(
    r: &Cholesky,
    center: &[Rational],
    level_bounds: &[Rational],
    n: usize,
    k: usize,
    partial_above: &Rational,
    vector: &mut [i64],
    visitor: &mut F,
) -> EnumResult<Flow>
where
    F: FnMut(&[i64]) -> Flow,
{
    let remaining = &level_bounds[n - k - 1] - partial_above;
    if remaining.is_negative() {
        return Ok(Flow::Continue);
    }

    let mut adjusted_center = center[k].clone();
    for j in (k + 1)..n {
        let offset = Rational::from(Integer::from(vector[j])) - &center[j];
        adjusted_center -= r.mu_exact(k, j) * offset;
    }
    let radius_sq = remaining / r.pivot_exact(k);
    let Some((minimum, maximum)) = integer_range(&adjusted_center, &radius_sq)? else {
        return Ok(Flow::Continue);
    };
    let nearest = nearest_integer(&adjusted_center)?.clamp(minimum, maximum);

    let mut step = 0_i128;
    loop {
        let left = nearest as i128 - step;
        let right = nearest as i128 + step;
        if left < minimum as i128 && right > maximum as i128 {
            break;
        }
        let mut visit = |candidate: i128| -> EnumResult<Flow> {
            if candidate < minimum as i128 || candidate > maximum as i128 {
                return Ok(Flow::Continue);
            }
            let candidate = candidate as i64;
            vector[k] = candidate;
            let delta = Rational::from(Integer::from(candidate)) - &adjusted_center;
            let partial = partial_above + r.pivot_exact(k) * &delta * delta;
            if k == 0 {
                Ok(visitor(vector))
            } else {
                recurse(r, center, level_bounds, n, k - 1, &partial, vector, visitor)
            }
        };

        if visit(left)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        if step != 0 && visit(right)? == Flow::Stop {
            vector[k] = 0;
            return Ok(Flow::Stop);
        }
        step += 1;
    }
    vector[k] = 0;
    Ok(Flow::Continue)
}

/// Canonical ± representative: negate so the first nonzero coordinate is
/// positive. The all-zero vector is returned unchanged.
pub fn canonical_sign(v: &[i64]) -> Vec<i64> {
    for &x in v {
        if x != 0 {
            if x < 0 {
                return v.iter().map(|&e| -e).collect();
            }
            break;
        }
    }
    v.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_range_keeps_integers_beyond_f64_unit_resolution() {
        let center = Rational::from(Integer::from(9_007_199_254_740_993_i64));
        assert_eq!(
            integer_range(&center, &Rational::zero()).unwrap(),
            Some((9_007_199_254_740_993, 9_007_199_254_740_993,))
        );
    }

    #[test]
    fn exact_range_handles_fractional_centers_and_radii() {
        let center = Rational::new(Integer::one(), Integer::from(3_u8)).unwrap();
        let radius = Rational::new(Integer::from(4_u8), Integer::from(9_u8)).unwrap();
        assert_eq!(integer_range(&center, &radius).unwrap(), Some((0, 1)));
    }
}
