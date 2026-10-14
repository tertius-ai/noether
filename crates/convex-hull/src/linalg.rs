// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact rational linear algebra over `Q` — the small, self-contained
//! kernel of Gaussian-elimination primitives the double-description
//! engine and the hull construction need: rank, matrix inverse, a
//! single null-space vector, and canonical (primitive-integer) scaling.
//!
//! Everything is exact: coordinates are [`crate::Q`]
//! (`flint_core::Rational`), so a degenerate incidence is *decided*,
//! never approximated.

// Gaussian elimination is inherently row/column-index driven: the pivot
// operations touch two rows simultaneously (`m[r]` against `m[pivot]`),
// which iterator adaptors cannot express without fighting the borrow
// checker. Index loops are the clear, correct form here.
#![allow(clippy::needless_range_loop)]

use crate::Q;
use flint_core::Integer;

/// Exact inner product `⟨a, b⟩`.
pub fn dot(a: &[Q], b: &[Q]) -> Q {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = Q::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Indices of a maximal set of linearly independent rows of `rows`
/// (each of width `ncols`), in input order. `rows[i]` is kept iff it is
/// independent of the rows kept before it. The returned length is the
/// rank of the matrix.
pub fn independent_rows(rows: &[Vec<Q>], ncols: usize) -> Vec<usize> {
    let mut basis: Vec<Vec<Q>> = Vec::new();
    let mut pivots: Vec<usize> = Vec::new();
    let mut chosen: Vec<usize> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let mut r = row.clone();
        // reduce against the current echelon basis (each basis row is
        // normalised to a leading 1 at its pivot column).
        for (bi, brow) in basis.iter().enumerate() {
            let pc = pivots[bi];
            if !r[pc].is_zero() {
                let factor = r[pc].clone();
                for c in 0..ncols {
                    r[c] = &r[c] - &(&factor * &brow[c]);
                }
            }
        }
        if let Some(pc) = (0..ncols).find(|&c| !r[c].is_zero()) {
            let piv = r[pc].clone();
            for c in 0..ncols {
                r[c] = &r[c] / &piv;
            }
            basis.push(r);
            pivots.push(pc);
            chosen.push(i);
        }
    }
    chosen
}

/// Rank of the matrix whose rows are `rows` (each of width `ncols`).
pub fn rank(rows: &[Vec<Q>], ncols: usize) -> usize {
    independent_rows(rows, ncols).len()
}

/// Affine rank (dimension of the affine hull) of a point set, or `-1`
/// for the empty set. A single point has affine rank `0`.
pub fn affine_rank(points: &[Vec<Q>], dim: usize) -> isize {
    if points.is_empty() {
        return -1;
    }
    let base = &points[0];
    let diffs: Vec<Vec<Q>> = points[1..]
        .iter()
        .map(|p| (0..dim).map(|c| &p[c] - &base[c]).collect())
        .collect();
    rank(&diffs, dim) as isize
}

/// Inverse of the `n × n` matrix `a` (rows), or `None` if singular.
pub fn inverse(a: &[Vec<Q>]) -> Option<Vec<Vec<Q>>> {
    let n = a.len();
    // augmented [A | I]
    let mut m: Vec<Vec<Q>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { Q::one() } else { Q::zero() });
            }
            r
        })
        .collect();
    // Gauss–Jordan: the pivot row for column `col` is row `col` (every
    // column is pivoted in order, so a separate pivot-row counter would
    // always equal `col`).
    for col in 0..n {
        let piv = (col..n).find(|&r| !m[r][col].is_zero())?;
        m.swap(col, piv);
        let d = m[col][col].clone();
        for c in 0..2 * n {
            m[col][c] = &m[col][c] / &d;
        }
        for r in 0..n {
            if r != col && !m[r][col].is_zero() {
                let f = m[r][col].clone();
                for c in 0..2 * n {
                    m[r][c] = &m[r][c] - &(&f * &m[col][c].clone());
                }
            }
        }
    }
    Some(m.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// One non-zero vector `x` in the (right) null space of the matrix whose
/// rows are `rows` (width `ncols`): `rows · x = 0`. Returns `None` when
/// the columns are independent (null space is `{0}`). When the null
/// space has dimension `> 1`, the vector obtained by setting the first
/// free column to `1` (and the rest to `0`) is returned.
pub fn nullspace_vector(rows: &[Vec<Q>], ncols: usize) -> Option<Vec<Q>> {
    // reduced row echelon form
    let mut m: Vec<Vec<Q>> = rows.to_vec();
    let mut pivots: Vec<usize> = Vec::new();
    let mut pr = 0;
    for col in 0..ncols {
        if pr >= m.len() {
            break;
        }
        if let Some(piv) = (pr..m.len()).find(|&r| !m[r][col].is_zero()) {
            m.swap(pr, piv);
            let d = m[pr][col].clone();
            for c in 0..ncols {
                m[pr][c] = &m[pr][c] / &d;
            }
            for r in 0..m.len() {
                if r != pr && !m[r][col].is_zero() {
                    let f = m[r][col].clone();
                    for c in 0..ncols {
                        m[r][c] = &m[r][c] - &(&f * &m[pr][c].clone());
                    }
                }
            }
            pivots.push(col);
            pr += 1;
        }
    }
    // free columns = those without a pivot
    let free: Vec<usize> = (0..ncols).filter(|c| !pivots.contains(c)).collect();
    let &f0 = free.first()?; // None ⇒ full column rank ⇒ null space {0}
    let mut x = vec![Q::zero(); ncols];
    x[f0] = Q::one();
    // back-substitute: pivot var = -(row · free-assignment)
    for (ri, &pc) in pivots.iter().enumerate() {
        // pivot row ri has a leading 1 at pc; x[pc] = -sum_{c != pc} m[ri][c] x[c]
        let mut s = Q::zero();
        if m[ri][f0] != Q::zero() {
            s += &m[ri][f0] * &x[f0];
        }
        x[pc] = -s;
    }
    Some(x)
}

// ── canonical scaling ──────────────────────────────────────────────

fn content(nums: &[Integer]) -> Integer {
    nums.iter().fold(Integer::zero(), |g, x| g.gcd(x))
}

/// Scale `vals` by a single positive rational so every entry becomes an
/// integer and their gcd is `1` (the *primitive integer* representative).
/// The **sign is preserved** — use this for inequality rows, where the
/// direction is load-bearing. An all-zero input is returned unchanged.
pub fn primitive_preserve_sign(vals: &[Q]) -> Vec<Q> {
    // common denominator
    let mut lcm = Integer::one();
    for v in vals {
        lcm = lcm.lcm(&v.denominator());
    }
    let ints: Vec<Integer> = vals
        .iter()
        .map(|v| v.numerator() * (&lcm / &v.denominator()))
        .collect();
    let g = content(&ints);
    if g.is_zero() {
        return vals.to_vec();
    }
    ints.into_iter().map(|x| Q::from(x / &g)).collect()
}

/// Primitive integer representative with a **canonical sign**: the first
/// non-zero entry is made positive. Use this for *rays* / directions,
/// where only the ray (positive scaling class) matters, so two encodings
/// of the same ray compare equal.
pub fn primitive_ray(vals: &[Q]) -> Vec<Q> {
    let mut v = primitive_preserve_sign(vals);
    if let Some(first) = v.iter().find(|x| !x.is_zero()) {
        if first.is_negative() {
            for x in &mut v {
                *x = -&*x;
            }
        }
    }
    v
}
