// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Exact determinant sign — the predicate's backstop.
//!
//! Reached only when [`super::interval`] cannot separate the determinant from
//! zero, which for generic input essentially never happens and for
//! deliberately degenerate input (a grid, a circle, a sphere) always does.
//! Speed is therefore not the design goal here; being unarguably right is.
//!
//! Every finite `f64` is a dyadic rational, so the input carries no error of
//! its own — the coordinates *are* exactly the numbers the caller gave us,
//! and the only question is whether the arithmetic on them rounds.
//! FLINT-backed [`Rational`] answers that by not rounding. Fraction-free elimination
//! would be faster, but this path is rare enough that clarity wins.

use flint_core::{Integer, Rational};

use super::Sign;

/// An `f64` as the exact rational it already is.
///
/// Panics on NaN or infinity; callers validate coordinates before reaching a
/// predicate, and a non-finite coordinate is a bug upstream rather than a
/// degeneracy to resolve here.
pub fn exact(x: f64) -> Rational {
    Rational::from_f64_exact(x).expect("coordinate must be finite")
}

/// Sign of the determinant of a square rational matrix.
///
/// Plain Gaussian elimination over `Rational`. Coefficient growth is real
/// but bounded by the matrix size, which is `d + 2` — small.
pub fn det_sign(mut m: Vec<Vec<Rational>>) -> Sign {
    let n = m.len();
    if n == 0 {
        return Sign::Positive;
    }
    let mut negative = false;

    for col in 0..n {
        let piv = (col..n).find(|&r| !m[r][col].is_zero());
        let piv = match piv {
            Some(r) => r,
            // A zero column means a zero determinant, exactly.
            None => return Sign::Zero,
        };
        if piv != col {
            m.swap(piv, col);
            negative = !negative;
        }
        let p = m[col][col].clone();
        let pivot_row = m[col].clone();
        for row in m.iter_mut().take(n).skip(col + 1) {
            if row[col].is_zero() {
                continue;
            }
            let f = &row[col] / &p;
            for (entry, pivot) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                let t = &f * &pivot.clone();
                *entry -= t;
            }
        }
    }

    let mut acc = Rational::one();
    for (i, row) in m.iter().enumerate() {
        acc *= &row[i];
    }
    if acc.is_zero() {
        return Sign::Zero;
    }
    let positive = acc.is_positive() != negative;
    if positive {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

/// An exact integer zero for predicate-matrix assembly.
pub fn zero_int() -> Integer {
    Integer::zero()
}
