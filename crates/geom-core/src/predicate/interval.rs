// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Midpoint–radius interval arithmetic — the predicate filter.
//!
//! Just enough of an interval type to bound a determinant. Every operation
//! returns an enclosure that provably contains the exact result, so an
//! interval excluding zero is a *proof* of the sign, and one straddling zero
//! is an admission of ignorance that sends the caller to exact arithmetic.
//! Nothing here can produce a wrong sign; the worst it can do is escalate
//! unnecessarily.
//!
//! # Why midpoint–radius rather than lower/upper
//!
//! A value is carried as a centre and a radius, `v ± e`, not as a pair of
//! endpoints. The endpoint form needs four products and a four-way min/max
//! for every multiplication, because either operand may straddle zero; this
//! form needs three products and three sums, and the sign logic disappears.
//! It is also a change of representation rather than a second implementation
//! to keep in step with the first.
//!
//! # Why the constants are generous
//!
//! Rounding modes are not settable from stable Rust, so each operation's own
//! error is bounded rather than rounded outward: `U` is the unit roundoff, so
//! a computed `v` is within `U|v|` of the exact product or sum. The radius is
//! itself computed in floating point, so it is inflated by `(1 + 8U)` to
//! cover the rounding in *its* arithmetic, and `ETA` guards underflow to
//! subnormals.
//!
//! Erring generous is free in the only sense that matters: a radius too large
//! costs an escalation to exact arithmetic, while a radius too small would be
//! a wrong sign. The asymmetry is total, so every constant here rounds the
//! safe way.

/// Unit roundoff: an IEEE operation is within `U` relative of exact.
const U: f64 = f64::EPSILON / 2.0;
/// Inflation covering the rounding in the radius computation itself.
const INFLATE: f64 = 1.0 + 8.0 * U;
/// Underflow guard, so a radius can never round down to exactly zero.
const ETA: f64 = f64::MIN_POSITIVE;

/// An enclosure `[v - e, v + e]` guaranteed to contain the exact value.
#[derive(Clone, Copy, Debug)]
pub struct Interval {
    /// Centre.
    pub v: f64,
    /// Radius, always non-negative.
    pub e: f64,
}

impl Interval {
    /// The exact interval `v ± 0` — an `f64` is exactly itself.
    pub fn exact(x: f64) -> Self {
        Interval { v: x, e: 0.0 }
    }

    /// Does this interval certainly exclude zero? If so, `v`'s sign is proven.
    pub fn excludes_zero(&self) -> bool {
        self.v.abs() > self.e
    }

    /// Is this exactly the number zero, with no uncertainty at all?
    pub fn is_exactly_zero(&self) -> bool {
        self.v == 0.0 && self.e == 0.0
    }

    /// Centre — the sign to report once [`Self::excludes_zero`] holds.
    pub fn midpoint(&self) -> f64 {
        self.v
    }

    /// Sum.
    pub fn add_interval(self, o: Self) -> Self {
        let v = self.v + o.v;
        Interval {
            v,
            e: (self.e + o.e + U * v.abs()) * INFLATE + ETA,
        }
    }

    /// Difference.
    pub fn sub_interval(self, o: Self) -> Self {
        let v = self.v - o.v;
        Interval {
            v,
            e: (self.e + o.e + U * v.abs()) * INFLATE + ETA,
        }
    }

    /// Product. `|a||db| + |b||da| + da·db`, plus the product's own rounding.
    pub fn mul_interval(self, o: Self) -> Self {
        let v = self.v * o.v;
        let e = self.v.abs() * o.e + o.v.abs() * self.e + self.e * o.e + U * v.abs();
        Interval {
            v,
            e: e * INFLATE + ETA,
        }
    }

    /// Quotient. `None` when the divisor straddles zero — the quotient is
    /// then unbounded and the caller must escalate.
    pub fn checked_div(self, o: Self) -> Option<Self> {
        if !o.excludes_zero() {
            return None;
        }
        let v = self.v / o.v;
        // d(a/b) = (da·b - a·db) / b^2, over the worst |b| in the enclosure.
        let denom = o.v.abs() * (o.v.abs() - o.e);
        let e = (self.e * o.v.abs() + self.v.abs() * o.e) / denom + U * v.abs();
        Some(Interval {
            v,
            e: e * INFLATE + ETA,
        })
    }

    /// Largest magnitude in the enclosure — used to choose a pivot.
    pub fn magnitude(&self) -> f64 {
        self.v.abs() + self.e
    }
}

/// Determinant of a small row-major interval matrix, by cofactor expansion.
///
/// Division-free, allocation-free, and unrolled. Two things follow from
/// having no division: it is fast, and the intervals stay far tighter than
/// elimination's do — so this path both answers quicker *and* escalates less
/// often. Only `n <= 4` is handled, which is what `d = 2, 3` needs
/// (orientation is `n = d`, in-sphere is `n = d + 1`); `5!` onward is where
/// cofactor expansion stops being the cheaper option and [`det`] takes over.
pub fn det_small(m: &[Interval], n: usize) -> Interval {
    let at = |r: usize, c: usize| m[r * n + c];
    match n {
        1 => at(0, 0),
        2 => at(0, 0)
            .mul_interval(at(1, 1))
            .sub_interval(at(0, 1).mul_interval(at(1, 0))),
        3 => {
            let c0 = at(1, 1)
                .mul_interval(at(2, 2))
                .sub_interval(at(1, 2).mul_interval(at(2, 1)));
            let c1 = at(1, 0)
                .mul_interval(at(2, 2))
                .sub_interval(at(1, 2).mul_interval(at(2, 0)));
            let c2 = at(1, 0)
                .mul_interval(at(2, 1))
                .sub_interval(at(1, 1).mul_interval(at(2, 0)));
            at(0, 0)
                .mul_interval(c0)
                .sub_interval(at(0, 1).mul_interval(c1))
                .add_interval(at(0, 2).mul_interval(c2))
        }
        4 => {
            // Laplace along the first two rows: the six 2x2 minors of rows
            // 0-1 against the complementary minors of rows 2-3. Fewer
            // multiplications than expanding a row into four 3x3s, and each
            // product is a shorter chain, so the intervals stay tighter.
            let m01 = |a: usize, b: usize| {
                at(0, a)
                    .mul_interval(at(1, b))
                    .sub_interval(at(0, b).mul_interval(at(1, a)))
            };
            let m23 = |a: usize, b: usize| {
                at(2, a)
                    .mul_interval(at(3, b))
                    .sub_interval(at(2, b).mul_interval(at(3, a)))
            };
            m01(0, 1)
                .mul_interval(m23(2, 3))
                .sub_interval(m01(0, 2).mul_interval(m23(1, 3)))
                .add_interval(m01(0, 3).mul_interval(m23(1, 2)))
                .add_interval(m01(1, 2).mul_interval(m23(0, 3)))
                .sub_interval(m01(1, 3).mul_interval(m23(0, 2)))
                .add_interval(m01(2, 3).mul_interval(m23(0, 1)))
        }
        _ => unreachable!("det_small handles n <= 4"),
    }
}

/// Determinant of a square interval matrix, by Gaussian elimination with
/// partial pivoting.
///
/// Returns `None` the moment the computation stops being decisive — a pivot
/// column whose every candidate straddles zero, so neither "singular" nor
/// "nonzero" can be asserted. A `Some` interval provably contains the exact
/// determinant; if it also excludes zero, its sign is proven.
///
/// Elimination is used rather than cofactor expansion because the matrices
/// reach `d + 2` and `(d + 2)!` becomes unaffordable around rank 7. The cost
/// is that division widens intervals faster than a division-free scheme
/// would, so near-degenerate input escalates sooner than strictly necessary.
pub fn det(mut m: Vec<Vec<Interval>>) -> Option<Interval> {
    let n = m.len();
    if n == 0 {
        return Some(Interval::exact(1.0));
    }
    let mut acc = Interval::exact(1.0);
    let mut sign_flips = 0usize;

    for col in 0..n {
        // Pivot on the largest magnitude that certainly is not zero. A row
        // whose entry merely *might* be nonzero cannot be divided by.
        let piv = (col..n)
            .filter(|&r| m[r][col].excludes_zero())
            .max_by(|&a, &b| {
                m[a][col]
                    .magnitude()
                    .partial_cmp(&m[b][col].magnitude())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let piv = match piv {
            Some(r) => r,
            None => {
                // Every remaining entry straddles zero. If they are all
                // exactly zero the determinant is zero, which we can assert;
                // otherwise we genuinely do not know.
                let all_zero = (col..n).all(|r| m[r][col].is_exactly_zero());
                return if all_zero {
                    Some(Interval::exact(0.0))
                } else {
                    None
                };
            }
        };
        if piv != col {
            m.swap(piv, col);
            sign_flips += 1;
        }
        let p = m[col][col];
        acc = acc.mul_interval(p);
        let pivot_row = m[col].clone();
        for row in m.iter_mut().take(n).skip(col + 1) {
            let f = row[col].checked_div(p)?;
            for (entry, pivot) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                let t = f.mul_interval(*pivot);
                *entry = entry.sub_interval(t);
            }
        }
    }
    if !sign_flips.is_multiple_of(2) {
        acc = Interval::exact(0.0).sub_interval(acc);
    }
    Some(acc)
}
