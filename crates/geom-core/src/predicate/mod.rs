// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Exact orientation and in-sphere predicates, in any dimension.
//!
//! A predicate returns a **sign**, not a number, and that distinction is the
//! reason this module exists. A triangulation is a combinatorial object, and
//! the algorithm that builds it branches on these answers. If rounding makes
//! two calls disagree about the same underlying fact, the result is not a
//! slightly-inaccurate triangulation — it is not a triangulation: cells
//! overlap, cavities fail to be star-shaped, adjacency does not close, and
//! the insertion walk can loop forever. The failures land on grids, circles
//! and spheres, which is to say on the inputs people actually have.
//!
//! So every sign here is exact. The cost is kept near zero by *filtering*:
//!
//! 1. Evaluate the determinant in outward-rounded interval arithmetic
//!    ([`interval`]), starting from the caller's coordinates so that the
//!    subtractions and squarings are themselves bounded rather than assumed.
//! 2. If the resulting interval excludes zero, its sign is *proven* — return.
//! 3. Otherwise fall back to exact rational arithmetic ([`exact`]).
//!
//! Step 3 is unreachable for generic input and unavoidable for degenerate
//! input, which is exactly the trade one wants. Nothing in step 1 can produce
//! a wrong answer; an over-wide interval costs an escalation, never
//! correctness.
//!
//! # Dimension
//!
//! General `d` by construction, with an allocation-free specialisation at
//! `d = 2, 3` — division-free cofactor expansion on stack arrays, where the
//! general route builds a `Vec<Vec<_>>` and eliminates. Being division-free
//! pays twice: it is quicker, and the intervals stay tighter, so it also
//! escalates less often. The crate-owned pinned benchmark measures the current
//! planar filtered paths directly.
//!
//! The specialisation is guarded, not trusted. A fast path that quietly
//! disagrees with the path it accelerates is a correctness bug wearing a
//! speedup's clothes, so both stay reachable — [`orientation_general`] and
//! [`in_sphere_general`] — and `tests/fastpath_agreement.rs` asserts they
//! never differ, across grids coarse enough to force the degenerate branch.
//! `tests/parity_robust.rs` independently pins both against Shewchuk's
//! `robust` at `d = 2, 3`.
//!
//! # Preconditions
//!
//! These are inner-loop functions and take no `Result`. Coordinates must be
//! finite and all points must share a dimension; both are `debug_assert`ed
//! and are caller bugs, not degeneracies to be resolved here.

pub mod exact;
pub mod interval;

use flint_core::Rational;
use interval::Interval;

/// The sign of a determinant: which side, or exactly on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    /// Strictly negative.
    Negative,
    /// Exactly zero — degenerate.
    Zero,
    /// Strictly positive.
    Positive,
}

impl Sign {
    /// Sign of an `f64`, with the interval convention that only an exact
    /// zero is [`Sign::Zero`].
    fn of(x: f64) -> Sign {
        if x > 0.0 {
            Sign::Positive
        } else if x < 0.0 {
            Sign::Negative
        } else {
            Sign::Zero
        }
    }

    /// Product of two signs.
    pub fn times(self, o: Sign) -> Sign {
        match (self, o) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (a, b) if a == b => Sign::Positive,
            _ => Sign::Negative,
        }
    }
}

/// The arithmetic a predicate matrix is assembled in.
///
/// Implemented by the interval filter and by exact rationals, so the matrix
/// construction below is written once and evaluated twice.
trait Arith: Clone {
    fn lift(x: f64) -> Self;
    fn plus(&self, o: &Self) -> Self;
    fn minus(&self, o: &Self) -> Self;
    fn times(&self, o: &Self) -> Self;
}

impl Arith for Interval {
    fn lift(x: f64) -> Self {
        Interval::exact(x)
    }
    fn plus(&self, o: &Self) -> Self {
        Interval::add_interval(*self, *o)
    }
    fn minus(&self, o: &Self) -> Self {
        Interval::sub_interval(*self, *o)
    }
    fn times(&self, o: &Self) -> Self {
        Interval::mul_interval(*self, *o)
    }
}

impl Arith for Rational {
    fn lift(x: f64) -> Self {
        exact::exact(x)
    }
    fn plus(&self, o: &Self) -> Self {
        self + o
    }
    fn minus(&self, o: &Self) -> Self {
        self - o
    }
    fn times(&self, o: &Self) -> Self {
        self * o
    }
}

/// Rows `p_i - p_0`, the orientation matrix of a `d`-simplex.
fn orientation_matrix<A: Arith>(points: &[&[f64]]) -> Vec<Vec<A>> {
    let d = points[0].len();
    (1..points.len())
        .map(|i| {
            (0..d)
                .map(|j| A::lift(points[i][j]).minus(&A::lift(points[0][j])))
                .collect()
        })
        .collect()
}

/// Rows `[p_i - q | |p_i - q|^2 - (w_i - w_q)]`.
///
/// This is the in-sphere matrix after translating to `q` and clearing the
/// linear terms out of the lifted column — a column operation, so the
/// determinant is unchanged, and the entries are relative to `q` rather than
/// to the origin, which is better conditioned. With all weights zero the last
/// column is the plain squared distance and this is the ordinary in-sphere
/// determinant; with weights it is the regular (power) one.
fn in_sphere_matrix<A: Arith>(
    simplex: &[&[f64]],
    weights: Option<&[f64]>,
    q: &[f64],
    wq: f64,
) -> Vec<Vec<A>> {
    let d = q.len();
    simplex
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let diff: Vec<A> = (0..d)
                .map(|j| A::lift(p[j]).minus(&A::lift(q[j])))
                .collect();
            let mut sq = diff[0].times(&diff[0]);
            for e in &diff[1..] {
                sq = sq.plus(&e.times(e));
            }
            if let Some(w) = weights {
                sq = sq.minus(&A::lift(w[i]).minus(&A::lift(wq)));
            }
            let mut row = diff;
            row.push(sq);
            row
        })
        .collect()
}

/// [`orientation`] with the small-dimension path disabled.
///
/// Exists so the specialised route can be differential-tested against the
/// general one. A fast path that quietly disagrees with the path it is meant
/// to accelerate is a correctness bug wearing a speedup's clothes, and the
/// only defence is to keep both reachable and assert they never differ.
#[doc(hidden)]
pub fn orientation_general(points: &[&[f64]]) -> Sign {
    decide(
        || orientation_matrix::<Interval>(points),
        || orientation_matrix::<Rational>(points),
    )
}

/// [`in_sphere_weighted`] with the small-dimension path disabled. See
/// [`orientation_general`].
#[doc(hidden)]
pub fn in_sphere_general(simplex: &[&[f64]], weights: Option<&[f64]>, q: &[f64], wq: f64) -> Sign {
    let orient = orientation_general(simplex);
    if orient == Sign::Zero {
        return Sign::Zero;
    }
    decide(
        || in_sphere_matrix::<Interval>(simplex, weights, q, wq),
        || in_sphere_matrix::<Rational>(simplex, weights, q, wq),
    )
    .times(orient)
    .times(dimension_parity(q.len()))
}

/// The in-sphere determinant's sign relative to "inside" alternates with the
/// ambient dimension, and forgetting that inverts the predicate on odd `d`.
///
/// For a positively oriented simplex and a point strictly inside its
/// circumsphere, `det [p_i - q | |p_i - q|^2]` is positive in `R^2`, negative
/// in `R^3`, positive in `R^4` — the factor is `(-1)^d`. It is not a
/// convention one is free to pick: it falls out of the expansion, and getting
/// it wrong makes `in_sphere` report "outside" for every point that is
/// actually inside, in exactly half of all dimensions.
fn dimension_parity(d: usize) -> Sign {
    if d.is_multiple_of(2) {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

/// Largest matrix the allocation-free path handles. `n = d` for orientation
/// and `n = d + 1` for in-sphere, so this is exactly `d <= 3`.
const SMALL: usize = 4;

/// Orientation in `R^2` and `R^3` without touching the heap.
///
/// `None` means "the filter could not decide", and the caller goes straight
/// to exact arithmetic — not to the general interval path, which uses
/// division and is strictly looser, so it could never succeed where this
/// failed.
fn orientation_small(points: &[&[f64]]) -> Option<Sign> {
    let d = points[0].len();
    if d > SMALL {
        return None;
    }
    let mut m = [Interval::exact(0.0); SMALL * SMALL];
    for i in 1..=d {
        for j in 0..d {
            m[(i - 1) * d + j] =
                Interval::exact(points[i][j]).sub_interval(Interval::exact(points[0][j]));
        }
    }
    let iv = interval::det_small(&m[..d * d], d);
    iv.excludes_zero().then(|| Sign::of(iv.midpoint()))
}

/// In-sphere in `R^2` and `R^3` without touching the heap. As
/// [`orientation_small`], `None` sends the caller to exact arithmetic.
fn in_sphere_small(
    simplex: &[&[f64]],
    weights: Option<&[f64]>,
    q: &[f64],
    wq: f64,
) -> Option<Sign> {
    let d = q.len();
    let n = d + 1;
    if n > SMALL {
        return None;
    }
    let mut m = [Interval::exact(0.0); SMALL * SMALL];
    for (i, p) in simplex.iter().enumerate() {
        let mut sq = Interval::exact(0.0);
        for j in 0..d {
            let e = Interval::exact(p[j]).sub_interval(Interval::exact(q[j]));
            m[i * n + j] = e;
            sq = sq.add_interval(e.mul_interval(e));
        }
        if let Some(w) = weights {
            sq = sq.sub_interval(Interval::exact(w[i]).sub_interval(Interval::exact(wq)));
        }
        m[i * n + d] = sq;
    }
    let iv = interval::det_small(&m[..n * n], n);
    iv.excludes_zero().then(|| Sign::of(iv.midpoint()))
}

/// Decide a sign from a matrix builder: interval first, exact on doubt.
fn decide<F, G>(build_interval: F, build_exact: G) -> Sign
where
    F: FnOnce() -> Vec<Vec<Interval>>,
    G: FnOnce() -> Vec<Vec<Rational>>,
{
    if let Some(iv) = interval::det(build_interval()) {
        if iv.excludes_zero() {
            return Sign::of(iv.midpoint());
        }
        if iv.is_exactly_zero() {
            return Sign::Zero;
        }
    }
    exact::det_sign(build_exact())
}

/// Orientation of `d + 1` points in `R^d`.
///
/// [`Sign::Positive`] for a positively oriented simplex, [`Sign::Zero`] when
/// the points are affinely dependent — which is the useful degenerate answer,
/// since such a simplex has no circumsphere and cannot be Delaunay.
///
/// ```
/// use geom_core::predicate::{orientation, Sign};
///
/// let ccw: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
/// assert_eq!(orientation(&ccw), Sign::Positive);
///
/// let collinear: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 1.0], &[2.0, 2.0]];
/// assert_eq!(orientation(&collinear), Sign::Zero);
/// ```
pub fn orientation(points: &[&[f64]]) -> Sign {
    debug_assert!(!points.is_empty(), "orientation needs at least one point");
    debug_assert_eq!(
        points.len(),
        points[0].len() + 1,
        "orientation needs exactly d + 1 points in R^d"
    );
    debug_assert!(
        points.iter().all(|p| p.iter().all(|x| x.is_finite())),
        "coordinates must be finite"
    );
    if let Some(s) = orientation_small(points) {
        return s;
    }
    decide(
        || orientation_matrix::<Interval>(points),
        || orientation_matrix::<Rational>(points),
    )
}

/// Is `q` inside the circumsphere of `simplex`?
///
/// [`Sign::Positive`] strictly inside, [`Sign::Negative`] strictly outside,
/// [`Sign::Zero`] exactly on it. A degenerate `simplex` has no circumsphere
/// and yields [`Sign::Zero`].
///
/// ```
/// use geom_core::predicate::{in_sphere, Sign};
///
/// let tri: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
/// assert_eq!(in_sphere(&tri, &[0.4, 0.4]), Sign::Positive);
/// assert_eq!(in_sphere(&tri, &[9.0, 9.0]), Sign::Negative);
/// // The fourth corner of the unit square is exactly cocircular.
/// assert_eq!(in_sphere(&tri, &[1.0, 1.0]), Sign::Zero);
/// ```
pub fn in_sphere(simplex: &[&[f64]], q: &[f64]) -> Sign {
    in_sphere_inner(simplex, None, q, 0.0)
}

/// The regular (power / weighted) in-sphere predicate.
///
/// Each site carries a weight, and "inside" means having smaller power
/// `|q - p|^2 - w_p` than the orthosphere. This is what makes a *regular*
/// triangulation well defined on cospherical input — points on a sphere,
/// which is the common case in topological data analysis rather than a
/// corner case — where the unweighted predicate is systematically zero and
/// every cospherical subset would otherwise be admitted at once.
///
/// With all weights equal this agrees with [`in_sphere`].
pub fn in_sphere_weighted(simplex: &[&[f64]], weights: &[f64], q: &[f64], wq: f64) -> Sign {
    debug_assert_eq!(simplex.len(), weights.len(), "one weight per site");
    in_sphere_inner(simplex, Some(weights), q, wq)
}

fn in_sphere_inner(simplex: &[&[f64]], weights: Option<&[f64]>, q: &[f64], wq: f64) -> Sign {
    debug_assert!(!simplex.is_empty(), "in_sphere needs a simplex");
    debug_assert_eq!(
        simplex.len(),
        q.len() + 1,
        "in_sphere needs a d-simplex and a point of R^d"
    );
    let orient = orientation(simplex);
    if orient == Sign::Zero {
        return Sign::Zero;
    }
    let raw = in_sphere_small(simplex, weights, q, wq).unwrap_or_else(|| {
        decide(
            || in_sphere_matrix::<Interval>(simplex, weights, q, wq),
            || in_sphere_matrix::<Rational>(simplex, weights, q, wq),
        )
    });
    raw.times(orient).times(dimension_parity(q.len()))
}
