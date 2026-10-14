// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Circumspheres and minimum enclosing balls.
//!
//! The two ball questions the filtrations ask. They sound alike and are not:
//! a **circumsphere** puts every point on the boundary, and a **minimum
//! enclosing ball** merely contains them. For an acute triangle they coincide;
//! for an obtuse one the enclosing ball is spanned by the longest edge alone
//! and the circumsphere is larger. Alpha wants the first (it reports a
//! circumradius), Čech the second (a simplex enters when its balls first share
//! a point, which is when the enclosing ball reaches the scale).
//!
//! # One monotonicity constraint governs both
//!
//! A filtration must have every face entering no later than its cofaces, and
//! Noether's filtration constructor rejects the complex outright if that fails. Two
//! routines computing the *same* ball by different arithmetic will differ in
//! the last ULP, and an ULP is enough: an edge whose value is computed one way
//! and a triangle spanned by that same edge computed another way can invert,
//! and a correct complex is rejected as malformed.
//!
//! Hence the two-point special case that appears in both functions below.
//! The ball on two points is the midpoint and half the distance, written
//! directly rather than routed through the linear solve, so that whenever a
//! larger simplex's answer *is* an edge's answer the two agree bit for bit
//! rather than nearly.

/// A closed ball: centre and **squared** radius.
///
/// The square is what is stored, because it is the quantity that is exactly
/// representable. For rational input the circumcentre is rational and `r^2`
/// is rational; `r` itself is a quadratic irrational. Keeping `r^2` therefore
/// costs one rounding fewer, removes a `sqrt` from every call, matches
/// GUDHI's convention, and is the representation an exact-rational tier would
/// need — so nothing has to change shape if one is added.
///
/// Consumers that order by radius can order by [`Self::radius_sq`] directly:
/// `x -> sqrt(x)` is monotone, so every comparison, minimum and sort agrees.
#[derive(Clone, Debug, PartialEq)]
pub struct Ball {
    /// Centre.
    pub center: Vec<f64>,
    /// Squared radius. See the type docs for why this rather than the radius.
    pub radius_sq: f64,
}

impl Ball {
    /// The radius. One `sqrt`, taken on demand rather than on construction.
    pub fn radius(&self) -> f64 {
        self.radius_sq.max(0.0).sqrt()
    }
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Midpoint and half-distance, the two-point ball, computed directly.
fn two_point_ball(a: &[f64], b: &[f64]) -> Ball {
    Ball {
        center: a.iter().zip(b).map(|(x, y)| (x + y) / 2.0).collect(),
        // Division by four is exact, so this carries only the rounding in
        // `dist2` — where the old `sqrt(dist2) / 2` carried that plus the
        // square root's.
        radius_sq: dist2(a, b) / 4.0,
    }
}

/// Relative tolerance below which a pivot is treated as an affine dependence.
///
/// Relative, and that matters. The Gram matrix entries are `2<v_i, v_j>`, so
/// they scale as the *square* of the coordinate magnitude: an absolute
/// threshold that correctly rejects a degenerate simplex on the unit square
/// silently accepts one at coordinate 100 and returns a garbage sphere
/// instead of `None`. Alpha reads circumradii straight out of this, so the
/// garbage would become a filtration value. Found by `tests/proptest.rs`.
const EPS: f64 = 1e-12;

/// The sphere through every point of `pts`, centred in their affine hull.
///
/// With `weights = None` this is the ordinary circumsphere. With weights it
/// is the **orthosphere** of the weighted points: the centre has equal *power*
/// `|c - p_i|^2 - w_i` with respect to every site rather than equal distance,
/// which is what makes a regular triangulation well defined where an ordinary
/// one is not. Equal weights reduce to the unweighted case.
///
/// `None` when the points are affinely dependent — no such sphere exists, and
/// in particular the simplex cannot be Delaunay.
///
/// Writing `c = p_0 + sum_j lambda_j (p_j - p_0)` and imposing equal power
/// gives a `k x k` system in the `lambda_j`, solved by Gaussian elimination
/// with partial pivoting.
///
/// ```
/// use geom_core::ball::circumsphere;
///
/// // A 3-4-5 right triangle: the circumcentre is the hypotenuse's midpoint
/// // and the circumradius is half of it.
/// let p = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]];
/// let rows: Vec<&[f64]> = p.iter().map(|q| &q[..]).collect();
/// let b = circumsphere(&rows, None).unwrap();
/// assert!((b.radius() - 2.5).abs() < 1e-12);
/// assert!((b.center[0] - 1.5).abs() < 1e-12);
/// assert!((b.center[1] - 2.0).abs() < 1e-12);
/// ```
pub fn circumsphere(pts: &[&[f64]], weights: Option<&[f64]>) -> Option<Ball> {
    let k = pts.len().checked_sub(1)?;
    let d = pts[0].len();
    if k == 0 {
        return Some(Ball {
            center: pts[0].to_vec(),
            radius_sq: 0.0,
        });
    }
    // Two equally-weighted points: the direct formula, for the bit-exactness
    // reason in the module docs. Unequal weights move the centre off the
    // midpoint, so that case goes through the solve like any other.
    if k == 1 && weights.is_none_or(|w| w[0] == w[1]) {
        return Some(two_point_ball(pts[0], pts[1]));
    }

    let v: Vec<Vec<f64>> = (1..=k)
        .map(|j| (0..d).map(|i| pts[j][i] - pts[0][i]).collect())
        .collect();
    // a[r][c] = 2<v_r, v_c>;  rhs = |v_r|^2 - (w_r - w_0)
    let mut a = vec![vec![0.0f64; k + 1]; k];
    for (r, row) in a.iter_mut().enumerate() {
        for c in 0..k {
            row[c] = 2.0 * v[r].iter().zip(&v[c]).map(|(x, y)| x * y).sum::<f64>();
        }
        row[k] = v[r].iter().map(|x| x * x).sum::<f64>();
        if let Some(w) = weights {
            row[k] -= w[r + 1] - w[0];
        }
    }
    // Scale of the system, for the relative dependence threshold below.
    //
    // No `.max(1.0)` floor. The Gram entries are `2<v_i, v_j>` and scale as
    // the *square* of the coordinates, so a floor of 1 is an absolute
    // threshold in disguise: below coordinate ~1e-6 every entry is smaller
    // than `EPS * 1.0`, and the pivot test then rejects every simplex as
    // affinely dependent. Alpha and the minimum enclosing ball both read
    // circumradii out of here, so that is not a `None` the caller sees — it is
    // a filtration value quietly computed from a fallback.
    //
    // A zero scale is the one case the floor was covering: every `v_j` is the
    // zero vector, the points coincide, and they are affinely dependent for
    // real.
    let scale = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return None;
    }
    for col in 0..k {
        let piv = (col..k).max_by(|&x, &y| {
            a[x][col]
                .abs()
                .partial_cmp(&a[y][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[piv][col].abs() < EPS * scale {
            return None; // affinely dependent
        }
        a.swap(col, piv);
        let pivot_row = a[col].clone();
        for row in a.iter_mut().take(k).skip(col + 1) {
            let f = row[col] / pivot_row[col];
            for (entry, pivot) in row[col..=k].iter_mut().zip(&pivot_row[col..=k]) {
                *entry -= f * pivot;
            }
        }
    }
    let mut lam = vec![0.0f64; k];
    for r in (0..k).rev() {
        let mut s = a[r][k];
        for c in (r + 1)..k {
            s -= a[r][c] * lam[c];
        }
        lam[r] = s / a[r][r];
    }
    let center: Vec<f64> = (0..d)
        .map(|i| pts[0][i] + (0..k).map(|j| lam[j] * v[j][i]).sum::<f64>())
        .collect();
    let mut r2 = dist2(&center, pts[0]);
    if let Some(w) = weights {
        r2 -= w[0];
    }
    Some(Ball {
        center,
        radius_sq: r2.max(0.0),
    })
}
/// The smallest ball containing every point of `pts`.
///
/// Not the circumsphere: for an obtuse triangle the enclosing ball is spanned
/// by the longest edge alone and the circumsphere is strictly larger. This is
/// the radius a simplex enters the Čech filtration at — the scale at which its
/// vertices' balls first share a point.
///
/// `None` only for empty input.
///
/// # Welzl's algorithm, and why the slack is relative to the spread
///
/// The ball is determined by at most `d + 1` of the points, so it can be found
/// by enumerating subsets — but that is `C(n, d+1)` candidates and intractable
/// past a simplex. Welzl's move-to-front randomised incremental algorithm
/// finds the same ball in expected linear time in `n`, and the brute-force
/// enumeration is kept as its differential-test oracle rather than as the
/// implementation.
///
/// The one tolerance is the "is this point already inside?" test, and it must
/// scale with the **spread** of the points — `max_i |p_i - p_0|^2` — not with
/// `1`. An absolute floor added to a squared radius is a slack of 30% once the
/// cloud has diameter `1e-4`, which admits a ball far too small and returns it
/// as the answer. Spread-relative slack is also the only form that respects
/// the problem's translation invariance: the answer for a cloud does not
/// change when it moves away from the origin, so the test deciding it must
/// not either.
pub fn min_enclosing_ball(pts: &[Vec<f64>]) -> Option<Ball> {
    if pts.is_empty() {
        return None;
    }
    let d = pts[0].len();
    if pts.len() == 1 {
        return Some(Ball {
            center: pts[0].clone(),
            radius_sq: 0.0,
        });
    }
    if pts.len() == 2 {
        return Some(two_point_ball(&pts[0], &pts[1]));
    }

    let slack = containment_slack(pts);
    let mut order: Vec<usize> = (0..pts.len()).collect();
    let mut boundary: Vec<usize> = Vec::with_capacity(d + 2);
    welzl(pts, &mut order, pts.len(), &mut boundary, d, slack)
}

/// Absolute slack for the containment test, set by the cloud's own spread.
///
/// Squared distances and circumcentres carry error proportional to the square
/// of the distances involved, so this is the only scale the test can be
/// relative to. `p_0` is an arbitrary reference: any point of the cloud gives
/// the same magnitude up to a factor of two, which a `1e-12` slack absorbs.
fn containment_slack(pts: &[Vec<f64>]) -> f64 {
    let spread2 = pts.iter().map(|p| dist2(p, &pts[0])).fold(0.0f64, f64::max);
    1e-12 * spread2
}

/// The smallest ball with every point of `boundary` on its surface.
///
/// `None` when they are affinely dependent, which exact arithmetic never
/// produces here — Welzl's boundary set is affinely independent by
/// construction — but floating point can on a near-degenerate cloud. The
/// caller treats it as "no improvement" and keeps the ball it had, which is
/// still a valid enclosing ball; the alternative is to propagate a failure out
/// of a function whose contract says it always has an answer.
fn ball_of_boundary(pts: &[Vec<f64>], boundary: &[usize], d: usize) -> Option<Ball> {
    match boundary.len() {
        // Nothing pinned yet: a ball that contains nothing, so the first point
        // examined is pushed onto the boundary. Never returned to a caller —
        // `min_enclosing_ball` handles `n < 3` before this is reached, and any
        // larger cloud pins at least one point.
        0 => Some(Ball {
            center: vec![0.0; d],
            radius_sq: -1.0,
        }),
        1 => Some(Ball {
            center: pts[boundary[0]].clone(),
            radius_sq: 0.0,
        }),
        // Through `two_point_ball`, not the solve — the module docs explain
        // why the two must agree bit for bit and not merely nearly.
        2 => Some(two_point_ball(&pts[boundary[0]], &pts[boundary[1]])),
        _ => {
            let rows: Vec<&[f64]> = boundary.iter().map(|&i| &pts[i][..]).collect();
            circumsphere(&rows, None)
        }
    }
}

/// Welzl's algorithm, move-to-front variant.
///
/// `order[..n]` is the prefix still under consideration. A point found outside
/// the current ball must lie on the boundary of the answer, so it is pinned
/// and the prefix before it re-solved. Moving it to the front afterwards is
/// what makes the expected cost linear: a point that forced a rebuild is
/// likely to force the next one too, and examining it first ends the search
/// sooner.
fn welzl(
    pts: &[Vec<f64>],
    order: &mut Vec<usize>,
    n: usize,
    boundary: &mut Vec<usize>,
    d: usize,
    slack: f64,
) -> Option<Ball> {
    let mut ball = ball_of_boundary(pts, boundary, d)?;
    // `d + 1` points already determine the ball; nothing further can move it.
    if boundary.len() > d {
        return Some(ball);
    }
    let mut i = 0;
    while i < n {
        let p = order[i];
        if dist2(&pts[p], &ball.center) > ball.radius_sq + slack {
            boundary.push(p);
            if let Some(b) = welzl(pts, order, i, boundary, d, slack) {
                ball = b;
            }
            boundary.pop();
            let x = order.remove(i);
            order.insert(0, x);
        }
        i += 1;
    }
    Some(ball)
}

/// The definitional answer: the smallest circumsphere of a subset that
/// contains everything. Exponential, and the oracle `min_enclosing_ball` is
/// tested against rather than the implementation.
#[cfg(test)]
fn min_enclosing_ball_brute(pts: &[Vec<f64>]) -> Option<Ball> {
    if pts.is_empty() {
        return None;
    }
    let d = pts[0].len();
    let n = pts.len();
    assert!(
        n < 20,
        "brute force is exponential; it is a test oracle only"
    );
    let slack = containment_slack(pts);
    let mut best: Option<Ball> = None;
    for mask in 1u32..(1u32 << n) {
        let subset: Vec<&[f64]> = (0..n)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| &pts[i][..])
            .collect();
        if subset.len() > d + 1 {
            continue;
        }
        let Some(b) = circumsphere(&subset, None) else {
            continue;
        };
        let encloses = pts
            .iter()
            .all(|p| dist2(p, &b.center) <= b.radius_sq + slack);
        if encloses && best.as_ref().is_none_or(|c| b.radius_sq < c.radius_sq) {
            best = Some(b);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(p: &[Vec<f64>]) -> Vec<&[f64]> {
        p.iter().map(|q| &q[..]).collect()
    }

    #[test]
    fn an_edge_and_a_triangle_spanning_it_agree_bit_for_bit() {
        // The property the two-point special case exists to protect. An
        // obtuse triangle's enclosing ball IS its longest edge's, and if the
        // two are computed by different arithmetic they differ in the last
        // ULP — enough for the face to enter after its coface and for a
        // correct filtration to be rejected as malformed.
        let a = vec![0.3, -1.7];
        let b = vec![2.9, 0.4];
        let apex = vec![1.6, -0.62];
        let edge = min_enclosing_ball(&[a.clone(), b.clone()]).unwrap();
        let tri = min_enclosing_ball(&[a, b, apex]).unwrap();
        assert_eq!(
            tri.radius_sq, edge.radius_sq,
            "obtuse triangle's MEB must equal its longest edge's exactly"
        );
    }

    /// A deterministic LCG. `Math::random` is not available and a fixed
    /// sequence is what makes a failure reproducible anyway.
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn cloud(n: usize, d: usize, scale: f64, offset: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..n)
            .map(|_| (0..d).map(|_| offset + scale * lcg(&mut s)).collect())
            .collect()
    }

    #[test]
    fn meb_encloses_everything_across_ten_decades_of_scale() {
        // The bug this exists for: the containment slack was
        // `1e-9 * (1.0 + r^2)`, whose `1.0` is an absolute floor. At scale
        // 1e-4 the slack is ~30% of r^2, so a ball far too small passes the
        // containment test and wins for being smallest. The returned radius
        // was 13% short of enclosing the cloud.
        for &scale in &[1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4] {
            for d in 2..=4 {
                let pts = cloud(40, d, scale, 0.0, 0x5eed + d as u64);
                let b = min_enclosing_ball(&pts).expect("non-empty cloud");
                for p in &pts {
                    let over = dist2(p, &b.center) - b.radius_sq;
                    assert!(
                        over <= 1e-9 * b.radius_sq,
                        "scale {scale}, d {d}: point outside by {over} on r^2 {}",
                        b.radius_sq
                    );
                }
            }
        }
    }

    #[test]
    fn meb_is_translation_invariant() {
        // The radius of an enclosing ball does not depend on where the cloud
        // sits, so neither may the computation. A slack relative to `1`
        // rather than to the spread fails this the moment the offset
        // dominates the diameter.
        for &offset in &[0.0, 1.0, 1e3, 1e6] {
            let here = min_enclosing_ball(&cloud(30, 3, 1.0, 0.0, 99)).unwrap();
            let there = min_enclosing_ball(&cloud(30, 3, 1.0, offset, 99)).unwrap();
            assert!(
                (there.radius_sq - here.radius_sq).abs() <= 1e-6 * here.radius_sq,
                "offset {offset}: r^2 moved from {} to {}",
                here.radius_sq,
                there.radius_sq
            );
        }
    }

    #[test]
    fn meb_matches_the_brute_force_oracle() {
        // Welzl against the definition — the smallest circumsphere of a
        // subset that contains everything — on clouds small enough to
        // enumerate.
        for d in 2..=4 {
            for n in [3, 5, 8, 12] {
                for &scale in &[1e-3, 1.0, 1e3] {
                    let pts = cloud(n, d, scale, 0.0, (n * 31 + d) as u64);
                    let fast = min_enclosing_ball(&pts).unwrap();
                    let slow = min_enclosing_ball_brute(&pts).unwrap();
                    assert!(
                        (fast.radius_sq - slow.radius_sq).abs()
                            <= 1e-9 * slow.radius_sq.max(f64::MIN_POSITIVE),
                        "n {n}, d {d}, scale {scale}: {} vs {}",
                        fast.radius_sq,
                        slow.radius_sq
                    );
                }
            }
        }
    }

    #[test]
    fn meb_handles_more_than_thirty_two_points() {
        // The subset enumeration shifted `1u32 << n`, which overflows at
        // n = 32 — a panic in debug and a wrapped, silently wrong answer in
        // release. Welzl has no such bound.
        for n in [31, 32, 33, 200] {
            let pts = cloud(n, 3, 1.0, 0.0, 7);
            let b = min_enclosing_ball(&pts).expect("non-empty cloud");
            for p in &pts {
                assert!(dist2(p, &b.center) <= b.radius_sq * (1.0 + 1e-9) + 1e-30);
            }
        }
    }

    #[test]
    fn meb_of_coincident_points_is_a_point() {
        let pts = vec![vec![2.0, -1.0]; 5];
        let b = min_enclosing_ball(&pts).unwrap();
        assert_eq!(b.radius_sq, 0.0);
        assert_eq!(b.center, vec![2.0, -1.0]);
    }

    #[test]
    fn meb_of_a_sphere_is_that_sphere() {
        // Cospherical input, where every point is on the boundary and the
        // answer is known independently of the algorithm.
        let n = 64;
        let pts: Vec<Vec<f64>> = (0..n)
            .map(|k| {
                let y = 1.0 - 2.0 * k as f64 / (n - 1) as f64;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let th = std::f64::consts::PI * (1.0 + 5.0f64.sqrt()) * k as f64;
                vec![th.cos() * r, y, th.sin() * r]
            })
            .collect();
        let b = min_enclosing_ball(&pts).unwrap();
        assert!((b.radius_sq - 1.0).abs() < 1e-9, "r^2 = {}", b.radius_sq);
        for c in &b.center {
            assert!(c.abs() < 1e-6, "centre {:?} is off the origin", b.center);
        }
    }

    #[test]
    fn meb_of_two_points_is_the_midpoint() {
        let b = min_enclosing_ball(&[vec![0.0, 0.0], vec![2.0, 0.0]]).unwrap();
        assert_eq!(b.center, vec![1.0, 0.0]);
        assert_eq!(b.radius_sq, 1.0);
    }

    #[test]
    fn meb_of_an_equilateral_triangle_is_its_circumcircle() {
        let h = (3.0f64).sqrt() / 2.0;
        let b = min_enclosing_ball(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, h]]).unwrap();
        assert!((b.radius() - 1.0 / (3.0f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn circumsphere_works_at_every_coordinate_scale() {
        // The pivot threshold was floored at `EPS * 1.0`, while the Gram
        // entries scale as the square of the coordinates. Below coordinate
        // ~1e-6 that floor exceeds every entry and the solve rejected
        // perfectly good simplices as affinely dependent. The circumcentre
        // of a right triangle is its hypotenuse's midpoint at every scale.
        for &s in &[1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6] {
            let p = [vec![0.0, 0.0], vec![3.0 * s, 0.0], vec![0.0, 4.0 * s]];
            let b = circumsphere(&rows(&p), None)
                .unwrap_or_else(|| panic!("scale {s}: reported affinely dependent"));
            assert!(
                (b.radius() - 2.5 * s).abs() <= 1e-9 * s,
                "scale {s}: r = {}",
                b.radius()
            );
        }
    }

    #[test]
    fn circumsphere_is_none_on_coincident_points() {
        let p = [vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert!(circumsphere(&rows(&p), None).is_none());
    }

    #[test]
    fn circumsphere_is_none_on_affinely_dependent_points() {
        let collinear = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        assert!(circumsphere(&rows(&collinear), None).is_none());
    }

    #[test]
    fn equal_weights_reduce_to_the_unweighted_circumsphere() {
        // If they diverged, alpha's weighted membership test and its
        // unweighted reported radius would describe different spheres.
        let p = vec![vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 4.0]];
        let plain = circumsphere(&rows(&p), None).unwrap();
        let weighted = circumsphere(&rows(&p), Some(&[0.7, 0.7, 0.7])).unwrap();
        for (a, b) in plain.center.iter().zip(&weighted.center) {
            assert!((a - b).abs() < 1e-12);
        }
        // The weighted radius is the *power* radius, offset by the weight.
        assert!((weighted.radius_sq - (plain.radius_sq - 0.7)).abs() < 1e-12);
    }

    #[test]
    fn a_weighted_centre_moves_toward_the_lighter_site() {
        // Sanity that weights do something, and in the right direction: the
        // orthocentre shifts away from the heavier point.
        let p = vec![vec![0.0, 0.0], vec![2.0, 0.0]];
        let heavy_left = circumsphere(&rows(&p), Some(&[1.0, 0.0])).unwrap();
        assert!(
            heavy_left.center[0] > 1.0,
            "a heavier left site pushes the equal-power point right, got {}",
            heavy_left.center[0]
        );
    }
}
