// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Where distances come from — the input abstraction under every filtration.
//!
//! A Vietoris–Rips filtration needs one thing from its input: the distance
//! between two indices. Point clouds are one way to supply that; a distance
//! matrix is another, and it is strictly more general — many real datasets
//! (phylogenetic, correlation, edit-distance, diffusion) have no Euclidean
//! embedding at all, so requiring points excludes them for no reason.
//!
//! [`Metric`] is that one method. It matters beyond input formats: the whole
//! of a VR complex is determined by pairwise distances, so a `Metric` is
//! enough to generate simplices **on demand** rather than materialise the
//! complex. That is the representation specialist implementations exploit for
//! speed, and having the trait is the precondition for adopting it.
//!
//! Čech, alpha and Delaunay want more than a metric — a circumcentre lives in
//! the ambient space and cannot be recovered from pairwise distances alone —
//! so they take a [`PointCloud`]. That is the real line between the two
//! abstractions, and it is why both live here.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A finite metric on `0..len()`.
///
/// Symmetry and `d(i,i) = 0` are assumed, not checked per call — a
/// constructor validates once instead ([`DistanceMatrix::from_full`]).
pub trait Metric {
    /// Number of points.
    fn len(&self) -> usize;

    /// Distance between two indices.
    fn distance(&self, i: usize, j: usize) -> f64;

    /// Is the metric empty?
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The smallest `t` at which every point lies within `t` of some chosen
    /// centre — ripser's default threshold, and a sane cap when the caller
    /// has no scale in mind. Above it the complex is a cone and every class
    /// has already died.
    fn enclosing_radius(&self) -> f64 {
        (0..self.len())
            .map(|i| {
                (0..self.len())
                    .map(|j| self.distance(i, j))
                    .fold(0.0f64, f64::max)
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// A finite point cloud in `R^d`, stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud {
    coords: Vec<f64>,
    dim: usize,
    n: usize,
}

impl PointCloud {
    /// Build from rows. Every point must have the same dimension.
    pub fn new(points: &[Vec<f64>]) -> Result<Self, String> {
        let Some(first) = points.first() else {
            return Ok(PointCloud {
                coords: Vec::new(),
                dim: 0,
                n: 0,
            });
        };
        let dim = first.len();
        if let Some(bad) = points.iter().position(|p| p.len() != dim) {
            return Err(format!(
                "point {bad} has dimension {} but point 0 has {dim}",
                points[bad].len()
            ));
        }
        // Non-finite coordinates are refused here because this is the last
        // place the caller can be told *which* point was wrong. A NaN makes
        // every comparison false and every predicate meaningless, and the
        // predicates only `debug_assert` finiteness — in release they would
        // propagate it silently into a triangulation.
        for (i, p) in points.iter().enumerate() {
            if let Some(j) = p.iter().position(|x| !x.is_finite()) {
                return Err(format!(
                    "point {i} coordinate {j} is {} — coordinates must be finite",
                    p[j]
                ));
            }
        }
        Ok(PointCloud {
            coords: points
                .iter()
                .flatten()
                .map(|&coordinate| if coordinate == 0.0 { 0.0 } else { coordinate })
                .collect(),
            dim,
            n: points.len(),
        })
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Is the cloud empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ambient dimension — the number of coordinates, which bounds but often
    /// exceeds the dimension the data occupies. See [`affine_rank`].
    pub fn ambient_dim(&self) -> usize {
        self.dim
    }

    /// Coordinates of point `i`.
    pub fn point(&self, i: usize) -> &[f64] {
        assert!(i < self.n, "point index out of bounds");
        &self.coords[i * self.dim..(i + 1) * self.dim]
    }

    /// Euclidean distance between two points.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.point(i)
            .iter()
            .zip(self.point(j))
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// The points as owned rows, for the routines that want them that way.
    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.len()).map(|i| self.point(i).to_vec()).collect()
    }

    /// Dimension of the affine hull of the cloud.
    pub fn affine_rank(&self) -> usize {
        affine_rank(&self.to_rows())
    }

    /// Re-express the cloud isometrically in coordinates for its affine hull.
    pub fn project_to_affine_hull(&self) -> Result<Self, String> {
        Self::new(&project_to_affine_hull(&self.to_rows()))
    }
}

impl Eq for PointCloud {}

impl fmt::Display for PointCloud {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PointCloud(points={}, ambient_dimension={})",
            self.n, self.dim
        )
    }
}

impl Serialize for PointCloud {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_rows().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PointCloud {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let rows = Vec::<Vec<f64>>::deserialize(deserializer)?;
        Self::new(&rows).map_err(serde::de::Error::custom)
    }
}

impl Metric for PointCloud {
    fn len(&self) -> usize {
        PointCloud::len(self)
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        PointCloud::distance(self, i, j)
    }
}

/// An explicit finite metric, stored as a lower-triangular distance matrix.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Row-major lower triangle, excluding the diagonal: entry `(i, j)` with
    /// `i > j` sits at `i*(i-1)/2 + j`.
    lower: Vec<f64>,
    n: usize,
}

impl DistanceMatrix {
    /// Build from the lower triangle, excluding the diagonal, row by row:
    /// `[d(1,0), d(2,0), d(2,1), d(3,0), ...]`.
    pub fn from_lower_triangle(lower: Vec<f64>) -> Result<Self, String> {
        // len = n(n-1)/2  =>  n = (1 + sqrt(1 + 8*len)) / 2
        let len = lower.len();
        let n = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
        if n * (n - 1) / 2 != len {
            return Err(format!(
                "{len} entries is not a lower triangle: the counts are \
                 n(n-1)/2 = 0, 1, 3, 6, 10, ... (nearest n is {n})"
            ));
        }
        if let Some(bad) = lower.iter().position(|d| *d < 0.0 || !d.is_finite()) {
            return Err(format!(
                "entry {bad} is {} — distances must be non-negative and finite",
                lower[bad]
            ));
        }
        Ok(DistanceMatrix { lower, n })
    }

    /// Build from a full square matrix, checking symmetry and a zero diagonal.
    ///
    /// Validated rather than assumed: an asymmetric input is not a metric, and
    /// silently reading its lower triangle would answer a question the caller
    /// did not ask.
    pub fn from_full(rows: &[Vec<f64>]) -> Result<Self, String> {
        let n = rows.len();
        if n == 0 {
            return Ok(Self {
                lower: Vec::new(),
                n: 0,
            });
        }
        if let Some(bad) = rows.iter().position(|r| r.len() != n) {
            return Err(format!(
                "row {bad} has {} entries but the matrix has {n} rows — not square",
                rows[bad].len()
            ));
        }
        for (i, row) in rows.iter().enumerate() {
            if row[i] != 0.0 {
                return Err(format!(
                    "d({i},{i}) = {} — the diagonal must be zero",
                    row[i]
                ));
            }
            for (j, previous_row) in rows.iter().enumerate().take(i) {
                if (row[j] - previous_row[i]).abs() > 1e-12 {
                    return Err(format!(
                        "d({i},{j}) = {} but d({j},{i}) = {} — a metric must be symmetric",
                        row[j], previous_row[i]
                    ));
                }
            }
        }
        let mut lower = Vec::with_capacity(n * (n - 1) / 2);
        for (i, row) in rows.iter().enumerate() {
            lower.extend_from_slice(&row[..i]);
        }
        DistanceMatrix::from_lower_triangle(lower)
    }

    /// Materialise the pairwise distances of a point cloud.
    ///
    /// Worth it whenever the metric is queried more than about `n` times per
    /// point — which the implicit Vietoris–Rips path always is. A cloud
    /// answers `distance` with a coordinate loop and a `sqrt`; this answers it
    /// with a load.
    pub fn from_point_cloud(cloud: &PointCloud) -> DistanceMatrix {
        let n = Metric::len(cloud);
        let mut lower = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in 0..i {
                lower.push(Metric::distance(cloud, i, j));
            }
        }
        DistanceMatrix { lower, n }
    }
}

impl Metric for DistanceMatrix {
    fn len(&self) -> usize {
        self.n
    }

    fn distance(&self, i: usize, j: usize) -> f64 {
        if i == j {
            return 0.0;
        }
        let (hi, lo) = if i > j { (i, j) } else { (j, i) };
        self.lower[hi * (hi - 1) / 2 + lo]
    }
}

/// Dimension of the affine hull of a point set.
///
/// The number that governs the cost of a triangulation, as against
/// [`PointCloud::ambient_dim`], which merely counts coordinates. A Delaunay
/// simplex spans at most the affine hull, so a subset of more than `rank + 1`
/// points is affinely dependent and cannot be Delaunay — building above the
/// rank is pure waste. Embedded clouds routinely carry spare coordinates: the
/// moment embedding of `CP^n` is pinned to `tr = 1`, so nine coordinates
/// carry a rank-eight cloud.
///
/// Modified Gram–Schmidt, with a tolerance that deliberately errs **high**:
/// an overestimate only costs time, an underestimate would silently drop
/// simplices.
///
/// ```
/// use geom_core::cloud::affine_rank;
///
/// // Three collinear points in the plane span a line, not the plane.
/// let line = [vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
/// assert_eq!(affine_rank(&line), 1);
///
/// let triangle = [vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
/// assert_eq!(affine_rank(&triangle), 2);
/// ```
pub fn affine_rank(pts: &[Vec<f64>]) -> usize {
    if pts.is_empty() {
        return 0;
    }
    let d = pts[0].len();
    let scale = pts
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return 0;
    }
    let tol = scale * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for p in pts.iter().skip(1) {
        let mut v: Vec<f64> = (0..d).map(|t| p[t] - pts[0][t]).collect();
        for b in &basis {
            let dot: f64 = v.iter().zip(b).map(|(a, c)| a * c).sum();
            for (value, basis_value) in v.iter_mut().zip(b) {
                *value -= dot * basis_value;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for value in &mut v {
                *value /= norm;
            }
            basis.push(v);
            if basis.len() == d {
                break;
            }
        }
    }
    basis.len()
}

/// Re-express the points in coordinates for their affine hull.
///
/// A triangulation is only defined in the dimension the data spans, and
/// embedded clouds routinely carry spare coordinates — the moment embedding
/// of `CP^n` is pinned to `tr = 1`, so nine coordinates hold a rank-eight
/// cloud, and a Delaunay build would reject it. Projecting onto an
/// orthonormal basis of the hull is an **isometry**, so every distance,
/// circumradius and filtration value is preserved; only the coordinate count
/// changes.
///
/// Returns points of dimension [`affine_rank`]. Distinct from a PCA
/// reduction, which discards directions the data really occupies and is
/// therefore lossy — this discards only directions it does not.
///
/// ```
/// use geom_core::cloud::{affine_rank, project_to_affine_hull};
///
/// // A triangle living on a plane inside R^3.
/// let flat = [vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
/// let out = project_to_affine_hull(&flat);
/// assert_eq!(affine_rank(&flat), 2);
/// assert_eq!(out[0].len(), 2);
///
/// // Distances survive.
/// let before = ((flat[1][0] - flat[2][0]).powi(2) + (flat[1][1] - flat[2][1]).powi(2)).sqrt();
/// let after = ((out[1][0] - out[2][0]).powi(2) + (out[1][1] - out[2][1]).powi(2)).sqrt();
/// assert!((before - after).abs() < 1e-12);
/// ```
pub fn project_to_affine_hull(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if pts.is_empty() {
        return Vec::new();
    }
    let d = pts[0].len();
    let scale = pts
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f64, |m, x| m.max(x.abs()));
    if scale == 0.0 {
        return pts.iter().map(|_| Vec::new()).collect();
    }
    let tol = scale * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for p in pts.iter().skip(1) {
        let mut v: Vec<f64> = (0..d).map(|t| p[t] - pts[0][t]).collect();
        for b in &basis {
            let dot: f64 = v.iter().zip(b).map(|(a, c)| a * c).sum();
            for (value, basis_value) in v.iter_mut().zip(b) {
                *value -= dot * basis_value;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for value in &mut v {
                *value /= norm;
            }
            basis.push(v);
            if basis.len() == d {
                break;
            }
        }
    }
    pts.iter()
        .map(|p| {
            basis
                .iter()
                .map(|b| (0..d).map(|t| (p[t] - pts[0][t]) * b[t]).sum())
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_triangle_round_trips() {
        // 4 points: 6 entries.
        let d = DistanceMatrix::from_lower_triangle(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(d.len(), 4);
        assert_eq!(d.distance(1, 0), 1.0);
        assert_eq!(d.distance(2, 0), 2.0);
        assert_eq!(d.distance(2, 1), 3.0);
        assert_eq!(d.distance(3, 2), 6.0);
        // symmetric, zero diagonal
        assert_eq!(d.distance(0, 1), d.distance(1, 0));
        assert_eq!(d.distance(2, 2), 0.0);
    }

    #[test]
    fn a_bad_entry_count_is_rejected_with_the_valid_ones() {
        let e = DistanceMatrix::from_lower_triangle(vec![1.0, 2.0]).unwrap_err();
        assert!(e.contains("0, 1, 3, 6, 10"), "{e}");
    }

    #[test]
    fn asymmetry_is_rejected_rather_than_silently_halved() {
        let e = DistanceMatrix::from_full(&[vec![0.0, 1.0], vec![2.0, 0.0]]).unwrap_err();
        assert!(e.contains("symmetric"), "{e}");
    }

    #[test]
    fn a_nonzero_diagonal_is_rejected() {
        let e = DistanceMatrix::from_full(&[vec![5.0, 1.0], vec![1.0, 0.0]]).unwrap_err();
        assert!(e.contains("diagonal"), "{e}");
    }

    #[test]
    fn a_non_finite_distance_is_rejected() {
        let error = DistanceMatrix::from_lower_triangle(vec![f64::INFINITY]).unwrap_err();
        assert!(error.contains("finite"), "{error}");
    }

    #[test]
    fn an_empty_full_matrix_is_an_empty_metric() {
        let metric = DistanceMatrix::from_full(&[]).unwrap();
        assert!(metric.is_empty());
        assert_eq!(metric.len(), 0);
    }

    #[test]
    fn a_point_cloud_is_a_metric_and_agrees_with_its_matrix() {
        let pts = vec![
            vec![0.0, 0.0],
            vec![3.0, 0.0],
            vec![0.0, 4.0],
            vec![1.0, 1.0],
        ];
        let cloud = PointCloud::new(&pts).unwrap();
        let full: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| Metric::distance(&cloud, i, j)).collect())
            .collect();
        let dm = DistanceMatrix::from_full(&full).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert!((Metric::distance(&cloud, i, j) - dm.distance(i, j)).abs() < 1e-12);
            }
        }
        // the 3-4-5 triangle, as a sanity anchor
        assert!((dm.distance(1, 2) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn enclosing_radius_is_the_min_over_centres_of_the_max_distance() {
        // Colinear 0,1,2: the midpoint sees everything within 1.
        let d = DistanceMatrix::from_full(&[
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ])
        .unwrap();
        assert_eq!(d.enclosing_radius(), 1.0);
    }

    #[test]
    fn affine_rank_sees_through_a_wasteful_embedding() {
        // A square in the plane, given five coordinates and pinned to a
        // hyperplane the way a moment map pins its image.
        let pts: Vec<Vec<f64>> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            .iter()
            .map(|p| vec![p[0], p[1], 0.0, 0.0, 1.0 - p[0] - p[1]])
            .collect();
        assert_eq!(pts[0].len(), 5, "five coordinates");
        assert_eq!(affine_rank(&pts), 2, "two dimensions of data");
    }

    #[test]
    fn a_single_point_and_a_constant_cloud_have_rank_zero() {
        assert_eq!(affine_rank(&[]), 0);
        assert_eq!(affine_rank(&[vec![3.0, 4.0]]), 0);
        assert_eq!(affine_rank(&[vec![0.0, 0.0], vec![0.0, 0.0]]), 0);
    }

    #[test]
    fn zero_dimensional_points_are_not_an_empty_cloud() {
        let cloud = PointCloud::new(&[vec![]]).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud.ambient_dim(), 0);
        assert_eq!(cloud.point(0), &[]);
        assert_eq!(cloud.to_rows(), vec![Vec::<f64>::new()]);

        let empty = PointCloud::new(&[]).unwrap();
        assert!(empty.is_empty());
        assert_ne!(cloud, empty);
    }
}
