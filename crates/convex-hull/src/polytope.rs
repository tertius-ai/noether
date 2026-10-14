// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The [`Polytope`] type and the two enumeration directions.
//!
//! A `Polytope` is a bounded convex polytope over `Q`, held in **both**
//! representations — V (its vertices, a subset of any generating point
//! set) and H (its facet inequalities `⟨a, x⟩ ≤ b`) — together with the
//! vertex/facet incidence the face lattice is read from.
//!
//! All facet arithmetic happens inside the polytope's **affine hull**: a
//! `k`-dimensional polytope embedded in `Rᵈ` is reduced to full-dimensional
//! `Rᵏ` coordinates before the double-description engine runs, so a
//! lower-dimensional hull is a first-class result rather than a
//! degenerate error. Vertices are always reported in the original ambient
//! coordinates; facet inequalities are ambient exactly when the polytope
//! is full-dimensional (`k == d`).

use crate::cone::extreme_rays;
use crate::error::HullError;
use crate::face_lattice::{f_vector_from, FaceLattice};
use crate::linalg::{affine_rank, dot, independent_rows, inverse, rank};
use crate::Q;
use std::cmp::Ordering;
use std::collections::HashSet;

/// A facet-defining inequality `⟨normal, x⟩ ≤ rhs`, in the polytope's
/// working (affine-hull) coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Facet {
    /// The (outer) facet normal.
    pub normal: Vec<Q>,
    /// The right-hand side.
    pub rhs: Q,
}

/// A bounded convex polytope over `Q` in both representations.
#[derive(Clone, Debug)]
pub struct Polytope {
    ambient_dim: usize,
    dim: isize,
    vertices: Vec<Vec<Q>>,
    /// Vertices in affine-hull (`Rᵏ`) coordinates; parallel to `vertices`.
    reduced_vertices: Vec<Vec<Q>>,
    facets: Vec<Facet>,
    /// `incidence[f]` = sorted indices of the vertices lying on facet `f`.
    incidence: Vec<Vec<usize>>,
    /// `true` when the hull is full-dimensional (`dim == ambient_dim`), in
    /// which case facet inequalities are in ambient coordinates.
    full_dim: bool,
}

/// Lexicographic comparison of two rational vectors.
fn cmp_vec(a: &[Q], b: &[Q]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.cmp(y) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    a.len().cmp(&b.len())
}

/// De-duplicate an exact point set, preserving first-seen order, and
/// validate a common non-zero ambient dimension.
pub(crate) fn clean_points(points: &[Vec<Q>]) -> Result<(usize, Vec<Vec<Q>>), HullError> {
    if points.is_empty() {
        return Err(HullError::EmptyInput);
    }
    let d = points[0].len();
    if d == 0 {
        return Err(HullError::ZeroAmbientDim);
    }
    for p in points {
        if p.len() != d {
            return Err(HullError::DimensionMismatch {
                expected: d,
                got: p.len(),
            });
        }
    }
    let mut seen: HashSet<&[Q]> = HashSet::with_capacity(points.len());
    let mut out: Vec<Vec<Q>> = Vec::with_capacity(points.len());
    for p in points {
        if seen.insert(p) {
            out.push(p.clone());
        }
    }
    Ok((d, out))
}

/// Reduce points to full-dimensional coordinates in their `k`-dimensional
/// affine hull: point `p₀` maps to the origin and the chosen basis
/// directions map to the standard basis of `Rᵏ`.
pub(crate) fn reduce_to_affine_hull(pts: &[Vec<Q>], d: usize, k: usize) -> Vec<Vec<Q>> {
    let o = &pts[0];
    let diffs: Vec<Vec<Q>> = pts[1..]
        .iter()
        .map(|p| (0..d).map(|c| &p[c] - &o[c]).collect())
        .collect();
    let basis_idx = independent_rows(&diffs, d);
    let basis: Vec<Vec<Q>> = basis_idx[..k].iter().map(|&i| diffs[i].clone()).collect();
    // bt[i] = i-th ambient coordinate across the k basis vectors.
    let bt: Vec<Vec<Q>> = (0..d)
        .map(|i| (0..k).map(|c| basis[c][i].clone()).collect())
        .collect();
    let rows_idx = independent_rows(&bt, k);
    let r_set: Vec<usize> = rows_idx[..k].to_vec();
    let b_r: Vec<Vec<Q>> = r_set.iter().map(|&i| bt[i].clone()).collect();
    let b_r_inv = inverse(&b_r).expect("basis rows are independent by construction");
    pts.iter()
        .map(|x| {
            let rhs: Vec<Q> = r_set.iter().map(|&r| &x[r] - &o[r]).collect();
            (0..k)
                .map(|c| {
                    let mut s = Q::zero();
                    for (j, rj) in rhs.iter().enumerate() {
                        s += &b_r_inv[c][j] * rj;
                    }
                    s
                })
                .collect()
        })
        .collect()
}

/// The facet inequalities of `conv(reduced)` for a **full-dimensional**
/// point set in `Rᵏ` (`k ≥ 1`), via the double-description method on the
/// polar cone `{ y : ⟨y, (1, uⱼ)⟩ ≤ 0 }`, whose extreme rays are the
/// facet normals. Returns `None` only if the polar cone is not pointed
/// (impossible for a full-dimensional hull).
pub(crate) fn facets_from_points(reduced: &[Vec<Q>], k: usize) -> Option<Vec<Facet>> {
    let gens: Vec<Vec<Q>> = reduced
        .iter()
        .map(|u| {
            let mut g = Vec::with_capacity(k + 1);
            g.push(Q::one());
            g.extend(u.iter().cloned());
            g
        })
        .collect();
    let rays = extreme_rays(&gens, k + 1)?;
    let mut facets: Vec<Facet> = Vec::with_capacity(rays.len());
    for y in rays {
        // ray y = (y₀, y₁..y_k); inequality ⟨a, u⟩ ≤ b with a = (y₁..y_k),
        // b = −y₀ (every generator (1,uⱼ) satisfies ⟨y,·⟩ ≤ 0).
        let normal: Vec<Q> = y[1..].to_vec();
        let rhs = -&y[0];
        facets.push(Facet { normal, rhs });
    }
    Some(facets)
}

impl Polytope {
    /// The convex hull of a finite rational point set — the V-rep in,
    /// the full polytope (V + H + incidence) out. Duplicate and interior
    /// points are silently absorbed; a point set of affine dimension
    /// `k < d` yields a valid `k`-dimensional polytope.
    pub fn convex_hull(points: &[Vec<Q>]) -> Result<Polytope, HullError> {
        let (d, pts) = clean_points(points)?;
        let k = affine_rank(&pts, d); // ≥ 0 for a non-empty set

        // A single point (all inputs equal): a 0-dimensional polytope.
        if k == 0 {
            return Ok(Polytope {
                ambient_dim: d,
                dim: 0,
                vertices: vec![pts[0].clone()],
                reduced_vertices: vec![vec![]],
                facets: vec![],
                incidence: vec![],
                full_dim: d == 0,
            });
        }
        let kk = k as usize;
        let full_dim = kk == d;
        let reduced = if full_dim {
            pts.clone()
        } else {
            reduce_to_affine_hull(&pts, d, kk)
        };

        let facets = facets_from_points(&reduced, kk).ok_or(HullError::Infeasible)?;

        // A reduced point is a vertex iff the normals of the facets tight
        // at it span Rᵏ (≥ k independent facets meet there).
        let mut keep: Vec<(Vec<Q>, Vec<Q>)> = Vec::new();
        for (j, u) in reduced.iter().enumerate() {
            let tight: Vec<Vec<Q>> = facets
                .iter()
                .filter(|f| dot(&f.normal, u) == f.rhs)
                .map(|f| f.normal.clone())
                .collect();
            if rank(&tight, kk) == kk {
                keep.push((pts[j].clone(), u.clone()));
            }
        }
        keep.sort_by(|a, b| cmp_vec(&a.0, &b.0));
        let vertices: Vec<Vec<Q>> = keep.iter().map(|(a, _)| a.clone()).collect();
        let reduced_vertices: Vec<Vec<Q>> = keep.iter().map(|(_, r)| r.clone()).collect();

        let incidence: Vec<Vec<usize>> = facets
            .iter()
            .map(|f| {
                (0..reduced_vertices.len())
                    .filter(|&vi| dot(&f.normal, &reduced_vertices[vi]) == f.rhs)
                    .collect()
            })
            .collect();

        Ok(Polytope {
            ambient_dim: d,
            dim: k,
            vertices,
            reduced_vertices,
            facets,
            incidence,
            full_dim,
        })
    }

    /// Build a polytope from an H-representation `⟨aᵢ, x⟩ ≤ bᵢ`. Runs
    /// [`vertex_enumeration`] and hulls the result, so the returned
    /// polytope is fully canonical (redundant inequalities dropped).
    pub fn from_inequalities(
        a_rows: &[Vec<Q>],
        b: &[Q],
        dim: usize,
    ) -> Result<Polytope, HullError> {
        let verts = vertex_enumeration(a_rows, b, dim)?;
        Polytope::convex_hull(&verts)
    }

    /// Ambient dimension `d`.
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }
    /// Affine dimension `k` (`-1` only for the empty polytope, never
    /// produced by a constructor here).
    pub fn dim(&self) -> isize {
        self.dim
    }
    /// The vertices, in ambient coordinates, lexicographically sorted.
    pub fn vertices(&self) -> &[Vec<Q>] {
        &self.vertices
    }
    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }
    /// The facet inequalities (`⟨normal, x⟩ ≤ rhs`). Ambient coordinates
    /// iff [`Polytope::is_full_dimensional`].
    pub fn facets(&self) -> &[Facet] {
        &self.facets
    }
    /// Number of facets.
    pub fn n_facets(&self) -> usize {
        self.facets.len()
    }
    /// `true` iff `dim == ambient_dim` (facet inequalities are ambient).
    pub fn is_full_dimensional(&self) -> bool {
        self.full_dim
    }
    /// `incidence[f]` = the vertex indices on facet `f`.
    pub fn vertex_facet_incidence(&self) -> &[Vec<usize>] {
        &self.incidence
    }

    /// The `f`-vector `(f₋₁, f₀, …, f_k)`: `f₋₁ = 1` (the empty face),
    /// `f₀ =` #vertices, …, `f_k = 1` (the whole polytope). Matches the
    /// Sage / Magma convention.
    pub fn f_vector(&self) -> Vec<usize> {
        f_vector_from(
            self.reduced_vertices.len(),
            &self.reduced_vertices,
            &self.incidence,
            self.dim,
        )
    }

    /// The face lattice (Hasse diagram) of the polytope.
    pub fn face_lattice(&self) -> FaceLattice {
        FaceLattice::compute(
            self.reduced_vertices.len(),
            &self.reduced_vertices,
            &self.incidence,
            self.dim,
        )
    }

    /// Is `point` (ambient coordinates) a vertex of this polytope?
    pub fn is_vertex(&self, point: &[Q]) -> bool {
        self.vertices.iter().any(|v| v.as_slice() == point)
    }

    /// Is `⟨normal, x⟩ ≤ rhs` (ambient coordinates) a facet-defining
    /// inequality of this polytope? Coordinate-free: the inequality must
    /// be valid on every vertex and its contact face must have dimension
    /// `k − 1`.
    pub fn is_facet(&self, normal: &[Q], rhs: &Q) -> bool {
        if normal.len() != self.ambient_dim || self.dim < 1 {
            return false;
        }
        let mut on: Vec<Vec<Q>> = Vec::new();
        for (vi, v) in self.vertices.iter().enumerate() {
            let val = dot(normal, v);
            match val.cmp(rhs) {
                Ordering::Greater => return false, // violated ⇒ not a valid inequality
                Ordering::Equal => on.push(self.reduced_vertices[vi].clone()),
                Ordering::Less => {}
            }
        }
        affine_rank(&on, self.dim as usize) == self.dim - 1
    }
}

/// The convex hull of a finite rational point set (free-function form of
/// [`Polytope::convex_hull`]).
pub fn convex_hull(points: &[Vec<Q>]) -> Result<Polytope, HullError> {
    Polytope::convex_hull(points)
}

/// **H → V by the double-description method.** Given `⟨aᵢ, x⟩ ≤ bᵢ`
/// (`i = 0..m`) in `Rᵈ`, return the vertices of the polytope. Returns
/// [`HullError::Unbounded`] if the region is unbounded (a recession ray
/// or lineality exists) and [`HullError::Infeasible`] if it is empty.
pub fn double_description(
    a_rows: &[Vec<Q>],
    b: &[Q],
    dim: usize,
) -> Result<Vec<Vec<Q>>, HullError> {
    if dim == 0 {
        return Err(HullError::ZeroAmbientDim);
    }
    if a_rows.len() != b.len() {
        return Err(HullError::DimensionMismatch {
            expected: a_rows.len(),
            got: b.len(),
        });
    }
    for row in a_rows {
        if row.len() != dim {
            return Err(HullError::DimensionMismatch {
                expected: dim,
                got: row.len(),
            });
        }
    }
    // Homogenise: variable (t, x) ∈ R^{dim+1}, cone
    //   -t ≤ 0,   ⟨aᵢ, x⟩ − bᵢ t ≤ 0.
    let n = dim + 1;
    let mut ineqs: Vec<Vec<Q>> = Vec::with_capacity(a_rows.len() + 1);
    let mut t_row = vec![Q::zero(); n];
    t_row[0] = -Q::one();
    ineqs.push(t_row);
    for (row, bi) in a_rows.iter().zip(b.iter()) {
        let mut r = Vec::with_capacity(n);
        r.push(-bi);
        r.extend(row.iter().cloned());
        ineqs.push(r);
    }
    // Not pointed ⇒ the polyhedron contains a line ⇒ unbounded.
    let rays = extreme_rays(&ineqs, n).ok_or(HullError::Unbounded)?;
    let mut verts: Vec<Vec<Q>> = Vec::new();
    for y in rays {
        let t = &y[0];
        if t.is_zero() {
            return Err(HullError::Unbounded); // recession ray
        }
        let v: Vec<Q> = y[1..].iter().map(|xi| xi / t).collect();
        if !verts.contains(&v) {
            verts.push(v);
        }
    }
    if verts.is_empty() {
        return Err(HullError::Infeasible);
    }
    verts.sort_by(|a, b| cmp_vec(a, b));
    Ok(verts)
}

/// **H → V vertex enumeration** — dispatches to [`double_description`].
pub fn vertex_enumeration(
    a_rows: &[Vec<Q>],
    b: &[Q],
    dim: usize,
) -> Result<Vec<Vec<Q>>, HullError> {
    double_description(a_rows, b, dim)
}

/// **V → H facet enumeration** — dispatches to the incremental
/// [`crate::beneath_beyond::beneath_beyond`] hull. Returns the facet
/// inequalities (`⟨a, x⟩ ≤ b`) together with the polytope's affine
/// dimension `k`; facet coordinates are ambient iff `k == d`.
pub fn facet_enumeration(points: &[Vec<Q>]) -> Result<(Vec<Facet>, isize), HullError> {
    crate::beneath_beyond::beneath_beyond(points)
}
