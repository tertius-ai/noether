// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The face lattice (Hasse diagram) of a polytope, and the `f`-vector
//! read off it.
//!
//! Every face of a polytope is an intersection of facets, and every such
//! intersection is a face. So the faces are exactly the **meet-closure**
//! of the facet vertex-sets under intersection (with the whole polytope
//! for the empty index set and the empty face for the full one). Each
//! face's dimension is the exact affine rank of its vertices.
//!
//! The `f`-vector needs only the closure and the per-face dimensions; the
//! `O(faces²)` covering relations are computed only for the full
//! [`FaceLattice`] (never on the `f`-vector hot path).

use crate::linalg::affine_rank;
use crate::Q;
use std::collections::HashSet;

/// One face: its dimension and the sorted indices of the vertices on it.
/// The empty face has `dim == -1` and no vertices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Face {
    /// Affine dimension of the face (`-1` for the empty face).
    pub dim: isize,
    /// Sorted vertex indices spanning the face.
    pub vertices: Vec<usize>,
}

/// The face lattice: all faces plus the covering relations of the Hasse
/// diagram, and the `f`-vector.
#[derive(Clone, Debug)]
pub struct FaceLattice {
    /// All faces, sorted by `(dim, vertices)`.
    pub faces: Vec<Face>,
    /// Covering pairs `(lower, upper)` as indices into `faces`, with
    /// `dim(upper) == dim(lower) + 1`.
    pub covers: Vec<(usize, usize)>,
    /// `f_vector[i]` = number of faces of dimension `i − 1`; so
    /// `f_vector[0] = 1` (empty face) and the last entry `= 1` (whole
    /// polytope).
    pub f_vector: Vec<usize>,
}

fn intersect_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn is_proper_subset(a: &[usize], b: &[usize]) -> bool {
    a.len() < b.len() && a.iter().all(|x| b.binary_search(x).is_ok())
}

/// The faces (dimension + vertex set), sorted by `(dim, vertices)`. No
/// covering relations — the cheap path shared by the `f`-vector.
fn closure(
    nverts: usize,
    reduced_vertices: &[Vec<Q>],
    incidence: &[Vec<usize>],
    poly_dim: isize,
) -> Vec<Face> {
    let red_dim = poly_dim.max(0) as usize;
    let dim_of = |vset: &[usize]| -> isize {
        let pts: Vec<Vec<Q>> = vset.iter().map(|&i| reduced_vertices[i].clone()).collect();
        affine_rank(&pts, red_dim)
    };

    // A single point (0-dimensional polytope) has no facets.
    if incidence.is_empty() {
        let mut faces = vec![Face {
            dim: -1,
            vertices: vec![],
        }];
        if nverts > 0 {
            faces.push(Face {
                dim: 0,
                vertices: (0..nverts).collect(),
            });
        }
        return faces;
    }

    let full: Vec<usize> = (0..nverts).collect();
    let facet_sets: Vec<Vec<usize>> = incidence
        .iter()
        .map(|s| {
            let mut v = s.clone();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    seen.insert(full.clone());
    for fs in &facet_sets {
        seen.insert(fs.clone());
    }
    let mut frontier: Vec<Vec<usize>> = seen.iter().cloned().collect();
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for face in &frontier {
            for fs in &facet_sets {
                let inter = intersect_sorted(face, fs);
                if seen.insert(inter.clone()) {
                    next.push(inter);
                }
            }
        }
        frontier = next;
    }
    // The empty face always exists.
    seen.insert(Vec::new());

    let mut faces: Vec<Face> = seen
        .into_iter()
        .map(|vset| Face {
            dim: dim_of(&vset),
            vertices: vset,
        })
        .collect();
    faces.sort_by(|a, b| a.dim.cmp(&b.dim).then_with(|| a.vertices.cmp(&b.vertices)));
    faces
}

fn f_vector_of(faces: &[Face], poly_dim: isize) -> Vec<usize> {
    let mut f = vec![0usize; (poly_dim.max(0) + 2) as usize];
    for face in faces {
        let idx = (face.dim + 1) as usize;
        if idx < f.len() {
            f[idx] += 1;
        }
    }
    f
}

impl FaceLattice {
    /// Compute the full face lattice — faces, `f`-vector, and the Hasse
    /// covering relations — from the vertex/facet incidence.
    pub fn compute(
        nverts: usize,
        reduced_vertices: &[Vec<Q>],
        incidence: &[Vec<usize>],
        poly_dim: isize,
    ) -> FaceLattice {
        let faces = closure(nverts, reduced_vertices, incidence, poly_dim);
        let f_vector = f_vector_of(&faces, poly_dim);
        // Covers: proper subset with a dimension gap of exactly 1.
        let mut covers = Vec::new();
        for (i, lo) in faces.iter().enumerate() {
            for (j, hi) in faces.iter().enumerate() {
                if hi.dim == lo.dim + 1 && is_proper_subset(&lo.vertices, &hi.vertices) {
                    covers.push((i, j));
                }
            }
        }
        FaceLattice {
            faces,
            covers,
            f_vector,
        }
    }
}

/// The `f`-vector, computed from the incidence via the meet-closure only
/// (no covering relations — see [`FaceLattice`]).
pub fn f_vector_from(
    nverts: usize,
    reduced_vertices: &[Vec<Q>],
    incidence: &[Vec<usize>],
    poly_dim: isize,
) -> Vec<usize> {
    let faces = closure(nverts, reduced_vertices, incidence, poly_dim);
    f_vector_of(&faces, poly_dim)
}
