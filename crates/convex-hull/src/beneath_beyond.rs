// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The **beneath–beyond** incremental hull (Grünbaum; Kallay): points are
//! inserted one at a time; for each new point the *visible* facets are
//! removed and the *horizon* ridges are joined to the point to form new
//! facets. It is the V→H enumeration algorithm dual to the batch
//! [`crate::cone`] double-description used for H→V.
//!
//! Runs inside the affine hull (full-dimensional `Rᵏ`), with a fixed
//! interior reference point — the centroid of the seed simplex, which is
//! strictly interior to every larger hull — orienting each facet. All
//! arithmetic is exact, so the visible/coplanar/beneath trichotomy is
//! *decided*, and coplanar facets are absorbed rather than duplicated
//! (the non-simplicial case).

use crate::error::HullError;
use crate::linalg::{affine_rank, dot, nullspace_vector, primitive_preserve_sign};
use crate::polytope::{clean_points, reduce_to_affine_hull, Facet};
use crate::Q;
use std::cmp::Ordering;
use std::collections::HashSet;

struct ActiveFacet {
    facet: Facet,
    /// Sorted processed-point indices on this facet.
    vertices: Vec<usize>,
}

/// V → H by beneath–beyond: the facet inequalities of `conv(points)`
/// together with the affine dimension `k`. Facet coordinates are ambient
/// iff the hull is full-dimensional (`k == d`); otherwise they are in the
/// polytope's affine-hull coordinate frame.
pub fn beneath_beyond(points: &[Vec<Q>]) -> Result<(Vec<Facet>, isize), HullError> {
    let (d, pts) = clean_points(points)?;
    let k = affine_rank(&pts, d);
    if k == 0 {
        // a single point — no facets.
        return Ok((vec![], 0));
    }
    let kk = k as usize;
    let reduced = if kk == d {
        pts
    } else {
        reduce_to_affine_hull(&pts, d, kk)
    };
    Ok((bb_hull(&reduced, kk), k))
}

/// Canonical key for a facet `⟨a, x⟩ ≤ b`: the primitive-integer form of
/// `[a … , b]`, sign preserved (the inequality direction is load-bearing).
fn facet_key(f: &Facet) -> Vec<Q> {
    let mut v = f.normal.clone();
    v.push(f.rhs.clone());
    primitive_preserve_sign(&v)
}

/// Indices (into `pts`) of `k + 1` affinely independent seed points.
fn seed_simplex(pts: &[Vec<Q>], k: usize) -> Vec<usize> {
    let mut chosen = vec![0usize];
    let mut r = 0usize;
    for j in 1..pts.len() {
        chosen.push(j);
        let sub: Vec<Vec<Q>> = chosen.iter().map(|&i| pts[i].clone()).collect();
        if affine_rank(&sub, k) as usize > r {
            r += 1;
        } else {
            chosen.pop();
        }
        if r == k {
            break;
        }
    }
    chosen
}

fn centroid(pts: &[Vec<Q>], idxs: &[usize], k: usize) -> Vec<Q> {
    let n = Q::from(idxs.len());
    (0..k)
        .map(|c| {
            let mut s = Q::zero();
            for &i in idxs {
                s += &pts[i][c];
            }
            s / &n
        })
        .collect()
}

/// The hyperplane `⟨a, x⟩ = b` through the affinely independent points
/// `idxs` (which must span a `(k−1)`-flat), oriented so the interior
/// reference `interior` is strictly *beneath* it (`⟨a, interior⟩ < b`).
fn oriented_hyperplane(
    pts: &[Vec<Q>],
    idxs: &[usize],
    interior: &[Q],
    k: usize,
) -> Option<(Vec<Q>, Q)> {
    let base = &pts[idxs[0]];
    let diffs: Vec<Vec<Q>> = idxs[1..]
        .iter()
        .map(|&i| (0..k).map(|c| &pts[i][c] - &base[c]).collect())
        .collect();
    let mut a = nullspace_vector(&diffs, k)?;
    let mut b = dot(&a, base);
    match dot(&a, interior).cmp(&b) {
        Ordering::Greater => {
            for x in &mut a {
                *x = -&*x;
            }
            b = -b;
        }
        Ordering::Equal => return None, // interior on the hyperplane — degenerate
        Ordering::Less => {}
    }
    Some((a, b))
}

/// The processed points lying on facet `f`'s hyperplane.
fn on_facet(pts: &[Vec<Q>], processed: &[usize], f: &Facet) -> Vec<usize> {
    let mut v: Vec<usize> = processed
        .iter()
        .copied()
        .filter(|&j| dot(&f.normal, &pts[j]) == f.rhs)
        .collect();
    v.sort_unstable();
    v
}

fn intersect_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) {
    if let Err(position) = values.binary_search(&value) {
        values.insert(position, value);
    }
}

/// The incremental hull in full-dimensional `Rᵏ` (`k ≥ 1`).
fn bb_hull(pts: &[Vec<Q>], k: usize) -> Vec<Facet> {
    let simplex = seed_simplex(pts, k);
    let interior = centroid(pts, &simplex, k);

    // Seed facets: each omits one simplex vertex.
    let mut facets: Vec<ActiveFacet> = Vec::new();
    for i in 0..simplex.len() {
        let face: Vec<usize> = simplex
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &v)| v)
            .collect();
        if let Some((a, b)) = oriented_hyperplane(pts, &face, &interior, k) {
            let facet = Facet { normal: a, rhs: b };
            let vertices = on_facet(pts, &simplex, &facet);
            facets.push(ActiveFacet { facet, vertices });
        }
    }

    let mut processed: Vec<usize> = simplex.clone();

    for q in 0..pts.len() {
        if simplex.contains(&q) {
            continue;
        }
        let vals: Vec<Ordering> = facets
            .iter()
            .map(|active| dot(&active.facet.normal, &pts[q]).cmp(&active.facet.rhs))
            .collect();
        let visible: Vec<usize> = (0..facets.len())
            .filter(|&i| vals[i] == Ordering::Greater)
            .collect();
        if visible.is_empty() {
            // beneath or on the boundary — q does not change the hull.
            for (active, value) in facets.iter_mut().zip(vals.iter()) {
                if *value == Ordering::Equal {
                    insert_sorted_unique(&mut active.vertices, q);
                }
            }
            processed.push(q);
            continue;
        }
        let strict_below: Vec<usize> = (0..facets.len())
            .filter(|&i| vals[i] == Ordering::Less)
            .collect();

        let mut new_facets: Vec<ActiveFacet> = Vec::new();
        let mut new_keys: HashSet<Vec<Q>> = HashSet::new();
        for &visible_index in &visible {
            for &below_index in &strict_below {
                let ridge = intersect_sorted(
                    &facets[visible_index].vertices,
                    &facets[below_index].vertices,
                );
                let ridge_pts: Vec<Vec<Q>> = ridge.iter().map(|&i| pts[i].clone()).collect();
                if affine_rank(&ridge_pts, k) == k as isize - 2 {
                    // horizon ridge — join it to q.
                    let mut fpts = ridge.clone();
                    fpts.push(q);
                    if let Some((a, b)) = oriented_hyperplane(pts, &fpts, &interior, k) {
                        let facet = Facet { normal: a, rhs: b };
                        let key = facet_key(&facet);
                        if new_keys.insert(key) {
                            let mut vertices = on_facet(pts, &processed, &facet);
                            insert_sorted_unique(&mut vertices, q);
                            new_facets.push(ActiveFacet { facet, vertices });
                        }
                    }
                }
            }
        }

        // Keep the non-visible facets; append the deduped new ones.
        let mut kept: Vec<ActiveFacet> = Vec::with_capacity(facets.len() + new_facets.len());
        let mut keys: HashSet<Vec<Q>> = HashSet::new();
        for (index, mut active) in facets.into_iter().enumerate() {
            if vals[index] != Ordering::Greater {
                if vals[index] == Ordering::Equal {
                    insert_sorted_unique(&mut active.vertices, q);
                }
                keys.insert(facet_key(&active.facet));
                kept.push(active);
            }
        }
        for active in new_facets {
            if keys.insert(facet_key(&active.facet)) {
                kept.push(active);
            }
        }
        facets = kept;
        processed.push(q);
    }

    // Canonicalise output (primitive-integer, sign preserved).
    let mut output = Vec::with_capacity(facets.len());
    for mut active in facets {
        let key = facet_key(&active.facet);
        active.facet.normal = key[..k].to_vec();
        active.facet.rhs = key[k].clone();
        output.push(active.facet);
    }
    output
}
