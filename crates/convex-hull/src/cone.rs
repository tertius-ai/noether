// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The **double-description** engine (Motzkin; Fukuda–Prodon revisited):
//! the extreme rays of a *pointed* polyhedral cone
//! `C = { x ∈ Rⁿ : ⟨aᵢ, x⟩ ≤ 0 ∀i }`, computed exactly over `Q`.
//!
//! This one primitive drives **both** polytope enumeration directions.
//! H→V homogenises the polytope to a cone and reads the
//! vertices off the extreme rays; V→H runs the same engine on the
//! **polar** cone, whose extreme rays are the facet normals. Adding the
//! inequalities one at a time is exactly the incremental step the
//! beneath–beyond method performs on the polar (see [`crate::beneath_beyond()`]).

// The double-description seed extracts columns of a matrix inverse and
// indexes inequality rows by position; index loops are the natural,
// clear form. (Elimination borrows two rows at once — see linalg.rs.)
#![allow(clippy::needless_range_loop)]

use crate::linalg::{dot, independent_rows, inverse, primitive_preserve_sign};
use crate::Q;
use std::cmp::Ordering;

/// A compact set of inequality indices tight at a ray (its *zero set*).
#[derive(Clone)]
struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    fn new(nbits: usize) -> Self {
        BitSet {
            bits: vec![0u64; nbits.div_ceil(64)],
        }
    }
    fn set(&mut self, i: usize) {
        self.bits[i / 64] |= 1u64 << (i % 64);
    }
    fn intersect(&self, other: &BitSet) -> BitSet {
        BitSet {
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(a, b)| a & b)
                .collect(),
        }
    }
    /// `self ⊆ other`?
    fn is_subset_of(&self, other: &BitSet) -> bool {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .all(|(a, b)| a & !b == 0)
    }
}

#[derive(Clone)]
struct Ray {
    coords: Vec<Q>,
    tight: BitSet,
}

/// Extreme rays of the pointed cone `{ x : ⟨row, x⟩ ≤ 0 for row in ineqs }`,
/// each `row` and each returned ray a vector in `Rⁿ`.
///
/// Returns `None` when the cone is **not pointed** (the inequality
/// normals do not span `Rⁿ`, i.e. `rank < n` — the cone contains a
/// line). Rays are returned as **primitive-integer** representatives with
/// a canonical sign; degenerate duplicates are removed.
pub fn extreme_rays(ineqs: &[Vec<Q>], n: usize) -> Option<Vec<Vec<Q>>> {
    if n == 0 {
        return None;
    }
    let m = ineqs.len();
    // A pointed cone needs n independent inequality rows to seed a
    // simplicial cone. If the rank is < n the cone has a lineality.
    let basis = independent_rows(ineqs, n);
    if basis.len() < n {
        return None;
    }
    let basis = &basis[..n];

    // Seed: the simplicial cone { A_B x ≤ 0 }. With A_B invertible its n
    // extreme rays are r_j = -(column j of A_B⁻¹): A_B · r_j = -e_j, so
    // basis row j is strict and every other basis row is tight.
    let a_b: Vec<Vec<Q>> = basis.iter().map(|&i| ineqs[i].clone()).collect();
    let inv = inverse(&a_b)?;
    let mut rays: Vec<Ray> = Vec::with_capacity(n);
    for j in 0..n {
        let coords: Vec<Q> = (0..n).map(|k| -&inv[k][j]).collect();
        let mut tight = BitSet::new(m);
        for (k, &bi) in basis.iter().enumerate() {
            if k != j {
                tight.set(bi);
            }
        }
        rays.push(Ray { coords, tight });
    }
    // The basis can contain every inequality, in which case the incremental
    // loop below has no work to do. Canonicalize the seed here so the public
    // promise of primitive-integer rays does not depend on seeing a later
    // halfspace.
    dedup_rays(&mut rays);

    // Incrementally intersect with each remaining halfspace.
    for i in 0..m {
        if basis.contains(&i) {
            continue;
        }
        let a = &ineqs[i];
        let vals: Vec<Q> = rays.iter().map(|r| dot(a, &r.coords)).collect();
        let sign: Vec<Ordering> = vals.iter().map(|v| v.cmp(&Q::zero())).collect();

        let pos: Vec<usize> = (0..rays.len())
            .filter(|&k| sign[k] == Ordering::Greater)
            .collect();
        let neg: Vec<usize> = (0..rays.len())
            .filter(|&k| sign[k] == Ordering::Less)
            .collect();

        let mut next: Vec<Ray> = Vec::new();
        // Keep the satisfied side; the ray stays extreme. A ray on the new
        // hyperplane gains i in its zero set.
        for (k, r) in rays.iter().enumerate() {
            match sign[k] {
                Ordering::Equal => {
                    let mut r2 = r.clone();
                    r2.tight.set(i);
                    next.push(r2);
                }
                Ordering::Less => next.push(r.clone()),
                Ordering::Greater => {}
            }
        }
        // Combine each *adjacent* (violated, satisfied) pair into a new
        // ray on the hyperplane ⟨a, ·⟩ = 0.
        for &p in &pos {
            for &q in &neg {
                if !adjacent(&rays, p, q) {
                    continue;
                }
                let sp = &vals[p]; // > 0
                let sq = &vals[q]; // < 0
                                   // r_new = sp·r_q − sq·r_p  (both coefficients positive) ⇒
                                   // ⟨a, r_new⟩ = 0, r_new in the cone.
                let coords: Vec<Q> = (0..n)
                    .map(|c| sp * &rays[q].coords[c] - sq * &rays[p].coords[c])
                    .collect();
                let mut tight = rays[p].tight.intersect(&rays[q].tight);
                tight.set(i);
                next.push(Ray { coords, tight });
            }
        }
        dedup_rays(&mut next);
        rays = next;
    }

    Some(rays.into_iter().map(|r| r.coords).collect())
}

/// Combinatorial adjacency: rays `p` and `q` are adjacent in the current
/// cone iff no *other* extreme ray's zero set contains `Z(p) ∩ Z(q)`.
fn adjacent(rays: &[Ray], p: usize, q: usize) -> bool {
    let z = rays[p].tight.intersect(&rays[q].tight);
    for (k, r) in rays.iter().enumerate() {
        if k == p || k == q {
            continue;
        }
        if z.is_subset_of(&r.tight) {
            return false;
        }
    }
    true
}

/// Remove degenerate duplicate rays (same primitive direction), keeping
/// the first occurrence, **and rescale every surviving ray to its
/// primitive-integer form**. Rescaling each step is what keeps the
/// double-description coefficient blow-up in check: a combined ray
/// `sp·r_q − sq·r_p` otherwise accumulates ever-larger numerators, and
/// the exact `Q` arithmetic slows to a crawl (an 8 s hull becomes 30 ms).
/// Identical directions share the same zero set, so dropping the later
/// copy loses nothing.
fn dedup_rays(rays: &mut Vec<Ray>) {
    let mut keys: Vec<Vec<Q>> = Vec::with_capacity(rays.len());
    let mut out: Vec<Ray> = Vec::with_capacity(rays.len());
    for mut r in rays.drain(..) {
        let key = primitive_preserve_sign(&r.coords);
        if keys.contains(&key) {
            continue;
        }
        r.coords = key.clone();
        keys.push(key);
        out.push(r);
    }
    *rays = out;
}
