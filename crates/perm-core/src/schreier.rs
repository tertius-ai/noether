// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Schreier orbit / transversal primitives — the substrate the
//! Schreier–Sims construction sits on.
//!
//! Given a base point `b` and a set of generators `S`, the *Schreier
//! tree* rooted at `b` records, for every point `p` in the orbit
//! `b^S`, a witness generator `s ∈ S` and a parent point `p'` with
//! `p' . s = p`. Composing the witnesses along the path back to `b`
//! gives a coset representative — the *transversal element* — mapping
//! `b` to `p`. The orbit, the transversal, and the Schreier
//! generators of the stabilizer of `b` are all read off this one
//! structure.

use crate::perm::{Perm, Point};

/// A Schreier tree rooted at a base point.
///
/// `pred[p]` is `Some((parent, gen_index))` when `p` lies in the
/// orbit, where `parent.apply(generators[gen_index]) == p`. `pred[b]
/// = None` for the root. Indices outside the orbit are also `None`.
#[derive(Debug, Clone)]
pub struct SchreierData {
    /// The base point.
    pub root: Point,
    /// `pred[p] = Some((parent, gen_index))` for non-root orbit
    /// points; `None` for the root and for points not in the orbit.
    pub pred: Vec<Option<(Point, u32)>>,
    /// The orbit, in BFS-from-`root` discovery order. The root is
    /// always `orbit[0]`.
    pub orbit: Vec<Point>,
}

impl SchreierData {
    /// Build the Schreier tree of `root` under `generators` (acting on
    /// the right) by BFS.
    pub fn build(root: Point, generators: &[Perm], degree: usize) -> SchreierData {
        let mut pred: Vec<Option<(Point, u32)>> = vec![None; degree];
        let mut in_orbit = vec![false; degree];
        let mut orbit: Vec<Point> = Vec::new();
        orbit.push(root);
        in_orbit[root as usize] = true;
        let mut head = 0usize;
        while head < orbit.len() {
            let p = orbit[head];
            head += 1;
            for (gi, g) in generators.iter().enumerate() {
                let q = g.apply(p);
                if !in_orbit[q as usize] {
                    in_orbit[q as usize] = true;
                    pred[q as usize] = Some((p, gi as u32));
                    orbit.push(q);
                }
            }
        }
        SchreierData { root, pred, orbit }
    }

    /// All coset representatives, indexed by point. `out[p]` is the
    /// representative mapping `root` to `p`, or `None` if `p` is not
    /// in the orbit.
    pub fn all_coset_reps(&self, generators: &[Perm], degree: usize) -> Vec<Option<Perm>> {
        let mut reps: Vec<Option<Perm>> = vec![None; degree];
        reps[self.root as usize] = Some(Perm::identity(degree));
        // Compute in BFS order: rep[q] = rep[p] * generators[gi].
        for &p in self.orbit.iter().skip(1) {
            let (parent, gi) = self.pred[p as usize].expect("orbit point has a predecessor");
            let parent_rep = reps[parent as usize]
                .as_ref()
                .expect("BFS order computes parent first")
                .clone();
            let rep = parent_rep.compose_unchecked(&generators[gi as usize]);
            reps[p as usize] = Some(rep);
        }
        reps
    }
}
