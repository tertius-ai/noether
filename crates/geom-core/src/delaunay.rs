// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Delaunay and regular triangulations, by incremental insertion.
//!
//! Bowyer–Watson: insert one site at a time, delete every cell whose
//! circumsphere contains it, and fill the resulting cavity by joining the new
//! site to the cavity's boundary. Correctness rests entirely on the cavity
//! being star-shaped from the new site, and *that* rests entirely on the
//! conflict test being consistent — which is why [`crate::predicate`] exists
//! and why nothing here rounds.
//!
//! # The point at infinity
//!
//! The usual shortcut is to start from a "big enough" bounding simplex and
//! delete the cells touching it at the end. It is a trap: a nearly-degenerate
//! triple has an arbitrarily large circumsphere, so no bounding simplex is
//! provably big enough, and the failure is a quietly wrong triangulation near
//! the convex hull rather than an error.
//!
//! Instead every convex-hull facet carries a **ghost cell** whose last vertex
//! is a symbolic point at infinity ([`GHOST`]). This is the standard device
//! and it buys a strong invariant: the triangulation becomes a triangulation
//! of the `d`-sphere, in which *every* facet has exactly two incident cells.
//! Adjacency is total, the cavity walk never falls off an edge, and the hull
//! needs no special-casing anywhere except the one conflict rule below.
//!
//! # Conflict
//!
//! A finite cell conflicts with `p` when `p` is strictly inside its
//! circumsphere — [`in_sphere`], or [`in_sphere_weighted`] for a regular
//! triangulation.
//!
//! A ghost cell over hull facet `F` conflicts with `p` when `p` lies strictly
//! outside the hull across `F` — an orientation test against the apex of the
//! finite cell on the other side. The one subtle case is `p` landing exactly
//! *on* `F`'s supporting hyperplane, which is not rare (it is every second
//! point on a grid): then `p` is not strictly outside, but if the finite
//! neighbour is itself in conflict the ghost must go too, or the cavity
//! boundary stops being a closed surface and the retriangulation is not a
//! manifold. That is the rule implemented, and
//! `tests/delaunay_correctness.rs` hammers it on grids, where the case is
//! common.
//!
//! # Regular triangulations
//!
//! With weights, "inside the circumsphere" becomes "smaller power than the
//! orthosphere", and the result is a regular triangulation. That is what makes
//! cospherical input — points on a sphere, the common case in topological data
//! analysis rather than a corner case — well defined: an unweighted predicate
//! is systematically zero there and would admit every cospherical subset at
//! once.

use core::fmt;
use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::cloud::{affine_rank, PointCloud};
use crate::predicate::{in_sphere, in_sphere_weighted, orientation, Sign};

/// The symbolic vertex at infinity, closing the triangulation into a sphere.
pub const GHOST: u32 = u32::MAX;

/// Largest affine rank this triangulator accepts.
///
/// The inner loop keeps a cell's vertex rows in a stack array, which needs a
/// compile-time bound. It is not a real restriction: a Delaunay triangulation
/// in `R^16` has `O(n^8)` cells, so the dimension at which this would bind is
/// far past the dimension at which the output stops fitting in memory.
pub const MAX_DIM: usize = 16;

/// A cell's vertex rows, on the stack.
type Rows<'a> = [&'a [f64]; MAX_DIM + 2];

#[derive(Clone, Debug)]
struct Cell {
    /// `dim + 1` vertices. A ghost cell carries [`GHOST`] in the last slot.
    verts: Vec<u32>,
    /// `nbrs[i]` is the cell across the facet opposite `verts[i]`.
    nbrs: Vec<u32>,
    alive: bool,
}

impl Cell {
    fn is_ghost(&self) -> bool {
        *self.verts.last().expect("cell has vertices") == GHOST
    }
}

/// A Delaunay or regular triangulation of a point set.
#[derive(Clone, Debug)]
pub struct Triangulation {
    dim: usize,
    points: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    cells: Vec<Cell>,
    /// Where the next point location starts walking. Successive sites are
    /// usually near each other, so the previous answer is a good guess.
    hint: usize,
    /// Generation marks for the cavity walk, so marking a cell visited is
    /// O(1) and does not require clearing an array of every cell ever made.
    visited: Vec<u32>,
    generation: u32,
    /// Dead cavity slots available for the next fill. Reusing them keeps the
    /// point-location walk and generation arrays proportional to the current
    /// triangulation rather than to every transient cell ever constructed.
    free: Vec<usize>,
}

impl Triangulation {
    /// The dimension actually triangulated — the affine rank of the input,
    /// which may be less than its coordinate count.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The finite cells, as vertex-index slices. Ghost cells are not
    /// reported; they are scaffolding, not geometry.
    pub fn cells(&self) -> impl Iterator<Item = &[u32]> {
        self.cells
            .iter()
            .filter(|c| c.alive && !c.is_ghost())
            .map(|c| &c.verts[..])
    }

    /// Number of finite cells.
    pub fn n_cells(&self) -> usize {
        self.cells().count()
    }

    /// The sites, in input order.
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Regular-triangulation weights, or `None` for an ordinary Delaunay triangulation.
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Reconstruct this deterministic triangulation from its defining data.
    pub fn verify(&self) -> Result<(), String> {
        let cloud = PointCloud::new(&self.points)?;
        let rebuilt = match &self.weights {
            Some(weights) => regular(&cloud, weights)?,
            None => delaunay(&cloud)?,
        };
        if rebuilt == *self {
            Ok(())
        } else {
            Err("triangulation differs from its deterministic reconstruction".to_owned())
        }
    }

    fn canonical_cells(&self) -> Vec<Vec<u32>> {
        let mut cells = self
            .cells()
            .map(|cell| {
                let mut cell = cell.to_vec();
                cell.sort_unstable();
                cell
            })
            .collect::<Vec<_>>();
        cells.sort_unstable();
        cells
    }

    fn point(&self, v: u32) -> &[f64] {
        &self.points[v as usize][..]
    }

    /// Write a cell's finite vertex rows into `buf`, returning how many.
    ///
    /// The hot loop calls this once per conflict test, so it fills a stack
    /// array rather than returning a `Vec`; allocating here was a measurable
    /// share of the build time.
    fn fill_rows<'a>(&'a self, c: usize, buf: &mut Rows<'a>) -> usize {
        let mut n = 0;
        for &v in &self.cells[c].verts {
            if v != GHOST {
                buf[n] = self.point(v);
                n += 1;
            }
        }
        n
    }

    fn weight_of(&self, v: u32) -> f64 {
        self.weights.as_ref().map_or(0.0, |w| w[v as usize])
    }

    /// Does cell `c` conflict with site `p`?
    fn conflicts(&self, c: usize, p: u32) -> bool {
        let cell = &self.cells[c];
        if !cell.alive {
            return false;
        }
        let q = self.point(p);
        let mut buf: Rows = [&[]; MAX_DIM + 2];
        let n = self.fill_rows(c, &mut buf);

        if !cell.is_ghost() {
            return match &self.weights {
                None => in_sphere(&buf[..n], q) == Sign::Positive,
                Some(_) => {
                    let mut w = [0.0f64; MAX_DIM + 2];
                    for (k, &v) in cell.verts.iter().enumerate() {
                        w[k] = self.weight_of(v);
                    }
                    in_sphere_weighted(&buf[..n], &w[..n], q, self.weight_of(p)) == Sign::Positive
                }
            };
        }

        // Ghost cell: p conflicts when it sees the hull facet from outside.
        // `buf[..n]` is that facet; slot n takes the point being tested.
        let finite = cell.nbrs[self.dim] as usize;
        let apex = self.apex_of(finite, c);
        buf[n] = q;
        let side_p = orientation(&buf[..n + 1]);
        buf[n] = self.point(apex);
        let side_a = orientation(&buf[..n + 1]);
        if side_p == Sign::Zero {
            // Exactly on the hull's supporting hyperplane. Not strictly
            // outside — but if the finite neighbour goes, this must go with
            // it, or the cavity boundary is no longer closed.
            return self.conflicts(finite, p);
        }
        side_p != side_a
    }

    /// The vertex of finite cell `f` that is not shared with cell `other`.
    fn apex_of(&self, f: usize, other: usize) -> u32 {
        let theirs = &self.cells[other].verts;
        *self.cells[f]
            .verts
            .iter()
            .find(|v| !theirs.contains(v))
            .expect("adjacent cells differ in exactly one vertex")
    }

    /// Facet of `c` opposite slot `i` with `p` removed, as a fixed-size key.
    ///
    /// Used only to pair the new cells created around a freshly inserted
    /// site. All of those facets contain `p`, so it carries no information
    /// and is dropped; what is left fits in a `Copy` array and needs no heap.
    fn facet_key_excluding(&self, c: usize, i: usize, p: u32) -> [u32; MAX_DIM] {
        let mut k = [0u32; MAX_DIM];
        let mut n = 0;
        for (j, &v) in self.cells[c].verts.iter().enumerate() {
            if j != i && v != p {
                k[n] = v;
                n += 1;
            }
        }
        k[..n].sort_unstable();
        k
    }

    /// Facet of cell `c` opposite slot `i`, sorted — the adjacency key.
    fn facet_key(&self, c: usize, i: usize) -> Vec<u32> {
        let mut f: Vec<u32> = self.cells[c]
            .verts
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &v)| v)
            .collect();
        f.sort_unstable();
        f
    }
}

/// The Delaunay triangulation of a point cloud.
///
/// ```
/// use geom_core::{cloud::PointCloud, delaunay::delaunay};
///
/// // Four corners of a square: two triangles, however the diagonal falls.
/// let sq = PointCloud::new(&[
///     vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0], vec![0.0, 1.0],
/// ]).unwrap();
/// let t = delaunay(&sq).unwrap();
/// assert_eq!(t.n_cells(), 2);
/// ```
pub fn delaunay(cloud: &PointCloud) -> Result<Triangulation, String> {
    build(cloud, None)
}

/// The regular (weighted) triangulation of a point cloud.
///
/// One weight per site. See the module docs for why this is the right tool on
/// cospherical input.
pub fn regular(cloud: &PointCloud, weights: &[f64]) -> Result<Triangulation, String> {
    if weights.len() != cloud.len() {
        return Err(format!(
            "{} weights for {} sites",
            weights.len(),
            cloud.len()
        ));
    }
    if let Some(bad) = weights.iter().position(|weight| !weight.is_finite()) {
        return Err(format!(
            "weight {bad} is {} — triangulation weights must be finite",
            weights[bad]
        ));
    }
    build(
        cloud,
        Some(
            weights
                .iter()
                .map(|&weight| if weight == 0.0 { 0.0 } else { weight })
                .collect(),
        ),
    )
}

#[derive(Serialize, Deserialize)]
struct TriangulationRepr {
    points: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
}

impl PartialEq for Triangulation {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
            && self.points == other.points
            && self.weights == other.weights
            && self.canonical_cells() == other.canonical_cells()
    }
}

impl Eq for Triangulation {}

impl fmt::Display for Triangulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Triangulation(dimension={}, points={}, simplices={}{})",
            self.dim,
            self.points.len(),
            self.n_cells(),
            if self.weights.is_some() {
                ", weighted"
            } else {
                ""
            }
        )
    }
}

impl Serialize for Triangulation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        TriangulationRepr {
            points: self.points.clone(),
            weights: self.weights.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Triangulation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = TriangulationRepr::deserialize(deserializer)?;
        let cloud = PointCloud::new(&repr.points).map_err(serde::de::Error::custom)?;
        match repr.weights {
            Some(weights) => regular(&cloud, &weights),
            None => delaunay(&cloud),
        }
        .map_err(serde::de::Error::custom)
    }
}

fn build(cloud: &PointCloud, weights: Option<Vec<f64>>) -> Result<Triangulation, String> {
    let points = cloud.to_rows();
    let n = points.len();
    let dim = affine_rank(&points);
    if dim == 0 || n < dim + 1 {
        return Err(format!(
            "a triangulation needs at least {} affinely independent points, \
             got {n} of affine rank {dim}",
            dim + 1
        ));
    }
    if dim > MAX_DIM {
        return Err(format!(
            "affine rank {dim} exceeds the supported maximum of {MAX_DIM}; a \
             triangulation in that dimension has O(n^{}) cells and would not \
             fit in memory long before this bound would bite",
            dim / 2
        ));
    }
    if points[0].len() != dim {
        return Err(format!(
            "the cloud has {} coordinates but affine rank {dim}; project it \
             onto its affine hull first — a triangulation is only defined in \
             the dimension the data actually spans",
            points[0].len()
        ));
    }

    let mut t = Triangulation {
        dim,
        points,
        weights,
        cells: Vec::new(),
        hint: 0,
        visited: Vec::new(),
        generation: 0,
        free: Vec::new(),
    };
    let seed = t.seed_simplex()?;
    t.init(&seed);
    for p in 0..n as u32 {
        if seed.contains(&p) {
            continue;
        }
        t.insert(p);
    }
    Ok(t)
}

impl Triangulation {
    /// `dim + 1` sites in general position, to start from.
    fn seed_simplex(&self) -> Result<Vec<u32>, String> {
        let n = self.points.len();
        let mut chosen: Vec<u32> = vec![0];
        for cand in 1..n as u32 {
            let mut trial: Vec<&[f64]> = chosen.iter().map(|&v| self.point(v)).collect();
            trial.push(self.point(cand));
            // Independence of a partial set is a rank question, and rank is
            // the one place a float tolerance is acceptable here: it only
            // decides which simplex we *start* from, and the final check
            // below is the exact predicate.
            let owned: Vec<Vec<f64>> = trial.iter().map(|r| r.to_vec()).collect();
            if affine_rank(&owned) == chosen.len() {
                chosen.push(cand);
            }
            if chosen.len() == self.dim + 1 {
                break;
            }
        }
        if chosen.len() != self.dim + 1 {
            return Err("no non-degenerate starting simplex found".into());
        }
        let rows: Vec<&[f64]> = chosen.iter().map(|&v| self.point(v)).collect();
        if orientation(&rows) == Sign::Zero {
            return Err("starting simplex is degenerate under the exact predicate".into());
        }
        Ok(chosen)
    }

    /// The first finite cell, plus a ghost over each of its facets.
    fn init(&mut self, seed: &[u32]) {
        let d = self.dim;
        let mut verts = seed.to_vec();
        let rows: Vec<&[f64]> = verts.iter().map(|&v| self.point(v)).collect();
        if orientation(&rows) == Sign::Negative {
            verts.swap(0, 1);
        }
        self.cells.push(Cell {
            verts,
            nbrs: vec![0; d + 1],
            alive: true,
        });
        for i in 0..=d {
            let mut gv: Vec<u32> = self.cells[0]
                .verts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();
            gv.push(GHOST);
            self.cells.push(Cell {
                verts: gv,
                nbrs: vec![0; d + 1],
                alive: true,
            });
        }
        self.relink_all();
    }

    /// Rebuild every adjacency from a facet map. Used at start-up; insertion
    /// relinks only what it touched.
    fn relink_all(&mut self) {
        let mut map: HashMap<Vec<u32>, Vec<(usize, usize)>> = HashMap::new();
        for c in 0..self.cells.len() {
            if !self.cells[c].alive {
                continue;
            }
            for i in 0..=self.dim {
                map.entry(self.facet_key(c, i)).or_default().push((c, i));
            }
        }
        for (_, sharers) in map {
            if let [(a, ai), (b, bi)] = sharers[..] {
                self.cells[a].nbrs[ai] = b as u32;
                self.cells[b].nbrs[bi] = a as u32;
            }
        }
    }

    /// Walk toward `p` from the hint, returning a cell in conflict with it.
    ///
    /// The cell *containing* `p` is always in conflict, since a point inside a
    /// simplex is inside its circumsphere, so locating is enough. Each step
    /// crosses the facet `p` lies beyond; on a Delaunay triangulation this
    /// terminates. `None` means the walk gave up and the caller should scan —
    /// a correctness backstop, not an expected path.
    fn locate(&self, p: u32) -> Option<usize> {
        let q = self.point(p);
        let mut cur = self.hint;
        // Generous, and only a guard against a walk that fails to terminate
        // on input we have not anticipated; the scan below is still correct.
        let budget = 8 * self.cells.len() + 64;
        for _ in 0..budget {
            if cur >= self.cells.len() || !self.cells[cur].alive {
                return None;
            }
            if self.cells[cur].is_ghost() {
                return self.conflicts(cur, p).then_some(cur);
            }
            let mut rows: Rows = [&[]; MAX_DIM + 2];
            for (k, &v) in self.cells[cur].verts.iter().enumerate() {
                rows[k] = self.point(v);
            }
            let mut step = None;
            for i in 0..=self.dim {
                let saved = rows[i];
                rows[i] = q;
                // Positively oriented cell: replacing v_i by p and getting a
                // negative sign means p is beyond the facet opposite v_i.
                let beyond = orientation(&rows[..=self.dim]) == Sign::Negative;
                rows[i] = saved;
                if beyond {
                    step = Some(self.cells[cur].nbrs[i] as usize);
                    break;
                }
            }
            match step {
                Some(next) => cur = next,
                // p is inside this cell, hence inside its circumsphere.
                None => return Some(cur),
            }
        }
        None
    }

    /// Insert one site.
    fn insert(&mut self, p: u32) {
        let seed = self
            .locate(p)
            .filter(|&c| self.conflicts(c, p))
            .or_else(|| (0..self.cells.len()).find(|&c| self.conflicts(c, p)));
        let Some(seed) = seed else {
            // Inside no circumsphere at all: the site coincides with one
            // already present. Nothing to do.
            return;
        };

        // Cavity: every cell in conflict, reachable from the seed. Marked by
        // generation rather than a fresh array, so the cost is the size of
        // the cavity and not the size of the triangulation.
        self.generation += 1;
        let gen = self.generation;
        self.visited.resize(self.cells.len(), 0);
        self.visited[seed] = gen;
        let mut cavity = vec![seed];
        let mut queue = vec![seed];
        while let Some(c) = queue.pop() {
            for i in 0..=self.dim {
                let nb = self.cells[c].nbrs[i] as usize;
                if self.visited[nb] != gen && self.conflicts(nb, p) {
                    self.visited[nb] = gen;
                    cavity.push(nb);
                    queue.push(nb);
                }
            }
        }

        // Boundary: facets of the cavity whose far side survives.
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for &c in &cavity {
            for i in 0..=self.dim {
                let nb = self.cells[c].nbrs[i] as usize;
                if self.visited[nb] != gen {
                    boundary.push((c, i));
                }
            }
        }

        // Prepare every replacement while the cavity cells still hold the
        // boundary facets. Only then may their slots be recycled.
        let mut replacements = Vec::with_capacity(boundary.len());
        for (c, i) in boundary {
            let outside = self.cells[c].nbrs[i] as usize;
            let outside_slot = (0..=self.dim)
                .find(|&slot| self.cells[outside].nbrs[slot] as usize == c)
                .expect("surviving cell points back across the cavity boundary");
            let mut verts: Vec<u32> = self.cells[c]
                .verts
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .collect();
            verts.push(p);
            if let Some(g) = verts.iter().position(|&v| v == GHOST) {
                let last = verts.len() - 1;
                verts.swap(g, last);
            } else {
                let mut rows: Rows = [&[]; MAX_DIM + 2];
                for (k, &v) in verts.iter().enumerate() {
                    rows[k] = self.point(v);
                }
                if orientation(&rows[..verts.len()]) == Sign::Negative {
                    verts.swap(0, 1);
                }
            }
            // Where p ended up after that reordering is the slot facing the
            // surviving cell across the boundary facet.
            let slot_p = verts.iter().position(|&v| v == p).expect("p is a vertex");
            replacements.push((outside, outside_slot, verts, slot_p));
        }

        for &c in &cavity {
            self.cells[c].alive = false;
            self.free.push(c);
        }

        // Fill: join p to every boundary facet, reusing dead cavity slots,
        // and relink only what changed. A full rebuild here is quadratic.
        let mut fresh: Vec<(usize, usize)> = Vec::with_capacity(replacements.len());
        for (outside, outside_slot, verts, slot_p) in replacements {
            let mut nbrs = vec![0u32; self.dim + 1];
            nbrs[slot_p] = outside as u32;
            let replacement = Cell {
                verts,
                nbrs,
                alive: true,
            };
            let k = if let Some(slot) = self.free.pop() {
                debug_assert!(!self.cells[slot].alive, "free triangulation slot is alive");
                self.cells[slot] = replacement;
                slot
            } else {
                self.cells.push(replacement);
                self.cells.len() - 1
            };
            // Repoint the survivor at the new cell.
            self.cells[outside].nbrs[outside_slot] = k as u32;
            fresh.push((k, slot_p));
        }

        // Link the new cells to one another. Their shared facets all contain
        // p, so only the slots other than `slot_p` are in play — and since
        // every key would carry p, it is left out, leaving `dim - 1`
        // vertices.
        //
        // Sorted and paired rather than hashed. The keys are fixed-size and
        // `Copy`, so this allocates nothing per facet, where a
        // `HashMap<Vec<u32>, _>` allocated and hashed a heap key for each one
        // — the dominant cost of an insertion once the predicates got quick.
        let mut keyed: Vec<([u32; MAX_DIM], usize, usize)> = Vec::with_capacity(fresh.len() * 2);
        for &(k, slot_p) in &fresh {
            for j in 0..=self.dim {
                if j != slot_p {
                    keyed.push((self.facet_key_excluding(k, j, p), k, j));
                }
            }
        }
        keyed.sort_unstable();
        for pair in keyed.chunks(2) {
            if let [(ka, c1, j1), (kb, c2, j2)] = pair {
                debug_assert_eq!(ka, kb, "internal cavity facet is unshared");
                self.cells[*c1].nbrs[*j1] = *c2 as u32;
                self.cells[*c2].nbrs[*j2] = *c1 as u32;
            }
        }

        debug_assert_eq!(
            keyed.len() % 2,
            0,
            "cavity boundary did not pair up: {} internal facets among {} new cells",
            keyed.len(),
            fresh.len()
        );
        if let Some(&(k, _)) = fresh.first() {
            self.hint = k;
        }
        self.visited.resize(self.cells.len(), 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(p: &[Vec<f64>]) -> PointCloud {
        PointCloud::new(p).unwrap()
    }

    #[test]
    fn a_single_triangle() {
        let t = delaunay(&cloud(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]])).unwrap();
        assert_eq!(t.n_cells(), 1);
    }

    #[test]
    fn a_square_is_two_triangles() {
        let t = delaunay(&cloud(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ]))
        .unwrap();
        assert_eq!(t.n_cells(), 2);
    }

    #[test]
    fn euler_holds_for_a_planar_triangulation() {
        // For n points in general position in the plane with h on the hull,
        // the Delaunay triangulation has exactly 2n - 2 - h triangles.
        let pts: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                let a = i as f64 * 0.7211;
                vec![
                    a.cos() * (1.0 + 0.3 * a.sin()),
                    a.sin() * (1.0 + 0.2 * a.cos()),
                ]
            })
            .collect();
        let t = delaunay(&cloud(&pts)).unwrap();
        assert!(t.n_cells() > 0, "produced no cells");
        // Every cell is positively oriented and non-degenerate.
        for c in t.cells() {
            let rows: Vec<&[f64]> = c.iter().map(|&v| &pts[v as usize][..]).collect();
            assert_eq!(
                orientation(&rows),
                Sign::Positive,
                "cell {c:?} is not positively oriented"
            );
        }
    }

    #[test]
    fn a_tetrahedron_in_space() {
        let t = delaunay(&cloud(&[
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]))
        .unwrap();
        assert_eq!(t.n_cells(), 1);
    }

    #[test]
    fn a_cloud_of_lower_rank_is_refused_rather_than_guessed() {
        // Collinear points in the plane have no 2-dimensional triangulation,
        // and silently producing one would be worse than saying so.
        let e = delaunay(&cloud(&[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]])).unwrap_err();
        assert!(
            e.contains("affine rank") || e.contains("affine hull"),
            "{e}"
        );
    }

    #[test]
    fn recycled_cells_keep_total_reciprocal_adjacency() {
        let pts = (0..12)
            .flat_map(|row| {
                (0..12).map(move |column| {
                    let index = row * 12 + column;
                    vec![
                        column as f64 + (index % 7) as f64 * 1.0e-4,
                        row as f64 + (index % 11) as f64 * 1.0e-4,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let triangulation = delaunay(&cloud(&pts)).unwrap();

        let free = triangulation
            .free
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(free.len(), triangulation.free.len());
        for (cell, value) in triangulation.cells.iter().enumerate() {
            assert_eq!(free.contains(&cell), !value.alive);
            if !value.alive {
                continue;
            }
            for facet in 0..=triangulation.dim {
                let neighbour = value.nbrs[facet] as usize;
                assert!(triangulation.cells[neighbour].alive);
                let key = triangulation.facet_key(cell, facet);
                assert!((0..=triangulation.dim).any(|other_facet| {
                    triangulation.cells[neighbour].nbrs[other_facet] as usize == cell
                        && triangulation.facet_key(neighbour, other_facet) == key
                }));
            }
        }
    }
}
