// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Partition-backtrack search over the stabilizer chain using Leon's
//! single-CPU algorithm.
//!
//! The BSGS substrate ([`crate::PermGroup`]) answers *membership*
//! questions in polynomial time. This module adds the questions that
//! need a **search over the base-image tree**: whether two subgroups
//! are conjugate (and a witness), whether one embeds into another up to
//! conjugacy, the symmetric-group normalizer, subgroup normalizers, and
//! set stabilizers.
//!
//! # The engine
//!
//! Every routine here is a depth-first walk of the base-image tree of
//! an *ambient* group `A` (either `S_n` or a caller-supplied `G`). A
//! group element `g ∈ A` factors uniquely as a product of one coset
//! representative per stabilizer-chain level, so the leaves of the tree
//! are exactly the elements of `A`, each reached once
//! ([`walk`]). Two ingredients cut the tree:
//!
//! * a **base-image prune** — a *necessary* pointwise condition on the
//!   final image `y = gᵇⁱ` of each base point `bᵢ` (image chosen at
//!   level `i` is final, since deeper levels fix `bᵢ`). Because the
//!   condition is necessary, pruning it never drops a solution.
//! * for subgroup searches, the running answer `K` is tracked so a
//!   found element that already lies in `K` is not re-added as a
//!   generator (the property is constant on cosets of `K`).
//!
//! A node cap bounds the walk: an exhausted budget returns
//! [`GroupError::SearchBudgetExceeded`] rather than a partial answer. Every
//! witness a search hands back is **sift-verified**
//! against its target first ([`verify_conjugate_eq`] /
//! [`verify_conjugate_leq`]); an unverified witness is
//! [`GroupError::WitnessUnverified`] and is never returned.
//!
//! # Convention
//!
//! Conjugation is right-action, matching [`Perm::conjugate`]:
//! `hᵍ = g⁻¹ · h · g`, and `Hᵍ = { hᵍ : h ∈ H }` relabels the points of
//! `H` through `g`. `conjugating_element(H, K)` returns `g` with
//! `Hᵍ = K`.

use std::collections::BTreeMap;

use crate::error::{GroupError, Result};
use crate::group::{disjoint_symmetric_group, symmetric_group_order, PermGroup};
use crate::perm::{Perm, Point};

/// Node-cap for a single backtrack search. Analogous to the coset cap
/// behind [`GroupError::IndexTooLarge`]: a search that visits more than
/// this many tree nodes returns [`GroupError::SearchBudgetExceeded`]
/// rather than run unbounded.
pub(crate) const NODE_CAP: usize = 60_000_000;

/// Orders at or below this bound let a conjugacy query pre-reject
/// non-conjugate pairs by comparing the full multiset of element cycle
/// types (an `S_n`-conjugacy invariant) without any tree search.
const CYCLE_TYPE_CAP: u64 = 8192;

/// Maximum permutation degree for the explicit cyclic normalizer lane.
const REGULAR_CYCLIC_NORMALIZER_CAP: usize = 1_000_000;

/// Maximum number of stored permutation cells used by the explicit cyclic
/// normalizer generators. Larger cycle decompositions retain the bounded
/// general backtracking path.
const CYCLIC_NORMALIZER_CELL_CAP: usize = 8_000_000;

// ── the base-image tree ─────────────────────────────────────────────

/// One level of the flattened base-image tree: the base point, its
/// orbit at this level, and the parallel coset representatives (in the
/// level's stabilizer `A^(i)`, each mapping `base` to `orbit[idx]`).
struct Lvl {
    base: Point,
    orbit: Vec<Point>,
    reps: Vec<Perm>,
}

/// The base-image tree of an ambient group, top level (`A`) first.
struct Chain {
    levels: Vec<Lvl>,
}

/// Flatten a group's public stabilizer chain into the walk structure.
fn chain_of(ambient: &PermGroup) -> Chain {
    let sc = ambient.stabilizer_chain();
    let levels = sc
        .steps
        .into_iter()
        .map(|step| Lvl {
            base: step.base,
            orbit: step.orbit,
            reps: step.transversal,
        })
        .collect();
    Chain { levels }
}

/// Depth-first walk of the base-image tree.
///
/// `g_partial` is the accumulated product of the reps chosen at levels
/// `0..level`; at a leaf it is a full element of the ambient group.
/// `prune(level, base, y)` returns `true` to skip a base image `y`;
/// `visit(g)` is called at each leaf and returns `Ok(true)` to stop the
/// whole walk early (single-witness searches). Returns `Ok(true)` when
/// a `visit` requested a stop.
fn walk(
    chain: &Chain,
    level: usize,
    g_partial: &Perm,
    prune: &mut dyn FnMut(usize, Point, Point) -> bool,
    visit: &mut dyn FnMut(&Perm) -> Result<bool>,
    nodes: &mut usize,
    cap: usize,
) -> Result<bool> {
    *nodes += 1;
    if *nodes > cap {
        return Err(GroupError::SearchBudgetExceeded { cap });
    }
    if level == chain.levels.len() {
        return visit(g_partial);
    }
    let lvl = &chain.levels[level];
    for (idx, &delta) in lvl.orbit.iter().enumerate() {
        // Final image of this level's base point under any completion of
        // `g_partial` down this branch (deeper reps fix it).
        let y = g_partial.apply(delta);
        if prune(level, lvl.base, y) {
            continue;
        }
        let g_new = lvl.reps[idx].compose(g_partial)?;
        if walk(chain, level + 1, &g_new, prune, visit, nodes, cap)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Collect the subgroup `{ g ∈ ambient : property(g) }`, given a
/// necessary base-image `prune`. Builds the answer as strong
/// generators: a satisfying element already in the running subgroup is
/// skipped, so the generator list stays short.
fn subgroup_search(
    ambient: &PermGroup,
    degree: usize,
    prune: &mut dyn FnMut(usize, Point, Point) -> bool,
    property: &mut dyn FnMut(&Perm) -> Result<bool>,
    cap: usize,
) -> Result<PermGroup> {
    let chain = chain_of(ambient);
    let mut gens: Vec<Perm> = Vec::new();
    let mut k: Option<PermGroup> = None;
    let mut nodes = 0usize;
    let identity = Perm::identity(degree);
    {
        let mut visit = |g: &Perm| -> Result<bool> {
            if property(g)? {
                let is_new = match &k {
                    None => true,
                    Some(kg) => !kg.contains(g)?,
                };
                if is_new {
                    gens.push(g.clone());
                    k = Some(PermGroup::from_generators(degree, &gens)?);
                }
            }
            Ok(false)
        };
        walk(&chain, 0, &identity, prune, &mut visit, &mut nodes, cap)?;
    }
    match k {
        Some(kg) => Ok(kg),
        None => PermGroup::from_generators(degree, &[]),
    }
}

/// Find one `g ∈ ambient` with `accept(g)`, or `None`.
fn witness_search(
    ambient: &PermGroup,
    degree: usize,
    prune: &mut dyn FnMut(usize, Point, Point) -> bool,
    accept: &mut dyn FnMut(&Perm) -> Result<bool>,
    cap: usize,
) -> Result<Option<Perm>> {
    let chain = chain_of(ambient);
    let mut nodes = 0usize;
    let identity = Perm::identity(degree);
    let mut found: Option<Perm> = None;
    {
        let mut visit = |g: &Perm| -> Result<bool> {
            if accept(g)? {
                found = Some(g.clone());
                Ok(true)
            } else {
                Ok(false)
            }
        };
        walk(&chain, 0, &identity, prune, &mut visit, &mut nodes, cap)?;
    }
    Ok(found)
}

// ── small helpers ───────────────────────────────────────────────────

/// The symmetric group `S_n` as a [`PermGroup`] on `0..n`.
fn symmetric_group(n: usize) -> Result<PermGroup> {
    if n <= 1 {
        return PermGroup::from_generators(n, &[]);
    }
    let mut transp: Vec<Point> = (0..n as Point).collect();
    transp.swap(0, 1);
    let cyc: Vec<Point> = (0..n).map(|i| ((i + 1) % n) as Point).collect();
    let g1 = Perm::from_images(&transp)?;
    let g2 = Perm::from_images(&cyc)?;
    PermGroup::from_generators(n, &[g1, g2])
}

/// The non-identity generators of `g`, cloned.
fn active_generators(g: &PermGroup) -> Vec<Perm> {
    g.generators()
        .iter()
        .filter(|p| !p.is_identity())
        .cloned()
        .collect()
}

/// Sorted cycle-length multiset of a permutation.
fn cycle_type(p: &Perm) -> Vec<usize> {
    let n = p.degree();
    let mut visited = vec![false; n];
    let mut lens = Vec::new();
    for s in 0..n {
        if visited[s] {
            continue;
        }
        let mut j = s;
        let mut l = 0usize;
        while !visited[j] {
            visited[j] = true;
            j = p.apply(j as Point) as usize;
            l += 1;
        }
        lens.push(l);
    }
    lens.sort_unstable();
    lens
}

/// The sorted multiset of element cycle types of `g`, or `None` when
/// `|g|` is too large to enumerate cheaply. Two `S_n`-conjugate groups
/// share this multiset, so a mismatch is a definite non-conjugacy.
fn cycle_type_multiset(g: &PermGroup) -> Option<Vec<Vec<usize>>> {
    if g.order() > crate::BigUInt::from(CYCLE_TYPE_CAP) {
        return None;
    }
    let mut types: Vec<Vec<usize>> = g.iter_elements().map(|e| cycle_type(&e)).collect();
    types.sort();
    Some(types)
}

/// Per-point orbit length under `h`, indexed by point.
fn orbit_lengths(h: &PermGroup, degree: usize) -> Result<Vec<usize>> {
    let mut lens = vec![0usize; degree];
    for p in 0..degree as Point {
        if lens[p as usize] == 0 {
            let ob = h.orbit(p)?;
            let l = ob.len();
            for &q in &ob {
                lens[q as usize] = l;
            }
        }
    }
    Ok(lens)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn cyclic_generator(group: &PermGroup) -> Option<Perm> {
    let degree = group.degree();
    if degree == 0 || degree > REGULAR_CYCLIC_NORMALIZER_CAP {
        return None;
    }
    if let Some(generator) = group
        .generators()
        .iter()
        .find(|generator| generator.order() == group.order())
    {
        return Some(generator.clone());
    }
    if degree > CYCLE_TYPE_CAP as usize {
        return None;
    }
    group
        .iter_elements()
        .find(|element| element.order() == group.order())
}

fn cycles_of(permutation: &Perm) -> Vec<Vec<Point>> {
    let degree = permutation.degree();
    let mut visited = vec![false; degree];
    let mut cycles = Vec::new();
    for start in 0..degree {
        if visited[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut point = start as Point;
        while !visited[point as usize] {
            visited[point as usize] = true;
            cycle.push(point);
            point = permutation.apply(point);
        }
        cycles.push(cycle);
    }
    cycles
}

fn checked_lcm(left: usize, right: usize) -> Option<usize> {
    left.checked_div(gcd(left, right))?.checked_mul(right)
}

fn unit_generators(modulus: usize) -> Option<(Vec<usize>, u64)> {
    if modulus == 1 {
        return Some((Vec::new(), 1));
    }
    let mut generators = Vec::new();
    let mut unit_count = 0_u64;
    let mut units = Vec::new();
    for candidate in 1..modulus {
        if gcd(candidate, modulus) == 1 {
            unit_count = unit_count.checked_add(1)?;
            units.push(candidate);
        }
    }
    let mut generated = vec![false; modulus];
    generated[1] = true;
    for candidate in units {
        if generated[candidate] {
            continue;
        }
        generators.push(candidate);
        generated.fill(false);
        generated[1] = true;
        let mut frontier = vec![1_usize];
        while let Some(value) = frontier.pop() {
            for &generator in &generators {
                let product = ((value as u128 * generator as u128) % modulus as u128) as usize;
                if !generated[product] {
                    generated[product] = true;
                    frontier.push(product);
                }
            }
        }
    }
    Some((generators, unit_count))
}

/// Construct `N_{S_n}(<cycle>)` from the exact cycle decomposition.
///
/// The centralizer is the product of the wreath factors
/// `C_l wr S_{a_l}` for the `a_l` cycles of length `l`. The quotient of the
/// normalizer by that centralizer is `(Z/mZ)^*`, acting simultaneously on
/// every cycle, where `m` is the permutation order. The caller has proved
/// that `h` is cyclic and generated by `cycle`.
fn cyclic_normalizer(h: &PermGroup, cycle: &Perm) -> Result<Option<PermGroup>> {
    let degree = h.degree();
    let cycles = cycles_of(cycle);
    let mut by_length = BTreeMap::<usize, Vec<&[Point]>>::new();
    let mut order = 1_usize;
    for orbit in &cycles {
        order = match checked_lcm(order, orbit.len()) {
            Some(order) if order <= REGULAR_CYCLIC_NORMALIZER_CAP => order,
            _ => return Ok(None),
        };
        by_length.entry(orbit.len()).or_default().push(orbit);
    }
    let Some((unit_generators, unit_count)) = unit_generators(order) else {
        return Ok(None);
    };
    let centralizer_generator_count = by_length
        .values()
        .map(|orbits| {
            orbits.iter().filter(|orbit| orbit.len() > 1).count() + orbits.len().saturating_sub(1)
        })
        .sum::<usize>();
    let generator_count = centralizer_generator_count.saturating_add(unit_generators.len());
    if generator_count.saturating_mul(degree) > CYCLIC_NORMALIZER_CELL_CAP {
        return Ok(None);
    }

    let mut generators = Vec::with_capacity(generator_count);
    let mut centralizer_order = crate::BigUInt::one();
    for (&length, orbits) in &by_length {
        for orbit in orbits {
            if length > 1 {
                let mut images = (0..degree as Point).collect::<Vec<_>>();
                for index in 0..length {
                    images[orbit[index] as usize] = orbit[(index + 1) % length];
                }
                generators.push(Perm::from_images(&images)?);
                centralizer_order.mul_assign_u64(
                    u64::try_from(length).map_err(|_| GroupError::WitnessUnverified)?,
                );
            }
        }
        if let Some(first) = orbits.first() {
            for other in orbits.iter().skip(1) {
                let mut images = (0..degree as Point).collect::<Vec<_>>();
                for index in 0..length {
                    images[first[index] as usize] = other[index];
                    images[other[index] as usize] = first[index];
                }
                generators.push(Perm::from_images(&images)?);
            }
        }
        for factor in 2..=orbits.len() {
            centralizer_order
                .mul_assign_u64(u64::try_from(factor).map_err(|_| GroupError::WitnessUnverified)?);
        }
    }
    for unit in unit_generators {
        let mut images: Vec<Point> = (0..degree as Point).collect();
        for orbit in &cycles {
            for (coordinate, &source) in orbit.iter().enumerate() {
                images[source as usize] = orbit[(coordinate * unit) % orbit.len()];
            }
        }
        let automorphism = Perm::from_images(&images)?;
        if !automorphism.is_identity() {
            generators.push(automorphism);
        }
    }
    let normalizer = PermGroup::from_generators(degree, &generators)?;
    let mut expected = centralizer_order;
    expected.mul_assign_u64(unit_count);
    if normalizer.order() != expected {
        return Err(GroupError::WitnessUnverified);
    }
    for generator in normalizer.generators() {
        verify_conjugate_eq(h, generator, h)?;
    }
    Ok(Some(normalizer))
}

fn symmetric_set_stabilizer(
    degree: usize,
    delta: &[Point],
    in_delta: &[bool],
) -> Result<PermGroup> {
    let complement = (0..degree as Point)
        .filter(|point| !in_delta[*point as usize])
        .collect::<Vec<_>>();
    let stabilizer = disjoint_symmetric_group(degree, &[delta.to_vec(), complement.clone()])?;
    let left_order = symmetric_group_order(delta.len()).ok_or(GroupError::WitnessUnverified)?;
    let right_order =
        symmetric_group_order(complement.len()).ok_or(GroupError::WitnessUnverified)?;
    if stabilizer.order() != left_order.mul(&right_order)
        || stabilizer.generators().iter().any(|generator| {
            delta
                .iter()
                .any(|point| !in_delta[generator.apply(*point) as usize])
        })
    {
        return Err(GroupError::WitnessUnverified);
    }
    Ok(stabilizer)
}

// ── witness verification ────────────────────────────────────────────

/// Verify a candidate conjugator: `Hᵍ = K`. Every generator of `h`,
/// conjugated by `g`, must lie in `k`, and `|h| = |k|`. On any failure
/// returns [`GroupError::WitnessUnverified`] — the search never hands
/// out an unverified conjugator.
pub(crate) fn verify_conjugate_eq(h: &PermGroup, g: &Perm, k: &PermGroup) -> Result<()> {
    if h.order() != k.order() {
        return Err(GroupError::WitnessUnverified);
    }
    for hg in active_generators(h) {
        let conj = hg.conjugate(g)?;
        if !k.contains(&conj)? {
            return Err(GroupError::WitnessUnverified);
        }
    }
    Ok(())
}

/// Verify a containment witness: `Hᵍ ≤ over`. Every generator of `h`,
/// conjugated by `g`, must lie in `over`; otherwise
/// [`GroupError::WitnessUnverified`].
pub(crate) fn verify_conjugate_leq(h: &PermGroup, g: &Perm, over: &PermGroup) -> Result<()> {
    for hg in active_generators(h) {
        let conj = hg.conjugate(g)?;
        if !over.contains(&conj)? {
            return Err(GroupError::WitnessUnverified);
        }
    }
    Ok(())
}

// ── public surface ──────────────────────────────────────────────────

impl PermGroup {
    /// A conjugating element `g` with `selfᵍ = other` (i.e.
    /// `g⁻¹ · self · g = other` as groups), or `None` if `self` and
    /// `other` are not conjugate in `S_n`.
    ///
    /// The returned witness is sift-verified: `selfᵍ = other` holds
    /// exactly.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] if `self` and `other` act on
    /// different degrees; [`GroupError::SearchBudgetExceeded`] if the
    /// backtrack exceeds its node cap.
    ///
    /// ```
    /// use perm_core::{PermGroup, Perm};
    /// // ⟨(0 1)⟩ and ⟨(2 3)⟩ are conjugate transpositions in S_4.
    /// let h = PermGroup::from_generators(4, &[Perm::from_images(&[1, 0, 2, 3]).unwrap()]).unwrap();
    /// let k = PermGroup::from_generators(4, &[Perm::from_images(&[0, 1, 3, 2]).unwrap()]).unwrap();
    /// assert!(h.conjugating_element(&k).unwrap().is_some());
    /// ```
    pub fn conjugating_element(&self, other: &PermGroup) -> Result<Option<Perm>> {
        self.conjugating_element_capped(other, NODE_CAP)
    }

    pub(crate) fn conjugating_element_capped(
        &self,
        other: &PermGroup,
        cap: usize,
    ) -> Result<Option<Perm>> {
        let n = self.degree();
        if other.degree() != n {
            return Err(GroupError::DegreeMismatch {
                got: other.degree(),
                expected: n,
            });
        }
        // Cheap necessary conditions before any search.
        if self.order() != other.order() {
            return Ok(None);
        }
        if let (Some(a), Some(b)) = (cycle_type_multiset(self), cycle_type_multiset(other)) {
            if a != b {
                return Ok(None);
            }
        }
        let ambient = symmetric_group(n)?;
        let hgens = active_generators(self);
        let mut accept = |g: &Perm| -> Result<bool> {
            for h in &hgens {
                if !other.contains(&h.conjugate(g)?)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        let mut prune = |_l: usize, _b: Point, _y: Point| false;
        let witness = witness_search(&ambient, n, &mut prune, &mut accept, cap)?;
        if let Some(g) = &witness {
            verify_conjugate_eq(self, g, other)?;
        }
        Ok(witness)
    }

    /// A witness `g` with `selfᵍ ≤ over` (some `S_n`-conjugate of
    /// `self` is a subgroup of `over`), or `None`.
    ///
    /// The returned witness is verified: every generator of `self`,
    /// conjugated by `g`, lies in `over`.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] on a degree mismatch;
    /// [`GroupError::SearchBudgetExceeded`] on an exhausted budget.
    ///
    /// ```
    /// use perm_core::{PermGroup, Perm};
    /// // ⟨(0 1)⟩ embeds (up to conjugacy) into S_3.
    /// let h = PermGroup::from_generators(3, &[Perm::from_images(&[1, 0, 2]).unwrap()]).unwrap();
    /// let s3 = PermGroup::from_generators(
    ///     3,
    ///     &[Perm::from_images(&[1, 0, 2]).unwrap(), Perm::from_images(&[1, 2, 0]).unwrap()],
    /// ).unwrap();
    /// assert!(h.embeds_up_to_conjugacy(&s3).unwrap().is_some());
    /// ```
    pub fn embeds_up_to_conjugacy(&self, over: &PermGroup) -> Result<Option<Perm>> {
        self.embeds_up_to_conjugacy_capped(over, NODE_CAP)
    }

    pub(crate) fn embeds_up_to_conjugacy_capped(
        &self,
        over: &PermGroup,
        cap: usize,
    ) -> Result<Option<Perm>> {
        let n = self.degree();
        if over.degree() != n {
            return Err(GroupError::DegreeMismatch {
                got: over.degree(),
                expected: n,
            });
        }
        if self.order() > over.order() {
            return Ok(None);
        }
        let ambient = symmetric_group(n)?;
        let hgens = active_generators(self);
        let mut accept = |g: &Perm| -> Result<bool> {
            for h in &hgens {
                if !over.contains(&h.conjugate(g)?)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        let mut prune = |_l: usize, _b: Point, _y: Point| false;
        let witness = witness_search(&ambient, n, &mut prune, &mut accept, cap)?;
        if let Some(g) = &witness {
            verify_conjugate_leq(self, g, over)?;
        }
        Ok(witness)
    }

    /// The normalizer of `self` in the full symmetric group,
    /// `N_{S_n}(self)`.
    ///
    /// # Errors
    /// [`GroupError::SearchBudgetExceeded`] on an exhausted budget.
    ///
    /// ```
    /// use perm_core::{PermGroup, Perm};
    /// // N_{S_3}(⟨(0 1 2)⟩) = S_3, order 6.
    /// let c3 = PermGroup::from_generators(3, &[Perm::from_images(&[1, 2, 0]).unwrap()]).unwrap();
    /// assert_eq!(c3.normalizer_in_symmetric().unwrap().order(), perm_core::BigUInt::from(6u64));
    /// ```
    pub fn normalizer_in_symmetric(&self) -> Result<PermGroup> {
        let n = self.degree();
        let ambient = symmetric_group(n)?;
        normalizer_impl(&ambient, self, NODE_CAP)
    }

    /// The normalizer of `h` inside `self`, `N_self(h) = { g ∈ self :
    /// hᵍ = h }`.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] if `self` and `h` act on different
    /// degrees; [`GroupError::SearchBudgetExceeded`] on an exhausted
    /// budget.
    pub fn normalizer(&self, h: &PermGroup) -> Result<PermGroup> {
        if h.degree() != self.degree() {
            return Err(GroupError::DegreeMismatch {
                got: h.degree(),
                expected: self.degree(),
            });
        }
        normalizer_impl(self, h, NODE_CAP)
    }

    /// The set stabilizer `Stab_self(set)` under the `OnSets` action —
    /// the subgroup of `self` fixing `set` setwise. Duplicate points in
    /// `set` are canonicalized (sort + dedup), not an error.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] if any point in `set` is `≥
    /// degree`; [`GroupError::SearchBudgetExceeded`] on an exhausted
    /// budget.
    ///
    /// ```
    /// use perm_core::{PermGroup, Perm};
    /// // Stab_{S_3}({0,1}) = ⟨(0 1)⟩, order 2.
    /// let s3 = PermGroup::from_generators(
    ///     3,
    ///     &[Perm::from_images(&[1, 0, 2]).unwrap(), Perm::from_images(&[1, 2, 0]).unwrap()],
    /// ).unwrap();
    /// assert_eq!(s3.set_stabilizer(&[0, 1]).unwrap().order(), perm_core::BigUInt::from(2u64));
    /// ```
    pub fn set_stabilizer(&self, set: &[Point]) -> Result<PermGroup> {
        self.set_stabilizer_capped(set, NODE_CAP)
    }

    pub(crate) fn set_stabilizer_capped(&self, set: &[Point], cap: usize) -> Result<PermGroup> {
        let degree = self.degree();
        let mut in_delta = vec![false; degree];
        for &p in set {
            if (p as usize) >= degree {
                return Err(GroupError::DegreeMismatch {
                    got: p as usize,
                    expected: degree,
                });
            }
            in_delta[p as usize] = true;
        }
        let delta: Vec<Point> = (0..degree as Point)
            .filter(|&p| in_delta[p as usize])
            .collect();
        if self.is_full_symmetric() {
            return symmetric_set_stabilizer(degree, &delta, &in_delta);
        }
        let mut property = |g: &Perm| -> Result<bool> {
            for &d in &delta {
                if !in_delta[g.apply(d) as usize] {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        let mut prune =
            |_l: usize, base: Point, y: Point| in_delta[base as usize] != in_delta[y as usize];
        subgroup_search(self, degree, &mut prune, &mut property, cap)
    }
}

/// Compute `N_ambient(h)` by a subgroup backtrack over `ambient`, using
/// the `h`-orbit-length necessary condition to prune base images.
fn normalizer_impl(ambient: &PermGroup, h: &PermGroup, cap: usize) -> Result<PermGroup> {
    let degree = ambient.degree();
    if h.order().is_one() || h.order() == ambient.order() {
        return Ok(ambient.clone());
    }
    if ambient.is_full_symmetric() {
        if let Some(cycle) = cyclic_generator(h) {
            if let Some(normalizer) = cyclic_normalizer(h, &cycle)? {
                return Ok(normalizer);
            }
        }
    }
    let h_orbit_len = orbit_lengths(h, degree)?;
    let hgens = active_generators(h);
    let mut property = |g: &Perm| -> Result<bool> {
        for hg in &hgens {
            if !h.contains(&hg.conjugate(g)?)? {
                return Ok(false);
            }
        }
        Ok(true)
    };
    let mut prune =
        |_l: usize, base: Point, y: Point| h_orbit_len[base as usize] != h_orbit_len[y as usize];
    subgroup_search(ambient, degree, &mut prune, &mut property, cap)
}

// ── free-function forms ─────────────────────────────────────────────

/// Free-function form of [`PermGroup::conjugating_element`].
pub fn subgroup_conjugacy(h: &PermGroup, k: &PermGroup) -> Result<Option<Perm>> {
    h.conjugating_element(k)
}

/// Free-function form of [`PermGroup::embeds_up_to_conjugacy`].
pub fn containment_up_to_conjugacy(h: &PermGroup, over: &PermGroup) -> Result<Option<Perm>> {
    h.embeds_up_to_conjugacy(over)
}

/// Free-function form of [`PermGroup::normalizer_in_symmetric`].
pub fn normalizer_in_symmetric_group(h: &PermGroup) -> Result<PermGroup> {
    h.normalizer_in_symmetric()
}

/// Free-function form of [`PermGroup::normalizer`] — `N_g(h)`.
pub fn normalizer(g: &PermGroup, h: &PermGroup) -> Result<PermGroup> {
    g.normalizer(h)
}

/// Free-function form of [`PermGroup::set_stabilizer`] — `Stab_g(set)`.
pub fn set_stabilizer(g: &PermGroup, set: &[Point]) -> Result<PermGroup> {
    g.set_stabilizer(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BigUInt;

    fn perm(images: &[Point]) -> Perm {
        Perm::from_images(images).unwrap()
    }

    fn group(degree: usize, gens: &[&[Point]]) -> PermGroup {
        let ps: Vec<Perm> = gens.iter().map(|g| perm(g)).collect();
        PermGroup::from_generators(degree, &ps).unwrap()
    }

    fn symmetric(n: usize) -> PermGroup {
        symmetric_group(n).unwrap()
    }

    fn big(n: u64) -> BigUInt {
        BigUInt::from(n)
    }

    // ── conjugacy ───────────────────────────────────────────────────

    #[test]
    fn conjugate_transpositions_have_witness() {
        let h = group(4, &[&[1, 0, 2, 3]]); // (0 1)
        let k = group(4, &[&[0, 1, 3, 2]]); // (2 3)
        let g = h.conjugating_element(&k).unwrap().expect("conjugate");
        // Verify the witness actually conjugates H onto K.
        verify_conjugate_eq(&h, &g, &k).unwrap();
    }

    #[test]
    fn non_conjugate_same_order_twins_return_none() {
        // In S_4: C_4 = ⟨(0 1 2 3)⟩ (order 4, transitive) vs the
        // intransitive V_4 = ⟨(0 1), (2 3)⟩ (order 4). Same order,
        // different cycle-type multiset → not conjugate.
        let c4 = group(4, &[&[1, 2, 3, 0]]);
        let v4 = group(4, &[&[1, 0, 2, 3], &[0, 1, 3, 2]]);
        assert_eq!(c4.order(), big(4));
        assert_eq!(v4.order(), big(4));
        assert!(c4.conjugating_element(&v4).unwrap().is_none());
    }

    #[test]
    fn conjugacy_degree_mismatch_errs() {
        let h = group(4, &[&[1, 0, 2, 3]]);
        let k = group(5, &[&[1, 0, 2, 3, 4]]);
        assert!(matches!(
            h.conjugating_element(&k),
            Err(GroupError::DegreeMismatch {
                got: 5,
                expected: 4
            })
        ));
    }

    #[test]
    fn trivial_is_conjugate_only_to_trivial() {
        let triv_a = PermGroup::from_generators(4, &[]).unwrap();
        let triv_b = PermGroup::from_generators(4, &[]).unwrap();
        assert!(triv_a.conjugating_element(&triv_b).unwrap().is_some());
        let nontriv = group(4, &[&[1, 0, 2, 3]]);
        assert!(triv_a.conjugating_element(&nontriv).unwrap().is_none());
    }

    // ── embedding up to conjugacy ───────────────────────────────────

    #[test]
    fn embeds_transposition_into_s3() {
        let h = group(3, &[&[1, 0, 2]]);
        let s3 = symmetric(3);
        let g = h.embeds_up_to_conjugacy(&s3).unwrap().expect("embeds");
        verify_conjugate_leq(&h, &g, &s3).unwrap();
    }

    #[test]
    fn does_not_embed_when_order_too_large() {
        // S_3 (order 6) cannot embed into C_4 (order 4).
        let s3 = symmetric(3);
        let s3_in_4: Vec<Perm> = s3
            .generators()
            .iter()
            .map(|p| {
                let mut im = p.images().to_vec();
                im.push(3);
                Perm::from_images(&im).unwrap()
            })
            .collect();
        let s3d4 = PermGroup::from_generators(4, &s3_in_4).unwrap();
        let c4 = group(4, &[&[1, 2, 3, 0]]);
        assert!(s3d4.embeds_up_to_conjugacy(&c4).unwrap().is_none());
    }

    // ── normalizer in symmetric ─────────────────────────────────────

    #[test]
    fn normalizer_of_cyclic_in_symmetric() {
        // N_{S_6}(⟨(0 1 2 3 4 5)⟩) has order 12 (the metacyclic
        // holomorph C_6 ⋊ C_2 — wait, |Aut(C_6)| = 2, so order 12).
        let c6 = group(6, &[&[1, 2, 3, 4, 5, 0]]);
        let n = c6.normalizer_in_symmetric().unwrap();
        assert_eq!(n.order(), big(12));
        // Every generator normalizes C_6.
        for gen in n.generators() {
            for h in c6.generators() {
                assert!(c6.contains(&h.conjugate(gen).unwrap()).unwrap());
            }
        }
    }

    #[test]
    fn normalizer_of_regular_cyclic_group_is_its_holomorph() {
        let c8 = group(8, &[&[1, 2, 3, 4, 5, 6, 7, 0]]);
        let normalizer = c8.normalizer_in_symmetric().unwrap();
        assert_eq!(normalizer.order(), big(32));
        for generator in normalizer.generators() {
            verify_conjugate_eq(&c8, generator, &c8).unwrap();
        }

        // C6 can be supplied by generators of orders two and three; the
        // regular generator then has to be recovered from the group.
        let square = perm(&[2, 3, 4, 5, 0, 1]);
        let cube = perm(&[3, 4, 5, 0, 1, 2]);
        let split_generators = PermGroup::from_generators(6, &[square, cube]).unwrap();
        assert_eq!(split_generators.order(), big(6));
        assert_eq!(
            split_generators.normalizer_in_symmetric().unwrap().order(),
            big(12)
        );
    }

    #[test]
    fn nonregular_cyclic_normalizer_uses_cycle_centralizers_and_unit_actions() {
        // A 6-cycle with two fixed points has normalizer
        // (C6 semidirect Aut(C6)) x S2 of order 24 in S8.
        let c6 = group(8, &[&[1, 2, 3, 4, 5, 0, 6, 7]]);
        let normalizer = c6.normalizer_in_symmetric().unwrap();
        assert_eq!(normalizer.order(), big(24));
        for generator in normalizer.generators() {
            verify_conjugate_eq(&c6, generator, &c6).unwrap();
        }
    }

    #[test]
    fn cyclic_normalizer_handles_repeated_nontrivial_cycles() {
        // For sigma=(0 1 2)(3 4 5), the centralizer is C3 wr S2 of
        // order 18 and Aut(<sigma>) contributes one further involution.
        let diagonal_c3 = group(6, &[&[1, 2, 0, 4, 5, 3]]);
        let normalizer = diagonal_c3.normalizer_in_symmetric().unwrap();
        assert_eq!(normalizer.order(), big(36));
        for generator in normalizer.generators() {
            verify_conjugate_eq(&diagonal_c3, generator, &diagonal_c3).unwrap();
        }
    }

    #[test]
    fn cyclic_normalizer_inside_a_proper_ambient_keeps_the_general_search() {
        let c3 = group(4, &[&[1, 2, 0, 3]]);
        let a4 = group(4, &[&[1, 2, 0, 3], &[1, 3, 2, 0]]);
        assert_eq!(a4.order(), big(12));
        let normalizer = a4.normalizer(&c3).unwrap();
        assert_eq!(normalizer.order(), big(3));
        for generator in normalizer.generators() {
            verify_conjugate_eq(&c3, generator, &c3).unwrap();
        }
    }

    #[test]
    fn normalizer_of_alternating_is_symmetric() {
        // A_5 = ⟨(0 1 2), (0 1 2 3 4)⟩; N_{S_5}(A_5) = S_5, order 120.
        let a5 = group(5, &[&[1, 2, 0, 3, 4], &[1, 2, 3, 4, 0]]);
        assert_eq!(a5.order(), big(60));
        let n = a5.normalizer_in_symmetric().unwrap();
        assert_eq!(n.order(), big(120));
    }

    #[test]
    fn normalizer_of_trivial_is_full_symmetric() {
        let triv = PermGroup::from_generators(4, &[]).unwrap();
        let n = triv.normalizer_in_symmetric().unwrap();
        assert_eq!(n.order(), big(24));
    }

    // ── subgroup normalizer ─────────────────────────────────────────

    #[test]
    fn normalizer_within_group() {
        // N_{S_4}(V_4) = S_4 (V_4 normal in S_4), order 24.
        let s4 = symmetric(4);
        let v4 = group(4, &[&[1, 0, 3, 2], &[2, 3, 0, 1]]);
        assert_eq!(v4.order(), big(4));
        let n = s4.normalizer(&v4).unwrap();
        assert_eq!(n.order(), big(24));
    }

    #[test]
    fn normalizer_degree_mismatch_errs() {
        let s4 = symmetric(4);
        let h = group(5, &[&[1, 0, 2, 3, 4]]);
        assert!(matches!(
            s4.normalizer(&h),
            Err(GroupError::DegreeMismatch {
                got: 5,
                expected: 4
            })
        ));
    }

    // ── set stabilizer ──────────────────────────────────────────────

    #[test]
    fn set_stabilizer_onsets_order() {
        // Stab_{S_5}({0,1}) has order 2! * 3! = 12.
        let s5 = symmetric(5);
        let stab = s5.set_stabilizer(&[0, 1]).unwrap();
        assert_eq!(stab.order(), big(12));
        // |Stab| * |orbit of {0,1}| = |G|: orbit size = C(5,2) = 10.
        // 12 * 10 = 120 = |S_5|.
        assert_eq!(stab.order().mul(&big(10)), big(120));
    }

    #[test]
    fn set_stabilizer_block_order() {
        // Stab_{S_8}({0,1,2,3}) = S_4 × S_4, order 24*24 = 576.
        let s8 = symmetric(8);
        let stab = s8.set_stabilizer(&[0, 1, 2, 3]).unwrap();
        assert_eq!(stab.order(), big(576));
    }

    #[test]
    fn symmetric_set_stabilizer_accepts_a_nonprefix_subset() {
        let s6 = symmetric(6);
        let subset = [0, 2, 5];
        let stabilizer = s6.set_stabilizer(&subset).unwrap();
        assert_eq!(stabilizer.order(), big(36));
        for generator in stabilizer.generators() {
            assert!(subset.contains(&(generator.apply(0))));
            assert!(subset.contains(&(generator.apply(2))));
            assert!(subset.contains(&(generator.apply(5))));
        }
    }

    #[test]
    fn nonsymmetric_set_stabilizer_keeps_the_general_path() {
        let a5 = group(5, &[&[1, 2, 0, 3, 4], &[1, 2, 3, 4, 0]]);
        assert_eq!(a5.set_stabilizer(&[0, 1]).unwrap().order(), big(6));
    }

    #[test]
    fn set_stabilizer_duplicate_points_canonicalized() {
        let s5 = symmetric(5);
        let a = s5.set_stabilizer(&[0, 1]).unwrap();
        let b = s5.set_stabilizer(&[1, 0, 1, 0]).unwrap();
        assert_eq!(a.order(), b.order());
    }

    #[test]
    fn set_stabilizer_out_of_range_point_errs() {
        let s4 = symmetric(4);
        assert!(matches!(
            s4.set_stabilizer(&[0, 9]),
            Err(GroupError::DegreeMismatch {
                got: 9,
                expected: 4
            })
        ));
    }

    // ── budget + witness verification ───────────────────────────────

    #[test]
    fn search_budget_exceeded_is_typed() {
        // A tiny cap on a non-trivial search trips the budget.
        let a5 = group(5, &[&[1, 2, 0, 3, 4], &[1, 2, 3, 4, 0]]);
        assert!(matches!(
            a5.set_stabilizer_capped(&[0, 1], 3),
            Err(GroupError::SearchBudgetExceeded { cap: 3 })
        ));
    }

    #[test]
    fn fault_injected_witness_is_rejected() {
        // Feed a deliberately-wrong conjugator to the verify helper:
        // the identity does NOT conjugate ⟨(0 1)⟩ onto ⟨(2 3)⟩.
        let h = group(4, &[&[1, 0, 2, 3]]);
        let k = group(4, &[&[0, 1, 3, 2]]);
        let wrong = Perm::identity(4);
        assert_eq!(
            verify_conjugate_eq(&h, &wrong, &k),
            Err(GroupError::WitnessUnverified)
        );
    }

    // ── free-function forms ─────────────────────────────────────────

    #[test]
    fn free_functions_match_methods() {
        let h = group(4, &[&[1, 0, 2, 3]]);
        let k = group(4, &[&[0, 1, 3, 2]]);
        assert!(subgroup_conjugacy(&h, &k).unwrap().is_some());
        let s4 = symmetric(4);
        assert!(containment_up_to_conjugacy(&h, &s4).unwrap().is_some());
        assert_eq!(
            normalizer_in_symmetric_group(&h).unwrap().order(),
            h.normalizer_in_symmetric().unwrap().order()
        );
        assert_eq!(
            normalizer(&s4, &h).unwrap().order(),
            s4.normalizer(&h).unwrap().order()
        );
        assert_eq!(
            set_stabilizer(&s4, &[0, 1]).unwrap().order(),
            s4.set_stabilizer(&[0, 1]).unwrap().order()
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::random::{seeded_rng, shuffle};
    use proptest::prelude::*;

    fn property_config(default_cases: u32) -> ProptestConfig {
        let configured = ProptestConfig::default();
        if std::env::var_os("PROPTEST_CASES").is_some() {
            configured
        } else {
            ProptestConfig {
                cases: default_cases,
                ..configured
            }
        }
    }

    /// A uniform random permutation of `0..n` from a seed.
    fn random_perm(n: usize, seed: u64) -> Perm {
        let mut rng = seeded_rng(seed);
        let mut images: Vec<Point> = (0..n as Point).collect();
        shuffle(&mut images, &mut rng);
        Perm::from_images(&images).unwrap()
    }

    proptest! {
        #![proptest_config(property_config(64))]

        /// Conjugating `H` by a random `g` yields `K = Hᵍ`, and
        /// `conjugating_element(H, K)` must recover a valid conjugator
        /// (not necessarily `g` itself, but one verified to work).
        #[test]
        fn conjugating_element_recovers_a_valid_witness(
            seed in any::<u64>(),
            variant in 0usize..3,
        ) {
            let n = 5;
            // A small family of source subgroups of S_5.
            let h = match variant {
                0 => PermGroup::from_generators(n, &[Perm::from_images(&[1, 0, 2, 3, 4]).unwrap()]).unwrap(),
                1 => PermGroup::from_generators(n, &[Perm::from_images(&[1, 2, 0, 3, 4]).unwrap()]).unwrap(),
                _ => PermGroup::from_generators(n, &[Perm::from_images(&[1, 2, 3, 4, 0]).unwrap()]).unwrap(),
            };
            let g = random_perm(n, seed);
            let kgens: Vec<Perm> = h
                .generators()
                .iter()
                .map(|p| p.conjugate(&g).unwrap())
                .collect();
            let k = PermGroup::from_generators(n, &kgens).unwrap();

            let witness = h.conjugating_element(&k).unwrap();
            prop_assert!(witness.is_some(), "H and H^g must be conjugate");
            let w = witness.unwrap();
            // The recovered witness verifies: H^w = K.
            prop_assert!(verify_conjugate_eq(&h, &w, &k).is_ok());
        }

        /// `N_{S_n}(H)` always contains `H` and every generator
        /// normalizes `H`.
        #[test]
        fn normalizer_contains_and_normalizes(variant in 0usize..3) {
            let n = 5;
            let h = match variant {
                0 => PermGroup::from_generators(n, &[Perm::from_images(&[1, 2, 3, 4, 0]).unwrap()]).unwrap(),
                1 => PermGroup::from_generators(n, &[Perm::from_images(&[1, 0, 2, 3, 4]).unwrap(), Perm::from_images(&[0, 1, 3, 4, 2]).unwrap()]).unwrap(),
                _ => PermGroup::from_generators(n, &[Perm::from_images(&[1, 2, 0, 3, 4]).unwrap()]).unwrap(),
            };
            let nrm = h.normalizer_in_symmetric().unwrap();
            // Every generator of H lies in its normalizer.
            for hg in h.generators() {
                prop_assert!(nrm.contains(hg).unwrap());
            }
            // Every generator of N normalizes H.
            for gen in nrm.generators() {
                for hg in h.generators() {
                    prop_assert!(h.contains(&hg.conjugate(gen).unwrap()).unwrap());
                }
            }
        }
    }
}
