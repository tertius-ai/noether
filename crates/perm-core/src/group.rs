// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! [`PermGroup`] — a permutation group carrying its base, strong generating
//! set, stabilizer chain, and derived operations.
//!
//! # Algorithm
//!
//! The default constructor [`PermGroup::from_generators`] runs the
//! **deterministic Schreier–Sims** procedure. Given a candidate base
//! and strong generating set, we sift every Schreier generator of
//! every level into the chain; whenever a sift residue is non-trivial
//! it becomes a new strong generator (and possibly extends the base
//! with a fresh moved point). The chain is correct when every
//! Schreier generator sifts to the identity.
//!
//! The randomized variant [`PermGroup::from_generators_randomized`]
//! follows the same shape but builds the chain by sifting *random*
//! products from a Sims-style product replacement on the input
//! generators, stopping once a configurable number of sift-to-
//! identity events in a row gives the Monte-Carlo BSGS. It then
//! **verifies** the result by the deterministic Schreier-generator sweep; an
//! unverified result is never returned.

use rand::Rng;
use std::sync::Arc;

use crate::bigint::BigUInt;
use crate::error::{GroupError, Result};
use crate::perm::{Perm, Point};
use crate::random::{sample_below, seeded_rng};
use crate::schreier::SchreierData;

/// A permutation group given by generators, plus the base, strong
/// generating set, and stabilizer chain Schreier–Sims computes.
///
/// `PermGroup` is **immutable once constructed** — every cached
/// structure (base, SGS, per-level Schreier tree, per-level coset
/// reps, order) is computed at construction and recorded. The hot
/// downstream operations (membership test, random element, lazy
/// iteration) are then constant-time-per-step.
#[derive(Debug, Clone)]
pub struct PermGroup {
    degree: usize,
    /// The user-supplied generators, kept verbatim.
    generators: Arc<Vec<Perm>>,
    /// Base + per-level Schreier tree / coset reps. Empty iff the
    /// group is trivial.
    levels: Arc<Vec<Level>>,
    /// First active level in `levels`. Point stabilizers at a prefix of the
    /// stored base share the immutable chain and advance this offset.
    level_start: usize,
    /// `|G|` as `∏ |orbit at level i|`.
    order: BigUInt,
}

/// Internal: combined output of [`PermGroup::build_point_stabilizer`].
/// Carries both the deduplicated Schreier generators (the surface
/// `schreier_generators` returns) and the BSGS sub-chain those
/// residues span (what `pointwise_stabilizer` and `orbit_stabilizer`
/// want as a [`PermGroup`]). Cached order alongside so we don't recompute.
struct PointStabilizerBuild {
    schreier_gens: Vec<SchreierGenerator>,
    sub_levels: Vec<Level>,
    sub_order: BigUInt,
}

/// One level of the stabilizer chain.
#[derive(Debug, Clone)]
struct Level {
    /// The base point of this level.
    base: Point,
    /// The strong generators stabilising every earlier base point
    /// (i.e. lying in `G^(i)`).
    generators: Vec<Perm>,
    /// The Schreier tree of `base` under `generators` (the orbit at
    /// this level).
    schreier: SchreierData,
    /// `reps[p]` = coset representative mapping `base` to `p`, or
    /// `None` if `p` is not in the orbit at this level.
    reps: Vec<Option<Arc<Perm>>>,
    /// Inverses of `reps`, cached because every membership sift and
    /// Schreier-generator sweep consumes them repeatedly.
    reps_inv: Vec<Option<Arc<Perm>>>,
}

impl PermGroup {
    /// Construct a permutation group on `0..degree` from a generator
    /// list, running deterministic Schreier–Sims to its BSGS.
    ///
    /// # Errors
    ///
    /// * [`GroupError::DegreeMismatch`] if any generator's length is
    ///   not `degree`.
    /// * [`GroupError::NotAPermutation`] if any generator is not a
    ///   permutation of `0..degree`.
    pub fn from_generators(degree: usize, generators: &[Perm]) -> Result<PermGroup> {
        validate_generators(degree, generators)?;
        if generators_certify_full_symmetric(degree, generators) {
            return Ok(full_symmetric_group_with_generators(degree, generators));
        }
        let mut levels: Vec<Level> = Vec::new();
        let mut strong: Vec<Perm> = generators
            .iter()
            .filter(|g| !g.is_identity())
            .cloned()
            .collect();
        schreier_sims_deterministic(degree, &mut levels, &mut strong)?;
        let order = compute_order(&levels);
        Ok(PermGroup {
            degree,
            generators: Arc::new(generators.to_vec()),
            levels: Arc::new(levels),
            level_start: 0,
            order,
        })
    }

    /// Construct via the **randomized, verified** Schreier–Sims
    /// variant.
    ///
    /// Internal Sims-style product replacement supplies random group
    /// elements; we sift each into the current chain, extend the
    /// chain on a non-trivial residue, and stop once `stabilize`
    /// random sifts in a row come back as the identity. The result is
    /// then verified by the deterministic Schreier-generator sweep —
    /// any sift that fails extends the chain. The verified output is
    /// returned; an unverified probabilistic result is never returned.
    ///
    /// The public seed is expanded by perm-core's fixed mapping into a
    /// `ChaCha20` seed, so the build is deterministic across dependency
    /// upgrades as well as within one binary.
    pub fn from_generators_randomized(
        degree: usize,
        generators: &[Perm],
        seed: u64,
        stabilize: u32,
    ) -> Result<PermGroup> {
        validate_generators(degree, generators)?;
        let mut levels: Vec<Level> = Vec::new();
        let mut strong: Vec<Perm> = generators
            .iter()
            .filter(|g| !g.is_identity())
            .cloned()
            .collect();
        // Bootstrap level 0 with the input generators so the
        // deterministic verification sweep at the end of this routine
        // sees the original group. Otherwise the Monte-Carlo phase
        // alone seeds level 0 with random products, and a
        // non-spanning random sample would silently produce an
        // under-order BSGS.
        if !strong.is_empty() {
            if let Some(base) = pick_base_point(degree, 0, &strong) {
                levels.push(build_level(degree, base, strong.clone()));
            }
        }
        let mut rng = seeded_rng(seed);
        if !strong.is_empty() {
            let mut sgsg = ProductReplacement::new(&strong, &mut rng);
            let mut consecutive_trivial: u32 = 0;
            while consecutive_trivial < stabilize {
                let g = sgsg.next(&mut rng);
                let (residue, extended) = sift_and_extend(degree, &mut levels, &mut strong, g);
                if extended {
                    consecutive_trivial = 0;
                } else if residue.is_identity() {
                    consecutive_trivial += 1;
                } else {
                    consecutive_trivial = 0;
                }
            }
        }
        // Verification — deterministic Schreier sweep extends anything
        // missed by the Monte-Carlo phase.
        schreier_sims_deterministic(degree, &mut levels, &mut strong)?;
        let order = compute_order(&levels);
        Ok(PermGroup {
            degree,
            generators: Arc::new(generators.to_vec()),
            levels: Arc::new(levels),
            level_start: 0,
            order,
        })
    }

    /// Return the subgroup generated by `self` and one additional
    /// permutation.
    ///
    /// The existing verified stabilizer chain is extended in place and then
    /// closed by the deterministic Schreier–Sims verifier. This is the
    /// natural path for searches that discover generators incrementally; it
    /// avoids rebuilding the already-proved part of the chain after every
    /// discovery.
    pub fn adjoin_generator(&self, generator: &Perm) -> Result<PermGroup> {
        validate_generators(self.degree, core::slice::from_ref(generator))?;
        if self.contains(generator)? {
            return Ok(self.clone());
        }
        if self.level_start != 0 {
            let mut generators = self.generators().to_vec();
            generators.push(generator.clone());
            return PermGroup::from_generators(self.degree, &generators);
        }

        let mut generators = self.generators().to_vec();
        generators.push(generator.clone());
        let mut levels = self.levels.as_ref().clone();
        let mut strong = self
            .strong_generating_set()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        strong.push(generator.clone());
        if levels.is_empty() {
            let base = pick_base_point(self.degree, 0, &strong)
                .expect("a non-member generator must move a point");
            levels.push(build_level(self.degree, base, strong.clone()));
        } else {
            levels[0].generators.push(generator.clone());
            let new_generator_index = levels[0].generators.len() - 1;
            extend_level_with_generator(self.degree, &mut levels[0], new_generator_index);
        }
        schreier_sims_deterministic(self.degree, &mut levels, &mut strong)?;
        let order = compute_order(&levels);
        Ok(PermGroup {
            degree: self.degree,
            generators: Arc::new(generators),
            levels: Arc::new(levels),
            level_start: 0,
            order,
        })
    }

    /// The degree — the size of the underlying domain `0..degree`.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The original generators, in input order.
    pub fn generators(&self) -> &[Perm] {
        self.generators.as_slice()
    }

    /// Whether the exact stabilizer-chain order proves this subgroup is the
    /// full symmetric group on its declared point set.
    pub(crate) fn is_full_symmetric(&self) -> bool {
        symmetric_group_order(self.degree).as_ref() == Some(&self.order)
    }

    /// Whether exact order proves this subgroup is the alternating group on
    /// its declared point set. For degree at least two, `A_n` is the unique
    /// index-two subgroup of `S_n`, so no generator-pattern heuristic enters
    /// the classification.
    pub(crate) fn is_full_alternating(&self) -> bool {
        if self.degree < 2 {
            return false;
        }
        let Some(symmetric_order) = symmetric_group_order(self.degree) else {
            return false;
        };
        let (alternating_order, remainder) = symmetric_order.divmod_u64(2);
        remainder == 0 && alternating_order == self.order
    }

    fn levels(&self) -> &[Level] {
        &self.levels[self.level_start..]
    }

    /// The base, in order — the points the stabilizer chain
    /// stabilises one level at a time. Empty for the trivial group.
    pub fn base(&self) -> Vec<Point> {
        self.levels().iter().map(|l| l.base).collect()
    }

    /// The strong generating set: the union, across levels, of the
    /// generators at each level. Returned in the order they were
    /// added (`G^(0)` first).
    pub fn strong_generating_set(&self) -> Vec<&Perm> {
        let mut sgs: Vec<&Perm> = Vec::new();
        for level in self.levels() {
            for g in &level.generators {
                if !sgs.iter().any(|s| **s == *g) {
                    sgs.push(g);
                }
            }
        }
        sgs
    }

    /// `|G|`, computed once at construction as the product of the
    /// transversal lengths.
    pub fn order(&self) -> BigUInt {
        self.order.clone()
    }

    /// Sift `g` down the stabilizer chain. Returns the *sift
    /// residue* and, on `true`, the sift terminated with the identity
    /// (i.e. `g ∈ G`). On `false`, the residue is the first
    /// non-trivial element encountered — a "witness of non-
    /// membership".
    pub fn sift(&self, g: &Perm) -> Result<(Perm, bool)> {
        if g.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                got: g.degree(),
                expected: self.degree,
            });
        }
        let (residue, _level) = sift(self.levels(), g.clone());
        let in_group = residue.is_identity();
        Ok((residue, in_group))
    }

    /// `true` iff `g` lies in the group. Sifts `g` and checks the residue is
    /// the identity.
    pub fn contains(&self, g: &Perm) -> Result<bool> {
        let (_residue, ok) = self.sift(g)?;
        Ok(ok)
    }

    /// The orbit of `point` under the **full group**, in BFS-from-
    /// `point` discovery order.
    ///
    /// # Errors
    ///
    /// [`GroupError::DegreeMismatch`] if `point >= degree`.
    pub fn orbit(&self, point: Point) -> Result<Vec<Point>> {
        if (point as usize) >= self.degree {
            return Err(GroupError::DegreeMismatch {
                got: point as usize,
                expected: self.degree,
            });
        }
        Ok(SchreierData::build(point, self.generators(), self.degree).orbit)
    }

    /// The Schreier tree of `point` under the full group. The result
    /// is a public, owned snapshot — `parents[p]` is `Some((parent,
    /// generator-index))` for non-root orbit points.
    pub fn schreier_tree(&self, point: Point) -> Result<SchreierTree> {
        if (point as usize) >= self.degree {
            return Err(GroupError::DegreeMismatch {
                got: point as usize,
                expected: self.degree,
            });
        }
        let raw = SchreierData::build(point, self.generators(), self.degree);
        Ok(SchreierTree {
            root: raw.root,
            orbit: raw.orbit,
            pred: raw.pred,
            generators_used: self.generators.as_ref().clone(),
        })
    }

    /// A transversal for the coset action of the **point stabilizer**
    /// of `point` in `G`. Returns one representative per coset, keyed
    /// to its image — i.e. `reps[i]` maps `point` to `orbit[i]`.
    ///
    /// Fast-path: when `point` is the chain's first base point, the
    /// level-0 Schreier tree + coset reps are already cached on
    /// `self`. Reuse them rather than rebuilding the BFS from scratch.
    /// (The cached level-0 generators are exactly `self.generators`,
    /// modulo trivial filtering, since deterministic Schreier–Sims
    /// seeds level 0 with the input generators — see the bootstrap in
    /// `schreier_sims_deterministic`.)
    pub fn transversal(&self, point: Point) -> Result<Transversal> {
        if (point as usize) >= self.degree {
            return Err(GroupError::DegreeMismatch {
                got: point as usize,
                expected: self.degree,
            });
        }
        if let Some(level0) = self.levels().first() {
            if level0.base == point {
                let mut reps: Vec<Perm> = Vec::with_capacity(level0.schreier.orbit.len());
                for &p in &level0.schreier.orbit {
                    reps.push(
                        level0.reps[p as usize]
                            .as_ref()
                            .expect("cached level-0 orbit rep present")
                            .as_ref()
                            .clone(),
                    );
                }
                return Ok(Transversal {
                    base: point,
                    orbit: level0.schreier.orbit.clone(),
                    reps,
                });
            }
        }
        let raw = SchreierData::build(point, self.generators(), self.degree);
        let reps_sparse = raw.all_coset_reps(self.generators(), self.degree);
        let mut reps: Vec<Perm> = Vec::with_capacity(raw.orbit.len());
        for &p in &raw.orbit {
            reps.push(reps_sparse[p as usize].clone().expect("orbit rep present"));
        }
        Ok(Transversal {
            base: point,
            orbit: raw.orbit,
            reps,
        })
    }

    /// The Schreier generators of `Stab_G(point)` — `g_i * s_j *
    /// g_{(i,s_j)}^{-1}` for every coset rep `g_i` and every
    /// generator `s_j`, where `g_{(i,s_j)}` is the rep of the image
    /// of `orbit[i]` under `s_j`.
    ///
    /// Per Schreier's lemma the raw multiset (`|orbit| * |gens|`
    /// entries minus trivials) generates the stabilizer; most of
    /// those generators are redundant. We **sift each candidate
    /// against an incremental sub-chain for the emerging stabilizer**
    /// and keep only candidates whose sift residue is non-trivial
    /// (i.e. that actually enlarge the subgroup). Output cardinality
    /// is then on the order of the sub-chain's strong-generator count
    /// rather than `|orbit| * |gens|`. The retained
    /// `(coset_index, generator_index, perm)` is the original Schreier
    /// generator, not the sift residue, so the metadata stays faithful to the
    /// construction.
    ///
    /// Early-termination optimisation: the stabilizer has known order
    /// `|G| / |orbit|` (orbit–stabilizer), so once the sub-chain's
    /// product of transversal lengths matches that, no remaining
    /// candidate can contribute and we return immediately. This is
    /// what turns the cost from `O(|orbit| * |gens|)` candidate
    /// products into `O(rank-of-stabilizer)` on groups with small
    /// stabilizers.
    ///
    /// The returned set always generates `Stab_G(point)` (the parity
    /// invariant `<schreier_gens> == Stab_G(point)` is preserved).
    pub fn schreier_generators(&self, point: Point) -> Result<Vec<SchreierGenerator>> {
        Ok(self.build_point_stabilizer(point)?.schreier_gens)
    }

    /// Internal: build the sift-deduplicated Schreier generators of
    /// `Stab_G(point)` *and* a BSGS sub-chain for the same group.
    ///
    /// The sub-chain is retained directly for [`pointwise_stabilizer`] and
    /// [`orbit_stabilizer`], avoiding a second construction from the returned
    /// Schreier generators.
    ///
    /// Two correctness subtleties:
    ///
    /// * **Sift-dedup alone is not enough** to give a *complete* chain.
    ///   Every cand sifting to identity guarantees only that
    ///   `<sub-strong-set> = Stab_G(point)`, not that
    ///   `product-of-orbits = |Stab_G(point)|`. PSL(2,5) acting on six
    ///   points with `point = 3` is a witness: two retained cands
    ///   suffice to span the order-10 stabilizer, but the sub-chain
    ///   bootstrap leaves a single level of orbit 5 — the dedup'd
    ///   cands all happen to coincide with level-0 transversal reps,
    ///   so no further install is triggered.
    /// * The **early-termination** branch *is* complete by orbit–
    ///   stabilizer (product equals `stab_target = |G|/|orbit|`), so
    ///   no further work is needed when that branch fires.
    ///
    /// Mitigation: when the sift-dedup loop exits without early
    /// termination, run a standard Schreier–Sims sweep on the partial
    /// chain to complete it. This is much cheaper than running SS from
    /// scratch on the cand list, because the level-0 BFS — usually the
    /// dominant cost on large orbits — is already done.
    fn build_point_stabilizer(&self, point: Point) -> Result<PointStabilizerBuild> {
        let trans = self.transversal(point)?;
        let mut idx: Vec<usize> = vec![usize::MAX; self.degree];
        for (i, &p) in trans.orbit.iter().enumerate() {
            idx[p as usize] = i;
        }
        // Memoise transversal-rep inverses on first touch. The naive
        // loop calls `trans.reps[qi].inverse()` per (orbit-point,
        // generator) pair; the inverse depends only on `qi`, so each
        // distinct `qi` is at most one inverse. We *lazy*-init rather
        // than pre-compute the full `orbit.len()` block, because the
        // early-termination check below frequently lets us return after
        // touching only a few `qi`s on groups with small stabilizers
        // (e.g. D_n: stabilizer order 2, one inverse needed).
        let mut reps_inv: Vec<Option<Perm>> = vec![None; trans.orbit.len()];
        // The stabilizer order target. By orbit–stabilizer,
        // `|Stab_G(point)| = |G| / |orbit_of_point|`. We can stop the
        // sweep as soon as the sub-chain we're building has this
        // order — every later Schreier-generator candidate is
        // redundant.
        let stab_target = self.order.divmod_u64(trans.orbit.len() as u64).0;
        let mut sub_levels: Vec<Level> = Vec::new();
        let mut sub_order = BigUInt::one();
        let mut out: Vec<SchreierGenerator> = Vec::new();
        let mut early_terminated = false;
        let mut scratch = Vec::with_capacity(self.degree);
        'outer: for (i, p) in trans.orbit.iter().enumerate() {
            let rep_p = &trans.reps[i];
            for (sj, s) in self.generators().iter().enumerate() {
                if sub_order == stab_target {
                    early_terminated = true;
                    break 'outer;
                }
                let q = s.apply(*p);
                let qi = idx[q as usize];
                debug_assert!(qi != usize::MAX, "orbit closure");
                if reps_inv[qi].is_none() {
                    reps_inv[qi] = Some(trans.reps[qi].inverse());
                }
                let rep_q_inv = reps_inv[qi].as_ref().expect("just initialised");
                let mut cand = rep_p.compose_unchecked(s);
                cand.compose_assign_unchecked(rep_q_inv, &mut scratch);
                if cand.is_identity() {
                    continue;
                }
                // Sift cand through the emerging sub-chain. If the
                // residue is the identity, cand lies in the subgroup
                // we already have — drop it.
                let (residue, stop_level) = sift(&sub_levels, cand.clone());
                if residue.is_identity() {
                    continue;
                }
                subchain_install(self.degree, &mut sub_levels, residue, stop_level);
                // Propagation to existing earlier levels can grow
                // multiple level orbits in one install (the residue
                // lives in `G^(j)` for every `j ≤ target`, and BFS-with-
                // residue at level `j` may discover new points via the
                // existing gens at that level). Recompute the order from
                // scratch: O(num_levels) u64 muls, with the chain rank
                // bounded by a small constant in practice, so this is
                // cheap.
                sub_order = compute_order(&sub_levels);
                out.push(SchreierGenerator {
                    coset_index: i,
                    generator_index: sj,
                    perm: cand,
                });
            }
        }
        // If the sift-dedup loop drained without hitting `stab_target`,
        // the sub-chain spans the right *group* but may not be a
        // *complete* BSGS — its product-of-orbits can be less than
        // `|Stab_G(point)|`. Run a Schreier–Sims completion sweep on
        // the partial chain; the standard fixpoint loop adds any
        // missing residues. Cheap because the level-0 BFS (the
        // dominant cost on large groups) is already done; this only
        // pays for the level-1+ residues the dedup loop didn't reach.
        if !early_terminated {
            complete_subchain(self.degree, &mut sub_levels);
            sub_order = compute_order(&sub_levels);
        }
        Ok(PointStabilizerBuild {
            schreier_gens: out,
            sub_levels,
            sub_order,
        })
    }

    /// Internal: turn the [`build_point_stabilizer`] output into the
    /// `PermGroup` that is `Stab_G(point)`. The sub-chain returned by
    /// `build_point_stabilizer` is a complete BSGS (see the
    /// completeness invariant on that method), so no Schreier–Sims
    /// rerun is needed — we just lift it onto the public `PermGroup`
    /// type with its own generators / levels / order.
    fn point_stabilizer_from_build(&self, build: PointStabilizerBuild) -> PermGroup {
        // The first level's strong set generates the whole subgroup. Lower
        // levels retain nested stabilizer generators for sifting; unioning
        // them here only duplicates permutations already present above.
        let gens = build
            .sub_levels
            .first()
            .map(|level| level.generators.clone())
            .unwrap_or_default();
        PermGroup {
            degree: self.degree,
            generators: Arc::new(gens),
            levels: Arc::new(build.sub_levels),
            level_start: 0,
            order: build.sub_order,
        }
    }

    /// The point stabilizer of `points` (jointly) in `G`. The
    /// result is itself a `PermGroup` with its own pre-computed
    /// chain.
    ///
    /// Fast-path: when `points` is a prefix of the existing base,
    /// the chain we want is just the suffix of `self`'s chain
    /// starting after that prefix — no Schreier–Sims rerun, no
    /// schreier-generator enumeration. Otherwise we fall back to
    /// the general path: rebuild from Schreier generators, point
    /// by point.
    pub fn pointwise_stabilizer(&self, points: &[Point]) -> Result<PermGroup> {
        for &p in points {
            if (p as usize) >= self.degree {
                return Err(GroupError::DegreeMismatch {
                    got: p as usize,
                    expected: self.degree,
                });
            }
        }
        if points.is_empty() {
            return Ok(self.clone());
        }

        // Fast-path: `points` is a prefix of `self`'s base. Then
        // `Stab_G(points)` = group at chain level `points.len()`.
        // The remaining levels form its complete chain.
        let base: Vec<Point> = self.base();
        if points.len() <= base.len() && points == &base[..points.len()] {
            let suffix_levels = &self.levels()[points.len()..];
            // The first active level's strong set already generates the
            // suffix subgroup. Lower levels are retained for sifting, not
            // duplicated into the public input-generator list.
            let gens = suffix_levels
                .first()
                .map(|level| level.generators.clone())
                .unwrap_or_default();
            let order = compute_order(suffix_levels);
            return Ok(PermGroup {
                degree: self.degree,
                generators: Arc::new(gens),
                levels: Arc::clone(&self.levels),
                level_start: self.level_start + points.len(),
                order,
            });
        }

        // General path: reuse the BSGS sub-chain
        // `build_point_stabilizer` already constructed.
        let mut current = self.clone();
        for &p in points {
            // If `p` is already the current chain's first base, use
            // the fast-path within the loop.
            if let Some(&first_base) = current.base().first() {
                if first_base == p {
                    let suffix_levels = &current.levels()[1..];
                    let gens = suffix_levels
                        .first()
                        .map(|level| level.generators.clone())
                        .unwrap_or_default();
                    let order = compute_order(suffix_levels);
                    current = PermGroup {
                        degree: current.degree,
                        generators: Arc::new(gens),
                        levels: Arc::clone(&current.levels),
                        level_start: current.level_start + 1,
                        order,
                    };
                    continue;
                }
            }
            let build = current.build_point_stabilizer(p)?;
            if build.sub_levels.is_empty() {
                // `p` is fixed by every generator of `current`, or
                // `current` is itself trivial — the stabilizer is the
                // trivial group acting on the same degree.
                current = PermGroup {
                    degree: self.degree,
                    generators: Arc::new(Vec::new()),
                    levels: Arc::new(Vec::new()),
                    level_start: 0,
                    order: BigUInt::one(),
                };
            } else {
                current = current.point_stabilizer_from_build(build);
            }
        }
        Ok(current)
    }

    /// The (orbit, stabilizer) pair of `point` under `G`. The
    /// orbit-stabilizer identity `|orbit| * |stab| = |G|` is an
    /// algebraic-invariant check exercised by the test suite.
    pub fn orbit_stabilizer(&self, point: Point) -> Result<OrbitStabilizer> {
        let orbit = self.orbit(point)?;
        let stab = self.pointwise_stabilizer(&[point])?;
        Ok(OrbitStabilizer {
            orbit,
            stabilizer: stab,
        })
    }

    /// The base + per-level transversal — i.e. the full stabilizer
    /// chain `G > G^(1) > G^(2) > ... > {1}`, with each level's base
    /// point, generating set, orbit (in BFS order), and transversal.
    pub fn stabilizer_chain(&self) -> StabilizerChain {
        let mut steps: Vec<StabilizerChainStep> = Vec::with_capacity(self.levels().len());
        for level in self.levels() {
            let mut reps: Vec<Perm> = Vec::with_capacity(level.schreier.orbit.len());
            for &p in &level.schreier.orbit {
                reps.push(
                    level.reps[p as usize]
                        .as_ref()
                        .expect("orbit reps populated at construction")
                        .as_ref()
                        .clone(),
                );
            }
            steps.push(StabilizerChainStep {
                base: level.base,
                generators: level.generators.clone(),
                orbit: level.schreier.orbit.clone(),
                transversal: reps,
            });
        }
        StabilizerChain { steps }
    }

    /// A lazy odometer iterator over every group element exactly
    /// once. Reverses the standard product-of-transversals formula:
    /// every `g ∈ G` writes uniquely as `t_k · t_{k-1} · ... · t_1`
    /// with `t_i ∈ T_i` (the level-`i` transversal).
    pub fn iter_elements(&self) -> ElementIter<'_> {
        let counters = vec![0usize; self.levels().len()];
        ElementIter {
            group: self,
            counters,
            done: false,
            cur: Vec::with_capacity(self.degree),
            scratch: Vec::with_capacity(self.degree),
        }
    }

    /// Sample a uniformly random group element via per-level coset
    /// choice. Returns the element plus the seed it was generated
    /// from (so the result is reproducible).
    pub fn uniform_random_element(&self, seed: u64) -> RandomElement {
        let mut rng = seeded_rng(seed);
        if self.levels().is_empty() {
            return RandomElement {
                element: Perm::identity(self.degree),
                seed,
            };
        }
        // Pick a uniform coset index at each level independently;
        // multiply the chosen representatives. This produces the
        // unique base-image vector (b_1, b_2, ..., b_k), which by
        // construction is a uniform draw from G.
        let mut images: Vec<Point> = (0..self.degree as Point).collect();
        let mut scratch = vec![0; self.degree];
        for level in self.levels().iter().rev() {
            let orbit_len = level.schreier.orbit.len();
            let idx = sample_below(&mut rng, orbit_len as u64) as usize;
            let rep = level.reps[level.schreier.orbit[idx] as usize]
                .as_ref()
                .expect("orbit rep present");
            let rep_images = rep.images();
            for (point, image) in scratch.iter_mut().enumerate() {
                *image = rep_images[images[point] as usize];
            }
            core::mem::swap(&mut images, &mut scratch);
        }
        RandomElement {
            element: Perm::from_images_unchecked(images),
            seed,
        }
    }
}

// ── public result types ─────────────────────────────────────────────

/// A transversal of `Stab_G(point)` in `G`, paired with the orbit it
/// indexes.
#[derive(Debug, Clone)]
pub struct Transversal {
    /// The point whose stabilizer's transversal this is.
    pub base: Point,
    /// The orbit of `base`, in BFS-from-base order.
    pub orbit: Vec<Point>,
    /// `reps[i]` maps `base` to `orbit[i]`.
    pub reps: Vec<Perm>,
}

/// A Schreier tree under the full group.
#[derive(Debug, Clone)]
pub struct SchreierTree {
    /// Root of the tree.
    pub root: Point,
    /// Orbit in BFS discovery order.
    pub orbit: Vec<Point>,
    /// Predecessor table — `pred[p] = Some((parent, gen_index))` for
    /// non-root orbit points.
    pub pred: Vec<Option<(Point, u32)>>,
    /// The generators referenced by `gen_index` slots.
    pub generators_used: Vec<Perm>,
}

impl SchreierTree {
    /// Number of points in the orbit.
    pub fn orbit_len(&self) -> usize {
        self.orbit.len()
    }

    /// `true` if `p` is in the orbit.
    pub fn contains(&self, p: Point) -> bool {
        (p as usize) < self.pred.len() && (p == self.root || self.pred[p as usize].is_some())
    }
}

/// One step of [`PermGroup::stabilizer_chain`].
#[derive(Debug, Clone)]
pub struct StabilizerChainStep {
    /// Base point at this level.
    pub base: Point,
    /// Strong generators stabilising every earlier base point.
    pub generators: Vec<Perm>,
    /// Orbit of `base` at this level, in BFS order.
    pub orbit: Vec<Point>,
    /// Coset representatives — `transversal[i]` maps `base` to
    /// `orbit[i]`.
    pub transversal: Vec<Perm>,
}

/// The full stabilizer chain returned by
/// [`PermGroup::stabilizer_chain`].
#[derive(Debug, Clone)]
pub struct StabilizerChain {
    /// The chain, top-level (`G`) first.
    pub steps: Vec<StabilizerChainStep>,
}

/// One Schreier generator returned by
/// [`PermGroup::schreier_generators`].
#[derive(Debug, Clone)]
pub struct SchreierGenerator {
    /// Index of the coset representative used.
    pub coset_index: usize,
    /// Index of the group generator used.
    pub generator_index: usize,
    /// The Schreier generator itself — an element of the point
    /// stabilizer.
    pub perm: Perm,
}

/// The (orbit, stabilizer) pair returned by
/// [`PermGroup::orbit_stabilizer`].
#[derive(Debug, Clone)]
pub struct OrbitStabilizer {
    /// The orbit of the requested point.
    pub orbit: Vec<Point>,
    /// The point stabilizer of the requested point.
    pub stabilizer: PermGroup,
}

/// A seeded uniform random element of a [`PermGroup`].
#[derive(Debug, Clone)]
pub struct RandomElement {
    /// The element itself.
    pub element: Perm,
    /// The seed that produced it.
    pub seed: u64,
}

/// A lazy odometer enumerator of every group element.
pub struct ElementIter<'a> {
    group: &'a PermGroup,
    counters: Vec<usize>,
    done: bool,
    /// Scratch buffers used to compose elements without allocating a
    /// fresh `Vec<Point>` per composition step. We ping-pong `cur` and
    /// `scratch` across the level-by-level multiply, then return `cur`
    /// as the yielded `Perm` and refill it on the next call.
    cur: Vec<Point>,
    scratch: Vec<Point>,
}

impl<'a> Iterator for ElementIter<'a> {
    type Item = Perm;

    fn next(&mut self) -> Option<Perm> {
        if self.done {
            return None;
        }
        let degree = self.group.degree;
        if self.group.levels().is_empty() {
            self.done = true;
            return Some(Perm::identity(degree));
        }
        // Build current element from counter state. The two buffers
        // `cur` and `scratch` ping-pong via `mem::swap`. After
        // composition the yielded `Perm` takes ownership of `cur`;
        // the iterator gets a fresh empty `Vec` to refill on the next
        // call. The remaining allocation is required because the yielded
        // element owns its image vector.
        self.cur.clear();
        self.cur.extend(0..degree as Point);
        for (i, level) in self.group.levels().iter().enumerate().rev() {
            let pt = level.schreier.orbit[self.counters[i]];
            let rep = level.reps[pt as usize].as_ref().expect("orbit rep present");
            // out = cur * rep (right action). With levels traversed from the
            // deepest stabilizer to the top, this constructs the unique
            // `t_k * ... * t_1` transversal decomposition. Write into scratch, then
            // swap to leave the result in `cur` for the next round.
            self.scratch.clear();
            self.scratch.resize(degree, 0);
            let rep_images = rep.images();
            for (j, slot) in self.scratch.iter_mut().enumerate() {
                *slot = rep_images[self.cur[j] as usize];
            }
            core::mem::swap(&mut self.cur, &mut self.scratch);
        }
        // Advance the counters.
        let mut carry = true;
        for (i, level) in self.group.levels().iter().enumerate() {
            if !carry {
                break;
            }
            self.counters[i] += 1;
            if self.counters[i] < level.schreier.orbit.len() {
                carry = false;
            } else {
                self.counters[i] = 0;
            }
        }
        if carry {
            self.done = true;
        }
        // Yield `cur` as a Perm — move the buffer out, replacing it
        // with a fresh empty Vec so the next call re-allocates as
        // needed. (The yielded element owns its image vector, which is
        // what callers expect.)
        let yielded = core::mem::take(&mut self.cur);
        Some(Perm::from_images_unchecked(yielded))
    }
}

// ── free functions ──────────────────────────────────────────────────

/// Computes a deterministic base and strong generating set.
pub fn base_and_strong_generating_set(
    degree: usize,
    generators: &[Perm],
) -> Result<(Vec<Point>, Vec<Perm>)> {
    let g = PermGroup::from_generators(degree, generators)?;
    let base = g.base();
    let sgs: Vec<Perm> = g.strong_generating_set().into_iter().cloned().collect();
    Ok((base, sgs))
}

/// Computes the order of the generated permutation group.
pub fn group_order(degree: usize, generators: &[Perm]) -> Result<BigUInt> {
    Ok(PermGroup::from_generators(degree, generators)?.order())
}

/// Tests whether a permutation belongs to the group.
pub fn element_membership_test(group: &PermGroup, g: &Perm) -> Result<bool> {
    group.contains(g)
}

/// Computes the orbit of a point.
pub fn orbit(group: &PermGroup, point: Point) -> Result<Vec<Point>> {
    group.orbit(point)
}

/// Computes a transversal for the orbit of a point.
pub fn transversal(group: &PermGroup, point: Point) -> Result<Transversal> {
    group.transversal(point)
}

/// Computes a Schreier tree rooted at a point.
pub fn schreier_tree(group: &PermGroup, point: Point) -> Result<SchreierTree> {
    group.schreier_tree(point)
}

/// Returns the group's stabilizer chain.
pub fn stabilizer_chain(group: &PermGroup) -> StabilizerChain {
    group.stabilizer_chain()
}

/// Computes the pointwise stabilizer of the given points.
pub fn pointwise_stabilizer(group: &PermGroup, points: &[Point]) -> Result<PermGroup> {
    group.pointwise_stabilizer(points)
}

/// Computes the orbit and stabilizer of a point.
pub fn orbit_stabilizer(group: &PermGroup, point: Point) -> Result<OrbitStabilizer> {
    group.orbit_stabilizer(point)
}

/// Computes Schreier generators for a point stabilizer.
pub fn schreier_generators(group: &PermGroup, point: Point) -> Result<Vec<SchreierGenerator>> {
    group.schreier_generators(point)
}

/// Iterates over the group elements.
pub fn iterate_elements(group: &PermGroup) -> ElementIter<'_> {
    group.iter_elements()
}

/// Samples a reproducible uniform random element.
pub fn uniform_random_element(group: &PermGroup, seed: u64) -> RandomElement {
    group.uniform_random_element(seed)
}

// ── internals ───────────────────────────────────────────────────────

pub(crate) fn symmetric_group_order(degree: usize) -> Option<BigUInt> {
    let mut order = BigUInt::one();
    for factor in 2..=degree {
        order.mul_assign_u64(u64::try_from(factor).ok()?);
    }
    Some(order)
}

fn validate_generators(degree: usize, generators: &[Perm]) -> Result<()> {
    for (i, g) in generators.iter().enumerate() {
        if g.degree() != degree {
            return Err(GroupError::DegreeMismatch {
                got: g.degree(),
                expected: degree,
            });
        }
        // The Perm constructor already enforces bijection on the
        // underlying domain — but a user-supplied vector that *was*
        // already a Perm could in principle have been built unchecked.
        // Re-validate cheaply.
        let mut seen = vec![false; degree];
        for &x in g.images() {
            let xi = x as usize;
            if xi >= degree {
                return Err(GroupError::NotAPermutation {
                    generator: i,
                    reason: crate::error::NotAPermutationReason::OutOfRange { image: xi, degree },
                });
            }
            if seen[xi] {
                return Err(GroupError::NotAPermutation {
                    generator: i,
                    reason: crate::error::NotAPermutationReason::RepeatedImage { image: xi },
                });
            }
            seen[xi] = true;
        }
    }
    Ok(())
}

/// Recognize the standard theorem that an `n`-cycle and a transposition
/// generate `S_n` exactly when the orbit of the transposed pair under the
/// cycle is a connected graph. In cycle coordinates this is the condition
/// `gcd(n, j - i) = 1`.
///
/// This is an exact certificate, not a generator-pattern guess: the conjugate
/// transpositions are the edges of a connected circulant graph, and edge
/// transpositions of any connected graph generate the full symmetric group.
fn generators_certify_full_symmetric(degree: usize, generators: &[Perm]) -> bool {
    if degree < 2 {
        return false;
    }
    let transpositions = generators
        .iter()
        .filter_map(transposition_points)
        .collect::<Vec<_>>();
    if transpositions.is_empty() {
        return false;
    }
    generators.iter().any(|cycle| {
        let Some(coordinates) = full_cycle_coordinates(cycle) else {
            return false;
        };
        transpositions.iter().any(|&(left, right)| {
            let distance = coordinates[left].abs_diff(coordinates[right]);
            greatest_common_divisor(distance, degree) == 1
        })
    })
}

fn transposition_points(permutation: &Perm) -> Option<(usize, usize)> {
    let moved = permutation
        .images()
        .iter()
        .enumerate()
        .filter_map(|(point, &image)| (point != image as usize).then_some(point))
        .collect::<Vec<_>>();
    if moved.len() == 2
        && permutation.apply(moved[0] as Point) as usize == moved[1]
        && permutation.apply(moved[1] as Point) as usize == moved[0]
    {
        Some((moved[0], moved[1]))
    } else {
        None
    }
}

fn full_cycle_coordinates(permutation: &Perm) -> Option<Vec<usize>> {
    let degree = permutation.degree();
    if degree == 0 {
        return None;
    }
    let mut coordinates = vec![usize::MAX; degree];
    let mut point = 0_usize;
    for coordinate in 0..degree {
        if coordinates[point] != usize::MAX {
            return None;
        }
        coordinates[point] = coordinate;
        point = permutation.apply(point as Point) as usize;
    }
    (point == 0).then_some(coordinates)
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Build the canonical stabilizer chain for `S_n` after
/// [`generators_certify_full_symmetric`] has proved the input group is all of
/// `S_n`. Each level uses a transposition and a cycle on the remaining suffix,
/// giving the exact chain `S_n > S_{n-1} > ... > S_1` without a redundant
/// Schreier-generator closure sweep.
fn full_symmetric_group_with_generators(degree: usize, generators: &[Perm]) -> PermGroup {
    let mut levels = Vec::with_capacity(degree.saturating_sub(1));
    for base in 0..degree.saturating_sub(1) {
        let mut transposition = (0..degree as Point).collect::<Vec<_>>();
        transposition.swap(base, base + 1);
        let transposition = Perm::from_images_unchecked(transposition);
        let level_generators = if base + 2 == degree {
            vec![transposition]
        } else {
            let mut cycle = (0..degree as Point).collect::<Vec<_>>();
            for (point, image) in cycle.iter_mut().enumerate().take(degree - 1).skip(base) {
                *image = (point + 1) as Point;
            }
            cycle[degree - 1] = base as Point;
            vec![transposition, Perm::from_images_unchecked(cycle)]
        };
        levels.push(build_level(degree, base as Point, level_generators));
    }
    PermGroup {
        degree,
        generators: Arc::new(generators.to_vec()),
        levels: Arc::new(levels),
        level_start: 0,
        order: symmetric_group_order(degree).expect("validated degree fits permutation points"),
    }
}

/// Build the direct product of full symmetric groups on disjoint blocks.
///
/// The blocks must partition `0..degree`.  The resulting stabilizer chain is
/// written down directly: within each block it is the canonical symmetric
/// chain, while every later block remains in the point stabilizer.  This is an
/// exact construction, not a recognition heuristic, and avoids rerunning
/// Schreier--Sims when a caller has already proved the block-product theorem.
pub(crate) fn disjoint_symmetric_group(degree: usize, blocks: &[Vec<Point>]) -> Result<PermGroup> {
    let mut seen = vec![false; degree];
    for block in blocks {
        for &point in block {
            let index = point as usize;
            if index >= degree || seen[index] {
                return Err(GroupError::WitnessUnverified);
            }
            seen[index] = true;
        }
    }
    if seen.iter().any(|present| !present) {
        return Err(GroupError::WitnessUnverified);
    }

    fn block_generators(degree: usize, block: &[Point]) -> Vec<Perm> {
        if block.len() < 2 {
            return Vec::new();
        }
        let mut transposition = (0..degree as Point).collect::<Vec<_>>();
        transposition.swap(block[0] as usize, block[1] as usize);
        let mut generators = vec![Perm::from_images_unchecked(transposition)];
        if block.len() > 2 {
            let mut cycle = (0..degree as Point).collect::<Vec<_>>();
            for (point, image) in block
                .iter()
                .copied()
                .zip(block.iter().copied().cycle().skip(1))
                .take(block.len())
            {
                cycle[point as usize] = image;
            }
            generators.push(Perm::from_images_unchecked(cycle));
        }
        generators
    }

    let generators = blocks
        .iter()
        .flat_map(|block| block_generators(degree, block))
        .collect::<Vec<_>>();
    let mut levels = Vec::with_capacity(
        blocks
            .iter()
            .map(|block| block.len().saturating_sub(1))
            .sum(),
    );
    for (block_index, block) in blocks.iter().enumerate() {
        for offset in 0..block.len().saturating_sub(1) {
            let mut level_generators = block_generators(degree, &block[offset..]);
            for later in &blocks[block_index + 1..] {
                level_generators.extend(block_generators(degree, later));
            }
            levels.push(build_level(degree, block[offset], level_generators));
        }
    }
    let order = compute_order(&levels);
    Ok(PermGroup {
        degree,
        generators: Arc::new(generators),
        levels: Arc::new(levels),
        level_start: 0,
        order,
    })
}

fn compute_order(levels: &[Level]) -> BigUInt {
    let mut order = BigUInt::one();
    for level in levels {
        order.mul_assign_u64(level.schreier.orbit.len() as u64);
    }
    order
}

fn pick_base_point(degree: usize, level_start: usize, generators: &[Perm]) -> Option<Point> {
    // Lowest-id point moved by any generator at this level. Stable
    // base ordering — feeds reproducible BSGS layouts.
    for p in level_start..degree {
        let pp = p as Point;
        if generators.iter().any(|g| g.apply(pp) != pp) {
            return Some(pp);
        }
    }
    None
}

fn build_level(degree: usize, base: Point, generators: Vec<Perm>) -> Level {
    let schreier = SchreierData::build(base, &generators, degree);
    let reps: Vec<Option<Arc<Perm>>> = schreier
        .all_coset_reps(&generators, degree)
        .into_iter()
        .map(|representative| representative.map(Arc::new))
        .collect();
    let reps_inv = reps
        .iter()
        .map(|representative| {
            representative
                .as_ref()
                .map(|value| Arc::new(value.inverse()))
        })
        .collect();
    Level {
        base,
        generators,
        schreier,
        reps,
        reps_inv,
    }
}

/// Sims-style incremental orbit extension after a single new generator
/// has been appended at `new_gen_index`. Pre-condition: the level's
/// existing `orbit` / `pred` / `reps` are already closed under
/// `generators[0..new_gen_index]`; we only need to discover the
/// **additional** points the new generator reaches (directly or
/// indirectly via the full updated generator set), reusing every cached
/// coset rep for points already in the orbit.
///
/// This replaces a full `rebuild_level` from scratch — important
/// because `install_residue` runs it once per strong generator added,
/// and on a degree-512 group the full BFS dominates cost.
fn extend_level_with_generator(degree: usize, level: &mut Level, new_gen_index: usize) {
    let mut in_orbit = vec![false; degree];
    for &p in &level.schreier.orbit {
        in_orbit[p as usize] = true;
    }
    // Phase 1: every existing orbit point under the *new* generator
    // alone — the existing generators already close the orbit.
    let new_gen = level.generators[new_gen_index].clone();
    let existing_len = level.schreier.orbit.len();
    let mut frontier_start = level.schreier.orbit.len();
    // Single scratch buffer reused across discovered points: the
    // parent rep is read by reference, composed into `scratch`, and
    // then the buffer is taken into a fresh `Perm` (with a new
    // `scratch` allocated for the next iteration). One `Vec<Point>`
    // allocation per discovered orbit point instead of cloning the parent
    // representative and allocating the composed result separately.
    let mut scratch: Vec<Point> = Vec::with_capacity(degree);
    for i in 0..existing_len {
        let p = level.schreier.orbit[i];
        let q = new_gen.apply(p);
        if !in_orbit[q as usize] {
            in_orbit[q as usize] = true;
            level.schreier.pred[q as usize] = Some((p, new_gen_index as u32));
            level.schreier.orbit.push(q);
            let parent_rep = level.reps[p as usize]
                .as_ref()
                .expect("phase-1 parent rep present");
            parent_rep.compose_into(&new_gen, &mut scratch);
            let q_rep_images = core::mem::replace(&mut scratch, Vec::with_capacity(degree));
            let q_rep = Perm::from_images_unchecked(q_rep_images);
            level.reps_inv[q as usize] = Some(Arc::new(q_rep.inverse()));
            level.reps[q as usize] = Some(Arc::new(q_rep));
        }
    }
    // Phase 2: BFS from the newly discovered points using *every*
    // generator — those points have not been explored before.
    while frontier_start < level.schreier.orbit.len() {
        let p = level.schreier.orbit[frontier_start];
        frontier_start += 1;
        for (gi, g) in level.generators.iter().enumerate() {
            let q = g.apply(p);
            if !in_orbit[q as usize] {
                in_orbit[q as usize] = true;
                level.schreier.pred[q as usize] = Some((p, gi as u32));
                level.schreier.orbit.push(q);
                let parent_rep = level.reps[p as usize]
                    .as_ref()
                    .expect("phase-2 parent rep present");
                parent_rep.compose_into(g, &mut scratch);
                let q_rep_images = core::mem::replace(&mut scratch, Vec::with_capacity(degree));
                let q_rep = Perm::from_images_unchecked(q_rep_images);
                level.reps_inv[q as usize] = Some(Arc::new(q_rep.inverse()));
                level.reps[q as usize] = Some(Arc::new(q_rep));
            }
        }
    }
}

/// Sift `g` through `levels`. Returns the residue and the index of
/// the first level whose orbit did not contain `g(base)` (i.e. the
/// level where the sift stopped) — or `levels.len()` if the sift
/// drained.
fn sift(levels: &[Level], mut g: Perm) -> (Perm, usize) {
    let mut scratch = Vec::with_capacity(g.degree());
    for (i, level) in levels.iter().enumerate() {
        let img = g.apply(level.base);
        match &level.reps_inv[img as usize] {
            Some(rep_inv) => {
                g.compose_assign_unchecked(rep_inv, &mut scratch);
            }
            None => return (g, i),
        }
    }
    (g, levels.len())
}

/// Install `residue` as a new strong generator. Schreier–Sims SGS
/// discipline: the strong gens form a global set `S`, with the
/// per-level subsets `S_i = S ∩ G^(i)`. A residue produced at level
/// `target` lies in `G^(target)`, hence in `G^(j)` for every `j ≤
/// target` — so we add it to **every** level `0..=target`. Levels
/// where the residue fixes the base may still see the orbit grow via
/// BFS chaining, which is the whole point: it is the missing
/// generator at level `target - 1` that the "sift residue at level
/// `target`" rule alone fails to surface.
///
/// Adding above level 0 (the user's input generators) is a no-op for
/// `G^(0) = G` and we skip it.
///
/// Returns `true` if the chain actually changed (orbit grew somewhere
/// or a new level opened).
fn install_residue(
    degree: usize,
    levels: &mut Vec<Level>,
    strong: &mut Vec<Perm>,
    residue: Perm,
    target: usize,
) -> bool {
    if residue.is_identity() {
        return false;
    }
    let mut changed = false;
    // If we need a new level, open it first so the loop below sees it.
    if target == levels.len() {
        let last_base_excluded: usize = levels.last().map(|l| l.base as usize + 1).unwrap_or(0);
        let mut pick: Option<Point> = None;
        for p in last_base_excluded..degree {
            let pp = p as Point;
            if residue.apply(pp) != pp {
                pick = Some(pp);
                break;
            }
        }
        if pick.is_none() {
            for p in 0..degree {
                let pp = p as Point;
                if residue.apply(pp) != pp {
                    pick = Some(pp);
                    break;
                }
            }
        }
        let new_base = pick.expect("non-identity residue moves some point");
        let new_level = build_level(degree, new_base, vec![residue.clone()]);
        levels.push(new_level);
        changed = true;
    }
    // Propagate the new strong gen to every level it lives at — i.e.,
    // every level `j ≥ 1` where `residue ∈ G^(j)` (residue fixes
    // base_0, ..., base_{j-1}). Level 0 (= G itself) gets skipped: we
    // do not want to enlarge the user's input generator set with
    // derived elements, since that would create new level-0 Schreier
    // generators with no algebraic content.
    for j in 1..=target.min(levels.len() - 1) {
        if !levels[j].generators.iter().any(|g| g == &residue) {
            levels[j].generators.push(residue.clone());
            let new_gen_index = levels[j].generators.len() - 1;
            let before = levels[j].schreier.orbit.len();
            extend_level_with_generator(degree, &mut levels[j], new_gen_index);
            let after = levels[j].schreier.orbit.len();
            if after != before {
                changed = true;
            } else {
                // The new generator was redundant — orbit unchanged.
                // We still report `changed = true` so the outer
                // schreier_sims_deterministic loop re-checks that
                // level's Schreier generators against the extended
                // strong set (the new generator may yet surface a
                // residue we hadn't enumerated). Schreier–Sims
                // terminates regardless: the strong set is finite.
                changed = true;
            }
        }
    }
    if !strong.iter().any(|s| s == &residue) {
        strong.push(residue);
    }
    changed
}

/// Like [`install_residue`], but adds the residue to **every** level
/// (including level 0) — used when the chain is being built from
/// scratch for a *derived* subgroup (e.g. the stabilizer the
/// `schreier_generators` routine is incrementally sifting against).
/// There is no "level 0 = user's input group" semantics to protect —
/// the level-0 strong gens of the sub-chain are themselves derived
/// residues, so adding more of them is normal.
fn subchain_install(degree: usize, levels: &mut Vec<Level>, residue: Perm, target: usize) {
    debug_assert!(!residue.is_identity(), "caller must filter trivials");
    let opened_new_level = target == levels.len();
    if opened_new_level {
        let last_base_excluded: usize = levels.last().map(|l| l.base as usize + 1).unwrap_or(0);
        let mut pick: Option<Point> = None;
        for p in last_base_excluded..degree {
            let pp = p as Point;
            if residue.apply(pp) != pp {
                pick = Some(pp);
                break;
            }
        }
        if pick.is_none() {
            for p in 0..degree {
                let pp = p as Point;
                if residue.apply(pp) != pp {
                    pick = Some(pp);
                    break;
                }
            }
        }
        let new_base = pick.expect("non-identity residue moves some point");
        levels.push(build_level(degree, new_base, vec![residue.clone()]));
    }
    // Propagate the new strong gen to every existing earlier level. The
    // residue fixes `b_0 .. b_{target-1}` by construction, so it lies
    // in `G^(j)` for every `j ≤ target` — the level-j strong set should
    // contain it. Skipping the propagation leaves earlier levels'
    // orbits under-closed (residue fixes `b_j` itself but may map
    // other orbit points to new ones, which BFS-with-residue picks up).
    let propagate_up_to = if opened_new_level {
        // existing earlier levels are `0..target` (new level is at
        // index `target`, so the length is `target + 1`).
        target
    } else {
        // existing levels include the one being extended at index
        // `target`; propagate inclusively up to it.
        target + 1
    };
    for level in levels.iter_mut().take(propagate_up_to) {
        if !level.generators.iter().any(|g| g == &residue) {
            level.generators.push(residue.clone());
            let new_idx = level.generators.len() - 1;
            extend_level_with_generator(degree, level, new_idx);
        }
    }
}

/// Complete a partial sub-chain via the standard Schreier–Sims
/// fixpoint sweep. Used by [`PermGroup::build_point_stabilizer`] after
/// the sift-dedup loop — the dedup'd chain spans the right *group* but
/// is not necessarily a *complete* BSGS (see the completeness
/// commentary on `build_point_stabilizer`). This routine adds any
/// missing residues until every level's Schreier generators sift to
/// identity, giving a chain whose product-of-orbits equals the group
/// order.
///
/// Uses [`subchain_install`] (not [`install_residue`]) so that derived
/// residues propagate to *every* level including level 0 — the
/// "level 0 = user's input generators" semantic is irrelevant here
/// because the entire chain is derived.
fn complete_subchain(degree: usize, levels: &mut Vec<Level>) {
    if levels.is_empty() {
        return;
    }
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < levels.len() {
            // Snapshot at start-of-level — installs below may mutate
            // `levels[i]` and we don't want to re-iterate over the
            // newly added entries in this pass.
            let gens_at_level = levels[i].generators.clone();
            let orbit_at_level = levels[i].schreier.orbit.clone();
            let reps_at_level = levels[i].reps.clone();
            let reps_inv_at_level = levels[i].reps_inv.clone();
            let mut scratch = Vec::with_capacity(degree);
            for &p in orbit_at_level.iter() {
                let rep_p = reps_at_level[p as usize]
                    .as_ref()
                    .expect("orbit rep present")
                    .clone();
                for s in gens_at_level.iter() {
                    let q = s.apply(p);
                    let rep_q_inv = reps_inv_at_level[q as usize]
                        .as_ref()
                        .expect("orbit closure");
                    let mut schreier_gen = rep_p.compose_unchecked(s);
                    schreier_gen.compose_assign_unchecked(rep_q_inv, &mut scratch);
                    if schreier_gen.is_identity() {
                        continue;
                    }
                    let (residue, stop_offset) = sift_subchain(levels, i + 1, schreier_gen);
                    if residue.is_identity() {
                        continue;
                    }
                    let abs_stop = i + 1 + stop_offset;
                    subchain_install(degree, levels, residue, abs_stop);
                    changed = true;
                }
            }
            i += 1;
        }
    }
}

/// Sift `g`; if the residue is non-trivial, install it into the
/// chain.
fn sift_and_extend(
    degree: usize,
    levels: &mut Vec<Level>,
    strong: &mut Vec<Perm>,
    g: Perm,
) -> (Perm, bool) {
    let (residue, stop_level) = sift(levels, g);
    if residue.is_identity() {
        return (residue, false);
    }
    let installed = install_residue(degree, levels, strong, residue.clone(), stop_level);
    (Perm::identity(degree), installed)
}

/// Deterministic Schreier–Sims: bootstrap levels off the strong
/// generators, then sweep every Schreier generator of every level
/// and sift it in until everything sifts to the identity.
fn schreier_sims_deterministic(
    degree: usize,
    levels: &mut Vec<Level>,
    strong: &mut Vec<Perm>,
) -> Result<()> {
    if strong.is_empty() {
        return Ok(());
    }
    // Bootstrap: if there are no levels yet, pick a base point and
    // open a level with every generator that moves any point at or
    // beyond it.
    if levels.is_empty() {
        let base = pick_base_point(degree, 0, strong)
            .expect("non-empty strong set has at least one moved point");
        levels.push(build_level(degree, base, strong.clone()));
    }
    let mut changed = true;
    while changed {
        changed = false;
        // For every level, generate Schreier generators and sift them.
        let mut i = 0;
        while i < levels.len() {
            // Snapshot the generator and orbit slice; mutations
            // happen via sift_and_extend on later iterations.
            let gens_at_level = levels[i].generators.clone();
            let orbit_at_level = levels[i].schreier.orbit.clone();
            let reps_at_level = levels[i].reps.clone();
            let reps_inv_at_level = levels[i].reps_inv.clone();
            let mut scratch = Vec::with_capacity(degree);
            for &p in orbit_at_level.iter() {
                let rep_p = reps_at_level[p as usize]
                    .as_ref()
                    .expect("orbit rep present")
                    .clone();
                for s in gens_at_level.iter() {
                    let q = s.apply(p);
                    let rep_q_inv = reps_inv_at_level[q as usize]
                        .as_ref()
                        .expect("orbit closure");
                    let mut schreier_gen = rep_p.compose_unchecked(s);
                    schreier_gen.compose_assign_unchecked(rep_q_inv, &mut scratch);
                    if schreier_gen.is_identity() {
                        continue;
                    }
                    // Sift starting at level i+1 (we already know it
                    // stabilises level i's base).
                    let (residue, stop_offset) = sift_subchain(levels, i + 1, schreier_gen.clone());
                    if residue.is_identity() {
                        continue;
                    }
                    let abs_stop = i + 1 + stop_offset;
                    if install_residue(degree, levels, strong, residue, abs_stop) {
                        changed = true;
                    }
                }
            }
            i += 1;
        }
    }
    Ok(())
}

/// Sift through the **tail** `levels[start..]`. Returns `(residue,
/// relative-offset)`.
fn sift_subchain(levels: &[Level], start: usize, g: Perm) -> (Perm, usize) {
    sift(&levels[start..], g)
}

/// Sims' product replacement — used by the randomized BSGS to draw
/// "random" group elements.
///
/// Maintains a state vector of accumulator generators; each `next`
/// call picks two distinct slots, replaces one with the product, and
/// returns the updated accumulator. The result quickly mixes towards
/// the uniform distribution on the group.
struct ProductReplacement {
    state: Vec<Perm>,
    accumulator: Perm,
}

impl ProductReplacement {
    fn new<R: Rng>(generators: &[Perm], _rng: &mut R) -> ProductReplacement {
        // Sims recommends a stash of at least 11 slots, padded by
        // repeating the input generators.
        let degree = generators[0].degree();
        let mut state: Vec<Perm> = Vec::with_capacity(11);
        for i in 0..11 {
            state.push(generators[i % generators.len()].clone());
        }
        // Discard a few rounds to mix.
        let mut pr = ProductReplacement {
            state,
            accumulator: Perm::identity(degree),
        };
        let mut prelim = seeded_rng(0xC0FF_EE12);
        for _ in 0..30 {
            pr.next(&mut prelim);
        }
        pr
    }

    fn next<R: Rng>(&mut self, rng: &mut R) -> Perm {
        let len = self.state.len();
        let i = sample_below(rng, len as u64) as usize;
        let mut j = sample_below(rng, (len - 1) as u64) as usize;
        if j >= i {
            j += 1;
        }
        // s[i] <- s[i] * s[j]
        let prod = self.state[i].compose_unchecked(&self.state[j]);
        self.state[i] = prod.clone();
        self.accumulator = self.accumulator.compose_unchecked(&self.state[i]);
        self.accumulator.clone()
    }
}

#[cfg(test)]
mod block_product_tests {
    use super::*;

    #[test]
    fn disjoint_symmetric_group_builds_three_independent_blocks() {
        let blocks = vec![vec![0, 3, 6], vec![1, 4], vec![2, 5, 7]];
        let group = disjoint_symmetric_group(8, &blocks).unwrap();
        assert_eq!(group.order(), BigUInt::from(72_u64));
        for generator in group.generators() {
            for block in &blocks {
                for &point in block {
                    assert!(block.contains(&generator.apply(point)));
                }
            }
            assert!(group.contains(generator).unwrap());
        }
        assert_eq!(group.orbit(0).unwrap(), vec![0, 3, 6]);
        assert_eq!(group.orbit(1).unwrap(), vec![1, 4]);
        assert_eq!(group.orbit(2).unwrap(), vec![2, 5, 7]);
    }

    #[test]
    fn disjoint_symmetric_group_rejects_a_nonpartition() {
        assert!(matches!(
            disjoint_symmetric_group(4, &[vec![0, 1], vec![1, 2]]),
            Err(GroupError::WitnessUnverified)
        ));
    }
}
