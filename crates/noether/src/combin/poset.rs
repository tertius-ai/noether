// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite partially ordered sets over `graph-core` storage.

use core::cmp::Ordering;
use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Graph, GraphError, VertexId};
use crate::groups::{GroupError, Permutation, PermutationGroup, Point};
use crate::homalg::FinitelyGeneratedAbelianGroup;
use crate::topology::{SimplicialComplex, TopologyError};
use crate::Integer;

/// Maximum number of nodes in the exact principal-downset canonical search.
const CANONICAL_SEARCH_NODE_CAP: usize = 10_000_000;

/// Maximum number of maximal chains materialized by [`Poset::order_complex`].
pub const MAX_ORDER_COMPLEX_FACETS: usize = 1_000_000;

/// Maximum number of nonempty chains materialized by [`Poset::order_complex`].
pub const MAX_ORDER_COMPLEX_SIMPLICES: usize = 1_000_000;

/// Largest order accepted by the dense 0.1 comparability representation.
///
/// The relation uses one bit per ordered pair. A deterministic limit makes an
/// oversized construction a typed failure instead of an attempted unbounded
/// allocation. A future sparse representation can raise this without changing
/// the mathematical interface.
pub const MAX_POSET_ORDER: usize = 16_384;

/// A failure while constructing or querying a finite poset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PosetError {
    /// The requested order exceeds the bounded dense representation.
    OrderTooLarge {
        /// Requested number of elements.
        order: usize,
        /// Largest accepted number of elements.
        max: usize,
    },
    /// A relation endpoint is not an element of the poset.
    RelationOutOfRange {
        /// Lower endpoint.
        lower: VertexId,
        /// Upper endpoint.
        upper: VertexId,
        /// Number of elements in the poset.
        order: usize,
    },
    /// A strict generating relation had equal endpoints.
    ReflexiveRelation {
        /// The repeated endpoint.
        element: VertexId,
    },
    /// The generating relations contain a directed cycle.
    CyclicRelations,
    /// A queried element is not in the poset.
    ElementOutOfRange {
        /// Queried element.
        element: VertexId,
        /// Number of elements in the poset.
        order: usize,
    },
    /// Building the canonical Hasse graph failed.
    Graph(GraphError),
    /// Exact permutation construction or group computation failed.
    Group(GroupError),
    /// Exact topology or homology computation failed.
    Topology(TopologyError),
    /// The exact principal-downset canonical search exhausted its node cap.
    CanonicalSearchBudgetExceeded {
        /// Maximum number of search nodes allowed.
        cap: usize,
    },
    /// Order-complex construction exceeded its maximal-chain limit.
    OrderComplexFacetLimitExceeded {
        /// Maximum number of facets allowed.
        cap: usize,
    },
    /// Order-complex construction exceeded its nonempty-simplex limit.
    OrderComplexSimplexLimitExceeded {
        /// Maximum number of simplices allowed.
        cap: usize,
    },
    /// An internally generated automorphism failed exact order verification.
    AutomorphismVerificationFailed,
}

impl fmt::Display for PosetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrderTooLarge { order, max } => {
                write!(f, "poset order {order} exceeds the 0.1 limit {max}")
            }
            Self::RelationOutOfRange {
                lower,
                upper,
                order,
            } => write!(
                f,
                "poset relation ({lower}, {upper}) has an endpoint out of range [0, {order})"
            ),
            Self::ReflexiveRelation { element } => write!(
                f,
                "poset generating relation ({element}, {element}) must be strict"
            ),
            Self::CyclicRelations => {
                f.write_str("poset generating relations contain a directed cycle")
            }
            Self::ElementOutOfRange { element, order } => {
                write!(f, "poset element {element} is out of range [0, {order})")
            }
            Self::Graph(error) => write!(f, "could not build the poset Hasse graph: {error}"),
            Self::Group(error) => write!(f, "could not compute poset symmetry: {error}"),
            Self::Topology(error) => write!(f, "could not compute poset topology: {error}"),
            Self::CanonicalSearchBudgetExceeded { cap } => {
                write!(f, "poset canonical-form search exceeded its {cap}-node cap")
            }
            Self::OrderComplexFacetLimitExceeded { cap } => {
                write!(f, "poset order complex exceeded its {cap}-facet limit")
            }
            Self::OrderComplexSimplexLimitExceeded { cap } => {
                write!(f, "poset order complex exceeded its {cap}-simplex limit")
            }
            Self::AutomorphismVerificationFailed => {
                f.write_str("a proposed poset automorphism failed exact verification")
            }
        }
    }
}

impl std::error::Error for PosetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Group(error) => Some(error),
            Self::Topology(error) => Some(error),
            Self::OrderTooLarge { .. }
            | Self::RelationOutOfRange { .. }
            | Self::ReflexiveRelation { .. }
            | Self::CyclicRelations
            | Self::ElementOutOfRange { .. }
            | Self::CanonicalSearchBudgetExceeded { .. }
            | Self::OrderComplexFacetLimitExceeded { .. }
            | Self::OrderComplexSimplexLimitExceeded { .. }
            | Self::AutomorphismVerificationFailed => None,
        }
    }
}

impl From<GraphError> for PosetError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<GroupError> for PosetError {
    fn from(error: GroupError) -> Self {
        Self::Group(error)
    }
}

impl From<TopologyError> for PosetError {
    fn from(error: TopologyError) -> Self {
        Self::Topology(error)
    }
}

/// Whether a beat-point deletion is certified from below or from above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BeatPointKind {
    /// The strict lower set has a greatest element.
    Down,
    /// The strict upper set has a least element.
    Up,
}

/// A checked beat point and the neighboring element witnessing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeatPoint {
    element: VertexId,
    kind: BeatPointKind,
    witness: VertexId,
}

impl BeatPoint {
    /// The element which may be deleted.
    #[must_use]
    pub fn element(&self) -> VertexId {
        self.element
    }

    /// Whether the certificate comes from the strict lower or upper set.
    #[must_use]
    pub fn kind(&self) -> BeatPointKind {
        self.kind
    }

    /// The greatest strict predecessor or least strict successor.
    #[must_use]
    pub fn witness(&self) -> VertexId {
        self.witness
    }
}

/// Exact canonical labeling and symmetry evidence for a finite poset.
///
/// The labeling maps original element ids to the canonical poset. Its
/// automorphism group instead acts on the original ids; the labeling itself
/// need not be an automorphism.
#[derive(Debug, Clone)]
pub struct PosetCanonicalLabel {
    poset: Poset,
    labeling: Permutation,
    incidence_sequence: Vec<Integer>,
    automorphism_group: PermutationGroup,
}

impl PosetCanonicalLabel {
    /// The canonically relabeled poset.
    #[must_use]
    pub fn poset(&self) -> &Poset {
        &self.poset
    }

    /// The permutation from original ids to canonical ids.
    #[must_use]
    pub fn labeling(&self) -> &Permutation {
        &self.labeling
    }

    /// Principal-downset integers in canonical element order.
    #[must_use]
    pub fn incidence_sequence(&self) -> &[Integer] {
        &self.incidence_sequence
    }

    /// The complete automorphism group on the original element ids.
    #[must_use]
    pub fn automorphism_group(&self) -> &PermutationGroup {
        &self.automorphism_group
    }

    /// Verify the canonical relabeling, incidence sequence, and generators.
    #[must_use]
    pub fn verify(&self, original: &Poset) -> bool {
        self.labeling.degree() == original.order()
            && relabeled_cover_pairs(original, &self.labeling) == self.poset.cover_pairs()
            && self.incidence_sequence == canonical_incidence_sequence(&self.poset)
            && self
                .automorphism_group
                .generators()
                .iter()
                .all(|generator| is_poset_automorphism(original, generator))
    }
}

/// Replayable evidence that beat-point deletions reach a canonical core.
#[derive(Debug, Clone)]
pub struct CoreReduction {
    reduced: Poset,
    core: Poset,
    deletions: Vec<BeatPoint>,
    retained_vertices: Vec<VertexId>,
    canonical_labeling: Permutation,
}

impl CoreReduction {
    /// The terminal core before its final canonical relabeling.
    #[must_use]
    pub fn reduced(&self) -> &Poset {
        &self.reduced
    }

    /// The canonical representative of the terminal core.
    #[must_use]
    pub fn core(&self) -> &Poset {
        &self.core
    }

    /// Beat-point deletions, in replay order and in original vertex ids.
    #[must_use]
    pub fn deletions(&self) -> &[BeatPoint] {
        &self.deletions
    }

    /// Map from reduced-poset ids to retained original vertex ids.
    #[must_use]
    pub fn retained_vertices(&self) -> &[VertexId] {
        &self.retained_vertices
    }

    /// Final permutation from reduced-poset ids to canonical-core ids.
    #[must_use]
    pub fn canonical_labeling(&self) -> &Permutation {
        &self.canonical_labeling
    }

    /// Compute lean integral homology from the canonical core.
    ///
    /// Beat-point deletion preserves the homotopy type of the order complex,
    /// so these groups are also the integral homology of the source poset.
    /// The returned vector is indexed by homological degree.
    pub fn graded_homology_invariants(
        &self,
    ) -> Result<Vec<FinitelyGeneratedAbelianGroup>, PosetError> {
        Ok(self.core.order_complex()?.graded_homology_invariants()?)
    }

    /// Replay every deletion and verify the stored reduced and canonical cores.
    #[must_use]
    pub fn verify(&self, original: &Poset) -> bool {
        let mut active = vec![true; original.order()];
        for deletion in &self.deletions {
            if !active
                .get(deletion.element as usize)
                .copied()
                .unwrap_or(false)
                || !beat_point_is_valid(original, &active, *deletion)
            {
                return false;
            }
            active[deletion.element as usize] = false;
        }
        let retained = active
            .iter()
            .enumerate()
            .filter_map(|(vertex, &is_active)| is_active.then_some(vertex as VertexId))
            .collect::<Vec<_>>();
        if retained != self.retained_vertices {
            return false;
        }
        let Ok(reduced) = induced_poset(original, &retained) else {
            return false;
        };
        if reduced != self.reduced {
            return false;
        }
        let relabeled = relabeled_cover_pairs(&reduced, &self.canonical_labeling);
        relabeled == self.core.cover_pairs() && self.core.beat_points().is_empty()
    }
}

/// A finite poset on the standard element set `0..order`.
///
/// The defining graph is the directed, unweighted Hasse diagram. Construction
/// accepts any acyclic set of strict generating relations, then computes the
/// transitive closure and removes redundant relations. Thus equal posets have
/// equal defining graphs even when one input includes transitive edges.
///
/// # Example
///
/// The four-element diamond is specified by its cover relations:
///
/// ```
/// use noether::combin::Poset;
///
/// let diamond = Poset::new(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])?;
/// assert!(diamond.leq(0, 3)?);
/// assert_eq!(diamond.covers().edge_count(), 4);
/// # Ok::<(), noether::combin::PosetError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Poset {
    covers: Graph,
    leq: Vec<u64>,
    words_per_row: usize,
}

#[derive(Serialize, Deserialize)]
struct PosetRepr {
    order: usize,
    covers: Vec<(VertexId, VertexId)>,
}

impl Poset {
    /// Construct a finite poset from strict generating relations `lower < upper`.
    pub fn new(order: usize, relations: &[(VertexId, VertexId)]) -> Result<Self, PosetError> {
        if order > MAX_POSET_ORDER {
            return Err(PosetError::OrderTooLarge {
                order,
                max: MAX_POSET_ORDER,
            });
        }

        let mut unique = BTreeSet::new();
        for &(lower, upper) in relations {
            if lower as usize >= order || upper as usize >= order {
                return Err(PosetError::RelationOutOfRange {
                    lower,
                    upper,
                    order,
                });
            }
            if lower == upper {
                return Err(PosetError::ReflexiveRelation { element: lower });
            }
            unique.insert((lower, upper));
        }

        let mut outgoing = vec![Vec::new(); order];
        let mut indegree = vec![0usize; order];
        for &(lower, upper) in &unique {
            outgoing[lower as usize].push(upper as usize);
            indegree[upper as usize] += 1;
        }

        let mut ready: VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(vertex, &degree)| (degree == 0).then_some(vertex))
            .collect();
        let mut topological = Vec::with_capacity(order);
        while let Some(vertex) = ready.pop_front() {
            topological.push(vertex);
            for &upper in &outgoing[vertex] {
                indegree[upper] -= 1;
                if indegree[upper] == 0 {
                    ready.push_back(upper);
                }
            }
        }
        if topological.len() != order {
            return Err(PosetError::CyclicRelations);
        }

        let words_per_row = order.div_ceil(u64::BITS as usize);
        let relation_words = order
            .checked_mul(words_per_row)
            .ok_or(PosetError::OrderTooLarge {
                order,
                max: MAX_POSET_ORDER,
            })?;
        let mut leq = vec![0u64; relation_words];
        for element in 0..order {
            set_relation(&mut leq, words_per_row, element, element);
        }

        // Successor rows are complete when visited because this is reverse
        // topological order. Packed rows make each closure union 64 relations
        // at a time while retaining one bit per ordered pair.
        for &lower in topological.iter().rev() {
            for &upper in &outgoing[lower] {
                for word in 0..words_per_row {
                    let inherited = leq[upper * words_per_row + word];
                    leq[lower * words_per_row + word] |= inherited;
                }
            }
        }

        // Every cover occurs among the generating edges: if a comparable pair
        // needs a path of length at least two, the interior vertex proves that
        // pair is not a cover. It is enough to test the other generating
        // successors of the lower endpoint against the packed closure.
        let covers: Vec<(VertexId, VertexId)> = unique
            .into_iter()
            .filter(|&(lower, upper)| {
                !outgoing[lower as usize].iter().any(|&middle| {
                    middle != upper as usize
                        && has_relation(&leq, words_per_row, middle, upper as usize)
                })
            })
            .collect();
        let covers = Graph::from_unweighted(order, &covers, true)?;

        Ok(Self {
            covers,
            leq,
            words_per_row,
        })
    }

    /// Number of elements.
    #[must_use]
    pub fn order(&self) -> usize {
        self.covers.order()
    }

    /// The directed Hasse diagram, with every cover edge oriented upward.
    #[must_use]
    pub fn covers(&self) -> &Graph {
        &self.covers
    }

    /// Return whether `lower <= upper`.
    pub fn leq(&self, lower: VertexId, upper: VertexId) -> Result<bool, PosetError> {
        self.validate_element(lower)?;
        self.validate_element(upper)?;
        Ok(has_relation(
            &self.leq,
            self.words_per_row,
            lower as usize,
            upper as usize,
        ))
    }

    /// Compute the exact principal-downset canonical labeling.
    ///
    /// For a labeling by `0..order`, associate to element `k` the integer
    /// `sum(2^j)` over every `j <= k` in the poset. The canonical form is the
    /// relabeling whose sequence of these integers is lexicographically least.
    /// The returned permutation maps the current element ids to those canonical
    /// ids; its automorphism group acts on the current ids.
    pub fn canonical_label(&self) -> Result<PosetCanonicalLabel, PosetError> {
        let search = PosetCanonicalSearch::new(self)?.run()?;
        let labeling = labeling_from_order(
            search
                .best_order
                .as_deref()
                .expect("every finite poset canonical search has a leaf"),
        )?;
        let canonical = Self::new(self.order(), &relabeled_cover_pairs(self, &labeling))?;
        let result = PosetCanonicalLabel {
            incidence_sequence: canonical_incidence_sequence(&canonical),
            poset: canonical,
            labeling,
            automorphism_group: search.group,
        };
        debug_assert!(result.verify(self));
        Ok(result)
    }

    /// Return the exact representative minimizing the principal-downset tuple.
    ///
    /// Isomorphic finite posets have equal canonical forms. Canonicalization is
    /// explicit: it does not change labeled equality or faithful serialization.
    pub fn canonical_form(&self) -> Result<Self, PosetError> {
        Ok(self.canonical_label()?.poset)
    }

    /// Compute the exact automorphism group acting on the element ids.
    pub fn automorphism_group(&self) -> Result<PermutationGroup, PosetError> {
        Ok(self.canonical_label()?.automorphism_group)
    }

    /// Number of strict predecessors of every element, indexed by element id.
    #[must_use]
    pub fn strict_lower_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.order()];
        for lower in 0..self.order() {
            for (word_index, &word) in self.relation_row(lower).iter().enumerate() {
                let mut strict = word;
                if word_index == lower / u64::BITS as usize {
                    strict &= !(1u64 << (lower % u64::BITS as usize));
                }
                while strict != 0 {
                    let bit = strict.trailing_zeros() as usize;
                    let upper = word_index * u64::BITS as usize + bit;
                    if upper < self.order() {
                        counts[upper] += 1;
                    }
                    strict &= strict - 1;
                }
            }
        }
        counts
    }

    /// Number of strict successors of every element, indexed by element id.
    #[must_use]
    pub fn strict_upper_counts(&self) -> Vec<usize> {
        (0..self.order())
            .map(|element| {
                self.relation_row(element)
                    .iter()
                    .map(|word| word.count_ones() as usize)
                    .sum::<usize>()
                    .saturating_sub(1)
            })
            .collect()
    }

    /// Minimal elements in ascending id order.
    #[must_use]
    pub fn minimal_elements(&self) -> Vec<VertexId> {
        let mut has_lower_cover = vec![false; self.order()];
        for edge in self.covers.edges() {
            has_lower_cover[edge.target as usize] = true;
        }
        has_lower_cover
            .iter()
            .enumerate()
            .filter_map(|(element, &has_lower)| (!has_lower).then_some(element as VertexId))
            .collect()
    }

    /// Maximal elements in ascending id order.
    #[must_use]
    pub fn maximal_elements(&self) -> Vec<VertexId> {
        let mut has_upper_cover = vec![false; self.order()];
        for edge in self.covers.edges() {
            has_upper_cover[edge.source as usize] = true;
        }
        has_upper_cover
            .iter()
            .enumerate()
            .filter_map(|(element, &has_upper)| (!has_upper).then_some(element as VertexId))
            .collect()
    }

    /// Number of strict comparable pairs.
    #[must_use]
    pub fn comparable_pair_count(&self) -> usize {
        self.strict_upper_counts().into_iter().sum()
    }

    /// Weakly connected components of the Hasse diagram.
    #[must_use]
    pub fn connected_components(&self) -> Vec<Vec<VertexId>> {
        let components = self.covers.connected_components();
        (0..components.count)
            .map(|component| components.members(component))
            .collect()
    }

    /// Number of elements in a longest chain; zero for the empty poset.
    #[must_use]
    pub fn height(&self) -> usize {
        let mut upper_covers = vec![Vec::new(); self.order()];
        let mut indegree = vec![0usize; self.order()];
        for edge in self.covers.edges() {
            upper_covers[edge.source as usize].push(edge.target as usize);
            indegree[edge.target as usize] += 1;
        }
        let mut ready = VecDeque::new();
        let mut longest = vec![1usize; self.order()];
        for (element, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                ready.push_back(element);
            }
        }
        let mut answer = 0;
        while let Some(lower) = ready.pop_front() {
            answer = answer.max(longest[lower]);
            for &upper in &upper_covers[lower] {
                longest[upper] = longest[upper].max(longest[lower] + 1);
                indegree[upper] -= 1;
                if indegree[upper] == 0 {
                    ready.push_back(upper);
                }
            }
        }
        answer
    }

    /// Maximum size of an antichain, via the exact Dilworth matching reduction.
    #[must_use]
    pub fn width(&self) -> usize {
        self.order() - maximum_strict_order_matching(self)
    }

    /// Reverse every comparison while retaining the current element ids.
    pub fn opposite(&self) -> Result<Self, PosetError> {
        let relations = self
            .covers
            .edges()
            .iter()
            .map(|edge| (edge.target, edge.source))
            .collect::<Vec<_>>();
        Self::new(self.order(), &relations)
    }

    /// Whether every pair has a unique meet and join.
    #[must_use]
    pub fn is_lattice(&self) -> bool {
        if self.order() == 0 {
            return false;
        }
        for left in 0..self.order() {
            for right in left..self.order() {
                if !self.has_meet(left, right) || !self.has_join(left, right) {
                    return false;
                }
            }
        }
        true
    }

    /// Whether this poset is isomorphic to its opposite.
    pub fn is_self_dual(&self) -> Result<bool, PosetError> {
        Ok(self.canonical_form()? == self.opposite()?.canonical_form()?)
    }

    /// All current up- and down-beat certificates in deterministic order.
    #[must_use]
    pub fn beat_points(&self) -> Vec<BeatPoint> {
        beat_points_with_active(self, &vec![true; self.order()])
    }

    /// Whether the poset has no beat points.
    #[must_use]
    pub fn is_core(&self) -> bool {
        self.beat_points().is_empty()
    }

    /// Delete beat points deterministically and return replayable core evidence.
    pub fn core(&self) -> Result<CoreReduction, PosetError> {
        let mut active = vec![true; self.order()];
        let mut deletions = Vec::new();
        loop {
            let Some(deletion) = beat_points_with_active(self, &active).into_iter().next() else {
                break;
            };
            active[deletion.element as usize] = false;
            deletions.push(deletion);
        }
        let retained_vertices = active
            .iter()
            .enumerate()
            .filter_map(|(vertex, &is_active)| is_active.then_some(vertex as VertexId))
            .collect::<Vec<_>>();
        let reduced = induced_poset(self, &retained_vertices)?;
        let canonical = reduced.canonical_label()?;
        let result = CoreReduction {
            reduced,
            core: canonical.poset,
            deletions,
            retained_vertices,
            canonical_labeling: canonical.labeling,
        };
        debug_assert!(result.verify(self));
        Ok(result)
    }

    /// Compute lean integral homology after deterministic beat-point reduction.
    ///
    /// This is the poset-specific homology playbook: reducible posets first
    /// pass to their canonical core, while core posets proceed directly to the
    /// sparse Smith-normal-form calculation. The returned vector is indexed by
    /// homological degree and carries no source-retaining boundary witnesses.
    pub fn graded_homology_invariants(
        &self,
    ) -> Result<Vec<FinitelyGeneratedAbelianGroup>, PosetError> {
        self.core()?.graded_homology_invariants()
    }

    /// Construct the order complex from its maximal chains.
    ///
    /// Vertex ids are retained. Deterministic limits bound both maximal-chain
    /// enumeration and the downward closure materialized by the existing
    /// simplicial-complex representation.
    pub fn order_complex(&self) -> Result<SimplicialComplex, PosetError> {
        let facets = maximal_chains(self)?;
        check_order_complex_simplex_limit(&facets)?;
        Ok(SimplicialComplex::from_facets(&facets))
    }

    fn validate_element(&self, element: VertexId) -> Result<(), PosetError> {
        if element as usize >= self.order() {
            return Err(PosetError::ElementOutOfRange {
                element,
                order: self.order(),
            });
        }
        Ok(())
    }

    fn cover_pairs(&self) -> Vec<(VertexId, VertexId)> {
        self.covers
            .edges()
            .iter()
            .map(|edge| (edge.source, edge.target))
            .collect()
    }

    fn relation_row(&self, lower: usize) -> &[u64] {
        &self.leq[lower * self.words_per_row..(lower + 1) * self.words_per_row]
    }

    fn has_meet(&self, left: usize, right: usize) -> bool {
        (0..self.order()).any(|candidate| {
            has_relation(&self.leq, self.words_per_row, candidate, left)
                && has_relation(&self.leq, self.words_per_row, candidate, right)
                && (0..self.order()).all(|other| {
                    !has_relation(&self.leq, self.words_per_row, other, left)
                        || !has_relation(&self.leq, self.words_per_row, other, right)
                        || has_relation(&self.leq, self.words_per_row, other, candidate)
                })
        })
    }

    fn has_join(&self, left: usize, right: usize) -> bool {
        (0..self.order()).any(|candidate| {
            has_relation(&self.leq, self.words_per_row, left, candidate)
                && has_relation(&self.leq, self.words_per_row, right, candidate)
                && (0..self.order()).all(|other| {
                    !has_relation(&self.leq, self.words_per_row, left, other)
                        || !has_relation(&self.leq, self.words_per_row, right, other)
                        || has_relation(&self.leq, self.words_per_row, candidate, other)
                })
        })
    }
}

struct PosetCanonicalSearch<'a> {
    poset: &'a Poset,
    lower_covers: Vec<Vec<VertexId>>,
    symmetry_cell: Vec<usize>,
    words: usize,
    nodes: usize,
    best_key: Option<Vec<Vec<u64>>>,
    best_order: Option<Vec<VertexId>>,
    generators: Vec<Permutation>,
    group: PermutationGroup,
}

impl<'a> PosetCanonicalSearch<'a> {
    fn new(poset: &'a Poset) -> Result<Self, PosetError> {
        let mut lower_covers = vec![Vec::new(); poset.order()];
        for edge in poset.covers.edges() {
            lower_covers[edge.target as usize].push(edge.source);
        }
        let cells = refined_symmetry_partition(poset);
        let mut symmetry_cell = vec![0; poset.order()];
        for (cell_index, cell) in cells.iter().enumerate() {
            for &vertex in cell {
                symmetry_cell[vertex as usize] = cell_index;
            }
        }
        Ok(Self {
            poset,
            lower_covers,
            symmetry_cell,
            words: poset.order().div_ceil(u64::BITS as usize),
            nodes: 0,
            best_key: None,
            best_order: None,
            generators: Vec::new(),
            group: PermutationGroup::from_generators(poset.order(), &[])?,
        })
    }

    fn run(mut self) -> Result<Self, PosetError> {
        let mut order = Vec::with_capacity(self.poset.order());
        let mut key = Vec::with_capacity(self.poset.order());
        let mut placed = vec![false; self.poset.order()];
        self.visit(&mut order, &mut key, &mut placed)?;
        debug_assert!(self.best_order.is_some());
        Ok(self)
    }

    fn visit(
        &mut self,
        order: &mut Vec<VertexId>,
        key: &mut Vec<Vec<u64>>,
        placed: &mut [bool],
    ) -> Result<(), PosetError> {
        self.nodes += 1;
        if self.nodes > CANONICAL_SEARCH_NODE_CAP {
            return Err(PosetError::CanonicalSearchBudgetExceeded {
                cap: CANONICAL_SEARCH_NODE_CAP,
            });
        }

        if order.len() == self.poset.order() {
            match self
                .best_key
                .as_ref()
                .map(|best| compare_incidence_keys(key, best))
            {
                None | Some(Ordering::Less) => {
                    self.best_key = Some(key.clone());
                    self.best_order = Some(order.clone());
                }
                Some(Ordering::Equal) => {
                    let best = self
                        .best_order
                        .as_ref()
                        .expect("a best key always has a labeling");
                    let best_labeling = labeling_from_order(best)?;
                    let labeling = labeling_from_order(order)?;
                    let automorphism = automorphism_between(&best_labeling, &labeling)?;
                    self.add_verified_automorphism(automorphism)?;
                }
                Some(Ordering::Greater) => {}
            }
            return Ok(());
        }

        // A minimizing labeling is a linear extension. Once labels `0..k`
        // have been fixed, the row for any minimal remaining element is known:
        // it is its already-labeled strict downset plus the common self bit k.
        let mut candidates = Vec::new();
        for candidate in 0..self.poset.order() {
            if placed[candidate]
                || !self.lower_covers[candidate]
                    .iter()
                    .all(|&lower| placed[lower as usize])
            {
                continue;
            }
            let candidate = candidate as VertexId;
            let mut mask = vec![0; self.words];
            for (label, &lower) in order.iter().enumerate() {
                if has_relation(
                    &self.poset.leq,
                    self.poset.words_per_row,
                    lower as usize,
                    candidate as usize,
                ) {
                    mask[label / u64::BITS as usize] |= 1u64 << (label % u64::BITS as usize);
                }
            }
            candidates.push((candidate, mask));
        }

        let least_mask = candidates
            .iter()
            .map(|(_, mask)| mask)
            .min_by(|left, right| compare_incidence_masks(left, right))
            .expect("a nonempty finite poset has a minimal remaining element")
            .clone();
        candidates.retain(|(_, mask)| compare_incidence_masks(mask, &least_mask).is_eq());
        // Stable refinement cells are necessary automorphism invariants. Their
        // ordering is not imposed on the canonical form; it merely keeps the
        // most plausible equivalent branches adjacent so generators are found
        // early and can prune later branches.
        candidates
            .sort_by_key(|(candidate, _)| (self.symmetry_cell[*candidate as usize], *candidate));

        // Choices in one orbit under the pointwise stabilizer of the prefix
        // have isomorphic continuation trees and the same incidence sequence.
        let mut explored = Vec::new();
        for (candidate, mask) in candidates {
            // Recursive exploration may have enlarged `self.group`, so compute
            // this stabilizer for each next candidate rather than once per node.
            let stabilizer = self.group.pointwise_stabilizer(order)?;
            let mut already_covered = false;
            for &representative in &explored {
                if stabilizer
                    .orbit(representative)
                    .map_err(PosetError::Group)?
                    .contains(&candidate)
                {
                    already_covered = true;
                    break;
                }
            }
            if already_covered {
                continue;
            }

            placed[candidate as usize] = true;
            order.push(candidate);
            key.push(mask);
            self.visit(order, key, placed)?;
            key.pop();
            order.pop();
            placed[candidate as usize] = false;
            explored.push(candidate);
        }
        Ok(())
    }

    fn add_verified_automorphism(&mut self, automorphism: Permutation) -> Result<(), PosetError> {
        if automorphism.is_identity() {
            return Ok(());
        }
        if !is_poset_automorphism(self.poset, &automorphism) {
            return Err(PosetError::AutomorphismVerificationFailed);
        }
        if self.group.contains(&automorphism)? {
            return Ok(());
        }
        self.generators.push(automorphism);
        self.group = PermutationGroup::from_generators(self.poset.order(), &self.generators)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct InitialSymmetryInvariant {
    strict_lower: usize,
    strict_upper: usize,
    lower_covers: usize,
    upper_covers: usize,
}

fn refined_symmetry_partition(poset: &Poset) -> Vec<Vec<VertexId>> {
    let strict_lower = poset.strict_lower_counts();
    let strict_upper = poset.strict_upper_counts();
    let mut lower_covers = vec![0usize; poset.order()];
    let mut upper_covers = vec![0usize; poset.order()];
    for edge in poset.covers.edges() {
        upper_covers[edge.source as usize] += 1;
        lower_covers[edge.target as usize] += 1;
    }
    let mut initial = BTreeMap::<InitialSymmetryInvariant, Vec<VertexId>>::new();
    for vertex in 0..poset.order() {
        initial
            .entry(InitialSymmetryInvariant {
                strict_lower: strict_lower[vertex],
                strict_upper: strict_upper[vertex],
                lower_covers: lower_covers[vertex],
                upper_covers: upper_covers[vertex],
            })
            .or_default()
            .push(vertex as VertexId);
    }
    let mut partition = initial.into_values().collect::<Vec<_>>();

    loop {
        let mut cell_of = vec![0usize; poset.order()];
        for (cell_index, cell) in partition.iter().enumerate() {
            for &vertex in cell {
                cell_of[vertex as usize] = cell_index;
            }
        }
        let cells = partition.len();
        let mut signatures = vec![vec![0usize; cells * 4]; poset.order()];
        for edge in poset.covers.edges() {
            signatures[edge.source as usize][cell_of[edge.target as usize]] += 1;
            signatures[edge.target as usize][cells + cell_of[edge.source as usize]] += 1;
        }
        for lower in 0..poset.order() {
            for upper in 0..poset.order() {
                if lower == upper || !has_relation(&poset.leq, poset.words_per_row, lower, upper) {
                    continue;
                }
                signatures[lower][2 * cells + cell_of[upper]] += 1;
                signatures[upper][3 * cells + cell_of[lower]] += 1;
            }
        }

        let mut refined = Vec::with_capacity(partition.len());
        for cell in &partition {
            let mut splits = BTreeMap::<Vec<usize>, Vec<VertexId>>::new();
            for &vertex in cell {
                splits
                    .entry(signatures[vertex as usize].clone())
                    .or_default()
                    .push(vertex);
            }
            refined.extend(splits.into_values());
        }
        if refined.len() == partition.len() {
            return refined;
        }
        partition = refined;
    }
}

fn compare_incidence_masks(left: &[u64], right: &[u64]) -> Ordering {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .find_map(|(left, right)| (left != right).then(|| left.cmp(right)))
        .unwrap_or(Ordering::Equal)
}

fn compare_incidence_keys(left: &[Vec<u64>], right: &[Vec<u64>]) -> Ordering {
    left.iter()
        .zip(right)
        .find_map(|(left, right)| {
            let ordering = compare_incidence_masks(left, right);
            (!ordering.is_eq()).then_some(ordering)
        })
        .unwrap_or(Ordering::Equal)
}

fn labeling_from_order(order: &[VertexId]) -> Result<Permutation, PosetError> {
    let mut images = vec![0; order.len()];
    for (canonical, &original) in order.iter().enumerate() {
        images[original as usize] = canonical as Point;
    }
    Ok(Permutation::from_images(&images)?)
}

fn automorphism_between(
    best: &Permutation,
    other: &Permutation,
) -> Result<Permutation, PosetError> {
    let best_inverse = best.inverse();
    let images = other
        .images()
        .iter()
        .map(|&canonical| best_inverse.apply(canonical))
        .collect::<Vec<_>>();
    Ok(Permutation::from_images(&images)?)
}

fn relabeled_cover_pairs(poset: &Poset, labeling: &Permutation) -> Vec<(VertexId, VertexId)> {
    let mut covers = poset
        .covers
        .edges()
        .iter()
        .map(|edge| {
            (
                labeling.images()[edge.source as usize],
                labeling.images()[edge.target as usize],
            )
        })
        .collect::<Vec<_>>();
    covers.sort_unstable();
    covers
}

fn is_poset_automorphism(poset: &Poset, permutation: &Permutation) -> bool {
    if permutation.degree() != poset.order() {
        return false;
    }
    for lower in 0..poset.order() {
        for upper in 0..poset.order() {
            if has_relation(&poset.leq, poset.words_per_row, lower, upper)
                != has_relation(
                    &poset.leq,
                    poset.words_per_row,
                    permutation.apply(lower as Point) as usize,
                    permutation.apply(upper as Point) as usize,
                )
            {
                return false;
            }
        }
    }
    true
}

fn canonical_incidence_sequence(poset: &Poset) -> Vec<Integer> {
    let powers = (0..poset.order())
        .map(|exponent| Integer::one() << exponent as u32)
        .collect::<Vec<_>>();
    (0..poset.order())
        .map(|upper| {
            let mut value = Integer::zero();
            for (lower, power) in powers.iter().enumerate() {
                if has_relation(&poset.leq, poset.words_per_row, lower, upper) {
                    value += power;
                }
            }
            value
        })
        .collect()
}

fn maximum_strict_order_matching(poset: &Poset) -> usize {
    fn augment(
        poset: &Poset,
        left: usize,
        seen_right: &mut [bool],
        matched_left: &mut [Option<usize>],
    ) -> bool {
        for right in 0..poset.order() {
            if left == right
                || seen_right[right]
                || !has_relation(&poset.leq, poset.words_per_row, left, right)
            {
                continue;
            }
            seen_right[right] = true;
            if matched_left[right]
                .is_none_or(|previous| augment(poset, previous, seen_right, matched_left))
            {
                matched_left[right] = Some(left);
                return true;
            }
        }
        false
    }

    let mut matched_left = vec![None; poset.order()];
    let mut size = 0;
    for left in 0..poset.order() {
        let mut seen_right = vec![false; poset.order()];
        if augment(poset, left, &mut seen_right, &mut matched_left) {
            size += 1;
        }
    }
    size
}

fn beat_points_with_active(poset: &Poset, active: &[bool]) -> Vec<BeatPoint> {
    let mut result = Vec::new();
    for element in 0..poset.order() {
        if !active[element] {
            continue;
        }
        let strict_lower = (0..poset.order())
            .filter(|&lower| {
                active[lower]
                    && lower != element
                    && has_relation(&poset.leq, poset.words_per_row, lower, element)
            })
            .collect::<Vec<_>>();
        if let Some(&witness) = strict_lower.iter().find(|&&candidate| {
            strict_lower
                .iter()
                .all(|&lower| has_relation(&poset.leq, poset.words_per_row, lower, candidate))
        }) {
            result.push(BeatPoint {
                element: element as VertexId,
                kind: BeatPointKind::Down,
                witness: witness as VertexId,
            });
        }

        let strict_upper = (0..poset.order())
            .filter(|&upper| {
                active[upper]
                    && upper != element
                    && has_relation(&poset.leq, poset.words_per_row, element, upper)
            })
            .collect::<Vec<_>>();
        if let Some(&witness) = strict_upper.iter().find(|&&candidate| {
            strict_upper
                .iter()
                .all(|&upper| has_relation(&poset.leq, poset.words_per_row, candidate, upper))
        }) {
            result.push(BeatPoint {
                element: element as VertexId,
                kind: BeatPointKind::Up,
                witness: witness as VertexId,
            });
        }
    }
    result
}

fn beat_point_is_valid(poset: &Poset, active: &[bool], beat_point: BeatPoint) -> bool {
    let element = beat_point.element as usize;
    let witness = beat_point.witness as usize;
    if element >= poset.order()
        || witness >= poset.order()
        || element == witness
        || !active[element]
        || !active[witness]
    {
        return false;
    }
    match beat_point.kind {
        BeatPointKind::Down => {
            has_relation(&poset.leq, poset.words_per_row, witness, element)
                && (0..poset.order()).all(|lower| {
                    !active[lower]
                        || lower == element
                        || !has_relation(&poset.leq, poset.words_per_row, lower, element)
                        || has_relation(&poset.leq, poset.words_per_row, lower, witness)
                })
        }
        BeatPointKind::Up => {
            has_relation(&poset.leq, poset.words_per_row, element, witness)
                && (0..poset.order()).all(|upper| {
                    !active[upper]
                        || upper == element
                        || !has_relation(&poset.leq, poset.words_per_row, element, upper)
                        || has_relation(&poset.leq, poset.words_per_row, witness, upper)
                })
        }
    }
}

fn induced_poset(poset: &Poset, retained: &[VertexId]) -> Result<Poset, PosetError> {
    let mut relations = Vec::new();
    for (new_lower, &old_lower) in retained.iter().enumerate() {
        for (new_upper, &old_upper) in retained.iter().enumerate() {
            if old_lower != old_upper
                && has_relation(
                    &poset.leq,
                    poset.words_per_row,
                    old_lower as usize,
                    old_upper as usize,
                )
            {
                relations.push((new_lower as VertexId, new_upper as VertexId));
            }
        }
    }
    Poset::new(retained.len(), &relations)
}

fn maximal_chains(poset: &Poset) -> Result<Vec<Vec<u32>>, PosetError> {
    fn visit(
        vertex: VertexId,
        upper_covers: &[Vec<VertexId>],
        path: &mut Vec<u32>,
        facets: &mut Vec<Vec<u32>>,
    ) -> Result<(), PosetError> {
        path.push(vertex);
        let prefix_subsets = 1usize.checked_shl(path.len() as u32).map(|n| n - 1);
        if prefix_subsets.is_none_or(|count| count > MAX_ORDER_COMPLEX_SIMPLICES) {
            path.pop();
            return Err(PosetError::OrderComplexSimplexLimitExceeded {
                cap: MAX_ORDER_COMPLEX_SIMPLICES,
            });
        }
        if upper_covers[vertex as usize].is_empty() {
            if facets.len() == MAX_ORDER_COMPLEX_FACETS {
                path.pop();
                return Err(PosetError::OrderComplexFacetLimitExceeded {
                    cap: MAX_ORDER_COMPLEX_FACETS,
                });
            }
            facets.push(path.clone());
        } else {
            for &upper in &upper_covers[vertex as usize] {
                visit(upper, upper_covers, path, facets)?;
            }
        }
        path.pop();
        Ok(())
    }

    let mut upper_covers = vec![Vec::new(); poset.order()];
    for edge in poset.covers.edges() {
        upper_covers[edge.source as usize].push(edge.target);
    }
    for successors in &mut upper_covers {
        successors.sort_unstable();
    }
    let mut facets = Vec::new();
    let mut path = Vec::new();
    for minimal in poset.minimal_elements() {
        visit(minimal, &upper_covers, &mut path, &mut facets)?;
    }
    facets.sort();
    Ok(facets)
}

fn check_order_complex_simplex_limit(facets: &[Vec<u32>]) -> Result<(), PosetError> {
    let mut simplices = BTreeSet::new();
    for facet in facets {
        let subset_count = 1usize.checked_shl(facet.len() as u32).ok_or(
            PosetError::OrderComplexSimplexLimitExceeded {
                cap: MAX_ORDER_COMPLEX_SIMPLICES,
            },
        )?;
        for mask in 1..subset_count {
            let simplex = facet
                .iter()
                .enumerate()
                .filter_map(|(index, &vertex)| ((mask >> index) & 1 == 1).then_some(vertex))
                .collect::<Vec<_>>();
            simplices.insert(simplex);
            if simplices.len() > MAX_ORDER_COMPLEX_SIMPLICES {
                return Err(PosetError::OrderComplexSimplexLimitExceeded {
                    cap: MAX_ORDER_COMPLEX_SIMPLICES,
                });
            }
        }
    }
    Ok(())
}

fn set_relation(bits: &mut [u64], words_per_row: usize, lower: usize, upper: usize) {
    bits[lower * words_per_row + upper / u64::BITS as usize] |=
        1u64 << (upper % u64::BITS as usize);
}

fn has_relation(bits: &[u64], words_per_row: usize, lower: usize, upper: usize) -> bool {
    bits[lower * words_per_row + upper / u64::BITS as usize]
        & (1u64 << (upper % u64::BITS as usize))
        != 0
}

impl PartialEq for Poset {
    fn eq(&self, other: &Self) -> bool {
        self.covers == other.covers
    }
}

impl Eq for Poset {}

impl fmt::Display for Poset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Poset(order={}, covers={})",
            self.order(),
            self.covers.edge_count()
        )
    }
}

impl Serialize for Poset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PosetRepr {
            order: self.order(),
            covers: self.cover_pairs(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Poset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PosetRepr::deserialize(deserializer)?;
        Self::new(repr.order, &repr.covers).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boolean_lattice() -> Poset {
        Poset::new(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap()
    }

    fn incidence_tuple(poset: &Poset, order: &[VertexId]) -> Vec<u64> {
        assert!(poset.order() <= u64::BITS as usize);
        let mut labels = vec![0; order.len()];
        for (label, &element) in order.iter().enumerate() {
            labels[element as usize] = label;
        }
        order
            .iter()
            .map(|&upper| {
                order
                    .iter()
                    .filter(|&&lower| poset.leq(lower, upper).unwrap())
                    .map(|&lower| 1u64 << labels[lower as usize])
                    .sum()
            })
            .collect()
    }

    fn minimum_incidence_tuple(poset: &Poset) -> Vec<u64> {
        fn visit(
            poset: &Poset,
            order: &mut Vec<VertexId>,
            remaining: &mut Vec<VertexId>,
            best: &mut Option<Vec<u64>>,
        ) {
            if remaining.is_empty() {
                let key = incidence_tuple(poset, order);
                if best.as_ref().is_none_or(|current| key < *current) {
                    *best = Some(key);
                }
                return;
            }
            for index in (0..remaining.len()).rev() {
                let element = remaining.remove(index);
                order.push(element);
                visit(poset, order, remaining, best);
                order.pop();
                remaining.insert(index, element);
            }
        }

        let mut remaining = (0..poset.order() as VertexId).collect();
        let mut best = None;
        visit(poset, &mut Vec::new(), &mut remaining, &mut best);
        best.unwrap()
    }

    fn all_permutations(order: usize) -> Vec<Permutation> {
        fn visit(
            order: usize,
            images: &mut Vec<Point>,
            used: &mut [bool],
            result: &mut Vec<Permutation>,
        ) {
            if images.len() == order {
                result.push(Permutation::from_images(images).unwrap());
                return;
            }
            for image in 0..order {
                if used[image] {
                    continue;
                }
                used[image] = true;
                images.push(image as Point);
                visit(order, images, used, result);
                images.pop();
                used[image] = false;
            }
        }

        let mut result = Vec::new();
        visit(
            order,
            &mut Vec::with_capacity(order),
            &mut vec![false; order],
            &mut result,
        );
        result
    }

    fn square_face_poset() -> Poset {
        Poset::new(
            8,
            &[
                (0, 4),
                (0, 5),
                (1, 4),
                (1, 6),
                (2, 5),
                (2, 7),
                (3, 6),
                (3, 7),
            ],
        )
        .unwrap()
    }

    fn subset_is_chain(poset: &Poset, subset: usize) -> bool {
        (0..poset.order()).all(|left| {
            (left + 1..poset.order()).all(|right| {
                subset & (1 << left) == 0
                    || subset & (1 << right) == 0
                    || has_relation(&poset.leq, poset.words_per_row, left, right)
                    || has_relation(&poset.leq, poset.words_per_row, right, left)
            })
        })
    }

    fn subset_is_antichain(poset: &Poset, subset: usize) -> bool {
        (0..poset.order()).all(|left| {
            (left + 1..poset.order()).all(|right| {
                subset & (1 << left) == 0
                    || subset & (1 << right) == 0
                    || (!has_relation(&poset.leq, poset.words_per_row, left, right)
                        && !has_relation(&poset.leq, poset.words_per_row, right, left))
            })
        })
    }

    fn brute_is_lattice(poset: &Poset) -> bool {
        if poset.order() == 0 {
            return false;
        }
        for left in 0..poset.order() {
            for right in 0..poset.order() {
                let common_lower = (0..poset.order())
                    .filter(|&candidate| {
                        has_relation(&poset.leq, poset.words_per_row, candidate, left)
                            && has_relation(&poset.leq, poset.words_per_row, candidate, right)
                    })
                    .collect::<Vec<_>>();
                let maximal_lower = common_lower
                    .iter()
                    .filter(|&&candidate| {
                        common_lower.iter().all(|&other| {
                            other == candidate
                                || !has_relation(&poset.leq, poset.words_per_row, candidate, other)
                        })
                    })
                    .count();
                let common_upper = (0..poset.order())
                    .filter(|&candidate| {
                        has_relation(&poset.leq, poset.words_per_row, left, candidate)
                            && has_relation(&poset.leq, poset.words_per_row, right, candidate)
                    })
                    .collect::<Vec<_>>();
                let minimal_upper = common_upper
                    .iter()
                    .filter(|&&candidate| {
                        common_upper.iter().all(|&other| {
                            other == candidate
                                || !has_relation(&poset.leq, poset.words_per_row, other, candidate)
                        })
                    })
                    .count();
                if maximal_lower != 1 || minimal_upper != 1 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn construction_normalizes_to_the_hasse_diagram() {
        let reduced = Poset::new(3, &[(0, 1), (1, 2)]).unwrap();
        let redundant = Poset::new(3, &[(0, 2), (1, 2), (0, 1), (0, 2)]).unwrap();

        assert_eq!(reduced, redundant);
        assert_eq!(reduced.cover_pairs(), vec![(0, 1), (1, 2)]);
        assert!(reduced.leq(0, 2).unwrap());
        assert!(!reduced.leq(2, 0).unwrap());
    }

    #[test]
    fn canonical_form_uses_the_minimum_principal_downset_tuple() {
        let vee = Poset::new(3, &[(0, 1), (0, 2)]).unwrap();
        let chain = Poset::new(3, &[(0, 1), (1, 2)]).unwrap();
        let diamond = boolean_lattice();

        assert_eq!(incidence_tuple(&vee, &[0, 1, 2]), vec![1, 3, 5]);
        assert_eq!(incidence_tuple(&chain, &[0, 1, 2]), vec![1, 3, 7]);
        assert_eq!(incidence_tuple(&diamond, &[0, 1, 2, 3]), vec![1, 3, 5, 15]);

        let relabeled = Poset::new(4, &[(3, 1), (3, 2), (1, 0), (2, 0)]).unwrap();
        assert_eq!(relabeled.canonical_form().unwrap(), diamond);
        assert_eq!(
            relabeled
                .canonical_form()
                .unwrap()
                .canonical_form()
                .unwrap(),
            diamond
        );
    }

    #[test]
    fn all_naturally_labeled_orders_on_four_points_match_brute_force() {
        let possible = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for mask in 0u8..64 {
            let relations: Vec<_> = possible
                .iter()
                .enumerate()
                .filter_map(|(bit, &relation)| ((mask >> bit) & 1 == 1).then_some(relation))
                .collect();
            let poset = Poset::new(4, &relations).unwrap();

            let mut expected = [[false; 4]; 4];
            for (element, row) in expected.iter_mut().enumerate() {
                row[element] = true;
            }
            for &(lower, upper) in &relations {
                expected[lower as usize][upper as usize] = true;
            }
            for middle in 0..4 {
                for lower in 0..4 {
                    for upper in 0..4 {
                        expected[lower][upper] |=
                            expected[lower][middle] && expected[middle][upper];
                    }
                }
            }

            for lower in 0..4 {
                for upper in 0..4 {
                    assert_eq!(
                        poset.leq(lower, upper).unwrap(),
                        expected[lower as usize][upper as usize],
                        "mask={mask}, pair=({lower}, {upper})"
                    );
                }
            }
            let expected_covers: Vec<_> = possible
                .iter()
                .copied()
                .filter(|&(lower, upper)| {
                    expected[lower as usize][upper as usize]
                        && !(0..4).any(|middle| {
                            middle != lower as usize
                                && middle != upper as usize
                                && expected[lower as usize][middle]
                                && expected[middle][upper as usize]
                        })
                })
                .collect();
            assert_eq!(poset.cover_pairs(), expected_covers, "mask={mask}");

            let canonical = poset.canonical_form().unwrap();
            assert_eq!(
                incidence_tuple(&canonical, &[0, 1, 2, 3]),
                minimum_incidence_tuple(&poset),
                "canonical form differs from exhaustive relabeling for mask={mask}"
            );
            assert!(canonical
                .cover_pairs()
                .iter()
                .all(|&(lower, upper)| lower < upper));
            assert_eq!(canonical.canonical_form().unwrap(), canonical);
        }
    }

    #[test]
    fn canonical_forms_and_automorphisms_match_every_permutation_through_order_five() {
        for order in 0..=5 {
            let possible = (0..order as VertexId)
                .flat_map(|lower| ((lower + 1)..order as VertexId).map(move |upper| (lower, upper)))
                .collect::<Vec<_>>();
            let permutations = all_permutations(order);
            for mask in 0usize..(1usize << possible.len()) {
                let relations = possible
                    .iter()
                    .enumerate()
                    .filter_map(|(bit, &relation)| ((mask >> bit) & 1 == 1).then_some(relation))
                    .collect::<Vec<_>>();
                let poset = Poset::new(order, &relations).unwrap();
                let canonical = poset.canonical_label().unwrap();
                assert!(canonical.verify(&poset), "order={order}, mask={mask}");
                assert_eq!(
                    incidence_tuple(
                        canonical.poset(),
                        &(0..order as VertexId).collect::<Vec<_>>()
                    ),
                    minimum_incidence_tuple(&poset),
                    "order={order}, mask={mask}"
                );
                for permutation in &permutations {
                    assert_eq!(
                        canonical
                            .automorphism_group()
                            .contains(permutation)
                            .unwrap(),
                        is_poset_automorphism(&poset, permutation),
                        "order={order}, mask={mask}, permutation={:?}",
                        permutation.images()
                    );
                }

                let strict_lower = (0..order)
                    .map(|upper| {
                        (0..order)
                            .filter(|&lower| {
                                lower != upper
                                    && has_relation(&poset.leq, poset.words_per_row, lower, upper)
                            })
                            .count()
                    })
                    .collect::<Vec<_>>();
                let strict_upper = (0..order)
                    .map(|lower| {
                        (0..order)
                            .filter(|&upper| {
                                lower != upper
                                    && has_relation(&poset.leq, poset.words_per_row, lower, upper)
                            })
                            .count()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(poset.strict_lower_counts(), strict_lower);
                assert_eq!(poset.strict_upper_counts(), strict_upper);
                assert_eq!(
                    poset.comparable_pair_count(),
                    strict_upper.iter().sum::<usize>()
                );
                let subsets = 0usize..(1usize << order);
                assert_eq!(
                    poset.height(),
                    subsets
                        .clone()
                        .filter(|&subset| subset_is_chain(&poset, subset))
                        .map(|subset| subset.count_ones() as usize)
                        .max()
                        .unwrap_or(0)
                );
                assert_eq!(
                    poset.width(),
                    subsets
                        .clone()
                        .filter(|&subset| subset_is_antichain(&poset, subset))
                        .map(|subset| subset.count_ones() as usize)
                        .max()
                        .unwrap_or(0)
                );
                assert_eq!(poset.is_lattice(), brute_is_lattice(&poset));

                let complex = poset.order_complex().unwrap();
                let actual_chains = (0..order)
                    .flat_map(|dimension| complex.simplices(dimension))
                    .map(|simplex| simplex.vertices().to_vec())
                    .collect::<BTreeSet<_>>();
                let expected_chains = subsets
                    .clone()
                    .filter(|&subset| subset != 0 && subset_is_chain(&poset, subset))
                    .map(|subset| {
                        (0..order as VertexId)
                            .filter(|&vertex| subset & (1 << vertex) != 0)
                            .collect::<Vec<_>>()
                    })
                    .collect::<BTreeSet<_>>();
                assert_eq!(actual_chains, expected_chains);

                let core = poset.core().unwrap();
                assert!(core.verify(&poset));
                assert!(core.core().is_core());
            }
        }
    }

    #[test]
    fn structural_invariants_cover_empty_lattice_and_circle_models() {
        let empty = Poset::new(0, &[]).unwrap();
        assert_eq!(empty.height(), 0);
        assert_eq!(empty.width(), 0);
        assert!(!empty.is_lattice());
        assert!(empty.connected_components().is_empty());

        let diamond = boolean_lattice();
        assert_eq!(diamond.strict_lower_counts(), vec![0, 1, 1, 3]);
        assert_eq!(diamond.strict_upper_counts(), vec![3, 1, 1, 0]);
        assert_eq!(diamond.minimal_elements(), vec![0]);
        assert_eq!(diamond.maximal_elements(), vec![3]);
        assert_eq!(diamond.comparable_pair_count(), 5);
        assert_eq!(diamond.connected_components(), vec![vec![0, 1, 2, 3]]);
        assert_eq!(diamond.height(), 3);
        assert_eq!(diamond.width(), 2);
        assert!(diamond.is_lattice());
        assert!(diamond.is_self_dual().unwrap());

        let square = square_face_poset();
        assert_eq!(square.strict_lower_counts(), vec![0, 0, 0, 0, 2, 2, 2, 2]);
        assert_eq!(square.strict_upper_counts(), vec![2, 2, 2, 2, 0, 0, 0, 0]);
        assert_eq!(square.comparable_pair_count(), 8);
        assert_eq!(square.height(), 2);
        assert_eq!(square.width(), 4);
        assert!(!square.is_lattice());
        assert!(square.is_self_dual().unwrap());
    }

    #[test]
    fn beat_point_reduction_is_replayable_and_dual_invariant() {
        let chain = Poset::new(3, &[(0, 1), (1, 2)]).unwrap();
        assert_eq!(
            chain.beat_points(),
            vec![
                BeatPoint {
                    element: 0,
                    kind: BeatPointKind::Up,
                    witness: 1,
                },
                BeatPoint {
                    element: 1,
                    kind: BeatPointKind::Down,
                    witness: 0,
                },
                BeatPoint {
                    element: 1,
                    kind: BeatPointKind::Up,
                    witness: 2,
                },
                BeatPoint {
                    element: 2,
                    kind: BeatPointKind::Down,
                    witness: 1,
                },
            ]
        );
        let reduction = chain.core().unwrap();
        assert!(reduction.verify(&chain));
        assert_eq!(reduction.core().order(), 1);
        assert!(reduction.core().is_core());
        assert_eq!(reduction.core().core().unwrap().core(), reduction.core());
        let opposite_core = reduction
            .core()
            .opposite()
            .unwrap()
            .canonical_form()
            .unwrap();
        assert_eq!(
            chain.opposite().unwrap().core().unwrap().core(),
            &opposite_core
        );

        let square = square_face_poset();
        assert!(square.is_core());
        assert!(square.core().unwrap().verify(&square));
        assert_eq!(
            square.core().unwrap().core(),
            &square.canonical_form().unwrap()
        );
    }

    #[test]
    fn order_complex_stores_maximal_chains_and_circle_homology() {
        let empty = Poset::new(0, &[]).unwrap().order_complex().unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.dimension(), -1);

        let square = square_face_poset();
        let complex = square.order_complex().unwrap();
        assert_eq!(
            complex
                .facets()
                .iter()
                .map(|simplex| simplex.vertices().to_vec())
                .collect::<Vec<_>>(),
            vec![
                vec![0, 4],
                vec![0, 5],
                vec![1, 4],
                vec![1, 6],
                vec![2, 5],
                vec![2, 7],
                vec![3, 6],
                vec![3, 7],
            ]
        );
        assert_eq!((complex.count(0), complex.count(1)), (8, 8));
        assert_eq!(complex.euler_characteristic(), 0);
        assert_eq!(complex.homology_invariants(0).unwrap().free_rank(), 1);
        assert_eq!(complex.homology_invariants(1).unwrap().free_rank(), 1);
    }

    #[test]
    fn core_reduction_preserves_integral_homology_and_verifies() {
        let chain = Poset::new(
            8,
            &(0..7).map(|lower| (lower, lower + 1)).collect::<Vec<_>>(),
        )
        .unwrap();
        let source_complex = chain.order_complex().unwrap();
        let reduction = chain.core().unwrap();
        assert!(reduction.verify(&chain));
        assert_eq!(reduction.core().order(), 1);

        let via_reduction = reduction.graded_homology_invariants().unwrap();
        let via_poset = chain.graded_homology_invariants().unwrap();
        assert_eq!(via_poset, via_reduction);
        assert_eq!(via_poset.len(), 1);
        assert_eq!(via_poset[0], source_complex.homology_invariants(0).unwrap());
        for degree in 1..=source_complex.dimension() as usize {
            let group = source_complex.homology_invariants(degree).unwrap();
            assert_eq!(group.free_rank(), 0);
            assert!(group.torsion_coefficients().is_empty());
        }

        let square = square_face_poset();
        assert!(square.is_core());
        assert_eq!(
            square.graded_homology_invariants().unwrap(),
            square
                .order_complex()
                .unwrap()
                .graded_homology_invariants()
                .unwrap()
        );
    }

    #[test]
    fn representative_order_ten_symmetry_models_stay_within_limits() {
        let antichain = Poset::new(10, &[]).unwrap();
        assert_eq!(
            antichain
                .canonical_label()
                .unwrap()
                .automorphism_group()
                .order()
                .to_decimal(),
            "3628800"
        );
        let layered_relations = (0..5)
            .flat_map(|lower| (5..10).map(move |upper| (lower, upper)))
            .collect::<Vec<_>>();
        let layered = Poset::new(10, &layered_relations).unwrap();
        assert_eq!(
            layered
                .canonical_label()
                .unwrap()
                .automorphism_group()
                .order()
                .to_decimal(),
            "14400"
        );
    }

    #[test]
    fn packed_relation_rows_cross_word_boundaries() {
        let poset = Poset::new(65, &[(64, 0), (0, 63), (64, 63)]).unwrap();

        assert!(poset.leq(64, 63).unwrap());
        assert!(!poset.leq(63, 64).unwrap());
        assert_eq!(poset.cover_pairs(), vec![(0, 63), (64, 0)]);
    }

    #[test]
    fn canonical_incidence_masks_cross_word_boundaries() {
        let descending: Vec<(VertexId, VertexId)> =
            (1..65).map(|element| (element, element - 1)).collect();
        let ascending: Vec<(VertexId, VertexId)> =
            (1..65).map(|element| (element - 1, element)).collect();

        assert_eq!(
            Poset::new(65, &descending)
                .unwrap()
                .canonical_form()
                .unwrap(),
            Poset::new(65, &ascending).unwrap()
        );
    }

    #[test]
    fn malformed_relations_fail_at_construction() {
        assert_eq!(
            Poset::new(2, &[(0, 2)]),
            Err(PosetError::RelationOutOfRange {
                lower: 0,
                upper: 2,
                order: 2,
            })
        );
        assert_eq!(
            Poset::new(2, &[(1, 1)]),
            Err(PosetError::ReflexiveRelation { element: 1 })
        );
        assert_eq!(
            Poset::new(3, &[(0, 1), (1, 2), (2, 0)]),
            Err(PosetError::CyclicRelations)
        );
    }

    #[test]
    fn serialization_contains_only_defining_content_and_revalidates() {
        let poset = boolean_lattice();
        let json = serde_json::to_string(&poset).unwrap();
        let rebuilt: Poset = serde_json::from_str(&json).unwrap();

        assert_eq!(rebuilt, poset);
        assert_eq!(json, r#"{"order":4,"covers":[[0,1],[0,2],[1,3],[2,3]]}"#);
        assert!(serde_json::from_str::<Poset>(r#"{"order":2,"covers":[[0,0]]}"#).is_err());
    }

    #[test]
    fn poset_symmetry_is_computed_with_canonical_labeling() {
        let poset = boolean_lattice();
        let canonical = poset.canonical_label().unwrap();
        let group = poset.automorphism_group().unwrap();

        assert!(canonical.verify(&poset));
        assert_eq!(group.order().to_decimal(), "2");
    }

    #[test]
    fn empty_poset_is_valid() {
        let empty = Poset::new(0, &[]).unwrap();
        assert_eq!(empty.order(), 0);
        assert_eq!(empty.covers().edge_count(), 0);
        assert_eq!(empty.canonical_form().unwrap(), empty);
        assert_eq!(
            empty.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
    }
}
