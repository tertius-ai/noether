// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact graph symmetries composed from `graph-core` and `perm-core`.

use core::fmt;
use std::collections::BTreeMap;

use super::{Graph, GraphError, VertexId};
use crate::groups::{BigUInt, GroupError, Permutation, PermutationGroup, Point};

/// Maximum number of individualization/refinement nodes in one symmetry search.
///
/// The cap makes pathological instances fail closed instead of turning a
/// convenience method into an unbounded computation. It is deliberately a
/// node count rather than a wall-clock timeout, so the result is deterministic
/// across machines.
const SEARCH_NODE_CAP: usize = 10_000_000;

/// A failure while computing an exact graph canonical label or automorphism group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphSymmetryError {
    /// The graph cannot be represented by `perm-core`'s `u32` point domain.
    VertexDomainTooLarge {
        /// The graph order that could not be represented.
        order: usize,
    },
    /// The deterministic individualization/refinement search exhausted its cap.
    SearchBudgetExceeded {
        /// Maximum number of search nodes allowed.
        cap: usize,
    },
    /// A generated relabeling could not be represented as a permutation.
    InvalidPermutation(GroupError),
    /// Constructing the returned permutation group failed.
    PermutationGroup(GroupError),
    /// Rebuilding the canonically relabeled graph failed.
    InvalidCanonicalGraph(GraphError),
    /// A proposed automorphism failed exact edge-multiset verification.
    AutomorphismVerificationFailed,
}

impl fmt::Display for GraphSymmetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexDomainTooLarge { order } => write!(
                f,
                "graph order {order} exceeds the permutation point domain"
            ),
            Self::SearchBudgetExceeded { cap } => {
                write!(f, "graph symmetry search exceeded its {cap}-node cap")
            }
            Self::InvalidPermutation(error) => {
                write!(f, "canonical labeling is not a permutation: {error}")
            }
            Self::PermutationGroup(error) => {
                write!(f, "could not construct the automorphism group: {error}")
            }
            Self::InvalidCanonicalGraph(error) => {
                write!(f, "could not construct the canonical graph: {error}")
            }
            Self::AutomorphismVerificationFailed => {
                f.write_str("a proposed graph automorphism failed exact verification")
            }
        }
    }
}

impl std::error::Error for GraphSymmetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPermutation(error) | Self::PermutationGroup(error) => Some(error),
            Self::InvalidCanonicalGraph(error) => Some(error),
            Self::VertexDomainTooLarge { .. }
            | Self::SearchBudgetExceeded { .. }
            | Self::AutomorphismVerificationFailed => None,
        }
    }
}

/// The exact output of graph canonical labeling.
///
/// `labeling` maps each original vertex to its vertex in `graph`. The
/// automorphism group acts on the original vertex set. Both are verified from
/// the complete directed, weighted edge multiset, including loops and parallel
/// edges.
#[derive(Debug, Clone)]
pub struct CanonicalLabel {
    graph: Graph,
    labeling: Permutation,
    automorphism_group: PermutationGroup,
}

impl CanonicalLabel {
    pub(super) fn from_verified_parts(
        graph: Graph,
        labeling: Permutation,
        automorphism_group: PermutationGroup,
        original: &Graph,
    ) -> Self {
        let result = Self {
            graph,
            labeling,
            automorphism_group,
        };
        debug_assert!(result.verify(original));
        result
    }

    /// The canonically relabeled graph.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The permutation mapping original vertex ids to canonical vertex ids.
    #[must_use]
    pub fn labeling(&self) -> &Permutation {
        &self.labeling
    }

    /// The exact automorphism group acting on the original vertex ids.
    #[must_use]
    pub fn automorphism_group(&self) -> &PermutationGroup {
        &self.automorphism_group
    }

    /// Verify that the labeling reconstructs the stored canonical graph.
    pub fn verify(&self, original: &Graph) -> bool {
        if original.order() != self.labeling.degree() {
            return false;
        }
        canonical_edge_key(original, self.labeling.images())
            == canonical_edge_key(&self.graph, &identity_labels(self.graph.order()))
    }
}

/// Exact symmetry operations supplied by Noether across its graph and group kernels.
///
/// `Graph` itself remains the representation owned by `graph-core`. Importing
/// this trait adds the cross-domain object methods without copying or wrapping
/// the graph on the Rust side.
pub trait GraphSymmetry {
    /// Compute an exact canonical labeling and the automorphism group found by
    /// the same individualization/refinement search.
    fn canonical_label(&self) -> Result<CanonicalLabel, GraphSymmetryError>;

    /// Compute the exact automorphism group as a native permutation group.
    fn automorphism_group(&self) -> Result<PermutationGroup, GraphSymmetryError>;
}

impl GraphSymmetry for Graph {
    fn canonical_label(&self) -> Result<CanonicalLabel, GraphSymmetryError> {
        canonicalize(self)
    }

    fn automorphism_group(&self) -> Result<PermutationGroup, GraphSymmetryError> {
        if let Some(simple) = dense_simple_graph(self) {
            if let Some(group) = two_subset_graph_automorphism_group(&simple)? {
                return Ok(group);
            }
        }
        Ok(search_symmetries(self)?.group)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RefinementSignature {
    outgoing: Vec<(usize, i64)>,
    incoming: Vec<(usize, i64)>,
}

type Partition = Vec<Vec<usize>>;
type EdgeKey = Vec<(VertexId, VertexId, i64)>;

/// Dense adjacency keys avoid allocating and sorting an edge list at every
/// leaf for the small simple graphs where individualization/refinement is
/// most useful. Larger or decorated graphs retain the sparse general key.
const DENSE_SIMPLE_KEY_MAX_ORDER: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SearchKey {
    DenseSimple(Vec<u64>),
    General(EdgeKey),
}

struct DenseSimpleGraph {
    neighbours: Vec<Vec<usize>>,
}

/// Maximum number of candidate `(n-1)`-subsets inspected while recognizing a
/// graph on the two-subsets of an `n`-set.
const TWO_SUBSET_RECOGNITION_CAP: usize = 1_000_000;

/// Recognize the Kneser graph `KG(n,2)` or its complement, the triangular
/// graph `T(n)`, and construct its exact automorphism group.
///
/// For `n >= 5`, Erdős--Ko--Rado identifies the `n` maximum intersecting
/// families as the stars of the hidden ground points. We reconstruct those
/// stars as independent sets (Kneser) or cliques (triangular), require every
/// graph vertex to be the unique intersection of two stars, and replay the
/// entire adjacency relation from disjointness/intersection. Once this exact
/// isomorphism is established, the classical `Aut(KG(n,2)) = Aut(T(n)) = S_n`
/// theorem supplies the full group, acting on the recovered two-subsets.
fn two_subset_graph_automorphism_group(
    graph: &DenseSimpleGraph,
) -> Result<Option<PermutationGroup>, GraphSymmetryError> {
    let vertex_count = graph.neighbours.len();
    if vertex_count > 127 {
        return Ok(None);
    }
    let Some(ground_order) = (5_usize..)
        .take_while(|&order| order.saturating_mul(order.saturating_sub(1)) / 2 <= vertex_count)
        .find(|&order| order * (order - 1) / 2 == vertex_count)
    else {
        return Ok(None);
    };
    let expected_degree = (ground_order - 2) * (ground_order - 3) / 2;
    let complement_degree = 2 * (ground_order - 2);
    if graph
        .neighbours
        .iter()
        .any(|row| row.len() != expected_degree && row.len() != complement_degree)
    {
        return Ok(None);
    }
    let star_size = ground_order - 1;
    if binomial_bounded(vertex_count, star_size, TWO_SUBSET_RECOGNITION_CAP).is_none() {
        return Ok(None);
    }

    let mut adjacency = vec![0_u128; vertex_count];
    for (vertex, neighbours) in graph.neighbours.iter().enumerate() {
        for &neighbour in neighbours {
            adjacency[vertex] |= 1_u128 << neighbour;
        }
    }
    let mut independent_stars = Vec::new();
    let mut clique_stars = Vec::new();
    for_fixed_size_subsets(vertex_count, star_size, |subset| {
        let mut independent = true;
        let mut clique = true;
        let mut remaining = subset;
        while remaining != 0 {
            let vertex = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            let others = subset & !(1_u128 << vertex);
            independent &= adjacency[vertex] & others == 0;
            clique &= adjacency[vertex] & others == others;
            if !independent && !clique {
                break;
            }
        }
        if independent {
            independent_stars.push(subset);
        }
        if clique {
            clique_stars.push(subset);
        }
    });

    for stars in [&independent_stars, &clique_stars] {
        let Some(vertex_pairs) = recover_two_subset_incidence(stars, vertex_count, ground_order)
        else {
            continue;
        };
        let edges_are_disjoint = graph.neighbours[0].len() == expected_degree;
        let relation_matches = (0..vertex_count).all(|left| {
            (left + 1..vertex_count).all(|right| {
                let left_pair = vertex_pairs[left];
                let right_pair = vertex_pairs[right];
                let disjoint = left_pair.0 != right_pair.0
                    && left_pair.0 != right_pair.1
                    && left_pair.1 != right_pair.0
                    && left_pair.1 != right_pair.1;
                let adjacent = adjacency[left] & (1_u128 << right) != 0;
                adjacent
                    == if edges_are_disjoint {
                        disjoint
                    } else {
                        !disjoint
                    }
            })
        });
        if !relation_matches {
            continue;
        }

        let mut pair_to_vertex = BTreeMap::new();
        for (vertex, &pair) in vertex_pairs.iter().enumerate() {
            pair_to_vertex.insert(pair, vertex);
        }
        let ground_generators = [
            (0..ground_order)
                .map(|point| match point {
                    0 => 1,
                    1 => 0,
                    _ => point,
                })
                .collect::<Vec<_>>(),
            (0..ground_order)
                .map(|point| (point + 1) % ground_order)
                .collect::<Vec<_>>(),
        ];
        let mut generators = Vec::with_capacity(2);
        for ground_generator in ground_generators {
            let images = vertex_pairs
                .iter()
                .map(|&(left, right)| {
                    let mut image = (ground_generator[left], ground_generator[right]);
                    if image.0 > image.1 {
                        image = (image.1, image.0);
                    }
                    pair_to_vertex[&image] as Point
                })
                .collect::<Vec<_>>();
            generators.push(
                Permutation::from_images(&images)
                    .map_err(GraphSymmetryError::InvalidPermutation)?,
            );
        }
        if generators
            .iter()
            .any(|generator| !is_dense_simple_automorphism(graph, generator))
        {
            return Err(GraphSymmetryError::AutomorphismVerificationFailed);
        }
        let group = PermutationGroup::from_generators(vertex_count, &generators)
            .map_err(GraphSymmetryError::PermutationGroup)?;
        let mut expected_order = BigUInt::one();
        for factor in 2..=ground_order {
            expected_order.mul_assign_u64(factor as u64);
        }
        if group.order() != expected_order {
            return Err(GraphSymmetryError::AutomorphismVerificationFailed);
        }
        return Ok(Some(group));
    }
    Ok(None)
}

fn recover_two_subset_incidence(
    stars: &[u128],
    vertex_count: usize,
    ground_order: usize,
) -> Option<Vec<(usize, usize)>> {
    if stars.len() != ground_order {
        return None;
    }
    let mut pairs = Vec::with_capacity(vertex_count);
    let mut seen = std::collections::BTreeSet::new();
    for vertex in 0..vertex_count {
        let memberships = stars
            .iter()
            .enumerate()
            .filter_map(|(star, &members)| (members & (1_u128 << vertex) != 0).then_some(star))
            .collect::<Vec<_>>();
        if memberships.len() != 2 {
            return None;
        }
        let pair = (memberships[0], memberships[1]);
        if !seen.insert(pair) {
            return None;
        }
        pairs.push(pair);
    }
    (seen.len() == vertex_count).then_some(pairs)
}

fn binomial_bounded(n: usize, k: usize, cap: usize) -> Option<usize> {
    let k = k.min(n.saturating_sub(k));
    let mut value = 1_u128;
    for index in 0..k {
        value = value.checked_mul((n - index) as u128)? / (index + 1) as u128;
        if value > cap as u128 {
            return None;
        }
    }
    Some(value as usize)
}

fn for_fixed_size_subsets(mut universe: usize, size: usize, mut visit: impl FnMut(u128)) {
    if size == 0 {
        visit(0);
        return;
    }
    if size > universe {
        return;
    }
    universe = universe.min(127);
    let limit = 1_u128 << universe;
    let mut subset = (1_u128 << size) - 1;
    while subset < limit {
        visit(subset);
        let lowest = subset & subset.wrapping_neg();
        let ripple = subset + lowest;
        subset = ripple | (((ripple ^ subset) >> 2) / lowest);
    }
}

struct SymmetrySearch<'a> {
    graph: &'a Graph,
    dense_simple: Option<DenseSimpleGraph>,
    nodes: usize,
    best_key: Option<SearchKey>,
    best_labeling: Option<Vec<Point>>,
    generators: Vec<Permutation>,
    group: PermutationGroup,
}

impl<'a> SymmetrySearch<'a> {
    fn new(graph: &'a Graph) -> Result<Self, GraphSymmetryError> {
        let group = PermutationGroup::from_generators(graph.order(), &[])
            .map_err(GraphSymmetryError::PermutationGroup)?;
        Ok(Self {
            graph,
            dense_simple: dense_simple_graph(graph),
            nodes: 0,
            best_key: None,
            best_labeling: None,
            generators: Vec::new(),
            group,
        })
    }

    fn visit(
        &mut self,
        partition: Partition,
        fixed: &mut Vec<Point>,
    ) -> Result<(), GraphSymmetryError> {
        self.nodes += 1;
        if self.nodes > SEARCH_NODE_CAP {
            return Err(GraphSymmetryError::SearchBudgetExceeded {
                cap: SEARCH_NODE_CAP,
            });
        }

        let partition = match &self.dense_simple {
            Some(simple) => refine_dense_simple_partition(simple, partition),
            None => refine_partition(self.graph, partition),
        };
        let Some(cell_index) = branch_cell(&partition) else {
            return self.visit_leaf(&partition);
        };

        let candidates = partition[cell_index].clone();
        let mut explored: Vec<Point> = Vec::new();
        let mut stabilizer_generation = usize::MAX;
        let mut stabilizer = None;
        let mut covered = vec![false; self.graph.order()];
        for candidate in candidates {
            let candidate = candidate as Point;
            if stabilizer_generation != self.generators.len() {
                let current = self
                    .group
                    .pointwise_stabilizer(fixed)
                    .map_err(GraphSymmetryError::PermutationGroup)?;
                covered.fill(false);
                for &representative in &explored {
                    for point in current
                        .orbit(representative)
                        .map_err(GraphSymmetryError::PermutationGroup)?
                    {
                        covered[point as usize] = true;
                    }
                }
                stabilizer_generation = self.generators.len();
                stabilizer = Some(current);
            }
            if covered[candidate as usize] {
                continue;
            }

            let child = individualize(&partition, cell_index, candidate as usize);
            fixed.push(candidate);
            self.visit(child, fixed)?;
            fixed.pop();
            explored.push(candidate);
            if stabilizer_generation == self.generators.len() {
                for point in stabilizer
                    .as_ref()
                    .expect("stabilizer is populated before recursion")
                    .orbit(candidate)
                    .map_err(GraphSymmetryError::PermutationGroup)?
                {
                    covered[point as usize] = true;
                }
            }
        }
        Ok(())
    }

    fn visit_leaf(&mut self, partition: &Partition) -> Result<(), GraphSymmetryError> {
        let labeling = labeling_from_discrete_partition(partition);
        let key = canonical_search_key(self.graph, &labeling, self.dense_simple.is_some());
        match &self.best_key {
            None => {
                self.best_key = Some(key);
                self.best_labeling = Some(labeling);
            }
            Some(best_key) if key < *best_key => {
                self.best_key = Some(key);
                self.best_labeling = Some(labeling);
            }
            Some(best_key) if key == *best_key => {
                let best = self
                    .best_labeling
                    .as_ref()
                    .expect("a best key always has a labeling");
                let automorphism = automorphism_between(best, &labeling)
                    .map_err(GraphSymmetryError::InvalidPermutation)?;
                self.add_verified_automorphism(automorphism)?;
            }
            Some(_) => {}
        }
        Ok(())
    }

    fn add_verified_automorphism(
        &mut self,
        automorphism: Permutation,
    ) -> Result<(), GraphSymmetryError> {
        if automorphism.is_identity() {
            return Ok(());
        }
        let verified = match &self.dense_simple {
            Some(simple) => is_dense_simple_automorphism(simple, &automorphism),
            None => is_automorphism(self.graph, &automorphism),
        };
        if !verified {
            return Err(GraphSymmetryError::AutomorphismVerificationFailed);
        }
        if self
            .group
            .contains(&automorphism)
            .map_err(GraphSymmetryError::PermutationGroup)?
        {
            return Ok(());
        }
        self.generators.push(automorphism);
        self.group = self
            .group
            .adjoin_generator(self.generators.last().expect("generator was just appended"))
            .map_err(GraphSymmetryError::PermutationGroup)?;
        Ok(())
    }
}

fn search_symmetries(graph: &Graph) -> Result<SymmetrySearch<'_>, GraphSymmetryError> {
    if graph.order() > Point::MAX as usize {
        return Err(GraphSymmetryError::VertexDomainTooLarge {
            order: graph.order(),
        });
    }

    let mut search = SymmetrySearch::new(graph)?;
    let initial = if graph.order() == 0 {
        Vec::new()
    } else {
        vec![(0..graph.order()).collect()]
    };
    search.visit(initial, &mut Vec::new())?;
    Ok(search)
}

fn canonicalize(graph: &Graph) -> Result<CanonicalLabel, GraphSymmetryError> {
    let search = search_symmetries(graph)?;
    let labeling = search
        .best_labeling
        .expect("every finite partition search has a leaf");
    let permutation =
        Permutation::from_images(&labeling).map_err(GraphSymmetryError::InvalidPermutation)?;
    let canonical_edges = canonical_edge_key(graph, &labeling);
    let canonical_graph = Graph::from_edges(graph.order(), &canonical_edges, graph.is_directed())
        .map_err(GraphSymmetryError::InvalidCanonicalGraph)?;
    Ok(CanonicalLabel::from_verified_parts(
        canonical_graph,
        permutation,
        search.group,
        graph,
    ))
}

fn refine_partition(graph: &Graph, mut partition: Partition) -> Partition {
    loop {
        let mut cell_of = vec![0usize; graph.order()];
        for (cell_index, cell) in partition.iter().enumerate() {
            for &vertex in cell {
                cell_of[vertex] = cell_index;
            }
        }

        let mut outgoing = vec![Vec::new(); graph.order()];
        let mut incoming = vec![Vec::new(); graph.order()];
        for edge in graph.edges() {
            let source = edge.source as usize;
            let target = edge.target as usize;
            outgoing[source].push((cell_of[target], edge.weight));
            if graph.is_directed() {
                incoming[target].push((cell_of[source], edge.weight));
            } else if source != target {
                outgoing[target].push((cell_of[source], edge.weight));
            }
        }
        for values in outgoing.iter_mut().chain(incoming.iter_mut()) {
            values.sort_unstable();
        }

        let mut refined = Vec::with_capacity(partition.len());
        for cell in &partition {
            let mut splits: BTreeMap<RefinementSignature, Vec<usize>> = BTreeMap::new();
            for &vertex in cell {
                splits
                    .entry(RefinementSignature {
                        outgoing: outgoing[vertex].clone(),
                        incoming: incoming[vertex].clone(),
                    })
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

fn refine_dense_simple_partition(graph: &DenseSimpleGraph, mut partition: Partition) -> Partition {
    loop {
        let mut cell_of = vec![0usize; graph.neighbours.len()];
        for (cell_index, cell) in partition.iter().enumerate() {
            for &vertex in cell {
                cell_of[vertex] = cell_index;
            }
        }

        let mut signatures = vec![vec![0_u16; partition.len()]; graph.neighbours.len()];
        for (vertex, neighbours) in graph.neighbours.iter().enumerate() {
            for &neighbour in neighbours {
                signatures[vertex][cell_of[neighbour]] += 1;
            }
        }

        let mut refined = Vec::with_capacity(partition.len());
        for cell in &partition {
            let mut sorted = cell.clone();
            sorted.sort_unstable_by(|left, right| {
                signatures[*left]
                    .cmp(&signatures[*right])
                    .then_with(|| left.cmp(right))
            });
            let mut start = 0;
            while start < sorted.len() {
                let mut end = start + 1;
                while end < sorted.len() && signatures[sorted[start]] == signatures[sorted[end]] {
                    end += 1;
                }
                refined.push(sorted[start..end].to_vec());
                start = end;
            }
        }
        if refined.len() == partition.len() {
            return refined;
        }
        partition = refined;
    }
}

fn branch_cell(partition: &Partition) -> Option<usize> {
    partition
        .iter()
        .enumerate()
        .filter(|(_, cell)| cell.len() > 1)
        .min_by_key(|(index, cell)| (cell.len(), *index))
        .map(|(index, _)| index)
}

fn individualize(partition: &Partition, cell_index: usize, vertex: usize) -> Partition {
    let mut result = Vec::with_capacity(partition.len() + 1);
    for (index, cell) in partition.iter().enumerate() {
        if index != cell_index {
            result.push(cell.clone());
            continue;
        }
        result.push(vec![vertex]);
        let remainder: Vec<usize> = cell
            .iter()
            .copied()
            .filter(|&candidate| candidate != vertex)
            .collect();
        if !remainder.is_empty() {
            result.push(remainder);
        }
    }
    result
}

fn labeling_from_discrete_partition(partition: &Partition) -> Vec<Point> {
    let mut labeling = vec![0; partition.len()];
    for (canonical, cell) in partition.iter().enumerate() {
        debug_assert_eq!(cell.len(), 1);
        labeling[cell[0]] = canonical as Point;
    }
    labeling
}

fn identity_labels(order: usize) -> Vec<Point> {
    (0..order as Point).collect()
}

fn canonical_edge_key(graph: &Graph, labeling: &[Point]) -> EdgeKey {
    let mut edges = Vec::with_capacity(graph.edge_count());
    for edge in graph.edges() {
        let mut source = labeling[edge.source as usize];
        let mut target = labeling[edge.target as usize];
        if !graph.is_directed() && source > target {
            core::mem::swap(&mut source, &mut target);
        }
        edges.push((source, target, edge.weight));
    }
    edges.sort_unstable();
    edges
}

fn dense_simple_graph(graph: &Graph) -> Option<DenseSimpleGraph> {
    if graph.is_directed() || graph.order() > DENSE_SIMPLE_KEY_MAX_ORDER {
        return None;
    }
    let bit_count = graph
        .order()
        .saturating_mul(graph.order().saturating_sub(1))
        / 2;
    let mut seen = vec![0_u64; bit_count.div_ceil(64)];
    let mut neighbours = vec![Vec::new(); graph.order()];
    for edge in graph.edges() {
        if edge.weight != 1 || edge.source == edge.target {
            return None;
        }
        let (source, target) = if edge.source < edge.target {
            (edge.source as usize, edge.target as usize)
        } else {
            (edge.target as usize, edge.source as usize)
        };
        let index = source * (2 * graph.order() - source - 1) / 2 + (target - source - 1);
        let mask = 1_u64 << (index % 64);
        if seen[index / 64] & mask != 0 {
            return None;
        }
        seen[index / 64] |= mask;
        neighbours[source].push(target);
        neighbours[target].push(source);
    }
    for row in &mut neighbours {
        row.sort_unstable();
    }
    Some(DenseSimpleGraph { neighbours })
}

fn canonical_search_key(graph: &Graph, labeling: &[Point], dense_simple: bool) -> SearchKey {
    if !dense_simple {
        return SearchKey::General(canonical_edge_key(graph, labeling));
    }
    let order = graph.order();
    let bit_count = order.saturating_mul(order.saturating_sub(1)) / 2;
    let mut words = vec![0_u64; bit_count.div_ceil(64)];
    for edge in graph.edges() {
        let mut source = labeling[edge.source as usize] as usize;
        let mut target = labeling[edge.target as usize] as usize;
        if source > target {
            core::mem::swap(&mut source, &mut target);
        }
        let index = source * (2 * order - source - 1) / 2 + (target - source - 1);
        words[index / 64] |= 1_u64 << (index % 64);
    }
    SearchKey::DenseSimple(words)
}

fn automorphism_between(best: &[Point], other: &[Point]) -> Result<Permutation, GroupError> {
    let mut best_inverse = vec![0; best.len()];
    for (vertex, &label) in best.iter().enumerate() {
        best_inverse[label as usize] = vertex as Point;
    }
    let images: Vec<Point> = other
        .iter()
        .map(|&label| best_inverse[label as usize])
        .collect();
    Permutation::from_images(&images)
}

fn is_automorphism(graph: &Graph, permutation: &Permutation) -> bool {
    permutation.degree() == graph.order()
        && canonical_edge_key(graph, permutation.images())
            == canonical_edge_key(graph, &identity_labels(graph.order()))
}

fn is_dense_simple_automorphism(graph: &DenseSimpleGraph, permutation: &Permutation) -> bool {
    if permutation.degree() != graph.neighbours.len() {
        return false;
    }
    graph.neighbours.iter().enumerate().all(|(source, row)| {
        row.iter().all(|&target| {
            let image_source = permutation.apply(source as Point) as usize;
            let image_target = permutation.apply(target as Point) as usize;
            graph.neighbours[image_source]
                .binary_search(&image_target)
                .is_ok()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use perm_core::BigUInt;

    #[test]
    fn path_canonical_label_is_invariant_under_relabeling() {
        let path = Graph::from_unweighted(4, &[(0, 1), (1, 2), (2, 3)], false).unwrap();
        let relabeled = Graph::from_unweighted(4, &[(2, 0), (0, 3), (3, 1)], false).unwrap();

        let left = path.canonical_label().unwrap();
        let right = relabeled.canonical_label().unwrap();

        assert_eq!(left.graph(), right.graph());
        assert!(left.verify(&path));
        assert!(right.verify(&relabeled));
        assert_eq!(left.automorphism_group().order(), BigUInt::from(2_u64));
    }

    #[test]
    fn cycle_automorphisms_are_the_dihedral_group() {
        let cycle = Graph::from_unweighted(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false).unwrap();
        let group = cycle.automorphism_group().unwrap();

        assert_eq!(group.degree(), 4);
        assert_eq!(group.order(), BigUInt::from(8_u64));
        for generator in group.generators() {
            assert!(is_automorphism(&cycle, generator));
        }
    }

    #[test]
    fn dense_simple_key_handles_petersen_and_cube_symmetries() {
        let petersen = Graph::from_unweighted(
            10,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 7),
                (7, 9),
                (9, 6),
                (6, 8),
                (8, 5),
            ],
            false,
        )
        .unwrap();
        assert!(dense_simple_graph(&petersen).is_some());
        assert_eq!(
            petersen.automorphism_group().unwrap().order(),
            BigUInt::from(120_u64)
        );

        let cube_edges = (0_u32..8)
            .flat_map(|vertex| {
                (0..3).filter_map(move |bit| {
                    let neighbour = vertex ^ (1 << bit);
                    (vertex < neighbour).then_some((vertex, neighbour))
                })
            })
            .collect::<Vec<_>>();
        let cube = Graph::from_unweighted(8, &cube_edges, false).unwrap();
        assert!(dense_simple_graph(&cube).is_some());
        assert_eq!(
            cube.automorphism_group().unwrap().order(),
            BigUInt::from(48_u64)
        );
    }

    #[test]
    fn complete_bipartite_graph_has_wreath_product_symmetry() {
        let edges = (0..3)
            .flat_map(|left| (3..6).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let graph = Graph::from_unweighted(6, &edges, false).unwrap();
        let group = graph.automorphism_group().unwrap();

        // (S_3 x S_3) semidirect C_2.
        assert_eq!(group.order(), BigUInt::from(72_u64));
        for generator in group.generators() {
            assert!(is_automorphism(&graph, generator));
        }
    }

    #[test]
    fn two_subset_recognizer_handles_kneser_and_triangular_families() {
        fn two_subsets(order: usize) -> Vec<(usize, usize)> {
            (0..order)
                .flat_map(|left| (left + 1..order).map(move |right| (left, right)))
                .collect()
        }

        let pairs = two_subsets(6);
        let kneser_edges = (0..pairs.len())
            .flat_map(|left| {
                let pairs = &pairs;
                (left + 1..pairs.len()).filter_map(move |right| {
                    let a = pairs[left];
                    let b = pairs[right];
                    (a.0 != b.0 && a.0 != b.1 && a.1 != b.0 && a.1 != b.1)
                        .then_some((left as Point, right as Point))
                })
            })
            .collect::<Vec<_>>();
        let triangular_edges = (0..pairs.len())
            .flat_map(|left| {
                let pairs = &pairs;
                (left + 1..pairs.len()).filter_map(move |right| {
                    let a = pairs[left];
                    let b = pairs[right];
                    (a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1)
                        .then_some((left as Point, right as Point))
                })
            })
            .collect::<Vec<_>>();

        for edges in [&kneser_edges, &triangular_edges] {
            let graph = Graph::from_unweighted(pairs.len(), edges, false).unwrap();
            let simple = dense_simple_graph(&graph).unwrap();
            let group = two_subset_graph_automorphism_group(&simple)
                .unwrap()
                .expect("recognized two-subset graph");
            assert_eq!(group.order(), BigUInt::from(720_u64));
            for generator in group.generators() {
                assert!(is_automorphism(&graph, generator));
            }
        }
    }

    #[test]
    fn dense_simple_key_rejects_decorated_and_parallel_graphs() {
        let weighted = Graph::from_edges(3, &[(0, 1, 2), (1, 2, 1)], false).unwrap();
        let parallel = Graph::from_unweighted(3, &[(0, 1), (0, 1)], false).unwrap();
        assert!(dense_simple_graph(&weighted).is_none());
        assert!(dense_simple_graph(&parallel).is_none());
        assert_eq!(
            weighted.automorphism_group().unwrap().order(),
            BigUInt::from(1_u64)
        );
        assert_eq!(
            parallel.automorphism_group().unwrap().order(),
            BigUInt::from(2_u64)
        );
    }

    #[test]
    fn direction_weights_loops_and_multiplicity_are_respected() {
        let graph =
            Graph::from_edges(3, &[(0, 0, 7), (0, 1, 2), (0, 1, 2), (1, 2, 3)], true).unwrap();

        assert_eq!(
            graph.automorphism_group().unwrap().order(),
            BigUInt::from(1_u64)
        );
    }

    #[test]
    fn edgeless_graph_has_full_symmetric_group() {
        let graph = Graph::from_edges(5, &[], false).unwrap();
        assert_eq!(
            graph.automorphism_group().unwrap().order(),
            BigUInt::from(120_u64)
        );
    }

    #[test]
    fn all_simple_graphs_of_order_four_match_brute_force() {
        let possible_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let relabeling = [2, 0, 3, 1];
        let permutations = all_permutations(4);

        for mask in 0..(1_u32 << possible_edges.len()) {
            let edges: Vec<(Point, Point)> = possible_edges
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(_, edge)| *edge)
                .collect();
            let graph = Graph::from_unweighted(4, &edges, false).unwrap();
            let expected_order = permutations
                .iter()
                .filter(|images| {
                    let permutation = Permutation::from_images(images).unwrap();
                    is_automorphism(&graph, &permutation)
                })
                .count() as u64;

            let moved_edges: Vec<(Point, Point)> = edges
                .iter()
                .map(|&(u, v)| (relabeling[u as usize], relabeling[v as usize]))
                .collect();
            let moved = Graph::from_unweighted(4, &moved_edges, false).unwrap();

            let canonical = graph.canonical_label().unwrap();
            assert_eq!(
                canonical.automorphism_group().order(),
                BigUInt::from(expected_order),
                "wrong automorphism order for edge mask {mask:06b}"
            );
            assert_eq!(
                canonical.graph(),
                moved.canonical_label().unwrap().graph(),
                "canonical graph changed under relabeling for edge mask {mask:06b}"
            );
        }
    }

    fn all_permutations(order: usize) -> Vec<Vec<Point>> {
        fn generate(index: usize, values: &mut [Point], output: &mut Vec<Vec<Point>>) {
            if index == values.len() {
                output.push(values.to_vec());
                return;
            }
            for swap_index in index..values.len() {
                values.swap(index, swap_index);
                generate(index + 1, values, output);
                values.swap(index, swap_index);
            }
        }

        let mut values: Vec<Point> = (0..order as Point).collect();
        let mut output = Vec::new();
        generate(0, &mut values, &mut output);
        output
    }
}
