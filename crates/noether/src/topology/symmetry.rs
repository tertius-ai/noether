// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact canonical forms and automorphisms of finite simplicial complexes.

use std::collections::BTreeSet;

use crate::combin::{Graph, GraphSymmetry, GraphSymmetryError};
use crate::groups::{Permutation, PermutationGroup, Point};

use super::{SimplicialComplex, TopologyError};

/// Legacy exhaustive-search cutoff retained for source compatibility.
///
/// The incidence-graph implementation no longer uses a fixed vertex-count
/// cutoff; it fails closed at the shared deterministic graph-search budget.
#[deprecated(
    since = "0.1.0",
    note = "simplicial symmetry now uses the shared graph-search budget instead of a vertex cutoff"
)]
pub const MAX_SIMPLICIAL_AUTOMORPHISM_VERTICES: usize = 9;

/// Largest permutation degree created from a sparse vertex-label domain.
pub const MAX_SIMPLICIAL_AUTOMORPHISM_DEGREE: usize = 4_096;

const INCIDENCE_EDGE_WEIGHT: i64 = 0;
const VERTEX_COLOR_WEIGHT: i64 = -1;
const FACET_COLOR_WEIGHT: i64 = -2;

impl SimplicialComplex {
    /// Canonically relabel the used vertices as `0,...,n-1`.
    ///
    /// The labeling is induced by exact canonicalization of the colored
    /// vertex--facet incidence graph. Vertex and facet nodes have distinct
    /// loop colors, so graph isomorphisms are exactly simplicial
    /// isomorphisms. This operation and [`Self::automorphism_group`] share the
    /// same individualization/refinement search.
    pub fn canonical_form(&self) -> Result<Self, TopologyError> {
        let data = incidence_symmetry(self)?;
        let mut ranks = (0..data.vertices.len()).collect::<Vec<_>>();
        ranks.sort_by_key(|local| data.labeling.apply(*local as Point));
        let mut canonical_rank = vec![0_u32; data.vertices.len()];
        for (rank, local) in ranks.into_iter().enumerate() {
            canonical_rank[local] =
                u32::try_from(rank).map_err(|_| GraphSymmetryError::VertexDomainTooLarge {
                    order: data.vertices.len(),
                })?;
        }
        let facets = self
            .facets()
            .into_iter()
            .map(|facet| {
                facet
                    .vertices()
                    .iter()
                    .map(|vertex| {
                        let local = data
                            .vertices
                            .binary_search(vertex)
                            .expect("every facet vertex is a stored vertex");
                        canonical_rank[local]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(Self::from_facets(&facets))
    }

    /// Compute the exact automorphism group acting on the original vertex ids.
    ///
    /// Missing ids below the largest used label are fixed. Automorphisms are
    /// projected from the exact colored incidence-graph group and reverified
    /// against the facet set.
    pub fn automorphism_group(&self) -> Result<PermutationGroup, TopologyError> {
        let data = incidence_symmetry(self)?;
        let degree = data
            .vertices
            .last()
            .map_or(0, |vertex| *vertex as usize + 1);
        if degree > MAX_SIMPLICIAL_AUTOMORPHISM_DEGREE {
            return Err(TopologyError::SymmetryDegreeTooLarge {
                degree,
                max: MAX_SIMPLICIAL_AUTOMORPHISM_DEGREE,
            });
        }

        let mut generators = Vec::new();
        for graph_generator in data.automorphism_group.generators() {
            let mut images = (0..degree as Point).collect::<Vec<_>>();
            for (local, vertex) in data.vertices.iter().enumerate() {
                let target_local = graph_generator.apply(local as Point) as usize;
                if target_local >= data.vertices.len() {
                    return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
                }
                images[*vertex as usize] = data.vertices[target_local];
            }
            let generator = Permutation::from_images(&images)?;
            if !is_simplicial_automorphism(self, &generator) {
                return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
            }
            generators.push(generator);
        }
        let group = PermutationGroup::from_generators(degree, &generators)?;
        if !group
            .generators()
            .iter()
            .all(|generator| is_simplicial_automorphism(self, generator))
        {
            return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
        }
        Ok(group)
    }
}

struct IncidenceSymmetry {
    vertices: Vec<u32>,
    labeling: Permutation,
    automorphism_group: PermutationGroup,
}

fn incidence_symmetry(complex: &SimplicialComplex) -> Result<IncidenceSymmetry, TopologyError> {
    let vertices = complex
        .simplices(0)
        .iter()
        .map(|simplex| simplex.vertices()[0])
        .collect::<Vec<_>>();
    let facets = complex.facets();
    let order = vertices
        .len()
        .checked_add(facets.len())
        .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: usize::MAX })?;
    if order > Point::MAX as usize {
        return Err(GraphSymmetryError::VertexDomainTooLarge { order }.into());
    }

    let mut edges = Vec::new();
    for vertex in 0..vertices.len() {
        let vertex = vertex as Point;
        edges.push((vertex, vertex, VERTEX_COLOR_WEIGHT));
    }
    for (facet_index, facet) in facets.iter().enumerate() {
        let facet_node = (vertices.len() + facet_index) as Point;
        edges.push((facet_node, facet_node, FACET_COLOR_WEIGHT));
        for vertex in facet.vertices() {
            let local = vertices
                .binary_search(vertex)
                .expect("every facet vertex is a stored vertex") as Point;
            edges.push((local, facet_node, INCIDENCE_EDGE_WEIGHT));
        }
    }
    let graph = Graph::from_edges(order, &edges, false)
        .map_err(GraphSymmetryError::InvalidCanonicalGraph)?;
    let canonical = graph.canonical_label()?;
    Ok(IncidenceSymmetry {
        vertices,
        labeling: canonical.labeling().clone(),
        automorphism_group: canonical.automorphism_group().clone(),
    })
}

fn is_simplicial_automorphism(complex: &SimplicialComplex, permutation: &Permutation) -> bool {
    let target = complex
        .facets()
        .into_iter()
        .map(|simplex| simplex.vertices().to_vec())
        .collect::<BTreeSet<_>>();
    let image = complex
        .facets()
        .into_iter()
        .map(|simplex| {
            let mut facet = simplex
                .vertices()
                .iter()
                .map(|vertex| permutation.apply(*vertex))
                .collect::<Vec<_>>();
            facet.sort_unstable();
            facet
        })
        .collect::<BTreeSet<_>>();
    image == target
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn triangle_boundary_has_dihedral_automorphisms() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![0, 2], vec![1, 2]]);
        assert_eq!(
            circle.automorphism_group().unwrap().order().to_decimal(),
            "6"
        );
    }

    #[test]
    fn sparse_labels_are_preserved_and_holes_are_fixed() {
        let edge = SimplicialComplex::from_facets(&[vec![1, 3]]);
        let group = edge.automorphism_group().unwrap();
        assert_eq!(group.degree(), 4);
        assert_eq!(group.order().to_decimal(), "2");
        assert!(group
            .generators()
            .iter()
            .all(|generator| { generator.apply(0) == 0 && generator.apply(2) == 2 }));
        assert_eq!(
            edge.canonical_form().unwrap(),
            SimplicialComplex::from_facets(&[vec![0, 1]])
        );
    }

    #[test]
    fn canonical_form_is_invariant_under_relabeling_and_idempotent() {
        let complex = SimplicialComplex::from_facets(&[vec![0, 1, 2], vec![0, 3], vec![1, 3]]);
        let relabeled = SimplicialComplex::from_facets(&[vec![4, 1, 3], vec![4, 2], vec![1, 2]]);
        let canonical = complex.canonical_form().unwrap();
        assert_eq!(canonical, relabeled.canonical_form().unwrap());
        assert_eq!(canonical, canonical.canonical_form().unwrap());
    }

    #[test]
    fn simplex_symmetry_is_no_longer_factorial_search_capped_at_nine() {
        let simplex = SimplicialComplex::from_facets(&[(0..10).collect()]);
        assert_eq!(
            simplex.automorphism_group().unwrap().order().to_decimal(),
            "3628800"
        );
    }

    #[test]
    fn empty_complex_has_canonical_empty_form_and_trivial_group() {
        let empty = SimplicialComplex::from_facets(&[]);
        assert_eq!(empty.canonical_form().unwrap(), empty);
        assert_eq!(
            empty.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
    }
}
