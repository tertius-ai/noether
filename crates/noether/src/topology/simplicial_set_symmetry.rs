// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact canonical forms and automorphisms of finite simplicial sets.

use crate::combin::{Graph, GraphSymmetry, GraphSymmetryError};
use crate::groups::{Permutation, PermutationGroup, Point};

use super::{SimplicialFace, SimplicialSet, TopologyError};

const STRUCTURE_EDGE_WEIGHT: i64 = 0;

impl SimplicialSet {
    /// Canonically number the non-degenerate simplices in every dimension.
    ///
    /// The exact encoding is a directed colored incidence graph. Its nodes
    /// record non-degenerate simplices, individual face occurrences, and the
    /// ordered entries of degeneracy surjections. Consequently graph
    /// isomorphisms are precisely simplicial-set isomorphisms, including all
    /// face positions and degeneracy data.
    pub fn canonical_form(&self) -> Result<Self, TopologyError> {
        let data = incidence_symmetry(self)?;
        let mut old_to_new = self
            .level_counts()
            .into_iter()
            .map(|count| vec![0_usize; count])
            .collect::<Vec<_>>();
        let mut new_to_old = old_to_new.clone();

        for dimension in 0..old_to_new.len() {
            let mut old_indices = (0..self.count(dimension)).collect::<Vec<_>>();
            old_indices.sort_by_key(|simplex| {
                data.labeling
                    .apply(data.offsets[dimension].saturating_add(*simplex) as Point)
            });
            for (new, old) in old_indices.into_iter().enumerate() {
                old_to_new[dimension][old] = new;
                new_to_old[dimension][new] = old;
            }
        }

        let faces = new_to_old
            .iter()
            .enumerate()
            .map(|(dimension, old_indices)| {
                old_indices
                    .iter()
                    .map(|old| {
                        self.faces_of(dimension, *old)
                            .expect("canonical ranks name existing simplices")
                            .iter()
                            .map(|face| relabel_face(face, dimension, &old_to_new))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::new(faces)
    }

    /// Exact automorphism group on all non-degenerate simplices.
    ///
    /// The permutation domain is the dimension-major order: first all
    /// vertices, then all non-degenerate edges, and so on. Every returned
    /// permutation preserves each dimensional block.
    pub fn automorphism_group(&self) -> Result<PermutationGroup, TopologyError> {
        let data = incidence_symmetry(self)?;
        let degree = self.total_cells();
        let mut generators = Vec::new();
        for graph_generator in data.automorphism_group.generators() {
            let mut images = Vec::with_capacity(degree);
            for cell in 0..degree {
                let target = graph_generator.apply(cell as Point) as usize;
                if target >= degree {
                    return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
                }
                images.push(target as Point);
            }
            let generator = Permutation::from_images(&images)?;
            if !is_simplicial_set_automorphism(self, &generator, &data.offsets) {
                return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
            }
            generators.push(generator);
        }
        let group = PermutationGroup::from_generators(degree, &generators)?;
        if !group
            .generators()
            .iter()
            .all(|generator| is_simplicial_set_automorphism(self, generator, &data.offsets))
        {
            return Err(GraphSymmetryError::AutomorphismVerificationFailed.into());
        }
        Ok(group)
    }

    fn level_counts(&self) -> Vec<usize> {
        (0..=self.dimension().max(0) as usize)
            .map(|dimension| self.count(dimension))
            .collect()
    }
}

struct IncidenceSymmetry {
    offsets: Vec<usize>,
    labeling: Permutation,
    automorphism_group: PermutationGroup,
}

fn incidence_symmetry(value: &SimplicialSet) -> Result<IncidenceSymmetry, TopologyError> {
    let counts = value.level_counts();
    let mut offsets = Vec::with_capacity(counts.len());
    let mut cell_nodes = 0_usize;
    for count in &counts {
        offsets.push(cell_nodes);
        cell_nodes = cell_nodes
            .checked_add(*count)
            .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: usize::MAX })?;
    }

    let mut edges = Vec::new();
    for (dimension, count) in counts.iter().enumerate() {
        let color = simplex_color(dimension)?;
        for simplex in 0..*count {
            let node = point(offsets[dimension].saturating_add(simplex))?;
            edges.push((node, node, color));
        }
    }

    let mut order = cell_nodes;
    for dimension in 1..counts.len() {
        for simplex in 0..counts[dimension] {
            let source = point(offsets[dimension].saturating_add(simplex))?;
            for (face_index, face) in value
                .faces_of(dimension, simplex)
                .expect("validated simplicial sets carry every face")
                .iter()
                .enumerate()
            {
                let face_node = point(order)?;
                order = order
                    .checked_add(1)
                    .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: usize::MAX })?;
                edges.push((face_node, face_node, face_color(face_index)?));
                edges.push((source, face_node, STRUCTURE_EDGE_WEIGHT));

                let (target_dimension, target_simplex) = match face {
                    SimplicialFace::Cell(target) => (dimension - 1, *target),
                    SimplicialFace::Degenerate {
                        simplex_dimension,
                        simplex,
                        ..
                    } => (*simplex_dimension, *simplex),
                };
                let target = point(offsets[target_dimension].saturating_add(target_simplex))?;
                edges.push((face_node, target, STRUCTURE_EDGE_WEIGHT));

                if let SimplicialFace::Degenerate { surjection, .. } = face {
                    for (position, value) in surjection.iter().enumerate() {
                        let entry_node = point(order)?;
                        order = order.checked_add(1).ok_or(
                            GraphSymmetryError::VertexDomainTooLarge { order: usize::MAX },
                        )?;
                        edges.push((entry_node, entry_node, entry_color(*value)?));
                        edges.push((face_node, entry_node, position_weight(position)?));
                    }
                }
            }
        }
    }
    if order > Point::MAX as usize {
        return Err(GraphSymmetryError::VertexDomainTooLarge { order }.into());
    }
    let graph = Graph::from_edges(order, &edges, true)
        .map_err(GraphSymmetryError::InvalidCanonicalGraph)?;
    let canonical = graph.canonical_label()?;
    Ok(IncidenceSymmetry {
        offsets,
        labeling: canonical.labeling().clone(),
        automorphism_group: canonical.automorphism_group().clone(),
    })
}

fn relabel_face(
    face: &SimplicialFace,
    source_dimension: usize,
    old_to_new: &[Vec<usize>],
) -> SimplicialFace {
    match face {
        SimplicialFace::Cell(simplex) => {
            SimplicialFace::Cell(old_to_new[source_dimension - 1][*simplex])
        }
        SimplicialFace::Degenerate {
            simplex_dimension,
            simplex,
            surjection,
        } => SimplicialFace::Degenerate {
            simplex_dimension: *simplex_dimension,
            simplex: old_to_new[*simplex_dimension][*simplex],
            surjection: surjection.clone(),
        },
    }
}

fn is_simplicial_set_automorphism(
    value: &SimplicialSet,
    permutation: &Permutation,
    offsets: &[usize],
) -> bool {
    for dimension in 0..offsets.len() {
        for simplex in 0..value.count(dimension) {
            let flat = offsets[dimension] + simplex;
            let image_flat = permutation.apply(flat as Point) as usize;
            let Some(image) = image_flat.checked_sub(offsets[dimension]) else {
                return false;
            };
            if image >= value.count(dimension) {
                return false;
            }
            let Some(source_faces) = value.faces_of(dimension, simplex) else {
                return false;
            };
            let Some(target_faces) = value.faces_of(dimension, image) else {
                return false;
            };
            if source_faces.len() != target_faces.len() {
                return false;
            }
            for (source, target) in source_faces.iter().zip(target_faces) {
                let expected = match source {
                    SimplicialFace::Cell(cell) => {
                        let target_flat =
                            permutation.apply((offsets[dimension - 1] + cell) as Point);
                        SimplicialFace::Cell(target_flat as usize - offsets[dimension - 1])
                    }
                    SimplicialFace::Degenerate {
                        simplex_dimension,
                        simplex,
                        surjection,
                    } => {
                        let target_flat =
                            permutation.apply((offsets[*simplex_dimension] + simplex) as Point);
                        SimplicialFace::Degenerate {
                            simplex_dimension: *simplex_dimension,
                            simplex: target_flat as usize - offsets[*simplex_dimension],
                            surjection: surjection.clone(),
                        }
                    }
                };
                if &expected != target {
                    return false;
                }
            }
        }
    }
    true
}

fn point(value: usize) -> Result<Point, GraphSymmetryError> {
    Point::try_from(value).map_err(|_| GraphSymmetryError::VertexDomainTooLarge { order: value })
}

fn simplex_color(dimension: usize) -> Result<i64, GraphSymmetryError> {
    let value = i64::try_from(dimension)
        .ok()
        .and_then(|dimension| dimension.checked_mul(2))
        .and_then(|dimension| dimension.checked_add(1))
        .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: dimension })?;
    Ok(-value)
}

fn face_color(face: usize) -> Result<i64, GraphSymmetryError> {
    let value = i64::try_from(face)
        .ok()
        .and_then(|face| face.checked_mul(2))
        .and_then(|face| face.checked_add(2))
        .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: face })?;
    Ok(-value)
}

fn entry_color(value: usize) -> Result<i64, GraphSymmetryError> {
    i64::try_from(value)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: value })
}

fn position_weight(position: usize) -> Result<i64, GraphSymmetryError> {
    i64::try_from(position)
        .ok()
        .and_then(|position| position.checked_add(1))
        .ok_or(GraphSymmetryError::VertexDomainTooLarge { order: position })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![], vec![]],
            vec![vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)]],
        ])
        .unwrap()
    }

    fn relabeled_triangle() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(2)],
                vec![SimplicialFace::Cell(1), SimplicialFace::Cell(2)],
                vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)],
            ],
            vec![vec![
                SimplicialFace::Cell(2),
                SimplicialFace::Cell(1),
                SimplicialFace::Cell(0),
            ]],
        ])
        .unwrap()
    }

    fn triangle() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(1)],
            ],
            vec![vec![
                SimplicialFace::Cell(2),
                SimplicialFace::Cell(1),
                SimplicialFace::Cell(0),
            ]],
        ])
        .unwrap()
    }

    #[test]
    fn ordered_interval_has_no_orientation_reversing_automorphism() {
        let value = interval();
        assert_eq!(
            value.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
        assert_eq!(
            value.canonical_form().unwrap().canonical_form().unwrap(),
            value.canonical_form().unwrap()
        );
    }

    #[test]
    fn canonical_form_ignores_numbering_in_each_dimension() {
        let standard = triangle();
        assert_eq!(
            standard.canonical_form().unwrap(),
            relabeled_triangle().canonical_form().unwrap()
        );
        assert_eq!(
            standard.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
    }

    #[test]
    fn degeneracy_data_is_part_of_the_exact_encoding() {
        let sphere = SimplicialSet::sphere(2);
        let canonical = sphere.canonical_form().unwrap();
        assert_eq!(canonical, sphere);
        assert_eq!(
            sphere.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
    }

    #[test]
    fn empty_set_has_a_trivial_degree_zero_group() {
        let empty = SimplicialSet::new(Vec::new()).unwrap();
        assert_eq!(empty.canonical_form().unwrap(), empty);
        assert_eq!(empty.automorphism_group().unwrap().degree(), 0);
        assert_eq!(
            empty.automorphism_group().unwrap().order().to_decimal(),
            "1"
        );
    }
}
