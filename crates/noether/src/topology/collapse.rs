// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Replayable elementary collapses of finite simplicial complexes.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{NoetherError, SemanticType, TypeDescription};

use super::{Simplex, SimplicialComplex, TopologyError};

static COLLAPSE_SEQUENCE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialCollapseSequence@1",
    rust_path: "noether::topology::SimplicialCollapseSequence",
    python_path: Some("noether.topology.SimplicialCollapseSequence"),
};

static SIMPLE_HOMOTOPY_ZIGZAG_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialSimpleHomotopyZigzag@1",
    rust_path: "noether::topology::SimplicialSimpleHomotopyZigzag",
    python_path: Some("noether.topology.SimplicialSimpleHomotopyZigzag"),
};

/// One elementary collapse deleting a free codimension-one face and its
/// unique maximal coface.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElementaryCollapse {
    free_face: Vec<u32>,
    coface: Vec<u32>,
}

impl ElementaryCollapse {
    /// Construct a proposed collapse. The source complex verifies it on use.
    pub fn new(free_face: Vec<u32>, coface: Vec<u32>) -> Self {
        Self {
            free_face: Simplex::new(free_face).vertices().to_vec(),
            coface: Simplex::new(coface).vertices().to_vec(),
        }
    }

    /// Free face removed by the collapse.
    pub fn free_face(&self) -> &[u32] {
        &self.free_face
    }

    /// Unique maximal coface removed with the free face.
    pub fn coface(&self) -> &[u32] {
        &self.coface
    }
}

/// An exact finite sequence of elementary simplicial collapses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialCollapseSequence {
    source: SimplicialComplex,
    steps: Vec<ElementaryCollapse>,
    target: SimplicialComplex,
}

/// One oriented move in a simplicial simple-homotopy zigzag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "direction", rename_all = "snake_case", deny_unknown_fields)]
pub enum SimplicialSimpleHomotopyMove {
    /// Apply the indicated elementary collapse to the current complex.
    Collapse {
        /// Checked on the current complex when the zigzag is constructed or
        /// replayed.
        step: ElementaryCollapse,
    },
    /// Replace the current complex by `expanded`, certified by collapsing
    /// `expanded` back to the current complex.
    Expansion {
        /// Complex after the elementary expansion.
        expanded: SimplicialComplex,
        /// Collapse of `expanded` whose target must equal the current complex.
        inverse_collapse: ElementaryCollapse,
    },
}

/// A replayable zigzag of elementary simplicial expansions and collapses.
///
/// This is exact evidence of simple-homotopy equivalence between its endpoints;
/// it does not claim that either endpoint is a canonical representative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialSimpleHomotopyZigzag {
    source: SimplicialComplex,
    moves: Vec<SimplicialSimpleHomotopyMove>,
    target: SimplicialComplex,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimpleHomotopyZigzagWire {
    #[serde(rename = "type")]
    type_id: String,
    source: SimplicialComplex,
    moves: Vec<SimplicialSimpleHomotopyMove>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CollapseSequenceWire {
    #[serde(rename = "type")]
    type_id: String,
    source: SimplicialComplex,
    steps: Vec<ElementaryCollapse>,
}

impl SimplicialComplex {
    /// All elementary free-face collapses available in this complex.
    pub fn elementary_collapses(&self) -> Vec<ElementaryCollapse> {
        let facets = self.facets();
        let mut output = Vec::new();
        for coface in &facets {
            if coface.vertices().len() <= 1 {
                continue;
            }
            for free_face in coface.faces() {
                let containing = facets
                    .iter()
                    .filter(|facet| free_face.is_face_of(facet))
                    .count();
                if containing == 1 {
                    output.push(ElementaryCollapse::new(
                        free_face.vertices().to_vec(),
                        coface.vertices().to_vec(),
                    ));
                }
            }
        }
        output.sort();
        output
    }

    /// Replay one proposed elementary collapse after verifying freeness.
    pub fn elementary_collapse(&self, step: &ElementaryCollapse) -> Result<Self, TopologyError> {
        let free_face = Simplex::new(step.free_face.clone());
        let coface = Simplex::new(step.coface.clone());
        let facets = self.facets();
        let coface_is_facet = facets.iter().any(|facet| **facet == coface);
        let containing = facets
            .iter()
            .filter(|facet| free_face.is_face_of(facet))
            .collect::<Vec<_>>();
        let valid = !free_face.vertices().is_empty()
            && free_face.vertices().len().checked_add(1) == Some(coface.vertices().len())
            && free_face.is_face_of(&coface)
            && coface_is_facet
            && containing.len() == 1
            && *containing[0] == &coface;
        if !valid {
            return Err(TopologyError::InvalidElementaryCollapse {
                reason: format!(
                    "{:?} is not a free codimension-one face of the unique maximal coface {:?}",
                    step.free_face, step.coface
                ),
            });
        }

        let mut new_facets = facets
            .into_iter()
            .filter(|facet| **facet != coface)
            .map(|facet| facet.vertices().to_vec())
            .collect::<Vec<_>>();
        new_facets.extend(
            coface
                .faces()
                .into_iter()
                .filter(|face| face != &free_face)
                .map(|face| face.vertices().to_vec()),
        );
        Ok(Self::from_facets(&new_facets))
    }

    /// Replay and retain a checked sequence of elementary collapses.
    pub fn collapse_sequence(
        &self,
        steps: Vec<ElementaryCollapse>,
    ) -> Result<SimplicialCollapseSequence, TopologyError> {
        let mut target = self.clone();
        for step in &steps {
            target = target.elementary_collapse(step)?;
        }
        let result = SimplicialCollapseSequence {
            source: self.clone(),
            steps,
            target,
        };
        result.verify()?;
        Ok(result)
    }

    /// Replay and retain a checked zigzag of elementary expansions and
    /// collapses beginning at this complex.
    pub fn simple_homotopy_zigzag(
        &self,
        moves: Vec<SimplicialSimpleHomotopyMove>,
    ) -> Result<SimplicialSimpleHomotopyZigzag, TopologyError> {
        let target = replay_simple_homotopy_moves(self, &moves)?;
        Ok(SimplicialSimpleHomotopyZigzag {
            source: self.clone(),
            moves,
            target,
        })
    }

    /// Deterministically collapse the least available free-face pair until no
    /// elementary collapse remains.
    ///
    /// Unlike a strong-homotopy core, this terminal target need not be unique
    /// up to isomorphism. The returned sequence is the mathematical evidence;
    /// callers must not treat the target as a canonical simple-homotopy type.
    pub fn greedy_collapse_reduction(&self) -> Result<SimplicialCollapseSequence, TopologyError> {
        let mut target = self.clone();
        let mut steps = Vec::new();
        while let Some(step) = target.elementary_collapses().first().cloned() {
            target = target.elementary_collapse(&step)?;
            steps.push(step);
        }
        let result = SimplicialCollapseSequence {
            source: self.clone(),
            steps,
            target,
        };
        result.verify()?;
        Ok(result)
    }
}

impl SimplicialSimpleHomotopyZigzag {
    /// Initial complex.
    pub fn source(&self) -> &SimplicialComplex {
        &self.source
    }

    /// Ordered elementary expansions and collapses.
    pub fn moves(&self) -> &[SimplicialSimpleHomotopyMove] {
        &self.moves
    }

    /// Endpoint reconstructed by exact replay.
    pub fn target(&self) -> &SimplicialComplex {
        &self.target
    }

    /// Recheck every elementary move and reconstruct the endpoint.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let target = replay_simple_homotopy_moves(&self.source, &self.moves)?;
        if target == self.target {
            Ok(())
        } else {
            Err(TopologyError::InvalidElementaryCollapse {
                reason: "replayed simple-homotopy zigzag changed its target".to_owned(),
            })
        }
    }

    /// Serialize the source and replayable moves; the target is derived.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly replay every move.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn replay_simple_homotopy_moves(
    source: &SimplicialComplex,
    moves: &[SimplicialSimpleHomotopyMove],
) -> Result<SimplicialComplex, TopologyError> {
    let mut current = source.clone();
    for movement in moves {
        match movement {
            SimplicialSimpleHomotopyMove::Collapse { step } => {
                current = current.elementary_collapse(step)?;
            }
            SimplicialSimpleHomotopyMove::Expansion {
                expanded,
                inverse_collapse,
            } => {
                if expanded.elementary_collapse(inverse_collapse)? != current {
                    return Err(TopologyError::InvalidElementaryCollapse {
                        reason:
                            "an elementary expansion does not collapse back to its declared source"
                                .to_owned(),
                    });
                }
                current = expanded.clone();
            }
        }
    }
    Ok(current)
}

impl SimplicialCollapseSequence {
    /// Source complex before any collapse.
    pub fn source(&self) -> &SimplicialComplex {
        &self.source
    }

    /// Ordered free-face/coface pairs.
    pub fn steps(&self) -> &[ElementaryCollapse] {
        &self.steps
    }

    /// Target reconstructed by replaying all steps.
    pub fn target(&self) -> &SimplicialComplex {
        &self.target
    }

    /// Whether no further elementary collapse is available at the target.
    pub fn is_terminal(&self) -> bool {
        self.target.elementary_collapses().is_empty()
    }

    /// Recheck every freeness condition and reconstruct the target.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let mut current = self.source.clone();
        for step in &self.steps {
            current = current.elementary_collapse(step)?;
        }
        if current != self.target {
            return Err(TopologyError::InvalidElementaryCollapse {
                reason: "replayed elementary collapses do not reconstruct the stored target"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Serialize only the source and replayable steps; the target is derived.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize, replay, and verify the complete collapse sequence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for SimplicialCollapseSequence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        CollapseSequenceWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            steps: self.steps.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialCollapseSequence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CollapseSequenceWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.source
            .collapse_sequence(wire.steps)
            .map_err(serde::de::Error::custom)
    }
}

impl Serialize for SimplicialSimpleHomotopyZigzag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SimpleHomotopyZigzagWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            moves: self.moves.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialSimpleHomotopyZigzag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SimpleHomotopyZigzagWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.source
            .simple_homotopy_zigzag(wire.moves)
            .map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for SimplicialCollapseSequence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "simplicial collapse sequence ({} cells -> {} cells, {} steps)",
            self.source.total_cells(),
            self.target.total_cells(),
            self.steps.len()
        )
    }
}

impl fmt::Display for SimplicialSimpleHomotopyZigzag {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "simplicial simple-homotopy zigzag ({} cells <-> {} cells, {} moves)",
            self.source.total_cells(),
            self.target.total_cells(),
            self.moves.len()
        )
    }
}

impl SemanticType for SimplicialCollapseSequence {
    const TYPE_ID: &'static str = "noether/topology/SimplicialCollapseSequence@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        let count = u64::try_from(self.steps.len()).map_err(|_| {
            NoetherError::Serialization("collapse step count does not fit u64".to_owned())
        })?;
        output.extend_from_slice(&count.to_be_bytes());
        for step in &self.steps {
            append_vertices(&mut output, &step.free_face)?;
            append_vertices(&mut output, &step.coface)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &COLLAPSE_SEQUENCE_DESCRIPTION
    }
}

impl SemanticType for SimplicialSimpleHomotopyZigzag {
    const TYPE_ID: &'static str = "noether/topology/SimplicialSimpleHomotopyZigzag@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        let count = u64::try_from(self.moves.len()).map_err(|_| {
            NoetherError::Serialization("simple-homotopy move count does not fit u64".to_owned())
        })?;
        output.extend_from_slice(&count.to_be_bytes());
        for movement in &self.moves {
            match movement {
                SimplicialSimpleHomotopyMove::Collapse { step } => {
                    output.push(0);
                    append_vertices(&mut output, &step.free_face)?;
                    append_vertices(&mut output, &step.coface)?;
                }
                SimplicialSimpleHomotopyMove::Expansion {
                    expanded,
                    inverse_collapse,
                } => {
                    output.push(1);
                    let expanded_bytes = expanded.canonical_bytes()?;
                    output.extend_from_slice(
                        &u64::try_from(expanded_bytes.len())
                            .map_err(|_| {
                                NoetherError::Serialization(
                                    "expanded complex bytes do not fit u64".to_owned(),
                                )
                            })?
                            .to_be_bytes(),
                    );
                    output.extend_from_slice(&expanded_bytes);
                    append_vertices(&mut output, &inverse_collapse.free_face)?;
                    append_vertices(&mut output, &inverse_collapse.coface)?;
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SIMPLE_HOMOTOPY_ZIGZAG_DESCRIPTION
    }
}

fn append_vertices(output: &mut Vec<u8>, vertices: &[u32]) -> Result<(), NoetherError> {
    let count = u64::try_from(vertices.len())
        .map_err(|_| NoetherError::Serialization("simplex size does not fit u64".to_owned()))?;
    output.extend_from_slice(&count.to_be_bytes());
    for vertex in vertices {
        output.extend_from_slice(&vertex.to_be_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_collapses_to_a_vertex_with_replayable_certificate() {
        let simplex = SimplicialComplex::from_facets(&[vec![0, 1, 2]]);
        let sequence = simplex.greedy_collapse_reduction().unwrap();
        assert_eq!(sequence.steps().len(), 3);
        assert_eq!(sequence.target().count(0), 1);
        assert!(sequence.is_terminal());
        sequence.verify().unwrap();
        assert_eq!(
            SimplicialCollapseSequence::from_json(&sequence.to_json().unwrap()).unwrap(),
            sequence
        );
    }

    #[test]
    fn circle_has_no_free_face() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![0, 2], vec![1, 2]]);
        assert!(circle.elementary_collapses().is_empty());
        assert!(circle.greedy_collapse_reduction().unwrap().is_terminal());
    }

    #[test]
    fn false_freeness_claim_fails_closed() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![0, 2], vec![1, 2]]);
        let step = ElementaryCollapse::new(vec![0], vec![0, 1]);
        assert!(matches!(
            circle.elementary_collapse(&step),
            Err(TopologyError::InvalidElementaryCollapse { .. })
        ));
    }

    #[test]
    fn collapse_preserves_integral_homology() {
        let disk = SimplicialComplex::from_facets(&[vec![0, 1, 2], vec![1, 2, 3]]);
        let sequence = disk.greedy_collapse_reduction().unwrap();
        for degree in 0..=disk.dimension() as usize {
            assert_eq!(
                disk.homology_invariants(degree).unwrap(),
                sequence.target().homology_invariants(degree).unwrap()
            );
        }
    }

    #[test]
    fn expansion_and_collapse_zigzag_replays_both_directions() {
        let point = SimplicialComplex::from_facets(&[vec![0]]);
        let interval = SimplicialComplex::from_facets(&[vec![0, 1]]);
        let inverse = ElementaryCollapse::new(vec![1], vec![0, 1]);
        let zigzag = point
            .simple_homotopy_zigzag(vec![
                SimplicialSimpleHomotopyMove::Expansion {
                    expanded: interval.clone(),
                    inverse_collapse: inverse.clone(),
                },
                SimplicialSimpleHomotopyMove::Collapse { step: inverse },
            ])
            .unwrap();
        assert_eq!(zigzag.target(), &point);
        zigzag.verify().unwrap();
        assert_eq!(
            SimplicialSimpleHomotopyZigzag::from_json(&zigzag.to_json().unwrap()).unwrap(),
            zigzag
        );
    }
}
