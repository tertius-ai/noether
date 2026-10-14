// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Replayable elementary horn collapses of finite simplicial sets.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{NoetherError, SemanticType, TypeDescription};

use super::{SimplicialFace, SimplicialSet, TopologyError};

static HORN_COLLAPSE_SEQUENCE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialSetHornCollapseSequence@1",
    rust_path: "noether::topology::SimplicialSetHornCollapseSequence",
    python_path: Some("noether.topology.SimplicialSetHornCollapseSequence"),
};

static ANODYNE_ZIGZAG_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialSetAnodyneZigzag@1",
    rust_path: "noether::topology::SimplicialSetAnodyneZigzag",
    python_path: Some("noether.topology.SimplicialSetAnodyneZigzag"),
};

/// One elementary collapse inverse to attaching a simplex along a horn.
///
/// The step removes `simplex` in `simplex_dimension` together with the
/// nondegenerate codimension-one simplex named by its `face`-th face map.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplicialSetElementaryHornCollapse {
    simplex_dimension: usize,
    simplex: usize,
    face: usize,
    free_face: usize,
}

impl SimplicialSetElementaryHornCollapse {
    /// Construct a proposed horn collapse. Its source verifies freeness on
    /// use.
    pub const fn new(
        simplex_dimension: usize,
        simplex: usize,
        face: usize,
        free_face: usize,
    ) -> Self {
        Self {
            simplex_dimension,
            simplex,
            face,
            free_face,
        }
    }

    /// Dimension of the removed horn filler.
    pub const fn simplex_dimension(&self) -> usize {
        self.simplex_dimension
    }

    /// Index of the removed nondegenerate horn filler.
    pub const fn simplex(&self) -> usize {
        self.simplex
    }

    /// Missing horn-face index `k`.
    pub const fn face(&self) -> usize {
        self.face
    }

    /// Index of the removed nondegenerate free face in dimension `n-1`.
    pub const fn free_face(&self) -> usize {
        self.free_face
    }
}

/// An exact finite sequence of elementary simplicial-set horn collapses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialSetHornCollapseSequence {
    source: SimplicialSet,
    steps: Vec<SimplicialSetElementaryHornCollapse>,
    target: SimplicialSet,
}

/// One oriented move in a checked zigzag of horn attachments and collapses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "direction", rename_all = "snake_case", deny_unknown_fields)]
pub enum SimplicialSetAnodyneMove {
    /// Apply an elementary horn collapse to the current simplicial set.
    Collapse {
        /// Checked on the current simplicial set during replay.
        step: SimplicialSetElementaryHornCollapse,
    },
    /// Replace the current simplicial set by `expanded`, certified by an
    /// elementary horn collapse back to the current value.
    Expansion {
        /// Simplicial set after the horn attachment.
        expanded: SimplicialSet,
        /// Collapse of `expanded` whose target is the current simplicial set.
        inverse_collapse: SimplicialSetElementaryHornCollapse,
    },
}

/// A replayable zigzag of elementary horn attachments and collapses.
///
/// Every move is a weak-homotopy equivalence. The endpoint is evidence for
/// this specific zigzag and is not asserted to be canonical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialSetAnodyneZigzag {
    source: SimplicialSet,
    moves: Vec<SimplicialSetAnodyneMove>,
    target: SimplicialSet,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnodyneZigzagWire {
    #[serde(rename = "type")]
    type_id: String,
    source: SimplicialSet,
    moves: Vec<SimplicialSetAnodyneMove>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HornCollapseSequenceWire {
    #[serde(rename = "type")]
    type_id: String,
    source: SimplicialSet,
    steps: Vec<SimplicialSetElementaryHornCollapse>,
}

impl SimplicialSet {
    /// All elementary horn collapses available in this simplicial set.
    ///
    /// A returned pair removes exactly one nondegenerate `n`-simplex and one
    /// nondegenerate `(n-1)`-simplex. The latter occurs as exactly the declared
    /// face of the former and neither removed simplex underlies any other face
    /// or degeneracy reference.
    pub fn elementary_horn_collapses(&self) -> Vec<SimplicialSetElementaryHornCollapse> {
        let mut output = Vec::new();
        for dimension in 1..self.faces.len() {
            for (simplex, faces) in self.faces[dimension].iter().enumerate() {
                for (face, value) in faces.iter().enumerate() {
                    let SimplicialFace::Cell(free_face) = value else {
                        continue;
                    };
                    let step = SimplicialSetElementaryHornCollapse::new(
                        dimension, simplex, face, *free_face,
                    );
                    if self.is_elementary_horn_collapse(&step) {
                        output.push(step);
                    }
                }
            }
        }
        output.sort();
        output
    }

    /// Replay one proposed elementary horn collapse after checking every
    /// nondegenerate and degenerate reference to the removed cells.
    pub fn elementary_horn_collapse(
        &self,
        step: &SimplicialSetElementaryHornCollapse,
    ) -> Result<Self, TopologyError> {
        if !self.is_elementary_horn_collapse(step) {
            return Err(TopologyError::InvalidElementaryCollapse {
                reason: format!(
                    "dimension-{} simplex {} and its d_{} face {} do not form a free horn pair",
                    step.simplex_dimension, step.simplex, step.face, step.free_face
                ),
            });
        }

        let mut faces = self.faces.clone();
        faces[step.simplex_dimension].remove(step.simplex);
        faces[step.simplex_dimension - 1].remove(step.free_face);
        for (owner_dimension, level) in faces.iter_mut().enumerate() {
            for simplex_faces in level {
                for value in simplex_faces {
                    reindex_face(value, owner_dimension, step.simplex_dimension, step.simplex);
                    reindex_face(
                        value,
                        owner_dimension,
                        step.simplex_dimension - 1,
                        step.free_face,
                    );
                }
            }
        }
        SimplicialSet::new(faces)
    }

    /// Replay and retain a checked sequence of elementary horn collapses.
    pub fn horn_collapse_sequence(
        &self,
        steps: Vec<SimplicialSetElementaryHornCollapse>,
    ) -> Result<SimplicialSetHornCollapseSequence, TopologyError> {
        let mut target = self.clone();
        for step in &steps {
            target = target.elementary_horn_collapse(step)?;
        }
        let result = SimplicialSetHornCollapseSequence {
            source: self.clone(),
            steps,
            target,
        };
        result.verify()?;
        Ok(result)
    }

    /// Replay and retain a checked zigzag of horn attachments and collapses
    /// beginning at this simplicial set.
    pub fn anodyne_zigzag(
        &self,
        moves: Vec<SimplicialSetAnodyneMove>,
    ) -> Result<SimplicialSetAnodyneZigzag, TopologyError> {
        let target = replay_anodyne_moves(self, &moves)?;
        Ok(SimplicialSetAnodyneZigzag {
            source: self.clone(),
            moves,
            target,
        })
    }

    /// Deterministically remove the least available free horn pair until no
    /// elementary horn collapse remains.
    ///
    /// The reverse of every step is a pushout of a horn inclusion and hence an
    /// anodyne extension. The terminal target is evidence for this particular
    /// zigzag; it is not asserted to be a canonical homotopy representative.
    pub fn greedy_horn_collapse_reduction(
        &self,
    ) -> Result<SimplicialSetHornCollapseSequence, TopologyError> {
        let mut target = self.clone();
        let mut steps = Vec::new();
        while let Some(step) = target.elementary_horn_collapses().first().cloned() {
            target = target.elementary_horn_collapse(&step)?;
            steps.push(step);
        }
        let result = SimplicialSetHornCollapseSequence {
            source: self.clone(),
            steps,
            target,
        };
        result.verify()?;
        Ok(result)
    }

    fn is_elementary_horn_collapse(&self, step: &SimplicialSetElementaryHornCollapse) -> bool {
        let dimension = step.simplex_dimension;
        if dimension == 0 || step.face > dimension {
            return false;
        }
        let Some(simplex_faces) = self
            .faces
            .get(dimension)
            .and_then(|level| level.get(step.simplex))
        else {
            return false;
        };
        if simplex_faces.get(step.face) != Some(&SimplicialFace::Cell(step.free_face))
            || step.free_face >= self.count(dimension - 1)
        {
            return false;
        }

        for (owner_dimension, level) in self.faces.iter().enumerate().skip(1) {
            for (simplex, faces) in level.iter().enumerate() {
                for (face, value) in faces.iter().enumerate() {
                    let declared_occurrence = owner_dimension == dimension
                        && simplex == step.simplex
                        && face == step.face;
                    if references_cell(value, owner_dimension, dimension - 1, step.free_face)
                        && !declared_occurrence
                    {
                        return false;
                    }
                    if references_cell(value, owner_dimension, dimension, step.simplex) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl SimplicialSetAnodyneZigzag {
    /// Initial simplicial set.
    pub fn source(&self) -> &SimplicialSet {
        &self.source
    }

    /// Ordered horn attachments and collapses.
    pub fn moves(&self) -> &[SimplicialSetAnodyneMove] {
        &self.moves
    }

    /// Endpoint reconstructed by exact replay.
    pub fn target(&self) -> &SimplicialSet {
        &self.target
    }

    /// Recheck every horn pair and reconstruct the endpoint.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let target = replay_anodyne_moves(&self.source, &self.moves)?;
        if target == self.target {
            Ok(())
        } else {
            Err(TopologyError::InvalidElementaryCollapse {
                reason: "replayed anodyne zigzag changed its target".to_owned(),
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

fn replay_anodyne_moves(
    source: &SimplicialSet,
    moves: &[SimplicialSetAnodyneMove],
) -> Result<SimplicialSet, TopologyError> {
    let mut current = source.clone();
    for movement in moves {
        match movement {
            SimplicialSetAnodyneMove::Collapse { step } => {
                current = current.elementary_horn_collapse(step)?;
            }
            SimplicialSetAnodyneMove::Expansion {
                expanded,
                inverse_collapse,
            } => {
                if expanded.elementary_horn_collapse(inverse_collapse)? != current {
                    return Err(TopologyError::InvalidElementaryCollapse {
                        reason: "a horn attachment does not collapse back to its declared source"
                            .to_owned(),
                    });
                }
                current = expanded.clone();
            }
        }
    }
    Ok(current)
}

impl SimplicialSetHornCollapseSequence {
    /// Source before any collapse.
    pub fn source(&self) -> &SimplicialSet {
        &self.source
    }

    /// Ordered horn-filler/free-face pairs.
    pub fn steps(&self) -> &[SimplicialSetElementaryHornCollapse] {
        &self.steps
    }

    /// Target reconstructed by replaying every step.
    pub fn target(&self) -> &SimplicialSet {
        &self.target
    }

    /// Whether the target admits no further elementary horn collapse.
    pub fn is_terminal(&self) -> bool {
        self.target.elementary_horn_collapses().is_empty()
    }

    /// Recheck every freeness condition and reconstruct the target.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let mut current = self.source.clone();
        for step in &self.steps {
            current = current.elementary_horn_collapse(step)?;
        }
        if current != self.target {
            return Err(TopologyError::InvalidElementaryCollapse {
                reason: "replayed horn collapses do not reconstruct the stored target".to_owned(),
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

impl Serialize for SimplicialSetHornCollapseSequence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        HornCollapseSequenceWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            steps: self.steps.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialSetHornCollapseSequence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = HornCollapseSequenceWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.source
            .horn_collapse_sequence(wire.steps)
            .map_err(serde::de::Error::custom)
    }
}

impl Serialize for SimplicialSetAnodyneZigzag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        AnodyneZigzagWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            moves: self.moves.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialSetAnodyneZigzag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AnodyneZigzagWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        wire.source
            .anodyne_zigzag(wire.moves)
            .map_err(serde::de::Error::custom)
    }
}

impl SemanticType for SimplicialSetHornCollapseSequence {
    const TYPE_ID: &'static str = "noether/topology/SimplicialSetHornCollapseSequence@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        output.extend_from_slice(&(self.steps.len() as u64).to_be_bytes());
        for step in &self.steps {
            output.extend_from_slice(&(step.simplex_dimension as u64).to_be_bytes());
            output.extend_from_slice(&(step.simplex as u64).to_be_bytes());
            output.extend_from_slice(&(step.face as u64).to_be_bytes());
            output.extend_from_slice(&(step.free_face as u64).to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &HORN_COLLAPSE_SEQUENCE_DESCRIPTION
    }
}

impl SemanticType for SimplicialSetAnodyneZigzag {
    const TYPE_ID: &'static str = "noether/topology/SimplicialSetAnodyneZigzag@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        output.extend_from_slice(
            &u64::try_from(self.moves.len())
                .map_err(|_| {
                    NoetherError::Serialization("anodyne move count does not fit u64".to_owned())
                })?
                .to_be_bytes(),
        );
        for movement in &self.moves {
            match movement {
                SimplicialSetAnodyneMove::Collapse { step } => {
                    output.push(0);
                    append_horn_collapse(&mut output, step)?;
                }
                SimplicialSetAnodyneMove::Expansion {
                    expanded,
                    inverse_collapse,
                } => {
                    output.push(1);
                    let expanded_bytes = expanded.canonical_bytes()?;
                    output.extend_from_slice(
                        &u64::try_from(expanded_bytes.len())
                            .map_err(|_| {
                                NoetherError::Serialization(
                                    "expanded simplicial-set bytes do not fit u64".to_owned(),
                                )
                            })?
                            .to_be_bytes(),
                    );
                    output.extend_from_slice(&expanded_bytes);
                    append_horn_collapse(&mut output, inverse_collapse)?;
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &ANODYNE_ZIGZAG_DESCRIPTION
    }
}

impl fmt::Display for SimplicialSetHornCollapseSequence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "simplicial-set horn collapse sequence ({} cells -> {} cells, {} steps)",
            self.source.total_cells(),
            self.target.total_cells(),
            self.steps.len()
        )
    }
}

impl fmt::Display for SimplicialSetAnodyneZigzag {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "simplicial-set anodyne zigzag ({} cells <-> {} cells, {} moves)",
            self.source.total_cells(),
            self.target.total_cells(),
            self.moves.len()
        )
    }
}

fn append_horn_collapse(
    output: &mut Vec<u8>,
    step: &SimplicialSetElementaryHornCollapse,
) -> Result<(), NoetherError> {
    for value in [
        step.simplex_dimension,
        step.simplex,
        step.face,
        step.free_face,
    ] {
        output.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| {
                    NoetherError::Serialization(
                        "horn-collapse coordinate does not fit u64".to_owned(),
                    )
                })?
                .to_be_bytes(),
        );
    }
    Ok(())
}

fn references_cell(
    value: &SimplicialFace,
    owner_dimension: usize,
    target_dimension: usize,
    target_simplex: usize,
) -> bool {
    match value {
        SimplicialFace::Cell(simplex) => {
            owner_dimension.checked_sub(1) == Some(target_dimension) && *simplex == target_simplex
        }
        SimplicialFace::Degenerate {
            simplex_dimension,
            simplex,
            ..
        } => *simplex_dimension == target_dimension && *simplex == target_simplex,
    }
}

fn reindex_face(
    value: &mut SimplicialFace,
    owner_dimension: usize,
    removed_dimension: usize,
    removed_simplex: usize,
) {
    let (named_dimension, named_simplex) = match value {
        SimplicialFace::Cell(simplex) => (owner_dimension.saturating_sub(1), simplex),
        SimplicialFace::Degenerate {
            simplex_dimension,
            simplex,
            ..
        } => (*simplex_dimension, simplex),
    };
    if named_dimension == removed_dimension && *named_simplex > removed_simplex {
        *named_simplex -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn interval_has_two_replayable_horn_collapses() {
        let interval = SimplicialSet::new(vec![
            vec![vec![], vec![]],
            vec![vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)]],
        ])
        .unwrap();
        let steps = interval.elementary_horn_collapses();
        assert_eq!(steps.len(), 2);
        for step in steps {
            let point = interval.elementary_horn_collapse(&step).unwrap();
            assert_eq!(point.total_cells(), 1);
            assert_eq!(point.dimension(), 0);
        }
    }

    #[test]
    fn loop_has_no_free_horn_face() {
        assert!(SimplicialSet::circle()
            .elementary_horn_collapses()
            .is_empty());
    }

    #[test]
    fn simplex_collapses_to_a_point_with_round_trip_evidence() {
        let simplex = triangle();
        let sequence = simplex.greedy_horn_collapse_reduction().unwrap();
        assert_eq!(sequence.steps().len(), 3);
        assert_eq!(sequence.target().total_cells(), 1);
        assert!(sequence.is_terminal());
        sequence.verify().unwrap();
        let restored =
            SimplicialSetHornCollapseSequence::from_json(&sequence.to_json().unwrap()).unwrap();
        assert_eq!(restored, sequence);
    }

    #[test]
    fn nonfree_declared_pair_is_rejected() {
        let loop_ = SimplicialSet::circle();
        let false_step = SimplicialSetElementaryHornCollapse::new(1, 0, 0, 0);
        assert!(matches!(
            loop_.elementary_horn_collapse(&false_step),
            Err(TopologyError::InvalidElementaryCollapse { .. })
        ));
    }

    #[test]
    fn horn_attachment_and_collapse_zigzag_replays_both_directions() {
        let interval = SimplicialSet::new(vec![
            vec![vec![], vec![]],
            vec![vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)]],
        ])
        .unwrap();
        let inverse = interval.elementary_horn_collapses()[0].clone();
        let point = interval.elementary_horn_collapse(&inverse).unwrap();
        let zigzag = point
            .anodyne_zigzag(vec![
                SimplicialSetAnodyneMove::Expansion {
                    expanded: interval,
                    inverse_collapse: inverse.clone(),
                },
                SimplicialSetAnodyneMove::Collapse { step: inverse },
            ])
            .unwrap();
        assert_eq!(zigzag.target(), &point);
        zigzag.verify().unwrap();
        assert_eq!(
            SimplicialSetAnodyneZigzag::from_json(&zigzag.to_json().unwrap()).unwrap(),
            zigzag
        );
    }
}
