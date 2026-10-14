// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Replayable dominated-vertex reductions of finite simplicial complexes.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{NoetherError, SemanticType, TypeDescription};

use super::{SimplicialComplex, TopologyError};

static STRONG_CORE_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/StrongCoreReduction@1",
    rust_path: "noether::topology::StrongCoreReduction",
    python_path: Some("noether.topology.StrongCoreReduction"),
};

/// One elementary strong collapse: delete `vertex`, which is dominated by
/// `dominator` in the current complex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DominatedVertex {
    vertex: u32,
    dominator: u32,
}

impl DominatedVertex {
    /// Construct a proposed dominated-vertex deletion.
    pub fn new(vertex: u32, dominator: u32) -> Self {
        Self { vertex, dominator }
    }

    /// Deleted vertex.
    pub fn vertex(self) -> u32 {
        self.vertex
    }

    /// A distinct vertex lying in every maximal simplex containing `vertex`.
    pub fn dominator(self) -> u32 {
        self.dominator
    }
}

/// A complete replayable reduction to a strong-homotopy core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrongCoreReduction {
    source: SimplicialComplex,
    steps: Vec<DominatedVertex>,
    reduced: SimplicialComplex,
    core: SimplicialComplex,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrongCoreReductionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: SimplicialComplex,
    steps: Vec<DominatedVertex>,
}

impl SimplicialComplex {
    /// Every currently dominated vertex, with its least-label dominator.
    ///
    /// A vertex `v` is dominated by `w` precisely when every maximal simplex
    /// containing `v` also contains `w`.
    pub fn dominated_vertices(&self) -> Vec<DominatedVertex> {
        let vertices = self
            .simplices(0)
            .iter()
            .map(|simplex| simplex.vertices()[0])
            .collect::<Vec<_>>();
        let facets = self.facets();
        let mut result = Vec::new();
        for vertex in &vertices {
            let containing = facets
                .iter()
                .filter(|facet| facet.vertices().binary_search(vertex).is_ok())
                .collect::<Vec<_>>();
            if let Some(dominator) = vertices.iter().copied().find(|candidate| {
                candidate != vertex
                    && containing
                        .iter()
                        .all(|facet| facet.vertices().binary_search(candidate).is_ok())
            }) {
                result.push(DominatedVertex {
                    vertex: *vertex,
                    dominator,
                });
            }
        }
        result
    }

    /// Delete a currently dominated vertex and verify the supplied dominator.
    pub fn strong_collapse(&self, step: DominatedVertex) -> Result<Self, TopologyError> {
        let containing = self
            .facets()
            .into_iter()
            .filter(|facet| facet.vertices().binary_search(&step.vertex).is_ok())
            .collect::<Vec<_>>();
        let valid = step.vertex != step.dominator
            && !containing.is_empty()
            && containing
                .iter()
                .all(|facet| facet.vertices().binary_search(&step.dominator).is_ok());
        if !valid {
            return Err(TopologyError::InvalidStrongCollapse {
                reason: format!(
                    "vertex {} is not dominated by {} in the current complex",
                    step.vertex, step.dominator
                ),
            });
        }
        Ok(self.delete_vertex(step.vertex))
    }

    /// Deterministically delete dominated vertices until none remain.
    ///
    /// The replayed `reduced` complex retains surviving labels. The published
    /// `core` is its exact canonical form, so isomorphic inputs have equal
    /// core values even when their deterministic deletion sequences differ.
    pub fn strong_core_reduction(&self) -> Result<StrongCoreReduction, TopologyError> {
        let mut reduced = self.clone();
        let mut steps = Vec::new();
        while let Some(step) = reduced.dominated_vertices().first().copied() {
            reduced = reduced.strong_collapse(step)?;
            steps.push(step);
        }
        let core = reduced.canonical_form()?;
        let result = StrongCoreReduction {
            source: self.clone(),
            steps,
            reduced,
            core,
        };
        result.verify()?;
        Ok(result)
    }

    /// Canonical representative of the strong-homotopy core.
    pub fn strong_core(&self) -> Result<Self, TopologyError> {
        Ok(self.strong_core_reduction()?.core)
    }

    fn delete_vertex(&self, vertex: u32) -> Self {
        let facets = self
            .facets()
            .into_iter()
            .map(|facet| {
                facet
                    .vertices()
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != vertex)
                    .collect::<Vec<_>>()
            })
            .filter(|facet| !facet.is_empty())
            .collect::<Vec<_>>();
        Self::from_facets(&facets)
    }
}

impl StrongCoreReduction {
    /// Source complex with its original labels.
    pub fn source(&self) -> &SimplicialComplex {
        &self.source
    }

    /// Ordered elementary strong collapses.
    pub fn steps(&self) -> &[DominatedVertex] {
        &self.steps
    }

    /// Replayed irreducible complex before final canonical relabeling.
    pub fn reduced(&self) -> &SimplicialComplex {
        &self.reduced
    }

    /// Canonically relabeled strong-homotopy core.
    pub fn core(&self) -> &SimplicialComplex {
        &self.core
    }

    /// Recheck every domination claim, replay every deletion, and recompute
    /// the canonical irreducible core.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let mut current = self.source.clone();
        for step in &self.steps {
            current = current.strong_collapse(*step)?;
        }
        if current != self.reduced {
            return Err(TopologyError::InvalidStrongCollapse {
                reason: "replayed deletions do not reconstruct the stored reduced complex"
                    .to_owned(),
            });
        }
        if !self.reduced.dominated_vertices().is_empty() {
            return Err(TopologyError::InvalidStrongCollapse {
                reason: "the stored reduced complex still has a dominated vertex".to_owned(),
            });
        }
        if self.reduced.canonical_form()? != self.core {
            return Err(TopologyError::InvalidStrongCollapse {
                reason: "canonicalizing the reduced complex does not reconstruct the core"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// Serialize the source and replayable steps; derived complexes are
    /// recomputed during deserialization.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize, replay, and revalidate the complete reduction.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for StrongCoreReduction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        StrongCoreReductionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            steps: self.steps.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StrongCoreReduction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StrongCoreReductionWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let mut reduced = wire.source.clone();
        for step in &wire.steps {
            reduced = reduced
                .strong_collapse(*step)
                .map_err(serde::de::Error::custom)?;
        }
        if !reduced.dominated_vertices().is_empty() {
            return Err(serde::de::Error::custom(
                "serialized strong-core reduction stops before reaching a core",
            ));
        }
        let core = reduced.canonical_form().map_err(serde::de::Error::custom)?;
        let result = Self {
            source: wire.source,
            steps: wire.steps,
            reduced,
            core,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl fmt::Display for StrongCoreReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "strong core reduction ({} vertices -> {} vertices, {} steps)",
            self.source.count(0),
            self.core.count(0),
            self.steps.len()
        )
    }
}

impl SemanticType for StrongCoreReduction {
    const TYPE_ID: &'static str = "noether/topology/StrongCoreReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        let count = u64::try_from(self.steps.len()).map_err(|_| {
            NoetherError::Serialization("strong-collapse step count does not fit u64".to_owned())
        })?;
        output.extend_from_slice(&count.to_be_bytes());
        for step in &self.steps {
            output.extend_from_slice(&step.vertex.to_be_bytes());
            output.extend_from_slice(&step.dominator.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &STRONG_CORE_REDUCTION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_reduces_to_a_canonical_point_with_replayable_steps() {
        let simplex = SimplicialComplex::from_facets(&[vec![4, 2, 7, 1]]);
        let reduction = simplex.strong_core_reduction().unwrap();
        assert_eq!(reduction.steps.len(), 3);
        assert_eq!(reduction.core, SimplicialComplex::from_facets(&[vec![0]]));
        reduction.verify().unwrap();
        assert_eq!(
            StrongCoreReduction::from_json(&reduction.to_json().unwrap()).unwrap(),
            reduction
        );
    }

    #[test]
    fn circle_has_no_dominated_vertex() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![1, 2], vec![0, 2]]);
        let reduction = circle.strong_core_reduction().unwrap();
        assert!(reduction.steps.is_empty());
        assert_eq!(reduction.core, circle.canonical_form().unwrap());
    }

    #[test]
    fn invalid_dominator_fails_closed() {
        let edge = SimplicialComplex::from_facets(&[vec![0, 1]]);
        assert!(matches!(
            edge.strong_collapse(DominatedVertex {
                vertex: 0,
                dominator: 2,
            }),
            Err(TopologyError::InvalidStrongCollapse { .. })
        ));
    }

    #[test]
    fn isomorphic_inputs_have_the_same_published_core() {
        let left = SimplicialComplex::from_facets(&[vec![0, 1, 2], vec![2, 3]]);
        let right = SimplicialComplex::from_facets(&[vec![8, 4, 6], vec![6, 1]]);
        assert_eq!(left.strong_core().unwrap(), right.strong_core().unwrap());
    }
}
