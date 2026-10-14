// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked embeddings between compatible finite-field presentations.

use core::fmt;

use serde::{Deserialize, Serialize};

use super::{FiniteField, FiniteFieldElement};
use crate::{
    CoefficientMap, ContextMismatchError, Integer, NoetherError, Ring, SemanticType,
    TypeDescription,
};

static FINITE_FIELD_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/FiniteFieldEmbedding@1",
    rust_path: "noether::numthy::FiniteFieldEmbedding",
    python_path: Some("noether.numthy.FiniteFieldEmbedding"),
};

/// Failure to construct, apply, or verify a finite-field embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteFieldEmbeddingError {
    /// Source and target have different prime characteristics.
    CharacteristicMismatch {
        /// Source characteristic.
        source: Integer,
        /// Target characteristic.
        target: Integer,
    },
    /// The source degree does not divide the target degree.
    IncompatibleDegrees {
        /// Source degree over the prime field.
        source: usize,
        /// Target degree over the prime field.
        target: usize,
    },
    /// A supplied value belongs to another field presentation.
    ContextMismatch(ContextMismatchError),
    /// The proposed generator image does not define the declared embedding.
    InvalidGeneratorImage(&'static str),
}

impl fmt::Display for FiniteFieldEmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CharacteristicMismatch { source, target } => write!(
                formatter,
                "finite-field embedding characteristics differ: source {source}, target {target}"
            ),
            Self::IncompatibleDegrees { source, target } => write!(
                formatter,
                "finite-field embedding requires source degree to divide target degree; received {source} and {target}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidGeneratorImage(reason) => {
                write!(formatter, "invalid finite-field generator image: {reason}")
            }
        }
    }
}

impl std::error::Error for FiniteFieldEmbeddingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for FiniteFieldEmbeddingError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

/// An explicitly presented embedding between compatible finite fields.
///
/// For a proper extension source, the image of its distinguished power-basis
/// generator determines the map. Different valid roots give different maps.
#[derive(Clone, PartialEq, Eq)]
pub struct FiniteFieldEmbedding {
    source: FiniteField,
    target: FiniteField,
    generator_image: FiniteFieldElement,
}

impl FiniteFieldEmbedding {
    /// Construct the embedding determined by one target root of the source's
    /// defining polynomial.
    pub fn new(
        source: FiniteField,
        target: FiniteField,
        generator_image: FiniteFieldElement,
    ) -> Result<Self, FiniteFieldEmbeddingError> {
        let embedding = Self {
            source,
            target,
            generator_image,
        };
        embedding.verify()?;
        Ok(embedding)
    }

    /// Source field presentation.
    pub fn source(&self) -> &FiniteField {
        &self.source
    }

    /// Target field presentation.
    pub fn target(&self) -> &FiniteField {
        &self.target
    }

    /// Image of the source's distinguished power-basis generator.
    pub fn generator_image(&self) -> &FiniteFieldElement {
        &self.generator_image
    }

    /// Apply the embedding after checking exact source identity.
    pub fn apply(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<FiniteFieldElement, FiniteFieldEmbeddingError> {
        if !self.source.contains(value) {
            return Err(ContextMismatchError {
                expected: self.source.to_string(),
                actual: value.field().to_string(),
            }
            .into());
        }
        let mut image = self.target.zero();
        for coefficient in value.coefficients().iter().rev() {
            image = self.target.mul(&image, &self.generator_image)?;
            image = self
                .target
                .add(&image, &self.target.element(coefficient.clone()))?;
        }
        Ok(image)
    }

    /// Verify endpoints, degree compatibility, and the generator relation.
    pub fn verify(&self) -> Result<(), FiniteFieldEmbeddingError> {
        if self.source.characteristic() != self.target.characteristic() {
            return Err(FiniteFieldEmbeddingError::CharacteristicMismatch {
                source: self.source.characteristic().clone(),
                target: self.target.characteristic().clone(),
            });
        }
        if !self.target.degree().is_multiple_of(self.source.degree()) {
            return Err(FiniteFieldEmbeddingError::IncompatibleDegrees {
                source: self.source.degree(),
                target: self.target.degree(),
            });
        }
        if !self.target.contains(&self.generator_image) {
            return Err(FiniteFieldEmbeddingError::InvalidGeneratorImage(
                "the image belongs to another target presentation",
            ));
        }
        if self.source.degree() == 1 {
            if self.generator_image != self.target.one() {
                return Err(FiniteFieldEmbeddingError::InvalidGeneratorImage(
                    "the distinguished generator of a prime field is one",
                ));
            }
        } else {
            let modulus = self.source.defining_modulus().ok_or(
                FiniteFieldEmbeddingError::InvalidGeneratorImage(
                    "a proper extension has no defining polynomial",
                ),
            )?;
            let mut value = self.target.zero();
            for coefficient in modulus.iter().rev() {
                value = self.target.mul(&value, &self.generator_image)?;
                value = self
                    .target
                    .add(&value, &self.target.element(*coefficient))?;
            }
            if !value.is_zero() {
                return Err(FiniteFieldEmbeddingError::InvalidGeneratorImage(
                    "the image is not a root of the source defining polynomial",
                ));
            }
        }
        Ok(())
    }

    /// Serialize endpoints and the exact generator image.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteFieldEmbeddingWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reverify the embedding.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: FiniteFieldEmbeddingWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.source, wire.target, wire.generator_image)
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl CoefficientMap for FiniteFieldEmbedding {
    type Source = FiniteField;
    type Target = FiniteField;
    type Error = FiniteFieldEmbeddingError;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &FiniteFieldElement) -> Result<FiniteFieldElement, Self::Error> {
        Self::apply(self, value)
    }
}

impl SemanticType for FiniteFieldEmbedding {
    const TYPE_ID: &'static str = "noether/numthy/FiniteFieldEmbedding@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_FIELD_EMBEDDING_DESCRIPTION
    }
}

impl Serialize for FiniteFieldEmbedding {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        FiniteFieldEmbeddingWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FiniteFieldEmbedding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = FiniteFieldEmbeddingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.source, wire.target, wire.generator_image).map_err(serde::de::Error::custom)
    }
}

impl fmt::Debug for FiniteFieldEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteFieldEmbedding")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("generator_image", &self.generator_image)
            .finish()
    }
}

impl fmt::Display for FiniteFieldEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FiniteFieldEmbedding({} -> {}, generator -> {})",
            self.source, self.target, self.generator_image
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteFieldEmbeddingWire {
    #[serde(rename = "type")]
    type_id: String,
    source: FiniteField,
    target: FiniteField,
    generator_image: FiniteFieldElement,
}

impl From<&FiniteFieldEmbedding> for FiniteFieldEmbeddingWire {
    fn from(embedding: &FiniteFieldEmbedding) -> Self {
        Self {
            type_id: FiniteFieldEmbedding::TYPE_ID.to_owned(),
            source: embedding.source.clone(),
            target: embedding.target.clone(),
            generator_image: embedding.generator_image.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FiniteContext;

    #[test]
    fn embeds_a_quadratic_field_into_a_quartic_field() {
        let source = FiniteField::with_modulus(2, vec![1, 1, 1]).unwrap();
        let target = FiniteField::extension(2, 4).unwrap();
        let root = target
            .elements()
            .find(|candidate| {
                let square = target.mul(candidate, candidate).unwrap();
                target
                    .add(&target.add(&square, candidate).unwrap(), &target.one())
                    .unwrap()
                    .is_zero()
            })
            .unwrap();
        let embedding = FiniteFieldEmbedding::new(source.clone(), target, root).unwrap();
        assert_eq!(
            embedding.apply(&source.one()).unwrap(),
            embedding.target.one()
        );
        assert_eq!(
            embedding.apply(&source.generator()).unwrap(),
            embedding.generator_image
        );
        embedding.verify().unwrap();
        assert_eq!(
            FiniteFieldEmbedding::from_json(&embedding.to_json().unwrap()).unwrap(),
            embedding
        );
    }
}
