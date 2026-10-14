// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked maps between explicit presentations of one finite field.

use core::fmt;

use serde::{Deserialize, Serialize};

use super::{FiniteField, FiniteFieldElement};
use crate::{
    ContextMismatchError, FiniteContext, Integer, NoetherError, Ring, SemanticType, TypeDescription,
};

const MAX_ISOMORPHISM_ORDER: u64 = 4_096;

static FINITE_FIELD_ISOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/FiniteFieldIsomorphism@1",
    rust_path: "noether::numthy::FiniteFieldIsomorphism",
    python_path: Some("noether.numthy.FiniteFieldIsomorphism"),
};

/// Failure to construct or apply a checked finite-field presentation map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteFieldIsomorphismError {
    /// The source and target do not have the same characteristic and degree.
    IncompatibleFields {
        /// Source characteristic.
        source_characteristic: Integer,
        /// Source extension degree.
        source_degree: usize,
        /// Target characteristic.
        target_characteristic: Integer,
        /// Target extension degree.
        target_degree: usize,
    },
    /// The bounded deterministic root search would exceed the 0.1 limit.
    ResourceLimit {
        /// Exact field order.
        order: Integer,
        /// Largest accepted field order.
        limit: u64,
    },
    /// An input element belongs to a different presentation.
    ContextMismatch(ContextMismatchError),
    /// Serialized or constructed witness data failed exact verification.
    InvalidWitness(String),
}

impl fmt::Display for FiniteFieldIsomorphismError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleFields {
                source_characteristic,
                source_degree,
                target_characteristic,
                target_degree,
            } => write!(
                formatter,
                "finite-field presentations are not isomorphic: source has characteristic \
                 {source_characteristic} and degree {source_degree}, target has characteristic \
                 {target_characteristic} and degree {target_degree}"
            ),
            Self::ResourceLimit { order, limit } => write!(
                formatter,
                "finite-field presentation isomorphisms currently require field order at most \
                 {limit}; received {order}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidWitness(reason) => {
                write!(
                    formatter,
                    "invalid finite-field isomorphism witness: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for FiniteFieldIsomorphismError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for FiniteFieldIsomorphismError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// An exactly verified isomorphism between two explicit finite-field contexts.
///
/// The image of each distinguished power-basis generator determines one map.
/// Both directions are stored so applying the inverse never searches. Different
/// valid choices of a root are different maps, even when their endpoints agree.
#[derive(Clone, PartialEq, Eq)]
pub struct FiniteFieldIsomorphism {
    source: FiniteField,
    target: FiniteField,
    generator_image: FiniteFieldElement,
    inverse_generator_image: FiniteFieldElement,
}

impl FiniteFieldIsomorphism {
    pub(crate) fn between(
        source: FiniteField,
        target: FiniteField,
    ) -> Result<Self, FiniteFieldIsomorphismError> {
        require_compatible(&source, &target)?;

        let (generator_image, inverse_generator_image) = if source == target {
            (target.generator(), source.generator())
        } else {
            require_bounded_order(&source)?;
            let source_modulus = source.defining_modulus().ok_or_else(|| {
                FiniteFieldIsomorphismError::InvalidWitness(
                    "distinct prime-field presentations cannot occur".to_owned(),
                )
            })?;
            let generator_image = target
                .elements()
                .find(|candidate| {
                    evaluate_word_polynomial(&target, source_modulus, candidate)
                        .is_ok_and(|value| value.is_zero())
                })
                .ok_or_else(|| {
                    FiniteFieldIsomorphismError::InvalidWitness(
                        "the source defining polynomial has no root in the target".to_owned(),
                    )
                })?;
            let target_generator = target.generator();
            let inverse_generator_image = source
                .elements()
                .find(|candidate| {
                    apply_with_generator(&source, &target, &generator_image, candidate)
                        .is_ok_and(|image| image == target_generator)
                })
                .ok_or_else(|| {
                    FiniteFieldIsomorphismError::InvalidWitness(
                        "the forward map has no inverse image for the target generator".to_owned(),
                    )
                })?;
            (generator_image, inverse_generator_image)
        };

        let isomorphism = Self {
            source,
            target,
            generator_image,
            inverse_generator_image,
        };
        isomorphism.verify()?;
        Ok(isomorphism)
    }

    /// Source presentation.
    pub fn source(&self) -> &FiniteField {
        &self.source
    }

    /// Target presentation.
    pub fn target(&self) -> &FiniteField {
        &self.target
    }

    /// Image in the target of the source's distinguished generator.
    pub fn generator_image(&self) -> &FiniteFieldElement {
        &self.generator_image
    }

    /// Image in the source of the target's distinguished generator.
    pub fn inverse_generator_image(&self) -> &FiniteFieldElement {
        &self.inverse_generator_image
    }

    /// Apply the checked source-to-target map.
    pub fn apply(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<FiniteFieldElement, FiniteFieldIsomorphismError> {
        apply_with_generator(&self.source, &self.target, &self.generator_image, value)
    }

    /// Apply the checked target-to-source inverse map.
    pub fn apply_inverse(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<FiniteFieldElement, FiniteFieldIsomorphismError> {
        apply_with_generator(
            &self.target,
            &self.source,
            &self.inverse_generator_image,
            value,
        )
    }

    /// Recheck endpoints, defining-polynomial roots and both generator identities.
    pub fn verify(&self) -> Result<(), FiniteFieldIsomorphismError> {
        require_compatible(&self.source, &self.target)?;
        if self.source != self.target {
            require_bounded_order(&self.source)?;
        }
        require_element(&self.target, &self.generator_image)?;
        require_element(&self.source, &self.inverse_generator_image)?;

        if let Some(modulus) = self.source.defining_modulus() {
            if !evaluate_word_polynomial(&self.target, modulus, &self.generator_image)?.is_zero() {
                return Err(FiniteFieldIsomorphismError::InvalidWitness(
                    "the recorded generator image is not a root of the source modulus".to_owned(),
                ));
            }
        }
        if let Some(modulus) = self.target.defining_modulus() {
            if !evaluate_word_polynomial(&self.source, modulus, &self.inverse_generator_image)?
                .is_zero()
            {
                return Err(FiniteFieldIsomorphismError::InvalidWitness(
                    "the recorded inverse generator image is not a root of the target modulus"
                        .to_owned(),
                ));
            }
        }

        let source_generator = self.source.generator();
        let target_generator = self.target.generator();
        if self.apply(&source_generator)? != self.generator_image
            || self.apply_inverse(&target_generator)? != self.inverse_generator_image
            || self.apply_inverse(&self.generator_image)? != source_generator
            || self.apply(&self.inverse_generator_image)? != target_generator
        {
            return Err(FiniteFieldIsomorphismError::InvalidWitness(
                "the forward and inverse maps do not compose to the generator identities"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize both contexts and the two checked generator images.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteFieldIsomorphismWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reverify the complete map witness.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FiniteFieldIsomorphismWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let isomorphism = Self {
            source: wire.source,
            target: wire.target,
            generator_image: wire.generator_image,
            inverse_generator_image: wire.inverse_generator_image,
        };
        isomorphism
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(isomorphism)
    }
}

impl FiniteField {
    /// Construct a deterministic checked isomorphism to another presentation.
    ///
    /// The bounded 0.1 implementation chooses the first target root in canonical
    /// element order and supports distinct presentations of field order at most
    /// 4096. Identity maps have no size bound.
    pub fn isomorphism_to(
        &self,
        target: &Self,
    ) -> Result<FiniteFieldIsomorphism, FiniteFieldIsomorphismError> {
        FiniteFieldIsomorphism::between(self.clone(), target.clone())
    }
}

impl SemanticType for FiniteFieldIsomorphism {
    const TYPE_ID: &'static str = "noether/numthy/FiniteFieldIsomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_FIELD_ISOMORPHISM_DESCRIPTION
    }
}

impl fmt::Debug for FiniteFieldIsomorphism {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteFieldIsomorphism")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("generator_image", &self.generator_image)
            .field("inverse_generator_image", &self.inverse_generator_image)
            .finish()
    }
}

impl fmt::Display for FiniteFieldIsomorphism {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FiniteFieldIsomorphism({} -> {}, generator -> {})",
            self.source, self.target, self.generator_image
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteFieldIsomorphismWire {
    #[serde(rename = "type")]
    type_id: String,
    source: FiniteField,
    target: FiniteField,
    generator_image: FiniteFieldElement,
    inverse_generator_image: FiniteFieldElement,
}

impl From<&FiniteFieldIsomorphism> for FiniteFieldIsomorphismWire {
    fn from(isomorphism: &FiniteFieldIsomorphism) -> Self {
        Self {
            type_id: FiniteFieldIsomorphism::TYPE_ID.to_owned(),
            source: isomorphism.source.clone(),
            target: isomorphism.target.clone(),
            generator_image: isomorphism.generator_image.clone(),
            inverse_generator_image: isomorphism.inverse_generator_image.clone(),
        }
    }
}

fn require_compatible(
    source: &FiniteField,
    target: &FiniteField,
) -> Result<(), FiniteFieldIsomorphismError> {
    if source.characteristic() == target.characteristic() && source.degree() == target.degree() {
        Ok(())
    } else {
        Err(FiniteFieldIsomorphismError::IncompatibleFields {
            source_characteristic: source.characteristic().clone(),
            source_degree: source.degree(),
            target_characteristic: target.characteristic().clone(),
            target_degree: target.degree(),
        })
    }
}

fn require_bounded_order(field: &FiniteField) -> Result<(), FiniteFieldIsomorphismError> {
    let order = field.order();
    if order <= MAX_ISOMORPHISM_ORDER {
        Ok(())
    } else {
        Err(FiniteFieldIsomorphismError::ResourceLimit {
            order,
            limit: MAX_ISOMORPHISM_ORDER,
        })
    }
}

fn require_element(
    field: &FiniteField,
    value: &FiniteFieldElement,
) -> Result<(), FiniteFieldIsomorphismError> {
    if field.contains(value) {
        Ok(())
    } else {
        Err(ContextMismatchError {
            expected: field.to_string(),
            actual: value.field().to_string(),
        }
        .into())
    }
}

fn apply_with_generator(
    source: &FiniteField,
    target: &FiniteField,
    generator_image: &FiniteFieldElement,
    value: &FiniteFieldElement,
) -> Result<FiniteFieldElement, FiniteFieldIsomorphismError> {
    require_element(source, value)?;
    require_element(target, generator_image)?;
    evaluate_integer_polynomial(target, &value.coefficients(), generator_image)
}

fn evaluate_word_polynomial(
    field: &FiniteField,
    coefficients: &[u64],
    value: &FiniteFieldElement,
) -> Result<FiniteFieldElement, FiniteFieldIsomorphismError> {
    let coefficients = coefficients
        .iter()
        .copied()
        .map(Integer::from)
        .collect::<Vec<_>>();
    evaluate_integer_polynomial(field, &coefficients, value)
}

fn evaluate_integer_polynomial(
    field: &FiniteField,
    coefficients: &[Integer],
    value: &FiniteFieldElement,
) -> Result<FiniteFieldElement, FiniteFieldIsomorphismError> {
    require_element(field, value)?;
    let mut result = field.zero();
    for coefficient in coefficients.iter().rev() {
        result = field.mul(&result, value)?;
        result = field.add(&result, &field.element(coefficient.clone()))?;
    }
    Ok(result)
}
