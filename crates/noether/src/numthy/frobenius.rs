// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked coefficient Frobenius maps on fixed-cap local contexts.

use core::fmt;

use serde::{Deserialize, Serialize};

use super::{PadicElement, PadicField, PadicRing, QadicElement, QadicField, QadicRing};
use crate::{
    CoefficientFrobeniusContext, CoefficientMap, ContextMismatchError, Integer, NoetherError, Ring,
    SemanticType, TypeDescription,
};

static PADIC_RING_FROBENIUS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicRingFrobenius@1",
    rust_path: "noether::numthy::PadicRingFrobenius",
    python_path: Some("noether.numthy.PadicRingFrobenius"),
};

static PADIC_FIELD_FROBENIUS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicFieldFrobenius@1",
    rust_path: "noether::numthy::PadicFieldFrobenius",
    python_path: Some("noether.numthy.PadicFieldFrobenius"),
};

static QADIC_RING_FROBENIUS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicRingFrobenius@1",
    rust_path: "noether::numthy::QadicRingFrobenius",
    python_path: Some("noether.numthy.QadicRingFrobenius"),
};

static QADIC_FIELD_FROBENIUS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicFieldFrobenius@1",
    rust_path: "noether::numthy::QadicFieldFrobenius",
    python_path: Some("noether.numthy.QadicFieldFrobenius"),
};

/// Failure to apply or compose a coefficient Frobenius map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoefficientFrobeniusError {
    /// A coefficient or second map belonged to another context presentation.
    ContextMismatch(ContextMismatchError),
    /// Composition could not represent the sum of the two exponents in `u64`.
    ExponentOverflow {
        /// Exponent of the outer map.
        left: u64,
        /// Exponent of the inner map.
        right: u64,
    },
}

impl fmt::Display for CoefficientFrobeniusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::ExponentOverflow { left, right } => write!(
                formatter,
                "cannot compose coefficient Frobenius exponents {left} and {right}: their sum exceeds u64"
            ),
        }
    }
}

impl std::error::Error for CoefficientFrobeniusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::ExponentOverflow { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for CoefficientFrobeniusError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Coefficient Frobenius on a fixed-cap p-adic ring.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicRingFrobenius {
    context: PadicRing,
    exponent: u64,
}

/// Coefficient Frobenius on a fixed-cap p-adic field.
#[derive(Clone, PartialEq, Eq)]
pub struct PadicFieldFrobenius {
    context: PadicField,
    exponent: u64,
}

/// Coefficient Frobenius on a fixed-cap unramified Witt ring.
#[derive(Clone, PartialEq, Eq)]
pub struct QadicRingFrobenius {
    context: QadicRing,
    exponent: u64,
}

/// Coefficient Frobenius on a fixed-cap unramified field.
#[derive(Clone, PartialEq, Eq)]
pub struct QadicFieldFrobenius {
    context: QadicField,
    exponent: u64,
}

fn padic_parent(value: &PadicElement) -> String {
    value
        .ring()
        .map(ToString::to_string)
        .or_else(|| value.field().map(ToString::to_string))
        .unwrap_or_else(|| "unknown p-adic parent".to_owned())
}

fn qadic_parent(value: &QadicElement) -> String {
    value
        .ring()
        .map(ToString::to_string)
        .or_else(|| value.field().map(ToString::to_string))
        .unwrap_or_else(|| "unknown q-adic parent".to_owned())
}

fn require_type(expected: &str, actual: &str) -> Result<(), NoetherError> {
    if actual == expected {
        Ok(())
    } else {
        Err(NoetherError::Serialization(format!(
            "expected type {expected}, received {actual}"
        )))
    }
}

macro_rules! define_frobenius_map {
    (
        $name:ident,
        $context:ty,
        $element:ty,
        $wire:ident,
        $type_id:literal,
        $description:ident,
        $actual_parent:ident,
        $apply:expr
    ) => {
        #[derive(Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct $wire {
            #[serde(rename = "type")]
            type_id: String,
            context: $context,
            exponent: u64,
        }

        impl $name {
            /// Construct the declared Frobenius power on one exact context.
            pub fn new(context: $context, exponent: u64) -> Self {
                Self { context, exponent }
            }

            /// Exact source and target context of this endomorphism.
            pub fn context(&self) -> &$context {
                &self.context
            }

            /// Unreduced nonnegative Frobenius exponent recorded by this map.
            pub fn exponent(&self) -> u64 {
                self.exponent
            }

            /// Apply coefficient Frobenius after checking the exact context.
            pub fn apply(&self, value: &$element) -> Result<$element, CoefficientFrobeniusError> {
                if !self.context.contains(value) {
                    return Err(ContextMismatchError {
                        expected: self.context.to_string(),
                        actual: $actual_parent(value),
                    }
                    .into());
                }
                Ok(($apply)(value, self.exponent))
            }

            /// Compose `self` after `inner`, adding the recorded exponents.
            pub fn compose(&self, inner: &Self) -> Result<Self, CoefficientFrobeniusError> {
                if self.context != inner.context {
                    return Err(ContextMismatchError {
                        expected: self.context.to_string(),
                        actual: inner.context.to_string(),
                    }
                    .into());
                }
                let exponent = self.exponent.checked_add(inner.exponent).ok_or(
                    CoefficientFrobeniusError::ExponentOverflow {
                        left: self.exponent,
                        right: inner.exponent,
                    },
                )?;
                Ok(Self::new(self.context.clone(), exponent))
            }

            /// Canonical JSON containing the context presentation and exponent.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(&$wire {
                    type_id: Self::TYPE_ID.to_owned(),
                    context: self.context.clone(),
                    exponent: self.exponent,
                })
                .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Deserialize and revalidate the complete context presentation.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                let wire = serde_json::from_str::<$wire>(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))?;
                require_type(Self::TYPE_ID, &wire.type_id)?;
                Ok(Self::new(wire.context, wire.exponent))
            }
        }

        impl CoefficientMap for $name {
            type Source = $context;
            type Target = $context;
            type Error = CoefficientFrobeniusError;

            fn source(&self) -> &Self::Source {
                &self.context
            }

            fn target(&self) -> &Self::Target {
                &self.context
            }

            fn apply(&self, value: &$element) -> Result<$element, Self::Error> {
                self.apply(value)
            }
        }

        impl SemanticType for $name {
            const TYPE_ID: &'static str = $type_id;

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &$description
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("context", &self.context)
                    .field("exponent", &self.exponent)
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "coefficient Frobenius^{} on {}",
                    self.exponent, self.context
                )
            }
        }
    };
}

define_frobenius_map!(
    PadicRingFrobenius,
    PadicRing,
    PadicElement,
    PadicRingFrobeniusWire,
    "noether/numthy/PadicRingFrobenius@1",
    PADIC_RING_FROBENIUS_DESCRIPTION,
    padic_parent,
    |value: &PadicElement, _exponent: u64| value.clone()
);

define_frobenius_map!(
    PadicFieldFrobenius,
    PadicField,
    PadicElement,
    PadicFieldFrobeniusWire,
    "noether/numthy/PadicFieldFrobenius@1",
    PADIC_FIELD_FROBENIUS_DESCRIPTION,
    padic_parent,
    |value: &PadicElement, _exponent: u64| value.clone()
);

define_frobenius_map!(
    QadicRingFrobenius,
    QadicRing,
    QadicElement,
    QadicRingFrobeniusWire,
    "noether/numthy/QadicRingFrobenius@1",
    QADIC_RING_FROBENIUS_DESCRIPTION,
    qadic_parent,
    |value: &QadicElement, exponent: u64| value.frobenius(exponent)
);

define_frobenius_map!(
    QadicFieldFrobenius,
    QadicField,
    QadicElement,
    QadicFieldFrobeniusWire,
    "noether/numthy/QadicFieldFrobenius@1",
    QADIC_FIELD_FROBENIUS_DESCRIPTION,
    qadic_parent,
    |value: &QadicElement, exponent: u64| value.frobenius(exponent)
);

impl PadicRing {
    /// Checked coefficient Frobenius power, the identity on `Z_p` coefficients.
    pub fn coefficient_frobenius(&self, exponent: u64) -> PadicRingFrobenius {
        PadicRingFrobenius::new(self.clone(), exponent)
    }
}

impl PadicField {
    /// Checked coefficient Frobenius power, the identity on `Q_p` coefficients.
    pub fn coefficient_frobenius(&self, exponent: u64) -> PadicFieldFrobenius {
        PadicFieldFrobenius::new(self.clone(), exponent)
    }
}

impl QadicRing {
    /// Checked lift of residue-field Frobenius on this Witt-ring presentation.
    pub fn coefficient_frobenius(&self, exponent: u64) -> QadicRingFrobenius {
        QadicRingFrobenius::new(self.clone(), exponent)
    }
}

impl QadicField {
    /// Checked lift of residue-field Frobenius on this unramified field.
    pub fn coefficient_frobenius(&self, exponent: u64) -> QadicFieldFrobenius {
        QadicFieldFrobenius::new(self.clone(), exponent)
    }
}

impl CoefficientFrobeniusContext for PadicRing {
    type Frobenius = PadicRingFrobenius;

    fn coefficient_frobenius(&self, exponent: u64) -> Self::Frobenius {
        PadicRingFrobenius::new(self.clone(), exponent)
    }

    fn residue_characteristic(&self) -> &Integer {
        self.prime()
    }
}

impl CoefficientFrobeniusContext for PadicField {
    type Frobenius = PadicFieldFrobenius;

    fn coefficient_frobenius(&self, exponent: u64) -> Self::Frobenius {
        PadicFieldFrobenius::new(self.clone(), exponent)
    }

    fn residue_characteristic(&self) -> &Integer {
        self.prime()
    }
}

impl CoefficientFrobeniusContext for QadicRing {
    type Frobenius = QadicRingFrobenius;

    fn coefficient_frobenius(&self, exponent: u64) -> Self::Frobenius {
        QadicRingFrobenius::new(self.clone(), exponent)
    }

    fn residue_characteristic(&self) -> &Integer {
        self.prime()
    }
}

impl CoefficientFrobeniusContext for QadicField {
    type Frobenius = QadicFieldFrobenius;

    fn coefficient_frobenius(&self, exponent: u64) -> Self::Frobenius {
        QadicFieldFrobenius::new(self.clone(), exponent)
    }

    fn residue_characteristic(&self) -> &Integer {
        self.prime()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;

    #[test]
    fn padic_coefficient_frobenius_records_power_but_acts_as_identity() {
        let ring = PadicRing::new(5_u8, 4).unwrap();
        let map = ring.coefficient_frobenius(7);
        assert_eq!(map.exponent(), 7);
        assert_eq!(map.apply(&ring.element(-1)).unwrap(), ring.element(-1));
        assert_eq!(
            PadicRingFrobenius::from_json(&map.to_json().unwrap()).unwrap(),
            map
        );
    }

    #[test]
    fn qadic_frobenius_has_residue_degree_order_and_checked_composition() {
        let residue = FiniteField::extension(3_u64, 2).unwrap();
        let ring = QadicRing::new(residue, 5).unwrap();
        let sigma = ring.coefficient_frobenius(1);
        let generator = ring.generator();
        assert_ne!(sigma.apply(&generator).unwrap(), generator);
        let square = sigma.compose(&sigma).unwrap();
        assert_eq!(square.exponent(), 2);
        assert_eq!(square.apply(&generator).unwrap(), generator);
        assert_eq!(
            QadicRingFrobenius::from_json(&sigma.to_json().unwrap()).unwrap(),
            sigma
        );

        let other = QadicRing::new(FiniteField::extension(5_u64, 2).unwrap(), 5).unwrap();
        assert!(matches!(
            sigma.apply(&other.generator()),
            Err(CoefficientFrobeniusError::ContextMismatch(_))
        ));
        assert!(matches!(
            sigma.compose(&other.coefficient_frobenius(1)),
            Err(CoefficientFrobeniusError::ContextMismatch(_))
        ));
    }

    #[test]
    fn field_frobenius_preserves_negative_valuation() {
        let residue = FiniteField::extension(3_u64, 2).unwrap();
        let field = QadicField::new(residue, 6).unwrap();
        let value = field
            .inverse(&field.mul(&field.element(3), &field.generator()).unwrap())
            .unwrap();
        let mapped = field.coefficient_frobenius(1).apply(&value).unwrap();
        assert_eq!(mapped.field(), Some(&field));
        assert_eq!(mapped.valuation(), value.valuation());
    }
}
