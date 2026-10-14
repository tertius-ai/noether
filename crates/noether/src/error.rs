// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use core::fmt;

pub use flint_core::DivisionByZeroError;

/// An operation is unavailable for the supplied mathematical context or regime.
///
/// This is a capability result, not a failed computation: callers can inspect
/// the stable operation and context identifiers and choose another operation
/// without parsing the display message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedOperationError {
    /// Stable semantic operation identifier.
    pub operation: String,
    /// Stable semantic identifier of the rejected context.
    pub context: String,
    /// Stable semantic identifiers of contexts accepted by the operation.
    ///
    /// An empty list means that no alternate context is being offered. When
    /// [`Self::detail`] is present, the context family is accepted but this
    /// bounded input regime is not.
    pub supported_contexts: Vec<String>,
    /// Optional bounded-regime explanation when the context family is valid.
    pub detail: Option<String>,
}

impl UnsupportedOperationError {
    /// Construct an unsupported-operation result from semantic identifiers.
    pub fn new(operation: impl Into<String>, context: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            context: context.into(),
            supported_contexts: Vec::new(),
            detail: None,
        }
    }

    /// Attach the complete accepted context set for capability discovery.
    #[must_use]
    pub fn with_supported_contexts<I, S>(mut self, supported_contexts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.supported_contexts = supported_contexts.into_iter().map(Into::into).collect();
        self
    }

    /// Attach a concise explanation of the rejected bounded regime.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

impl fmt::Display for UnsupportedOperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operation {} is unsupported for context {}",
            self.operation, self.context
        )?;
        if let Some(detail) = &self.detail {
            write!(f, ": {detail}")?;
        }
        if self.supported_contexts.is_empty() {
            Ok(())
        } else {
            write!(
                f,
                "; supported contexts: {}",
                self.supported_contexts.join(", ")
            )
        }
    }
}

impl std::error::Error for UnsupportedOperationError {}

/// An operation received values belonging to different mathematical contexts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMismatchError {
    /// The context required by the operation.
    pub expected: String,
    /// The context carried by the rejected value.
    pub actual: String,
}

impl fmt::Display for ContextMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "context mismatch: expected {}, received {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for ContextMismatchError {}

/// Failure while inducing a checked scalar map through a composite value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoefficientMappingError {
    /// The composite value's coefficient context was not the map's source.
    ContextMismatch(ContextMismatchError),
    /// Reconstructing the corresponding target container context failed.
    TargetContext(String),
    /// Applying the scalar map to a source coefficient failed.
    Coefficient(String),
    /// Constructing the mapped composite value from valid mapped coefficients failed.
    Value(String),
}

impl fmt::Display for CoefficientMappingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::TargetContext(detail) => {
                write!(
                    formatter,
                    "cannot construct mapped coefficient context: {detail}"
                )
            }
            Self::Coefficient(detail) => {
                write!(formatter, "coefficient map application failed: {detail}")
            }
            Self::Value(detail) => {
                write!(
                    formatter,
                    "cannot construct coefficient-mapped value: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for CoefficientMappingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::TargetContext(_) | Self::Coefficient(_) | Self::Value(_) => None,
        }
    }
}

impl From<ContextMismatchError> for CoefficientMappingError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A field inversion received zero or a value from another context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InversionError {
    /// Zero has no multiplicative inverse.
    DivisionByZero(DivisionByZeroError),
    /// The value did not belong to the field used for inversion.
    ContextMismatch(ContextMismatchError),
}

impl fmt::Display for InversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero(error) => error.fmt(f),
            Self::ContextMismatch(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for InversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DivisionByZero(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
        }
    }
}

impl From<DivisionByZeroError> for InversionError {
    fn from(value: DivisionByZeroError) -> Self {
        Self::DivisionByZero(value)
    }
}

impl From<ContextMismatchError> for InversionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to invert an element in a ring's group of units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitError {
    /// Zero cannot be a unit in a nontrivial ring.
    DivisionByZero(DivisionByZeroError),
    /// The value belongs to the ring but is not invertible there.
    NotAUnit {
        /// Stable identity of the ring in which inversion was attempted.
        context: &'static str,
    },
    /// The value did not belong to the ring used for inversion.
    ContextMismatch(ContextMismatchError),
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero(error) => error.fmt(f),
            Self::NotAUnit { context } => write!(f, "element is not a unit in {context}"),
            Self::ContextMismatch(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for UnitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DivisionByZero(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::NotAUnit { .. } => None,
        }
    }
}

impl From<DivisionByZeroError> for UnitError {
    fn from(value: DivisionByZeroError) -> Self {
        Self::DivisionByZero(value)
    }
}

impl From<ContextMismatchError> for UnitError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<InversionError> for UnitError {
    fn from(value: InversionError) -> Self {
        match value {
            InversionError::DivisionByZero(error) => Self::DivisionByZero(error),
            InversionError::ContextMismatch(error) => Self::ContextMismatch(error),
        }
    }
}

/// A mathematical witness failed an explicit verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    /// One of the values does not belong to the context used for verification.
    ContextMismatch(ContextMismatchError),
    /// The claimed witness does not satisfy its defining identity.
    InvalidWitness {
        /// Stable, concise description of the failed claim.
        claim: &'static str,
    },
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::InvalidWitness { claim } => write!(f, "witness verification failed: {claim}"),
        }
    }
}

impl std::error::Error for VerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::InvalidWitness { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for VerificationError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Error shared by cross-domain facade operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoetherError {
    /// Canonical serialization failed.
    Serialization(String),
    /// Values from incompatible mathematical contexts were combined.
    ContextMismatch(ContextMismatchError),
    /// A mathematical witness did not verify.
    Verification(VerificationError),
    /// An operation is unavailable for a mathematical context or regime.
    UnsupportedOperation(UnsupportedOperationError),
}

impl fmt::Display for NoetherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialization(message) => write!(f, "serialization failed: {message}"),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Verification(error) => error.fmt(f),
            Self::UnsupportedOperation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for NoetherError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(_) => None,
            Self::ContextMismatch(error) => Some(error),
            Self::Verification(error) => Some(error),
            Self::UnsupportedOperation(error) => Some(error),
        }
    }
}

impl From<ContextMismatchError> for NoetherError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<VerificationError> for NoetherError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

impl From<UnsupportedOperationError> for NoetherError {
    fn from(value: UnsupportedOperationError) -> Self {
        Self::UnsupportedOperation(value)
    }
}

impl From<flint_core::ParseIntegerError> for NoetherError {
    fn from(value: flint_core::ParseIntegerError) -> Self {
        Self::Serialization(value.to_string())
    }
}

impl From<flint_core::ParseRationalError> for NoetherError {
    fn from(value: flint_core::ParseRationalError) -> Self {
        Self::Serialization(value.to_string())
    }
}
