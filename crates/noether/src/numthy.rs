// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Arithmetic fields, number fields, curves, and modular arithmetic.

mod elliptic_curve;
mod finite_field;
mod finite_field_embedding;
mod finite_field_isomorphism;
mod fractional_ideal;
mod frobenius;
mod integer;
mod number_field;
mod number_field_order;
mod padic;
mod padic_local_field;
mod padic_local_ideal;
mod padic_local_order;
mod prime_ideal;
mod qadic;

pub use crate::point_count::PointCountCertificate;
pub use elliptic_curve::{
    EllipticCurve, EllipticCurveError, EllipticCurvePoint, MillerFunction, NthPowerClass,
    NthPowerClassGroup, PowerClassError,
};
pub(crate) use finite_field::{
    DefaultExtensionElement, ExtensionContext, ExtensionElement, SmallExtensionTables,
};
pub use finite_field::{
    FiniteField, FiniteFieldConstructionError, FiniteFieldElement, InvalidCharacteristicError,
    InvalidFiniteFieldElementError,
};
pub use finite_field_embedding::{FiniteFieldEmbedding, FiniteFieldEmbeddingError};
pub use finite_field_isomorphism::{FiniteFieldIsomorphism, FiniteFieldIsomorphismError};
pub use fractional_ideal::{FractionalIdeal, FractionalIdealError, IdealFactorizationError};
pub use frobenius::{
    CoefficientFrobeniusError, PadicFieldFrobenius, PadicRingFrobenius, QadicFieldFrobenius,
    QadicRingFrobenius,
};
pub use integer::{
    cyclotomic_polynomial, factor, is_prime, partial_factor, CofactorClassification,
    CyclotomicPolynomialError, FactorError, IntegerFactorization, PartialFactorError,
    PartialIntegerFactorization,
};
pub use number_field::{NumberField, NumberFieldElement, NumberFieldError};
pub use number_field_order::{NumberFieldOrder, NumberFieldOrderError};
pub use padic::{
    HenselLiftError, PadicContextError, PadicElement, PadicEmbeddingError, PadicField,
    PadicFieldRestriction, PadicIntegralEmbedding, PadicPrecisionError, PadicRestrictionError,
    PadicRing, PadicRingRestriction, PadicUniformizerDivisionError, PadicValuation,
};
pub use padic_local_field::{
    PadicLocalCoefficientFrobeniusError, PadicLocalField, PadicLocalFieldCoefficientFrobenius,
    PadicLocalFieldElement, PadicLocalFieldError, PadicLocalFieldRestriction,
    PadicLocalFieldRestrictionError, PadicLocalInvariantError, PadicLocalPrecision,
    PadicLocalPrecisionError,
};
pub use padic_local_ideal::{PadicLocalFractionalIdeal, PadicLocalIdealError};
pub use padic_local_order::{PadicLocalFieldOrder, PadicLocalOrderError};
pub use prime_ideal::PrimeDecompositionError;
pub use qadic::{
    InvalidQadicCoordinatesError, QadicContextError, QadicElement, QadicEmbeddingError, QadicField,
    QadicFieldRestriction, QadicIntegralEmbedding, QadicRestrictionError, QadicRing,
    QadicRingRestriction, QadicUniformizerDivisionError,
};
