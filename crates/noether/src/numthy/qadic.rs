// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Fixed-cap Witt rings of finite fields.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::mem::MaybeUninit;
use std::sync::Arc;

use flint_sys::{
    fmpz_fdiv_ui, fmpz_poly_clear, fmpz_poly_init, fmpz_poly_set_coeff_fmpz, fmpz_poly_struct,
    nmod_poly_clear, nmod_poly_init, nmod_poly_set_coeff_ui, nmod_poly_struct,
    padic_print_mode_PADIC_TERSE, qadic_add, qadic_clear, qadic_ctx_clear,
    qadic_ctx_init_modulus_nmod, qadic_ctx_struct, qadic_frobenius, qadic_init2, qadic_inv,
    qadic_is_zero, qadic_mul, qadic_neg, qadic_norm, qadic_one, qadic_set, qadic_set_fmpz_poly,
    qadic_struct, qadic_sub, qadic_teichmuller, qadic_trace, qadic_zero,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::{
    FiniteField, FiniteFieldElement, PadicContextError, PadicElement, PadicField, PadicRing,
    PadicValuation,
};
use crate::{
    CommutativeRing, ContextMismatchError, Integer, InversionError, NoetherError, PrecisionContext,
    Ring, SemanticType, TypeDescription, UnitError, UnitInversion, ValuedRing,
};

mod context_maps;

pub use context_maps::{
    QadicEmbeddingError, QadicFieldRestriction, QadicIntegralEmbedding, QadicRingRestriction,
    QadicUniformizerDivisionError,
};

static QADIC_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicRing@1",
    rust_path: "noether::numthy::QadicRing",
    python_path: Some("noether.numthy.QadicRing"),
};

static QADIC_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicElement@1",
    rust_path: "noether::numthy::QadicElement",
    python_path: Some("noether.numthy.QadicElement"),
};

static QADIC_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/QadicField@1",
    rust_path: "noether::numthy::QadicField",
    python_path: Some("noether.numthy.QadicField"),
};

/// Failure to construct a fixed-cap Witt-ring context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QadicContextError {
    /// Absolute precision zero carries no q-adic digit information.
    ZeroAbsolutePrecision,
    /// The precision did not fit FLINT's signed machine-word interface.
    PrecisionTooLarge {
        /// Rejected absolute precision.
        absolute_precision: u64,
    },
    /// FLINT's q-adic implementation currently requires a word-size prime.
    CharacteristicTooLarge {
        /// Rejected residue characteristic.
        characteristic: Integer,
    },
    /// The base p-adic coefficient context could not be constructed.
    BaseContext(PadicContextError),
}

impl fmt::Display for QadicContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAbsolutePrecision => {
                f.write_str("q-adic absolute precision must be at least one")
            }
            Self::PrecisionTooLarge { absolute_precision } => write!(
                f,
                "q-adic absolute precision {absolute_precision} exceeds FLINT's supported range"
            ),
            Self::CharacteristicTooLarge { characteristic } => write!(
                f,
                "q-adic residue characteristic {characteristic} does not fit FLINT's word-size backend"
            ),
            Self::BaseContext(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for QadicContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BaseContext(error) => Some(error),
            _ => None,
        }
    }
}

/// Failure to discard q-adic digits through an explicit precision restriction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QadicRestrictionError {
    /// The value did not belong to the source Witt-ring context.
    ContextMismatch(ContextMismatchError),
    /// The requested precision was zero or exceeded the available precision.
    InvalidPrecision {
        /// Requested absolute precision.
        requested: u64,
        /// Absolute precision available on the source value.
        available: u64,
    },
    /// The target FLINT context could not be constructed.
    Context(QadicContextError),
    /// A source value could not be represented canonically at the target precision.
    Representation(String),
}

/// Failure to construct a q-adic value from p-adic power-basis coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidQadicCoordinatesError {
    /// More coordinates were supplied than the unramified degree permits.
    TooManyCoordinates {
        /// Number of supplied coordinates.
        received: usize,
        /// Degree of the q-adic context over its p-adic coefficient context.
        degree: usize,
    },
    /// One coordinate belonged to another p-adic context.
    CoordinateContext {
        /// Zero-based coordinate index.
        index: usize,
        /// Exact context mismatch.
        source: ContextMismatchError,
    },
    /// The coordinate data could not be represented canonically by the backend.
    Representation(String),
}

impl fmt::Display for InvalidQadicCoordinatesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyCoordinates { received, degree } => write!(
                formatter,
                "received {received} power-basis coordinates for a degree-{degree} q-adic context"
            ),
            Self::CoordinateContext { index, source } => {
                write!(formatter, "q-adic coordinate {index}: {source}")
            }
            Self::Representation(reason) => {
                write!(
                    formatter,
                    "invalid q-adic power-basis coordinates: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for InvalidQadicCoordinatesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CoordinateContext { source, .. } => Some(source),
            Self::TooManyCoordinates { .. } | Self::Representation(_) => None,
        }
    }
}

impl fmt::Display for QadicRestrictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::InvalidPrecision {
                requested,
                available,
            } => write!(
                f,
                "cannot restrict q-adic precision {available} to {requested}; the target must lie in 1..={available}"
            ),
            Self::Context(error) => error.fmt(f),
            Self::Representation(reason) => {
                write!(f, "cannot restrict q-adic value: {reason}")
            }
        }
    }
}

impl std::error::Error for QadicRestrictionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Context(error) => Some(error),
            Self::InvalidPrecision { .. } | Self::Representation(_) => None,
        }
    }
}

struct NmodPolynomial {
    raw: nmod_poly_struct,
}

impl NmodPolynomial {
    fn new(characteristic: u64, coefficients: &[u64]) -> Self {
        let mut raw = MaybeUninit::<nmod_poly_struct>::uninit();
        unsafe { nmod_poly_init(raw.as_mut_ptr(), characteristic) };
        let mut output = Self {
            raw: unsafe { raw.assume_init() },
        };
        for (degree, coefficient) in coefficients.iter().copied().enumerate() {
            unsafe { nmod_poly_set_coeff_ui(&mut output.raw, degree as i64, coefficient) };
        }
        output
    }
}

impl Drop for NmodPolynomial {
    fn drop(&mut self) {
        unsafe { nmod_poly_clear(&mut self.raw) };
    }
}

struct FmpzPolynomial {
    raw: fmpz_poly_struct,
}

impl FmpzPolynomial {
    fn from_coefficients(coefficients: &[Integer]) -> Self {
        let mut raw = MaybeUninit::<fmpz_poly_struct>::uninit();
        unsafe { fmpz_poly_init(raw.as_mut_ptr()) };
        let mut output = Self {
            raw: unsafe { raw.assume_init() },
        };
        for (degree, coefficient) in coefficients.iter().enumerate() {
            unsafe {
                fmpz_poly_set_coeff_fmpz(&mut output.raw, degree as i64, coefficient.as_ptr())
            };
        }
        output
    }
}

impl Drop for FmpzPolynomial {
    fn drop(&mut self) {
        unsafe { fmpz_poly_clear(&mut self.raw) };
    }
}

struct QadicContext {
    raw: qadic_ctx_struct,
    characteristic: u64,
    precision: i64,
}

impl QadicContext {
    fn new(residue_field: &FiniteField, characteristic: u64, precision: i64) -> Self {
        let modulus = residue_field
            .defining_modulus()
            .map_or_else(|| vec![0, 1], <[u64]>::to_vec);
        let modulus = NmodPolynomial::new(characteristic, &modulus);
        let mut raw = MaybeUninit::<qadic_ctx_struct>::uninit();
        unsafe {
            qadic_ctx_init_modulus_nmod(
                raw.as_mut_ptr(),
                characteristic,
                &modulus.raw,
                0,
                precision,
                c"a".as_ptr(),
                padic_print_mode_PADIC_TERSE,
            )
        };
        Self {
            raw: unsafe { raw.assume_init() },
            characteristic,
            precision,
        }
    }

    fn as_ptr(&self) -> *const qadic_ctx_struct {
        &self.raw
    }
}

impl Drop for QadicContext {
    fn drop(&mut self) {
        unsafe { qadic_ctx_clear(&mut self.raw) };
    }
}

unsafe impl Send for QadicContext {}
unsafe impl Sync for QadicContext {}

struct QadicValue {
    raw: qadic_struct,
    context: Arc<QadicContext>,
}

impl QadicValue {
    fn initialized(context: Arc<QadicContext>) -> Self {
        let precision = context.precision;
        Self::initialized_at(context, precision)
    }

    fn initialized_at(context: Arc<QadicContext>, precision: i64) -> Self {
        let mut raw = MaybeUninit::<qadic_struct>::uninit();
        unsafe { qadic_init2(raw.as_mut_ptr(), precision) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    fn zero(context: Arc<QadicContext>) -> Self {
        let mut output = Self::initialized(context);
        unsafe { qadic_zero(&mut output.raw) };
        output
    }

    fn one(context: Arc<QadicContext>) -> Self {
        let mut output = Self::initialized(context);
        unsafe { qadic_one(&mut output.raw) };
        output
    }

    fn from_coefficients(context: Arc<QadicContext>, coefficients: &[Integer]) -> Self {
        let polynomial = FmpzPolynomial::from_coefficients(coefficients);
        let mut output = Self::initialized(context);
        unsafe { qadic_set_fmpz_poly(&mut output.raw, &polynomial.raw, output.context.as_ptr()) };
        output
    }

    fn from_unit_coordinates(
        context: Arc<QadicContext>,
        coefficients: &[Integer],
        valuation: i64,
    ) -> Result<Self, String> {
        if coefficients.iter().all(Integer::is_zero) {
            if valuation != 0 {
                return Err("zero q-adic coordinates must have valuation zero".to_owned());
            }
            return Ok(Self::zero(context));
        }
        let unit_precision = context
            .precision
            .checked_sub(valuation)
            .filter(|precision| *precision > 0)
            .ok_or_else(|| {
                "q-adic valuation is incompatible with the fixed absolute precision".to_owned()
            })?;
        let polynomial = FmpzPolynomial::from_coefficients(coefficients);
        let mut output = Self::initialized_at(context, unit_precision);
        unsafe { qadic_set_fmpz_poly(&mut output.raw, &polynomial.raw, output.context.as_ptr()) };
        if output.raw.val != 0 || output.unit_coefficients(coefficients.len()) != coefficients {
            return Err("q-adic unit coordinates are not in canonical FLINT form".to_owned());
        }
        output.raw.N = output.context.precision;
        output.raw.val = valuation;
        Ok(output)
    }

    fn binary(
        context: Arc<QadicContext>,
        left: &Self,
        right: &Self,
        operation: unsafe extern "C" fn(
            *mut qadic_struct,
            *const qadic_struct,
            *const qadic_struct,
            *const qadic_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::initialized(context);
        unsafe {
            operation(
                &mut output.raw,
                &left.raw,
                &right.raw,
                output.context.as_ptr(),
            )
        };
        output
    }

    fn unary(
        context: Arc<QadicContext>,
        value: &Self,
        operation: unsafe extern "C" fn(
            *mut qadic_struct,
            *const qadic_struct,
            *const qadic_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::initialized(context);
        unsafe { operation(&mut output.raw, &value.raw, output.context.as_ptr()) };
        output
    }

    fn unit_coefficients(&self, degree: usize) -> Vec<Integer> {
        (0..degree)
            .map(|index| {
                if index < self.raw.length as usize {
                    unsafe { Integer::clone_from_ptr(self.raw.coeffs.add(index)) }
                } else {
                    Integer::zero()
                }
            })
            .collect()
    }

    fn integral_coefficients(&self, degree: usize) -> Option<Vec<Integer>> {
        if self.raw.length == 0 {
            return Some(vec![Integer::zero(); degree]);
        }
        let valuation = u64::try_from(self.raw.val).ok()?;
        let scale = Integer::from(self.context.characteristic).pow(valuation);
        Some(
            self.unit_coefficients(degree)
                .into_iter()
                .map(|unit| &unit * &scale)
                .collect(),
        )
    }
}

impl Clone for QadicValue {
    fn clone(&self) -> Self {
        let mut output = Self::initialized(Arc::clone(&self.context));
        unsafe { qadic_set(&mut output.raw, &self.raw, output.context.as_ptr()) };
        output
    }
}

impl Drop for QadicValue {
    fn drop(&mut self) {
        unsafe { qadic_clear(&mut self.raw) };
    }
}

unsafe impl Send for QadicValue {}
unsafe impl Sync for QadicValue {}

/// The fixed-cap Witt ring `W(F_q) / p^N`.
///
/// The complete residue-field presentation and the absolute precision cap are
/// context identity. Elements use FLINT's `qadic_t` backend but are restricted
/// to nonnegative valuation, so every power-basis coefficient has a fixed
/// `N`-digit p-adic storage bound.
#[derive(Clone)]
pub struct QadicRing {
    residue_field: FiniteField,
    coefficient_ring: PadicRing,
    absolute_precision: u64,
    context: Arc<QadicContext>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicRingWire {
    #[serde(rename = "type")]
    type_id: String,
    residue_field: super::finite_field::FiniteFieldWire,
    absolute_precision: u64,
}

impl QadicRing {
    /// Construct `W(residue_field) / p^N` from a verified finite-field
    /// presentation.
    pub fn new(
        residue_field: FiniteField,
        absolute_precision: u64,
    ) -> Result<Self, QadicContextError> {
        if absolute_precision == 0 {
            return Err(QadicContextError::ZeroAbsolutePrecision);
        }
        let precision = i64::try_from(absolute_precision)
            .map_err(|_| QadicContextError::PrecisionTooLarge { absolute_precision })?;
        let characteristic = residue_field.word_characteristic().ok_or_else(|| {
            QadicContextError::CharacteristicTooLarge {
                characteristic: residue_field.characteristic().clone(),
            }
        })?;
        let coefficient_ring =
            PadicRing::new(residue_field.characteristic().clone(), absolute_precision)
                .map_err(QadicContextError::BaseContext)?;
        let context = Arc::new(QadicContext::new(&residue_field, characteristic, precision));
        Ok(Self {
            residue_field,
            coefficient_ring,
            absolute_precision,
            context,
        })
    }

    /// Residue field `F_q`, including its normalized defining presentation.
    pub fn residue_field(&self) -> &FiniteField {
        &self.residue_field
    }

    /// Residue characteristic `p`.
    pub fn prime(&self) -> &Integer {
        self.residue_field.characteristic()
    }

    /// Residue degree `[F_q : F_p]`.
    pub fn residue_degree(&self) -> usize {
        self.residue_field.degree()
    }

    /// Base p-adic integer ring supplying the power-basis coefficients.
    pub fn coefficient_ring(&self) -> &PadicRing {
        &self.coefficient_ring
    }

    /// Immutable absolute precision cap `N`.
    pub fn absolute_precision(&self) -> u64 {
        self.absolute_precision
    }

    /// Exact cardinality `q^N` of the truncated Witt ring.
    pub fn cardinality(&self) -> Integer {
        self.residue_field.order().pow(self.absolute_precision)
    }

    /// Embed an integer through `Z_p -> W(F_q)`.
    pub fn element<T: Into<Integer>>(&self, value: T) -> QadicElement {
        self.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.context),
            &[value.into()],
        ))
    }

    /// Construct an element from ascending `Z_p` coordinates in the declared
    /// q-adic power basis.
    ///
    /// Missing high coordinates are zero, and at most [`residue_degree`](Self::residue_degree)
    /// coordinates may be supplied. Every coordinate must belong to this
    /// context's exact coefficient ring.
    pub fn element_from_coefficients(
        &self,
        coefficients: &[PadicElement],
    ) -> Result<QadicElement, InvalidQadicCoordinatesError> {
        validate_coordinate_count(coefficients.len(), self.residue_degree())?;
        let mut representatives = Vec::with_capacity(coefficients.len());
        for (index, coefficient) in coefficients.iter().enumerate() {
            if coefficient.ring() != Some(&self.coefficient_ring) {
                return Err(InvalidQadicCoordinatesError::CoordinateContext {
                    index,
                    source: ContextMismatchError {
                        expected: self.coefficient_ring.to_string(),
                        actual: padic_parent(coefficient),
                    },
                });
            }
            representatives.push(
                coefficient
                    .representative()
                    .expect("a coordinate in a p-adic ring has an integral representative"),
            );
        }
        Ok(self.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.context),
            &representatives,
        )))
    }

    /// Serialize the residue-field presentation and absolute precision as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision Witt-ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> QadicRingWire {
        QadicRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            residue_field: self.residue_field.to_wire(),
            absolute_precision: self.absolute_precision,
        }
    }

    fn from_wire(wire: QadicRingWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let residue_field = FiniteField::from_wire(wire.residue_field)?;
        Self::new(residue_field, wire.absolute_precision).map_err(|error| error.to_string())
    }

    /// Reduce an integral q-adic value to `F_q`.
    pub fn residue(
        &self,
        value: &QadicElement,
    ) -> Result<FiniteFieldElement, ContextMismatchError> {
        self.require(value)?;
        self.residue_value(value)
            .ok_or_else(|| ContextMismatchError {
                expected: format!("an integral value in {self}"),
                actual: format!("a value of valuation {}", value.value.raw.val),
            })
    }

    fn residue_value(&self, value: &QadicElement) -> Option<FiniteFieldElement> {
        if value.value.raw.val < 0 {
            return None;
        }
        if value.value.raw.length == 0 || value.value.raw.val > 0 {
            return Some(self.residue_field.zero());
        }
        let coefficients = (0..self.residue_degree())
            .map(|index| {
                if index < value.value.raw.length as usize {
                    unsafe {
                        fmpz_fdiv_ui(
                            value.value.raw.coeffs.add(index),
                            value.value.context.characteristic,
                        )
                    }
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        self.residue_field
            .element_from_coefficients(&coefficients)
            .ok()
    }

    /// Lift a residue-field value to its Teichmüller representative.
    pub fn teichmuller(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<QadicElement, ContextMismatchError> {
        if value.field() != &self.residue_field {
            return Err(ContextMismatchError {
                expected: self.residue_field.to_string(),
                actual: value.field().to_string(),
            });
        }
        let coefficients = value.coefficients();
        let lift = QadicValue::from_coefficients(Arc::clone(&self.context), &coefficients);
        Ok(self.wrap(QadicValue::unary(
            Arc::clone(&self.context),
            &lift,
            qadic_teichmuller,
        )))
    }

    /// Distinguished Teichmüller lift of the residue-field power-basis
    /// generator.
    pub fn generator(&self) -> QadicElement {
        self.teichmuller(&self.residue_field.generator())
            .expect("a residue-field generator belongs to its own field")
    }

    /// Explicitly discard high p-adic digits.
    pub fn restrict(
        &self,
        value: &QadicElement,
        absolute_precision: u64,
    ) -> Result<QadicElement, QadicRestrictionError> {
        self.precision_restriction(absolute_precision)?.apply(value)
    }

    fn require(&self, value: &QadicElement) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.parent.to_string(),
            })
        }
    }

    fn wrap(&self, value: QadicValue) -> QadicElement {
        debug_assert!(value.raw.length == 0 || value.raw.val >= 0);
        QadicElement {
            parent: QadicParent::Ring(self.clone()),
            value,
        }
    }
}

impl Serialize for QadicRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QadicRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(QadicRingWire::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl PartialEq for QadicRing {
    fn eq(&self, other: &Self) -> bool {
        self.residue_field == other.residue_field
            && self.absolute_precision == other.absolute_precision
    }
}

impl Eq for QadicRing {}

impl fmt::Debug for QadicRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QadicRing")
            .field("residue_field", &self.residue_field)
            .field("absolute_precision", &self.absolute_precision)
            .finish()
    }
}

impl fmt::Display for QadicRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "W({}) modulo {}^{}",
            self.residue_field,
            self.prime(),
            self.absolute_precision
        )
    }
}

impl SemanticType for QadicRing {
    const TYPE_ID: &'static str = "noether/numthy/QadicRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.residue_field.canonical_bytes()?)?;
        output.extend_from_slice(&self.absolute_precision.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &QADIC_RING_DESCRIPTION
    }
}

impl Ring for QadicRing {
    type Element = QadicElement;

    fn contains(&self, value: &Self::Element) -> bool {
        matches!(&value.parent, QadicParent::Ring(ring) if self == ring)
    }

    fn zero(&self) -> Self::Element {
        self.wrap(QadicValue::zero(Arc::clone(&self.context)))
    }

    fn one(&self) -> Self::Element {
        self.wrap(QadicValue::one(Arc::clone(&self.context)))
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            qadic_add,
        )))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            qadic_sub,
        )))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            qadic_mul,
        )))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(self.wrap(QadicValue::unary(
            Arc::clone(&self.context),
            &value.value,
            qadic_neg,
        )))
    }
}

impl CommutativeRing for QadicRing {}

impl UnitInversion for QadicRing {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)?;
        value.inverse().map_err(|_| UnitError::NotAUnit {
            context: Self::TYPE_ID,
        })
    }
}

impl PrecisionContext for QadicRing {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.absolute_precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

impl ValuedRing for QadicRing {
    fn uniformizer(&self) -> Self::Element {
        self.element(self.prime().clone())
    }

    fn valuation_lower_bound(&self, value: &Self::Element) -> Option<i64> {
        match value.valuation() {
            PadicValuation::Exact(valuation) => Some(valuation),
            PadicValuation::AtLeast(_) => None,
        }
    }
}

/// A finite-precision model of the unramified p-adic field with the given residue field.
///
/// The complete `F_q` presentation and absolute precision cap are context
/// identity. Negative valuations are permitted, so this context does not have
/// the context-only storage bound of [`QadicRing`]. Values indistinguishable at
/// the cap compare equal, so it deliberately does not implement the exact
/// [`crate::Field`] trait. Exact field algorithms must use precision-aware
/// contracts.
///
/// ```compile_fail
/// use noether::numthy::{FiniteField, QadicField};
/// use noether::Field;
///
/// fn needs_exact_field<R: Field>(_: &R) {}
/// let fq = FiniteField::extension(3_u8, 2).unwrap();
/// needs_exact_field(&QadicField::new(fq, 8).unwrap());
/// ```
#[derive(Clone)]
pub struct QadicField {
    integral_ring: QadicRing,
    coefficient_field: PadicField,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicFieldWire {
    #[serde(rename = "type")]
    type_id: String,
    residue_field: super::finite_field::FiniteFieldWire,
    absolute_precision: u64,
}

impl QadicField {
    /// Construct the unramified field with residue field `F_q` at absolute
    /// precision `N`.
    pub fn new(
        residue_field: FiniteField,
        absolute_precision: u64,
    ) -> Result<Self, QadicContextError> {
        let integral_ring = QadicRing::new(residue_field, absolute_precision)?;
        let coefficient_field =
            PadicField::from_integral_ring(integral_ring.coefficient_ring.clone());
        Ok(Self {
            integral_ring,
            coefficient_field,
        })
    }

    /// Residue field `F_q`, including its normalized defining presentation.
    pub fn residue_field(&self) -> &FiniteField {
        self.integral_ring.residue_field()
    }

    /// Base field `Q_p` supplying the power-basis coefficients.
    pub fn coefficient_field(&self) -> &PadicField {
        &self.coefficient_field
    }

    /// Fixed-cap Witt ring canonically embedded in this field context.
    pub fn integral_ring(&self) -> &QadicRing {
        &self.integral_ring
    }

    /// Residue characteristic `p`.
    pub fn prime(&self) -> &Integer {
        self.integral_ring.prime()
    }

    /// Residue degree `[F_q : F_p]`.
    pub fn residue_degree(&self) -> usize {
        self.integral_ring.residue_degree()
    }

    /// Immutable absolute precision cap `N`.
    pub fn absolute_precision(&self) -> u64 {
        self.integral_ring.absolute_precision()
    }

    /// Embed an integer through `Q_p -> Q_q`.
    pub fn element<T: Into<Integer>>(&self, value: T) -> QadicElement {
        self.wrap(QadicValue::from_coefficients(
            Arc::clone(&self.integral_ring.context),
            &[value.into()],
        ))
    }

    /// Construct an element from ascending `Q_p` coordinates in the declared
    /// q-adic power basis.
    ///
    /// Missing high coordinates are zero, and at most [`residue_degree`](Self::residue_degree)
    /// coordinates may be supplied. Every coordinate must belong to this
    /// context's exact coefficient field.
    pub fn element_from_coefficients(
        &self,
        coefficients: &[PadicElement],
    ) -> Result<QadicElement, InvalidQadicCoordinatesError> {
        validate_coordinate_count(coefficients.len(), self.residue_degree())?;
        let mut result = self.zero();
        for (index, coefficient) in coefficients.iter().enumerate() {
            if coefficient.field() != Some(&self.coefficient_field) {
                return Err(InvalidQadicCoordinatesError::CoordinateContext {
                    index,
                    source: ContextMismatchError {
                        expected: self.coefficient_field.to_string(),
                        actual: padic_parent(coefficient),
                    },
                });
            }
            let Some((unit, valuation)) = coefficient.unit_and_valuation() else {
                continue;
            };
            let mut unit_coordinates = vec![Integer::zero(); self.residue_degree()];
            unit_coordinates[index] = unit;
            let value = QadicValue::from_unit_coordinates(
                Arc::clone(&self.integral_ring.context),
                &unit_coordinates,
                valuation,
            )
            .map_err(InvalidQadicCoordinatesError::Representation)?;
            let term = self.wrap(value);
            result = self
                .add(&result, &term)
                .expect("q-adic coordinate terms share one field context");
        }
        Ok(result)
    }

    /// Invert a value that is nonzero at the available precision.
    ///
    /// This is finite-precision arithmetic, not a proof that the represented
    /// value is nonzero in the complete unramified field. Consequently
    /// [`QadicField`] deliberately does not implement the exact
    /// [`crate::Field`] trait.
    pub fn inverse(&self, value: &QadicElement) -> Result<QadicElement, InversionError> {
        self.require(value)?;
        if unsafe { qadic_is_zero(&value.value.raw) != 0 } {
            return Err(crate::DivisionByZeroError.into());
        }
        Ok(self.wrap(QadicValue::unary(
            Arc::clone(&self.integral_ring.context),
            &value.value,
            qadic_inv,
        )))
    }

    /// Explicitly discard high p-adic digits into a distinct field context.
    pub fn restrict(
        &self,
        value: &QadicElement,
        absolute_precision: u64,
    ) -> Result<QadicElement, QadicRestrictionError> {
        self.precision_restriction(absolute_precision)?.apply(value)
    }

    /// Serialize the residue-field presentation and absolute precision as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate an unramified fixed-precision field context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> QadicFieldWire {
        QadicFieldWire {
            type_id: Self::TYPE_ID.to_owned(),
            residue_field: self.residue_field().to_wire(),
            absolute_precision: self.absolute_precision(),
        }
    }

    fn from_wire(wire: QadicFieldWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let residue_field = FiniteField::from_wire(wire.residue_field)?;
        Self::new(residue_field, wire.absolute_precision).map_err(|error| error.to_string())
    }

    /// Reduce an integral value to `F_q`; nonintegral values have no residue.
    pub fn residue(
        &self,
        value: &QadicElement,
    ) -> Result<Option<FiniteFieldElement>, ContextMismatchError> {
        self.require(value)?;
        Ok(self.integral_ring.residue_value(value))
    }

    /// Lift a residue-field value to its Teichmüller representative.
    pub fn teichmuller(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<QadicElement, ContextMismatchError> {
        if value.field() != self.residue_field() {
            return Err(ContextMismatchError {
                expected: self.residue_field().to_string(),
                actual: value.field().to_string(),
            });
        }
        let coefficients = value.coefficients();
        let lift =
            QadicValue::from_coefficients(Arc::clone(&self.integral_ring.context), &coefficients);
        Ok(self.wrap(QadicValue::unary(
            Arc::clone(&self.integral_ring.context),
            &lift,
            qadic_teichmuller,
        )))
    }

    /// Distinguished Teichmüller lift of the residue-field generator.
    pub fn generator(&self) -> QadicElement {
        self.teichmuller(&self.residue_field().generator())
            .expect("a residue-field generator belongs to its own field")
    }

    fn require(&self, value: &QadicElement) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.parent.to_string(),
            })
        }
    }

    fn wrap(&self, value: QadicValue) -> QadicElement {
        QadicElement {
            parent: QadicParent::Field(self.clone()),
            value,
        }
    }
}

impl Serialize for QadicField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QadicField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(QadicFieldWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for QadicField {
    fn eq(&self, other: &Self) -> bool {
        self.integral_ring == other.integral_ring
    }
}

impl Eq for QadicField {}

impl fmt::Debug for QadicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QadicField")
            .field("residue_field", &self.residue_field())
            .field("absolute_precision", &self.absolute_precision())
            .finish()
    }
}

impl fmt::Display for QadicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unramified degree-{} extension of Q_{} at absolute precision {} with residue field {}",
            self.residue_degree(),
            self.prime(),
            self.absolute_precision(),
            self.residue_field()
        )
    }
}

impl SemanticType for QadicField {
    const TYPE_ID: &'static str = "noether/numthy/QadicField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.integral_ring.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &QADIC_FIELD_DESCRIPTION
    }
}

impl Ring for QadicField {
    type Element = QadicElement;

    fn contains(&self, value: &Self::Element) -> bool {
        matches!(&value.parent, QadicParent::Field(field) if self == field)
    }

    fn zero(&self) -> Self::Element {
        self.wrap(QadicValue::zero(Arc::clone(&self.integral_ring.context)))
    }

    fn one(&self) -> Self::Element {
        self.wrap(QadicValue::one(Arc::clone(&self.integral_ring.context)))
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            qadic_add,
        )))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            qadic_sub,
        )))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(QadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            qadic_mul,
        )))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(self.wrap(QadicValue::unary(
            Arc::clone(&self.integral_ring.context),
            &value.value,
            qadic_neg,
        )))
    }
}

impl CommutativeRing for QadicField {}

impl UnitInversion for QadicField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(Into::into)
    }
}

impl PrecisionContext for QadicField {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.integral_ring.absolute_precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

impl ValuedRing for QadicField {
    fn uniformizer(&self) -> Self::Element {
        self.element(self.prime().clone())
    }

    fn valuation_lower_bound(&self, value: &Self::Element) -> Option<i64> {
        match value.valuation() {
            PadicValuation::Exact(valuation) => Some(valuation),
            PadicValuation::AtLeast(_) => None,
        }
    }
}

#[derive(Clone)]
enum QadicParent {
    Ring(QadicRing),
    Field(QadicField),
}

impl fmt::Display for QadicParent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ring(ring) => ring.fmt(f),
            Self::Field(field) => field.fmt(f),
        }
    }
}

/// A fixed-cap q-adic value carrying its concrete ring or field context.
#[derive(Clone)]
pub struct QadicElement {
    parent: QadicParent,
    value: QadicValue,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QadicElementWire {
    #[serde(rename = "type")]
    type_id: String,
    value: QadicElementValueWire,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "parent", rename_all = "snake_case", deny_unknown_fields)]
enum QadicElementValueWire {
    Ring {
        context: QadicRingWire,
        coefficients: Vec<String>,
    },
    Field {
        context: QadicFieldWire,
        unit_coefficients: Vec<String>,
        valuation: i64,
    },
}

impl QadicElement {
    /// Concrete Witt-ring context containing this value, or `None` for a
    /// [`QadicField`] element.
    pub fn ring(&self) -> Option<&QadicRing> {
        match &self.parent {
            QadicParent::Ring(ring) => Some(ring),
            QadicParent::Field(_) => None,
        }
    }

    /// Concrete unramified field containing this value, or `None` for a
    /// [`QadicRing`] element.
    pub fn field(&self) -> Option<&QadicField> {
        match &self.parent {
            QadicParent::Ring(_) => None,
            QadicParent::Field(field) => Some(field),
        }
    }

    /// Power-basis coefficients in `Z_p` or `Q_p`, matching the concrete
    /// parent context.
    pub fn coefficients(&self) -> Vec<PadicElement> {
        match &self.parent {
            QadicParent::Ring(ring) => self
                .value
                .integral_coefficients(ring.residue_degree())
                .expect("Witt-ring values retain nonnegative valuation")
                .into_iter()
                .map(|coefficient| ring.coefficient_ring.element(coefficient))
                .collect(),
            QadicParent::Field(field) => self
                .value
                .unit_coefficients(field.residue_degree())
                .into_iter()
                .map(|unit| {
                    field
                        .coefficient_field
                        .element_from_unit_and_valuation(&unit, self.value.raw.val)
                })
                .collect(),
        }
    }

    /// Split a nonzero field value into its q-adic valuation and normalized
    /// integral unit at the full surviving relative precision.
    ///
    /// This is crate-internal support for ramified local-field embeddings. It
    /// does not manufacture a public precision-refinement map: the returned
    /// ring may have a larger absolute cap than the source field because it
    /// stores the unit after removing a negative power of `p`.
    pub(crate) fn normalized_integral_unit(&self) -> Option<(i64, QadicElement)> {
        let field = self.field()?;
        if self.value.raw.length == 0 {
            return None;
        }
        let valuation = self.value.raw.val;
        let relative_precision = i128::from(field.absolute_precision()) - i128::from(valuation);
        let relative_precision = u64::try_from(relative_precision)
            .expect("a nonzero q-adic value has positive representable relative precision");
        let ring = QadicRing::new(field.residue_field().clone(), relative_precision)
            .expect("a surviving q-adic relative precision is positive and representable");
        let unit = ring.wrap(QadicValue::from_coefficients(
            Arc::clone(&ring.context),
            &self.value.unit_coefficients(field.residue_degree()),
        ));
        Some((valuation, unit))
    }

    /// Known absolute precision, equal to the immutable context cap.
    pub fn known_absolute_precision(&self) -> u64 {
        match &self.parent {
            QadicParent::Ring(ring) => ring.absolute_precision(),
            QadicParent::Field(field) => field.absolute_precision(),
        }
    }

    /// Residue in the explicit finite residue field, or `None` when this value
    /// has negative valuation.
    pub fn residue(&self) -> Option<FiniteFieldElement> {
        match &self.parent {
            QadicParent::Ring(ring) => ring.residue_value(self),
            QadicParent::Field(field) => field.integral_ring.residue_value(self),
        }
    }

    /// Precision-aware p-adic valuation.
    pub fn valuation(&self) -> PadicValuation {
        if self.value.raw.length == 0 {
            PadicValuation::AtLeast(self.known_absolute_precision() as i64)
        } else {
            PadicValuation::Exact(self.value.raw.val)
        }
    }

    /// Whether this value is invertible in its concrete parent context.
    pub fn is_unit(&self) -> bool {
        match &self.parent {
            QadicParent::Ring(_) => self.valuation() == PadicValuation::Exact(0),
            QadicParent::Field(_) => unsafe { qadic_is_zero(&self.value.raw) == 0 },
        }
    }

    /// Multiplicative inverse in the concrete parent context.
    pub fn inverse(&self) -> Result<Self, UnitError> {
        if !self.is_unit() {
            return Err(if unsafe { qadic_is_zero(&self.value.raw) != 0 } {
                crate::DivisionByZeroError.into()
            } else {
                UnitError::NotAUnit {
                    context: QadicRing::TYPE_ID,
                }
            });
        }
        let context = Arc::clone(&self.value.context);
        let value = QadicValue::unary(context, &self.value, qadic_inv);
        Ok(match &self.parent {
            QadicParent::Ring(ring) => ring.wrap(value),
            QadicParent::Field(field) => field.wrap(value),
        })
    }

    /// Raise this value to a nonnegative word-sized power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = match &self.parent {
            QadicParent::Ring(ring) => ring.one(),
            QadicParent::Field(field) => field.one(),
        };
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self
                    .mul_in_parent(&result, &base)
                    .expect("values constructed by one q-adic context remain compatible");
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self
                    .mul_in_parent(&base, &base)
                    .expect("values constructed by one q-adic context remain compatible");
            }
        }
        result
    }

    /// Serialize the concrete parent and canonical power-basis coordinates.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a q-adic value and its complete parent context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> QadicElementWire {
        let value = match &self.parent {
            QadicParent::Ring(ring) => QadicElementValueWire::Ring {
                context: ring.to_wire(),
                coefficients: self
                    .value
                    .integral_coefficients(ring.residue_degree())
                    .expect("Witt-ring values retain nonnegative valuation")
                    .into_iter()
                    .map(|coefficient| coefficient.to_string())
                    .collect(),
            },
            QadicParent::Field(field) => QadicElementValueWire::Field {
                context: field.to_wire(),
                unit_coefficients: self
                    .value
                    .unit_coefficients(field.residue_degree())
                    .into_iter()
                    .map(|coefficient| coefficient.to_string())
                    .collect(),
                valuation: self.value.raw.val,
            },
        };
        QadicElementWire {
            type_id: Self::TYPE_ID.to_owned(),
            value,
        }
    }

    fn from_wire(wire: QadicElementWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        match wire.value {
            QadicElementValueWire::Ring {
                context,
                coefficients,
            } => {
                let ring = QadicRing::from_wire(context)?;
                let coefficients = parse_coordinates(
                    coefficients,
                    ring.residue_degree(),
                    "q-adic integral coefficient",
                )?;
                if coefficients.iter().any(|coefficient| {
                    coefficient < &Integer::zero() || coefficient >= ring.coefficient_ring.modulus()
                }) {
                    return Err(format!(
                        "q-adic integral coefficients must lie in [0, {})",
                        ring.coefficient_ring.modulus()
                    ));
                }
                let value = QadicValue::from_coefficients(Arc::clone(&ring.context), &coefficients);
                if value.integral_coefficients(ring.residue_degree()).as_ref()
                    != Some(&coefficients)
                {
                    return Err(
                        "q-adic integral coordinates are not in canonical FLINT form".to_owned(),
                    );
                }
                Ok(ring.wrap(value))
            }
            QadicElementValueWire::Field {
                context,
                unit_coefficients,
                valuation,
            } => {
                let field = QadicField::from_wire(context)?;
                let unit_coefficients = parse_coordinates(
                    unit_coefficients,
                    field.residue_degree(),
                    "q-adic unit coefficient",
                )?;
                let value = QadicValue::from_unit_coordinates(
                    Arc::clone(&field.integral_ring.context),
                    &unit_coefficients,
                    valuation,
                )?;
                Ok(field.wrap(value))
            }
        }
    }

    /// Apply the lifted residue-field Frobenius automorphism.
    pub fn frobenius(&self, exponent: u64) -> Self {
        let degree = match &self.parent {
            QadicParent::Ring(ring) => ring.residue_degree(),
            QadicParent::Field(field) => field.residue_degree(),
        } as u64;
        let reduced = exponent % degree;
        if reduced == 0 {
            return self.clone();
        }
        let mut output = QadicValue::initialized(Arc::clone(&self.value.context));
        unsafe {
            qadic_frobenius(
                &mut output.raw,
                &self.value.raw,
                reduced as i64,
                output.context.as_ptr(),
            )
        };
        match &self.parent {
            QadicParent::Ring(ring) => ring.wrap(output),
            QadicParent::Field(field) => field.wrap(output),
        }
    }

    /// Trace to the coefficient p-adic ring or field.
    pub fn trace(&self) -> PadicElement {
        match &self.parent {
            QadicParent::Ring(ring) => ring.coefficient_ring.element_with_flint(|output| unsafe {
                qadic_trace(output, &self.value.raw, self.value.context.as_ptr())
            }),
            QadicParent::Field(field) => {
                field.coefficient_field.element_with_flint(|output| unsafe {
                    qadic_trace(output, &self.value.raw, self.value.context.as_ptr())
                })
            }
        }
    }

    /// Norm to the coefficient p-adic ring or field.
    pub fn norm(&self) -> PadicElement {
        match &self.parent {
            QadicParent::Ring(ring) => ring.coefficient_ring.element_with_flint(|output| unsafe {
                qadic_norm(output, &self.value.raw, self.value.context.as_ptr())
            }),
            QadicParent::Field(field) => {
                field.coefficient_field.element_with_flint(|output| unsafe {
                    qadic_norm(output, &self.value.raw, self.value.context.as_ptr())
                })
            }
        }
    }

    fn mul_in_parent(&self, left: &Self, right: &Self) -> Result<Self, ContextMismatchError> {
        match &self.parent {
            QadicParent::Ring(ring) => ring.mul(left, right),
            QadicParent::Field(field) => field.mul(left, right),
        }
    }

    fn coefficients_degree(&self) -> usize {
        match &self.parent {
            QadicParent::Ring(ring) => ring.residue_degree(),
            QadicParent::Field(field) => field.residue_degree(),
        }
    }

    fn prime(&self) -> &Integer {
        match &self.parent {
            QadicParent::Ring(ring) => ring.prime(),
            QadicParent::Field(field) => field.prime(),
        }
    }
}

impl Serialize for QadicElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QadicElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(QadicElementWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for QadicElement {
    fn eq(&self, other: &Self) -> bool {
        let same_parent = match (&self.parent, &other.parent) {
            (QadicParent::Ring(left), QadicParent::Ring(right)) => left == right,
            (QadicParent::Field(left), QadicParent::Field(right)) => left == right,
            _ => false,
        };
        same_parent && unsafe { flint_sys::qadic_equal(&self.value.raw, &other.value.raw) != 0 }
    }
}

impl Eq for QadicElement {}

impl fmt::Debug for QadicElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QadicElement")
            .field(
                "unit_coefficients",
                &self.value.unit_coefficients(self.coefficients_degree()),
            )
            .field("valuation", &self.valuation())
            .field("parent", &self.parent.to_string())
            .finish()
    }
}

impl fmt::Display for QadicElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}*{}^{} + O({}^{})",
            self.value.unit_coefficients(self.coefficients_degree()),
            self.prime(),
            self.value.raw.val,
            self.prime(),
            self.known_absolute_precision()
        )
    }
}

impl SemanticType for QadicElement {
    const TYPE_ID: &'static str = "noether/numthy/QadicElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        match &self.parent {
            QadicParent::Ring(ring) => {
                output.push(0);
                append_component(&mut output, ring.canonical_bytes()?)?;
            }
            QadicParent::Field(field) => {
                output.push(1);
                append_component(&mut output, field.canonical_bytes()?)?;
            }
        }
        for coefficient in self.value.unit_coefficients(self.coefficients_degree()) {
            append_component(&mut output, coefficient.canonical_bytes()?)?;
        }
        output.extend_from_slice(&self.value.raw.val.to_be_bytes());
        output.extend_from_slice(&self.value.raw.N.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &QADIC_ELEMENT_DESCRIPTION
    }
}

impl Add<&QadicElement> for &QadicElement {
    type Output = Result<QadicElement, ContextMismatchError>;

    fn add(self, rhs: &QadicElement) -> Self::Output {
        match &self.parent {
            QadicParent::Ring(ring) => ring.add(self, rhs),
            QadicParent::Field(field) => field.add(self, rhs),
        }
    }
}

impl Sub<&QadicElement> for &QadicElement {
    type Output = Result<QadicElement, ContextMismatchError>;

    fn sub(self, rhs: &QadicElement) -> Self::Output {
        match &self.parent {
            QadicParent::Ring(ring) => ring.sub(self, rhs),
            QadicParent::Field(field) => field.sub(self, rhs),
        }
    }
}

impl Mul<&QadicElement> for &QadicElement {
    type Output = Result<QadicElement, ContextMismatchError>;

    fn mul(self, rhs: &QadicElement) -> Self::Output {
        match &self.parent {
            QadicParent::Ring(ring) => ring.mul(self, rhs),
            QadicParent::Field(field) => field.mul(self, rhs),
        }
    }
}

impl Neg for &QadicElement {
    type Output = Result<QadicElement, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        match &self.parent {
            QadicParent::Ring(ring) => ring.neg(self),
            QadicParent::Field(field) => field.neg(self),
        }
    }
}

fn validate_coordinate_count(
    received: usize,
    degree: usize,
) -> Result<(), InvalidQadicCoordinatesError> {
    if received <= degree {
        Ok(())
    } else {
        Err(InvalidQadicCoordinatesError::TooManyCoordinates { received, degree })
    }
}

fn padic_parent(value: &PadicElement) -> String {
    value
        .ring()
        .map(ToString::to_string)
        .or_else(|| value.field().map(ToString::to_string))
        .unwrap_or_else(|| "unknown p-adic parent".to_owned())
}

fn append_component(output: &mut Vec<u8>, component: Vec<u8>) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("q-adic component is too large to frame".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(&component);
    Ok(())
}

fn parse_coordinates(
    coordinates: Vec<String>,
    degree: usize,
    label: &str,
) -> Result<Vec<Integer>, String> {
    if coordinates.len() != degree {
        return Err(format!(
            "expected {degree} {label}s, received {}",
            coordinates.len()
        ));
    }
    coordinates
        .into_iter()
        .enumerate()
        .map(|(index, coefficient)| {
            coefficient
                .parse::<Integer>()
                .map_err(|error| format!("invalid {label} {index}: {error}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residue_field() -> FiniteField {
        FiniteField::extension(3_u64, 2).unwrap()
    }

    #[test]
    fn power_basis_coordinates_round_trip_for_ring_and_field_values() {
        let ring = QadicRing::new(residue_field(), 6).unwrap();
        let ring_coordinates = [
            ring.coefficient_ring().element(7),
            ring.coefficient_ring().element(11),
        ];
        let ring_value = ring.element_from_coefficients(&ring_coordinates).unwrap();
        assert_eq!(ring_value.coefficients(), ring_coordinates);

        let field = QadicField::new(residue_field(), 6).unwrap();
        let inverse_uniformizer = field
            .coefficient_field()
            .inverse(&field.coefficient_field().element(3))
            .unwrap();
        let field_coordinates = [inverse_uniformizer, field.coefficient_field().element(2)];
        let field_value = field.element_from_coefficients(&field_coordinates).unwrap();
        assert_eq!(field_value.coefficients(), field_coordinates);
    }

    #[test]
    fn power_basis_construction_rejects_wrong_length_and_context() {
        let ring = QadicRing::new(residue_field(), 4).unwrap();
        assert!(matches!(
            ring.element_from_coefficients(&[
                ring.coefficient_ring().one(),
                ring.coefficient_ring().zero(),
                ring.coefficient_ring().zero(),
            ]),
            Err(InvalidQadicCoordinatesError::TooManyCoordinates {
                received: 3,
                degree: 2
            })
        ));

        let wrong_ring = PadicRing::new(Integer::from(5), 4).unwrap();
        assert!(matches!(
            ring.element_from_coefficients(&[wrong_ring.one()]),
            Err(InvalidQadicCoordinatesError::CoordinateContext { index: 0, .. })
        ));
    }
}
