// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Fixed-cap p-adic integer contexts.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::mem::MaybeUninit;
use std::sync::Arc;

use flint_sys::{
    fmpz_mod, fmpz_set, padic_add, padic_clear, padic_ctx_clear, padic_ctx_init, padic_ctx_struct,
    padic_equal, padic_get_fmpz, padic_init2, padic_inv, padic_is_zero, padic_mul, padic_neg,
    padic_print_mode_PADIC_TERSE, padic_reduce, padic_set, padic_set_fmpz, padic_struct, padic_sub,
    padic_teichmuller,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{Polynomial, PolynomialError};
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    CommutativeRing, ContextMismatchError, FiniteContext, Integer, InversionError, NoetherError,
    PrecisionContext, Ring, SemanticType, TypeDescription, UnitError, UnitInversion, ValuedRing,
};

mod context_maps;

pub use context_maps::{
    PadicEmbeddingError, PadicFieldRestriction, PadicIntegralEmbedding, PadicRingRestriction,
    PadicUniformizerDivisionError,
};

static PADIC_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicRing@1",
    rust_path: "noether::numthy::PadicRing",
    python_path: Some("noether.numthy.PadicRing"),
};

static PADIC_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicElement@1",
    rust_path: "noether::numthy::PadicElement",
    python_path: Some("noether.numthy.PadicElement"),
};

static PADIC_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicField@1",
    rust_path: "noether::numthy::PadicField",
    python_path: Some("noether.numthy.PadicField"),
};

/// Failure to construct a fixed-cap p-adic integer context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicContextError {
    /// The proposed residue characteristic was not prime.
    InvalidPrime {
        /// Rejected residue characteristic.
        prime: Integer,
    },
    /// Absolute precision zero carries no p-adic digit information.
    ZeroAbsolutePrecision,
    /// The precision did not fit FLINT's signed machine-word interface.
    PrecisionTooLarge {
        /// Rejected absolute precision.
        absolute_precision: u64,
    },
}

impl fmt::Display for PadicContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrime { prime } => {
                write!(
                    f,
                    "p-adic residue characteristic must be prime; received {prime}"
                )
            }
            Self::ZeroAbsolutePrecision => {
                write!(f, "p-adic absolute precision must be at least one")
            }
            Self::PrecisionTooLarge { absolute_precision } => write!(
                f,
                "p-adic absolute precision {absolute_precision} exceeds FLINT's supported range"
            ),
        }
    }
}

impl std::error::Error for PadicContextError {}

/// Failure to discard p-adic digits through an explicit precision restriction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadicPrecisionError {
    /// Requested absolute precision.
    pub requested: u64,
    /// Absolute precision available on the source value.
    pub available: u64,
}

impl fmt::Display for PadicPrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cannot restrict p-adic precision {} to {}; the target must lie in 1..={}",
            self.available, self.requested, self.available
        )
    }
}

impl std::error::Error for PadicPrecisionError {}

/// Failure to restrict a p-adic value to a smaller precision context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicRestrictionError {
    /// The value did not belong to the source context.
    ContextMismatch(ContextMismatchError),
    /// The requested precision was zero or exceeded the available precision.
    Precision(PadicPrecisionError),
}

impl fmt::Display for PadicRestrictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Precision(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PadicRestrictionError {}

impl From<ContextMismatchError> for PadicRestrictionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PadicPrecisionError> for PadicRestrictionError {
    fn from(value: PadicPrecisionError) -> Self {
        Self::Precision(value)
    }
}

/// Failure to lift a simple residue-field root to a fixed-cap p-adic root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HenselLiftError {
    /// The polynomial or residue root belonged to another context.
    ContextMismatch(ContextMismatchError),
    /// Generic polynomial construction or evaluation failed.
    Polynomial(PolynomialError),
    /// Hensel root lifting is defined here only for univariate polynomials.
    NotUnivariate {
        /// Number of variables in the supplied polynomial ring.
        arity: usize,
    },
    /// The proposed residue value was not a root modulo `p`.
    NotAResidueRoot,
    /// The derivative vanished modulo `p`, so the simple-root lemma does not apply.
    SingularDerivative,
    /// Newton lifting did not produce a root at the context's full precision.
    DidNotConverge {
        /// Number of Newton updates performed.
        iterations: u32,
    },
}

impl fmt::Display for HenselLiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Polynomial(error) => error.fmt(f),
            Self::NotUnivariate { arity } => write!(
                f,
                "Hensel root lifting requires a univariate polynomial; received {arity} variables"
            ),
            Self::NotAResidueRoot => {
                f.write_str("the proposed residue value is not a root modulo p")
            }
            Self::SingularDerivative => f.write_str(
                "the polynomial derivative vanishes modulo p at the proposed residue root",
            ),
            Self::DidNotConverge { iterations } => write!(
                f,
                "Hensel lifting did not reach the fixed precision after {iterations} Newton updates"
            ),
        }
    }
}

impl std::error::Error for HenselLiftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::NotUnivariate { .. }
            | Self::NotAResidueRoot
            | Self::SingularDerivative
            | Self::DidNotConverge { .. } => None,
        }
    }
}

impl From<ContextMismatchError> for HenselLiftError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<PolynomialError> for HenselLiftError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

/// Precision-aware p-adic valuation information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadicValuation {
    /// The represented ball has exact valuation strictly below its precision.
    Exact(i64),
    /// The value is zero modulo `p^N`, so only the lower bound `v_p(x) >= N`
    /// is known; it must not be reported as exact mathematical zero.
    AtLeast(i64),
}

struct PadicContext {
    raw: padic_ctx_struct,
    precision: i64,
}

impl PadicContext {
    fn new(prime: &Integer, precision: i64) -> Self {
        let mut raw = MaybeUninit::<padic_ctx_struct>::uninit();
        unsafe {
            padic_ctx_init(
                raw.as_mut_ptr(),
                prime.as_ptr(),
                0,
                precision,
                padic_print_mode_PADIC_TERSE,
            )
        };
        Self {
            raw: unsafe { raw.assume_init() },
            precision,
        }
    }

    fn as_ptr(&self) -> *const padic_ctx_struct {
        &self.raw
    }
}

impl Drop for PadicContext {
    fn drop(&mut self) {
        unsafe { padic_ctx_clear(&mut self.raw) };
    }
}

// An initialized FLINT p-adic context is immutable during element arithmetic.
// Operations write only to their separately allocated output values.
unsafe impl Send for PadicContext {}
unsafe impl Sync for PadicContext {}

struct PadicValue {
    raw: padic_struct,
    context: Arc<PadicContext>,
}

impl PadicValue {
    fn initialized(context: Arc<PadicContext>) -> Self {
        let mut raw = MaybeUninit::<padic_struct>::uninit();
        unsafe { padic_init2(raw.as_mut_ptr(), context.precision) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    fn from_integer(context: Arc<PadicContext>, value: &Integer) -> Self {
        let mut output = Self::initialized(context);
        unsafe { padic_set_fmpz(&mut output.raw, value.as_ptr(), output.context.as_ptr()) };
        output
    }

    fn integer(&self) -> Integer {
        let mut output = Integer::zero();
        unsafe { padic_get_fmpz(output.as_mut_ptr(), &self.raw, self.context.as_ptr()) };
        output
    }

    fn is_zero(&self) -> bool {
        unsafe { padic_is_zero(&self.raw) != 0 }
    }

    fn unit(&self) -> Integer {
        unsafe { Integer::clone_from_ptr(&self.raw.u) }
    }

    fn binary(
        context: Arc<PadicContext>,
        left: &Self,
        right: &Self,
        operation: unsafe extern "C" fn(
            *mut padic_struct,
            *const padic_struct,
            *const padic_struct,
            *const padic_ctx_struct,
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
        context: Arc<PadicContext>,
        value: &Self,
        operation: unsafe extern "C" fn(
            *mut padic_struct,
            *const padic_struct,
            *const padic_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::initialized(context);
        unsafe { operation(&mut output.raw, &value.raw, output.context.as_ptr()) };
        output
    }
}

impl Clone for PadicValue {
    fn clone(&self) -> Self {
        let mut output = Self::initialized(Arc::clone(&self.context));
        unsafe { padic_set(&mut output.raw, &self.raw, output.context.as_ptr()) };
        output
    }
}

impl Drop for PadicValue {
    fn drop(&mut self) {
        unsafe { padic_clear(&mut self.raw) };
    }
}

unsafe impl Send for PadicValue {}
unsafe impl Sync for PadicValue {}

impl fmt::Display for PadicValuation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(value) => value.fmt(f),
            Self::AtLeast(bound) => write!(f, ">={bound}"),
        }
    }
}

/// The p-adic integers known modulo `p^N`, with `N` fixed at construction.
///
/// This is a finite-precision context, not the infinite ring `Z_p` disguised
/// as exact. Every element is a canonical FLINT-backed residue modulo `p^N`,
/// so its coefficient storage is bounded before allocation by the context's
/// [`Self::modulus`]. Arithmetic preserves the same absolute precision cap.
#[derive(Clone)]
pub struct PadicRing {
    prime: Integer,
    absolute_precision: u64,
    modulus: Integer,
    residue_field: FiniteField,
    context: Arc<PadicContext>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicRingWire {
    #[serde(rename = "type")]
    type_id: String,
    prime: String,
    absolute_precision: u64,
}

impl PadicRing {
    /// Construct fixed-cap p-adic integers known modulo `p^N`.
    pub fn new<T: Into<Integer>>(
        prime: T,
        absolute_precision: u64,
    ) -> Result<Self, PadicContextError> {
        let prime = prime.into();
        if absolute_precision == 0 {
            return Err(PadicContextError::ZeroAbsolutePrecision);
        }
        let precision = i64::try_from(absolute_precision)
            .map_err(|_| PadicContextError::PrecisionTooLarge { absolute_precision })?;
        let residue_field =
            FiniteField::prime(prime.clone()).map_err(|error| PadicContextError::InvalidPrime {
                prime: error.characteristic,
            })?;
        let modulus = prime.pow(absolute_precision);
        let context = Arc::new(PadicContext::new(&prime, precision));
        Ok(Self {
            prime,
            absolute_precision,
            modulus,
            residue_field,
            context,
        })
    }

    /// Residue characteristic `p`.
    pub fn prime(&self) -> &Integer {
        &self.prime
    }

    /// Immutable absolute precision cap `N`.
    pub fn absolute_precision(&self) -> u64 {
        self.absolute_precision
    }

    /// Storage modulus `p^N`; every element has a representative in
    /// `[0, p^N)`, giving the promised per-element coefficient bound.
    pub fn modulus(&self) -> &Integer {
        &self.modulus
    }

    /// Prime residue field `F_p`.
    pub fn residue_field(&self) -> &FiniteField {
        &self.residue_field
    }

    /// Embed an integer as a value known modulo `p^N`.
    pub fn element<T: Into<Integer>>(&self, value: T) -> PadicElement {
        let representative = self.canonicalize(&value.into());
        self.wrap(PadicValue::from_integer(
            Arc::clone(&self.context),
            &representative,
        ))
    }

    /// Serialize the fixed prime and absolute precision as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision p-adic ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> PadicRingWire {
        PadicRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            prime: self.prime.to_string(),
            absolute_precision: self.absolute_precision,
        }
    }

    fn from_wire(wire: PadicRingWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let prime = wire
            .prime
            .parse::<Integer>()
            .map_err(|error| format!("invalid p-adic residue characteristic: {error}"))?;
        Self::new(prime, wire.absolute_precision).map_err(|error| error.to_string())
    }

    /// Reduce a p-adic integer to its prime residue field.
    pub fn residue(
        &self,
        value: &PadicElement,
    ) -> Result<FiniteFieldElement, ContextMismatchError> {
        self.require(value)?;
        Ok(self
            .residue_field
            .element(self.canonicalize(&value.value.integer())))
    }

    /// Lift a residue-field element to its Teichmüller representative modulo
    /// `p^N`, characterized by `[a]^p = [a]` and `[a] mod p = a`.
    pub fn teichmuller(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<PadicElement, ContextMismatchError> {
        if value.field() != &self.residue_field {
            return Err(ContextMismatchError {
                expected: self.residue_field.to_string(),
                actual: value.field().to_string(),
            });
        }
        let mut lift = self.element(value.representative().clone());
        let raw = PadicValue::unary(Arc::clone(&self.context), &lift.value, padic_teichmuller);
        lift = self.wrap(raw);
        Ok(lift)
    }

    /// Lift a simple root modulo `p` to a root at this context's full precision.
    ///
    /// The polynomial must be univariate over this exact [`PadicRing`], the
    /// supplied value must be a root in its residue field, and the derivative
    /// must be nonzero there. The returned value is checked to annihilate the
    /// polynomial modulo `p^N`; unmet hypotheses fail closed.
    pub fn hensel_lift(
        &self,
        polynomial: &Polynomial<Self>,
        residue_root: &FiniteFieldElement,
    ) -> Result<PadicElement, HenselLiftError> {
        if polynomial.ring().coefficient_ring() != self {
            return Err(ContextMismatchError {
                expected: self.to_string(),
                actual: polynomial.ring().coefficient_ring().to_string(),
            }
            .into());
        }
        let arity = polynomial.ring().variables().len();
        if arity != 1 {
            return Err(HenselLiftError::NotUnivariate { arity });
        }
        if residue_root.field() != &self.residue_field {
            return Err(ContextMismatchError {
                expected: self.residue_field.to_string(),
                actual: residue_root.field().to_string(),
            }
            .into());
        }

        let derivative = polynomial.derivative(0)?;
        let mut root = self.element(residue_root.representative().clone());
        let initial_value = polynomial.evaluate(&[root.clone()])?;
        if self.residue(&initial_value)? != self.residue_field.zero() {
            return Err(HenselLiftError::NotAResidueRoot);
        }
        let initial_derivative = derivative.evaluate(&[root.clone()])?;
        if !initial_derivative.is_unit() {
            return Err(HenselLiftError::SingularDerivative);
        }

        let iterations = hensel_newton_iterations(self.absolute_precision);
        for _ in 0..iterations {
            let value = polynomial.evaluate(&[root.clone()])?;
            if value == self.zero() {
                return Ok(root);
            }
            let slope = derivative.evaluate(&[root.clone()])?;
            let inverse = slope
                .inverse()
                .map_err(|_| HenselLiftError::SingularDerivative)?;
            let correction = self.mul(&value, &inverse)?;
            root = self.sub(&root, &correction)?;
        }

        if polynomial.evaluate(&[root.clone()])? == self.zero()
            && self.residue(&root)? == *residue_root
        {
            Ok(root)
        } else {
            Err(HenselLiftError::DidNotConverge { iterations })
        }
    }

    /// Discard high p-adic digits explicitly, producing a value in the
    /// distinct context `(p, requested_absolute_precision)`.
    pub fn restrict(
        &self,
        value: &PadicElement,
        requested_absolute_precision: u64,
    ) -> Result<PadicElement, PadicRestrictionError> {
        self.precision_restriction(requested_absolute_precision)?
            .apply(value)
    }

    fn require(&self, value: &PadicElement) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.parent.to_string(),
            })
        }
    }

    fn canonicalize(&self, value: &Integer) -> Integer {
        let mut representative = Integer::zero();
        unsafe {
            fmpz_mod(
                representative.as_mut_ptr(),
                value.as_ptr(),
                self.modulus.as_ptr(),
            )
        };
        representative
    }

    fn wrap(&self, value: PadicValue) -> PadicElement {
        PadicElement {
            parent: PadicParent::Ring(self.clone()),
            value,
        }
    }

    pub(crate) fn element_with_flint(
        &self,
        operation: impl FnOnce(*mut padic_struct),
    ) -> PadicElement {
        let mut value = PadicValue::initialized(Arc::clone(&self.context));
        operation(&mut value.raw);
        self.wrap(value)
    }

    pub(crate) fn raw_context_ptr(&self) -> *const padic_ctx_struct {
        self.context.as_ptr()
    }
}

fn hensel_newton_iterations(absolute_precision: u64) -> u32 {
    let mut remaining = absolute_precision;
    let mut iterations = 0;
    while remaining > 1 {
        remaining = remaining / 2 + remaining % 2;
        iterations += 1;
    }
    iterations
}

impl Serialize for PadicRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicRingWire::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl PartialEq for PadicRing {
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime && self.absolute_precision == other.absolute_precision
    }
}

impl Eq for PadicRing {}

impl fmt::Debug for PadicRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PadicRing")
            .field("prime", &self.prime)
            .field("absolute_precision", &self.absolute_precision)
            .finish()
    }
}

impl fmt::Display for PadicRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Z_{} modulo {}^{}",
            self.prime, self.prime, self.absolute_precision
        )
    }
}

impl SemanticType for PadicRing {
    const TYPE_ID: &'static str = "noether/numthy/PadicRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.prime.canonical_bytes()?)?;
        output.extend_from_slice(&self.absolute_precision.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_RING_DESCRIPTION
    }
}

impl Ring for PadicRing {
    type Element = PadicElement;

    fn contains(&self, value: &Self::Element) -> bool {
        matches!(&value.parent, PadicParent::Ring(ring) if self == ring)
    }

    fn zero(&self) -> Self::Element {
        self.element(Integer::zero())
    }

    fn one(&self) -> Self::Element {
        self.element(Integer::one())
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            padic_add,
        )))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            padic_sub,
        )))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.context),
            &a.value,
            &b.value,
            padic_mul,
        )))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(self.wrap(PadicValue::unary(
            Arc::clone(&self.context),
            &value.value,
            padic_neg,
        )))
    }
}

impl CommutativeRing for PadicRing {}

impl UnitInversion for PadicRing {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)?;
        if value.value.is_zero() {
            return Err(crate::DivisionByZeroError.into());
        }
        value.inverse().map_err(|_| UnitError::NotAUnit {
            context: Self::TYPE_ID,
        })
    }
}

impl PrecisionContext for PadicRing {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.absolute_precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

impl ValuedRing for PadicRing {
    fn uniformizer(&self) -> Self::Element {
        self.element(self.prime.clone())
    }

    fn valuation_lower_bound(&self, value: &Self::Element) -> Option<i64> {
        match value.valuation() {
            PadicValuation::Exact(valuation) => Some(valuation),
            PadicValuation::AtLeast(_) => None,
        }
    }
}

/// A finite-precision model of the p-adic field `Q_p`.
///
/// The cap is fixed before allocation and is retained by every value. Unlike
/// [`PadicRing`], arbitrary negative valuations mean the context alone cannot
/// bound an element's storage: the unit part may require `N - v_p(x)` digits.
/// Values indistinguishable at the cap compare equal, so this context
/// deliberately does not implement the exact [`crate::Field`] trait. Exact
/// field algorithms must use precision-aware contracts.
///
/// ```compile_fail
/// use noether::numthy::PadicField;
/// use noether::Field;
///
/// fn needs_exact_field<R: Field>(_: &R) {}
/// needs_exact_field(&PadicField::new(3_u8, 8).unwrap());
/// ```
#[derive(Clone)]
pub struct PadicField {
    integral_ring: PadicRing,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicFieldWire {
    #[serde(rename = "type")]
    type_id: String,
    prime: String,
    absolute_precision: u64,
}

impl PadicField {
    /// Construct `Q_p` with values known through absolute precision `N`.
    pub fn new<T: Into<Integer>>(
        prime: T,
        absolute_precision: u64,
    ) -> Result<Self, PadicContextError> {
        Ok(Self {
            integral_ring: PadicRing::new(prime, absolute_precision)?,
        })
    }

    pub(crate) fn from_integral_ring(integral_ring: PadicRing) -> Self {
        Self { integral_ring }
    }

    /// Residue characteristic `p`.
    pub fn prime(&self) -> &Integer {
        self.integral_ring.prime()
    }

    /// Immutable absolute precision cap `N`.
    pub fn absolute_precision(&self) -> u64 {
        self.integral_ring.absolute_precision()
    }

    /// Fixed-cap integral ring canonically embedded in this field context.
    pub fn integral_ring(&self) -> &PadicRing {
        &self.integral_ring
    }

    /// Prime residue field `F_p`.
    pub fn residue_field(&self) -> &FiniteField {
        self.integral_ring.residue_field()
    }

    /// Embed an integer into this fixed-cap field.
    pub fn element<T: Into<Integer>>(&self, value: T) -> PadicElement {
        self.wrap(PadicValue::from_integer(
            Arc::clone(&self.integral_ring.context),
            &value.into(),
        ))
    }

    /// Invert a value that is nonzero at the available precision.
    ///
    /// This is finite-precision arithmetic, not a proof that the represented
    /// value is nonzero in `Q_p`. Consequently [`PadicField`] deliberately
    /// does not implement the exact [`crate::Field`] trait.
    pub fn inverse(&self, value: &PadicElement) -> Result<PadicElement, InversionError> {
        self.require(value)?;
        if value.value.is_zero() {
            return Err(crate::DivisionByZeroError.into());
        }
        Ok(self.wrap(PadicValue::unary(
            Arc::clone(&self.integral_ring.context),
            &value.value,
            padic_inv,
        )))
    }

    /// Explicitly discard high p-adic digits into a distinct field context.
    pub fn restrict(
        &self,
        value: &PadicElement,
        requested_absolute_precision: u64,
    ) -> Result<PadicElement, PadicRestrictionError> {
        self.precision_restriction(requested_absolute_precision)?
            .apply(value)
    }

    /// Serialize the fixed prime and absolute precision as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a fixed-precision p-adic field context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> PadicFieldWire {
        PadicFieldWire {
            type_id: Self::TYPE_ID.to_owned(),
            prime: self.prime().to_string(),
            absolute_precision: self.absolute_precision(),
        }
    }

    fn from_wire(wire: PadicFieldWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let prime = wire
            .prime
            .parse::<Integer>()
            .map_err(|error| format!("invalid p-adic residue characteristic: {error}"))?;
        Self::new(prime, wire.absolute_precision).map_err(|error| error.to_string())
    }

    fn require(&self, value: &PadicElement) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.parent.to_string(),
            })
        }
    }

    fn wrap(&self, value: PadicValue) -> PadicElement {
        PadicElement {
            parent: PadicParent::Field(self.clone()),
            value,
        }
    }

    pub(crate) fn element_with_flint(
        &self,
        operation: impl FnOnce(*mut padic_struct),
    ) -> PadicElement {
        let mut value = PadicValue::initialized(Arc::clone(&self.integral_ring.context));
        operation(&mut value.raw);
        self.wrap(value)
    }

    pub(crate) fn raw_context_ptr(&self) -> *const padic_ctx_struct {
        self.integral_ring.raw_context_ptr()
    }

    pub(crate) fn element_from_unit_and_valuation(
        &self,
        unit: &Integer,
        valuation: i64,
    ) -> PadicElement {
        let mut value = PadicValue::initialized(Arc::clone(&self.integral_ring.context));
        unsafe {
            fmpz_set(&mut value.raw.u, unit.as_ptr());
            value.raw.v = valuation;
            padic_reduce(&mut value.raw, value.context.as_ptr());
        }
        self.wrap(value)
    }
}

impl Serialize for PadicField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicFieldWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for PadicField {
    fn eq(&self, other: &Self) -> bool {
        self.integral_ring == other.integral_ring
    }
}

impl Eq for PadicField {}

impl fmt::Debug for PadicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PadicField")
            .field("prime", &self.prime())
            .field("absolute_precision", &self.absolute_precision())
            .finish()
    }
}

impl fmt::Display for PadicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q_{} at absolute precision {}",
            self.prime(),
            self.absolute_precision()
        )
    }
}

impl SemanticType for PadicField {
    const TYPE_ID: &'static str = "noether/numthy/PadicField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.prime().canonical_bytes()?)?;
        output.extend_from_slice(&self.absolute_precision().to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_FIELD_DESCRIPTION
    }
}

impl Ring for PadicField {
    type Element = PadicElement;

    fn contains(&self, value: &Self::Element) -> bool {
        matches!(&value.parent, PadicParent::Field(field) if self == field)
    }

    fn zero(&self) -> Self::Element {
        self.element(Integer::zero())
    }

    fn one(&self) -> Self::Element {
        self.element(Integer::one())
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            padic_add,
        )))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            padic_sub,
        )))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.wrap(PadicValue::binary(
            Arc::clone(&self.integral_ring.context),
            &a.value,
            &b.value,
            padic_mul,
        )))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        Ok(self.wrap(PadicValue::unary(
            Arc::clone(&self.integral_ring.context),
            &value.value,
            padic_neg,
        )))
    }
}

impl CommutativeRing for PadicField {}

impl UnitInversion for PadicField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(Into::into)
    }
}

impl PrecisionContext for PadicField {
    type Precision = u64;

    fn precision(&self) -> &Self::Precision {
        &self.integral_ring.absolute_precision
    }

    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool {
        candidate >= reference
    }
}

impl ValuedRing for PadicField {
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
enum PadicParent {
    Ring(PadicRing),
    Field(PadicField),
}

impl fmt::Display for PadicParent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ring(ring) => ring.fmt(f),
            Self::Field(field) => field.fmt(f),
        }
    }
}

impl FiniteContext for PadicRing {
    fn cardinality(&self) -> Integer {
        self.modulus().clone()
    }

    fn elements(&self) -> Box<dyn Iterator<Item = Self::Element> + '_> {
        Box::new(PadicElements {
            ring: self.clone(),
            next: Integer::zero(),
        })
    }
}

struct PadicElements {
    ring: PadicRing,
    next: Integer,
}

impl Iterator for PadicElements {
    type Item = PadicElement;

    fn next(&mut self) -> Option<Self::Item> {
        if &self.next >= self.ring.modulus() {
            return None;
        }
        let value = self.ring.element(self.next.clone());
        self.next = &self.next + &Integer::one();
        Some(value)
    }
}

/// A fixed-cap p-adic value carrying its concrete ring or field context.
#[derive(Clone)]
pub struct PadicElement {
    parent: PadicParent,
    value: PadicValue,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicElementWire {
    #[serde(rename = "type")]
    type_id: String,
    value: PadicElementValueWire,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "parent", rename_all = "snake_case", deny_unknown_fields)]
enum PadicElementValueWire {
    Ring {
        context: PadicRingWire,
        representative: String,
    },
    Field {
        context: PadicFieldWire,
        unit: String,
        valuation: i64,
    },
}

impl PadicElement {
    pub(crate) fn raw_ptr(&self) -> *const padic_struct {
        &self.value.raw
    }

    pub(crate) fn unit_and_valuation(&self) -> Option<(Integer, i64)> {
        (!self.value.is_zero()).then(|| (self.value.unit(), self.value.raw.v))
    }

    /// Concrete fixed-cap p-adic ring containing this value, or `None` when
    /// the value belongs to a [`PadicField`].
    pub fn ring(&self) -> Option<&PadicRing> {
        match &self.parent {
            PadicParent::Ring(ring) => Some(ring),
            PadicParent::Field(_) => None,
        }
    }

    /// Concrete fixed-cap p-adic field containing this value, when it is a
    /// field element rather than an integral-ring element.
    pub fn field(&self) -> Option<&PadicField> {
        match &self.parent {
            PadicParent::Ring(_) => None,
            PadicParent::Field(field) => Some(field),
        }
    }

    /// Canonical representative in `[0, p^N)`.
    pub fn representative(&self) -> Option<Integer> {
        self.ring()
            .map(|ring| ring.canonicalize(&self.value.integer()))
    }

    /// Known absolute precision, equal to the immutable context cap.
    pub fn known_absolute_precision(&self) -> u64 {
        match &self.parent {
            PadicParent::Ring(ring) => ring.absolute_precision(),
            PadicParent::Field(field) => field.absolute_precision(),
        }
    }

    /// Residue in `F_p`, or `None` when the value has negative valuation.
    pub fn residue(&self) -> Option<FiniteFieldElement> {
        if self.value.is_zero() || self.value.raw.v > 0 {
            let field = match &self.parent {
                PadicParent::Ring(ring) => ring.residue_field(),
                PadicParent::Field(field) => field.residue_field(),
            };
            return Some(field.zero());
        }
        if self.value.raw.v < 0 {
            return None;
        }
        let field = match &self.parent {
            PadicParent::Ring(ring) => ring.residue_field(),
            PadicParent::Field(field) => field.residue_field(),
        };
        Some(field.element(self.value.unit()))
    }

    /// Precision-aware p-adic valuation.
    pub fn valuation(&self) -> PadicValuation {
        if self.value.is_zero() {
            return PadicValuation::AtLeast(self.known_absolute_precision() as i64);
        }
        PadicValuation::Exact(self.value.raw.v)
    }

    /// Whether this value is invertible in its concrete parent context.
    pub fn is_unit(&self) -> bool {
        match &self.parent {
            PadicParent::Ring(_) => self.valuation() == PadicValuation::Exact(0),
            PadicParent::Field(_) => !self.value.is_zero(),
        }
    }

    /// Multiplicative inverse in the concrete parent context.
    pub fn inverse(&self) -> Result<Self, UnitError> {
        if !self.is_unit() {
            return Err(if self.value.is_zero() {
                crate::DivisionByZeroError.into()
            } else {
                UnitError::NotAUnit {
                    context: PadicRing::TYPE_ID,
                }
            });
        }
        let context = Arc::clone(&self.value.context);
        let value = PadicValue::unary(context, &self.value, padic_inv);
        Ok(match &self.parent {
            PadicParent::Ring(ring) => ring.wrap(value),
            PadicParent::Field(field) => field.wrap(value),
        })
    }

    /// Raise this value to a nonnegative word-sized power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = match &self.parent {
            PadicParent::Ring(ring) => ring.one(),
            PadicParent::Field(field) => field.one(),
        };
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining != 0 {
            if remaining & 1 == 1 {
                result = self
                    .mul_in_parent(&result, &base)
                    .expect("values constructed by one p-adic context remain compatible");
            }
            remaining >>= 1;
            if remaining != 0 {
                base = self
                    .mul_in_parent(&base, &base)
                    .expect("values constructed by one p-adic context remain compatible");
            }
        }
        result
    }

    /// Serialize the concrete parent and canonical FLINT coordinates as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a p-adic value and its fixed-precision parent.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> PadicElementWire {
        let value = match &self.parent {
            PadicParent::Ring(ring) => PadicElementValueWire::Ring {
                context: ring.to_wire(),
                representative: ring.canonicalize(&self.value.integer()).to_string(),
            },
            PadicParent::Field(field) => PadicElementValueWire::Field {
                context: field.to_wire(),
                unit: self.value.unit().to_string(),
                valuation: self.value.raw.v,
            },
        };
        PadicElementWire {
            type_id: Self::TYPE_ID.to_owned(),
            value,
        }
    }

    fn from_wire(wire: PadicElementWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        match wire.value {
            PadicElementValueWire::Ring {
                context,
                representative,
            } => {
                let ring = PadicRing::from_wire(context)?;
                let representative = representative
                    .parse::<Integer>()
                    .map_err(|error| format!("invalid p-adic representative: {error}"))?;
                if representative < Integer::zero() || representative >= *ring.modulus() {
                    return Err(format!(
                        "p-adic representative must lie in [0, {})",
                        ring.modulus()
                    ));
                }
                Ok(ring.element(representative))
            }
            PadicElementValueWire::Field {
                context,
                unit,
                valuation,
            } => {
                let field = PadicField::from_wire(context)?;
                let unit = unit
                    .parse::<Integer>()
                    .map_err(|error| format!("invalid p-adic unit coordinate: {error}"))?;
                let value = field.element_from_unit_and_valuation(&unit, valuation);
                if value.value.unit() != unit || value.value.raw.v != valuation {
                    return Err(
                        "p-adic field coordinates are not in canonical FLINT form".to_owned()
                    );
                }
                Ok(value)
            }
        }
    }

    fn mul_in_parent(&self, left: &Self, right: &Self) -> Result<Self, ContextMismatchError> {
        match &self.parent {
            PadicParent::Ring(ring) => ring.mul(left, right),
            PadicParent::Field(field) => field.mul(left, right),
        }
    }
}

impl Serialize for PadicElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(PadicElementWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for PadicElement {
    fn eq(&self, other: &Self) -> bool {
        let same_parent = match (&self.parent, &other.parent) {
            (PadicParent::Ring(left), PadicParent::Ring(right)) => left == right,
            (PadicParent::Field(left), PadicParent::Field(right)) => left == right,
            _ => false,
        };
        same_parent && unsafe { padic_equal(&self.value.raw, &other.value.raw) != 0 }
    }
}

impl Eq for PadicElement {}

impl fmt::Debug for PadicElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PadicElement")
            .field("unit", &self.value.unit())
            .field("valuation", &self.valuation())
            .field("parent", &self.parent.to_string())
            .finish()
    }
}

impl fmt::Display for PadicElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prime = match &self.parent {
            PadicParent::Ring(ring) => ring.prime(),
            PadicParent::Field(field) => field.prime(),
        };
        write!(
            f,
            "{}*{}^{} + O({}^{})",
            self.value.unit(),
            prime,
            self.value.raw.v,
            prime,
            self.known_absolute_precision()
        )
    }
}

impl SemanticType for PadicElement {
    const TYPE_ID: &'static str = "noether/numthy/PadicElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        match &self.parent {
            PadicParent::Ring(ring) => {
                output.push(0);
                append_component(&mut output, ring.canonical_bytes()?)?;
            }
            PadicParent::Field(field) => {
                output.push(1);
                append_component(&mut output, field.canonical_bytes()?)?;
            }
        }
        append_component(&mut output, self.value.unit().canonical_bytes()?)?;
        output.extend_from_slice(&self.value.raw.v.to_be_bytes());
        output.extend_from_slice(&self.value.raw.N.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_ELEMENT_DESCRIPTION
    }
}

impl Add<&PadicElement> for &PadicElement {
    type Output = Result<PadicElement, ContextMismatchError>;

    fn add(self, rhs: &PadicElement) -> Self::Output {
        match &self.parent {
            PadicParent::Ring(ring) => ring.add(self, rhs),
            PadicParent::Field(field) => field.add(self, rhs),
        }
    }
}

impl Sub<&PadicElement> for &PadicElement {
    type Output = Result<PadicElement, ContextMismatchError>;

    fn sub(self, rhs: &PadicElement) -> Self::Output {
        match &self.parent {
            PadicParent::Ring(ring) => ring.sub(self, rhs),
            PadicParent::Field(field) => field.sub(self, rhs),
        }
    }
}

impl Mul<&PadicElement> for &PadicElement {
    type Output = Result<PadicElement, ContextMismatchError>;

    fn mul(self, rhs: &PadicElement) -> Self::Output {
        match &self.parent {
            PadicParent::Ring(ring) => ring.mul(self, rhs),
            PadicParent::Field(field) => field.mul(self, rhs),
        }
    }
}

impl Neg for &PadicElement {
    type Output = Result<PadicElement, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        match &self.parent {
            PadicParent::Ring(ring) => ring.neg(self),
            PadicParent::Field(field) => field.neg(self),
        }
    }
}

fn append_component(output: &mut Vec<u8>, component: Vec<u8>) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("p-adic component is too large to frame".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(&component);
    Ok(())
}
