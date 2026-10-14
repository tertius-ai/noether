// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact arbitrary-modulus residue rings.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::mem::MaybeUninit;
use std::sync::Arc;

use flint_sys::{
    fmpz_invmod, fmpz_mod_add, fmpz_mod_ctx_clear, fmpz_mod_ctx_init, fmpz_mod_ctx_struct,
    fmpz_mod_mul, fmpz_mod_neg, fmpz_mod_set_fmpz, fmpz_mod_sub,
};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CommutativeRing, ContextMismatchError, EuclideanDomain, FiniteContext, Integer, NoetherError,
    PrincipalIdealDomain, Ring, SemanticType, TypeDescription, UnitError, UnitInversion, Z,
};

use crate::numthy::{factor, FactorError};

static RESIDUE_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ResidueRing@1",
    rust_path: "noether::commalg::ResidueRing",
    python_path: Some("noether.commalg.ResidueRing"),
};

static RESIDUE_CLASS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ResidueClass@1",
    rust_path: "noether::commalg::ResidueClass",
    python_path: Some("noether.commalg.ResidueClass"),
};

static CRT_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CrtDecomposition@1",
    rust_path: "noether::commalg::CrtDecomposition",
    python_path: Some("noether.commalg.CrtDecomposition"),
};

/// A residue-ring modulus was not an integer greater than one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidModulusError {
    /// Rejected modulus.
    pub modulus: Integer,
}

impl fmt::Display for InvalidModulusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "residue-ring modulus must be greater than one; received {}",
            self.modulus
        )
    }
}

impl std::error::Error for InvalidModulusError {}

/// A residue class was not invertible modulo its ring's modulus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonUnitError {
    /// Canonical representative of the rejected class.
    pub value: Integer,
    /// Modulus of its residue ring.
    pub modulus: Integer,
}

impl fmt::Display for NonUnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} is not a unit modulo {}", self.value, self.modulus)
    }
}

impl std::error::Error for NonUnitError {}

/// Failure to construct, verify, project through, or reconstruct from a CRT decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrtError {
    /// Complete prime factorization of the source modulus failed.
    Factorization(FactorError),
    /// A projected value belonged to another source ring.
    SourceContext(ContextMismatchError),
    /// Reconstruction received the wrong number of components.
    ComponentCount {
        /// Required number of prime-power components.
        expected: usize,
        /// Supplied number of components.
        actual: usize,
    },
    /// One reconstruction component belonged to the wrong residue ring.
    ComponentContext {
        /// Zero-based component position.
        index: usize,
        /// Required prime-power modulus.
        expected: Integer,
        /// Supplied modulus.
        actual: Integer,
    },
    /// Serialized or internally constructed CRT data violated its defining invariants.
    InvalidDecomposition(&'static str),
}

impl fmt::Display for CrtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Factorization(error) => error.fmt(formatter),
            Self::SourceContext(error) => error.fmt(formatter),
            Self::ComponentCount { expected, actual } => write!(
                formatter,
                "CRT reconstruction requires {expected} components; received {actual}"
            ),
            Self::ComponentContext {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "CRT component {index} has modulus {actual}; expected {expected}"
            ),
            Self::InvalidDecomposition(reason) => {
                write!(formatter, "invalid CRT decomposition: {reason}")
            }
        }
    }
}

impl std::error::Error for CrtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Factorization(error) => Some(error),
            Self::SourceContext(error) => Some(error),
            Self::ComponentCount { .. }
            | Self::ComponentContext { .. }
            | Self::InvalidDecomposition(_) => None,
        }
    }
}

impl From<FactorError> for CrtError {
    fn from(error: FactorError) -> Self {
        Self::Factorization(error)
    }
}

struct ResidueContext {
    raw: fmpz_mod_ctx_struct,
    modulus: Integer,
}

impl ResidueContext {
    fn new(modulus: Integer) -> Self {
        let mut raw = MaybeUninit::<fmpz_mod_ctx_struct>::uninit();
        unsafe { fmpz_mod_ctx_init(raw.as_mut_ptr(), modulus.as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            modulus,
        }
    }

    pub(crate) fn raw_ptr(&self) -> *const fmpz_mod_ctx_struct {
        &self.raw
    }

    fn reduce(&self, value: &Integer) -> Integer {
        let mut reduced = Integer::zero();
        unsafe { fmpz_mod_set_fmpz(reduced.as_mut_ptr(), value.as_ptr(), self.raw_ptr()) };
        reduced
    }
}

impl Drop for ResidueContext {
    fn drop(&mut self) {
        unsafe { fmpz_mod_ctx_clear(&mut self.raw) };
    }
}

// FLINT's fmpz_mod arithmetic takes a const context and does not mutate its
// cached reciprocal after initialization.
unsafe impl Send for ResidueContext {}
unsafe impl Sync for ResidueContext {}

/// The exact finite ring `Z/nZ` for an arbitrary-precision modulus `n`.
///
/// The modulus is the complete context identity. FLINT owns a cached modular
/// reduction context internally; this is storage, not part of the public type.
#[derive(Clone)]
pub struct ResidueRing {
    context: Arc<ResidueContext>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidueRingWire {
    #[serde(rename = "type")]
    type_id: String,
    modulus: String,
}

impl ResidueRing {
    /// Construct `Z/nZ`, rejecting `n <= 1`.
    pub fn new<T: Into<Integer>>(modulus: T) -> Result<Self, InvalidModulusError> {
        let modulus = modulus.into();
        if modulus <= Integer::one() {
            return Err(InvalidModulusError { modulus });
        }
        Ok(Self {
            context: Arc::new(ResidueContext::new(modulus)),
        })
    }

    /// The positive arbitrary-precision modulus.
    pub fn modulus(&self) -> &Integer {
        &self.context.modulus
    }

    /// Borrow the private FLINT reduction context for another internal
    /// whole-object backend. The returned pointer remains valid while this
    /// cloned ring context is alive.
    pub(crate) fn raw_context_ptr(&self) -> *const fmpz_mod_ctx_struct {
        self.context.raw_ptr()
    }

    /// Decompose this ring into its canonical prime-power CRT factors.
    pub fn crt(&self) -> Result<CrtDecomposition, CrtError> {
        CrtDecomposition::for_ring(self.clone())
    }

    /// Reduce an integer to its canonical class in this ring.
    pub fn element<T: Into<Integer>>(&self, value: T) -> ResidueClass {
        ResidueClass {
            ring: self.clone(),
            value: self.context.reduce(&value.into()),
        }
    }

    /// Serialize the complete residue-ring context as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a residue-ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> ResidueRingWire {
        ResidueRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            modulus: self.modulus().to_string(),
        }
    }

    fn from_wire(wire: ResidueRingWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let modulus = wire
            .modulus
            .parse::<Integer>()
            .map_err(|error| format!("invalid residue-ring modulus: {error}"))?;
        Self::new(modulus).map_err(|error| error.to_string())
    }

    fn inverse_value(&self, value: &ResidueClass) -> Result<ResidueClass, NonUnitError> {
        let mut inverse = Integer::zero();
        let invertible = unsafe {
            fmpz_invmod(
                inverse.as_mut_ptr(),
                value.value.as_ptr(),
                self.modulus().as_ptr(),
            ) != 0
        };
        if invertible {
            Ok(self.element(inverse))
        } else {
            Err(NonUnitError {
                value: value.value.clone(),
                modulus: self.modulus().clone(),
            })
        }
    }

    fn require(&self, value: &ResidueClass) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.ring.to_string(),
            })
        }
    }

    fn binary(
        &self,
        a: &ResidueClass,
        b: &ResidueClass,
        operation: unsafe extern "C" fn(
            *mut flint_sys::fmpz,
            *const flint_sys::fmpz,
            *const flint_sys::fmpz,
            *const fmpz_mod_ctx_struct,
        ),
    ) -> Result<ResidueClass, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        Ok(self.binary_unchecked(a, b, operation))
    }

    fn binary_unchecked(
        &self,
        a: &ResidueClass,
        b: &ResidueClass,
        operation: unsafe extern "C" fn(
            *mut flint_sys::fmpz,
            *const flint_sys::fmpz,
            *const flint_sys::fmpz,
            *const fmpz_mod_ctx_struct,
        ),
    ) -> ResidueClass {
        let mut value = Integer::zero();
        unsafe {
            operation(
                value.as_mut_ptr(),
                a.value.as_ptr(),
                b.value.as_ptr(),
                self.context.raw_ptr(),
            )
        };
        ResidueClass {
            ring: self.clone(),
            value,
        }
    }
}

/// The canonical Chinese-remainder decomposition of `Z/nZ`.
///
/// Factors are the prime-power rings `Z/p^e Z`, ordered by increasing proved
/// prime `p`. Projection and reconstruction are exact mutually inverse maps.
#[derive(Clone, PartialEq, Eq)]
pub struct CrtDecomposition {
    source: ResidueRing,
    factors: Vec<ResidueRing>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrtDecompositionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: ResidueRing,
    factors: Vec<ResidueRing>,
}

impl CrtDecomposition {
    fn for_ring(source: ResidueRing) -> Result<Self, CrtError> {
        let factors = canonical_crt_factors(&source)?;
        let decomposition = Self { source, factors };
        decomposition.verify()?;
        Ok(decomposition)
    }

    /// The residue ring being decomposed.
    pub fn source(&self) -> &ResidueRing {
        &self.source
    }

    /// Canonical prime-power factor rings in increasing-prime order.
    pub fn factors(&self) -> &[ResidueRing] {
        &self.factors
    }

    /// Project one source value to every prime-power component.
    pub fn project(&self, value: &ResidueClass) -> Result<Vec<ResidueClass>, CrtError> {
        self.verify()?;
        self.source
            .require(value)
            .map_err(CrtError::SourceContext)?;
        Ok(self
            .factors
            .iter()
            .map(|factor| factor.element(value.representative().clone()))
            .collect())
    }

    /// Reconstruct the unique source value with the supplied components.
    pub fn reconstruct(&self, components: &[ResidueClass]) -> Result<ResidueClass, CrtError> {
        self.verify()?;
        if components.len() != self.factors.len() {
            return Err(CrtError::ComponentCount {
                expected: self.factors.len(),
                actual: components.len(),
            });
        }

        let mut combined = Integer::zero();
        for (index, (component, factor_ring)) in components.iter().zip(&self.factors).enumerate() {
            if component.ring() != factor_ring {
                return Err(CrtError::ComponentContext {
                    index,
                    expected: factor_ring.modulus().clone(),
                    actual: component.ring().modulus().clone(),
                });
            }
            let (cofactor, remainder) = Z
                .div_rem(self.source.modulus(), factor_ring.modulus())
                .map_err(|_| CrtError::InvalidDecomposition("a factor modulus cannot be zero"))?;
            if !remainder.is_zero() {
                return Err(CrtError::InvalidDecomposition(
                    "a component modulus does not divide the source modulus",
                ));
            }
            let inverse_witness = Z.extended_gcd(&cofactor, factor_ring.modulus());
            if !inverse_witness.gcd.is_one() {
                return Err(CrtError::InvalidDecomposition(
                    "component moduli are not pairwise coprime",
                ));
            }
            let scaled = component.representative() * &cofactor;
            let contribution = &scaled * &inverse_witness.coefficient_a;
            combined = &combined + &contribution;
        }
        Ok(self.source.element(combined))
    }

    /// Verify factor order, prime-power structure, and exact reconstruction modulus.
    pub fn verify(&self) -> Result<(), CrtError> {
        let expected = canonical_crt_factors(&self.source)?;
        if expected.len() != self.factors.len() {
            return Err(CrtError::InvalidDecomposition(
                "factor count does not match the proved prime factorization",
            ));
        }
        if expected != self.factors {
            return Err(CrtError::InvalidDecomposition(
                "factors are not the canonical increasing-prime powers",
            ));
        }
        Ok(())
    }

    /// Serialize the source and canonical factor rings.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a canonical CRT decomposition.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn canonical_crt_factors(source: &ResidueRing) -> Result<Vec<ResidueRing>, CrtError> {
    factor(source.modulus())?
        .factors()
        .iter()
        .map(|(prime, exponent)| {
            ResidueRing::new(prime.pow(*exponent)).map_err(|_| {
                CrtError::InvalidDecomposition("a proved prime power was not a valid modulus")
            })
        })
        .collect()
}

impl Serialize for CrtDecomposition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CrtDecomposition", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("factors", &self.factors)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CrtDecomposition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CrtDecompositionWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let decomposition = Self {
            source: wire.source,
            factors: wire.factors,
        };
        decomposition.verify().map_err(serde::de::Error::custom)?;
        Ok(decomposition)
    }
}

impl fmt::Debug for CrtDecomposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CrtDecomposition")
            .field("source_modulus", self.source.modulus())
            .field(
                "factor_moduli",
                &self
                    .factors
                    .iter()
                    .map(ResidueRing::modulus)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl fmt::Display for CrtDecomposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ≅ ", self.source)?;
        for (index, factor) in self.factors.iter().enumerate() {
            if index != 0 {
                formatter.write_str(" × ")?;
            }
            write!(formatter, "{factor}")?;
        }
        Ok(())
    }
}

impl SemanticType for CrtDecomposition {
    const TYPE_ID: &'static str = "noether/commalg/CrtDecomposition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &CRT_DECOMPOSITION_DESCRIPTION
    }
}

impl Serialize for ResidueRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ResidueRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(ResidueRingWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for ResidueRing {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context) || self.modulus() == other.modulus()
    }
}

impl Eq for ResidueRing {}

impl fmt::Debug for ResidueRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ResidueRing").field(self.modulus()).finish()
    }
}

impl fmt::Display for ResidueRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z/{}Z", self.modulus())
    }
}

impl SemanticType for ResidueRing {
    const TYPE_ID: &'static str = "noether/commalg/ResidueRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.modulus().canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &RESIDUE_RING_DESCRIPTION
    }
}

impl Ring for ResidueRing {
    type Element = ResidueClass;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring
    }

    fn zero(&self) -> Self::Element {
        self.element(0_u8)
    }

    fn one(&self) -> Self::Element {
        self.element(1_u8)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, fmpz_mod_add)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, fmpz_mod_sub)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, fmpz_mod_mul)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        let mut result = Integer::zero();
        unsafe {
            fmpz_mod_neg(
                result.as_mut_ptr(),
                value.value.as_ptr(),
                self.context.raw_ptr(),
            )
        };
        Ok(ResidueClass {
            ring: self.clone(),
            value: result,
        })
    }
}

impl CommutativeRing for ResidueRing {}

impl UnitInversion for ResidueRing {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)?;
        if value.is_zero() {
            return Err(crate::DivisionByZeroError.into());
        }
        value.inverse().map_err(|_| UnitError::NotAUnit {
            context: Self::TYPE_ID,
        })
    }
}

impl FiniteContext for ResidueRing {
    fn cardinality(&self) -> Integer {
        self.modulus().clone()
    }

    fn elements(&self) -> Box<dyn Iterator<Item = Self::Element> + '_> {
        Box::new(ResidueElements {
            ring: self.clone(),
            next: Integer::zero(),
        })
    }
}

struct ResidueElements {
    ring: ResidueRing,
    next: Integer,
}

impl Iterator for ResidueElements {
    type Item = ResidueClass;

    fn next(&mut self) -> Option<Self::Item> {
        if &self.next >= self.ring.modulus() {
            return None;
        }
        let value = self.ring.element(self.next.clone());
        self.next = &self.next + &Integer::one();
        Some(value)
    }
}

/// An exact residue class carrying its concrete [`ResidueRing`] context.
#[derive(Clone)]
pub struct ResidueClass {
    ring: ResidueRing,
    value: Integer,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResidueClassWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: ResidueRingWire,
    representative: String,
}

impl ResidueClass {
    /// The concrete residue ring containing this value.
    pub fn ring(&self) -> &ResidueRing {
        &self.ring
    }

    /// The canonical representative in `[0, n)`.
    pub fn representative(&self) -> &Integer {
        &self.value
    }

    /// Whether this class is zero.
    pub fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    /// Whether this class is one.
    pub fn is_one(&self) -> bool {
        self.value.is_one()
    }

    /// Whether this class has a multiplicative inverse.
    pub fn is_unit(&self) -> bool {
        Z.gcd(&self.value, self.ring.modulus()).is_one()
    }

    /// Return the multiplicative inverse of this class when it is a unit.
    pub fn inverse(&self) -> Result<Self, NonUnitError> {
        self.ring.inverse_value(self)
    }

    /// Raise this class to a nonnegative power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = self.ring.one();
        let mut base = self.clone();
        let mut power = exponent;
        while power != 0 {
            if power & 1 == 1 {
                result = self.ring.binary_unchecked(&result, &base, fmpz_mod_mul);
            }
            power >>= 1;
            if power != 0 {
                base = self.ring.binary_unchecked(&base, &base, fmpz_mod_mul);
            }
        }
        result
    }

    /// Serialize the ring context and canonical representative as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a residue class and its ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> ResidueClassWire {
        ResidueClassWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.to_wire(),
            representative: self.value.to_string(),
        }
    }

    fn from_wire(wire: ResidueClassWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let ring = ResidueRing::from_wire(wire.ring)?;
        let representative = wire
            .representative
            .parse::<Integer>()
            .map_err(|error| format!("invalid residue-class representative: {error}"))?;
        if representative < Integer::zero() || representative >= *ring.modulus() {
            return Err(format!(
                "residue-class representative must lie in [0, {})",
                ring.modulus()
            ));
        }
        Ok(ring.element(representative))
    }
}

impl Serialize for ResidueClass {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ResidueClass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(ResidueClassWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for ResidueClass {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring && self.value == other.value
    }
}

impl Eq for ResidueClass {}

impl fmt::Debug for ResidueClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidueClass")
            .field("representative", &self.value)
            .field("modulus", self.ring.modulus())
            .finish()
    }
}

impl fmt::Display for ResidueClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, self.ring.modulus())
    }
}

impl SemanticType for ResidueClass {
    const TYPE_ID: &'static str = "noether/commalg/ResidueClass@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let modulus = self.ring.canonical_bytes()?;
        let value = self.value.canonical_bytes()?;
        let mut output = Vec::with_capacity(16 + modulus.len() + value.len());
        for component in [modulus, value] {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "residue-class component is too large for canonical framing".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &RESIDUE_CLASS_DESCRIPTION
    }
}

impl Add<&ResidueClass> for &ResidueClass {
    type Output = Result<ResidueClass, ContextMismatchError>;

    fn add(self, rhs: &ResidueClass) -> Self::Output {
        self.ring.add(self, rhs)
    }
}

impl Sub<&ResidueClass> for &ResidueClass {
    type Output = Result<ResidueClass, ContextMismatchError>;

    fn sub(self, rhs: &ResidueClass) -> Self::Output {
        self.ring.sub(self, rhs)
    }
}

impl Mul<&ResidueClass> for &ResidueClass {
    type Output = Result<ResidueClass, ContextMismatchError>;

    fn mul(self, rhs: &ResidueClass) -> Self::Output {
        self.ring.mul(self, rhs)
    }
}

impl Neg for &ResidueClass {
    type Output = Result<ResidueClass, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.ring.neg(self)
    }
}
