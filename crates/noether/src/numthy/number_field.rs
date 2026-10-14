// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact absolute number fields backed by FLINT's Antic implementation.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use flint_sys::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{MonomialOrder, Polynomial, PolynomialError, PolynomialRing};
use crate::{
    CommutativeRing, ComplexBall, ComplexBallField, ContextMismatchError, DivisionByZeroError,
    Field, IntegralDomain, InversionError, NoetherError, Rational, RationalField, Ring,
    SemanticType, TypeDescription, UnitError, UnitInversion, Q,
};

use super::number_field_order::{NumberFieldOrderCache, NumberFieldOrderError};

static NUMBER_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/NumberField@1",
    rust_path: "noether::numthy::NumberField",
    python_path: Some("noether.numthy.NumberField"),
};

static NUMBER_FIELD_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/NumberFieldElement@1",
    rust_path: "noether::numthy::NumberFieldElement",
    python_path: Some("noether.numthy.NumberFieldElement"),
};

/// Failure to construct or operate on an absolute number field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberFieldError {
    /// The defining polynomial was not univariate.
    DefiningPolynomialNotUnivariate {
        /// Number of variables in the supplied polynomial context.
        variables: usize,
    },
    /// The defining polynomial had degree less than two.
    DegreeTooSmall,
    /// A polynomial exponent could not be represented by this build.
    DegreeTooLarge,
    /// The defining polynomial was not monic.
    DefiningPolynomialNotMonic,
    /// The defining polynomial was reducible over the rationals.
    ReducibleDefiningPolynomial,
    /// The display name for the distinguished generator was invalid.
    InvalidGeneratorName,
    /// Too many power-basis coefficients were supplied for an element.
    TooManyCoefficients {
        /// Degree of the field.
        degree: usize,
        /// Number of supplied coefficients.
        actual: usize,
    },
    /// A generic polynomial operation failed while building an invariant.
    Polynomial(PolynomialError),
    /// Values from unequal number fields were combined.
    ContextMismatch(ContextMismatchError),
    /// Zero was used as a divisor.
    DivisionByZero(DivisionByZeroError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for NumberFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DefiningPolynomialNotUnivariate { variables } => write!(
                f,
                "number-field defining polynomial must be univariate; received {variables} variables"
            ),
            Self::DegreeTooSmall => {
                f.write_str("number-field defining polynomial must have degree at least two")
            }
            Self::DegreeTooLarge => {
                f.write_str("number-field degree does not fit this platform's FLINT interface")
            }
            Self::DefiningPolynomialNotMonic => {
                f.write_str("number-field defining polynomial must be monic")
            }
            Self::ReducibleDefiningPolynomial => {
                f.write_str("number-field defining polynomial is reducible over Q")
            }
            Self::InvalidGeneratorName => f.write_str(
                "number-field generator name must be nonempty, at most 64 characters, and contain no control characters",
            ),
            Self::TooManyCoefficients { degree, actual } => write!(
                f,
                "number-field element has {actual} coefficients; field degree is {degree}"
            ),
            Self::Polynomial(error) => error.fmt(f),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::DivisionByZero(error) => error.fmt(f),
            Self::InvalidEncoding(message) => {
                write!(f, "invalid number-field encoding: {message}")
            }
        }
    }
}

impl std::error::Error for NumberFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::DivisionByZero(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolynomialError> for NumberFieldError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<ContextMismatchError> for NumberFieldError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<DivisionByZeroError> for NumberFieldError {
    fn from(value: DivisionByZeroError) -> Self {
        Self::DivisionByZero(value)
    }
}

struct NumberFieldInner {
    raw: nf_t,
    defining_polynomial: Polynomial<RationalField>,
    coefficients: Vec<Rational>,
    generator_name: String,
    maximal_order: OnceLock<Result<NumberFieldOrderCache, NumberFieldOrderError>>,
}

impl NumberFieldInner {
    fn as_ptr(&self) -> *const nf_struct {
        self.raw.as_ptr()
    }
}

impl Drop for NumberFieldInner {
    fn drop(&mut self) {
        unsafe { nf_clear(self.raw.as_mut_ptr()) };
    }
}

// SAFETY: Antic's number-field context is immutable after construction.
unsafe impl Send for NumberFieldInner {}
unsafe impl Sync for NumberFieldInner {}

/// An exact absolute number field `Q[a] / (f(a))`.
///
/// The defining polynomial determines context identity. The generator name is
/// bounded display metadata, so `Q[a]/(f(a))` and `Q[b]/(f(b))` compare equal
/// and their elements interoperate without confusing notation with algebra.
#[derive(Clone)]
pub struct NumberField {
    inner: Arc<NumberFieldInner>,
}

impl NumberField {
    /// Construct a number field from a monic irreducible polynomial over `Q`.
    ///
    /// The polynomial must be univariate and have degree at least two. `name`
    /// changes only the displayed distinguished generator; when omitted, the
    /// polynomial ring's variable name is used.
    pub fn new(
        defining_polynomial: Polynomial<RationalField>,
        name: Option<String>,
    ) -> Result<Self, NumberFieldError> {
        let variable_count = defining_polynomial.ring().variables().len();
        if variable_count != 1 {
            return Err(NumberFieldError::DefiningPolynomialNotUnivariate {
                variables: variable_count,
            });
        }
        let generator_name =
            name.unwrap_or_else(|| defining_polynomial.ring().variables()[0].clone());
        validate_generator_name(&generator_name)?;
        let coefficients = dense_coefficients(&defining_polynomial)?;
        if coefficients.len() < 3 {
            return Err(NumberFieldError::DegreeTooSmall);
        }
        if coefficients.last() != Some(&Rational::one()) {
            return Err(NumberFieldError::DefiningPolynomialNotMonic);
        }
        let degree = coefficients.len() - 1;
        if i64::try_from(degree).is_err() {
            return Err(NumberFieldError::DegreeTooLarge);
        }

        let normalized_ring =
            PolynomialRing::new(Q, [generator_name.clone()], MonomialOrder::Lexicographic)
                .map_err(|error| NumberFieldError::InvalidEncoding(error.to_string()))?;
        let normalized = polynomial_from_coefficients(&normalized_ring, &coefficients)?;

        let raw_polynomial = RawFmpqPolynomial::from_coefficients(&coefficients);
        if !raw_polynomial.is_irreducible() {
            return Err(NumberFieldError::ReducibleDefiningPolynomial);
        }
        let mut raw = MaybeUninit::<nf_t>::uninit();
        unsafe { nf_init(raw.as_mut_ptr().cast(), raw_polynomial.as_ptr()) };
        Ok(Self {
            inner: Arc::new(NumberFieldInner {
                raw: unsafe { raw.assume_init() },
                defining_polynomial: normalized,
                coefficients,
                generator_name,
                maximal_order: OnceLock::new(),
            }),
        })
    }

    /// The normalized monic defining polynomial over `Q`.
    pub fn defining_polynomial(&self) -> &Polynomial<RationalField> {
        &self.inner.defining_polynomial
    }

    /// Display name of the distinguished power-basis generator.
    pub fn generator_name(&self) -> &str {
        &self.inner.generator_name
    }

    /// The distinguished generator, the image of the polynomial variable.
    pub fn generator(&self) -> NumberFieldElement {
        self.element(&[Rational::zero(), Rational::one()])
            .expect("two coefficients fit a number field of degree at least two")
    }

    /// Degree `[K:Q]`.
    pub fn degree(&self) -> usize {
        self.inner.coefficients.len() - 1
    }

    pub(super) fn cached_maximal_order(
        &self,
    ) -> Option<Result<NumberFieldOrderCache, NumberFieldOrderError>> {
        self.inner.maximal_order.get().cloned()
    }

    pub(super) fn cache_maximal_order(
        &self,
        value: Result<NumberFieldOrderCache, NumberFieldOrderError>,
    ) {
        let _ = self.inner.maximal_order.set(value);
    }

    pub(crate) fn defining_coefficients(&self) -> &[Rational] {
        &self.inner.coefficients
    }

    /// Construct an element from ascending rational power-basis coefficients.
    pub fn element(
        &self,
        coefficients: &[Rational],
    ) -> Result<NumberFieldElement, NumberFieldError> {
        if coefficients.len() > self.degree() {
            return Err(NumberFieldError::TooManyCoefficients {
                degree: self.degree(),
                actual: coefficients.len(),
            });
        }
        let mut element = NumberFieldElement::initialized(self.clone());
        let polynomial = RawFmpqPolynomial::from_coefficients(coefficients);
        unsafe {
            nf_elem_set_fmpq_poly(
                element.raw.as_mut_ptr(),
                polynomial.as_ptr(),
                self.inner.as_ptr(),
            )
        };
        Ok(element)
    }

    /// Embed one rational scalar in this field.
    pub fn scalar(&self, value: Rational) -> NumberFieldElement {
        self.element(&[value])
            .expect("one coefficient fits a number field of degree at least two")
    }

    /// Signature `(r1, r2)` with `r1 + 2*r2 = degree`.
    pub fn signature(&self) -> Result<(usize, usize), NumberFieldError> {
        let polynomial = RawFmpqPolynomial::from_coefficients(&self.inner.coefficients);
        let numerator = polynomial.numerator();
        let mut real_roots = 0;
        let mut complex_root_pairs = 0;
        unsafe {
            fmpz_poly_signature(&mut real_roots, &mut complex_root_pairs, numerator.as_ptr());
        }
        Ok((real_roots as usize, complex_root_pairs as usize))
    }

    fn as_ptr(&self) -> *const nf_struct {
        self.inner.as_ptr()
    }

    fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn mismatch(&self, actual: &Self) -> ContextMismatchError {
        ContextMismatchError {
            expected: format!("NumberField(degree={})", self.degree()),
            actual: format!("NumberField(degree={})", actual.degree()),
        }
    }

    fn normalize<'a>(
        &self,
        value: &'a NumberFieldElement,
    ) -> Result<Cow<'a, NumberFieldElement>, ContextMismatchError> {
        if self != value.field() {
            return Err(self.mismatch(value.field()));
        }
        if self.same_storage(value.field()) {
            Ok(Cow::Borrowed(value))
        } else {
            Ok(Cow::Owned(self.element(&value.coefficients()).expect(
                "equal fields accept the same degree-many coordinates",
            )))
        }
    }

    fn binary(
        &self,
        left: &NumberFieldElement,
        right: &NumberFieldElement,
        operation: unsafe extern "C" fn(
            *mut nf_elem_struct,
            *const nf_elem_struct,
            *const nf_elem_struct,
            *const nf_struct,
        ),
    ) -> Result<NumberFieldElement, ContextMismatchError> {
        let left = self.normalize(left)?;
        let right = self.normalize(right)?;
        let mut output = NumberFieldElement::initialized(self.clone());
        unsafe {
            operation(
                output.raw.as_mut_ptr(),
                left.raw.as_ptr(),
                right.raw.as_ptr(),
                self.as_ptr(),
            )
        };
        Ok(output)
    }

    /// Serialize this context to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and validate a number-field context from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PartialEq for NumberField {
    fn eq(&self, other: &Self) -> bool {
        self.inner.coefficients == other.inner.coefficients
    }
}

impl Eq for NumberField {}

impl fmt::Debug for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumberField")
            .field("degree", &self.degree())
            .field("generator", &self.generator_name())
            .finish()
    }
}

impl fmt::Display for NumberField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NumberField(degree={}, generator={})",
            self.degree(),
            self.generator_name()
        )
    }
}

impl SemanticType for NumberField {
    const TYPE_ID: &'static str = "noether/numthy/NumberField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        coefficients_bytes(&self.inner.coefficients)
    }

    fn describe() -> &'static TypeDescription {
        &NUMBER_FIELD_DESCRIPTION
    }
}

impl Ring for NumberField {
    type Element = NumberFieldElement;

    fn contains(&self, value: &Self::Element) -> bool {
        self == value.field()
    }

    fn zero(&self) -> Self::Element {
        NumberFieldElement::initialized(self.clone())
    }

    fn one(&self) -> Self::Element {
        let mut output = NumberFieldElement::initialized(self.clone());
        unsafe { nf_elem_one(output.raw.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, nf_elem_add)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, nf_elem_sub)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.binary(a, b, nf_elem_mul)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        let value = self.normalize(value)?;
        let mut output = NumberFieldElement::initialized(self.clone());
        unsafe { nf_elem_neg(output.raw.as_mut_ptr(), value.raw.as_ptr(), self.as_ptr()) };
        Ok(output)
    }
}

impl CommutativeRing for NumberField {}
impl IntegralDomain for NumberField {}

impl UnitInversion for NumberField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(UnitError::from)
    }
}

impl Field for NumberField {
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError> {
        let value = self.normalize(value)?;
        if value.is_zero() {
            return Err(DivisionByZeroError.into());
        }
        let mut output = NumberFieldElement::initialized(self.clone());
        unsafe { nf_elem_inv(output.raw.as_mut_ptr(), value.raw.as_ptr(), self.as_ptr()) };
        Ok(output)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumberFieldRepr {
    #[serde(rename = "type")]
    type_id: String,
    coefficients: Vec<String>,
    generator: String,
}

impl Serialize for NumberField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NumberFieldRepr {
            type_id: Self::TYPE_ID.to_owned(),
            coefficients: self
                .inner
                .coefficients
                .iter()
                .map(ToString::to_string)
                .collect(),
            generator: self.generator_name().to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NumberField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NumberFieldRepr::deserialize(deserializer)?;
        number_field_from_repr(repr).map_err(serde::de::Error::custom)
    }
}

/// One exact element of a concrete [`NumberField`].
pub struct NumberFieldElement {
    raw: nf_elem_t,
    field: NumberField,
    invariants: OnceLock<Box<NumberFieldElementInvariantCache>>,
}

#[derive(Default)]
struct NumberFieldElementInvariantCache {
    norm: OnceLock<Rational>,
    trace: OnceLock<Rational>,
    minimal_polynomial: OnceLock<Result<Polynomial<RationalField>, NumberFieldError>>,
    characteristic_polynomial: OnceLock<Result<Polynomial<RationalField>, NumberFieldError>>,
}

impl NumberFieldElement {
    fn initialized(field: NumberField) -> Self {
        let mut raw = MaybeUninit::<nf_elem_t>::uninit();
        unsafe { nf_elem_init(raw.as_mut_ptr().cast(), field.as_ptr()) };
        let mut output = Self {
            raw: unsafe { raw.assume_init() },
            field,
            invariants: OnceLock::new(),
        };
        unsafe { nf_elem_zero(output.raw.as_mut_ptr(), output.field.as_ptr()) };
        output
    }

    /// Concrete field context containing this element.
    pub fn field(&self) -> &NumberField {
        &self.field
    }

    /// Exact ascending coordinates in the distinguished power basis.
    pub fn coefficients(&self) -> Vec<Rational> {
        let polynomial = RawFmpqPolynomial::from_element(self);
        (0..self.field.degree())
            .map(|index| polynomial.coefficient(index))
            .collect()
    }

    /// Whether this element is zero.
    pub fn is_zero(&self) -> bool {
        unsafe { nf_elem_is_zero(self.raw.as_ptr(), self.field.as_ptr()) != 0 }
    }

    /// Whether this element is one.
    pub fn is_one(&self) -> bool {
        self == &self.field.one()
    }

    /// Add another element of the same number field.
    pub fn add(&self, other: &Self) -> Result<Self, ContextMismatchError> {
        self.field.add(self, other)
    }

    /// Subtract another element of the same number field.
    pub fn sub(&self, other: &Self) -> Result<Self, ContextMismatchError> {
        self.field.sub(self, other)
    }

    /// Multiply by another element of the same number field.
    pub fn mul(&self, other: &Self) -> Result<Self, ContextMismatchError> {
        self.field.mul(self, other)
    }

    /// Additive inverse.
    pub fn neg(&self) -> Self {
        self.field
            .neg(self)
            .expect("an element always belongs to its own field")
    }

    /// Multiplicative inverse, failing on zero.
    pub fn inverse(&self) -> Result<Self, InversionError> {
        self.field.inverse(self)
    }

    /// Divide by a nonzero element of the same number field.
    pub fn checked_div(&self, other: &Self) -> Result<Self, NumberFieldError> {
        let inverse = other.inverse().map_err(|error| match error {
            InversionError::DivisionByZero(error) => NumberFieldError::DivisionByZero(error),
            InversionError::ContextMismatch(error) => NumberFieldError::ContextMismatch(error),
        })?;
        self.mul(&inverse).map_err(NumberFieldError::from)
    }

    /// Raise to a nonnegative power.
    pub fn pow(&self, exponent: u64) -> Result<Self, ContextMismatchError> {
        let mut result = self.field.one();
        let mut base = self.clone();
        let mut power = exponent;
        while power != 0 {
            if power & 1 == 1 {
                result = result.mul(&base)?;
            }
            power >>= 1;
            if power != 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    /// Exact field norm to `Q`.
    pub fn norm(&self) -> Rational {
        self.invariant_cache()
            .norm
            .get_or_init(|| {
                let mut output = Rational::zero();
                unsafe {
                    nf_elem_norm(output.as_mut_ptr(), self.raw.as_ptr(), self.field.as_ptr())
                };
                output
            })
            .clone()
    }

    /// Exact field trace to `Q`.
    pub fn trace(&self) -> Rational {
        self.invariant_cache()
            .trace
            .get_or_init(|| {
                let mut output = Rational::zero();
                unsafe {
                    nf_elem_trace(output.as_mut_ptr(), self.raw.as_ptr(), self.field.as_ptr())
                };
                output
            })
            .clone()
    }

    /// Minimal polynomial over `Q`, in the canonical variable `t`.
    pub fn minimal_polynomial(&self) -> Result<Polynomial<RationalField>, NumberFieldError> {
        self.invariant_cache()
            .minimal_polynomial
            .get_or_init(|| self.representation_polynomial(true))
            .clone()
    }

    /// Characteristic polynomial over `Q`, in the canonical variable `t`.
    pub fn characteristic_polynomial(&self) -> Result<Polynomial<RationalField>, NumberFieldError> {
        self.invariant_cache()
            .characteristic_polynomial
            .get_or_init(|| self.representation_polynomial(false))
            .clone()
    }

    fn invariant_cache(&self) -> &NumberFieldElementInvariantCache {
        self.invariants
            .get_or_init(|| Box::new(NumberFieldElementInvariantCache::default()))
    }

    fn representation_polynomial(
        &self,
        minimal: bool,
    ) -> Result<Polynomial<RationalField>, NumberFieldError> {
        let degree = self.field.degree() as slong;
        unsafe {
            let mut matrix: fmpz_mat_t = std::mem::zeroed();
            fmpz_mat_init(matrix.as_mut_ptr(), degree, degree);
            let mut denominator = crate::Integer::zero();
            nf_elem_rep_mat_fmpz_mat_den(
                matrix.as_mut_ptr(),
                denominator.as_mut_ptr(),
                self.raw.as_ptr(),
                self.field.as_ptr(),
            );
            let mut polynomial: fmpz_poly_t = std::mem::zeroed();
            fmpz_poly_init(polynomial.as_mut_ptr());
            if minimal {
                fmpz_mat_minpoly(polynomial.as_mut_ptr(), matrix.as_ptr());
            } else {
                fmpz_mat_charpoly(polynomial.as_mut_ptr(), matrix.as_ptr());
            }
            let output_degree = fmpz_poly_degree(polynomial.as_ptr());
            let coefficients = if output_degree < 0 {
                Vec::new()
            } else {
                (0..=output_degree as usize)
                    .map(|index| {
                        let mut numerator = crate::Integer::zero();
                        fmpz_poly_get_coeff_fmpz(
                            numerator.as_mut_ptr(),
                            polynomial.as_ptr(),
                            index as slong,
                        );
                        let scale = denominator.pow((output_degree as usize - index) as u64);
                        Rational::new(numerator, scale)
                            .expect("Antic representation denominator is nonzero")
                    })
                    .collect::<Vec<_>>()
            };
            fmpz_poly_clear(polynomial.as_mut_ptr());
            fmpz_mat_clear(matrix.as_mut_ptr());
            let ring = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic)
                .map_err(|error| NumberFieldError::InvalidEncoding(error.to_string()))?;
            polynomial_from_coefficients(&ring, &coefficients)
        }
    }

    /// Certified complex values under every embedding of this field.
    pub fn conjugates(
        &self,
        ball_field: ComplexBallField,
    ) -> Result<Vec<ComplexBall>, NumberFieldError> {
        let degree = self.field.degree();
        let precision = ball_field.precision_bits() as slong;
        let element_polynomial = RawFmpqPolynomial::from_element(self);
        let defining_polynomial =
            RawFmpqPolynomial::from_coefficients(&self.field.inner.coefficients);
        let numerator = defining_polynomial.numerator();
        let mut output = Vec::with_capacity(degree);
        unsafe {
            let roots = _acb_vec_init(degree as slong);
            arb_fmpz_poly_complex_roots(roots, numerator.as_ptr(), 0, precision);
            let mut accumulator = MaybeUninit::<acb_struct>::uninit();
            let mut coefficient = MaybeUninit::<acb_struct>::uninit();
            acb_init(accumulator.as_mut_ptr());
            acb_init(coefficient.as_mut_ptr());
            let mut accumulator = accumulator.assume_init();
            let mut coefficient = coefficient.assume_init();
            for root_index in 0..degree {
                acb_zero(&mut accumulator);
                for coefficient_index in (0..degree).rev() {
                    let exact = element_polynomial.coefficient(coefficient_index);
                    acb_set_fmpq(&mut coefficient, exact.as_ptr(), precision);
                    acb_mul(
                        &mut accumulator,
                        &accumulator as *const _,
                        roots.add(root_index),
                        precision,
                    );
                    acb_add(
                        &mut accumulator,
                        &accumulator as *const _,
                        &coefficient,
                        precision,
                    );
                }
                output.push(ComplexBall::clone_from_ptr(ball_field, &accumulator));
            }
            acb_clear(&mut accumulator);
            acb_clear(&mut coefficient);
            _acb_vec_clear(roots, degree as slong);
        }
        Ok(output)
    }

    /// Serialize this exact element to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and validate an exact element from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Drop for NumberFieldElement {
    fn drop(&mut self) {
        unsafe { nf_elem_clear(self.raw.as_mut_ptr(), self.field.as_ptr()) };
    }
}

impl Clone for NumberFieldElement {
    fn clone(&self) -> Self {
        let mut output = Self::initialized(self.field.clone());
        unsafe {
            nf_elem_set(
                output.raw.as_mut_ptr(),
                self.raw.as_ptr(),
                self.field.as_ptr(),
            )
        };
        output
    }
}

// SAFETY: an Antic element has no shared mutable state, and its immutable
// context remains alive through the owned `NumberField` handle.
unsafe impl Send for NumberFieldElement {}
unsafe impl Sync for NumberFieldElement {}

impl PartialEq for NumberFieldElement {
    fn eq(&self, other: &Self) -> bool {
        if self.field != other.field {
            return false;
        }
        if self.field.same_storage(&other.field) {
            unsafe {
                nf_elem_equal(self.raw.as_ptr(), other.raw.as_ptr(), self.field.as_ptr()) != 0
            }
        } else {
            self.coefficients() == other.coefficients()
        }
    }
}

impl Eq for NumberFieldElement {}

impl fmt::Debug for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumberFieldElement")
            .field("coefficients", &bounded_coefficients(&self.coefficients()))
            .field("field", &self.field)
            .finish()
    }
}

impl fmt::Display for NumberFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} in {}",
            bounded_coefficients(&self.coefficients()),
            self.field
        )
    }
}

impl SemanticType for NumberFieldElement {
    const TYPE_ID: &'static str = "noether/numthy/NumberFieldElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        append_component(&mut output, &coefficients_bytes(&self.coefficients())?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &NUMBER_FIELD_ELEMENT_DESCRIPTION
    }
}

impl Add<&NumberFieldElement> for &NumberFieldElement {
    type Output = Result<NumberFieldElement, ContextMismatchError>;

    fn add(self, rhs: &NumberFieldElement) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub<&NumberFieldElement> for &NumberFieldElement {
    type Output = Result<NumberFieldElement, ContextMismatchError>;

    fn sub(self, rhs: &NumberFieldElement) -> Self::Output {
        self.sub(rhs)
    }
}

impl Mul<&NumberFieldElement> for &NumberFieldElement {
    type Output = Result<NumberFieldElement, ContextMismatchError>;

    fn mul(self, rhs: &NumberFieldElement) -> Self::Output {
        self.mul(rhs)
    }
}

impl Neg for &NumberFieldElement {
    type Output = NumberFieldElement;

    fn neg(self) -> Self::Output {
        self.neg()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumberFieldElementRepr {
    #[serde(rename = "type")]
    type_id: String,
    field: NumberFieldRepr,
    coefficients: Vec<String>,
}

impl Serialize for NumberFieldElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NumberFieldElementRepr {
            type_id: Self::TYPE_ID.to_owned(),
            field: NumberFieldRepr {
                type_id: NumberField::TYPE_ID.to_owned(),
                coefficients: self
                    .field
                    .inner
                    .coefficients
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                generator: self.field.generator_name().to_owned(),
            },
            coefficients: self
                .coefficients()
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NumberFieldElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NumberFieldElementRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        let field = number_field_from_repr(repr.field).map_err(serde::de::Error::custom)?;
        let coefficients = parse_rationals(&repr.coefficients).map_err(serde::de::Error::custom)?;
        field
            .element(&coefficients)
            .map_err(serde::de::Error::custom)
    }
}

struct RawFmpqPolynomial {
    raw: fmpq_poly_t,
}

impl RawFmpqPolynomial {
    fn from_coefficients(coefficients: &[Rational]) -> Self {
        let mut raw: fmpq_poly_t = unsafe { std::mem::zeroed() };
        unsafe {
            fmpq_poly_init(raw.as_mut_ptr());
            for (index, coefficient) in coefficients.iter().enumerate() {
                fmpq_poly_set_coeff_fmpq(raw.as_mut_ptr(), index as slong, coefficient.as_ptr());
            }
        }
        Self { raw }
    }

    fn from_element(element: &NumberFieldElement) -> Self {
        let mut raw: fmpq_poly_t = unsafe { std::mem::zeroed() };
        unsafe {
            fmpq_poly_init(raw.as_mut_ptr());
            nf_elem_get_fmpq_poly(
                raw.as_mut_ptr(),
                element.raw.as_ptr(),
                element.field.as_ptr(),
            );
        }
        Self { raw }
    }

    fn as_ptr(&self) -> *const fmpq_poly_struct {
        self.raw.as_ptr()
    }

    fn coefficient(&self, index: usize) -> Rational {
        let mut output = Rational::zero();
        unsafe { fmpq_poly_get_coeff_fmpq(output.as_mut_ptr(), self.as_ptr(), index as slong) };
        output
    }

    fn numerator(&self) -> RawFmpzPolynomial {
        let mut raw: fmpz_poly_t = unsafe { std::mem::zeroed() };
        unsafe {
            fmpz_poly_init(raw.as_mut_ptr());
            fmpq_poly_get_numerator(raw.as_mut_ptr(), self.as_ptr());
        }
        RawFmpzPolynomial { raw }
    }

    fn is_irreducible(&self) -> bool {
        let numerator = self.numerator();
        unsafe {
            let mut factors: fmpz_poly_factor_t = std::mem::zeroed();
            fmpz_poly_factor_init(factors.as_mut_ptr());
            fmpz_poly_factor(factors.as_mut_ptr(), numerator.as_ptr());
            let irreducible = factors[0].num == 1 && *factors[0].exp == 1;
            fmpz_poly_factor_clear(factors.as_mut_ptr());
            irreducible
        }
    }
}

impl Drop for RawFmpqPolynomial {
    fn drop(&mut self) {
        unsafe { fmpq_poly_clear(self.raw.as_mut_ptr()) };
    }
}

struct RawFmpzPolynomial {
    raw: fmpz_poly_t,
}

impl RawFmpzPolynomial {
    fn as_ptr(&self) -> *const fmpz_poly_struct {
        self.raw.as_ptr()
    }
}

impl Drop for RawFmpzPolynomial {
    fn drop(&mut self) {
        unsafe { fmpz_poly_clear(self.raw.as_mut_ptr()) };
    }
}

fn validate_generator_name(name: &str) -> Result<(), NumberFieldError> {
    if name.is_empty() || name.chars().count() > 64 || name.chars().any(char::is_control) {
        Err(NumberFieldError::InvalidGeneratorName)
    } else {
        Ok(())
    }
}

fn dense_coefficients(
    polynomial: &Polynomial<RationalField>,
) -> Result<Vec<Rational>, NumberFieldError> {
    let mut terms = Vec::new();
    let mut degree = 0usize;
    for (exponent, coefficient) in polynomial.terms() {
        let power = exponent[0]
            .to_string()
            .parse::<usize>()
            .map_err(|_| NumberFieldError::DegreeTooLarge)?;
        degree = degree.max(power);
        terms.push((power, coefficient.clone()));
    }
    if terms.is_empty() {
        return Err(NumberFieldError::DegreeTooSmall);
    }
    let mut coefficients = vec![Rational::zero(); degree + 1];
    for (power, coefficient) in terms {
        coefficients[power] = coefficient;
    }
    Ok(coefficients)
}

fn polynomial_from_coefficients(
    ring: &PolynomialRing<RationalField>,
    coefficients: &[Rational],
) -> Result<Polynomial<RationalField>, NumberFieldError> {
    ring.from_u64_terms(
        coefficients
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| !coefficient.is_zero())
            .map(|(index, coefficient)| (vec![index as u64], coefficient.clone())),
    )
    .map_err(NumberFieldError::from)
}

fn parse_rationals(values: &[String]) -> Result<Vec<Rational>, NumberFieldError> {
    values
        .iter()
        .map(|value| {
            value
                .parse::<Rational>()
                .map_err(|error| NumberFieldError::InvalidEncoding(error.to_string()))
        })
        .collect()
}

fn number_field_from_repr(repr: NumberFieldRepr) -> Result<NumberField, NumberFieldError> {
    if repr.type_id != NumberField::TYPE_ID {
        return Err(NumberFieldError::InvalidEncoding(format!(
            "expected type {}, received {}",
            NumberField::TYPE_ID,
            repr.type_id
        )));
    }
    let coefficients = parse_rationals(&repr.coefficients)?;
    let ring = PolynomialRing::new(Q, [repr.generator.clone()], MonomialOrder::Lexicographic)
        .map_err(|error| NumberFieldError::InvalidEncoding(error.to_string()))?;
    let polynomial = polynomial_from_coefficients(&ring, &coefficients)?;
    NumberField::new(polynomial, Some(repr.generator))
}

fn coefficients_bytes(coefficients: &[Rational]) -> Result<Vec<u8>, NoetherError> {
    let mut output = Vec::new();
    let count = u64::try_from(coefficients.len()).map_err(|_| {
        NoetherError::Serialization(
            "number-field coefficient count does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&count.to_be_bytes());
    for coefficient in coefficients {
        append_component(&mut output, &coefficient.canonical_bytes()?)?;
    }
    Ok(output)
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "number-field component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

fn bounded_coefficients(coefficients: &[Rational]) -> String {
    const COEFFICIENT_LIMIT: usize = 8;
    const CHARACTER_LIMIT: usize = 32;
    let mut rendered = coefficients
        .iter()
        .take(COEFFICIENT_LIMIT)
        .map(|coefficient| {
            let mut value = coefficient.to_string();
            if value.len() > CHARACTER_LIMIT {
                value.truncate(CHARACTER_LIMIT);
                value.push_str("...");
            }
            value
        })
        .collect::<Vec<_>>();
    if coefficients.len() > COEFFICIENT_LIMIT {
        rendered.push("...".to_owned());
    }
    format!("[{}]", rendered.join(", "))
}
