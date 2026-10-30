// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite fields with explicit, verified presentations.

#![allow(unsafe_code)]

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use flint_sys::{
    _fq_nmod_ctx_init_conway_ui, fq_default_clear, fq_default_ctx_clear,
    fq_default_ctx_init_modulus_nmod, fq_default_ctx_struct, fq_default_get_nmod_poly,
    fq_default_init, fq_default_set_nmod_poly, fq_default_struct, fq_nmod_add, fq_nmod_clear,
    fq_nmod_ctx_clear, fq_nmod_ctx_init_modulus, fq_nmod_ctx_modulus, fq_nmod_ctx_struct,
    fq_nmod_equal, fq_nmod_frobenius, fq_nmod_init, fq_nmod_inv, fq_nmod_is_one, fq_nmod_is_zero,
    fq_nmod_mul, fq_nmod_neg, fq_nmod_norm, fq_nmod_one, fq_nmod_pow_ui, fq_nmod_reduce,
    fq_nmod_set, fq_nmod_set_fmpz, fq_nmod_struct, fq_nmod_sub, fq_nmod_trace, fq_nmod_zero,
    nmod_poly_clear, nmod_poly_get_coeff_ui, nmod_poly_init, nmod_poly_is_irreducible,
    nmod_poly_set_coeff_ui, nmod_poly_struct,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{ResidueClass, ResidueRing};
use crate::{
    CommutativeRing, ContextMismatchError, DivisionByZeroError, Field, FiniteContext, Integer,
    InversionError, NoetherError, PrimeSubfieldEmbedding, Ring, SemanticType, TypeDescription,
    UnitError, UnitInversion,
};

static FINITE_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/FiniteField@1",
    rust_path: "noether::numthy::FiniteField",
    python_path: Some("noether.numthy.FiniteField"),
};

static FINITE_FIELD_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/FiniteFieldElement@1",
    rust_path: "noether::numthy::FiniteFieldElement",
    python_path: Some("noether.numthy.FiniteFieldElement"),
};

/// A proposed finite-field characteristic was not prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidCharacteristicError {
    /// Rejected characteristic.
    pub characteristic: Integer,
}

impl fmt::Display for InvalidCharacteristicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "finite-field characteristic must be prime; received {}",
            self.characteristic
        )
    }
}

impl std::error::Error for InvalidCharacteristicError {}

/// Failure to construct a verified extension-field presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteFieldConstructionError {
    /// The proposed characteristic was not prime.
    InvalidCharacteristic(InvalidCharacteristicError),
    /// The extension degree was zero or could not be represented by FLINT.
    InvalidDegree {
        /// Rejected extension degree.
        degree: usize,
    },
    /// The explicit defining polynomial was not a normalized irreducible
    /// polynomial over the requested prime field.
    InvalidModulus {
        /// Precise failed validation.
        reason: String,
    },
    /// FLINT's deterministic Conway table has no entry for this field order.
    ConwayPolynomialUnavailable {
        /// Prime characteristic.
        characteristic: u64,
        /// Extension degree.
        degree: usize,
    },
}

impl fmt::Display for FiniteFieldConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCharacteristic(error) => error.fmt(f),
            Self::InvalidDegree { degree } => write!(
                f,
                "finite-field extension degree must be positive and fit FLINT; received {degree}"
            ),
            Self::InvalidModulus { reason } => {
                write!(f, "invalid finite-field defining polynomial: {reason}")
            }
            Self::ConwayPolynomialUnavailable {
                characteristic,
                degree,
            } => write!(
                f,
                "no deterministic Conway polynomial is available for characteristic \
                 {characteristic} and degree {degree}; supply an explicit monic irreducible modulus"
            ),
        }
    }
}

impl std::error::Error for FiniteFieldConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCharacteristic(error) => Some(error),
            _ => None,
        }
    }
}

impl From<InvalidCharacteristicError> for FiniteFieldConstructionError {
    fn from(value: InvalidCharacteristicError) -> Self {
        Self::InvalidCharacteristic(value)
    }
}

/// A coefficient list did not define an element in the requested basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidFiniteFieldElementError {
    /// Precise failed validation.
    pub reason: String,
}

impl fmt::Display for InvalidFiniteFieldElementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid finite-field element coefficients: {}",
            self.reason
        )
    }
}

impl std::error::Error for InvalidFiniteFieldElementError {}

struct ModulusPolynomial {
    raw: nmod_poly_struct,
}

impl ModulusPolynomial {
    fn new(characteristic: u64, coefficients: &[u64]) -> Self {
        let mut raw = MaybeUninit::<nmod_poly_struct>::uninit();
        unsafe { nmod_poly_init(raw.as_mut_ptr(), characteristic) };
        let mut polynomial = Self {
            raw: unsafe { raw.assume_init() },
        };
        for (degree, coefficient) in coefficients.iter().copied().enumerate() {
            unsafe { nmod_poly_set_coeff_ui(&mut polynomial.raw, degree as i64, coefficient) };
        }
        polynomial
    }

    fn as_ptr(&self) -> *const nmod_poly_struct {
        &self.raw
    }
}

impl Drop for ModulusPolynomial {
    fn drop(&mut self) {
        unsafe { nmod_poly_clear(&mut self.raw) };
    }
}

pub(crate) struct ExtensionContext {
    raw: fq_nmod_ctx_struct,
    default_raw: fq_default_ctx_struct,
    small_extension_tables: Option<SmallExtensionTables>,
    characteristic: Integer,
    characteristic_word: u64,
    modulus: Vec<u64>,
}

/// Direct arithmetic tables for byte-sized odd-characteristic extensions.
///
/// Codes are the canonical base-`p` encodings of power-basis coefficients, so
/// this supports both Conway and explicit moduli without changing the public
/// field presentation. Larger fields continue through FLINT.
pub(crate) struct SmallExtensionTables {
    order: usize,
    characteristic: usize,
    degree: usize,
    add: Box<[u8]>,
    subtract: Box<[u8]>,
    multiply: Box<[u8]>,
    subtract_product: Box<[u8]>,
    inverse: Box<[u8]>,
}

impl SmallExtensionTables {
    // The fused row-operation table has q^3 bytes. At q=64 it is 256 KiB.
    const MAX_ORDER: usize = 64;

    fn from_modulus(characteristic: u64, modulus: &[u64]) -> Option<Self> {
        // Characteristic two has the denser, SIMD-capable m4rste backend.
        if characteristic == 2 {
            return None;
        }
        let characteristic = usize::try_from(characteristic).ok()?;
        let degree = modulus.len().checked_sub(1)?;
        let order = characteristic.checked_pow(u32::try_from(degree).ok()?)?;
        if order > Self::MAX_ORDER {
            return None;
        }
        let decode = |mut value: usize| {
            let mut coefficients = vec![0; degree];
            for coefficient in &mut coefficients {
                *coefficient = value % characteristic;
                value /= characteristic;
            }
            coefficients
        };
        let encode = |coefficients: &[usize]| {
            coefficients
                .iter()
                .rev()
                .fold(0, |value, coefficient| value * characteristic + coefficient)
        };
        let coefficient_vectors = (0..order).map(decode).collect::<Vec<_>>();
        let mut add = vec![0_u8; order * order];
        let mut subtract = vec![0_u8; order * order];
        let mut multiply = vec![0_u8; order * order];
        for left in 0..order {
            for right in 0..order {
                let left_coefficients = &coefficient_vectors[left];
                let right_coefficients = &coefficient_vectors[right];
                let sum = (0..degree)
                    .map(|index| {
                        (left_coefficients[index] + right_coefficients[index]) % characteristic
                    })
                    .collect::<Vec<_>>();
                let difference = (0..degree)
                    .map(|index| {
                        (left_coefficients[index] + characteristic - right_coefficients[index])
                            % characteristic
                    })
                    .collect::<Vec<_>>();
                add[left * order + right] = encode(&sum) as u8;
                subtract[left * order + right] = encode(&difference) as u8;

                let mut product = vec![0; degree * 2 - 1];
                for (left_degree, left_coefficient) in left_coefficients.iter().enumerate() {
                    for (right_degree, right_coefficient) in right_coefficients.iter().enumerate() {
                        product[left_degree + right_degree] = (product[left_degree + right_degree]
                            + left_coefficient * right_coefficient)
                            % characteristic;
                    }
                }
                for high_degree in (degree..product.len()).rev() {
                    let coefficient = product[high_degree];
                    for (low_degree, modulus_coefficient) in
                        modulus.iter().copied().take(degree).enumerate()
                    {
                        let reduction = coefficient * usize::try_from(modulus_coefficient).ok()?
                            % characteristic;
                        let target = high_degree - degree + low_degree;
                        product[target] =
                            (product[target] + characteristic - reduction) % characteristic;
                    }
                }
                multiply[left * order + right] = encode(&product[..degree]) as u8;
            }
        }
        let mut inverse = vec![0_u8; order];
        for value in 1..order {
            inverse[value] = (1..order)
                .find(|candidate| multiply[value * order + candidate] == 1)
                .expect("irreducible modulus defines a field") as u8;
        }
        let mut subtract_product = vec![0_u8; order * order * order];
        for factor in 0..order {
            for pivot in 0..order {
                let product = multiply[factor * order + pivot] as usize;
                for current in 0..order {
                    subtract_product[(factor * order + pivot) * order + current] =
                        subtract[current * order + product];
                }
            }
        }
        Some(Self {
            order,
            characteristic,
            degree,
            add: add.into_boxed_slice(),
            subtract: subtract.into_boxed_slice(),
            multiply: multiply.into_boxed_slice(),
            subtract_product: subtract_product.into_boxed_slice(),
            inverse: inverse.into_boxed_slice(),
        })
    }

    pub(crate) fn decode(&self, mut value: u8) -> Vec<u64> {
        (0..self.degree)
            .map(|_| {
                let coefficient = usize::from(value) % self.characteristic;
                value = (usize::from(value) / self.characteristic) as u8;
                coefficient as u64
            })
            .collect()
    }

    pub(crate) fn inverse(&self, value: u8) -> u8 {
        self.inverse[usize::from(value)]
    }

    #[inline(always)]
    pub(crate) fn add(&self, left: u8, right: u8) -> u8 {
        self.add[usize::from(left) * self.order + usize::from(right)]
    }

    #[inline(always)]
    pub(crate) fn subtract(&self, left: u8, right: u8) -> u8 {
        self.subtract[usize::from(left) * self.order + usize::from(right)]
    }

    #[inline(always)]
    pub(crate) fn multiply(&self, left: u8, right: u8) -> u8 {
        debug_assert!((left as usize) < self.order);
        debug_assert!((right as usize) < self.order);
        self.multiply[usize::from(left) * self.order + usize::from(right)]
    }

    #[inline(always)]
    pub(crate) fn subtract_product(&self, current: u8, factor: u8, pivot: u8) -> u8 {
        debug_assert!((current as usize) < self.order);
        debug_assert!((factor as usize) < self.order);
        debug_assert!((pivot as usize) < self.order);
        let index = (usize::from(factor) * self.order + usize::from(pivot)) * self.order
            + usize::from(current);
        self.subtract_product[index]
    }
}

impl ExtensionContext {
    fn conway(characteristic: u64, degree: i64) -> Option<Self> {
        let mut raw = MaybeUninit::<fq_nmod_ctx_struct>::uninit();
        let initialized = unsafe {
            _fq_nmod_ctx_init_conway_ui(raw.as_mut_ptr(), characteristic, degree, c"a".as_ptr())
                != 0
        };
        if !initialized {
            return None;
        }
        let raw = unsafe { raw.assume_init() };
        let modulus_pointer = unsafe { fq_nmod_ctx_modulus(&raw) };
        let mut default_raw = MaybeUninit::<fq_default_ctx_struct>::uninit();
        unsafe {
            fq_default_ctx_init_modulus_nmod(
                default_raw.as_mut_ptr(),
                modulus_pointer,
                c"a".as_ptr(),
            )
        };
        let default_raw = unsafe { default_raw.assume_init() };
        let modulus = (0..=degree)
            .map(|index| unsafe { nmod_poly_get_coeff_ui(modulus_pointer, index) })
            .collect::<Vec<_>>();
        let small_extension_tables = SmallExtensionTables::from_modulus(characteristic, &modulus);
        Some(Self {
            raw,
            default_raw,
            small_extension_tables,
            characteristic: Integer::from(characteristic),
            characteristic_word: characteristic,
            modulus,
        })
    }

    fn explicit(characteristic: u64, polynomial: &ModulusPolynomial, modulus: Vec<u64>) -> Self {
        let mut raw = MaybeUninit::<fq_nmod_ctx_struct>::uninit();
        unsafe { fq_nmod_ctx_init_modulus(raw.as_mut_ptr(), polynomial.as_ptr(), c"a".as_ptr()) };
        let mut default_raw = MaybeUninit::<fq_default_ctx_struct>::uninit();
        unsafe {
            fq_default_ctx_init_modulus_nmod(
                default_raw.as_mut_ptr(),
                polynomial.as_ptr(),
                c"a".as_ptr(),
            )
        };
        let default_raw = unsafe { default_raw.assume_init() };
        let small_extension_tables = SmallExtensionTables::from_modulus(characteristic, &modulus);
        Self {
            raw: unsafe { raw.assume_init() },
            default_raw,
            small_extension_tables,
            characteristic: Integer::from(characteristic),
            characteristic_word: characteristic,
            modulus,
        }
    }

    pub(crate) fn as_ptr(&self) -> *const fq_nmod_ctx_struct {
        &self.raw
    }

    pub(crate) fn default_as_ptr(&self) -> *const fq_default_ctx_struct {
        &self.default_raw
    }

    pub(crate) fn small_extension_tables(&self) -> Option<&SmallExtensionTables> {
        self.small_extension_tables.as_ref()
    }

    fn degree(&self) -> usize {
        self.modulus.len() - 1
    }
}

impl Drop for ExtensionContext {
    fn drop(&mut self) {
        unsafe {
            fq_default_ctx_clear(&mut self.default_raw);
            fq_nmod_ctx_clear(&mut self.raw);
        };
    }
}

// An initialized fq_nmod context is immutable during element arithmetic. Each
// operation writes only to its distinct output element.
unsafe impl Send for ExtensionContext {}
unsafe impl Sync for ExtensionContext {}

pub(crate) struct ExtensionElement {
    pub(crate) raw: fq_nmod_struct,
    context: Arc<ExtensionContext>,
}

impl ExtensionElement {
    pub(crate) fn initialized(context: Arc<ExtensionContext>) -> Self {
        let mut raw = MaybeUninit::<fq_nmod_struct>::uninit();
        unsafe { fq_nmod_init(raw.as_mut_ptr(), context.as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    pub(crate) fn to_default(&self) -> DefaultExtensionElement {
        let mut output = DefaultExtensionElement::initialized(Arc::clone(&self.context));
        unsafe {
            fq_default_set_nmod_poly(&mut output.raw, &self.raw, output.context.default_as_ptr())
        };
        output
    }

    fn zero(context: Arc<ExtensionContext>) -> Self {
        let mut output = Self::initialized(context);
        unsafe { fq_nmod_zero(&mut output.raw, output.context.as_ptr()) };
        output
    }

    fn one(context: Arc<ExtensionContext>) -> Self {
        let mut output = Self::initialized(context);
        unsafe { fq_nmod_one(&mut output.raw, output.context.as_ptr()) };
        output
    }

    fn from_integer(context: Arc<ExtensionContext>, value: &Integer) -> Self {
        let mut output = Self::initialized(context);
        unsafe { fq_nmod_set_fmpz(&mut output.raw, value.as_ptr(), output.context.as_ptr()) };
        output
    }

    fn from_coefficients(context: Arc<ExtensionContext>, coefficients: &[u64]) -> Self {
        let mut output = Self::zero(context);
        for (degree, coefficient) in coefficients.iter().copied().enumerate() {
            unsafe { nmod_poly_set_coeff_ui(&mut output.raw, degree as i64, coefficient) };
        }
        unsafe { fq_nmod_reduce(&mut output.raw, output.context.as_ptr()) };
        output
    }

    fn coefficients(&self) -> Vec<u64> {
        (0..self.context.degree())
            .map(|degree| unsafe { nmod_poly_get_coeff_ui(&self.raw, degree as i64) })
            .collect()
    }

    fn binary(
        context: Arc<ExtensionContext>,
        left: &Self,
        right: &Self,
        operation: unsafe extern "C" fn(
            *mut fq_nmod_struct,
            *const fq_nmod_struct,
            *const fq_nmod_struct,
            *const fq_nmod_ctx_struct,
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
        context: Arc<ExtensionContext>,
        value: &Self,
        operation: unsafe extern "C" fn(
            *mut fq_nmod_struct,
            *const fq_nmod_struct,
            *const fq_nmod_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::initialized(context);
        unsafe { operation(&mut output.raw, &value.raw, output.context.as_ptr()) };
        output
    }
}

/// An owned element in FLINT's implementation-selecting `fq_default`
/// representation. It is private dispatch storage: the canonical public
/// finite-field element remains [`ExtensionElement`].
pub(crate) struct DefaultExtensionElement {
    pub(crate) raw: fq_default_struct,
    context: Arc<ExtensionContext>,
}

impl DefaultExtensionElement {
    pub(crate) fn initialized(context: Arc<ExtensionContext>) -> Self {
        let mut raw = MaybeUninit::<fq_default_struct>::uninit();
        unsafe { fq_default_init(raw.as_mut_ptr(), context.default_as_ptr()) };
        Self {
            raw: unsafe { raw.assume_init() },
            context,
        }
    }

    pub(crate) fn to_extension(&self) -> ExtensionElement {
        let mut output = ExtensionElement::initialized(Arc::clone(&self.context));
        unsafe {
            fq_default_get_nmod_poly(&mut output.raw, &self.raw, self.context.default_as_ptr())
        };
        output
    }
}

impl Drop for DefaultExtensionElement {
    fn drop(&mut self) {
        unsafe { fq_default_clear(&mut self.raw, self.context.default_as_ptr()) };
    }
}

unsafe impl Send for DefaultExtensionElement {}
unsafe impl Sync for DefaultExtensionElement {}

impl Clone for ExtensionElement {
    fn clone(&self) -> Self {
        let mut output = Self::initialized(Arc::clone(&self.context));
        unsafe { fq_nmod_set(&mut output.raw, &self.raw, output.context.as_ptr()) };
        output
    }
}

impl Drop for ExtensionElement {
    fn drop(&mut self) {
        unsafe { fq_nmod_clear(&mut self.raw, self.context.as_ptr()) };
    }
}

unsafe impl Send for ExtensionElement {}
unsafe impl Sync for ExtensionElement {}

#[derive(Clone)]
enum FiniteFieldBackend {
    Prime(ResidueRing),
    Extension(Arc<ExtensionContext>),
}

/// A finite field with one explicit, serialization-stable presentation.
///
/// Prime fields support arbitrary-size characteristic. Extension fields use
/// FLINT's word-characteristic `fq_nmod` backend and carry their normalized
/// monic irreducible modulus in the context identity. Equal-order fields with
/// different moduli therefore remain distinct until a checked isomorphism is
/// requested.
#[derive(Clone)]
pub struct FiniteField {
    backend: FiniteFieldBackend,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FiniteFieldWire {
    #[serde(rename = "type")]
    type_id: String,
    characteristic: String,
    modulus: Option<Vec<u64>>,
}

impl FiniteField {
    /// Construct the prime field `F_p` after a proved primality test.
    pub fn prime<T: Into<Integer>>(characteristic: T) -> Result<Self, InvalidCharacteristicError> {
        let characteristic = characteristic.into();
        if !characteristic.is_prime() {
            return Err(InvalidCharacteristicError { characteristic });
        }
        ResidueRing::new(characteristic.clone())
            .map(|residues| Self {
                backend: FiniteFieldBackend::Prime(residues),
            })
            .map_err(|_| InvalidCharacteristicError { characteristic })
    }

    /// Construct FLINT's deterministic Conway presentation of `F_{p^degree}`.
    ///
    /// Degree one returns the prime field. Higher degrees are currently
    /// word-characteristic, matching FLINT `fq_nmod`. A missing Conway-table
    /// entry is a typed error rather than a FLINT process abort; callers may
    /// then use [`Self::with_modulus`].
    pub fn extension(
        characteristic: u64,
        degree: usize,
    ) -> Result<Self, FiniteFieldConstructionError> {
        let characteristic_integer = Integer::from(characteristic);
        if !characteristic_integer.is_prime() {
            return Err(InvalidCharacteristicError {
                characteristic: characteristic_integer,
            }
            .into());
        }
        if degree == 0 {
            return Err(FiniteFieldConstructionError::InvalidDegree { degree });
        }
        if degree == 1 {
            return Self::prime(characteristic).map_err(Into::into);
        }
        let degree_ffi = i64::try_from(degree)
            .map_err(|_| FiniteFieldConstructionError::InvalidDegree { degree })?;
        let context = ExtensionContext::conway(characteristic, degree_ffi).ok_or(
            FiniteFieldConstructionError::ConwayPolynomialUnavailable {
                characteristic,
                degree,
            },
        )?;
        Ok(Self {
            backend: FiniteFieldBackend::Extension(Arc::new(context)),
        })
    }

    /// Construct `F_p[a] / (modulus)` from ascending coefficients.
    ///
    /// The modulus must have degree at least two, be monic, have every
    /// coefficient in `[0, p)`, and be irreducible over `F_p`. These
    /// requirements make the coefficient list a normalized context identity.
    pub fn with_modulus(
        characteristic: u64,
        modulus: Vec<u64>,
    ) -> Result<Self, FiniteFieldConstructionError> {
        let characteristic_integer = Integer::from(characteristic);
        if !characteristic_integer.is_prime() {
            return Err(InvalidCharacteristicError {
                characteristic: characteristic_integer,
            }
            .into());
        }
        if modulus.len() < 3 {
            return Err(FiniteFieldConstructionError::InvalidModulus {
                reason: "the modulus must have degree at least two".to_owned(),
            });
        }
        if modulus.last().copied() != Some(1) {
            return Err(FiniteFieldConstructionError::InvalidModulus {
                reason: "the modulus must be monic".to_owned(),
            });
        }
        if let Some((index, coefficient)) = modulus
            .iter()
            .copied()
            .enumerate()
            .find(|(_, coefficient)| *coefficient >= characteristic)
        {
            return Err(FiniteFieldConstructionError::InvalidModulus {
                reason: format!(
                    "coefficient {index} is {coefficient}, outside [0, {characteristic})"
                ),
            });
        }
        if i64::try_from(modulus.len()).is_err() {
            return Err(FiniteFieldConstructionError::InvalidDegree {
                degree: modulus.len() - 1,
            });
        }
        let polynomial = ModulusPolynomial::new(characteristic, &modulus);
        if unsafe { nmod_poly_is_irreducible(polynomial.as_ptr()) } == 0 {
            return Err(FiniteFieldConstructionError::InvalidModulus {
                reason: "the modulus is reducible over the prime field".to_owned(),
            });
        }
        let context = ExtensionContext::explicit(characteristic, &polynomial, modulus);
        Ok(Self {
            backend: FiniteFieldBackend::Extension(Arc::new(context)),
        })
    }

    /// The prime characteristic.
    pub fn characteristic(&self) -> &Integer {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => residues.modulus(),
            FiniteFieldBackend::Extension(context) => &context.characteristic,
        }
    }

    pub(crate) fn word_characteristic(&self) -> Option<u64> {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => residues.modulus().to_string().parse().ok(),
            FiniteFieldBackend::Extension(context) => Some(context.characteristic_word),
        }
    }

    pub(crate) fn extension_context(&self) -> Option<Arc<ExtensionContext>> {
        match &self.backend {
            FiniteFieldBackend::Prime(_) => None,
            FiniteFieldBackend::Extension(context) => Some(Arc::clone(context)),
        }
    }

    /// Extension degree over the prime field.
    pub fn degree(&self) -> usize {
        match &self.backend {
            FiniteFieldBackend::Prime(_) => 1,
            FiniteFieldBackend::Extension(context) => context.degree(),
        }
    }

    /// Canonical embedding of this field's prime subfield.
    pub fn prime_subfield_embedding(&self) -> PrimeSubfieldEmbedding {
        PrimeSubfieldEmbedding::for_target(self.clone())
    }

    /// Exact field order.
    pub fn order(&self) -> Integer {
        self.characteristic().pow(self.degree() as u64)
    }

    /// The ascending coefficients of the normalized defining modulus.
    /// Prime fields return `None` because they need no extension presentation.
    pub fn defining_modulus(&self) -> Option<&[u64]> {
        match &self.backend {
            FiniteFieldBackend::Prime(_) => None,
            FiniteFieldBackend::Extension(context) => Some(&context.modulus),
        }
    }

    /// Embed an integer through the prime subfield.
    pub fn element<T: Into<Integer>>(&self, value: T) -> FiniteFieldElement {
        let value = value.into();
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => self.wrap_prime(residues.element(value)),
            FiniteFieldBackend::Extension(context) => {
                self.wrap_extension(ExtensionElement::from_integer(Arc::clone(context), &value))
            }
        }
    }

    /// Construct an element from ascending coefficients in the power basis.
    ///
    /// Missing high coefficients are zero. Each supplied coefficient must be
    /// the canonical integer in `[0, p)`, and at most `degree` coefficients may
    /// be supplied.
    pub fn element_from_coefficients(
        &self,
        coefficients: &[u64],
    ) -> Result<FiniteFieldElement, InvalidFiniteFieldElementError> {
        if coefficients.len() > self.degree() {
            return Err(InvalidFiniteFieldElementError {
                reason: format!(
                    "received {} coefficients for a degree-{} field",
                    coefficients.len(),
                    self.degree()
                ),
            });
        }
        let characteristic = match &self.backend {
            FiniteFieldBackend::Prime(residues) => {
                let rendered = residues.modulus().to_string();
                rendered.parse::<u64>().ok()
            }
            FiniteFieldBackend::Extension(context) => Some(context.characteristic_word),
        };
        if let Some(characteristic) = characteristic {
            if let Some((index, coefficient)) = coefficients
                .iter()
                .copied()
                .enumerate()
                .find(|(_, coefficient)| *coefficient >= characteristic)
            {
                return Err(InvalidFiniteFieldElementError {
                    reason: format!(
                        "coefficient {index} is {coefficient}, outside [0, {characteristic})"
                    ),
                });
            }
        }
        match &self.backend {
            FiniteFieldBackend::Prime(_) => {
                Ok(self.element(coefficients.first().copied().unwrap_or(0)))
            }
            FiniteFieldBackend::Extension(context) => Ok(self.wrap_extension(
                ExtensionElement::from_coefficients(Arc::clone(context), coefficients),
            )),
        }
    }

    /// Random access to the canonical `elements()` order without retaining
    /// all preceding elements. Used by bounded geometric point enumeration.
    pub(crate) fn element_at_index(&self, index: u64) -> Option<FiniteFieldElement> {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => {
                let value = Integer::from(index);
                (&value < residues.modulus()).then(|| self.wrap_prime(residues.element(value)))
            }
            FiniteFieldBackend::Extension(context) => {
                let mut remaining = index;
                let mut coefficients = vec![0; context.degree()];
                for coefficient in &mut coefficients {
                    *coefficient = remaining % context.characteristic_word;
                    remaining /= context.characteristic_word;
                }
                (remaining == 0).then(|| {
                    self.wrap_extension(ExtensionElement::from_coefficients(
                        Arc::clone(context),
                        &coefficients,
                    ))
                })
            }
        }
    }

    /// The distinguished power-basis generator. It is `1` for a prime field.
    pub fn generator(&self) -> FiniteFieldElement {
        match &self.backend {
            FiniteFieldBackend::Prime(_) => self.one(),
            FiniteFieldBackend::Extension(context) => self.wrap_extension(
                ExtensionElement::from_coefficients(Arc::clone(context), &[0, 1]),
            ),
        }
    }

    /// Serialize the complete verified presentation as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a finite-field presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    pub(crate) fn to_wire(&self) -> FiniteFieldWire {
        FiniteFieldWire {
            type_id: Self::TYPE_ID.to_owned(),
            characteristic: self.characteristic().to_string(),
            modulus: self.defining_modulus().map(<[u64]>::to_vec),
        }
    }

    pub(crate) fn from_wire(wire: FiniteFieldWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let characteristic = wire
            .characteristic
            .parse::<Integer>()
            .map_err(|error| format!("invalid finite-field characteristic: {error}"))?;
        match wire.modulus {
            None => Self::prime(characteristic).map_err(|error| error.to_string()),
            Some(modulus) => {
                let characteristic = wire.characteristic.parse::<u64>().map_err(|_| {
                    "extension-field characteristic does not fit the word-sized FLINT backend"
                        .to_owned()
                })?;
                Self::with_modulus(characteristic, modulus).map_err(|error| error.to_string())
            }
        }
    }

    fn require(&self, value: &FiniteFieldElement) -> Result<(), ContextMismatchError> {
        if self.contains(value) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: self.to_string(),
                actual: value.field.to_string(),
            })
        }
    }

    fn wrap_prime(&self, value: ResidueClass) -> FiniteFieldElement {
        let representative = value.representative().clone();
        FiniteFieldElement {
            field: self.clone(),
            value: FiniteFieldElementValue::Prime(value),
            representative: OnceLock::from(representative),
        }
    }

    pub(crate) fn wrap_extension(&self, value: ExtensionElement) -> FiniteFieldElement {
        FiniteFieldElement {
            field: self.clone(),
            value: FiniteFieldElementValue::Extension(value),
            representative: OnceLock::new(),
        }
    }
}

impl Serialize for FiniteField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FiniteField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(FiniteFieldWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for FiniteField {
    fn eq(&self, other: &Self) -> bool {
        match (&self.backend, &other.backend) {
            (FiniteFieldBackend::Prime(left), FiniteFieldBackend::Prime(right)) => left == right,
            (FiniteFieldBackend::Extension(left), FiniteFieldBackend::Extension(right)) => {
                left.characteristic_word == right.characteristic_word
                    && left.modulus == right.modulus
            }
            _ => false,
        }
    }
}

impl Eq for FiniteField {}

impl fmt::Debug for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.defining_modulus() {
            Some(modulus) => f
                .debug_struct("FiniteField")
                .field("characteristic", self.characteristic())
                .field("modulus", &modulus)
                .finish(),
            None => f
                .debug_tuple("FiniteField")
                .field(self.characteristic())
                .finish(),
        }
    }
}

impl fmt::Display for FiniteField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.degree() == 1 {
            write!(f, "F_{}", self.characteristic())
        } else {
            write!(
                f,
                "F_{}^{} with modulus {:?}",
                self.characteristic(),
                self.degree(),
                self.defining_modulus().unwrap_or(&[])
            )
        }
    }
}

impl SemanticType for FiniteField {
    const TYPE_ID: &'static str = "noether/numthy/FiniteField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = if self.degree() == 1 {
            b"prime".to_vec()
        } else {
            b"polynomial".to_vec()
        };
        append_component(&mut output, self.characteristic().canonical_bytes()?)?;
        if let Some(modulus) = self.defining_modulus() {
            let count = u64::try_from(modulus.len()).map_err(|_| {
                NoetherError::Serialization(
                    "finite-field modulus is too large for canonical framing".to_owned(),
                )
            })?;
            output.extend_from_slice(&count.to_be_bytes());
            for coefficient in modulus {
                output.extend_from_slice(&coefficient.to_be_bytes());
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_FIELD_DESCRIPTION
    }
}

impl Ring for FiniteField {
    type Element = FiniteFieldElement;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.field
    }

    fn zero(&self) -> Self::Element {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => self.wrap_prime(residues.zero()),
            FiniteFieldBackend::Extension(context) => {
                self.wrap_extension(ExtensionElement::zero(Arc::clone(context)))
            }
        }
    }

    fn one(&self) -> Self::Element {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => self.wrap_prime(residues.one()),
            FiniteFieldBackend::Extension(context) => {
                self.wrap_extension(ExtensionElement::one(Arc::clone(context)))
            }
        }
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        match (&self.backend, &a.value, &b.value) {
            (
                FiniteFieldBackend::Prime(residues),
                FiniteFieldElementValue::Prime(left),
                FiniteFieldElementValue::Prime(right),
            ) => Ok(self.wrap_prime(residues.add(left, right)?)),
            (
                FiniteFieldBackend::Extension(context),
                FiniteFieldElementValue::Extension(left),
                FiniteFieldElementValue::Extension(right),
            ) => Ok(self.wrap_extension(ExtensionElement::binary(
                Arc::clone(context),
                left,
                right,
                fq_nmod_add,
            ))),
            _ => Err(internal_context_mismatch(self, a)),
        }
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        match (&self.backend, &a.value, &b.value) {
            (
                FiniteFieldBackend::Prime(residues),
                FiniteFieldElementValue::Prime(left),
                FiniteFieldElementValue::Prime(right),
            ) => Ok(self.wrap_prime(residues.sub(left, right)?)),
            (
                FiniteFieldBackend::Extension(context),
                FiniteFieldElementValue::Extension(left),
                FiniteFieldElementValue::Extension(right),
            ) => Ok(self.wrap_extension(ExtensionElement::binary(
                Arc::clone(context),
                left,
                right,
                fq_nmod_sub,
            ))),
            _ => Err(internal_context_mismatch(self, a)),
        }
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.require(a)?;
        self.require(b)?;
        match (&self.backend, &a.value, &b.value) {
            (
                FiniteFieldBackend::Prime(residues),
                FiniteFieldElementValue::Prime(left),
                FiniteFieldElementValue::Prime(right),
            ) => Ok(self.wrap_prime(residues.mul(left, right)?)),
            (
                FiniteFieldBackend::Extension(context),
                FiniteFieldElementValue::Extension(left),
                FiniteFieldElementValue::Extension(right),
            ) => Ok(self.wrap_extension(ExtensionElement::binary(
                Arc::clone(context),
                left,
                right,
                fq_nmod_mul,
            ))),
            _ => Err(internal_context_mismatch(self, a)),
        }
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.require(value)?;
        match (&self.backend, &value.value) {
            (FiniteFieldBackend::Prime(residues), FiniteFieldElementValue::Prime(inner)) => {
                Ok(self.wrap_prime(residues.neg(inner)?))
            }
            (FiniteFieldBackend::Extension(context), FiniteFieldElementValue::Extension(inner)) => {
                Ok(self.wrap_extension(ExtensionElement::unary(
                    Arc::clone(context),
                    inner,
                    fq_nmod_neg,
                )))
            }
            _ => Err(internal_context_mismatch(self, value)),
        }
    }
}

impl CommutativeRing for FiniteField {}

impl UnitInversion for FiniteField {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(UnitError::from)
    }
}

impl Field for FiniteField {
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError> {
        self.require(value)?;
        if value.is_zero() {
            return Err(DivisionByZeroError.into());
        }
        match (&self.backend, &value.value) {
            (FiniteFieldBackend::Prime(_), FiniteFieldElementValue::Prime(inner)) => inner
                .inverse()
                .map(|inverse| self.wrap_prime(inverse))
                .map_err(|_| InversionError::from(DivisionByZeroError)),
            (FiniteFieldBackend::Extension(context), FiniteFieldElementValue::Extension(inner)) => {
                Ok(self.wrap_extension(ExtensionElement::unary(
                    Arc::clone(context),
                    inner,
                    fq_nmod_inv,
                )))
            }
            _ => Err(internal_context_mismatch(self, value).into()),
        }
    }
}

impl FiniteContext for FiniteField {
    fn cardinality(&self) -> Integer {
        self.order()
    }

    fn elements(&self) -> Box<dyn Iterator<Item = Self::Element> + '_> {
        match &self.backend {
            FiniteFieldBackend::Prime(residues) => {
                Box::new(residues.elements().map(|value| self.wrap_prime(value)))
            }
            FiniteFieldBackend::Extension(context) => Box::new(ExtensionElements {
                field: self,
                characteristic: context.characteristic_word,
                next_coefficients: Some(vec![0; context.degree()]),
            }),
        }
    }
}

#[derive(Clone)]
enum FiniteFieldElementValue {
    Prime(ResidueClass),
    Extension(ExtensionElement),
}

/// An exact value carrying its concrete [`FiniteField`] presentation.
#[derive(Clone)]
pub struct FiniteFieldElement {
    field: FiniteField,
    value: FiniteFieldElementValue,
    // Lazily cached base-p encoding of the canonical power-basis coefficients.
    // FLINT storage remains authoritative; ordinary extension-field arithmetic
    // therefore does not unpack every result merely to maintain a duplicate.
    representative: OnceLock<Integer>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteFieldElementWire {
    #[serde(rename = "type")]
    type_id: String,
    field: FiniteFieldWire,
    coefficients: Vec<String>,
}

impl FiniteFieldElement {
    /// The concrete finite field containing this value.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Canonical base-`p` encoding of the power-basis coefficients in
    /// `[0, |F|)`. For a prime field this is the usual residue representative.
    pub fn representative(&self) -> &Integer {
        self.representative.get_or_init(|| match &self.value {
            FiniteFieldElementValue::Prime(value) => value.representative().clone(),
            FiniteFieldElementValue::Extension(value) => {
                encode_coefficients(value.context.characteristic_word, &value.coefficients())
            }
        })
    }

    pub(crate) fn extension_value(&self) -> Option<&ExtensionElement> {
        match &self.value {
            FiniteFieldElementValue::Prime(_) => None,
            FiniteFieldElementValue::Extension(value) => Some(value),
        }
    }

    /// Ascending coefficients in the field's canonical power basis.
    pub fn coefficients(&self) -> Vec<Integer> {
        match &self.value {
            FiniteFieldElementValue::Prime(value) => {
                vec![value.representative().clone()]
            }
            FiniteFieldElementValue::Extension(value) => value
                .coefficients()
                .into_iter()
                .map(Integer::from)
                .collect(),
        }
    }

    /// Whether this value is zero.
    pub fn is_zero(&self) -> bool {
        match &self.value {
            FiniteFieldElementValue::Prime(value) => value.is_zero(),
            FiniteFieldElementValue::Extension(value) => unsafe {
                fq_nmod_is_zero(&value.raw, value.context.as_ptr()) != 0
            },
        }
    }

    /// Whether this value is one.
    pub fn is_one(&self) -> bool {
        match &self.value {
            FiniteFieldElementValue::Prime(value) => value.is_one(),
            FiniteFieldElementValue::Extension(value) => unsafe {
                fq_nmod_is_one(&value.raw, value.context.as_ptr()) != 0
            },
        }
    }

    /// Multiplicative inverse, failing on zero.
    pub fn inverse(&self) -> Result<Self, InversionError> {
        self.field.inverse(self)
    }

    /// Raise this value to a nonnegative power.
    pub fn pow(&self, exponent: u64) -> Self {
        match &self.value {
            FiniteFieldElementValue::Prime(value) => self.field.wrap_prime(value.pow(exponent)),
            FiniteFieldElementValue::Extension(value) => {
                let mut output = ExtensionElement::initialized(Arc::clone(&value.context));
                unsafe {
                    fq_nmod_pow_ui(
                        &mut output.raw,
                        &value.raw,
                        exponent,
                        output.context.as_ptr(),
                    )
                };
                self.field.wrap_extension(output)
            }
        }
    }

    /// Apply a power of the characteristic Frobenius.
    pub fn frobenius(&self, exponent: u64) -> Self {
        match &self.value {
            FiniteFieldElementValue::Prime(_) => self.clone(),
            FiniteFieldElementValue::Extension(value) => {
                let reduced = exponent % self.field.degree() as u64;
                let mut output = ExtensionElement::initialized(Arc::clone(&value.context));
                unsafe {
                    fq_nmod_frobenius(
                        &mut output.raw,
                        &value.raw,
                        reduced as i64,
                        output.context.as_ptr(),
                    )
                };
                self.field.wrap_extension(output)
            }
        }
    }

    /// Field trace to the prime subfield, embedded back into this field.
    pub fn trace(&self) -> Self {
        match &self.value {
            FiniteFieldElementValue::Prime(_) => self.clone(),
            FiniteFieldElementValue::Extension(value) => {
                let mut trace = Integer::zero();
                unsafe { fq_nmod_trace(trace.as_mut_ptr(), &value.raw, value.context.as_ptr()) };
                self.field.element(trace)
            }
        }
    }

    /// Field norm to the prime subfield, embedded back into this field.
    pub fn norm(&self) -> Self {
        match &self.value {
            FiniteFieldElementValue::Prime(_) => self.clone(),
            FiniteFieldElementValue::Extension(value) => {
                let mut norm = Integer::zero();
                unsafe { fq_nmod_norm(norm.as_mut_ptr(), &value.raw, value.context.as_ptr()) };
                self.field.element(norm)
            }
        }
    }

    /// Serialize the field presentation and canonical power-basis coordinates.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate both the field and element coordinates.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn to_wire(&self) -> FiniteFieldElementWire {
        FiniteFieldElementWire {
            type_id: Self::TYPE_ID.to_owned(),
            field: self.field.to_wire(),
            coefficients: self
                .coefficients()
                .into_iter()
                .map(|coefficient| coefficient.to_string())
                .collect(),
        }
    }

    fn from_wire(wire: FiniteFieldElementWire) -> Result<Self, String> {
        if wire.type_id != Self::TYPE_ID {
            return Err(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            ));
        }
        let field = FiniteField::from_wire(wire.field)?;
        if wire.coefficients.len() != field.degree() {
            return Err(format!(
                "expected {} canonical power-basis coefficients, received {}",
                field.degree(),
                wire.coefficients.len()
            ));
        }
        if field.degree() == 1 {
            let representative = wire.coefficients[0]
                .parse::<Integer>()
                .map_err(|error| format!("invalid prime-field representative: {error}"))?;
            if representative < Integer::zero() || representative >= *field.characteristic() {
                return Err(format!(
                    "prime-field representative must lie in [0, {})",
                    field.characteristic()
                ));
            }
            return Ok(field.element(representative));
        }
        let coefficients = wire
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(index, coefficient)| {
                coefficient.parse::<u64>().map_err(|_| {
                    format!(
                        "finite-field coefficient {index} is not a canonical word-sized integer"
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        field
            .element_from_coefficients(&coefficients)
            .map_err(|error| error.to_string())
    }
}

impl Serialize for FiniteFieldElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_wire().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FiniteFieldElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_wire(FiniteFieldElementWire::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for FiniteFieldElement {
    fn eq(&self, other: &Self) -> bool {
        if self.field != other.field {
            return false;
        }
        match (&self.value, &other.value) {
            (FiniteFieldElementValue::Prime(left), FiniteFieldElementValue::Prime(right)) => {
                left == right
            }
            (
                FiniteFieldElementValue::Extension(left),
                FiniteFieldElementValue::Extension(right),
            ) => unsafe { fq_nmod_equal(&left.raw, &right.raw, left.context.as_ptr()) != 0 },
            _ => false,
        }
    }
}

impl Eq for FiniteFieldElement {}

impl fmt::Debug for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FiniteFieldElement")
            .field("coefficients", &self.coefficients())
            .field("field", &self.field)
            .finish()
    }
}

impl fmt::Display for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.field.degree() == 1 {
            write!(f, "{} in {}", self.representative(), self.field)
        } else {
            write!(f, "{:?} in {}", self.coefficients(), self.field)
        }
    }
}

impl SemanticType for FiniteFieldElement {
    const TYPE_ID: &'static str = "noether/numthy/FiniteFieldElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, self.field.canonical_bytes()?)?;
        let coefficients = self.coefficients();
        let count = u64::try_from(coefficients.len()).map_err(|_| {
            NoetherError::Serialization(
                "finite-field element has too many coefficients to serialize".to_owned(),
            )
        })?;
        output.extend_from_slice(&count.to_be_bytes());
        for coefficient in coefficients {
            append_component(&mut output, coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_FIELD_ELEMENT_DESCRIPTION
    }
}

impl Add<&FiniteFieldElement> for &FiniteFieldElement {
    type Output = Result<FiniteFieldElement, ContextMismatchError>;

    fn add(self, rhs: &FiniteFieldElement) -> Self::Output {
        self.field.add(self, rhs)
    }
}

impl Sub<&FiniteFieldElement> for &FiniteFieldElement {
    type Output = Result<FiniteFieldElement, ContextMismatchError>;

    fn sub(self, rhs: &FiniteFieldElement) -> Self::Output {
        self.field.sub(self, rhs)
    }
}

impl Mul<&FiniteFieldElement> for &FiniteFieldElement {
    type Output = Result<FiniteFieldElement, ContextMismatchError>;

    fn mul(self, rhs: &FiniteFieldElement) -> Self::Output {
        self.field.mul(self, rhs)
    }
}

impl Neg for &FiniteFieldElement {
    type Output = Result<FiniteFieldElement, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.field.neg(self)
    }
}

struct ExtensionElements<'a> {
    field: &'a FiniteField,
    characteristic: u64,
    next_coefficients: Option<Vec<u64>>,
}

impl Iterator for ExtensionElements<'_> {
    type Item = FiniteFieldElement;

    fn next(&mut self) -> Option<Self::Item> {
        let coefficients = self.next_coefficients.clone()?;
        let result = match &self.field.backend {
            FiniteFieldBackend::Extension(context) => {
                self.field
                    .wrap_extension(ExtensionElement::from_coefficients(
                        Arc::clone(context),
                        &coefficients,
                    ))
            }
            FiniteFieldBackend::Prime(_) => return None,
        };

        let mut following = coefficients;
        let mut carry = true;
        for coefficient in &mut following {
            if !carry {
                break;
            }
            *coefficient += 1;
            if *coefficient == self.characteristic {
                *coefficient = 0;
            } else {
                carry = false;
            }
        }
        self.next_coefficients = if carry { None } else { Some(following) };
        Some(result)
    }
}

fn encode_coefficients(characteristic: u64, coefficients: &[u64]) -> Integer {
    let characteristic = Integer::from(characteristic);
    let mut result = Integer::zero();
    let mut place = Integer::one();
    for coefficient in coefficients {
        result = &result + &(&place * &Integer::from(*coefficient));
        place = &place * &characteristic;
    }
    result
}

fn append_component(output: &mut Vec<u8>, component: Vec<u8>) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("finite-field component is too large to frame".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(&component);
    Ok(())
}

fn internal_context_mismatch(
    field: &FiniteField,
    value: &FiniteFieldElement,
) -> ContextMismatchError {
    ContextMismatchError {
        expected: field.to_string(),
        actual: value.field.to_string(),
    }
}

#[cfg(test)]
mod enumeration_index_tests {
    use super::*;

    #[test]
    fn indexed_elements_match_canonical_iteration_and_reject_out_of_range_indices() {
        for field in [
            FiniteField::prime(7).unwrap(),
            FiniteField::extension(2, 3).unwrap(),
            FiniteField::extension(3, 3).unwrap(),
            FiniteField::with_modulus(3, vec![1, 0, 1]).unwrap(),
        ] {
            for (index, value) in field.elements().enumerate() {
                assert_eq!(field.element_at_index(index as u64), Some(value));
            }
            assert!(field
                .element_at_index(field.order().to_u64_checked().unwrap())
                .is_none());
            assert!(field.element_at_index(u64::MAX).is_none());
        }
        let huge = FiniteField::prime(
            "170141183460469231731687303715884105727"
                .parse::<Integer>()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            huge.element_at_index(u64::MAX),
            Some(huge.element(u64::MAX))
        );
    }
}
