// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! FLINT-owned arbitrary-precision rational numbers.

#![allow(unsafe_code)]

use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;
use std::ffi::CStr;
use std::mem::MaybeUninit;

use flint_sys::{
    flint_free, fmpq, fmpq_add, fmpq_canonicalise, fmpq_clear, fmpq_cmp, fmpq_equal, fmpq_get_d,
    fmpq_get_str, fmpq_init, fmpq_inv, fmpq_is_one, fmpq_is_zero, fmpq_mat_clear, fmpq_mat_init,
    fmpq_mat_rref, fmpq_mat_struct, fmpq_mul, fmpq_neg, fmpq_reconstruct_fmpz, fmpq_set,
    fmpq_set_fmpz_frac, fmpq_set_si, fmpq_sub, fmpq_swap, fmpz_sgn, slong,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Integer;

/// Division or inversion was requested for zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DivisionByZeroError;

impl fmt::Display for DivisionByZeroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("division by zero")
    }
}

impl std::error::Error for DivisionByZeroError {}

/// Error returned when a rational literal is malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRationalError {
    literal: String,
    reason: String,
}

impl ParseRationalError {
    fn new(literal: &str, reason: impl Into<String>) -> Self {
        Self {
            literal: literal.to_owned(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ParseRationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid rational literal {:?}: {}",
            self.literal, self.reason
        )
    }
}

impl std::error::Error for ParseRationalError {}

/// A flat rational matrix does not match its declared dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RationalMatrixShapeError {
    rows: usize,
    columns: usize,
    entries: usize,
}

impl fmt::Display for RationalMatrixShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a {} by {} rational matrix requires {} entries, received {}",
            self.rows,
            self.columns,
            self.rows.saturating_mul(self.columns),
            self.entries
        )
    }
}

impl std::error::Error for RationalMatrixShapeError {}

/// An arbitrary-precision rational backed directly by FLINT `fmpq` storage.
///
/// Values are always reduced and have a positive denominator.
pub struct Rational {
    inner: fmpq,
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalWire {
    numerator: String,
    denominator: String,
}

#[cfg(feature = "serde")]
impl Serialize for Rational {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        RationalWire {
            numerator: self.numerator().to_string(),
            denominator: self.denominator().to_string(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Rational {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RationalWire::deserialize(deserializer)?;
        let numerator = wire
            .numerator
            .parse::<Integer>()
            .map_err(serde::de::Error::custom)?;
        let denominator = wire
            .denominator
            .parse::<Integer>()
            .map_err(serde::de::Error::custom)?;
        let value =
            Self::new(numerator.clone(), denominator.clone()).map_err(serde::de::Error::custom)?;
        if value.numerator() != numerator || value.denominator() != denominator {
            return Err(serde::de::Error::custom(
                "rational coordinates are not canonical",
            ));
        }
        Ok(value)
    }
}

impl Rational {
    /// Construct and canonicalize `numerator / denominator`.
    pub fn new(numerator: Integer, denominator: Integer) -> Result<Self, DivisionByZeroError> {
        if denominator.is_zero() {
            return Err(DivisionByZeroError);
        }
        let mut output = Self::initialized();
        unsafe {
            fmpq_set_fmpz_frac(
                output.as_mut_ptr(),
                numerator.as_ptr(),
                denominator.as_ptr(),
            );
            fmpq_canonicalise(output.as_mut_ptr());
        }
        Ok(output)
    }

    /// The rational zero.
    pub fn zero() -> Self {
        Self::initialized()
    }

    /// The rational one.
    pub fn one() -> Self {
        Self::from(1_i64)
    }

    /// Reconstruct the unique reduced rational with numerator and denominator
    /// bounded by `floor(sqrt((modulus - 1) / 2))`.
    ///
    /// Returns `None` unless `modulus` is positive, `residue` lies in
    /// `0..modulus`, and such a symmetric-bound reconstruction exists.
    pub fn reconstruct_symmetric(residue: &Integer, modulus: &Integer) -> Option<Self> {
        if !modulus.is_positive() || residue.is_negative() || residue >= modulus {
            return None;
        }
        let mut output = Self::initialized();
        let reconstructed = unsafe {
            fmpq_reconstruct_fmpz(output.as_mut_ptr(), residue.as_ptr(), modulus.as_ptr()) != 0
        };
        if reconstructed {
            unsafe { fmpq_canonicalise(output.as_mut_ptr()) };
            Some(output)
        } else {
            None
        }
    }

    /// Whether this rational is zero.
    pub fn is_zero(&self) -> bool {
        unsafe { fmpq_is_zero(self.as_ptr()) != 0 }
    }

    /// Whether this rational is one.
    pub fn is_one(&self) -> bool {
        unsafe { fmpq_is_one(self.as_ptr()) != 0 }
    }

    /// Whether this value is positive.
    pub fn is_positive(&self) -> bool {
        unsafe { fmpz_sgn(&self.inner.num) > 0 }
    }

    /// Whether this value is negative.
    pub fn is_negative(&self) -> bool {
        unsafe { fmpz_sgn(&self.inner.num) < 0 }
    }

    /// Convert a finite `f64` to its exact dyadic rational value.
    pub fn from_f64_exact(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        if value == 0.0 {
            return Some(Self::zero());
        }
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (significand, exponent) = if encoded_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1_u64 << 52), encoded_exponent - 1023 - 52)
        };
        let mut numerator = Integer::from(significand);
        if negative {
            numerator = -numerator;
        }
        if exponent >= 0 {
            Some(Self::from(numerator << exponent as u32))
        } else {
            Self::new(numerator, Integer::one() << (-exponent) as u32).ok()
        }
    }

    /// Round to the nearest `f64`, returning `None` on overflow.
    pub fn to_f64(&self) -> Option<f64> {
        let value = unsafe { fmpq_get_d(self.as_ptr()) };
        value.is_finite().then_some(value)
    }

    /// The canonical numerator.
    pub fn numerator(&self) -> Integer {
        unsafe { Integer::clone_from_ptr(&self.inner.num) }
    }

    /// The positive canonical denominator.
    pub fn denominator(&self) -> Integer {
        unsafe { Integer::clone_from_ptr(&self.inner.den) }
    }

    /// The multiplicative inverse, failing on zero.
    pub fn inverse(&self) -> Result<Self, DivisionByZeroError> {
        if self.is_zero() {
            return Err(DivisionByZeroError);
        }
        let mut output = Self::initialized();
        unsafe { fmpq_inv(output.as_mut_ptr(), self.as_ptr()) };
        Ok(output)
    }

    /// Divide by a nonzero rational.
    pub fn checked_div(&self, divisor: &Self) -> Result<Self, DivisionByZeroError> {
        Ok(self * &divisor.inverse()?)
    }

    /// Raise this rational to a nonnegative power.
    pub fn pow(&self, exponent: u64) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut power = exponent;
        while power != 0 {
            if power & 1 == 1 {
                result = &result * &base;
            }
            power >>= 1;
            if power != 0 {
                base = &base * &base;
            }
        }
        result
    }

    /// Construct an initialized zero value.
    #[doc(hidden)]
    pub fn initialized() -> Self {
        let mut inner = MaybeUninit::<fmpq>::uninit();
        unsafe { fmpq_init(inner.as_mut_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    #[doc(hidden)]
    pub fn as_ptr(&self) -> *const fmpq {
        &self.inner
    }

    #[doc(hidden)]
    pub fn as_mut_ptr(&mut self) -> *mut fmpq {
        &mut self.inner
    }

    /// Deep-copy a valid initialized FLINT rational.
    ///
    /// # Safety
    ///
    /// `value` must point to a live initialized `fmpq` for this call.
    #[doc(hidden)]
    pub unsafe fn clone_from_ptr(value: *const fmpq) -> Self {
        let mut output = Self::initialized();
        unsafe { fmpq_set(output.as_mut_ptr(), value) };
        output
    }
}

/// Compute the reduced row-echelon form of a row-major dense rational matrix.
///
/// This is a safe bulk adapter around FLINT's fraction-free rational matrix
/// elimination. The returned flat matrix has the same row-major shape.
pub fn rational_matrix_rref(
    rows: usize,
    columns: usize,
    entries: &[Rational],
) -> Result<(Vec<Rational>, usize), RationalMatrixShapeError> {
    rational_matrix_rref_owned(rows, columns, entries.to_vec())
}

/// Compute rational reduced row-echelon form while consuming flat entries.
///
/// Native rational values are swapped into and out of FLINT's matrix storage,
/// avoiding one arbitrary-precision clone at each side of the bulk operation.
pub fn rational_matrix_rref_owned(
    rows: usize,
    columns: usize,
    entries: Vec<Rational>,
) -> Result<(Vec<Rational>, usize), RationalMatrixShapeError> {
    let Some(expected) = rows.checked_mul(columns) else {
        return Err(RationalMatrixShapeError {
            rows,
            columns,
            entries: entries.len(),
        });
    };
    let Ok(rows_ffi) = slong::try_from(rows) else {
        return Err(RationalMatrixShapeError {
            rows,
            columns,
            entries: entries.len(),
        });
    };
    let Ok(columns_ffi) = slong::try_from(columns) else {
        return Err(RationalMatrixShapeError {
            rows,
            columns,
            entries: entries.len(),
        });
    };
    if entries.len() != expected {
        return Err(RationalMatrixShapeError {
            rows,
            columns,
            entries: entries.len(),
        });
    }
    let input = RawRationalMatrix::from_entries(rows_ffi, columns_ffi, rows, columns, entries);
    let mut output = RawRationalMatrix::new(rows_ffi, columns_ffi, rows, columns);
    let rank = unsafe { fmpq_mat_rref(output.as_mut_ptr(), input.as_ptr()) } as usize;
    Ok((output.into_entries(), rank))
}

struct RawRationalMatrix {
    inner: fmpq_mat_struct,
    rows: usize,
    columns: usize,
}

impl RawRationalMatrix {
    fn new(rows_ffi: slong, columns_ffi: slong, rows: usize, columns: usize) -> Self {
        let mut inner = MaybeUninit::<fmpq_mat_struct>::uninit();
        unsafe { fmpq_mat_init(inner.as_mut_ptr(), rows_ffi, columns_ffi) };
        Self {
            inner: unsafe { inner.assume_init() },
            rows,
            columns,
        }
    }

    fn from_entries(
        rows_ffi: slong,
        columns_ffi: slong,
        rows: usize,
        columns: usize,
        mut entries: Vec<Rational>,
    ) -> Self {
        let matrix = Self::new(rows_ffi, columns_ffi, rows, columns);
        for (index, entry) in entries.iter_mut().enumerate() {
            unsafe {
                fmpq_swap(
                    matrix.entry_ptr(index / columns, index % columns),
                    entry.as_mut_ptr(),
                )
            };
        }
        matrix
    }

    fn into_entries(self) -> Vec<Rational> {
        let mut entries = Vec::with_capacity(self.rows.saturating_mul(self.columns));
        for row in 0..self.rows {
            for column in 0..self.columns {
                let mut entry = Rational::zero();
                unsafe { fmpq_swap(entry.as_mut_ptr(), self.entry_ptr(row, column)) };
                entries.push(entry);
            }
        }
        entries
    }

    fn entry_ptr(&self, row: usize, column: usize) -> *mut fmpq {
        let offset = row * self.inner.stride as usize + column;
        unsafe { self.inner.entries.add(offset) }
    }

    fn as_ptr(&self) -> *const fmpq_mat_struct {
        &self.inner
    }

    fn as_mut_ptr(&mut self) -> *mut fmpq_mat_struct {
        &mut self.inner
    }
}

impl Drop for RawRationalMatrix {
    fn drop(&mut self) {
        unsafe { fmpq_mat_clear(self.as_mut_ptr()) };
    }
}

impl Default for Rational {
    fn default() -> Self {
        Self::zero()
    }
}

impl Drop for Rational {
    fn drop(&mut self) {
        unsafe { fmpq_clear(self.as_mut_ptr()) };
    }
}

impl Clone for Rational {
    fn clone(&self) -> Self {
        unsafe { Self::clone_from_ptr(self.as_ptr()) }
    }
}

impl PartialEq for Rational {
    fn eq(&self, other: &Self) -> bool {
        unsafe { fmpq_equal(self.as_ptr(), other.as_ptr()) != 0 }
    }
}

impl Eq for Rational {}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        unsafe { fmpq_cmp(self.as_ptr(), other.as_ptr()) }.cmp(&0)
    }
}

impl Hash for Rational {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.numerator().hash(state);
        self.denominator().hash(state);
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pointer = unsafe { fmpq_get_str(core::ptr::null_mut(), 10, self.as_ptr()) };
        if pointer.is_null() {
            return Err(fmt::Error);
        }
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { flint_free(pointer.cast()) };
        f.write_str(&value)
    }
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rational({self})")
    }
}

impl FromStr for Rational {
    type Err = ParseRationalError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut parts = value.split('/');
        let numerator = parts
            .next()
            .unwrap_or_default()
            .parse::<Integer>()
            .map_err(|error| ParseRationalError::new(value, error.to_string()))?;
        let denominator = match parts.next() {
            Some(part) => part
                .parse::<Integer>()
                .map_err(|error| ParseRationalError::new(value, error.to_string()))?,
            None => Integer::one(),
        };
        if parts.next().is_some() {
            return Err(ParseRationalError::new(value, "too many '/' separators"));
        }
        Self::new(numerator, denominator)
            .map_err(|error| ParseRationalError::new(value, error.to_string()))
    }
}

impl From<Integer> for Rational {
    fn from(value: Integer) -> Self {
        Self::new(value, Integer::one()).expect("one is nonzero")
    }
}

macro_rules! impl_binary_operator {
    ($type:ty, $trait:ident, $method:ident, $function:path) => {
        impl $trait for $type {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        impl $trait<&$type> for &$type {
            type Output = $type;

            fn $method(self, rhs: &$type) -> Self::Output {
                let mut output = <$type>::default();
                unsafe { $function(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
                output
            }
        }

        impl $trait<&$type> for $type {
            type Output = $type;

            fn $method(self, rhs: &$type) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl $trait<$type> for &$type {
            type Output = $type;

            fn $method(self, rhs: $type) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

macro_rules! impl_rational_from_signed {
    ($($primitive:ty),+ $(,)?) => {
        $(
            impl From<$primitive> for Rational {
                fn from(value: $primitive) -> Self {
                    let mut output = Self::initialized();
                    unsafe { fmpq_set_si(output.as_mut_ptr(), value as i64, 1) };
                    output
                }
            }
        )+
    };
}

impl_rational_from_signed!(i8, i16, i32, i64);

macro_rules! impl_rational_via_integer {
    ($($primitive:ty),+ $(,)?) => {
        $(
            impl From<$primitive> for Rational {
                fn from(value: $primitive) -> Self {
                    Self::from(Integer::from(value))
                }
            }
        )+
    };
}

impl_rational_via_integer!(isize, i128, u8, u16, u32, u64, u128, usize);

impl_binary_operator!(Rational, Add, add, fmpq_add);
impl_binary_operator!(Rational, Sub, sub, fmpq_sub);
impl_binary_operator!(Rational, Mul, mul, fmpq_mul);

impl Div<&Rational> for &Rational {
    type Output = Rational;

    fn div(self, rhs: &Rational) -> Self::Output {
        self.checked_div(rhs).expect("rational division by zero")
    }
}

impl Div<Rational> for Rational {
    type Output = Rational;

    fn div(self, rhs: Rational) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl Div<&Rational> for Rational {
    type Output = Rational;

    fn div(self, rhs: &Rational) -> Self::Output {
        (&self).div(rhs)
    }
}

impl Div<Rational> for &Rational {
    type Output = Rational;

    fn div(self, rhs: Rational) -> Self::Output {
        self.div(&rhs)
    }
}

macro_rules! impl_assign_operator {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait<&Rational> for Rational {
            fn $method(&mut self, rhs: &Rational) {
                *self = &*self $operator rhs;
            }
        }

        impl $trait<Rational> for Rational {
            fn $method(&mut self, rhs: Rational) {
                self.$method(&rhs);
            }
        }
    };
}

impl_assign_operator!(AddAssign, add_assign, +);
impl_assign_operator!(SubAssign, sub_assign, -);
impl_assign_operator!(MulAssign, mul_assign, *);
impl_assign_operator!(DivAssign, div_assign, /);

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &Rational {
    type Output = Rational;

    fn neg(self) -> Self::Output {
        let mut output = Rational::initialized();
        unsafe { fmpq_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_sign_and_common_factor() {
        let value = Rational::new(Integer::from(-6), Integer::from(-8)).unwrap();
        assert_eq!(value.to_string(), "3/4");
        assert_eq!(value.numerator(), Integer::from(3));
        assert_eq!(value.denominator(), Integer::from(4));
    }

    #[test]
    fn exact_f64_conversion_handles_normal_and_subnormal_values() {
        assert_eq!(Rational::from_f64_exact(0.5).unwrap().to_string(), "1/2");
        assert_eq!(Rational::from_f64_exact(-0.0).unwrap(), Rational::zero());
        assert_eq!(
            Rational::from_f64_exact(f64::from_bits(1))
                .unwrap()
                .to_f64(),
            Some(f64::from_bits(1))
        );
        assert!(Rational::from_f64_exact(f64::INFINITY).is_none());
        assert!(Rational::from_f64_exact(f64::NAN).is_none());
    }

    #[test]
    fn checked_division_fails_on_zero() {
        assert_eq!(
            Rational::one().checked_div(&Rational::zero()),
            Err(DivisionByZeroError)
        );
    }

    #[test]
    fn symmetric_rational_reconstruction_uses_the_exact_modular_identity() {
        let modulus = Integer::from(1_000_003_u32) * Integer::from(1_000_033_u32);
        let expected = Rational::new(Integer::from(-29), Integer::from(37)).unwrap();
        let residue = (&expected.numerator()
            * expected
                .denominator()
                .invert(&modulus)
                .expect("the denominator is invertible"))
            % &modulus;
        let residue = if residue.is_negative() {
            residue + &modulus
        } else {
            residue
        };

        assert_eq!(
            Rational::reconstruct_symmetric(&residue, &modulus),
            Some(expected)
        );
        assert!(Rational::reconstruct_symmetric(&Integer::from(-1), &modulus).is_none());
        assert!(Rational::reconstruct_symmetric(&modulus, &modulus).is_none());
    }

    #[test]
    fn arithmetic_is_exact() {
        let third: Rational = "1/3".parse().unwrap();
        let sixth: Rational = "1/6".parse().unwrap();
        assert_eq!(&third + &sixth, "1/2".parse().unwrap());
        assert_eq!(&third * &sixth, "1/18".parse().unwrap());
        assert_eq!(&third / &sixth, Rational::from(2));
        assert_eq!(third.pow(3), "1/27".parse().unwrap());
    }

    #[test]
    fn dense_rational_rref_is_exact() {
        let entries = [1_i64, 2, 3, 2, 4, 7].map(Rational::from);
        let (reduced, rank) = rational_matrix_rref(2, 3, &entries).unwrap();
        assert_eq!(rank, 2);
        assert_eq!(
            reduced,
            [1_i64, 2, 0, 0, 0, 1]
                .map(Rational::from)
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(rational_matrix_rref(2, 4, &entries).is_err());
    }
}
