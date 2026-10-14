// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! FLINT-owned arbitrary-precision integers used by sparse `Z` kernels.

#![allow(unsafe_code)]

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, RemAssign, Shl, Sub, SubAssign};
use core::str::FromStr;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;

use flint_sys::{
    flint_free, fmpz, fmpz_abs, fmpz_add, fmpz_bits, fmpz_clear, fmpz_cmp, fmpz_divisible,
    fmpz_equal, fmpz_fdiv_q, fmpz_fdiv_ui, fmpz_get_si, fmpz_get_str, fmpz_get_ui, fmpz_init,
    fmpz_invmod, fmpz_is_one, fmpz_is_prime, fmpz_is_zero, fmpz_lcm, fmpz_mul, fmpz_mul_2exp,
    fmpz_neg, fmpz_root, fmpz_set, fmpz_set_si, fmpz_set_str, fmpz_set_ui, fmpz_sgn, fmpz_sqrt,
    fmpz_sub, fmpz_tdiv_q,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Error returned when FLINT rejects an integer literal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseIntegerError {
    literal: String,
    radix: u8,
}

impl fmt::Display for ParseIntegerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid base-{} integer literal {:?}",
            self.radix, self.literal
        )
    }
}

impl std::error::Error for ParseIntegerError {}

/// Failure to render an integer in a requested radix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerRadixError {
    radix: u8,
}

impl fmt::Display for IntegerRadixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "integer radix {} is outside 2..=62", self.radix)
    }
}

impl std::error::Error for IntegerRadixError {}

/// An arbitrary-precision integer backed directly by FLINT `fmpz` storage.
///
/// Small values are stored inline. Large values own GMP-managed limbs through
/// FLINT and are deep-cloned and cleared by this wrapper.
pub struct Integer {
    inner: fmpz,
}

#[cfg(feature = "serde")]
impl Serialize for Integer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Integer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse::<Self>()
            .map_err(serde::de::Error::custom)
    }
}

impl Integer {
    /// Construct zero.
    pub fn new() -> Self {
        Self::initialized()
    }

    /// Construct zero.
    pub fn zero() -> Self {
        Self::new()
    }

    /// Construct one.
    pub fn one() -> Self {
        Self::from(1_u8)
    }

    /// Whether this value is zero.
    pub fn is_zero(&self) -> bool {
        unsafe { fmpz_is_zero(self.as_ptr()) != 0 }
    }

    /// Whether this value is one.
    pub fn is_one(&self) -> bool {
        unsafe { fmpz_is_one(self.as_ptr()) != 0 }
    }

    /// Whether this integer is prime, using FLINT's proved primality test.
    pub fn is_prime(&self) -> bool {
        unsafe { fmpz_is_prime(self.as_ptr()) != 0 }
    }

    /// The sign of this integer: `-1`, `0`, or `1`.
    pub fn sign(&self) -> i8 {
        unsafe { fmpz_sgn(self.as_ptr()) as i8 }
    }

    /// Whether this integer is strictly positive.
    pub fn is_positive(&self) -> bool {
        self.sign() > 0
    }

    /// Whether this integer is strictly negative.
    pub fn is_negative(&self) -> bool {
        self.sign() < 0
    }

    /// Compare with zero.
    pub fn cmp0(&self) -> Ordering {
        unsafe { fmpz_sgn(self.as_ptr()) }.cmp(&0)
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        let mut output = Self::new();
        unsafe { fmpz_abs(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    /// Number of significant bits in the absolute value, with zero having
    /// bit length zero.
    pub fn bit_length(&self) -> usize {
        usize::try_from(unsafe { fmpz_bits(self.as_ptr()) })
            .expect("FLINT bit counts fit the host address space")
    }

    /// Nonnegative greatest common divisor.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut output = Self::zero();
        unsafe { flint_sys::fmpz_gcd(output.as_mut_ptr(), self.as_ptr(), other.as_ptr()) };
        output
    }

    /// Nonnegative least common multiple.
    pub fn lcm(&self, other: &Self) -> Self {
        let mut output = Self::zero();
        unsafe { fmpz_lcm(output.as_mut_ptr(), self.as_ptr(), other.as_ptr()) };
        output
    }

    /// Square this value.
    pub fn square(&self) -> Self {
        self * self
    }

    /// Raise this integer to a nonnegative power.
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

    /// Floor of the nonnegative square root.
    ///
    /// # Panics
    ///
    /// Panics when this value is negative.
    pub fn sqrt_floor(&self) -> Self {
        assert!(
            self.cmp0() != Ordering::Less,
            "square root of a negative integer"
        );
        let mut output = Self::new();
        unsafe { fmpz_sqrt(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    /// Exact `exponent`-th root, or `None` when none exists in `Z`.
    ///
    /// Exponent zero is undefined. A negative integer has an exact root only
    /// for odd exponents.
    pub fn nth_root_exact(&self, exponent: u64) -> Option<Self> {
        let exponent = i64::try_from(exponent).ok()?;
        if exponent == 0 || (self.sign() < 0 && exponent % 2 == 0) {
            return None;
        }
        let mut output = Self::new();
        let exact = unsafe { fmpz_root(output.as_mut_ptr(), self.as_ptr(), exponent) };
        (exact != 0).then_some(output)
    }

    /// Floor division, rounding the quotient toward negative infinity.
    ///
    /// # Panics
    ///
    /// Panics when `divisor` is zero.
    pub fn div_floor(&self, divisor: &Self) -> Self {
        assert!(!divisor.is_zero(), "integer division by zero");
        let mut output = Self::new();
        unsafe { fmpz_fdiv_q(output.as_mut_ptr(), self.as_ptr(), divisor.as_ptr()) };
        output
    }

    /// Whether `divisor` divides this integer exactly.
    pub fn is_divisible(&self, divisor: &Self) -> bool {
        !divisor.is_zero() && unsafe { fmpz_divisible(self.as_ptr(), divisor.as_ptr()) != 0 }
    }

    /// Multiplicative inverse modulo `modulus`, if it exists.
    pub fn invert(self, modulus: &Self) -> Option<Self> {
        if modulus.is_zero() {
            return None;
        }
        let mut output = Self::new();
        let invertible =
            unsafe { fmpz_invmod(output.as_mut_ptr(), self.as_ptr(), modulus.as_ptr()) };
        (invertible != 0).then_some(output)
    }

    /// Convert to `i64` when the value is in range.
    pub fn to_i64(&self) -> Option<i64> {
        if self < &Self::from(i64::MIN) || self > &Self::from(i64::MAX) {
            None
        } else {
            Some(unsafe { fmpz_get_si(self.as_ptr()) as i64 })
        }
    }

    /// Convert to `u64` when the value is in range.
    pub fn to_u64_checked(&self) -> Option<u64> {
        if self.sign() < 0 || self > &Self::from(u64::MAX) {
            None
        } else {
            Some(unsafe { fmpz_get_ui(self.as_ptr()) as u64 })
        }
    }

    /// Euclidean remainder modulo a nonzero `u32`.
    pub fn mod_u(&self, modulus: u32) -> u32 {
        assert_ne!(modulus, 0, "integer remainder by zero");
        unsafe { fmpz_fdiv_ui(self.as_ptr(), modulus.into()) as u32 }
    }

    /// Euclidean remainder modulo a nonzero `u64`.
    pub fn modulo_u64(&self, modulus: u64) -> u64 {
        assert_ne!(modulus, 0, "integer remainder by zero");
        unsafe { fmpz_fdiv_ui(self.as_ptr(), modulus) }
    }

    /// Replace this value with a deep copy of `other`.
    pub fn assign(&mut self, other: &Self) {
        unsafe { fmpz_set(self.as_mut_ptr(), other.as_ptr()) };
    }

    /// Borrow the underlying FLINT integer pointer for zero-copy FFI calls.
    ///
    /// The pointer is valid only while `self` is alive and must never be
    /// cleared or mutated through this const pointer.
    pub fn as_fmpz(&self) -> *const fmpz {
        self.as_ptr()
    }

    /// Borrow the underlying FLINT integer pointer for FFI calls.
    #[doc(hidden)]
    pub fn as_ptr(&self) -> *const fmpz {
        &self.inner
    }

    /// Mutably borrow the underlying FLINT integer pointer for FFI calls.
    #[doc(hidden)]
    pub fn as_mut_ptr(&mut self) -> *mut fmpz {
        &mut self.inner
    }

    /// Deep-copy a valid initialized FLINT integer.
    ///
    /// # Safety
    ///
    /// `value` must be non-null and point to an initialized `fmpz` that stays
    /// alive for the duration of this call.
    pub unsafe fn clone_from_fmpz(value: *const fmpz) -> Self {
        assert!(!value.is_null(), "cannot clone a null fmpz pointer");
        let mut output = Self::new();
        unsafe { fmpz_set(output.as_mut_ptr(), value) };
        output
    }

    /// Deep-copy a valid initialized FLINT integer.
    ///
    /// # Safety
    ///
    /// `value` must point to a live initialized `fmpz` for this call.
    #[doc(hidden)]
    pub unsafe fn clone_from_ptr(value: *const fmpz) -> Self {
        unsafe { Self::clone_from_fmpz(value) }
    }

    /// Parse using a FLINT radix in `2..=62`.
    pub fn from_str_radix(value: &str, radix: u8) -> Result<Self, ParseIntegerError> {
        if !(2..=62).contains(&radix) {
            return Err(ParseIntegerError {
                literal: value.to_owned(),
                radix,
            });
        }
        let Ok(value_c) = CString::new(value) else {
            return Err(ParseIntegerError {
                literal: value.to_owned(),
                radix,
            });
        };
        let mut output = Self::new();
        let status = unsafe { fmpz_set_str(output.as_mut_ptr(), value_c.as_ptr(), radix.into()) };
        if status == 0 {
            Ok(output)
        } else {
            Err(ParseIntegerError {
                literal: value.to_owned(),
                radix,
            })
        }
    }

    /// Render this integer in a FLINT radix in `2..=62`.
    pub fn to_string_radix(&self, radix: u8) -> Result<String, IntegerRadixError> {
        if !(2..=62).contains(&radix) {
            return Err(IntegerRadixError { radix });
        }
        let pointer = unsafe { fmpz_get_str(core::ptr::null_mut(), radix.into(), self.as_ptr()) };
        if pointer.is_null() {
            return Err(IntegerRadixError { radix });
        }
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { flint_free(pointer.cast()) };
        Ok(value)
    }

    /// Construct an initialized zero value.
    #[doc(hidden)]
    pub fn initialized() -> Self {
        let mut inner = MaybeUninit::<fmpz>::uninit();
        unsafe { fmpz_init(inner.as_mut_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn from_decimal(value: &str) -> Self {
        value
            .parse()
            .expect("a primitive integer has a valid decimal rendering")
    }

    fn trunc_div(&self, divisor: &Self) -> Self {
        assert!(!divisor.is_zero(), "integer division by zero");
        let mut output = Self::new();
        unsafe { fmpz_tdiv_q(output.as_mut_ptr(), self.as_ptr(), divisor.as_ptr()) };
        output
    }

    fn trunc_rem(&self, divisor: &Self) -> Self {
        assert!(!divisor.is_zero(), "integer remainder by zero");
        self - &(&self.trunc_div(divisor) * divisor)
    }
}

impl Default for Integer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Integer {
    fn drop(&mut self) {
        unsafe { fmpz_clear(self.as_mut_ptr()) };
    }
}

impl Clone for Integer {
    fn clone(&self) -> Self {
        let mut output = Self::new();
        output.assign(self);
        output
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        unsafe { fmpz_equal(self.as_ptr(), other.as_ptr()) != 0 }
    }
}

impl Eq for Integer {}

impl Hash for Integer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> Ordering {
        unsafe { fmpz_cmp(self.as_ptr(), other.as_ptr()) }.cmp(&0)
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_string_radix(10).map_err(|_| fmt::Error)?)
    }
}

impl fmt::Debug for Integer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Integer({self})")
    }
}

impl FromStr for Integer {
    type Err = ParseIntegerError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(value, 10)
    }
}

macro_rules! integer_from_signed {
    ($($source:ty),+ $(,)?) => {$ (
        impl From<$source> for Integer {
            fn from(value: $source) -> Self {
                let mut output = Self::new();
                unsafe { fmpz_set_si(output.as_mut_ptr(), value as i64) };
                output
            }
        }
    )+ };
}

macro_rules! integer_from_unsigned {
    ($($source:ty),+ $(,)?) => {$ (
        impl From<$source> for Integer {
            fn from(value: $source) -> Self {
                let mut output = Self::new();
                unsafe { fmpz_set_ui(output.as_mut_ptr(), value as u64) };
                output
            }
        }
    )+ };
}

integer_from_signed!(i8, i16, i32, i64);
integer_from_unsigned!(u8, u16, u32, u64);

impl From<isize> for Integer {
    fn from(value: isize) -> Self {
        Self::from_decimal(&value.to_string())
    }
}

impl From<usize> for Integer {
    fn from(value: usize) -> Self {
        Self::from_decimal(&value.to_string())
    }
}

impl From<i128> for Integer {
    fn from(value: i128) -> Self {
        Self::from_decimal(&value.to_string())
    }
}

impl From<u128> for Integer {
    fn from(value: u128) -> Self {
        Self::from_decimal(&value.to_string())
    }
}

macro_rules! binary_operator {
    ($trait:ident, $method:ident, $function:path) => {
        impl $trait<&Integer> for &Integer {
            type Output = Integer;

            fn $method(self, rhs: &Integer) -> Self::Output {
                let mut output = Integer::new();
                unsafe { $function(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
                output
            }
        }

        impl $trait<Integer> for Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        impl $trait<&Integer> for Integer {
            type Output = Integer;
            fn $method(self, rhs: &Integer) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl $trait<Integer> for &Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

binary_operator!(Add, add, fmpz_add);
binary_operator!(Sub, sub, fmpz_sub);
binary_operator!(Mul, mul, fmpz_mul);

impl Add<i32> for Integer {
    type Output = Integer;
    fn add(self, rhs: i32) -> Self::Output {
        self + Integer::from(rhs)
    }
}

impl Mul<u32> for Integer {
    type Output = Integer;
    fn mul(self, rhs: u32) -> Self::Output {
        self * Integer::from(rhs)
    }
}

impl Div<&Integer> for &Integer {
    type Output = Integer;
    fn div(self, rhs: &Integer) -> Self::Output {
        self.trunc_div(rhs)
    }
}

impl Div<Integer> for Integer {
    type Output = Integer;
    fn div(self, rhs: Integer) -> Self::Output {
        (&self).div(&rhs)
    }
}

impl Div<&Integer> for Integer {
    type Output = Integer;
    fn div(self, rhs: &Integer) -> Self::Output {
        (&self).div(rhs)
    }
}

impl Div<Integer> for &Integer {
    type Output = Integer;
    fn div(self, rhs: Integer) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<u32> for &Integer {
    type Output = Integer;
    fn div(self, rhs: u32) -> Self::Output {
        self.trunc_div(&Integer::from(rhs))
    }
}

impl Rem<&Integer> for &Integer {
    type Output = Integer;
    fn rem(self, rhs: &Integer) -> Self::Output {
        self.trunc_rem(rhs)
    }
}

impl Rem<Integer> for Integer {
    type Output = Integer;
    fn rem(self, rhs: Integer) -> Self::Output {
        (&self).rem(&rhs)
    }
}

impl Rem<&Integer> for Integer {
    type Output = Integer;
    fn rem(self, rhs: &Integer) -> Self::Output {
        (&self).rem(rhs)
    }
}

impl Rem<Integer> for &Integer {
    type Output = Integer;
    fn rem(self, rhs: Integer) -> Self::Output {
        self.rem(&rhs)
    }
}

impl Rem<i32> for &Integer {
    type Output = Integer;
    fn rem(self, rhs: i32) -> Self::Output {
        self.trunc_rem(&Integer::from(rhs))
    }
}

impl Neg for Integer {
    type Output = Integer;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl Neg for &Integer {
    type Output = Integer;
    fn neg(self) -> Self::Output {
        let mut output = Integer::new();
        unsafe { fmpz_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl AddAssign<&Integer> for Integer {
    fn add_assign(&mut self, rhs: &Integer) {
        unsafe { fmpz_add(self.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
    }
}

impl AddAssign<Integer> for Integer {
    fn add_assign(&mut self, rhs: Integer) {
        *self += &rhs;
    }
}

impl AddAssign<u32> for Integer {
    fn add_assign(&mut self, rhs: u32) {
        *self += &Integer::from(rhs);
    }
}

impl SubAssign<&Integer> for Integer {
    fn sub_assign(&mut self, rhs: &Integer) {
        unsafe { fmpz_sub(self.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
    }
}

impl SubAssign<Integer> for Integer {
    fn sub_assign(&mut self, rhs: Integer) {
        *self -= &rhs;
    }
}

impl MulAssign<&Integer> for Integer {
    fn mul_assign(&mut self, rhs: &Integer) {
        unsafe { fmpz_mul(self.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
    }
}

impl MulAssign<Integer> for Integer {
    fn mul_assign(&mut self, rhs: Integer) {
        *self *= &rhs;
    }
}

impl MulAssign<u32> for Integer {
    fn mul_assign(&mut self, rhs: u32) {
        *self *= &Integer::from(rhs);
    }
}

impl RemAssign<&Integer> for Integer {
    fn rem_assign(&mut self, rhs: &Integer) {
        let output = self.trunc_rem(rhs);
        *self = output;
    }
}

impl Shl<u32> for Integer {
    type Output = Integer;
    fn shl(self, rhs: u32) -> Self::Output {
        let mut output = Integer::new();
        unsafe { fmpz_mul_2exp(output.as_mut_ptr(), self.as_ptr(), rhs.into()) };
        output
    }
}

macro_rules! compare_primitive {
    ($($primitive:ty),+ $(,)?) => {$ (
        impl PartialEq<$primitive> for Integer {
            fn eq(&self, other: &$primitive) -> bool {
                self == &Integer::from(*other)
            }
        }

        impl PartialOrd<$primitive> for Integer {
            fn partial_cmp(&self, other: &$primitive) -> Option<Ordering> {
                self.partial_cmp(&Integer::from(*other))
            }
        }
    )+ };
}

compare_primitive!(i32, i64, u32, u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_arithmetic_and_floor_division_are_exact() {
        let value: Integer = "123456789012345678901234567890".parse().unwrap();
        assert_eq!((&value * &value).sqrt_floor(), value);
        assert_eq!(
            Integer::from(-7).div_floor(&Integer::from(3)),
            Integer::from(-3)
        );
        assert_eq!(&Integer::from(-7) % Integer::from(3), Integer::from(-1));
    }

    #[test]
    fn modular_inverse_and_checked_word_conversion_work() {
        assert_eq!(
            Integer::from(3).invert(&Integer::from(7)),
            Some(Integer::from(5))
        );
        assert_eq!(Integer::from(i64::MIN).to_i64(), Some(i64::MIN));
        assert_eq!((Integer::one() << 100).to_i64(), None);
    }

    #[test]
    fn bit_length_is_sign_independent_and_exact_at_boundaries() {
        assert_eq!(Integer::zero().bit_length(), 0);
        assert_eq!(Integer::one().bit_length(), 1);
        assert_eq!(Integer::from(-1).bit_length(), 1);
        assert_eq!((Integer::one() << 127).bit_length(), 128);
        assert_eq!((-(Integer::one() << 127)).bit_length(), 128);
    }
}
