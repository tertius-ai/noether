// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Validated field lanes used by Gröbner hot loops.

use core::fmt;
use std::sync::Arc;

use m4rste::{Gf2e, Gf2eTables};
use poly_core::CoefficientRing;

/// Narrow field contract for validated kernel coefficients.
pub trait Field: CoefficientRing + Sync {
    /// Multiplicative identity.
    fn one(&self) -> Self::Element;
    /// Invert a known nonzero value.
    fn inverse(&self, value: &Self::Element) -> Self::Element;
}

/// Coefficient-free binary field lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BinaryField;

impl CoefficientRing for BinaryField {
    type Element = u8;

    fn zero(&self) -> Self::Element {
        0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value & 1 == 0
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (left ^ right) & 1
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (left & right) & 1
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        value & 1
    }
}

impl Field for BinaryField {
    fn one(&self) -> Self::Element {
        1
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        assert_eq!(value & 1, 1, "zero has no inverse in F2");
        1
    }
}

/// Failure to construct a packed binary extension-field lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinaryExtensionFieldError {
    minpoly: u64,
}

impl fmt::Display for BinaryExtensionFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "binary extension Groebner lane requires a degree 2..=16 modulus; received {:#x}",
            self.minpoly
        )
    }
}

impl std::error::Error for BinaryExtensionFieldError {}

/// Packed `F_{2^e}` coefficient lane for `2 <= e <= 16`.
///
/// Elements use the low `e` bits of `u16` in the polynomial basis declared by
/// the irreducible modulus. Multiplication and inversion reuse one immutable
/// logarithm table for the whole computation.
#[derive(Clone)]
pub struct BinaryExtensionField16 {
    field: Gf2e,
    tables: Arc<Gf2eTables>,
}

impl BinaryExtensionField16 {
    /// Construct from the irreducible modulus bit pattern, including its
    /// leading degree-`e` bit.
    pub fn new(minpoly: u64) -> Result<Self, BinaryExtensionFieldError> {
        let degree = minpoly.checked_ilog2();
        if !degree.is_some_and(|degree| (2..=16).contains(&degree)) {
            return Err(BinaryExtensionFieldError { minpoly });
        }
        let field = Gf2e::new(minpoly);
        Ok(Self {
            field,
            tables: Arc::new(field.tables()),
        })
    }

    /// Extension degree over `F_2`.
    pub fn degree(&self) -> u32 {
        self.field.degree()
    }

    /// Defining irreducible polynomial as a bit pattern.
    pub fn minpoly(&self) -> u64 {
        self.field.minpoly()
    }

    pub(crate) fn matrix_context(&self) -> Gf2e {
        self.field
    }
}

impl fmt::Debug for BinaryExtensionField16 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BinaryExtensionField16")
            .field("degree", &self.degree())
            .field("minpoly", &format_args!("{:#x}", self.minpoly()))
            .finish()
    }
}

impl PartialEq for BinaryExtensionField16 {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
    }
}

impl Eq for BinaryExtensionField16 {}

impl CoefficientRing for BinaryExtensionField16 {
    type Element = u16;

    fn zero(&self) -> Self::Element {
        0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value == 0
    }

    #[inline]
    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left ^ right
    }

    #[inline]
    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        self.tables.mul(*left, *right)
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        *value
    }
}

impl Field for BinaryExtensionField16 {
    fn one(&self) -> Self::Element {
        1
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        assert_ne!(*value, 0, "zero has no inverse in a binary extension field");
        self.tables.inv(*value)
    }
}

/// Failure to construct a word-prime field lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeFieldError {
    modulus: u32,
}

impl fmt::Display for PrimeFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "word-prime Gröbner lane requires a prime below 2^31; received {}",
            self.modulus
        )
    }
}

impl std::error::Error for PrimeFieldError {}

/// Optimized field `F_p` for prime `p < 2^31`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeField32 {
    modulus: u32,
}

impl PrimeField32 {
    /// Validate one prime modulus in the optimized lane.
    pub fn new(modulus: u32) -> Result<Self, PrimeFieldError> {
        if modulus >= (1_u32 << 31) || !is_prime(modulus) {
            return Err(PrimeFieldError { modulus });
        }
        Ok(Self { modulus })
    }

    /// Prime modulus.
    pub fn modulus(self) -> u32 {
        self.modulus
    }

    /// Reduce a signed integer into `[0,p)`.
    pub fn element(self, value: i64) -> u32 {
        value.rem_euclid(i64::from(self.modulus)) as u32
    }
}

impl CoefficientRing for PrimeField32 {
    type Element = u32;

    fn zero(&self) -> Self::Element {
        0
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        *value == 0
    }

    #[inline]
    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        let sum = u64::from(*left) + u64::from(*right);
        let modulus = u64::from(self.modulus);
        if sum >= modulus {
            (sum - modulus) as u32
        } else {
            sum as u32
        }
    }

    #[inline]
    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        ((u64::from(*left) * u64::from(*right)) % u64::from(self.modulus)) as u32
    }

    #[inline]
    fn neg(&self, value: &Self::Element) -> Self::Element {
        if *value == 0 {
            0
        } else {
            self.modulus - *value
        }
    }
}

impl Field for PrimeField32 {
    fn one(&self) -> Self::Element {
        1
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        assert_ne!(*value, 0, "zero has no inverse in a prime field");
        pow_mod(*value, self.modulus - 2, self.modulus)
    }
}

fn pow_mod(mut base: u32, mut exponent: u32, modulus: u32) -> u32 {
    let mut result = 1_u32;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = ((u64::from(result) * u64::from(base)) % u64::from(modulus)) as u32;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = ((u64::from(base) * u64::from(base)) % u64::from(modulus)) as u32;
        }
    }
    result
}

fn is_prime(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3_u32;
    while u64::from(divisor) * u64::from(divisor) <= u64::from(value) {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_lane_inverts() {
        let field = PrimeField32::new(65_521).unwrap();
        for value in [1, 2, 17, 65_520] {
            assert_eq!(field.mul(&value, &field.inverse(&value)), 1);
        }
    }

    #[test]
    fn rejects_composite_and_oversized_moduli() {
        assert!(PrimeField32::new(21).is_err());
        assert!(PrimeField32::new(1_u32 << 31).is_err());
    }

    #[test]
    fn binary_extension_lane_reuses_the_declared_power_basis() {
        let field = BinaryExtensionField16::new(0b1_0011).unwrap();
        assert_eq!(field.degree(), 4);
        for value in 1..16 {
            assert_eq!(field.mul(&value, &field.inverse(&value)), 1);
        }
        assert!(BinaryExtensionField16::new(0b11).is_err());
    }
}
