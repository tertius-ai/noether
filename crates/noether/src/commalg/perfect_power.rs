// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact perfect-power tests and extraction for supported univariate values.

use core::fmt;

use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{CommutativeRing, IntegerRing, Rational, RationalField, Ring};

use super::{
    MonomialOrder, Polynomial, PolynomialError, PolynomialFactorization,
    PolynomialFactorizationError, PolynomialRing, RationalFunction, RationalFunctionError,
};

/// Failure to test or extract an exact univariate perfect power.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerfectPowerError {
    /// Zeroth roots are undefined.
    ZeroExponent,
    /// Canonical factorization failed.
    Factorization(PolynomialFactorizationError),
    /// Polynomial reconstruction failed.
    Polynomial(PolynomialError),
    /// Rational-function normalization failed.
    RationalFunction(RationalFunctionError),
}

impl fmt::Display for PerfectPowerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroExponent => formatter.write_str("perfect-power exponent must be positive"),
            Self::Factorization(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::RationalFunction(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PerfectPowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Factorization(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::RationalFunction(error) => Some(error),
            Self::ZeroExponent => None,
        }
    }
}

impl From<PolynomialFactorizationError> for PerfectPowerError {
    fn from(error: PolynomialFactorizationError) -> Self {
        Self::Factorization(error)
    }
}

impl From<PolynomialError> for PerfectPowerError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<RationalFunctionError> for PerfectPowerError {
    fn from(error: RationalFunctionError) -> Self {
        Self::RationalFunction(error)
    }
}

impl Polynomial<IntegerRing> {
    /// Whether this polynomial is an exact `exponent`-th power over `Z`.
    pub fn is_nth_power(&self, exponent: u64) -> Result<bool, PerfectPowerError> {
        Ok(self.nth_root(exponent)?.is_some())
    }

    /// Return an exact `exponent`-th root over `Z`, when one exists.
    pub fn nth_root(&self, exponent: u64) -> Result<Option<Self>, PerfectPowerError> {
        validate_exponent(exponent)?;
        if self.is_zero() || exponent == 1 {
            return Ok(Some(self.clone()));
        }
        let factorization = self.factor()?;
        root_from_factorization(&factorization, exponent, |coefficient| {
            coefficient.nth_root_exact(exponent)
        })
    }
}

impl Polynomial<RationalField> {
    /// Whether this polynomial is an exact `exponent`-th power over `Q`.
    pub fn is_nth_power(&self, exponent: u64) -> Result<bool, PerfectPowerError> {
        Ok(self.nth_root(exponent)?.is_some())
    }

    /// Return an exact `exponent`-th root over `Q`, when one exists.
    pub fn nth_root(&self, exponent: u64) -> Result<Option<Self>, PerfectPowerError> {
        validate_exponent(exponent)?;
        if self.is_zero() || exponent == 1 {
            return Ok(Some(self.clone()));
        }
        let factorization = self.factor()?;
        root_from_factorization(&factorization, exponent, |coefficient| {
            rational_nth_root(coefficient, exponent)
        })
    }
}

impl Polynomial<FiniteField> {
    /// Whether this polynomial is an exact `exponent`-th power in its finite field.
    pub fn is_nth_power(&self, exponent: u64) -> Result<bool, PerfectPowerError> {
        Ok(self.nth_root(exponent)?.is_some())
    }

    /// Return a deterministic exact `exponent`-th root in its finite field.
    ///
    /// Coefficient roots are found by exact finite-field factorization of
    /// `X^exponent-a`, so this also handles inseparable exponents divisible by
    /// the characteristic.
    pub fn nth_root(&self, exponent: u64) -> Result<Option<Self>, PerfectPowerError> {
        validate_exponent(exponent)?;
        if self.is_zero() || exponent == 1 {
            return Ok(Some(self.clone()));
        }
        let factorization = self.factor()?;
        root_from_factorization(&factorization, exponent, |coefficient| {
            finite_field_nth_root(coefficient, exponent)
        })
    }
}

impl RationalFunction<RationalField> {
    /// Whether this rational function is an exact `exponent`-th power over `Q`.
    pub fn is_nth_power(&self, exponent: u64) -> Result<bool, PerfectPowerError> {
        Ok(self.nth_root(exponent)?.is_some())
    }

    /// Extract an exact rational-function root, checking numerator and denominator valuations.
    pub fn nth_root(&self, exponent: u64) -> Result<Option<Self>, PerfectPowerError> {
        validate_exponent(exponent)?;
        let Some(numerator) = self.numerator().nth_root(exponent)? else {
            return Ok(None);
        };
        let Some(denominator) = self.denominator().nth_root(exponent)? else {
            return Ok(None);
        };
        Ok(Some(self.field().fraction(numerator, denominator)?))
    }
}

impl RationalFunction<FiniteField> {
    /// Whether this rational function is an exact `exponent`-th power.
    pub fn is_nth_power(&self, exponent: u64) -> Result<bool, PerfectPowerError> {
        Ok(self.nth_root(exponent)?.is_some())
    }

    /// Extract an exact rational-function root, checking numerator and denominator valuations.
    pub fn nth_root(&self, exponent: u64) -> Result<Option<Self>, PerfectPowerError> {
        validate_exponent(exponent)?;
        let Some(numerator) = self.numerator().nth_root(exponent)? else {
            return Ok(None);
        };
        let Some(denominator) = self.denominator().nth_root(exponent)? else {
            return Ok(None);
        };
        Ok(Some(self.field().fraction(numerator, denominator)?))
    }
}

fn validate_exponent(exponent: u64) -> Result<(), PerfectPowerError> {
    if exponent == 0 {
        Err(PerfectPowerError::ZeroExponent)
    } else {
        Ok(())
    }
}

fn root_from_factorization<R>(
    factorization: &PolynomialFactorization<R>,
    exponent: u64,
    coefficient_root: impl FnOnce(&R::Element) -> Option<R::Element>,
) -> Result<Option<Polynomial<R>>, PerfectPowerError>
where
    R: CommutativeRing,
{
    if factorization
        .factors()
        .iter()
        .any(|(_, multiplicity)| multiplicity % exponent != 0)
    {
        return Ok(None);
    }
    let Some(coefficient) = coefficient_root(factorization.coefficient()) else {
        return Ok(None);
    };
    let mut root = factorization.source().ring().constant(coefficient)?;
    for (factor, multiplicity) in factorization.factors() {
        let power = factor.pow(multiplicity / exponent)?;
        root = (&root * &power).map_err(PolynomialError::from)?;
    }
    if root.pow(exponent)? == *factorization.source() {
        Ok(Some(root))
    } else {
        Err(PerfectPowerError::Factorization(
            PolynomialFactorizationError::InvalidCertificate(
                "the extracted root does not reproduce the source".to_owned(),
            ),
        ))
    }
}

fn rational_nth_root(value: &Rational, exponent: u64) -> Option<Rational> {
    let numerator = value.numerator().nth_root_exact(exponent)?;
    let denominator = value.denominator().nth_root_exact(exponent)?;
    Rational::new(numerator, denominator).ok()
}

fn finite_field_nth_root(value: &FiniteFieldElement, exponent: u64) -> Option<FiniteFieldElement> {
    if value.is_zero() {
        return Some(value.clone());
    }
    let field = value.field().clone();
    let ring = PolynomialRing::new(
        field.clone(),
        ["coefficient_root"],
        MonomialOrder::Lexicographic,
    )
    .ok()?;
    let variable = ring.generator(0).ok()?;
    let binomial = (&variable.pow(exponent).ok()? - &ring.constant(value.clone()).ok()?).ok()?;
    let factorization = binomial.factor().ok()?;
    factorization
        .factors()
        .iter()
        .find(|(factor, _)| factor.degree(Some(0)).ok().flatten() == Some(1.into()))
        .and_then(|(factor, _)| field.neg(&factor.constant_coefficient()).ok())
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialRing, RationalFunctionField};
    use crate::numthy::FiniteField;
    use crate::{Integer, Rational, Ring, Q, Z};

    #[test]
    fn extracts_integer_and_rational_polynomial_powers() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let cubic = (&x.pow(3).unwrap() - &ring.constant(Integer::from(2)).unwrap()).unwrap();
        let square = cubic.pow(2).unwrap();
        assert_eq!(square.nth_root(2).unwrap(), Some(cubic));
        assert!(x.nth_root(2).unwrap().is_none());
        assert!(x.nth_root(0).is_err());

        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let root = (&ring
            .constant(Rational::new(2.into(), 3.into()).unwrap())
            .unwrap()
            * &(&x + &ring.one()).unwrap())
            .unwrap();
        assert_eq!(root.pow(3).unwrap().nth_root(3).unwrap(), Some(root));
    }

    #[test]
    fn finite_field_roots_cover_all_backend_lanes_and_inseparability() {
        let fields = [
            FiniteField::prime(2_u64).unwrap(),
            FiniteField::prime("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
        ];
        for field in fields {
            let ring =
                PolynomialRing::new(field.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
            let x = ring.generator(0).unwrap();
            let root = (&x + &ring.constant(field.generator()).unwrap()).unwrap();
            let exponent = if field.characteristic() == &Integer::from(2) {
                2
            } else {
                3
            };
            let power = root.pow(exponent).unwrap();
            let extracted = power.nth_root(exponent).unwrap().unwrap();
            assert_eq!(extracted.pow(exponent).unwrap(), power);
        }
    }

    #[test]
    fn rational_function_roots_check_both_valuations() {
        let field = RationalFunctionField::new(Q, "t").unwrap();
        let t = field.generator();
        let numerator = (&t + &field.constant(1.into()).unwrap()).unwrap();
        let denominator = (&t - &field.constant(1.into()).unwrap()).unwrap();
        let root = (&numerator / &denominator).unwrap();
        let square = root.pow(2).unwrap();
        assert_eq!(square.nth_root(2).unwrap(), Some(root));
        assert!(numerator.nth_root(2).unwrap().is_none());
    }
}
