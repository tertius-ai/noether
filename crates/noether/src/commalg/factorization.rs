// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical exact factorization of supported polynomials.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};

use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    CommutativeRing, Field, Integer, IntegerRing, NoetherError, Rational, RationalField,
    SemanticType, TypeDescription,
};

use super::{MonomialOrder, Polynomial, PolynomialError, PolynomialRing};

const MAX_DENSE_FACTOR_DEGREE: usize = 1_000_000;

static POLYNOMIAL_FACTORIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialFactorization@1",
    rust_path: "noether::commalg::PolynomialFactorization",
    python_path: Some("noether.commalg.PolynomialFactorization"),
};

/// Failure to factor a supported exact polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialFactorizationError {
    /// The zero polynomial has no canonical finite factorization.
    ZeroPolynomial,
    /// A sparse exponent cannot be represented by the selected FLINT operation.
    ExponentOutOfRange(Integer),
    /// Densifying a sparse polynomial would exceed the declared safety limit.
    ResourceLimit {
        /// Requested dense degree.
        degree: Integer,
        /// Largest admitted dense degree.
        limit: usize,
    },
    /// The native multivariate factorization backend failed or was unavailable.
    BackendFailure(String),
    /// Polynomial construction or arithmetic failed.
    Polynomial(PolynomialError),
    /// Serialized or backend evidence contradicted the canonical result.
    InvalidCertificate(String),
}

impl fmt::Display for PolynomialFactorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPolynomial => {
                formatter.write_str("factorization of the zero polynomial is undefined")
            }
            Self::ExponentOutOfRange(exponent) => write!(
                formatter,
                "polynomial exponent {exponent} does not fit the factorization backend"
            ),
            Self::ResourceLimit { degree, limit } => write!(
                formatter,
                "dense polynomial factorization degree {degree} exceeds the safety limit of {limit}"
            ),
            Self::BackendFailure(reason) => {
                write!(
                    formatter,
                    "multivariate factorization backend failed: {reason}"
                )
            }
            Self::Polynomial(error) => error.fmt(formatter),
            Self::InvalidCertificate(reason) => {
                write!(
                    formatter,
                    "invalid polynomial factorization certificate: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for PolynomialFactorizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial(error) => Some(error),
            Self::ZeroPolynomial
            | Self::ExponentOutOfRange(_)
            | Self::ResourceLimit { .. }
            | Self::BackendFailure(_)
            | Self::InvalidCertificate(_) => None,
        }
    }
}

impl From<PolynomialError> for PolynomialFactorizationError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

/// Canonical complete factorization of one supported nonzero polynomial.
///
/// The represented value is `coefficient * product(factor^multiplicity)`. Over
/// `Z`, `coefficient` is the signed content and every factor is primitive with
/// positive leading coefficient. Over a field, every factor is monic and the
/// coefficient is the source polynomial's leading scalar after normalization.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialFactorization<R: CommutativeRing> {
    source: Polynomial<R>,
    coefficient: R::Element,
    factors: Vec<(Polynomial<R>, u64)>,
}

impl<R: CommutativeRing> PolynomialFactorization<R> {
    /// Polynomial whose canonical factorization is recorded.
    pub fn source(&self) -> &Polynomial<R> {
        &self.source
    }

    /// Signed content over `Z`, or normalized leading scalar over a field.
    pub fn coefficient(&self) -> &R::Element {
        &self.coefficient
    }

    /// Deterministically ordered irreducible factors and positive multiplicities.
    pub fn factors(&self) -> &[(Polynomial<R>, u64)] {
        &self.factors
    }

    /// Reconstruct the exact represented polynomial.
    pub fn value(&self) -> Result<Polynomial<R>, PolynomialFactorizationError> {
        let mut value = self.source.ring().constant(self.coefficient.clone())?;
        for (factor, multiplicity) in &self.factors {
            let power = factor.pow(*multiplicity)?;
            value = (&value * &power).map_err(PolynomialError::from)?;
        }
        Ok(value)
    }

    /// Verify exact reconstruction of the source polynomial.
    pub fn verify(&self) -> Result<(), PolynomialFactorizationError> {
        if self.factors.iter().any(|(_, exponent)| *exponent == 0) {
            return Err(PolynomialFactorizationError::InvalidCertificate(
                "a factor has multiplicity zero".to_owned(),
            ));
        }
        if self.value()? != self.source {
            return Err(PolynomialFactorizationError::InvalidCertificate(
                "the factors do not reconstruct the source".to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize the source and its canonical exact factors as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> fmt::Debug for PolynomialFactorization<R>
where
    R: CommutativeRing,
    R::Element: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialFactorization")
            .field("coefficient", &self.coefficient)
            .field("factors", &self.factors)
            .finish()
    }
}

impl<R> fmt::Display for PolynomialFactorization<R>
where
    R: CommutativeRing,
    R::Element: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.coefficient)?;
        for (factor, exponent) in &self.factors {
            if *exponent == 1 {
                write!(formatter, " * ({factor})")?;
            } else {
                write!(formatter, " * ({factor})^{exponent}")?;
            }
        }
        Ok(())
    }
}

impl<R> SemanticType for PolynomialFactorization<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/PolynomialFactorization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_FACTORIZATION_DESCRIPTION
    }
}

impl<R> Serialize for PolynomialFactorization<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialFactorization", 4)?;
        state.serialize_field("type", "noether/commalg/PolynomialFactorization@1")?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("coefficient", &self.coefficient)?;
        state.serialize_field("factors", &self.factors)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PolynomialFactorizationWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    source: Polynomial<R>,
    coefficient: R::Element,
    factors: Vec<(Polynomial<R>, u64)>,
}

impl Polynomial<IntegerRing> {
    /// Return signed content and canonical primitive irreducible factors over `Z`.
    pub fn factor(
        &self,
    ) -> Result<PolynomialFactorization<IntegerRing>, PolynomialFactorizationError> {
        factor_integer(self)
    }
}

impl Polynomial<RationalField> {
    /// Return the canonical monic irreducible factorization over `Q`.
    pub fn factor(
        &self,
    ) -> Result<PolynomialFactorization<RationalField>, PolynomialFactorizationError> {
        factor_rational(self)
    }
}

impl Polynomial<FiniteField> {
    /// Return the canonical monic irreducible factorization over this finite field.
    pub fn factor(
        &self,
    ) -> Result<PolynomialFactorization<FiniteField>, PolynomialFactorizationError> {
        factor_finite_field(self)
    }
}

impl PolynomialFactorization<IntegerRing> {
    /// Deserialize and recompute the canonical integer-polynomial factorization.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        checked_from_json(value, factor_integer)
    }
}

impl PolynomialFactorization<RationalField> {
    /// Deserialize and recompute the canonical rational-polynomial factorization.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        checked_from_json(value, factor_rational)
    }
}

impl PolynomialFactorization<FiniteField> {
    /// Deserialize and recompute the canonical finite-field factorization.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        checked_from_json(value, factor_finite_field)
    }
}

fn checked_from_json<R>(
    value: &str,
    factor: impl FnOnce(
        &Polynomial<R>,
    ) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>,
) -> Result<PolynomialFactorization<R>, NoetherError>
where
    R: CommutativeRing + for<'de> Deserialize<'de>,
    R::Element: for<'de> Deserialize<'de>,
{
    let wire: PolynomialFactorizationWire<R> = serde_json::from_str(value)
        .map_err(|error| NoetherError::Serialization(error.to_string()))?;
    if wire.type_id != "noether/commalg/PolynomialFactorization@1" {
        return Err(NoetherError::Serialization(format!(
            "expected type noether/commalg/PolynomialFactorization@1, received {}",
            wire.type_id
        )));
    }
    let canonical =
        factor(&wire.source).map_err(|error| NoetherError::Serialization(error.to_string()))?;
    if canonical.coefficient != wire.coefficient || canonical.factors != wire.factors {
        return Err(NoetherError::Serialization(
            "serialized factors are not the canonical factorization of the source".to_owned(),
        ));
    }
    Ok(canonical)
}

fn factor_integer(
    polynomial: &Polynomial<IntegerRing>,
) -> Result<PolynomialFactorization<IntegerRing>, PolynomialFactorizationError> {
    if polynomial.ring().variables().len() > 1 {
        return factor_integer_multivariate(polynomial);
    }
    let dense = dense_coefficients(polynomial)?;
    let raw = super::polynomial_flint::factor_integer_dense(&dense);
    let factors = canonical_factors(polynomial.ring(), raw.factors, |coefficient: &Integer| {
        coefficient.is_zero()
    })?;
    let result = PolynomialFactorization {
        source: polynomial.clone(),
        coefficient: raw.coefficient,
        factors,
    };
    result.verify()?;
    Ok(result)
}

fn factor_rational(
    polynomial: &Polynomial<RationalField>,
) -> Result<PolynomialFactorization<RationalField>, PolynomialFactorizationError> {
    if polynomial.ring().variables().len() > 1 {
        return factor_field_multivariate(polynomial, |native, _| {
            native.factor_rational_multivariate()
        });
    }
    let dense = dense_coefficients(polynomial)?;
    let denominator = dense.iter().fold(Integer::one(), |value, coefficient| {
        value.lcm(&coefficient.denominator())
    });
    let integral = dense
        .iter()
        .map(|coefficient| {
            &coefficient.numerator() * &denominator.div_floor(&coefficient.denominator())
        })
        .collect::<Vec<_>>();
    let raw = super::polynomial_flint::factor_integer_dense(&integral);
    let mut coefficient = Rational::new(raw.coefficient, denominator)
        .map_err(|_| PolynomialFactorizationError::InvalidCertificate("zero denominator".into()))?;
    let mut factors = Vec::with_capacity(raw.factors.len());
    for (factor, multiplicity) in raw.factors {
        let leading = factor.last().cloned().ok_or_else(|| {
            PolynomialFactorizationError::InvalidCertificate(
                "FLINT returned an empty factor".to_owned(),
            )
        })?;
        let leading = Rational::from(leading);
        coefficient = &coefficient * &rational_pow(&leading, multiplicity);
        let monic = factor
            .into_iter()
            .map(|value| &Rational::from(value) / &leading)
            .collect::<Vec<_>>();
        factors.push((
            polynomial_from_dense(polynomial.ring(), monic)?,
            multiplicity,
        ));
    }
    sort_factors(&mut factors)?;
    let result = PolynomialFactorization {
        source: polynomial.clone(),
        coefficient,
        factors,
    };
    result.verify()?;
    Ok(result)
}

fn factor_finite_field(
    polynomial: &Polynomial<FiniteField>,
) -> Result<PolynomialFactorization<FiniteField>, PolynomialFactorizationError> {
    if polynomial.ring().variables().len() > 1 {
        return factor_field_multivariate(polynomial, |native, field| {
            native.factor_finite_field_multivariate(field)
        });
    }
    let dense = dense_coefficients(polynomial)?;
    let raw = super::polynomial_flint::factor_finite_field_dense(
        polynomial.ring().coefficient_ring(),
        &dense,
    );
    let factors = canonical_factors(polynomial.ring(), raw.factors, FiniteFieldElement::is_zero)?;
    let result = PolynomialFactorization {
        source: polynomial.clone(),
        coefficient: raw.coefficient,
        factors,
    };
    result.verify()?;
    Ok(result)
}

fn factor_integer_multivariate(
    polynomial: &Polynomial<IntegerRing>,
) -> Result<PolynomialFactorization<IntegerRing>, PolynomialFactorizationError> {
    let working = multivariate_working_polynomial(polynomial)?;
    let raw = working
        .native()
        .ok_or_else(|| {
            PolynomialFactorizationError::BackendFailure(
                "the working polynomial has no native FLINT storage".to_owned(),
            )
        })?
        .factor_integer_multivariate()
        .map_err(|reason| PolynomialFactorizationError::BackendFailure(reason.to_owned()))?;
    let mut coefficient = raw.coefficient;
    let mut factors = restore_native_factors(polynomial, &working, raw.factors)?;
    for (factor, multiplicity) in &mut factors {
        let leading = factor.leading_coefficient().ok_or_else(|| {
            PolynomialFactorizationError::InvalidCertificate(
                "FLINT returned an empty multivariate factor".to_owned(),
            )
        })?;
        if leading.is_negative() {
            *factor = (-&*factor).map_err(|error| {
                PolynomialFactorizationError::InvalidCertificate(error.to_string())
            })?;
            if *multiplicity & 1 == 1 {
                coefficient = -coefficient;
            }
        }
    }
    sort_factors(&mut factors)?;
    let result = PolynomialFactorization {
        source: polynomial.clone(),
        coefficient,
        factors,
    };
    result.verify()?;
    Ok(result)
}

fn factor_field_multivariate<R>(
    polynomial: &Polynomial<R>,
    factor: impl FnOnce(
        &super::polynomial_flint::FlintPolynomial,
        &R,
    ) -> Result<
        super::polynomial_flint::NativeFactorization<R::Element>,
        &'static str,
    >,
) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    let working = multivariate_working_polynomial(polynomial)?;
    let native = working.native().ok_or_else(|| {
        PolynomialFactorizationError::BackendFailure(
            "the working polynomial has no native FLINT storage".to_owned(),
        )
    })?;
    let raw = factor(native, working.ring().coefficient_ring())
        .map_err(|reason| PolynomialFactorizationError::BackendFailure(reason.to_owned()))?;
    let mut coefficient = raw.coefficient;
    let mut factors = restore_native_factors(polynomial, &working, raw.factors)?;
    for (value, multiplicity) in &mut factors {
        let leading = value.leading_coefficient().ok_or_else(|| {
            PolynomialFactorizationError::InvalidCertificate(
                "FLINT returned an empty multivariate factor".to_owned(),
            )
        })?;
        let scalar = scalar_power(
            polynomial.ring().coefficient_ring(),
            &leading,
            *multiplicity,
        )?;
        coefficient = polynomial
            .ring()
            .coefficient_ring()
            .mul(&coefficient, &scalar)
            .map_err(|error| PolynomialFactorizationError::InvalidCertificate(error.to_string()))?;
        *value = value.monic()?;
    }
    sort_factors(&mut factors)?;
    let result = PolynomialFactorization {
        source: polynomial.clone(),
        coefficient,
        factors,
    };
    result.verify()?;
    Ok(result)
}

fn multivariate_working_polynomial<R>(
    polynomial: &Polynomial<R>,
) -> Result<Polynomial<R>, PolynomialFactorizationError>
where
    R: CommutativeRing,
{
    if polynomial.is_zero() {
        return Err(PolynomialFactorizationError::ZeroPolynomial);
    }
    if polynomial.native().is_some() {
        return Ok(polynomial.clone());
    }
    if let Some(exponent) = polynomial
        .terms()
        .flat_map(|(exponents, _)| exponents)
        .find(|exponent| exponent.to_u64_checked().is_none())
    {
        return Err(PolynomialFactorizationError::ExponentOutOfRange(
            exponent.clone(),
        ));
    }
    let working_ring = PolynomialRing::with_grading(
        polynomial.ring().coefficient_ring().clone(),
        polynomial.ring().variables().to_vec(),
        MonomialOrder::GradedReverseLexicographic,
        polynomial.ring().grading().clone(),
    )
    .map_err(|error| PolynomialFactorizationError::BackendFailure(error.to_string()))?;
    let working = working_ring.from_terms(
        polynomial
            .terms()
            .map(|(exponents, coefficient)| (exponents.to_vec(), coefficient.clone())),
    )?;
    if working.native().is_none() {
        return Err(PolynomialFactorizationError::BackendFailure(
            "the coefficient context has no native multivariate factorization storage".to_owned(),
        ));
    }
    Ok(working)
}

fn restore_native_factors<R>(
    source: &Polynomial<R>,
    working: &Polynomial<R>,
    factors: Vec<(super::polynomial_flint::FlintPolynomial, u64)>,
) -> Result<Vec<(Polynomial<R>, u64)>, PolynomialFactorizationError>
where
    R: CommutativeRing,
{
    factors
        .into_iter()
        .map(|(native, multiplicity)| {
            let factor = working.ring().polynomial_from_native(native);
            let restored = if working.ring() == source.ring() {
                factor
            } else {
                source.ring().from_terms(
                    factor
                        .terms()
                        .map(|(exponents, coefficient)| (exponents.to_vec(), coefficient.clone())),
                )?
            };
            Ok((restored, multiplicity))
        })
        .collect()
}

fn scalar_power<R: Field>(
    field: &R,
    value: &R::Element,
    mut exponent: u64,
) -> Result<R::Element, PolynomialFactorizationError> {
    let mut result = field.one();
    let mut base = value.clone();
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = field.mul(&result, &base).map_err(|error| {
                PolynomialFactorizationError::InvalidCertificate(error.to_string())
            })?;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = field.mul(&base, &base).map_err(|error| {
                PolynomialFactorizationError::InvalidCertificate(error.to_string())
            })?;
        }
    }
    Ok(result)
}

fn canonical_factors<R>(
    ring: &PolynomialRing<R>,
    factors: Vec<(Vec<R::Element>, u64)>,
    is_zero: impl Fn(&R::Element) -> bool + Copy,
) -> Result<Vec<(Polynomial<R>, u64)>, PolynomialFactorizationError>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    let mut result = Vec::with_capacity(factors.len());
    for (factor, multiplicity) in factors {
        if factor.is_empty() || factor.iter().all(is_zero) || multiplicity == 0 {
            return Err(PolynomialFactorizationError::InvalidCertificate(
                "the factorization backend returned an empty factor".to_owned(),
            ));
        }
        result.push((polynomial_from_dense(ring, factor)?, multiplicity));
    }
    sort_factors(&mut result)?;
    Ok(result)
}

fn sort_factors<R>(factors: &mut [(Polynomial<R>, u64)]) -> Result<(), PolynomialFactorizationError>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    let mut keyed = factors
        .iter()
        .map(|(factor, _)| factor.to_json())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| PolynomialFactorizationError::InvalidCertificate(error.to_string()))?;
    let mut permutation = (0..factors.len()).collect::<Vec<_>>();
    permutation.sort_by(|left, right| keyed[*left].cmp(&keyed[*right]));
    keyed.clear();
    let original = factors.to_vec();
    for (destination, source) in permutation.into_iter().enumerate() {
        factors[destination] = original[source].clone();
    }
    Ok(())
}

fn dense_coefficients<R>(
    polynomial: &Polynomial<R>,
) -> Result<Vec<R::Element>, PolynomialFactorizationError>
where
    R: CommutativeRing,
{
    debug_assert_eq!(polynomial.ring().variables().len(), 1);
    let Some(degree) = polynomial.degree(Some(0))? else {
        return Err(PolynomialFactorizationError::ZeroPolynomial);
    };
    let degree_word = degree
        .to_u64_checked()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| PolynomialFactorizationError::ExponentOutOfRange(degree.clone()))?;
    if degree_word > MAX_DENSE_FACTOR_DEGREE {
        return Err(PolynomialFactorizationError::ResourceLimit {
            degree,
            limit: MAX_DENSE_FACTOR_DEGREE,
        });
    }
    let mut coefficients = vec![polynomial.ring().coefficient_ring().zero(); degree_word + 1];
    for (exponent, coefficient) in polynomial.terms() {
        let exponent = exponent[0]
            .to_u64_checked()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| PolynomialFactorizationError::ExponentOutOfRange(exponent[0].clone()))?;
        coefficients[exponent] = coefficient.clone();
    }
    Ok(coefficients)
}

fn polynomial_from_dense<R>(
    ring: &PolynomialRing<R>,
    coefficients: Vec<R::Element>,
) -> Result<Polynomial<R>, PolynomialFactorizationError>
where
    R: CommutativeRing,
{
    ring.from_u64_terms(
        coefficients
            .into_iter()
            .enumerate()
            .map(|(exponent, coefficient)| (vec![exponent as u64], coefficient)),
    )
    .map_err(Into::into)
}

fn rational_pow(value: &Rational, mut exponent: u64) -> Rational {
    let mut result = Rational::one();
    let mut base = value.clone();
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = &result * &base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = &base * &base;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialFactorization, PolynomialRing};
    use crate::numthy::FiniteField;
    use crate::{Integer, Rational, Ring, Q, Z};

    #[test]
    fn factors_integer_and_rational_polynomials_canonically() {
        let integers = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = integers.generator(0).unwrap();
        let x_minus_one = (&x - &integers.one()).unwrap();
        let x_plus_two = (&x + &integers.constant(Integer::from(2)).unwrap()).unwrap();
        let product = (&x_minus_one.pow(2).unwrap() * &x_plus_two).unwrap();
        let source = (&integers.constant(Integer::from(-6)).unwrap() * &product).unwrap();
        let factorization = source.factor().unwrap();
        assert_eq!(factorization.coefficient(), &Integer::from(-6));
        assert_eq!(factorization.factors().len(), 2);
        assert_eq!(factorization.value().unwrap(), source);

        let rationals = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = rationals.generator(0).unwrap();
        let x_minus_one = (&x - &rationals.one()).unwrap();
        let source = (&rationals
            .constant(Rational::new(3.into(), 5.into()).unwrap())
            .unwrap()
            * &x_minus_one.pow(3).unwrap())
            .unwrap();
        let factorization = source.factor().unwrap();
        assert_eq!(
            factorization.coefficient(),
            &Rational::new(3.into(), 5.into()).unwrap()
        );
        assert_eq!(factorization.factors()[0].1, 3);
        assert_eq!(factorization.value().unwrap(), source);
    }

    #[test]
    fn factors_prime_big_prime_and_extension_field_polynomials() {
        let fields = [
            FiniteField::prime(5_u64).unwrap(),
            FiniteField::prime("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
        ];
        for field in fields {
            let ring =
                PolynomialRing::new(field.clone(), ["x"], MonomialOrder::Lexicographic).unwrap();
            let x = ring.generator(0).unwrap();
            let linear = (&x + &ring.constant(field.generator()).unwrap()).unwrap();
            let source =
                (&ring.constant(field.element(3)).unwrap() * &linear.pow(4).unwrap()).unwrap();
            let factorization = source.factor().unwrap();
            assert_eq!(factorization.coefficient(), &field.element(3));
            assert_eq!(factorization.factors(), &[(linear, 4)]);
            assert_eq!(factorization.value().unwrap(), source);
        }
    }

    #[test]
    fn multivariate_factors_are_canonical_over_every_native_field_lane() {
        let fields = [
            FiniteField::prime(5_u64).unwrap(),
            FiniteField::prime("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
        ];
        for field in fields {
            let ring = PolynomialRing::new(
                field.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap();
            let variables = ring.generators().unwrap();
            let first = (&variables[0] + &variables[1]).unwrap();
            let second =
                (&variables[0] + &variables[1].scalar_mul(&field.element(2)).unwrap()).unwrap();
            let source = (&ring.constant(field.element(3)).unwrap()
                * &(&first.pow(2).unwrap() * &second).unwrap())
                .unwrap();

            let factorization = source.factor().unwrap();
            factorization.verify().unwrap();
            assert_eq!(factorization.coefficient(), &field.element(3));
            assert_eq!(factorization.factors().len(), 2);
            assert_eq!(factorization.value().unwrap(), source);
            assert_eq!(
                PolynomialFactorization::<FiniteField>::from_json(
                    &factorization.to_json().unwrap()
                )
                .unwrap(),
                factorization
            );
        }
    }

    #[test]
    fn multivariate_integer_and_rational_factorization_restore_nonflint_orders() {
        let order = MonomialOrder::elimination(2, [0]).unwrap();
        let integers = PolynomialRing::new(Z, ["x", "y"], order.clone()).unwrap();
        let variables = integers.generators().unwrap();
        let first = (&variables[0] + &variables[1]).unwrap();
        let second = (&variables[0] - &variables[1]).unwrap();
        let source = (&integers.constant(Integer::from(-6)).unwrap()
            * &(&first.pow(2).unwrap() * &second).unwrap())
            .unwrap();
        let factorization = source.factor().unwrap();
        factorization.verify().unwrap();
        assert_eq!(factorization.coefficient(), &Integer::from(-6));
        assert_eq!(factorization.factors().len(), 2);
        assert!(factorization
            .factors()
            .iter()
            .all(|(factor, _)| factor.ring() == &integers));

        let rationals = PolynomialRing::new(Q, ["x", "y"], order).unwrap();
        let variables = rationals.generators().unwrap();
        let first = (&variables[0] + &variables[1]).unwrap();
        let second = (&variables[0] - &variables[1]).unwrap();
        let scalar = Rational::new(3.into(), 5.into()).unwrap();
        let source = (&rationals.constant(scalar.clone()).unwrap()
            * &(&first.pow(2).unwrap() * &second).unwrap())
            .unwrap();
        let factorization = source.factor().unwrap();
        factorization.verify().unwrap();
        assert_eq!(factorization.coefficient(), &scalar);
        assert_eq!(factorization.factors().len(), 2);
        assert!(factorization
            .factors()
            .iter()
            .all(|(factor, _)| factor.ring() == &rationals));
    }

    #[test]
    fn typed_json_is_recomputed_and_tampering_is_rejected() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let source = (&x.pow(2).unwrap() - &ring.one()).unwrap();
        let factorization = source.factor().unwrap();
        let json = factorization.to_json().unwrap();
        assert_eq!(
            super::PolynomialFactorization::<crate::IntegerRing>::from_json(&json).unwrap(),
            factorization
        );
        let tampered = json.replacen("\"coefficient\":\"1\"", "\"coefficient\":\"2\"", 1);
        assert!(
            super::PolynomialFactorization::<crate::IntegerRing>::from_json(&tampered).is_err()
        );
    }
}
