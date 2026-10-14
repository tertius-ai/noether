// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified real roots of exact univariate polynomials.

use core::cmp::Ordering;
use core::fmt;
use std::collections::VecDeque;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    BallError, Integer, IntegerRing, NoetherError, Rational, RationalField, RealBall,
    RealBallField, SemanticType, TypeDescription, Q,
};

use super::{Polynomial, PolynomialError, PolynomialFactorizationError, PolynomialRing};

const MAX_ROOT_DEGREE: usize = 4_096;
const MAX_ISOLATION_SPLITS: usize = 1_000_000;
const MAX_REFINEMENT_BITS: u64 = 16_384;

static REAL_ROOT_ISOLATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RealRootIsolation@1",
    rust_path: "noether::commalg::RealRootIsolation",
    python_path: Some("noether.commalg.RealRootIsolation"),
};

/// Failure to isolate or refine exact real roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealRootError {
    /// Real-root isolation is defined here only for a one-variable context.
    NotUnivariate {
        /// Number of variables in the rejected context.
        variables: usize,
    },
    /// The zero polynomial has no finite discrete root set.
    ZeroPolynomial,
    /// Exact polynomial construction or conversion failed.
    Polynomial(PolynomialError),
    /// Canonical defining-factor computation failed.
    Factorization(PolynomialFactorizationError),
    /// A derived polynomial contradicted an exact division or isolation invariant.
    InvalidCertificate(&'static str),
    /// A bounded exact computation exhausted its declared limit.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
    /// Arb could not construct the requested certified enclosure.
    Ball(BallError),
}

impl fmt::Display for RealRootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotUnivariate { variables } => write!(
                formatter,
                "real-root isolation requires one variable; received {variables}"
            ),
            Self::ZeroPolynomial => {
                formatter.write_str("the zero polynomial does not have a discrete real-root set")
            }
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Factorization(error) => error.fmt(formatter),
            Self::InvalidCertificate(reason) => {
                write!(formatter, "invalid real-root certificate: {reason}")
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "real-root isolation exceeded its {resource} limit of {limit}"
            ),
            Self::Ball(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RealRootError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial(error) => Some(error),
            Self::Factorization(error) => Some(error),
            Self::Ball(error) => Some(error),
            Self::NotUnivariate { .. }
            | Self::ZeroPolynomial
            | Self::InvalidCertificate(_)
            | Self::ResourceLimit { .. } => None,
        }
    }
}

impl From<PolynomialError> for RealRootError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<PolynomialFactorizationError> for RealRootError {
    fn from(error: PolynomialFactorizationError) -> Self {
        Self::Factorization(error)
    }
}

impl From<BallError> for RealRootError {
    fn from(error: BallError) -> Self {
        Self::Ball(error)
    }
}

/// One exact real root isolated in an open rational interval.
///
/// Both the original polynomial and a monic square-free defining factor are
/// stored over `Q`, even when isolation was requested from an integer
/// polynomial. The interval endpoints are not roots, the defining factor has
/// exactly one root between them, and `multiplicity` is the root's
/// multiplicity in the original polynomial.
#[derive(Clone, PartialEq, Eq)]
pub struct RealRootIsolation {
    polynomial: Polynomial<RationalField>,
    defining_factor: Polynomial<RationalField>,
    lower: Rational,
    upper: Rational,
    multiplicity: usize,
}

impl RealRootIsolation {
    /// Original rational polynomial whose root is isolated.
    pub fn polynomial(&self) -> &Polynomial<RationalField> {
        &self.polynomial
    }

    /// Monic square-free defining factor carrying the isolated root.
    pub fn defining_factor(&self) -> &Polynomial<RationalField> {
        &self.defining_factor
    }

    /// Strict lower rational endpoint.
    pub fn lower(&self) -> &Rational {
        &self.lower
    }

    /// Strict upper rational endpoint.
    pub fn upper(&self) -> &Rational {
        &self.upper
    }

    /// Multiplicity in the stored polynomial.
    pub fn multiplicity(&self) -> usize {
        self.multiplicity
    }

    /// Refine the exact interval to width at most `2^-bits`.
    pub fn refine(&self, bits: u64) -> Result<Self, RealRootError> {
        if bits > MAX_REFINEMENT_BITS {
            return Err(resource_limit(
                "refinement bit count",
                MAX_REFINEMENT_BITS as usize,
            ));
        }
        self.verify()?;
        let defining_factor = DensePolynomial::from_rational(&self.defining_factor)?;
        let sturm = defining_factor.sturm_sequence()?;
        let denominator = Integer::one()
            << u32::try_from(bits).map_err(|_| {
                resource_limit("refinement bit count", MAX_REFINEMENT_BITS as usize)
            })?;
        let target = Rational::new(Integer::one(), denominator)
            .map_err(|_| RealRootError::InvalidCertificate("invalid refinement width"))?;
        let mut lower = self.lower.clone();
        let mut upper = self.upper.clone();
        let mut steps = 0_usize;
        while &upper - &lower > target {
            steps = steps.saturating_add(1);
            if steps > MAX_ISOLATION_SPLITS {
                return Err(resource_limit(
                    "refinement split count",
                    MAX_ISOLATION_SPLITS,
                ));
            }
            let split = choose_nonroot_split(&defining_factor, &lower, &upper)?;
            match sturm_root_count(&sturm, &lower, &split)? {
                1 => upper = split,
                0 => lower = split,
                _ => {
                    return Err(RealRootError::InvalidCertificate(
                        "a refining interval ceased to isolate one root",
                    ));
                }
            }
        }
        Ok(Self {
            polynomial: self.polynomial.clone(),
            defining_factor: self.defining_factor.clone(),
            lower,
            upper,
            multiplicity: self.multiplicity,
        })
    }

    /// Return an Arb enclosure after exact rational refinement.
    ///
    /// Exact Sturm refinement is performed to eight guard bits beyond the Arb
    /// working precision. Arb then encloses both rational endpoints; it is an
    /// accelerator and representation layer, not the source of the isolation
    /// proof.
    pub fn ball(&self, precision: u64) -> Result<RealBall, RealRootError> {
        let field = RealBallField::new(precision)?;
        let refined = self.refine(precision.saturating_add(8))?;
        field
            .interval(refined.lower, refined.upper)
            .map_err(Into::into)
    }

    /// Compare this exact root with a rational number.
    pub fn compare_rational(&self, value: &Rational) -> Result<Ordering, RealRootError> {
        self.verify()?;
        let defining_factor = DensePolynomial::from_rational(&self.defining_factor)?;
        let sturm = defining_factor.sturm_sequence()?;
        let mut lower = self.lower.clone();
        let mut upper = self.upper.clone();
        for _ in 0..MAX_REFINEMENT_BITS {
            if value <= &lower {
                return Ok(Ordering::Greater);
            }
            if value >= &upper {
                return Ok(Ordering::Less);
            }
            if defining_factor.evaluate(value).is_zero() {
                return Ok(Ordering::Equal);
            }
            let split = choose_nonroot_split(&defining_factor, &lower, &upper)?;
            if sturm_root_count(&sturm, &lower, &split)? == 1 {
                upper = split;
            } else {
                lower = split;
            }
        }
        Err(resource_limit(
            "rational comparison refinement count",
            MAX_REFINEMENT_BITS as usize,
        ))
    }

    /// Compare two exact isolated roots, recognizing equal algebraic values.
    pub fn compare(&self, other: &Self) -> Result<Ordering, RealRootError> {
        self.verify()?;
        other.verify()?;
        let mut left = self.clone();
        let mut right = other.clone();
        let same_factor = left.defining_factor == right.defining_factor;
        let sturm = same_factor
            .then(|| DensePolynomial::from_rational(&left.defining_factor))
            .transpose()?
            .map(|factor| factor.sturm_sequence())
            .transpose()?;
        for bits in 0..=MAX_REFINEMENT_BITS {
            if left.upper <= right.lower {
                return Ok(Ordering::Less);
            }
            if right.upper <= left.lower {
                return Ok(Ordering::Greater);
            }
            if let Some(sturm) = &sturm {
                let lower = std::cmp::max(left.lower.clone(), right.lower.clone());
                let upper = std::cmp::min(left.upper.clone(), right.upper.clone());
                if lower < upper && sturm_root_count(sturm, &lower, &upper)? == 1 {
                    return Ok(Ordering::Equal);
                }
            }
            if bits == MAX_REFINEMENT_BITS {
                break;
            }
            let next_bits = bits.saturating_add(1);
            left = left.refine(next_bits)?;
            right = right.refine(next_bits)?;
        }
        Err(resource_limit(
            "algebraic comparison refinement count",
            MAX_REFINEMENT_BITS as usize,
        ))
    }

    /// Sign of this exact root: `-1`, `0`, or `1`.
    pub fn sign(&self) -> Result<i8, RealRootError> {
        Ok(match self.compare_rational(&Rational::zero())? {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        })
    }

    /// Greatest integer not exceeding this exact root.
    pub fn floor(&self) -> Result<Integer, RealRootError> {
        let refined = self.refine(2)?;
        let lower_floor = rational_floor(&refined.lower);
        let upper_floor = rational_floor(&refined.upper);
        if lower_floor == upper_floor {
            return Ok(lower_floor);
        }
        let separator = &lower_floor + &Integer::one();
        let separator_rational = Rational::from(separator.clone());
        Ok(match self.compare_rational(&separator_rational)? {
            Ordering::Less => &separator - &Integer::one(),
            Ordering::Equal | Ordering::Greater => separator,
        })
    }

    /// Least integer not smaller than this exact root.
    pub fn ceiling(&self) -> Result<Integer, RealRootError> {
        let floor = self.floor()?;
        if self.compare_rational(&Rational::from(floor.clone()))? == Ordering::Equal {
            Ok(floor)
        } else {
            Ok(&floor + &Integer::one())
        }
    }

    /// Verify the Sturm root count and exact multiplicity certificate.
    pub fn verify(&self) -> Result<(), RealRootError> {
        if self.lower >= self.upper {
            return Err(RealRootError::InvalidCertificate(
                "the lower endpoint is not strictly below the upper endpoint",
            ));
        }
        if self.polynomial.ring() != self.defining_factor.ring() {
            return Err(RealRootError::InvalidCertificate(
                "the source and defining factor use different polynomial contexts",
            ));
        }
        let dense = DensePolynomial::from_rational(&self.polynomial)?;
        let defining_factor = DensePolynomial::from_rational(&self.defining_factor)?;
        if dense.is_zero() {
            return Err(RealRootError::ZeroPolynomial);
        }
        if dense.degree() == Some(0) || defining_factor.degree().is_none_or(|degree| degree == 0) {
            return Err(RealRootError::InvalidCertificate(
                "a constant polynomial cannot certify a root",
            ));
        }
        if defining_factor.monic()? != defining_factor {
            return Err(RealRootError::InvalidCertificate(
                "the defining factor is not monic",
            ));
        }
        if defining_factor.gcd(&defining_factor.derivative())?.degree() != Some(0) {
            return Err(RealRootError::InvalidCertificate(
                "the defining factor is not square-free",
            ));
        }
        if !dense.div_rem(&defining_factor)?.1.is_zero() {
            return Err(RealRootError::InvalidCertificate(
                "the defining factor does not divide the source polynomial",
            ));
        }
        let count = sturm_root_count(&defining_factor.sturm_sequence()?, &self.lower, &self.upper)?;
        if count != 1 {
            return Err(RealRootError::InvalidCertificate(
                "the interval does not contain exactly one distinct root",
            ));
        }
        if root_multiplicity(&dense, &self.lower, &self.upper)? != self.multiplicity {
            return Err(RealRootError::InvalidCertificate(
                "the stored root multiplicity is incorrect",
            ));
        }
        Ok(())
    }

    /// Serialize this exact certificate as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly verify a typed certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for RealRootIsolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RealRootIsolation")
            .field("defining_factor", &self.defining_factor)
            .field("lower", &self.lower)
            .field("upper", &self.upper)
            .field("multiplicity", &self.multiplicity)
            .finish()
    }
}

impl fmt::Display for RealRootIsolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RealRootIsolation(({}, {}), multiplicity={})",
            self.lower, self.upper, self.multiplicity
        )
    }
}

impl SemanticType for RealRootIsolation {
    const TYPE_ID: &'static str = "noether/commalg/RealRootIsolation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &REAL_ROOT_ISOLATION_DESCRIPTION
    }
}

impl Polynomial<RationalField> {
    /// Isolate every distinct real root, including exact multiplicities.
    pub fn real_roots(&self) -> Result<Vec<RealRootIsolation>, RealRootError> {
        isolate_real_roots(self)
    }
}

impl Polynomial<IntegerRing> {
    /// Isolate every distinct real root after the canonical embedding into `Q`.
    pub fn real_roots(&self) -> Result<Vec<RealRootIsolation>, RealRootError> {
        let rational_ring = PolynomialRing::with_grading(
            Q,
            self.ring().variables().iter().cloned(),
            self.ring().term_order(),
            self.ring().grading().clone(),
        )
        .map_err(|_| {
            RealRootError::InvalidCertificate("the rational root context could not be built")
        })?;
        let rational = rational_ring.from_terms(self.terms().map(|(exponent, coefficient)| {
            (exponent.to_vec(), Rational::from(coefficient.clone()))
        }))?;
        isolate_real_roots(&rational)
    }
}

fn isolate_real_roots(
    polynomial: &Polynomial<RationalField>,
) -> Result<Vec<RealRootIsolation>, RealRootError> {
    let dense = DensePolynomial::from_rational(polynomial)?;
    if dense.is_zero() {
        return Err(RealRootError::ZeroPolynomial);
    }
    if dense.degree() == Some(0) {
        return Ok(Vec::new());
    }
    let factorization = polynomial.factor()?;
    let square_free = dense.square_free_part()?;
    let sturm = square_free.sturm_sequence()?;
    let bound = square_free.cauchy_bound()?;
    let lower = -&Rational::from(bound.clone());
    let upper = Rational::from(bound);
    let count = sturm_root_count(&sturm, &lower, &upper)?;
    let mut queue = VecDeque::from([(lower, upper, count)]);
    let mut intervals = Vec::with_capacity(count);
    let mut splits = 0_usize;
    while let Some((lower, upper, count)) = queue.pop_front() {
        match count {
            0 => {}
            1 => intervals.push((lower, upper)),
            _ => {
                splits = splits.saturating_add(1);
                if splits > MAX_ISOLATION_SPLITS {
                    return Err(resource_limit(
                        "isolation split count",
                        MAX_ISOLATION_SPLITS,
                    ));
                }
                let split = choose_nonroot_split(&square_free, &lower, &upper)?;
                let left_count = sturm_root_count(&sturm, &lower, &split)?;
                let right_count = sturm_root_count(&sturm, &split, &upper)?;
                if left_count.saturating_add(right_count) != count {
                    return Err(RealRootError::InvalidCertificate(
                        "Sturm subdivision changed the root count",
                    ));
                }
                queue.push_back((lower, split.clone(), left_count));
                queue.push_back((split, upper, right_count));
            }
        }
    }
    intervals.sort_by(|left, right| left.0.cmp(&right.0));
    intervals
        .into_iter()
        .map(|(lower, upper)| {
            let multiplicity = root_multiplicity(&dense, &lower, &upper)?;
            let mut defining_factors = Vec::new();
            for (factor, factor_multiplicity) in factorization.factors() {
                let dense_factor = DensePolynomial::from_rational(factor)?;
                let count = sturm_root_count(&dense_factor.sturm_sequence()?, &lower, &upper)?;
                if count == 1 {
                    defining_factors.push((factor, *factor_multiplicity));
                }
            }
            let [(defining_factor, factor_multiplicity)] = defining_factors.as_slice() else {
                return Err(RealRootError::InvalidCertificate(
                    "the isolated root did not select exactly one defining factor",
                ));
            };
            if usize::try_from(*factor_multiplicity).ok() != Some(multiplicity) {
                return Err(RealRootError::InvalidCertificate(
                    "the defining-factor and Sturm multiplicities disagree",
                ));
            }
            Ok(RealRootIsolation {
                polynomial: polynomial.clone(),
                defining_factor: (*defining_factor).clone(),
                lower,
                upper,
                multiplicity,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DensePolynomial {
    coefficients: Vec<Rational>,
}

impl DensePolynomial {
    fn new(coefficients: Vec<Rational>) -> Self {
        let mut result = Self { coefficients };
        result.trim();
        result
    }

    fn from_rational(polynomial: &Polynomial<RationalField>) -> Result<Self, RealRootError> {
        let variables = polynomial.ring().variables().len();
        if variables != 1 {
            return Err(RealRootError::NotUnivariate { variables });
        }
        let mut terms = Vec::new();
        let mut degree = 0_usize;
        for (exponent, coefficient) in polynomial.terms() {
            let exponent = exponent[0]
                .to_string()
                .parse::<usize>()
                .map_err(|_| resource_limit("polynomial degree", MAX_ROOT_DEGREE))?;
            if exponent > MAX_ROOT_DEGREE {
                return Err(resource_limit("polynomial degree", MAX_ROOT_DEGREE));
            }
            degree = degree.max(exponent);
            terms.push((exponent, coefficient.clone()));
        }
        if terms.is_empty() {
            return Ok(Self::new(Vec::new()));
        }
        let mut coefficients = vec![Rational::zero(); degree + 1];
        for (exponent, coefficient) in terms {
            coefficients[exponent] = coefficient;
        }
        Ok(Self::new(coefficients))
    }

    fn trim(&mut self) {
        while self.coefficients.last().is_some_and(Rational::is_zero) {
            self.coefficients.pop();
        }
    }

    fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0].is_one()
    }

    fn degree(&self) -> Option<usize> {
        self.coefficients.len().checked_sub(1)
    }

    fn leading(&self) -> Option<&Rational> {
        self.coefficients.last()
    }

    fn derivative(&self) -> Self {
        Self::new(
            self.coefficients
                .iter()
                .enumerate()
                .skip(1)
                .map(|(power, coefficient)| coefficient * &Rational::from(Integer::from(power)))
                .collect(),
        )
    }

    fn evaluate(&self, value: &Rational) -> Rational {
        self.coefficients
            .iter()
            .rev()
            .fold(Rational::zero(), |result, coefficient| {
                &(&result * value) + coefficient
            })
    }

    fn monic(&self) -> Result<Self, RealRootError> {
        let leading = self.leading().ok_or(RealRootError::InvalidCertificate(
            "the zero polynomial cannot be made monic",
        ))?;
        Ok(Self::new(
            self.coefficients
                .iter()
                .map(|coefficient| {
                    coefficient.checked_div(leading).map_err(|_| {
                        RealRootError::InvalidCertificate("a leading coefficient was zero")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), RealRootError> {
        let divisor_degree = divisor.degree().ok_or(RealRootError::InvalidCertificate(
            "polynomial division used a zero divisor",
        ))?;
        let divisor_leading = divisor.leading().ok_or(RealRootError::InvalidCertificate(
            "polynomial division used a zero divisor",
        ))?;
        let mut remainder = self.clone();
        let quotient_length = self
            .degree()
            .map_or(0, |degree| degree.saturating_sub(divisor_degree) + 1);
        let mut quotient = vec![Rational::zero(); quotient_length];
        while let Some(remainder_degree) = remainder.degree() {
            if remainder_degree < divisor_degree {
                break;
            }
            let shift = remainder_degree - divisor_degree;
            let factor = remainder
                .leading()
                .ok_or(RealRootError::InvalidCertificate(
                    "a polynomial remainder lost its leading coefficient",
                ))?
                .checked_div(divisor_leading)
                .map_err(|_| {
                    RealRootError::InvalidCertificate("a divisor leading coefficient was zero")
                })?;
            quotient[shift] = &quotient[shift] + &factor;
            for (index, coefficient) in divisor.coefficients.iter().enumerate() {
                let product = &factor * coefficient;
                remainder.coefficients[index + shift] =
                    &remainder.coefficients[index + shift] - &product;
            }
            remainder.trim();
        }
        Ok((Self::new(quotient), remainder))
    }

    fn exact_div(&self, divisor: &Self) -> Result<Self, RealRootError> {
        let (quotient, remainder) = self.div_rem(divisor)?;
        if !remainder.is_zero() {
            return Err(RealRootError::InvalidCertificate(
                "a derived polynomial quotient was not exact",
            ));
        }
        Ok(quotient)
    }

    fn gcd(&self, other: &Self) -> Result<Self, RealRootError> {
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        if left.is_zero() {
            Ok(Self::new(Vec::new()))
        } else {
            left.monic()
        }
    }

    fn square_free_part(&self) -> Result<Self, RealRootError> {
        if self.degree().is_none_or(|degree| degree == 0) {
            return Ok(self.clone());
        }
        let gcd = self.gcd(&self.derivative())?;
        self.exact_div(&gcd)?.monic()
    }

    fn sturm_sequence(&self) -> Result<Vec<Self>, RealRootError> {
        let polynomial = self.monic()?;
        if polynomial.degree() == Some(0) {
            return Ok(vec![polynomial]);
        }
        let derivative = polynomial.derivative();
        if derivative.is_zero() {
            return Err(RealRootError::InvalidCertificate(
                "a nonconstant characteristic-zero polynomial had zero derivative",
            ));
        }
        let mut sequence = vec![polynomial, derivative];
        loop {
            let length = sequence.len();
            let (_, remainder) = sequence[length - 2].div_rem(&sequence[length - 1])?;
            if remainder.is_zero() {
                break;
            }
            let mut next = Self::new(
                remainder
                    .coefficients
                    .into_iter()
                    .map(|coefficient| -coefficient)
                    .collect(),
            );
            let scale = rational_abs(next.leading().ok_or(RealRootError::InvalidCertificate(
                "a Sturm remainder was empty",
            ))?);
            next = Self::new(
                next.coefficients
                    .iter()
                    .map(|coefficient| {
                        coefficient.checked_div(&scale).map_err(|_| {
                            RealRootError::InvalidCertificate("a Sturm scale was zero")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
            sequence.push(next);
        }
        Ok(sequence)
    }

    fn cauchy_bound(&self) -> Result<Integer, RealRootError> {
        let degree = self.degree().ok_or(RealRootError::InvalidCertificate(
            "a constant polynomial has no root bound",
        ))?;
        let leading = rational_abs(self.leading().ok_or(RealRootError::InvalidCertificate(
            "a polynomial has no leading coefficient",
        ))?);
        let mut maximum = Integer::zero();
        for coefficient in &self.coefficients[..degree] {
            let ratio = rational_abs(coefficient)
                .checked_div(&leading)
                .map_err(|_| {
                    RealRootError::InvalidCertificate("the leading coefficient was zero")
                })?;
            maximum = maximum.max(ceil_nonnegative(&ratio));
        }
        Ok(&maximum + &Integer::one())
    }
}

fn rational_abs(value: &Rational) -> Rational {
    if value.is_negative() {
        -value
    } else {
        value.clone()
    }
}

fn ceil_nonnegative(value: &Rational) -> Integer {
    let numerator = value.numerator();
    let denominator = value.denominator();
    (&(&numerator + &denominator) - &Integer::one()).div_floor(&denominator)
}

fn rational_floor(value: &Rational) -> Integer {
    value.numerator().div_floor(&value.denominator())
}

fn sturm_variations(sequence: &[DensePolynomial], point: &Rational) -> usize {
    let mut previous = 0_i8;
    let mut variations = 0_usize;
    for polynomial in sequence {
        let value = polynomial.evaluate(point);
        let sign = if value.is_positive() {
            1
        } else if value.is_negative() {
            -1
        } else {
            0
        };
        if sign == 0 {
            continue;
        }
        if previous != 0 && previous != sign {
            variations = variations.saturating_add(1);
        }
        previous = sign;
    }
    variations
}

fn sturm_root_count(
    sequence: &[DensePolynomial],
    lower: &Rational,
    upper: &Rational,
) -> Result<usize, RealRootError> {
    if lower >= upper {
        return Err(RealRootError::InvalidCertificate(
            "a Sturm interval is not increasing",
        ));
    }
    sturm_variations(sequence, lower)
        .checked_sub(sturm_variations(sequence, upper))
        .ok_or(RealRootError::InvalidCertificate(
            "Sturm variations increased across an interval",
        ))
}

fn choose_nonroot_split(
    polynomial: &DensePolynomial,
    lower: &Rational,
    upper: &Rational,
) -> Result<Rational, RealRootError> {
    let width = upper - lower;
    let two = Rational::from(2_i64);
    let midpoint = lower
        + &width
            .checked_div(&two)
            .map_err(|_| RealRootError::InvalidCertificate("invalid interval midpoint"))?;
    if !polynomial.evaluate(&midpoint).is_zero() {
        return Ok(midpoint);
    }
    let degree = polynomial.degree().unwrap_or(0);
    let denominator = Rational::from(Integer::from(degree.saturating_add(2)));
    for numerator in 1..=degree.saturating_add(1) {
        let fraction = Rational::from(Integer::from(numerator))
            .checked_div(&denominator)
            .map_err(|_| RealRootError::InvalidCertificate("invalid subdivision fraction"))?;
        let candidate = lower + &(&width * &fraction);
        if !polynomial.evaluate(&candidate).is_zero() {
            return Ok(candidate);
        }
    }
    Err(RealRootError::InvalidCertificate(
        "could not choose a nonroot subdivision point",
    ))
}

fn root_multiplicity(
    polynomial: &DensePolynomial,
    lower: &Rational,
    upper: &Rational,
) -> Result<usize, RealRootError> {
    let mut current = polynomial.monic()?;
    let mut multiplicity = 0_usize;
    loop {
        if current.degree().is_none_or(|degree| degree == 0) {
            break;
        }
        let square_free = current.square_free_part()?;
        if sturm_root_count(&square_free.sturm_sequence()?, lower, upper)? != 1 {
            break;
        }
        multiplicity = multiplicity.saturating_add(1);
        current = current.gcd(&current.derivative())?;
        if current.is_one() {
            break;
        }
    }
    if multiplicity == 0 {
        return Err(RealRootError::InvalidCertificate(
            "an isolated root had zero multiplicity",
        ));
    }
    Ok(multiplicity)
}

fn resource_limit(resource: &'static str, limit: usize) -> RealRootError {
    RealRootError::ResourceLimit { resource, limit }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealRootIsolationWire {
    #[serde(rename = "type")]
    type_id: String,
    polynomial: Polynomial<RationalField>,
    defining_factor: Polynomial<RationalField>,
    lower: Rational,
    upper: Rational,
    multiplicity: usize,
}

impl<'de> Deserialize<'de> for RealRootIsolation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RealRootIsolationWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let root = Self {
            polynomial: wire.polynomial,
            defining_factor: wire.defining_factor,
            lower: wire.lower,
            upper: wire.upper,
            multiplicity: wire.multiplicity,
        };
        root.verify().map_err(serde::de::Error::custom)?;
        Ok(root)
    }
}

impl Serialize for RealRootIsolation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RealRootIsolation", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("polynomial", &self.polynomial)?;
        state.serialize_field("defining_factor", &self.defining_factor)?;
        state.serialize_field("lower", &self.lower)?;
        state.serialize_field("upper", &self.upper)?;
        state.serialize_field("multiplicity", &self.multiplicity)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::Ring;

    #[test]
    fn isolates_simple_and_multiple_roots_exactly() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let one = ring.one();
        let left = ring.sub(&x, &one).unwrap();
        let right = ring
            .add(&x, &ring.constant(Rational::from(2_i64)).unwrap())
            .unwrap();
        let polynomial = ring.mul(&ring.mul(&left, &left).unwrap(), &right).unwrap();
        let roots = polynomial.real_roots().unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].multiplicity(), 1);
        assert_eq!(roots[1].multiplicity(), 2);
        assert!(roots[0].lower() < &Rational::from(-2_i64));
        assert!(roots[0].upper() > &Rational::from(-2_i64));
        assert!(roots[1].lower() < &Rational::from(1_i64));
        assert!(roots[1].upper() > &Rational::from(1_i64));
        for root in roots {
            root.verify().unwrap();
            let refined = root.refine(80).unwrap();
            assert!(
                &refined.upper - &refined.lower
                    <= Rational::new(Integer::one(), Integer::one() << 80).unwrap()
            );
            assert!(!root.ball(128).unwrap().is_exact());
        }
    }

    #[test]
    fn constants_and_zero_are_distinguished() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        assert!(ring.one().real_roots().unwrap().is_empty());
        assert_eq!(ring.zero().real_roots(), Err(RealRootError::ZeroPolynomial));
    }

    #[test]
    fn irrational_roots_and_a_root_at_the_first_split_are_isolated() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let x_squared = ring.mul(&x, &x).unwrap();
        let two = ring.constant(Rational::from(2_i64)).unwrap();
        let polynomial = ring.mul(&x, &ring.sub(&x_squared, &two).unwrap()).unwrap();
        let roots = polynomial.real_roots().unwrap();
        assert_eq!(roots.len(), 3);
        assert!(roots[0].upper() < &Rational::zero());
        assert!(roots[1].lower() < &Rational::zero());
        assert!(roots[1].upper() > &Rational::zero());
        assert!(roots[2].lower() > &Rational::zero());

        let no_real_roots = ring.add(&x_squared, &ring.one()).unwrap();
        assert!(no_real_roots.real_roots().unwrap().is_empty());
    }

    #[test]
    fn multivariate_polynomials_fail_closed() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        assert_eq!(
            ring.generator(0).unwrap().real_roots(),
            Err(RealRootError::NotUnivariate { variables: 2 })
        );
    }

    #[test]
    fn defining_factors_comparisons_and_integer_rounding_are_exact() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let quadratic =
            (&x.pow(2).unwrap() - &ring.constant(Rational::from(2_i64)).unwrap()).unwrap();
        let roots = quadratic.real_roots().unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].defining_factor(), &quadratic.monic().unwrap());
        assert_eq!(roots[0].sign().unwrap(), -1);
        assert_eq!(roots[0].floor().unwrap(), Integer::from(-2));
        assert_eq!(roots[0].ceiling().unwrap(), Integer::from(-1));
        assert_eq!(roots[1].sign().unwrap(), 1);
        assert_eq!(roots[1].floor().unwrap(), Integer::from(1));
        assert_eq!(roots[1].ceiling().unwrap(), Integer::from(2));
        assert_eq!(
            roots[0].compare_rational(&Rational::from(-1_i64)).unwrap(),
            Ordering::Less
        );
        assert_eq!(roots[0].compare(&roots[1]).unwrap(), Ordering::Less);

        let augmented =
            (&quadratic * &(&x + &ring.constant(Rational::from(3_i64)).unwrap()).unwrap()).unwrap();
        let augmented_roots = augmented.real_roots().unwrap();
        assert_eq!(
            roots[0].compare(&augmented_roots[1]).unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            roots[1].compare(&augmented_roots[2]).unwrap(),
            Ordering::Equal
        );

        let integer_root = x.real_roots().unwrap().pop().unwrap();
        assert_eq!(integer_root.sign().unwrap(), 0);
        assert_eq!(integer_root.floor().unwrap(), Integer::zero());
        assert_eq!(integer_root.ceiling().unwrap(), Integer::zero());
        assert_eq!(
            RealRootIsolation::from_json(&integer_root.to_json().unwrap()).unwrap(),
            integer_root
        );
    }
}
