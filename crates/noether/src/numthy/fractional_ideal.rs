// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact nonzero fractional ideals of number-field orders.

use core::fmt;
use core::ops::{Add, Mul};
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::integer::{factor as factor_integer, remove_factor, FactorError};
use super::number_field_order::{integer_lcm, ExactLattice};
use super::{NumberFieldElement, NumberFieldOrder, NumberFieldOrderError, PrimeDecompositionError};
use crate::commalg::{Ideal, PrimeIdeal};
use crate::{
    ContextMismatchError, Integer, NoetherError, Rational, Ring, SemanticType, TypeDescription,
};

static FRACTIONAL_IDEAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/FractionalIdeal@1",
    rust_path: "noether::numthy::FractionalIdeal",
    python_path: Some("noether.numthy.FractionalIdeal"),
};

/// Failure to construct or operate on a fractional ideal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FractionalIdealError {
    /// Fractional ideals in this API are nonzero and need a nonzero generator.
    ZeroIdeal,
    /// An exact lattice failed a full-rank invariant.
    InvalidLattice,
    /// An integral ideal was required, but the lattice was not contained in its order.
    NotIntegral,
    /// Values from different number-field orders were combined.
    ContextMismatch(ContextMismatchError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for FractionalIdealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroIdeal => {
                f.write_str("a fractional ideal needs at least one nonzero number-field generator")
            }
            Self::InvalidLattice => {
                f.write_str("fractional-ideal generators did not span a full-rank exact lattice")
            }
            Self::NotIntegral => {
                f.write_str("fractional ideal is not contained in its coefficient order")
            }
            Self::ContextMismatch(error) => error.fmt(f),
            Self::InvalidEncoding(message) => {
                write!(f, "invalid fractional-ideal encoding: {message}")
            }
        }
    }
}

impl std::error::Error for FractionalIdealError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for FractionalIdealError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Failure to value or factor a number-field fractional ideal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdealFactorizationError {
    /// Unique prime-ideal factorization requires the maximal order.
    NonMaximalOrder,
    /// Values from different number-field orders were combined.
    ContextMismatch(ContextMismatchError),
    /// Exact order arithmetic failed.
    Order(NumberFieldOrderError),
    /// Exact ideal arithmetic failed.
    Ideal(FractionalIdealError),
    /// Rational-prime decomposition failed.
    PrimeDecomposition(PrimeDecompositionError),
    /// Complete integer factorization failed.
    IntegerFactorization(FactorError),
    /// A signed public exponent did not fit in `i64`.
    ExponentOverflow,
    /// An exact reconstruction invariant failed.
    Invariant(&'static str),
}

impl fmt::Display for IdealFactorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonMaximalOrder => f.write_str(
                "ideal valuation and factorization require a maximal number-field order",
            ),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Order(error) => error.fmt(f),
            Self::Ideal(error) => error.fmt(f),
            Self::PrimeDecomposition(error) => error.fmt(f),
            Self::IntegerFactorization(error) => error.fmt(f),
            Self::ExponentOverflow => {
                f.write_str("ideal-factorization exponent does not fit in signed 64 bits")
            }
            Self::Invariant(claim) => write!(f, "ideal-factorization invariant failed: {claim}"),
        }
    }
}

impl std::error::Error for IdealFactorizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Order(error) => Some(error),
            Self::Ideal(error) => Some(error),
            Self::PrimeDecomposition(error) => Some(error),
            Self::IntegerFactorization(error) => Some(error),
            Self::NonMaximalOrder | Self::ExponentOverflow | Self::Invariant(_) => None,
        }
    }
}

impl From<NumberFieldOrderError> for IdealFactorizationError {
    fn from(value: NumberFieldOrderError) -> Self {
        Self::Order(value)
    }
}

impl From<FractionalIdealError> for IdealFactorizationError {
    fn from(value: FractionalIdealError) -> Self {
        Self::Ideal(value)
    }
}

impl From<PrimeDecompositionError> for IdealFactorizationError {
    fn from(value: PrimeDecompositionError) -> Self {
        Self::PrimeDecomposition(value)
    }
}

impl From<FactorError> for IdealFactorizationError {
    fn from(value: FactorError) -> Self {
        Self::IntegerFactorization(value)
    }
}

/// A nonzero fractional ideal of an exact number-field order.
///
/// The ideal is stored as a canonical row-Hermite rational lattice in the
/// ambient field's distinguished power basis. Construction from generators
/// forms their `NumberFieldOrder`-span; equality therefore compares ideals,
/// not the supplied generator list.
#[derive(Clone)]
pub struct FractionalIdeal {
    order: NumberFieldOrder,
    lattice: ExactLattice,
}

impl FractionalIdeal {
    /// Construct the fractional ideal generated over `order` by exact field elements.
    pub fn new(
        order: NumberFieldOrder,
        generators: &[NumberFieldElement],
    ) -> Result<Self, FractionalIdealError> {
        if generators.is_empty() || generators.iter().all(NumberFieldElement::is_zero) {
            return Err(FractionalIdealError::ZeroIdeal);
        }
        if generators
            .iter()
            .any(|element| !order.field().contains(element))
        {
            return Err(FractionalIdealError::ContextMismatch(
                ContextMismatchError {
                    expected: format!("NumberFieldOrder(discriminant={})", order.discriminant()),
                    actual: "number-field element from another field".to_owned(),
                },
            ));
        }

        let order_basis = order.basis();
        let mut spanning = Vec::with_capacity(generators.len() * order_basis.len());
        for generator in generators {
            if generator.is_zero() {
                continue;
            }
            for basis_element in &order_basis {
                spanning.push(generator.mul(basis_element)?);
            }
        }
        Self::from_spanning_elements(order, &spanning)
    }

    fn from_spanning_elements(
        order: NumberFieldOrder,
        elements: &[NumberFieldElement],
    ) -> Result<Self, FractionalIdealError> {
        let lattice = ExactLattice::from_spanning_elements(order.field(), elements)
            .ok_or(FractionalIdealError::InvalidLattice)?;
        Ok(Self { order, lattice })
    }

    fn mismatch(&self, other: &Self) -> ContextMismatchError {
        ContextMismatchError {
            expected: format!(
                "NumberFieldOrder(discriminant={})",
                self.order.discriminant()
            ),
            actual: format!(
                "NumberFieldOrder(discriminant={})",
                other.order.discriminant()
            ),
        }
    }

    /// The coefficient order of this ideal.
    pub fn order(&self) -> &NumberFieldOrder {
        &self.order
    }

    /// Canonical `Z`-basis of this fractional ideal.
    pub fn basis(&self) -> Vec<NumberFieldElement> {
        self.lattice.basis(self.order.field())
    }

    /// Whether an exact field element lies in this fractional ideal.
    pub fn contains(&self, element: &NumberFieldElement) -> bool {
        self.order.field().contains(element) && self.lattice.contains(element)
    }

    /// Whether this fractional ideal is contained in its coefficient order.
    pub fn is_integral(&self) -> bool {
        self.basis()
            .iter()
            .all(|element| self.order.contains(element))
    }

    /// View this value as a generic integral ideal when it lies in its order.
    pub fn as_ideal(&self) -> Result<Ideal<NumberFieldOrder>, FractionalIdealError> {
        if !self.is_integral() {
            return Err(FractionalIdealError::NotIntegral);
        }
        Ideal::from_canonical_generators(self.order.clone(), self.basis())
            .map_err(|_| FractionalIdealError::InvalidLattice)
    }

    /// Sum of two fractional ideals of the same order.
    pub fn add(&self, other: &Self) -> Result<Self, FractionalIdealError> {
        if self.order != other.order {
            return Err(self.mismatch(other).into());
        }
        let mut spanning = self.basis();
        spanning.extend(other.basis());
        Self::from_spanning_elements(self.order.clone(), &spanning)
    }

    /// Product of two fractional ideals of the same order.
    pub fn mul(&self, other: &Self) -> Result<Self, FractionalIdealError> {
        if self.order != other.order {
            return Err(self.mismatch(other).into());
        }
        let left_basis = self.basis();
        let right_basis = other.basis();
        let mut spanning = Vec::with_capacity(left_basis.len() * right_basis.len());
        for left in &left_basis {
            for right in &right_basis {
                spanning.push(left.mul(right)?);
            }
        }
        Self::from_spanning_elements(self.order.clone(), &spanning)
    }

    /// Exact absolute ideal norm, extended multiplicatively to fractional ideals.
    pub fn norm(&self) -> Rational {
        let degree = self.order.field().degree();
        let (ideal_numerator, ideal_denominator) = self.lattice.covolume_parts(degree);
        let (order_numerator, order_denominator) = self.order.covolume_parts();
        Rational::new(
            &ideal_numerator * &order_denominator,
            &ideal_denominator * &order_numerator,
        )
        .expect("an exact full-rank lattice has positive nonzero covolume")
    }

    fn subset_of(&self, other: &Self) -> bool {
        self.order == other.order && self.basis().iter().all(|element| other.contains(element))
    }

    fn scaled_by_integer(&self, scalar: &Integer) -> Result<Self, FractionalIdealError> {
        let scalar = self.order.field().scalar(Rational::from(scalar.clone()));
        let basis = self
            .basis()
            .iter()
            .map(|element| element.mul(&scalar))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_spanning_elements(self.order.clone(), &basis)
    }

    fn integral_multiple(&self) -> Result<(Integer, Self), IdealFactorizationError> {
        let mut denominator = Integer::one();
        for element in self.basis() {
            let coordinates = self.order.rational_coordinates(&element).ok_or(
                IdealFactorizationError::Invariant(
                    "fractional-ideal basis left its ambient order lattice",
                ),
            )?;
            for coordinate in coordinates {
                denominator = integer_lcm(&denominator, &coordinate.denominator());
            }
        }
        let integral = self.scaled_by_integer(&denominator)?;
        if !integral.is_integral() {
            return Err(IdealFactorizationError::Invariant(
                "clearing order-coordinate denominators did not make the ideal integral",
            ));
        }
        Ok((denominator, integral))
    }

    fn unit_ideal(&self) -> Result<Self, FractionalIdealError> {
        self.order.fractional_ideal(&[self.order.one()])
    }

    fn power(ideal: &Self, mut exponent: u64) -> Result<Self, IdealFactorizationError> {
        let mut output = ideal.unit_ideal()?;
        let mut base = ideal.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                output = output.mul(&base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(output)
    }

    fn integral_valuation(
        &self,
        prime: &PrimeIdeal<NumberFieldOrder>,
    ) -> Result<u64, IdealFactorizationError> {
        if !self.is_integral() {
            return Err(IdealFactorizationError::Invariant(
                "integral valuation received a nonintegral ideal",
            ));
        }
        let norm = self.norm();
        if !norm.denominator().is_one() {
            return Err(IdealFactorizationError::Invariant(
                "integral ideal had a nonintegral norm",
            ));
        }
        let (_, norm_exponent) = remove_factor(&norm.numerator(), &prime.rational_prime());
        let residue_degree = u64::try_from(prime.residue_degree())
            .map_err(|_| IdealFactorizationError::ExponentOverflow)?;
        let upper_bound = norm_exponent / residue_degree;
        if upper_bound == 0 {
            return Ok(0);
        }

        let prime_ideal = prime.as_fractional_ideal();
        let mut powers = vec![(1_u64, prime_ideal.clone())];
        while powers.last().expect("one prime power exists").0 <= upper_bound / 2 {
            let (exponent, power) = powers.last().expect("one prime power exists");
            powers.push((exponent * 2, power.mul(power)?));
        }

        let mut valuation = 0_u64;
        let mut contained_power = self.unit_ideal()?;
        for (step, power) in powers.iter().rev() {
            if *step > upper_bound - valuation {
                continue;
            }
            let candidate = contained_power.mul(power)?;
            if self.subset_of(&candidate) {
                valuation += *step;
                contained_power = candidate;
            }
        }
        if valuation < upper_bound && self.subset_of(&contained_power.mul(&prime_ideal)?) {
            return Err(IdealFactorizationError::Invariant(
                "binary ideal containment did not find the maximal valuation",
            ));
        }
        Ok(valuation)
    }

    /// The signed `P`-adic valuation of this fractional ideal.
    pub fn valuation(
        &self,
        prime: &PrimeIdeal<NumberFieldOrder>,
    ) -> Result<i64, IdealFactorizationError> {
        if self.order != *prime.order() {
            return Err(IdealFactorizationError::ContextMismatch(
                ContextMismatchError {
                    expected: format!(
                        "NumberFieldOrder(discriminant={})",
                        self.order.discriminant()
                    ),
                    actual: format!(
                        "NumberFieldOrder(discriminant={})",
                        prime.order().discriminant()
                    ),
                },
            ));
        }
        if !self.order.is_maximal()? {
            return Err(IdealFactorizationError::NonMaximalOrder);
        }
        let (denominator, integral) = self.integral_multiple()?;
        let integral_valuation = integral.integral_valuation(prime)?;
        let (_, denominator_exponent) = remove_factor(&denominator, &prime.rational_prime());
        let ramification = u64::try_from(prime.ramification_index())
            .map_err(|_| IdealFactorizationError::ExponentOverflow)?;
        let denominator_valuation = denominator_exponent
            .checked_mul(ramification)
            .ok_or(IdealFactorizationError::ExponentOverflow)?;
        let signed = i128::from(integral_valuation) - i128::from(denominator_valuation);
        i64::try_from(signed).map_err(|_| IdealFactorizationError::ExponentOverflow)
    }

    /// Complete signed prime-ideal factorization in this maximal order.
    ///
    /// The result is reconstructed exactly after moving negative powers to
    /// the left side, so no unchecked factor list can escape this method.
    pub fn factor(
        &self,
    ) -> Result<Vec<(PrimeIdeal<NumberFieldOrder>, i64)>, IdealFactorizationError> {
        if !self.order.is_maximal()? {
            return Err(IdealFactorizationError::NonMaximalOrder);
        }
        let (denominator, integral) = self.integral_multiple()?;
        let norm = integral.norm();
        if !norm.denominator().is_one() {
            return Err(IdealFactorizationError::Invariant(
                "integral ideal had a nonintegral norm",
            ));
        }

        let denominator_factorization = factor_integer(&denominator)?;
        let denominator_exponents = denominator_factorization
            .factors()
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();
        let mut rational_primes = denominator_exponents
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        rational_primes.extend(
            factor_integer(&norm.numerator())?
                .factors()
                .iter()
                .map(|(prime, _)| prime.clone()),
        );

        let mut factors = Vec::new();
        for rational_prime in rational_primes {
            let denominator_exponent = denominator_exponents
                .get(&rational_prime)
                .copied()
                .unwrap_or(0);
            for prime in self.order.prime_ideals_above(&rational_prime)? {
                let integral_valuation = integral.integral_valuation(&prime)?;
                let ramification = u64::try_from(prime.ramification_index())
                    .map_err(|_| IdealFactorizationError::ExponentOverflow)?;
                let cleared_valuation = denominator_exponent
                    .checked_mul(ramification)
                    .ok_or(IdealFactorizationError::ExponentOverflow)?;
                let exponent = i128::from(integral_valuation) - i128::from(cleared_valuation);
                let exponent = i64::try_from(exponent)
                    .map_err(|_| IdealFactorizationError::ExponentOverflow)?;
                if exponent != 0 {
                    factors.push((prime, exponent));
                }
            }
        }

        let mut left = self.clone();
        let mut right = self.unit_ideal()?;
        for (prime, exponent) in &factors {
            let power = Self::power(&prime.as_fractional_ideal(), exponent.unsigned_abs())?;
            if *exponent < 0 {
                left = left.mul(&power)?;
            } else {
                right = right.mul(&power)?;
            }
        }
        if left != right {
            return Err(IdealFactorizationError::Invariant(
                "prime powers did not reconstruct the fractional ideal",
            ));
        }
        Ok(factors)
    }

    /// Serialize this exact ideal to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and validate an exact fractional ideal from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PartialEq for FractionalIdeal {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order && self.lattice == other.lattice
    }
}

impl Eq for FractionalIdeal {}

impl fmt::Display for FractionalIdeal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FractionalIdeal(degree={}, norm={})",
            self.order.field().degree(),
            self.norm()
        )
    }
}

impl fmt::Debug for FractionalIdeal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl SemanticType for FractionalIdeal {
    const TYPE_ID: &'static str = "noether/numthy/FractionalIdeal@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.order.canonical_bytes()?)?;
        for element in self.basis() {
            append_component(&mut output, &element.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FRACTIONAL_IDEAL_DESCRIPTION
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "fractional-ideal component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

impl Add<&FractionalIdeal> for &FractionalIdeal {
    type Output = Result<FractionalIdeal, FractionalIdealError>;

    fn add(self, rhs: &FractionalIdeal) -> Self::Output {
        self.add(rhs)
    }
}

impl Mul<&FractionalIdeal> for &FractionalIdeal {
    type Output = Result<FractionalIdeal, FractionalIdealError>;

    fn mul(self, rhs: &FractionalIdeal) -> Self::Output {
        self.mul(rhs)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FractionalIdealRepr {
    order: NumberFieldOrder,
    basis: Vec<Vec<String>>,
}

impl Serialize for FractionalIdeal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        FractionalIdealRepr {
            order: self.order.clone(),
            basis: self
                .basis()
                .into_iter()
                .map(|element| {
                    element
                        .coefficients()
                        .into_iter()
                        .map(|coefficient| coefficient.to_string())
                        .collect()
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FractionalIdeal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = FractionalIdealRepr::deserialize(deserializer)?;
        let basis = repr
            .basis
            .iter()
            .map(|coefficients| {
                let coefficients = coefficients
                    .iter()
                    .map(|coefficient| {
                        coefficient
                            .parse::<Rational>()
                            .map_err(serde::de::Error::custom)
                    })
                    .collect::<Result<Vec<_>, D::Error>>()?;
                repr.order
                    .field()
                    .element(&coefficients)
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::new(repr.order, &basis).map_err(serde::de::Error::custom)
    }
}

impl NumberFieldOrder {
    /// Construct the nonzero fractional ideal generated by `generators`.
    pub fn fractional_ideal(
        &self,
        generators: &[NumberFieldElement],
    ) -> Result<FractionalIdeal, FractionalIdealError> {
        FractionalIdeal::new(self.clone(), generators)
    }
}
