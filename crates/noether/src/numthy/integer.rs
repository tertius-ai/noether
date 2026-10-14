// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Proven integer arithmetic and canonical integer polynomials.

#![allow(unsafe_code)]

use core::fmt;
use std::mem::MaybeUninit;
use std::str::FromStr;

use flint_sys::{
    fmpz_bits, fmpz_factor, fmpz_factor_clear, fmpz_factor_init, fmpz_factor_t,
    fmpz_is_probabprime_BPSW, fmpz_poly_clear, fmpz_poly_cyclotomic, fmpz_poly_get_coeff_fmpz,
    fmpz_poly_init, fmpz_poly_length, fmpz_poly_struct, fmpz_remove, n_nextprime,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{MonomialOrder, Polynomial, PolynomialRing};
use crate::{Integer, IntegerRing, NoetherError, SemanticType, TypeDescription, Z};

static INTEGER_FACTORIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/IntegerFactorization@1",
    rust_path: "noether::numthy::IntegerFactorization",
    python_path: Some("noether.numthy.IntegerFactorization"),
};

static PARTIAL_INTEGER_FACTORIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PartialIntegerFactorization@1",
    rust_path: "noether::numthy::PartialIntegerFactorization",
    python_path: Some("noether.numthy.PartialIntegerFactorization"),
};

/// Failure to factor an integer or validate a serialized factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorError {
    /// Zero has no finite prime factorization.
    Zero,
    /// The unit must be `-1` or `1`.
    InvalidUnit(i8),
    /// A claimed factor is not a positive proved prime.
    InvalidPrime(Integer),
    /// Every listed prime must have positive multiplicity.
    ZeroExponent(Integer),
    /// Prime factors must be strictly increasing and unique.
    NonIncreasingFactors,
}

impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => f.write_str("factorization of zero is undefined"),
            Self::InvalidUnit(unit) => write!(f, "factorization unit must be -1 or 1, got {unit}"),
            Self::InvalidPrime(prime) => {
                write!(f, "factorization contains a non-prime factor {prime}")
            }
            Self::ZeroExponent(prime) => {
                write!(f, "factorization gives prime {prime} exponent zero")
            }
            Self::NonIncreasingFactors => {
                f.write_str("factorization primes must be strictly increasing")
            }
        }
    }
}

impl std::error::Error for FactorError {}

/// The complete proved prime factorization of one nonzero integer.
///
/// The represented value is `unit * product(prime^exponent)`. Factors are
/// positive proved primes in strictly increasing order, so the representation
/// is unique and reconstructs negative inputs and units exactly.
#[derive(Clone, PartialEq, Eq)]
pub struct IntegerFactorization {
    unit: i8,
    factors: Vec<(Integer, u64)>,
}

impl IntegerFactorization {
    fn from_parts(unit: i8, factors: Vec<(Integer, u64)>) -> Result<Self, FactorError> {
        if !matches!(unit, -1 | 1) {
            return Err(FactorError::InvalidUnit(unit));
        }
        let mut previous: Option<&Integer> = None;
        for (prime, exponent) in &factors {
            if prime.sign() <= 0 || !prime.is_prime() {
                return Err(FactorError::InvalidPrime(prime.clone()));
            }
            if *exponent == 0 {
                return Err(FactorError::ZeroExponent(prime.clone()));
            }
            if previous.is_some_and(|value| value >= prime) {
                return Err(FactorError::NonIncreasingFactors);
            }
            previous = Some(prime);
        }
        Ok(Self { unit, factors })
    }

    /// The multiplicative unit, either `-1` or `1`.
    pub fn unit(&self) -> i8 {
        self.unit
    }

    /// Strictly increasing `(prime, exponent)` pairs.
    pub fn factors(&self) -> &[(Integer, u64)] {
        &self.factors
    }

    /// Reconstruct the exact integer represented by this factorization.
    pub fn value(&self) -> Integer {
        let mut value = Integer::from(self.unit);
        for (prime, exponent) in &self.factors {
            value = &value * &prime.pow(*exponent);
        }
        value
    }

    /// Verify that this factorization reconstructs `value`.
    pub fn verify(&self, value: &Integer) -> bool {
        self.value() == *value
    }

    /// Serialize this factorization to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate a canonical integer factorization from JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for IntegerFactorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.factors.is_empty() {
            return write!(f, "{}", self.unit);
        }
        if self.unit < 0 {
            f.write_str("-1 * ")?;
        }
        for (index, (prime, exponent)) in self.factors.iter().enumerate() {
            if index != 0 {
                f.write_str(" * ")?;
            }
            if *exponent == 1 {
                write!(f, "{prime}")?;
            } else {
                write!(f, "{prime}^{exponent}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for IntegerFactorization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IntegerFactorization({self})")
    }
}

impl SemanticType for IntegerFactorization {
    const TYPE_ID: &'static str = "noether/numthy/IntegerFactorization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_FACTORIZATION_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactorRepr {
    prime: String,
    exponent: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerFactorizationRepr {
    unit: i8,
    factors: Vec<FactorRepr>,
}

impl Serialize for IntegerFactorization {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        IntegerFactorizationRepr {
            unit: self.unit,
            factors: self
                .factors
                .iter()
                .map(|(prime, exponent)| FactorRepr {
                    prime: prime.to_string(),
                    exponent: *exponent,
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for IntegerFactorization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = IntegerFactorizationRepr::deserialize(deserializer)?;
        let factors = repr
            .factors
            .into_iter()
            .map(|factor| {
                Integer::from_str(&factor.prime)
                    .map(|prime| (prime, factor.exponent))
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::from_parts(repr.unit, factors).map_err(serde::de::Error::custom)
    }
}

/// Proof status of the positive cofactor left by bounded integer factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CofactorClassification {
    /// The remaining cofactor is exactly one.
    Unit,
    /// The remaining cofactor passed a deterministic primality proof.
    ProvedPrime,
    /// A deterministic test proved the remaining cofactor composite.
    ProvedComposite,
    /// The work bound did not prove primality or compositeness.
    Unclassified,
}

impl fmt::Display for CofactorClassification {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unit => "unit",
            Self::ProvedPrime => "proved prime",
            Self::ProvedComposite => "proved composite",
            Self::Unclassified => "unclassified",
        })
    }
}

/// Failure to perform or validate bounded integer factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialFactorError {
    /// Zero has no finite factorization.
    Zero,
    /// The requested trial-division bound exceeds the operation's safety cap.
    TrialBoundTooLarge {
        /// Rejected bound.
        requested: u64,
        /// Largest supported bound.
        limit: u64,
    },
    /// Serialized evidence violated the result contract.
    InvalidCertificate(String),
}

impl fmt::Display for PartialFactorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str("factorization of zero is undefined"),
            Self::TrialBoundTooLarge { requested, limit } => write!(
                formatter,
                "trial-division bound {requested} exceeds the safety limit of {limit}"
            ),
            Self::InvalidCertificate(reason) => {
                write!(
                    formatter,
                    "invalid partial factorization certificate: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for PartialFactorError {}

/// Deterministic bounded factorization of one nonzero integer.
///
/// Every listed factor is a proved prime. The positive cofactor is retained
/// explicitly and its proof status never promotes a probable prime to a
/// proved prime.
#[derive(Clone, PartialEq, Eq)]
pub struct PartialIntegerFactorization {
    source: Integer,
    unit: i8,
    factors: Vec<(Integer, u64)>,
    cofactor: Integer,
    classification: CofactorClassification,
    trial_bound: u64,
    primality_proof_bits: u64,
}

impl PartialIntegerFactorization {
    /// Original nonzero integer.
    pub fn source(&self) -> &Integer {
        &self.source
    }

    /// Multiplicative unit, either `-1` or `1`.
    pub fn unit(&self) -> i8 {
        self.unit
    }

    /// Strictly increasing proved prime factors found within the work bound.
    pub fn factors(&self) -> &[(Integer, u64)] {
        &self.factors
    }

    /// Positive remaining cofactor.
    pub fn cofactor(&self) -> &Integer {
        &self.cofactor
    }

    /// Honest proof status of the remaining cofactor.
    pub fn classification(&self) -> CofactorClassification {
        self.classification
    }

    /// Inclusive upper bound used for deterministic trial division.
    pub fn trial_bound(&self) -> u64 {
        self.trial_bound
    }

    /// Largest cofactor bit length admitted to deterministic primality proof.
    pub fn primality_proof_bits(&self) -> u64 {
        self.primality_proof_bits
    }

    /// Reconstruct the exact source integer.
    pub fn value(&self) -> Integer {
        let mut value = Integer::from(self.unit);
        for (prime, exponent) in &self.factors {
            value = &value * &prime.pow(*exponent);
        }
        &value * &self.cofactor
    }

    /// Recompute the bounded operation and validate every proof-status claim.
    pub fn verify(&self) -> bool {
        partial_factor(&self.source, self.trial_bound, self.primality_proof_bits)
            .is_ok_and(|canonical| canonical == *self)
    }

    /// Serialize the complete bounded result as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recompute the bounded result before accepting it.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: PartialIntegerFactorizationRepr = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let source = wire
            .source
            .parse::<Integer>()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let canonical = partial_factor(&source, wire.trial_bound, wire.primality_proof_bits)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let factors = parse_factor_repr(wire.factors)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let cofactor = wire
            .cofactor
            .parse::<Integer>()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if canonical.unit != wire.unit
            || canonical.factors != factors
            || canonical.cofactor != cofactor
            || canonical.classification != wire.classification
        {
            return Err(NoetherError::Serialization(
                "serialized partial factorization is not the canonical bounded result".to_owned(),
            ));
        }
        Ok(canonical)
    }
}

impl fmt::Display for PartialIntegerFactorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unit < 0 {
            formatter.write_str("-1 * ")?;
        }
        for (index, (prime, exponent)) in self.factors.iter().enumerate() {
            if index != 0 {
                formatter.write_str(" * ")?;
            }
            if *exponent == 1 {
                write!(formatter, "{prime}")?;
            } else {
                write!(formatter, "{prime}^{exponent}")?;
            }
        }
        if !self.factors.is_empty() {
            formatter.write_str(" * ")?;
        }
        write!(formatter, "{} [{}]", self.cofactor, self.classification)
    }
}

impl fmt::Debug for PartialIntegerFactorization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "PartialIntegerFactorization({self})")
    }
}

impl SemanticType for PartialIntegerFactorization {
    const TYPE_ID: &'static str = "noether/numthy/PartialIntegerFactorization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &PARTIAL_INTEGER_FACTORIZATION_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialIntegerFactorizationRepr {
    #[serde(rename = "type")]
    type_id: String,
    source: String,
    unit: i8,
    factors: Vec<FactorRepr>,
    cofactor: String,
    classification: CofactorClassification,
    trial_bound: u64,
    primality_proof_bits: u64,
}

impl Serialize for PartialIntegerFactorization {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PartialIntegerFactorizationRepr {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.to_string(),
            unit: self.unit,
            factors: self
                .factors
                .iter()
                .map(|(prime, exponent)| FactorRepr {
                    prime: prime.to_string(),
                    exponent: *exponent,
                })
                .collect(),
            cofactor: self.cofactor.to_string(),
            classification: self.classification,
            trial_bound: self.trial_bound,
            primality_proof_bits: self.primality_proof_bits,
        }
        .serialize(serializer)
    }
}

fn parse_factor_repr(factors: Vec<FactorRepr>) -> Result<Vec<(Integer, u64)>, String> {
    factors
        .into_iter()
        .map(|factor| {
            factor
                .prime
                .parse::<Integer>()
                .map(|prime| (prime, factor.exponent))
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// Return deterministic partial information within explicit work bounds.
///
/// Trial division visits every proved machine-word prime at most
/// `trial_bound`. A remaining cofactor of at most `primality_proof_bits` bits
/// receives FLINT's deterministic primality proof. Larger cofactors receive a
/// deterministic BPSW compositeness test: failure proves composite, while a
/// pass is reported only as unclassified.
pub fn partial_factor(
    value: &Integer,
    trial_bound: u64,
    primality_proof_bits: u64,
) -> Result<PartialIntegerFactorization, PartialFactorError> {
    const MAX_TRIAL_BOUND: u64 = 1_000_000_000;
    if value.is_zero() {
        return Err(PartialFactorError::Zero);
    }
    if trial_bound > MAX_TRIAL_BOUND {
        return Err(PartialFactorError::TrialBoundTooLarge {
            requested: trial_bound,
            limit: MAX_TRIAL_BOUND,
        });
    }
    let source = value.clone();
    let unit = if value.sign() < 0 { -1 } else { 1 };
    let mut cofactor = value.abs();
    let mut factors = Vec::new();
    if trial_bound >= 2 {
        let mut prime = 2_u64;
        while prime <= trial_bound && !cofactor.is_one() {
            let prime_integer = Integer::from(prime);
            let (remaining, exponent) = remove_factor(&cofactor, &prime_integer);
            cofactor = remaining;
            if exponent != 0 {
                factors.push((prime_integer, exponent));
            }
            if prime >= trial_bound {
                break;
            }
            let next = unsafe { n_nextprime(prime, 1) };
            if next <= prime {
                break;
            }
            prime = next;
        }
    }
    let classification = classify_cofactor(&cofactor, primality_proof_bits);
    let result = PartialIntegerFactorization {
        source,
        unit,
        factors,
        cofactor,
        classification,
        trial_bound,
        primality_proof_bits,
    };
    if result.value() != *value {
        return Err(PartialFactorError::InvalidCertificate(
            "the bounded factors do not reconstruct the source".to_owned(),
        ));
    }
    Ok(result)
}

fn classify_cofactor(cofactor: &Integer, primality_proof_bits: u64) -> CofactorClassification {
    if cofactor.is_one() {
        return CofactorClassification::Unit;
    }
    let bits = unsafe { fmpz_bits(cofactor.as_ptr()) };
    if bits <= primality_proof_bits {
        if cofactor.is_prime() {
            CofactorClassification::ProvedPrime
        } else {
            CofactorClassification::ProvedComposite
        }
    } else if unsafe { fmpz_is_probabprime_BPSW(cofactor.as_ptr()) } == 0 {
        CofactorClassification::ProvedComposite
    } else {
        CofactorClassification::Unclassified
    }
}

/// Return the complete proved prime factorization of a nonzero integer.
pub fn factor(value: &Integer) -> Result<IntegerFactorization, FactorError> {
    if value.is_zero() {
        return Err(FactorError::Zero);
    }
    let mut factorization = MaybeUninit::<fmpz_factor_t>::zeroed();
    let mut factors = Vec::new();
    unsafe {
        fmpz_factor_init(factorization.as_mut_ptr().cast());
        let mut factorization = factorization.assume_init();
        fmpz_factor(factorization.as_mut_ptr(), value.as_ptr());
        let raw = &*factorization.as_ptr();
        factors.reserve(raw.num as usize);
        for index in 0..raw.num as usize {
            factors.push((
                Integer::clone_from_ptr(raw.p.add(index)),
                *raw.exp.add(index),
            ));
        }
        fmpz_factor_clear(factorization.as_mut_ptr());
    }
    IntegerFactorization::from_parts(if value.sign() < 0 { -1 } else { 1 }, factors)
}

/// Test primality with FLINT's deterministic primality prover.
pub fn is_prime(value: &Integer) -> bool {
    value.is_prime()
}

pub(super) fn remove_factor(value: &Integer, factor: &Integer) -> (Integer, u64) {
    let mut cofactor = Integer::zero();
    let exponent = unsafe { fmpz_remove(cofactor.as_mut_ptr(), value.as_ptr(), factor.as_ptr()) };
    (
        cofactor,
        u64::try_from(exponent).expect("a positive factor has nonnegative multiplicity"),
    )
}

/// Failure to construct a cyclotomic polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CyclotomicPolynomialError {
    /// Cyclotomic polynomial indices start at one.
    ZeroIndex,
    /// The fixed univariate integer-polynomial context could not be built.
    InternalInvariant,
}

impl fmt::Display for CyclotomicPolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroIndex => f.write_str("cyclotomic polynomial index must be positive"),
            Self::InternalInvariant => {
                f.write_str("failed to construct the canonical integer polynomial context")
            }
        }
    }
}

impl std::error::Error for CyclotomicPolynomialError {}

/// Return the `index`-th cyclotomic polynomial in the canonical ring `Z[x]`.
pub fn cyclotomic_polynomial(
    index: u64,
) -> Result<Polynomial<IntegerRing>, CyclotomicPolynomialError> {
    if index == 0 {
        return Err(CyclotomicPolynomialError::ZeroIndex);
    }

    let mut raw = MaybeUninit::<fmpz_poly_struct>::uninit();
    let coefficients = unsafe {
        fmpz_poly_init(raw.as_mut_ptr());
        let mut raw = raw.assume_init();
        fmpz_poly_cyclotomic(&mut raw, index);
        let length = fmpz_poly_length(&raw);
        let mut coefficients = Vec::with_capacity(length as usize);
        for exponent in 0..length {
            let mut coefficient = Integer::zero();
            fmpz_poly_get_coeff_fmpz(coefficient.as_mut_ptr(), &raw, exponent);
            coefficients.push(coefficient);
        }
        fmpz_poly_clear(&mut raw);
        coefficients
    };

    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic)
        .map_err(|_| CyclotomicPolynomialError::InternalInvariant)?;
    ring.from_u64_terms(
        coefficients
            .into_iter()
            .enumerate()
            .map(|(exponent, coefficient)| (vec![exponent as u64], coefficient)),
    )
    .map_err(|_| CyclotomicPolynomialError::InternalInvariant)
}
