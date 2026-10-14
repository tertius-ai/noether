// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Arithmetic certificates for integral polynomial ideals.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::numthy::{
    partial_factor, CofactorClassification, PartialFactorError, PartialIntegerFactorization,
};
use crate::{
    Integer, IntegerRing, NoetherError, RationalField, Ring, SemanticType, TypeDescription, Z,
};

use super::groebner::unit_ideal_multipliers;
use super::{BaseChangeError, GroebnerError, Ideal, Polynomial, PolynomialError, PolynomialRing};

static NULLSTELLENSATZ_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ArithmeticNullstellensatzCertificate@1",
    rust_path: "noether::commalg::ArithmeticNullstellensatzCertificate",
    python_path: Some("noether.commalg.ArithmeticNullstellensatzCertificate"),
};

static FIBER_SUPPORT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ArithmeticFiberSupport@1",
    rust_path: "noether::commalg::ArithmeticFiberSupport",
    python_path: Some("noether.commalg.ArithmeticFiberSupport"),
};

/// Failure to construct or verify arithmetic ideal evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithmeticIdealError {
    /// Integral strong-basis or normal-form computation failed.
    Groebner(GroebnerError),
    /// Exact polynomial construction failed.
    Polynomial(PolynomialError),
    /// A checked coefficient extension failed.
    BaseChange(BaseChangeError),
    /// The requested bounded factorization failed.
    PartialFactor(PartialFactorError),
    /// The generic fiber is nonempty, so no nonzero integer lies in the ideal.
    GenericFiberNonempty,
    /// Supplied evidence contradicted an exact invariant.
    InvalidCertificate(&'static str),
}

impl fmt::Display for ArithmeticIdealError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Groebner(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::BaseChange(error) => error.fmt(formatter),
            Self::PartialFactor(error) => error.fmt(formatter),
            Self::GenericFiberNonempty => {
                formatter.write_str("the integral polynomial ideal has nonempty generic fiber")
            }
            Self::InvalidCertificate(reason) => {
                write!(formatter, "invalid arithmetic ideal certificate: {reason}")
            }
        }
    }
}

impl std::error::Error for ArithmeticIdealError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Groebner(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::BaseChange(error) => Some(error),
            Self::PartialFactor(error) => Some(error),
            Self::GenericFiberNonempty | Self::InvalidCertificate(_) => None,
        }
    }
}

impl From<GroebnerError> for ArithmeticIdealError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<PolynomialError> for ArithmeticIdealError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<BaseChangeError> for ArithmeticIdealError {
    fn from(error: BaseChangeError) -> Self {
        Self::BaseChange(error)
    }
}

impl From<PartialFactorError> for ArithmeticIdealError {
    fn from(error: PartialFactorError) -> Self {
        Self::PartialFactor(error)
    }
}

/// Exact arithmetic Nullstellensatz identity for an integral polynomial ideal.
///
/// For the recorded generator family `f_i`, this value proves
/// `integer = sum(multipliers[i] * f_i)`. Its principal ideal is a certified
/// containing support for non-unit fibers. It need not equal the exact
/// coefficient contraction: that stronger claim is exposed and checked
/// separately.
#[derive(Clone, PartialEq, Eq)]
pub struct ArithmeticNullstellensatzCertificate {
    ring: PolynomialRing<IntegerRing>,
    generators: Vec<Polynomial<IntegerRing>>,
    integer: Integer,
    principal_ideal: Ideal<IntegerRing>,
    multipliers: Vec<Polynomial<IntegerRing>>,
}

impl ArithmeticNullstellensatzCertificate {
    /// Integral polynomial context of the certified generator family.
    pub fn ring(&self) -> &PolynomialRing<IntegerRing> {
        &self.ring
    }

    /// Generator family appearing in the exact identity.
    pub fn generators(&self) -> &[Polynomial<IntegerRing>] {
        &self.generators
    }

    /// Positive nonzero integer represented by the multiplier identity.
    pub fn integer(&self) -> &Integer {
        &self.integer
    }

    /// Principal witness ideal `(integer)` in `Z`.
    pub fn principal_ideal(&self) -> &Ideal<IntegerRing> {
        &self.principal_ideal
    }

    /// Multipliers in the recorded generator order.
    pub fn multipliers(&self) -> &[Polynomial<IntegerRing>] {
        &self.multipliers
    }

    /// Construct and verify an exact multiplier identity.
    ///
    /// A negative witness and its multipliers are normalized simultaneously,
    /// so the stored integer is always positive.
    pub fn from_identity(
        source: Ideal<PolynomialRing<IntegerRing>>,
        integer: Integer,
        multipliers: Vec<Polynomial<IntegerRing>>,
    ) -> Result<Self, ArithmeticIdealError> {
        Self::from_generator_identity(
            source.ring().clone(),
            source.generators().to_vec(),
            integer,
            multipliers,
        )
    }

    /// Construct and verify an identity against an explicit generator family
    /// without first computing a strong integral basis.
    pub fn from_generator_identity(
        ring: PolynomialRing<IntegerRing>,
        generators: Vec<Polynomial<IntegerRing>>,
        integer: Integer,
        mut multipliers: Vec<Polynomial<IntegerRing>>,
    ) -> Result<Self, ArithmeticIdealError> {
        if integer.is_zero() {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the witness integer is zero",
            ));
        }
        let integer = if integer.is_negative() {
            for multiplier in &mut multipliers {
                *multiplier = ring.neg(multiplier).map_err(GroebnerError::from)?;
            }
            -integer
        } else {
            integer
        };
        let certificate = Self {
            ring,
            generators,
            principal_ideal: Z.ideal([integer.clone()]),
            integer,
            multipliers,
        };
        certificate.verify()?;
        Ok(certificate)
    }

    /// Verify the exact multiplier identity.
    pub fn verify(&self) -> Result<(), ArithmeticIdealError> {
        if !self.integer.is_positive() {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the witness integer is not positive",
            ));
        }
        if self.principal_ideal != Z.ideal([self.integer.clone()]) {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the principal ideal does not match the witness integer",
            ));
        }
        if self
            .generators
            .iter()
            .any(|generator| !self.ring.contains(generator))
            || self.multipliers.len() != self.generators.len()
            || self
                .multipliers
                .iter()
                .any(|multiplier| !self.ring.contains(multiplier))
        {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the multiplier family does not match the source ideal",
            ));
        }
        let represented = self.multipliers.iter().zip(&self.generators).try_fold(
            self.ring.zero(),
            |sum, (multiplier, generator)| {
                let product = self.ring.mul(multiplier, generator)?;
                self.ring.add(&sum, &product).map_err(PolynomialError::from)
            },
        )?;
        if represented != self.ring.constant(self.integer.clone())? {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the multipliers do not represent the witness integer",
            ));
        }
        Ok(())
    }

    /// Check whether this witness generates the exact coefficient contraction.
    pub fn is_exact_contraction(&self) -> Result<bool, ArithmeticIdealError> {
        let ideal = self.ring.ideal(self.generators.clone())?;
        Ok(ideal.contract_to_coefficients()? == self.principal_ideal)
    }

    /// Partially factor the witness and expose its certified containing support.
    pub fn arithmetic_fiber_support(
        &self,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> Result<ArithmeticFiberSupport, ArithmeticIdealError> {
        ArithmeticFiberSupport::from_certificate(self.clone(), trial_bound, primality_proof_bits)
    }

    /// Serialize the exact identity as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and verify the supplied exact identity.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: ArithmeticNullstellensatzWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = Self {
            ring: wire.ring,
            generators: wire.generators,
            integer: wire.integer,
            principal_ideal: wire.principal_ideal,
            multipliers: wire.multipliers,
        };
        supplied
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(supplied)
    }
}

impl fmt::Debug for ArithmeticNullstellensatzCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArithmeticNullstellensatzCertificate")
            .field("integer", &self.integer)
            .field("multipliers", &self.multipliers.len())
            .finish()
    }
}

impl fmt::Display for ArithmeticNullstellensatzCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ArithmeticNullstellensatzCertificate(integer={}, multipliers={})",
            self.integer,
            self.multipliers.len()
        )
    }
}

impl SemanticType for ArithmeticNullstellensatzCertificate {
    const TYPE_ID: &'static str = "noether/commalg/ArithmeticNullstellensatzCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &NULLSTELLENSATZ_DESCRIPTION
    }
}

impl Serialize for ArithmeticNullstellensatzCertificate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ArithmeticNullstellensatzCertificate", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("generators", &self.generators)?;
        state.serialize_field("integer", &self.integer)?;
        state.serialize_field("principal_ideal", &self.principal_ideal)?;
        state.serialize_field("multipliers", &self.multipliers)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArithmeticNullstellensatzWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<IntegerRing>,
    generators: Vec<Polynomial<IntegerRing>>,
    integer: Integer,
    principal_ideal: Ideal<IntegerRing>,
    multipliers: Vec<Polynomial<IntegerRing>>,
}

impl PolynomialRing<IntegerRing> {
    /// Compute a rational Nullstellensatz identity for an explicit integral
    /// generator family without first constructing a strong basis over `Z`.
    pub fn nullstellensatz_certificate<I>(
        &self,
        generators: I,
    ) -> Result<ArithmeticNullstellensatzCertificate, ArithmeticIdealError>
    where
        I: IntoIterator<Item = Polynomial<IntegerRing>>,
    {
        let generators = generators.into_iter().collect::<Vec<_>>();
        if generators.iter().any(|generator| !self.contains(generator)) {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "a generator belongs to another polynomial context",
            ));
        }
        let embedding = Z.rational_embedding();
        let rational_ring = self.base_change(&embedding)?;
        let rational_generators = generators
            .iter()
            .map(|generator| generator.base_change(&embedding))
            .collect::<Result<Vec<_>, _>>()?;
        let rational_multipliers = unit_ideal_multipliers(&rational_ring, &rational_generators)?
            .ok_or(ArithmeticIdealError::GenericFiberNonempty)?;
        let (integer, multipliers) = clear_rational_denominators(self, &rational_multipliers)?;
        ArithmeticNullstellensatzCertificate::from_generator_identity(
            self.clone(),
            generators,
            integer,
            multipliers,
        )
    }
}

fn clear_rational_denominators(
    ring: &PolynomialRing<IntegerRing>,
    multipliers: &[Polynomial<RationalField>],
) -> Result<(Integer, Vec<Polynomial<IntegerRing>>), ArithmeticIdealError> {
    let denominator = multipliers
        .iter()
        .flat_map(|multiplier| multiplier.terms())
        .fold(Integer::one(), |common, (_, coefficient)| {
            common.lcm(&coefficient.denominator())
        });
    let integral = multipliers
        .iter()
        .map(|multiplier| {
            multiplier
                .terms()
                .map(|(exponent, coefficient)| {
                    let scale = &denominator / &coefficient.denominator();
                    (exponent.to_vec(), coefficient.numerator() * scale)
                })
                .collect::<Vec<_>>()
        })
        .map(|terms| ring.from_terms(terms).map_err(ArithmeticIdealError::from))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((denominator, integral))
}

impl Ideal<PolynomialRing<IntegerRing>> {
    /// Produce the exact positive coefficient-contraction generator and
    /// multiplier identity when the generic fiber is empty.
    pub fn nullstellensatz_certificate(
        &self,
    ) -> Result<ArithmeticNullstellensatzCertificate, ArithmeticIdealError> {
        let principal_ideal = self.contract_to_coefficients()?;
        let integer = principal_ideal.principal_generator();
        if integer.is_zero() {
            return Err(ArithmeticIdealError::GenericFiberNonempty);
        }
        let value = self.ring().constant(integer.clone())?;
        let reduction = self.reduce(&value)?;
        if !reduction.is_member() {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the contraction generator did not reduce to zero",
            ));
        }
        let certificate = ArithmeticNullstellensatzCertificate::from_identity(
            self.clone(),
            integer,
            reduction.multipliers().to_vec(),
        )?;
        if certificate.principal_ideal != principal_ideal {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "the exact contraction changed during certificate construction",
            ));
        }
        certificate.verify()?;
        Ok(certificate)
    }
}

/// Mathematical strength of an arithmetic fiber-support result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArithmeticFiberSupportClaim {
    /// The principal ideal is the exact contraction to `Z`.
    ExactContraction,
    /// A multiplier identity proves only that the actual support is contained.
    CertifiedContainment,
}

impl fmt::Display for ArithmeticFiberSupportClaim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ExactContraction => "exact contraction",
            Self::CertifiedContainment => "certified containment",
        })
    }
}

/// Certified arithmetic support of non-unit fibers of an integral ideal.
///
/// `claim` records whether `support_ideal` is exact or only a certified
/// containing support. Independently, `factorization` records whether every
/// prime in that principal ideal has been identified.
#[derive(Clone, PartialEq, Eq)]
pub struct ArithmeticFiberSupport {
    ring: PolynomialRing<IntegerRing>,
    generators: Vec<Polynomial<IntegerRing>>,
    support_ideal: Ideal<IntegerRing>,
    claim: ArithmeticFiberSupportClaim,
    certificate: Option<ArithmeticNullstellensatzCertificate>,
    factorization: PartialIntegerFactorization,
}

impl ArithmeticFiberSupport {
    fn exact(
        source: Ideal<PolynomialRing<IntegerRing>>,
        support_ideal: Ideal<IntegerRing>,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> Result<Self, ArithmeticIdealError> {
        let factorization = partial_factor(
            &support_ideal.principal_generator(),
            trial_bound,
            primality_proof_bits,
        )?;
        let result = Self {
            ring: source.ring().clone(),
            generators: source.generators().to_vec(),
            support_ideal,
            claim: ArithmeticFiberSupportClaim::ExactContraction,
            certificate: None,
            factorization,
        };
        result.verify()?;
        Ok(result)
    }

    fn from_certificate(
        certificate: ArithmeticNullstellensatzCertificate,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> Result<Self, ArithmeticIdealError> {
        certificate.verify()?;
        let factorization =
            partial_factor(certificate.integer(), trial_bound, primality_proof_bits)?;
        let result = Self {
            ring: certificate.ring().clone(),
            generators: certificate.generators().to_vec(),
            support_ideal: certificate.principal_ideal().clone(),
            claim: ArithmeticFiberSupportClaim::CertifiedContainment,
            certificate: Some(certificate),
            factorization,
        };
        result.verify()?;
        Ok(result)
    }

    /// Integral polynomial context whose fibers are being studied.
    pub fn ring(&self) -> &PolynomialRing<IntegerRing> {
        &self.ring
    }

    /// Generator family defining the fibers under study.
    pub fn generators(&self) -> &[Polynomial<IntegerRing>] {
        &self.generators
    }

    /// Exact or containing principal support ideal, as specified by `claim`.
    pub fn support_ideal(&self) -> &Ideal<IntegerRing> {
        &self.support_ideal
    }

    /// Mathematical strength of the support ideal.
    pub fn claim(&self) -> ArithmeticFiberSupportClaim {
        self.claim
    }

    /// Whether the support ideal is the exact contraction to `Z`.
    pub fn is_exact(&self) -> bool {
        self.claim == ArithmeticFiberSupportClaim::ExactContraction
    }

    /// Multiplier identity when the result is a containing support.
    pub fn certificate(&self) -> Option<&ArithmeticNullstellensatzCertificate> {
        self.certificate.as_ref()
    }

    /// Bounded factorization evidence for the support generator.
    pub fn factorization(&self) -> &PartialIntegerFactorization {
        &self.factorization
    }

    /// Proved prime factors currently known, including a proved-prime cofactor.
    pub fn proved_primes(&self) -> Vec<(Integer, u64)> {
        let mut factors = self.factorization.factors().to_vec();
        if self.factorization.classification() == CofactorClassification::ProvedPrime {
            factors.push((self.factorization.cofactor().clone(), 1));
        }
        factors
    }

    /// Whether every prime dividing the support generator has been identified.
    pub fn prime_support_is_complete(&self) -> bool {
        matches!(
            self.factorization.classification(),
            CofactorClassification::Unit | CofactorClassification::ProvedPrime
        )
    }

    /// Positive cofactor whose prime divisors have not yet been identified.
    pub fn unfactored_cofactor(&self) -> Option<&Integer> {
        if self.prime_support_is_complete() {
            None
        } else {
            Some(self.factorization.cofactor())
        }
    }

    /// Verify support strength, multiplier evidence, and bounded factorization.
    pub fn verify(&self) -> Result<(), ArithmeticIdealError> {
        let generator = self.support_ideal.principal_generator();
        if generator.is_zero() {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "arithmetic fiber support has zero generator",
            ));
        }
        if self.factorization.source() != &generator || !self.factorization.verify() {
            return Err(ArithmeticIdealError::InvalidCertificate(
                "partial factorization does not certify the support generator",
            ));
        }
        match self.claim {
            ArithmeticFiberSupportClaim::ExactContraction => {
                if self.certificate.is_some() {
                    return Err(ArithmeticIdealError::InvalidCertificate(
                        "an exact contraction unexpectedly carries a containment witness",
                    ));
                }
                let source = self.ring.ideal(self.generators.clone())?;
                if source.contract_to_coefficients()? != self.support_ideal {
                    return Err(ArithmeticIdealError::InvalidCertificate(
                        "the support ideal is not the exact coefficient contraction",
                    ));
                }
            }
            ArithmeticFiberSupportClaim::CertifiedContainment => {
                let certificate =
                    self.certificate
                        .as_ref()
                        .ok_or(ArithmeticIdealError::InvalidCertificate(
                            "a containing support lacks a Nullstellensatz witness",
                        ))?;
                certificate.verify()?;
                if certificate.ring() != &self.ring
                    || certificate.generators() != self.generators
                    || certificate.principal_ideal() != &self.support_ideal
                {
                    return Err(ArithmeticIdealError::InvalidCertificate(
                        "the containment witness does not match the support result",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Serialize the complete support evidence as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and verify support, witness, and bounded factorization.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: ArithmeticFiberSupportWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let certificate = wire
            .certificate
            .map(|certificate| {
                serde_json::to_string(&certificate)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
                    .and_then(|value| ArithmeticNullstellensatzCertificate::from_json(&value))
            })
            .transpose()?;
        let factorization_json = serde_json::to_string(&wire.factorization)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let factorization = PartialIntegerFactorization::from_json(&factorization_json)?;
        let supplied = Self {
            ring: wire.ring,
            generators: wire.generators,
            support_ideal: wire.support_ideal,
            claim: wire.claim,
            certificate,
            factorization,
        };
        supplied
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(supplied)
    }
}

impl Ideal<PolynomialRing<IntegerRing>> {
    /// Compute the exact coefficient contraction and factor its generator with
    /// explicit deterministic work bounds.
    pub fn arithmetic_fiber_support(
        &self,
        trial_bound: u64,
        primality_proof_bits: u64,
    ) -> Result<ArithmeticFiberSupport, ArithmeticIdealError> {
        let support_ideal = self.contract_to_coefficients()?;
        if support_ideal.principal_generator().is_zero() {
            return Err(ArithmeticIdealError::GenericFiberNonempty);
        }
        ArithmeticFiberSupport::exact(
            self.clone(),
            support_ideal,
            trial_bound,
            primality_proof_bits,
        )
    }
}

impl fmt::Debug for ArithmeticFiberSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArithmeticFiberSupport")
            .field("claim", &self.claim)
            .field("generator", &self.support_ideal.principal_generator())
            .field("prime_support_complete", &self.prime_support_is_complete())
            .finish()
    }
}

impl fmt::Display for ArithmeticFiberSupport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ArithmeticFiberSupport(generator={}, claim={}, prime_support_complete={})",
            self.support_ideal.principal_generator(),
            self.claim,
            self.prime_support_is_complete()
        )
    }
}

impl SemanticType for ArithmeticFiberSupport {
    const TYPE_ID: &'static str = "noether/commalg/ArithmeticFiberSupport@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &FIBER_SUPPORT_DESCRIPTION
    }
}

impl Serialize for ArithmeticFiberSupport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ArithmeticFiberSupport", 7)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("generators", &self.generators)?;
        state.serialize_field("support_ideal", &self.support_ideal)?;
        state.serialize_field("claim", &self.claim)?;
        state.serialize_field("certificate", &self.certificate)?;
        state.serialize_field("factorization", &self.factorization)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArithmeticFiberSupportWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<IntegerRing>,
    generators: Vec<Polynomial<IntegerRing>>,
    support_ideal: Ideal<IntegerRing>,
    claim: ArithmeticFiberSupportClaim,
    certificate: Option<serde_json::Value>,
    factorization: serde_json::Value,
}
