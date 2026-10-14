// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite birational normalizations of explicitly presented affine domains.

use core::fmt;
use std::collections::BTreeMap;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    ContextMismatchError, Field, IdentityCoefficientMap, Integer, NoetherError, RationalField,
    Ring, SemanticType, TypeDescription,
};

use super::{
    DecompositionError, GroebnerError, Ideal, MonomialOrder, Polynomial, PolynomialContextError,
    PolynomialError, PolynomialHomomorphismError, PolynomialPrimeCertificate, PolynomialRing,
    PolynomialRingHomomorphism, QuotientRing, QuotientRingMap, QuotientRingMapError,
};
use crate::numthy::FiniteField;

static AFFINE_NORMALIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/AffineNormalization@1",
    rust_path: "noether::commalg::AffineNormalization",
    python_path: Some("noether.commalg.AffineNormalization"),
};

static GRAUERT_REMMERT_STEP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GrauertRemmertStep@1",
    rust_path: "noether::commalg::GrauertRemmertStep",
    python_path: Some("noether.commalg.GrauertRemmertStep"),
};
const GRAUERT_REMMERT_STEP_TYPE_ID: &str = "noether/commalg/GrauertRemmertStep@1";

const MAX_CONDUCTOR_MODULE_GENERATORS: usize = 1_000_000;

/// The checked quotient map retained by an affine normalization.
pub type AffineNormalizationMap<R> = QuotientRingMap<
    PolynomialRing<R>,
    PolynomialRing<R>,
    PolynomialRingHomomorphism<R, R, IdentityCoefficientMap<R>>,
>;

/// The checked quotient map retained by one Grauert--Remmert end-ring step.
pub type GrauertRemmertStepMap<R> = AffineNormalizationMap<R>;

/// Failure to discover, construct, or verify an affine normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineNormalizationError {
    /// The defining ideal was not certified prime.
    NotADomain,
    /// The domain is outside the currently admitted native discovery routes.
    Unsupported(String),
    /// A required exponent exceeded the bounded exact route.
    ResourceLimit {
        /// Name of the bounded resource.
        resource: &'static str,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// Prime-ideal discovery or verification failed.
    Decomposition(DecompositionError),
    /// Polynomial-context construction failed.
    PolynomialContext(PolynomialContextError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Canonical ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Ambient polynomial-map construction failed.
    Homomorphism(PolynomialHomomorphismError),
    /// Quotient-map construction failed.
    QuotientMap(QuotientRingMapError),
    /// Exact context identity was violated.
    ContextMismatch(ContextMismatchError),
    /// Retained evidence failed exact replay.
    InvalidEvidence(String),
}

impl fmt::Display for AffineNormalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotADomain => formatter
                .write_str("affine normalization requires a defining ideal certified as prime"),
            Self::Unsupported(reason) => {
                write!(
                    formatter,
                    "unsupported native affine normalization: {reason}"
                )
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "affine normalization exceeded its {resource} limit of {limit}"
            ),
            Self::Decomposition(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Homomorphism(error) => error.fmt(formatter),
            Self::QuotientMap(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid affine-normalization evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for AffineNormalizationError {}

macro_rules! error_conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for AffineNormalizationError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}

error_conversion!(DecompositionError, Decomposition);
error_conversion!(PolynomialContextError, PolynomialContext);
error_conversion!(PolynomialError, Polynomial);
error_conversion!(GroebnerError, Groebner);
error_conversion!(PolynomialHomomorphismError, Homomorphism);
error_conversion!(QuotientRingMapError, QuotientMap);
error_conversion!(ContextMismatchError, ContextMismatch);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum AffineNormalizationProof {
    PolynomialGraph {
        pivot_variables: Vec<usize>,
        free_variables: Vec<usize>,
    },
    CoprimePlaneMonomialCurve {
        first_variable: usize,
        second_variable: usize,
        first_relation_power: u64,
        second_relation_power: u64,
        conductor_exponent: u64,
    },
    SmoothAffineModel,
}

/// One verified finite birational Grauert--Remmert end-ring extension.
///
/// For a radical Jacobian test ideal `J` and a nonzero `g` in `J`, the
/// retained numerator `u` lies in `(gJ:J)` but not in `(g)`. Thus `u/g` is a
/// nontrivial element of `End_A(J)`. The target is the saturated presentation
/// obtained by adjoining that fraction. It need not yet be normal; its
/// defining ideal can be used as the source of the next exact step.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "R: Serialize, R::Element: Serialize, Ideal<PolynomialRing<R>>: Serialize, PolynomialPrimeCertificate<R>: Serialize, QuotientRing<PolynomialRing<R>>: Serialize, GrauertRemmertStepMap<R>: Serialize, Polynomial<R>: Serialize"
    )
)]
pub struct GrauertRemmertStep<R: Field + fmt::Display> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    test_ideal: Ideal<PolynomialRing<R>>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    target_algebra: QuotientRing<PolynomialRing<R>>,
    target_prime: PolynomialPrimeCertificate<R>,
    structure_map: GrauertRemmertStepMap<R>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
}

impl<R: Field + fmt::Display> GrauertRemmertStep<R> {
    /// Defining prime ideal of the source domain.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }
    /// Exact primality evidence for the source.
    pub fn source_prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.source_prime
    }
    /// Radical Jacobian test ideal `J`.
    pub fn test_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.test_ideal
    }
    /// Numerator `u` of the adjoined endomorphism `u/g`.
    pub fn numerator(&self) -> &Polynomial<R> {
        &self.numerator
    }
    /// Nonzero denominator `g` of the adjoined endomorphism `u/g`.
    pub fn denominator(&self) -> &Polynomial<R> {
        &self.denominator
    }
    /// Source coordinate algebra.
    pub fn source_algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.source_algebra
    }
    /// Finite birational target coordinate algebra.
    pub fn target_algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.target_algebra
    }
    /// Defining prime ideal of the target presentation.
    pub fn target(&self) -> &Ideal<PolynomialRing<R>> {
        self.target_algebra.defining_ideal()
    }
    /// Exact primality evidence for the target.
    pub fn target_prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.target_prime
    }
    /// Checked finite birational map from source to target.
    pub fn structure_map(&self) -> &GrauertRemmertStepMap<R> {
        &self.structure_map
    }
    /// Images of the source coordinates in the target presentation.
    pub fn coordinate_images(&self) -> &[Polynomial<R>] {
        self.structure_map.base_map().images()
    }
    /// Monic equations for the target ambient generators over the source.
    pub fn integral_equations(&self) -> &[Vec<Polynomial<R>>] {
        &self.integral_equations
    }
    /// Fraction numerators for target ambient generators in `Frac(A)`.
    pub fn fraction_numerators(&self) -> &[Polynomial<R>] {
        &self.fraction_numerators
    }
    /// Fraction denominators for target ambient generators in `Frac(A)`.
    pub fn fraction_denominators(&self) -> &[Polynomial<R>] {
        &self.fraction_denominators
    }
    /// Stable proof-route name.
    pub const fn proof_kind(&self) -> &'static str {
        "grauert_remmert_end_ring"
    }
    /// Exact theorem certified by this step.
    pub const fn theorem(&self) -> &'static str {
        "u in (gJ:J) makes u/g integral in End_A(J); the saturated prime presentation, monic equations, and fraction identities prove a finite birational extension"
    }
}

impl<R: Field + fmt::Display> fmt::Debug for GrauertRemmertStep<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrauertRemmertStep")
            .field("source_variables", &self.source.ring().variables().len())
            .field(
                "target_variables",
                &self.target_algebra.base_ring().variables().len(),
            )
            .finish_non_exhaustive()
    }
}

impl<R: Field + fmt::Display> fmt::Display for GrauertRemmertStep<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GrauertRemmertStep(source_variables={}, target_variables={})",
            self.source.ring().variables().len(),
            self.target_algebra.base_ring().variables().len()
        )
    }
}

/// A verified finite birational map from an affine domain to a normal algebra.
///
/// The source is `A = k[x]/I`. The retained normal algebra `B` is an explicit
/// polynomial quotient, and `structure_map` is a checked map `A -> B`.
/// For each ambient generator of `B`, a monic equation over `A` proves
/// integrality and a numerator/denominator pair proves that the same generator
/// lies in `Frac(A)`. A prime certificate for `I`, equality of dimensions, and
/// finiteness prove injectivity; the fraction witnesses prove birationality.
/// Native discovery includes smooth affine complete intersections and a first
/// Grauert--Remmert end-ring step for singular curves. A separately supplied
/// smooth affine complete-intersection model is admitted by the same exact
/// finite/birational/conductor checks.
#[derive(Clone, PartialEq, Eq)]
pub struct AffineNormalization<R: Field + fmt::Display> {
    source: Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    normal_algebra: QuotientRing<PolynomialRing<R>>,
    normal_prime: Option<PolynomialPrimeCertificate<R>>,
    normal_jacobian_minors: Vec<Polynomial<R>>,
    normal_singular_locus: Option<Ideal<PolynomialRing<R>>>,
    structure_map: AffineNormalizationMap<R>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
    source_conductor: Ideal<PolynomialRing<R>>,
    normal_conductor: Ideal<PolynomialRing<R>>,
    nonnormal_locus: Ideal<PolynomialRing<R>>,
    proof: AffineNormalizationProof,
}

impl<R: Field + fmt::Display> AffineNormalization<R> {
    /// Defining prime ideal of the source affine domain.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Exact primality evidence for the source ideal.
    pub fn source_prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.source_prime
    }

    /// Source coordinate algebra `A`.
    pub fn source_algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.source_algebra
    }

    /// Integral closure `B` in the source fraction field.
    pub fn normal_algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.normal_algebra
    }

    /// Prime certificate for a non-polynomial normal target model.
    pub fn normal_prime(&self) -> Option<&PolynomialPrimeCertificate<R>> {
        self.normal_prime.as_ref()
    }

    /// Expected-codimension Jacobian minors proving a supplied target smooth.
    pub fn normal_jacobian_minors(&self) -> &[Polynomial<R>] {
        &self.normal_jacobian_minors
    }

    /// Empty singular-locus ideal of a supplied smooth target model.
    pub fn normal_singular_locus(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        self.normal_singular_locus.as_ref()
    }

    /// Checked finite birational structure map `A -> B`.
    pub fn structure_map(&self) -> &AffineNormalizationMap<R> {
        &self.structure_map
    }

    /// Images of the source ambient coordinates in the normal algebra.
    pub fn coordinate_images(&self) -> &[Polynomial<R>] {
        self.structure_map.base_map().images()
    }

    /// Monic equations for the normal ambient generators.
    ///
    /// Each inner vector contains coefficients in increasing degree and omits
    /// the leading coefficient `1`.
    pub fn integral_equations(&self) -> &[Vec<Polynomial<R>>] {
        &self.integral_equations
    }

    /// Fraction numerators for the normal ambient generators.
    pub fn fraction_numerators(&self) -> &[Polynomial<R>] {
        &self.fraction_numerators
    }

    /// Nonzero fraction denominators for the normal ambient generators.
    pub fn fraction_denominators(&self) -> &[Polynomial<R>] {
        &self.fraction_denominators
    }

    /// Preimage in the source polynomial presentation of the conductor ideal.
    pub fn source_conductor(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source_conductor
    }

    /// Conductor ideal in the normal polynomial presentation.
    pub fn normal_conductor(&self) -> &Ideal<PolynomialRing<R>> {
        &self.normal_conductor
    }

    /// Ideal defining the source nonnormal locus.
    pub fn nonnormal_locus(&self) -> &Ideal<PolynomialRing<R>> {
        &self.nonnormal_locus
    }

    /// Whether the structure map is an isomorphism by the retained conductor.
    pub fn is_normal(&self) -> bool {
        self.source_conductor
            .generators()
            .first()
            .is_some_and(Polynomial::is_one)
    }

    /// Canonical name of the exact discovery/proof route.
    pub const fn proof_kind(&self) -> &'static str {
        match self.proof {
            AffineNormalizationProof::PolynomialGraph { .. } => "polynomial_graph",
            AffineNormalizationProof::CoprimePlaneMonomialCurve { .. } => {
                "coprime_plane_monomial_curve"
            }
            AffineNormalizationProof::SmoothAffineModel => "smooth_affine_model",
        }
    }

    /// Exact theorem certified by this value.
    pub const fn theorem(&self) -> &'static str {
        "monic equations prove finiteness, exact prime and dimension evidence proves injectivity, fraction witnesses prove birationality, and the target is certified normal"
    }

    /// Exact conductor theorem used by the retained route.
    pub const fn conductor_theorem(&self) -> &'static str {
        match self.proof {
            AffineNormalizationProof::PolynomialGraph { .. } => {
                "an isomorphism has unit conductor and empty nonnormal locus"
            }
            AffineNormalizationProof::CoprimePlaneMonomialCurve { .. } => {
                "the conductor of k[t^m,t^n] in k[t] is t^((m-1)(n-1))k[t] for coprime m and n"
            }
            AffineNormalizationProof::SmoothAffineModel => {
                "fraction monomials bounded by the retained monic equations span the finite target over the source; exact ideal quotients compute the source elements carrying that whole span into the source"
            }
        }
    }
}

impl<R: Field + fmt::Display> fmt::Debug for AffineNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AffineNormalization")
            .field("proof", &self.proof_kind())
            .field("source_variables", &self.source.ring().variables().len())
            .field(
                "normal_variables",
                &self.normal_algebra.base_ring().variables().len(),
            )
            .field("is_normal", &self.is_normal())
            .finish_non_exhaustive()
    }
}

impl<R: Field + fmt::Display> fmt::Display for AffineNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "AffineNormalization(proof={}, normal_variables={})",
            self.proof_kind(),
            self.normal_algebra.base_ring().variables().len()
        )
    }
}

fn unique_prime_certificate_q(
    source: &Ideal<PolynomialRing<RationalField>>,
) -> Result<PolynomialPrimeCertificate<RationalField>, AffineNormalizationError> {
    let decomposition = source.minimal_primes()?;
    decomposition.verify()?;
    if decomposition.components().len() != 1 || decomposition.components()[0] != *source {
        return Err(AffineNormalizationError::NotADomain);
    }
    Ok(decomposition.prime_certificates()[0].clone())
}

fn unique_prime_certificate_finite(
    source: &Ideal<PolynomialRing<FiniteField>>,
) -> Result<PolynomialPrimeCertificate<FiniteField>, AffineNormalizationError> {
    let decomposition = source.minimal_primes()?;
    decomposition.verify()?;
    if decomposition.components().len() != 1 || decomposition.components()[0] != *source {
        return Err(AffineNormalizationError::NotADomain);
    }
    Ok(decomposition.prime_certificates()[0].clone())
}

pub(super) fn construct_affine_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
) -> Result<AffineNormalization<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    if source_prime.ideal() != source {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the source prime certificate belongs to another ideal".to_owned(),
        ));
    }
    if let Some((pivots, presentation)) = discover_polynomial_graph(source)? {
        return polynomial_graph_normalization(source, source_prime, pivots, &presentation);
    }
    if let Some(data) = coprime_plane_monomial_curve(source)? {
        return monomial_curve_normalization(source, source_prime, data);
    }
    if smooth_affine_complete_intersection_data(source)?.is_some() {
        let coordinates = source.ring().generators()?;
        let integral_equations = coordinates
            .iter()
            .map(|coordinate| source.ring().neg(coordinate).map(|value| vec![value]))
            .collect::<Result<Vec<_>, _>>()?;
        let denominators = vec![source.ring().one(); coordinates.len()];
        return construct_smooth_affine_model(
            source,
            source_prime.clone(),
            source.clone(),
            source_prime,
            coordinates.clone(),
            integral_equations,
            coordinates,
            denominators,
        );
    }
    Err(AffineNormalizationError::Unsupported(
        "the native floor covers smooth affine complete intersections, polynomial graphs, and coprime plane monomial curves".to_owned(),
    ))
}

fn singular_test_ideal<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Ideal<PolynomialRing<R>>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    if source.dimension()? != 1 {
        return Ok(None);
    }
    let codimension = source
        .ring()
        .variables()
        .len()
        .checked_sub(1)
        .ok_or_else(|| {
            AffineNormalizationError::InvalidEvidence(
                "a one-dimensional affine domain has no ambient variables".to_owned(),
            )
        })?;
    let jacobian_minors =
        super::jacobian::jacobian_minors(source, codimension).map_err(map_jacobian_error)?;
    let singular = source
        .ring()
        .ideal(source.generators().iter().cloned().chain(jacobian_minors))?;
    Ok((!singular.contains(&source.ring().one())?).then_some(singular))
}

struct FractionTargetPresentation<R: Field> {
    ideal: Ideal<PolynomialRing<R>>,
    coordinate_images: Vec<Polynomial<R>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
    variable_sources: Vec<Option<usize>>,
    adjoined_integral_equation: Option<Vec<Polynomial<R>>>,
}

type PrimeCertifier<R> = fn(
    &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimeCertificate<R>, AffineNormalizationError>;

fn construct_grauert_remmert_step<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: &PolynomialPrimeCertificate<R>,
    test_ideal: &Ideal<PolynomialRing<R>>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
    certify_prime: PrimeCertifier<R>,
) -> Result<Option<GrauertRemmertStep<R>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    if source_prime.ideal() != source
        || test_ideal.ring() != source.ring()
        || numerator.ring() != source.ring()
        || denominator.ring() != source.ring()
    {
        return Err(AffineNormalizationError::InvalidEvidence(
            "a Grauert--Remmert witness crosses polynomial contexts".to_owned(),
        ));
    }
    if !test_ideal.contains(&denominator)? || source.contains(&denominator)? {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the Grauert--Remmert denominator is not a nonzero test-ideal element".to_owned(),
        ));
    }
    let denominator_ideal = source.ring().ideal([denominator.clone()])?;
    let end_numerators = test_ideal
        .product(&denominator_ideal)?
        .sum(source)?
        .colon(test_ideal)?;
    let trivial_numerators = denominator_ideal.sum(source)?;
    if !end_numerators.contains(&numerator)? || trivial_numerators.contains(&numerator)? {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the retained fraction is not a nontrivial element of End_A(J)".to_owned(),
        ));
    }
    let Some(presentation) =
        fraction_target_presentation(source, numerator.clone(), denominator.clone())?
    else {
        return Ok(None);
    };
    let target_prime = certify_prime(&presentation.ideal)?;
    let Some(integral_equations) = fraction_integral_equations(source, &presentation)? else {
        return Ok(None);
    };
    if source.dimension()? != presentation.ideal.dimension()? {
        return Err(AffineNormalizationError::InvalidEvidence(
            "a Grauert--Remmert finite extension changed dimension".to_owned(),
        ));
    }
    let source_algebra = source.ring().quotient(source)?;
    let target_ring = presentation.ideal.ring().clone();
    let target_algebra = target_ring.quotient(&presentation.ideal)?;
    let ambient_map = source.ring().homomorphism(
        &target_ring,
        IdentityCoefficientMap::new(source.ring().coefficient_ring().clone()),
        presentation.coordinate_images,
    )?;
    let structure_map = source_algebra.induced_map(&target_algebra, ambient_map)?;
    Ok(Some(GrauertRemmertStep {
        type_id: GRAUERT_REMMERT_STEP_TYPE_ID.to_owned(),
        source: source.clone(),
        source_prime: source_prime.clone(),
        test_ideal: test_ideal.clone(),
        numerator,
        denominator,
        source_algebra,
        target_algebra,
        target_prime,
        structure_map,
        integral_equations,
        fraction_numerators: presentation.fraction_numerators,
        fraction_denominators: presentation.fraction_denominators,
    }))
}

fn discover_grauert_remmert_step<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: &PolynomialPrimeCertificate<R>,
    test_ideal: &Ideal<PolynomialRing<R>>,
    certify_prime: PrimeCertifier<R>,
) -> Result<Option<GrauertRemmertStep<R>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    for denominator in test_ideal.generators() {
        if source.contains(denominator)? {
            continue;
        }
        let denominator_ideal = source.ring().ideal([denominator.clone()])?;
        let end_numerators = test_ideal
            .product(&denominator_ideal)?
            .sum(source)?
            .colon(test_ideal)?;
        let trivial_numerators = denominator_ideal.sum(source)?;
        for numerator in end_numerators.generators() {
            if trivial_numerators.contains(numerator)? {
                continue;
            }
            match construct_grauert_remmert_step(
                source,
                source_prime,
                test_ideal,
                numerator.clone(),
                denominator.clone(),
                certify_prime,
            ) {
                Ok(Some(step)) => return Ok(Some(step)),
                Ok(None) | Err(AffineNormalizationError::NotADomain) => continue,
                Err(AffineNormalizationError::Decomposition(_)) => continue,
                Err(error) => return Err(error),
            }
        }
    }
    Ok(None)
}

fn one_step_smooth_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: &PolynomialPrimeCertificate<R>,
    test_ideal: &Ideal<PolynomialRing<R>>,
    certify_prime: PrimeCertifier<R>,
) -> Result<Option<AffineNormalization<R>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let source_ring = source.ring();
    for denominator in test_ideal.generators() {
        if source.contains(denominator)? {
            continue;
        }
        let denominator_ideal = source_ring.ideal([denominator.clone()])?;
        let end_numerators = test_ideal
            .product(&denominator_ideal)?
            .sum(source)?
            .colon(test_ideal)?;
        let trivial_numerators = denominator_ideal.sum(source)?;
        for numerator in end_numerators.generators() {
            if trivial_numerators.contains(numerator)? {
                continue;
            }
            let Some(presentation) =
                fraction_target_presentation(source, numerator.clone(), denominator.clone())?
            else {
                continue;
            };
            let Ok(normal_prime) = certify_prime(&presentation.ideal) else {
                continue;
            };
            if smooth_affine_complete_intersection_data(&presentation.ideal)?.is_none() {
                continue;
            }
            let Some(integral_equations) = fraction_integral_equations(source, &presentation)?
            else {
                continue;
            };
            let result = construct_smooth_affine_model(
                source,
                source_prime.clone(),
                presentation.ideal,
                normal_prime,
                presentation.coordinate_images,
                integral_equations,
                presentation.fraction_numerators,
                presentation.fraction_denominators,
            )?;
            return Ok(Some(result));
        }
    }
    Ok(None)
}

fn fraction_target_presentation<R>(
    source: &Ideal<PolynomialRing<R>>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
) -> Result<Option<FractionTargetPresentation<R>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let source_ring = source.ring();
    let mut variable_names = source_ring.variables().to_vec();
    let mut fraction_name = "_gr0".to_owned();
    while variable_names.contains(&fraction_name) {
        fraction_name.push('_');
    }
    variable_names.push(fraction_name);
    let target_ring = PolynomialRing::new(
        source_ring.coefficient_ring().clone(),
        variable_names,
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let target_variables = target_ring.generators()?;
    let source_arity = source_ring.variables().len();
    let embedding = source_ring.homomorphism(
        &target_ring,
        IdentityCoefficientMap::new(source_ring.coefficient_ring().clone()),
        target_variables[..source_arity].to_vec(),
    )?;
    let embedded_denominator = embedding.apply(&denominator)?;
    let embedded_numerator = embedding.apply(&numerator)?;
    let fraction_relation = target_ring.sub(
        &target_ring.mul(&embedded_denominator, &target_variables[source_arity])?,
        &embedded_numerator,
    )?;
    let raw_ideal = target_ring.ideal(
        source
            .generators()
            .iter()
            .map(|generator| embedding.apply(generator))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .chain([fraction_relation]),
    )?;
    let denominator_ideal = target_ring.ideal([embedded_denominator])?;
    let ideal = raw_ideal.saturate(&denominator_ideal)?;
    let mut presentation = FractionTargetPresentation {
        ideal,
        coordinate_images: target_variables[..source_arity].to_vec(),
        fraction_numerators: source_ring
            .generators()?
            .into_iter()
            .chain([numerator])
            .collect(),
        fraction_denominators: (0..source_arity)
            .map(|_| source_ring.one())
            .chain([denominator])
            .collect(),
        variable_sources: (0..source_arity).map(Some).chain([None]).collect(),
        adjoined_integral_equation: None,
    };
    presentation.adjoined_integral_equation = presentation
        .ideal
        .generators()
        .iter()
        .find_map(|relation| {
            integral_equation_from_relation(source, &presentation, source_arity, relation)
                .transpose()
        })
        .transpose()?;
    if presentation.adjoined_integral_equation.is_none() {
        return Ok(None);
    }
    while eliminate_one_graph_variable(&mut presentation)? {}
    Ok(Some(presentation))
}

fn eliminate_one_graph_variable<R>(
    presentation: &mut FractionTargetPresentation<R>,
) -> Result<bool, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let old_ring = presentation.ideal.ring();
    let Some((pivot, pivot_image)) = presentation
        .ideal
        .generators()
        .iter()
        .find_map(|relation| linear_graph_pivot(relation, &presentation.variable_sources))
        .transpose()?
    else {
        return Ok(false);
    };
    let new_ring = pivot_image.ring().clone();
    let new_variables = new_ring.generators()?;
    let mut old_images = Vec::with_capacity(old_ring.variables().len());
    let mut next = 0;
    for variable in 0..old_ring.variables().len() {
        if variable == pivot {
            old_images.push(pivot_image.clone());
        } else {
            old_images.push(new_variables[next].clone());
            next += 1;
        }
    }
    let elimination_map = old_ring.homomorphism(
        &new_ring,
        IdentityCoefficientMap::new(old_ring.coefficient_ring().clone()),
        old_images,
    )?;
    let new_ideal = new_ring.ideal(
        presentation
            .ideal
            .generators()
            .iter()
            .map(|generator| elimination_map.apply(generator))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    presentation.coordinate_images = presentation
        .coordinate_images
        .iter()
        .map(|image| elimination_map.apply(image))
        .collect::<Result<Vec<_>, _>>()?;
    presentation.ideal = new_ideal;
    presentation.fraction_numerators.remove(pivot);
    presentation.fraction_denominators.remove(pivot);
    presentation.variable_sources.remove(pivot);
    Ok(true)
}

fn linear_graph_pivot<R>(
    relation: &Polynomial<R>,
    variable_sources: &[Option<usize>],
) -> Option<Result<(usize, Polynomial<R>), AffineNormalizationError>>
where
    R: Field + fmt::Display,
{
    let old_ring = relation.ring();
    for pivot in 0..old_ring.variables().len() {
        if variable_sources[pivot].is_none() {
            continue;
        }
        let mut pivot_coefficient = None;
        let mut remaining_terms = Vec::new();
        let mut valid = true;
        for (exponent, coefficient) in relation.terms() {
            if exponent[pivot].is_zero() {
                remaining_terms.push((exponent.to_vec(), coefficient.clone()));
            } else if exponent[pivot].is_one()
                && exponent
                    .iter()
                    .enumerate()
                    .all(|(index, power)| index == pivot || power.is_zero())
                && pivot_coefficient.is_none()
            {
                pivot_coefficient = Some(coefficient.clone());
            } else {
                valid = false;
                break;
            }
        }
        let Some(pivot_coefficient) = pivot_coefficient.filter(|_| valid) else {
            continue;
        };
        return Some((|| {
            let new_ring = PolynomialRing::new(
                old_ring.coefficient_ring().clone(),
                old_ring
                    .variables()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != pivot)
                    .map(|(_, name)| name.clone()),
                old_ring.term_order().without_variables(&[pivot]),
            )?;
            let inverse = old_ring
                .coefficient_ring()
                .inverse(&pivot_coefficient)
                .map_err(|_| {
                    AffineNormalizationError::InvalidEvidence(
                        "a graph pivot coefficient is not invertible".to_owned(),
                    )
                })?;
            let negative_inverse = old_ring.coefficient_ring().neg(&inverse)?;
            let pivot_image = new_ring.from_terms(
                remaining_terms
                    .into_iter()
                    .map(|(mut exponent, coefficient)| {
                        exponent.remove(pivot);
                        old_ring
                            .coefficient_ring()
                            .mul(&coefficient, &negative_inverse)
                            .map(|coefficient| (exponent, coefficient))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            Ok((pivot, pivot_image))
        })());
    }
    None
}

fn fraction_integral_equations<R>(
    source: &Ideal<PolynomialRing<R>>,
    presentation: &FractionTargetPresentation<R>,
) -> Result<Option<Vec<Vec<Polynomial<R>>>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let mut equations = Vec::with_capacity(presentation.variable_sources.len());
    for variable in 0..presentation.variable_sources.len() {
        if presentation.fraction_denominators[variable].is_one() {
            equations.push(vec![source
                .ring()
                .neg(&presentation.fraction_numerators[variable])?]);
            continue;
        }
        if presentation.variable_sources[variable].is_none() {
            let Some(equation) = presentation.adjoined_integral_equation.clone() else {
                return Ok(None);
            };
            equations.push(equation);
            continue;
        }
        let Some(equation) = presentation
            .ideal
            .generators()
            .iter()
            .find_map(|relation| {
                integral_equation_from_relation(source, presentation, variable, relation)
                    .transpose()
            })
            .transpose()?
        else {
            return Ok(None);
        };
        equations.push(equation);
    }
    Ok(Some(equations))
}

fn integral_equation_from_relation<R>(
    source: &Ideal<PolynomialRing<R>>,
    presentation: &FractionTargetPresentation<R>,
    integral_variable: usize,
    relation: &Polynomial<R>,
) -> Result<Option<Vec<Polynomial<R>>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let mut grouped = BTreeMap::<usize, Vec<(Vec<Integer>, R::Element)>>::new();
    for (target_exponent, coefficient) in relation.terms() {
        let Some(power) = target_exponent[integral_variable]
            .to_u64_checked()
            .and_then(|power| usize::try_from(power).ok())
        else {
            return Ok(None);
        };
        let mut source_exponent = vec![Integer::zero(); source.ring().variables().len()];
        for (target_variable, target_power) in target_exponent.iter().enumerate() {
            if target_variable == integral_variable {
                continue;
            }
            let Some(source_variable) = presentation.variable_sources[target_variable] else {
                return Ok(None);
            };
            source_exponent[source_variable] += target_power;
        }
        grouped
            .entry(power)
            .or_default()
            .push((source_exponent, coefficient.clone()));
    }
    let Some((&degree, leading_terms)) = grouped.last_key_value() else {
        return Ok(None);
    };
    if degree == 0 || leading_terms.len() != 1 || leading_terms[0].0.iter().any(|e| !e.is_zero()) {
        return Ok(None);
    }
    let inverse_leading = source
        .ring()
        .coefficient_ring()
        .inverse(&leading_terms[0].1)
        .map_err(|_| {
            AffineNormalizationError::InvalidEvidence(
                "an integral relation has a noninvertible leading coefficient".to_owned(),
            )
        })?;
    let coefficients = (0..degree)
        .map(|power| {
            source
                .ring()
                .from_terms(grouped.remove(&power).unwrap_or_default())?
                .scalar_mul(&inverse_leading)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, AffineNormalizationError>>()?;
    Ok(Some(coefficients))
}

type PolynomialGraphDiscovery<R> = Option<(Vec<usize>, Ideal<PolynomialRing<R>>)>;

fn discover_polynomial_graph<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialGraphDiscovery<R>, AffineNormalizationError> {
    let dimension = source.dimension()?;
    if dimension < 0 {
        return Ok(None);
    }
    let arity = source.ring().variables().len();
    let dimension = usize::try_from(dimension).map_err(|_| {
        AffineNormalizationError::InvalidEvidence(
            "the source dimension does not fit its polynomial arity".to_owned(),
        )
    })?;
    let pivot_count = arity.checked_sub(dimension).ok_or_else(|| {
        AffineNormalizationError::InvalidEvidence(
            "the source dimension exceeds its polynomial arity".to_owned(),
        )
    })?;
    for pivots in index_combinations(arity, pivot_count) {
        let free = (0..arity)
            .filter(|index| !pivots.contains(index))
            .collect::<Vec<_>>();
        let order = if pivots.is_empty() {
            MonomialOrder::GradedReverseLexicographic
        } else if free.is_empty() {
            MonomialOrder::Lexicographic
        } else {
            MonomialOrder::block([
                (pivots.clone(), MonomialOrder::Lexicographic),
                (free, MonomialOrder::GradedReverseLexicographic),
            ])
            .map_err(|error| {
                AffineNormalizationError::InvalidEvidence(format!(
                    "polynomial-graph block order failed: {error}"
                ))
            })?
        };
        let presentation = source.change_order(order)?;
        if super::radical::polynomial_graph_pivots(&presentation).as_ref() == Some(&pivots) {
            return Ok(Some((pivots, presentation)));
        }
    }
    Ok(None)
}

fn index_combinations(arity: usize, count: usize) -> Vec<Vec<usize>> {
    fn recurse(
        next: usize,
        arity: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(current.clone());
            return;
        }
        if next >= arity || arity - next < remaining {
            return;
        }
        for index in next..=arity - remaining {
            current.push(index);
            recurse(index + 1, arity, remaining - 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    recurse(0, arity, count, &mut Vec::new(), &mut output);
    output
}

fn polynomial_graph_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    pivot_variables: Vec<usize>,
    graph_presentation: &Ideal<PolynomialRing<R>>,
) -> Result<AffineNormalization<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let arity = source.ring().variables().len();
    let free_variables = (0..arity)
        .filter(|index| !pivot_variables.contains(index))
        .collect::<Vec<_>>();
    let normal_ring = PolynomialRing::new(
        source.ring().coefficient_ring().clone(),
        free_variables
            .iter()
            .map(|index| source.ring().variables()[*index].clone()),
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let normal_coordinates = normal_ring.generators()?;
    let mut coordinate_images = vec![normal_ring.zero(); arity];
    for (target_index, &source_index) in free_variables.iter().enumerate() {
        coordinate_images[source_index] = normal_coordinates[target_index].clone();
    }
    for relation in graph_presentation.generators() {
        let mut terms = relation.terms();
        let Some((leading_exponent, leading_coefficient)) = terms.next() else {
            return Err(AffineNormalizationError::InvalidEvidence(
                "a polynomial-graph relation is zero".to_owned(),
            ));
        };
        let Some(pivot) = leading_exponent.iter().position(Integer::is_one) else {
            return Err(AffineNormalizationError::InvalidEvidence(
                "a polynomial-graph relation has no pivot".to_owned(),
            ));
        };
        let inverse_leading = source
            .ring()
            .coefficient_ring()
            .inverse(leading_coefficient)
            .map_err(|_| {
                AffineNormalizationError::InvalidEvidence(
                    "a polynomial-graph leading coefficient is not invertible".to_owned(),
                )
            })?;
        let image_terms = terms
            .map(|(exponent, coefficient)| {
                let exponent = free_variables
                    .iter()
                    .map(|index| exponent[*index].clone())
                    .collect::<Vec<_>>();
                let scaled = source
                    .ring()
                    .coefficient_ring()
                    .mul(coefficient, &inverse_leading)?;
                let coefficient = source.ring().coefficient_ring().neg(&scaled)?;
                Ok((exponent, coefficient))
            })
            .collect::<Result<Vec<_>, ContextMismatchError>>()?;
        coordinate_images[pivot] = normal_ring.from_terms(image_terms)?;
    }
    let source_coordinates = source.ring().generators()?;
    let integral_equations = free_variables
        .iter()
        .map(|index| {
            source
                .ring()
                .neg(&source_coordinates[*index])
                .map(|value| vec![value])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fraction_numerators = free_variables
        .iter()
        .map(|index| source_coordinates[*index].clone())
        .collect::<Vec<_>>();
    let fraction_denominators = vec![source.ring().one(); free_variables.len()];
    let source_conductor = source.ring().ideal([source.ring().one()])?;
    let normal_ideal = normal_ring.ideal(Vec::<Polynomial<R>>::new())?;
    assemble_affine_normalization(
        source,
        source_prime,
        normal_ring,
        normal_ideal,
        None,
        Vec::new(),
        None,
        coordinate_images,
        integral_equations,
        fraction_numerators,
        fraction_denominators,
        source_conductor.clone(),
        source_conductor,
        AffineNormalizationProof::PolynomialGraph {
            pivot_variables,
            free_variables,
        },
    )
}

#[derive(Clone, Copy)]
struct MonomialCurveData {
    first_variable: usize,
    second_variable: usize,
    first_relation_power: u64,
    second_relation_power: u64,
}

fn coprime_plane_monomial_curve<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<MonomialCurveData>, AffineNormalizationError> {
    if source.ring().variables().len() != 2 || source.generators().len() != 1 {
        return Ok(None);
    }
    let relation = &source.generators()[0];
    if relation.nonzero_term_count() != 2 {
        return Ok(None);
    }
    let terms = relation.terms().collect::<Vec<_>>();
    let Some((first_variable, first_power)) = pure_power_term(terms[0].0) else {
        return Ok(None);
    };
    let Some((second_variable, second_power)) = pure_power_term(terms[1].0) else {
        return Ok(None);
    };
    if first_variable == second_variable || first_power < 2 || second_power < 2 {
        return Ok(None);
    }
    let coefficient_sum = source
        .ring()
        .coefficient_ring()
        .add(terms[0].1, terms[1].1)?;
    if coefficient_sum != source.ring().coefficient_ring().zero()
        || gcd_u64(first_power, second_power) != 1
    {
        return Ok(None);
    }
    Ok(Some(MonomialCurveData {
        first_variable,
        second_variable,
        first_relation_power: first_power,
        second_relation_power: second_power,
    }))
}

fn pure_power_term(exponent: &[Integer]) -> Option<(usize, u64)> {
    let mut support = exponent
        .iter()
        .enumerate()
        .filter(|(_, power)| !power.is_zero());
    let (variable, power) = support.next()?;
    if support.next().is_some() {
        return None;
    }
    Some((variable, power.to_string().parse().ok()?))
}

fn monomial_curve_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    data: MonomialCurveData,
) -> Result<AffineNormalization<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let normal_ring = PolynomialRing::new(
        source.ring().coefficient_ring().clone(),
        ["t"],
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let parameter = normal_ring.generator(0)?;
    let first_image_power = data.second_relation_power;
    let second_image_power = data.first_relation_power;
    let mut coordinate_images = vec![normal_ring.zero(); 2];
    coordinate_images[data.first_variable] = parameter.pow(first_image_power)?;
    coordinate_images[data.second_variable] = parameter.pow(second_image_power)?;

    let source_coordinates = source.ring().generators()?;
    let equation_degree = usize::try_from(first_image_power).map_err(|_| {
        AffineNormalizationError::ResourceLimit {
            resource: "integral-equation degree",
            limit: usize::MAX,
        }
    })?;
    let mut equation = vec![source.ring().zero(); equation_degree];
    equation[0] = source
        .ring()
        .neg(&source_coordinates[data.first_variable])?;
    let (bezout_first, bezout_second) = extended_gcd_u64(first_image_power, second_image_power);
    let positive_first = nonnegative_i128_to_u64(bezout_first.max(0))?;
    let positive_second = nonnegative_i128_to_u64(bezout_second.max(0))?;
    let negative_first = nonnegative_i128_to_u64((-bezout_first).max(0))?;
    let negative_second = nonnegative_i128_to_u64((-bezout_second).max(0))?;
    let numerator = signed_monomial(
        source.ring(),
        &source_coordinates,
        data.first_variable,
        positive_first,
        data.second_variable,
        positive_second,
    )?;
    let denominator = signed_monomial(
        source.ring(),
        &source_coordinates,
        data.first_variable,
        negative_first,
        data.second_variable,
        negative_second,
    )?;

    let conductor_exponent = data
        .first_relation_power
        .checked_sub(1)
        .and_then(|left| {
            data.second_relation_power
                .checked_sub(1)
                .and_then(|right| left.checked_mul(right))
        })
        .ok_or(AffineNormalizationError::ResourceLimit {
            resource: "monomial-curve conductor exponent",
            limit: usize::MAX,
        })?;
    let multiplicity = first_image_power.min(second_image_power);
    let conductor_stop = conductor_exponent.checked_add(multiplicity).ok_or(
        AffineNormalizationError::ResourceLimit {
            resource: "monomial-curve conductor generators",
            limit: usize::MAX,
        },
    )?;
    let conductor_lifts = (conductor_exponent..conductor_stop)
        .map(|exponent| {
            semigroup_monomial(
                source.ring(),
                &source_coordinates,
                data.first_variable,
                first_image_power,
                data.second_variable,
                second_image_power,
                exponent,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source_conductor = source
        .ring()
        .ideal(source.generators().iter().cloned().chain(conductor_lifts))?;
    let normal_ideal = normal_ring.ideal(Vec::<Polynomial<R>>::new())?;
    assemble_affine_normalization(
        source,
        source_prime,
        normal_ring,
        normal_ideal,
        None,
        Vec::new(),
        None,
        coordinate_images,
        vec![equation],
        vec![numerator],
        vec![denominator],
        source_conductor.clone(),
        source_conductor,
        AffineNormalizationProof::CoprimePlaneMonomialCurve {
            first_variable: data.first_variable,
            second_variable: data.second_variable,
            first_relation_power: data.first_relation_power,
            second_relation_power: data.second_relation_power,
            conductor_exponent,
        },
    )
}

fn map_jacobian_error(error: super::jacobian::JacobianError) -> AffineNormalizationError {
    match error {
        super::jacobian::JacobianError::Polynomial(error) => {
            AffineNormalizationError::Polynomial(error)
        }
        super::jacobian::JacobianError::ContextMismatch(error) => {
            AffineNormalizationError::ContextMismatch(error)
        }
        super::jacobian::JacobianError::ResourceLimit { resource, limit } => {
            AffineNormalizationError::ResourceLimit { resource, limit }
        }
        super::jacobian::JacobianError::Invalid(reason) => {
            AffineNormalizationError::InvalidEvidence(reason.to_owned())
        }
    }
}

type SmoothAffineData<R> = Option<(Vec<Polynomial<R>>, Ideal<PolynomialRing<R>>)>;

fn smooth_affine_complete_intersection_data<R>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<SmoothAffineData<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let ring = ideal.ring();
    let dimension = ideal.dimension()?;
    if dimension < 0 {
        return Ok(None);
    }
    let ambient_dimension = i64::try_from(ring.variables().len()).map_err(|_| {
        AffineNormalizationError::ResourceLimit {
            resource: "normal-model ambient variable count",
            limit: usize::MAX,
        }
    })?;
    let codimension = usize::try_from(ambient_dimension - dimension).map_err(|_| {
        AffineNormalizationError::InvalidEvidence(
            "the normal model has impossible dimension".to_owned(),
        )
    })?;
    if ideal.generators().len() != codimension {
        return Ok(None);
    }
    let jacobian_minors =
        super::jacobian::jacobian_minors(ideal, codimension).map_err(map_jacobian_error)?;
    let singular_locus = ring.ideal(
        ideal
            .generators()
            .iter()
            .cloned()
            .chain(jacobian_minors.iter().cloned()),
    )?;
    Ok(singular_locus
        .contains(&ring.one())?
        .then_some((jacobian_minors, singular_locus)))
}

#[allow(clippy::too_many_arguments)]
fn construct_smooth_affine_model<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    normal_ideal: Ideal<PolynomialRing<R>>,
    normal_prime: PolynomialPrimeCertificate<R>,
    coordinate_images: Vec<Polynomial<R>>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
) -> Result<AffineNormalization<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    if normal_prime.ideal() != &normal_ideal {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the normal-model prime certificate belongs to another ideal".to_owned(),
        ));
    }
    let normal_ring = normal_ideal.ring().clone();
    let Some((normal_jacobian_minors, normal_singular_locus)) =
        smooth_affine_complete_intersection_data(&normal_ideal)?
    else {
        return Err(AffineNormalizationError::Unsupported(
            "the supplied target is not a smooth affine complete intersection".to_owned(),
        ));
    };

    let separator = source.ring().ideal([source.ring().one()])?;
    let source_conductor = finite_fraction_conductor_preimage(
        source,
        &separator,
        &integral_equations,
        &fraction_numerators,
        &fraction_denominators,
    )?;
    assemble_affine_normalization(
        source,
        source_prime,
        normal_ring,
        normal_ideal,
        Some(normal_prime),
        normal_jacobian_minors,
        Some(normal_singular_locus),
        coordinate_images,
        integral_equations,
        fraction_numerators,
        fraction_denominators,
        source_conductor.clone(),
        source_conductor,
        AffineNormalizationProof::SmoothAffineModel,
    )
}

/// Pull back the largest target ideal contained in a source ideal using the
/// rectangular finite-module span supplied by monic equations and fractions.
pub(super) fn finite_fraction_conductor_preimage<R>(
    source_prime: &Ideal<PolynomialRing<R>>,
    separator: &Ideal<PolynomialRing<R>>,
    equations: &[Vec<Polynomial<R>>],
    numerators: &[Polynomial<R>],
    denominators: &[Polynomial<R>],
) -> Result<Ideal<PolynomialRing<R>>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    if equations.len() != numerators.len() || equations.len() != denominators.len() {
        return Err(AffineNormalizationError::InvalidEvidence(
            "integral equations and fraction witnesses have different lengths".to_owned(),
        ));
    }
    let requested = equations.iter().try_fold(1_usize, |product, equation| {
        product.checked_mul(equation.len())
    });
    let Some(requested) = requested else {
        return Err(AffineNormalizationError::ResourceLimit {
            resource: "conductor finite-module generators",
            limit: MAX_CONDUCTOR_MODULE_GENERATORS,
        });
    };
    if requested == 0 {
        return Err(AffineNormalizationError::InvalidEvidence(
            "an integral equation has degree zero".to_owned(),
        ));
    }
    if requested > MAX_CONDUCTOR_MODULE_GENERATORS {
        return Err(AffineNormalizationError::ResourceLimit {
            resource: "conductor finite-module generators",
            limit: MAX_CONDUCTOR_MODULE_GENERATORS,
        });
    }

    let ring = source_prime.ring();
    let mut fractions = vec![(ring.one(), ring.one())];
    for ((equation, numerator), denominator) in equations.iter().zip(numerators).zip(denominators) {
        let mut numerator_power = ring.one();
        let mut denominator_power = ring.one();
        let mut expanded = Vec::with_capacity(fractions.len() * equation.len());
        for _ in 0..equation.len() {
            for (old_numerator, old_denominator) in &fractions {
                expanded.push((
                    ring.mul(old_numerator, &numerator_power)?,
                    ring.mul(old_denominator, &denominator_power)?,
                ));
            }
            numerator_power = ring.mul(&numerator_power, numerator)?;
            denominator_power = ring.mul(&denominator_power, denominator)?;
        }
        fractions = expanded;
    }

    let mut condition = ring.ideal([ring.one()])?;
    for (numerator, denominator) in fractions {
        let denominator_ideal = ring.ideal([denominator])?;
        let required_numerators = separator.product(&denominator_ideal)?.sum(source_prime)?;
        let numerator_ideal = ring.ideal([numerator])?;
        let quotient = required_numerators.colon(&numerator_ideal)?;
        condition = condition.intersection(&quotient)?;
    }
    Ok(condition)
}

#[allow(clippy::too_many_arguments)]
fn assemble_affine_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    normal_ring: PolynomialRing<R>,
    normal_ideal: Ideal<PolynomialRing<R>>,
    normal_prime: Option<PolynomialPrimeCertificate<R>>,
    normal_jacobian_minors: Vec<Polynomial<R>>,
    normal_singular_locus: Option<Ideal<PolynomialRing<R>>>,
    coordinate_images: Vec<Polynomial<R>>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
    source_conductor: Ideal<PolynomialRing<R>>,
    nonnormal_locus: Ideal<PolynomialRing<R>>,
    proof: AffineNormalizationProof,
) -> Result<AffineNormalization<R>, AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    let source_algebra = source.ring().quotient(source)?;
    let normal_algebra = normal_ring.quotient(&normal_ideal)?;
    let ambient_map = source.ring().homomorphism(
        &normal_ring,
        IdentityCoefficientMap::new(source.ring().coefficient_ring().clone()),
        coordinate_images,
    )?;
    let normal_conductor_generators = source_conductor
        .generators()
        .iter()
        .map(|generator| ambient_map.apply(generator))
        .collect::<Result<Vec<_>, _>>()?;
    let normal_conductor = normal_ring
        .ideal(normal_conductor_generators)?
        .sum(&normal_ideal)?;
    let structure_map = source_algebra.induced_map(&normal_algebra, ambient_map)?;
    Ok(AffineNormalization {
        source: source.clone(),
        source_prime,
        source_algebra,
        normal_algebra,
        normal_prime,
        normal_jacobian_minors,
        normal_singular_locus,
        structure_map,
        integral_equations,
        fraction_numerators,
        fraction_denominators,
        source_conductor,
        normal_conductor,
        nonnormal_locus,
        proof,
    })
}

fn signed_monomial<R: Field>(
    ring: &PolynomialRing<R>,
    coordinates: &[Polynomial<R>],
    first_variable: usize,
    first_power: u64,
    second_variable: usize,
    second_power: u64,
) -> Result<Polynomial<R>, AffineNormalizationError> {
    let first = coordinates[first_variable].pow(first_power)?;
    let second = coordinates[second_variable].pow(second_power)?;
    Ok(ring.mul(&first, &second)?)
}

fn semigroup_monomial<R: Field>(
    ring: &PolynomialRing<R>,
    coordinates: &[Polynomial<R>],
    first_variable: usize,
    first_weight: u64,
    second_variable: usize,
    second_weight: u64,
    target: u64,
) -> Result<Polynomial<R>, AffineNormalizationError> {
    for first_power in 0..=target / first_weight {
        let remainder = target - first_power * first_weight;
        if remainder.is_multiple_of(second_weight) {
            return signed_monomial(
                ring,
                coordinates,
                first_variable,
                first_power,
                second_variable,
                remainder / second_weight,
            );
        }
    }
    Err(AffineNormalizationError::InvalidEvidence(
        "a numerical-semigroup conductor exponent has no source monomial".to_owned(),
    ))
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn extended_gcd_u64(left: u64, right: u64) -> (i128, i128) {
    let (mut old_r, mut r) = (i128::from(left), i128::from(right));
    let (mut old_s, mut s) = (1_i128, 0_i128);
    let (mut old_t, mut t) = (0_i128, 1_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
        (old_t, t) = (t, old_t - quotient * t);
    }
    (old_s, old_t)
}

fn nonnegative_i128_to_u64(value: i128) -> Result<u64, AffineNormalizationError> {
    u64::try_from(value).map_err(|_| AffineNormalizationError::ResourceLimit {
        resource: "monomial-curve Bezout coefficient",
        limit: usize::MAX,
    })
}

fn verify_grauert_remmert_step<R>(
    value: &GrauertRemmertStep<R>,
    certify_prime: PrimeCertifier<R>,
) -> Result<(), AffineNormalizationError>
where
    R: super::minimal_primes::MinimalPrimeCoefficientField + fmt::Display,
{
    if value.type_id != GRAUERT_REMMERT_STEP_TYPE_ID {
        return Err(AffineNormalizationError::InvalidEvidence(
            "a Grauert--Remmert step has the wrong type identity".to_owned(),
        ));
    }
    let singular = singular_test_ideal(&value.source)?.ok_or_else(|| {
        AffineNormalizationError::InvalidEvidence(
            "a Grauert--Remmert step has a nonsingular source".to_owned(),
        )
    })?;
    let radical = super::radical::construct_radical(&singular)?;
    if radical.ideal() != &value.test_ideal {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the retained test ideal is not the radical Jacobian ideal".to_owned(),
        ));
    }
    let recomputed = construct_grauert_remmert_step(
        &value.source,
        &value.source_prime,
        &value.test_ideal,
        value.numerator.clone(),
        value.denominator.clone(),
        certify_prime,
    )?
    .ok_or_else(|| {
        AffineNormalizationError::InvalidEvidence(
            "the retained end-ring witness no longer produces an integral target".to_owned(),
        )
    })?;
    if &recomputed != value {
        return Err(AffineNormalizationError::InvalidEvidence(
            "recomputation changed the Grauert--Remmert extension".to_owned(),
        ));
    }
    Ok(())
}

fn verify_common<R>(value: &AffineNormalization<R>) -> Result<(), AffineNormalizationError>
where
    R: Field + fmt::Display,
{
    value.source.verify().map_err(|error| {
        AffineNormalizationError::InvalidEvidence(format!("source ideal failed: {error}"))
    })?;
    if value.source_prime.ideal() != &value.source
        || value.source_algebra.defining_ideal() != &value.source
        || value.structure_map.source() != &value.source_algebra
        || value.structure_map.target() != &value.normal_algebra
    {
        return Err(AffineNormalizationError::InvalidEvidence(
            "the source, target, prime certificate, and structure-map endpoints disagree"
                .to_owned(),
        ));
    }
    match value.proof {
        AffineNormalizationProof::PolynomialGraph { .. }
        | AffineNormalizationProof::CoprimePlaneMonomialCurve { .. } => {
            if !value
                .normal_algebra
                .defining_ideal()
                .generators()
                .is_empty()
                || value.normal_prime.is_some()
                || !value.normal_jacobian_minors.is_empty()
                || value.normal_singular_locus.is_some()
            {
                return Err(AffineNormalizationError::InvalidEvidence(
                    "a polynomial normal target retained incompatible smooth-model evidence"
                        .to_owned(),
                ));
            }
        }
        AffineNormalizationProof::SmoothAffineModel => {
            if value
                .normal_prime
                .as_ref()
                .map(PolynomialPrimeCertificate::ideal)
                != Some(value.normal_algebra.defining_ideal())
                || value.normal_singular_locus.is_none()
            {
                return Err(AffineNormalizationError::InvalidEvidence(
                    "a smooth affine model omitted its target prime or Jacobian evidence"
                        .to_owned(),
                ));
            }
        }
    }
    value.structure_map.verify().map_err(|error| {
        AffineNormalizationError::InvalidEvidence(format!("structure map failed: {error}"))
    })?;
    let target_coordinates = value.normal_algebra.base_ring().generators()?;
    let target_elements = target_coordinates
        .iter()
        .map(|coordinate| value.normal_algebra.element(coordinate))
        .collect::<Result<Vec<_>, _>>()?;
    if value.integral_equations.len() != target_elements.len()
        || value.fraction_numerators.len() != target_elements.len()
        || value.fraction_denominators.len() != target_elements.len()
    {
        return Err(AffineNormalizationError::InvalidEvidence(
            "integral and birational evidence has the wrong target arity".to_owned(),
        ));
    }
    for (index, target) in target_elements.iter().enumerate() {
        let mut equation_value = target.pow(value.integral_equations[index].len() as u64);
        let mut power = value.normal_algebra.one();
        for coefficient in &value.integral_equations[index] {
            let source_coefficient = value.source_algebra.element(coefficient)?;
            let image = value.structure_map.apply(&source_coefficient)?;
            equation_value = value
                .normal_algebra
                .add(&equation_value, &value.normal_algebra.mul(&image, &power)?)?;
            power = value.normal_algebra.mul(&power, target)?;
        }
        if !equation_value.is_zero() {
            return Err(AffineNormalizationError::InvalidEvidence(format!(
                "target generator {index} does not satisfy its monic equation"
            )));
        }
        let numerator = value
            .source_algebra
            .element(&value.fraction_numerators[index])?;
        let denominator = value
            .source_algebra
            .element(&value.fraction_denominators[index])?;
        if denominator.is_zero() {
            return Err(AffineNormalizationError::InvalidEvidence(format!(
                "target generator {index} has a zero fraction denominator"
            )));
        }
        let numerator_image = value.structure_map.apply(&numerator)?;
        let denominator_image = value.structure_map.apply(&denominator)?;
        if value.normal_algebra.mul(&denominator_image, target)? != numerator_image {
            return Err(AffineNormalizationError::InvalidEvidence(format!(
                "target generator {index} fails its birational fraction identity"
            )));
        }
    }
    if value.source.dimension()? != value.normal_algebra.dimension()? {
        return Err(AffineNormalizationError::InvalidEvidence(
            "finite source and target dimensions differ".to_owned(),
        ));
    }
    let recomputed = match value.proof {
        AffineNormalizationProof::SmoothAffineModel => construct_smooth_affine_model(
            &value.source,
            value.source_prime.clone(),
            value.normal_algebra.defining_ideal().clone(),
            value.normal_prime.clone().ok_or_else(|| {
                AffineNormalizationError::InvalidEvidence(
                    "a smooth affine model has no target prime certificate".to_owned(),
                )
            })?,
            value.coordinate_images().to_vec(),
            value.integral_equations.clone(),
            value.fraction_numerators.clone(),
            value.fraction_denominators.clone(),
        )?,
        AffineNormalizationProof::PolynomialGraph { .. }
        | AffineNormalizationProof::CoprimePlaneMonomialCurve { .. } => {
            construct_affine_normalization(&value.source, value.source_prime.clone())?
        }
    };
    if &recomputed != value {
        return Err(AffineNormalizationError::InvalidEvidence(
            "recomputation changed the normalization or conductor evidence".to_owned(),
        ));
    }
    Ok(())
}

macro_rules! concrete_methods {
    ($coefficient:ty, $prime:ident) => {
        impl Ideal<PolynomialRing<$coefficient>> {
            /// Compute one verified Grauert--Remmert end-ring extension.
            ///
            /// A smooth source returns `None`. A singular source yields a
            /// finite birational target that can be used as the source of a
            /// subsequent step; failure to find an admissible target is an
            /// explicit unsupported error.
            pub fn grauert_remmert_step(
                &self,
            ) -> Result<Option<GrauertRemmertStep<$coefficient>>, AffineNormalizationError> {
                let prime = $prime(self)?;
                let Some(singular) = singular_test_ideal(self)? else {
                    return Ok(None);
                };
                let radical = super::radical::construct_radical(&singular)?;
                let step = discover_grauert_remmert_step(
                    self,
                    &prime,
                    radical.ideal(),
                    $prime,
                )?
                .ok_or_else(|| {
                    AffineNormalizationError::Unsupported(
                        "the bounded Grauert--Remmert search found no admissible nontrivial end-ring generator"
                            .to_owned(),
                    )
                })?;
                step.verify()?;
                Ok(Some(step))
            }

            /// Compute a verified finite birational normalization when a native route applies.
            pub fn normalization(
                &self,
            ) -> Result<AffineNormalization<$coefficient>, AffineNormalizationError> {
                let prime = $prime(self)?;
                let result = match construct_affine_normalization(self, prime.clone()) {
                    Ok(result) => result,
                    Err(AffineNormalizationError::Unsupported(_)) => {
                        let Some(singular) = singular_test_ideal(self)? else {
                            return Err(AffineNormalizationError::Unsupported(
                                "the native discovery routes did not produce a normal target"
                                    .to_owned(),
                            ));
                        };
                        let radical = super::radical::construct_radical(&singular)?;
                        one_step_smooth_normalization(
                            self,
                            &prime,
                            radical.ideal(),
                            $prime,
                        )?
                        .ok_or_else(|| {
                            AffineNormalizationError::Unsupported(
                                "one Grauert--Remmert end-ring step did not produce a smooth affine complete-intersection target".to_owned(),
                            )
                        })?
                    }
                    Err(error) => return Err(error),
                };
                result.verify()?;
                Ok(result)
            }

            /// Admit a supplied smooth affine complete-intersection model as
            /// the normalization after exact finite, birational, normality,
            /// and conductor checks.
            pub fn normalization_from_smooth_model(
                &self,
                normal_ideal: Ideal<PolynomialRing<$coefficient>>,
                coordinate_images: Vec<Polynomial<$coefficient>>,
                integral_equations: Vec<Vec<Polynomial<$coefficient>>>,
                fraction_numerators: Vec<Polynomial<$coefficient>>,
                fraction_denominators: Vec<Polynomial<$coefficient>>,
            ) -> Result<AffineNormalization<$coefficient>, AffineNormalizationError> {
                let source_prime = $prime(self)?;
                let normal_prime = $prime(&normal_ideal)?;
                let result = construct_smooth_affine_model(
                    self,
                    source_prime,
                    normal_ideal,
                    normal_prime,
                    coordinate_images,
                    integral_equations,
                    fraction_numerators,
                    fraction_denominators,
                )?;
                result.verify()?;
                Ok(result)
            }
        }

        impl AffineNormalization<$coefficient> {
            /// Replay primality, finiteness, birationality, normality, and conductor evidence.
            pub fn verify(&self) -> Result<(), AffineNormalizationError> {
                self.source_prime.verify()?;
                if let Some(normal_prime) = &self.normal_prime {
                    normal_prime.verify()?;
                }
                verify_common(self)
            }

            /// Deserialize and exactly reverify an affine normalization.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl GrauertRemmertStep<$coefficient> {
            /// Replay the test ideal, end-ring witness, prime presentations,
            /// finite map, monic equations, and fraction identities.
            pub fn verify(&self) -> Result<(), AffineNormalizationError> {
                self.source_prime.verify()?;
                self.target_prime.verify()?;
                verify_grauert_remmert_step(self, $prime)
            }

            /// Deserialize and exactly reverify one end-ring step.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

concrete_methods!(RationalField, unique_prime_certificate_q);
concrete_methods!(FiniteField, unique_prime_certificate_finite);

impl<R> GrauertRemmertStep<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    GrauertRemmertStepMap<R>: Serialize,
    Polynomial<R>: Serialize,
{
    /// Serialize the complete end-ring extension evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> SemanticType for GrauertRemmertStep<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    GrauertRemmertStepMap<R>: Serialize,
    Polynomial<R>: Serialize,
{
    const TYPE_ID: &'static str = GRAUERT_REMMERT_STEP_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &GRAUERT_REMMERT_STEP_DESCRIPTION
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + fmt::Display + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>, GrauertRemmertStepMap<R>: Deserialize<'de>, Polynomial<R>: Deserialize<'de>"
    )
)]
struct GrauertRemmertStepWire<R: Field + fmt::Display> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    test_ideal: Ideal<PolynomialRing<R>>,
    numerator: Polynomial<R>,
    denominator: Polynomial<R>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    target_algebra: QuotientRing<PolynomialRing<R>>,
    target_prime: PolynomialPrimeCertificate<R>,
    structure_map: GrauertRemmertStepMap<R>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
}

fn deserialize_grauert_remmert_step<'de, R, D>(
    deserializer: D,
) -> Result<GrauertRemmertStep<R>, D::Error>
where
    R: Field + fmt::Display + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Serialize + Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    GrauertRemmertStepMap<R>: Serialize + Deserialize<'de>,
    Polynomial<R>: Serialize + Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = GrauertRemmertStepWire::<R>::deserialize(deserializer)?;
    if wire.type_id != GRAUERT_REMMERT_STEP_TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            GRAUERT_REMMERT_STEP_TYPE_ID, wire.type_id
        )));
    }
    Ok(GrauertRemmertStep {
        type_id: wire.type_id,
        source: wire.source,
        source_prime: wire.source_prime,
        test_ideal: wire.test_ideal,
        numerator: wire.numerator,
        denominator: wire.denominator,
        source_algebra: wire.source_algebra,
        target_algebra: wire.target_algebra,
        target_prime: wire.target_prime,
        structure_map: wire.structure_map,
        integral_equations: wire.integral_equations,
        fraction_numerators: wire.fraction_numerators,
        fraction_denominators: wire.fraction_denominators,
    })
}

impl<'de> Deserialize<'de> for GrauertRemmertStep<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: GrauertRemmertStep<RationalField> =
            deserialize_grauert_remmert_step(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for GrauertRemmertStep<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: GrauertRemmertStep<FiniteField> =
            deserialize_grauert_remmert_step(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R> AffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    Polynomial<R>: Serialize,
{
    /// Serialize the complete finite birational map and conductor evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> SemanticType for AffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    Polynomial<R>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/AffineNormalization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &AFFINE_NORMALIZATION_DESCRIPTION
    }
}

impl<R> Serialize for AffineNormalization<R>
where
    R: Field + fmt::Display + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    AffineNormalizationMap<R>: Serialize,
    Polynomial<R>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AffineNormalization", 16)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("source_prime", &self.source_prime)?;
        state.serialize_field("source_algebra", &self.source_algebra)?;
        state.serialize_field("normal_algebra", &self.normal_algebra)?;
        state.serialize_field("normal_prime", &self.normal_prime)?;
        state.serialize_field("normal_jacobian_minors", &self.normal_jacobian_minors)?;
        state.serialize_field("normal_singular_locus", &self.normal_singular_locus)?;
        state.serialize_field("structure_map", &self.structure_map)?;
        state.serialize_field("integral_equations", &self.integral_equations)?;
        state.serialize_field("fraction_numerators", &self.fraction_numerators)?;
        state.serialize_field("fraction_denominators", &self.fraction_denominators)?;
        state.serialize_field("source_conductor", &self.source_conductor)?;
        state.serialize_field("normal_conductor", &self.normal_conductor)?;
        state.serialize_field("nonnormal_locus", &self.nonnormal_locus)?;
        state.serialize_field("proof", &self.proof)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + fmt::Display + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>, AffineNormalizationMap<R>: Deserialize<'de>, Polynomial<R>: Deserialize<'de>"
    )
)]
struct AffineNormalizationWire<R: Field + fmt::Display> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    source_prime: PolynomialPrimeCertificate<R>,
    source_algebra: QuotientRing<PolynomialRing<R>>,
    normal_algebra: QuotientRing<PolynomialRing<R>>,
    normal_prime: Option<PolynomialPrimeCertificate<R>>,
    normal_jacobian_minors: Vec<Polynomial<R>>,
    normal_singular_locus: Option<Ideal<PolynomialRing<R>>>,
    structure_map: AffineNormalizationMap<R>,
    integral_equations: Vec<Vec<Polynomial<R>>>,
    fraction_numerators: Vec<Polynomial<R>>,
    fraction_denominators: Vec<Polynomial<R>>,
    source_conductor: Ideal<PolynomialRing<R>>,
    normal_conductor: Ideal<PolynomialRing<R>>,
    nonnormal_locus: Ideal<PolynomialRing<R>>,
    proof: AffineNormalizationProof,
}

fn deserialize_affine_normalization<'de, R, D>(
    deserializer: D,
) -> Result<AffineNormalization<R>, D::Error>
where
    R: Field + fmt::Display + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Serialize + Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    AffineNormalizationMap<R>: Serialize + Deserialize<'de>,
    Polynomial<R>: Serialize + Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = AffineNormalizationWire::<R>::deserialize(deserializer)?;
    if wire.type_id != AffineNormalization::<R>::TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            AffineNormalization::<R>::TYPE_ID,
            wire.type_id
        )));
    }
    Ok(AffineNormalization {
        source: wire.source,
        source_prime: wire.source_prime,
        source_algebra: wire.source_algebra,
        normal_algebra: wire.normal_algebra,
        normal_prime: wire.normal_prime,
        normal_jacobian_minors: wire.normal_jacobian_minors,
        normal_singular_locus: wire.normal_singular_locus,
        structure_map: wire.structure_map,
        integral_equations: wire.integral_equations,
        fraction_numerators: wire.fraction_numerators,
        fraction_denominators: wire.fraction_denominators,
        source_conductor: wire.source_conductor,
        normal_conductor: wire.normal_conductor,
        nonnormal_locus: wire.nonnormal_locus,
        proof: wire.proof,
    })
}

impl<'de> Deserialize<'de> for AffineNormalization<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: AffineNormalization<RationalField> =
            deserialize_affine_normalization(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de> Deserialize<'de> for AffineNormalization<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let result: AffineNormalization<FiniteField> =
            deserialize_affine_normalization(deserializer)?;
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q;

    #[test]
    fn cusp_normalization_retains_integral_fraction_and_conductor_evidence() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(&variables[1].pow(2).unwrap(), &variables[0].pow(3).unwrap())
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();

        assert_eq!(normalization.proof_kind(), "coprime_plane_monomial_curve");
        assert!(!normalization.is_normal());
        assert_eq!(normalization.coordinate_images().len(), 2);
        let parameter = normalization
            .normal_algebra()
            .base_ring()
            .generator(0)
            .unwrap();
        assert_eq!(
            normalization.coordinate_images(),
            &[parameter.pow(2).unwrap(), parameter.pow(3).unwrap()]
        );
        assert_eq!(normalization.integral_equations().len(), 1);
        assert_eq!(
            normalization.integral_equations(),
            &[vec![ring.neg(&variables[0]).unwrap(), ring.zero()]]
        );
        assert_eq!(normalization.fraction_numerators(), &[variables[1].clone()]);
        assert_eq!(
            normalization.fraction_denominators(),
            &[variables[0].clone()]
        );
        assert_eq!(
            normalization.source_conductor(),
            &ring
                .ideal([variables[0].clone(), variables[1].clone()])
                .unwrap()
        );
        assert_eq!(
            normalization.normal_conductor(),
            &normalization
                .normal_algebra()
                .base_ring()
                .ideal([parameter.pow(2).unwrap()])
                .unwrap()
        );
        assert_eq!(normalization.source_conductor().generators().len(), 2);
        assert_eq!(
            normalization.nonnormal_locus(),
            normalization.source_conductor()
        );
        normalization.verify().unwrap();
        assert_eq!(
            AffineNormalization::<RationalField>::from_json(&normalization.to_json().unwrap())
                .unwrap(),
            normalization
        );
        let mut tampered: serde_json::Value =
            serde_json::from_str(&normalization.to_json().unwrap()).unwrap();
        tampered["proof"]["conductor_exponent"] = serde_json::Value::from(3_u64);
        assert!(AffineNormalization::<RationalField>::from_json(&tampered.to_string()).is_err());
    }

    #[test]
    fn polynomial_graph_normalization_is_an_isomorphism() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(&variables[1], &variables[0].pow(2).unwrap())
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();

        assert_eq!(normalization.proof_kind(), "polynomial_graph");
        assert!(normalization.is_normal());
        assert_eq!(
            normalization.normal_algebra().base_ring().variables(),
            &["x"]
        );
        normalization.verify().unwrap();
    }

    #[test]
    fn supplied_smooth_affine_model_normalizes_a_nonrational_curve() {
        let source_ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let source_variables = source_ring.generators().unwrap();
        let x = &source_variables[0];
        let y = &source_variables[1];
        let source_equation = source_ring
            .add(
                &source_ring
                    .sub(&y.pow(2).unwrap(), &x.pow(5).unwrap())
                    .unwrap(),
                &x.pow(3).unwrap(),
            )
            .unwrap();
        let source = source_ring.ideal([source_equation]).unwrap();

        let normal_ring = PolynomialRing::new(Q, ["u", "v"], MonomialOrder::default()).unwrap();
        let normal_variables = normal_ring.generators().unwrap();
        let u = &normal_variables[0];
        let v = &normal_variables[1];
        let normal_equation = normal_ring
            .add(
                &normal_ring
                    .sub(&v.pow(2).unwrap(), &u.pow(3).unwrap())
                    .unwrap(),
                u,
            )
            .unwrap();
        let normal_ideal = normal_ring.ideal([normal_equation]).unwrap();
        let normalization = source
            .normalization_from_smooth_model(
                normal_ideal,
                vec![u.clone(), normal_ring.mul(u, v).unwrap()],
                vec![
                    vec![source_ring.neg(x).unwrap()],
                    vec![
                        source_ring.sub(x, &x.pow(3).unwrap()).unwrap(),
                        source_ring.zero(),
                    ],
                ],
                vec![x.clone(), y.clone()],
                vec![source_ring.one(), x.clone()],
            )
            .unwrap();

        assert_eq!(normalization.proof_kind(), "smooth_affine_model");
        assert!(normalization.normal_prime().is_some());
        assert_eq!(normalization.normal_jacobian_minors().len(), 2);
        assert!(normalization
            .normal_singular_locus()
            .unwrap()
            .contains(&normal_ring.one())
            .unwrap());
        assert_eq!(
            normalization.source_conductor(),
            &source_ring.ideal([x.clone(), y.clone()]).unwrap()
        );
        normalization.verify().unwrap();
        assert_eq!(
            AffineNormalization::<RationalField>::from_json(&normalization.to_json().unwrap())
                .unwrap(),
            normalization
        );
    }

    #[test]
    fn one_end_ring_step_discovers_a_nonrational_normalization() {
        let source_ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = source_ring.generators().unwrap();
        let x = &variables[0];
        let y = &variables[1];
        let source = source_ring
            .ideal([source_ring
                .add(
                    &source_ring
                        .sub(&y.pow(2).unwrap(), &x.pow(5).unwrap())
                        .unwrap(),
                    &x.pow(3).unwrap(),
                )
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();

        assert_eq!(normalization.proof_kind(), "smooth_affine_model");
        assert_eq!(
            normalization.normal_algebra().base_ring().variables(),
            &["x", "_gr0"]
        );
        assert_eq!(normalization.coordinate_images().len(), 2);
        assert_eq!(normalization.fraction_numerators(), &[x.clone(), y.clone()]);
        assert_eq!(
            normalization.fraction_denominators(),
            &[source_ring.one(), x.clone()]
        );
        assert_eq!(
            normalization.source_conductor(),
            &source_ring.ideal([x.clone(), y.clone()]).unwrap()
        );
        normalization.verify().unwrap();
    }

    #[test]
    fn one_end_ring_step_can_simplify_to_a_polynomial_target() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let x = &variables[0];
        let y = &variables[1];
        let source = ring
            .ideal([ring
                .sub(
                    &y.pow(2).unwrap(),
                    &ring
                        .mul(&x.pow(2).unwrap(), &ring.add(x, &ring.one()).unwrap())
                        .unwrap(),
                )
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();

        assert_eq!(
            normalization.normal_algebra().base_ring().variables(),
            &["_gr0"]
        );
        assert!(normalization
            .normal_algebra()
            .defining_ideal()
            .generators()
            .is_empty());
        assert_eq!(normalization.fraction_numerators(), std::slice::from_ref(y));
        assert_eq!(
            normalization.fraction_denominators(),
            std::slice::from_ref(x)
        );
        assert_eq!(
            normalization.source_conductor(),
            &ring.ideal([x.clone(), y.clone()]).unwrap()
        );
        normalization.verify().unwrap();
    }

    #[test]
    fn one_end_ring_step_is_coefficient_generic() {
        let field = FiniteField::prime(5_u8).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let x = &variables[0];
        let y = &variables[1];
        let source = ring
            .ideal([ring
                .add(
                    &ring.sub(&y.pow(2).unwrap(), &x.pow(5).unwrap()).unwrap(),
                    &x.pow(3).unwrap(),
                )
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();

        assert_eq!(normalization.proof_kind(), "smooth_affine_model");
        assert_eq!(
            normalization.source_conductor(),
            &ring.ideal([x.clone(), y.clone()]).unwrap()
        );
        normalization.verify().unwrap();
    }

    #[test]
    fn reducible_and_unsupported_domains_fail_closed() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let reducible = ring
            .ideal([ring.mul(&variables[0], &variables[1]).unwrap()])
            .unwrap();
        assert!(matches!(
            reducible.normalization(),
            Err(AffineNormalizationError::NotADomain)
        ));

        let surface_ring =
            PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let surface_variables = surface_ring.generators().unwrap();
        let normal_singular_surface = surface_ring
            .ideal([surface_ring
                .sub(
                    &surface_variables[2].pow(2).unwrap(),
                    &surface_ring
                        .mul(&surface_variables[0], &surface_variables[1])
                        .unwrap(),
                )
                .unwrap()])
            .unwrap();
        assert!(matches!(
            normal_singular_surface.normalization(),
            Err(AffineNormalizationError::Unsupported(_))
        ));
    }

    #[test]
    fn smooth_affine_complete_intersection_is_its_own_normalization() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let elliptic = ring
            .ideal([ring
                .sub(
                    &variables[1].pow(2).unwrap(),
                    &ring
                        .add(&variables[0].pow(3).unwrap(), &variables[0])
                        .unwrap(),
                )
                .unwrap()])
            .unwrap();

        let normalization = elliptic.normalization().unwrap();

        assert_eq!(normalization.proof_kind(), "smooth_affine_model");
        assert!(normalization.is_normal());
        assert_eq!(normalization.normal_algebra().defining_ideal(), &elliptic);
        assert_eq!(
            normalization.source_conductor(),
            &ring.ideal([ring.one()]).unwrap()
        );
        normalization.verify().unwrap();
    }

    #[test]
    fn arbitrary_coprime_monomial_curves_work_over_finite_fields() {
        let field = FiniteField::prime(5_u8).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(&variables[1].pow(3).unwrap(), &variables[0].pow(4).unwrap())
                .unwrap()])
            .unwrap();

        let normalization = source.normalization().unwrap();
        let parameter = normalization
            .normal_algebra()
            .base_ring()
            .generator(0)
            .unwrap();

        assert_eq!(
            normalization.coordinate_images(),
            &[parameter.pow(3).unwrap(), parameter.pow(4).unwrap()]
        );
        assert_eq!(
            normalization.source_conductor(),
            &ring
                .ideal([
                    variables[0].pow(2).unwrap(),
                    ring.mul(&variables[0], &variables[1]).unwrap(),
                    variables[1].pow(2).unwrap(),
                ])
                .unwrap()
        );
        assert_eq!(
            normalization.normal_conductor(),
            &normalization
                .normal_algebra()
                .base_ring()
                .ideal([parameter.pow(6).unwrap()])
                .unwrap()
        );
        normalization.verify().unwrap();
        assert_eq!(
            AffineNormalization::<FiniteField>::from_json(&normalization.to_json().unwrap())
                .unwrap(),
            normalization
        );
    }

    #[test]
    fn grauert_remmert_steps_can_be_iterated_before_the_target_is_normal() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([(&variables[1].pow(2).unwrap() - &variables[0].pow(7).unwrap()).unwrap()])
            .unwrap();

        let first = source.grauert_remmert_step().unwrap().unwrap();
        first.verify().unwrap();
        assert_eq!(first.proof_kind(), "grauert_remmert_end_ring");
        assert_ne!(first.target(), &source);
        assert!(first.target().grauert_remmert_step().unwrap().is_some());
        assert_eq!(
            GrauertRemmertStep::<RationalField>::from_json(&first.to_json().unwrap()).unwrap(),
            first
        );

        let mut current = first.target().clone();
        let mut step_count = 1;
        while let Some(step) = current.grauert_remmert_step().unwrap() {
            step.verify().unwrap();
            current = step.target().clone();
            step_count += 1;
            assert!(step_count <= 4);
        }
        assert!(step_count >= 2);
        assert!(current.normalization().unwrap().is_normal());
    }
}
