// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Cohomologically justified zeta and Frobenius reconstruction.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::linalg::{LinalgError, Matrix};
use crate::numthy::FiniteField;
use crate::numthy::{cyclotomic_polynomial, CyclotomicPolynomialError};
use crate::{
    ContextMismatchError, IdentityCoefficientMap, Integer, IntegerRing, NoetherError, Rational,
    RationalField, Ring, SemanticType, TypeDescription, Q, Z,
};

use super::{
    CyclicCoverRamification, CyclicHypersurfaceCover, MonomialOrder, PerfectPowerError,
    PointCountError, PointCountSequence, PointCountTarget, Polynomial, PolynomialContextError,
    PolynomialError, PolynomialFactorizationError, PolynomialHomomorphismError, PolynomialRing,
    WeightedProjectiveError,
};

static SURFACE_COHOMOLOGY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/SmoothProjectiveSurfaceCohomology@1",
    rust_path: "noether::commalg::SmoothProjectiveSurfaceCohomology",
    python_path: Some("noether.commalg.SmoothProjectiveSurfaceCohomology"),
};

static FROBENIUS_POLYNOMIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/FrobeniusPolynomial@1",
    rust_path: "noether::commalg::FrobeniusPolynomial",
    python_path: Some("noether.commalg.FrobeniusPolynomial"),
};

static ZETA_FUNCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ZetaFunction@1",
    rust_path: "noether::commalg::ZetaFunction",
    python_path: Some("noether.commalg.ZetaFunction"),
};

static DIVISOR_RANK_UPPER_BOUND_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GeometricDivisorRankUpperBound@1",
    rust_path: "noether::commalg::GeometricDivisorRankUpperBound",
    python_path: Some("noether.commalg.GeometricDivisorRankUpperBound"),
};

static SPLIT_CYCLIC_PULLBACK_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/SplitCyclicPullback@1",
    rust_path: "noether::commalg::SplitCyclicPullback",
    python_path: Some("noether.commalg.SplitCyclicPullback"),
};

static DIVISOR_RANK_LOWER_BOUND_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/DivisorRankLowerBound@1",
    rust_path: "noether::commalg::DivisorRankLowerBound",
    python_path: Some("noether.commalg.DivisorRankLowerBound"),
};

static GEOMETRIC_DIVISOR_RANK_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/GeometricDivisorRank@1",
    rust_path: "noether::commalg::GeometricDivisorRank",
    python_path: Some("noether.commalg.GeometricDivisorRank"),
};

/// Failure to establish cohomological evidence or reconstruct Frobenius.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrobeniusError {
    /// Cyclic-cover smoothness or geometry failed.
    WeightedProjective(WeightedProjectiveError),
    /// Point-count evidence failed.
    PointCount(PointCountError),
    /// An exact polynomial context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Exact polynomial factorization failed.
    Factorization(PolynomialFactorizationError),
    /// A polynomial substitution failed.
    PolynomialHomomorphism(PolynomialHomomorphismError),
    /// Exact perfect-power extraction failed.
    PerfectPower(PerfectPowerError),
    /// A cyclotomic polynomial could not be constructed.
    Cyclotomic(CyclotomicPolynomialError),
    /// Exact matrix arithmetic failed.
    LinearAlgebra(LinalgError),
    /// The admitted cohomological theorem does not apply.
    UnsupportedCohomology(&'static str),
    /// Counts did not supply all power sums needed by Newton identities.
    InsufficientPointCounts {
        /// Required consecutive extension degree.
        required_through: usize,
    },
    /// Duality and the known hyperplane class left both signs possible.
    AmbiguousFunctionalEquationSign {
        /// A further extension degree that can distinguish the candidates.
        additional_degree: usize,
    },
    /// Newton identities did not divide exactly over the integers.
    NonIntegralReconstruction {
        /// Coefficient index that failed.
        coefficient: usize,
    },
    /// The reconstructed polynomial contradicted a required property.
    InvalidFrobeniusPolynomial(String),
    /// Canonical evidence was malformed.
    InvalidEvidence(String),
}

impl fmt::Display for FrobeniusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightedProjective(error) => error.fmt(formatter),
            Self::PointCount(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Factorization(error) => error.fmt(formatter),
            Self::PolynomialHomomorphism(error) => error.fmt(formatter),
            Self::PerfectPower(error) => error.fmt(formatter),
            Self::Cyclotomic(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::UnsupportedCohomology(reason) => {
                write!(formatter, "unsupported cohomological reconstruction: {reason}")
            }
            Self::InsufficientPointCounts { required_through } => write!(
                formatter,
                "Frobenius reconstruction needs exact counts through extension degree {required_through}"
            ),
            Self::AmbiguousFunctionalEquationSign { additional_degree } => write!(
                formatter,
                "Frobenius reconstruction needs an additional exact count (for example in extension degree {additional_degree}) to determine the functional-equation sign"
            ),
            Self::NonIntegralReconstruction { coefficient } => write!(
                formatter,
                "Newton reconstruction produced a nonintegral coefficient at index {coefficient}"
            ),
            Self::InvalidFrobeniusPolynomial(message) => {
                write!(formatter, "invalid Frobenius polynomial: {message}")
            }
            Self::InvalidEvidence(message) => {
                write!(formatter, "invalid Frobenius evidence: {message}")
            }
        }
    }
}

impl std::error::Error for FrobeniusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WeightedProjective(error) => Some(error),
            Self::PointCount(error) => Some(error),
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Factorization(error) => Some(error),
            Self::PolynomialHomomorphism(error) => Some(error),
            Self::PerfectPower(error) => Some(error),
            Self::Cyclotomic(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WeightedProjectiveError> for FrobeniusError {
    fn from(value: WeightedProjectiveError) -> Self {
        Self::WeightedProjective(value)
    }
}

impl From<PointCountError> for FrobeniusError {
    fn from(value: PointCountError) -> Self {
        Self::PointCount(value)
    }
}

impl From<PolynomialContextError> for FrobeniusError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for FrobeniusError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<ContextMismatchError> for FrobeniusError {
    fn from(value: ContextMismatchError) -> Self {
        Self::Polynomial(value.into())
    }
}

impl From<PolynomialFactorizationError> for FrobeniusError {
    fn from(value: PolynomialFactorizationError) -> Self {
        Self::Factorization(value)
    }
}

impl From<PolynomialHomomorphismError> for FrobeniusError {
    fn from(value: PolynomialHomomorphismError) -> Self {
        Self::PolynomialHomomorphism(value)
    }
}

impl From<PerfectPowerError> for FrobeniusError {
    fn from(value: PerfectPowerError) -> Self {
        Self::PerfectPower(value)
    }
}

impl From<CyclotomicPolynomialError> for FrobeniusError {
    fn from(value: CyclotomicPolynomialError) -> Self {
        Self::Cyclotomic(value)
    }
}

impl From<LinalgError> for FrobeniusError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

/// Verified cohomological profile for a smooth projective cyclic-cover surface.
///
/// In the admitted tame regime, the cover is a smooth cyclic cover of `P^2`
/// along a smooth plane curve. The standard Euler-characteristic formula and
/// irregularity vanishing establish `b0=b4=1`, `b1=b3=0`, and the recorded
/// second Betti number. This is a general surface statement, not a K3 type.
#[derive(Clone, PartialEq, Eq)]
pub struct SmoothProjectiveSurfaceCohomology {
    cover: CyclicHypersurfaceCover<FiniteField>,
    branch_degree: u64,
    branch_genus: Integer,
    euler_characteristic: Integer,
    second_betti_number: usize,
}

impl SmoothProjectiveSurfaceCohomology {
    /// Construct and verify the admitted cyclic-surface profile.
    pub fn for_cyclic_cover(
        cover: &CyclicHypersurfaceCover<FiniteField>,
    ) -> Result<Self, FrobeniusError> {
        build_surface_cohomology(cover)
    }

    /// Cyclic-cover presentation establishing this profile.
    pub fn cover(&self) -> &CyclicHypersurfaceCover<FiniteField> {
        &self.cover
    }

    /// Degree of the smooth plane branch curve.
    pub fn branch_degree(&self) -> u64 {
        self.branch_degree
    }

    /// Genus `(d-1)(d-2)/2` of the smooth plane branch curve.
    pub fn branch_genus(&self) -> &Integer {
        &self.branch_genus
    }

    /// Euler characteristic of the covering surface.
    pub fn euler_characteristic(&self) -> &Integer {
        &self.euler_characteristic
    }

    /// Dimension of second etale cohomology.
    pub fn second_betti_number(&self) -> usize {
        self.second_betti_number
    }

    /// Number of consecutive counts normally sufficient for reconstruction.
    ///
    /// The pullback of a line is defined over the base field and contributes
    /// the known Frobenius eigenvalue `q`. Poincare duality then reconstructs
    /// the remaining coefficients from `ceil(b_2/2)` traces. In a degenerate
    /// zero-coefficient case, one further count may be requested to distinguish
    /// the two possible functional-equation signs.
    pub fn required_point_count_degree(&self) -> usize {
        self.second_betti_number.div_ceil(2)
    }

    /// Frobenius eigenvalue supplied by the pullback of a base hyperplane.
    pub fn hyperplane_frobenius_eigenvalue(&self) -> Integer {
        self.cover.base().base().order()
    }

    /// Recompute the smoothness and cohomological formula.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        let rebuilt = build_surface_cohomology(&self.cover)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(FrobeniusError::InvalidEvidence(
                "recomputation changed the cyclic-cover cohomology profile".to_owned(),
            ))
        }
    }

    /// Recover `det(T-F | H^2)` from sufficient consecutive exact counts.
    pub fn frobenius_polynomial(
        &self,
        counts: &PointCountSequence,
    ) -> Result<FrobeniusPolynomial, FrobeniusError> {
        reconstruct_frobenius(self, counts)
    }

    /// Serialize the complete geometric evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&SurfaceCohomologyWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and recompute the complete geometric evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<SurfaceCohomologyWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let rebuilt = build_surface_cohomology(&wire.cover)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if rebuilt.branch_degree != wire.branch_degree
            || rebuilt.branch_genus != wire.branch_genus
            || rebuilt.euler_characteristic != wire.euler_characteristic
            || rebuilt.second_betti_number != wire.second_betti_number
        {
            return Err(NoetherError::Serialization(
                "serialized surface cohomology contradicts its cover".to_owned(),
            ));
        }
        Ok(rebuilt)
    }
}

impl CyclicHypersurfaceCover<FiniteField> {
    /// Establish the smooth projective surface cohomology used for exact
    /// Frobenius reconstruction, when the tame cyclic-cover theorem applies.
    pub fn surface_cohomology(&self) -> Result<SmoothProjectiveSurfaceCohomology, FrobeniusError> {
        SmoothProjectiveSurfaceCohomology::for_cyclic_cover(self)
    }
}

fn build_surface_cohomology(
    cover: &CyclicHypersurfaceCover<FiniteField>,
) -> Result<SmoothProjectiveSurfaceCohomology, FrobeniusError> {
    if cover.base().dimension() != 2 {
        return Err(FrobeniusError::UnsupportedCohomology(
            "the current theorem requires a cyclic cover of P^2",
        ));
    }
    if cover.ramification() != CyclicCoverRamification::Tame {
        return Err(FrobeniusError::UnsupportedCohomology(
            "the current cyclic-surface cohomology theorem requires tame ramification",
        ));
    }
    let smoothness = cover.smoothness_evidence()?;
    if !smoothness.is_smooth() || smoothness.branch_is_smooth() != Some(true) {
        return Err(FrobeniusError::UnsupportedCohomology(
            "the cyclic surface and its branch curve must be proved smooth",
        ));
    }
    let branch_degree = cover
        .branch_equation()
        .homogeneous_degree()?
        .ok_or(FrobeniusError::UnsupportedCohomology(
            "the branch equation must be nonzero and homogeneous",
        ))?
        .to_string()
        .parse::<u64>()
        .map_err(|_| FrobeniusError::UnsupportedCohomology("the branch degree must fit u64"))?;
    let d = Integer::from(branch_degree);
    let branch_genus = &(&(&d - &Integer::one()) * &(&d - &Integer::from(2))) / &Integer::from(2);
    let branch_euler = &Integer::from(2) - &(&Integer::from(2) * &branch_genus);
    let m = Integer::from(cover.cover_degree());
    let euler_characteristic =
        &(&Integer::from(3) * &m) - &(&(&m - &Integer::one()) * &branch_euler);
    let second_betti = &euler_characteristic - &Integer::from(2);
    if second_betti.sign() < 0 {
        return Err(FrobeniusError::InvalidEvidence(
            "cyclic-cover Euler formula produced a negative second Betti number".to_owned(),
        ));
    }
    let second_betti_number = second_betti.to_string().parse::<usize>().map_err(|_| {
        FrobeniusError::UnsupportedCohomology("the second Betti number must fit usize")
    })?;
    Ok(SmoothProjectiveSurfaceCohomology {
        cover: cover.clone(),
        branch_degree,
        branch_genus,
        euler_characteristic,
        second_betti_number,
    })
}

/// Exact characteristic polynomial of Frobenius with reconstruction evidence.
#[derive(Clone, PartialEq, Eq)]
pub struct FrobeniusPolynomial {
    polynomial: Polynomial<IntegerRing>,
    field_order: Integer,
    cohomological_degree: u32,
    cohomological_weight: u32,
    cohomology: SmoothProjectiveSurfaceCohomology,
    point_counts: PointCountSequence,
}

impl FrobeniusPolynomial {
    /// Monic polynomial `det(T-F | H^i)` over the integers.
    pub fn polynomial(&self) -> &Polynomial<IntegerRing> {
        &self.polynomial
    }

    /// Order of the explicit finite base field.
    pub fn field_order(&self) -> &Integer {
        &self.field_order
    }

    /// Cohomological degree on which Frobenius acts.
    pub fn cohomological_degree(&self) -> u32 {
        self.cohomological_degree
    }

    /// Weil weight certified for the eigenvalues.
    pub fn cohomological_weight(&self) -> u32 {
        self.cohomological_weight
    }

    /// Geometric evidence fixing the required polynomial degree.
    pub fn cohomology(&self) -> &SmoothProjectiveSurfaceCohomology {
        &self.cohomology
    }

    /// Exact point counts used by Newton reconstruction.
    pub fn point_counts(&self) -> &PointCountSequence {
        &self.point_counts
    }

    /// Degree of the characteristic polynomial.
    pub fn degree(&self) -> usize {
        self.cohomology.second_betti_number
    }

    /// Exact trace of the requested positive Frobenius power.
    pub fn trace(&self, power: u64) -> Result<Integer, FrobeniusError> {
        if power == 0 {
            return Err(FrobeniusError::InvalidFrobeniusPolynomial(
                "a Frobenius trace power must be positive".to_owned(),
            ));
        }
        let power = usize::try_from(power).map_err(|_| {
            FrobeniusError::InvalidFrobeniusPolynomial(
                "the trace power does not fit the recurrence".to_owned(),
            )
        })?;
        trace_from_coefficients(&characteristic_coefficients(&self.polynomial)?, power)
    }

    /// Verify geometry, Newton identities, all recorded counts, and duality.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        self.cohomology.verify()?;
        self.point_counts.verify()?;
        let rebuilt = reconstruct_frobenius(&self.cohomology, &self.point_counts)?;
        if rebuilt.polynomial != self.polynomial
            || rebuilt.field_order != self.field_order
            || self.cohomological_degree != 2
            || self.cohomological_weight != 2
        {
            return Err(FrobeniusError::InvalidFrobeniusPolynomial(
                "reconstruction or declared cohomological metadata changed".to_owned(),
            ));
        }
        verify_surface_counts(self)?;
        verify_functional_equation(self)?;
        Ok(())
    }

    /// Canonical zeta function of the supported smooth projective surface.
    pub fn zeta_function(&self) -> Result<ZetaFunction, FrobeniusError> {
        ZetaFunction::from_frobenius(self.clone())
    }

    /// Serialize polynomial, geometry, and reconstruction transcript.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FrobeniusPolynomialWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and fully reverify a Frobenius polynomial.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FrobeniusPolynomialWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = Self {
            polynomial: wire.polynomial,
            field_order: wire.field_order,
            cohomological_degree: wire.cohomological_degree,
            cohomological_weight: wire.cohomological_weight,
            cohomology: wire.cohomology,
            point_counts: wire.point_counts,
        };
        value
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(value)
    }
}

fn reconstruct_frobenius(
    cohomology: &SmoothProjectiveSurfaceCohomology,
    counts: &PointCountSequence,
) -> Result<FrobeniusPolynomial, FrobeniusError> {
    cohomology.verify()?;
    if counts.target() != &PointCountTarget::CyclicCover(Box::new(cohomology.cover.clone())) {
        return Err(FrobeniusError::InvalidEvidence(
            "point counts belong to another geometric presentation".to_owned(),
        ));
    }
    let rank = cohomology.second_betti_number;
    let required = cohomology.required_point_count_degree();
    let mut primitive_traces = Vec::with_capacity(required);
    let q = counts.base().order();
    for degree in 1..=required {
        let degree_u64 =
            u64::try_from(degree).map_err(|_| FrobeniusError::InsufficientPointCounts {
                required_through: required,
            })?;
        let count = counts
            .count(degree_u64)
            .ok_or(FrobeniusError::InsufficientPointCounts {
                required_through: required,
            })?;
        let top_trace = q.pow(2 * degree_u64);
        let middle_trace = count - &Integer::one() - &top_trace;
        primitive_traces.push(&middle_trace - &q.pow(degree_u64));
    }
    let primitive_prefix = newton_characteristic_coefficients(&primitive_traces)?;
    let primitive_degree = rank
        .checked_sub(1)
        .ok_or(FrobeniusError::UnsupportedCohomology(
            "the hyperplane class requires positive second Betti number",
        ))?;
    let mut candidates = [-1_i8, 1_i8]
        .into_iter()
        .filter_map(|sign| {
            complete_reciprocal_coefficients(&primitive_prefix, primitive_degree, &q, sign)
        })
        .map(|primitive_coefficients| {
            let coefficients = attach_hyperplane_factor(&primitive_coefficients, &q);
            let polynomial = polynomial_from_characteristic_coefficients(&coefficients)?;
            Ok(FrobeniusPolynomial {
                polynomial,
                field_order: q.clone(),
                cohomological_degree: 2,
                cohomological_weight: 2,
                cohomology: cohomology.clone(),
                point_counts: counts.clone(),
            })
        })
        .collect::<Result<Vec<_>, FrobeniusError>>()?;
    candidates.retain(|candidate| {
        verify_surface_counts(candidate).is_ok() && verify_functional_equation(candidate).is_ok()
    });
    let result = match candidates.len() {
        1 => candidates.remove(0),
        2 => {
            return Err(FrobeniusError::AmbiguousFunctionalEquationSign {
                additional_degree: required + 1,
            });
        }
        _ => {
            return Err(FrobeniusError::InvalidFrobeniusPolynomial(
                "no functional-equation sign is compatible with the exact counts and hyperplane class"
                    .to_owned(),
            ));
        }
    };
    Ok(result)
}

fn complete_reciprocal_coefficients(
    prefix: &[Integer],
    degree: usize,
    q: &Integer,
    sign: i8,
) -> Option<Vec<Integer>> {
    if prefix.is_empty() || prefix[0] != Integer::one() || prefix.len() > degree + 1 {
        return None;
    }
    let mut coefficients = vec![Integer::zero(); degree + 1];
    coefficients[..prefix.len()].clone_from_slice(prefix);
    for index in 0..=degree / 2 {
        if index >= prefix.len() {
            return None;
        }
        let mut reciprocal = &prefix[index] * &q.pow((degree - 2 * index) as u64);
        if sign < 0 {
            reciprocal = -reciprocal;
        }
        let target = degree - index;
        if target < prefix.len() {
            if prefix[target] != reciprocal {
                return None;
            }
        } else {
            coefficients[target] = reciprocal;
        }
    }
    Some(coefficients)
}

fn attach_hyperplane_factor(primitive: &[Integer], q: &Integer) -> Vec<Integer> {
    let mut coefficients = Vec::with_capacity(primitive.len() + 1);
    coefficients.push(primitive[0].clone());
    for index in 1..primitive.len() {
        coefficients.push(&primitive[index] - &(q * &primitive[index - 1]));
    }
    coefficients.push(-(q * primitive.last().expect("primitive polynomial is nonempty")));
    coefficients
}

fn newton_characteristic_coefficients(traces: &[Integer]) -> Result<Vec<Integer>, FrobeniusError> {
    let mut coefficients = Vec::with_capacity(traces.len() + 1);
    coefficients.push(Integer::one());
    for k in 1..=traces.len() {
        let mut numerator = Integer::zero();
        for index in 1..=k {
            numerator = &numerator + &(&coefficients[k - index] * &traces[index - 1]);
        }
        numerator = -numerator;
        let divisor = Integer::from(k as u64);
        if &numerator % &divisor != Integer::zero() {
            return Err(FrobeniusError::NonIntegralReconstruction { coefficient: k });
        }
        coefficients.push(&numerator / &divisor);
    }
    Ok(coefficients)
}

fn polynomial_from_characteristic_coefficients(
    coefficients: &[Integer],
) -> Result<Polynomial<IntegerRing>, FrobeniusError> {
    let degree = coefficients.len().saturating_sub(1);
    let ring = PolynomialRing::new(Z, ["T"], MonomialOrder::Lexicographic)?;
    ring.from_terms(coefficients.iter().enumerate().map(|(index, coefficient)| {
        (
            vec![Integer::from((degree - index) as u64)],
            coefficient.clone(),
        )
    }))
    .map_err(Into::into)
}

fn characteristic_coefficients(
    polynomial: &Polynomial<IntegerRing>,
) -> Result<Vec<Integer>, FrobeniusError> {
    let degree = polynomial
        .degree(Some(0))?
        .ok_or_else(|| {
            FrobeniusError::InvalidFrobeniusPolynomial("the polynomial is zero".to_owned())
        })?
        .to_string()
        .parse::<usize>()
        .map_err(|_| {
            FrobeniusError::InvalidFrobeniusPolynomial(
                "the polynomial degree does not fit usize".to_owned(),
            )
        })?;
    (0..=degree)
        .map(|index| {
            polynomial
                .coefficient(&[Integer::from((degree - index) as u64)])
                .map_err(Into::into)
        })
        .collect()
}

fn trace_from_coefficients(
    coefficients: &[Integer],
    requested_power: usize,
) -> Result<Integer, FrobeniusError> {
    let degree = coefficients.len().checked_sub(1).ok_or_else(|| {
        FrobeniusError::InvalidFrobeniusPolynomial("missing leading coefficient".to_owned())
    })?;
    if degree == 0 || coefficients[0] != Integer::one() {
        return Err(FrobeniusError::InvalidFrobeniusPolynomial(
            "the characteristic polynomial must be monic of positive degree".to_owned(),
        ));
    }
    let mut traces = Vec::<Integer>::with_capacity(requested_power);
    for k in 1..=requested_power {
        let mut sum = Integer::zero();
        let previous_count = if k <= degree { k - 1 } else { degree };
        for index in 1..=previous_count {
            sum = &sum + &(&coefficients[index] * &traces[k - index - 1]);
        }
        if k <= degree {
            sum = &sum + &(&Integer::from(k as u64) * &coefficients[k]);
        }
        traces.push(-sum);
    }
    traces.last().cloned().ok_or_else(|| {
        FrobeniusError::InvalidFrobeniusPolynomial("trace power must be positive".to_owned())
    })
}

fn verify_surface_counts(frobenius: &FrobeniusPolynomial) -> Result<(), FrobeniusError> {
    for record in frobenius.point_counts.counts() {
        let trace = frobenius.trace(record.degree())?;
        let expected =
            &(&Integer::one() + &trace) + &frobenius.field_order.pow(2 * record.degree());
        if &expected != record.count() {
            return Err(FrobeniusError::InvalidFrobeniusPolynomial(format!(
                "Frobenius trace does not recover the point count in degree {}",
                record.degree()
            )));
        }
    }
    Ok(())
}

fn verify_functional_equation(frobenius: &FrobeniusPolynomial) -> Result<(), FrobeniusError> {
    let coefficients = characteristic_coefficients(&frobenius.polynomial)?;
    let degree = coefficients.len() - 1;
    if degree != frobenius.degree() {
        return Err(FrobeniusError::InvalidFrobeniusPolynomial(
            "polynomial degree differs from the established Betti number".to_owned(),
        ));
    }
    let constant = &coefficients[degree];
    let q_squared = frobenius.field_order.pow(2);
    for index in 0..=degree {
        let left = &coefficients[index] * &q_squared.pow((degree - index) as u64);
        let right = constant * &coefficients[degree - index];
        if left != right {
            return Err(FrobeniusError::InvalidFrobeniusPolynomial(
                "Poincare-duality functional equation failed".to_owned(),
            ));
        }
    }
    Ok(())
}

/// Rational zeta function represented by exact integral numerator and denominator.
#[derive(Clone, PartialEq, Eq)]
pub struct ZetaFunction {
    numerator: Polynomial<IntegerRing>,
    denominator: Polynomial<IntegerRing>,
    frobenius: FrobeniusPolynomial,
}

impl ZetaFunction {
    fn from_frobenius(frobenius: FrobeniusPolynomial) -> Result<Self, FrobeniusError> {
        frobenius.verify()?;
        let coefficients = characteristic_coefficients(&frobenius.polynomial)?;
        let ring = PolynomialRing::new(Z, ["T"], MonomialOrder::Lexicographic)?;
        let middle = ring.from_terms(coefficients.iter().enumerate().map(
            |(degree, coefficient)| (vec![Integer::from(degree as u64)], coefficient.clone()),
        ))?;
        let variable = ring.generator(0)?;
        let h0 = ring.sub(&ring.one(), &variable)?;
        let q_squared_t = ring.mul(&ring.constant(frobenius.field_order.pow(2))?, &variable)?;
        let h4 = ring.sub(&ring.one(), &q_squared_t)?;
        let denominator = ring.mul(&ring.mul(&h0, &middle)?, &h4)?;
        let result = Self {
            numerator: ring.one(),
            denominator,
            frobenius,
        };
        result.verify()?;
        Ok(result)
    }

    /// Exact numerator in the canonical variable `T`.
    pub fn numerator(&self) -> &Polynomial<IntegerRing> {
        &self.numerator
    }

    /// Exact denominator in the canonical variable `T`.
    pub fn denominator(&self) -> &Polynomial<IntegerRing> {
        &self.denominator
    }

    /// Frobenius polynomial supplying the nontrivial Euler factor.
    pub fn frobenius_polynomial(&self) -> &FrobeniusPolynomial {
        &self.frobenius
    }

    /// Rebuild the Euler factors and recheck every recorded point count.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        self.frobenius.verify()?;
        let rebuilt = Self::from_frobenius_unchecked(self.frobenius.clone())?;
        if self.numerator != rebuilt.numerator || self.denominator != rebuilt.denominator {
            return Err(FrobeniusError::InvalidEvidence(
                "zeta numerator or denominator does not match its Frobenius polynomial".to_owned(),
            ));
        }
        Ok(())
    }

    fn from_frobenius_unchecked(frobenius: FrobeniusPolynomial) -> Result<Self, FrobeniusError> {
        let coefficients = characteristic_coefficients(&frobenius.polynomial)?;
        let ring = PolynomialRing::new(Z, ["T"], MonomialOrder::Lexicographic)?;
        let middle = ring.from_terms(coefficients.iter().enumerate().map(
            |(degree, coefficient)| (vec![Integer::from(degree as u64)], coefficient.clone()),
        ))?;
        let variable = ring.generator(0)?;
        let h0 = ring.sub(&ring.one(), &variable)?;
        let q_squared_t = ring.mul(&ring.constant(frobenius.field_order.pow(2))?, &variable)?;
        let h4 = ring.sub(&ring.one(), &q_squared_t)?;
        Ok(Self {
            numerator: ring.one(),
            denominator: ring.mul(&ring.mul(&h0, &middle)?, &h4)?,
            frobenius,
        })
    }

    /// Serialize all Euler factors and their evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&ZetaFunctionWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify a zeta function.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<ZetaFunctionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = Self {
            numerator: wire.numerator,
            denominator: wire.denominator,
            frobenius: wire.frobenius,
        };
        value
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(value)
    }
}

/// One cyclotomic factor of the Tate-normalized Frobenius polynomial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TateCyclotomicFactor {
    index: u64,
    degree: usize,
    multiplicity: u64,
}

impl TateCyclotomicFactor {
    /// Cyclotomic index `n` for the factor `Phi_n`.
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Degree `phi(n)` of this factor.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Multiplicity in the normalized Frobenius polynomial.
    pub fn multiplicity(&self) -> u64 {
        self.multiplicity
    }
}

/// Tate-style upper bound on the geometric divisor rank of a surface.
///
/// The bound counts Frobenius eigenvalues that become roots of unity after
/// division by the field order. It is deliberately not reported as an exact
/// rank without independent divisor classes of matching rank.
#[derive(Clone, PartialEq, Eq)]
pub struct GeometricDivisorRankUpperBound {
    frobenius: FrobeniusPolynomial,
    tate_normalized_polynomial: Polynomial<RationalField>,
    cyclotomic_factors: Vec<TateCyclotomicFactor>,
    upper_bound: usize,
}

impl GeometricDivisorRankUpperBound {
    /// Frobenius polynomial from which the bound was obtained.
    pub fn frobenius_polynomial(&self) -> &FrobeniusPolynomial {
        &self.frobenius
    }

    /// Monic polynomial whose roots are the Frobenius eigenvalues divided by `q`.
    pub fn tate_normalized_polynomial(&self) -> &Polynomial<RationalField> {
        &self.tate_normalized_polynomial
    }

    /// Complete cyclotomic factors and multiplicities.
    pub fn cyclotomic_factors(&self) -> &[TateCyclotomicFactor] {
        &self.cyclotomic_factors
    }

    /// Certified upper bound on geometric divisor rank.
    pub fn upper_bound(&self) -> usize {
        self.upper_bound
    }

    /// Recompute normalization, factorization, and the complete cyclotomic search.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        let rebuilt = divisor_rank_upper_bound(&self.frobenius)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(FrobeniusError::InvalidEvidence(
                "recomputation changed the geometric divisor-rank upper bound".to_owned(),
            ))
        }
    }

    /// Combine with independently verified divisor classes when the bounds match.
    pub fn certify_exact(
        &self,
        lower: &DivisorRankLowerBound,
    ) -> Result<GeometricDivisorRank, FrobeniusError> {
        self.verify()?;
        lower.verify()?;
        if self.frobenius.cohomology.cover != lower.split_pullback.cover {
            return Err(FrobeniusError::InvalidEvidence(
                "upper and lower divisor-rank evidence concern different surfaces".to_owned(),
            ));
        }
        if self.upper_bound != lower.lower_bound {
            return Err(FrobeniusError::InvalidEvidence(
                "geometric divisor rank is not exact because the bounds do not match".to_owned(),
            ));
        }
        Ok(GeometricDivisorRank {
            rank: self.upper_bound,
            upper: self.clone(),
            lower: lower.clone(),
        })
    }

    /// Serialize the normalized factorization evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&DivisorRankUpperBoundWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify the upper bound.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<DivisorRankUpperBoundWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let candidate = Self {
            frobenius: wire.frobenius,
            tate_normalized_polynomial: wire.tate_normalized_polynomial,
            cyclotomic_factors: wire.cyclotomic_factors,
            upper_bound: wire.upper_bound,
        };
        candidate
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(candidate)
    }
}

impl FrobeniusPolynomial {
    /// Compute the Tate-style geometric divisor-rank upper bound.
    pub fn geometric_divisor_rank_upper_bound(
        &self,
    ) -> Result<GeometricDivisorRankUpperBound, FrobeniusError> {
        divisor_rank_upper_bound(self)
    }
}

fn divisor_rank_upper_bound(
    frobenius: &FrobeniusPolynomial,
) -> Result<GeometricDivisorRankUpperBound, FrobeniusError> {
    frobenius.verify()?;
    if frobenius.cohomological_degree != 2 || frobenius.cohomological_weight != 2 {
        return Err(FrobeniusError::UnsupportedCohomology(
            "divisor-rank bounds require weight-two Frobenius on H^2",
        ));
    }
    let coefficients = characteristic_coefficients(&frobenius.polynomial)?;
    let degree = coefficients.len() - 1;
    let ring = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic)?;
    let normalized =
        ring.from_terms(coefficients.iter().enumerate().map(|(index, coefficient)| {
            let denominator = frobenius.field_order.pow(index as u64);
            let value = Rational::new(coefficient.clone(), denominator)
                .expect("a finite-field order is nonzero");
            (vec![Integer::from((degree - index) as u64)], value)
        }))?;
    let factorization = normalized.factor()?;
    let search_limit = (degree as u64)
        .checked_mul(degree as u64)
        .and_then(|square| square.checked_mul(2))
        .unwrap_or(u64::MAX)
        .max(6);
    let mut cyclotomic_factors = Vec::new();
    for (factor, multiplicity) in factorization.factors() {
        let factor_degree = factor
            .degree(Some(0))?
            .ok_or_else(|| {
                FrobeniusError::InvalidFrobeniusPolynomial(
                    "a nonconstant factor had no degree".to_owned(),
                )
            })?
            .to_string()
            .parse::<usize>()
            .map_err(|_| {
                FrobeniusError::InvalidFrobeniusPolynomial(
                    "a factor degree does not fit usize".to_owned(),
                )
            })?;
        if let Some(index) = cyclotomic_index(factor, search_limit)? {
            cyclotomic_factors.push(TateCyclotomicFactor {
                index,
                degree: factor_degree,
                multiplicity: *multiplicity,
            });
        }
    }
    cyclotomic_factors.sort_by_key(TateCyclotomicFactor::index);
    let upper_bound = cyclotomic_factors
        .iter()
        .try_fold(0_usize, |sum, factor| {
            usize::try_from(factor.multiplicity)
                .ok()
                .and_then(|multiplicity| factor.degree.checked_mul(multiplicity))
                .and_then(|contribution| sum.checked_add(contribution))
        })
        .ok_or_else(|| {
            FrobeniusError::InvalidEvidence(
                "cyclotomic multiplicities exceed the cohomological dimension".to_owned(),
            )
        })?;
    if upper_bound > degree {
        return Err(FrobeniusError::InvalidEvidence(
            "cyclotomic factor degrees exceed the Frobenius degree".to_owned(),
        ));
    }
    Ok(GeometricDivisorRankUpperBound {
        frobenius: frobenius.clone(),
        tate_normalized_polynomial: normalized,
        cyclotomic_factors,
        upper_bound,
    })
}

fn cyclotomic_index(
    factor: &Polynomial<RationalField>,
    search_limit: u64,
) -> Result<Option<u64>, FrobeniusError> {
    let ring = factor.ring();
    for index in 1..=search_limit {
        let cyclotomic = cyclotomic_polynomial(index)?;
        let candidate = ring.from_terms(cyclotomic.terms().map(|(exponent, coefficient)| {
            (exponent.to_vec(), Rational::from(coefficient.clone()))
        }))?;
        if &candidate == factor {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

/// Verified splitting of the pullback of a line to a cyclic surface cover.
///
/// The line is given on an affine parameter `t` by three affine-linear base
/// coordinates. The exact identity `f(line(t)) = g(t)^m`, together with all
/// `m`-th roots of unity in the base field, proves that its pullback has `m`
/// distinct geometrically defined components.
#[derive(Clone, PartialEq, Eq)]
pub struct SplitCyclicPullback {
    cover: CyclicHypersurfaceCover<FiniteField>,
    line_parameterization: Vec<Polynomial<FiniteField>>,
    restricted_branch: Polynomial<FiniteField>,
    branch_root: Polynomial<FiniteField>,
}

impl SplitCyclicPullback {
    /// Verify a split line pullback from three affine-linear coordinate images.
    pub fn line(
        cover: &CyclicHypersurfaceCover<FiniteField>,
        coordinate_images: Vec<Polynomial<FiniteField>>,
    ) -> Result<Self, FrobeniusError> {
        build_split_cyclic_pullback(cover, coordinate_images)
    }

    /// Cyclic surface carrying the split pullback.
    pub fn cover(&self) -> &CyclicHypersurfaceCover<FiniteField> {
        &self.cover
    }

    /// Affine-linear base coordinates as functions of `t`.
    pub fn line_parameterization(&self) -> &[Polynomial<FiniteField>] {
        &self.line_parameterization
    }

    /// Exact restriction of the branch equation to the line.
    pub fn restricted_branch(&self) -> &Polynomial<FiniteField> {
        &self.restricted_branch
    }

    /// Deterministic polynomial `m`-th root of the restricted branch equation.
    pub fn branch_root(&self) -> &Polynomial<FiniteField> {
        &self.branch_root
    }

    /// Number of distinct components of the split pullback.
    pub fn component_count(&self) -> usize {
        self.cover.cover_degree() as usize
    }

    /// Recompute the substitution and perfect-power certificate.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        let rebuilt = build_split_cyclic_pullback(&self.cover, self.line_parameterization.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(FrobeniusError::InvalidEvidence(
                "recomputation changed the split cyclic pullback".to_owned(),
            ))
        }
    }

    /// Intersection pairing of the split components.
    ///
    /// If `a=d/m`, distinct components meet in the `a` zeros of the branch
    /// root. Since their sum is the pullback of a line and has intersection
    /// one with each component, diagonal entries are `1-(m-1)a`.
    pub fn intersection_matrix(&self) -> Result<Matrix<IntegerRing>, FrobeniusError> {
        let component_count = self.component_count();
        let branch_degree = self
            .cover
            .branch_equation()
            .homogeneous_degree()?
            .expect("a cyclic cover has a homogeneous branch equation");
        let m = Integer::from(self.cover.cover_degree());
        let a = &branch_degree / &m;
        let diagonal = &Integer::one() - &(&(&m - &Integer::one()) * &a);
        Matrix::from_rows(
            Z,
            (0..component_count).map(|row| {
                (0..component_count)
                    .map(|column| {
                        if row == column {
                            diagonal.clone()
                        } else {
                            a.clone()
                        }
                    })
                    .collect::<Vec<_>>()
            }),
        )
        .map_err(Into::into)
    }

    /// Exact lower bound furnished by the nondegenerate component pairing.
    pub fn divisor_rank_lower_bound(&self) -> Result<DivisorRankLowerBound, FrobeniusError> {
        let matrix = self.intersection_matrix()?;
        let lower_bound = matrix.smith_normal_form().rank();
        let result = DivisorRankLowerBound {
            split_pullback: self.clone(),
            intersection_matrix: matrix,
            lower_bound,
        };
        result.verify()?;
        Ok(result)
    }

    /// Serialize the parameterization and exact power identity.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&SplitCyclicPullbackWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify a split pullback.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<SplitCyclicPullbackWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let candidate = Self {
            cover: wire.cover,
            line_parameterization: wire.line_parameterization,
            restricted_branch: wire.restricted_branch,
            branch_root: wire.branch_root,
        };
        candidate
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(candidate)
    }
}

fn build_split_cyclic_pullback(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    coordinate_images: Vec<Polynomial<FiniteField>>,
) -> Result<SplitCyclicPullback, FrobeniusError> {
    if cover.base().dimension() != 2 || coordinate_images.len() != 3 {
        return Err(FrobeniusError::UnsupportedCohomology(
            "a split line pullback currently requires a cyclic surface over P^2",
        ));
    }
    if cover.ramification() != CyclicCoverRamification::Tame {
        return Err(FrobeniusError::UnsupportedCohomology(
            "split components require a tame cyclic cover",
        ));
    }
    let target = coordinate_images
        .first()
        .ok_or(FrobeniusError::InvalidEvidence(
            "a line parameterization has no coordinate images".to_owned(),
        ))?
        .ring()
        .clone();
    if target.variables().len() != 1
        || coordinate_images
            .iter()
            .any(|image| image.ring() != &target)
    {
        return Err(FrobeniusError::InvalidEvidence(
            "line coordinate images must share one univariate polynomial context".to_owned(),
        ));
    }
    let mut has_degree_one = false;
    let mut has_unit_constant_chart = false;
    for image in &coordinate_images {
        let degree = image.degree(Some(0))?;
        if degree
            .as_ref()
            .is_some_and(|degree| degree > &Integer::one())
        {
            return Err(FrobeniusError::InvalidEvidence(
                "a line coordinate image is not affine linear".to_owned(),
            ));
        }
        has_degree_one |= degree == Some(Integer::one());
        has_unit_constant_chart |= image == &target.one();
    }
    if !has_degree_one || !has_unit_constant_chart {
        return Err(FrobeniusError::InvalidEvidence(
            "the line parameterization must be nonconstant and use a unit affine chart".to_owned(),
        ));
    }
    let group_order = cover.base().base().order() - Integer::one();
    if &group_order % Integer::from(cover.cover_degree()) != Integer::zero() {
        return Err(FrobeniusError::UnsupportedCohomology(
            "the base field must contain every root of unity for the cyclic cover degree",
        ));
    }
    let substitution = cover.base().coordinate_ring().homomorphism(
        &target,
        IdentityCoefficientMap::new(cover.base().base().clone()),
        coordinate_images.clone(),
    )?;
    let restricted_branch = substitution.apply(cover.branch_equation())?;
    let branch_root = restricted_branch.nth_root(cover.cover_degree())?.ok_or(
        FrobeniusError::InvalidEvidence(
            "the branch restriction is not an exact power of the cover degree".to_owned(),
        ),
    )?;
    if branch_root.pow(cover.cover_degree())? != restricted_branch {
        return Err(FrobeniusError::InvalidEvidence(
            "the extracted branch root does not reproduce the restriction".to_owned(),
        ));
    }
    Ok(SplitCyclicPullback {
        cover: cover.clone(),
        line_parameterization: coordinate_images,
        restricted_branch,
        branch_root,
    })
}

/// Lower bound from an exact nondegenerate divisor intersection pairing.
#[derive(Clone, PartialEq, Eq)]
pub struct DivisorRankLowerBound {
    split_pullback: SplitCyclicPullback,
    intersection_matrix: Matrix<IntegerRing>,
    lower_bound: usize,
}

impl DivisorRankLowerBound {
    /// Geometric splitting evidence for the represented divisor classes.
    pub fn split_pullback(&self) -> &SplitCyclicPullback {
        &self.split_pullback
    }

    /// Exact component intersection matrix.
    pub fn intersection_matrix(&self) -> &Matrix<IntegerRing> {
        &self.intersection_matrix
    }

    /// Rank proved by that pairing.
    pub fn lower_bound(&self) -> usize {
        self.lower_bound
    }

    /// Recompute the intersection formula and exact integer rank.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        self.split_pullback.verify()?;
        let matrix = self.split_pullback.intersection_matrix()?;
        let rank = matrix.smith_normal_form().rank();
        if matrix != self.intersection_matrix || rank != self.lower_bound {
            return Err(FrobeniusError::InvalidEvidence(
                "divisor intersection pairing or its rank changed".to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize the split geometry and exact pairing.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&DivisorRankLowerBoundWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify a lower bound.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<DivisorRankLowerBoundWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let candidate = Self {
            split_pullback: wire.split_pullback,
            intersection_matrix: wire.intersection_matrix,
            lower_bound: wire.lower_bound,
        };
        candidate
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(candidate)
    }
}

/// Exact geometric divisor rank certified by matching independent bounds.
#[derive(Clone, PartialEq, Eq)]
pub struct GeometricDivisorRank {
    rank: usize,
    upper: GeometricDivisorRankUpperBound,
    lower: DivisorRankLowerBound,
}

impl GeometricDivisorRank {
    /// Exact geometric divisor rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Frobenius/Tate upper-bound evidence.
    pub fn upper_bound_evidence(&self) -> &GeometricDivisorRankUpperBound {
        &self.upper
    }

    /// Explicit-divisor lower-bound evidence.
    pub fn lower_bound_evidence(&self) -> &DivisorRankLowerBound {
        &self.lower
    }

    /// Reverify both independent bounds and their equality.
    pub fn verify(&self) -> Result<(), FrobeniusError> {
        let rebuilt = self.upper.certify_exact(&self.lower)?;
        if rebuilt.rank == self.rank {
            Ok(())
        } else {
            Err(FrobeniusError::InvalidEvidence(
                "matching divisor-rank bounds changed".to_owned(),
            ))
        }
    }

    /// Serialize both independent rank bounds.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&GeometricDivisorRankWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify an exact divisor rank.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<GeometricDivisorRankWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let candidate = Self {
            rank: wire.rank,
            upper: wire.upper,
            lower: wire.lower,
        };
        candidate
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(candidate)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DivisorRankUpperBoundWire {
    #[serde(rename = "type")]
    type_id: String,
    frobenius: FrobeniusPolynomial,
    tate_normalized_polynomial: Polynomial<RationalField>,
    cyclotomic_factors: Vec<TateCyclotomicFactor>,
    upper_bound: usize,
}

impl From<&GeometricDivisorRankUpperBound> for DivisorRankUpperBoundWire {
    fn from(value: &GeometricDivisorRankUpperBound) -> Self {
        Self {
            type_id: GeometricDivisorRankUpperBound::TYPE_ID.to_owned(),
            frobenius: value.frobenius.clone(),
            tate_normalized_polynomial: value.tate_normalized_polynomial.clone(),
            cyclotomic_factors: value.cyclotomic_factors.clone(),
            upper_bound: value.upper_bound,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitCyclicPullbackWire {
    #[serde(rename = "type")]
    type_id: String,
    cover: CyclicHypersurfaceCover<FiniteField>,
    line_parameterization: Vec<Polynomial<FiniteField>>,
    restricted_branch: Polynomial<FiniteField>,
    branch_root: Polynomial<FiniteField>,
}

impl From<&SplitCyclicPullback> for SplitCyclicPullbackWire {
    fn from(value: &SplitCyclicPullback) -> Self {
        Self {
            type_id: SplitCyclicPullback::TYPE_ID.to_owned(),
            cover: value.cover.clone(),
            line_parameterization: value.line_parameterization.clone(),
            restricted_branch: value.restricted_branch.clone(),
            branch_root: value.branch_root.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DivisorRankLowerBoundWire {
    #[serde(rename = "type")]
    type_id: String,
    split_pullback: SplitCyclicPullback,
    intersection_matrix: Matrix<IntegerRing>,
    lower_bound: usize,
}

impl From<&DivisorRankLowerBound> for DivisorRankLowerBoundWire {
    fn from(value: &DivisorRankLowerBound) -> Self {
        Self {
            type_id: DivisorRankLowerBound::TYPE_ID.to_owned(),
            split_pullback: value.split_pullback.clone(),
            intersection_matrix: value.intersection_matrix.clone(),
            lower_bound: value.lower_bound,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GeometricDivisorRankWire {
    #[serde(rename = "type")]
    type_id: String,
    rank: usize,
    upper: GeometricDivisorRankUpperBound,
    lower: DivisorRankLowerBound,
}

impl From<&GeometricDivisorRank> for GeometricDivisorRankWire {
    fn from(value: &GeometricDivisorRank) -> Self {
        Self {
            type_id: GeometricDivisorRank::TYPE_ID.to_owned(),
            rank: value.rank,
            upper: value.upper.clone(),
            lower: value.lower.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceCohomologyWire {
    #[serde(rename = "type")]
    type_id: String,
    cover: CyclicHypersurfaceCover<FiniteField>,
    branch_degree: u64,
    branch_genus: Integer,
    euler_characteristic: Integer,
    second_betti_number: usize,
}

impl From<&SmoothProjectiveSurfaceCohomology> for SurfaceCohomologyWire {
    fn from(value: &SmoothProjectiveSurfaceCohomology) -> Self {
        Self {
            type_id: SmoothProjectiveSurfaceCohomology::TYPE_ID.to_owned(),
            cover: value.cover.clone(),
            branch_degree: value.branch_degree,
            branch_genus: value.branch_genus.clone(),
            euler_characteristic: value.euler_characteristic.clone(),
            second_betti_number: value.second_betti_number,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrobeniusPolynomialWire {
    #[serde(rename = "type")]
    type_id: String,
    polynomial: Polynomial<IntegerRing>,
    field_order: Integer,
    cohomological_degree: u32,
    cohomological_weight: u32,
    cohomology: SmoothProjectiveSurfaceCohomology,
    point_counts: PointCountSequence,
}

impl From<&FrobeniusPolynomial> for FrobeniusPolynomialWire {
    fn from(value: &FrobeniusPolynomial) -> Self {
        Self {
            type_id: FrobeniusPolynomial::TYPE_ID.to_owned(),
            polynomial: value.polynomial.clone(),
            field_order: value.field_order.clone(),
            cohomological_degree: value.cohomological_degree,
            cohomological_weight: value.cohomological_weight,
            cohomology: value.cohomology.clone(),
            point_counts: value.point_counts.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ZetaFunctionWire {
    #[serde(rename = "type")]
    type_id: String,
    numerator: Polynomial<IntegerRing>,
    denominator: Polynomial<IntegerRing>,
    frobenius: FrobeniusPolynomial,
}

impl From<&ZetaFunction> for ZetaFunctionWire {
    fn from(value: &ZetaFunction) -> Self {
        Self {
            type_id: ZetaFunction::TYPE_ID.to_owned(),
            numerator: value.numerator.clone(),
            denominator: value.denominator.clone(),
            frobenius: value.frobenius.clone(),
        }
    }
}

macro_rules! impl_typed_serialization {
    ($type:ty, $wire:ty, $description:ident, $type_id:literal, $debug_name:literal) => {
        impl Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                <$wire>::from(self).serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let wire = <$wire>::deserialize(deserializer)?;
                let encoded = serde_json::to_string(&wire).map_err(serde::de::Error::custom)?;
                Self::from_json(&encoded).map_err(serde::de::Error::custom)
            }
        }

        impl SemanticType for $type {
            const TYPE_ID: &'static str = $type_id;

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &$description
            }
        }

        impl fmt::Debug for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str($debug_name)
            }
        }
    };
}

impl_typed_serialization!(
    SmoothProjectiveSurfaceCohomology,
    SurfaceCohomologyWire,
    SURFACE_COHOMOLOGY_DESCRIPTION,
    "noether/commalg/SmoothProjectiveSurfaceCohomology@1",
    "SmoothProjectiveSurfaceCohomology(...)"
);
impl_typed_serialization!(
    FrobeniusPolynomial,
    FrobeniusPolynomialWire,
    FROBENIUS_POLYNOMIAL_DESCRIPTION,
    "noether/commalg/FrobeniusPolynomial@1",
    "FrobeniusPolynomial(...)"
);
impl_typed_serialization!(
    ZetaFunction,
    ZetaFunctionWire,
    ZETA_FUNCTION_DESCRIPTION,
    "noether/commalg/ZetaFunction@1",
    "ZetaFunction(...)"
);
impl_typed_serialization!(
    GeometricDivisorRankUpperBound,
    DivisorRankUpperBoundWire,
    DIVISOR_RANK_UPPER_BOUND_DESCRIPTION,
    "noether/commalg/GeometricDivisorRankUpperBound@1",
    "GeometricDivisorRankUpperBound(...)"
);
impl_typed_serialization!(
    SplitCyclicPullback,
    SplitCyclicPullbackWire,
    SPLIT_CYCLIC_PULLBACK_DESCRIPTION,
    "noether/commalg/SplitCyclicPullback@1",
    "SplitCyclicPullback(...)"
);
impl_typed_serialization!(
    DivisorRankLowerBound,
    DivisorRankLowerBoundWire,
    DIVISOR_RANK_LOWER_BOUND_DESCRIPTION,
    "noether/commalg/DivisorRankLowerBound@1",
    "DivisorRankLowerBound(...)"
);
impl_typed_serialization!(
    GeometricDivisorRank,
    GeometricDivisorRankWire,
    GEOMETRIC_DIVISOR_RANK_DESCRIPTION,
    "noether/commalg/GeometricDivisorRank@1",
    "GeometricDivisorRank(...)"
);

impl fmt::Display for FrobeniusPolynomial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.polynomial)
    }
}

impl fmt::Display for ZetaFunction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}) / ({})", self.numerator, self.denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Polynomial, ProjectiveSpace};
    use super::*;

    fn smooth_quadric_cover() -> CyclicHypersurfaceCover<FiniteField> {
        let field = FiniteField::prime(5).unwrap();
        let base = ProjectiveSpace::with_names(field, 2, ["x", "y", "z"]).unwrap();
        let [x, y, z]: [Polynomial<FiniteField>; 3] = base
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let ring = base.coordinate_ring();
        let branch = ring
            .sub(&ring.mul(&x, &z).unwrap(), &y.pow(2).unwrap())
            .unwrap();
        CyclicHypersurfaceCover::<FiniteField>::new(base, 2, branch).unwrap()
    }

    #[test]
    fn cyclic_surface_formula_establishes_the_needed_degree() {
        let cover = smooth_quadric_cover();
        let cohomology = cover.surface_cohomology().unwrap();
        assert_eq!(cohomology.branch_genus(), &Integer::zero());
        assert_eq!(cohomology.euler_characteristic(), &Integer::from(4));
        assert_eq!(cohomology.second_betti_number(), 2);
        cohomology.verify().unwrap();
    }

    #[test]
    fn point_counts_reconstruct_frobenius_and_zeta() {
        let cover = smooth_quadric_cover();
        let cohomology = cover.surface_cohomology().unwrap();
        assert_eq!(cohomology.required_point_count_degree(), 1);
        assert_eq!(
            cohomology.hyperplane_frobenius_eigenvalue(),
            Integer::from(5)
        );
        let counts = cover.point_counts([1]).unwrap();
        let frobenius = cohomology.frobenius_polynomial(&counts).unwrap();
        frobenius.verify().unwrap();
        for degree in 1..=1 {
            let trace = frobenius.trace(degree).unwrap();
            assert_eq!(
                counts.count(degree).unwrap(),
                &(&Integer::one() + &trace + cover.base().base().order().pow(2 * degree))
            );
        }
        let zeta = frobenius.zeta_function().unwrap();
        zeta.verify().unwrap();
        assert_eq!(
            FrobeniusPolynomial::from_json(&frobenius.to_json().unwrap()).unwrap(),
            frobenius
        );
        assert_eq!(
            ZetaFunction::from_json(&zeta.to_json().unwrap()).unwrap(),
            zeta
        );
    }

    #[test]
    fn reconstruction_refuses_missing_cohomological_traces() {
        let cover = smooth_quadric_cover();
        let cohomology = cover.surface_cohomology().unwrap();
        let counts = cover.point_counts([2]).unwrap();
        assert_eq!(
            cohomology.frobenius_polynomial(&counts),
            Err(FrobeniusError::InsufficientPointCounts {
                required_through: 1
            })
        );
    }

    #[test]
    fn cyclotomic_upper_bound_and_split_line_pairing_certify_rank() {
        let cover = smooth_quadric_cover();
        let cohomology = cover.surface_cohomology().unwrap();
        let counts = cover.point_counts([1]).unwrap();
        let frobenius = cohomology.frobenius_polynomial(&counts).unwrap();
        let upper = frobenius.geometric_divisor_rank_upper_bound().unwrap();
        assert_eq!(upper.upper_bound(), 2);

        let field = cover.base().base().clone();
        let parameter_ring =
            PolynomialRing::new(field, ["t"], MonomialOrder::Lexicographic).unwrap();
        let t = parameter_ring.generator(0).unwrap();
        let split =
            SplitCyclicPullback::line(&cover, vec![parameter_ring.one(), t, parameter_ring.zero()])
                .unwrap();
        let lower = split.divisor_rank_lower_bound().unwrap();
        assert_eq!(lower.lower_bound(), 2);
        assert_eq!(
            lower.intersection_matrix().get(0, 0).unwrap(),
            Integer::zero()
        );
        assert_eq!(
            lower.intersection_matrix().get(0, 1).unwrap(),
            Integer::one()
        );

        let exact = upper.certify_exact(&lower).unwrap();
        assert_eq!(exact.rank(), 2);
        exact.verify().unwrap();
        assert_eq!(
            GeometricDivisorRank::from_json(&exact.to_json().unwrap()).unwrap(),
            exact
        );
    }
}
