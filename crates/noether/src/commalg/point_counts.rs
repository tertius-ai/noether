// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact extension-field point-count sequences and their evidence.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::linalg::LinalgError;
use crate::numthy::{
    FiniteField, FiniteFieldConstructionError, FiniteFieldEmbedding, FiniteFieldEmbeddingError,
    PointCountCertificate,
};
use crate::{
    ContextMismatchError, FiniteContext, Integer, NoetherError, Ring, SemanticType, TypeDescription,
};

use super::padic_point_count::count_projective_double_cover;
use super::scheme::deterministic_finite_field_embedding;
use super::{
    BaseChangeError, CyclicHypersurfaceCover, PolynomialContextError, PolynomialError, Scheme,
    SchemeError, WeightedProjectiveError,
};

const MAX_EXTENSION_FIELD_ORDER: u64 = 4_096;
const MAX_PROJECTIVE_CANDIDATES: u64 = 10_000_000;
const MAX_INDEPENDENT_FIBER_TESTS: u64 = 1_000_000;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

static POINT_COUNT_SEQUENCE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PointCountSequence@1",
    rust_path: "noether::commalg::PointCountSequence",
    python_path: Some("noether.commalg.PointCountSequence"),
};

/// Failure to construct or verify an exact extension-field point count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PointCountError {
    /// An exact scheme operation failed.
    Scheme(SchemeError),
    /// A cyclic-cover operation failed.
    WeightedProjective(WeightedProjectiveError),
    /// A coefficient extension failed.
    BaseChange(BaseChangeError),
    /// An extension field could not be constructed.
    FiniteFieldConstruction(FiniteFieldConstructionError),
    /// A checked finite-field embedding failed.
    FiniteFieldEmbedding(FiniteFieldEmbeddingError),
    /// A polynomial context required by a trace formula could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic required by a trace formula failed.
    Polynomial(PolynomialError),
    /// Exact matrix arithmetic required by a trace formula failed.
    LinearAlgebra(LinalgError),
    /// Two internal exact values unexpectedly carried unequal contexts.
    ContextMismatch(ContextMismatchError),
    /// Extension degrees must be positive and distinct.
    InvalidExtensionDegrees,
    /// The exhaustive correctness floor reached an explicit limit.
    ResourceLimit {
        /// Bounded quantity.
        resource: &'static str,
        /// Fixed limit.
        limit: u64,
    },
    /// No accelerated theorem currently admits this input regime.
    TraceFormulaUnavailable(&'static str),
    /// A p-adic trace-formula invariant or reconstruction failed.
    TraceFormulaFailure(String),
    /// Canonical evidence was internally inconsistent.
    InvalidEvidence(String),
}

impl fmt::Display for PointCountError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheme(error) => error.fmt(formatter),
            Self::WeightedProjective(error) => error.fmt(formatter),
            Self::BaseChange(error) => error.fmt(formatter),
            Self::FiniteFieldConstruction(error) => error.fmt(formatter),
            Self::FiniteFieldEmbedding(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidExtensionDegrees => formatter.write_str(
                "point-count extension degrees must be positive, strictly increasing, and distinct",
            ),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "exact point counting exceeded its {resource} limit of {limit}"
            ),
            Self::TraceFormulaUnavailable(reason) => {
                write!(formatter, "p-adic trace formula unavailable: {reason}")
            }
            Self::TraceFormulaFailure(message) => {
                write!(formatter, "p-adic trace formula failed: {message}")
            }
            Self::InvalidEvidence(message) => {
                write!(formatter, "invalid point-count evidence: {message}")
            }
        }
    }
}

impl std::error::Error for PointCountError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scheme(error) => Some(error),
            Self::WeightedProjective(error) => Some(error),
            Self::BaseChange(error) => Some(error),
            Self::FiniteFieldConstruction(error) => Some(error),
            Self::FiniteFieldEmbedding(error) => Some(error),
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SchemeError> for PointCountError {
    fn from(value: SchemeError) -> Self {
        Self::Scheme(value)
    }
}

impl From<WeightedProjectiveError> for PointCountError {
    fn from(value: WeightedProjectiveError) -> Self {
        Self::WeightedProjective(value)
    }
}

impl From<BaseChangeError> for PointCountError {
    fn from(value: BaseChangeError) -> Self {
        Self::BaseChange(value)
    }
}

impl From<FiniteFieldConstructionError> for PointCountError {
    fn from(value: FiniteFieldConstructionError) -> Self {
        Self::FiniteFieldConstruction(value)
    }
}

impl From<FiniteFieldEmbeddingError> for PointCountError {
    fn from(value: FiniteFieldEmbeddingError) -> Self {
        Self::FiniteFieldEmbedding(value)
    }
}

impl From<PolynomialContextError> for PointCountError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for PointCountError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<LinalgError> for PointCountError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<ContextMismatchError> for PointCountError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// Geometric presentation whose extension-field points are being counted.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PointCountTarget {
    /// An ordinary affine or projective finite-field scheme.
    Scheme(Box<Scheme<FiniteField>>),
    /// A cyclic cover counted through its projective base and exact fibers.
    CyclicCover(Box<CyclicHypersurfaceCover<FiniteField>>),
}

impl PointCountTarget {
    /// Explicit finite base field.
    pub fn base(&self) -> &FiniteField {
        match self {
            Self::Scheme(scheme) => scheme.base(),
            Self::CyclicCover(cover) => cover.base().base(),
        }
    }
}

impl fmt::Debug for PointCountTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheme(scheme) => formatter.debug_tuple("Scheme").field(scheme).finish(),
            Self::CyclicCover(cover) => formatter.debug_tuple("CyclicCover").field(cover).finish(),
        }
    }
}

/// Mathematical route used for one exact count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointCountMethod {
    /// Canonical rational-point enumeration of an ordinary scheme.
    ExhaustiveScheme,
    /// Canonical projective-base enumeration plus exact cyclic fiber sizes.
    ExhaustiveCyclicFibers,
    /// Elsenhans--Jahnel/Harvey p-adic trace formula for a projective double cover.
    PadicDoubleCoverTrace,
}

/// Method-specific replayable evidence for one extension-field count.
///
/// Construction and standalone deserialization are intentionally sealed: the
/// evidence is meaningful only together with a target and count, and
/// [`PointCountSequence`] is the validating serialization boundary.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct PointCountEvidence {
    inner: PointCountEvidenceWire,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
enum PointCountEvidenceWire {
    /// Complete ordinary-scheme enumeration transcript.
    ExhaustiveScheme {
        /// Extension field used for the count.
        extension_field: FiniteField,
        /// Checked embedding of the declared base field.
        embedding: FiniteFieldEmbedding,
        /// Deterministic enumeration transcript.
        certificate: PointCountCertificate,
    },
    /// Complete cyclic-fiber transcript over every base projective point.
    ExhaustiveCyclicFibers {
        /// Extension field used for the count.
        extension_field: FiniteField,
        /// Checked embedding of the declared base field.
        embedding: FiniteFieldEmbedding,
        /// Transcript covering every base projective point.
        base_certificate: PointCountCertificate,
        /// Stable checksum of the ordered fiber sizes.
        fiber_checksum: u64,
        /// Whether direct enumeration of every possible fiber coordinate was
        /// also affordable and agreed with the group-theoretic root count.
        independently_verified: bool,
    },
    /// Exact lift of a p-adic trace-formula congruence using a Weil bound.
    PadicDoubleCoverTrace {
        /// Canonical extension field associated with the relative degree.
        extension_field: FiniteField,
        /// Checked embedding of the declared finite base field.
        embedding: FiniteFieldEmbedding,
        /// Number of p-adic digits used for this count.
        precision: u64,
        /// Exact modulus `p^precision`.
        modulus: Integer,
        /// Canonical point-count residue modulo the exact modulus.
        count_residue: Integer,
        /// Weil bound used to select the unique Frobenius-trace lift.
        trace_bound: Integer,
        /// Number of disjoint projective coordinate-support strata.
        stratum_count: usize,
        /// Largest finite trace-operator matrix dimension.
        largest_matrix_dimension: usize,
        /// Stable checksum of all dense-simplex coefficients consumed.
        coefficient_checksum: u64,
        /// Whether exhaustive cyclic fibers were affordable and agreed.
        independently_verified: bool,
    },
}

impl PointCountEvidence {
    /// Mathematical route used by this evidence.
    pub fn method(&self) -> PointCountMethod {
        match self.inner {
            PointCountEvidenceWire::ExhaustiveScheme { .. } => PointCountMethod::ExhaustiveScheme,
            PointCountEvidenceWire::ExhaustiveCyclicFibers { .. } => {
                PointCountMethod::ExhaustiveCyclicFibers
            }
            PointCountEvidenceWire::PadicDoubleCoverTrace { .. } => {
                PointCountMethod::PadicDoubleCoverTrace
            }
        }
    }

    /// Exact extension field used for this count.
    pub fn extension_field(&self) -> &FiniteField {
        match &self.inner {
            PointCountEvidenceWire::ExhaustiveScheme {
                extension_field, ..
            }
            | PointCountEvidenceWire::ExhaustiveCyclicFibers {
                extension_field, ..
            }
            | PointCountEvidenceWire::PadicDoubleCoverTrace {
                extension_field, ..
            } => extension_field,
        }
    }

    /// Checked embedding of the declared finite base field.
    pub fn embedding(&self) -> &FiniteFieldEmbedding {
        match &self.inner {
            PointCountEvidenceWire::ExhaustiveScheme { embedding, .. }
            | PointCountEvidenceWire::ExhaustiveCyclicFibers { embedding, .. }
            | PointCountEvidenceWire::PadicDoubleCoverTrace { embedding, .. } => embedding,
        }
    }

    /// Enumeration certificate retained by an exhaustive route.
    pub fn certificate(&self) -> Option<&PointCountCertificate> {
        match &self.inner {
            PointCountEvidenceWire::ExhaustiveScheme { certificate, .. } => Some(certificate),
            PointCountEvidenceWire::ExhaustiveCyclicFibers {
                base_certificate, ..
            } => Some(base_certificate),
            PointCountEvidenceWire::PadicDoubleCoverTrace { .. } => None,
        }
    }

    /// Number of p-adic digits used by a trace-formula route.
    pub fn precision(&self) -> Option<u64> {
        match self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace { precision, .. } => Some(precision),
            _ => None,
        }
    }

    /// Exact p-adic modulus used by a trace-formula route.
    pub fn modulus(&self) -> Option<&Integer> {
        match &self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace { modulus, .. } => Some(modulus),
            _ => None,
        }
    }

    /// Canonical count residue used by a trace-formula route.
    pub fn count_residue(&self) -> Option<&Integer> {
        match &self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace { count_residue, .. } => {
                Some(count_residue)
            }
            _ => None,
        }
    }

    /// Weil trace bound used by a trace-formula route.
    pub fn trace_bound(&self) -> Option<&Integer> {
        match &self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace { trace_bound, .. } => Some(trace_bound),
            _ => None,
        }
    }

    /// Number of projective coordinate-support strata in a trace route.
    pub fn stratum_count(&self) -> Option<usize> {
        match self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace { stratum_count, .. } => {
                Some(stratum_count)
            }
            _ => None,
        }
    }

    /// Largest trace-operator matrix dimension in a trace route.
    pub fn largest_matrix_dimension(&self) -> Option<usize> {
        match self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace {
                largest_matrix_dimension,
                ..
            } => Some(largest_matrix_dimension),
            _ => None,
        }
    }

    /// Stable checksum of coefficients consumed by a trace route.
    pub fn coefficient_checksum(&self) -> Option<u64> {
        match self.inner {
            PointCountEvidenceWire::PadicDoubleCoverTrace {
                coefficient_checksum,
                ..
            } => Some(coefficient_checksum),
            _ => None,
        }
    }

    /// Whether the count is exhaustively replayable or an additional route
    /// independently checked an accelerated computation.
    pub fn independently_verified(&self) -> bool {
        match self.inner {
            PointCountEvidenceWire::ExhaustiveScheme { .. } => true,
            PointCountEvidenceWire::ExhaustiveCyclicFibers {
                independently_verified,
                ..
            }
            | PointCountEvidenceWire::PadicDoubleCoverTrace {
                independently_verified,
                ..
            } => independently_verified,
        }
    }
}

/// One exact point count over a relative extension degree.
///
/// Records are obtained from a verified [`PointCountSequence`]. They do not
/// implement standalone deserialization because their evidence cannot be
/// replayed without the sequence's retained target.
#[derive(Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtensionPointCount {
    degree: u64,
    count: Integer,
    evidence: PointCountEvidence,
}

impl ExtensionPointCount {
    /// Degree over the declared finite base field.
    pub fn degree(&self) -> u64 {
        self.degree
    }

    /// Exact number of rational points over that extension.
    pub fn count(&self) -> &Integer {
        &self.count
    }

    /// Replayable method-specific evidence.
    pub fn evidence(&self) -> &PointCountEvidence {
        &self.evidence
    }
}

/// Exact counts over a requested increasing sequence of extension degrees.
#[derive(Clone, PartialEq, Eq)]
pub struct PointCountSequence {
    target: PointCountTarget,
    counts: Vec<ExtensionPointCount>,
}

impl PointCountSequence {
    /// Scheme presentation being counted.
    pub fn target(&self) -> &PointCountTarget {
        &self.target
    }

    /// Explicit finite base field.
    pub fn base(&self) -> &FiniteField {
        self.target.base()
    }

    /// Counts in strictly increasing requested degree order.
    pub fn counts(&self) -> &[ExtensionPointCount] {
        &self.counts
    }

    /// Return the exact count at one relative degree, when present.
    pub fn count(&self, degree: u64) -> Option<&Integer> {
        self.counts
            .binary_search_by_key(&degree, ExtensionPointCount::degree)
            .ok()
            .map(|index| self.counts[index].count())
    }

    /// Re-run every exact count and compare all evidence.
    pub fn verify(&self) -> Result<(), PointCountError> {
        validate_degrees(self.counts.iter().map(ExtensionPointCount::degree))?;
        let rebuilt = count_target(
            self.target.clone(),
            self.counts.iter().map(ExtensionPointCount::degree),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(PointCountError::InvalidEvidence(
                "recomputation changed a count or its transcript".to_owned(),
            ))
        }
    }

    /// Serialize the target, counts, and complete evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PointCountSequenceWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and exactly reverify a point-count sequence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PointCountSequenceWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let sequence = Self {
            target: wire.target,
            counts: wire.counts.into_iter().map(Into::into).collect(),
        };
        sequence
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(sequence)
    }
}

impl Scheme<FiniteField> {
    /// Count points over each requested relative extension degree.
    pub fn point_counts<I>(&self, degrees: I) -> Result<PointCountSequence, PointCountError>
    where
        I: IntoIterator<Item = u64>,
    {
        count_target(PointCountTarget::Scheme(Box::new(self.clone())), degrees)
    }
}

impl CyclicHypersurfaceCover<FiniteField> {
    /// Count points over each requested relative extension degree.
    ///
    /// The operation chooses the admitted mathematical route; callers do not
    /// name a backend. Exhaustive cyclic fibers are the current correctness
    /// floor and retain an independent direct-root scan when affordable.
    pub fn point_counts<I>(&self, degrees: I) -> Result<PointCountSequence, PointCountError>
    where
        I: IntoIterator<Item = u64>,
    {
        count_target(
            PointCountTarget::CyclicCover(Box::new(self.clone())),
            degrees,
        )
    }
}

fn count_target<I>(
    target: PointCountTarget,
    degrees: I,
) -> Result<PointCountSequence, PointCountError>
where
    I: IntoIterator<Item = u64>,
{
    let degrees = degrees.into_iter().collect::<Vec<_>>();
    validate_degrees(degrees.iter().copied())?;
    let counts = match &target {
        PointCountTarget::Scheme(scheme) => degrees
            .into_iter()
            .map(|degree| count_scheme_extension(scheme, degree))
            .collect::<Result<Vec<_>, _>>()?,
        PointCountTarget::CyclicCover(cover) => count_cyclic_degrees(cover, &degrees)?,
    };
    Ok(PointCountSequence { target, counts })
}

fn validate_degrees<I>(degrees: I) -> Result<(), PointCountError>
where
    I: IntoIterator<Item = u64>,
{
    let mut previous = 0_u64;
    let mut any = false;
    for degree in degrees {
        if degree == 0 || degree <= previous {
            return Err(PointCountError::InvalidExtensionDegrees);
        }
        previous = degree;
        any = true;
    }
    if !any {
        return Err(PointCountError::InvalidExtensionDegrees);
    }
    Ok(())
}

fn count_scheme_extension(
    scheme: &Scheme<FiniteField>,
    degree: u64,
) -> Result<ExtensionPointCount, PointCountError> {
    let (extension_field, embedding) = extension_field(scheme.base(), degree)?;
    require_exhaustive_field(&extension_field)?;
    let changed = scheme.base_change(&embedding)?;
    let (count, certificate) = changed.point_count()?;
    certificate.verify(&changed)?;
    Ok(ExtensionPointCount {
        degree,
        count,
        evidence: PointCountEvidence {
            inner: PointCountEvidenceWire::ExhaustiveScheme {
                extension_field,
                embedding,
                certificate,
            },
        },
    })
}

pub(super) fn count_cyclic_extension(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    degree: u64,
) -> Result<ExtensionPointCount, PointCountError> {
    let (extension_field, embedding) = extension_field(cover.base().base(), degree)?;
    let field_order = require_exhaustive_field(&extension_field)?;
    let base = cover.base().base_change(&embedding)?;
    let branch = cover.branch_equation().base_change(&embedding)?;
    let empty_ideal = base
        .coordinate_ring()
        .ideal([])
        .map_err(SchemeError::from)?;
    let projective_scheme = base.closed_subscheme(empty_ideal)?;
    let (base_count, base_certificate) = projective_scheme.point_count()?;
    let base_count_u64 = integer_to_u64(&base_count, "projective base point count")?;
    if base_count_u64 > MAX_PROJECTIVE_CANDIDATES {
        return Err(PointCountError::ResourceLimit {
            resource: "projective base point count",
            limit: MAX_PROJECTIVE_CANDIDATES,
        });
    }

    let direct_tests = base_count_u64.checked_mul(field_order);
    let independently_verified =
        direct_tests.is_some_and(|tests| tests <= MAX_INDEPENDENT_FIBER_TESTS);
    let mut count = Integer::zero();
    let mut fiber_checksum = FNV_OFFSET;
    for (index, point) in projective_scheme.rational_points()?.enumerate() {
        let value = branch
            .evaluate(point.coordinates())
            .map_err(SchemeError::from)?;
        let roots = cyclic_root_count(&extension_field, &value, cover.cover_degree(), field_order)?;
        if independently_verified {
            let direct = extension_field
                .elements()
                .filter(|candidate| candidate.pow(cover.cover_degree()) == value)
                .count();
            if direct as u64 != roots {
                return Err(PointCountError::InvalidEvidence(
                    "cyclic root formula disagrees with direct fiber enumeration".to_owned(),
                ));
            }
        }
        count = &count + &Integer::from(roots);
        checksum_u64(&mut fiber_checksum, index as u64);
        checksum_u64(&mut fiber_checksum, roots);
    }
    base_certificate.verify(&projective_scheme)?;
    Ok(ExtensionPointCount {
        degree,
        count,
        evidence: PointCountEvidence {
            inner: PointCountEvidenceWire::ExhaustiveCyclicFibers {
                extension_field,
                embedding,
                base_certificate,
                fiber_checksum,
                independently_verified,
            },
        },
    })
}

fn count_cyclic_degrees(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    degrees: &[u64],
) -> Result<Vec<ExtensionPointCount>, PointCountError> {
    if degrees
        .iter()
        .copied()
        .all(|degree| cyclic_exhaustive_is_affordable(cover, degree))
    {
        return degrees
            .iter()
            .copied()
            .map(|degree| count_cyclic_extension(cover, degree))
            .collect();
    }
    let transcript = count_projective_double_cover(cover, degrees)?;
    transcript
        .counts
        .into_iter()
        .map(|record| {
            let (extension_field, embedding) = extension_field(cover.base().base(), record.degree)?;
            let independently_verified = if cyclic_exhaustive_is_affordable(cover, record.degree) {
                let exhaustive = count_cyclic_extension(cover, record.degree)?;
                if exhaustive.count != record.count {
                    return Err(PointCountError::TraceFormulaFailure(format!(
                        "accelerated and exhaustive counts disagree in extension degree {}",
                        record.degree
                    )));
                }
                true
            } else {
                false
            };
            Ok(ExtensionPointCount {
                degree: record.degree,
                count: record.count,
                evidence: PointCountEvidence {
                    inner: PointCountEvidenceWire::PadicDoubleCoverTrace {
                        extension_field,
                        embedding,
                        precision: record.precision,
                        modulus: record.modulus,
                        count_residue: record.count_residue,
                        trace_bound: record.trace_bound,
                        stratum_count: transcript.stratum_count,
                        largest_matrix_dimension: transcript.largest_matrix_dimension,
                        coefficient_checksum: transcript.coefficient_checksum,
                        independently_verified,
                    },
                },
            })
        })
        .collect()
}

fn cyclic_exhaustive_is_affordable(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    degree: u64,
) -> bool {
    let order = cover.base().base().order().pow(degree);
    let Some(order_word) = order.to_u64_checked() else {
        return false;
    };
    if order_word > MAX_EXTENSION_FIELD_ORDER {
        return false;
    }
    let dimension = cover.base().dimension() as u64;
    let numerator = &order.pow(dimension + 1) - &Integer::one();
    let denominator = &order - &Integer::one();
    (&numerator / &denominator)
        .to_u64_checked()
        .is_some_and(|candidates| candidates <= MAX_PROJECTIVE_CANDIDATES)
}

fn cyclic_root_count(
    field: &FiniteField,
    value: &crate::numthy::FiniteFieldElement,
    exponent: u64,
    field_order: u64,
) -> Result<u64, PointCountError> {
    if value.is_zero() {
        return Ok(1);
    }
    let group_order = field_order - 1;
    let divisor = gcd_u64(exponent, group_order);
    if value.pow(group_order / divisor) == field.one() {
        Ok(divisor)
    } else {
        Ok(0)
    }
}

fn extension_field(
    base: &FiniteField,
    relative_degree: u64,
) -> Result<(FiniteField, FiniteFieldEmbedding), PointCountError> {
    let relative_degree =
        usize::try_from(relative_degree).map_err(|_| PointCountError::ResourceLimit {
            resource: "relative extension degree",
            limit: usize::MAX as u64,
        })?;
    let absolute_degree =
        base.degree()
            .checked_mul(relative_degree)
            .ok_or(PointCountError::ResourceLimit {
                resource: "absolute extension degree",
                limit: usize::MAX as u64,
            })?;
    let characteristic = base
        .characteristic()
        .to_string()
        .parse::<u64>()
        .map_err(|_| PointCountError::ResourceLimit {
            resource: "finite-field word characteristic",
            limit: u64::MAX,
        })?;
    let extension = if relative_degree == 1 {
        base.clone()
    } else {
        FiniteField::extension(characteristic, absolute_degree)?
    };
    let embedding = deterministic_finite_field_embedding(base, &extension)?;
    Ok((extension, embedding))
}

fn require_exhaustive_field(field: &FiniteField) -> Result<u64, PointCountError> {
    let order = integer_to_u64(&field.order(), "extension field order")?;
    if order > MAX_EXTENSION_FIELD_ORDER {
        return Err(PointCountError::ResourceLimit {
            resource: "extension field order",
            limit: MAX_EXTENSION_FIELD_ORDER,
        });
    }
    Ok(order)
}

fn integer_to_u64(value: &Integer, resource: &'static str) -> Result<u64, PointCountError> {
    value
        .to_string()
        .parse::<u64>()
        .map_err(|_| PointCountError::ResourceLimit {
            resource,
            limit: u64::MAX,
        })
}

fn checksum_u64(checksum: &mut u64, value: u64) {
    for byte in value.to_be_bytes() {
        *checksum ^= u64::from(byte);
        *checksum = checksum.wrapping_mul(FNV_PRIME);
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointCountSequenceWire {
    #[serde(rename = "type")]
    type_id: String,
    target: PointCountTarget,
    counts: Vec<ExtensionPointCountWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtensionPointCountWire {
    degree: u64,
    count: Integer,
    evidence: PointCountEvidenceWire,
}

impl From<&ExtensionPointCount> for ExtensionPointCountWire {
    fn from(record: &ExtensionPointCount) -> Self {
        Self {
            degree: record.degree,
            count: record.count.clone(),
            evidence: record.evidence.inner.clone(),
        }
    }
}

impl From<ExtensionPointCountWire> for ExtensionPointCount {
    fn from(record: ExtensionPointCountWire) -> Self {
        Self {
            degree: record.degree,
            count: record.count,
            evidence: PointCountEvidence {
                inner: record.evidence,
            },
        }
    }
}

impl From<&PointCountSequence> for PointCountSequenceWire {
    fn from(sequence: &PointCountSequence) -> Self {
        Self {
            type_id: PointCountSequence::TYPE_ID.to_owned(),
            target: sequence.target.clone(),
            counts: sequence
                .counts
                .iter()
                .map(ExtensionPointCountWire::from)
                .collect(),
        }
    }
}

impl Serialize for PointCountSequence {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        PointCountSequenceWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PointCountSequence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PointCountSequenceWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let sequence = Self {
            target: wire.target,
            counts: wire.counts.into_iter().map(Into::into).collect(),
        };
        sequence.verify().map_err(serde::de::Error::custom)?;
        Ok(sequence)
    }
}

impl SemanticType for PointCountSequence {
    const TYPE_ID: &'static str = "noether/commalg/PointCountSequence@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &POINT_COUNT_SEQUENCE_DESCRIPTION
    }
}

impl fmt::Debug for PointCountSequence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PointCountSequence")
            .field("target", &self.target)
            .field(
                "degrees",
                &self
                    .counts
                    .iter()
                    .map(ExtensionPointCount::degree)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl fmt::Display for PointCountSequence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PointCountSequence(base={}, degrees={:?})",
            self.base(),
            self.counts
                .iter()
                .map(ExtensionPointCount::degree)
                .collect::<Vec<_>>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{MonomialOrder, Polynomial, PolynomialRing, ProjectiveSpace};

    #[test]
    fn ordinary_scheme_counts_extend_and_replay() {
        let field = FiniteField::prime(3).unwrap();
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let affine = super::super::AffineSpace::with_names(field, 1, ["x"]).unwrap();
        let equation = ring.sub(&x.pow(2).unwrap(), &ring.one()).unwrap();
        let scheme = affine
            .closed_subscheme(ring.ideal([equation]).unwrap())
            .unwrap();
        let counts = scheme.point_counts([1, 2]).unwrap();
        assert_eq!(counts.count(1), Some(&Integer::from(2)));
        assert_eq!(counts.count(2), Some(&Integer::from(2)));
        counts.verify().unwrap();
        assert_eq!(
            PointCountSequence::from_json(&counts.to_json().unwrap()).unwrap(),
            counts
        );

        let mut forged =
            serde_json::from_str::<serde_json::Value>(&counts.to_json().unwrap()).unwrap();
        forged["counts"][0]["count"] = serde_json::json!("3");
        assert!(PointCountSequence::from_json(&forged.to_string()).is_err());
        assert!(serde_json::from_str::<PointCountSequence>(&forged.to_string()).is_err());
    }

    #[test]
    fn cyclic_fibers_have_an_independent_small_field_check() {
        let field = FiniteField::prime(3).unwrap();
        let base = ProjectiveSpace::with_names(field, 1, ["x", "y"]).unwrap();
        let [x, y]: [Polynomial<FiniteField>; 2] = base
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let branch = base
            .coordinate_ring()
            .add(&x.pow(4).unwrap(), &y.pow(4).unwrap())
            .unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(base, 2, branch).unwrap();
        let counts = cover.point_counts([1, 2]).unwrap();
        assert!(counts
            .counts()
            .iter()
            .all(|count| count.evidence().independently_verified()));
        counts.verify().unwrap();

        let mut forged =
            serde_json::from_str::<serde_json::Value>(&counts.to_json().unwrap()).unwrap();
        forged["counts"][0]["evidence"]["independently_verified"] = serde_json::json!(false);
        assert!(PointCountSequence::from_json(&forged.to_string()).is_err());
        assert!(serde_json::from_str::<PointCountSequence>(&forged.to_string()).is_err());
    }

    #[test]
    fn cyclic_cover_dispatches_to_the_trace_formula_beyond_the_exhaustive_floor() {
        let field = FiniteField::prime(11_u8).unwrap();
        let base = ProjectiveSpace::with_names(field, 2, ["x", "y", "z"]).unwrap();
        let [x, y, z]: [Polynomial<FiniteField>; 3] = base
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let branch = (&(&x * &z).unwrap() - &y.pow(2).unwrap()).unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(base, 2, branch).unwrap();

        let exhaustive = cover.point_counts([1]).unwrap();
        assert_eq!(
            exhaustive.counts()[0].evidence().method(),
            PointCountMethod::ExhaustiveCyclicFibers
        );
        let frobenius = cover
            .surface_cohomology()
            .unwrap()
            .frobenius_polynomial(&exhaustive)
            .unwrap();
        let expected =
            &(&Integer::one() + &frobenius.trace(4).unwrap()) + &Integer::from(11).pow(8);

        let accelerated = cover.point_counts([4]).unwrap();
        assert_eq!(
            accelerated.counts()[0].evidence().method(),
            PointCountMethod::PadicDoubleCoverTrace
        );
        assert!(!accelerated.counts()[0].evidence().independently_verified());
        assert_eq!(accelerated.count(4), Some(&expected));
        accelerated.verify().unwrap();
        assert_eq!(
            PointCountSequence::from_json(&accelerated.to_json().unwrap()).unwrap(),
            accelerated
        );
    }
}
