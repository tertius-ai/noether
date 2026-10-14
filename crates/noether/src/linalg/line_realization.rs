// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified realization of integral lines as a complete lattice minimum shell.

use core::fmt;
use std::time::Instant;

use crate::geometry::exact_lp::{
    certified_upper_bound_from_dual_free, maximize_free, LinearProgramResult, NativeLpProposal,
};
use crate::{Integer, NoetherError, Rational, SemanticType, TypeDescription, Q, Z};

use super::{
    IntegralLineConfiguration, Lattice, LatticeError, LineConfigurationError, Matrix,
    MinimumVectorConfiguration,
};

static MINIMUM_REALIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumRealization@1",
    rust_path: "noether::linalg::MinimumRealization",
    python_path: Some("noether.linalg.MinimumRealization"),
};

static MINIMUM_NONREALIZABILITY_CERTIFICATE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumNonrealizabilityCertificate@1",
    rust_path: "noether::linalg::MinimumNonrealizabilityCertificate",
    python_path: Some("noether.linalg.MinimumNonrealizabilityCertificate"),
};

static MINIMUM_REALIZATION_DIAGNOSTIC_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumRealizationDiagnostic@1",
    rust_path: "noether::linalg::MinimumRealizationDiagnostic",
    python_path: Some("noether.linalg.MinimumRealizationDiagnostic"),
};

static MINIMUM_COMPLETION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumCompletion@1",
    rust_path: "noether::linalg::MinimumCompletion",
    python_path: Some("noether.linalg.MinimumCompletion"),
};

static MINIMUM_COMPLETION_CERTIFICATE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/MinimumCompletionCertificate@1",
    rust_path: "noether::linalg::MinimumCompletionCertificate",
    python_path: Some("noether.linalg.MinimumCompletionCertificate"),
};

/// Opt-in timing for diagnosing hard cutting-plane cases. This stays outside
/// the mathematical evidence: it writes only process-local phase measurements
/// to stderr when `NOETHER_LINE_REALIZATION_PROFILE=1`.
struct LineRealizationProfile {
    operation: &'static str,
    started: Option<Instant>,
}

impl LineRealizationProfile {
    fn new(operation: &'static str) -> Self {
        let enabled =
            std::env::var_os("NOETHER_LINE_REALIZATION_PROFILE").is_some_and(|value| value == "1");
        Self {
            operation,
            started: enabled.then(Instant::now),
        }
    }

    fn enabled(&self) -> bool {
        self.started.is_some()
    }

    fn phase_start(&self) -> Option<Instant> {
        self.started.as_ref().map(|_| Instant::now())
    }

    fn event(
        &self,
        iteration: Option<usize>,
        phase: &str,
        phase_started: Option<Instant>,
        line_count: usize,
        obstacle_count: usize,
        detail: fmt::Arguments<'_>,
    ) {
        let Some(started) = &self.started else {
            return;
        };
        let iteration = iteration
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_owned());
        let phase_ms = phase_started.map_or(0, |value| value.elapsed().as_millis());
        eprintln!(
            "noether_line_realization_profile operation={} iteration={} phase={} \
             phase_ms={} total_ms={} lines={} obstacles={} {}",
            self.operation,
            iteration,
            phase,
            phase_ms,
            started.elapsed().as_millis(),
            line_count,
            obstacle_count,
            detail,
        );
    }
}

fn integer_decimal_digits(value: &Integer) -> usize {
    value.to_string().trim_start_matches('-').len()
}

fn rational_gram_decimal_digits(gram: &[Vec<Rational>]) -> (usize, usize) {
    gram.iter()
        .flatten()
        .fold((1, 1), |(maximum_numerator, maximum_denominator), entry| {
            (
                maximum_numerator.max(integer_decimal_digits(&entry.numerator())),
                maximum_denominator.max(integer_decimal_digits(&entry.denominator())),
            )
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NegativeCertificateKind {
    EqualityInconsistent,
    Infeasible,
    NonpositiveMargin,
    HomogeneousMinimum,
}

/// Exact rational evidence that no positive-definite form has the requested
/// complete minimum-vector configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimumNonrealizabilityCertificate {
    configuration: IntegralLineConfiguration,
    obstacles: Vec<Vec<Integer>>,
    multipliers: Vec<Rational>,
    kind: NegativeCertificateKind,
}

impl MinimumNonrealizabilityCertificate {
    /// Candidate configuration proved nonrealizable.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        &self.configuration
    }

    /// Stable kind of rational certificate.
    pub fn kind(&self) -> &'static str {
        match self.kind {
            NegativeCertificateKind::EqualityInconsistent => "inconsistent_equal_norms",
            NegativeCertificateKind::Infeasible => "farkas_infeasible",
            NegativeCertificateKind::NonpositiveMargin => "nonpositive_outsider_margin",
            NegativeCertificateKind::HomogeneousMinimum => "homogeneous_minimum_at_least_one",
        }
    }

    /// Number of finite separating vectors used by the certificate.
    pub fn obstacle_count(&self) -> usize {
        self.obstacles.len()
    }

    /// Recheck the rational Farkas or dual-bound identity exactly.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        if self.kind == NegativeCertificateKind::EqualityInconsistent {
            if self.multipliers.len() != self.configuration.line_count() {
                return Err(realization_error(
                    "equal-norm inconsistency multipliers have the wrong shape",
                ));
            }
            let symmetric_dimension = self.configuration.symmetric_dimension();
            let columns = (0..symmetric_dimension)
                .map(|column| {
                    self.configuration
                        .vectors()
                        .iter()
                        .zip(&self.multipliers)
                        .fold(Rational::zero(), |sum, (vector, multiplier)| {
                            &sum + &(&quadratic_row(vector, self.configuration.ambient_rank())
                                [column]
                                * multiplier)
                        })
                })
                .collect::<Vec<_>>();
            let rhs = self
                .multipliers
                .iter()
                .fold(Rational::zero(), |sum, multiplier| &sum + multiplier);
            if columns.iter().any(|entry| !entry.is_zero()) || rhs.is_zero() {
                return Err(realization_error(
                    "invalid equal-norm inconsistency identity",
                ));
            }
            return Ok(());
        }
        if self.kind == NegativeCertificateKind::HomogeneousMinimum {
            let problem = HomogeneousProblem::new(&self.configuration, &self.obstacles);
            verify_dual_bound(
                &problem.rows,
                &problem.rhs,
                &problem.objective,
                &self.multipliers,
                &Rational::from(-1_i64),
                "invalid dual bound for the homogeneous minimum",
            )?;
            return Ok(());
        }
        let problem = RealizationProblem::new(&self.configuration, self.obstacles.clone())
            .map_err(|_| realization_error("certificate unexpectedly has inconsistent norms"))?;
        if self.multipliers.len() != problem.rhs.len()
            || self.multipliers.iter().any(Rational::is_negative)
        {
            return Err(realization_error(
                "nonrealizability multipliers have the wrong shape or sign",
            ));
        }
        let dual_columns = (0..problem.objective.len())
            .map(|column| {
                problem
                    .rows
                    .iter()
                    .zip(&self.multipliers)
                    .fold(Rational::zero(), |sum, (row, multiplier)| {
                        &sum + &(&row[column] * multiplier)
                    })
            })
            .collect::<Vec<_>>();
        let dual_value = problem
            .rhs
            .iter()
            .zip(&self.multipliers)
            .fold(Rational::zero(), |sum, (rhs, multiplier)| {
                &sum + &(rhs * multiplier)
            });
        match self.kind {
            NegativeCertificateKind::EqualityInconsistent => {
                return Err(realization_error(
                    "equal-norm certificate reached the finite-LP verifier",
                ));
            }
            NegativeCertificateKind::Infeasible => {
                if dual_columns.iter().any(|entry| !entry.is_zero()) || !dual_value.is_negative() {
                    return Err(realization_error("invalid Farkas infeasibility identity"));
                }
            }
            NegativeCertificateKind::NonpositiveMargin => {
                if dual_columns != problem.objective || dual_value.is_positive() {
                    return Err(realization_error(
                        "invalid dual bound for the outsider margin",
                    ));
                }
            }
            NegativeCertificateKind::HomogeneousMinimum => {
                return Err(realization_error(
                    "homogeneous certificate reached the outsider-margin verifier",
                ));
            }
        }
        Ok(())
    }
}

impl SemanticType for MinimumNonrealizabilityCertificate {
    const TYPE_ID: &'static str = "noether/linalg/MinimumNonrealizabilityCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let obstacles = integer_rows_as_strings(&self.obstacles);
        let multipliers = self
            .multipliers
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.configuration.canonical_bytes()?,
            self.kind(),
            obstacles,
            multipliers,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_NONREALIZABILITY_CERTIFICATE_DESCRIPTION
    }
}

impl fmt::Debug for MinimumNonrealizabilityCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumNonrealizabilityCertificate")
            .field("kind", &self.kind())
            .field("obstacle_count", &self.obstacle_count())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for MinimumNonrealizabilityCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumNonrealizabilityCertificate(kind={}, obstacles={})",
            self.kind(),
            self.obstacle_count()
        )
    }
}

/// Evidence-free diagnostic for an inconclusive minimum-realization run.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimumRealizationDiagnostic {
    configuration: IntegralLineConfiguration,
    obstacle_count: usize,
    iterations: usize,
    reason: String,
}

impl MinimumRealizationDiagnostic {
    /// Candidate whose realization remains undecided.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        &self.configuration
    }

    /// Completed cutting-plane iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Number of exact separating vectors accumulated before stopping.
    pub fn obstacle_count(&self) -> usize {
        self.obstacle_count
    }

    /// Stable explanation of why no mathematical verdict was returned.
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Check that the diagnostic makes no false terminal claim.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        if self.reason.is_empty() {
            return Err(realization_error("inconclusive diagnostic has no reason"));
        }
        Ok(())
    }
}

impl SemanticType for MinimumRealizationDiagnostic {
    const TYPE_ID: &'static str = "noether/linalg/MinimumRealizationDiagnostic@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.configuration.canonical_bytes()?,
            self.obstacle_count,
            self.iterations,
            &self.reason,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_REALIZATION_DIAGNOSTIC_DESCRIPTION
    }
}

impl fmt::Debug for MinimumRealizationDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumRealizationDiagnostic")
            .field("iterations", &self.iterations)
            .field("obstacle_count", &self.obstacle_count())
            .field("reason", &self.reason)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for MinimumRealizationDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumRealizationDiagnostic(iterations={}, obstacles={}, reason={})",
            self.iterations,
            self.obstacle_count(),
            self.reason
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
enum MinimumRealizationOutcome {
    Realized(MinimumVectorConfiguration),
    Nonrealizable(MinimumNonrealizabilityCertificate),
    Inconclusive(MinimumRealizationDiagnostic),
}

/// Three-way certified result of the minimum-vector realization playbook.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimumRealization {
    outcome: MinimumRealizationOutcome,
}

impl MinimumRealization {
    /// Terminal state: `realized`, `nonrealizable`, or `inconclusive`.
    pub fn status(&self) -> &'static str {
        match self.outcome {
            MinimumRealizationOutcome::Realized(_) => "realized",
            MinimumRealizationOutcome::Nonrealizable(_) => "nonrealizable",
            MinimumRealizationOutcome::Inconclusive(_) => "inconclusive",
        }
    }

    /// Candidate configuration common to every outcome.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        match &self.outcome {
            MinimumRealizationOutcome::Realized(value) => value.configuration(),
            MinimumRealizationOutcome::Nonrealizable(value) => value.configuration(),
            MinimumRealizationOutcome::Inconclusive(value) => value.configuration(),
        }
    }

    /// Positive Gram and complete-enumeration evidence, when realized.
    pub fn realization(&self) -> Option<&MinimumVectorConfiguration> {
        match &self.outcome {
            MinimumRealizationOutcome::Realized(value) => Some(value),
            _ => None,
        }
    }

    /// Rational negative evidence, when proved nonrealizable.
    pub fn certificate(&self) -> Option<&MinimumNonrealizabilityCertificate> {
        match &self.outcome {
            MinimumRealizationOutcome::Nonrealizable(value) => Some(value),
            _ => None,
        }
    }

    /// Diagnostic information, when the computation is inconclusive.
    pub fn diagnostic(&self) -> Option<&MinimumRealizationDiagnostic> {
        match &self.outcome {
            MinimumRealizationOutcome::Inconclusive(value) => Some(value),
            _ => None,
        }
    }

    /// Verify the mathematical evidence appropriate to the outcome.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        match &self.outcome {
            MinimumRealizationOutcome::Realized(value) => value
                .verify()
                .map_err(|error| realization_error(error.to_string())),
            MinimumRealizationOutcome::Nonrealizable(value) => value.verify(),
            MinimumRealizationOutcome::Inconclusive(value) => value.verify(),
        }
    }
}

impl SemanticType for MinimumRealization {
    const TYPE_ID: &'static str = "noether/linalg/MinimumRealization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let evidence = match &self.outcome {
            MinimumRealizationOutcome::Realized(value) => value.canonical_bytes()?,
            MinimumRealizationOutcome::Nonrealizable(value) => value.canonical_bytes()?,
            MinimumRealizationOutcome::Inconclusive(value) => value.canonical_bytes()?,
        };
        serde_json::to_vec(&(Self::TYPE_ID, self.status(), evidence))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_REALIZATION_DESCRIPTION
    }
}

impl fmt::Debug for MinimumRealization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumRealization")
            .field("status", &self.status())
            .field("configuration", self.configuration())
            .finish()
    }
}

impl fmt::Display for MinimumRealization {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumRealization(status={}, rank={}, lines={})",
            self.status(),
            self.configuration().ambient_rank(),
            self.configuration().line_count()
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
enum MinimumCompletionCertificateEvidence {
    Obstructed(MinimumNonrealizabilityCertificate),
    ForcedShorter {
        vector: Vec<Integer>,
        coefficients: Vec<Rational>,
    },
}

/// Exact evidence that a line configuration has no minimum-shell completion
/// of the same perfection rank.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimumCompletionCertificate {
    seed: IntegralLineConfiguration,
    evidence: MinimumCompletionCertificateEvidence,
}

impl MinimumCompletionCertificate {
    /// Configuration whose fixed-rank completion was requested.
    pub fn seed(&self) -> &IntegralLineConfiguration {
        &self.seed
    }

    /// Stable certificate kind.
    pub fn kind(&self) -> &'static str {
        match &self.evidence {
            MinimumCompletionCertificateEvidence::Obstructed(evidence) => evidence.kind(),
            MinimumCompletionCertificateEvidence::ForcedShorter { .. } => "forced_shorter_line",
        }
    }

    /// Number of exact separating vectors retained by the evidence.
    pub fn obstacle_count(&self) -> usize {
        match &self.evidence {
            MinimumCompletionCertificateEvidence::Obstructed(evidence) => evidence.obstacle_count(),
            MinimumCompletionCertificateEvidence::ForcedShorter { .. } => 1,
        }
    }

    /// Recheck the fixed-rank obstruction exactly.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        match &self.evidence {
            MinimumCompletionCertificateEvidence::Obstructed(evidence) => {
                evidence.verify()?;
                if !configuration_contains(evidence.configuration(), &self.seed)
                    || evidence.configuration().perfection_rank() != self.seed.perfection_rank()
                {
                    return Err(realization_error(
                        "completion obstruction does not extend the seed at fixed rank",
                    ));
                }
                for obstacle in &evidence.obstacles {
                    if rank_one_relation(&self.seed, obstacle)?.is_some() {
                        return Err(realization_error(
                            "completion obstruction excludes a line inside the seed rank-one span",
                        ));
                    }
                }
                Ok(())
            }
            MinimumCompletionCertificateEvidence::ForcedShorter {
                vector,
                coefficients,
            } => {
                let expected = quadratic_row(vector, self.seed.ambient_rank());
                if coefficients.len() != self.seed.line_count() {
                    return Err(realization_error(
                        "forced-shorter relation has the wrong coefficient count",
                    ));
                }
                let rebuilt = (0..expected.len())
                    .map(|column| {
                        self.seed.vectors().iter().zip(coefficients).fold(
                            Rational::zero(),
                            |sum, (line, coefficient)| {
                                &sum + &(&quadratic_row(line, self.seed.ambient_rank())[column]
                                    * coefficient)
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                let forced_value = coefficients
                    .iter()
                    .fold(Rational::zero(), |sum, value| &sum + value);
                if rebuilt != expected || forced_value >= Rational::one() {
                    return Err(realization_error(
                        "invalid relation proving a forced shorter line",
                    ));
                }
                Ok(())
            }
        }
    }
}

impl SemanticType for MinimumCompletionCertificate {
    const TYPE_ID: &'static str = "noether/linalg/MinimumCompletionCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let evidence = match &self.evidence {
            MinimumCompletionCertificateEvidence::Obstructed(value) => value.canonical_bytes()?,
            MinimumCompletionCertificateEvidence::ForcedShorter {
                vector,
                coefficients,
            } => serde_json::to_vec(&(
                integer_rows_as_strings(core::slice::from_ref(vector)),
                coefficients
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            ))
            .map_err(|error| NoetherError::Serialization(error.to_string()))?,
        };
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.seed.canonical_bytes()?,
            self.kind(),
            evidence,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_COMPLETION_CERTIFICATE_DESCRIPTION
    }
}

impl fmt::Debug for MinimumCompletionCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumCompletionCertificate")
            .field("kind", &self.kind())
            .field("obstacle_count", &self.obstacle_count())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for MinimumCompletionCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumCompletionCertificate(kind={}, obstacles={})",
            self.kind(),
            self.obstacle_count()
        )
    }
}

#[derive(Clone, PartialEq, Eq)]
enum MinimumCompletionOutcome {
    Completed(MinimumVectorConfiguration),
    Nonrealizable(MinimumCompletionCertificate),
    Inconclusive(MinimumRealizationDiagnostic),
}

/// Three-way certified result of fixed-perfection-rank minimum completion.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimumCompletion {
    seed: IntegralLineConfiguration,
    outcome: MinimumCompletionOutcome,
}

impl MinimumCompletion {
    /// Terminal state: `completed`, `nonrealizable`, or `inconclusive`.
    pub fn status(&self) -> &'static str {
        match self.outcome {
            MinimumCompletionOutcome::Completed(_) => "completed",
            MinimumCompletionOutcome::Nonrealizable(_) => "nonrealizable",
            MinimumCompletionOutcome::Inconclusive(_) => "inconclusive",
        }
    }

    /// Configuration whose completion was requested.
    pub fn seed(&self) -> &IntegralLineConfiguration {
        &self.seed
    }

    /// Positive complete minimum-shell evidence, when found.
    pub fn realization(&self) -> Option<&MinimumVectorConfiguration> {
        match &self.outcome {
            MinimumCompletionOutcome::Completed(value) => Some(value),
            _ => None,
        }
    }

    /// Exact fixed-rank negative evidence, when proved impossible.
    pub fn certificate(&self) -> Option<&MinimumCompletionCertificate> {
        match &self.outcome {
            MinimumCompletionOutcome::Nonrealizable(value) => Some(value),
            _ => None,
        }
    }

    /// Diagnostic information, when the computation exhausted its work limit.
    pub fn diagnostic(&self) -> Option<&MinimumRealizationDiagnostic> {
        match &self.outcome {
            MinimumCompletionOutcome::Inconclusive(value) => Some(value),
            _ => None,
        }
    }

    /// Recheck the appropriate completion evidence exactly.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        match &self.outcome {
            MinimumCompletionOutcome::Completed(value) => {
                value
                    .verify()
                    .map_err(|error| realization_error(error.to_string()))?;
                if !configuration_contains(value.configuration(), &self.seed)
                    || value.configuration().perfection_rank() != self.seed.perfection_rank()
                {
                    return Err(realization_error(
                        "completed minimum shell does not extend the seed at fixed rank",
                    ));
                }
                Ok(())
            }
            MinimumCompletionOutcome::Nonrealizable(value) => value.verify(),
            MinimumCompletionOutcome::Inconclusive(value) => {
                value.verify()?;
                if !configuration_contains(value.configuration(), &self.seed)
                    || value.configuration().perfection_rank() != self.seed.perfection_rank()
                {
                    return Err(realization_error(
                        "completion diagnostic does not preserve fixed-rank state",
                    ));
                }
                Ok(())
            }
        }
    }
}

impl SemanticType for MinimumCompletion {
    const TYPE_ID: &'static str = "noether/linalg/MinimumCompletion@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let evidence = match &self.outcome {
            MinimumCompletionOutcome::Completed(value) => value.canonical_bytes()?,
            MinimumCompletionOutcome::Nonrealizable(value) => value.canonical_bytes()?,
            MinimumCompletionOutcome::Inconclusive(value) => value.canonical_bytes()?,
        };
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.seed.canonical_bytes()?,
            self.status(),
            evidence,
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MINIMUM_COMPLETION_DESCRIPTION
    }
}

impl fmt::Debug for MinimumCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimumCompletion")
            .field("status", &self.status())
            .field("seed", &self.seed)
            .finish()
    }
}

impl fmt::Display for MinimumCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimumCompletion(status={}, rank={}, lines={})",
            self.status(),
            self.seed.ambient_rank(),
            self.seed.line_count()
        )
    }
}

impl IntegralLineConfiguration {
    /// Run the exact cutting-plane playbook for realizing these lines as the
    /// complete minimum shell of a positive-definite integral lattice.
    ///
    /// A work limit produces an explicit inconclusive diagnostic. It is never
    /// interpreted as mathematical nonrealizability.
    pub fn realize_minimum(
        &self,
        max_iterations: usize,
    ) -> Result<MinimumRealization, LineConfigurationError> {
        let profile = LineRealizationProfile::new("exact_shell");
        profile.event(
            None,
            "start",
            None,
            self.line_count(),
            0,
            format_args!(
                "ambient_rank={} max_iterations={}",
                self.ambient_rank(),
                max_iterations
            ),
        );
        if self.ambient_rank() == 0 {
            return Err(LineConfigurationError::RankZeroRealization);
        }
        if !self.is_spanning() {
            return Err(LineConfigurationError::NonspanningRealization);
        }
        if max_iterations == 0 {
            return inconclusive(
                self,
                seed_obstacles(self),
                0,
                "cutting-plane iteration limit reached",
            );
        }
        let basis_started = profile.phase_start();
        let (basis_realization, basis_obstacles) = try_basis_realization(self)?;
        profile.event(
            None,
            "basis_probe",
            basis_started,
            self.line_count(),
            basis_obstacles.len(),
            format_args!("realized={}", basis_realization.is_some()),
        );
        if let Some(realization) = basis_realization {
            return Ok(MinimumRealization {
                outcome: MinimumRealizationOutcome::Realized(realization),
            });
        }
        let mut obstacles = seed_obstacles(self);
        for obstacle in &basis_obstacles {
            insert_obstacle(&mut obstacles, obstacle.clone());
        }
        let mut positivity_directions = Vec::new();
        let mut native_lp_proposal = NativeLpProposal::new();
        for iteration in 0..max_iterations {
            profile.event(
                Some(iteration),
                "iteration_start",
                None,
                self.line_count(),
                obstacles.len(),
                format_args!(""),
            );
            let problem_started = profile.phase_start();
            let problem = match RealizationProblem::new(self, obstacles.clone()) {
                Ok(problem) => {
                    profile.event(
                        Some(iteration),
                        "problem_build",
                        problem_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(
                            "lp_rows={} lp_variables={} positivity_cuts={}",
                            problem.rows.len(),
                            problem.objective.len(),
                            positivity_directions.len()
                        ),
                    );
                    problem
                }
                Err(multipliers) => {
                    profile.event(
                        Some(iteration),
                        "terminal_equality_inconsistent",
                        problem_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!("multiplier_count={}", multipliers.len()),
                    );
                    let evidence = MinimumNonrealizabilityCertificate {
                        configuration: self.clone(),
                        obstacles: Vec::new(),
                        multipliers,
                        kind: NegativeCertificateKind::EqualityInconsistent,
                    };
                    evidence.verify()?;
                    return Ok(MinimumRealization {
                        outcome: MinimumRealizationOutcome::Nonrealizable(evidence),
                    });
                }
            };
            let proposal_problem = problem.with_positive_directions(&positivity_directions);
            let proposal_started = profile.phase_start();
            profile.event(
                Some(iteration),
                "native_lp_proposal_start",
                None,
                self.line_count(),
                obstacles.len(),
                format_args!(
                    "lp_rows={} lp_variables={} positivity_cuts={}",
                    proposal_problem.rows.len(),
                    proposal_problem.objective.len(),
                    positivity_directions.len()
                ),
            );
            let proposal = match proposed_realization(&proposal_problem, &mut native_lp_proposal) {
                ProposedRealization::Gram(gram) => ProposedRealization::Gram(gram),
                proposal_outcome if !positivity_directions.is_empty() => {
                    profile.event(
                        Some(iteration),
                        "positive_proposal_fallback",
                        proposal_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(
                            "reason={}",
                            match proposal_outcome {
                                ProposedRealization::Nonrealizable(_) => "nonrealizable",
                                ProposedRealization::Unavailable => "unavailable",
                                ProposedRealization::Gram(_) => unreachable!(),
                            }
                        ),
                    );
                    let native_proposal_started = profile.phase_start();
                    profile.event(
                        Some(iteration),
                        "native_positive_proposal_start",
                        None,
                        self.line_count(),
                        obstacles.len(),
                        format_args!("lp_rows={}", proposal_problem.rows.len()),
                    );
                    if let Some(gram) =
                        native_positive_proposal(&proposal_problem).map_err(realization_error)?
                    {
                        profile.event(
                            Some(iteration),
                            "native_positive_proposal_gram",
                            native_proposal_started,
                            self.line_count(),
                            obstacles.len(),
                            format_args!(""),
                        );
                        ProposedRealization::Gram(gram)
                    } else {
                        profile.event(
                            Some(iteration),
                            "native_positive_proposal_rejected",
                            native_proposal_started,
                            self.line_count(),
                            obstacles.len(),
                            format_args!(""),
                        );
                        ProposedRealization::Unavailable
                    }
                }
                outcome => outcome,
            };
            let rational_gram = match proposal {
                ProposedRealization::Gram(gram) => {
                    profile.event(
                        Some(iteration),
                        "native_lp_proposal_gram",
                        proposal_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(""),
                    );
                    gram
                }
                ProposedRealization::Nonrealizable(multipliers) => {
                    profile.event(
                        Some(iteration),
                        "native_lp_proposal_nonrealizable",
                        proposal_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!("multiplier_count={}", multipliers.len()),
                    );
                    let evidence = MinimumNonrealizabilityCertificate {
                        configuration: self.clone(),
                        obstacles,
                        multipliers,
                        kind: NegativeCertificateKind::NonpositiveMargin,
                    };
                    evidence.verify()?;
                    return Ok(MinimumRealization {
                        outcome: MinimumRealizationOutcome::Nonrealizable(evidence),
                    });
                }
                ProposedRealization::Unavailable => {
                    profile.event(
                        Some(iteration),
                        "native_lp_proposal_unavailable",
                        proposal_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(""),
                    );
                    let native_started = profile.phase_start();
                    profile.event(
                        Some(iteration),
                        "native_lp_start",
                        None,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(
                            "lp_rows={} lp_variables={}",
                            problem.rows.len(),
                            problem.objective.len()
                        ),
                    );
                    let result = maximize_free(&problem.rows, &problem.rhs, &problem.objective)
                        .map_err(realization_error)?;
                    let primal = match result {
                        LinearProgramResult::Infeasible { certificate } => {
                            profile.event(
                                Some(iteration),
                                "native_lp_infeasible",
                                native_started,
                                self.line_count(),
                                obstacles.len(),
                                format_args!("certificate_count={}", certificate.len()),
                            );
                            let evidence = MinimumNonrealizabilityCertificate {
                                configuration: self.clone(),
                                obstacles,
                                multipliers: certificate,
                                kind: NegativeCertificateKind::Infeasible,
                            };
                            evidence.verify()?;
                            return Ok(MinimumRealization {
                                outcome: MinimumRealizationOutcome::Nonrealizable(evidence),
                            });
                        }
                        LinearProgramResult::Unbounded => {
                            return Err(realization_error(
                                "the seeded outsider-margin objective was unexpectedly unbounded",
                            ));
                        }
                        LinearProgramResult::Optimal {
                            primal,
                            optimum,
                            dual,
                        } => {
                            if !optimum.is_positive() {
                                profile.event(
                                    Some(iteration),
                                    "native_lp_nonpositive",
                                    native_started,
                                    self.line_count(),
                                    obstacles.len(),
                                    format_args!("dual_count={}", dual.len()),
                                );
                                let evidence = MinimumNonrealizabilityCertificate {
                                    configuration: self.clone(),
                                    obstacles,
                                    multipliers: dual,
                                    kind: NegativeCertificateKind::NonpositiveMargin,
                                };
                                evidence.verify()?;
                                return Ok(MinimumRealization {
                                    outcome: MinimumRealizationOutcome::Nonrealizable(evidence),
                                });
                            }
                            profile.event(
                                Some(iteration),
                                "native_lp_gram",
                                native_started,
                                self.line_count(),
                                obstacles.len(),
                                format_args!("primal_count={}", primal.len()),
                            );
                            primal
                        }
                    };
                    problem.gram_from_primal(&primal)
                }
            };
            if profile.enabled() {
                let (numerator_digits, denominator_digits) =
                    rational_gram_decimal_digits(&rational_gram);
                profile.event(
                    Some(iteration),
                    "gram_size",
                    None,
                    self.line_count(),
                    obstacles.len(),
                    format_args!(
                        "max_numerator_decimal_digits={} max_denominator_decimal_digits={}",
                        numerator_digits, denominator_digits
                    ),
                );
            }
            let small_nonpositive_started = profile.phase_start();
            let small_nonpositive = small_nonpositive_integral_directions(&rational_gram);
            let mut added_nonpositive = false;
            let small_nonpositive_count = small_nonpositive.len();
            for separator in small_nonpositive {
                added_nonpositive |= insert_obstacle(&mut obstacles, separator);
            }
            profile.event(
                Some(iteration),
                "small_nonpositive_scan",
                small_nonpositive_started,
                self.line_count(),
                obstacles.len(),
                format_args!(
                    "separator_count={} changed={}",
                    small_nonpositive_count, added_nonpositive
                ),
            );
            if added_nonpositive {
                continue;
            }
            let nonpositive_started = profile.phase_start();
            let nonpositive = nonpositive_rational_direction(&rational_gram);
            profile.event(
                Some(iteration),
                "positive_definite_check",
                nonpositive_started,
                self.line_count(),
                obstacles.len(),
                format_args!("separator_found={}", nonpositive.is_some()),
            );
            if let Some(direction) = nonpositive {
                if let Some(separator) =
                    compact_nonpositive_integral_direction(&direction, &rational_gram)
                {
                    if insert_obstacle(&mut obstacles, separator) {
                        continue;
                    }
                    return Err(realization_error(
                        "compact non-positive Gram separator was already present in the finite LP",
                    ));
                }
                let direction = normalize_rational_direction(direction);
                if insert_positivity_direction(&mut positivity_directions, direction.clone()) {
                    profile.event(
                        Some(iteration),
                        "positivity_cut_added",
                        None,
                        self.line_count(),
                        obstacles.len(),
                        format_args!("positivity_cuts={}", positivity_directions.len()),
                    );
                    continue;
                }
                let separator = rational_vector_to_primitive_integer(&direction);
                if insert_obstacle(&mut obstacles, separator) {
                    continue;
                }
                return Err(realization_error(
                    "non-positive Gram direction was already present in the proposal LP",
                ));
            }
            let lattice_started = profile.phase_start();
            let (integer_gram, scale) = scale_rational_gram(&rational_gram)?;
            let max_integer_digits = profile.enabled().then(|| {
                integer_gram
                    .rows()
                    .flatten()
                    .map(|entry| integer_decimal_digits(&entry))
                    .max()
                    .unwrap_or(1)
            });
            let lattice = Lattice::from_gram(integer_gram.clone()).map_err(|error| {
                realization_error(format!("positive rational Gram was rejected: {error}"))
            })?;
            if profile.enabled() {
                profile.event(
                    Some(iteration),
                    "lattice_build",
                    lattice_started,
                    self.line_count(),
                    obstacles.len(),
                    format_args!(
                        "scale_decimal_digits={} max_gram_decimal_digits={}",
                        integer_decimal_digits(&scale),
                        max_integer_digits.unwrap_or(0)
                    ),
                );
            }
            // A bounded batch amortizes LP rounds while still preventing a
            // poor proposal Gram from triggering an unboundedly large
            // ellipsoid walk.  Since the batch is larger than `line_count`,
            // every incomplete prefix necessarily contains an outsider.
            let separator_cap = self
                .line_count()
                .checked_add(512)
                .ok_or_else(|| realization_error("separator count overflow"))?;
            let enumeration_started = profile.phase_start();
            profile.event(
                Some(iteration),
                "short_vectors_start",
                None,
                self.line_count(),
                obstacles.len(),
                format_args!("separator_cap={}", separator_cap),
            );
            let enumeration = match lattice.shortest_vectors(Some(scale), Some(separator_cap)) {
                Ok(value) => {
                    profile.event(
                        Some(iteration),
                        "short_vectors_end",
                        enumeration_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(
                            "pair_count={} complete={}",
                            value.pair_count(),
                            value.is_complete()
                        ),
                    );
                    value
                }
                Err(LatticeError::WorkLimitExceeded { .. } | LatticeError::SearchRangeExceeded) => {
                    profile.event(
                        Some(iteration),
                        "short_vectors_work_limit",
                        enumeration_started,
                        self.line_count(),
                        obstacles.len(),
                        format_args!(""),
                    );
                    return inconclusive(
                        self,
                        obstacles,
                        iteration + 1,
                        "complete short-vector separation exceeded its work limit",
                    );
                }
                Err(error) => return Err(realization_error(error.to_string())),
            };
            let classification_started = profile.phase_start();
            let mut added = false;
            for vector in enumeration.vectors() {
                let row = vector
                    .iter()
                    .copied()
                    .map(Integer::from)
                    .collect::<Vec<_>>();
                if !contains_line(self.vectors(), &row) {
                    added |= insert_obstacle(&mut obstacles, row);
                }
            }
            profile.event(
                Some(iteration),
                "short_vectors_classify",
                classification_started,
                self.line_count(),
                obstacles.len(),
                format_args!("changed={}", added),
            );
            if added {
                continue;
            }
            if !enumeration.is_complete() {
                return inconclusive(
                    self,
                    obstacles,
                    iteration + 1,
                    "capped short-vector separation found no new outsider",
                );
            }
            let verification_started = profile.phase_start();
            let realization = lattice.minimum_vector_configuration().map_err(|error| {
                realization_error(format!("minimum-shell verification failed: {error}"))
            })?;
            profile.event(
                Some(iteration),
                "minimum_shell_verify",
                verification_started,
                self.line_count(),
                obstacles.len(),
                format_args!(
                    "realized_lines={}",
                    realization.configuration().line_count()
                ),
            );
            if realization.configuration() != self {
                return Err(realization_error(
                    "separator terminated with a different complete minimum shell",
                ));
            }
            return Ok(MinimumRealization {
                outcome: MinimumRealizationOutcome::Realized(realization),
            });
        }
        inconclusive(
            self,
            obstacles,
            max_iterations,
            "cutting-plane iteration limit reached",
        )
    }

    /// Complete these equal-length lines to a realizable minimum shell without
    /// increasing their perfection rank.
    ///
    /// Separating vectors whose rank-one forms lie in the seed span are either
    /// forced shorter, which yields exact negative evidence, or forced to the
    /// same length and admitted to the working shell. Only vectors that would
    /// increase the seed's perfection rank are constrained to be longer. This
    /// is the fixed-rank completion variant used by Voronoi coface gluing.
    pub fn complete_minimum(
        &self,
        max_iterations: usize,
    ) -> Result<MinimumCompletion, LineConfigurationError> {
        if self.ambient_rank() == 0 {
            return Err(LineConfigurationError::RankZeroRealization);
        }
        if !self.is_spanning() {
            return Err(LineConfigurationError::NonspanningRealization);
        }
        let seed = self.clone();
        let mut working = self.clone();
        let mut obstacles = Vec::new();
        if max_iterations == 0 {
            return completion_inconclusive(
                &seed,
                working,
                obstacles,
                0,
                "cutting-plane iteration limit reached",
            );
        }

        let (basis_realization, basis_obstacles) = try_basis_realization(&working)?;
        if let Some(realization) = basis_realization {
            let result = MinimumCompletion {
                seed,
                outcome: MinimumCompletionOutcome::Completed(realization),
            };
            result.verify()?;
            return Ok(result);
        }
        let initial_vectors = seed_obstacles(&seed)
            .into_iter()
            .chain(basis_obstacles)
            .collect::<Vec<_>>();
        for vector in initial_vectors {
            if let Some(certificate) =
                classify_completion_vector(&seed, &mut working, &mut obstacles, vector)?
            {
                return completion_negative(seed, certificate);
            }
        }

        let mut native_lp_proposal = NativeLpProposal::new();
        for iteration in 0..max_iterations {
            let problem = match RealizationProblem::new(&working, obstacles.clone()) {
                Ok(problem) => problem,
                Err(multipliers) => {
                    let evidence = MinimumNonrealizabilityCertificate {
                        configuration: working,
                        obstacles: Vec::new(),
                        multipliers,
                        kind: NegativeCertificateKind::EqualityInconsistent,
                    };
                    evidence.verify()?;
                    return completion_obstructed(seed, evidence);
                }
            };
            let proposal = proposed_realization(&problem, &mut native_lp_proposal);
            let rational_gram = match proposal {
                ProposedRealization::Gram(gram) => gram,
                ProposedRealization::Nonrealizable(multipliers) => {
                    let evidence = MinimumNonrealizabilityCertificate {
                        configuration: working,
                        obstacles,
                        multipliers,
                        kind: NegativeCertificateKind::NonpositiveMargin,
                    };
                    evidence.verify()?;
                    return completion_obstructed(seed, evidence);
                }
                ProposedRealization::Unavailable => {
                    let result = maximize_free(&problem.rows, &problem.rhs, &problem.objective)
                        .map_err(realization_error)?;
                    let primal = match result {
                        LinearProgramResult::Infeasible { certificate } => {
                            let evidence = MinimumNonrealizabilityCertificate {
                                configuration: working,
                                obstacles,
                                multipliers: certificate,
                                kind: NegativeCertificateKind::Infeasible,
                            };
                            evidence.verify()?;
                            return completion_obstructed(seed, evidence);
                        }
                        LinearProgramResult::Unbounded => {
                            return Err(realization_error(
                                "the seeded completion objective was unexpectedly unbounded",
                            ));
                        }
                        LinearProgramResult::Optimal {
                            primal,
                            optimum,
                            dual,
                        } => {
                            if !optimum.is_positive() {
                                let evidence = MinimumNonrealizabilityCertificate {
                                    configuration: working,
                                    obstacles,
                                    multipliers: dual,
                                    kind: NegativeCertificateKind::NonpositiveMargin,
                                };
                                evidence.verify()?;
                                return completion_obstructed(seed, evidence);
                            }
                            primal
                        }
                    };
                    problem.gram_from_primal(&primal)
                }
            };

            let mut changed = false;
            for separator in small_nonpositive_integral_directions(&rational_gram) {
                let previous_lines = working.line_count();
                let previous_obstacles = obstacles.len();
                if let Some(certificate) =
                    classify_completion_vector(&seed, &mut working, &mut obstacles, separator)?
                {
                    return completion_negative(seed, certificate);
                }
                changed |=
                    working.line_count() != previous_lines || obstacles.len() != previous_obstacles;
            }
            if changed {
                continue;
            }
            if let Some(separator) = nonpositive_integral_direction(&rational_gram) {
                let previous_lines = working.line_count();
                let previous_obstacles = obstacles.len();
                if let Some(certificate) =
                    classify_completion_vector(&seed, &mut working, &mut obstacles, separator)?
                {
                    return completion_negative(seed, certificate);
                }
                if working.line_count() != previous_lines || obstacles.len() != previous_obstacles {
                    continue;
                }
                return Err(realization_error(
                    "non-positive completion separator was already classified",
                ));
            }

            let (integer_gram, scale) = scale_rational_gram(&rational_gram)?;
            let lattice = Lattice::from_gram(integer_gram).map_err(|error| {
                realization_error(format!("positive completion Gram was rejected: {error}"))
            })?;
            let separator_cap = working
                .line_count()
                .checked_add(512)
                .ok_or_else(|| realization_error("completion separator count overflow"))?;
            let enumeration = match lattice.shortest_vectors(Some(scale), Some(separator_cap)) {
                Ok(value) => value,
                Err(LatticeError::WorkLimitExceeded { .. } | LatticeError::SearchRangeExceeded) => {
                    return completion_inconclusive(
                        &seed,
                        working,
                        obstacles,
                        iteration + 1,
                        "complete short-vector separation exceeded its work limit",
                    );
                }
                Err(error) => return Err(realization_error(error.to_string())),
            };
            let mut changed = false;
            for vector in enumeration.vectors() {
                let row = vector
                    .iter()
                    .copied()
                    .map(Integer::from)
                    .collect::<Vec<_>>();
                let previous_lines = working.line_count();
                let previous_obstacles = obstacles.len();
                if let Some(certificate) =
                    classify_completion_vector(&seed, &mut working, &mut obstacles, row)?
                {
                    return completion_negative(seed, certificate);
                }
                changed |=
                    working.line_count() != previous_lines || obstacles.len() != previous_obstacles;
            }
            if changed {
                continue;
            }
            if !enumeration.is_complete() {
                return completion_inconclusive(
                    &seed,
                    working,
                    obstacles,
                    iteration + 1,
                    "capped short-vector separation found no new completion separator",
                );
            }
            let realization = lattice.minimum_vector_configuration().map_err(|error| {
                realization_error(format!(
                    "completion minimum-shell verification failed: {error}"
                ))
            })?;
            if realization.configuration() != &working {
                return Err(realization_error(
                    "completion separator terminated with a different minimum shell",
                ));
            }
            let result = MinimumCompletion {
                seed,
                outcome: MinimumCompletionOutcome::Completed(realization),
            };
            result.verify()?;
            return Ok(result);
        }
        completion_inconclusive(
            &seed,
            working,
            obstacles,
            max_iterations,
            "cutting-plane iteration limit reached",
        )
    }
}

fn configuration_contains(
    configuration: &IntegralLineConfiguration,
    subset: &IntegralLineConfiguration,
) -> bool {
    configuration.ambient_rank() == subset.ambient_rank()
        && subset
            .vectors()
            .iter()
            .all(|line| configuration.vectors().binary_search(line).is_ok())
}

fn primitive_line(mut vector: Vec<Integer>) -> Result<Vec<Integer>, LineConfigurationError> {
    let content = vector
        .iter()
        .fold(Integer::zero(), |gcd, entry| gcd.gcd(&entry.abs()));
    if content.is_zero() {
        return Err(realization_error(
            "completion separator was the zero vector",
        ));
    }
    if !content.is_one() {
        for entry in &mut vector {
            *entry = &*entry / &content;
        }
    }
    Ok(canonical_sign(vector))
}

fn rank_one_relation(
    seed: &IntegralLineConfiguration,
    vector: &[Integer],
) -> Result<Option<Vec<Rational>>, LineConfigurationError> {
    let rows = seed
        .vectors()
        .iter()
        .map(|line| quadratic_row(line, seed.ambient_rank()))
        .collect::<Vec<_>>();
    let operator = Matrix::from_rows(Q, rows)
        .map_err(|error| realization_error(error.to_string()))?
        .transpose();
    let rhs = super::Vector::from_values(Q, quadratic_row(vector, seed.ambient_rank()))
        .map_err(|error| realization_error(error.to_string()))?;
    match operator.solve(&rhs) {
        Ok(solution) => Ok(Some(solution.particular().values().collect::<Vec<_>>())),
        Err(super::LinalgError::InconsistentSystem) => Ok(None),
        Err(error) => Err(realization_error(error.to_string())),
    }
}

fn classify_completion_vector(
    seed: &IntegralLineConfiguration,
    working: &mut IntegralLineConfiguration,
    obstacles: &mut Vec<Vec<Integer>>,
    vector: Vec<Integer>,
) -> Result<Option<MinimumCompletionCertificate>, LineConfigurationError> {
    let vector = primitive_line(vector)?;
    if contains_line(working.vectors(), &vector) {
        return Ok(None);
    }
    let Some(coefficients) = rank_one_relation(seed, &vector)? else {
        insert_obstacle(obstacles, vector);
        return Ok(None);
    };
    let forced_value = coefficients
        .iter()
        .fold(Rational::zero(), |sum, value| &sum + value);
    if forced_value < Rational::one() {
        let certificate = MinimumCompletionCertificate {
            seed: seed.clone(),
            evidence: MinimumCompletionCertificateEvidence::ForcedShorter {
                vector,
                coefficients,
            },
        };
        certificate.verify()?;
        return Ok(Some(certificate));
    }
    if forced_value == Rational::one() {
        let mut lines = working.vectors().to_vec();
        lines.push(vector);
        *working = IntegralLineConfiguration::new(working.ambient_rank(), lines)?;
        if working.perfection_rank() != seed.perfection_rank() {
            return Err(realization_error(
                "admitted completion line changed the perfection rank",
            ));
        }
    }
    Ok(None)
}

fn completion_negative(
    seed: IntegralLineConfiguration,
    certificate: MinimumCompletionCertificate,
) -> Result<MinimumCompletion, LineConfigurationError> {
    let result = MinimumCompletion {
        seed,
        outcome: MinimumCompletionOutcome::Nonrealizable(certificate),
    };
    result.verify()?;
    Ok(result)
}

fn completion_obstructed(
    seed: IntegralLineConfiguration,
    evidence: MinimumNonrealizabilityCertificate,
) -> Result<MinimumCompletion, LineConfigurationError> {
    let certificate = MinimumCompletionCertificate {
        seed: seed.clone(),
        evidence: MinimumCompletionCertificateEvidence::Obstructed(evidence),
    };
    completion_negative(seed, certificate)
}

fn completion_inconclusive(
    seed: &IntegralLineConfiguration,
    configuration: IntegralLineConfiguration,
    obstacles: Vec<Vec<Integer>>,
    iterations: usize,
    reason: impl Into<String>,
) -> Result<MinimumCompletion, LineConfigurationError> {
    let diagnostic = MinimumRealizationDiagnostic {
        configuration,
        obstacle_count: obstacles.len(),
        iterations,
        reason: reason.into(),
    };
    let result = MinimumCompletion {
        seed: seed.clone(),
        outcome: MinimumCompletionOutcome::Inconclusive(diagnostic),
    };
    result.verify()?;
    Ok(result)
}

enum ProposedRealization {
    Gram(Vec<Vec<Rational>>),
    Nonrealizable(Vec<Rational>),
    Unavailable,
}

/// Ask the native revised-simplex engine for an exactly reconstructed proposal.
fn proposed_realization(
    problem: &RealizationProblem,
    native: &mut NativeLpProposal,
) -> ProposedRealization {
    if let Some((primal, dual)) =
        native.optimized_solution_free(&problem.rows, &problem.rhs, &problem.objective)
    {
        return checked_proposal_realization(problem, primal, dual);
    }
    ProposedRealization::Unavailable
}

fn checked_proposal_realization(
    problem: &RealizationProblem,
    primal: Vec<Rational>,
    dual: Vec<Rational>,
) -> ProposedRealization {
    let optimum = dot_rational(&problem.objective, &primal);
    if !optimum.is_positive() {
        if let Some(multipliers) = certified_upper_bound_from_dual_free(
            &problem.rows,
            &problem.rhs,
            &problem.objective,
            &Rational::zero(),
            &dual,
        ) {
            return ProposedRealization::Nonrealizable(multipliers);
        }
        return ProposedRealization::Unavailable;
    }
    ProposedRealization::Gram(problem.gram_from_primal(&primal))
}

/// Exact fallback for a proposal carrying non-evidentiary positivity cuts.
/// Only a positive feasible Gram is consumed; terminal outcomes are left to
/// the integer-obstacle problem and its public certificate contract.
fn native_positive_proposal(
    problem: &RealizationProblem,
) -> Result<Option<Vec<Vec<Rational>>>, String> {
    match maximize_free(&problem.rows, &problem.rhs, &problem.objective)? {
        LinearProgramResult::Optimal {
            primal, optimum, ..
        } if optimum.is_positive() => Ok(Some(problem.gram_from_primal(&primal))),
        LinearProgramResult::Optimal { .. }
        | LinearProgramResult::Infeasible { .. }
        | LinearProgramResult::Unbounded => Ok(None),
    }
}

/// Legacy homogeneous formulation retained solely to verify previously issued
/// `homogeneous_minimum_at_least_one` certificates.
struct HomogeneousProblem {
    rows: Vec<Vec<Rational>>,
    rhs: Vec<Rational>,
    objective: Vec<Rational>,
}

impl HomogeneousProblem {
    fn new(configuration: &IntegralLineConfiguration, obstacles: &[Vec<Integer>]) -> Self {
        let rank = configuration.ambient_rank();
        let symmetric_dimension = configuration.symmetric_dimension();
        let variable_count = symmetric_dimension + 1;
        let mut rows = Vec::new();
        let mut rhs = Vec::new();
        for vector in configuration.vectors() {
            let mut equality = quadratic_row(vector, rank);
            equality.push(-Rational::one());
            rows.push(equality.clone());
            rhs.push(Rational::zero());
            rows.push(equality.into_iter().map(|entry| -entry).collect());
            rhs.push(Rational::zero());
        }
        for vector in obstacles {
            let mut inequality = quadratic_row(vector, rank)
                .into_iter()
                .map(|entry| -entry)
                .collect::<Vec<_>>();
            inequality.push(Rational::zero());
            rows.push(inequality);
            rhs.push(Rational::from(-1_i64));
        }
        let mut positive_lambda = vec![Rational::zero(); variable_count];
        positive_lambda[symmetric_dimension] = -Rational::one();
        rows.push(positive_lambda);
        rhs.push(Rational::zero());
        let mut objective = vec![Rational::zero(); variable_count];
        objective[symmetric_dimension] = -Rational::one();
        Self {
            rows,
            rhs,
            objective,
        }
    }
}

struct RealizationProblem {
    rank: usize,
    symmetric_dimension: usize,
    particular: Vec<Rational>,
    directions: Vec<Vec<Rational>>,
    rows: Vec<Vec<Rational>>,
    rhs: Vec<Rational>,
    objective: Vec<Rational>,
}

impl RealizationProblem {
    fn new(
        configuration: &IntegralLineConfiguration,
        obstacles: Vec<Vec<Integer>>,
    ) -> Result<Self, Vec<Rational>> {
        let rank = configuration.ambient_rank();
        let symmetric_dimension = configuration.symmetric_dimension();
        let equalities = configuration
            .vectors()
            .iter()
            .map(|vector| quadratic_row(vector, rank))
            .collect::<Vec<_>>();
        let (particular, directions) = affine_equal_norm_space(equalities)?;
        let variable_count = directions.len() + 1;
        let mut rows = Vec::new();
        let mut rhs = Vec::new();
        for vector in &obstacles {
            let quadratic = quadratic_row(vector, rank);
            let mut inequality = directions
                .iter()
                .map(|direction| -dot_rational(&quadratic, direction))
                .collect::<Vec<_>>();
            inequality.push(Rational::one());
            let mut bound = &dot_rational(&quadratic, &particular) - Rational::one();
            normalize_inequality(&mut inequality, &mut bound);
            rows.push(inequality);
            rhs.push(bound);
        }
        let mut objective = vec![Rational::zero(); variable_count];
        objective[variable_count - 1] = Rational::one();
        Ok(Self {
            rank,
            symmetric_dimension,
            particular,
            directions,
            rows,
            rhs,
            objective,
        })
    }

    fn gram_from_primal(&self, primal: &[Rational]) -> Vec<Vec<Rational>> {
        let mut coordinates = self.particular.clone();
        for (coefficient, direction) in primal.iter().zip(&self.directions) {
            for (coordinate, entry) in coordinates.iter_mut().zip(direction) {
                *coordinate = &*coordinate + &(coefficient * entry);
            }
        }
        debug_assert_eq!(coordinates.len(), self.symmetric_dimension);
        symmetric_matrix_from_coordinates(self.rank, &coordinates)
    }

    /// Add scale-free positive-definiteness cuts to the proposal LP. These
    /// steer the optional solver away from singular faces but are not part of
    /// the integer-obstacle evidence used for a terminal negative result.
    fn with_positive_directions(&self, positive_directions: &[Vec<Rational>]) -> Self {
        let mut rows = self.rows.clone();
        let mut rhs = self.rhs.clone();
        for vector in positive_directions {
            debug_assert_eq!(vector.len(), self.rank);
            let quadratic = rational_quadratic_row(vector, self.rank);
            let mut inequality = self
                .directions
                .iter()
                .map(|direction| -dot_rational(&quadratic, direction))
                .collect::<Vec<_>>();
            inequality.push(Rational::one());
            let mut bound = dot_rational(&quadratic, &self.particular);
            normalize_inequality(&mut inequality, &mut bound);
            rows.push(inequality);
            rhs.push(bound);
        }
        Self {
            rank: self.rank,
            symmetric_dimension: self.symmetric_dimension,
            particular: self.particular.clone(),
            directions: self.directions.clone(),
            rows,
            rhs,
            objective: self.objective.clone(),
        }
    }
}

/// Rescale one exact weak inequality when its dynamic range would overflow the
/// floating proposal phase.  Dividing both sides by the same positive rational
/// leaves the feasible half-space, and therefore every exact certificate,
/// unchanged.  Small rows retain their historical representation.
fn normalize_inequality(row: &mut [Rational], rhs: &mut Rational) {
    const NORMALIZATION_THRESHOLD: i64 = 1_000_000;

    let scale = row
        .iter()
        .chain(core::iter::once(&*rhs))
        .map(|entry| {
            if entry.is_negative() {
                -entry
            } else {
                entry.clone()
            }
        })
        .max()
        .unwrap_or_else(Rational::zero);
    if scale <= Rational::from(NORMALIZATION_THRESHOLD) {
        return;
    }
    for entry in row {
        *entry = &*entry / &scale;
    }
    *rhs = &*rhs / &scale;
}

fn symmetric_matrix_from_coordinates(rank: usize, coordinates: &[Rational]) -> Vec<Vec<Rational>> {
    let mut matrix = vec![vec![Rational::zero(); rank]; rank];
    for ((row, column), coordinate) in (0..rank)
        .flat_map(|row| (row..rank).map(move |column| (row, column)))
        .zip(coordinates)
    {
        matrix[row][column] = coordinate.clone();
        matrix[column][row] = coordinate.clone();
    }
    matrix
}

type AffineEqualNormResult = Result<(Vec<Rational>, Vec<Vec<Rational>>), Vec<Rational>>;

fn affine_equal_norm_space(coefficients: Vec<Vec<Rational>>) -> AffineEqualNormResult {
    let row_count = coefficients.len();
    let column_count = coefficients.first().map_or(0, Vec::len);
    let mut matrix = coefficients;
    let mut rhs = vec![Rational::one(); row_count];
    let mut combinations = (0..row_count)
        .map(|row| {
            (0..row_count)
                .map(|column| {
                    if row == column {
                        Rational::one()
                    } else {
                        Rational::zero()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..column_count {
        if pivot_row == row_count {
            break;
        }
        let Some(found) = (pivot_row..row_count).find(|row| !matrix[*row][column].is_zero()) else {
            continue;
        };
        matrix.swap(pivot_row, found);
        rhs.swap(pivot_row, found);
        combinations.swap(pivot_row, found);
        let divisor = matrix[pivot_row][column].clone();
        for entry in &mut matrix[pivot_row] {
            *entry = &*entry / &divisor;
        }
        rhs[pivot_row] = &rhs[pivot_row] / &divisor;
        for entry in &mut combinations[pivot_row] {
            *entry = &*entry / &divisor;
        }
        let normalized_matrix_row = matrix[pivot_row].clone();
        let normalized_combination = combinations[pivot_row].clone();
        for row in 0..row_count {
            if row == pivot_row || matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column].clone();
            for (entry, pivot_entry) in matrix[row].iter_mut().zip(&normalized_matrix_row) {
                *entry = &*entry - &(&factor * pivot_entry);
            }
            rhs[row] = &rhs[row] - &(&factor * &rhs[pivot_row].clone());
            for (entry, pivot_entry) in combinations[row].iter_mut().zip(&normalized_combination) {
                *entry = &*entry - &(&factor * pivot_entry);
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }
    for row in pivot_row..row_count {
        if matrix[row].iter().all(Rational::is_zero) && !rhs[row].is_zero() {
            return Err(combinations[row].clone());
        }
    }
    let free_columns = (0..column_count)
        .filter(|column| !pivot_columns.contains(column))
        .collect::<Vec<_>>();
    let mut particular = vec![Rational::zero(); column_count];
    for (row, column) in pivot_columns.iter().enumerate() {
        particular[*column] = rhs[row].clone();
    }
    let directions = free_columns
        .iter()
        .map(|free| {
            let mut direction = vec![Rational::zero(); column_count];
            direction[*free] = Rational::one();
            for (row, pivot) in pivot_columns.iter().enumerate() {
                direction[*pivot] = -&matrix[row][*free];
            }
            direction
        })
        .collect();
    Ok((particular, directions))
}

fn dot_rational(left: &[Rational], right: &[Rational]) -> Rational {
    left.iter()
        .zip(right)
        .fold(Rational::zero(), |sum, (left, right)| {
            &sum + &(left * right)
        })
}

fn quadratic_row(vector: &[Integer], rank: usize) -> Vec<Rational> {
    let mut row = Vec::new();
    for left in 0..rank {
        for right in left..rank {
            let mut coefficient = &vector[left] * &vector[right];
            if left != right {
                coefficient *= 2_u32;
            }
            row.push(Rational::from(coefficient));
        }
    }
    row
}

fn rational_quadratic_row(vector: &[Rational], rank: usize) -> Vec<Rational> {
    let mut row = Vec::new();
    for left in 0..rank {
        for right in left..rank {
            let coefficient = &vector[left] * &vector[right];
            row.push(if left == right {
                coefficient
            } else {
                &coefficient * &Rational::from(2_i64)
            });
        }
    }
    row
}

fn try_basis_realization(
    configuration: &IntegralLineConfiguration,
) -> Result<(Option<MinimumVectorConfiguration>, Vec<Vec<Integer>>), LineConfigurationError> {
    let rank = configuration.ambient_rank();
    if configuration.line_count() != rank {
        return Ok((None, Vec::new()));
    }
    let basis = configuration
        .vectors()
        .iter()
        .map(|row| row.iter().cloned().map(Rational::from).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut inverse = vec![vec![Rational::zero(); rank]; rank];
    for column in 0..rank {
        let mut unit = vec![Rational::zero(); rank];
        unit[column] = Rational::one();
        let solution = solve_square(basis.clone(), unit).ok_or_else(|| {
            realization_error("spanning candidate basis was unexpectedly singular")
        })?;
        for row in 0..rank {
            inverse[row][column] = solution[row].clone();
        }
    }
    let gram = (0..rank)
        .map(|row| {
            (0..rank)
                .map(|column| {
                    (0..rank).fold(Rational::zero(), |sum, inner| {
                        &sum + &(&inverse[row][inner] * &inverse[column][inner])
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (integer_gram, scale) = scale_rational_gram(&gram)?;
    let lattice = Lattice::from_gram(integer_gram)
        .map_err(|error| realization_error(format!("basis Gram construction failed: {error}")))?;
    let separator_cap = configuration
        .line_count()
        .checked_add(512)
        .ok_or_else(|| realization_error("basis separator count overflow"))?;
    let enumeration = match lattice.shortest_vectors(Some(scale), Some(separator_cap)) {
        Ok(enumeration) => enumeration,
        Err(LatticeError::WorkLimitExceeded { .. } | LatticeError::SearchRangeExceeded) => {
            return Ok((None, Vec::new()));
        }
        Err(error) => {
            return Err(realization_error(format!(
                "basis short-vector separation failed: {error}"
            )));
        }
    };
    let mut obstacles = Vec::new();
    for vector in enumeration.vectors() {
        let row = vector
            .iter()
            .copied()
            .map(Integer::from)
            .collect::<Vec<_>>();
        if !contains_line(configuration.vectors(), &row) {
            insert_obstacle(&mut obstacles, row);
        }
    }
    if enumeration.is_complete() && obstacles.is_empty() {
        let realization = lattice.minimum_vector_configuration().map_err(|error| {
            realization_error(format!("basis minimum-shell verification failed: {error}"))
        })?;
        if realization.configuration() == configuration {
            return Ok((Some(realization), Vec::new()));
        }
    }
    Ok((None, obstacles))
}

fn seed_obstacles(configuration: &IntegralLineConfiguration) -> Vec<Vec<Integer>> {
    let rank = configuration.ambient_rank();
    let mut obstacles = Vec::new();
    for vector in configuration.vectors() {
        insert_obstacle(
            &mut obstacles,
            vector.iter().map(|entry| entry.clone() * 2_u32).collect(),
        );
    }
    for left in 0..rank {
        let mut unit = vec![Integer::zero(); rank];
        unit[left] = Integer::one();
        if !contains_line(configuration.vectors(), &unit) {
            insert_obstacle(&mut obstacles, unit);
        }
        for right in left + 1..rank {
            for sign in [1_i64, -1_i64] {
                let mut pair = vec![Integer::zero(); rank];
                pair[left] = Integer::one();
                pair[right] = Integer::from(sign);
                if !contains_line(configuration.vectors(), &pair) {
                    insert_obstacle(&mut obstacles, pair);
                }
            }
        }
    }
    // Direct control of off-diagonal entries in a basis chosen from the
    // candidate lines keeps the native proposal well conditioned. The seed is
    // part of the playbook and therefore does not vary with build features.
    let basis = independent_line_basis(configuration);
    for left in 0..basis.len() {
        for right in left + 1..basis.len() {
            for sign in [1_i64, -1_i64] {
                let pair = basis[left]
                    .iter()
                    .zip(&basis[right])
                    .map(|(left, right)| left + &(right * &Integer::from(sign)))
                    .collect::<Vec<_>>();
                if !contains_line(configuration.vectors(), &pair) {
                    insert_obstacle(&mut obstacles, pair);
                }
            }
        }
    }
    obstacles
}

fn independent_line_basis(configuration: &IntegralLineConfiguration) -> Vec<Vec<Integer>> {
    let mut basis = Vec::new();
    let mut rank = 0;
    for vector in configuration.vectors() {
        let mut trial = basis.clone();
        trial.push(vector.clone());
        let matrix = Matrix::from_rows(Z, trial.clone())
            .expect("configuration rows have a checked common dimension");
        let trial_rank = matrix.rank_backend();
        if trial_rank > rank {
            basis.push(vector.clone());
            rank = trial_rank;
            if rank == configuration.ambient_rank() {
                break;
            }
        }
    }
    basis
}

fn contains_line(rows: &[Vec<Integer>], vector: &[Integer]) -> bool {
    let canonical = canonical_sign(vector.to_vec());
    rows.binary_search(&canonical).is_ok()
}

fn insert_obstacle(obstacles: &mut Vec<Vec<Integer>>, vector: Vec<Integer>) -> bool {
    let vector = canonical_sign(vector);
    match obstacles.binary_search(&vector) {
        Ok(_) => false,
        Err(index) => {
            obstacles.insert(index, vector);
            true
        }
    }
}

fn canonical_sign(mut vector: Vec<Integer>) -> Vec<Integer> {
    if vector
        .iter()
        .find(|entry| !entry.is_zero())
        .is_some_and(Integer::is_negative)
    {
        for entry in &mut vector {
            *entry = -entry.clone();
        }
    }
    vector
}

fn insert_positivity_direction(
    directions: &mut Vec<Vec<Rational>>,
    direction: Vec<Rational>,
) -> bool {
    match directions.binary_search(&direction) {
        Ok(_) => false,
        Err(index) => {
            directions.insert(index, direction);
            true
        }
    }
}

fn normalize_rational_direction(mut direction: Vec<Rational>) -> Vec<Rational> {
    let magnitude = direction
        .iter()
        .map(|entry| {
            if entry.is_negative() {
                -entry
            } else {
                entry.clone()
            }
        })
        .max()
        .unwrap_or_else(Rational::zero);
    debug_assert!(!magnitude.is_zero());
    for entry in &mut direction {
        *entry = &*entry / &magnitude;
    }
    if direction
        .iter()
        .find(|entry| !entry.is_zero())
        .is_some_and(Rational::is_negative)
    {
        for entry in &mut direction {
            *entry = -&*entry;
        }
    }
    direction
}

fn nonpositive_rational_direction(gram: &[Vec<Rational>]) -> Option<Vec<Rational>> {
    for size in 1..=gram.len() {
        let previous = size - 1;
        let mut direction = vec![Rational::zero(); gram.len()];
        direction[previous] = Rational::one();
        let schur = if previous == 0 {
            gram[0][0].clone()
        } else {
            let leading = gram[..previous]
                .iter()
                .map(|row| row[..previous].to_vec())
                .collect::<Vec<_>>();
            let column = (0..previous)
                .map(|row| gram[row][previous].clone())
                .collect::<Vec<_>>();
            let solution = solve_square(leading, column)?;
            for index in 0..previous {
                direction[index] = -&solution[index];
            }
            (0..previous).fold(gram[previous][previous].clone(), |value, index| {
                &value - &(&gram[previous][index] * &solution[index])
            })
        };
        if !schur.is_positive() {
            return Some(direction);
        }
    }
    None
}

fn nonpositive_integral_direction(gram: &[Vec<Rational>]) -> Option<Vec<Integer>> {
    let direction = nonpositive_rational_direction(gram)?;
    if let Some(candidate) = compact_nonpositive_integral_direction(&direction, gram) {
        return Some(candidate);
    }
    Some(rational_vector_to_primitive_integer(&direction))
}

fn compact_nonpositive_integral_direction(
    direction: &[Rational],
    gram: &[Vec<Rational>],
) -> Option<Vec<Integer>> {
    // Prefer a compact, exactly checked integral separator to clearing the
    // potentially enormous denominators of the Schur direction. Rank-eight
    // tail cases can need a few hundred coordinates before rounding lands
    // inside the nonpositive cone.
    for max_coordinate in 1_i64..=4_096_i64 {
        if let Some(candidate) = bounded_nonpositive_approximation(direction, gram, max_coordinate)
        {
            return Some(candidate);
        }
    }
    None
}

fn bounded_nonpositive_approximation(
    direction: &[Rational],
    gram: &[Vec<Rational>],
    max_coordinate: i64,
) -> Option<Vec<Integer>> {
    let magnitude = direction
        .iter()
        .map(|entry| {
            if entry.is_negative() {
                -entry
            } else {
                entry.clone()
            }
        })
        .max()?;
    if magnitude.is_zero() {
        return None;
    }
    let scale = Rational::from(max_coordinate);
    let candidate = direction
        .iter()
        .map(|entry| nearest_rational_integer(&(entry * &scale / &magnitude)))
        .collect::<Vec<_>>();
    if candidate.iter().all(Integer::is_zero) {
        return None;
    }
    let norm = (0..candidate.len()).fold(Rational::zero(), |sum, row| {
        (0..candidate.len()).fold(sum, |sum, column| {
            &sum + &(&gram[row][column] * &Rational::from(&candidate[row] * &candidate[column]))
        })
    });
    (!norm.is_positive()).then(|| canonical_sign(candidate))
}

fn nearest_rational_integer(value: &Rational) -> Integer {
    let denominator = value.denominator();
    let floor = value.numerator().div_floor(&denominator);
    let remainder = value.numerator() - &(&floor * &denominator);
    if &remainder * &Integer::from(2_u8) > denominator {
        floor + Integer::one()
    } else {
        floor
    }
}

fn small_nonpositive_integral_directions(gram: &[Vec<Rational>]) -> Vec<Vec<Integer>> {
    let denominator = gram
        .iter()
        .flatten()
        .fold(Integer::one(), |lcm, entry| lcm.lcm(&entry.denominator()));
    let integral_gram = gram
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| entry.numerator() * (&denominator / &entry.denominator()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let coordinate_bound = 1_i64;

    fn visit(
        gram: &[Vec<Integer>],
        vector: &mut [i64],
        coordinate: usize,
        coordinate_bound: i64,
    ) -> Vec<(Integer, Vec<Integer>)> {
        if coordinate < vector.len() {
            let mut best = Vec::new();
            for value in -coordinate_bound..=coordinate_bound {
                vector[coordinate] = value;
                best.extend(visit(gram, vector, coordinate + 1, coordinate_bound));
                best.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
                best.truncate(8);
            }
            return best;
        }
        let Some(first) = vector.iter().copied().find(|entry| *entry != 0) else {
            return Vec::new();
        };
        if first < 0 {
            return Vec::new();
        }
        let norm = (0..vector.len()).fold(Integer::zero(), |sum, row| {
            let diagonal = &gram[row][row] * &Integer::from(vector[row] * vector[row]);
            (row + 1..vector.len()).fold(&sum + &diagonal, |sum, column| {
                let coefficient = Integer::from(2_i64 * vector[row] * vector[column]);
                &sum + &(&gram[row][column] * &coefficient)
            })
        });
        if norm.is_positive() {
            Vec::new()
        } else {
            vec![(norm, vector.iter().copied().map(Integer::from).collect())]
        }
    }

    visit(
        &integral_gram,
        &mut vec![0_i64; gram.len()],
        0,
        coordinate_bound,
    )
    .into_iter()
    .map(|(_, vector)| vector)
    .collect()
}

fn solve_square(mut matrix: Vec<Vec<Rational>>, mut rhs: Vec<Rational>) -> Option<Vec<Rational>> {
    let dimension = matrix.len();
    for column in 0..dimension {
        let pivot = (column..dimension).find(|row| !matrix[*row][column].is_zero())?;
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let divisor = matrix[column][column].clone();
        for entry in matrix[column].iter_mut().skip(column) {
            *entry = &*entry / &divisor;
        }
        rhs[column] = &rhs[column] / &divisor;
        let pivot_values = matrix[column].clone();
        for row in 0..dimension {
            if row == column || matrix[row][column].is_zero() {
                continue;
            }
            let factor = matrix[row][column].clone();
            for (entry, pivot_entry) in matrix[row].iter_mut().zip(&pivot_values).skip(column) {
                *entry = &*entry - &(&factor * pivot_entry);
            }
            rhs[row] = &rhs[row] - &(&factor * &rhs[column].clone());
        }
    }
    Some(rhs)
}

fn rational_vector_to_primitive_integer(vector: &[Rational]) -> Vec<Integer> {
    let denominator = vector
        .iter()
        .fold(Integer::one(), |lcm, entry| lcm.lcm(&entry.denominator()));
    let mut integers = vector
        .iter()
        .map(|entry| entry.numerator() * (&denominator / &entry.denominator()))
        .collect::<Vec<_>>();
    let content = integers
        .iter()
        .fold(Integer::zero(), |gcd, entry| gcd.gcd(&entry.abs()));
    if !content.is_zero() && !content.is_one() {
        for entry in &mut integers {
            *entry = &*entry / &content;
        }
    }
    canonical_sign(integers)
}

fn scale_rational_gram(
    gram: &[Vec<Rational>],
) -> Result<(Matrix<crate::IntegerRing>, Integer), LineConfigurationError> {
    let scale = gram
        .iter()
        .flatten()
        .fold(Integer::one(), |lcm, entry| lcm.lcm(&entry.denominator()));
    let rows = gram
        .iter()
        .map(|row| {
            row.iter()
                .map(|entry| entry.numerator() * (&scale / &entry.denominator()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Matrix::from_rows(Z, rows)
        .map(|matrix| (matrix, scale))
        .map_err(|error| realization_error(error.to_string()))
}

fn inconclusive(
    configuration: &IntegralLineConfiguration,
    obstacles: Vec<Vec<Integer>>,
    iterations: usize,
    reason: impl Into<String>,
) -> Result<MinimumRealization, LineConfigurationError> {
    let diagnostic = MinimumRealizationDiagnostic {
        configuration: configuration.clone(),
        obstacle_count: obstacles.len(),
        iterations,
        reason: reason.into(),
    };
    diagnostic.verify()?;
    Ok(MinimumRealization {
        outcome: MinimumRealizationOutcome::Inconclusive(diagnostic),
    })
}

fn realization_error(message: impl Into<String>) -> LineConfigurationError {
    LineConfigurationError::RealizationComputation {
        message: message.into(),
    }
}

fn verify_dual_bound(
    rows: &[Vec<Rational>],
    rhs: &[Rational],
    objective: &[Rational],
    multipliers: &[Rational],
    upper_bound: &Rational,
    message: &str,
) -> Result<(), LineConfigurationError> {
    if multipliers.len() != rhs.len() || multipliers.iter().any(Rational::is_negative) {
        return Err(realization_error(message));
    }
    let columns = (0..objective.len())
        .map(|column| {
            rows.iter()
                .zip(multipliers)
                .fold(Rational::zero(), |sum, (row, multiplier)| {
                    &sum + &(&row[column] * multiplier)
                })
        })
        .collect::<Vec<_>>();
    let bound = rhs
        .iter()
        .zip(multipliers)
        .fold(Rational::zero(), |sum, (value, multiplier)| {
            &sum + &(value * multiplier)
        });
    if columns != objective || bound > upper_bound.clone() {
        return Err(realization_error(message));
    }
    Ok(())
}

fn integer_rows_as_strings(rows: &[Vec<Integer>]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| row.iter().map(ToString::to_string).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configuration(rows: &[&[i64]]) -> IntegralLineConfiguration {
        IntegralLineConfiguration::new(
            rows[0].len(),
            rows.iter()
                .map(|row| row.iter().copied().map(Integer::from).collect()),
        )
        .unwrap()
    }

    #[test]
    fn realization_inequality_normalization_is_exact_and_scale_invariant() {
        let scale = Rational::from(12_000_000_i64);
        let original_row = vec![&scale * &Rational::from(2_i64), -&scale, Rational::one()];
        let original_rhs = &scale * &Rational::from(3_i64);
        let mut row = original_row.clone();
        let mut rhs = original_rhs.clone();

        normalize_inequality(&mut row, &mut rhs);

        let divisor = original_rhs;
        assert_eq!(rhs, Rational::one());
        assert!(row.iter().all(|entry| {
            let magnitude = if entry.is_negative() {
                -entry
            } else {
                entry.clone()
            };
            magnitude <= Rational::one()
        }));
        assert_eq!(
            row.iter().map(|entry| entry * &divisor).collect::<Vec<_>>(),
            original_row
        );

        let mut small_row = vec![Rational::from(2_i64), Rational::from(-3_i64)];
        let mut small_rhs = Rational::from(4_i64);
        let unchanged_row = small_row.clone();
        normalize_inequality(&mut small_row, &mut small_rhs);
        assert_eq!(small_row, unchanged_row);
        assert_eq!(small_rhs, Rational::from(4_i64));
    }

    #[test]
    fn scale_free_positivity_cut_rejects_its_indefinite_source() {
        let candidate = configuration(&[&[1, 0], &[0, 1]]);
        let problem = RealizationProblem::new(&candidate, Vec::new()).unwrap();
        assert_eq!(problem.directions.len(), 1);
        let bad_gram = vec![
            vec![Rational::from(1_i64), Rational::from(2_i64)],
            vec![Rational::from(2_i64), Rational::from(1_i64)],
        ];
        let direction = normalize_rational_direction(
            nonpositive_rational_direction(&bad_gram).expect("the form is indefinite"),
        );
        let maximum = direction
            .iter()
            .map(|entry| {
                if entry.is_negative() {
                    -entry
                } else {
                    entry.clone()
                }
            })
            .max()
            .unwrap();
        assert_eq!(maximum, Rational::one());

        let proposal = problem.with_positive_directions(&[direction]);
        assert_eq!(proposal.rows.len(), 1);
        let off_diagonal = 1;
        let bad_coefficient = &(&Rational::from(2_i64) - &problem.particular[off_diagonal])
            / &problem.directions[0][off_diagonal];
        let bad_primal = vec![bad_coefficient, Rational::new(1.into(), 4.into()).unwrap()];
        assert!(dot_rational(&proposal.rows[0], &bad_primal) > proposal.rhs[0]);

        let good_coefficient =
            -&problem.particular[off_diagonal] / &problem.directions[0][off_diagonal];
        let good_primal = vec![good_coefficient, Rational::new(1.into(), 4.into()).unwrap()];
        assert!(dot_rational(&proposal.rows[0], &good_primal) <= proposal.rhs[0]);
    }

    #[test]
    fn realizes_coordinate_axes_with_complete_enumeration_evidence() {
        let candidate = configuration(&[&[1, 0], &[0, 1]]);
        let result = candidate.realize_minimum(20).unwrap();
        assert_eq!(result.status(), "realized");
        assert_eq!(result.configuration(), &candidate);
        result.verify().unwrap();
    }

    #[test]
    fn minimum_playbooks_enforce_rank_preconditions() {
        let candidate = IntegralLineConfiguration::new(0, Vec::<Vec<Integer>>::new()).unwrap();
        assert_eq!(
            candidate.realize_minimum(1),
            Err(LineConfigurationError::RankZeroRealization)
        );
        assert_eq!(
            candidate.complete_minimum(1),
            Err(LineConfigurationError::RankZeroRealization)
        );

        let nonspanning = configuration(&[&[1, 0]]);
        assert_eq!(
            nonspanning.realize_minimum(1),
            Err(LineConfigurationError::NonspanningRealization)
        );
        assert_eq!(
            nonspanning.complete_minimum(1),
            Err(LineConfigurationError::NonspanningRealization)
        );
    }

    #[test]
    fn realizes_rank_eight_coordinate_axes_without_tableau_blowup() {
        let rows = (0..8)
            .map(|index| {
                (0..8)
                    .map(|column| if index == column { 1_i64 } else { 0_i64 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let references = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let candidate = configuration(&references);
        let result = candidate.realize_minimum(20).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn realizes_a_nonorthogonal_published_shape_without_cut_blowup() {
        let candidate = configuration(&[
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[0, 0, 0, 0, 0, 0, 1, 0],
            &[0, 0, 0, 0, 0, 1, 0, 0],
            &[0, 0, 0, 1, 1, 0, 0, 0],
            &[0, 0, 1, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, -1, 1, 0, 0, 0],
        ]);
        let result = candidate.realize_minimum(200).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn realizes_a_deeper_published_shape_without_cut_blowup() {
        let candidate = configuration(&[
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[0, 0, 0, 1, -1, 0, 0, 0],
            &[0, 0, 1, 0, -1, 0, 1, 0],
            &[0, 0, 1, 0, -1, 1, 0, 0],
            &[0, 0, 1, 1, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 0, 0, 0, 1, 1, 0],
        ]);
        let result = candidate.realize_minimum(200).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn realizes_a_coefficient_two_separator_shape() {
        let candidate = configuration(&[
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[0, 0, 0, 0, 0, 0, 1, 0],
            &[0, 0, 0, 0, 0, 1, 0, 0],
            &[0, 0, 0, 0, 1, 0, 0, 0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[0, 0, 1, 0, 0, 0, 0, 0],
            &[1, -3, 1, 1, 1, 1, 1, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
        ]);
        let result = candidate.realize_minimum(200).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn realizes_a_multi_separator_published_shape() {
        let candidate = configuration(&[
            &[0, 0, 0, 0, 1, -1, 0, 0],
            &[0, 0, 0, 1, 0, -1, 0, 1],
            &[0, 0, 0, 1, 0, -1, 1, 0],
            &[0, 0, 0, 1, 1, 0, 0, 0],
            &[0, 0, 1, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 1, 1, 0, 0, 0, 1, 1],
        ]);
        let result = candidate.realize_minimum(200).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn equality_reduced_margin_resolves_a_large_coefficient_tail() {
        let candidate = configuration(&[
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[0, 0, 0, 0, 0, 0, 1, 0],
            &[0, 0, 0, 0, 0, 1, 0, 0],
            &[0, 0, 0, 0, 1, 0, 0, 0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[0, 0, 1, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[1, 4, -1, -3, 2, -1, -1, 1],
            &[2, 2, -5, 1, 1, 1, -3, 4],
        ]);
        let result = candidate.realize_minimum(200).unwrap();
        assert_eq!(result.status(), "nonrealizable");
        assert_eq!(
            result.certificate().unwrap().kind(),
            "nonpositive_outsider_margin"
        );
        result.verify().unwrap();
    }

    #[test]
    #[ignore = "historical rounding-sensitive exact-LP tail regression"]
    fn compact_exact_separators_resolve_rounding_sensitive_tails() {
        for rows in [
            vec![
                vec![0, 0, 0, 0, 0, 0, 0, 1],
                vec![0, 0, 0, 0, 0, 0, 1, 0],
                vec![0, 0, 0, 0, 0, 1, 0, 0],
                vec![0, 0, 0, 0, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 1, 0, 0, 0, 0, 0],
                vec![0, 1, 0, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0, 0, 0, 0],
                vec![2, -3, -4, -1, -1, -1, 1, -2],
                vec![4, -3, -1, -1, -2, -1, 2, 0],
            ],
            vec![
                vec![0, 0, 0, 0, 0, 0, 0, 1],
                vec![0, 0, 0, 0, 0, 0, 1, 0],
                vec![0, 0, 0, 0, 0, 1, 0, 0],
                vec![0, 0, 0, 0, 1, 0, 0, 0],
                vec![0, 0, 0, 1, 0, 0, 0, 0],
                vec![0, 0, 1, 0, 0, 0, 0, 0],
                vec![0, 1, 0, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0, 0, 0, 0],
                vec![2, -3, -1, 2, 2, 2, -3, 1],
                vec![2, -3, 1, 2, 2, -1, -1, 0],
            ],
        ] {
            let references = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let candidate = configuration(&references);
            let result = candidate.realize_minimum(200).unwrap();
            assert_eq!(result.status(), "nonrealizable");
            assert_eq!(
                result.certificate().unwrap().kind(),
                "nonpositive_outsider_margin"
            );
            result.verify().unwrap();
        }
    }

    #[test]
    fn classifies_all_watson_index_four_candidates() {
        let mut realized = 0_usize;
        let mut nonrealizable = 0_usize;
        for twos in 0_usize..=8 {
            for ones in 1_usize..=8 - twos {
                if 2 * twos + ones <= 8 {
                    continue;
                }
                let code = [
                    vec![2_i64; twos],
                    vec![1_i64; ones],
                    vec![0_i64; 8 - twos - ones],
                ]
                .concat();
                let pivot = twos;
                let rows = (0..8)
                    .map(|old_coordinate| {
                        let mut row = vec![0_i64; 8];
                        if old_coordinate == pivot {
                            row[0] = 4;
                            for (index, coefficient) in code.iter().enumerate() {
                                if index != pivot {
                                    let new_index = if index < pivot { index + 1 } else { index };
                                    row[new_index] = -*coefficient;
                                }
                            }
                        } else {
                            let new_index = if old_coordinate < pivot {
                                old_coordinate + 1
                            } else {
                                old_coordinate
                            };
                            row[new_index] = 1;
                        }
                        row
                    })
                    .collect::<Vec<_>>();
                let Ok(candidate) = IntegralLineConfiguration::new(
                    8,
                    rows.into_iter()
                        .map(|row| row.into_iter().map(Integer::from).collect::<Vec<_>>()),
                ) else {
                    continue;
                };
                let result = candidate.realize_minimum(200).unwrap();
                result.verify().unwrap();
                match result.status() {
                    "realized" => realized += 1,
                    "nonrealizable" => nonrealizable += 1,
                    status => panic!("unexpected Watson classification {status}"),
                }
            }
        }
        assert_eq!(realized, 4);
        assert_eq!(nonrealizable, 8);
    }

    #[test]
    fn realizes_the_a2_minimum_shell() {
        let candidate = configuration(&[&[1, 0], &[0, 1], &[1, -1]]);
        let result = candidate.realize_minimum(20).unwrap();
        assert_eq!(result.status(), "realized");
        result.verify().unwrap();
    }

    #[test]
    fn proves_the_four_rank_two_lines_nonrealizable() {
        let candidate = configuration(&[&[1, 0], &[0, 1], &[1, 1], &[1, -1]]);
        let result = candidate.realize_minimum(20).unwrap();
        assert_eq!(result.status(), "nonrealizable");
        assert!(result.certificate().is_some());
        result.verify().unwrap();
    }

    #[test]
    fn work_limit_is_inconclusive_not_negative() {
        let candidate = configuration(&[&[1, 0], &[0, 1]]);
        let result = candidate.realize_minimum(0).unwrap();
        assert_eq!(result.status(), "inconclusive");
        assert_eq!(result.diagnostic().unwrap().iterations(), 0);
        result.verify().unwrap();
    }

    #[test]
    fn fixed_rank_completion_accepts_an_already_complete_shell() {
        let seed = configuration(&[&[1, 0], &[0, 1]]);
        let result = seed.complete_minimum(20).unwrap();
        assert_eq!(result.status(), "completed");
        assert_eq!(result.realization().unwrap().configuration(), &seed);
        result.verify().unwrap();
    }

    #[test]
    fn fixed_rank_completion_rejects_inconsistent_equal_lines() {
        let seed = configuration(&[&[1, 0], &[0, 1], &[1, 1], &[1, -1]]);
        let result = seed.complete_minimum(20).unwrap();
        assert_eq!(result.status(), "nonrealizable");
        assert_eq!(
            result.certificate().unwrap().kind(),
            "inconsistent_equal_norms"
        );
        result.verify().unwrap();
    }

    #[test]
    fn fixed_rank_completion_work_limit_is_inconclusive() {
        let seed = configuration(&[&[1, 0], &[0, 1]]);
        let result = seed.complete_minimum(0).unwrap();
        assert_eq!(result.status(), "inconclusive");
        assert_eq!(result.diagnostic().unwrap().iterations(), 0);
        result.verify().unwrap();
    }

    #[test]
    #[ignore = "historical 1800-second completion-tail regression"]
    fn native_completion_resolves_the_extreme_voronoi_tail() {
        // sha256:9d2230f7b6cc8d61bf3a3205047fe436031d9c390672b9b632c14a7cab513993
        let seed = configuration(&[
            &[0, 0, 0, 0, 0, 0, 0, 1],
            &[0, 0, 0, 0, 0, 0, 1, 0],
            &[0, 0, 0, 0, 0, 1, 0, 0],
            &[0, 0, 0, 0, 1, 0, 0, 0],
            &[0, 0, 0, 1, 0, 0, 0, 0],
            &[0, 0, 1, 0, 0, 0, 0, 0],
            &[0, 1, 0, 0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 0, 0, 0, 0],
            &[2, 2, -5, 1, 1, 1, 2, -1],
            &[3, 3, -5, 1, 1, 2, 2, -2],
        ]);

        let result = seed.complete_minimum(1_000).unwrap();

        assert_eq!(result.status(), "nonrealizable");
        assert_eq!(
            result.certificate().unwrap().kind(),
            "nonpositive_outsider_margin"
        );
        result.verify().unwrap();
    }
}
