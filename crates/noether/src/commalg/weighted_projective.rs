// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Weighted projective presentations and cyclic hypersurface covers.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::{factor, FiniteField};
use crate::{
    CommutativeRing, ContextMismatchError, Field, Integer, NoetherError, RationalField, Ring,
    SemanticType, TypeDescription,
};

use super::scheme::{affine_closed_subscheme, AffineSpace};
use super::{
    GradingError, GroebnerError, Ideal, MonomialOrder, Polynomial, PolynomialContextError,
    PolynomialError, PolynomialRing, ProjectiveSpace, Scheme, SchemeError, WeightedGrading,
};

static WEIGHTED_PROJECTIVE_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/WeightedProjectiveSpace@1",
    rust_path: "noether::commalg::WeightedProjectiveSpace",
    python_path: Some("noether.commalg.WeightedProjectiveSpace"),
};

static WEIGHTED_PROJECTIVE_SCHEME_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/WeightedProjectiveScheme@1",
    rust_path: "noether::commalg::WeightedProjectiveScheme",
    python_path: Some("noether.commalg.WeightedProjectiveScheme"),
};

static WEIGHTED_PROJECTIVE_HYPERSURFACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/WeightedProjectiveHypersurface@1",
    rust_path: "noether::commalg::WeightedProjectiveHypersurface",
    python_path: Some("noether.commalg.WeightedProjectiveHypersurface"),
};

static CYCLIC_COVER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CyclicHypersurfaceCover@1",
    rust_path: "noether::commalg::CyclicHypersurfaceCover",
    python_path: Some("noether.commalg.CyclicHypersurfaceCover"),
};

/// Failure to construct or analyze a weighted projective presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightedProjectiveError {
    /// A polynomial-ring context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Exact ideal arithmetic failed.
    Groebner(GroebnerError),
    /// An ordinary affine/projective scheme operation failed.
    Scheme(SchemeError),
    /// A grading could not be constructed.
    Grading(GradingError),
    /// A value belonged to another explicit context.
    ContextMismatch(ContextMismatchError),
    /// The coordinate count did not match the weight vector.
    CoordinateNameCount {
        /// Number of weights.
        expected: usize,
        /// Number of coordinate names.
        actual: usize,
    },
    /// A defining equation was not homogeneous for the ambient weights.
    NonHomogeneousEquation,
    /// An indexed coordinate does not exist.
    CoordinateIndex {
        /// Rejected coordinate index.
        index: usize,
        /// Number of coordinates.
        count: usize,
    },
    /// A cyclic cover degree must be at least two.
    InvalidCoverDegree(u64),
    /// The branch degree is not divisible by the cyclic cover degree.
    IncompatibleBranchDegree {
        /// Degree of the branch equation.
        branch_degree: Integer,
        /// Degree of the cyclic cover.
        cover_degree: u64,
    },
    /// A branch equation was zero.
    ZeroBranchEquation,
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for WeightedProjectiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Scheme(error) => error.fmt(formatter),
            Self::Grading(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::CoordinateNameCount { expected, actual } => write!(
                formatter,
                "weighted projective space needs {expected} coordinate names, received {actual}"
            ),
            Self::NonHomogeneousEquation => formatter.write_str(
                "a weighted projective closed subscheme requires weighted-homogeneous equations",
            ),
            Self::CoordinateIndex { index, count } => write!(
                formatter,
                "weighted projective coordinate {index} is outside coordinate count {count}"
            ),
            Self::InvalidCoverDegree(degree) => {
                write!(formatter, "cyclic cover degree must be at least two; received {degree}")
            }
            Self::IncompatibleBranchDegree {
                branch_degree,
                cover_degree,
            } => write!(
                formatter,
                "branch degree {branch_degree} is not divisible by cyclic cover degree {cover_degree}"
            ),
            Self::ZeroBranchEquation => {
                formatter.write_str("a cyclic cover requires a nonzero branch equation")
            }
            Self::InvalidEncoding(message) => {
                write!(formatter, "invalid weighted-projective encoding: {message}")
            }
        }
    }
}

impl std::error::Error for WeightedProjectiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Scheme(error) => Some(error),
            Self::Grading(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolynomialContextError> for WeightedProjectiveError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for WeightedProjectiveError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<GroebnerError> for WeightedProjectiveError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<SchemeError> for WeightedProjectiveError {
    fn from(value: SchemeError) -> Self {
        Self::Scheme(value)
    }
}

impl From<GradingError> for WeightedProjectiveError {
    fn from(value: GradingError) -> Self {
        Self::Grading(value)
    }
}

impl From<ContextMismatchError> for WeightedProjectiveError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A weighted projective space over one explicit coefficient context.
///
/// The constructor divides the complete weight vector by its gcd. Thus the
/// stored grading is primitive, and equality includes that primitive vector,
/// the ordered coordinate names, term order, and base context.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedProjectiveSpace<R: CommutativeRing> {
    coordinate_ring: PolynomialRing<R>,
    weights: Vec<u64>,
}

impl<R: CommutativeRing> WeightedProjectiveSpace<R> {
    /// Construct a weighted projective space with default names `x0, ...`.
    pub fn new(base: R, weights: Vec<u64>) -> Result<Self, WeightedProjectiveError> {
        let names = (0..weights.len()).map(|index| format!("x{index}"));
        Self::with_names(base, weights, names)
    }

    /// Construct a weighted projective space with explicit ordered names.
    pub fn with_names<I, S>(
        base: R,
        weights: Vec<u64>,
        names: I,
    ) -> Result<Self, WeightedProjectiveError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names = names.into_iter().map(Into::into).collect::<Vec<_>>();
        if names.len() != weights.len() {
            return Err(WeightedProjectiveError::CoordinateNameCount {
                expected: weights.len(),
                actual: names.len(),
            });
        }
        let weights = primitive_weights(weights)?;
        let grading = WeightedGrading::new(weights.clone())?;
        Ok(Self {
            coordinate_ring: PolynomialRing::with_grading(
                base,
                names,
                MonomialOrder::default(),
                grading,
            )?,
            weights,
        })
    }

    fn from_coordinate_ring(
        coordinate_ring: PolynomialRing<R>,
    ) -> Result<Self, WeightedProjectiveError> {
        if coordinate_ring.variables().is_empty() {
            return Err(WeightedProjectiveError::CoordinateNameCount {
                expected: 1,
                actual: 0,
            });
        }
        let weights = coordinate_ring
            .weighted_grading()
            .ok_or_else(|| {
                WeightedProjectiveError::InvalidEncoding(
                    "weighted-projective coordinate ring requires a positive rank-one grading"
                        .to_owned(),
                )
            })?
            .weights()
            .to_vec();
        let primitive = primitive_weights(weights.clone())?;
        if primitive != weights {
            return Err(WeightedProjectiveError::InvalidEncoding(
                "weighted-projective coordinate ring does not use primitive weights".to_owned(),
            ));
        }
        Ok(Self {
            coordinate_ring,
            weights,
        })
    }

    /// Base coefficient context.
    pub fn base(&self) -> &R {
        self.coordinate_ring.coefficient_ring()
    }

    /// Dimension of the ambient weighted projective space.
    pub fn dimension(&self) -> usize {
        self.coordinate_ring.variables().len() - 1
    }

    /// Ordered homogeneous coordinate names.
    pub fn coordinates(&self) -> &[String] {
        self.coordinate_ring.variables()
    }

    /// Primitive positive weights in coordinate order.
    pub fn weights(&self) -> &[u64] {
        &self.weights
    }

    /// Exact weighted homogeneous coordinate ring.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        &self.coordinate_ring
    }

    /// Whether the primitive weight vector is well formed.
    ///
    /// This is the usual condition that omitting any one weight leaves gcd
    /// one. It is distinct from primitivity of the complete vector.
    pub fn is_well_formed(&self) -> bool {
        (0..self.weights().len()).all(|omitted| {
            self.weights()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != omitted)
                .map(|(_, weight)| *weight)
                .reduce(gcd_u64)
                .unwrap_or(1)
                == 1
        })
    }

    /// One standard quotient chart `x_index != 0`.
    ///
    /// It is an affine-space chart exactly when the indexed weight is one.
    /// Otherwise it records `A^n / mu_weight` and the diagonal action weights;
    /// no false affine-space identification is made.
    pub fn chart(
        &self,
        index: usize,
    ) -> Result<WeightedProjectiveChart<R>, WeightedProjectiveError> {
        let Some(&stabilizer_order) = self.weights().get(index) else {
            return Err(WeightedProjectiveError::CoordinateIndex {
                index,
                count: self.weights().len(),
            });
        };
        let coordinates = self
            .coordinates()
            .iter()
            .enumerate()
            .filter(|(coordinate, _)| *coordinate != index)
            .map(|(_, name)| name.clone())
            .collect::<Vec<_>>();
        let action_weights = self
            .weights()
            .iter()
            .enumerate()
            .filter(|(coordinate, _)| *coordinate != index)
            .map(|(_, weight)| weight % stabilizer_order)
            .collect::<Vec<_>>();
        let affine_space =
            AffineSpace::with_names(self.base().clone(), coordinates.len(), coordinates)?;
        Ok(WeightedProjectiveChart {
            ambient: self.clone(),
            coordinate_index: index,
            affine_space,
            stabilizer_order,
            action_weights,
        })
    }

    /// Coordinate strata carrying a nontrivial generic quotient stabilizer.
    ///
    /// Only maximal support sets are returned. These are quotient strata, not
    /// automatically singular strata of the coarse space; the distinction is
    /// intentional (for example in dimension one).
    pub fn quotient_strata(&self) -> Vec<WeightedProjectiveStratum> {
        // Every maximal support with nontrivial stabilizer is precisely the
        // set of coordinates whose weights are divisible by one prime.  This
        // characterization avoids enumerating the power set of coordinates
        // and remains correct when there are more coordinates than bits in a
        // machine word.
        let mut primes = self
            .weights()
            .iter()
            .flat_map(|weight| {
                factor(&Integer::from(*weight))
                    .expect("positive normalized weights always admit factorization")
                    .factors()
                    .iter()
                    .map(|(prime, _)| {
                        prime
                            .to_u64_checked()
                            .expect("a prime divisor of a u64 weight fits u64")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        primes.sort_unstable();
        primes.dedup();

        let mut candidates = primes
            .into_iter()
            .map(|prime| {
                let support = self
                    .weights()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, weight)| (weight % prime == 0).then_some(index))
                    .collect::<Vec<_>>();
                let stabilizer_order = support
                    .iter()
                    .map(|index| self.weights()[*index])
                    .reduce(gcd_u64)
                    .unwrap_or(1);
                WeightedProjectiveStratum {
                    support,
                    stabilizer_order,
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| left.support.cmp(&right.support));
        candidates.dedup_by(|left, right| left.support == right.support);
        let all_supports = candidates
            .iter()
            .map(|candidate| candidate.support.clone())
            .collect::<Vec<_>>();
        candidates.retain(|candidate| {
            !all_supports.iter().any(|other| {
                candidate.support.len() < other.len()
                    && candidate.support.iter().all(|index| other.contains(index))
            })
        });
        candidates.sort_by(|left, right| left.support.cmp(&right.support));
        candidates
    }
}

/// A maximal coordinate stratum with nontrivial generic quotient stabilizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedProjectiveStratum {
    support: Vec<usize>,
    stabilizer_order: u64,
}

impl WeightedProjectiveStratum {
    /// Coordinates allowed to be nonzero on the stratum.
    pub fn support(&self) -> &[usize] {
        &self.support
    }

    /// Gcd of the supported coordinate weights.
    pub fn stabilizer_order(&self) -> u64 {
        self.stabilizer_order
    }
}

/// A standard quotient chart of a weighted projective space.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedProjectiveChart<R: CommutativeRing> {
    ambient: WeightedProjectiveSpace<R>,
    coordinate_index: usize,
    affine_space: AffineSpace<R>,
    stabilizer_order: u64,
    action_weights: Vec<u64>,
}

impl<R: CommutativeRing> WeightedProjectiveChart<R> {
    /// Ambient weighted projective space.
    pub fn ambient(&self) -> &WeightedProjectiveSpace<R> {
        &self.ambient
    }

    /// Coordinate inverted on this chart.
    pub fn coordinate_index(&self) -> usize {
        self.coordinate_index
    }

    /// Affine cover before taking the residual cyclic quotient.
    pub fn affine_cover(&self) -> &AffineSpace<R> {
        &self.affine_space
    }

    /// Order of the residual diagonal cyclic action.
    pub fn stabilizer_order(&self) -> u64 {
        self.stabilizer_order
    }

    /// Residual action weights modulo the stabilizer order.
    pub fn action_weights(&self) -> &[u64] {
        &self.action_weights
    }

    /// Whether this quotient chart is honestly an affine-space chart.
    pub fn is_affine_space(&self) -> bool {
        self.stabilizer_order == 1
    }
}

/// A weighted-projective closed subscheme with homogeneous defining ideal.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedProjectiveScheme<R: CommutativeRing> {
    ambient: WeightedProjectiveSpace<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
}

impl<R: CommutativeRing> WeightedProjectiveScheme<R> {
    /// Ambient weighted projective space.
    pub fn ambient(&self) -> &WeightedProjectiveSpace<R> {
        &self.ambient
    }

    /// Canonical weighted-homogeneous defining ideal.
    pub fn defining_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.defining_ideal
    }

    /// Weighted homogeneous coordinate ring.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        self.ambient.coordinate_ring()
    }

    /// Base coefficient context.
    pub fn base(&self) -> &R {
        self.ambient.base()
    }
}

impl<R: Field> WeightedProjectiveSpace<R> {
    /// Construct a weighted-homogeneous closed subscheme.
    pub fn closed_subscheme(
        &self,
        ideal: Ideal<PolynomialRing<R>>,
    ) -> Result<WeightedProjectiveScheme<R>, WeightedProjectiveError> {
        if ideal.ring() != self.coordinate_ring() {
            return Err(ContextMismatchError {
                expected: self.coordinate_ring().to_string(),
                actual: ideal.ring().to_string(),
            }
            .into());
        }
        if ideal.generators().iter().any(|generator| {
            !generator.is_zero() && generator.homogeneous_degree().ok().flatten().is_none()
        }) {
            return Err(WeightedProjectiveError::NonHomogeneousEquation);
        }
        Ok(WeightedProjectiveScheme {
            ambient: self.clone(),
            defining_ideal: ideal,
        })
    }

    /// Construct a single weighted-projective hypersurface.
    pub fn hypersurface(
        &self,
        equation: Polynomial<R>,
    ) -> Result<WeightedProjectiveHypersurface<R>, WeightedProjectiveError> {
        if equation.ring() != self.coordinate_ring() {
            return Err(ContextMismatchError {
                expected: self.coordinate_ring().to_string(),
                actual: equation.ring().to_string(),
            }
            .into());
        }
        if equation.is_zero() || equation.homogeneous_degree()?.is_none() {
            return Err(WeightedProjectiveError::NonHomogeneousEquation);
        }
        let ideal = self.coordinate_ring().ideal([equation.clone()])?;
        Ok(WeightedProjectiveHypersurface {
            scheme: self.closed_subscheme(ideal)?,
            equation,
        })
    }
}

impl<R: Field> WeightedProjectiveScheme<R> {
    /// Dimension of this weighted projective presentation.
    pub fn dimension(&self) -> Result<i64, WeightedProjectiveError> {
        let cone_dimension = self.defining_ideal.dimension()?;
        Ok(if cone_dimension >= 0 {
            cone_dimension - 1
        } else {
            cone_dimension
        })
    }

    /// Pull this presentation back to one affine cover of a quotient chart.
    pub fn chart(
        &self,
        index: usize,
    ) -> Result<WeightedProjectiveSchemeChart<R>, WeightedProjectiveError> {
        let quotient_chart = self.ambient.chart(index)?;
        let ring = quotient_chart.affine_cover().coordinate_ring();
        let equations = self
            .defining_ideal
            .generators()
            .iter()
            .map(|equation| dehomogenize(equation, ring, index))
            .collect::<Result<Vec<_>, _>>()?;
        let ideal = ring.ideal(equations)?;
        let affine_cover = affine_closed_subscheme(quotient_chart.affine_cover(), ideal)?;
        Ok(WeightedProjectiveSchemeChart {
            quotient_chart,
            affine_cover,
        })
    }
}

/// Pullback of a weighted-projective scheme to a standard quotient chart.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedProjectiveSchemeChart<R: CommutativeRing> {
    quotient_chart: WeightedProjectiveChart<R>,
    affine_cover: Scheme<R>,
}

impl<R: CommutativeRing> WeightedProjectiveSchemeChart<R> {
    /// Ambient quotient-chart data.
    pub fn quotient_chart(&self) -> &WeightedProjectiveChart<R> {
        &self.quotient_chart
    }

    /// Closed subscheme of the affine cover before the cyclic quotient.
    pub fn affine_cover(&self) -> &Scheme<R> {
        &self.affine_cover
    }
}

/// A single weighted-homogeneous hypersurface.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedProjectiveHypersurface<R: CommutativeRing> {
    scheme: WeightedProjectiveScheme<R>,
    equation: Polynomial<R>,
}

impl<R: CommutativeRing> WeightedProjectiveHypersurface<R> {
    /// Underlying weighted-projective closed subscheme.
    pub fn scheme(&self) -> &WeightedProjectiveScheme<R> {
        &self.scheme
    }

    /// Canonical defining equation as supplied in the ambient ring.
    pub fn equation(&self) -> &Polynomial<R> {
        &self.equation
    }

    /// Ambient weighted projective space.
    pub fn ambient(&self) -> &WeightedProjectiveSpace<R> {
        self.scheme.ambient()
    }

    /// Weighted degree of the equation.
    pub fn degree(&self) -> Integer {
        self.equation
            .homogeneous_degree()
            .expect("a checked hypersurface has measurable weighted degree")
            .expect("a checked hypersurface is nonzero and homogeneous")
    }
}

impl<R: Field> WeightedProjectiveHypersurface<R> {
    /// Prove or disprove quasismoothness by the affine-cone Jacobian criterion.
    ///
    /// The critical ideal is saturated by the irrelevant ideal. Hence a true
    /// result says that the affine cone is smooth away from its vertex. It
    /// does not claim the coarse weighted-projective variety avoids ambient
    /// quotient singularities; those are recorded separately.
    pub fn jacobian_evidence(
        &self,
    ) -> Result<WeightedHypersurfaceJacobianEvidence<R>, WeightedProjectiveError> {
        let ring = self.ambient().coordinate_ring();
        let mut critical_generators = vec![self.equation.clone()];
        for variable in 0..ring.variables().len() {
            critical_generators.push(self.equation.derivative(variable)?);
        }
        let critical = ring.ideal(critical_generators)?;
        let irrelevant = ring.ideal(ring.generators()?)?;
        let away_from_vertex = critical.saturate(&irrelevant)?;
        let quasismooth = away_from_vertex.contains(&ring.one())?;
        let evidence = WeightedHypersurfaceJacobianEvidence {
            hypersurface: self.clone(),
            saturated_critical_ideal: away_from_vertex,
            quasismooth,
            ambient_quotient_strata: self.ambient().quotient_strata(),
        };
        evidence.verify()?;
        Ok(evidence)
    }

    /// Whether the affine cone is smooth away from its vertex.
    pub fn is_quasismooth(&self) -> Result<bool, WeightedProjectiveError> {
        self.jacobian_evidence()
            .map(|evidence| evidence.is_quasismooth())
    }
}

/// Exact cone-Jacobian evidence for one weighted hypersurface.
#[derive(Clone, PartialEq, Eq)]
pub struct WeightedHypersurfaceJacobianEvidence<R: CommutativeRing> {
    hypersurface: WeightedProjectiveHypersurface<R>,
    saturated_critical_ideal: Ideal<PolynomialRing<R>>,
    quasismooth: bool,
    ambient_quotient_strata: Vec<WeightedProjectiveStratum>,
}

impl<R: Field> WeightedHypersurfaceJacobianEvidence<R> {
    /// Hypersurface checked by this evidence.
    pub fn hypersurface(&self) -> &WeightedProjectiveHypersurface<R> {
        &self.hypersurface
    }

    /// Critical ideal after removing the cone vertex.
    pub fn saturated_critical_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.saturated_critical_ideal
    }

    /// Whether the cone has no critical point away from its vertex.
    pub fn is_quasismooth(&self) -> bool {
        self.quasismooth
    }

    /// Ambient quotient strata, kept distinct from cone singularities.
    pub fn ambient_quotient_strata(&self) -> &[WeightedProjectiveStratum] {
        &self.ambient_quotient_strata
    }

    /// Recompute and compare the complete Jacobian result.
    pub fn verify(&self) -> Result<(), WeightedProjectiveError> {
        let ring = self.hypersurface.ambient().coordinate_ring();
        if self.saturated_critical_ideal.ring() != ring {
            return Err(WeightedProjectiveError::InvalidEncoding(
                "saturated critical ideal belongs to another ring".to_owned(),
            ));
        }
        let expected_quasismooth = self.saturated_critical_ideal.contains(&ring.one())?;
        if expected_quasismooth != self.quasismooth
            || self.ambient_quotient_strata != self.hypersurface.ambient().quotient_strata()
        {
            return Err(WeightedProjectiveError::InvalidEncoding(
                "weighted hypersurface Jacobian evidence does not match its presentation"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Ramification regime of a cyclic cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CyclicCoverRamification {
    /// The cover degree is invertible in the base field.
    Tame,
    /// The characteristic divides the cover degree.
    Wild,
}

/// A cyclic cover of projective space branched along a hypersurface.
#[derive(Clone, PartialEq, Eq)]
pub struct CyclicHypersurfaceCover<R: CommutativeRing> {
    base: ProjectiveSpace<R>,
    cover_degree: u64,
    branch_equation: Polynomial<R>,
    branch: Scheme<R>,
    weighted_realization: WeightedProjectiveHypersurface<R>,
    ramification: CyclicCoverRamification,
}

impl CyclicHypersurfaceCover<RationalField> {
    /// Construct a characteristic-zero cyclic hypersurface cover.
    pub fn new(
        base: ProjectiveSpace<RationalField>,
        cover_degree: u64,
        branch_equation: Polynomial<RationalField>,
    ) -> Result<Self, WeightedProjectiveError> {
        construct_cyclic_cover(
            base,
            cover_degree,
            branch_equation,
            CyclicCoverRamification::Tame,
        )
    }
}

impl CyclicHypersurfaceCover<FiniteField> {
    /// Construct a finite-field cyclic hypersurface cover.
    pub fn new(
        base: ProjectiveSpace<FiniteField>,
        cover_degree: u64,
        branch_equation: Polynomial<FiniteField>,
    ) -> Result<Self, WeightedProjectiveError> {
        let characteristic_divides_degree =
            Integer::from(cover_degree) % base.base().characteristic() == Integer::zero();
        construct_cyclic_cover(
            base,
            cover_degree,
            branch_equation,
            if characteristic_divides_degree {
                CyclicCoverRamification::Wild
            } else {
                CyclicCoverRamification::Tame
            },
        )
    }
}

impl<R: CommutativeRing> CyclicHypersurfaceCover<R> {
    /// Base projective space.
    pub fn base(&self) -> &ProjectiveSpace<R> {
        &self.base
    }

    /// Cyclic cover degree.
    pub fn cover_degree(&self) -> u64 {
        self.cover_degree
    }

    /// Homogeneous branch equation in the base coordinates.
    pub fn branch_equation(&self) -> &Polynomial<R> {
        &self.branch_equation
    }

    /// Branch hypersurface in the base projective space.
    pub fn branch(&self) -> &Scheme<R> {
        &self.branch
    }

    /// Equation `w^m-f=0` in its weighted projective ambient space.
    pub fn weighted_realization(&self) -> &WeightedProjectiveHypersurface<R> {
        &self.weighted_realization
    }

    /// Tame or wild ramification regime.
    pub fn ramification(&self) -> CyclicCoverRamification {
        self.ramification
    }
}

macro_rules! impl_cyclic_cover_smoothness {
    ($field:ty) => {
        impl CyclicHypersurfaceCover<$field> {
            /// Compute smoothness evidence using the criterion valid in this regime.
            ///
            /// Tame covers check both branch smoothness and the cone Jacobian. Wild
            /// covers deliberately do not apply the tame branch criterion and rely on
            /// the direct equation Jacobian only.
            pub fn smoothness_evidence(
                &self,
            ) -> Result<CyclicCoverSmoothnessEvidence<$field>, WeightedProjectiveError> {
                cyclic_cover_smoothness(self)
            }
        }
    };
}

impl_cyclic_cover_smoothness!(RationalField);
impl_cyclic_cover_smoothness!(FiniteField);

/// Regime-aware smoothness evidence for a cyclic hypersurface cover.
#[derive(Clone, PartialEq, Eq)]
pub struct CyclicCoverSmoothnessEvidence<R: CommutativeRing> {
    cover: CyclicHypersurfaceCover<R>,
    branch_smooth: Option<bool>,
    weighted_jacobian: WeightedHypersurfaceJacobianEvidence<R>,
    smooth: bool,
}

impl<R: CommutativeRing> CyclicCoverSmoothnessEvidence<R> {
    /// Checked cover.
    pub fn cover(&self) -> &CyclicHypersurfaceCover<R> {
        &self.cover
    }

    /// Branch criterion result in the tame regime, absent in the wild regime.
    pub fn branch_is_smooth(&self) -> Option<bool> {
        self.branch_smooth
    }

    /// Direct weighted cone-Jacobian evidence in every regime.
    pub fn weighted_jacobian(&self) -> &WeightedHypersurfaceJacobianEvidence<R> {
        &self.weighted_jacobian
    }

    /// Smoothness conclusion under the regime-specific rule.
    pub fn is_smooth(&self) -> bool {
        self.smooth
    }
}

fn cyclic_cover_smoothness<R>(
    cover: &CyclicHypersurfaceCover<R>,
) -> Result<CyclicCoverSmoothnessEvidence<R>, WeightedProjectiveError>
where
    R: SupportedWeightedField,
{
    let branch_smooth = match cover.ramification {
        CyclicCoverRamification::Tame => Some(R::scheme_is_smooth(&cover.branch)?),
        CyclicCoverRamification::Wild => None,
    };
    let jacobian = cover.weighted_realization.jacobian_evidence()?;
    let smooth = match branch_smooth {
        Some(branch_smooth) => branch_smooth && jacobian.is_quasismooth(),
        None => jacobian.is_quasismooth(),
    };
    Ok(CyclicCoverSmoothnessEvidence {
        cover: cover.clone(),
        branch_smooth,
        weighted_jacobian: jacobian,
        smooth,
    })
}

trait SupportedWeightedField: Field {
    fn projective_closed_subscheme(
        space: &ProjectiveSpace<Self>,
        ideal: Ideal<PolynomialRing<Self>>,
    ) -> Result<Scheme<Self>, SchemeError>
    where
        Self: Sized;

    fn scheme_is_smooth(scheme: &Scheme<Self>) -> Result<bool, SchemeError>
    where
        Self: Sized;
}

impl SupportedWeightedField for RationalField {
    fn projective_closed_subscheme(
        space: &ProjectiveSpace<Self>,
        ideal: Ideal<PolynomialRing<Self>>,
    ) -> Result<Scheme<Self>, SchemeError> {
        space.closed_subscheme(ideal)
    }

    fn scheme_is_smooth(scheme: &Scheme<Self>) -> Result<bool, SchemeError> {
        scheme.is_smooth()
    }
}

impl SupportedWeightedField for FiniteField {
    fn projective_closed_subscheme(
        space: &ProjectiveSpace<Self>,
        ideal: Ideal<PolynomialRing<Self>>,
    ) -> Result<Scheme<Self>, SchemeError> {
        space.closed_subscheme(ideal)
    }

    fn scheme_is_smooth(scheme: &Scheme<Self>) -> Result<bool, SchemeError> {
        scheme.is_smooth()
    }
}

fn construct_cyclic_cover<R>(
    base: ProjectiveSpace<R>,
    cover_degree: u64,
    branch_equation: Polynomial<R>,
    ramification: CyclicCoverRamification,
) -> Result<CyclicHypersurfaceCover<R>, WeightedProjectiveError>
where
    R: SupportedWeightedField,
{
    if cover_degree < 2 {
        return Err(WeightedProjectiveError::InvalidCoverDegree(cover_degree));
    }
    if branch_equation.ring() != base.coordinate_ring() {
        return Err(ContextMismatchError {
            expected: base.coordinate_ring().to_string(),
            actual: branch_equation.ring().to_string(),
        }
        .into());
    }
    if branch_equation.is_zero() {
        return Err(WeightedProjectiveError::ZeroBranchEquation);
    }
    let branch_degree = branch_equation
        .homogeneous_degree()?
        .ok_or(WeightedProjectiveError::NonHomogeneousEquation)?;
    let cover_degree_integer = Integer::from(cover_degree);
    let remainder = &branch_degree % &cover_degree_integer;
    let cover_weight = &branch_degree / &cover_degree_integer;
    if !remainder.is_zero() || cover_weight.sign() <= 0 {
        return Err(WeightedProjectiveError::IncompatibleBranchDegree {
            branch_degree,
            cover_degree,
        });
    }
    let cover_weight = cover_weight.to_string().parse::<u64>().map_err(|_| {
        WeightedProjectiveError::InvalidEncoding(
            "cyclic-cover weight does not fit the weighted grading".to_owned(),
        )
    })?;
    let branch_ideal = base.coordinate_ring().ideal([branch_equation.clone()])?;
    let branch = R::projective_closed_subscheme(&base, branch_ideal)?;

    let mut names = Vec::with_capacity(base.coordinates().len() + 1);
    names.push("w".to_owned());
    names.extend(base.coordinates().iter().cloned());
    let mut weights = Vec::with_capacity(names.len());
    weights.push(cover_weight);
    weights.extend(core::iter::repeat_n(1_u64, base.coordinates().len()));
    let ambient = WeightedProjectiveSpace::with_names(base.base().clone(), weights, names)?;
    let ring = ambient.coordinate_ring();
    let w_power = ring.generator(0)?.pow(cover_degree)?;
    let lifted_branch =
        ring.from_terms(branch_equation.terms().map(|(exponent, coefficient)| {
            let mut lifted = Vec::with_capacity(exponent.len() + 1);
            lifted.push(Integer::zero());
            lifted.extend(exponent.iter().cloned());
            (lifted, coefficient.clone())
        }))?;
    let equation = ring.sub(&w_power, &lifted_branch)?;
    let weighted_realization = ambient.hypersurface(equation)?;
    Ok(CyclicHypersurfaceCover {
        base,
        cover_degree,
        branch_equation,
        branch,
        weighted_realization,
        ramification,
    })
}

fn dehomogenize<R: CommutativeRing>(
    polynomial: &Polynomial<R>,
    target: &PolynomialRing<R>,
    coordinate: usize,
) -> Result<Polynomial<R>, WeightedProjectiveError> {
    target
        .from_terms(polynomial.terms().map(|(exponent, coefficient)| {
            let exponent = exponent
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != coordinate)
                .map(|(_, power)| power.clone())
                .collect::<Vec<_>>();
            (exponent, coefficient.clone())
        }))
        .map_err(Into::into)
}

fn primitive_weights(mut weights: Vec<u64>) -> Result<Vec<u64>, GradingError> {
    if weights.is_empty() {
        return Err(GradingError::NoVariables);
    }
    if let Some(variable) = weights.iter().position(|weight| *weight == 0) {
        return Err(GradingError::ZeroWeight { variable });
    }
    let gcd = weights.iter().copied().reduce(gcd_u64).unwrap_or(1);
    for weight in &mut weights {
        *weight /= gcd;
    }
    Ok(weights)
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

impl<R> Serialize for WeightedProjectiveSpace<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("WeightedProjectiveSpace", 2)?;
        state.serialize_field("type", "noether/commalg/WeightedProjectiveSpace@1")?;
        state.serialize_field("coordinate_ring", &self.coordinate_ring)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "R: CommutativeRing + Deserialize<'de>"))]
struct WeightedSpaceWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    coordinate_ring: PolynomialRing<R>,
}

impl<R> Serialize for WeightedProjectiveScheme<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("WeightedProjectiveScheme", 3)?;
        state.serialize_field("type", "noether/commalg/WeightedProjectiveScheme@1")?;
        state.serialize_field("ambient", &self.ambient)?;
        state.serialize_field("defining_ideal", &self.defining_ideal)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(bound(
    deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, WeightedProjectiveSpace<R>: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>"
))]
struct WeightedSchemeWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ambient: WeightedProjectiveSpace<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
}

impl<R> Serialize for WeightedProjectiveHypersurface<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("WeightedProjectiveHypersurface", 3)?;
        state.serialize_field("type", "noether/commalg/WeightedProjectiveHypersurface@1")?;
        state.serialize_field("ambient", self.ambient())?;
        state.serialize_field("equation", &self.equation)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(bound(
    deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, WeightedProjectiveSpace<R>: Deserialize<'de>"
))]
struct WeightedHypersurfaceWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ambient: WeightedProjectiveSpace<R>,
    equation: Polynomial<R>,
}

impl<R> Serialize for CyclicHypersurfaceCover<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CyclicHypersurfaceCover", 5)?;
        state.serialize_field("type", "noether/commalg/CyclicHypersurfaceCover@1")?;
        state.serialize_field("base", &self.base)?;
        state.serialize_field("cover_degree", &self.cover_degree)?;
        state.serialize_field("branch_equation", &self.branch_equation)?;
        state.serialize_field("ramification", &self.ramification)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(bound(
    deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, ProjectiveSpace<R>: Deserialize<'de>"
))]
struct CyclicCoverWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    base: ProjectiveSpace<R>,
    cover_degree: u64,
    branch_equation: Polynomial<R>,
    ramification: CyclicCoverRamification,
}

macro_rules! impl_weighted_serialization {
    ($field:ty, $construct:path, $expected_ramification:expr) => {
        impl<'de> Deserialize<'de> for WeightedProjectiveSpace<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = WeightedSpaceWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != Self::TYPE_ID {
                    return Err(serde::de::Error::custom(
                        "wrong weighted-projective-space type identity",
                    ));
                }
                Self::from_coordinate_ring(wire.coordinate_ring).map_err(serde::de::Error::custom)
            }
        }

        impl<'de> Deserialize<'de> for WeightedProjectiveScheme<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = WeightedSchemeWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != Self::TYPE_ID {
                    return Err(serde::de::Error::custom(
                        "wrong weighted-projective-scheme type identity",
                    ));
                }
                wire.ambient
                    .closed_subscheme(wire.defining_ideal)
                    .map_err(serde::de::Error::custom)
            }
        }

        impl<'de> Deserialize<'de> for WeightedProjectiveHypersurface<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = WeightedHypersurfaceWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != Self::TYPE_ID {
                    return Err(serde::de::Error::custom(
                        "wrong weighted-projective-hypersurface type identity",
                    ));
                }
                wire.ambient
                    .hypersurface(wire.equation)
                    .map_err(serde::de::Error::custom)
            }
        }

        impl<'de> Deserialize<'de> for CyclicHypersurfaceCover<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = CyclicCoverWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != Self::TYPE_ID {
                    return Err(serde::de::Error::custom("wrong cyclic-cover type identity"));
                }
                let rebuilt = $construct(wire.base, wire.cover_degree, wire.branch_equation)
                    .map_err(serde::de::Error::custom)?;
                if rebuilt.ramification != wire.ramification
                    || rebuilt.ramification != $expected_ramification(&rebuilt)
                {
                    return Err(serde::de::Error::custom(
                        "cyclic-cover ramification tag is inconsistent",
                    ));
                }
                Ok(rebuilt)
            }
        }

        impl WeightedProjectiveSpace<$field> {
            /// Serialize this weighted-projective context.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and revalidate this weighted-projective context.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl WeightedProjectiveScheme<$field> {
            /// Serialize this weighted-projective scheme.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and revalidate this weighted-projective scheme.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl WeightedProjectiveHypersurface<$field> {
            /// Serialize this weighted-projective hypersurface.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and revalidate this weighted-projective hypersurface.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl CyclicHypersurfaceCover<$field> {
            /// Serialize this cyclic cover and its exact branch presentation.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and revalidate this cyclic cover.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl SemanticType for WeightedProjectiveSpace<$field> {
            const TYPE_ID: &'static str = "noether/commalg/WeightedProjectiveSpace@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &WEIGHTED_PROJECTIVE_SPACE_DESCRIPTION
            }
        }

        impl SemanticType for WeightedProjectiveScheme<$field> {
            const TYPE_ID: &'static str = "noether/commalg/WeightedProjectiveScheme@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &WEIGHTED_PROJECTIVE_SCHEME_DESCRIPTION
            }
        }

        impl SemanticType for WeightedProjectiveHypersurface<$field> {
            const TYPE_ID: &'static str = "noether/commalg/WeightedProjectiveHypersurface@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &WEIGHTED_PROJECTIVE_HYPERSURFACE_DESCRIPTION
            }
        }

        impl SemanticType for CyclicHypersurfaceCover<$field> {
            const TYPE_ID: &'static str = "noether/commalg/CyclicHypersurfaceCover@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &CYCLIC_COVER_DESCRIPTION
            }
        }
    };
}

fn rational_cover(
    base: ProjectiveSpace<RationalField>,
    degree: u64,
    equation: Polynomial<RationalField>,
) -> Result<CyclicHypersurfaceCover<RationalField>, WeightedProjectiveError> {
    CyclicHypersurfaceCover::<RationalField>::new(base, degree, equation)
}

fn finite_cover(
    base: ProjectiveSpace<FiniteField>,
    degree: u64,
    equation: Polynomial<FiniteField>,
) -> Result<CyclicHypersurfaceCover<FiniteField>, WeightedProjectiveError> {
    CyclicHypersurfaceCover::<FiniteField>::new(base, degree, equation)
}

impl_weighted_serialization!(
    RationalField,
    rational_cover,
    |_: &CyclicHypersurfaceCover<RationalField>| CyclicCoverRamification::Tame
);
impl_weighted_serialization!(
    FiniteField,
    finite_cover,
    |cover: &CyclicHypersurfaceCover<FiniteField>| cover.ramification()
);

impl<R: CommutativeRing> fmt::Debug for WeightedProjectiveSpace<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedProjectiveSpace")
            .field("weights", &self.weights())
            .field("coordinates", &self.coordinates())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for WeightedProjectiveSpace<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "WeightedProjectiveSpace({:?}, {})",
            self.weights(),
            R::TYPE_ID
        )
    }
}

impl<R: CommutativeRing> fmt::Debug for WeightedProjectiveScheme<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedProjectiveScheme")
            .field("ambient", &self.ambient)
            .field("generators", &self.defining_ideal.generators().len())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Debug for WeightedProjectiveHypersurface<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedProjectiveHypersurface")
            .field("ambient", self.ambient())
            .field("degree", &self.degree())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Debug for CyclicHypersurfaceCover<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CyclicHypersurfaceCover")
            .field("degree", &self.cover_degree)
            .field("base_dimension", &self.base.dimension())
            .field("ramification", &self.ramification)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q;

    #[test]
    fn weights_are_primitive_and_well_formedness_is_separate() {
        let primitive =
            WeightedProjectiveSpace::with_names(Q, vec![6, 2, 2, 2], ["w", "x", "y", "z"]).unwrap();
        assert_eq!(primitive.weights(), &[3, 1, 1, 1]);
        assert!(primitive.is_well_formed());
        assert_eq!(primitive.chart(0).unwrap().stabilizer_order(), 3);
        assert!(!primitive.chart(0).unwrap().is_affine_space());
        assert!(primitive.chart(1).unwrap().is_affine_space());

        let not_well_formed = WeightedProjectiveSpace::new(Q, vec![1, 2, 2]).unwrap();
        assert!(!not_well_formed.is_well_formed());
        assert_eq!(not_well_formed.quotient_strata()[0].support(), &[1, 2]);
    }

    #[test]
    fn quotient_strata_do_not_depend_on_machine_word_bitmasks() {
        let mut weights = vec![2; usize::BITS as usize + 1];
        weights.push(3);
        let space = WeightedProjectiveSpace::new(Q, weights).unwrap();
        let strata = space.quotient_strata();
        assert_eq!(strata.len(), 2);
        assert_eq!(strata[0].support().len(), usize::BITS as usize + 1);
        assert_eq!(strata[0].stabilizer_order(), 2);
        assert_eq!(strata[1].support(), &[usize::BITS as usize + 1]);
        assert_eq!(strata[1].stabilizer_order(), 3);
    }

    #[test]
    fn cyclic_double_cover_and_weighted_hypersurface_are_the_same_presentation() {
        let base = ProjectiveSpace::with_names(Q, 2, ["x", "y", "z"]).unwrap();
        let [x, y, z]: [Polynomial<RationalField>; 3] = base
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let ring = base.coordinate_ring();
        let branch = ring
            .add(&x.pow(6).unwrap(), &y.pow(6).unwrap())
            .and_then(|sum| ring.add(&sum, &z.pow(6).unwrap()))
            .unwrap();
        let cover = CyclicHypersurfaceCover::<RationalField>::new(base, 2, branch).unwrap();
        assert_eq!(
            cover.weighted_realization().ambient().weights(),
            &[3, 1, 1, 1]
        );
        assert_eq!(cover.weighted_realization().degree(), Integer::from(6));
        assert_eq!(
            CyclicHypersurfaceCover::<RationalField>::from_json(&cover.to_json().unwrap()).unwrap(),
            cover
        );
    }

    #[test]
    fn tame_and_wild_smoothness_do_not_share_a_false_branch_rule() {
        let field = FiniteField::prime(5).unwrap();
        let base = ProjectiveSpace::with_names(field.clone(), 1, ["x", "y"]).unwrap();
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
        let tame = CyclicHypersurfaceCover::<FiniteField>::new(base, 2, branch).unwrap();
        assert_eq!(tame.ramification(), CyclicCoverRamification::Tame);
        assert!(tame
            .smoothness_evidence()
            .unwrap()
            .branch_is_smooth()
            .is_some());

        let base = ProjectiveSpace::with_names(field, 1, ["x", "y"]).unwrap();
        let [x, y]: [Polynomial<FiniteField>; 2] = base
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let ring = base.coordinate_ring();
        let mixed = ring.mul(&x, &y.pow(9).unwrap()).unwrap();
        let branch = ring.add(&x.pow(10).unwrap(), &mixed).unwrap();
        let wild = CyclicHypersurfaceCover::<FiniteField>::new(base, 5, branch).unwrap();
        assert_eq!(wild.ramification(), CyclicCoverRamification::Wild);
        assert!(wild
            .smoothness_evidence()
            .unwrap()
            .branch_is_smooth()
            .is_none());
    }

    #[test]
    fn weighted_jacobian_removes_only_the_cone_vertex() {
        let field = FiniteField::prime(7).unwrap();
        let ambient =
            WeightedProjectiveSpace::with_names(field, vec![2, 1, 1], ["w", "x", "y"]).unwrap();
        let [w, x, y]: [Polynomial<FiniteField>; 3] = ambient
            .coordinate_ring()
            .generators()
            .unwrap()
            .try_into()
            .unwrap();
        let ring = ambient.coordinate_ring();
        let equation = ring
            .sub(&w.pow(2).unwrap(), &x.pow(4).unwrap())
            .and_then(|difference| ring.sub(&difference, &y.pow(4).unwrap()))
            .unwrap();
        let hypersurface = ambient.hypersurface(equation).unwrap();
        let evidence = hypersurface.jacobian_evidence().unwrap();
        assert!(evidence.is_quasismooth());
        assert_eq!(evidence.ambient_quotient_strata().len(), 1);
    }
}
