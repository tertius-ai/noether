// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Polynomial rings, ideals, modules, schemes, and morphisms.

mod affine_normalization;
mod arithmetic_ideal;
mod associated_primes;
mod base_change;
mod construction;
mod curve;
mod decomposition;
mod differentials;
mod equidimensional;
mod equidimensional_hull;
#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
mod external_groebner;
mod factorization;
mod frobenius;
mod geometry;
mod graded_commutative;
mod grading;
mod groebner;
mod hilbert;
mod hilbert_series;
mod homomorphism;
mod hyperelliptic;
mod ideal;
mod intersection_theory;
mod jacobian;
mod kedlaya;
mod local;
mod minimal_primes;
mod monomial;
mod monomial_ideal;
mod normalization;
mod padic_point_count;
mod perfect_power;
mod point_counts;
mod polynomial;
mod polynomial_flint;
mod primary;
mod prime_local_ideal;
mod prime_localization;
mod quotient;
mod radical;
mod rational_function;
mod real_roots;
mod real_solutions;
mod reduced_normalization;
mod residue;
pub(crate) mod scheme;
mod series;
mod weighted_projective;

pub use affine_normalization::{
    AffineNormalization, AffineNormalizationError, AffineNormalizationMap, GrauertRemmertStep,
    GrauertRemmertStepMap,
};
pub use arithmetic_ideal::{
    ArithmeticFiberSupport, ArithmeticFiberSupportClaim, ArithmeticIdealError,
    ArithmeticNullstellensatzCertificate,
};
pub(crate) use associated_primes::{
    quotient_codimension, sorted_associated_prime_certificates, stage_from_ext,
};
pub use associated_primes::{AssociatedPrimeError, AssociatedPrimeExtStage, AssociatedPrimeSet};
pub use base_change::BaseChangeError;
pub use construction::{
    AssociatedGradedAlgebra, ConstructionAlgebraError, ReesAlgebra, SymmetricAlgebra,
};
pub use curve::{
    Curve, CurveClosedPoint, CurveError, CurveFunction, CurveFunctionEvaluation,
    CurveFunctionField, CurveFunctionOrder, CurveMorphism, CurvePoint, CurvePresentationKind,
    CurveSmoothnessEvidence, RiemannRochSpace, WeilDivisor,
};
pub use decomposition::{DecompositionError, IdealDecomposition};
pub use differentials::{
    ConormalSequenceEvidence, CotangentComplexAmplitude, CotangentComplexHypothesis,
    CotangentComplexPresentation, CotangentSpace, KahlerDifferential, KahlerDifferentialError,
    KahlerDifferentials,
};
pub use equidimensional::{EquidimensionalRadicalComponent, EquidimensionalRadicalDecomposition};
pub use equidimensional_hull::EquidimensionalHull;
pub use factorization::{PolynomialFactorization, PolynomialFactorizationError};
pub use frobenius::{
    DivisorRankLowerBound, FrobeniusError, FrobeniusPolynomial, GeometricDivisorRank,
    GeometricDivisorRankUpperBound, SmoothProjectiveSurfaceCohomology, SplitCyclicPullback,
    TateCyclotomicFactor, ZetaFunction,
};
pub use geometry::{BlowUp, BlowUpChart, GeometryError, SchemeCompletion};
pub use graded_commutative::{
    GradedCommutativeError, GradedCommutativePolynomial, GradedCommutativePolynomialRing,
    GradedCommutativePresentation, GradedGenerator, GradedMonomial,
};
pub use grading::{Degree, FreeAbelianGrading, GradingError, PositiveGrading, WeightedGrading};
pub(crate) use groebner::{
    exponent_to_kernel, from_integer_kernel_polynomial, from_kernel_basis,
    from_prime_kernel_polynomial, from_rational_kernel_terms, kernel_groebner_basis_with_transform,
    to_kernel_order, to_kernel_polynomial, to_prime_kernel_polynomial,
    to_rational_kernel_polynomial, NoetherFieldLane, RationalKernelLane,
};
pub use groebner::{
    FiniteQuotientAlgebra, GroebnerError, IdealElimination, IdealReduction,
    RationalUnivariateRepresentation, StrongGroebnerBasis,
};
pub use hilbert_series::{HilbertSeries, HilbertSeriesError};
pub use homomorphism::{
    PolynomialConstantEmbedding, PolynomialHomomorphism, PolynomialHomomorphismError,
    PolynomialRingHomomorphism,
};
pub use hyperelliptic::{HyperellipticCurve, HyperellipticError, HyperellipticFrobeniusEvidence};
pub use ideal::{Ideal, IdealError, PrimeIdeal};
pub use intersection_theory::{
    ChowClass, ChowRing, CompleteIntersectionHypotheses, HirzebruchRiemannRochResult,
    IntersectionTheoryError, ProjectiveCompleteIntersection, RationalChowClass,
};
pub use local::{
    LocalAlgebraError, LocalCompletionMap, LocalIdeal, LocalMonomialOrder, LocalPolynomial,
    LocalPolynomialRing, LocalReduction, LocalStandardBasis,
};
pub(crate) use minimal_primes::MinimalPrimeCoefficientField;
pub use minimal_primes::{MinimalPrimeDecomposition, PolynomialPrimeCertificate};
pub use monomial::{MonomialMap, MonomialMapError, SemilinearSubstitutionError};
pub use monomial_ideal::{MonomialIdeal, MonomialIdealError};
pub use normalization::{NoetherNormalization, NoetherNormalizationError};
pub use perfect_power::PerfectPowerError;
pub use point_counts::{
    ExtensionPointCount, PointCountError, PointCountEvidence, PointCountMethod, PointCountSequence,
    PointCountTarget,
};
pub use polynomial::{
    MonomialOrder, MonomialOrderError, Polynomial, PolynomialContextError, PolynomialError,
    PolynomialRing,
};
pub use primary::{PolynomialPrimaryCertificate, PrimaryDecomposition};
pub use prime_local_ideal::{PolynomialPrimeLocalIdeal, PolynomialPrimeLocalMembership};
pub use prime_localization::{
    PolynomialPrimeLocalElement, PolynomialPrimeLocalization, PrimeLocalizationError,
};
pub use quotient::{
    QuotientElement, QuotientError, QuotientRing, QuotientRingMap, QuotientRingMapError,
};
pub use radical::RadicalIdeal;
pub use rational_function::{
    RationalFunction, RationalFunctionError, RationalFunctionField, RationalFunctionFieldMap,
    RationalFunctionFieldMapError,
};
pub use real_roots::{RealRootError, RealRootIsolation};
pub use real_solutions::{RealSolutionBox, RealSolutionError};
pub use reduced_normalization::{ReducedAffineNormalization, ReducedNormalizationError};
pub use residue::{
    CrtDecomposition, CrtError, InvalidModulusError, NonUnitError, ResidueClass, ResidueRing,
};
pub use scheme::{
    AffineSpace, FiniteSchemeClosedPoint, FiniteSchemeClosedPointSet, ProjectiveSpace,
    RationalPointEnumeration, Scheme, SchemeError, SchemePoint,
};
pub use series::{
    PowerSeries, PowerSeriesError, PowerSeriesRing, SeriesContextError, SeriesTruncation,
    SeriesTruncationError, SeriesTruncationMode,
};
pub use weighted_projective::{
    CyclicCoverRamification, CyclicCoverSmoothnessEvidence, CyclicHypersurfaceCover,
    WeightedHypersurfaceJacobianEvidence, WeightedProjectiveChart, WeightedProjectiveError,
    WeightedProjectiveHypersurface, WeightedProjectiveScheme, WeightedProjectiveSchemeChart,
    WeightedProjectiveSpace, WeightedProjectiveStratum,
};
