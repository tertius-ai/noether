# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Commutative rings and their algebraic constructions."""

from noether._engine_gateway import _import_engine

_engine = _import_engine()
ArithmeticFiberSupport = _engine.ArithmeticFiberSupport
ArithmeticNullstellensatzCertificate = _engine.ArithmeticNullstellensatzCertificate
AffineNormalization = _engine.AffineNormalization
AssociatedGradedAlgebra = _engine.AssociatedGradedAlgebra
AssociatedPrimeExtStage = _engine.AssociatedPrimeExtStage
AssociatedPrimeSet = _engine.AssociatedPrimeSet
CrtDecomposition = _engine.CrtDecomposition
ConormalSequenceEvidence = _engine.ConormalSequenceEvidence
CotangentComplexAmplitude = _engine.CotangentComplexAmplitude
CotangentComplexHypothesis = _engine.CotangentComplexHypothesis
CotangentComplexPresentation = _engine.CotangentComplexPresentation
CotangentSpace = _engine.CotangentSpace
CyclicCoverSmoothnessEvidence = _engine.CyclicCoverSmoothnessEvidence
CyclicHypersurfaceCover = _engine.CyclicHypersurfaceCover
Curve = _engine.Curve
CurveClosedPoint = _engine.CurveClosedPoint
CurveFunction = _engine.CurveFunction
CurveFunctionEvaluation = _engine.CurveFunctionEvaluation
CurveFunctionField = _engine.CurveFunctionField
CurveFunctionOrder = _engine.CurveFunctionOrder
CurveMorphism = _engine.CurveMorphism
CurvePoint = _engine.CurvePoint
DivisorRankLowerBound = _engine.DivisorRankLowerBound
Degree = _engine.Degree
ExtensionPointCount = _engine.ExtensionPointCount
EquidimensionalHull = _engine.EquidimensionalHull
EquidimensionalRadicalComponent = _engine.EquidimensionalRadicalComponent
EquidimensionalRadicalDecomposition = _engine.EquidimensionalRadicalDecomposition
FrobeniusPolynomial = _engine.FrobeniusPolynomial
HyperellipticCurve = _engine.HyperellipticCurve
HyperellipticFrobeniusEvidence = _engine.HyperellipticFrobeniusEvidence
FiniteQuotientAlgebra = _engine.FiniteQuotientAlgebra
FreeAbelianGrading = _engine.FreeAbelianGrading
GeometricDivisorRank = _engine.GeometricDivisorRank
GeometricDivisorRankUpperBound = _engine.GeometricDivisorRankUpperBound
GrauertRemmertStep = _engine.GrauertRemmertStep
HilbertSeries = _engine.HilbertSeries
HirzebruchRiemannRochResult = _engine.HirzebruchRiemannRochResult
KahlerDifferential = _engine.KahlerDifferential
KahlerDifferentials = _engine.KahlerDifferentials
GradedCommutativePolynomial = _engine.GradedCommutativePolynomial
GradedCommutativePolynomialRing = _engine.GradedCommutativePolynomialRing
GradedCommutativePresentation = _engine.GradedCommutativePresentation
GradedGenerator = _engine.GradedGenerator
GradedMonomial = _engine.GradedMonomial
AffineSpace = _engine.AffineSpace
BlowUp = _engine.BlowUp
BlowUpChart = _engine.BlowUpChart
ChowClass = _engine.ChowClass
ChowRing = _engine.ChowRing
CompleteIntersectionHypotheses = _engine.CompleteIntersectionHypotheses
Ideal = _engine.Ideal
IdealElimination = _engine.IdealElimination
IdealDecomposition = _engine.IdealDecomposition
IdealReduction = _engine.IdealReduction
PrimeIdeal = _engine.PrimeIdeal
PowerSeries = _engine.PowerSeries
PowerSeriesRing = _engine.PowerSeriesRing
MonomialIdeal = _engine.MonomialIdeal
MonomialMap = _engine.MonomialMap
MonomialOrder = _engine.MonomialOrder
LocalMonomialOrder = _engine.LocalMonomialOrder
LocalCompletionMap = _engine.LocalCompletionMap
LocalIdeal = _engine.LocalIdeal
LocalPolynomial = _engine.LocalPolynomial
LocalPolynomialRing = _engine.LocalPolynomialRing
LocalReduction = _engine.LocalReduction
LocalStandardBasis = _engine.LocalStandardBasis
NoetherNormalization = _engine.NoetherNormalization
MinimalPrimeDecomposition = _engine.MinimalPrimeDecomposition
Polynomial = _engine.Polynomial
PolynomialPrimeLocalElement = _engine.PolynomialPrimeLocalElement
PolynomialPrimeLocalIdeal = _engine.PolynomialPrimeLocalIdeal
PolynomialPrimeLocalMembership = _engine.PolynomialPrimeLocalMembership
PolynomialPrimeLocalization = _engine.PolynomialPrimeLocalization
PolynomialFactorization = _engine.PolynomialFactorization
PolynomialHomomorphism = _engine.PolynomialHomomorphism
PolynomialPrimaryCertificate = _engine.PolynomialPrimaryCertificate
PolynomialPrimeCertificate = _engine.PolynomialPrimeCertificate
PolynomialRing = _engine.PolynomialRing
PrimaryDecomposition = _engine.PrimaryDecomposition
QuotientElement = _engine.QuotientElement
QuotientRing = _engine.QuotientRing
QuotientRingMap = _engine.QuotientRingMap
RadicalIdeal = _engine.RadicalIdeal
PositiveGrading = _engine.PositiveGrading
PointCountEvidence = _engine.PointCountEvidence
PointCountMethod = _engine.PointCountMethod
PointCountSequence = _engine.PointCountSequence
ProjectiveSpace = _engine.ProjectiveSpace
ProjectiveCompleteIntersection = _engine.ProjectiveCompleteIntersection
RationalPointEnumeration = _engine.RationalPointEnumeration
RationalChowClass = _engine.RationalChowClass
RationalFunction = _engine.RationalFunction
RationalFunctionField = _engine.RationalFunctionField
RationalFunctionFieldMap = _engine.RationalFunctionFieldMap
RationalUnivariateRepresentation = _engine.RationalUnivariateRepresentation
ReesAlgebra = _engine.ReesAlgebra
ReducedAffineNormalization = _engine.ReducedAffineNormalization
RealRootIsolation = _engine.RealRootIsolation
RealSolutionBox = _engine.RealSolutionBox
ResidueClass = _engine.ResidueClass
ResidueRing = _engine.ResidueRing
SeriesTruncation = _engine.SeriesTruncation
StrongGroebnerBasis = _engine.StrongGroebnerBasis
SymmetricAlgebra = _engine.SymmetricAlgebra
RiemannRochSpace = _engine.RiemannRochSpace
Scheme = _engine.Scheme
SchemeCompletion = _engine.SchemeCompletion
SchemePoint = _engine.SchemePoint
SmoothProjectiveSurfaceCohomology = _engine.SmoothProjectiveSurfaceCohomology
SplitCyclicPullback = _engine.SplitCyclicPullback
TateCyclotomicFactor = _engine.TateCyclotomicFactor
FiniteSchemeClosedPoint = _engine.FiniteSchemeClosedPoint
FiniteSchemeClosedPointSet = _engine.FiniteSchemeClosedPointSet
WeightedGrading = _engine.WeightedGrading
WeightedHypersurfaceJacobianEvidence = _engine.WeightedHypersurfaceJacobianEvidence
WeightedProjectiveChart = _engine.WeightedProjectiveChart
WeightedProjectiveHypersurface = _engine.WeightedProjectiveHypersurface
WeightedProjectiveScheme = _engine.WeightedProjectiveScheme
WeightedProjectiveSchemeChart = _engine.WeightedProjectiveSchemeChart
WeightedProjectiveSpace = _engine.WeightedProjectiveSpace
WeightedProjectiveStratum = _engine.WeightedProjectiveStratum
WeilDivisor = _engine.WeilDivisor
ZetaFunction = _engine.ZetaFunction

__all__ = [
    "ArithmeticFiberSupport",
    "ArithmeticNullstellensatzCertificate",
    "AffineNormalization",
    "AssociatedGradedAlgebra",
    "AssociatedPrimeExtStage",
    "AssociatedPrimeSet",
    "AffineSpace",
    "BlowUp",
    "BlowUpChart",
    "ChowClass",
    "ChowRing",
    "CompleteIntersectionHypotheses",
    "CrtDecomposition",
    "ConormalSequenceEvidence",
    "CotangentComplexAmplitude",
    "CotangentComplexHypothesis",
    "CotangentComplexPresentation",
    "CotangentSpace",
    "CyclicCoverSmoothnessEvidence",
    "CyclicHypersurfaceCover",
    "Curve",
    "CurveClosedPoint",
    "CurveFunction",
    "CurveFunctionEvaluation",
    "CurveFunctionField",
    "CurveFunctionOrder",
    "CurveMorphism",
    "CurvePoint",
    "DivisorRankLowerBound",
    "Degree",
    "EquidimensionalHull",
    "EquidimensionalRadicalComponent",
    "EquidimensionalRadicalDecomposition",
    "ExtensionPointCount",
    "FrobeniusPolynomial",
    "HyperellipticCurve",
    "HyperellipticFrobeniusEvidence",
    "FiniteQuotientAlgebra",
    "FreeAbelianGrading",
    "GeometricDivisorRank",
    "GeometricDivisorRankUpperBound",
    "GrauertRemmertStep",
    "HilbertSeries",
    "HirzebruchRiemannRochResult",
    "KahlerDifferential",
    "KahlerDifferentials",
    "GradedCommutativePolynomial",
    "GradedCommutativePolynomialRing",
    "GradedCommutativePresentation",
    "GradedGenerator",
    "GradedMonomial",
    "Ideal",
    "IdealElimination",
    "IdealDecomposition",
    "IdealReduction",
    "PrimeIdeal",
    "PowerSeries",
    "PowerSeriesRing",
    "MonomialIdeal",
    "MonomialMap",
    "MonomialOrder",
    "LocalMonomialOrder",
    "LocalCompletionMap",
    "LocalIdeal",
    "LocalPolynomial",
    "LocalPolynomialRing",
    "LocalReduction",
    "LocalStandardBasis",
    "NoetherNormalization",
    "MinimalPrimeDecomposition",
    "Polynomial",
    "PolynomialPrimeLocalElement",
    "PolynomialPrimeLocalIdeal",
    "PolynomialPrimeLocalMembership",
    "PolynomialPrimeLocalization",
    "PolynomialFactorization",
    "PolynomialHomomorphism",
    "PolynomialPrimaryCertificate",
    "PolynomialPrimeCertificate",
    "PolynomialRing",
    "PrimaryDecomposition",
    "QuotientElement",
    "QuotientRing",
    "QuotientRingMap",
    "RadicalIdeal",
    "PositiveGrading",
    "PointCountEvidence",
    "PointCountMethod",
    "PointCountSequence",
    "ProjectiveSpace",
    "ProjectiveCompleteIntersection",
    "RationalPointEnumeration",
    "RationalChowClass",
    "RationalFunction",
    "RationalFunctionField",
    "RationalFunctionFieldMap",
    "RationalUnivariateRepresentation",
    "ReesAlgebra",
    "ReducedAffineNormalization",
    "RiemannRochSpace",
    "RealRootIsolation",
    "RealSolutionBox",
    "ResidueClass",
    "ResidueRing",
    "SeriesTruncation",
    "StrongGroebnerBasis",
    "SymmetricAlgebra",
    "Scheme",
    "SchemeCompletion",
    "SchemePoint",
    "SmoothProjectiveSurfaceCohomology",
    "SplitCyclicPullback",
    "TateCyclotomicFactor",
    "FiniteSchemeClosedPoint",
    "FiniteSchemeClosedPointSet",
    "WeightedGrading",
    "WeightedHypersurfaceJacobianEvidence",
    "WeightedProjectiveChart",
    "WeightedProjectiveHypersurface",
    "WeightedProjectiveScheme",
    "WeightedProjectiveSchemeChart",
    "WeightedProjectiveSpace",
    "WeightedProjectiveStratum",
    "WeilDivisor",
    "ZetaFunction",
]
