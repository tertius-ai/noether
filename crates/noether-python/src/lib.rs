// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Private PyO3 transport for the public `noether` Rust facade.

#![deny(unused_crate_dependencies)]
// PyO3 exposes established Python instance-method names such as `to_json` and
// `from_terms`; their receivers intentionally follow Python rather than Rust
// constructor conventions. Exact Python tuple signatures are also useful at
// the binding boundary even when Clippy would prefer local aliases.
#![allow(clippy::wrong_self_convention, clippy::type_complexity)]

pub mod py;

// Canonical serde used by the transport adapters.
pub use serde_json;

use pyo3::prelude::*;

/// Register the exact public mathematical projection and private build helpers.
#[pymodule]
fn _engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", noether::VERSION)?;
    py::errors::register(module)?;
    py::discovery::register(module)?;
    module.add_class::<py::tableau::PySemistandardTableaux>()?;

    module.add_class::<py::root::PyIntegerRing>()?;
    module.add_class::<py::root::PyRationalField>()?;
    module.add_class::<py::root::PyBezoutIdentity>()?;
    module.add_class::<py::coefficient_maps::PyIntegerRationalEmbedding>()?;
    module.add_class::<py::coefficient_maps::PyIntegerReduction>()?;
    module.add_class::<py::coefficient_maps::PyIntegerFiniteFieldMap>()?;
    module.add_class::<py::coefficient_maps::PyPrimeSubfieldEmbedding>()?;
    module.add_class::<py::validity::PyAssumption>()?;
    module.add_class::<py::validity::PyMathematicalValidity>()?;
    module.add_class::<py::ball::PyRealBallField>()?;
    module.add_class::<py::ball::PyRealBall>()?;
    module.add_class::<py::ball::PyComplexBallField>()?;
    module.add_class::<py::ball::PyComplexBall>()?;

    module.add_class::<py::category::PyAtomId>()?;
    module.add_class::<py::category::PySignedBoundary>()?;
    module.add_class::<py::category::PySteinerCell>()?;
    module.add_class::<py::category::PySteinerCanonicalLabel>()?;
    module.add_class::<py::category::PyFiniteOmegaCategory>()?;
    module.add_class::<py::category::PyOrientedBasepoint>()?;
    module.add_class::<py::category::PyStreetSimplex>()?;
    module.add_class::<py::category::PyStreetNerveSkeleton>()?;
    module.add_class::<py::category::PyCategoricalHomologyComponentSlice>()?;
    module.add_class::<py::category::PyCategoricalHomologyPresentation>()?;

    module.add_class::<py::linalg::PyMatrix>()?;
    module.add_class::<py::linalg::PyFrobeniusSemilinearMap>()?;
    module.add_class::<py::linalg::PySparseMatrix>()?;
    module.add_class::<py::linalg::PyVector>()?;
    module.add_class::<py::linalg::PySolutionSpace>()?;
    module.add_class::<py::linalg::PyIntegerSolutionModule>()?;
    module.add_class::<py::linalg::PyIntegerInconsistencyCertificate>()?;
    module.add_class::<py::linalg::PyIntegerLinearSystemResult>()?;
    module.add_class::<py::linalg::PyCongruenceSolutionModule>()?;
    module.add_class::<py::linalg::PyCongruenceInconsistencyCertificate>()?;
    module.add_class::<py::linalg::PyCongruenceSystemResult>()?;
    module.add_class::<py::linalg::PyPLUQFactorization>()?;
    module.add_class::<py::linalg::PyPadicSmithInvariants>()?;
    module.add_class::<py::linalg::PySmithNormalForm>()?;
    module.add_class::<py::linalg::PySmithNormalFormWithTransforms>()?;
    module.add_class::<py::linalg::PySmithVerification>()?;
    module.add_class::<py::linalg::PyIntegralLineConfiguration>()?;
    module.add_class::<py::linalg::PyIntegralLineEquivalence>()?;
    module.add_class::<py::linalg::PyIntegralLineAutomorphismGroup>()?;
    module.add_class::<py::linalg::PyIntegralLineOrbitAction>()?;
    module.add_class::<py::linalg::PyIntegralLineSubgroup>()?;
    module.add_class::<py::linalg::PyMinimumCompletion>()?;
    module.add_class::<py::linalg::PyMinimumCompletionCertificate>()?;
    module.add_class::<py::linalg::PyMinimumNonrealizabilityCertificate>()?;
    module.add_class::<py::linalg::PyMinimumRealization>()?;
    module.add_class::<py::linalg::PyMinimumRealizationDiagnostic>()?;
    module.add_class::<py::linalg::PyMinimumVectorConfiguration>()?;
    module.add_class::<py::linalg::PyLattice>()?;
    module.add_class::<py::linalg::PyLatticeEnumeration>()?;
    module.add_class::<py::linalg::PyClosestLatticeVectors>()?;
    #[cfg(feature = "distributed")]
    {
        module.add_class::<py::distributed::PyMpiThreadSupport>()?;
        module.add_class::<py::distributed::PyWorld>()?;
        module.add_class::<py::distributed::PyDistributedMatrix>()?;
        module.add_class::<py::distributed::PyDistributedPLUQFactorization>()?;
    }

    module.add_class::<py::homalg::PyChainCancellation>()?;
    module.add_class::<py::homalg::PyChainComplex>()?;
    module.add_class::<py::homalg::PyChainReduction>()?;
    module.add_class::<py::homalg::PyPadicChainCancellation>()?;
    module.add_class::<py::homalg::PyPadicChainReduction>()?;
    module.add_class::<py::homalg::PyHomologyGroup>()?;
    module.add_class::<py::homalg::PyGradedHomology>()?;
    module.add_class::<py::homalg::PyPadicCyclicSummand>()?;
    module.add_class::<py::homalg::PyPadicModuleInvariants>()?;
    module.add_class::<py::homalg::PyRankBounds>()?;
    module.add_class::<py::module_facade::PyFreeModule>()?;
    module.add_class::<py::module_facade::PyFreeResolution>()?;
    module.add_class::<py::module_facade::PyDerivedModule>()?;
    module.add_class::<py::module_facade::PyDerivedModuleHomomorphism>()?;
    module.add_class::<py::module_facade::PyGradedBettiTable>()?;
    module.add_class::<py::module_facade::PyGradedModuleInvariants>()?;
    module.add_class::<py::module_facade::PyModuleElement>()?;
    module.add_class::<py::module_facade::PyModuleHomomorphism>()?;
    module.add_class::<py::module_facade::PyKernelSubmodule>()?;
    module.add_class::<py::module_facade::PyImageSubmodule>()?;
    module.add_class::<py::module_facade::PySubmodule>()?;
    module.add_class::<py::module_facade::PySubmoduleMembership>()?;
    module.add_class::<py::module_facade::PyPresentedModule>()?;
    module.add_class::<py::module_facade::PyModuleAssociatedPrimeSet>()?;
    module.add_class::<py::module_facade::PyModulePrimaryComponent>()?;
    module.add_class::<py::module_facade::PyModulePrimaryDecomposition>()?;
    module.add_class::<py::module_facade::PyPresentedModuleHomomorphism>()?;
    module.add_class::<py::module_facade::PyPresentedModuleHomology>()?;
    module.add_class::<py::module_facade::PyResolutionCompleteness>()?;
    module.add_class::<py::module_facade::PyResolutionMinimality>()?;
    module.add_class::<py::module_facade::PyModuleMonomialOrder>()?;
    module.add_class::<py::module_facade::PyModuleMonomial>()?;
    module.add_class::<py::module_facade::PyModuleGroebnerBasis>()?;
    module.add_class::<py::module_facade::PyModuleReduction>()?;
    module.add_class::<py::module_facade::PyStrongModuleTerm>()?;
    module.add_class::<py::module_facade::PyStrongModuleGroebnerBasis>()?;
    module.add_class::<py::module_facade::PyStrongModuleReduction>()?;
    module.add_class::<py::module_power::PyModulePowerKind>()?;
    module.add_class::<py::module_power::PyFreeModulePower>()?;
    module.add_class::<py::module_power::PyPresentedModulePower>()?;
    module.add_class::<py::module_power::PyExteriorAlgebra>()?;
    module.add_class::<py::module_power::PyExteriorElement>()?;
    module.add_class::<py::module_power::PyModulePowerHomomorphism>()?;
    module.add_class::<py::module_power::PyPresentedModulePowerHomomorphism>()?;
    module.add_class::<py::construction::PySymmetricAlgebra>()?;
    module.add_class::<py::construction::PyReesAlgebra>()?;
    module.add_class::<py::construction::PyAssociatedGradedAlgebra>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologyTruncation>()?;
    module.add_class::<py::homalg::PyFiniteGroupIntegralCohomologyTruncation>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologyAlgebraGenerator>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologyAlgebraTruncation>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologyElementaryAbelianRestriction>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologySymondsParameterCertificate>()?;
    module.add_class::<py::homalg::PyFiniteGroupCohomologyCompletePresentation>()?;
    module.add_class::<py::homalg::PyFiniteAbelianGroupCohomologyPresentation>()?;
    module.add_class::<py::homalg::PyFiniteAbelianGroupIntegralCohomologyPresentation>()?;
    module.add_class::<py::homalg::PyFiniteCyclicGroupIntegralCohomologyPresentation>()?;
    module.add_class::<py::topology_facade::PySimplicialComplex>()?;
    module.add_class::<py::topology_facade::PySimplicialCohomologyBasisElement>()?;
    module.add_class::<py::topology_facade::PySimplicialCohomologyProductTerm>()?;
    module.add_class::<py::topology_facade::PySimplicialCohomologyProduct>()?;
    module.add_class::<py::topology_facade::PySimplicialCohomologyRing>()?;
    module.add_class::<py::topology_facade::PyAdmissibleCohomologyOperation>()?;
    module.add_class::<py::topology_facade::PyElementaryCollapse>()?;
    module.add_class::<py::topology_facade::PySimplicialCollapseSequence>()?;
    module.add_class::<py::topology_facade::PySimplicialSimpleHomotopyMove>()?;
    module.add_class::<py::topology_facade::PySimplicialSimpleHomotopyZigzag>()?;
    module.add_class::<py::topology_facade::PyDominatedVertex>()?;
    module.add_class::<py::topology_facade::PyStrongCoreReduction>()?;
    module.add_class::<py::topology_facade::PySimplicialMorseReduction>()?;
    module.add_class::<py::topology_facade::PySimplicialSet>()?;
    module.add_class::<py::topology_facade::PySimplicialSetCohomologyRing>()?;
    module.add_class::<py::topology_facade::PySimplicialSetElementaryHornCollapse>()?;
    module.add_class::<py::topology_facade::PySimplicialSetHornCollapseSequence>()?;
    module.add_class::<py::topology_facade::PySimplicialSetAnodyneMove>()?;
    module.add_class::<py::topology_facade::PySimplicialSetAnodyneZigzag>()?;
    module.add_class::<py::topology_facade::PyFilteredComplex>()?;
    module.add_class::<py::topology_facade::PyBarcode>()?;
    module.add_class::<py::topology_facade::PyEilenbergMacLaneIntegralCohomologyGenerator>()?;
    module.add_class::<py::topology_facade::PyEilenbergMacLaneIntegralCohomologyRelationTerm>()?;
    module.add_class::<py::topology_facade::PyEilenbergMacLaneIntegralCohomologyRelation>()?;
    module.add_class::<py::topology_facade::PyEilenbergMacLaneCohomologyGenerator>()?;
    module.add_class::<py::topology_facade::PyEilenbergMacLaneCohomologyOperation>()?;

    module.add_class::<py::geometry_facade::PyPointCloud>()?;
    module.add_class::<py::geometry_facade::PyTriangulation>()?;
    module.add_class::<py::convex_hull::PyPolytope>()?;
    module.add_function(wrap_pyfunction!(
        py::geometry_facade::facade_delaunay,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py::geometry_facade::facade_regular_triangulation,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py::convex_hull::facade_convex_hull,
        module
    )?)?;

    module.add_class::<py::commalg::PyResidueRing>()?;
    module.add_class::<py::commalg::PyResidueClass>()?;
    module.add_class::<py::commalg::PyCrtDecomposition>()?;
    module.add_class::<py::polynomial_facade::PyMonomialMap>()?;
    module.add_class::<py::commalg::PyGradedGenerator>()?;
    module.add_class::<py::commalg::PyGradedMonomial>()?;
    module.add_class::<py::commalg::PyGradedCommutativePolynomialRing>()?;
    module.add_class::<py::commalg::PyGradedCommutativePolynomial>()?;
    module.add_class::<py::commalg::PyGradedCommutativePresentation>()?;
    module.add_class::<py::intersection_theory::PyCompleteIntersectionHypotheses>()?;
    module.add_class::<py::intersection_theory::PyChowRing>()?;
    module.add_class::<py::intersection_theory::PyChowClass>()?;
    module.add_class::<py::intersection_theory::PyRationalChowClass>()?;
    module.add_class::<py::intersection_theory::PyProjectiveCompleteIntersection>()?;
    module.add_class::<py::intersection_theory::PyHirzebruchRiemannRochResult>()?;
    module.add_class::<py::polynomial_facade::PyMonomialOrder>()?;
    module.add_class::<py::local::PyLocalMonomialOrder>()?;
    module.add_class::<py::local::PyLocalPolynomialRing>()?;
    module.add_class::<py::local::PyLocalPolynomial>()?;
    module.add_class::<py::local::PyLocalCompletionMap>()?;
    module.add_class::<py::local::PyLocalIdeal>()?;
    module.add_class::<py::local::PyLocalStandardBasis>()?;
    module.add_class::<py::local::PyLocalReduction>()?;
    module.add_class::<py::local_module::PyLocalModuleOrder>()?;
    module.add_class::<py::local_module::PyLocalModuleStandardBasis>()?;
    module.add_class::<py::local_module::PyLocalModuleReduction>()?;
    module.add_class::<py::prime_localization::PyPolynomialPrimeLocalization>()?;
    module.add_class::<py::prime_localization::PyPolynomialPrimeLocalElement>()?;
    module.add_class::<py::prime_localization::PyPolynomialPrimeLocalIdeal>()?;
    module.add_class::<py::prime_localization::PyPolynomialPrimeLocalMembership>()?;
    module.add_class::<py::polynomial_facade::PyPolynomialFactorization>()?;
    module.add_class::<py::polynomial_facade::PyRealRootIsolation>()?;
    module.add_class::<py::polynomial_facade::PyPolynomialRing>()?;
    module.add_class::<py::polynomial_facade::PyPolynomial>()?;
    module.add_class::<py::quotient::PyQuotientRing>()?;
    module.add_class::<py::quotient::PyQuotientElement>()?;
    module.add_class::<py::quotient::PyQuotientRingMap>()?;
    module.add_class::<py::differentials::PyKahlerDifferentials>()?;
    module.add_class::<py::differentials::PyKahlerDifferential>()?;
    module.add_class::<py::differentials::PyConormalSequenceEvidence>()?;
    module.add_class::<py::differentials::PyCotangentSpace>()?;
    module.add_class::<py::differentials::PyCotangentComplexPresentation>()?;
    module.add_class::<py::differentials::PyCotangentComplexAmplitude>()?;
    module.add_class::<py::differentials::PyCotangentComplexHypothesis>()?;
    module.add_class::<py::monomial_ideal::PyMonomialIdeal>()?;
    module.add_class::<py::polynomial_facade::PyStrongGroebnerBasis>()?;
    module.add_class::<py::series::PyDegree>()?;
    module.add_class::<py::series::PyFreeAbelianGrading>()?;
    module.add_class::<py::series::PyHilbertSeries>()?;
    module.add_class::<py::series::PyPositiveGrading>()?;
    module.add_class::<py::rational_function::PyRationalFunctionField>()?;
    module.add_class::<py::rational_function::PyRationalFunction>()?;
    module.add_class::<py::series::PySeriesTruncation>()?;
    module.add_class::<py::series::PyWeightedGrading>()?;
    module.add_class::<py::series::PyPowerSeriesRing>()?;
    module.add_class::<py::series::PyPowerSeries>()?;
    module.add_class::<py::curve::PyCurve>()?;
    module.add_class::<py::curve::PyCurveMorphism>()?;
    module.add_class::<py::curve::PyCurvePoint>()?;
    module.add_class::<py::curve::PyCurveClosedPoint>()?;
    module.add_class::<py::curve::PyWeilDivisor>()?;
    module.add_class::<py::curve::PyCurveFunctionField>()?;
    module.add_class::<py::curve::PyCurveFunction>()?;
    module.add_class::<py::curve::PyCurveFunctionOrder>()?;
    module.add_class::<py::curve::PyCurveFunctionEvaluation>()?;
    module.add_class::<py::curve::PyRiemannRochSpace>()?;

    module.add_class::<py::elliptic_curve::PyEllipticCurve>()?;
    module.add_class::<py::elliptic_curve::PyEllipticCurvePoint>()?;
    module.add_class::<py::elliptic_curve::PyMillerFunction>()?;
    module.add_class::<py::elliptic_curve::PyNthPowerClassGroup>()?;
    module.add_class::<py::elliptic_curve::PyNthPowerClass>()?;
    module.add_class::<py::numthy::PyFiniteField>()?;
    module.add_class::<py::numthy::PyFiniteFieldElement>()?;
    module.add_class::<py::numthy::PyFiniteFieldIsomorphism>()?;
    module.add_class::<py::numthy::PyIntegerFactorization>()?;
    module.add_class::<py::numthy::PyPartialIntegerFactorization>()?;
    module.add_class::<py::numthy::PyPadicRing>()?;
    module.add_class::<py::numthy::PyPadicField>()?;
    module.add_class::<py::numthy::PyPadicRingFrobenius>()?;
    module.add_class::<py::numthy::PyPadicFieldFrobenius>()?;
    module.add_class::<py::numthy::PyPadicRingRestriction>()?;
    module.add_class::<py::numthy::PyPadicFieldRestriction>()?;
    module.add_class::<py::numthy::PyPadicIntegralEmbedding>()?;
    module.add_class::<py::numthy::PyPadicElement>()?;
    module.add_class::<py::numthy::PyPadicValuation>()?;
    module.add_class::<py::numthy::PyQadicRing>()?;
    module.add_class::<py::numthy::PyQadicField>()?;
    module.add_class::<py::numthy::PyQadicRingFrobenius>()?;
    module.add_class::<py::numthy::PyQadicFieldFrobenius>()?;
    module.add_class::<py::numthy::PyQadicRingRestriction>()?;
    module.add_class::<py::numthy::PyQadicFieldRestriction>()?;
    module.add_class::<py::numthy::PyQadicIntegralEmbedding>()?;
    module.add_class::<py::numthy::PyQadicElement>()?;
    module.add_class::<py::numthy::PyPadicLocalPrecision>()?;
    module.add_class::<py::numthy::PyPadicLocalField>()?;
    module.add_class::<py::numthy::PyPadicLocalFieldElement>()?;
    module.add_class::<py::numthy::PyPadicLocalFieldOrder>()?;
    module.add_class::<py::numthy::PyPadicLocalFractionalIdeal>()?;
    module.add_class::<py::numthy::PyPadicLocalFieldRestriction>()?;
    module.add_class::<py::numthy::PyPadicLocalFieldCoefficientFrobenius>()?;
    module.add_class::<py::number_field::PyNumberField>()?;
    module.add_class::<py::number_field::PyNumberFieldElement>()?;
    module.add_class::<py::number_field::PyNumberFieldOrder>()?;
    module.add_class::<py::number_field::PyFractionalIdeal>()?;
    module.add_class::<py::number_field::PyIdeal>()?;
    module.add_class::<py::ideal_evidence::PyIdealReduction>()?;
    module.add_class::<py::ideal_evidence::PyIdealElimination>()?;
    module.add_class::<py::ideal_evidence::PyArithmeticNullstellensatzCertificate>()?;
    module.add_class::<py::ideal_evidence::PyArithmeticFiberSupport>()?;
    module.add_class::<py::number_field::PyRationalUnivariateRepresentation>()?;
    module.add_class::<py::number_field::PyFiniteQuotientAlgebra>()?;
    module.add_class::<py::number_field::PyRealSolutionBox>()?;
    module.add_class::<py::number_field::PyIdealDecomposition>()?;
    module.add_class::<py::number_field::PyRadicalIdeal>()?;
    module.add_class::<py::number_field::PyAssociatedPrimeExtStage>()?;
    module.add_class::<py::number_field::PyAssociatedPrimeSet>()?;
    module.add_class::<py::number_field::PyPolynomialPrimeCertificate>()?;
    module.add_class::<py::number_field::PyPolynomialPrimaryCertificate>()?;
    module.add_class::<py::number_field::PyMinimalPrimeDecomposition>()?;
    module.add_class::<py::number_field::PyPrimaryDecomposition>()?;
    module.add_class::<py::number_field::PyEquidimensionalHull>()?;
    module.add_class::<py::number_field::PyAffineNormalization>()?;
    module.add_class::<py::number_field::PyGrauertRemmertStep>()?;
    module.add_class::<py::number_field::PyReducedAffineNormalization>()?;
    module.add_class::<py::number_field::PyNoetherNormalization>()?;
    module.add_class::<py::number_field::PyEquidimensionalRadicalComponent>()?;
    module.add_class::<py::number_field::PyEquidimensionalRadicalDecomposition>()?;
    module.add_class::<py::number_field::PyPrimeIdeal>()?;
    module.add_function(wrap_pyfunction!(py::numthy::integer_factor, module)?)?;
    module.add_function(wrap_pyfunction!(
        py::numthy::integer_partial_factor,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py::numthy::integer_is_prime, module)?)?;
    module.add_function(wrap_pyfunction!(
        py::numthy::facade_cyclotomic_polynomial,
        module
    )?)?;

    module.add_class::<py::scheme::PyAffineSpace>()?;
    module.add_class::<py::scheme::PyProjectiveSpace>()?;
    module.add_class::<py::scheme::PyScheme>()?;
    module.add_class::<py::scheme::PySchemePoint>()?;
    module.add_class::<py::scheme::PyFiniteSchemeClosedPoint>()?;
    module.add_class::<py::scheme::PyFiniteSchemeClosedPointSet>()?;
    module.add_class::<py::scheme::PyBlowUpChart>()?;
    module.add_class::<py::scheme::PyBlowUp>()?;
    module.add_class::<py::scheme::PySchemeCompletion>()?;
    module.add_class::<py::scheme::PyRationalPointEnumeration>()?;
    module.add_class::<py::scheme::PyPointCountCertificate>()?;
    module.add_class::<py::surface::PyWeightedProjectiveSpace>()?;
    module.add_class::<py::surface::PyWeightedProjectiveStratum>()?;
    module.add_class::<py::surface::PyWeightedProjectiveChart>()?;
    module.add_class::<py::surface::PyWeightedProjectiveScheme>()?;
    module.add_class::<py::surface::PyWeightedProjectiveSchemeChart>()?;
    module.add_class::<py::surface::PyWeightedProjectiveHypersurface>()?;
    module.add_class::<py::surface::PyWeightedHypersurfaceJacobianEvidence>()?;
    module.add_class::<py::surface::PyCyclicHypersurfaceCover>()?;
    module.add_class::<py::surface::PyCyclicCoverSmoothnessEvidence>()?;
    module.add_class::<py::surface::PyPointCountMethod>()?;
    module.add_class::<py::surface::PyPointCountEvidence>()?;
    module.add_class::<py::surface::PyExtensionPointCount>()?;
    module.add_class::<py::surface::PyPointCountSequence>()?;
    module.add_class::<py::surface::PySmoothProjectiveSurfaceCohomology>()?;
    module.add_class::<py::surface::PyFrobeniusPolynomial>()?;
    module.add_class::<py::surface::PyZetaFunction>()?;
    module.add_class::<py::surface::PyTateCyclotomicFactor>()?;
    module.add_class::<py::surface::PyGeometricDivisorRankUpperBound>()?;
    module.add_class::<py::surface::PySplitCyclicPullback>()?;
    module.add_class::<py::surface::PyDivisorRankLowerBound>()?;
    module.add_class::<py::surface::PyGeometricDivisorRank>()?;

    module.add_class::<py::groups::PyPermutation>()?;
    module.add_class::<py::groups::PyPermGroup>()?;
    module.add_class::<py::groups::PyConjugacyClass>()?;
    module.add_class::<py::groups::PySubsetOrbit>()?;
    module.add_class::<py::groups::PyPairOrbit>()?;
    module.add_class::<py::groups::PyDoubleCoset>()?;
    module.add_class::<py::groups::PyTransversal>()?;
    module.add_class::<py::groups::PySchreierTree>()?;
    module.add_class::<py::groups::PyStabilizerChain>()?;
    module.add_class::<py::groups::PyStabilizerChainStep>()?;
    module.add_class::<py::groups::PySchreierGenerator>()?;
    module.add_class::<py::groups::PyOrbitStabilizer>()?;
    module.add_class::<py::groups::PyRandomElement>()?;
    module.add_class::<py::groups::PyElementIter>()?;

    module.add_class::<py::graph::PyTraversal>()?;
    module.add_class::<py::graph::PyShortestPath>()?;
    module.add_class::<py::graph::PyComponents>()?;
    module.add_class::<py::graph::PyLabeling>()?;
    module.add_class::<py::graph::PyCanonicalLabel>()?;
    module.add_class::<py::poset::PyBeatPoint>()?;
    module.add_class::<py::poset::PyPosetCanonicalLabel>()?;
    module.add_class::<py::poset::PyCoreReduction>()?;
    // These are private optional-backend probes and adapters, not mathematical
    // identities. Public operations reach them only through Rust dispatch.
    module.add_function(wrap_pyfunction!(
        py::ca_bridges::engine_msolve_available,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py::ca_bridges::engine_m2_available,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        py::ca_bridges::engine_singular_available,
        module
    )?)?;
    // Canonical combinatorial values use the small serde/schema registry.
    py::types_registry::register_all_types(module)?;
    py::types_registry::register_functions(module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_coefficient_maps_are_registered_on_the_extension_module() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_engine")?;
            _engine(&module)?;

            for name in [
                "IntegerRationalEmbedding",
                "IntegerReduction",
                "IntegerFiniteFieldMap",
                "PrimeSubfieldEmbedding",
            ] {
                assert!(module.hasattr(name)?, "missing extension class {name}");
            }
            Ok(())
        })
    }
}
