# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Chain complexes and checked derived invariants."""

from __future__ import annotations

from typing import Literal, TypeAlias

from noether._engine_gateway import _import_engine

UniversalUnit: TypeAlias = Literal[-1, 1]

ChainCancellation = _import_engine().ChainCancellation
ChainComplex = _import_engine().ChainComplex
ChainReduction = _import_engine().ChainReduction
DerivedModule = _import_engine().DerivedModule
DerivedModuleHomomorphism = _import_engine().DerivedModuleHomomorphism
ExteriorAlgebra = _import_engine().ExteriorAlgebra
ExteriorElement = _import_engine().ExteriorElement
FinitelyGeneratedAbelianGroup = _import_engine().FinitelyGeneratedAbelianGroup
FreeModule = _import_engine().FreeModule
FreeModulePower = _import_engine().FreeModulePower
FreeResolution = _import_engine().FreeResolution
GradedBettiTable = _import_engine().GradedBettiTable
FiniteAbelianGroupCohomologyPresentation = _import_engine().FiniteAbelianGroupCohomologyPresentation
FiniteAbelianGroupIntegralCohomologyPresentation = (
    _import_engine().FiniteAbelianGroupIntegralCohomologyPresentation
)
FiniteCyclicGroupIntegralCohomologyPresentation = (
    _import_engine().FiniteCyclicGroupIntegralCohomologyPresentation
)
FiniteGroupCohomologyAlgebraGenerator = _import_engine().FiniteGroupCohomologyAlgebraGenerator
FiniteGroupCohomologyAlgebraTruncation = _import_engine().FiniteGroupCohomologyAlgebraTruncation
FiniteGroupCohomologyCompletePresentation = (
    _import_engine().FiniteGroupCohomologyCompletePresentation
)
FiniteGroupCohomologyElementaryAbelianRestriction = (
    _import_engine().FiniteGroupCohomologyElementaryAbelianRestriction
)
FiniteGroupCohomologySymondsParameterCertificate = (
    _import_engine().FiniteGroupCohomologySymondsParameterCertificate
)
FiniteGroupCohomologyTruncation = _import_engine().FiniteGroupCohomologyTruncation
FiniteGroupIntegralCohomologyTruncation = _import_engine().FiniteGroupIntegralCohomologyTruncation
GradedHomology = _import_engine().GradedHomology
GradedModuleInvariants = _import_engine().GradedModuleInvariants
HomologyGroup = _import_engine().HomologyGroup
ImageSubmodule = _import_engine().ImageSubmodule
KernelSubmodule = _import_engine().KernelSubmodule
LocalModuleOrder = _import_engine().LocalModuleOrder
LocalModuleReduction = _import_engine().LocalModuleReduction
LocalModuleStandardBasis = _import_engine().LocalModuleStandardBasis
ModuleElement = _import_engine().ModuleElement
ModuleAssociatedPrimeSet = _import_engine().ModuleAssociatedPrimeSet
ModulePrimaryComponent = _import_engine().ModulePrimaryComponent
ModulePrimaryDecomposition = _import_engine().ModulePrimaryDecomposition
ModuleHomomorphism = _import_engine().ModuleHomomorphism
ModuleGroebnerBasis = _import_engine().ModuleGroebnerBasis
ModuleMonomial = _import_engine().ModuleMonomial
ModuleMonomialOrder = _import_engine().ModuleMonomialOrder
ModulePowerKind = _import_engine().ModulePowerKind
ModulePowerHomomorphism = _import_engine().ModulePowerHomomorphism
ModuleReduction = _import_engine().ModuleReduction
PadicChainCancellation = _import_engine().PadicChainCancellation
PadicChainReduction = _import_engine().PadicChainReduction
PadicCyclicSummand = _import_engine().PadicCyclicSummand
PadicModuleInvariants = _import_engine().PadicModuleInvariants
PresentedModule = _import_engine().PresentedModule
PresentedModuleHomology = _import_engine().PresentedModuleHomology
PresentedModuleHomomorphism = _import_engine().PresentedModuleHomomorphism
PresentedModulePower = _import_engine().PresentedModulePower
PresentedModulePowerHomomorphism = _import_engine().PresentedModulePowerHomomorphism
RankBounds = _import_engine().RankBounds
ResolutionCompleteness = _import_engine().ResolutionCompleteness
ResolutionMinimality = _import_engine().ResolutionMinimality
Submodule = _import_engine().Submodule
SubmoduleMembership = _import_engine().SubmoduleMembership
StrongModuleGroebnerBasis = _import_engine().StrongModuleGroebnerBasis
StrongModuleReduction = _import_engine().StrongModuleReduction
StrongModuleTerm = _import_engine().StrongModuleTerm

__all__ = [
    "ChainCancellation",
    "ChainComplex",
    "ChainReduction",
    "DerivedModule",
    "DerivedModuleHomomorphism",
    "ExteriorAlgebra",
    "ExteriorElement",
    "FreeModule",
    "FreeModulePower",
    "FreeResolution",
    "GradedBettiTable",
    "FiniteAbelianGroupCohomologyPresentation",
    "FiniteAbelianGroupIntegralCohomologyPresentation",
    "FiniteCyclicGroupIntegralCohomologyPresentation",
    "FinitelyGeneratedAbelianGroup",
    "FiniteGroupCohomologyAlgebraGenerator",
    "FiniteGroupCohomologyAlgebraTruncation",
    "FiniteGroupCohomologyCompletePresentation",
    "FiniteGroupCohomologyElementaryAbelianRestriction",
    "FiniteGroupCohomologySymondsParameterCertificate",
    "FiniteGroupCohomologyTruncation",
    "FiniteGroupIntegralCohomologyTruncation",
    "GradedHomology",
    "GradedModuleInvariants",
    "HomologyGroup",
    "ImageSubmodule",
    "KernelSubmodule",
    "LocalModuleOrder",
    "LocalModuleReduction",
    "LocalModuleStandardBasis",
    "ModuleElement",
    "ModuleAssociatedPrimeSet",
    "ModulePrimaryComponent",
    "ModulePrimaryDecomposition",
    "ModuleHomomorphism",
    "ModuleGroebnerBasis",
    "ModuleMonomial",
    "ModuleMonomialOrder",
    "ModulePowerKind",
    "ModulePowerHomomorphism",
    "ModuleReduction",
    "PadicChainCancellation",
    "PadicChainReduction",
    "PadicCyclicSummand",
    "PadicModuleInvariants",
    "PresentedModule",
    "PresentedModuleHomology",
    "PresentedModuleHomomorphism",
    "PresentedModulePower",
    "PresentedModulePowerHomomorphism",
    "RankBounds",
    "ResolutionCompleteness",
    "ResolutionMinimality",
    "Submodule",
    "SubmoduleMembership",
    "StrongModuleGroebnerBasis",
    "StrongModuleReduction",
    "StrongModuleTerm",
    "UniversalUnit",
]
