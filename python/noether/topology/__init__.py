# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Simplicial objects, filtrations, persistence, and homology."""

from __future__ import annotations

from noether._engine_gateway import _import_engine

AdmissibleCohomologyOperation = _import_engine().AdmissibleCohomologyOperation
SimplicialComplex = _import_engine().SimplicialComplex
SimplicialCohomologyBasisElement = _import_engine().SimplicialCohomologyBasisElement
SimplicialCohomologyProduct = _import_engine().SimplicialCohomologyProduct
SimplicialCohomologyProductTerm = _import_engine().SimplicialCohomologyProductTerm
SimplicialCohomologyRing = _import_engine().SimplicialCohomologyRing
DominatedVertex = _import_engine().DominatedVertex
ElementaryCollapse = _import_engine().ElementaryCollapse
SimplicialCollapseSequence = _import_engine().SimplicialCollapseSequence
SimplicialSimpleHomotopyMove = _import_engine().SimplicialSimpleHomotopyMove
SimplicialSimpleHomotopyZigzag = _import_engine().SimplicialSimpleHomotopyZigzag
StrongCoreReduction = _import_engine().StrongCoreReduction
SimplicialMorseReduction = _import_engine().SimplicialMorseReduction
SimplicialSet = _import_engine().SimplicialSet
SimplicialSetCohomologyRing = _import_engine().SimplicialSetCohomologyRing
SimplicialSetElementaryHornCollapse = _import_engine().SimplicialSetElementaryHornCollapse
SimplicialSetHornCollapseSequence = _import_engine().SimplicialSetHornCollapseSequence
SimplicialSetAnodyneMove = _import_engine().SimplicialSetAnodyneMove
SimplicialSetAnodyneZigzag = _import_engine().SimplicialSetAnodyneZigzag
FilteredComplex = _import_engine().FilteredComplex
Barcode = _import_engine().Barcode
EilenbergMacLaneSpace = _import_engine().EilenbergMacLaneSpace
EilenbergMacLaneIntegralHomology = _import_engine().EilenbergMacLaneIntegralHomology
EilenbergMacLaneIntegralCohomology = _import_engine().EilenbergMacLaneIntegralCohomology
EilenbergMacLaneIntegralCohomologyGenerator = (
    _import_engine().EilenbergMacLaneIntegralCohomologyGenerator
)
EilenbergMacLaneIntegralCohomologyRelationTerm = (
    _import_engine().EilenbergMacLaneIntegralCohomologyRelationTerm
)
EilenbergMacLaneIntegralCohomologyRelation = (
    _import_engine().EilenbergMacLaneIntegralCohomologyRelation
)
EilenbergMacLaneIntegralCohomologyPresentation = (
    _import_engine().EilenbergMacLaneIntegralCohomologyPresentation
)
EilenbergMacLaneCohomologyGenerator = _import_engine().EilenbergMacLaneCohomologyGenerator
EilenbergMacLaneCohomologyOperation = _import_engine().EilenbergMacLaneCohomologyOperation
EilenbergMacLaneModularCohomology = _import_engine().EilenbergMacLaneModularCohomology
EilenbergMacLaneModularCohomologyTruncation = (
    _import_engine().EilenbergMacLaneModularCohomologyTruncation
)

__all__ = [
    "AdmissibleCohomologyOperation",
    "Barcode",
    "DominatedVertex",
    "ElementaryCollapse",
    "EilenbergMacLaneCohomologyGenerator",
    "EilenbergMacLaneCohomologyOperation",
    "EilenbergMacLaneIntegralCohomology",
    "EilenbergMacLaneIntegralCohomologyGenerator",
    "EilenbergMacLaneIntegralCohomologyPresentation",
    "EilenbergMacLaneIntegralCohomologyRelation",
    "EilenbergMacLaneIntegralCohomologyRelationTerm",
    "EilenbergMacLaneIntegralHomology",
    "EilenbergMacLaneModularCohomology",
    "EilenbergMacLaneModularCohomologyTruncation",
    "EilenbergMacLaneSpace",
    "FilteredComplex",
    "SimplicialCollapseSequence",
    "SimplicialCohomologyBasisElement",
    "SimplicialCohomologyProduct",
    "SimplicialCohomologyProductTerm",
    "SimplicialCohomologyRing",
    "SimplicialComplex",
    "SimplicialMorseReduction",
    "SimplicialSet",
    "SimplicialSetAnodyneMove",
    "SimplicialSetAnodyneZigzag",
    "SimplicialSetCohomologyRing",
    "SimplicialSetElementaryHornCollapse",
    "SimplicialSetHornCollapseSequence",
    "SimplicialSimpleHomotopyMove",
    "SimplicialSimpleHomotopyZigzag",
    "StrongCoreReduction",
]
