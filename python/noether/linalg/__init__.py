# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Exact linear algebra over explicit coefficient contexts."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from noether._engine_gateway import _import_engine

Matrix = _import_engine().Matrix
CongruenceInconsistencyCertificate = _import_engine().CongruenceInconsistencyCertificate
CongruenceSolutionModule = _import_engine().CongruenceSolutionModule
CongruenceSystemResult = _import_engine().CongruenceSystemResult
FrobeniusSemilinearMap = _import_engine().FrobeniusSemilinearMap
IntegerInconsistencyCertificate = _import_engine().IntegerInconsistencyCertificate
IntegerLinearSystemResult = _import_engine().IntegerLinearSystemResult
IntegerSolutionModule = _import_engine().IntegerSolutionModule
IntegralLineConfiguration = _import_engine().IntegralLineConfiguration
IntegralLineEquivalence = _import_engine().IntegralLineEquivalence
IntegralLineAutomorphismGroup = _import_engine().IntegralLineAutomorphismGroup
IntegralLineOrbitAction = _import_engine().IntegralLineOrbitAction
IntegralLineSubgroup = _import_engine().IntegralLineSubgroup
MinimumCompletion = _import_engine().MinimumCompletion
MinimumCompletionCertificate = _import_engine().MinimumCompletionCertificate
MinimumNonrealizabilityCertificate = _import_engine().MinimumNonrealizabilityCertificate
MinimumRealization = _import_engine().MinimumRealization
MinimumRealizationDiagnostic = _import_engine().MinimumRealizationDiagnostic
MinimumVectorConfiguration = _import_engine().MinimumVectorConfiguration
Lattice = _import_engine().Lattice
LatticeEnumeration = _import_engine().LatticeEnumeration
ClosestLatticeVectors = _import_engine().ClosestLatticeVectors
SparseMatrix = _import_engine().SparseMatrix
PLUQFactorization = _import_engine().PLUQFactorization
PadicSmithInvariants = _import_engine().PadicSmithInvariants
SmithNormalForm = _import_engine().SmithNormalForm
SmithNormalFormWithTransforms = _import_engine().SmithNormalFormWithTransforms
SmithVerification = _import_engine().SmithVerification
SolutionSpace = _import_engine().SolutionSpace
Vector = _import_engine().Vector

try:
    DistributedError = _import_engine().DistributedError
    DistributedMatrix = _import_engine().DistributedMatrix
    DistributedPLUQFactorization = _import_engine().DistributedPLUQFactorization
    MpiThreadSupport = _import_engine().MpiThreadSupport
    World = _import_engine().World
except AttributeError:
    # Portable wheels intentionally omit the optional MPI surface.
    pass

R = TypeVar("R")


@runtime_checkable
class LinearOperator(Protocol[R]):
    """A linear map acting on vectors or dense blocks over one context."""

    @property
    def ring(self) -> R: ...

    @property
    def input_dimension(self) -> int: ...

    @property
    def output_dimension(self) -> int: ...

    def apply(self, vector: Vector[R]) -> Vector[R]: ...

    def apply_block(self, block: Matrix[R]) -> Matrix[R]: ...


__all__ = [
    "CongruenceInconsistencyCertificate",
    "CongruenceSolutionModule",
    "CongruenceSystemResult",
    "IntegerInconsistencyCertificate",
    "IntegerLinearSystemResult",
    "IntegerSolutionModule",
    "LinearOperator",
    "IntegralLineConfiguration",
    "IntegralLineEquivalence",
    "IntegralLineAutomorphismGroup",
    "IntegralLineOrbitAction",
    "IntegralLineSubgroup",
    "MinimumCompletion",
    "MinimumCompletionCertificate",
    "MinimumNonrealizabilityCertificate",
    "MinimumRealization",
    "MinimumRealizationDiagnostic",
    "MinimumVectorConfiguration",
    "Lattice",
    "LatticeEnumeration",
    "ClosestLatticeVectors",
    "FrobeniusSemilinearMap",
    "Matrix",
    "SparseMatrix",
    "PLUQFactorization",
    "PadicSmithInvariants",
    "SmithNormalForm",
    "SmithNormalFormWithTransforms",
    "SmithVerification",
    "SolutionSpace",
    "Vector",
]

if "DistributedMatrix" in globals():
    __all__ += [
        "DistributedError",
        "DistributedMatrix",
        "DistributedPLUQFactorization",
        "MpiThreadSupport",
        "World",
    ]
