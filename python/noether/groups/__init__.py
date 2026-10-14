# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Finite groups, actions, and representation theory.

GAP is a validation oracle, and ``nTj`` labels are external names. Catalog
lookup and GAP-backed convenience functions therefore do not form part of this
package boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from noether._engine_gateway import _import_engine

_engine = _import_engine()

Permutation = _engine.Permutation
PermutationGroup = _engine.PermutationGroup
PolynomialGlRepresentation = _engine.PolynomialGlRepresentation
PairOrbit = _engine.PairOrbit
DoubleCoset = _engine.DoubleCoset
SubsetOrbit = _engine.SubsetOrbit
OrbitStabilizer = _engine.OrbitStabilizer
RandomElement = _engine.RandomElement
SchreierGenerator = _engine.SchreierGenerator
SchreierTree = _engine.SchreierTree
StabilizerChain = _engine.StabilizerChain
StabilizerChainStep = _engine.StabilizerChainStep
Transversal = _engine.Transversal
WeightCharacter = _engine.WeightCharacter
WeightLattice = _engine.WeightLattice
WeightLatticeMap = _engine.WeightLatticeMap

ConjugacyClass = _engine.ConjugacyClass


@runtime_checkable
class FiniteGroup(Protocol):
    """Structural contract shared by concrete finite groups."""

    @property
    def order(self) -> int: ...

    def identity(self): ...

    def contains(self, value) -> bool: ...

    def multiply(self, left, right): ...

    def inverse(self, value): ...


__all__ = [
    "ConjugacyClass",
    "DoubleCoset",
    "FiniteGroup",
    "OrbitStabilizer",
    "PairOrbit",
    "Permutation",
    "PermutationGroup",
    "PolynomialGlRepresentation",
    "RandomElement",
    "SchreierGenerator",
    "SchreierTree",
    "StabilizerChain",
    "StabilizerChainStep",
    "SubsetOrbit",
    "Transversal",
    "WeightCharacter",
    "WeightLattice",
    "WeightLatticeMap",
]
