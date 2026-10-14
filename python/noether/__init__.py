# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Noether: deterministic scientific mathematics backed by Rust.

The package root intentionally contains only shared vocabulary. Mathematical
objects and operations live in their owning domain packages; the compiled
extension remains a private implementation detail.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from noether._discovery import (
    Capabilities,
    LocalConcurrency,
    MpiRuntime,
    OperationDescription,
    capabilities,
    describe,
)
from noether._engine_gateway import _import_engine

_Element = TypeVar("_Element")


@runtime_checkable
class Ring(Protocol[_Element]):
    """Structural Python view of a coefficient context."""

    def zero(self) -> _Element: ...

    def one(self) -> _Element: ...

    def __call__(self, value: object) -> _Element: ...


@runtime_checkable
class Field(Ring[_Element], Protocol[_Element]):
    """A coefficient context whose nonzero elements are invertible."""

    def inverse(self, value: _Element) -> _Element: ...


_engine = _import_engine()
NoetherError = _engine.NoetherError
UnsupportedOperationError = _engine.UnsupportedOperationError
ContextMismatchError = _engine.ContextMismatchError
VerificationError = _engine.VerificationError
BezoutIdentity = _engine.BezoutIdentity
IntegerFiniteFieldMap = _engine.IntegerFiniteFieldMap
IntegerRationalEmbedding = _engine.IntegerRationalEmbedding
IntegerReduction = _engine.IntegerReduction
IntegerRing = _engine.IntegerRing
PrimeSubfieldEmbedding = _engine.PrimeSubfieldEmbedding
RationalField = _engine.RationalField
Assumption = _engine.Assumption
MathematicalValidity = _engine.MathematicalValidity
RealBallField = _engine.RealBallField
RealBall = _engine.RealBall
ComplexBallField = _engine.ComplexBallField
ComplexBall = _engine.ComplexBall
Z = IntegerRing()
Q = RationalField()

__version__ = str(_engine.__version__)

__all__ = [
    "Assumption",
    "BezoutIdentity",
    "Capabilities",
    "LocalConcurrency",
    "MpiRuntime",
    "ComplexBall",
    "ComplexBallField",
    "ContextMismatchError",
    "Field",
    "IntegerFiniteFieldMap",
    "IntegerRationalEmbedding",
    "IntegerReduction",
    "IntegerRing",
    "MathematicalValidity",
    "NoetherError",
    "OperationDescription",
    "PrimeSubfieldEmbedding",
    "Q",
    "RationalField",
    "RealBall",
    "RealBallField",
    "Ring",
    "UnsupportedOperationError",
    "VerificationError",
    "Z",
    "__version__",
    "capabilities",
    "describe",
]
