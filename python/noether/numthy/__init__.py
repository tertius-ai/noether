# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Number theory, arithmetic fields, and arithmetic geometry."""

from noether._engine_gateway import _import_engine

_engine = _import_engine()
EllipticCurve = _engine.EllipticCurve
EllipticCurvePoint = _engine.EllipticCurvePoint
FiniteField = _engine.FiniteField
FiniteFieldElement = _engine.FiniteFieldElement
FiniteFieldEmbedding = _engine.FiniteFieldEmbedding
FiniteFieldIsomorphism = _engine.FiniteFieldIsomorphism
IntegerFactorization = _engine.IntegerFactorization
MillerFunction = _engine.MillerFunction
NthPowerClass = _engine.NthPowerClass
NthPowerClassGroup = _engine.NthPowerClassGroup
PartialIntegerFactorization = _engine.PartialIntegerFactorization
FractionalIdeal = _engine.FractionalIdeal
NumberField = _engine.NumberField
NumberFieldElement = _engine.NumberFieldElement
NumberFieldOrder = _engine.NumberFieldOrder
PadicRing = _engine.PadicRing
PadicField = _engine.PadicField
PadicRingFrobenius = _engine.PadicRingFrobenius
PadicFieldFrobenius = _engine.PadicFieldFrobenius
PadicRingRestriction = _engine.PadicRingRestriction
PadicFieldRestriction = _engine.PadicFieldRestriction
PadicIntegralEmbedding = _engine.PadicIntegralEmbedding
PadicElement = _engine.PadicElement
PadicValuation = _engine.PadicValuation
PadicLocalPrecision = _engine.PadicLocalPrecision
PadicLocalField = _engine.PadicLocalField
PadicLocalFieldElement = _engine.PadicLocalFieldElement
PadicLocalFieldOrder = _engine.PadicLocalFieldOrder
PadicLocalFractionalIdeal = _engine.PadicLocalFractionalIdeal
PadicLocalFieldRestriction = _engine.PadicLocalFieldRestriction
PadicLocalFieldCoefficientFrobenius = _engine.PadicLocalFieldCoefficientFrobenius
PointCountCertificate = _engine.PointCountCertificate
QadicRing = _engine.QadicRing
QadicField = _engine.QadicField
QadicRingFrobenius = _engine.QadicRingFrobenius
QadicFieldFrobenius = _engine.QadicFieldFrobenius
QadicRingRestriction = _engine.QadicRingRestriction
QadicFieldRestriction = _engine.QadicFieldRestriction
QadicIntegralEmbedding = _engine.QadicIntegralEmbedding
QadicElement = _engine.QadicElement
factor = _engine._facade_integer_factor
partial_factor = _engine._facade_integer_partial_factor
is_prime = _engine._facade_integer_is_prime
cyclotomic_polynomial = _engine._facade_cyclotomic_polynomial

__all__ = [
    "EllipticCurve",
    "EllipticCurvePoint",
    "FiniteField",
    "FiniteFieldElement",
    "FiniteFieldEmbedding",
    "FiniteFieldIsomorphism",
    "IntegerFactorization",
    "MillerFunction",
    "NthPowerClass",
    "NthPowerClassGroup",
    "PartialIntegerFactorization",
    "FractionalIdeal",
    "NumberField",
    "NumberFieldElement",
    "NumberFieldOrder",
    "PadicField",
    "PadicFieldFrobenius",
    "PadicFieldRestriction",
    "PadicIntegralEmbedding",
    "PadicLocalField",
    "PadicLocalFieldCoefficientFrobenius",
    "PadicLocalFieldElement",
    "PadicLocalFieldOrder",
    "PadicLocalFractionalIdeal",
    "PadicLocalFieldRestriction",
    "PadicLocalPrecision",
    "PadicElement",
    "PadicRing",
    "PadicRingFrobenius",
    "PadicRingRestriction",
    "PadicValuation",
    "PointCountCertificate",
    "QadicField",
    "QadicFieldFrobenius",
    "QadicFieldRestriction",
    "QadicIntegralEmbedding",
    "QadicElement",
    "QadicRing",
    "QadicRingFrobenius",
    "QadicRingRestriction",
    "cyclotomic_polynomial",
    "factor",
    "partial_factor",
    "is_prime",
]
