# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Finite combinatorics backed by Noether's Rust kernels."""

from __future__ import annotations

from typing import Literal, TypeAlias

from noether._engine_gateway import _import_engine

SymmetricBasis: TypeAlias = Literal[
    "monomial",
    "power_sum",
    "elementary",
    "complete_homogeneous",
    "schur",
]
BeatPointKind: TypeAlias = Literal["down", "up"]

_engine = _import_engine()

CanonicalLabel = _engine.CanonicalLabel
BeatPoint = _engine.BeatPoint
Composition = _engine.Composition
CoreReduction = _engine.CoreReduction
Graph = _engine.Graph
GraphTraversal = _engine.GraphTraversal
ConnectedComponents = _engine.ConnectedComponents
KashiwaraCrystal = _engine.KashiwaraCrystal
SemistandardTableaux = _engine.SemistandardTableaux
ShortestPath = _engine.ShortestPath
SymmetricFunction = _engine.SymmetricFunction
Tableau = _engine.Tableau
GraphLabeling = _engine.GraphLabeling
Matroid = _engine.Matroid
Partition = _engine.Partition
Tree = _engine.Tree
Poset = _engine.Poset
PosetCanonicalLabel = _engine.PosetCanonicalLabel

__all__ = [
    "CanonicalLabel",
    "BeatPoint",
    "BeatPointKind",
    "Composition",
    "CoreReduction",
    "ConnectedComponents",
    "Graph",
    "GraphLabeling",
    "GraphTraversal",
    "KashiwaraCrystal",
    "Matroid",
    "Partition",
    "Poset",
    "PosetCanonicalLabel",
    "SemistandardTableaux",
    "ShortestPath",
    "SymmetricBasis",
    "SymmetricFunction",
    "Tableau",
    "Tree",
]
