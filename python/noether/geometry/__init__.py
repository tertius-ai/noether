# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Checked Euclidean geometry and exact combinatorial constructions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from noether._engine_gateway import _import_engine

_engine = _import_engine()
PointCloud = _engine.PointCloud
Triangulation = _engine.Triangulation
Polytope = _engine.Polytope


@runtime_checkable
class Metric(Protocol):
    """Finite metric contract consumed by Vietoris–Rips constructions."""

    def __len__(self) -> int: ...

    def distance(self, i: int, j: int) -> float: ...

    def enclosing_radius(self) -> float: ...


_PointRows = Iterable[Iterable[float]]


def _as_point_cloud(points: PointCloud | _PointRows) -> PointCloud:
    if isinstance(points, PointCloud):
        return points
    return PointCloud([[float(coordinate) for coordinate in point] for point in points])


def delaunay(points: PointCloud | _PointRows) -> Triangulation:
    """Return the exact-predicate Delaunay triangulation of ``points``."""

    return _engine._facade_delaunay(_as_point_cloud(points))


def regular_triangulation(
    points: PointCloud | _PointRows, weights: Sequence[float]
) -> Triangulation:
    """Return the regular triangulation with one finite weight per site."""

    return _engine._facade_regular_triangulation(
        _as_point_cloud(points), [float(weight) for weight in weights]
    )


def convex_hull(points: Iterable[Iterable[object]]) -> Polytope:
    """Return the exact rational convex hull of a finite point family."""

    return _engine._facade_convex_hull([list(point) for point in points])


__all__ = [
    "Metric",
    "PointCloud",
    "Polytope",
    "Triangulation",
    "convex_hull",
    "delaunay",
    "regular_triangulation",
]
