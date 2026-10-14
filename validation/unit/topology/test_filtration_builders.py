# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import math

import pytest

from noether.geometry import Metric, PointCloud
from noether.topology import FilteredComplex


class PathMetric:
    def __len__(self) -> int:
        return 3

    def distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        if {i, j} == {0, 2}:
            return 2.0
        return 1.0

    def enclosing_radius(self) -> float:
        return 1.0


def test_rips_accepts_the_metric_protocol_without_coordinates() -> None:
    metric = PathMetric()
    filtration = FilteredComplex.vietoris_rips(metric, max_scale=1.0, max_dimension=2)

    assert isinstance(metric, Metric)
    assert filtration.entries == [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([0, 1], 1.0),
        ([1, 2], 1.0),
    ]


def test_point_cloud_uses_the_direct_rust_metric_path() -> None:
    points = PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    filtration = FilteredComplex.vietoris_rips(points, max_scale=2.0, max_dimension=2)

    assert filtration.dimension == 2
    assert filtration.values[-1] == pytest.approx(math.sqrt(2.0))


def test_cech_uses_radius_and_minimum_enclosing_balls() -> None:
    edge = PointCloud([[0.0], [2.0]])
    edge_filtration = FilteredComplex.cech(edge, max_scale=1.0, max_dimension=1)
    assert edge_filtration.values[-1] == 1.0

    obtuse = PointCloud([[0.0, 0.0], [4.0, 0.0], [2.0, 0.1]])
    filtration = FilteredComplex.cech(obtuse, max_scale=2.0, max_dimension=2)
    assert ([0, 1, 2], 2.0) in filtration.entries


def test_alpha_reuses_a_checked_triangulation() -> None:
    points = PointCloud([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    triangulation = points.delaunay()
    filtration = FilteredComplex.alpha(triangulation, max_scale=3.0, max_dimension=2)

    assert ([0, 1, 2], 2.5) in filtration.entries
    assert triangulation.n_cells == 1
    triangulation.verify()


def test_alpha_cost_and_weighted_preconditions_fail_closed() -> None:
    points = PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triangulation = points.delaunay()
    with pytest.raises(ValueError, match="face slots"):
        FilteredComplex.alpha(
            triangulation,
            max_scale=1.0,
            max_dimension=2,
            max_face_slots=0,
        )

    weighted = points.regular_triangulation([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="ordinary Delaunay"):
        FilteredComplex.alpha(weighted, max_scale=1.0, max_dimension=2)


def test_invalid_scales_and_python_metrics_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        FilteredComplex.vietoris_rips(PathMetric(), max_scale=math.nan, max_dimension=1)

    class AsymmetricMetric(PathMetric):
        def distance(self, i: int, j: int) -> float:
            if (i, j) == (1, 0):
                return 3.0
            return super().distance(i, j)

    with pytest.raises(ValueError, match="symmetric"):
        FilteredComplex.vietoris_rips(AsymmetricMetric(), max_scale=1.0, max_dimension=1)

    class InfiniteMetric(PathMetric):
        def distance(self, i: int, j: int) -> float:
            return 0.0 if i == j else math.inf

    with pytest.raises(ValueError, match="finite"):
        FilteredComplex.vietoris_rips(InfiniteMetric(), max_scale=1.0, max_dimension=1)


def test_empty_structural_metric_stays_empty() -> None:
    class EmptyMetric:
        def __len__(self) -> int:
            return 0

        def distance(self, i: int, j: int) -> float:
            raise AssertionError(f"no distance exists between {i} and {j}")

        def enclosing_radius(self) -> float:
            return math.inf

    filtration = FilteredComplex.vietoris_rips(EmptyMetric(), max_scale=1.0, max_dimension=1)
    assert filtration.is_empty()
