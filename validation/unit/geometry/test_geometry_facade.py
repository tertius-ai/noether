# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import math

import pytest

from noether.geometry import Metric, PointCloud, Triangulation, delaunay, regular_triangulation


def square() -> PointCloud:
    return PointCloud([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_point_cloud_is_the_checked_metric_value() -> None:
    cloud = PointCloud([[0.0, 0.0], [3.0, 4.0]])

    assert PointCloud.__module__ == "noether.geometry"
    assert PointCloud.type_id == "noether/geometry/PointCloud@1"
    assert isinstance(cloud, Metric)
    assert cloud.points == [[0.0, 0.0], [3.0, 4.0]]
    assert cloud.ambient_dimension == 2
    assert cloud.affine_rank() == 1
    assert cloud.distance(0, 1) == 5.0
    assert cloud.enclosing_radius() == 5.0
    assert cloud[-1] == [3.0, 4.0]
    assert PointCloud.from_json(cloud.to_json()) == cloud


def test_point_cloud_rejects_invalid_rows_and_distinguishes_r0() -> None:
    with pytest.raises(ValueError, match="dimension"):
        PointCloud([[0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        PointCloud([[math.nan]])

    assert len(PointCloud([])) == 0
    assert len(PointCloud([[]])) == 1
    assert PointCloud([]) != PointCloud([[]])


def test_affine_hull_projection_is_a_point_cloud_isometry() -> None:
    cloud = PointCloud([[0.0, 0.0, 1.0], [3.0, 4.0, 1.0], [6.0, 8.0, 1.0]])
    projected = cloud.project_to_affine_hull()

    assert projected.ambient_dimension == 1
    assert projected.affine_rank() == 1
    assert projected.distance(0, 2) == pytest.approx(cloud.distance(0, 2))


def test_delaunay_and_regular_results_are_canonical_kernel_values() -> None:
    cloud = square()
    ordinary = delaunay(cloud)
    weighted = regular_triangulation(cloud, [0.0] * len(cloud))

    assert Triangulation.__module__ == "noether.geometry"
    assert Triangulation.type_id == "noether/geometry/Triangulation@1"
    assert ordinary == cloud.delaunay()
    assert ordinary.dimension == ordinary.dim == 2
    assert ordinary.simplices == ordinary.cells
    assert ordinary.n_cells == 2
    assert not ordinary.weighted and ordinary.weights is None
    assert weighted.weighted and weighted.weights == [0.0] * len(cloud)
    assert weighted.simplices == ordinary.simplices
    ordinary.verify()
    weighted.verify()
    assert Triangulation.from_json(ordinary.to_json()) == ordinary
    assert Triangulation.from_json(weighted.to_json()) == weighted


def test_triangulation_construction_fails_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        regular_triangulation(square(), [0.0, 0.0, math.inf, 0.0])
    with pytest.raises(ValueError, match="weights for"):
        regular_triangulation(square(), [0.0])


def test_duplicate_sites_do_not_create_degenerate_simplices() -> None:
    triangulation = delaunay([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])

    assert triangulation.n_cells == 2
    assert all(len(set(simplex)) == 3 for simplex in triangulation.simplices)
    triangulation.verify()
