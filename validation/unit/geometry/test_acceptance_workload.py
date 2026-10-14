# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

from fractions import Fraction

import pytest

from noether.geometry import PointCloud, Polytope, Triangulation, convex_hull
from noether.topology import FilteredComplex


def test_point_cloud_to_alpha_filtration_acceptance_workload() -> None:
    cloud = PointCloud(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    assert cloud.affine_rank() == 3
    ordinary = cloud.delaunay()
    weighted = cloud.regular_triangulation([0.0, 1.0, 2.0, 3.0, 4.0])
    ordinary.verify()
    weighted.verify()
    assert PointCloud.from_json(cloud.to_json()) == cloud
    assert Triangulation.from_json(ordinary.to_json()) == ordinary
    assert Triangulation.from_json(weighted.to_json()) == weighted
    assert not ordinary.weighted
    assert weighted.weighted

    filtration = FilteredComplex.alpha(
        ordinary,
        max_scale=2.0,
        max_dimension=3,
    )
    filtration.verify()
    assert filtration.dimension == 3
    assert set(map(tuple, ordinary.simplices)).issubset(set(map(tuple, filtration.simplices)))

    triangle = PointCloud([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]).delaunay()
    assert triangle.circumcentres() == pytest.approx([([1.0, 1.0], 2.0**0.5)])


def test_exact_polytope_acceptance_workload() -> None:
    polytope = convex_hull(
        [
            [0, 0],
            [2, 0],
            [0, 1],
            [2, 1],
            [Fraction(1, 2), Fraction(1, 2)],
        ]
    )
    polytope.verify()

    normals, rhs = zip(*polytope.facets, strict=True)
    from_halfspaces = Polytope.from_halfspaces(normals, rhs)
    from_vertices = Polytope.from_vertices(from_halfspaces.vertices)
    reconstructed = Polytope.from_json(from_vertices.to_json())

    def incidence_relation(
        value: Polytope,
    ) -> frozenset[
        tuple[
            tuple[tuple[Fraction, ...], Fraction],
            frozenset[tuple[Fraction, ...]],
        ]
    ]:
        return frozenset(
            (
                (tuple(normal), rhs),
                frozenset(tuple(value.vertices[index]) for index in incident),
            )
            for (normal, rhs), incident in zip(
                value.facets, value.vertex_facet_incidence, strict=True
            )
        )

    assert reconstructed == polytope
    assert incidence_relation(reconstructed) == incidence_relation(polytope)
    assert reconstructed.vertex_facet_incidence == from_vertices.vertex_facet_incidence
    assert reconstructed.face_lattice() == from_vertices.face_lattice()
    assert reconstructed.face_lattice_covers() == from_vertices.face_lattice_covers()
    reconstructed.verify()
