# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import json
from fractions import Fraction

import pytest

from noether.geometry import Polytope, convex_hull


def square() -> Polytope:
    return convex_hull(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [Fraction(1, 2), Fraction(1, 2)],
        ]
    )


def test_exact_polytope_is_the_checked_domain_value() -> None:
    polytope = square()

    assert Polytope.__module__ == "noether.geometry"
    assert Polytope.type_id == "noether/geometry/Polytope@1"
    assert polytope.dimension == polytope.ambient_dimension == 2
    assert polytope.is_full_dimensional
    assert set(map(tuple, polytope.vertices)) == {
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1)),
    }
    assert polytope.f_vector == [1, 4, 4, 1]
    assert len(polytope.facets) == 4
    assert len(polytope.vertex_facet_incidence) == 4
    assert polytope.is_vertex([1, 1])
    assert not polytope.is_vertex([Fraction(1, 2), Fraction(1, 2)])
    assert polytope.is_facet([1, 0], 1)
    polytope.verify()


def test_halfspaces_and_json_reconstruct_the_same_polytope() -> None:
    polytope = Polytope.from_halfspaces(
        [[-1, 0], [1, 0], [0, -1], [0, 1]],
        [0, 1, 0, 1],
    )

    assert polytope == square()
    assert Polytope.from_vertices(polytope.vertices) == polytope
    assert Polytope.from_json(polytope.to_json()) == polytope

    payload = json.loads(polytope.to_json())
    payload["derived_facets"] = []
    with pytest.raises(ValueError):
        Polytope.from_json(json.dumps(payload))
    payload.pop("derived_facets")
    payload["type"] = "noether/geometry/PointCloud@1"
    with pytest.raises(ValueError, match="expected type"):
        Polytope.from_json(json.dumps(payload))


def test_unbounded_inexact_and_reduced_facets_fail_closed() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        Polytope.from_halfspaces([[-1]], [0])
    with pytest.raises(ValueError, match="infeasible"):
        Polytope.from_halfspaces([[1], [-1]], [0, -1])
    with pytest.raises(ValueError, match="dimension mismatch"):
        Polytope.from_halfspaces([[1], [-1]], [0])
    with pytest.raises(TypeError, match="int or fractions.Fraction"):
        convex_hull([[0.0], [1.0]])

    segment = convex_hull([[0, 0, 0], [1, 0, 0]])
    assert segment.dimension == 1
    assert segment.f_vector == [1, 2, 1]
    assert segment.face_lattice()
    with pytest.raises(ValueError, match="full-dimensional"):
        _ = segment.facets


def test_record_placeholders_and_loose_kernel_functions_are_not_shipped() -> None:
    import noether._engine as engine

    removed = {
        "Polyhedron",
        "PolyhedralCone",
        "PolyhedralFan",
        "convex_hull",
        "vertex_enumeration",
        "facet_enumeration",
        "double_description",
        "beneath_beyond",
    }
    assert not removed.intersection(dir(engine))
