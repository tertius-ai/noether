# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import importlib.util
from pathlib import Path

from noether._engine_gateway import _import_engine

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "scripts" / "gen_py_api_doc.py"
USAGE_GENERATOR = ROOT / "scripts" / "gen_usage_index.py"
SCHEMA_GENERATOR = ROOT / "scripts" / "gen_engine_schemas.py"


def _generator_module():
    spec = importlib.util.spec_from_file_location("noether_public_api_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _usage_generator_module():
    spec = importlib.util.spec_from_file_location("noether_usage_index_generator", USAGE_GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema_generator_module():
    spec = importlib.util.spec_from_file_location("noether_schema_generator", SCHEMA_GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_index_contains_only_declared_facade_domains() -> None:
    entries = _generator_module().build_index()

    assert {entry["module"] for entry in entries} == {
        "noether",
        "noether.category",
        "noether.combin",
        "noether.commalg",
        "noether.geometry",
        "noether.groups",
        "noether.homalg",
        "noether.linalg",
        "noether.numthy",
        "noether.topology",
    }
    canonical_names = [entry["canonical_name"] for entry in entries]
    assert len(canonical_names) == len(set(canonical_names))
    assert all("_engine" not in name for name in canonical_names)


def test_index_exactly_projects_the_rust_acceptance_lists() -> None:
    entries = _generator_module().build_index()
    _, accepted_types, accepted_operations, accepted_values = _import_engine()._accepted_surface()

    assert {
        entry["canonical_name"] for entry in entries if entry["kind"] in {"class", "type_alias"}
    } == set(accepted_types)
    assert {
        entry["canonical_name"]
        for entry in entries
        if entry["kind"] not in {"class", "type_alias", "value"}
    } == set(accepted_operations)
    assert {entry["canonical_name"] for entry in entries if entry["kind"] == "value"} == set(
        accepted_values
    )


def test_matrix_smith_description_comes_from_typed_stub_and_rust_docstring() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    smith = entries["linalg.Matrix.smith_normal_form"]
    assert len(smith["overloads"]) == 2
    assert smith["signatures"] == [
        "(self: Matrix[IntegerRing], *, algorithm: Literal['auto', 'exact']='auto', smooth_bound: None=None) -> SmithNormalForm",
        "(self: Matrix[IntegerRing], *, algorithm: Literal['smooth'], smooth_bound: int) -> SmithNormalForm",
    ]
    assert smith["overloads"][1]["input_schema"] == {
        "type": "object",
        "properties": {
            "algorithm": {
                "enum": ["smooth"],
                "x-python-type": "Literal['smooth']",
            },
            "smooth_bound": {"type": "integer"},
        },
        "required": ["algorithm", "smooth_bound"],
        "additionalProperties": False,
    }
    assert smith["result_schema"] == {"x-python-type": "SmithNormalForm"}
    assert "Smith normal form" in smith["doc"]


def test_overloads_retain_distinct_typed_schemas() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    determinant = entries["linalg.Matrix.det"]
    assert len(determinant["overloads"]) == 17
    assert {
        overload["result_schema"].get("x-python-type", overload["result_schema"].get("type"))
        for overload in determinant["overloads"]
    } == {
        "FiniteFieldElement",
        "ComplexBall",
        "NumberFieldElement",
        "PadicElement",
        "PadicLocalFieldElement",
        "Polynomial[_PolynomialCoefficientContext]",
        "PowerSeries[C]",
        "QadicElement",
        "QuotientElement[C]",
        "RationalFunction[RationalField]",
        "RealBall",
        "ResidueClass",
        "_Fraction",
        "integer",
    }


def test_editorial_metadata_only_names_rust_accepted_operations() -> None:
    generator = _generator_module()
    entries = generator.build_index()
    names = {entry["canonical_name"] for entry in entries}

    assert set(generator._DISCOVERY_METADATA) <= names
    assert {entry["domain"] for entry in entries if entry.get("example") is not None} == {
        "category",
        "combin",
        "commalg",
        "geometry",
        "groups",
        "homalg",
        "linalg",
        "numthy",
        "topology",
    }
    for entry in entries:
        example = entry.get("example")
        if example is not None:
            compile(example, f"<{entry['canonical_name']} example>", "exec")


def test_each_shipped_example_has_a_documented_headline_operation() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}
    headlines = (
        "IntegerRing.extended_gcd",
        "RealBallField.__call__",
        "linalg.Matrix.solve",
        "linalg.Matrix.smith_normal_form",
        "linalg.SparseMatrix.apply",
        "linalg.Lattice.shortest_vectors",
        "numthy.factor",
        "numthy.FiniteField.extension",
        "numthy.FiniteField.isomorphism_to",
        "numthy.PadicRing.hensel_lift",
        "numthy.QadicRing.teichmuller",
        "numthy.NumberField.maximal_order",
        "commalg.CrtDecomposition.project",
        "commalg.Polynomial.homogeneous_components",
        "commalg.Ideal.groebner_basis",
        "commalg.RationalFunctionField.__call__",
        "commalg.PowerSeries.pow",
        "commalg.Scheme.point_count",
        "groups.PermutationGroup.from_generators",
        "groups.PermutationGroup.conjugacy_classes",
        "groups.PolynomialGlRepresentation.second_chern_index",
        "combin.Graph.canonical_label",
        "combin.Matroid.tutte_polynomial",
        "combin.KashiwaraCrystal.tensor",
        "topology.FilteredComplex.persistent_homology",
        "geometry.convex_hull",
        "numthy.EllipticCurve.miller_function",
        "category.SteinerCategory.oriental",
        "linalg.IntegralLineConfiguration.automorphism_group",
        "homalg.PresentedModule.minimal_free_resolution",
        "numthy.PadicLocalField.maximal_order",
        "combin.Poset.core",
        "topology.SimplicialComplex.cohomology_ring",
        "groups.PermutationGroup.modular_cohomology",
        "topology.EilenbergMacLaneSpace.integral_homology",
        "commalg.Ideal.normalization",
        "commalg.ProjectiveCompleteIntersection.assuming_smooth",
        "linalg.Matrix.solve_exact",
        "commalg.QuotientRing.kahler_differentials",
        "homalg.FreeModule.exterior_algebra",
    )

    assert len(headlines) == 40
    assert all(entries[name]["summary"] for name in headlines)


def test_missing_descriptions_are_reported_without_invented_prose() -> None:
    entries = _generator_module().build_index()
    assert any(entry["summary_source"] == "missing" for entry in entries)
    for entry in entries:
        assert (not entry["summary"]) == (entry["summary_source"] == "missing")
        assert entry["summary_source"] != "derived projection vocabulary"


def test_missing_operator_and_evidence_descriptions_stay_empty() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    multiplication = entries["ComplexBall.__mul__"]
    assert multiplication["summary"] == ""
    assert multiplication["summary_source"] == "missing"

    replay = entries["IntegerFiniteFieldMap.verify"]
    assert replay["summary"] == ""
    assert replay["summary_source"] == "missing"

    authored = entries["linalg.Matrix.rank"]
    assert authored["summary"] == "Rank over a field context."
    assert authored["summary_source"] == "public docstring"


def test_authored_failure_and_dispatch_paragraphs_are_not_truncated() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    assert "Raises `ValueError`" in entries["groups.PermutationGroup.from_generators"]["doc"]
    persistence_doc = " ".join(
        entries["topology.FilteredComplex.persistent_homology"]["doc"].split()
    )
    assert "Extension fields use the generic facade reduction" in persistence_doc


def test_delegated_kahler_constructions_report_their_gil_release() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}
    delegated = {
        "commalg.PolynomialRing.kahler_differentials",
        "commalg.PolynomialRing.cotangent_space_at",
        "commalg.QuotientRing.kahler_differentials",
        "commalg.QuotientRing.cotangent_space_at",
        "commalg.QuotientRing.cotangent_complex",
        "commalg.QuotientRing.cotangent_complex_from_generators",
        "commalg.QuotientRing.conormal_sequence_from_generators",
    }

    assert all(entries[name]["releases_gil"] for name in delegated)


def test_delegated_module_functions_report_their_gil_release() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    assert entries["numthy.partial_factor"]["releases_gil"]
    assert entries["geometry.convex_hull"]["releases_gil"]


def test_type_aliases_and_values_retain_python_declarations() -> None:
    entries = {entry["canonical_name"]: entry for entry in _generator_module().build_index()}

    assert entries["combin.BeatPointKind"]["python_declaration"].startswith("type BeatPointKind =")
    assert entries["Z"]["python_declaration"] == "Z: IntegerRing"


def test_engine_schemas_have_identical_python_wire_identity() -> None:
    entries = _generator_module().build_index()
    schemas = _schema_generator_module().build_index()
    classes = {
        entry["type_id"]: entry["name"]
        for entry in entries
        if entry["kind"] == "class" and entry.get("type_id")
    }

    schema_classes = {type_id: record["py_class"] for type_id, record in schemas.items()}

    assert set(schema_classes) <= set(classes)
    assert {type_id: classes[type_id] for type_id in schema_classes} == schema_classes
    assert classes["noether/commalg/RationalFunction@1"] == "RationalFunction"
    assert classes["noether/commalg/RationalFunctionFieldMap@1"] == "RationalFunctionFieldMap"
    assert classes["noether/homalg/KernelSubmodule@1"] == "KernelSubmodule"
    assert classes["noether/homalg/ImageSubmodule@1"] == "ImageSubmodule"


def test_usage_vocabulary_preserves_canonical_rust_acceptance_names() -> None:
    engine = _import_engine()
    _, accepted_types, accepted_operations, accepted_values = engine._accepted_surface()
    accepted = set((*accepted_types, *accepted_operations, *accepted_values))

    symbols = _usage_generator_module().public_symbols()

    assert set(symbols) == accepted
    assert {"capabilities", "describe"} <= set(symbols)
    assert "linalg.Matrix.__add__" in symbols
    assert "schema_for" not in symbols
    assert "engine_m2_available" not in symbols
