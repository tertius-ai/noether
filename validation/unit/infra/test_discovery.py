# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
from __future__ import annotations

import pytest

import noether
from noether.linalg import Matrix


def test_describe_resolves_canonical_string_and_public_method() -> None:
    by_name = noether.describe("noether.linalg.Matrix.smith_normal_form")
    by_method = noether.describe(Matrix.smith_normal_form)

    assert by_name == by_method
    assert by_name.name == "linalg.Matrix.smith_normal_form"
    assert by_name.domain == "linalg"
    assert by_name.determinism == "deterministic"
    assert by_name.result_schema == {"x-python-type": "SmithNormalForm"}
    assert "Smith normal form" in by_name.summary

    transforms = noether.describe("linalg.Matrix.smith_normal_form_with_transforms")
    assert transforms.result_schema == {"x-python-type": "SmithNormalFormWithTransforms"}


def test_describe_preserves_overload_schemas() -> None:
    determinant = noether.describe("linalg.Matrix.det")

    assert "(self: Matrix[IntegerRing]) -> int" in determinant.signatures
    assert any("_PadicLocalIntegralContext" in item for item in determinant.signatures)
    assert {
        result.get("x-python-type", result.get("type")) for result in determinant.result_schemas
    } == {
        "ComplexBall",
        "FiniteFieldElement",
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


def test_one_discoverable_example_per_domain_executes() -> None:
    examples = {
        description.domain: description.example
        for description in (
            noether.describe("category.SteinerCategory.point"),
            noether.describe("combin.Poset.canonical_form"),
            noether.describe("commalg.ResidueRing.__call__"),
            noether.describe("geometry.convex_hull"),
            noether.describe("groups.PermutationGroup.from_generators"),
            noether.describe("homalg.ChainComplex.from_boundaries"),
            noether.describe("linalg.Matrix.smith_normal_form"),
            noether.describe("numthy.factor"),
            noether.describe("topology.SimplicialComplex.graded_homology"),
        )
    }

    assert set(examples) == set(noether.capabilities().domains)
    for domain, example in examples.items():
        assert example is not None
        exec(compile(example, f"<noether {domain} example>", "exec"), {})


def test_smith_discovery_names_strategy_validity_and_evidence() -> None:
    smith = noether.describe("linalg.Matrix.smith_normal_form")

    assert smith.supported_rings == ("Z",)
    assert smith.algorithms == ("exact", "smooth_local")
    assert smith.backends == ("flint", "native")
    assert smith.validity == (
        "unconditional",
        "conditional:TorsionOrderIsSmooth",
    )
    assert smith.evidence == ("SmithVerification",)
    assert smith.releases_gil
    assert noether.describe("linalg.Matrix.shape").releases_gil is False


def test_material_python_operations_advertise_gil_release() -> None:
    material_operations = (
        "numthy.factor",
        "numthy.partial_factor",
        "numthy.is_prime",
        "commalg.Polynomial.factor",
        "commalg.Polynomial.real_roots",
        "commalg.Scheme.point_count",
        "geometry.convex_hull",
        "combin.Poset.canonical_label",
        "combin.Poset.canonical_form",
        "combin.Poset.automorphism_group",
        "homalg.Submodule.groebner_basis",
        "homalg.Submodule.strong_groebner_basis",
        "homalg.Submodule.groebner_basis_with_hilbert_series",
        "homalg.Submodule.minimal_homogeneous_generators",
        "homalg.PresentedModule.hom",
        "homalg.PresentedModule.tor",
        "homalg.PresentedModule.ext",
        "homalg.PresentedModule.free_resolution",
        "homalg.PresentedModule.minimal_free_resolution",
        "homalg.FreeResolution.minimalize",
        "homalg.FreeResolution.tor",
        "homalg.FreeResolution.ext",
        "homalg.FreeResolution.verify",
        "groups.PermutationGroup.modular_cohomology",
        "groups.PermutationGroup.integral_cohomology",
        "groups.PermutationGroup.double_cosets",
        "groups.PermutationGroup.iterate_elements",
        "linalg.IntegralLineConfiguration.automorphism_group",
        "linalg.IntegralLineConfiguration.equivalence_to",
        "numthy.FractionalIdeal.factor",
        "commalg.Ideal.primary_decomposition",
        "topology.SimplicialComplex.cohomology_ring",
        "topology.SimplicialComplex.canonical_form",
        "topology.SimplicialComplex.strong_core",
        "topology.SimplicialSet.cohomology_ring",
        "topology.SimplicialSet.canonical_form",
        "topology.SimplicialSet.automorphism_group",
        "commalg.PolynomialRing.strong_groebner_basis",
        "commalg.PolynomialRing.nullstellensatz_certificate",
        "commalg.Ideal.nullstellensatz_certificate",
        "commalg.Scheme.closed_points",
        "homalg.ChainComplex.algebraic_morse_reduction",
        "homalg.ChainComplex.padic_unit_morse_reduction",
        "homalg.ChainComplex.reduction_from_cancellations",
        "topology.SimplicialComplex.cochain_complex",
        "topology.SimplicialComplex.morse_reduction",
        "topology.SimplicialComplex.morse_chain_complex",
        "topology.SimplicialComplex.cup_product",
        "topology.SimplicialComplex.cup_i",
        "topology.SimplicialComplex.steenrod_square",
        "topology.SimplicialComplex.steenrod_reduced_power",
        "topology.SimplicialComplex.bockstein_reduced_power",
        "topology.SimplicialSet.cochain_complex",
        "topology.SimplicialSet.normalized_chain_reduction",
        "topology.SimplicialSet.greedy_horn_collapse_reduction",
        "topology.SimplicialSet.cup_product",
        "topology.SimplicialSet.cup_i",
        "topology.SimplicialSet.steenrod_square",
        "topology.SimplicialSet.steenrod_reduced_power",
        "topology.SimplicialSet.bockstein_reduced_power",
        "linalg.Lattice.minimum_vector_configuration",
        "combin.Matroid.tutte_polynomial",
        "combin.Poset.graded_homology_invariants",
        "combin.CoreReduction.graded_homology_invariants",
        "homalg.PresentedModuleHomomorphism.kernel",
        "homalg.PresentedModuleHomomorphism.image",
        "homalg.PresentedModuleHomomorphism.coimage",
        "homalg.PresentedModuleHomomorphism.homology",
        "homalg.PresentedModuleHomomorphism.is_equal",
        "homalg.PresentedModuleHomomorphism.is_zero",
        "homalg.PresentedModuleHomomorphism.is_exact_at_source",
        "homalg.Submodule.colon_by_element",
        "homalg.Submodule.saturate_by_element",
        "commalg.Ideal.finite_quotient_algebra",
        "commalg.Ideal.equidimensional_hull",
        "commalg.Ideal.noether_normalization",
        "commalg.Ideal.rees_algebra",
        "commalg.Ideal.associated_graded_algebra",
        "commalg.Ideal.grauert_remmert_step",
        "commalg.Ideal.normalization",
        "commalg.Ideal.normalization_from_smooth_model",
        "commalg.Ideal.reduced_normalization",
        "commalg.Ideal.equidimensional_radical_decomposition",
        "commalg.Ideal.certify_primary",
        "commalg.Ideal.admit_primary_decomposition",
        "commalg.Ideal.radical",
        "commalg.Ideal.eliminate",
        "commalg.Ideal.eliminate_with_evidence",
        "commalg.Ideal.rational_univariate_representation",
        "commalg.Ideal.normal_forms",
        "commalg.Ideal.reduce",
        "category.CategoricalHomologyPresentation.skeleton",
        "category.CategoricalHomologyPresentation.component_poset_slice",
        "category.CategoricalHomologyPresentation.fundamental_poset",
        "category.CategoricalHomologyPresentation.particle_homotopy_poset",
        "category.FiniteOmegaCategory.strong_steiner_model",
        "category.FiniteOmegaCategory.mapping_category",
        "category.FiniteOmegaCategory.component_poset",
        "category.FiniteOmegaCategory.exact_cells",
        "category.FiniteOmegaCategory.exact_cell_counts",
        "category.OrientedBasepoint.iterated_mapping_category",
        "category.OrientedBasepoint.homotopy_poset",
        "combin.Poset.width",
        "combin.Poset.is_lattice",
        "combin.Poset.is_self_dual",
        "combin.Poset.core",
        "combin.Poset.order_complex",
        "geometry.Triangulation.verify",
        "linalg.Matrix.apply_block",
        "linalg.Matrix.__matmul__",
        "groups.PermutationGroup.__contains__",
        "groups.PermutationGroup.stabilizer",
        "groups.PermutationGroup.is_transitive",
    )

    if "distributed" in noether.capabilities().compiled_features:
        material_operations += (
            "linalg.DistributedMatrix.transpose",
            "linalg.DistributedMatrix.paste",
            "linalg.DistributedMatrix.matmul",
            "linalg.DistributedMatrix.nullspace",
            "linalg.DistributedMatrix.__add__",
            "linalg.DistributedMatrix.__sub__",
            "linalg.DistributedMatrix.__matmul__",
        )

    assert all(noether.describe(name).releases_gil for name in material_operations)


def test_describe_rejects_unknown_and_ambiguous_short_names() -> None:
    with pytest.raises(ValueError, match="unknown public"):
        noether.describe("linalg.Matrix.not_an_operation")
    with pytest.raises(ValueError, match="ambiguous"):
        noether.describe("from_json")


def test_capabilities_report_the_installed_wheel_profile() -> None:
    installed = noether.capabilities()

    base_domains = (
        "category",
        "combin",
        "commalg",
        "geometry",
        "groups",
        "homalg",
        "linalg",
        "numthy",
        "topology",
    )
    assert installed.domains == base_domains
    assert installed.supports("linalg.Matrix.smith_normal_form")
    assert installed.supports("linalg.Matrix.shape")
    assert installed.supports("noether.numthy.factor")
    assert not installed.supports("linalg.Matrix.not_an_operation")
    assert "numthy.PadicLocalFieldOrder" in installed.types
    assert "numthy.PadicLocalFractionalIdeal" in installed.types
    assert installed.supports("numthy.PadicLocalFieldElement.minimal_polynomial")
    assert len(installed.operations) > len(installed.types)
    if "distributed" in installed.compiled_features:
        assert installed.supports("linalg.DistributedMatrix.rank")
    else:
        assert not installed.supports("linalg.DistributedMatrix.rank")
    assert installed.values == ("Q", "Z", "__version__")
    assert installed.backend("native").available
    assert installed.backend("flint").available
    for name in (
        "convex-hull",
        "flint",
        "geom-core",
        "graph-core",
        "lattice-enum",
        "m4rst",
        "m4rste",
        "mzs",
        "perm-core",
        "pwvla",
        "sparb",
        "sparla",
    ):
        assert installed.backend(name).compiled
        assert installed.backend(name).available
    distributed = installed.backend("m4rst-distributed")
    assert distributed.compiled == ("distributed" in installed.compiled_features)
    assert distributed.available == distributed.compiled
    for name in ("msolve", "macaulay2", "singular"):
        assert not installed.backend(name).compiled
        assert not installed.backend(name).available
    concurrency = installed.local_concurrency
    assert concurrency.schedulable_threads >= 1
    assert 1 <= concurrency.effective_threads <= concurrency.schedulable_threads
    assert concurrency.affinity_source == "std::thread::available_parallelism"
    assert concurrency.override_name == "NOETHER_NUM_THREADS"
    assert concurrency.phase_owner == "noether-rayon"


def test_discovery_reprs_are_bounded() -> None:
    installed = noether.capabilities()
    description = noether.describe("linalg.Matrix.from_rows")

    assert len(repr(installed).encode()) < 1_024
    assert len(repr(description).encode()) < 1_024
    assert f"types={len(installed.types)}" in repr(installed)
    assert f"operations={len(installed.operations)}" in repr(installed)
    assert "linalg.Matrix.from_rows" in repr(description)


def test_python_version_is_owned_by_the_rust_facade() -> None:
    from noether import _engine

    assert noether.__version__ == _engine.__version__
    assert noether.capabilities().version == _engine.__version__


def test_conveyed_profile_exposes_no_raw_subprocess_evaluators() -> None:
    from noether import _engine

    assert not hasattr(_engine, "engine_msolve_run")
    assert not hasattr(_engine, "engine_m2_eval")
    assert not hasattr(_engine, "engine_singular_eval")


def test_discovery_names_are_small_root_vocabulary() -> None:
    assert "OperationDescription" in noether.__all__
    assert "Capabilities" in noether.__all__
    assert "describe" in noether.__all__
    assert "capabilities" in noether.__all__
    assert "BackendStatus" not in noether.__all__
