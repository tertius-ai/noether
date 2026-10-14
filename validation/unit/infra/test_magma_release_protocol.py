# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The release Magma session is deterministic, complete, and fail-closed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "validation" / "bench"


def _load(name: str, path: Path):
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _driver():
    return _load("noether_magma_release_driver", BENCH / "magma_release.py")


def _runtime():
    return _load("noether_magma_release_runtime", BENCH / "_magma.py")


def _protocol(driver, samples: int = 2) -> str:
    lines = [driver.PROTOCOL, "VERSION|2.29.1", "META|setup_wall_s|0.25"]
    for index, identifier in enumerate(driver.EXPECTED_CASES, start=1):
        lines.append(f"RESULT|{identifier}|{json.dumps([index])}")
        lines.append(f"LOOPS|{identifier}|8")
        for sample in range(samples):
            lines.append(f"SAMPLE|{identifier}|{0.001 * index + sample * 1e-6}|0.0005")
    lines.extend(["META|script_wall_s|1.5", f"END_{driver.PROTOCOL}"])
    return "\n".join(lines)


def _magma_function(script: str, identifier: str) -> str:
    start = script.index(f"function Op_{identifier}()")
    end = script.index("end function;", start) + len("end function;")
    return script[start:end]


def _worker(driver, backend: str, wall: float, result_delta: int = 0) -> dict:
    cases = {}
    for index, identifier in enumerate(driver.EXPECTED_CASES, start=1):
        cases[identifier] = {
            "result": [index + (result_delta if identifier == "integer_smith" else 0)],
            "loops_per_sample": 8,
            "wall_samples_s": [wall, wall * 1.01, wall * 0.99],
            "cpu_samples_s": [wall / 2, wall / 2, wall / 2],
        }
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    return {
        "schema": driver.WORKER_SCHEMA,
        "backend": backend,
        "manifest_sha256": driver.sha256_json(manifest),
        "cases": cases,
    }


def test_manifest_and_materialized_inputs_are_frozen() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    assert manifest["measurement"]["magma_seed"] == 5929
    assert driver.sha256_json(manifest) == (
        "32cb920ec7bd2d3be5048b7520c371642472cfbb05b2bae609a40099dc392a9c"
    )
    materialized = driver.materialize_manifest(manifest)
    input_hashes = {name: driver.sha256_json(value) for name, value in materialized.items()}
    assert len(input_hashes) == 71
    assert driver.sha256_json(input_hashes) == (
        "ecdb7f9bdf1cf99e493584c7377ba4fac655a72ceec316c484620d978a9f0cea"
    )
    assert input_hashes["integer_smith"] == (
        "92da9a69ca1740b27c7494f8ca0d7d31fa98303faba65b7e760bfb8eb34f3f5d"
    )
    assert input_hashes["extension_field_elimination"] == (
        "0a4c250a594975092b1ba3be352c4c759e8fda50af3fe9665efa88dea1040fcf"
    )
    assert input_hashes["permutation_group_structure_nonsymmetric"] == (
        "beea7395c163ce1b0378cad5ec433561d235df63a13b8b542493a46557240f3c"
    )
    assert input_hashes["f2_rectangular_elimination"] == (
        "75964d4498b38cae56910d955f68a8f265a3f71965afc1a745fad6784f2bcdce"
    )
    assert input_hashes["extension_field_rectangular_elimination"] == (
        "655ce613ea6e7faa4e18bd79b13610fff5edf6fdd8943cce0e37d3ff3a648c21"
    )
    assert input_hashes["f2_rectangular_matmul"] == (
        "b0ad238f150a5ef2049205355df7cebf36b91267924dbe5a1a28650ca81c35b3"
    )
    assert input_hashes["f2_large_matmul"] == (
        "92ee9fa3cc1d1d47e52b87833a9d89912ac6bf240f132a323de82c90da130cb4"
    )
    assert input_hashes["f2_large_rectangular_matmul"] == (
        "db44568e37811413726c477a24b62020c99d081cb1172606b2cf5e8d5e7bbdae"
    )
    assert input_hashes["elliptic_curve_generalized_arithmetic"] == (
        "ce9f11348dcfb757d4cd6dc1ff3d1e991359f3c53b93da7de37d16c7c7f947b0"
    )
    assert input_hashes["odd_prime_higher_homology"] == (
        "2cce5372f098c70db00f390d3111b4762559870501512a255de52c54de17d083"
    )
    assert input_hashes["prime_field_generic_determinantal_resolution"] == (
        "c7c79ea759cd0cf7f80f9f1bce66c2ac6a949215b82d5417f907c94a8e3b3fbf"
    )
    assert input_hashes["prime_field_module_groebner_syzygy_nonmonomial"] == (
        "2a9a480b41f6eed0f18b5489f030b8f157da201effccaf0bf3c21ddf66941d4f"
    )
    assert input_hashes["kahler_cotangent"] == (
        "96202cf74bbf661ff86c9e0f76bdaef048898db735b5a1f448dd3b0c87e11106"
    )
    smith = materialized["integer_smith"]
    assert len(smith["matrix"]) == 32
    assert all(len(row) == 40 for row in smith["matrix"])
    assert len(materialized["f2_elimination"]["matrix"]) == 192
    assert len(materialized["f2_rectangular_matmul"]["left"]) == 320
    assert all(len(row) == 448 for row in materialized["f2_rectangular_matmul"]["left"])
    assert len(materialized["f2_rectangular_matmul"]["right"]) == 448
    assert all(len(row) == 288 for row in materialized["f2_rectangular_matmul"]["right"])
    assert len(materialized["f2_large_matmul"]["left"]) == 1536
    assert all(len(row) == 1536 for row in materialized["f2_large_matmul"]["left"])
    assert len(materialized["f2_large_matmul"]["right"]) == 1536
    assert all(len(row) == 1536 for row in materialized["f2_large_matmul"]["right"])
    assert len(materialized["f2_large_rectangular_matmul"]["left"]) == 1025
    assert all(len(row) == 1537 for row in materialized["f2_large_rectangular_matmul"]["left"])
    assert len(materialized["f2_large_rectangular_matmul"]["right"]) == 1537
    assert all(len(row) == 769 for row in materialized["f2_large_rectangular_matmul"]["right"])
    assert len(materialized["f2_rectangular_elimination"]["matrix"]) == 128
    assert all(len(row) == 192 for row in materialized["f2_rectangular_elimination"]["matrix"])
    assert len(materialized["extension_field_rectangular_elimination"]["matrix"]) == 30
    assert all(
        len(row) == 36 for row in materialized["extension_field_rectangular_elimination"]["matrix"]
    )
    assert len(materialized["prime_field_homology"]["matrix"]) == 160
    assert len(materialized["integer_higher_homology"]["outgoing"]) == 20
    assert len(materialized["integer_higher_homology"]["incoming"]) == 64
    assert len(materialized["prime_field_higher_homology"]["outgoing"]) == 64
    assert len(materialized["prime_field_higher_homology"]["incoming"]) == 192
    for identifier, modulus in (
        ("integer_higher_homology", None),
        ("prime_field_higher_homology", 2),
        ("odd_prime_higher_homology", 7),
    ):
        outgoing = materialized[identifier]["outgoing"]
        incoming = materialized[identifier]["incoming"]
        product = [
            [
                sum(
                    outgoing[row][inner] * incoming[inner][column] for inner in range(len(incoming))
                )
                for column in range(len(incoming[0]))
            ]
            for row in range(len(outgoing))
        ]
        assert all(
            entry == 0 if modulus is None else entry % modulus == 0
            for row in product
            for entry in row
        )


def test_generated_magma_script_has_complete_and_fair_timing_scope() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    script = driver.magma_script(manifest, driver.materialize_manifest(manifest), 2)

    assert script.count(f'print "{driver.PROTOCOL}"') == 1
    assert script.count(f'print "END_{driver.PROTOCOL}"') == 1
    assert "SmithForm(IntegerSmithInput)" in script
    assert "smith, left" not in script.lower()
    assert "Kernel(Transpose(IntegerH1Outgoing))" in script
    assert "Kernel(Transpose(FieldH1Outgoing))" in script
    assert "return Moduli(homology);" in script
    assert script.count("FreeResolution(module)") == 9
    assert script.count("SyzygyModule(module)") == 3
    assert script.count("Groebner(module)") == 3
    assert script.count("AutomorphismGroup(") == 2
    normalizer = _magma_function(script, "permutation_group_normalizer")
    assert "ambient := Sym(8);" in normalizer
    assert "subgroup := sub<ambient" in normalizer
    assert "Normalizer(ambient, subgroup)" in normalizer
    nonregular_normalizer = _magma_function(script, "permutation_group_normalizer_nonregular")
    assert "ambient := Sym(8);" in nonregular_normalizer
    assert "ambient!(1,2,3,4,5,6)" in nonregular_normalizer
    nonsymmetric_set_stabilizer = _magma_function(
        script, "permutation_group_set_stabilizer_nonsymmetric"
    )
    assert "ambient := Sym(6);" in nonsymmetric_set_stabilizer
    assert "group := sub<ambient" in nonsymmetric_set_stabilizer
    intersection = _magma_function(script, "permutation_group_intersection")
    assert "ambient := Sym(5);" in intersection
    assert "alternating := sub<ambient" in intersection
    assert "point_stabilizer := sub<ambient" in intersection
    assert "alternating meet point_stabilizer" in intersection
    set_stabilizer = _magma_function(script, "permutation_group_set_stabilizer")
    assert "ambient := Sym(8);" in set_stabilizer
    assert "group := sub<ambient" in set_stabilizer
    petersen = _magma_function(script, "graph_petersen_automorphisms")
    asymmetric_tree = _magma_function(script, "graph_asymmetric_tree_automorphisms")
    assert "graph := Graph< 10 |" in petersen
    assert "graph := Graph< 7 |" in asymmetric_tree
    for stale_global in (
        "GroupSetSymmetric",
        "GroupNormalizerCyclic",
        "GroupIntersectionAlternating",
        "PetersenGraphInput",
        "AsymmetricTreeInput",
    ):
        assert stale_global not in script
    assert "DoubleCosetRepresentatives(" in script
    assert "k0^2+2*k1^2" not in script
    assert (
        "RationalKatsuraGenerators := [k0+2*k1+2*k2+2*k3-1, 2*k0*k1+2*k1*k2+2*k2*k3-k1"
    ) in script
    assert "setup_wall := ChangePrecision(Realtime(setup_start), 15)" in script
    assert "cpu_start := Cputime()" in script
    assert "wall_start := Realtime()" in script
    assert "ChangePrecision(Cputime(cpu_start) / loops, 15)" in script
    assert "ChangePrecision(Realtime(wall_start) / loops, 15)" in script
    assert "ChangePrecision(Realtime(setup_start), 15)" in script
    assert "ChangePrecision(Realtime(script_start), 15)" in script
    assert "procedure PrintJSON(value)" in script
    assert script.count("function Op_") == len(driver.EXPECTED_CASES)
    assert script.count("procedure Emit_") == len(driver.EXPECTED_CASES)
    for identifier in driver.EXPECTED_CASES:
        assert script.count(f"function Op_{identifier}()") == 1
        assert script.count(f"procedure Emit_{identifier}(value)") == 1
        assert script.count(f"LOOPS|{identifier}|") == 1
        assert script.count(f"SAMPLE|{identifier}|") == 1


def test_cache_sensitive_noether_cases_reconstruct_their_target_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import noether.combin as combin
    import noether.groups as groups

    real_graph = combin.Graph
    real_permutation_group = groups.PermutationGroup
    graph_constructions: list[int] = []
    group_constructions: list[int] = []

    class TrackingGraph:
        def __new__(cls, vertices, edges):
            graph_constructions.append(int(vertices))
            return real_graph(vertices, edges)

    class TrackingPermutationGroup:
        @staticmethod
        def from_generators(degree, generators):
            group_constructions.append(int(degree))
            return real_permutation_group.from_generators(degree, generators)

    monkeypatch.setattr(combin, "Graph", TrackingGraph)
    monkeypatch.setattr(groups, "PermutationGroup", TrackingPermutationGroup)

    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    workloads = driver.noether_workloads(driver.materialize_manifest(manifest))
    assert graph_constructions == []

    group_constructions.clear()
    for identifier in (
        "permutation_group_set_stabilizer",
        "permutation_group_set_stabilizer_nonsymmetric",
        "permutation_group_normalizer",
        "permutation_group_normalizer_nonregular",
        "permutation_group_intersection",
    ):
        workloads[identifier].operation()
    assert group_constructions == [8, 6, 8, 8, 8, 8, 5, 5]

    workloads["graph_petersen_automorphisms"].operation()
    workloads["graph_asymmetric_tree_automorphisms"].operation()
    assert graph_constructions == [10, 7]


def test_noether_side_executes_every_frozen_mathematical_workload() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    workloads = driver.noether_workloads(driver.materialize_manifest(manifest))
    results = {
        identifier: workloads[identifier].normalize(workloads[identifier].operation())
        for identifier in driver.EXPECTED_CASES
    }

    assert tuple(workloads) == driver.EXPECTED_CASES
    assert driver.sha256_json(results) == (
        "3887547d6fd818b81c00ed5e25289fbb29e0bcc792130bd25d5c48bcf2de23c6"
    )
    assert len(results["f2_large_matmul"]) == 1536 * 1536
    assert driver.sha256_json(results["f2_large_matmul"]) == (
        "8d9621f3c7be97eaf68c200f006ef840725ba3f055a86f9d65cf93e71752031c"
    )
    assert len(results["f2_large_rectangular_matmul"]) == 1025 * 769
    assert driver.sha256_json(results["f2_large_rectangular_matmul"]) == (
        "1140f5c434e746713269e3a64aa9291c91d927b15dccc913c8cc242b95f019d8"
    )
    assert results["f2_rectangular_elimination"][0] == 96
    assert results["extension_field_rectangular_elimination"][0] == [18]
    assert results["extension_field_arithmetic"] == [
        [3, 1],
        [4, 0],
        [3, 1],
        [1, 0],
        [1, 0],
    ]
    assert results["number_field_order"] == [5, 1]
    assert results["number_field_prime_decomposition"] == [[5, 1, 1], [5, 1, 1]]
    assert results["number_field_cubic_order"] == [-23, 1]
    assert results["number_field_cubic_prime_decomposition"] == [[8, 1, 3]]
    assert results["elliptic_curve_arithmetic"] == [76, 65, 1]
    assert results["elliptic_curve_generalized_arithmetic"] == [78, 54, 1]
    assert results["permutation_group_cyclic_nonprefix"] == [64, 64, 1]
    assert results["permutation_group_structure"][:7] == [40320, 0, 0, 0, 1, 128, 22]
    assert results["permutation_group_structure_nonsymmetric"] == [
        60,
        0,
        0,
        1,
        1,
        4,
        5,
        1,
        12,
        12,
        15,
        20,
    ]
    expected_group_structures = {
        "permutation_group_structure_s4": [24, 1, 0, 0, 1, 8, 5, 1, 3, 6, 6, 8],
        "permutation_group_structure_dihedral": [
            16,
            1,
            1,
            0,
            1,
            16,
            7,
            1,
            1,
            2,
            2,
            2,
            4,
            4,
        ],
        "permutation_group_structure_cyclic": [15, 1, 1, 0, 1, 3, 15, *([1] * 15)],
        "permutation_group_structure_quaternion": [8, 1, 1, 0, 1, 8, 5, 1, 1, 2, 2, 2],
        "permutation_group_structure_a6": [
            360,
            0,
            0,
            1,
            1,
            9,
            7,
            1,
            40,
            40,
            45,
            72,
            72,
            90,
        ],
        "permutation_group_structure_product": [
            36,
            1,
            0,
            0,
            1,
            4,
            9,
            1,
            2,
            2,
            3,
            3,
            4,
            6,
            6,
            9,
        ],
    }
    assert {
        identifier: results[identifier] for identifier in expected_group_structures
    } == expected_group_structures
    assert results["permutation_group_set_stabilizer"] == 576
    assert results["permutation_group_set_stabilizer_nonsymmetric"] == 18
    assert results["permutation_group_normalizer"] == 32
    assert results["permutation_group_normalizer_nonregular"] == 24
    assert results["permutation_group_intersection"] == 12
    assert results["permutation_group_double_cosets"] == [3, 24]
    assert results["integer_homology"][:2] == [8, 16]
    assert results["prime_field_homology"] == 32
    assert results["integer_higher_homology"][:2] == [12, 16]
    assert results["prime_field_higher_homology"] == 32
    assert results["odd_prime_higher_homology"] == 24
    expected_betti = [[0, 0, 1], [1, 2, 3], [2, 3, 2]]
    assert results["rational_graded_resolution"] == expected_betti
    assert results["prime_field_graded_resolution"] == expected_betti
    assert results["prime_field_generic_determinantal_resolution"] == expected_betti
    expected_complete_intersection_betti = [
        [0, 0, 1],
        [1, 2, 3],
        [2, 4, 3],
        [3, 6, 1],
    ]
    assert results["rational_complete_intersection_resolution"] == (
        expected_complete_intersection_betti
    )
    assert results["prime_field_complete_intersection_resolution"] == (
        expected_complete_intersection_betti
    )
    expected_mixed_complete_intersection_betti = [
        [0, 0, 1],
        [1, 2, 1],
        [1, 3, 1],
        [1, 4, 1],
        [2, 5, 1],
        [2, 6, 1],
        [2, 7, 1],
        [3, 9, 1],
    ]
    assert results["rational_mixed_complete_intersection_resolution"] == (
        expected_mixed_complete_intersection_betti
    )
    assert results["prime_field_mixed_complete_intersection_resolution"] == (
        expected_mixed_complete_intersection_betti
    )
    expected_cycle_betti = [[0, 0, 1], [1, 2, 5], [2, 3, 5], [3, 5, 1]]
    assert results["rational_cycle_resolution"] == expected_cycle_betti
    assert results["prime_field_cycle_resolution"] == expected_cycle_betti
    module_basis, module_syzygies = results["rational_module_groebner_syzygy"]
    assert len(module_basis) == 4
    assert len(module_syzygies) == 1
    prime_module_basis, prime_module_syzygies = results["prime_field_module_groebner_syzygy"]
    assert len(prime_module_basis) == 4
    assert len(prime_module_syzygies) == 1
    nonmonomial_basis, nonmonomial_syzygies = results[
        "prime_field_module_groebner_syzygy_nonmonomial"
    ]
    assert len(nonmonomial_basis) == 4
    assert len(nonmonomial_syzygies) == 1
    assert len(results["rational_lex_groebner"]) == 6
    assert len(results["prime_field_lex_groebner"]) == 6
    assert len(results["rational_katsura_groebner"]) == 7
    assert len(results["prime_field_katsura_groebner"]) == 7
    assert driver.sha256_json(results["rational_katsura_groebner"]) == (
        "12540ee1a799b501ea87dd45047e299e6ec28793d8482e12b62e1b321bafc82a"
    )
    assert driver.sha256_json(results["prime_field_katsura_groebner"]) == (
        "be52d6433c7fa315ce9d698add69a8c2a8f58c51506d4e19e80bf72ac8b7bd49"
    )
    assert results["graph_petersen_automorphisms"] == 120
    assert results["graph_asymmetric_tree_automorphisms"] == 1
    assert results["kahler_cotangent"] == [[0, 2], [1, 1]]


def test_protocol_requires_every_result_and_every_sample() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    parsed = driver.parse_magma_protocol(_protocol(driver), manifest, 2)
    assert parsed["runtime"]["script_version"] == "2.29.1"
    assert tuple(parsed["cases"]) == driver.EXPECTED_CASES
    assert all(len(case["wall_samples_s"]) == 2 for case in parsed["cases"].values())

    with pytest.raises(ValueError, match="produced 1 samples"):
        driver.parse_magma_protocol(
            _protocol(driver).replace("\nSAMPLE|integer_smith|0.003|0.0005", "", 1),
            manifest,
            2,
        )
    with pytest.raises(ValueError, match="sentinels are incomplete"):
        driver.parse_magma_protocol(
            _protocol(driver).replace(f"END_{driver.PROTOCOL}", ""), manifest, 2
        )
    with pytest.raises(ValueError, match="unexpected output"):
        driver.parse_magma_protocol(
            _protocol(driver).replace(
                "META|setup_wall_s|0.25",
                "META|setup_wall_s|0.25\nRuntime error: intrinsic failed",
            ),
            manifest,
            2,
        )


def test_comparison_checks_mathematics_before_performance() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    noether = _worker(driver, "noether", 0.002)
    magma = _worker(driver, "magma", 0.001)
    comparison, status = driver.compare_workers(manifest, noether, magma)
    assert comparison["all_results_agree"]
    assert status == "needs-investigation"
    assert comparison["comparisons"]["integer_determinant"]["material_slowdown"]

    mismatch = _worker(driver, "noether", 0.0005, result_delta=1)
    comparison, status = driver.compare_workers(manifest, mismatch, magma)
    assert status == "failed"
    assert not comparison["performance_compared"]
    assert comparison["correctness_failures"] == ["integer_smith"]
    assert "noether_wall" not in comparison["comparisons"]["integer_smith"]


def test_comparison_canonicalizes_finite_module_generator_units() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    noether = _worker(driver, "noether", 0.001)
    magma = _worker(driver, "magma", 0.001)
    modulus = 32_003

    noether["cases"]["prime_field_module_groebner_syzygy"]["result"] = [
        [[[[0, 1, 0, 1], [1, 0, 0, 2]], []]],
        [],
    ]
    magma["cases"]["prime_field_module_groebner_syzygy"]["result"] = [
        [[[[0, 1, 0, 3], [1, 0, 0, 6]], []]],
        [],
    ]
    noether["cases"]["prime_field_module_groebner_syzygy_nonmonomial"]["result"] = [
        [],
        [[[[0, 2, 0, 1], [1, 1, 0, 1]], [[1, 1, 0, modulus - 1]]]],
    ]
    magma["cases"]["prime_field_module_groebner_syzygy_nonmonomial"]["result"] = [
        [],
        [[[[0, 2, 0, modulus - 1], [1, 1, 0, modulus - 1]], [[1, 1, 0, 1]]]],
    ]

    comparison, status = driver.compare_workers(manifest, noether, magma)

    assert status == "passed"
    for identifier in driver.FINITE_MODULE_WIRE_CASES:
        case = comparison["comparisons"][identifier]
        assert case["correctness"] == "agree"
        assert case["noether_result"] == case["magma_result"]


def test_comparison_table_handles_correctness_failures(capsys: pytest.CaptureFixture[str]) -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    noether = _worker(driver, "noether", 0.001, result_delta=1)
    magma = _worker(driver, "magma", 0.001)
    comparison, status = driver.compare_workers(manifest, noether, magma)

    assert status == "failed"
    driver.print_comparison_table(comparison)
    output = capsys.readouterr().out
    smith_line = next(line for line in output.splitlines() if line.startswith("integer_smith"))
    assert smith_line.count("n/a") == 3
    assert smith_line.endswith("mismatch")


def test_zero_magma_timing_is_json_safe_and_requires_investigation() -> None:
    driver = _driver()
    manifest = driver.load_manifest(driver.DEFAULT_MANIFEST)
    noether = _worker(driver, "noether", 0.001)
    magma = _worker(driver, "magma", 0.0)

    comparison, status = driver.compare_workers(manifest, noether, magma)

    assert status == "needs-investigation"
    assert not comparison["performance_compared"]
    integer_factorization = comparison["comparisons"]["integer_factorization"]
    assert integer_factorization["noether_over_magma_median_wall_ratio"] is None
    assert integer_factorization["material_slowdown"] is None
    assert "performance ratio unavailable" in comparison["investigations"][0]
    json.dumps(comparison, allow_nan=False)


def test_runtime_fingerprints_selected_binary_and_cleans_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime()
    binary = tmp_path / "magma"
    binary.write_text("#!/bin/sh\necho 'Magma V2.29-1 test build'\n")
    binary.chmod(0o755)
    monkeypatch.setenv("NOETHER_MAGMA_BIN", str(binary))
    monkeypatch.setenv("LD_PRELOAD", "/private/noether/libflint.so")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/private/noether")
    monkeypatch.setenv("PYTHONPATH", "/private/noether/python")
    monkeypatch.setenv("MAGMA_STARTUP_FILE", "/private/magma/startup")

    assert runtime.magma_command(1, 5929) == [
        str(binary),
        "-n",
        "-S",
        "5929",
        "-t",
        "1",
        "-b",
    ]
    provenance = runtime.magma_provenance()
    assert provenance["version"] == "Magma V2.29-1 test build"
    assert provenance["executable"] == str(binary.resolve())
    assert len(provenance["executable_sha256"]) == 64
    environment = runtime.magma_environment()
    assert "LD_PRELOAD" not in environment
    assert "DYLD_LIBRARY_PATH" not in environment
    assert "PYTHONPATH" not in environment
    assert "MAGMA_STARTUP_FILE" not in environment


def test_source_provenance_rejects_untracked_executable_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Noether Test",
            "-c",
            "user.email=noether@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    driver = _driver()
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    assert driver.source_provenance()["clean"] is True

    (tmp_path / "untracked.py").write_text("value = 2\n", encoding="utf-8")
    provenance = driver.source_provenance()
    assert provenance["tracked_clean"] is True
    assert provenance["untracked_file_count"] == 1
    assert provenance["clean"] is False


def test_release_entry_point_is_one_command_and_outputs_are_untracked() -> None:
    makefile = (ROOT / "Makefile").read_text()
    validation_readme = (ROOT / "validation" / "README.md").read_text()
    assert "validate: engine-portable" in makefile
    assert "validate-sage: engine-portable" in makefile
    assert "validate-magma-release: require-magma engine-native" in makefile
    assert "rm -f $(CURDIR)/python/noether/_engine*.so" in makefile
    assert "$(CURDIR)/python/noether/_engine*.pyd" in makefile
    assert "expected exactly one" in makefile
    assert "manual removal of extension artifacts is neither required" in validation_readme
    assert "NOETHER_MAGMA_BIN=/path/to/magma make validate-magma-release" in validation_readme
    assert "target/magma-validation/core.json" in validation_readme
