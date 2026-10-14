# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""The Macaulay2 comparison keeps truncated resolutions honest."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = ROOT / "validation" / "bench"
DRIVER_PATH = BENCH_ROOT / "macaulay2_core.py"
PRESENTED_DRIVER_PATH = BENCH_ROOT / "macaulay2_presented_modules.py"
DECOMPOSITION_DRIVER_PATH = BENCH_ROOT / "macaulay2_decomposition.py"
NORMALIZATION_DRIVER_PATH = BENCH_ROOT / "macaulay2_normalization.py"
AFFINE_NORMALIZATION_DRIVER_PATH = BENCH_ROOT / "macaulay2_affine_normalization.py"
DERIVED_DRIVER_PATH = BENCH_ROOT / "macaulay2_derived.py"
RUNTIME_PATH = BENCH_ROOT / "_macaulay2.py"
SCORECARD_PATH = BENCH_ROOT / "macaulay2_scorecard.py"
M2_DRIVER_PATHS = (
    DRIVER_PATH,
    PRESENTED_DRIVER_PATH,
    DECOMPOSITION_DRIVER_PATH,
    NORMALIZATION_DRIVER_PATH,
    AFFINE_NORMALIZATION_DRIVER_PATH,
    BENCH_ROOT / "macaulay2_quotient.py",
    BENCH_ROOT / "macaulay2_derived.py",
    BENCH_ROOT / "groebner_compare.py",
)


def _runtime():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location("noether_macaulay2_runtime", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scorecard_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location("noether_macaulay2_scorecard", SCORECARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location("noether_macaulay2_core_benchmark", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _presented_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location(
        "noether_macaulay2_presented_module_benchmark", PRESENTED_DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decomposition_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location(
        "noether_macaulay2_decomposition_benchmark", DECOMPOSITION_DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalization_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location(
        "noether_macaulay2_normalization_benchmark", NORMALIZATION_DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _affine_normalization_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location(
        "noether_macaulay2_affine_normalization_benchmark",
        AFFINE_NORMALIZATION_DRIVER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _derived_driver():
    sys.path.insert(0, str(BENCH_ROOT))
    spec = importlib.util.spec_from_file_location(
        "noether_macaulay2_derived_benchmark", DERIVED_DRIVER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_macaulay2_runtime_records_the_selected_binary_and_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "M2-1.26.06"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ -n "$LD_PRELOAD$LD_LIBRARY_PATH$DYLD_INSERT_LIBRARIES$DYLD_LIBRARY_PATH" ]; '
        "then echo contaminated-native-loader; exit 0; fi\n"
        'if [ "$1" = "--version" ]; then echo 1.26.06; exit 0; fi\n'
        "printf 'Depth|1.6\\nPrimaryDecomposition|2.0\\n'\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("NOETHER_M2_BIN", str(binary))
    monkeypatch.setenv("NOETHER_M2_REVISION", "release-1.26.06-test")
    monkeypatch.setenv("LD_PRELOAD", "/private/noether/libflint.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/private/noether")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/private/noether/libflint.dylib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/private/noether")

    runtime = _runtime()
    assert runtime.m2_available()
    assert runtime.m2_command("--version") == [str(binary), "--version"]
    provenance = runtime.m2_provenance(("Depth", "PrimaryDecomposition"))
    assert provenance["macaulay2_executable"] == str(binary.resolve())
    assert provenance["macaulay2_version"] == "1.26.06"
    assert provenance["macaulay2_source_revision"] == "release-1.26.06-test"
    assert provenance["macaulay2_package_versions"] == {
        "Depth": "1.6",
        "PrimaryDecomposition": "2.0",
    }
    assert len(provenance["macaulay2_executable_sha256"]) == 64


def test_macaulay2_runtime_rejects_package_name_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "M2"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("NOETHER_M2_BIN", str(binary))
    runtime = _runtime()
    with pytest.raises(ValueError, match="invalid Macaulay2 package name"):
        runtime.m2_package_versions(('Depth"); error "escaped',))


def test_every_permanent_macaulay2_driver_uses_the_shared_runtime() -> None:
    for path in M2_DRIVER_PATHS:
        source = path.read_text()
        assert "m2_command(" in source, path.name
        assert 'return ["M2",' not in source, path.name


def test_normalization_protocol_retains_noncanonical_linear_maps() -> None:
    driver = _normalization_driver()
    case = driver.CASES["hyperbola-q"]
    noether_output = "\n".join(
        [
            driver.PROTOCOL,
            "METRIC|operation_cpu_s|0.001",
            "COORDINATES|2",
            "POLY|C/0",
            "TERM|C/0|1|{1,0}",
            "TERM|C/0|1|{0,1}",
            "POLY|C/1",
            "TERM|C/1|1|{0,1}",
            "PARAMETERS|1",
            "INDEX|0",
            f"END_{driver.PROTOCOL}",
        ]
    )
    m2_output = "\n".join(
        [
            driver.PROTOCOL,
            "METRIC|operation_cpu_s|0.002",
            "COORDINATES|2",
            "POLY|C/0",
            "TERM|C/0|1|{1,0}",
            "POLY|C/1",
            "TERM|C/1|-1|{1,0}",
            "TERM|C/1|1|{0,1}",
            "PARAMETERS|1",
            "POLY|P/0",
            "TERM|P/0|1|{0,1}",
            f"END_{driver.PROTOCOL}",
        ]
    )

    noether = driver.parse_protocol(noether_output, case)
    m2 = driver.parse_protocol(m2_output, case)

    assert noether.parameter_indices == [0]
    assert m2.parameter_indices == []
    assert m2.parameter_polynomials == [{(0, 1): Fraction(1)}]
    with pytest.raises(ValueError, match="singular"):
        driver._invert_matrix([[Fraction(1), Fraction(0)], [Fraction(1), Fraction(0)]], 0)


def test_affine_normalization_protocol_admits_only_shared_invariants() -> None:
    driver = _affine_normalization_driver()
    case = driver.CASES["node-q"]
    output = "\n".join(
        [
            driver.PROTOCOL,
            "METRIC|operation_cpu_s|0.001",
            "CONDUCTOR|2",
            "POLY|C/0",
            "TERM|C/0|1|{1,0}",
            "POLY|C/1",
            "TERM|C/1|1|{0,1}",
            "DIMENSION|1",
            "COMPONENTS|2",
            "NORMAL|true",
            "MAP|true",
            f"END_{driver.PROTOCOL}",
        ]
    )

    parsed = driver.parse_protocol(output, case)
    admitted, checks = driver.admitted_hash(case, parsed)
    assert len(admitted) == 64
    assert "canonical-source-conductor" in checks
    assert parsed.component_count == 2

    parsed.normal_target = False
    with pytest.raises(ValueError, match="not certified normal"):
        driver.admitted_hash(case, parsed)

    assert "betterNormalizationMap" in driver.m2_script(case, "warm", 2)
    assert "icMap" in driver.m2_script(driver.CASES["cusp-q"], "warm", 2)


def test_scorecard_requires_one_fingerprinted_oracle_and_exact_agreement() -> None:
    scorecard = _scorecard_driver()
    schema = "noether/macaulay2-decomposition-benchmark@1"
    common = {
        "schema": schema,
        "record": "run",
        "status": "ok",
        "case": "fat-point",
        "operation": "primary-decomposition",
    }
    records = [
        {**common, "backend": "noether", "backend_version": "test"},
        {
            **common,
            "backend": "M2",
            "backend_version": "1.26.06",
            "macaulay2_executable": "/opt/M2-1.26.06",
            "macaulay2_executable_sha256": "a" * 64,
            "macaulay2_source_revision": "release-1.26.06",
            "macaulay2_package_versions": {"PrimaryDecomposition": "2.0"},
        },
        {
            "schema": schema,
            "record": "comparison",
            "status": "ok",
            "case": "fat-point",
            "operation": "primary-decomposition",
            "semantic_output_agreement": {"noether": "same", "M2": "same"},
        },
    ]
    result = scorecard.build_scorecard(records, "1.26.06")
    assert result["status"] == "ok"
    assert result["semantic_cells"] == 1
    assert result["verified_comparisons"] == 1
    assert result["exact_agreement_records"] == 1
    assert result["independent_identity_records"] == 0
    assert result["unpaired_records"] == 0
    assert result["macaulay2_package_versions"] == {"PrimaryDecomposition": "2.0"}

    mixed = [*records, {**records[1], "macaulay2_executable_sha256": "b" * 64}]
    with pytest.raises(ValueError, match="mix Macaulay2 macaulay2_executable_sha256"):
        scorecard.build_scorecard(mixed)

    missing_revision = [*records[:1], {**records[1], "macaulay2_source_revision": None}, records[2]]
    with pytest.raises(ValueError, match="lack required provenance macaulay2_source_revision"):
        scorecard.build_scorecard(missing_revision)

    disagreement = [
        *records[:2],
        {**records[2], "semantic_output_agreement": {"noether": "left", "M2": "right"}},
    ]
    with pytest.raises(ValueError, match="failed semantic comparisons"):
        scorecard.build_scorecard(disagreement)


def test_scorecard_recognizes_every_macaulay2_driver_family() -> None:
    scorecard = _scorecard_driver()
    schemas = set()
    for path in M2_DRIVER_PATHS:
        source = path.read_text()
        marker = 'REPORT_SCHEMA = "'
        start = source.find(marker)
        assert start >= 0, path
        start += len(marker)
        end = source.find('"', start)
        schemas.add(source[start:end])

    assert schemas <= scorecard.KNOWN_FAMILIES.keys()


def test_scorecard_distinguishes_independent_witnesses_from_unpaired_cells() -> None:
    scorecard = _scorecard_driver()
    schema = "noether/macaulay2-core-benchmark@1"
    provenance = {
        "backend_version": "1.26.06",
        "macaulay2_executable": "/opt/M2-1.26.06",
        "macaulay2_executable_sha256": "a" * 64,
        "macaulay2_source_revision": "release-1.26.06",
    }
    records = []
    for backend, semantic_level in (("noether", "verified"), ("M2", "witnessed")):
        records.append(
            {
                "schema": schema,
                "record": "run",
                "status": "ok",
                "case": "module-chain",
                "operation": "gb-witness",
                "mode": "warm",
                "backend": backend,
                "semantic_level": semantic_level,
                "checks": ["change-matrix-identity"],
                **(provenance if backend == "M2" else {}),
            }
        )
    records.extend(
        [
            {
                "schema": schema,
                "record": "comparison",
                "status": "ok",
                "case": "module-chain",
                "operation": "gb-witness",
                "mode": "warm",
                "semantic_output_agreement": {},
            },
            {
                "schema": schema,
                "record": "comparison",
                "status": "ok",
                "case": "unsupported-hilbert",
                "operation": "hilbert",
                "mode": "warm",
                "semantic_output_agreement": {},
            },
        ]
    )

    result = scorecard.build_scorecard(records, "1.26.06")

    assert result["semantic_cells"] == 1
    assert result["verified_comparisons"] == 1
    assert result["exact_agreement_records"] == 0
    assert result["independent_identity_records"] == 1
    assert result["unpaired_records"] == 1

    bad_records = [
        *records,
        {
            "schema": schema,
            "record": "run",
            "status": "ok",
            "case": "paired-without-proof",
            "operation": "hilbert",
            "mode": "warm",
            "backend": "noether",
        },
        {
            "schema": schema,
            "record": "run",
            "status": "ok",
            "case": "paired-without-proof",
            "operation": "hilbert",
            "mode": "warm",
            "backend": "M2",
            **provenance,
        },
        {
            "schema": schema,
            "record": "comparison",
            "status": "ok",
            "case": "paired-without-proof",
            "operation": "hilbert",
            "mode": "warm",
            "semantic_output_agreement": {},
        },
    ]
    with pytest.raises(ValueError, match="failed semantic comparisons"):
        scorecard.build_scorecard(bad_records, "1.26.06")


def _twisted_cubic_protocol(driver) -> str:
    case = driver.CASES["twisted-cubic-q"]
    lines = [
        driver.PROTOCOL,
        "META|length|1",
        "META|completeness|through_homological_degree",
        "META|through_degree|1",
        "DEGREES|C0|1",
        "DEGREE|C0|{0}",
        "DEGREES|C1|3",
        "DEGREE|C1|{2}",
        "DEGREE|C1|{2}",
        "DEGREE|C1|{2}",
        "MATRIX|d1|1|3",
    ]
    for column, polynomial in enumerate(case.presentation.entries[0]):
        label = f"d1/0/{column}"
        lines.append(f"POLY|{label}")
        for numerator, denominator, exponent in polynomial.terms:
            powers = ",".join(str(value) for value in exponent)
            lines.append(f"TERM|{label}|{Fraction(numerator, denominator)}|{{{powers}}}")
    lines.append(f"END_{driver.PROTOCOL}")
    return "\n".join(lines)


def _redundant_mingens_protocol(driver) -> str:
    return "\n".join(
        [
            driver.PROTOCOL,
            "META|generator_count|2",
            "DEGREES|M|2",
            "DEGREE|M|{1}",
            "DEGREE|M|{1}",
            "MATRIX|M|1|2",
            "POLY|M/0/0",
            "TERM|M/0/0|1|{1,0}",
            "POLY|M/0/1",
            "TERM|M/0/1|1|{0,1}",
            f"END_{driver.PROTOCOL}",
        ]
    )


def _matrix_protocol(driver, case, operation: str) -> str:
    right = case.matrix_vector if operation == "matrix-application" else case.matrix_right
    assert right is not None
    result = driver.matrix_product(
        driver.matrix_spec_value(case.presentation, case.ring),
        driver.matrix_spec_value(right, case.ring),
        case.ring,
    )
    lines = [driver.PROTOCOL, f"MATRIX|P|{len(result)}|{len(result[0])}"]
    for row, values in enumerate(result):
        for column, polynomial in enumerate(values):
            label = f"P/{row}/{column}"
            lines.append(f"POLY|{label}")
            for exponent, coefficient in polynomial.items():
                powers = ",".join(str(value) for value in exponent)
                lines.append(f"TERM|{label}|{coefficient}|{{{powers}}}")
    lines.append(f"END_{driver.PROTOCOL}")
    return "\n".join(lines)


def _scalar_protocol(driver, case, operation: str) -> str:
    result = driver.scalar_operation_expected(case, operation)
    lines = [driver.PROTOCOL, "POLY|P"]
    for exponent, coefficient in result.items():
        powers = ",".join(str(value) for value in exponent)
        lines.append(f"TERM|P|{coefficient}|{{{powers}}}")
    lines.append(f"END_{driver.PROTOCOL}")
    return "\n".join(lines)


def test_truncated_resolution_sources_use_matched_degree_one_claims() -> None:
    driver = _driver()
    case = driver.CASES["twisted-cubic-q"]
    operation = "resolution-through-1"

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    candidate = driver.parse_protocol(_twisted_cubic_protocol(driver), case, operation)
    verifier = driver.noether_resolution_verifier_script(case, candidate, operation, "warm", 1)

    assert "LengthLimit => 1" in m2
    assert "META|completeness|through_homological_degree" in m2
    assert "minimal_free_resolution_through(order, 1)" in noether
    assert "ResolutionCompleteness.through(1)" in verifier
    assert "ResolutionCompleteness.complete()" not in verifier


def test_truncated_protocol_checks_the_known_prefix_and_rejects_a_complete_label() -> None:
    driver = _driver()
    case = driver.CASES["twisted-cubic-q"]
    operation = "resolution-through-1"
    protocol = _twisted_cubic_protocol(driver)

    parsed = driver.parse_protocol(protocol, case, operation)

    assert parsed.metadata == {
        "length": "1",
        "completeness": "through_homological_degree",
        "through_degree": "1",
    }
    assert "resolution-completeness-metadata" in parsed.checks
    assert "known-betti-fixture" in parsed.checks
    with pytest.raises(ValueError, match="inconsistent completeness metadata"):
        driver.parse_protocol(
            protocol.replace(
                "META|completeness|through_homological_degree", "META|completeness|complete"
            ),
            case,
            operation,
        )


def test_every_frozen_truncated_case_has_a_hand_known_tail_beyond_degree_one() -> None:
    driver = _driver()

    cases = [case for case in driver.CASES.values() if "resolution-through-1" in case.operations]

    assert len(cases) == 8
    assert all(dict(case.expected_betti).keys() >= {0, 1, 2} for case in cases)
    assert all(driver.grading_data(case)[0] != "ungraded" for case in cases)


def test_minimal_generator_lane_has_matched_construction_and_exact_admission() -> None:
    driver = _driver()
    case = driver.CASES["redundant-plane-generators-q"]
    operation = "mingens"
    protocol = _redundant_mingens_protocol(driver)

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    candidate = driver.parse_protocol(protocol, case, operation)
    verifier = driver.noether_mingens_verifier_script(case, candidate, "warm", 1)

    assert "mingens image" in m2
    assert ".minimal_homogeneous_generators()" in noether
    assert "candidate.minimal_homogeneous_generators()" in verifier
    assert "candidate_basis.contains(value)" in verifier
    assert "original_basis.contains(value)" in verifier
    assert candidate.metadata["generator_count"] == "2"
    assert candidate.degrees["M"] == [(1,), (1,)]
    assert "known-minimal-generator-fixture" in candidate.checks
    with pytest.raises(ValueError, match="minimal-generator fixture mismatch"):
        driver.parse_protocol(protocol.replace("DEGREE|M|{1}", "DEGREE|M|{2}", 1), case, operation)


def test_every_frozen_minimal_generator_case_has_a_known_first_betti_row() -> None:
    driver = _driver()

    cases = [case for case in driver.CASES.values() if "mingens" in case.operations]

    assert len(cases) == 9
    assert all(1 in dict(case.expected_betti) for case in cases)
    assert all(driver.grading_data(case)[0] != "ungraded" for case in cases)


def test_integral_module_lane_preserves_nonunit_leading_coefficients() -> None:
    driver = _driver()
    case = driver.CASES["module-coefficient-gcd-z"]

    m2 = driver.m2_script(case, "gb", "warm", 1)
    noether = driver.noether_script(case, "gb", "warm", 1)
    protocol = "\n".join(
        [
            driver.PROTOCOL,
            "MATRIX|G|2|2",
            "POLY|G/0/0",
            "TERM|G/0/0|1|{1}",
            "POLY|G/0/1",
            "POLY|G/1/0",
            "POLY|G/1/1",
            "TERM|G/1/1|5|{1}",
            f"END_{driver.PROTOCOL}",
        ]
    )

    parsed = driver.parse_protocol(protocol, case, "gb")

    assert case.ring.coefficient_context == "Z"
    assert "R = ZZ[x" in m2
    assert "coefficient_ring = Z" in noether
    assert ".strong_groebner_basis(order)" in noether
    assert parsed.matrices["G"][1][1][(1,)] == 5
    assert not isinstance(parsed.matrices["G"][1][1][(1,)], Fraction)
    assert driver.noether_supported(case, "syzygy")
    assert not driver.noether_supported(case, "resolution")


@pytest.mark.parametrize("operation", ["matrix-application", "matrix-composition"])
def test_polynomial_matrix_lane_times_matched_public_operations(operation: str) -> None:
    driver = _driver()
    case = driver.CASES["polynomial-matrix-48-q"]

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    parsed = driver.parse_protocol(_matrix_protocol(driver, case, operation), case, operation)

    assert m2.index("FMatrix =") < m2.index("Started0 = cpuTime()")
    assert noether.index("left = build_sparse") < noether.index("def compute():")
    if operation == "matrix-application":
        assert "FMatrix * VMatrix" in m2
        assert "return left.apply(vector)" in noether
    else:
        assert "FMatrix * RMatrix" in m2
        assert "return left @ right" in noether
    assert parsed.checks == ["matrix-product-identity"]
    assert parsed.output_terms > 0
    assert driver.noether_supported(case, operation)


@pytest.mark.parametrize(
    ("case_name", "operation"),
    [
        ("polynomial-scalar-256-q", "scalar-addition"),
        ("polynomial-scalar-256-f65521", "scalar-multiplication"),
        ("polynomial-scalar-256-q", "scalar-derivative"),
        ("polynomial-scalar-256-f65521", "scalar-evaluation"),
        ("polynomial-scalar-256-q", "scalar-substitution"),
        ("polynomial-univariate-56-f65521", "scalar-exact-division"),
        ("polynomial-univariate-56-q", "scalar-gcd"),
        ("polynomial-normal-form-256-f65521", "scalar-normal-form"),
    ],
)
def test_scalar_lanes_time_matched_claims_and_check_independent_results(
    case_name: str, operation: str
) -> None:
    driver = _driver()
    case = driver.CASES[case_name]

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    parsed = driver.parse_protocol(_scalar_protocol(driver, case, operation), case, operation)

    assert m2.index("ScalarValues =") < m2.index("Started0 = cpuTime()")
    assert noether.index("scalar_values =") < noether.index("def compute():")
    assert parsed.checks == ["independent-polynomial-arithmetic"]
    assert parsed.output_terms > 0
    assert driver.noether_supported(case, operation)
    if operation == "scalar-gcd":
        assert "leadCoefficient" in m2
    if operation == "scalar-normal-form":
        assert m2.index("ScalarReducers =") < m2.index("Started0 = cpuTime()")
        assert noether.index("scalar_ideal =") < noether.index("def compute():")


def _presented_matrix_protocol(driver, case, exponents: list[tuple[int, ...]]) -> str:
    lines = [driver.PROTOCOL, f"MATRIX|G|1|{len(exponents)}"]
    for column, exponent in enumerate(exponents):
        powers = ",".join(str(value) for value in exponent)
        lines.extend(
            [
                f"POLY|G/0/{column}",
                f"TERM|G/0/{column}|1|{{{powers}}}",
            ]
        )
    lines.extend(["METRIC|operation_cpu_s|0.001", f"END_{driver.PROTOCOL}"])
    return "\n".join(lines)


def test_presented_module_corpus_covers_polynomial_quotient_and_nonzero_homology() -> None:
    driver = _presented_driver()

    assert len(driver.CASES) == 8
    assert all(case.operations == driver.OPERATIONS for case in driver.CASES.values())
    assert {case.ring_shape for case in driver.CASES.values()} == {
        "polynomial",
        "polynomial-quotient",
    }
    assert {case.coefficient_context for case in driver.CASES.values()} == {"Q", "F_65521"}
    assert any(case.middle_rank == 8 for case in driver.CASES.values())
    assert any(case.expected_homology_relations is not None for case in driver.CASES.values())


@pytest.mark.parametrize("operation", ["kernel", "image", "homology"])
def test_presented_module_lane_times_forced_public_results(operation: str) -> None:
    driver = _presented_driver()
    case = driver.CASES["cyclic-presented-r1-q"]

    m2 = driver.m2_script(case, operation, "warm", 2)
    noether = driver.noether_script(case, operation, "warm", 2)

    assert m2.index("Source=") < m2.index("Started0=cpuTime()")
    assert noether.index("source=presented") < noether.index("def compute")
    assert "build_maps()" in noether
    assert "    result=None\n    started_wall=" in noether
    if operation in {"kernel", "image"}:
        assert "presentation ResultRun0" in m2
        assert f"return outgoing.{operation}(order)" in noether
    else:
        assert "presentation ResultRun0" in m2
        assert "return outgoing.homology(incoming,order)" in noether


def test_presented_module_admission_ignores_redundant_generator_choices() -> None:
    driver = _presented_driver()
    case = driver.CASES["cyclic-presented-r1-q"]
    minimal = driver.parse_protocol(_presented_matrix_protocol(driver, case, [(7, 0)]), case)
    redundant = driver.parse_protocol(
        _presented_matrix_protocol(driver, case, [(7, 0), (12, 0)]), case
    )

    minimal_hash, minimal_checks = driver.admitted_hash(case, "kernel", minimal)
    redundant_hash, redundant_checks = driver.admitted_hash(case, "kernel", redundant)

    assert minimal_hash == redundant_hash
    assert minimal_checks == redundant_checks == ("canonical-fixed-ambient-submodule",)


def test_presented_module_protocol_rejects_a_false_exactness_fixture() -> None:
    driver = _presented_driver()
    case = driver.CASES["cyclic-presented-r1-q"]
    protocol = "\n".join(
        [
            driver.PROTOCOL,
            "META|exact|false",
            "METRIC|operation_cpu_s|0.001",
            f"END_{driver.PROTOCOL}",
        ]
    )

    parsed = driver.parse_protocol(protocol, case)
    with pytest.raises(ValueError, match="wrong exactness predicate"):
        driver.admitted_hash(case, "exactness", parsed)


def test_presented_module_campaign_names_its_only_approved_warm_gap() -> None:
    driver = _presented_driver()

    assert driver.DOCUMENTED_WARM_DISPATCH_GAPS == {
        ("nonzero-cyclic-homology-quotient-f65521", "homology")
    }
    assert all(case in driver.CASES for case, _ in driver.DOCUMENTED_WARM_DISPATCH_GAPS)


def _module_associated_prime_protocol(driver, case) -> str:
    lines = [driver.PROTOCOL, f"META|count|{len(case.expected_associated_primes)}"]
    for index, matrix in enumerate(case.expected_associated_primes):
        label = f"A{index}"
        lines.append(f"MATRIX|{label}|{matrix.rows}|{matrix.columns}")
        for row, values in enumerate(matrix.entries):
            for column, polynomial in enumerate(values):
                polynomial_label = f"{label}/{row}/{column}"
                lines.append(f"POLY|{polynomial_label}")
                for numerator, denominator, exponent in polynomial.terms:
                    powers = ",".join(str(value) for value in exponent)
                    lines.append(
                        f"TERM|{polynomial_label}|{Fraction(numerator, denominator)}|{{{powers}}}"
                    )
    lines.extend(["METRIC|operation_cpu_s|0.001", f"END_{driver.PROTOCOL}"])
    return "\n".join(lines)


def test_module_associated_prime_lane_is_matched_and_typed() -> None:
    driver = _derived_driver()
    case = driver.CASES["module-associated-primes-q"]
    operation = "associated-primes"
    protocol = _module_associated_prime_protocol(driver, case)

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    parsed = driver.parse_protocol(protocol, case)
    admitted_hash, checks = driver.admitted_hash(case, operation, parsed)

    assert "associatedPrimes LeftRun0" in m2
    assert "return left.associated_primes()" in noether
    assert "result.verify()" in noether
    assert len(admitted_hash) == 64
    assert checks == ("hand-known-associated-prime-set", "typed-prime-admission")
    with pytest.raises(ValueError, match="returned 1 associated primes"):
        bad = driver.parse_protocol(protocol.replace("META|count|2", "META|count|1"), case)
        driver.admitted_hash(case, operation, bad)


def _module_primary_protocol(driver, components) -> str:
    lines = [driver.PROTOCOL, f"META|count|{len(components)}"]
    for index, matrix in enumerate(components):
        label = f"C{index}"
        lines.append(f"MATRIX|{label}|{matrix.rows}|{matrix.columns}")
        for row, values in enumerate(matrix.entries):
            for column, polynomial in enumerate(values):
                polynomial_label = f"{label}/{row}/{column}"
                lines.append(f"POLY|{polynomial_label}")
                for numerator, denominator, exponent in polynomial.terms:
                    powers = ",".join(str(value) for value in exponent)
                    lines.append(
                        f"TERM|{polynomial_label}|{Fraction(numerator, denominator)}|{{{powers}}}"
                    )
    lines.extend(["METRIC|operation_cpu_s|0.001", f"END_{driver.PROTOCOL}"])
    return "\n".join(lines)


def test_module_primary_decomposition_lane_admits_invariant_component_claim() -> None:
    driver = _derived_driver()
    case = driver.CASES["module-primary-decomposition-q"]
    operation = "primary-decomposition"
    linear = driver.polynomial(3, (1, {0: 1}), (1, {1: 1}))
    z = driver.polynomial(3, (1, {2: 1}))
    one = driver.polynomial(3, (1, {}))
    zero = driver.polynomial(3)
    minimal_component = driver.MatrixSpec(((linear, zero, zero, zero), (zero, linear, z, one)))
    embedded_component = driver.MatrixSpec(((linear, zero, zero, z), (zero, linear, z, zero)))
    protocol = _module_primary_protocol(driver, (minimal_component, embedded_component))

    m2 = driver.m2_script(case, operation, "warm", 1)
    noether = driver.noether_script(case, operation, "warm", 1)
    parsed = driver.parse_protocol(protocol, case)
    admitted_hash, checks = driver.admitted_hash(case, operation, parsed)

    assert "primaryDecomposition LeftRun0" in m2
    assert "presentation(LeftRun0/Qp)" in m2
    assert "return left.primary_decomposition()" in noether
    assert "result.verify()" in noether
    assert len(admitted_hash) == 64
    assert checks == (
        "hand-known-associated-prime-set",
        "singleton-quotient-associated-prime-admission",
        "exact-component-intersection",
        "irredundant-primary-components",
    )

    nonprimary = driver.MatrixSpec(((linear, zero, zero), (zero, linear, z)))
    bad = driver.parse_protocol(
        _module_primary_protocol(driver, (nonprimary, embedded_component)), case
    )
    with pytest.raises(ValueError, match="nonprimary"):
        driver.admitted_hash(case, operation, bad)


def _decomposition_protocol(driver, ideals) -> str:
    lines = [driver.PROTOCOL]
    if len(ideals) > 1 or ideals[0][0] != "I":
        lines.append(f"COMPONENTS|P|{len(ideals)}")
    for ideal_label, values in ideals:
        lines.append(f"IDEAL|{ideal_label}|{len(values)}")
        for index, polynomial in enumerate(values):
            label = f"{ideal_label}/{index}"
            lines.append(f"POLY|{label}")
            for numerator, denominator, exponent in polynomial.terms:
                powers = ",".join(str(value) for value in exponent)
                lines.append(f"TERM|{label}|{Fraction(numerator, denominator)}|{{{powers}}}")
    lines.extend(["METRIC|operation_cpu_s|0.001", f"END_{driver.PROTOCOL}"])
    return "\n".join(lines)


def test_decomposition_radical_lane_has_matched_inputs_and_exact_admission() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["monomial-fat-4x3-q"]

    m2 = driver.m2_script(case, "radical", "warm", 1)
    noether = driver.noether_script(case, "radical", "warm", 1)
    parsed = driver.parse_protocol(
        _decomposition_protocol(driver, (("I", case.expected_radical),)), case, "radical"
    )
    admitted, checks = driver.admitted_hash(case, "radical", parsed)

    assert len(driver.CASES) == 36
    assert {value.coefficient_context for value in driver.CASES.values()} == {
        "Q",
        "F_2",
        "F_101",
        "F_65521",
    }
    saturation = driver.CASES["saturation-component-split-f101"]
    assert saturation.operations == (
        "radical",
        "minimal-primes",
        "equidimensional-radical",
        "primary-decomposition",
    )
    assert saturation.expected_equidimensional_count == 2
    complete_intersection = driver.CASES["graph-complete-intersection-powers-q"]
    assert complete_intersection.operations == ("primary-decomposition",)
    assert complete_intersection.expected_component_count == 4
    assert complete_intersection.dimension == 3
    common_power = driver.CASES["common-factor-power-embedded-q"]
    assert common_power.operations == ("primary-decomposition",)
    assert common_power.expected_component_count == 3
    assert common_power.dimension == 3
    graph_monomial = driver.CASES["graph-monomial-embedded-q"]
    assert graph_monomial.operations == ("primary-decomposition",)
    assert graph_monomial.expected_component_count == 2
    assert graph_monomial.expected_equidimensional_count == 2
    assert graph_monomial.dimension == 2
    ehv = driver.CASES["ehv-embedded-associated-prime-q"]
    assert ehv.operations == (
        "associated-primes",
        "equidimensional-hull",
        "primary-decomposition",
    )
    assert ehv.expected_component_count == 2
    assert ehv.dimension == 2
    localized = driver.CASES["ehv-incomparable-embedded-q"]
    assert localized.operations == ("associated-primes", "primary-decomposition")
    assert localized.expected_component_count == 3
    assert localized.dimension == 2
    assert sum(len(value.operations) for value in driver.CASES.values()) == 113
    assert m2.index("PreBasis0=gens gb IRun0") < m2.index("Started0=cpuTime()")
    assert noether.index("source=build_ideal()") < noether.index("started_cpu=time.process_time()")
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "hand-known-radical",
        "exact-source-quotient-dimension",
    )

    wrong = driver.parse_protocol(
        _decomposition_protocol(driver, (("I", case.generators),)), case, "radical"
    )
    with pytest.raises(ValueError, match="wrong radical ideal"):
        driver.admitted_hash(case, "radical", wrong)


def test_positive_dimensional_radical_lane_uses_exact_dimension_admission() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["principal-hypersurface-q"]
    parsed = driver.parse_protocol(
        _decomposition_protocol(driver, (("I", case.expected_radical),)), case, "radical"
    )

    admitted, checks = driver.admitted_hash(case, "radical", parsed)

    assert case.dimension == 2
    assert case.quotient_dimension is None
    assert case.operations == driver.OPERATIONS
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "hand-known-radical",
        "exact-source-dimension",
    )


def test_ehv_associated_prime_lane_has_matched_ext_annihilator_admission() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["ehv-embedded-associated-prime-q"]
    x, y, z = (driver.variable(3, index) for index in range(3))
    linear = driver.add(x, y)
    parsed = driver.parse_protocol(
        _decomposition_protocol(driver, (("P/0", (linear,)), ("P/1", (linear, z)))),
        case,
        "associated-primes",
    )

    admitted, checks = driver.admitted_hash(case, "associated-primes", parsed)
    m2 = driver.m2_script(case, "associated-primes", "warm", 1)
    noether = driver.noether_script(case, "associated-primes", "warm", 1)

    assert "associatedPrimes IRun0" in m2
    assert "source.associated_primes()" in noether
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-ehv-ext-annihilator-and-prime-certificates",
        "exact-source-dimension",
    )


def test_equidimensional_hull_lane_has_matched_ext_annihilator_admission() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["ehv-embedded-associated-prime-q"]
    x, y, _ = (driver.variable(3, index) for index in range(3))
    linear = driver.add(x, y)
    parsed = driver.parse_protocol(
        _decomposition_protocol(driver, (("I", (linear,)),)),
        case,
        "equidimensional-hull",
    )

    admitted, checks = driver.admitted_hash(case, "equidimensional-hull", parsed)
    m2 = driver.m2_script(case, "equidimensional-hull", "warm", 1)
    noether = driver.noether_script(case, "equidimensional-hull", "warm", 1)

    assert "topComponents IRun0" in m2
    assert "source.equidimensional_hull()" in noether
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-ehv-top-dimensional-ext-annihilator",
        "exact-source-dimension",
    )


def test_decomposition_minimal_prime_lane_has_prime_and_intersection_admission() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["translated-fat-q"]
    x, y = driver.variable(2, 0), driver.variable(2, 1)
    components = (
        (driver.subtract(x, driver.constant(2, 1)), y),
        (driver.add(x, driver.constant(2, 2)), y),
    )
    protocol = _decomposition_protocol(
        driver, tuple((f"P/{index}", values) for index, values in enumerate(components))
    )

    m2 = driver.m2_script(case, "minimal-primes", "warm", 1)
    noether = driver.noether_script(case, "minimal-primes", "warm", 1)
    parsed = driver.parse_protocol(protocol, case, "minimal-primes")
    admitted, checks = driver.admitted_hash(case, "minimal-primes", parsed)

    assert "minimalPrimes IRun0" in m2
    assert "Forced0=apply(Result0,J->gens J)" in m2
    assert ".minimal_primes()" in noether
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-prime-certificates",
        "exact-source-quotient-dimension",
    )

    nonprime = driver.parse_protocol(
        _decomposition_protocol(driver, (("P/0", case.expected_radical),)),
        case,
        "minimal-primes",
    )
    with pytest.raises(ValueError, match="wrong number of minimal primes"):
        driver.admitted_hash(case, "minimal-primes", nonprime)


def test_positive_dimensional_minimal_prime_lane_uses_typed_certificates() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["principal-hypersurface-q"]
    x, y, z = (driver.variable(3, index) for index in range(3))
    components = (
        (driver.add(x, y),),
        (driver.subtract(x, z),),
        (driver.add(y, driver.multiply(driver.constant(3, 2), z)),),
    )
    parsed = driver.parse_protocol(
        _decomposition_protocol(
            driver, tuple((f"P/{index}", values) for index, values in enumerate(components))
        ),
        case,
        "minimal-primes",
    )

    admitted, checks = driver.admitted_hash(case, "minimal-primes", parsed)

    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-prime-certificates",
        "exact-source-dimension",
    )


def test_equidimensional_radical_lane_preserves_dimension_groups() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["mixed-dimensional-monomial-q"]
    x, y, z = (driver.variable(3, index) for index in range(3))
    parsed = driver.parse_protocol(
        _decomposition_protocol(driver, (("P/0", (x,)), ("P/1", (y, z)))),
        case,
        "equidimensional-radical",
    )

    admitted, checks = driver.admitted_hash(case, "equidimensional-radical", parsed)

    assert case.expected_component_count == 2
    assert case.expected_equidimensional_count == 2
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-equidimensional-prime-grouping",
        "exact-source-dimension",
    )


def test_primary_lane_admits_components_and_compares_associated_primes() -> None:
    driver = _decomposition_driver()
    case = driver.CASES["mixed-dimensional-monomial-q"]
    x, y, z = (driver.variable(3, index) for index in range(3))
    components = (
        (x,),
        (driver.power(x, 3), driver.power(z, 4)),
        (y, driver.power(z, 4)),
    )
    parsed = driver.parse_protocol(
        _decomposition_protocol(
            driver, tuple((f"P/{index}", values) for index, values in enumerate(components))
        ),
        case,
        "primary-decomposition",
    )

    m2 = driver.m2_script(case, "primary-decomposition", "warm", 1)
    noether = driver.noether_script(case, "primary-decomposition", "warm", 1)
    admitted, checks = driver.admitted_hash(case, "primary-decomposition", parsed)

    assert "primaryDecomposition IRun0" in m2
    assert ".primary_decomposition()" in noether
    assert len(admitted) == 64
    assert checks == (
        "canonical-noether-ideal-admission",
        "typed-primary-admission-and-associated-primes",
        "exact-source-dimension",
    )

    nonprimary = driver.parse_protocol(
        _decomposition_protocol(driver, (("P/0", case.generators),)),
        case,
        "primary-decomposition",
    )
    with pytest.raises(ValueError, match="not admitted as primary"):
        driver.admitted_hash(case, "primary-decomposition", nonprimary)
