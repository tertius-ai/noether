// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Compile-time audit of private implementation routes into accepted facade calls.
//!
//! This is deliberately test-only.  Users choose mathematical operations and
//! coefficient contexts; they do not choose storage engines.  The table makes
//! the private bookkeeping reviewable without turning it into another public
//! planning or dispatch API.

use std::collections::BTreeSet;

use crate::{accepted_operations, COMPILED_BACKENDS};

struct DispatchRoute {
    backend: &'static str,
    operations: &'static [&'static str],
    condition: &'static str,
    correctness_floor: &'static str,
    differential_test: &'static str,
}

const DENSE_RING_OPERATIONS: &[&str] = &[
    "linalg.Matrix.__add__",
    "linalg.Matrix.__sub__",
    "linalg.Matrix.__neg__",
    "linalg.Matrix.__mul__",
    "linalg.Matrix.__rmul__",
    "linalg.Matrix.__matmul__",
    "linalg.Matrix.matmul",
    "linalg.Matrix.transpose",
];

const DENSE_FIELD_ELIMINATION_OPERATIONS: &[&str] = &["linalg.Matrix.rank", "linalg.Matrix.rref"];

const DENSE_FIELD_SOLUTION_OPERATIONS: &[&str] = &[
    "linalg.Matrix.solve",
    "linalg.Matrix.kernel",
    "linalg.Matrix.image",
    "linalg.Matrix.pluq",
];

const DENSE_FIELD_INVARIANT_OPERATIONS: &[&str] = &["linalg.Matrix.det", "linalg.Matrix.charpoly"];

const SPARSE_RING_OPERATIONS: &[&str] = &[
    "linalg.SparseMatrix.__add__",
    "linalg.SparseMatrix.__sub__",
    "linalg.SparseMatrix.__neg__",
    "linalg.SparseMatrix.__mul__",
    "linalg.SparseMatrix.__rmul__",
    "linalg.SparseMatrix.__matmul__",
    "linalg.SparseMatrix.transpose",
];

const POLYNOMIAL_RING_OPERATIONS: &[&str] = &[
    "commalg.Polynomial.__add__",
    "commalg.Polynomial.__sub__",
    "commalg.Polynomial.__neg__",
    "commalg.Polynomial.__mul__",
];

const SERIES_RING_OPERATIONS: &[&str] = &[
    "commalg.PowerSeries.__add__",
    "commalg.PowerSeries.__sub__",
    "commalg.PowerSeries.__neg__",
    "commalg.PowerSeries.__mul__",
    "commalg.PowerSeries.scalar_mul",
];

const FIELD_GROEBNER_OPERATIONS: &[&str] = &[
    "commalg.Ideal.__contains__",
    "commalg.Ideal.change_order",
    "commalg.Ideal.groebner_basis",
    "commalg.Ideal.normal_form",
    "commalg.Ideal.rational_univariate_representation",
    "commalg.Ideal.verify",
    "commalg.PolynomialRing.ideal",
];

const INTEGER_GROEBNER_OPERATIONS: &[&str] = &[
    "commalg.PolynomialRing.strong_groebner_basis",
    "commalg.StrongGroebnerBasis.__contains__",
    "commalg.StrongGroebnerBasis.normal_form",
    "commalg.StrongGroebnerBasis.verify",
];

const CERTIFIED_REAL_SOLVING_OPERATIONS: &[&str] = &[
    "commalg.Polynomial.real_roots",
    "commalg.RationalUnivariateRepresentation.real_solutions",
    "commalg.RealRootIsolation.ball",
    "commalg.RealRootIsolation.refine",
];

#[cfg(feature = "distributed")]
const DISTRIBUTED_F2_OPERATIONS: &[&str] = &[
    "linalg.DistributedMatrix.__add__",
    "linalg.DistributedMatrix.__matmul__",
    "linalg.DistributedMatrix.__sub__",
    "linalg.DistributedMatrix.echelon",
    "linalg.DistributedMatrix.matmul",
    "linalg.DistributedMatrix.nullspace",
    "linalg.DistributedMatrix.ple_rank",
    "linalg.DistributedMatrix.pluq",
    "linalg.DistributedMatrix.rank",
    "linalg.DistributedMatrix.rref",
    "linalg.DistributedMatrix.solve_left",
];

const DISPATCH_ROUTES: &[DispatchRoute] = &[
    DispatchRoute {
        backend: "flint",
        operations: DENSE_RING_OPERATIONS,
        condition: "dense matrices over Z or Q",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::{integer,rational}_flint_operations_match_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: &[
            "linalg.Matrix.rank",
            "linalg.Matrix.rref",
            "linalg.Matrix.det",
            "linalg.Matrix.charpoly",
        ],
        condition: "whole-matrix elimination and invariants over Q",
        correctness_floor: "generic field elimination and Berkowitz characteristic polynomial",
        differential_test: "linalg::matrix::tests::rational_flint_operations_match_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: &[
            "homalg.ChainComplex.graded_cohomology_invariants",
            "homalg.ChainComplex.graded_homology_invariants",
            "linalg.Matrix.charpoly",
            "linalg.Matrix.hermite_normal_form",
            "linalg.Matrix.smith_normal_form",
            "linalg.Matrix.smith_normal_form_with_transforms",
        ],
        condition: "dense integer matrices, with no-transform normal forms unless requested",
        correctness_floor: "generic determinants/characteristic polynomials plus verified normal-form witnesses",
        differential_test: "linalg::matrix::tests::integer_flint_charpoly_matches_berkowitz_through_dimension_five",
    },
    DispatchRoute {
        backend: "m4rst",
        operations: DENSE_RING_OPERATIONS,
        condition: "dense matrices over Z/2Z or a prime field of characteristic two",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::residue_flint_storage_matches_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "m4rst",
        operations: &[
            "linalg.Matrix.rank",
            "linalg.Matrix.rref",
            "linalg.Matrix.det",
            "linalg.Matrix.solve",
            "linalg.Matrix.kernel",
            "linalg.Matrix.image",
            "linalg.Matrix.pluq",
        ],
        condition: "packed rank, RREF, determinant, solve, kernel, image and PLUQ over F_2",
        correctness_floor: "generic field elimination and Berkowitz characteristic polynomial",
        differential_test: "tests/field_linear_algebra.rs::packed_binary_fields_solve_and_find_kernels_without_changing_the_surface",
    },
    #[cfg(feature = "distributed")]
    DispatchRoute {
        backend: "m4rst-distributed",
        operations: DISTRIBUTED_F2_OPERATIONS,
        condition: "explicit MPI-distributed dense matrices over F_2",
        correctness_floor: "the local packed m4rst operations after an explicitly bounded collect",
        differential_test: "examples/{operations,factorization}_acceptance.rs under one, two and four MPI ranks",
    },
    DispatchRoute {
        backend: "pwvla",
        operations: DENSE_RING_OPERATIONS,
        condition: "dense matrices over word-sized Z/2^N with N > 1",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::residue_flint_storage_matches_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: &[
            "linalg.Matrix.howell_normal_form",
            "linalg.Matrix.det",
            "linalg.Matrix.charpoly",
        ],
        condition: "word and arbitrary-size residue rings; power-of-two determinant/charpoly convert terminally and Howell changes authoritative storage once",
        correctness_floor: "verified Howell form and generic Berkowitz characteristic polynomial",
        differential_test: "linalg::matrix::tests::residue_flint_storage_matches_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: DENSE_RING_OPERATIONS,
        condition: "dense matrices over word-sized nonbinary or arbitrary-size residue rings",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::residue_flint_storage_matches_the_generic_ring_floor",
    },
    DispatchRoute {
        backend: "m4rste",
        operations: DENSE_RING_OPERATIONS,
        condition: "dense matrices over binary extension fields of degree 2 through 16",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::finite_field_flint_storage_matches_the_generic_field_floor",
    },
    DispatchRoute {
        backend: "m4rste",
        operations: &[
            "linalg.Matrix.rank",
            "linalg.Matrix.rref",
            "linalg.Matrix.solve",
            "linalg.Matrix.kernel",
            "linalg.Matrix.image",
            "linalg.Matrix.pluq",
        ],
        condition: "packed elimination, solve, kernel, image and PLUQ over binary extension fields of degree 2 through 16",
        correctness_floor: "generic field elimination",
        differential_test: "tests/field_linear_algebra.rs::packed_binary_fields_solve_and_find_kernels_without_changing_the_surface",
    },
    DispatchRoute {
        backend: "flint",
        operations: DENSE_FIELD_INVARIANT_OPERATIONS,
        condition: "finite-field invariants; binary-extension storage converts terminally once to fq_default",
        correctness_floor: "generic field elimination and Berkowitz characteristic polynomial",
        differential_test: "linalg::matrix::tests::finite_field_flint_storage_matches_the_generic_field_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: DENSE_FIELD_ELIMINATION_OPERATIONS,
        condition: "prime fields other than F_2 and nonbinary finite extension fields",
        correctness_floor: "generic field elimination",
        differential_test: "linalg::matrix::tests::finite_field_flint_storage_matches_the_generic_field_floor",
    },
    DispatchRoute {
        backend: "flint",
        operations: DENSE_RING_OPERATIONS,
        condition: "nonbinary finite extension fields, fixed-cap p-adics, and real or complex balls",
        correctness_floor: "the context-generic dense scalar loops",
        differential_test: "linalg::matrix::tests::{finite_field,padic,ball}_flint_storage_matches_the_generic_*_floor",
    },
    DispatchRoute {
        backend: "native",
        operations: DENSE_FIELD_SOLUTION_OPERATIONS,
        condition: "generic field floor when no retained whole-matrix solve, kernel, image or PLUQ route exists",
        correctness_floor: "deterministic scalar field elimination",
        differential_test: "tests/field_linear_algebra.rs::{rational_elimination_is_exact_and_hand_checkable,the_same_surface_works_over_f9,the_same_surface_works_over_a_number_field}",
    },
    DispatchRoute {
        backend: "native",
        operations: &[
            "linalg.Vector.__add__",
            "linalg.Vector.__sub__",
            "linalg.Vector.__neg__",
            "linalg.Vector.__mul__",
            "linalg.Vector.__rmul__",
            "linalg.Vector.dot",
        ],
        condition: "vectors delegate whole-object arithmetic to the corresponding Matrix<R> route",
        correctness_floor: "the context-generic Matrix<R> route",
        differential_test: "tests/vectors.rs::specialized_context_vectors_reuse_matrix_arithmetic",
    },
    DispatchRoute {
        backend: "mzs",
        operations: SPARSE_RING_OPERATIONS,
        condition: "persistent sparse F_2 storage and arithmetic",
        correctness_floor: "generic canonical CSR arithmetic",
        differential_test: "linalg::sparse_matrix::tests::binary_sparse_storage_stays_in_mzs_and_matches_the_generic_floor",
    },
    DispatchRoute {
        backend: "sparla",
        operations: SPARSE_RING_OPERATIONS,
        condition: "integer or word-prime-field CSR arithmetic, including add/sub/neg/scaling",
        correctness_floor: "generic canonical CSR arithmetic",
        differential_test: "linalg::sparse_matrix::tests::{integer,prime_field}_storage_and_products_stay_in_sparla",
    },
    DispatchRoute {
        backend: "sparla",
        operations: &[
            "topology.SimplicialComplex.betti_number",
            "topology.SimplicialComplex.graded_homology_invariants",
            "topology.SimplicialComplex.homology",
            "topology.SimplicialComplex.homology_invariants",
            "topology.SimplicialSet.betti_number",
            "topology.SimplicialSet.homology",
            "topology.SimplicialSet.homology_invariants",
        ],
        condition: "rational Betti numbers and integral homology invariants of canonical sparse simplicial boundary operators",
        correctness_floor: "dense Matrix<R> chain-complex homology and its independent verifier",
        differential_test: "topology::tests::{betti_numbers_match_full_homology_on_spheres,integral_homology_names_projective_plane_torsion} and properties::integral_homology_invariants_match_source_retaining_results",
    },
    DispatchRoute {
        backend: "sparb",
        operations: SPARSE_RING_OPERATIONS,
        condition: "real or complex ball CSR",
        correctness_floor: "generic canonical CSR arithmetic",
        differential_test: "linalg::sparse_matrix::tests::ball_storage_and_arithmetic_stay_in_sparb",
    },
    DispatchRoute {
        backend: "flint",
        operations: POLYNOMIAL_RING_OPERATIONS,
        condition: "polynomials over Z, Q, residue rings, or finite fields with machine-size exponents",
        correctness_floor: "generic sparse multivariate polynomial arithmetic",
        differential_test: "commalg::polynomial::storage_tests::supported_exact_contexts_keep_flint_storage_across_arithmetic",
    },
    DispatchRoute {
        backend: "flint",
        operations: SERIES_RING_OPERATIONS,
        condition: "supported exact univariate fixed-truncation power series",
        correctness_floor: "generic sparse truncation-aware arithmetic, always used for weighted multivariate series",
        differential_test: "commalg::series::storage_tests::supported_exact_contexts_keep_univariate_series_in_flint",
    },
    DispatchRoute {
        backend: "poly-core",
        operations: FIELD_GROEBNER_OPERATIONS,
        condition: "exact field polynomials whose machine-size monomials enter a Groebner computation",
        correctness_floor: "the existing bounded sparse Buchberger implementation over the public coefficient context",
        differential_test: "tests/polynomial_ideals.rs::{rational_linear_ideal_has_a_canonical_reduced_basis,extension_fields_use_the_same_verified_ideal_floor,binary_field_kernel_lanes_preserve_verified_facade_semantics}",
    },
    DispatchRoute {
        backend: "poly-core",
        operations: INTEGER_GROEBNER_OPERATIONS,
        condition: "integer polynomials with machine-size monomials entering a strong-basis computation",
        correctness_floor: "exact S/G-pair and original-generator verification over public polynomials",
        differential_test: "tests/integer_groebner.rs::integer_strong_basis_has_verified_membership_and_round_trips",
    },
    DispatchRoute {
        backend: "groebner-kernel",
        operations: FIELD_GROEBNER_OPERATIONS,
        condition: "exact field ideals with machine-size monomials; F_2, word-prime, binary-extension, modular-Q, and generic lanes dispatch privately",
        correctness_floor: "the bounded public sparse implementation, followed by exact reduced-basis and original-generator verification",
        differential_test: "tests/polynomial_ideals.rs::{rational_linear_ideal_has_a_canonical_reduced_basis,modular_rational_lane_returns_a_canonical_exact_ideal,extension_fields_use_the_same_verified_ideal_floor,binary_field_kernel_lanes_preserve_verified_facade_semantics}",
    },
    DispatchRoute {
        backend: "groebner-kernel",
        operations: INTEGER_GROEBNER_OPERATIONS,
        condition: "strong Groebner bases over Z with machine-size monomials",
        correctness_floor: "exact S/G-pair and original-generator verification over public polynomials",
        differential_test: "tests/integer_groebner.rs::integer_strong_basis_has_verified_membership_and_round_trips",
    },
    DispatchRoute {
        backend: "flint",
        operations: CERTIFIED_REAL_SOLVING_OPERATIONS,
        condition: "exact rational Sturm isolation and RUR projection, with Arb used for rigorous refined enclosures",
        correctness_floor: "exact rational Sturm counts, multiplicity chains, and deterministic RUR recomputation",
        differential_test: "commalg::real_roots::tests::isolates_simple_and_multiple_roots_exactly and commalg::real_solutions::tests::rur_projects_two_real_points_to_certified_boxes",
    },
    DispatchRoute {
        backend: "convex-hull",
        operations: &["geometry.Polytope.from_vertices"],
        condition: "exact rational V/H polytope conversion",
        correctness_floor: "exact facade verification rejects inconsistent incidence data",
        differential_test: "geometry::tests::exact_polytopes_use_flint_rationals_and_recompute_json",
    },
    DispatchRoute {
        backend: "geom-core",
        operations: &["geometry.PointCloud.delaunay"],
        condition: "Euclidean Delaunay triangulation with filtered exact predicates",
        correctness_floor: "exact predicate fallback inside geom-core",
        differential_test: "geometry::tests::triangulations_are_typed_facade_results",
    },
    DispatchRoute {
        backend: "graph-core",
        operations: &["combin.Graph.bfs"],
        condition: "canonical graph storage and traversal",
        correctness_floor: "typed traversal witnesses are checked against the source graph",
        differential_test: "tests/facade.rs::graph_and_permutation_kernels_compose_through_the_facade",
    },
    DispatchRoute {
        backend: "lattice-enum",
        operations: &["linalg.Lattice.shortest_vectors"],
        condition: "positive-definite exact lattice enumeration",
        correctness_floor: "exact norm and completeness verification",
        differential_test: "linalg::lattice::tests::integral_gram_dispatches_to_verified_enumeration",
    },
    DispatchRoute {
        backend: "perm-core",
        operations: &["groups.PermutationGroup.order"],
        condition: "permutation-group values and stabilizer-chain algorithms",
        correctness_floor: "checked permutation action and stabilizer-chain witnesses",
        differential_test: "tests/permutation_groups.rs::permutation_group_is_the_finite_group_facade",
    },
];

struct MathematicalPlaybook {
    operation: &'static str,
    strategies: &'static [&'static str],
    precondition_test: &'static str,
    strategy_test: &'static str,
    witness_test: &'static str,
}

struct KernelBenchmark {
    backend: &'static str,
    command: &'static str,
    release_case: &'static str,
}

// Every compiled route has a pinned, repeatable performance owner.  Commands
// are intentionally executable documentation: release measurements wrap each
// one in `scripts/bench-pin.sh` and keep the resulting numbers outside this
// living source tree.
const KERNEL_BENCHMARKS: &[KernelBenchmark] = &[
    KernelBenchmark {
        backend: "flint",
        command: "cargo run --release -p pwvla --features flint --example bench_z2n_flint",
        release_case: "dense modular arithmetic against the shared FLINT stack",
    },
    KernelBenchmark {
        backend: "m4rst",
        command: "cargo bench -p m4rst --bench matmul",
        release_case: "dense F_2 matrix multiplication",
    },
    #[cfg(feature = "distributed")]
    KernelBenchmark {
        backend: "m4rst-distributed",
        command: "cargo run --release -p m4rst-distributed --example benchmark -- 4096 3",
        release_case:
            "MPI-distributed dense F_2 matrix multiplication under the selected rank count",
    },
    KernelBenchmark {
        backend: "m4rste",
        command: "cargo bench -p m4rste --bench m4rste",
        release_case: "dense binary-extension-field matrix arithmetic",
    },
    KernelBenchmark {
        backend: "mzs",
        command: "cargo bench -p mzs --bench matmul",
        release_case: "sparse F_2 matrix multiplication",
    },
    KernelBenchmark {
        backend: "pwvla",
        command: "cargo run --release -p pwvla --features flint --example bench_z2n_flint",
        release_case: "dense power-of-two residue-ring matrix multiplication",
    },
    KernelBenchmark {
        backend: "sparla",
        command: "cargo bench -p sparla --bench matmul",
        release_case: "exact sparse integer and prime-field matrix multiplication",
    },
    KernelBenchmark {
        backend: "sparb",
        command: "cargo bench -p sparb --bench spmv",
        release_case: "rigorous sparse real-ball matrix-vector multiplication",
    },
    KernelBenchmark {
        backend: "convex-hull",
        command: "cargo bench -p noether --bench facade_kernels convex-hull",
        release_case: "exact rational convex hull",
    },
    KernelBenchmark {
        backend: "geom-core",
        command: "cargo bench -p noether --bench facade_kernels geom-core",
        release_case: "filtered-predicate Delaunay triangulation",
    },
    KernelBenchmark {
        backend: "graph-core",
        command: "cargo bench -p noether --bench facade_kernels graph-core",
        release_case: "canonical graph traversal",
    },
    KernelBenchmark {
        backend: "lattice-enum",
        command: "cargo bench -p noether --bench facade_kernels lattice-enum",
        release_case: "exact shortest-vector enumeration",
    },
    KernelBenchmark {
        backend: "perm-core",
        command: "cargo bench -p noether --bench facade_kernels perm-core",
        release_case: "stabilizer-chain group order",
    },
    KernelBenchmark {
        backend: "poly-core",
        command: "cargo bench -p poly-core --bench polynomial",
        release_case: "packed-monomial sparse multiplication, F_2 support arithmetic, and packed truncated series",
    },
    KernelBenchmark {
        backend: "groebner-kernel",
        command: "cargo bench -p groebner-kernel --bench groebner",
        release_case: "verified F4 Groebner bases over word-prime fields",
    },
];

// Routine storage selection above is dispatch bookkeeping.  A playbook earns
// this separate table only when it chooses between mathematically different
// strategies with different proof obligations.
const MATHEMATICAL_PLAYBOOKS: &[MathematicalPlaybook] = &[
    MathematicalPlaybook {
        operation: "combin.Poset.graded_homology_invariants",
        strategies: &["beat_core", "core_sparse_smith"],
        precondition_test:
            "combin::poset::tests::beat_point_reduction_is_replayable_and_dual_invariant",
        strategy_test:
            "combin::poset::tests::core_reduction_preserves_integral_homology_and_verifies",
        witness_test:
            "combin::poset::tests::core_reduction_preserves_integral_homology_and_verifies",
    },
    MathematicalPlaybook {
        operation: "homalg.ChainComplex.graded_homology_invariants",
        strategies: &["direct_smith", "algebraic_morse_then_smith"],
        precondition_test:
            "homalg::reduction::tests::plus_or_minus_one_is_cancelled_over_unrelated_rings",
        strategy_test:
            "homalg::tests::algebraic_reduction_playbook_uses_the_benchmarked_cutover",
        witness_test: "homalg::tests::algebraic_reduction_preserves_graded_integral_invariants",
    },
    MathematicalPlaybook {
        operation: "linalg.IntegralLineConfiguration.complete_minimum",
        strategies: &["basis_gram", "fixed_rank_cutting_plane"],
        precondition_test:
            "linalg::line_realization::tests::minimum_playbooks_enforce_rank_preconditions",
        strategy_test:
            "linalg::line_realization::tests::fixed_rank_completion_accepts_an_already_complete_shell",
        witness_test:
            "linalg::line_realization::tests::fixed_rank_completion_rejects_inconsistent_equal_lines",
    },
    MathematicalPlaybook {
        operation: "linalg.IntegralLineConfiguration.realize_minimum",
        strategies: &["basis_gram", "cutting_plane"],
        precondition_test:
            "linalg::line_realization::tests::minimum_playbooks_enforce_rank_preconditions",
        strategy_test:
            "linalg::line_realization::tests::realizes_coordinate_axes_with_complete_enumeration_evidence",
        witness_test: "linalg::line_realization::tests::realizes_the_a2_minimum_shell",
    },
    MathematicalPlaybook {
        operation: "linalg.Matrix.smith_normal_form",
        strategies: &["exact", "smooth_local"],
        precondition_test:
            "tests/integer_normal_forms.rs::smooth_smith_fails_closed_at_invalid_bounds_and_word_precision",
        strategy_test:
            "tests/integer_normal_forms.rs::smooth_local_strategy_records_its_exact_assumption_and_prime_checks",
        witness_test:
            "tests/integer_normal_forms.rs::explicit_transform_result_verifies_both_unimodular_witnesses",
    },
    MathematicalPlaybook {
        operation: "topology.SimplicialComplex.graded_homology_invariants",
        strategies: &["ordinary_sparse_boundaries", "simplicial_morse_model"],
        precondition_test:
            "topology::tests::simplicial_morse_models_are_verified_and_preserve_homology",
        strategy_test:
            "topology::tests::simplicial_homology_playbook_uses_the_benchmarked_cutover",
        witness_test:
            "topology::tests::simplicial_morse_models_are_verified_and_preserve_homology",
    },
];

#[test]
fn every_dispatch_route_names_accepted_operations_and_a_checked_floor() {
    let accepted = accepted_operations();
    for route in DISPATCH_ROUTES {
        assert!(!route.condition.is_empty());
        assert!(!route.correctness_floor.is_empty());
        assert!(!route.differential_test.is_empty());
        assert!(!route.operations.is_empty());
        for operation in route.operations {
            assert!(
                accepted.binary_search(operation).is_ok(),
                "{} route has no accepted facade caller {operation}",
                route.backend,
            );
        }
    }
}

#[test]
fn every_compiled_backend_has_a_real_facade_route() {
    let routed = DISPATCH_ROUTES
        .iter()
        .map(|route| route.backend)
        .filter(|backend| *backend != "native")
        .collect::<BTreeSet<_>>();
    let compiled = COMPILED_BACKENDS.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(compiled, routed);
}

#[test]
fn every_compiled_backend_has_a_pinned_release_benchmark() {
    let benchmarks = KERNEL_BENCHMARKS
        .iter()
        .map(|benchmark| benchmark.backend)
        .collect::<BTreeSet<_>>();
    let compiled = COMPILED_BACKENDS.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(compiled, benchmarks);
    for benchmark in KERNEL_BENCHMARKS {
        assert!(benchmark.command.starts_with("cargo "));
        assert!(!benchmark.release_case.is_empty());
    }
}

#[test]
fn only_genuine_multi_strategy_mathematics_is_a_playbook() {
    let accepted = accepted_operations();
    assert_eq!(MATHEMATICAL_PLAYBOOKS.len(), 6);
    for playbook in MATHEMATICAL_PLAYBOOKS {
        assert!(accepted.binary_search(&playbook.operation).is_ok());
        assert!(playbook.strategies.len() > 1);
        assert!(!playbook.precondition_test.is_empty());
        assert!(!playbook.strategy_test.is_empty());
        assert!(!playbook.witness_test.is_empty());
    }
}
