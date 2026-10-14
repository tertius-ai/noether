// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::category::SteinerCategory;
use noether::combin::{CanonicalLabel, Graph, GraphSymmetry, GraphSymmetryError, Tree};
use noether::geometry::{PointCloud, Polytope, Triangulation};
use noether::groups::{Permutation, PermutationGroup};
#[allow(unused_imports)]
use noether::{category, commalg, homalg, linalg, numthy, topology};
use noether::{
    CheckpointableEnumeration, CommutativeRing, EuclideanDomain, Field, FiniteContext,
    IntegralDomain, NoetherError, PrecisionContext, PrincipalIdealDomain, Ring, SemanticType,
    UnsupportedOperationError, ValuedRing,
};

#[test]
fn version_matches_workspace_package() {
    assert_eq!(noether::VERSION, env!("CARGO_PKG_VERSION"));
}

#[test]
fn unsupported_operations_keep_machine_readable_identifiers() {
    let error =
        UnsupportedOperationError::new("noether/linalg/Matrix/svd@1", "noether/root/IntegerRing@1")
            .with_supported_contexts([
                "noether/root/RealBallField@1",
                "noether/root/ComplexBallField@1",
            ]);
    assert_eq!(error.operation, "noether/linalg/Matrix/svd@1");
    assert_eq!(error.context, "noether/root/IntegerRing@1");
    assert_eq!(
        error.supported_contexts,
        [
            "noether/root/RealBallField@1",
            "noether/root/ComplexBallField@1"
        ]
    );
    assert_eq!(
        error.to_string(),
        "operation noether/linalg/Matrix/svd@1 is unsupported for context noether/root/IntegerRing@1; supported contexts: noether/root/RealBallField@1, noether/root/ComplexBallField@1"
    );
    assert!(matches!(
        NoetherError::from(error),
        NoetherError::UnsupportedOperation(_)
    ));
}

#[test]
fn self_contained_kernel_types_have_no_facade_copy() {
    fn graph_identity(value: graph_core::Graph) -> Graph {
        value
    }
    fn permutation_identity(value: perm_core::Perm) -> Permutation {
        value
    }
    fn group_identity(value: perm_core::PermGroup) -> PermutationGroup {
        value
    }

    let graph = graph_core::Graph::from_edges(1, &[], false).expect("valid graph");
    assert_eq!(graph_identity(graph).order(), 1);

    let permutation = perm_core::Perm::identity(3);
    let group = perm_core::PermGroup::from_generators(3, std::slice::from_ref(&permutation))
        .expect("valid permutation group");
    assert!(permutation_identity(permutation).is_identity());
    assert_eq!(group_identity(group).degree(), 3);
}

#[test]
fn all_nine_domain_modules_are_public() {
    let _ = core::any::type_name::<SteinerCategory>();
    let _ = core::any::type_name::<CanonicalLabel>();
    let _ = core::any::type_name::<Graph>();
    let _ = core::any::type_name::<GraphSymmetryError>();
    let _ = core::any::type_name::<Tree>();
    let _ = core::any::type_name::<Permutation>();
    let _ = core::any::type_name::<PermutationGroup>();
    let _ = core::any::type_name::<PointCloud>();
    let _ = core::any::type_name::<Triangulation>();
    let _ = core::any::type_name::<Polytope>();
}

#[test]
fn graph_and_permutation_kernels_compose_through_the_facade() {
    let graph = Graph::from_unweighted(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], false).unwrap();
    let label = graph.canonical_label().unwrap();

    assert!(label.verify(&graph));
    assert_eq!(label.automorphism_group().degree(), graph.order());
    assert_eq!(label.automorphism_group().order().to_decimal(), "8");
}

#[allow(dead_code)]
fn root_trait_contracts_compile<
    S: SemanticType,
    R: Ring
        + CommutativeRing
        + IntegralDomain
        + Field
        + EuclideanDomain
        + PrincipalIdealDomain
        + PrecisionContext
        + ValuedRing
        + FiniteContext,
    E: CheckpointableEnumeration,
>() {
}
