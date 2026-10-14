// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::groups::{ConjugacyClass, FiniteGroup, GroupError, Permutation, PermutationGroup};
use noether::SemanticType;

fn symmetric_group_3() -> PermutationGroup {
    let transposition = Permutation::from_images(&[1, 0, 2]).unwrap();
    let three_cycle = Permutation::from_images(&[1, 2, 0]).unwrap();
    PermutationGroup::from_generators(3, &[transposition, three_cycle]).unwrap()
}

#[test]
fn permutation_group_is_the_finite_group_facade() {
    let group = symmetric_group_3();
    let transposition = Permutation::from_images(&[1, 0, 2]).unwrap();

    assert_eq!(FiniteGroup::order(&group).to_string(), "6");
    assert!(FiniteGroup::contains(&group, &transposition).unwrap());
    assert_eq!(group.orbit(0).unwrap(), vec![0, 1, 2]);
    assert_eq!(
        group
            .pointwise_stabilizer(&[0])
            .unwrap()
            .order()
            .to_decimal(),
        "2"
    );

    let inverse = FiniteGroup::inverse(&group, &transposition).unwrap();
    assert_eq!(
        FiniteGroup::multiply(&group, &transposition, &inverse).unwrap(),
        FiniteGroup::identity(&group)
    );
}

#[test]
fn checked_group_operations_reject_nonmembers_and_wrong_degrees() {
    let group =
        PermutationGroup::from_generators(3, &[Permutation::from_images(&[1, 2, 0]).unwrap()])
            .unwrap();
    let transposition = Permutation::from_images(&[1, 0, 2]).unwrap();
    let wrong_degree = Permutation::identity(4);

    assert_eq!(
        FiniteGroup::inverse(&group, &transposition),
        Err(GroupError::ElementNotInGroup)
    );
    assert_eq!(
        FiniteGroup::contains(&group, &wrong_degree),
        Err(GroupError::DegreeMismatch {
            got: 4,
            expected: 3,
        })
    );
}

#[test]
fn public_group_values_have_stable_semantic_descriptions() {
    let group = symmetric_group_3();
    let permutation = Permutation::identity(3);

    assert_eq!(Permutation::TYPE_ID, "noether/groups/Permutation@1");
    assert_eq!(
        PermutationGroup::TYPE_ID,
        "noether/groups/PermutationGroup@1"
    );
    assert_eq!(
        PermutationGroup::describe().python_path,
        Some("noether.groups.PermutationGroup")
    );
    assert_eq!(
        group.canonical_bytes().unwrap(),
        group.canonical_bytes().unwrap()
    );
    assert_eq!(
        permutation.canonical_bytes().unwrap(),
        permutation.canonical_bytes().unwrap()
    );
}

#[test]
fn conjugacy_classes_are_kernel_values_with_noether_identity() {
    let classes = symmetric_group_3().conjugacy_classes().unwrap();
    let mut sizes: Vec<_> = classes
        .iter()
        .map(|class| class.size().to_decimal())
        .collect();
    sizes.sort();

    assert_eq!(sizes, vec!["1", "2", "3"]);
    assert_eq!(ConjugacyClass::TYPE_ID, "noether/groups/ConjugacyClass@1");
    for class in classes {
        assert_eq!(
            class.centralizer().unwrap().order().mul(&class.size()),
            symmetric_group_3().order()
        );
        class.verify().unwrap();
        assert!(!class.canonical_bytes().unwrap().is_empty());
    }
}
