// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Public partition, tableau, and algebraic-combinatorics contracts.

use noether::combin::{
    CombinError, Composition, KashiwaraCrystal, Partition, SymmetricBasis, SymmetricFunction,
    Tableau, MAX_PARTITION_CELLS,
};
use noether::{Integer, Rational, SemanticType};

#[test]
fn partition_is_a_canonical_public_value() {
    let value = Partition::new(vec![4, 2, 1]).unwrap();

    assert_eq!(Partition::TYPE_ID, "noether/combin/Partition@1");
    assert_eq!(value.parts(), &[4, 2, 1]);
    assert_eq!(value.size(), 7);
    assert_eq!(value.conjugate().parts(), &[3, 2, 1, 1]);
    assert_eq!(
        serde_json::from_str::<Partition>(&serde_json::to_string(&value).unwrap()).unwrap(),
        value
    );
}

#[test]
fn partition_hooks_are_row_major_and_total_on_the_accepted_type() {
    let value = Partition::new(vec![3, 2]).unwrap();

    assert_eq!(value.hook_lengths(), vec![4, 3, 1, 2, 1]);
    assert!(matches!(
        Partition::new(vec![MAX_PARTITION_CELLS + 1]),
        Err(CombinError::PartitionTooLarge { .. })
    ));
}

#[test]
fn composition_keeps_order_and_derives_size() {
    let value = Composition::new(vec![1, 3, 1]).unwrap();

    assert_eq!(Composition::TYPE_ID, "noether/combin/Composition@1");
    assert_eq!(value.parts(), &[1, 3, 1]);
    assert_eq!(value.size(), 5);
    assert!(matches!(
        Composition::new(vec![1, 0, 2]),
        Err(CombinError::ZeroPart { index: 1 })
    ));
}

#[test]
fn tableau_owns_semistandard_validation_and_forward_rsk() {
    let rows = vec![
        vec![Integer::from(-3), Integer::from(10)],
        vec![Integer::from(2)],
    ];
    let value = Tableau::new(rows).unwrap();

    assert_eq!(Tableau::TYPE_ID, "noether/combin/Tableau@1");
    assert_eq!(value.shape(), &Partition::new(vec![2, 1]).unwrap());
    assert_eq!(value.rows()[0][0], Integer::from(-3));

    let word: Vec<_> = [3, 1, 2].into_iter().map(Integer::from).collect();
    let (insertion, recording) = Tableau::rsk(&word).unwrap();
    assert_eq!(insertion.shape(), recording.shape());
    assert_eq!(
        insertion.rows()[0],
        vec![Integer::from(1), Integer::from(2)]
    );
    assert_eq!(insertion.rows()[1], vec![Integer::from(3)]);
}

#[test]
fn symmetric_functions_use_typed_partitions_flint_rationals_and_matrix_conversion() {
    let partition = Partition::new(vec![2, 1]).unwrap();
    let value =
        SymmetricFunction::new(SymmetricBasis::Schur, [(partition, Rational::one())]).unwrap();
    let power = value.convert_basis(SymmetricBasis::PowerSum).unwrap();
    let terms = power
        .terms()
        .map(|(partition, coefficient)| (partition.parts().to_vec(), coefficient.to_string()))
        .collect::<Vec<_>>();

    assert_eq!(
        SymmetricFunction::TYPE_ID,
        "noether/combin/SymmetricFunction@1"
    );
    assert_eq!(
        terms,
        vec![
            (vec![1, 1, 1], "1/3".to_owned()),
            (vec![3], "-1/3".to_owned())
        ]
    );
    assert_eq!(power.convert_basis(SymmetricBasis::Schur).unwrap(), value);
}

#[test]
fn crystals_carry_cartan_context_and_checked_vertex_operators() {
    let shape = Partition::new(vec![2, 1]).unwrap();
    let crystal = KashiwaraCrystal::type_a_tableaux(2, &shape).unwrap();

    assert_eq!(
        KashiwaraCrystal::TYPE_ID,
        "noether/combin/KashiwaraCrystal@1"
    );
    assert_eq!(crystal.cartan_matrix(), &[vec![2, -1], vec![-1, 2]]);
    assert_eq!(crystal.vertex_count(), 8);
    assert_eq!(crystal.highest_weight_vertices().len(), 1);
    assert_eq!(crystal.character().get(&vec![0, 0]), Some(&2));
    assert!(matches!(
        crystal.f(2, 0),
        Err(CombinError::CrystalIndexOutOfRange { .. })
    ));

    let rebuilt: KashiwaraCrystal =
        serde_json::from_str(&serde_json::to_string(&crystal).unwrap()).unwrap();
    assert_eq!(rebuilt, crystal);
}

#[test]
fn crystal_tensor_products_are_native_and_deterministic() {
    let letters = KashiwaraCrystal::type_a_letters(2).unwrap();
    let tensor = letters.tensor(&letters).unwrap();
    let mut highest_weights = tensor
        .highest_weight_vertices()
        .into_iter()
        .map(|vertex| tensor.weight(vertex).unwrap().to_vec())
        .collect::<Vec<_>>();
    highest_weights.sort();

    assert_eq!(tensor.vertex_count(), 9);
    assert_eq!(highest_weights, vec![vec![0, 1], vec![2, 0]]);
    assert_eq!(
        tensor.canonical_bytes().unwrap(),
        letters.tensor(&letters).unwrap().canonical_bytes().unwrap()
    );
}
