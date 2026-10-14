// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::ResidueRing;
use noether::homalg::{
    ExteriorAlgebra, ExteriorAlgebraError, ExteriorElement, FreeModule, FreeModulePower,
    ModuleElement, ModuleHomomorphism, ModulePowerError, ModulePowerHomomorphism, PresentedModule,
    PresentedModuleHomomorphism, PresentedModulePower, PresentedModulePowerHomomorphism,
};
use noether::linalg::{Matrix, Vector};
use noether::numthy::FiniteField;
use noether::{Integer, Ring, Z};
use proptest::prelude::*;

fn property_config(default_cases: u32) -> ProptestConfig {
    let configured = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_some() {
        configured
    } else {
        ProptestConfig {
            cases: default_cases,
            ..configured
        }
    }
}

fn integer_exterior(rank: usize) -> ExteriorAlgebra<noether::IntegerRing> {
    ExteriorAlgebra::new(FreeModule::new(Z, rank))
}

fn subset(mask: usize, rank: usize) -> Vec<usize> {
    (0..rank)
        .filter(|position| mask & (1 << position) != 0)
        .collect()
}

fn integer_element(
    algebra: &ExteriorAlgebra<noether::IntegerRing>,
    coefficients: &[i8],
) -> ExteriorElement<noether::IntegerRing> {
    algebra
        .from_terms(
            coefficients
                .iter()
                .copied()
                .enumerate()
                .map(|(mask, coefficient)| {
                    (subset(mask, algebra.rank()), Integer::from(coefficient))
                }),
        )
        .unwrap()
}

fn homogeneous_integer_element(
    algebra: &ExteriorAlgebra<noether::IntegerRing>,
    coefficients: &[i8],
    degree: usize,
) -> ExteriorElement<noether::IntegerRing> {
    integer_element(algebra, coefficients).homogeneous_component(degree)
}

fn integer_vector(
    module: &FreeModule<noether::IntegerRing>,
    coordinates: impl IntoIterator<Item = i8>,
) -> ModuleElement<noether::IntegerRing> {
    ModuleElement::from_coordinates(
        module.clone(),
        Vector::from_values(Z, coordinates.into_iter().map(Integer::from)).unwrap(),
    )
    .unwrap()
}

#[test]
fn composite_residue_coefficients_support_exterior_and_presented_powers() {
    let ring = ResidueRing::new(8_u8).unwrap();
    let module = FreeModule::new(ring.clone(), 3);
    let algebra = module.exterior_algebra();
    let e0 = algebra.generator(0).unwrap();
    let e1 = algebra.generator(1).unwrap();
    let phi = ModuleElement::from_coordinates(
        module.dual().unwrap(),
        Vector::from_values(ring.clone(), [ring.one(), ring.zero(), ring.zero()]).unwrap(),
    )
    .unwrap();

    assert!(e0.wedge(&e0).unwrap().is_zero());
    assert_eq!(
        e1.wedge(&e0).unwrap(),
        e0.wedge(&e1).unwrap().neg().unwrap()
    );
    assert_eq!(e0.wedge(&e1).unwrap().contract(&phi).unwrap(), e1);

    let identity = ModuleHomomorphism::identity(module.clone()).unwrap();
    identity.exterior_power(2).unwrap().verify().unwrap();
    identity.symmetric_power(2).unwrap().verify().unwrap();

    let presented = PresentedModule::free(module).unwrap();
    let presented_identity = PresentedModuleHomomorphism::identity(presented).unwrap();
    presented_identity
        .exterior_power(2)
        .unwrap()
        .verify()
        .unwrap();
    presented_identity
        .symmetric_power(2)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn ranks_zero_one_and_two_have_the_expected_complete_bases() {
    let rank_zero = integer_exterior(0);
    assert_eq!(rank_zero.basis(0).unwrap(), vec![Vec::<usize>::new()]);
    assert!(rank_zero.basis(1).unwrap().is_empty());
    assert_eq!(rank_zero.one(), rank_zero.basis_element([]).unwrap());

    let rank_one = integer_exterior(1);
    let e0 = rank_one.generator(0).unwrap();
    assert_eq!(rank_one.basis(1).unwrap(), vec![vec![0]]);
    assert!(rank_one.basis(2).unwrap().is_empty());
    assert!(e0.wedge(&e0).unwrap().is_zero());

    let rank_two = integer_exterior(2);
    let e0 = rank_two.generator(0).unwrap();
    let e1 = rank_two.generator(1).unwrap();
    assert_eq!(rank_two.basis(0).unwrap(), vec![Vec::<usize>::new()]);
    assert_eq!(rank_two.basis(1).unwrap(), vec![vec![0], vec![1]]);
    assert_eq!(rank_two.basis(2).unwrap(), vec![vec![0, 1]]);
    assert!(rank_two.basis(3).unwrap().is_empty());
    assert_eq!(
        e1.wedge(&e0).unwrap(),
        e0.wedge(&e1).unwrap().neg().unwrap()
    );
}

#[test]
fn inhomogeneous_arithmetic_is_canonical_and_power_conversion_is_explicit() {
    let algebra = integer_exterior(3);
    let value = algebra
        .one()
        .add(&algebra.generator(0).unwrap())
        .unwrap()
        .add(
            &algebra
                .basis_element([1, 2])
                .unwrap()
                .scalar_mul(&Integer::from(4))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(value.degrees(), vec![0, 1, 2]);
    assert!(matches!(
        value.to_power_element(2),
        Err(ExteriorAlgebraError::DegreeMismatch {
            expected: 2,
            actual: None,
            ..
        })
    ));

    let degree_two = value.homogeneous_component(2);
    let coordinates = degree_two.to_power_element(2).unwrap();
    assert_eq!(
        algebra.from_power_element(2, &coordinates).unwrap(),
        degree_two
    );
    algebra.zero().to_power_element(17).unwrap();
}

#[test]
fn contraction_is_the_signed_interior_product_and_fails_on_a_wrong_dual() {
    let algebra = integer_exterior(3);
    let covector = integer_vector(&algebra.source().dual().unwrap(), [2, 3, 5]);
    let top = algebra.basis_element([0, 1, 2]).unwrap();
    let contracted = top.contract(&covector).unwrap();
    let expected = algebra
        .from_terms([
            (vec![1, 2], Integer::from(2)),
            (vec![0, 2], Integer::from(-3)),
            (vec![0, 1], Integer::from(5)),
        ])
        .unwrap();
    assert_eq!(contracted, expected);
    assert!(algebra.one().contract(&covector).unwrap().is_zero());

    let wrong = integer_vector(&FreeModule::new(Z, 2), [1, 0]);
    assert_eq!(
        top.contract(&wrong),
        Err(ExteriorAlgebraError::CovectorMismatch)
    );
}

#[test]
fn alternating_product_agrees_with_wedge_and_repeated_vectors_vanish() {
    let algebra = integer_exterior(3);
    let x = integer_vector(algebra.source(), [1, 2, 3]);
    let y = integer_vector(algebra.source(), [2, 0, 1]);
    assert_eq!(
        algebra.alternating_product([&x, &y]).unwrap(),
        algebra
            .from_vector(&x)
            .unwrap()
            .wedge(&algebra.from_vector(&y).unwrap())
            .unwrap()
    );
    assert!(algebra.alternating_product([&x, &x]).unwrap().is_zero());
    assert_eq!(algebra.alternating_product([]).unwrap(), algebra.one());
}

#[test]
fn characteristic_two_is_commutative_but_still_alternating() {
    let field = FiniteField::prime(2_u8).unwrap();
    let algebra = ExteriorAlgebra::new(FreeModule::new(field, 2));
    let e0 = algebra.generator(0).unwrap();
    let e1 = algebra.generator(1).unwrap();
    assert_eq!(e0.wedge(&e1).unwrap(), e1.wedge(&e0).unwrap());
    assert!(e0.wedge(&e0).unwrap().is_zero());
    let linear_form = e0.add(&e1).unwrap();
    assert!(linear_form.wedge(&linear_form).unwrap().is_zero());

    let dual = algebra.source().dual().unwrap();
    let phi = ModuleElement::from_coordinates(
        dual.clone(),
        Vector::from_values(
            algebra.ring().clone(),
            [algebra.ring().one(), algebra.ring().one()],
        )
        .unwrap(),
    )
    .unwrap();
    let psi = ModuleElement::from_coordinates(
        dual,
        Vector::from_values(
            algebra.ring().clone(),
            [algebra.ring().one(), algebra.ring().zero()],
        )
        .unwrap(),
    )
    .unwrap();
    let top = e0.wedge(&e1).unwrap();
    assert_eq!(top.contract(&phi).unwrap(), linear_form);
    assert!(top
        .contract(&phi)
        .unwrap()
        .contract(&phi)
        .unwrap()
        .is_zero());
    assert_eq!(
        top.contract(&phi).unwrap().contract(&psi).unwrap(),
        top.contract(&psi).unwrap().contract(&phi).unwrap()
    );
    assert!(algebra
        .from_terms([(vec![0, 0], algebra.ring().one())])
        .is_err());
}

#[test]
fn exterior_resource_limits_fail_before_large_tables_or_products_are_built() {
    assert!(matches!(
        FreeModule::new(Z, 1_000).exterior_power(2),
        Err(ModulePowerError::BasisTableLimit {
            requested: Some(499_500_000),
            limit: 8_000_000,
        })
    ));

    let module = FreeModule::new(Z, 70);
    let zero = ModuleHomomorphism::zero(module.clone(), module).unwrap();
    assert!(matches!(
        zero.exterior_power(2),
        Err(ModulePowerError::MatrixCellLimit {
            operation: "exterior-power induced matrix",
            requested: Some(5_832_225),
            limit: 4_000_000,
        })
    ));

    let relations = FreeModule::new(Z, 30);
    let generators = FreeModule::new(Z, 70);
    let presentation = ModuleHomomorphism::zero(relations, generators).unwrap();
    let presented = PresentedModule::cokernel(presentation);
    assert!(matches!(
        presented.exterior_power(2),
        Err(ModulePowerError::MatrixCellLimit {
            operation: "presented module-power relation matrix",
            requested: Some(5_071_500),
            limit: 4_000_000,
        })
    ));

    let mut state = 0x9e37_79b9_u64;
    let rows = (0..100)
        .map(|_| {
            (0..100)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    Integer::from((state % 19) as i64 - 9)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let module = FreeModule::new(Z, 100);
    let expansive = ModuleHomomorphism::from_matrix(
        module.clone(),
        module,
        Matrix::from_rows(Z, rows).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        expansive.exterior_power(100),
        Err(ModulePowerError::ExpansionWorkLimit {
            operation: "exterior-power coefficient expansion",
            requested: Some(_),
            limit: 16_000_000,
        })
    ));

    let algebra = integer_exterior(64);
    let terms = algebra
        .from_terms(
            algebra
                .basis(2)
                .unwrap()
                .into_iter()
                .map(|basis| (basis, Integer::from(1))),
        )
        .unwrap();
    assert!(matches!(
        terms.wedge(&terms),
        Err(ExteriorAlgebraError::TermPairLimit {
            requested: Some(4_064_256),
            limit: 4_000_000,
        })
    ));
}

#[test]
fn context_degree_and_basis_mismatches_are_typed() {
    let rank_two = integer_exterior(2);
    let rank_three = integer_exterior(3);
    assert!(matches!(
        rank_two
            .generator(0)
            .unwrap()
            .wedge(&rank_three.generator(0).unwrap()),
        Err(ExteriorAlgebraError::AlgebraMismatch { .. })
    ));
    assert!(matches!(
        rank_two.basis_element([2]),
        Err(ExteriorAlgebraError::BasisPosition {
            position: 2,
            rank: 2,
            ..
        })
    ));
    assert!(matches!(
        rank_two.basis_element([1, 0]),
        Err(ExteriorAlgebraError::BasisNotIncreasing { .. })
    ));

    let module = FreeModule::new(Z, 2);
    let identity = ModuleHomomorphism::identity(module).unwrap();
    let exterior_square = identity.exterior_power(2).unwrap();
    assert!(matches!(
        exterior_square.apply_exterior(&rank_two.generator(0).unwrap()),
        Err(ModulePowerError::DegreeMismatch {
            expected: 2,
            actual: Some(1)
        })
    ));
    assert!(matches!(
        identity
            .symmetric_power(2)
            .unwrap()
            .apply_exterior(&rank_two.basis_element([0, 1]).unwrap()),
        Err(ModulePowerError::KindMismatch { .. })
    ));
    assert!(matches!(
        exterior_square.compose(&identity.exterior_power(1).unwrap()),
        Err(ModulePowerError::DegreeMismatch { .. })
    ));
    assert!(matches!(
        exterior_square.compose(&identity.symmetric_power(2).unwrap()),
        Err(ModulePowerError::KindMismatch { .. })
    ));
}

#[test]
fn induced_exterior_map_is_the_alternating_map_on_vectors() {
    let source = FreeModule::new(Z, 3);
    let target = FreeModule::new(Z, 2);
    let map = ModuleHomomorphism::from_matrix(
        source.clone(),
        target.clone(),
        Matrix::from_rows(
            Z,
            [
                [Integer::from(1), Integer::from(2), Integer::from(0)],
                [Integer::from(0), Integer::from(3), Integer::from(1)],
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let source_algebra = ExteriorAlgebra::new(source);
    let target_algebra = ExteriorAlgebra::new(target);
    let source_term = source_algebra.basis_element([0, 1]).unwrap();
    let induced = map.exterior_power(2).unwrap();
    let image = induced.apply_exterior(&source_term).unwrap();
    assert_eq!(
        image,
        target_algebra
            .basis_element([0, 1])
            .unwrap()
            .scalar_mul(&Integer::from(3))
            .unwrap()
    );
    induced.verify().unwrap();
    let restored = noether::homalg::ModulePowerHomomorphism::<noether::IntegerRing>::from_json(
        &induced.to_json().unwrap(),
    )
    .unwrap();
    assert_eq!(restored, induced);
}

#[test]
fn degree_zero_induced_map_verifies_and_maps_the_unit_to_the_unit() {
    let module = FreeModule::new(Z, 3);
    let map = ModuleHomomorphism::identity(module.clone()).unwrap();
    let induced = map.exterior_power(0).unwrap();
    assert!(induced.induced().is_sparse());
    induced.verify().unwrap();

    let algebra = module.exterior_algebra();
    assert_eq!(
        induced.apply_exterior(&algebra.one()).unwrap(),
        algebra.one()
    );
    assert!(induced.apply_exterior(&algebra.zero()).unwrap().is_zero());
}

#[test]
fn degree_one_induced_maps_preserve_the_original_sparse_storage() {
    let module = FreeModule::new(Z, 4);
    let map = ModuleHomomorphism::identity(module).unwrap();
    assert!(map.is_sparse());
    let exterior = map.exterior_power(1).unwrap();
    let symmetric = map.symmetric_power(1).unwrap();
    assert!(exterior.induced().is_sparse());
    assert!(symmetric.induced().is_sparse());
    assert_eq!(exterior.induced(), &map);
    assert_eq!(symmetric.induced(), &map);
}

#[test]
fn induced_map_deserialization_replays_evidence_and_rejects_tampering() {
    let module = FreeModule::new(Z, 3);
    let map = ModuleHomomorphism::identity(module.clone()).unwrap();
    let exterior = map.exterior_power(2).unwrap();
    let mut wire = serde_json::to_value(&exterior).unwrap();
    wire["degree"] = serde_json::json!(1);
    assert!(serde_json::from_value::<ModulePowerHomomorphism<noether::IntegerRing>>(wire).is_err());

    let presented = PresentedModule::free(module).unwrap();
    let presented_map = PresentedModuleHomomorphism::identity(presented)
        .unwrap()
        .exterior_power(2)
        .unwrap();
    let mut wire = serde_json::to_value(&presented_map).unwrap();
    wire["kind"] = serde_json::json!("symmetric");
    assert!(
        serde_json::from_value::<PresentedModulePowerHomomorphism<noether::IntegerRing>>(wire)
            .is_err()
    );
}

#[test]
fn module_power_deserialization_replays_canonical_objects_and_rejects_tampering() {
    let module = FreeModule::new(Z, 3);
    let power = module.symmetric_power(2).unwrap();

    let mut wrong_degree = serde_json::to_value(&power).unwrap();
    wrong_degree["degree"] = serde_json::json!(1);
    assert!(serde_json::from_value::<FreeModulePower<noether::IntegerRing>>(wrong_degree).is_err());

    let mut wrong_basis = serde_json::to_value(&power).unwrap();
    wrong_basis["basis_exponents"][0] = serde_json::json!([2, 0, 1]);
    assert!(serde_json::from_value::<FreeModulePower<noether::IntegerRing>>(wrong_basis).is_err());

    let mut wrong_module = serde_json::to_value(&power).unwrap();
    wrong_module["module"] = serde_json::to_value(&module).unwrap();
    assert!(serde_json::from_value::<FreeModulePower<noether::IntegerRing>>(wrong_module).is_err());

    let presented = PresentedModule::free(module.clone()).unwrap();
    let presented_power = presented.symmetric_power(2).unwrap();

    let mut wrong_presented_degree = serde_json::to_value(&presented_power).unwrap();
    wrong_presented_degree["degree"] = serde_json::json!(1);
    assert!(
        serde_json::from_value::<PresentedModulePower<noether::IntegerRing>>(
            wrong_presented_degree
        )
        .is_err()
    );

    let mut wrong_presented_module = serde_json::to_value(&presented_power).unwrap();
    wrong_presented_module["module"] = serde_json::to_value(&presented).unwrap();
    assert!(
        serde_json::from_value::<PresentedModulePower<noether::IntegerRing>>(
            wrong_presented_module
        )
        .is_err()
    );

    let induced = ModuleHomomorphism::identity(module)
        .unwrap()
        .symmetric_power(2)
        .unwrap();
    let mut nested = serde_json::to_value(&induced).unwrap();
    nested["source_power"]["basis_exponents"][0] = serde_json::json!([0, 0, 0]);
    assert!(
        serde_json::from_value::<ModulePowerHomomorphism<noether::IntegerRing>>(nested).is_err()
    );
}

#[test]
fn nontrivial_presented_power_maps_descend_through_the_relation_square() {
    let generators = FreeModule::new(Z, 2)
        .with_basis_labels(vec!["x".to_owned(), "y".to_owned()])
        .unwrap();
    let relations = FreeModule::new(Z, 2)
        .with_basis_labels(vec!["r0".to_owned(), "r1".to_owned()])
        .unwrap();
    let presentation = ModuleHomomorphism::from_matrix(
        relations.clone(),
        generators.clone(),
        Matrix::from_rows(
            Z,
            [
                [Integer::from(2), Integer::from(0)],
                [Integer::from(0), Integer::from(2)],
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let module = PresentedModule::cokernel(presentation);
    let identity_power = PresentedModuleHomomorphism::identity(module.clone())
        .unwrap()
        .exterior_power(2)
        .unwrap();
    assert!(identity_power.induced().relation_lift().is_sparse());
    identity_power.verify().unwrap();

    let matrix = Matrix::from_rows(
        Z,
        [
            [Integer::from(1), Integer::from(2)],
            [Integer::from(0), Integer::from(1)],
        ],
    )
    .unwrap();
    let generator_lift =
        ModuleHomomorphism::from_matrix(generators.clone(), generators, matrix.clone()).unwrap();
    let relation_lift =
        ModuleHomomorphism::from_matrix(relations.clone(), relations, matrix).unwrap();
    let map = PresentedModuleHomomorphism::from_lifts(
        module.clone(),
        module,
        generator_lift,
        relation_lift,
    )
    .unwrap();

    for induced in [
        map.exterior_power(2).unwrap(),
        map.symmetric_power(2).unwrap(),
    ] {
        induced.verify().unwrap();
        induced.induced().verify().unwrap();
    }
    let induced = map.exterior_power(2).unwrap();
    let restored =
        noether::homalg::PresentedModulePowerHomomorphism::<noether::IntegerRing>::from_json(
            &induced.to_json().unwrap(),
        )
        .unwrap();
    assert_eq!(restored, induced);
}

#[test]
fn presented_power_maps_are_functorial_under_composition() {
    let generators = FreeModule::new(Z, 2);
    let relations = FreeModule::new(Z, 2);
    let presentation = ModuleHomomorphism::from_matrix(
        relations.clone(),
        generators.clone(),
        Matrix::from_rows(
            Z,
            [
                [Integer::from(2), Integer::from(0)],
                [Integer::from(0), Integer::from(2)],
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let module = PresentedModule::cokernel(presentation);
    let checked_map = |entries: [[i64; 2]; 2]| {
        let matrix = Matrix::from_rows(Z, entries.map(|row| row.map(Integer::from))).unwrap();
        PresentedModuleHomomorphism::from_lifts(
            module.clone(),
            module.clone(),
            ModuleHomomorphism::from_matrix(generators.clone(), generators.clone(), matrix.clone())
                .unwrap(),
            ModuleHomomorphism::from_matrix(relations.clone(), relations.clone(), matrix).unwrap(),
        )
        .unwrap()
    };
    let first = checked_map([[1, 2], [0, 1]]);
    let second = checked_map([[2, 0], [1, 1]]);
    let composite = second.compose(&first).unwrap();

    let exterior_stages = second
        .exterior_power(2)
        .unwrap()
        .compose(&first.exterior_power(2).unwrap())
        .unwrap();
    assert_eq!(composite.exterior_power(2).unwrap(), exterior_stages);
    exterior_stages.verify().unwrap();

    let symmetric_stages = second
        .symmetric_power(2)
        .unwrap()
        .compose(&first.symmetric_power(2).unwrap())
        .unwrap();
    assert_eq!(composite.symmetric_power(2).unwrap(), symmetric_stages);
    symmetric_stages.verify().unwrap();
}

proptest! {
    #![proptest_config(property_config(96))]

    #[test]
    fn wedge_is_associative_and_bilinear(
        left in prop::collection::vec(-3i8..=3, 16),
        middle in prop::collection::vec(-3i8..=3, 16),
        right in prop::collection::vec(-3i8..=3, 16),
    ) {
        let algebra = integer_exterior(4);
        let x = integer_element(&algebra, &left);
        let y = integer_element(&algebra, &middle);
        let z = integer_element(&algebra, &right);
        prop_assert_eq!(
            x.wedge(&y).unwrap().wedge(&z).unwrap(),
            x.wedge(&y.wedge(&z).unwrap()).unwrap()
        );
        prop_assert_eq!(
            x.add(&y).unwrap().wedge(&z).unwrap(),
            x.wedge(&z).unwrap().add(&y.wedge(&z).unwrap()).unwrap()
        );
    }

    #[test]
    fn homogeneous_wedge_is_graded_commutative_and_odd_squares_vanish(
        left in prop::collection::vec(-3i8..=3, 16),
        right in prop::collection::vec(-3i8..=3, 16),
        left_degree in 0usize..=4,
        right_degree in 0usize..=4,
    ) {
        let algebra = integer_exterior(4);
        let x = homogeneous_integer_element(&algebra, &left, left_degree);
        let y = homogeneous_integer_element(&algebra, &right, right_degree);
        let swapped = y.wedge(&x).unwrap();
        let expected = if left_degree * right_degree % 2 == 1 {
            swapped.scalar_mul(&Integer::from(-1)).unwrap()
        } else {
            swapped
        };
        prop_assert_eq!(x.wedge(&y).unwrap(), expected);
        if left_degree % 2 == 1 {
            prop_assert!(x.wedge(&x).unwrap().is_zero());
        }
    }

    #[test]
    fn contraction_satisfies_the_graded_leibniz_rule(
        left in prop::collection::vec(-3i8..=3, 16),
        right in prop::collection::vec(-3i8..=3, 16),
        covector in prop::array::uniform4(-3i8..=3),
        left_degree in 0usize..=4,
        right_degree in 0usize..=4,
    ) {
        let algebra = integer_exterior(4);
        let x = homogeneous_integer_element(&algebra, &left, left_degree);
        let y = homogeneous_integer_element(&algebra, &right, right_degree);
        let phi = integer_vector(&algebra.source().dual().unwrap(), covector);
        let actual = x.wedge(&y).unwrap().contract(&phi).unwrap();
        let first = x.contract(&phi).unwrap().wedge(&y).unwrap();
        let mut second = x.wedge(&y.contract(&phi).unwrap()).unwrap();
        if left_degree % 2 == 1 {
            second = second.scalar_mul(&Integer::from(-1)).unwrap();
        }
        prop_assert_eq!(actual, first.add(&second).unwrap());
    }

    #[test]
    fn contractions_square_to_zero_and_anticommute(
        coefficients in prop::collection::vec(-3i8..=3, 16),
        left_covector in prop::array::uniform4(-3i8..=3),
        right_covector in prop::array::uniform4(-3i8..=3),
    ) {
        let algebra = integer_exterior(4);
        let value = integer_element(&algebra, &coefficients);
        let dual = algebra.source().dual().unwrap();
        let phi = integer_vector(&dual, left_covector);
        let psi = integer_vector(&dual, right_covector);

        prop_assert!(value
            .contract(&phi).unwrap()
            .contract(&phi).unwrap()
            .is_zero());
        let phi_then_psi = value.contract(&phi).unwrap().contract(&psi).unwrap();
        let psi_then_phi = value.contract(&psi).unwrap().contract(&phi).unwrap();
        prop_assert!(phi_then_psi.add(&psi_then_phi).unwrap().is_zero());
    }

    #[test]
    fn exterior_and_symmetric_power_maps_are_functorial(
        first_entries in prop::collection::vec(-2i8..=2, 9),
        second_entries in prop::collection::vec(-2i8..=2, 9),
    ) {
        let module = FreeModule::new(Z, 3);
        let first = ModuleHomomorphism::from_matrix(
            module.clone(),
            module.clone(),
            Matrix::from_rows(
                Z,
                first_entries.chunks_exact(3).map(|row| row.iter().copied().map(Integer::from)),
            ).unwrap(),
        ).unwrap();
        let second = ModuleHomomorphism::from_matrix(
            module.clone(),
            module,
            Matrix::from_rows(
                Z,
                second_entries.chunks_exact(3).map(|row| row.iter().copied().map(Integer::from)),
            ).unwrap(),
        ).unwrap();
        let composite = second.compose(&first).unwrap();

        let exterior_composite = composite.exterior_power(2).unwrap();
        let first_exterior = first.exterior_power(2).unwrap();
        let second_exterior = second.exterior_power(2).unwrap();
        let exterior_stages = second_exterior.compose(&first_exterior).unwrap();
        prop_assert_eq!(exterior_composite, exterior_stages);

        let symmetric_composite = composite.symmetric_power(2).unwrap();
        let first_symmetric = first.symmetric_power(2).unwrap();
        let second_symmetric = second.symmetric_power(2).unwrap();
        let symmetric_stages = second_symmetric.compose(&first_symmetric).unwrap();
        prop_assert_eq!(symmetric_composite, symmetric_stages);
    }

    #[test]
    fn homogeneous_coordinate_round_trips(
        coefficients in prop::collection::vec(-5i8..=5, 16),
        degree in 0usize..=4,
    ) {
        let algebra = integer_exterior(4);
        let element = homogeneous_integer_element(&algebra, &coefficients, degree);
        let coordinates = element.to_power_element(degree).unwrap();
        prop_assert_eq!(algebra.from_power_element(degree, &coordinates).unwrap(), element);
    }
}
