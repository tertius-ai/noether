// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use groebner_kernel::{
    GroebnerOptions, Monomial, MonomialOrder, PrimeField32, SparseModuleElement, SparseModuleOrder,
    SparseModuleTerm,
};
use module_core::ModuleTerm;
use poly_core::MonomialArena;
use resolution_kernel::{
    schreyer_degree_cells, standard_graded_betti_prime, standard_graded_resolution_prime,
    StandardGradedBettiEntry, StandardGradedResolutionFrame, StandardGradedResolutionOutput,
};

fn assert_is_complex(field: &PrimeField32, output: &StandardGradedResolutionOutput<u32>) {
    for adjacent in output.differentials.windows(2) {
        let preceding = &adjacent[0];
        let current = &adjacent[1];
        for column in &current.columns {
            let coefficients = column.coordinates(field);
            assert_eq!(coefficients.len(), preceding.columns.len());
            let mut composite = vec![
                groebner_kernel::SparsePolynomial::new(
                    column.arity(),
                    column.polynomial_order(),
                    [],
                    field,
                );
                preceding.columns[0].rank()
            ];
            for (coefficient, image) in coefficients.iter().zip(&preceding.columns) {
                for (value, image_coordinate) in composite.iter_mut().zip(image.coordinates(field))
                {
                    *value =
                        value.add(&coefficient.multiply(&image_coordinate, field, None), field);
                }
            }
            assert!(composite.iter().all(|value| value.is_zero()));
        }
    }
}

#[test]
fn public_packed_betti_lane_resolves_a_principal_quadratic() {
    let field = PrimeField32::new(101).unwrap();
    let polynomial_order = MonomialOrder::GradedReverseLexicographic;
    let presentation = vec![SparseModuleElement::new(
        1,
        1,
        polynomial_order,
        [(
            SparseModuleTerm::new(Monomial::new(vec![2].into_boxed_slice()), 0),
            1,
        )],
        &field,
    )];
    let output = standard_graded_betti_prime(
        &field,
        &presentation,
        &[0],
        &[2],
        &SparseModuleOrder::TermOverPosition(polynomial_order),
        &GroebnerOptions::default(),
    )
    .unwrap();

    assert_eq!(
        output.entries,
        vec![
            StandardGradedBettiEntry {
                homological_degree: 0,
                internal_degree: 0,
                multiplicity: 1,
            },
            StandardGradedBettiEntry {
                homological_degree: 1,
                internal_degree: 2,
                multiplicity: 1,
            },
        ],
    );
    assert_eq!(output.length, 1);
}

#[test]
fn public_packed_resolution_retains_exact_sparse_frames() {
    let field = PrimeField32::new(101).unwrap();
    let polynomial_order = MonomialOrder::GradedReverseLexicographic;
    let presentation = vec![SparseModuleElement::new(
        1,
        1,
        polynomial_order,
        [(
            SparseModuleTerm::new(Monomial::new(vec![2].into_boxed_slice()), 0),
            1,
        )],
        &field,
    )];
    let output = standard_graded_resolution_prime(
        &field,
        &presentation,
        &[0],
        &[2],
        &SparseModuleOrder::TermOverPosition(polynomial_order),
        &GroebnerOptions::default(),
    )
    .unwrap();

    assert_eq!(output.differentials.len(), 1);
    assert_eq!(output.differentials[0].columns, presentation);
    assert_eq!(output.differentials[0].source_generator_degrees, [2]);
}

#[test]
fn public_packed_resolution_cancels_higher_units_without_breaking_the_complex() {
    let field = PrimeField32::new(65_521).unwrap();
    let polynomial_order = MonomialOrder::GradedReverseLexicographic;
    let presentation = [
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
    ]
    .into_iter()
    .map(|exponents| {
        SparseModuleElement::new(
            5,
            1,
            polynomial_order,
            [(
                SparseModuleTerm::new(Monomial::new(Vec::from(exponents).into_boxed_slice()), 0),
                1,
            )],
            &field,
        )
    })
    .collect::<Vec<_>>();
    let mut output = standard_graded_resolution_prime(
        &field,
        &presentation,
        &[0],
        &[2; 5],
        &SparseModuleOrder::TermOverPosition(polynomial_order),
        &GroebnerOptions::default(),
    )
    .unwrap();

    assert_eq!(
        output
            .differentials
            .iter()
            .map(|frame| frame.columns.len())
            .collect::<Vec<_>>(),
        [5, 6, 2]
    );
    assert_is_complex(&field, &output);

    output.minimalize_higher_differentials(&field).unwrap();

    assert_eq!(
        output
            .differentials
            .iter()
            .map(|frame| frame.columns.len())
            .collect::<Vec<_>>(),
        [5, 5, 1]
    );
    assert_eq!(output.differentials[0].source_generator_degrees, [2; 5]);
    assert_eq!(output.differentials[1].source_generator_degrees, [3; 5]);
    assert_eq!(output.differentials[2].source_generator_degrees, [5]);
    assert_is_complex(&field, &output);
}

#[test]
fn public_packed_resolution_discards_a_contractible_trailing_pair() {
    let field = PrimeField32::new(101).unwrap();
    let polynomial_order = MonomialOrder::GradedReverseLexicographic;
    let zero = SparseModuleElement::new(1, 1, polynomial_order, [], &field);
    let unit = SparseModuleElement::new(
        1,
        1,
        polynomial_order,
        [(
            SparseModuleTerm::new(Monomial::new(vec![0].into_boxed_slice()), 0),
            1,
        )],
        &field,
    );
    let mut output = StandardGradedResolutionOutput {
        differentials: vec![
            StandardGradedResolutionFrame {
                columns: vec![zero],
                source_generator_degrees: vec![0],
            },
            StandardGradedResolutionFrame {
                columns: vec![unit],
                source_generator_degrees: vec![0],
            },
        ],
    };

    output.minimalize_higher_differentials(&field).unwrap();

    assert!(output.differentials.is_empty());
}

#[test]
fn public_degree_cells_filter_components_and_keep_stable_pair_order() {
    let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
    let x2 = arena.intern_exponents(&[2, 0]).unwrap();
    let xy = arena.intern_exponents(&[1, 1]).unwrap();
    let y2 = arena.intern_exponents(&[0, 2]).unwrap();
    let other = arena.intern_exponents(&[1, 0]).unwrap();
    let cells = schreyer_degree_cells(
        &mut arena,
        &[
            ModuleTerm::new(x2, 0),
            ModuleTerm::new(xy, 0),
            ModuleTerm::new(y2, 0),
            ModuleTerm::new(other, 1),
        ],
        3,
    )
    .unwrap();

    assert_eq!(cells.len(), 2);
    assert_eq!(cells[0].degree(), 3);
    assert_eq!(
        cells[0]
            .pairs()
            .iter()
            .map(|pair| (pair.left(), pair.right()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 2)],
    );
    assert_eq!(cells[1].degree(), 4);
    assert_eq!(
        cells[1]
            .pairs()
            .iter()
            .map(|pair| (pair.left(), pair.right()))
            .collect::<Vec<_>>(),
        vec![(0, 2)],
    );
}
