// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::ResidueRing;
use noether::homalg::{FreeModule, ModuleElement, Submodule};
use noether::linalg::{CongruenceSystemResult, IntegerLinearSystemResult, Matrix, Vector};
use noether::{Integer, Z};

#[test]
fn integer_affine_lattice_and_obstruction_are_public_and_replayable() {
    let operator = Matrix::from_rows(
        Z,
        [
            [Integer::from(2), Integer::from(4), Integer::from(6)],
            [Integer::from(0), Integer::from(3), Integer::from(3)],
        ],
    )
    .unwrap();
    let rhs = Vector::from_values(Z, [Integer::from(8), Integer::from(6)]).unwrap();
    let result = operator.solve_exact(&rhs).unwrap();
    let IntegerLinearSystemResult::Solutions(solution) = &result else {
        panic!("the displayed system has the solution (1, 0, 1)");
    };
    assert_eq!(solution.homogeneous_rank(), 1);
    assert_eq!(
        operator
            .matmul(&solution.particular_solution().to_column_matrix())
            .unwrap(),
        rhs.to_column_matrix()
    );
    result.verify().unwrap();

    let impossible = Matrix::from_rows(
        Z,
        [
            [Integer::from(2), Integer::from(0)],
            [Integer::from(0), Integer::from(3)],
        ],
    )
    .unwrap()
    .solve_exact(&Vector::from_values(Z, [Integer::from(1), Integer::from(0)]).unwrap())
    .unwrap();
    assert!(impossible.obstruction().is_some());
    impossible.verify().unwrap();
}

#[test]
fn composite_congruence_solution_module_is_complete_on_an_exhaustive_small_case() {
    let modulus = 6_u64;
    let ring = ResidueRing::new(modulus).unwrap();
    let operator = Matrix::from_rows(ring.clone(), [[ring.element(2), ring.element(3)]]).unwrap();
    let rhs = Vector::from_values(ring.clone(), [ring.element(1)]).unwrap();
    let result = operator.solve_exact(&rhs).unwrap();
    let CongruenceSystemResult::Solutions(solution) = &result else {
        panic!("2x+3y=1 modulo 6 has solutions");
    };
    let particular = solution.particular_solution().to_values();
    let generators = solution.homogeneous_generators();

    for x in 0..modulus {
        for y in 0..modulus {
            let satisfies = (2 * x + 3 * y) % modulus == 1;
            let delta = [
                (x + modulus - particular[0].representative().to_u64_checked().unwrap()) % modulus,
                (y + modulus - particular[1].representative().to_u64_checked().unwrap()) % modulus,
            ];
            let generated = generated_by_columns_modulo(&delta, generators, modulus);
            assert_eq!(generated, satisfies, "candidate ({x}, {y})");
        }
    }
    result.verify().unwrap();
}

#[test]
fn integer_and_residue_submodule_quotients_have_expected_invariants() {
    let ambient = FreeModule::new(Z, 2);
    let element = |entries: [i64; 2]| {
        ModuleElement::from_coordinates(
            ambient.clone(),
            Vector::from_values(Z, entries.map(Integer::from)).unwrap(),
        )
        .unwrap()
    };
    let lattice =
        Submodule::from_generators(ambient.clone(), vec![element([2, 0]), element([0, 4])])
            .unwrap();
    assert_eq!(
        lattice.quotient_invariants().torsion_coefficients(),
        [Integer::from(2), Integer::from(4)]
    );

    let ring = ResidueRing::new(12).unwrap();
    let ambient = FreeModule::new(ring.clone(), 1);
    let four = ModuleElement::from_coordinates(
        ambient.clone(),
        Vector::from_values(ring.clone(), [ring.element(4)]).unwrap(),
    )
    .unwrap();
    let submodule = Submodule::from_generators(ambient, vec![four]).unwrap();
    assert_eq!(
        submodule.quotient_invariants().torsion_coefficients(),
        [Integer::from(4)]
    );
}

fn generated_by_columns_modulo(
    target: &[u64; 2],
    generators: &Matrix<ResidueRing>,
    modulus: u64,
) -> bool {
    fn visit(
        column: usize,
        sums: [u64; 2],
        target: &[u64; 2],
        generators: &Matrix<ResidueRing>,
        modulus: u64,
    ) -> bool {
        if column == generators.shape().1 {
            return sums == *target;
        }
        let first = generators
            .get(0, column)
            .unwrap()
            .representative()
            .to_u64_checked()
            .unwrap();
        let second = generators
            .get(1, column)
            .unwrap()
            .representative()
            .to_u64_checked()
            .unwrap();
        (0..modulus).any(|coefficient| {
            visit(
                column + 1,
                [
                    (sums[0] + coefficient * first) % modulus,
                    (sums[1] + coefficient * second) % modulus,
                ],
                target,
                generators,
                modulus,
            )
        })
    }
    visit(0, [0, 0], target, generators, modulus)
}
