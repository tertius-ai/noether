// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use noether::commalg::{Degree, MonomialOrder, PolynomialRing};
use noether::homalg::{
    FreeModule, ModuleElement, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule, Submodule,
};
use noether::linalg::Matrix;
use noether::numthy::FiniteField;
use noether::{Field, Integer, Rational, Ring, Q};

fn cycle_edge_module<C: Field>(
    ring: &PolynomialRing<C>,
) -> (PresentedModule<PolynomialRing<C>>, ModuleMonomialOrder) {
    let variables = ring.generators().unwrap();
    let edges = variables
        .iter()
        .zip(variables.iter().cycle().skip(1))
        .take(variables.len())
        .map(|(left, right)| (left * right).unwrap())
        .collect::<Vec<_>>();
    let target = FreeModule::new(ring.clone(), 1)
        .with_degree_shifts(1, vec![Degree::zero(1)])
        .unwrap();
    let source = FreeModule::new(ring.clone(), edges.len())
        .with_degree_shifts(
            1,
            vec![Degree::new(vec![Integer::from(-2_i64)]); edges.len()],
        )
        .unwrap();
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), [edges]).unwrap(),
    )
    .unwrap();
    (
        PresentedModule::cokernel(presentation),
        ModuleMonomialOrder::TermOverPosition(ring.term_order()),
    )
}

fn twisted_cubic_module<C: Field>(
    ring: &PolynomialRing<C>,
) -> (PresentedModule<PolynomialRing<C>>, ModuleMonomialOrder) {
    let [x, y, z, w] = ring.generators().unwrap().try_into().unwrap();
    let relations = vec![
        (&(&x * &z).unwrap() - &y.pow(2).unwrap()).unwrap(),
        (&(&x * &w).unwrap() - &(&y * &z).unwrap()).unwrap(),
        (&(&y * &w).unwrap() - &z.pow(2).unwrap()).unwrap(),
    ];
    quotient_by_relations(ring, relations)
}

fn generic_two_by_three_minors_module<C: Field>(
    ring: &PolynomialRing<C>,
) -> (PresentedModule<PolynomialRing<C>>, ModuleMonomialOrder) {
    let [a, b, c, d, e, f] = ring.generators().unwrap().try_into().unwrap();
    let relations = vec![
        (&(&a * &e).unwrap() - &(&b * &d).unwrap()).unwrap(),
        (&(&a * &f).unwrap() - &(&c * &d).unwrap()).unwrap(),
        (&(&b * &f).unwrap() - &(&c * &e).unwrap()).unwrap(),
    ];
    quotient_by_relations(ring, relations)
}

fn quotient_by_relations<C: Field>(
    ring: &PolynomialRing<C>,
    relations: Vec<noether::commalg::Polynomial<C>>,
) -> (PresentedModule<PolynomialRing<C>>, ModuleMonomialOrder) {
    let target = FreeModule::new(ring.clone(), 1)
        .with_degree_shifts(1, vec![Degree::zero(1)])
        .unwrap();
    let source = FreeModule::new(ring.clone(), relations.len())
        .with_degree_shifts(
            1,
            vec![Degree::new(vec![Integer::from(-2_i64)]); relations.len()],
        )
        .unwrap();
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), [relations]).unwrap(),
    )
    .unwrap();
    (
        PresentedModule::cokernel(presentation),
        ModuleMonomialOrder::TermOverPosition(ring.term_order()),
    )
}

fn semiregular_three_rational_presentation() -> (
    ModuleHomomorphism<PolynomialRing<noether::RationalField>>,
    ModuleMonomialOrder,
) {
    let ring = PolynomialRing::new(
        Q,
        ["x0", "x1", "x2"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let mut state = 0x4D59_5DF4_D0F3_3173_u64;
    let mut coefficient = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let residue = state % 65_520 + 1;
        Rational::from(if residue <= 32_760 {
            residue as i64
        } else {
            residue as i64 - 65_521
        })
    };
    let generators = (0..3)
        .map(|_| {
            let mut terms = Vec::new();
            for left in 0..3 {
                for right in left..3 {
                    let mut exponent = vec![0_u64; 3];
                    exponent[left] += 1;
                    exponent[right] += 1;
                    terms.push((exponent, coefficient()));
                }
            }
            for variable in 0..3 {
                let mut exponent = vec![0_u64; 3];
                exponent[variable] = 1;
                terms.push((exponent, coefficient()));
            }
            terms.push((vec![0_u64; 3], coefficient()));
            ring.from_u64_terms(terms).unwrap()
        })
        .collect::<Vec<_>>();
    let source = FreeModule::new(ring.clone(), generators.len());
    let target = FreeModule::new(ring.clone(), 1);
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), [generators]).unwrap(),
    )
    .unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    (presentation, order)
}

fn sixth_power_three_variable_presentation() -> (
    ModuleHomomorphism<PolynomialRing<FiniteField>>,
    ModuleMonomialOrder,
) {
    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(
        field.clone(),
        ["x0", "x1", "x2"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let generators = (0_u64..=6)
        .flat_map(|x_degree| {
            let ring = ring.clone();
            let field = field.clone();
            (0_u64..=6 - x_degree).map(move |y_degree| {
                ring.from_u64_terms([(
                    vec![x_degree, y_degree, 6 - x_degree - y_degree],
                    field.one(),
                )])
                .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let source = FreeModule::new(ring.clone(), generators.len());
    let target = FreeModule::new(ring.clone(), 1);
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), [generators]).unwrap(),
    )
    .unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    (presentation, order)
}

fn mixed_rank_eight_presentation() -> (
    ModuleHomomorphism<PolynomialRing<FiniteField>>,
    ModuleMonomialOrder,
) {
    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(
        field,
        ["x0", "x1", "x2"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let x0 = ring.generator(0).unwrap();
    let x1 = ring.generator(1).unwrap();
    let x0_plus_x1 = (&x0 + &x1).unwrap();
    let zero = ring.zero();
    let local = [
        [x0.clone(), x0_plus_x1.clone(), x1.clone()],
        [zero.clone(), x0, x0_plus_x1],
    ];
    let blocks = 4;
    let mut rows = vec![vec![zero; 3 * blocks]; 2 * blocks];
    for new_block in 0..blocks {
        for old_block in 0..=new_block {
            for row in 0..2 {
                for column in 0..3 {
                    rows[2 * old_block + row][3 * new_block + column] = local[row][column].clone();
                }
            }
        }
    }
    let source = FreeModule::new(ring.clone(), 3 * blocks)
        .with_degree_shifts(
            1,
            (0..3 * blocks)
                .map(|_| Degree::new(vec![Integer::from(-1_i64)]))
                .collect(),
        )
        .unwrap();
    let target = FreeModule::new(ring.clone(), 2 * blocks)
        .with_degree_shifts(1, vec![Degree::zero(1); 2 * blocks])
        .unwrap();
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), rows).unwrap(),
    )
    .unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    (presentation, order)
}

fn noncyclic_embedded_rational_module() -> PresentedModule<PolynomialRing<noether::RationalField>> {
    let ring = PolynomialRing::new(
        Q,
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let variables = ring.generators().unwrap();
    let linear = (&variables[0] + &variables[1]).unwrap();
    let zero = ring.zero();
    let relations = FreeModule::new(ring.clone(), 3);
    let generators = FreeModule::new(ring.clone(), 2);
    PresentedModule::cokernel(
        ModuleHomomorphism::from_matrix(
            relations,
            generators,
            Matrix::from_rows(
                ring,
                [
                    [linear.clone(), zero.clone(), zero.clone()],
                    [zero, linear, variables[2].clone()],
                ],
            )
            .unwrap(),
        )
        .unwrap(),
    )
}

fn rational_source_changed_presentation(
    blocks: usize,
) -> (
    ModuleHomomorphism<PolynomialRing<noether::RationalField>>,
    ModuleMonomialOrder,
) {
    let ring = PolynomialRing::new(
        Q,
        ["x0", "x1", "x2"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let source_rank = 3 * blocks;
    let source_change = |row: usize, column: usize| {
        let Some(offset) = column.checked_sub(row) else {
            return Rational::zero();
        };
        if offset > 3 {
            return Rational::zero();
        }
        if offset == 0 {
            return Rational::one();
        }
        let numerator = ((row + 2) * (column + 3) + 5 * offset) % 11;
        let numerator = if numerator == 5 {
            if (row + column).is_multiple_of(2) {
                1_i64
            } else {
                -1_i64
            }
        } else {
            i64::try_from(numerator).unwrap() - 5
        };
        Rational::new(
            Integer::from(numerator),
            Integer::from([2_i64, 3, 5, 7][(2 * row + column) % 4]),
        )
        .unwrap()
    };
    let linear_form = |x0_coefficient: Rational, x1_coefficient: Rational| {
        ring.from_u64_terms([
            (vec![1, 0, 0], x0_coefficient),
            (vec![0, 1, 0], x1_coefficient),
        ])
        .unwrap()
    };
    let rows = (0..blocks)
        .flat_map(|block| {
            let first = 3 * block;
            [
                (0..source_rank)
                    .map(|column| {
                        linear_form(
                            source_change(first, column),
                            source_change(first + 1, column),
                        )
                    })
                    .collect::<Vec<_>>(),
                (0..source_rank)
                    .map(|column| {
                        linear_form(
                            source_change(first + 1, column),
                            source_change(first + 2, column),
                        )
                    })
                    .collect::<Vec<_>>(),
            ]
        })
        .collect::<Vec<_>>();
    let source = FreeModule::new(ring.clone(), source_rank);
    let target = FreeModule::new(ring.clone(), 2 * blocks);
    let presentation = ModuleHomomorphism::from_matrix(
        source,
        target,
        Matrix::from_rows(ring.clone(), rows).unwrap(),
    )
    .unwrap();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    (presentation, order)
}

fn bench_macaulay2_commalg(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("macaulay2-commalg");
    let cyclic_ring = PolynomialRing::new(
        Q,
        (0..6).map(|index| format!("x{index}")),
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let mut cyclic_generators = (1..6)
        .map(|degree| {
            cyclic_ring
                .from_u64_terms((0..6).map(|start| {
                    let mut exponent = vec![0_u64; 6];
                    for offset in 0..degree {
                        exponent[(start + offset) % 6] += 1;
                    }
                    (exponent, Rational::one())
                }))
                .unwrap()
        })
        .collect::<Vec<_>>();
    cyclic_generators.push(
        cyclic_ring
            .from_u64_terms([
                (vec![1_u64; 6], Rational::one()),
                (vec![0_u64; 6], -Rational::one()),
            ])
            .unwrap(),
    );
    group.bench_function("groebner-basis/cyclic-6-q", |b| {
        b.iter(|| {
            let ideal = black_box(&cyclic_ring)
                .ideal(black_box(cyclic_generators.iter().cloned()))
                .unwrap();
            black_box(ideal.generators().len())
        })
    });
    let (rational_presentation, rational_order) = semiregular_three_rational_presentation();
    let rational_image = Submodule::from_generator_map(
        rational_presentation.target().clone(),
        rational_presentation,
    )
    .unwrap();
    group.bench_function("groebner-basis/semiregular-3-q-transform", |b| {
        b.iter(|| {
            let basis = black_box(&rational_image)
                .groebner_basis(black_box(rational_order.clone()))
                .unwrap();
            black_box(basis.basis().generator_count())
        })
    });
    for blocks in [4_usize, 8] {
        let (presentation, order) = rational_source_changed_presentation(blocks);
        let image =
            Submodule::from_generator_map(presentation.target().clone(), presentation).unwrap();
        group.bench_function(
            format!(
                "module-groebner-basis/rational-source-changed-rank{}",
                2 * blocks
            ),
            |b| {
                b.iter(|| {
                    let basis = black_box(&image)
                        .groebner_basis(black_box(order.clone()))
                        .unwrap();
                    black_box(basis.basis().generator_count())
                })
            },
        );
    }

    let (presentation, order) = sixth_power_three_variable_presentation();
    let image =
        Submodule::from_generator_map(presentation.target().clone(), presentation.clone()).unwrap();
    let basis = image.groebner_basis(order.clone()).unwrap();
    let kernel = basis.syzygies().unwrap();
    group.bench_function("groebner-basis/power-ideal-3vars-degree6-f65521", |b| {
        b.iter(|| {
            let basis = black_box(&image)
                .groebner_basis(black_box(order.clone()))
                .unwrap();
            black_box(basis.basis().generator_count())
        })
    });
    group.bench_function("syzygy-from-basis/power-ideal-3vars-degree6-f65521", |b| {
        b.iter(|| {
            let kernel = black_box(&basis).syzygies().unwrap();
            black_box(kernel.generator_count())
        })
    });
    group.bench_function("syzygy/power-ideal-3vars-degree6-f65521", |b| {
        b.iter(|| {
            let kernel = black_box(&presentation)
                .kernel(black_box(order.clone()))
                .unwrap();
            black_box(kernel.generator_count())
        })
    });
    group.bench_function("dense-output/power-ideal-3vars-degree6-f65521", |b| {
        b.iter(|| black_box(&kernel).generator_map().matrix())
    });

    let (mixed_presentation, mixed_order) = mixed_rank_eight_presentation();
    let mixed_image = Submodule::from_generator_map(
        mixed_presentation.target().clone(),
        mixed_presentation.clone(),
    )
    .unwrap();
    let mixed_basis = mixed_image.groebner_basis(mixed_order.clone()).unwrap();
    let mixed_module = PresentedModule::cokernel(mixed_presentation.clone());
    group.bench_function("module-groebner-basis/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let basis = black_box(&mixed_image)
                .groebner_basis(black_box(mixed_order.clone()))
                .unwrap();
            black_box(basis.basis().generator_count())
        })
    });
    group.bench_function("module-minimal-generators/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let minimal = black_box(&mixed_image)
                .minimal_homogeneous_generators()
                .unwrap();
            black_box(minimal.generator_count())
        })
    });
    group.bench_function("module-syzygy-from-basis/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let kernel = black_box(&mixed_basis).syzygies().unwrap();
            black_box(kernel.generator_count())
        })
    });
    group.bench_function("module-syzygy/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let kernel = black_box(&mixed_presentation)
                .kernel(black_box(mixed_order.clone()))
                .unwrap();
            black_box(kernel.generator_count())
        })
    });

    let prime_chain_ring = PolynomialRing::new(
        FiniteField::prime(32_003_u64).unwrap(),
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let [prime_x, prime_y, _] = prime_chain_ring.generators().unwrap().try_into().unwrap();
    let prime_chain_ambient = FreeModule::new(prime_chain_ring.clone(), 2);
    let prime_chain = Submodule::from_generators(
        prime_chain_ambient.clone(),
        [
            [prime_x.clone(), prime_chain_ring.zero()],
            [prime_y.clone(), prime_x],
            [prime_chain_ring.zero(), prime_y],
        ]
        .into_iter()
        .map(|coordinates| {
            ModuleElement::from_coordinates(
                prime_chain_ambient.clone(),
                noether::linalg::Vector::from_values(prime_chain_ring.clone(), coordinates)
                    .unwrap(),
            )
            .unwrap()
        })
        .collect(),
    )
    .unwrap();
    let prime_chain_order =
        ModuleMonomialOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
    let prime_chain_basis = prime_chain
        .groebner_basis(prime_chain_order.clone())
        .unwrap();
    group.bench_function("module-groebner-basis/rank2-chain-f32003", |b| {
        b.iter(|| {
            black_box(&prime_chain)
                .groebner_basis(black_box(prime_chain_order.clone()))
                .unwrap()
        })
    });
    group.bench_function("module-syzygy-from-basis/rank2-chain-f32003", |b| {
        b.iter(|| black_box(&prime_chain_basis).syzygies().unwrap())
    });
    group.bench_function("module-groebner-syzygy/rank2-chain-f32003", |b| {
        b.iter(|| {
            let basis = black_box(&prime_chain)
                .groebner_basis(black_box(prime_chain_order.clone()))
                .unwrap();
            let syzygies = basis.syzygies().unwrap();
            (basis, syzygies)
        })
    });

    group.bench_function("module-betti/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let betti = black_box(&mixed_module)
                .graded_betti_table(black_box(mixed_order.clone()))
                .unwrap();
            black_box(betti.projective_dimension())
        })
    });
    group.bench_function("module-resolution/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let resolution = black_box(&mixed_module)
                .free_resolution(black_box(mixed_order.clone()))
                .unwrap();
            black_box(resolution.length())
        })
    });
    group.bench_function("module-resolution-through1/mixed-chain-rank8-f65521", |b| {
        b.iter(|| {
            let resolution = black_box(&mixed_module)
                .minimal_free_resolution_through(black_box(mixed_order.clone()), 1)
                .unwrap();
            black_box(resolution.length())
        })
    });

    let prime_twisted_cubic_ring = PolynomialRing::new(
        FiniteField::prime(32_003_u64).unwrap(),
        ["x", "y", "z", "w"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let (prime_twisted_cubic, prime_twisted_cubic_order) =
        twisted_cubic_module(&prime_twisted_cubic_ring);
    group.bench_function("minimal-resolution/twisted-cubic-f32003-kernel", |b| {
        b.iter(|| {
            black_box(&prime_twisted_cubic)
                .minimal_free_resolution(black_box(prime_twisted_cubic_order.clone()))
                .unwrap()
        })
    });
    group.bench_function(
        "minimal-resolution/twisted-cubic-f32003-construction",
        |b| b.iter(|| black_box(twisted_cubic_module(black_box(&prime_twisted_cubic_ring)))),
    );
    group.bench_function("minimal-resolution/twisted-cubic-f32003-scorecard", |b| {
        b.iter(|| {
            let (module, order) = twisted_cubic_module(black_box(&prime_twisted_cubic_ring));
            black_box(module.minimal_free_resolution(order).unwrap())
        })
    });

    let prime_generic_minors_ring = PolynomialRing::new(
        FiniteField::prime(32_003_u64).unwrap(),
        ["a", "b", "c", "d", "e", "f"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    group.bench_function("minimal-resolution/generic-2x3-minors-f32003", |b| {
        b.iter(|| {
            let (module, order) =
                generic_two_by_three_minors_module(black_box(&prime_generic_minors_ring));
            black_box(module.minimal_free_resolution(order).unwrap())
        })
    });

    let rational_cycle_ring = PolynomialRing::new(
        Q,
        ["a", "b", "c", "d", "e"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let (rational_cycle, rational_cycle_order) = cycle_edge_module(&rational_cycle_ring);
    let rational_cycle_resolution = rational_cycle
        .free_resolution(rational_cycle_order.clone())
        .unwrap();
    group.bench_function("minimal-resolution/cycle5-q", |b| {
        b.iter(|| {
            let resolution = black_box(&rational_cycle)
                .minimal_free_resolution(black_box(rational_cycle_order.clone()))
                .unwrap();
            black_box(resolution.length())
        })
    });
    group.bench_function("minimalize/cycle5-q", |b| {
        b.iter(|| {
            let resolution = black_box(&rational_cycle_resolution).minimalize().unwrap();
            black_box(resolution.length())
        })
    });
    group.bench_function("betti/cycle5-q", |b| {
        b.iter(|| {
            let table = black_box(&rational_cycle)
                .graded_betti_table(black_box(rational_cycle_order.clone()))
                .unwrap();
            black_box(table.projective_dimension())
        })
    });
    let prime_cycle_ring = PolynomialRing::new(
        FiniteField::prime(65_521_u64).unwrap(),
        ["a", "b", "c", "d", "e", "f"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let prime_cycle_five_ring = PolynomialRing::new(
        FiniteField::prime(32_003_u64).unwrap(),
        ["a", "b", "c", "d", "e"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let (prime_cycle_five, prime_cycle_five_order) = cycle_edge_module(&prime_cycle_five_ring);
    group.bench_function("minimal-resolution/cycle5-f32003", |b| {
        b.iter(|| {
            let resolution = black_box(&prime_cycle_five)
                .minimal_free_resolution(black_box(prime_cycle_five_order.clone()))
                .unwrap();
            black_box(resolution.length())
        })
    });
    let (prime_cycle, prime_cycle_order) = cycle_edge_module(&prime_cycle_ring);
    group.bench_function("minimal-resolution/cycle6-f65521", |b| {
        b.iter(|| {
            let resolution = black_box(&prime_cycle)
                .minimal_free_resolution(black_box(prime_cycle_order.clone()))
                .unwrap();
            black_box(resolution.length())
        })
    });

    let associated_prime_module = noncyclic_embedded_rational_module();
    group.bench_function("module-associated-primes/noncyclic-embedded-q", |b| {
        b.iter(|| {
            let associated = black_box(&associated_prime_module)
                .associated_primes()
                .unwrap();
            black_box(associated.prime_certificates().len())
        })
    });
    group.bench_function("module-primary-decomposition/noncyclic-embedded-q", |b| {
        b.iter(|| {
            let decomposition = black_box(&associated_prime_module)
                .primary_decomposition()
                .unwrap();
            black_box(decomposition.components().len())
        })
    });

    let decomposition_ring = PolynomialRing::new(
        Q,
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let decomposition_variables = decomposition_ring.generators().unwrap();
    let linear = (&decomposition_variables[0] + &decomposition_variables[1]).unwrap();
    let embedded_source = decomposition_ring
        .ideal([
            linear.pow(2).unwrap(),
            (&linear * &decomposition_variables[2]).unwrap(),
        ])
        .unwrap();
    group.bench_function("associated-primes/nonmonomial-embedded-q", |b| {
        b.iter(|| {
            let associated = black_box(&embedded_source).associated_primes().unwrap();
            black_box(associated.prime_certificates().len())
        })
    });
    group.bench_function("primary-decomposition/nonmonomial-embedded-q", |b| {
        b.iter(|| {
            let decomposition = black_box(&embedded_source).primary_decomposition().unwrap();
            black_box(decomposition.components().len())
        })
    });

    let incomparable_minimal = decomposition_ring.ideal([linear.clone()]).unwrap();
    let incomparable_first = decomposition_ring
        .ideal([linear.clone(), decomposition_variables[1].clone()])
        .unwrap()
        .pow(2)
        .unwrap();
    let incomparable_second = decomposition_ring
        .ideal([linear.clone(), decomposition_variables[2].clone()])
        .unwrap()
        .pow(2)
        .unwrap();
    let incomparable_embedded_source = incomparable_minimal
        .intersection(&incomparable_first)
        .unwrap()
        .intersection(&incomparable_second)
        .unwrap();
    group.bench_function(
        "primary-decomposition/incomparable-nonmonomial-embedded-q",
        |b| {
            b.iter(|| {
                let decomposition = black_box(&incomparable_embedded_source)
                    .primary_decomposition()
                    .unwrap();
                black_box(decomposition.components().len())
            })
        },
    );

    let top_component = decomposition_ring
        .ideal([
            decomposition_variables[0].pow(2).unwrap(),
            decomposition_variables[1].pow(2).unwrap(),
        ])
        .unwrap();
    let closed_component = decomposition_ring
        .ideal([
            decomposition_variables[0].pow(3).unwrap(),
            decomposition_variables[1].pow(3).unwrap(),
            decomposition_variables[2].pow(3).unwrap(),
        ])
        .unwrap();
    let mixed_dimensional_source = top_component.intersection(&closed_component).unwrap();
    group.bench_function("equidimensional-hull/nonradical-mixed-dimensional-q", |b| {
        b.iter(|| {
            let hull = black_box(&mixed_dimensional_source)
                .equidimensional_hull()
                .unwrap();
            black_box(hull.ideal().generators().len())
        })
    });

    let normalization_ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let normalization_variables = normalization_ring.generators().unwrap();
    let hyperbola = normalization_ring
        .ideal([normalization_ring
            .from_u64_terms([
                (vec![1, 1], Rational::one()),
                (vec![0, 0], -Rational::one()),
            ])
            .unwrap()])
        .unwrap();
    group.bench_function("noether-normalization/hyperbola-q", |b| {
        b.iter(|| {
            let normalization = black_box(&hyperbola).noether_normalization().unwrap();
            black_box(normalization.module_generator_count())
        })
    });

    let artinian = normalization_ring
        .ideal([
            normalization_variables[0].pow(3).unwrap(),
            normalization_variables[1].pow(2).unwrap(),
        ])
        .unwrap();
    group.bench_function("noether-normalization/artinian-3x2-q", |b| {
        b.iter(|| {
            let normalization = black_box(&artinian).noether_normalization().unwrap();
            black_box(normalization.module_generator_count())
        })
    });

    let cusp = normalization_ring
        .ideal([normalization_ring
            .sub(
                &normalization_variables[1].pow(2).unwrap(),
                &normalization_variables[0].pow(3).unwrap(),
            )
            .unwrap()])
        .unwrap();
    group.bench_function("affine-normalization/cusp-q", |b| {
        b.iter(|| {
            let normalization = black_box(&cusp).normalization().unwrap();
            black_box(normalization.source_conductor().generators().len())
        })
    });

    let monomial_curve = normalization_ring
        .ideal([normalization_ring
            .sub(
                &normalization_variables[1].pow(3).unwrap(),
                &normalization_variables[0].pow(4).unwrap(),
            )
            .unwrap()])
        .unwrap();
    group.bench_function("affine-normalization/monomial-3-4-q", |b| {
        b.iter(|| {
            let normalization = black_box(&monomial_curve).normalization().unwrap();
            black_box(normalization.source_conductor().generators().len())
        })
    });

    let polynomial_graph = normalization_ring
        .ideal([normalization_ring
            .sub(
                &normalization_variables[1],
                &normalization_variables[0].pow(2).unwrap(),
            )
            .unwrap()])
        .unwrap();
    group.bench_function("affine-normalization/parabola-q", |b| {
        b.iter(|| {
            let normalization = black_box(&polynomial_graph).normalization().unwrap();
            black_box(normalization.is_normal())
        })
    });

    let node = normalization_ring
        .ideal([normalization_ring
            .mul(&normalization_variables[0], &normalization_variables[1])
            .unwrap()])
        .unwrap();
    group.bench_function("reduced-normalization/node-q", |b| {
        b.iter(|| {
            let normalization = black_box(&node).reduced_normalization().unwrap();
            black_box(normalization.source_conductor().generators().len())
        })
    });

    let cusp_and_line = normalization_ring
        .ideal([normalization_ring
            .mul(
                &normalization_variables[1],
                &normalization_ring
                    .sub(
                        &normalization_variables[1].pow(2).unwrap(),
                        &normalization_variables[0].pow(3).unwrap(),
                    )
                    .unwrap(),
            )
            .unwrap()])
        .unwrap();
    group.bench_function("reduced-normalization/cusp-and-line-q", |b| {
        b.iter(|| {
            let normalization = black_box(&cusp_and_line).reduced_normalization().unwrap();
            black_box(normalization.source_conductor().generators().len())
        })
    });

    // Pinned with scripts/bench-pin.sh, these three cases separate the exact
    // kernels behind the public cotangent surface: ideal/module Gröbner work
    // for Omega and I/I², symbolic differentiation/Jacobian construction, and
    // exact field rank for the fiber. The quotient itself is prepared outside
    // the timed loop so the measurements are construction kernels, not setup.
    let differential_ring = PolynomialRing::new(
        Q,
        ["x", "y", "z"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let differential_variables = differential_ring.generators().unwrap();
    let cusp = differential_ring
        .sub(
            &differential_variables[1].pow(2).unwrap(),
            &differential_variables[0].pow(3).unwrap(),
        )
        .unwrap();
    let graph = differential_ring
        .sub(
            &differential_variables[2],
            &differential_ring
                .mul(&differential_variables[0], &differential_variables[1])
                .unwrap(),
        )
        .unwrap();
    let differential_equations = vec![cusp, graph];
    let differential_ideal = differential_ring
        .ideal(differential_equations.clone())
        .unwrap();
    let differential_algebra = differential_ring.quotient(&differential_ideal).unwrap();
    group.bench_function("kahler-differentials/ci-curve-q", |b| {
        b.iter(|| {
            let omega = black_box(&differential_algebra)
                .kahler_differentials()
                .unwrap();
            black_box(omega.module().relation_count())
        })
    });
    group.bench_function("conormal-sequence/ci-curve-q", |b| {
        b.iter(|| {
            let evidence = black_box(&differential_algebra)
                .conormal_sequence_from_generators(black_box(&differential_equations))
                .unwrap();
            black_box(evidence.conormal_module().relation_count())
        })
    });
    group.bench_function("cotangent-space/ci-curve-smooth-point-q", |b| {
        b.iter(|| {
            let fiber = black_box(&differential_algebra)
                .cotangent_space_at(black_box(vec![
                    Rational::one(),
                    Rational::one(),
                    Rational::one(),
                ]))
                .unwrap();
            black_box(fiber.embedding_dimension())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_macaulay2_commalg);
criterion_main!(benches);
