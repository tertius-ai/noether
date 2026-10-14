// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use groebner_kernel::{
    fglm_basis, groebner_basis_prime, groebner_basis_prime_profiled, groebner_basis_rational,
    groebner_basis_with_transform_rational, module_groebner_basis_prime,
    module_groebner_basis_with_transform, module_groebner_basis_with_transform_binary_extension,
    module_groebner_basis_with_transform_f2, module_groebner_basis_with_transform_prime,
    module_groebner_basis_with_transform_rational, module_membership, module_normal_forms,
    module_relations_from_groebner_data_prime, module_schreyer_syzygies_prime,
    module_schreyer_syzygies_prime_with_hilbert_target, rational_univariate_representation,
    BinaryExtensionField16, BinaryField, Field, GroebnerOptions, GroebnerPhaseTimings, Integer,
    Monomial, MonomialOrder, PrimeField32, Rational, SparseModuleElement, SparseModuleOrder,
    SparsePolynomial, StandardModuleHilbertTarget,
};
use poly_core::CoefficientRing;

struct RationalRing;

impl CoefficientRing for RationalRing {
    type Element = Rational;

    fn zero(&self) -> Self::Element {
        Rational::zero()
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.is_zero()
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left + right
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        left * right
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        -value
    }
}

impl Field for RationalRing {
    fn one(&self) -> Self::Element {
        Rational::one()
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        value.inverse().expect("a module pivot is nonzero")
    }
}

fn cyclic(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
    let order = MonomialOrder::GradedReverseLexicographic;
    let mut generators = Vec::with_capacity(size);
    for degree in 1..size {
        generators.push(SparsePolynomial::new(
            size,
            order,
            (0..size).map(|start| {
                let mut exponent = vec![0_u32; size];
                for offset in 0..degree {
                    exponent[(start + offset) % size] += 1;
                }
                (Monomial::new(exponent.into_boxed_slice()), 1_u32)
            }),
            field,
        ));
    }
    generators.push(SparsePolynomial::new(
        size,
        order,
        [
            (Monomial::new(vec![1_u32; size].into_boxed_slice()), 1_u32),
            (Monomial::one(size), field.modulus() - 1),
        ],
        field,
    ));
    generators
}

fn power_cycle(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
    let order = MonomialOrder::GradedReverseLexicographic;
    (0..size)
        .map(|variable| {
            let mut square = vec![0_u32; size];
            square[variable] = 2;
            let mut next = vec![0_u32; size];
            next[(variable + 1) % size] = 1;
            SparsePolynomial::new(
                size,
                order,
                [
                    (Monomial::new(square.into_boxed_slice()), 1_u32),
                    (Monomial::new(next.into_boxed_slice()), field.modulus() - 1),
                ],
                field,
            )
        })
        .collect()
}

fn rank_two_chain(field: &PrimeField32) -> Vec<SparseModuleElement<u32>> {
    let order = MonomialOrder::GradedReverseLexicographic;
    let polynomial = |terms| SparsePolynomial::new(3, order, terms, field);
    let zero = || polynomial(Vec::<(Monomial, u32)>::new());
    let x = || polynomial(vec![(Monomial::new(vec![1, 0, 0].into_boxed_slice()), 1)]);
    let y = || polynomial(vec![(Monomial::new(vec![0, 1, 0].into_boxed_slice()), 1)]);
    vec![
        SparseModuleElement::from_coordinates(vec![x(), zero()], field),
        SparseModuleElement::from_coordinates(vec![y(), x()], field),
        SparseModuleElement::from_coordinates(vec![zero(), y()], field),
    ]
}

fn cyclic_rational(size: usize) -> Vec<SparsePolynomial<Rational>> {
    let order = MonomialOrder::GradedReverseLexicographic;
    let mut generators = Vec::with_capacity(size);
    for degree in 1..size {
        generators.push(SparsePolynomial::new(
            size,
            order,
            (0..size).map(|start| {
                let mut exponent = vec![0_u32; size];
                for offset in 0..degree {
                    exponent[(start + offset) % size] += 1;
                }
                (Monomial::new(exponent.into_boxed_slice()), Rational::one())
            }),
            &RationalRing,
        ));
    }
    generators.push(SparsePolynomial::new(
        size,
        order,
        [
            (
                Monomial::new(vec![1_u32; size].into_boxed_slice()),
                Rational::one(),
            ),
            (Monomial::one(size), -Rational::one()),
        ],
        &RationalRing,
    ));
    generators
}

fn semiregular_rational(size: usize) -> Vec<SparsePolynomial<Rational>> {
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
    (0..size)
        .map(|_| {
            let mut terms = Vec::new();
            for left in 0..size {
                for right in left..size {
                    let mut exponent = vec![0_u32; size];
                    exponent[left] += 1;
                    exponent[right] += 1;
                    terms.push((Monomial::new(exponent.into_boxed_slice()), coefficient()));
                }
            }
            for variable in 0..size {
                let mut exponent = vec![0_u32; size];
                exponent[variable] = 1;
                terms.push((Monomial::new(exponent.into_boxed_slice()), coefficient()));
            }
            terms.push((Monomial::one(size), coefficient()));
            SparsePolynomial::new(
                size,
                MonomialOrder::GradedReverseLexicographic,
                terms,
                &RationalRing,
            )
        })
        .collect()
}

fn rational_source_changed_module_chains(blocks: usize) -> Vec<SparseModuleElement<Rational>> {
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
        let denominator = [2_i64, 3, 5, 7][(2 * row + column) % 4];
        Rational::new(Integer::from(numerator), Integer::from(denominator)).unwrap()
    };
    let linear_form = |x0_coefficient: Rational, x1_coefficient: Rational| {
        SparsePolynomial::new(
            3,
            MonomialOrder::GradedReverseLexicographic,
            [
                (
                    Monomial::new(vec![1, 0, 0].into_boxed_slice()),
                    x0_coefficient,
                ),
                (
                    Monomial::new(vec![0, 1, 0].into_boxed_slice()),
                    x1_coefficient,
                ),
            ],
            &RationalRing,
        )
    };
    (0..source_rank)
        .map(|column| {
            let coordinates = (0..blocks)
                .flat_map(|block| {
                    let first = 3 * block;
                    [
                        linear_form(
                            source_change(first, column),
                            source_change(first + 1, column),
                        ),
                        linear_form(
                            source_change(first + 1, column),
                            source_change(first + 2, column),
                        ),
                    ]
                })
                .collect();
            SparseModuleElement::from_coordinates(coordinates, &RationalRing)
        })
        .collect()
}

fn groebner_benchmark(criterion: &mut Criterion) {
    let field = PrimeField32::new(65_521).unwrap();
    let options = GroebnerOptions::default();
    let mut group = criterion.benchmark_group("prime_field_f4");
    for size in [4, 5, 6] {
        let generators = cyclic(size, &field);
        group.bench_with_input(BenchmarkId::new("cyclic", size), &size, |bencher, _| {
            bencher
                .iter(|| black_box(groebner_basis_prime(&field, &generators, &options).unwrap()));
        });
    }
    group.finish();

    let module_polynomial = |terms: &[(Vec<u32>, u32)]| {
        SparsePolynomial::new(
            3,
            MonomialOrder::GradedReverseLexicographic,
            terms.iter().cloned().map(|(exponents, coefficient)| {
                (Monomial::new(exponents.into_boxed_slice()), coefficient)
            }),
            &field,
        )
    };
    let zero = module_polynomial(&[]);
    let module_generators = vec![
        SparseModuleElement::from_coordinates(
            vec![
                module_polynomial(&[(vec![2, 0, 0], 1), (vec![0, 1, 0], 1)]),
                module_polynomial(&[(vec![0, 0, 1], 1)]),
            ],
            &field,
        ),
        SparseModuleElement::from_coordinates(
            vec![
                module_polynomial(&[(vec![1, 1, 0], 1)]),
                module_polynomial(&[(vec![1, 0, 0], 1)]),
            ],
            &field,
        ),
        SparseModuleElement::from_coordinates(
            vec![
                zero,
                module_polynomial(&[(vec![0, 2, 0], 1), (vec![0, 0, 1], 1)]),
            ],
            &field,
        ),
    ];
    criterion.bench_function("module_f4/rank-2-witness", |bencher| {
        bencher.iter(|| {
            black_box(
                module_groebner_basis_with_transform_prime(
                    &field,
                    &module_generators,
                    &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
                    &options,
                )
                .unwrap(),
            )
        });
    });
    let module_order =
        SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
    let module_basis = module_groebner_basis_with_transform_prime(
        &field,
        &module_generators,
        &module_order,
        &options,
    )
    .unwrap()
    .basis;
    criterion.bench_function("module_f4/rank-2-syzygies", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime(&field, &module_basis, &module_order, &options)
                    .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/rank-2-membership-batch", |bencher| {
        bencher.iter(|| {
            black_box(
                module_membership(
                    &field,
                    &module_basis,
                    &module_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });

    let mixed_rank8_generators = (0..4)
        .flat_map(|new_block| {
            (0..3).map(move |local_column| {
                let coordinates = (0..8)
                    .map(|component| {
                        let old_block = component / 2;
                        let local_row = component % 2;
                        if old_block > new_block {
                            return module_polynomial(&[]);
                        }
                        match (local_row, local_column) {
                            (0, 0) | (1, 1) => module_polynomial(&[(vec![1, 0, 0], 1)]),
                            (0, 1) | (1, 2) => {
                                module_polynomial(&[(vec![1, 0, 0], 1), (vec![0, 1, 0], 1)])
                            }
                            (0, 2) => module_polynomial(&[(vec![0, 1, 0], 1)]),
                            (1, 0) => module_polynomial(&[]),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                SparseModuleElement::from_coordinates(coordinates, &field)
            })
        })
        .collect::<Vec<_>>();
    let mixed_rank8_output = module_groebner_basis_with_transform_prime(
        &field,
        &mixed_rank8_generators,
        &module_order,
        &options,
    )
    .unwrap();
    let mixed_rank8_transformations = mixed_rank8_output.transformations.clone();
    let mixed_rank8_basis = mixed_rank8_output.basis;
    assert_eq!(
        module_groebner_basis_prime(&field, &mixed_rank8_generators, &module_order, &options,)
            .unwrap()
            .basis,
        mixed_rank8_basis
    );
    criterion.bench_function("module_f4/mixed-chain-rank8-basis-only", |bencher| {
        bencher.iter(|| {
            black_box(
                module_groebner_basis_prime(
                    &field,
                    &mixed_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-witness", |bencher| {
        bencher.iter(|| {
            black_box(
                module_groebner_basis_with_transform_prime(
                    &field,
                    &mixed_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-schreyer", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime(&field, &mixed_rank8_basis, &module_order, &options)
                    .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-membership", |bencher| {
        bencher.iter(|| {
            black_box(
                module_membership(
                    &field,
                    &mixed_rank8_basis,
                    &mixed_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-lifts", |bencher| {
        bencher.iter(|| {
            black_box(
                module_normal_forms(
                    &field,
                    &mixed_rank8_basis,
                    &mixed_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-original-kernel", |bencher| {
        bencher.iter(|| {
            black_box(
                module_relations_from_groebner_data_prime(
                    &field,
                    &mixed_rank8_basis,
                    &mixed_rank8_transformations,
                    &mixed_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });

    let binary = BinaryField;
    let binary_polynomial = |terms: &[(Vec<u32>, u8)]| {
        SparsePolynomial::new(
            3,
            MonomialOrder::GradedReverseLexicographic,
            terms.iter().cloned().map(|(exponents, coefficient)| {
                (Monomial::new(exponents.into_boxed_slice()), coefficient)
            }),
            &binary,
        )
    };
    let binary_rank8_generators = (0..4)
        .flat_map(|new_block| {
            (0..3).map(move |local_column| {
                let coordinates = (0..8)
                    .map(|component| {
                        let old_block = component / 2;
                        let local_row = component % 2;
                        if old_block > new_block {
                            return binary_polynomial(&[]);
                        }
                        match (local_row, local_column) {
                            (0, 0) | (1, 1) => binary_polynomial(&[(vec![1, 0, 0], 1)]),
                            (0, 1) | (1, 2) => {
                                binary_polynomial(&[(vec![1, 0, 0], 1), (vec![0, 1, 0], 1)])
                            }
                            (0, 2) => binary_polynomial(&[(vec![0, 1, 0], 1)]),
                            (1, 0) => binary_polynomial(&[]),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                SparseModuleElement::from_coordinates(coordinates, &binary)
            })
        })
        .collect::<Vec<_>>();
    criterion.bench_function("module_f4/mixed-chain-rank8-f2-generic", |bencher| {
        bencher.iter(|| {
            black_box(
                module_groebner_basis_with_transform(
                    &binary,
                    &binary_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/mixed-chain-rank8-f2-packed", |bencher| {
        bencher.iter(|| {
            black_box(
                module_groebner_basis_with_transform_f2(
                    &binary_rank8_generators,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });

    let binary_extension = BinaryExtensionField16::new(0b1_0011).unwrap();
    let binary_extension_ref = &binary_extension;
    let extension_polynomial = |terms: &[(Vec<u32>, u16)]| {
        SparsePolynomial::new(
            3,
            MonomialOrder::GradedReverseLexicographic,
            terms.iter().cloned().map(|(exponents, coefficient)| {
                (Monomial::new(exponents.into_boxed_slice()), coefficient)
            }),
            binary_extension_ref,
        )
    };
    let extension_rank8_generators = (0..4)
        .flat_map(|new_block| {
            (0..3).map(move |local_column| {
                let coordinates = (0..8)
                    .map(|component| {
                        let old_block = component / 2;
                        let local_row = component % 2;
                        if old_block > new_block {
                            return extension_polynomial(&[]);
                        }
                        match (local_row, local_column) {
                            (0, 0) | (1, 1) => extension_polynomial(&[(vec![1, 0, 0], 2)]),
                            (0, 1) | (1, 2) => {
                                extension_polynomial(&[(vec![1, 0, 0], 3), (vec![0, 1, 0], 2)])
                            }
                            (0, 2) => extension_polynomial(&[(vec![0, 1, 0], 1)]),
                            (1, 0) => extension_polynomial(&[]),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                SparseModuleElement::from_coordinates(coordinates, binary_extension_ref)
            })
        })
        .collect::<Vec<_>>();
    criterion.bench_function(
        "module_f4/mixed-chain-rank8-binary-extension-generic",
        |bencher| {
            bencher.iter(|| {
                black_box(
                    module_groebner_basis_with_transform(
                        &binary_extension,
                        &extension_rank8_generators,
                        &module_order,
                        &options,
                    )
                    .unwrap(),
                )
            });
        },
    );
    criterion.bench_function(
        "module_f4/mixed-chain-rank8-binary-extension-packed",
        |bencher| {
            bencher.iter(|| {
                black_box(
                    module_groebner_basis_with_transform_binary_extension(
                        &binary_extension,
                        &extension_rank8_generators,
                        &module_order,
                        &options,
                    )
                    .unwrap(),
                )
            });
        },
    );

    let single_coordinate = |exponents| {
        SparseModuleElement::from_coordinates(vec![module_polynomial(&[(exponents, 1)])], &field)
    };
    let redundant_pair_basis = vec![
        single_coordinate(vec![2, 0, 0]),
        single_coordinate(vec![1, 1, 0]),
        single_coordinate(vec![0, 2, 0]),
    ];
    let redundant_pair_target = StandardModuleHilbertTarget::new(
        vec![2, 2, 2],
        vec![
            Integer::zero(),
            Integer::zero(),
            Integer::from(3),
            Integer::from(-2),
        ],
    );
    criterion.bench_function("module_f4/redundant-schreyer-cell/ordinary", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime(
                    &field,
                    &redundant_pair_basis,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/redundant-schreyer-cell/hilbert", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime_with_hilbert_target(
                    &field,
                    &redundant_pair_basis,
                    &module_order,
                    &options,
                    &redundant_pair_target,
                )
                .unwrap(),
            )
        });
    });
    let seventh_power_basis = (0..=7)
        .map(|y_degree| single_coordinate(vec![7 - y_degree, y_degree, 0]))
        .collect::<Vec<_>>();
    let seventh_power_target = StandardModuleHilbertTarget::new(
        vec![7; 8],
        vec![
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
            Integer::zero(),
            Integer::from(8),
            Integer::from(-7),
        ],
    );
    criterion.bench_function("module_f4/seventh-power-schreyer/ordinary", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime(
                    &field,
                    &seventh_power_basis,
                    &module_order,
                    &options,
                )
                .unwrap(),
            )
        });
    });
    criterion.bench_function("module_f4/seventh-power-schreyer/hilbert", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime_with_hilbert_target(
                    &field,
                    &seventh_power_basis,
                    &module_order,
                    &options,
                    &seventh_power_target,
                )
                .unwrap(),
            )
        });
    });

    // All first syzygies of the homogeneous maximal ideal land in one degree
    // cell.  Unlike the small public-surface cases above, this is large enough
    // to expose scheduler and row-reducer crossover costs.
    for variables in [16_usize, 24] {
        let linear_basis = (0..variables)
            .map(|variable| {
                let mut exponent = vec![0_u32; variables];
                exponent[variable] = 1;
                SparseModuleElement::from_coordinates(
                    vec![SparsePolynomial::new(
                        variables,
                        MonomialOrder::GradedReverseLexicographic,
                        [(Monomial::new(exponent.into_boxed_slice()), 1_u32)],
                        &field,
                    )],
                    &field,
                )
            })
            .collect::<Vec<_>>();
        let linear_order =
            SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        criterion.bench_function(
            &format!("module_f4/maximal-ideal-schreyer/{variables}"),
            |bencher| {
                bencher.iter(|| {
                    black_box(
                        module_schreyer_syzygies_prime(
                            &field,
                            &linear_basis,
                            &linear_order,
                            &options,
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    let weighted_pure_power_basis = (0..24)
        .map(|variable| {
            let mut exponent = vec![0_u32; 24];
            exponent[variable] = 1 + (variable % 6) as u32;
            SparseModuleElement::from_coordinates(
                vec![SparsePolynomial::new(
                    24,
                    MonomialOrder::GradedReverseLexicographic,
                    [(Monomial::new(exponent.into_boxed_slice()), 1_u32)],
                    &field,
                )],
                &field,
            )
        })
        .collect::<Vec<_>>();
    criterion.bench_function("module_f4/weighted-pure-powers-schreyer/24", |bencher| {
        bencher.iter(|| {
            black_box(
                module_schreyer_syzygies_prime(
                    &field,
                    &weighted_pure_power_basis,
                    &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
                    &options,
                )
                .unwrap(),
            )
        });
    });

    let sixth_power_three_variables = (0..=6)
        .flat_map(|x_degree| {
            (0..=(6 - x_degree)).map(move |y_degree| {
                let exponent = vec![x_degree, y_degree, 6 - x_degree - y_degree];
                SparseModuleElement::from_coordinates(
                    vec![SparsePolynomial::new(
                        3,
                        MonomialOrder::GradedReverseLexicographic,
                        [(Monomial::new(exponent.into_boxed_slice()), 1_u32)],
                        &field,
                    )],
                    &field,
                )
            })
        })
        .collect::<Vec<_>>();
    criterion.bench_function(
        "module_f4/three-variable-sixth-power-schreyer/ordinary",
        |bencher| {
            bencher.iter(|| {
                black_box(
                    module_schreyer_syzygies_prime(
                        &field,
                        &sixth_power_three_variables,
                        &module_order,
                        &options,
                    )
                    .unwrap(),
                )
            });
        },
    );

    for size in [4_usize, 6] {
        let rational_generators = cyclic_rational(size);
        criterion.bench_function(&format!("rational_f4/cyclic/{size}"), |bencher| {
            bencher.iter(|| {
                black_box(
                    groebner_basis_rational(&rational_generators, &GroebnerOptions::default())
                        .unwrap(),
                )
            });
        });
    }
    let semiregular_rational = semiregular_rational(3);
    criterion.bench_function("rational_f4/semiregular-3-transform", |bencher| {
        bencher.iter(|| {
            black_box(
                groebner_basis_with_transform_rational(
                    &semiregular_rational,
                    &GroebnerOptions::default(),
                )
                .unwrap(),
            )
        });
    });
    let mut rational_modules = criterion.benchmark_group("rational_module_f4");
    rational_modules.sample_size(20);
    for blocks in [4_usize, 8] {
        let generators = rational_source_changed_module_chains(blocks);
        rational_modules.bench_with_input(
            BenchmarkId::new("source-changed-chain-witness", 2 * blocks),
            &blocks,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        module_groebner_basis_with_transform_rational(
                            &generators,
                            &SparseModuleOrder::TermOverPosition(
                                MonomialOrder::GradedReverseLexicographic,
                            ),
                            &GroebnerOptions::default(),
                        )
                        .unwrap(),
                    )
                });
            },
        );
    }
    rational_modules.finish();

    let generators = power_cycle(4, &field);
    let source = groebner_basis_prime(&field, &generators, &options)
        .unwrap()
        .basis;
    criterion.bench_function("fglm/power-cycle-4/grevlex-to-lex", |bencher| {
        bencher.iter(|| {
            black_box(fglm_basis(&field, &source, MonomialOrder::Lexicographic, &options).unwrap())
        });
    });
    criterion.bench_function("rur/power-cycle-4", |bencher| {
        bencher.iter(|| {
            black_box(rational_univariate_representation(&field, &source, &options).unwrap())
        });
    });

    let phase_generators = cyclic(6, &field);
    let mut phases = criterion.benchmark_group("prime_field_f4_phases/cyclic-6");
    phases.sample_size(10);
    for (name, select) in [
        (
            "pair-selection",
            (|profile: &GroebnerPhaseTimings| profile.pair_selection)
                as fn(&GroebnerPhaseTimings) -> std::time::Duration,
        ),
        (
            "symbolic-preprocessing",
            |profile: &GroebnerPhaseTimings| profile.symbolic_preprocessing,
        ),
        (
            "monomial-table-and-row-assembly",
            |profile: &GroebnerPhaseTimings| profile.matrix_construction,
        ),
        ("modular-row-reduction", |profile: &GroebnerPhaseTimings| {
            profile.row_reduction
        }),
        ("new-basis-extraction", |profile: &GroebnerPhaseTimings| {
            profile.basis_extraction
        }),
        (
            "final-canonicalization-and-verification",
            |profile: &GroebnerPhaseTimings| profile.final_canonicalization_and_verification,
        ),
    ] {
        phases.bench_function(name, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = std::time::Duration::ZERO;
                for _ in 0..iterations {
                    let (output, profile) =
                        groebner_basis_prime_profiled(&field, &phase_generators, &options).unwrap();
                    black_box(output.basis.len());
                    elapsed += select(&profile);
                }
                elapsed
            })
        });
    }
    phases.finish();

    let module_prime = PrimeField32::new(32_003).unwrap();
    let module_generators = rank_two_chain(&module_prime);
    let module_order =
        SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
    criterion.bench_function(
        "module-groebner-with-transform/rank2-chain-f32003",
        |bencher| {
            bencher.iter(|| {
                black_box(
                    module_groebner_basis_with_transform_prime(
                        black_box(&module_prime),
                        black_box(&module_generators),
                        black_box(&module_order),
                        black_box(&options),
                    )
                    .unwrap(),
                )
            });
        },
    );
}

criterion_group!(benches, groebner_benchmark);
criterion_main!(benches);
