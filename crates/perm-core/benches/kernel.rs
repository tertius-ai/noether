// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use perm_core::{is_nilpotent, is_perfect, is_solvable, Perm, PermGroup, Point};

fn permutation(images: Vec<usize>) -> Perm {
    Perm::from_images(
        &images
            .into_iter()
            .map(|image| image as Point)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn symmetric_generators(degree: usize) -> [Perm; 2] {
    let mut transposition = (0..degree).collect::<Vec<_>>();
    transposition.swap(0, 1);
    let cycle = (0..degree).map(|point| (point + 1) % degree).collect();
    [permutation(transposition), permutation(cycle)]
}

fn dihedral_generators(degree: usize) -> [Perm; 2] {
    let rotation = (0..degree).map(|point| (point + 1) % degree).collect();
    let reflection = (0..degree).map(|point| (degree - point) % degree).collect();
    [permutation(rotation), permutation(reflection)]
}

fn psl_2_31_generators() -> [Perm; 2] {
    const PRIME: usize = 31;
    let mut translation = vec![0; PRIME + 1];
    for (point, image) in translation.iter_mut().enumerate().take(PRIME) {
        *image = (point + 1) % PRIME;
    }
    translation[PRIME] = PRIME;

    let mut inversion = vec![0; PRIME + 1];
    inversion[0] = PRIME;
    inversion[PRIME] = 0;
    for (point, image) in inversion.iter_mut().enumerate().take(PRIME).skip(1) {
        let inverse = (1..PRIME)
            .find(|candidate| point * candidate % PRIME == 1)
            .unwrap();
        *image = (PRIME - inverse) % PRIME;
    }
    [permutation(translation), permutation(inversion)]
}

fn bench_kernel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("perm-core");

    let symmetric_12 = symmetric_generators(12);
    group.bench_function("bsgs/symmetric-12", |bencher| {
        bencher.iter(|| PermGroup::from_generators(12, black_box(&symmetric_12)).unwrap())
    });
    group.bench_function("bsgs/scorecard-symmetric-8", |bencher| {
        bencher.iter(|| PermGroup::from_generators(8, black_box(&symmetric_generators(8))).unwrap())
    });
    group.bench_function("bsgs/scorecard-cyclic-8", |bencher| {
        bencher.iter(|| {
            PermGroup::from_generators(
                8,
                black_box(&[permutation((0..8).map(|point| (point + 1) % 8).collect())]),
            )
            .unwrap()
        })
    });

    let dihedral_512 = dihedral_generators(512);
    group.bench_function("bsgs/dihedral-512", |bencher| {
        bencher.iter(|| PermGroup::from_generators(512, black_box(&dihedral_512)).unwrap())
    });

    let symmetric_12_group = PermGroup::from_generators(12, &symmetric_12).unwrap();
    group.bench_function("stabilizer/symmetric-12-fix-3", |bencher| {
        bencher.iter(|| {
            symmetric_12_group
                .pointwise_stabilizer(black_box(&[0, 1, 2]))
                .unwrap()
        })
    });

    let symmetric_48_group = PermGroup::from_generators(48, &symmetric_generators(48)).unwrap();
    group.bench_function("stabilizer/symmetric-48-prefix", |bencher| {
        bencher.iter(|| {
            symmetric_48_group
                .pointwise_stabilizer(black_box(&[0]))
                .unwrap()
        })
    });

    let dihedral_64_group = PermGroup::from_generators(64, &dihedral_generators(64)).unwrap();
    group.bench_function("stabilizer/dihedral-64-nonprefix", |bencher| {
        bencher.iter(|| {
            dihedral_64_group
                .pointwise_stabilizer(black_box(&[37]))
                .unwrap()
        })
    });

    let psl = PermGroup::from_generators(32, &psl_2_31_generators()).unwrap();
    let probe = psl.uniform_random_element(91).element;
    group.bench_function("membership/psl-2-31", |bencher| {
        bencher.iter(|| psl.contains(black_box(&probe)).unwrap())
    });

    let symmetric_32 = PermGroup::from_generators(32, &symmetric_generators(32)).unwrap();
    group.bench_function("sample/symmetric-32", |bencher| {
        bencher.iter(|| symmetric_32.uniform_random_element(black_box(17)))
    });

    let symmetric_8 = PermGroup::from_generators(8, &symmetric_generators(8)).unwrap();
    group.bench_function("enumeration/symmetric-8", |bencher| {
        bencher.iter(|| symmetric_8.iter_elements().count())
    });
    group.bench_function("sylow/symmetric-8-p2", |bencher| {
        bencher.iter(|| symmetric_8.sylow_subgroup(black_box(2)).unwrap())
    });
    group.bench_function("conjugacy/symmetric-8", |bencher| {
        bencher.iter(|| symmetric_8.conjugacy_classes().unwrap())
    });

    let cyclic_8 = PermGroup::from_generators(
        8,
        &[permutation((0..8).map(|point| (point + 1) % 8).collect())],
    )
    .unwrap();
    group.bench_function("normalizer/regular-cyclic-8", |bencher| {
        bencher.iter(|| symmetric_8.normalizer(black_box(&cyclic_8)).unwrap())
    });
    let nonregular_cyclic_6 =
        PermGroup::from_generators(8, &[permutation(vec![1, 2, 3, 4, 5, 0, 6, 7])]).unwrap();
    group.bench_function("normalizer/nonregular-cyclic-6-in-symmetric-8", |bencher| {
        bencher.iter(|| {
            symmetric_8
                .normalizer(black_box(&nonregular_cyclic_6))
                .unwrap()
        })
    });
    group.bench_function("normalizer/scorecard-regular-cyclic-8", |bencher| {
        bencher.iter(|| {
            let ambient =
                PermGroup::from_generators(8, black_box(&symmetric_generators(8))).unwrap();
            let subgroup = PermGroup::from_generators(
                8,
                black_box(&[permutation((0..8).map(|point| (point + 1) % 8).collect())]),
            )
            .unwrap();
            ambient.normalizer(&subgroup).unwrap()
        })
    });
    group.bench_function(
        "normalizer/scorecard-nonregular-cyclic-6-in-symmetric-8",
        |bencher| {
            bencher.iter(|| {
                let ambient =
                    PermGroup::from_generators(8, black_box(&symmetric_generators(8))).unwrap();
                let subgroup = PermGroup::from_generators(
                    8,
                    black_box(&[permutation(vec![1, 2, 3, 4, 5, 0, 6, 7])]),
                )
                .unwrap();
                ambient.normalizer(&subgroup).unwrap()
            })
        },
    );
    group.bench_function("set-stabilizer/symmetric-8-four-set", |bencher| {
        bencher.iter(|| {
            symmetric_8
                .set_stabilizer(black_box(&[0, 1, 2, 3]))
                .unwrap()
        })
    });
    group.bench_function("set-stabilizer/scorecard-symmetric-8-four-set", |bencher| {
        bencher.iter(|| {
            PermGroup::from_generators(8, black_box(&symmetric_generators(8)))
                .unwrap()
                .set_stabilizer(black_box(&[0, 1, 2, 3]))
                .unwrap()
        })
    });

    let alternating_5 = PermGroup::from_generators(
        5,
        &[
            permutation(vec![1, 2, 0, 3, 4]),
            permutation(vec![1, 2, 3, 4, 0]),
        ],
    )
    .unwrap();
    let symmetric_4_fixing_last = PermGroup::from_generators(
        5,
        &[
            permutation(vec![1, 0, 2, 3, 4]),
            permutation(vec![1, 2, 3, 0, 4]),
        ],
    )
    .unwrap();
    group.bench_function(
        "intersection/scorecard-alternating-5-symmetric-4",
        |bencher| {
            bencher.iter(|| {
                alternating_5
                    .intersection(black_box(&symmetric_4_fixing_last))
                    .unwrap()
            })
        },
    );
    group.bench_function(
        "intersection/scorecard-with-group-construction",
        |bencher| {
            bencher.iter(|| {
                let left = PermGroup::from_generators(
                    5,
                    black_box(&[
                        permutation(vec![1, 2, 0, 3, 4]),
                        permutation(vec![1, 2, 3, 4, 0]),
                    ]),
                )
                .unwrap();
                let right = PermGroup::from_generators(
                    5,
                    black_box(&[
                        permutation(vec![1, 0, 2, 3, 4]),
                        permutation(vec![1, 2, 3, 0, 4]),
                    ]),
                )
                .unwrap();
                left.intersection(&right).unwrap()
            })
        },
    );

    let alternating_6 = PermGroup::from_generators(
        6,
        &[
            permutation(vec![1, 2, 0, 3, 4, 5]),
            permutation(vec![0, 2, 3, 4, 5, 1]),
        ],
    )
    .unwrap();
    group.bench_function("structure/alternating-6", |bencher| {
        bencher.iter(|| {
            let group = black_box(&alternating_6);
            (
                is_solvable(group).unwrap(),
                is_nilpotent(group).unwrap(),
                is_perfect(group).unwrap(),
                group.sylow_subgroup(3).unwrap(),
                group.conjugacy_classes().unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kernel);
criterion_main!(benches);
