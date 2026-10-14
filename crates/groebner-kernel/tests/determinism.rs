// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use std::fs;
use std::process::Command;

use groebner_kernel::{
    groebner_basis_prime, groebner_basis_with_transform,
    module_groebner_basis_with_transform_prime, GroebnerOptions, Monomial, MonomialOrder,
    PrimeField32, SparseModuleElement, SparseModuleOrder, SparsePolynomial,
};

const SNAPSHOT_PATH_ENV: &str = "NOETHER_GROEBNER_SNAPSHOT_PATH";

fn monomial(exponents: &[u32]) -> Monomial {
    Monomial::new(exponents.to_vec().into_boxed_slice())
}

fn canonical_snapshot() -> String {
    let field = PrimeField32::new(101).unwrap();
    let order = MonomialOrder::GradedReverseLexicographic;
    let polynomial = |terms| SparsePolynomial::new(3, order, terms, &field);
    let generators = vec![
        polynomial([
            (monomial(&[2, 0, 0]), 1),
            (monomial(&[0, 1, 0]), 1),
            (monomial(&[0, 0, 0]), 3),
        ]),
        polynomial([
            (monomial(&[1, 1, 0]), 1),
            (monomial(&[0, 0, 1]), 1),
            (monomial(&[0, 0, 0]), 5),
        ]),
        polynomial([
            (monomial(&[0, 2, 0]), 1),
            (monomial(&[1, 0, 1]), 1),
            (monomial(&[0, 0, 0]), 7),
        ]),
    ];
    let options = GroebnerOptions::default();
    let basis = groebner_basis_prime(&field, &generators, &options).unwrap();
    let witnessed = groebner_basis_with_transform(&field, &generators, &options).unwrap();

    let module_generators = vec![
        SparseModuleElement::from_coordinates(
            vec![generators[0].clone(), generators[1].clone()],
            &field,
        ),
        SparseModuleElement::from_coordinates(
            vec![generators[1].clone(), generators[2].clone()],
            &field,
        ),
    ];
    let module = module_groebner_basis_with_transform_prime(
        &field,
        &module_generators,
        &SparseModuleOrder::TermOverPosition(order),
        &options,
    )
    .unwrap();

    format!(
        "basis={:?}\nwitnessed={:?}\ntransformations={:?}\nmodule={:?}\nmodule_transformations={:?}\n",
        basis.basis,
        witnessed.basis,
        witnessed.transformations,
        module.basis,
        module.transformations,
    )
}

#[test]
fn hash_seed_child_writes_canonical_snapshot() {
    let Ok(path) = std::env::var(SNAPSHOT_PATH_ENV) else {
        return;
    };
    fs::write(path, canonical_snapshot()).unwrap();
}

#[test]
fn canonical_outputs_are_stable_across_process_hash_seeds_and_thread_counts() {
    let expected = canonical_snapshot();
    let executable = std::env::current_exe().unwrap();
    for (run, threads) in [1, 2, 4, 3].into_iter().enumerate() {
        let path = std::env::temp_dir().join(format!(
            "noether-groebner-determinism-{}-{run}.txt",
            std::process::id(),
        ));
        let status = Command::new(&executable)
            .args([
                "--exact",
                "hash_seed_child_writes_canonical_snapshot",
                "--nocapture",
            ])
            .env(SNAPSHOT_PATH_ENV, &path)
            .env("RAYON_NUM_THREADS", threads.to_string())
            .status()
            .unwrap();
        assert!(status.success(), "determinism child {run} failed");
        let observed = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(observed, expected, "determinism child {run} diverged");
    }
}
