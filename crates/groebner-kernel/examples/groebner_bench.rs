// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Small deterministic end-to-end driver used by cross-CAS benchmarks.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use groebner_kernel::{
    groebner_basis_prime_profiled, GroebnerOptions, Monomial, MonomialOrder, PrimeField32,
    SparsePolynomial,
};

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.len() != 3 {
        eprintln!("usage: groebner_bench cyclic-N|katsura-N|semiregular-N PRIME");
        return ExitCode::FAILURE;
    }
    let Some((family, size)) = arguments[1].rsplit_once('-') else {
        eprintln!("system must be cyclic-N, katsura-N, or semiregular-N");
        return ExitCode::FAILURE;
    };
    let Ok(size) = size.parse::<usize>() else {
        eprintln!("system size must be a positive integer");
        return ExitCode::FAILURE;
    };
    let Ok(modulus) = arguments[2].parse::<u32>() else {
        eprintln!("prime must fit u32");
        return ExitCode::FAILURE;
    };
    let Ok(field) = PrimeField32::new(modulus) else {
        eprintln!("modulus must be prime and less than 2^31");
        return ExitCode::FAILURE;
    };
    let generators = match family {
        "cyclic" if size >= 2 => cyclic(size, &field),
        "katsura" if size >= 1 => katsura(size, &field),
        "semiregular" if size >= 1 => semiregular(size, &field),
        _ => {
            eprintln!("system must be cyclic-N (N >= 2), katsura-N, or semiregular-N (N >= 1)");
            return ExitCode::FAILURE;
        }
    };

    let started = Instant::now();
    let (output, profile) =
        match groebner_basis_prime_profiled(&field, &generators, &GroebnerOptions::default()) {
            Ok(output) => output,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        };
    let elapsed = started.elapsed();
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let output_terms = output
        .basis
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let output_coefficient_digits = output
        .basis
        .iter()
        .flat_map(|polynomial| polynomial.terms())
        .map(|(_, coefficient)| {
            let balanced = (*coefficient).min(modulus - *coefficient);
            balanced.to_string().len()
        })
        .max()
        .unwrap_or(0);
    println!(
        "system={} prime={} seconds={:.9} phase_input_s={:.9} phase_pairs_s={:.9} phase_symbolic_s={:.9} phase_matrix_s={:.9} phase_reduction_s={:.9} phase_extraction_s={:.9} phase_reconstruction_s={:.9} phase_final_s={:.9} input_terms={} basis={} output_terms={} output_coefficient_digits={} rounds={} degree_min={} degree_max={} degree_sum={} pairs={} gm={} equal_lcm={} monomials={} rows={} columns={} nonzeros={} peak_rows={} peak_columns={} peak_nonzeros={} monomial_bytes={} pair_bytes={} sparse_row_bytes={} dense_row_bytes={} coefficient_bytes={} workspace_bytes={} abcd_left={} abcd_right={} abcd_dense_cells={} abcd_full_cells={} parallel_rounds={} parallel_rows={} backend={:?}",
        arguments[1],
        modulus,
        elapsed.as_secs_f64(),
        profile.input_conversion.as_secs_f64(),
        profile.pair_selection.as_secs_f64(),
        profile.symbolic_preprocessing.as_secs_f64(),
        profile.matrix_construction.as_secs_f64(),
        profile.row_reduction.as_secs_f64(),
        profile.basis_extraction.as_secs_f64(),
        profile.rational_reconstruction.as_secs_f64(),
        profile
            .final_canonicalization_and_verification
            .as_secs_f64(),
        input_terms,
        output.basis.len(),
        output_terms,
        output_coefficient_digits,
        output.metrics.rounds,
        output.metrics.minimum_round_degree.unwrap_or(0),
        output.metrics.maximum_round_degree.unwrap_or(0),
        output.metrics.round_degree_sum,
        output.metrics.pairs_selected,
        output.metrics.gebauer_moeller_pairs,
        output.metrics.equal_lcm_pairs,
        output.metrics.monomials_interned,
        output.metrics.matrix_rows,
        output.metrics.matrix_columns,
        output.metrics.matrix_nonzeros,
        output.metrics.peak_matrix_rows,
        output.metrics.peak_matrix_columns,
        output.metrics.peak_matrix_nonzeros,
        output.metrics.peak_monomial_bytes,
        output.metrics.peak_pair_bytes,
        output.metrics.peak_sparse_row_bytes,
        output.metrics.peak_dense_row_bytes,
        output.metrics.peak_coefficient_bytes,
        output.metrics.peak_workspace_bytes,
        output.metrics.abcd_left_columns,
        output.metrics.abcd_right_columns,
        output.metrics.abcd_dense_cells,
        output.metrics.abcd_full_dense_cells,
        output.metrics.abcd_parallel_projection_rounds,
        output.metrics.abcd_parallel_projection_rows,
        output.metrics.last_row_backend,
    );
    ExitCode::SUCCESS
}

fn cyclic(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
    let order = MonomialOrder::GradedReverseLexicographic;
    let mut generators = Vec::with_capacity(size);
    for degree in 1..size {
        let terms = (0..size).map(|start| {
            let mut exponent = vec![0_u32; size];
            for offset in 0..degree {
                exponent[(start + offset) % size] += 1;
            }
            (Monomial::new(exponent.into_boxed_slice()), 1_u32)
        });
        generators.push(SparsePolynomial::new(size, order, terms, field));
    }
    let product = vec![1_u32; size];
    generators.push(SparsePolynomial::new(
        size,
        order,
        [
            (Monomial::new(product.clone().into_boxed_slice()), 1_u32),
            (Monomial::one(size), field.modulus() - 1),
        ],
        field,
    ));
    generators
}

fn katsura(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
    let arity = size + 1;
    let order = MonomialOrder::GradedReverseLexicographic;
    let mut linear = Vec::with_capacity(arity + 1);
    linear.push((variable(arity, 0), 1_u32));
    for variable_index in 1..arity {
        linear.push((variable(arity, variable_index), 2_u32));
    }
    linear.push((Monomial::one(arity), field.modulus() - 1));
    let mut generators = vec![SparsePolynomial::new(arity, order, linear, field)];

    for equation in 1..=size {
        let mut terms = Vec::new();
        for signed_index in -(size as isize)..=(size as isize) {
            let left = signed_index.unsigned_abs();
            let right = (equation as isize - signed_index).unsigned_abs();
            if left <= size && right <= size {
                let monomial = variable(arity, left)
                    .checked_product(&variable(arity, right))
                    .expect("Katsura exponents fit the benchmark kernel");
                terms.push((monomial, 1_u32));
            }
        }
        terms.push((variable(arity, equation), field.modulus() - 1));
        generators.push(SparsePolynomial::new(arity, order, terms, field));
    }
    generators
}

fn semiregular(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut coefficient = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state % u64::from(field.modulus() - 1) + 1) as u32
    };
    (0..size)
        .map(|_| {
            let mut terms = Vec::with_capacity(size * (size + 1) / 2 + size + 1);
            for left in 0..size {
                for right in left..size {
                    let monomial = variable(size, left)
                        .checked_product(&variable(size, right))
                        .expect("semi-regular benchmark exponents fit the kernel");
                    terms.push((monomial, coefficient()));
                }
            }
            for index in 0..size {
                terms.push((variable(size, index), coefficient()));
            }
            terms.push((Monomial::one(size), coefficient()));
            SparsePolynomial::new(
                size,
                MonomialOrder::GradedReverseLexicographic,
                terms,
                field,
            )
        })
        .collect()
}

fn variable(arity: usize, index: usize) -> Monomial {
    let mut exponent = vec![0_u32; arity];
    exponent[index] = 1;
    Monomial::new(exponent.into_boxed_slice())
}
