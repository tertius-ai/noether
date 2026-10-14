// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Small deterministic rational driver used by cross-CAS benchmarks.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use groebner_kernel::{
    groebner_basis_rational_profiled, GroebnerOptions, Monomial, MonomialOrder, Rational,
    SparsePolynomial,
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

fn main() -> ExitCode {
    let arguments = env::args().collect::<Vec<_>>();
    if !(2..=3).contains(&arguments.len()) {
        eprintln!(
            "usage: groebner_rational_bench cyclic-N|modular-cyclic-N|katsura-N|semiregular-N [RECIPE_MODULUS]"
        );
        return ExitCode::FAILURE;
    }
    let Some((family, size)) = arguments[1].rsplit_once('-') else {
        eprintln!("system must be cyclic-N, modular-cyclic-N, katsura-N, or semiregular-N");
        return ExitCode::FAILURE;
    };
    let Ok(size) = size.parse::<usize>() else {
        eprintln!("system size must be a positive integer");
        return ExitCode::FAILURE;
    };
    let recipe_modulus = arguments
        .get(2)
        .map_or(Ok(65_521_u32), |value| value.parse::<u32>());
    let Ok(recipe_modulus) = recipe_modulus else {
        eprintln!("recipe modulus must fit u32");
        return ExitCode::FAILURE;
    };
    if recipe_modulus < 3 {
        eprintln!("recipe modulus must be at least three");
        return ExitCode::FAILURE;
    }
    let generators = match family {
        "cyclic" if size >= 2 => cyclic(size),
        "modular-cyclic" if (2..=8).contains(&size) => pad_arity(cyclic(size), 9),
        "katsura" if size >= 1 => katsura(size),
        "semiregular" if size >= 1 => semiregular(size, recipe_modulus),
        _ => {
            eprintln!("system must be cyclic-N (N >= 2), modular-cyclic-N (2 <= N <= 8), katsura-N, or semiregular-N (N >= 1)");
            return ExitCode::FAILURE;
        }
    };

    let started = Instant::now();
    let (output, profile) =
        match groebner_basis_rational_profiled(&generators, &GroebnerOptions::default()) {
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
        .flat_map(|(_, coefficient)| {
            [
                coefficient
                    .numerator()
                    .to_string()
                    .trim_start_matches('-')
                    .len(),
                coefficient.denominator().to_string().len(),
            ]
        })
        .max()
        .unwrap_or(0);
    println!(
        "system={} coefficient_context=Q seconds={:.9} phase_input_s={:.9} phase_primitive_s={:.9} phase_modular_images_s={:.9} phase_reconstruction_s={:.9} phase_final_s={:.9} input_terms={} basis={} output_terms={} output_coefficient_digits={} rounds={} degree_min={} degree_max={} degree_sum={} modular_primes={} reconstruction_attempts={} monomials={} rows={} columns={} nonzeros={} peak_rows={} peak_columns={} peak_nonzeros={} monomial_bytes={} pair_bytes={} sparse_row_bytes={} dense_row_bytes={} coefficient_bytes={} workspace_bytes={} backend={:?} rational_backend={:?}",
        arguments[1],
        elapsed.as_secs_f64(),
        profile.input_conversion.as_secs_f64(),
        profile.primitive_integer_construction.as_secs_f64(),
        profile.modular_image_construction.as_secs_f64(),
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
        output.metrics.modular_primes,
        output.metrics.reconstruction_attempts,
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
        output.metrics.last_row_backend,
        output.metrics.rational_backend,
    );
    ExitCode::SUCCESS
}

fn cyclic(size: usize) -> Vec<SparsePolynomial<Rational>> {
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

fn pad_arity(
    values: Vec<SparsePolynomial<Rational>>,
    arity: usize,
) -> Vec<SparsePolynomial<Rational>> {
    values
        .into_iter()
        .map(|polynomial| {
            let old_arity = polynomial.arity();
            SparsePolynomial::from_sorted_terms(
                arity,
                polynomial.order(),
                polynomial.terms().iter().map(|(monomial, coefficient)| {
                    let mut exponents = vec![0; arity];
                    for (variable, exponent) in exponents.iter_mut().enumerate().take(old_arity) {
                        *exponent = monomial.exponent(variable);
                    }
                    (
                        Monomial::new(exponents.into_boxed_slice()),
                        coefficient.clone(),
                    )
                }),
                &RationalRing,
            )
        })
        .collect()
}

fn katsura(size: usize) -> Vec<SparsePolynomial<Rational>> {
    let arity = size + 1;
    let order = MonomialOrder::GradedReverseLexicographic;
    let mut linear = Vec::with_capacity(arity + 1);
    linear.push((variable(arity, 0), Rational::one()));
    for variable_index in 1..arity {
        linear.push((variable(arity, variable_index), Rational::from(2)));
    }
    linear.push((Monomial::one(arity), -Rational::one()));
    let mut generators = vec![SparsePolynomial::new(arity, order, linear, &RationalRing)];
    for equation in 1..=size {
        let mut terms = Vec::new();
        for signed_index in -(size as isize)..=(size as isize) {
            let left = signed_index.unsigned_abs();
            let right = (equation as isize - signed_index).unsigned_abs();
            if left <= size && right <= size {
                let monomial = variable(arity, left)
                    .checked_product(&variable(arity, right))
                    .expect("Katsura exponents fit the benchmark kernel");
                terms.push((monomial, Rational::one()));
            }
        }
        terms.push((variable(arity, equation), -Rational::one()));
        generators.push(SparsePolynomial::new(arity, order, terms, &RationalRing));
    }
    generators
}

fn semiregular(size: usize, recipe_modulus: u32) -> Vec<SparsePolynomial<Rational>> {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut coefficient = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let residue = (state % u64::from(recipe_modulus - 1) + 1) as u32;
        let signed = if residue <= recipe_modulus / 2 {
            i64::from(residue)
        } else {
            i64::from(residue) - i64::from(recipe_modulus)
        };
        Rational::from(signed)
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
                &RationalRing,
            )
        })
        .collect()
}

fn variable(arity: usize, index: usize) -> Monomial {
    let mut exponent = vec![0_u32; arity];
    exponent[index] = 1;
    Monomial::new(exponent.into_boxed_slice())
}
