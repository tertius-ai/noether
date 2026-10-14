// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Primitive-integer Gröbner construction for modest rational systems.

use core::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use ahash::AHashSet as HashSet;
use flint_core::{Integer, Rational};
use poly_core::{
    ArenaPolynomial, MonomialArena, MonomialId, PolynomialWorkspace, SparsePolynomial,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::engine::{
    interreduce, interreduce_tracked, verification_pairs, verify_basis_transform,
    verify_construction_output, verify_monic_and_reduced,
};
use crate::integer::IntegerCoefficientRing;
use crate::rational::RationalCoefficientField;
use crate::{
    GroebnerError, GroebnerEvidence, GroebnerMetrics, GroebnerOptions, GroebnerOutput,
    GroebnerTransformOutput, RationalBackend, RationalTransformationBackend,
};

const MAX_PRIMITIVE_COEFFICIENT_BITS: usize = 16_384;
const MAX_PRIMITIVE_INPUT_TERMS: usize = 512;
const MAX_PRIMITIVE_INPUT_POLYNOMIALS: usize = 64;
const MAX_PRIMITIVE_ARITY: usize = 8;
const MAX_PRIMITIVE_SHAPE_SCORE: usize = 6_000;
const PRIMITIVE_CONTENT_INTERVAL: usize = 128;
const PRIMITIVE_CONTENT_TRIGGER_BITS: usize = 4_096;
const MAX_PRIMITIVE_VERIFICATION_TERMS: usize = 250_000;
const MAX_PRIMITIVE_VERIFICATION_PAIRS: usize = 8_192;
#[cfg(feature = "parallel")]
const MIN_PARALLEL_VERIFICATION_PAIRS: usize = 128;

type PendingPair = Reverse<(u64, u64, usize, usize, MonomialId)>;
type VerificationPair = (u64, usize, usize, MonomialId);

struct PrimitiveTrackedWorkspace {
    value: PolynomialWorkspace<Integer>,
    representation: PolynomialWorkspace<Rational>,
}

impl PrimitiveTrackedWorkspace {
    fn new() -> Self {
        Self {
            value: PolynomialWorkspace::new(),
            representation: PolynomialWorkspace::new(),
        }
    }
}

/// Whether the bounded primitive lane is a plausible win over modular F4.
pub(crate) fn should_try_primitive(generators: &[SparsePolynomial<Rational>]) -> bool {
    let Some(first) = generators.first() else {
        return false;
    };
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let Some(maximum_degree) = generators
        .iter()
        .flat_map(|polynomial| polynomial.terms())
        .map(|(monomial, _)| monomial.total_degree())
        .collect::<Option<Vec<_>>>()
        .and_then(|degrees| degrees.into_iter().max())
        .and_then(|degree| usize::try_from(degree).ok())
    else {
        return false;
    };
    // This dimension/degree/support score separates the measured direct
    // fraction-free and modular-F4 regimes without recognizing a named
    // benchmark family. In particular, dense high-dimensional full-degree
    // systems cross to modular arithmetic before pseudo-reduction growth
    // dominates, while small high-coefficient systems retain the direct lane.
    let shape_score = first
        .arity()
        .saturating_mul(generators.len())
        .saturating_mul(input_terms)
        .saturating_mul(maximum_degree);
    first.arity() <= MAX_PRIMITIVE_ARITY
        && generators.len() <= MAX_PRIMITIVE_INPUT_POLYNOMIALS
        && input_terms <= MAX_PRIMITIVE_INPUT_TERMS
        && shape_score < MAX_PRIMITIVE_SHAPE_SCORE
        && generators.iter().all(|polynomial| {
            polynomial.terms().iter().all(|(_, coefficient)| {
                coefficient.numerator().bit_length() <= MAX_PRIMITIVE_COEFFICIENT_BITS
                    && coefficient.denominator().bit_length() <= MAX_PRIMITIVE_COEFFICIENT_BITS
            })
        })
}

/// Construct a reduced rational basis through primitive integer
/// pseudo-reduction.
///
/// Every intermediate polynomial is identified up to a nonzero rational
/// scalar. S-polynomials and head reductions use gcd-cancelled cross
/// multiplication, after which coefficient content and leading sign are
/// normalized. Exhausting the exact critical-pair queue is construction
/// evidence; final rational interreduction checks reducedness and source
/// containment without repeating a second all-pairs sweep.
pub(crate) fn groebner_basis_rational_primitive(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<Rational>, GroebnerError> {
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        order,
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let mut basis = generators
        .iter()
        .map(|polynomial| clear_denominators(polynomial, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    basis.retain(|polynomial| !polynomial.is_zero());
    deduplicate(&mut basis);
    let mut basis_sugar = basis
        .iter()
        .map(|polynomial| polynomial_total_degree(polynomial, &arena))
        .collect::<Vec<_>>();
    if basis.len() > options.max_basis_size {
        return Err(resource(
            "primitive rational basis size",
            options.max_basis_size,
        ));
    }

    let mut metrics = GroebnerMetrics {
        rational_backend: Some(RationalBackend::PrimitiveInteger),
        ..GroebnerMetrics::default()
    };
    let mut pending = BinaryHeap::<PendingPair>::new();
    let mut resolved = HashSet::<(usize, usize)>::default();
    let mut serial = 0_u64;
    for right in 0..basis.len() {
        update_pairs(
            right,
            &basis,
            &basis_sugar,
            &mut pending,
            &mut resolved,
            &mut serial,
            &mut metrics,
            options,
            &mut arena,
        )?;
    }
    observe_workspace(&basis, &pending, &arena, &mut metrics);

    while let Some(Reverse((round_degree, _, _, _, _))) = pending.peek().copied() {
        if metrics.rounds == options.max_rounds {
            return Err(resource("primitive rational rounds", options.max_rounds));
        }
        metrics.rounds += 1;
        metrics.minimum_round_degree = Some(
            metrics
                .minimum_round_degree
                .map_or(round_degree, |degree| degree.min(round_degree)),
        );
        metrics.maximum_round_degree = Some(
            metrics
                .maximum_round_degree
                .map_or(round_degree, |degree| degree.max(round_degree)),
        );
        metrics.round_degree_sum = metrics
            .round_degree_sum
            .saturating_add(u128::from(round_degree));
        let mut selected = 0_usize;
        while pending
            .peek()
            .is_some_and(|Reverse((degree, _, _, _, _))| *degree == round_degree)
            && (options.max_pairs_per_round == 0 || selected < options.max_pairs_per_round)
        {
            let Reverse((_, _, left, right, lcm)) =
                pending.pop().expect("a pending pair was inspected");
            selected += 1;
            metrics.pairs_selected += 1;
            let key = pair_key(left, right);
            if chain_criterion(left, right, lcm, &basis, &resolved, &arena) {
                metrics.chain_criterion_pairs += 1;
                resolved.insert(key);
                continue;
            }
            let s = primitive_s_polynomial(&basis[left], &basis[right], lcm, &mut arena)?;
            let remainder =
                primitive_head_normal_form(s, &basis, options.max_reduction_steps, &mut arena)?;
            resolved.insert(key);
            if remainder.is_zero() {
                continue;
            }
            if basis.len() == options.max_basis_size {
                return Err(resource(
                    "primitive rational basis size",
                    options.max_basis_size,
                ));
            }
            let new_index = basis.len();
            basis.push(remainder);
            basis_sugar.push(round_degree);
            update_pairs(
                new_index,
                &basis,
                &basis_sugar,
                &mut pending,
                &mut resolved,
                &mut serial,
                &mut metrics,
                options,
                &mut arena,
            )?;
            observe_workspace(&basis, &pending, &arena, &mut metrics);
        }
    }

    let field = RationalCoefficientField;
    let rational_basis = basis
        .iter()
        .map(|polynomial| {
            ArenaPolynomial::from_sorted_terms(
                &arena,
                polynomial
                    .iter()
                    .map(|(monomial, coefficient)| (monomial, Rational::from(coefficient.clone()))),
                &field,
            )
        })
        .collect::<Vec<_>>();
    let rational_basis = interreduce(
        &field,
        rational_basis,
        options.max_reduction_steps,
        &mut arena,
    )?;
    let rational_generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_construction_output(
        &field,
        &rational_generators,
        &rational_basis,
        options,
        &mut arena,
    )?;
    metrics.monomials_interned = arena.len();
    observe_workspace(&basis, &pending, &arena, &mut metrics);
    Ok(GroebnerOutput {
        basis: rational_basis
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(&arena, &field))
            .collect(),
        metrics,
        evidence: GroebnerEvidence::ExactConstruction,
    })
}

/// Construct a reduced rational basis through primitive integer arithmetic
/// while retaining exact rational expressions in the input generators.
pub(crate) fn groebner_basis_rational_primitive_with_transform(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<GroebnerTransformOutput<Rational>, GroebnerError> {
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        order,
        input_terms
            .max(input_terms.saturating_mul(8).min(65_536))
            .max(256),
    );
    let rational_field = RationalCoefficientField;
    let one_monomial = arena
        .intern_exponents(&vec![0; arity])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut basis = Vec::new();
    let mut representations = Vec::new();
    for (index, generator) in generators.iter().enumerate() {
        let value = clear_denominators(generator, &mut arena)?;
        let Some((_, integer_leading)) = value.leading_term() else {
            continue;
        };
        if basis.contains(&value) {
            continue;
        }
        let rational_leading = generator
            .terms()
            .first()
            .map(|(_, coefficient)| coefficient)
            .ok_or(GroebnerError::VerificationFailed)?;
        let scale = Rational::from(integer_leading.clone())
            * rational_leading
                .inverse()
                .map_err(|_| GroebnerError::VerificationFailed)?;
        let mut representation = vec![ArenaPolynomial::zero(&arena); generators.len()];
        representation[index] =
            ArenaPolynomial::new(&arena, [(one_monomial, scale)], &rational_field);
        basis.push(value);
        representations.push(representation);
    }
    let mut basis_sugar = basis
        .iter()
        .map(|polynomial| polynomial_total_degree(polynomial, &arena))
        .collect::<Vec<_>>();
    if basis.len() > options.max_basis_size {
        return Err(resource(
            "tracked primitive rational basis size",
            options.max_basis_size,
        ));
    }

    let mut metrics = GroebnerMetrics {
        rational_backend: Some(RationalBackend::PrimitiveInteger),
        rational_transformation_backend: Some(RationalTransformationBackend::PrimitiveInteger),
        ..GroebnerMetrics::default()
    };
    let mut pending = BinaryHeap::<PendingPair>::new();
    let mut resolved = HashSet::<(usize, usize)>::default();
    let mut serial = 0_u64;
    let mut s_workspace = PrimitiveTrackedWorkspace::new();
    for right in 0..basis.len() {
        update_pairs(
            right,
            &basis,
            &basis_sugar,
            &mut pending,
            &mut resolved,
            &mut serial,
            &mut metrics,
            options,
            &mut arena,
        )?;
    }
    observe_workspace(&basis, &pending, &arena, &mut metrics);
    observe_tracked_workspace(
        &basis,
        &representations,
        &pending,
        &arena,
        options,
        &mut metrics,
    )?;

    while let Some(Reverse((round_degree, _, _, _, _))) = pending.peek().copied() {
        if metrics.rounds == options.max_rounds {
            return Err(resource(
                "tracked primitive rational rounds",
                options.max_rounds,
            ));
        }
        metrics.rounds += 1;
        metrics.minimum_round_degree = Some(
            metrics
                .minimum_round_degree
                .map_or(round_degree, |degree| degree.min(round_degree)),
        );
        metrics.maximum_round_degree = Some(
            metrics
                .maximum_round_degree
                .map_or(round_degree, |degree| degree.max(round_degree)),
        );
        metrics.round_degree_sum = metrics
            .round_degree_sum
            .saturating_add(u128::from(round_degree));
        let mut selected = 0_usize;
        while pending
            .peek()
            .is_some_and(|Reverse((degree, _, _, _, _))| *degree == round_degree)
            && (options.max_pairs_per_round == 0 || selected < options.max_pairs_per_round)
        {
            let Reverse((_, _, left, right, lcm)) =
                pending.pop().expect("a pending tracked pair was inspected");
            selected += 1;
            metrics.pairs_selected += 1;
            let key = pair_key(left, right);
            if chain_criterion(left, right, lcm, &basis, &resolved, &arena) {
                metrics.chain_criterion_pairs += 1;
                resolved.insert(key);
                continue;
            }
            let (s, representation) = primitive_s_polynomial_tracked(
                &basis[left],
                &representations[left],
                &basis[right],
                &representations[right],
                lcm,
                &mut arena,
                &mut s_workspace,
            )?;
            let (remainder, representation) = primitive_head_normal_form_tracked(
                s,
                representation,
                &basis,
                &representations,
                options.max_reduction_steps,
                &mut arena,
            )?;
            resolved.insert(key);
            if remainder.is_zero() {
                continue;
            }
            if basis.len() == options.max_basis_size {
                return Err(resource(
                    "tracked primitive rational basis size",
                    options.max_basis_size,
                ));
            }
            let new_index = basis.len();
            basis.push(remainder);
            representations.push(representation);
            basis_sugar.push(round_degree);
            update_pairs(
                new_index,
                &basis,
                &basis_sugar,
                &mut pending,
                &mut resolved,
                &mut serial,
                &mut metrics,
                options,
                &mut arena,
            )?;
            observe_workspace(&basis, &pending, &arena, &mut metrics);
            observe_tracked_workspace(
                &basis,
                &representations,
                &pending,
                &arena,
                options,
                &mut metrics,
            )?;
        }
    }

    let rational_basis = basis
        .iter()
        .map(|polynomial| {
            ArenaPolynomial::from_sorted_terms(
                &arena,
                polynomial
                    .iter()
                    .map(|(monomial, coefficient)| (monomial, Rational::from(coefficient.clone()))),
                &rational_field,
            )
        })
        .collect::<Vec<_>>();
    let (rational_basis, representations) = interreduce_tracked(
        &rational_field,
        rational_basis,
        representations,
        options.max_reduction_steps,
        &mut arena,
    )?;
    let rational_generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &rational_field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_construction_output(
        &rational_field,
        &rational_generators,
        &rational_basis,
        options,
        &mut arena,
    )?;
    if verify_primitive_basis_transform(
        &rational_generators,
        &rational_basis,
        &representations,
        &mut arena,
    )?
    .is_none()
    {
        verify_basis_transform(
            &rational_field,
            &rational_generators,
            &rational_basis,
            &representations,
            &mut arena,
        )?;
    }
    metrics.monomials_interned = arena.len();
    observe_workspace(&basis, &pending, &arena, &mut metrics);
    observe_tracked_workspace(
        &basis,
        &representations,
        &pending,
        &arena,
        options,
        &mut metrics,
    )?;
    Ok(GroebnerTransformOutput {
        basis: rational_basis
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(&arena, &rational_field))
            .collect(),
        transformations: representations
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|polynomial| polynomial.to_sparse(&arena, &rational_field))
                    .collect()
            })
            .collect(),
        metrics,
        evidence: GroebnerEvidence::ExactConstruction,
    })
}

/// Verify a reduced rational candidate with exact primitive-integer normal
/// forms and coefficient-independent Buchberger criteria.
pub(crate) fn verify_rational_candidate_primitive(
    generators: &[SparsePolynomial<Rational>],
    basis: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<(), GroebnerError> {
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .chain(basis)
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let terms = generators
        .iter()
        .chain(basis)
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    if terms > MAX_PRIMITIVE_VERIFICATION_TERMS {
        return Err(resource(
            "primitive rational verification terms",
            MAX_PRIMITIVE_VERIFICATION_TERMS,
        ));
    }

    let field = RationalCoefficientField;
    let mut arena = MonomialArena::with_capacity(
        arity,
        order,
        terms.max(terms.saturating_mul(2).min(1_048_576)).max(256),
    );
    let rational_basis = basis
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_monic_and_reduced(&field, &rational_basis, &arena)?;
    let pairs = verification_pairs(&rational_basis, &mut arena)?;
    if pairs.len() > MAX_PRIMITIVE_VERIFICATION_PAIRS {
        return Err(resource(
            "primitive rational verification pairs",
            MAX_PRIMITIVE_VERIFICATION_PAIRS,
        ));
    }
    let integer_basis = basis
        .iter()
        .map(|polynomial| clear_denominators(polynomial, &mut arena))
        .collect::<Result<Vec<_>, _>>()?;
    verify_primitive_pairs(&pairs, &integer_basis, options, &mut arena)?;
    for generator in generators {
        let generator = clear_denominators(generator, &mut arena)?;
        if !primitive_head_normal_form(
            generator,
            &integer_basis,
            options.max_reduction_steps,
            &mut arena,
        )?
        .is_zero()
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn verify_primitive_pairs(
    pairs: &[VerificationPair],
    basis: &[ArenaPolynomial<Integer>],
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    #[cfg(feature = "parallel")]
    if pairs.len() >= MIN_PARALLEL_VERIFICATION_PAIRS && rayon::current_num_threads() > 1 {
        // Every worker owns an arena because reductions may intern quotient
        // monomials. Existing IDs are stable across forks, and rebuilding the
        // polynomial headers binds their immutable terms to the fresh arena.
        let coefficient_bytes = basis.iter().fold(0_usize, |total, polynomial| {
            polynomial
                .coefficients()
                .iter()
                .fold(total, |total, coefficient| {
                    total.saturating_add(coefficient.bit_length().div_ceil(8).max(1))
                })
        });
        let basis_bytes = basis.iter().fold(0_usize, |total, polynomial| {
            total.saturating_add(polynomial.allocated_bytes())
        });
        let worker_bytes = arena
            .allocated_bytes()
            .saturating_add(basis_bytes)
            .saturating_add(coefficient_bytes)
            .max(1);
        let memory_workers = (options.max_workspace_bytes / worker_bytes).max(1);
        let workers = rayon::current_num_threads()
            .min(memory_workers)
            .min(pairs.len());
        if workers > 1 {
            let chunk_size = pairs.len().div_ceil(workers);
            pairs
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut worker_arena = arena.fork();
                    let worker_basis = rehome_integer_basis(basis, &worker_arena);
                    verify_primitive_pair_slice(
                        chunk,
                        &worker_basis,
                        options.max_reduction_steps,
                        &mut worker_arena,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(());
        }
    }

    verify_primitive_pair_slice(pairs, basis, options.max_reduction_steps, arena)
}

#[cfg(feature = "parallel")]
fn rehome_integer_basis(
    basis: &[ArenaPolynomial<Integer>],
    arena: &MonomialArena,
) -> Vec<ArenaPolynomial<Integer>> {
    let ring = IntegerCoefficientRing;
    basis
        .iter()
        .map(|polynomial| {
            ArenaPolynomial::from_sorted_terms(
                arena,
                polynomial
                    .support()
                    .iter()
                    .copied()
                    .zip(polynomial.coefficients().iter().cloned()),
                &ring,
            )
        })
        .collect()
}

fn verify_primitive_pair_slice(
    pairs: &[VerificationPair],
    basis: &[ArenaPolynomial<Integer>],
    max_reduction_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for &(_, left, right, lcm) in pairs {
        let s = primitive_s_polynomial(&basis[left], &basis[right], lcm, arena)?;
        if !primitive_head_normal_form(s, basis, max_reduction_steps, arena)?.is_zero() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn clear_denominators(
    polynomial: &SparsePolynomial<Rational>,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<Integer>, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let denominator = polynomial
        .terms()
        .iter()
        .fold(Integer::one(), |value, (_, coefficient)| {
            value.lcm(&coefficient.denominator())
        });
    let terms = polynomial
        .terms()
        .iter()
        .map(|(monomial, coefficient)| {
            arena
                .intern(monomial.clone())
                .map(|monomial| {
                    (
                        monomial,
                        coefficient.numerator() * (&denominator / coefficient.denominator()),
                    )
                })
                .map_err(|_| GroebnerError::ExponentOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let value = ArenaPolynomial::from_sorted_terms(arena, terms, &ring);
    primitive_normalize(value, &ring, arena)
}

fn clear_arena_denominators(
    polynomial: &ArenaPolynomial<Rational>,
    arena: &MonomialArena,
) -> (ArenaPolynomial<Integer>, Integer) {
    let ring = IntegerCoefficientRing;
    let denominator = polynomial
        .coefficients()
        .iter()
        .fold(Integer::one(), |value, coefficient| {
            value.lcm(&coefficient.denominator())
        });
    let terms = polynomial.iter().map(|(monomial, coefficient)| {
        (
            monomial,
            coefficient.numerator() * (&denominator / coefficient.denominator()),
        )
    });
    (
        ArenaPolynomial::from_sorted_terms(arena, terms, &ring),
        denominator,
    )
}

fn verify_primitive_basis_transform(
    generators: &[ArenaPolynomial<Rational>],
    basis: &[ArenaPolynomial<Rational>],
    transformations: &[Vec<ArenaPolynomial<Rational>>],
    arena: &mut MonomialArena,
) -> Result<Option<()>, GroebnerError> {
    if basis.len() != transformations.len()
        || transformations
            .iter()
            .any(|row| row.len() != generators.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let integer_generators = generators
        .iter()
        .map(|generator| clear_arena_denominators(generator, arena))
        .collect::<Vec<_>>();
    let one_monomial = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let ring = IntegerCoefficientRing;
    let one = ArenaPolynomial::new(arena, [(one_monomial, Integer::one())], &ring);
    let zero = ArenaPolynomial::zero(arena);
    let mut workspace = PolynomialWorkspace::new();

    for (expected, row) in basis.iter().zip(transformations) {
        let (integer_expected, expected_denominator) = clear_arena_denominators(expected, arena);
        let integer_multipliers = row
            .iter()
            .map(|multiplier| clear_arena_denominators(multiplier, arena))
            .collect::<Vec<_>>();
        let product_denominators = integer_multipliers
            .iter()
            .zip(&integer_generators)
            .map(
                |((_, multiplier_denominator), (_, generator_denominator))| {
                    multiplier_denominator * generator_denominator
                },
            )
            .collect::<Vec<_>>();
        let mut common_denominator = expected_denominator.clone();
        for denominator in &product_denominators {
            common_denominator = common_denominator.lcm(denominator);
            if common_denominator.bit_length() > MAX_PRIMITIVE_COEFFICIENT_BITS {
                return Ok(None);
            }
        }
        let product_scalars = product_denominators
            .iter()
            .map(|denominator| &common_denominator / denominator)
            .collect::<Vec<_>>();
        let expected_scalar = -(&common_denominator / &expected_denominator);
        let mut products = integer_multipliers
            .iter()
            .zip(&integer_generators)
            .zip(&product_scalars)
            .map(|(((multiplier, _), (generator, _)), scalar)| (multiplier, generator, scalar))
            .collect::<Vec<_>>();
        products.push((&integer_expected, &one, &expected_scalar));
        let matches = zero
            .equals_sum_scaled_products(products, arena, &ring, &mut workspace)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        if !matches {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(Some(()))
}

fn primitive_normalize(
    polynomial: ArenaPolynomial<Integer>,
    ring: &IntegerCoefficientRing,
    arena: &MonomialArena,
) -> Result<ArenaPolynomial<Integer>, GroebnerError> {
    if polynomial.is_zero() {
        return Ok(polynomial);
    }
    let mut content = Integer::zero();
    for coefficient in polynomial.coefficients() {
        content = content.gcd(&coefficient.abs());
        if content.is_one() {
            break;
        }
    }
    let leading_negative = polynomial
        .leading_term()
        .is_some_and(|(_, coefficient)| coefficient.is_negative());
    if content.is_one() && !leading_negative {
        if polynomial
            .coefficients()
            .iter()
            .any(|coefficient| coefficient.bit_length() > MAX_PRIMITIVE_COEFFICIENT_BITS)
        {
            return Err(resource(
                "primitive rational coefficient bits",
                MAX_PRIMITIVE_COEFFICIENT_BITS,
            ));
        }
        return Ok(polynomial);
    }
    let divisor = if leading_negative { -content } else { content };
    let coefficients = polynomial
        .coefficients()
        .iter()
        .map(|coefficient| coefficient / &divisor)
        .collect::<Vec<_>>();
    if coefficients
        .iter()
        .any(|coefficient| coefficient.bit_length() > MAX_PRIMITIVE_COEFFICIENT_BITS)
    {
        return Err(resource(
            "primitive rational coefficient bits",
            MAX_PRIMITIVE_COEFFICIENT_BITS,
        ));
    }
    Ok(ArenaPolynomial::from_sorted_terms(
        arena,
        polynomial.support().iter().copied().zip(coefficients),
        ring,
    ))
}

fn primitive_normalize_tracked(
    polynomial: ArenaPolynomial<Integer>,
    representation: Vec<ArenaPolynomial<Rational>>,
    arena: &MonomialArena,
) -> Result<(ArenaPolynomial<Integer>, Vec<ArenaPolynomial<Rational>>), GroebnerError> {
    let Some((_, before)) = polynomial.leading_term() else {
        return Ok((polynomial, representation));
    };
    let before = before.clone();
    let integer_ring = IntegerCoefficientRing;
    let polynomial = primitive_normalize(polynomial, &integer_ring, arena)?;
    let after = polynomial
        .leading_term()
        .map(|(_, coefficient)| coefficient.clone())
        .ok_or(GroebnerError::VerificationFailed)?;
    if before == after {
        return Ok((polynomial, representation));
    }
    let scale = Rational::from(after)
        * Rational::from(before)
            .inverse()
            .map_err(|_| GroebnerError::VerificationFailed)?;
    let field = RationalCoefficientField;
    Ok((
        polynomial,
        representation
            .into_iter()
            .map(|coordinate| coordinate.scalar_mul(&scale, arena, &field))
            .collect(),
    ))
}

fn primitive_s_polynomial_tracked(
    left: &ArenaPolynomial<Integer>,
    left_representation: &[ArenaPolynomial<Rational>],
    right: &ArenaPolynomial<Integer>,
    right_representation: &[ArenaPolynomial<Rational>],
    lcm: MonomialId,
    arena: &mut MonomialArena,
    workspace: &mut PrimitiveTrackedWorkspace,
) -> Result<(ArenaPolynomial<Integer>, Vec<ArenaPolynomial<Rational>>), GroebnerError> {
    if left_representation.len() != right_representation.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let integer_ring = IntegerCoefficientRing;
    let rational_field = RationalCoefficientField;
    let (left_leading, left_coefficient) = left.leading_term().expect("nonzero basis member");
    let (right_leading, right_coefficient) = right.leading_term().expect("nonzero basis member");
    let coefficient_gcd = left_coefficient.gcd(right_coefficient);
    let left_scalar = right_coefficient / &coefficient_gcd;
    let right_scalar = left_coefficient / &coefficient_gcd;
    let left_multiplier = arena
        .quotient(lcm, left_leading)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("least common multiple is divisible by the left leading monomial");
    let right_multiplier = arena
        .quotient(lcm, right_leading)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("least common multiple is divisible by the right leading monomial");
    let mut value = ArenaPolynomial::zero(arena);
    left.shifted_scaled_sub_shifted_into(
        left_multiplier,
        &left_scalar,
        right,
        right_multiplier,
        &right_scalar,
        arena,
        &integer_ring,
        &mut workspace.value,
        &mut value,
    )
    .map_err(|_| GroebnerError::ExponentOverflow)?;
    let left_scalar = Rational::from(left_scalar);
    let right_scalar = Rational::from(right_scalar);
    let mut representation = Vec::with_capacity(left_representation.len());
    for (left, right) in left_representation.iter().zip(right_representation) {
        let mut coordinate = ArenaPolynomial::zero(arena);
        left.shifted_scaled_sub_shifted_into(
            left_multiplier,
            &left_scalar,
            right,
            right_multiplier,
            &right_scalar,
            arena,
            &rational_field,
            &mut workspace.representation,
            &mut coordinate,
        )
        .map_err(|_| GroebnerError::ExponentOverflow)?;
        representation.push(coordinate);
    }
    primitive_normalize_tracked(value, representation, arena)
}

fn primitive_head_normal_form_tracked(
    mut value: ArenaPolynomial<Integer>,
    mut representation: Vec<ArenaPolynomial<Rational>>,
    basis: &[ArenaPolynomial<Integer>],
    basis_representations: &[Vec<ArenaPolynomial<Rational>>],
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(ArenaPolynomial<Integer>, Vec<ArenaPolynomial<Rational>>), GroebnerError> {
    if basis.len() != basis_representations.len()
        || basis_representations
            .iter()
            .any(|row| row.len() != representation.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let integer_ring = IntegerCoefficientRing;
    let rational_field = RationalCoefficientField;
    let mut steps = 0_usize;
    let mut workspace = PolynomialWorkspace::new();
    let mut combined = ArenaPolynomial::zero(arena);
    let mut representation_workspace = PolynomialWorkspace::new();
    let mut representation_combined = ArenaPolynomial::zero(arena);
    while let Some((leading, coefficient)) = value.leading_term() {
        let Some((reducer_index, reducer)) = basis.iter().enumerate().find(|(_, polynomial)| {
            polynomial
                .leading_term()
                .is_some_and(|(reducer_leading, _)| arena.divides(reducer_leading, leading))
        }) else {
            return primitive_normalize_tracked(value, representation, arena);
        };
        if steps == max_steps {
            return Err(resource(
                "tracked primitive rational reduction steps",
                max_steps,
            ));
        }
        let (reducer_leading, reducer_coefficient) =
            reducer.leading_term().expect("nonzero reducer");
        let multiplier = arena
            .quotient(leading, reducer_leading)
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .expect("selected reducer divides the leading monomial");
        let coefficient_gcd = coefficient.gcd(reducer_coefficient);
        let value_scalar = reducer_coefficient / &coefficient_gcd;
        let reducer_scalar = coefficient / &coefficient_gcd;
        value
            .scaled_sub_term_into(
                &value_scalar,
                reducer,
                multiplier,
                &reducer_scalar,
                arena,
                &integer_ring,
                &mut workspace,
                &mut combined,
            )
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        core::mem::swap(&mut value, &mut combined);
        let value_scalar = Rational::from(value_scalar);
        let reducer_scalar = Rational::from(reducer_scalar);
        for (coordinate, reducer_coordinate) in representation
            .iter_mut()
            .zip(&basis_representations[reducer_index])
        {
            coordinate
                .scaled_sub_term_into(
                    &value_scalar,
                    reducer_coordinate,
                    multiplier,
                    &reducer_scalar,
                    arena,
                    &rational_field,
                    &mut representation_workspace,
                    &mut representation_combined,
                )
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            core::mem::swap(coordinate, &mut representation_combined);
        }
        steps += 1;
        let coefficient_bits = value
            .coefficients()
            .iter()
            .map(Integer::bit_length)
            .max()
            .unwrap_or(0);
        if steps.is_multiple_of(PRIMITIVE_CONTENT_INTERVAL)
            || coefficient_bits > PRIMITIVE_CONTENT_TRIGGER_BITS
        {
            (value, representation) = primitive_normalize_tracked(value, representation, arena)?;
        }
    }
    primitive_normalize_tracked(value, representation, arena)
}

#[derive(Clone, Copy)]
struct PairCandidate {
    left: usize,
    lcm: MonomialId,
    degree: u64,
    product: bool,
    active: bool,
}

fn compare_pair_candidates(left: &PairCandidate, right: &PairCandidate) -> Ordering {
    left.degree
        .cmp(&right.degree)
        .then_with(|| left.lcm.cmp(&right.lcm))
        .then_with(|| left.left.cmp(&right.left))
}

#[allow(clippy::too_many_arguments)]
fn update_pairs(
    new_index: usize,
    basis: &[ArenaPolynomial<Integer>],
    basis_sugar: &[u64],
    pending: &mut BinaryHeap<PendingPair>,
    resolved: &mut HashSet<(usize, usize)>,
    serial: &mut u64,
    metrics: &mut GroebnerMetrics,
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    let new_leading = basis[new_index]
        .leading_term()
        .expect("nonzero basis member")
        .0;
    let mut candidates = Vec::with_capacity(new_index);
    for (left, polynomial) in basis.iter().enumerate().take(new_index) {
        let left_leading = polynomial.leading_term().expect("nonzero basis member").0;
        let lcm = arena
            .lcm(left_leading, new_leading)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        let lcm_degree = arena.total_degree(lcm);
        let degree = basis_sugar[left]
            .saturating_add(lcm_degree.saturating_sub(arena.total_degree(left_leading)))
            .max(
                basis_sugar[new_index]
                    .saturating_add(lcm_degree.saturating_sub(arena.total_degree(new_leading))),
            );
        let product = Some(lcm_degree)
            == arena
                .total_degree(left_leading)
                .checked_add(arena.total_degree(new_leading));
        candidates.push(PairCandidate {
            left,
            lcm,
            degree,
            product,
            active: true,
        });
    }

    // Remove old pairs made redundant by the newly installed leading term.
    // The unequal-LCM side conditions are the Gebauer--Moeller old-pair
    // criterion and keep the equal-LCM representative needed below.
    pending.retain(|Reverse((_, _, left, right, lcm))| {
        let removed = arena.divides(new_leading, *lcm)
            && candidates[*left].lcm != *lcm
            && candidates[*right].lcm != *lcm;
        if removed {
            metrics.gebauer_moeller_pairs += 1;
        }
        !removed
    });

    candidates.sort_unstable_by(compare_pair_candidates);
    for index in 0..candidates.len() {
        if !candidates[index].active {
            continue;
        }
        for earlier in 0..index {
            if candidates[earlier].active
                && candidates[earlier].lcm != candidates[index].lcm
                && arena.divides(candidates[earlier].lcm, candidates[index].lcm)
            {
                candidates[index].active = false;
                metrics.gebauer_moeller_pairs += 1;
                break;
            }
        }
    }

    let mut group_start = 0;
    while group_start < candidates.len() {
        let lcm = candidates[group_start].lcm;
        let mut group_end = group_start + 1;
        while group_end < candidates.len() && candidates[group_end].lcm == lcm {
            group_end += 1;
        }
        let product_group = candidates[group_start..group_end]
            .iter()
            .any(|candidate| candidate.active && candidate.product);
        let mut retained_one = false;
        for candidate in &mut candidates[group_start..group_end] {
            if !candidate.active {
                continue;
            }
            if product_group {
                candidate.active = false;
                metrics.product_criterion_pairs += 1;
                if candidate.product {
                    resolved.insert(pair_key(candidate.left, new_index));
                }
            } else if retained_one {
                candidate.active = false;
                metrics.equal_lcm_pairs += 1;
            } else {
                retained_one = true;
            }
        }
        group_start = group_end;
    }

    for candidate in candidates.into_iter().filter(|candidate| candidate.active) {
        if pending.len() == options.max_pending_pairs {
            return Err(resource(
                "primitive rational pending pairs",
                options.max_pending_pairs,
            ));
        }
        pending.push(Reverse((
            candidate.degree,
            *serial,
            candidate.left,
            new_index,
            candidate.lcm,
        )));
        *serial = serial.saturating_add(1);
        metrics.pairs_generated += 1;
    }
    Ok(())
}

fn polynomial_total_degree(polynomial: &ArenaPolynomial<Integer>, arena: &MonomialArena) -> u64 {
    polynomial
        .support()
        .iter()
        .map(|monomial| arena.total_degree(*monomial))
        .max()
        .unwrap_or(0)
}

fn chain_criterion(
    left: usize,
    right: usize,
    lcm: MonomialId,
    basis: &[ArenaPolynomial<Integer>],
    resolved: &HashSet<(usize, usize)>,
    arena: &MonomialArena,
) -> bool {
    basis.iter().enumerate().any(|(middle, polynomial)| {
        middle != left
            && middle != right
            && polynomial
                .leading_term()
                .is_some_and(|(leading, _)| arena.divides(leading, lcm))
            && resolved.contains(&pair_key(left, middle))
            && resolved.contains(&pair_key(middle, right))
    })
}

fn primitive_s_polynomial(
    left: &ArenaPolynomial<Integer>,
    right: &ArenaPolynomial<Integer>,
    lcm: MonomialId,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<Integer>, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let (left_leading, left_coefficient) = left.leading_term().expect("nonzero basis member");
    let (right_leading, right_coefficient) = right.leading_term().expect("nonzero basis member");
    let coefficient_gcd = left_coefficient.gcd(right_coefficient);
    let left_scalar = right_coefficient / &coefficient_gcd;
    let right_scalar = left_coefficient / &coefficient_gcd;
    let left_multiplier = arena
        .quotient(lcm, left_leading)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("least common multiple is divisible by the left leading monomial");
    let right_multiplier = arena
        .quotient(lcm, right_leading)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("least common multiple is divisible by the right leading monomial");
    let left = left
        .term_mul(left_multiplier, &left_scalar, arena, &ring)
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let right = right
        .term_mul(right_multiplier, &right_scalar, arena, &ring)
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    primitive_normalize(left.sub(&right, arena, &ring), &ring, arena)
}

fn primitive_head_normal_form(
    mut value: ArenaPolynomial<Integer>,
    basis: &[ArenaPolynomial<Integer>],
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<Integer>, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let mut steps = 0_usize;
    let mut workspace = PolynomialWorkspace::new();
    let mut combined = ArenaPolynomial::zero(arena);
    while let Some((leading, coefficient)) = value.leading_term() {
        let Some(reducer) = basis.iter().find(|polynomial| {
            polynomial
                .leading_term()
                .is_some_and(|(reducer_leading, _)| arena.divides(reducer_leading, leading))
        }) else {
            return primitive_normalize(value, &ring, arena);
        };
        if steps == max_steps {
            return Err(resource("primitive rational reduction steps", max_steps));
        }
        let (reducer_leading, reducer_coefficient) =
            reducer.leading_term().expect("nonzero reducer");
        let multiplier = arena
            .quotient(leading, reducer_leading)
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .expect("selected reducer divides the leading monomial");
        let coefficient_gcd = coefficient.gcd(reducer_coefficient);
        let value_scalar = reducer_coefficient / &coefficient_gcd;
        let reducer_scalar = coefficient / &coefficient_gcd;
        value
            .scaled_sub_term_into(
                &value_scalar,
                reducer,
                multiplier,
                &reducer_scalar,
                arena,
                &ring,
                &mut workspace,
                &mut combined,
            )
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        core::mem::swap(&mut value, &mut combined);
        steps += 1;
        let coefficient_bits = value
            .coefficients()
            .iter()
            .map(Integer::bit_length)
            .max()
            .unwrap_or(0);
        if steps.is_multiple_of(PRIMITIVE_CONTENT_INTERVAL)
            || coefficient_bits > PRIMITIVE_CONTENT_TRIGGER_BITS
        {
            value = primitive_normalize(value, &ring, arena)?;
        }
    }
    primitive_normalize(value, &ring, arena)
}

fn pair_key(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn deduplicate(values: &mut Vec<ArenaPolynomial<Integer>>) {
    let mut unique = Vec::with_capacity(values.len());
    for value in values.drain(..) {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    *values = unique;
}

fn observe_workspace(
    basis: &[ArenaPolynomial<Integer>],
    pending: &BinaryHeap<PendingPair>,
    arena: &MonomialArena,
    metrics: &mut GroebnerMetrics,
) {
    let sparse = basis.iter().fold(0_usize, |total, polynomial| {
        total.saturating_add(polynomial.allocated_support_bytes())
    });
    let coefficients = basis.iter().fold(0_usize, |total, polynomial| {
        total.saturating_add(polynomial.allocated_coefficient_bytes())
    });
    let pairs = pending
        .capacity()
        .saturating_mul(core::mem::size_of::<PendingPair>());
    metrics.peak_monomial_bytes = metrics.peak_monomial_bytes.max(arena.allocated_bytes());
    metrics.peak_pair_bytes = metrics.peak_pair_bytes.max(pairs);
    metrics.peak_sparse_row_bytes = metrics.peak_sparse_row_bytes.max(sparse);
    metrics.peak_coefficient_bytes = metrics.peak_coefficient_bytes.max(coefficients);
    metrics.peak_workspace_bytes = metrics.peak_workspace_bytes.max(
        arena
            .allocated_bytes()
            .saturating_add(pairs)
            .saturating_add(sparse)
            .saturating_add(coefficients),
    );
}

fn observe_tracked_workspace(
    basis: &[ArenaPolynomial<Integer>],
    representations: &[Vec<ArenaPolynomial<Rational>>],
    pending: &BinaryHeap<PendingPair>,
    arena: &MonomialArena,
    options: &GroebnerOptions,
    metrics: &mut GroebnerMetrics,
) -> Result<(), GroebnerError> {
    let representation_structural = representations.iter().fold(0_usize, |total, row| {
        row.iter().fold(
            total.saturating_add(
                row.capacity()
                    .saturating_mul(core::mem::size_of::<ArenaPolynomial<Rational>>()),
            ),
            |total, polynomial| total.saturating_add(polynomial.allocated_support_bytes()),
        )
    });
    let representation_coefficients = representations.iter().fold(0_usize, |total, row| {
        row.iter().fold(total, |total, polynomial| {
            polynomial
                .coefficients()
                .iter()
                .fold(total, |total, coefficient| {
                    total
                        .saturating_add(coefficient.numerator().bit_length().div_ceil(8).max(1))
                        .saturating_add(coefficient.denominator().bit_length().div_ceil(8).max(1))
                })
        })
    });
    let basis_structural = basis.iter().fold(0_usize, |total, polynomial| {
        total.saturating_add(polynomial.allocated_support_bytes())
    });
    let basis_coefficients = basis.iter().fold(0_usize, |total, polynomial| {
        polynomial
            .coefficients()
            .iter()
            .fold(total, |total, coefficient| {
                total.saturating_add(coefficient.bit_length().div_ceil(8).max(1))
            })
    });
    let sparse = basis_structural.saturating_add(representation_structural);
    let coefficients = basis_coefficients.saturating_add(representation_coefficients);
    let pairs = pending
        .capacity()
        .saturating_mul(core::mem::size_of::<PendingPair>());
    metrics.peak_sparse_row_bytes = metrics.peak_sparse_row_bytes.max(sparse);
    metrics.peak_coefficient_bytes = metrics.peak_coefficient_bytes.max(coefficients);
    let workspace = arena
        .allocated_bytes()
        .saturating_add(pairs)
        .saturating_add(sparse)
        .saturating_add(coefficients);
    metrics.peak_workspace_bytes = metrics.peak_workspace_bytes.max(workspace);
    if workspace > options.max_workspace_bytes {
        Err(resource(
            "tracked primitive rational workspace bytes",
            options.max_workspace_bytes,
        ))
    } else {
        Ok(())
    }
}

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use poly_core::{Monomial, MonomialOrder};
    use proptest::prelude::*;

    use super::*;

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

    fn monomial(exponents: &[u32]) -> Monomial {
        Monomial::new(exponents.to_vec().into_boxed_slice())
    }

    fn cyclic(size: usize) -> Vec<SparsePolynomial<Rational>> {
        let field = RationalCoefficientField;
        let mut generators = Vec::with_capacity(size);
        for degree in 1..size {
            generators.push(SparsePolynomial::new(
                size,
                MonomialOrder::GradedReverseLexicographic,
                (0..size).map(|start| {
                    let mut exponents = vec![0; size];
                    for offset in 0..degree {
                        exponents[(start + offset) % size] += 1;
                    }
                    (Monomial::new(exponents.into_boxed_slice()), Rational::one())
                }),
                &field,
            ));
        }
        generators.push(SparsePolynomial::new(
            size,
            MonomialOrder::GradedReverseLexicographic,
            [
                (
                    Monomial::new(vec![1; size].into_boxed_slice()),
                    Rational::one(),
                ),
                (Monomial::one(size), Rational::from(-1)),
            ],
            &field,
        ));
        generators
    }

    #[test]
    fn primitive_integer_basis_matches_the_direct_rational_field() {
        let field = RationalCoefficientField;
        let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
        let third = Rational::new(Integer::one(), Integer::from(3)).unwrap();
        let generators = vec![
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[2, 0]), half),
                    (monomial(&[0, 1]), Rational::one()),
                ],
                &field,
            ),
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[1, 1]), Rational::one()),
                    (monomial(&[0, 0]), third),
                ],
                &field,
            ),
        ];
        let options = GroebnerOptions::default();
        let primitive = groebner_basis_rational_primitive(&generators, &options).unwrap();
        let direct = crate::groebner_basis(&field, &generators, &options).unwrap();
        assert_eq!(primitive.basis, direct.basis);
        assert_eq!(primitive.evidence, GroebnerEvidence::ExactConstruction);
        assert_eq!(
            primitive.metrics.rational_backend,
            Some(RationalBackend::PrimitiveInteger)
        );
        assert_eq!(primitive.metrics.modular_primes, 0);
    }

    #[test]
    fn rational_dispatch_selects_the_primitive_lane_for_a_modest_system() {
        let field = RationalCoefficientField;
        let generators = vec![
            SparsePolynomial::new(
                2,
                MonomialOrder::Lexicographic,
                [
                    (monomial(&[1, 0]), Rational::from(2)),
                    (monomial(&[0, 1]), Rational::from(3)),
                ],
                &field,
            ),
            SparsePolynomial::new(
                2,
                MonomialOrder::Lexicographic,
                [
                    (monomial(&[2, 0]), Rational::one()),
                    (monomial(&[0, 1]), Rational::from(-1)),
                ],
                &field,
            ),
        ];
        let output =
            crate::groebner_basis_rational(&generators, &GroebnerOptions::default()).unwrap();
        assert_eq!(
            output.metrics.rational_backend,
            Some(RationalBackend::PrimitiveInteger)
        );
    }

    #[test]
    fn shape_dispatch_keeps_cyclic_five_direct_and_sends_cyclic_six_modular() {
        assert!(should_try_primitive(&cyclic(5)));
        assert!(!should_try_primitive(&cyclic(6)));
        let output =
            crate::groebner_basis_rational(&cyclic(6), &GroebnerOptions::default()).unwrap();
        assert_eq!(
            output.metrics.rational_backend,
            Some(RationalBackend::Modular)
        );
    }

    #[test]
    fn large_monomial_basis_verifies_in_independent_pair_batches() {
        let field = RationalCoefficientField;
        let basis = (0_u32..130)
            .map(|x| {
                SparsePolynomial::new(
                    2,
                    MonomialOrder::GradedReverseLexicographic,
                    [(monomial(&[x, 129 - x]), Rational::one())],
                    &field,
                )
            })
            .collect::<Vec<_>>();

        let mut arena =
            MonomialArena::with_capacity(2, MonomialOrder::GradedReverseLexicographic, basis.len());
        let arena_basis = basis
            .iter()
            .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(verification_pairs(&arena_basis, &mut arena).unwrap().len() >= 128);

        verify_rational_candidate_primitive(&basis, &basis, &GroebnerOptions::default()).unwrap();
    }

    #[test]
    fn primitive_transform_verifier_is_exact_and_bounded() {
        let field = RationalCoefficientField;
        let mut arena = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let x = arena.intern_exponents(&[1]).unwrap();
        let one = arena.intern_exponents(&[0]).unwrap();
        let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
        let generator = ArenaPolynomial::new(&arena, [(x, half)], &field);
        let expected = ArenaPolynomial::new(&arena, [(x, Rational::one())], &field);
        let multiplier = ArenaPolynomial::new(&arena, [(one, Rational::from(2))], &field);
        assert_eq!(
            verify_primitive_basis_transform(
                &[generator],
                &[expected],
                &[vec![multiplier]],
                &mut arena,
            )
            .unwrap(),
            Some(())
        );

        let power_of_two = Integer::one() << 9_000_u32;
        let power_of_three = Integer::from(3).pow(5_000);
        let large_generator = ArenaPolynomial::new(
            &arena,
            [(
                x,
                Rational::new(Integer::one(), power_of_two.clone()).unwrap(),
            )],
            &field,
        );
        let large_multiplier = ArenaPolynomial::new(
            &arena,
            [(
                one,
                Rational::new(Integer::one(), power_of_three.clone()).unwrap(),
            )],
            &field,
        );
        let large_expected = ArenaPolynomial::new(
            &arena,
            [(
                x,
                Rational::new(Integer::one(), power_of_two * power_of_three).unwrap(),
            )],
            &field,
        );
        assert_eq!(
            verify_primitive_basis_transform(
                &[large_generator],
                &[large_expected],
                &[vec![large_multiplier]],
                &mut arena,
            )
            .unwrap(),
            None
        );
    }

    proptest! {
        #![proptest_config(property_config(48))]

        #[test]
        fn random_small_primitive_systems_match_direct_rational_f4(
            raw_generators in prop::collection::vec(
                prop::collection::vec((-3_i32..4, 1_u32..5, 0_u32..4, 0_u32..4), 1..5),
                1..5,
            )
        ) {
            let field = RationalCoefficientField;
            let generators = raw_generators
                .into_iter()
                .map(|terms| {
                    SparsePolynomial::new(
                        2,
                        MonomialOrder::GradedReverseLexicographic,
                        terms.into_iter().map(|(numerator, denominator, x, y)| {
                            (
                                monomial(&[x, y]),
                                Rational::new(
                                    Integer::from(numerator),
                                    Integer::from(denominator),
                                )
                                .unwrap(),
                            )
                        }),
                        &field,
                    )
                })
                .collect::<Vec<_>>();
            let options = GroebnerOptions::default();
            let primitive = groebner_basis_rational_primitive(&generators, &options).unwrap();
            let direct = crate::groebner_basis(&field, &generators, &options).unwrap();

            prop_assert_eq!(primitive.basis, direct.basis);
            prop_assert_eq!(primitive.evidence, GroebnerEvidence::ExactConstruction);
        }
    }
}
