// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Modular Gröbner construction over the rationals.

use std::collections::HashMap;

use flint_core::{Integer, Rational};
use poly_core::{ArenaPolynomial, CoefficientRing, Monomial, MonomialArena, SparsePolynomial};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sparla::CsrFp;

use crate::engine::{
    apply_prime_trace, learn_prime_trace, profile_phase, unit_ideal_certificate, verify_output,
    PrimeTraceSession, ProfilePhase, UnitIdealCertificateOutput,
};
use crate::primitive::{
    groebner_basis_rational_primitive, groebner_basis_rational_primitive_with_transform,
    should_try_primitive, verify_rational_candidate_primitive,
};
use crate::reducer::GenericReducer;
use crate::{
    groebner_basis_prime, groebner_basis_with_transform, Field, GroebnerError, GroebnerEvidence,
    GroebnerMetrics, GroebnerOptions, GroebnerOutput, GroebnerPhaseTimings,
    GroebnerTransformOutput, PrimeField32, RationalBackend, RationalTransformationBackend,
};

const MIN_STABILITY_FILTER_TERMS: usize = 4_096;

pub(crate) struct RationalCoefficientField;

impl CoefficientRing for RationalCoefficientField {
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

impl Field for RationalCoefficientField {
    fn one(&self) -> Self::Element {
        Rational::one()
    }

    fn inverse(&self, value: &Self::Element) -> Self::Element {
        value
            .inverse()
            .expect("the rational Gröbner lane only inverts nonzero coefficients")
    }
}

struct ReconstructionState {
    support: Vec<Monomial>,
    row_offsets: Vec<usize>,
    residues: Vec<Integer>,
    stable: Vec<Option<Rational>>,
    modulus: Integer,
    matching_images: usize,
}

type PrimeMacaulayCertificate = (usize, Vec<SparsePolynomial<u32>>);

struct PreparedPrime {
    prime: u32,
    field: PrimeField32,
    images: Vec<SparsePolynomial<u32>>,
}

struct EvaluatedPrime {
    prepared: PreparedPrime,
    replayed: Option<(Vec<SparsePolynomial<u32>>, GroebnerMetrics)>,
}

impl ReconstructionState {
    fn from_image(basis: &[SparsePolynomial<u32>], prime: u32) -> Self {
        let mut row_offsets = Vec::with_capacity(basis.len() + 1);
        let term_count = basis
            .iter()
            .map(|polynomial| polynomial.terms().len())
            .sum();
        let mut support = Vec::with_capacity(term_count);
        let mut residues = Vec::with_capacity(term_count);
        row_offsets.push(0);
        for polynomial in basis {
            for (monomial, coefficient) in polynomial.terms() {
                support.push(monomial.clone());
                residues.push(Integer::from(*coefficient));
            }
            row_offsets.push(support.len());
        }
        Self {
            support,
            row_offsets,
            stable: vec![None; residues.len()],
            residues,
            modulus: Integer::from(prime),
            matching_images: 1,
        }
    }

    fn support_matches(&self, basis: &[SparsePolynomial<u32>]) -> bool {
        self.row_offsets.len() == basis.len() + 1
            && self
                .row_offsets
                .windows(2)
                .zip(basis)
                .all(|(range, polynomial)| range[1] - range[0] == polynomial.terms().len())
            && self.support.len()
                == basis
                    .iter()
                    .map(|polynomial| polynomial.terms().len())
                    .sum()
            && self
                .support
                .iter()
                .zip(
                    basis
                        .iter()
                        .flat_map(|polynomial| polynomial.terms().iter()),
                )
                .all(|(left, (right, _))| left == right)
    }

    fn incorporate(&mut self, basis: &[SparsePolynomial<u32>], prime: u32) {
        debug_assert!(self.support_matches(basis));
        let field = PrimeField32::new(prime).expect("a reconstruction image uses a prime field");
        let modulus_inverse = field.inverse(&self.modulus.mod_u(prime));
        for ((residue, stable), (_, coefficient)) in
            self.residues.iter_mut().zip(&mut self.stable).zip(
                basis
                    .iter()
                    .flat_map(|polynomial| polynomial.terms().iter()),
            )
        {
            if stable
                .as_ref()
                .and_then(|value| rational_coefficient_mod_prime(value, prime))
                != Some(*coefficient)
            {
                *stable = None;
            }
            *residue = crt_pair(residue, &self.modulus, *coefficient, prime, modulus_inverse);
        }
        self.modulus *= Integer::from(prime);
        self.matching_images += 1;
    }

    fn reconstruct(
        &mut self,
        arity: usize,
        order: poly_core::MonomialOrder,
    ) -> Option<Vec<SparsePolynomial<Rational>>> {
        let field = RationalCoefficientField;
        for (stable, residue) in self.stable.iter_mut().zip(&self.residues) {
            if stable.is_none() {
                *stable = rational_reconstruct_candidate(residue, &self.modulus);
            }
        }
        if self.stable.iter().any(Option::is_none) {
            return None;
        }
        Some(
            self.row_offsets
                .windows(2)
                .map(|range| {
                    let start = range[0];
                    let end = range[1];
                    SparsePolynomial::from_sorted_terms(
                        arity,
                        order,
                        self.support[start..end].iter().cloned().zip(
                            self.stable[start..end]
                                .iter()
                                .map(|value| value.clone().expect("stability was checked")),
                        ),
                        &field,
                    )
                })
                .collect(),
        )
    }
}

fn rational_coefficient_mod_prime(coefficient: &Rational, prime: u32) -> Option<u32> {
    let field = PrimeField32::new(prime).ok()?;
    let denominator = coefficient.denominator().mod_u(prime);
    if denominator == 0 {
        return None;
    }
    Some(field.mul(
        &coefficient.numerator().mod_u(prime),
        &field.inverse(&denominator),
    ))
}

fn rational_prime_batch_size(
    remaining: usize,
    measured_worker_bytes: usize,
    maximum_workspace_bytes: usize,
) -> usize {
    let memory_lanes = if measured_worker_bytes == 0 {
        1
    } else {
        maximum_workspace_bytes / measured_worker_bytes
    }
    .max(1);
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
            .min(8)
            .min(memory_lanes)
            .min(remaining)
            .max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (remaining, memory_lanes);
        1
    }
}

fn replay_prime_batch(
    session: &PrimeTraceSession,
    prepared: Vec<PreparedPrime>,
    options: &GroebnerOptions,
) -> Result<Vec<EvaluatedPrime>, GroebnerError> {
    let replay_one = |prepared: PreparedPrime| {
        let mut fork = session.fork();
        let replayed = apply_prime_trace(&mut fork, &prepared.field, &prepared.images, options)?
            .map(|image| (image.basis, image.metrics));
        Ok(EvaluatedPrime { prepared, replayed })
    };
    #[cfg(feature = "parallel")]
    if prepared.len() > 1 {
        return prepared
            .into_par_iter()
            .map(replay_one)
            .collect::<Result<Vec<_>, _>>();
    }
    prepared.into_iter().map(replay_one).collect()
}

/// Compute a reduced Gröbner basis over `Q` through checked prime images,
/// CRT/rational reconstruction, and an exact characteristic-zero verifier.
/// If the prime budget cannot certify reconstruction, the exact direct field
/// engine remains the correctness floor.
pub fn groebner_basis_rational(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<Rational>, GroebnerError> {
    groebner_basis_rational_impl(generators, options, None)
}

/// Compute a reduced rational Gröbner basis while collecting benchmark-only
/// phase timings separately from the mathematical result.
pub fn groebner_basis_rational_profiled(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<(GroebnerOutput<Rational>, GroebnerPhaseTimings), GroebnerError> {
    let mut profile = GroebnerPhaseTimings::default();
    let output = groebner_basis_rational_impl(generators, options, Some(&mut profile))?;
    Ok((output, profile))
}

/// Compute a fast reduced rational basis and reconstruct an exact polynomial
/// change matrix from checked word-prime images.
///
/// The reduced basis uses the ordinary primitive/modular rational dispatch.
/// Transformation-aware F4 is then performed over word-prime fields, where
/// carrying polynomial vectors is cheap, and its stable support is lifted by
/// CRT and rational reconstruction. The complete characteristic-zero change
/// identity is checked before return. Direct rational tracked F4 remains the
/// correctness floor when the separate transformation-prime budget is not
/// sufficient.
pub fn groebner_basis_with_transform_rational(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<GroebnerTransformOutput<Rational>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }

    if should_try_primitive(generators) {
        let mut primitive_options = options.clone();
        primitive_options.max_basis_size = primitive_options.max_basis_size.min(2_048);
        primitive_options.max_pending_pairs = primitive_options.max_pending_pairs.min(250_000);
        primitive_options.max_rounds = primitive_options.max_rounds.min(2_048);
        primitive_options.max_reduction_steps =
            primitive_options.max_reduction_steps.min(1_000_000);
        match groebner_basis_rational_primitive_with_transform(generators, &primitive_options) {
            Ok(output) => return Ok(output),
            Err(GroebnerError::ResourceLimit { .. }) => {}
            Err(error) => return Err(error),
        }
    }

    let basis_output = groebner_basis_rational(generators, options)?;
    let mut metrics = basis_output.metrics;
    let basis = basis_output.basis;
    let evidence = basis_output.evidence;
    let mut candidate = (1_u32 << 30) - 1;
    let mut state = None::<ReconstructionState>;
    let mut transformation_primes = 0;
    while transformation_primes < options.max_transformation_primes {
        let Some((prime, field)) = next_usable_prime(&mut candidate, generators) else {
            break;
        };
        let images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        let image = groebner_basis_with_transform(&field, &images, options)?;
        absorb_metrics(&mut metrics, &image.metrics);
        metrics.modular_primes = metrics.modular_primes.saturating_add(1);
        transformation_primes += 1;
        if !rational_basis_matches_prime_image(&basis, &image.basis, &field) {
            metrics.unlucky_primes = metrics.unlucky_primes.saturating_add(1);
            continue;
        }
        let flattened = image
            .transformations
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if let Some(current) = &mut state {
            if current.support_matches(&flattened) {
                current.incorporate(&flattened, prime);
            } else {
                metrics.unlucky_primes = metrics.unlucky_primes.saturating_add(1);
                if current.matching_images == 1 {
                    *current = ReconstructionState::from_image(&flattened, prime);
                }
                continue;
            }
        } else {
            state = Some(ReconstructionState::from_image(&flattened, prime));
            continue;
        }
        let current = state
            .as_mut()
            .expect("a tracked prime image initialized reconstruction");
        if !current.matching_images.is_power_of_two() {
            continue;
        }
        let Some(flattened) = current.reconstruct(arity, order) else {
            continue;
        };
        metrics.reconstruction_attempts = metrics.reconstruction_attempts.saturating_add(1);
        let transformations = flattened
            .chunks_exact(generators.len())
            .map(<[SparsePolynomial<Rational>]>::to_vec)
            .collect::<Vec<_>>();
        if transformations.len() == basis.len()
            && verify_transform_candidate(generators, &basis, &transformations).is_ok()
        {
            metrics.rational_transformation_backend = Some(RationalTransformationBackend::Modular);
            return Ok(GroebnerTransformOutput {
                basis,
                transformations,
                metrics,
                evidence,
            });
        }
    }

    let direct = groebner_basis_with_transform(&RationalCoefficientField, generators, options)?;
    absorb_metrics(&mut metrics, &direct.metrics);
    metrics.rational_transformation_backend = Some(RationalTransformationBackend::DirectField);
    Ok(GroebnerTransformOutput {
        basis: direct.basis,
        transformations: direct.transformations,
        metrics,
        evidence: direct.evidence,
    })
}

fn groebner_basis_rational_impl(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
    mut profile: Option<&mut GroebnerPhaseTimings>,
) -> Result<GroebnerOutput<Rational>, GroebnerError> {
    let (arity, order) = profile_phase(
        &mut profile,
        ProfilePhase::InputConversion,
        || -> Result<_, GroebnerError> {
            let Some(first) = generators.first() else {
                return Err(GroebnerError::ContextMismatch);
            };
            let arity = first.arity();
            let order = first.order();
            if generators
                .iter()
                .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
            {
                return Err(GroebnerError::ContextMismatch);
            }
            Ok((arity, order))
        },
    )?;
    if should_try_primitive(generators) {
        let mut primitive_options = options.clone();
        primitive_options.max_basis_size = primitive_options.max_basis_size.min(2_048);
        primitive_options.max_pending_pairs = primitive_options.max_pending_pairs.min(250_000);
        primitive_options.max_rounds = primitive_options.max_rounds.min(2_048);
        primitive_options.max_reduction_steps =
            primitive_options.max_reduction_steps.min(1_000_000);
        match profile_phase(
            &mut profile,
            ProfilePhase::PrimitiveIntegerConstruction,
            || groebner_basis_rational_primitive(generators, &primitive_options),
        ) {
            Ok(output) => return Ok(output),
            Err(GroebnerError::ResourceLimit { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    let mut candidate = (1_u32 << 30) - 1;
    let mut state = None::<ReconstructionState>;
    let mut trace = None::<PrimeTraceSession>;
    let mut stability_candidate = None::<Vec<SparsePolynomial<Rational>>>;
    let mut stability_filter_used = false;
    let mut metrics = GroebnerMetrics::default();
    while metrics.modular_primes < options.max_modular_primes {
        let remaining = options
            .max_modular_primes
            .saturating_sub(metrics.modular_primes);
        let batch_size = if options.algorithm == crate::Algorithm::F4
            && trace.is_some()
            && state.as_ref().is_some_and(|current| {
                current.matching_images >= 2 && current.support.len() >= MIN_STABILITY_FILTER_TERMS
            }) {
            rational_prime_batch_size(
                remaining,
                metrics.peak_workspace_bytes,
                options.max_workspace_bytes,
            )
        } else {
            1
        };
        let prepared = profile_phase(&mut profile, ProfilePhase::InputConversion, || {
            let mut prepared = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let Some((prime, field)) = next_usable_prime(&mut candidate, generators) else {
                    break;
                };
                let images = generators
                    .iter()
                    .map(|polynomial| rational_image(polynomial, &field))
                    .collect::<Vec<_>>();
                prepared.push(PreparedPrime {
                    prime,
                    field,
                    images,
                });
            }
            prepared
        });
        if prepared.is_empty() {
            break;
        }
        let evaluated = if options.algorithm == crate::Algorithm::F4 {
            if let Some(session) = &mut trace {
                if prepared.len() == 1 {
                    let prepared = prepared.into_iter().next().expect("one prime was prepared");
                    let replayed = profile_phase(
                        &mut profile,
                        ProfilePhase::ModularImageConstruction,
                        || apply_prime_trace(session, &prepared.field, &prepared.images, options),
                    )?
                    .map(|image| (image.basis, image.metrics));
                    vec![EvaluatedPrime { prepared, replayed }]
                } else {
                    profile_phase(&mut profile, ProfilePhase::ModularImageConstruction, || {
                        replay_prime_batch(session, prepared, options)
                    })?
                }
            } else {
                let prepared = prepared.into_iter().next().expect("one prime was prepared");
                let (image, session) =
                    profile_phase(&mut profile, ProfilePhase::ModularImageConstruction, || {
                        learn_prime_trace(&prepared.field, &prepared.images, options)
                    })?;
                trace = Some(session);
                vec![EvaluatedPrime {
                    prepared,
                    replayed: Some((image.basis, image.metrics)),
                }]
            }
        } else {
            profile_phase(&mut profile, ProfilePhase::ModularImageConstruction, || {
                prepared
                    .into_iter()
                    .map(|prepared| {
                        let image =
                            groebner_basis_prime(&prepared.field, &prepared.images, options)?;
                        Ok(EvaluatedPrime {
                            prepared,
                            replayed: Some((image.basis, image.metrics)),
                        })
                    })
                    .collect::<Result<Vec<_>, GroebnerError>>()
            })?
        };

        for evaluated in evaluated {
            let PreparedPrime {
                prime,
                field,
                images,
            } = evaluated.prepared;
            let (basis, image_metrics) = if let Some(replayed) = evaluated.replayed {
                replayed
            } else {
                metrics.trace_rejections += 1;
                let (image, replacement) =
                    profile_phase(&mut profile, ProfilePhase::ModularImageConstruction, || {
                        learn_prime_trace(&field, &images, options)
                    })?;
                trace = Some(replacement);
                (image.basis, image.metrics)
            };
            absorb_metrics(&mut metrics, &image_metrics);
            metrics.modular_primes += 1;

            if let Some(stable_basis) = stability_candidate.take() {
                let matches =
                    profile_phase(&mut profile, ProfilePhase::RationalReconstruction, || {
                        rational_basis_matches_prime_image(&stable_basis, &basis, &field)
                    });
                if matches {
                    match profile_phase(
                        &mut profile,
                        ProfilePhase::FinalCanonicalizationAndVerification,
                        || verify_candidate(generators, &stable_basis, options),
                    ) {
                        Ok(()) => {
                            metrics.rational_backend = Some(RationalBackend::Modular);
                            return Ok(GroebnerOutput {
                                basis: stable_basis,
                                metrics,
                                evidence: GroebnerEvidence::IndependentBuchbergerSweep,
                            });
                        }
                        Err(GroebnerError::VerificationFailed) => {}
                        Err(error) => return Err(error),
                    }
                }
            }

            let incorporated =
                profile_phase(&mut profile, ProfilePhase::RationalReconstruction, || {
                    if let Some(current) = &mut state {
                        if current.support_matches(&basis) {
                            current.incorporate(&basis, prime);
                            true
                        } else {
                            metrics.unlucky_primes += 1;
                            if current.matching_images == 1 {
                                *current = ReconstructionState::from_image(&basis, prime);
                            }
                            false
                        }
                    } else {
                        state = Some(ReconstructionState::from_image(&basis, prime));
                        false
                    }
                });
            if !incorporated {
                continue;
            }

            let current = state
                .as_mut()
                .expect("a prime image initialized reconstruction");
            let Some(basis) =
                profile_phase(&mut profile, ProfilePhase::RationalReconstruction, || {
                    current.reconstruct(arity, order)
                })
            else {
                continue;
            };
            metrics.reconstruction_attempts += 1;
            let output_terms = basis
                .iter()
                .map(|polynomial| polynomial.terms().len())
                .sum::<usize>();
            if output_terms >= MIN_STABILITY_FILTER_TERMS && !stability_filter_used {
                stability_filter_used = true;
                stability_candidate = Some(basis);
                continue;
            }
            match profile_phase(
                &mut profile,
                ProfilePhase::FinalCanonicalizationAndVerification,
                || verify_candidate(generators, &basis, options),
            ) {
                Ok(()) => {
                    metrics.rational_backend = Some(RationalBackend::Modular);
                    return Ok(GroebnerOutput {
                        basis,
                        metrics,
                        evidence: GroebnerEvidence::IndependentBuchbergerSweep,
                    });
                }
                Err(GroebnerError::VerificationFailed) => {}
                Err(error) => return Err(error),
            }
        }
    }
    let direct = profile_phase(&mut profile, ProfilePhase::ModularImageConstruction, || {
        crate::groebner_basis(&RationalCoefficientField, generators, options)
    })?;
    absorb_metrics(&mut metrics, &direct.metrics);
    metrics.rational_backend = Some(RationalBackend::DirectField);
    Ok(GroebnerOutput {
        basis: direct.basis,
        metrics,
        evidence: direct.evidence,
    })
}

/// Express one in a rational-polynomial generator family through checked
/// prime images, CRT/rational reconstruction, and an exact characteristic-zero
/// identity check.
///
/// The modular images use the same deterministic transformation-aware F4
/// computation. Images whose certificate support differs from the current
/// reconstruction are treated as unlucky. If the configured prime budget is
/// insufficient, direct exact F4 remains the correctness floor.
pub fn unit_ideal_certificate_rational(
    generators: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<Option<UnitIdealCertificateOutput<Rational>>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Ok(None);
    };
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }

    // A nonunit prime image is not conclusive: a rational certificate can
    // lose its denominator modulo finitely many primes. Decide the generic
    // fiber once with the independently verified modular basis path.
    let decision = groebner_basis_rational(generators, options)?;
    if !is_unit_basis(&decision.basis, arity) {
        return Ok(None);
    }

    let mut candidate = (1_u32 << 30) - 1;
    let mut state = None::<ReconstructionState>;
    let mut metrics = decision.metrics;
    while metrics.modular_primes < options.max_modular_primes {
        let Some((prime, field)) = next_usable_prime(&mut candidate, generators) else {
            break;
        };
        let images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        let Some(image) = unit_ideal_certificate(&field, &images, options)? else {
            metrics.unlucky_primes += 1;
            metrics.modular_primes += 1;
            continue;
        };
        absorb_metrics(&mut metrics, &image.metrics);
        metrics.modular_primes += 1;

        if let Some(current) = &mut state {
            if current.support_matches(&image.multipliers) {
                current.incorporate(&image.multipliers, prime);
            } else {
                metrics.unlucky_primes += 1;
                if current.matching_images == 1 {
                    *current = ReconstructionState::from_image(&image.multipliers, prime);
                }
                continue;
            }
        } else {
            state = Some(ReconstructionState::from_image(&image.multipliers, prime));
            continue;
        }

        let current = state
            .as_mut()
            .expect("a prime certificate initialized reconstruction");
        // Certificate vectors are much larger than reduced bases. Rebuilding
        // rational candidates and multiplying them exactly after every image
        // would dominate the modular computation, so use geometric early
        // termination checkpoints.
        if !current.matching_images.is_power_of_two() {
            continue;
        }
        let Some(multipliers) = current.reconstruct(arity, order) else {
            continue;
        };
        metrics.reconstruction_attempts += 1;
        if let Some(verified) =
            verify_rational_certificate_at_fresh_prime(generators, &multipliers, &mut candidate)
        {
            metrics.modular_primes += 1;
            if !verified {
                continue;
            }
        }
        if verify_certificate_candidate(generators, &multipliers).is_ok() {
            return Ok(Some(UnitIdealCertificateOutput {
                multipliers,
                metrics,
            }));
        }
    }

    let direct = unit_ideal_certificate(&RationalCoefficientField, generators, options)?;
    Ok(direct.map(|mut output| {
        absorb_metrics(&mut metrics, &output.metrics);
        output.metrics = metrics;
        output
    }))
}

/// Learn the least admitted Macaulay certificate degree from a usable
/// word-prime image of a rational generator family.
///
/// This is an acceleration hint rather than evidence: characteristic-zero
/// code must still solve and verify the exact system. Up to eight usable
/// images are tried so a denominator or unlucky rank does not make the hint
/// fragile.
pub fn rational_macaulay_certificate_degree(
    generators: &[SparsePolynomial<Rational>],
    maximum_degree: usize,
    maximum_cells: usize,
) -> Result<Option<usize>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Ok(None);
    };
    if generators.iter().any(|polynomial| {
        polynomial.arity() != first.arity() || polynomial.order() != first.order()
    }) {
        return Err(GroebnerError::ContextMismatch);
    }
    let mut candidate = (1_u32 << 30) - 1;
    for _ in 0..8 {
        let Some((_, field)) = next_usable_prime(&mut candidate, generators) else {
            break;
        };
        let images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        if let Some((degree, _)) =
            search_prime_macaulay_certificate(&images, maximum_degree, maximum_cells, &field)?
        {
            return Ok(Some(degree));
        }
    }
    Ok(None)
}

/// Construct a bounded low-degree rational Nullstellensatz certificate from
/// sparse Macaulay systems over word-prime images.
///
/// The first usable image learns the least admitted total-degree bound. Later
/// images solve the same canonical sparse system, after which CRT/rational
/// reconstruction is accepted only when exact characteristic-zero
/// multiplication verifies `1 = sum(h_i f_i)`. A unit ideal whose certificate
/// exceeds either explicit bound returns a resource-limit error.
pub fn macaulay_unit_ideal_certificate_rational(
    generators: &[SparsePolynomial<Rational>],
    maximum_degree: usize,
    maximum_cells: usize,
    options: &GroebnerOptions,
) -> Result<Option<UnitIdealCertificateOutput<Rational>>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Ok(None);
    };
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let decision = groebner_basis_rational(generators, options)?;
    if !is_unit_basis(&decision.basis, arity) {
        return Ok(None);
    }

    let mut candidate = (1_u32 << 30) - 1;
    let mut learned_degree = None::<usize>;
    let mut degree_misses = 0_usize;
    let mut state = None::<ReconstructionState>;
    let mut metrics = decision.metrics;
    while metrics.modular_primes < options.max_modular_primes {
        let Some((prime, field)) = next_usable_prime(&mut candidate, generators) else {
            break;
        };
        let images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        let certificate = if let Some(degree) = learned_degree {
            prime_macaulay_certificate(&images, degree, maximum_cells, &field)?
        } else {
            let result =
                search_prime_macaulay_certificate(&images, maximum_degree, maximum_cells, &field)?;
            if let Some((degree, certificate)) = result {
                learned_degree = Some(degree);
                Some(certificate)
            } else {
                None
            }
        };
        metrics.modular_primes += 1;
        let Some(certificate) = certificate else {
            metrics.unlucky_primes += 1;
            degree_misses += 1;
            if learned_degree.is_none() && degree_misses == 8 {
                return Err(GroebnerError::ResourceLimit {
                    resource: "Macaulay certificate degree",
                    limit: maximum_degree,
                });
            }
            continue;
        };

        if let Some(current) = &mut state {
            if current.support_matches(&certificate) {
                current.incorporate(&certificate, prime);
            } else {
                metrics.unlucky_primes += 1;
                if current.matching_images == 1 {
                    *current = ReconstructionState::from_image(&certificate, prime);
                }
                continue;
            }
        } else {
            state = Some(ReconstructionState::from_image(&certificate, prime));
            continue;
        }

        let current = state
            .as_mut()
            .expect("a Macaulay prime image initialized reconstruction");
        if !current.matching_images.is_power_of_two() {
            continue;
        }
        let Some(multipliers) = current.reconstruct(arity, order) else {
            continue;
        };
        metrics.reconstruction_attempts += 1;
        if let Some(verified) =
            verify_rational_certificate_at_fresh_prime(generators, &multipliers, &mut candidate)
        {
            metrics.modular_primes += 1;
            if !verified {
                continue;
            }
        }
        if verify_certificate_candidate(generators, &multipliers).is_ok() {
            return Ok(Some(UnitIdealCertificateOutput {
                multipliers,
                metrics,
            }));
        }
    }
    Err(GroebnerError::ResourceLimit {
        resource: "Macaulay certificate primes",
        limit: options.max_modular_primes,
    })
}

fn search_prime_macaulay_certificate(
    generators: &[SparsePolynomial<u32>],
    maximum_degree: usize,
    maximum_cells: usize,
    field: &PrimeField32,
) -> Result<Option<PrimeMacaulayCertificate>, GroebnerError> {
    let Some(initial_degree) = polynomial_family_maximum_degree(generators)? else {
        return Ok(None);
    };
    if initial_degree > maximum_degree {
        return Ok(None);
    }
    for degree in initial_degree..=maximum_degree {
        if let Some(certificate) =
            prime_macaulay_certificate(generators, degree, maximum_cells, field)?
        {
            return Ok(Some((degree, certificate)));
        }
    }
    Ok(None)
}

fn prime_macaulay_certificate(
    generators: &[SparsePolynomial<u32>],
    degree: usize,
    maximum_cells: usize,
    field: &PrimeField32,
) -> Result<Option<Vec<SparsePolynomial<u32>>>, GroebnerError> {
    let Some(first) = generators.first() else {
        return Ok(None);
    };
    let arity = first.arity();
    let order = first.order();
    let generator_degrees = generators
        .iter()
        .map(polynomial_maximum_degree)
        .collect::<Result<Vec<_>, _>>()?;
    let equations = bounded_monomials(arity, degree, maximum_cells)?;
    let equation_index = equations
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, exponent)| (exponent, index))
        .collect::<HashMap<_, _>>();
    let mut unknowns = Vec::<(usize, Vec<u32>)>::new();
    for (generator, generator_degree) in generator_degrees.iter().enumerate() {
        let Some(generator_degree) = generator_degree else {
            continue;
        };
        if *generator_degree > degree {
            continue;
        }
        unknowns.extend(
            bounded_monomials(arity, degree - generator_degree, maximum_cells)?
                .into_iter()
                .map(|monomial| (generator, monomial)),
        );
    }
    let cells = equations.len().saturating_mul(unknowns.len());
    if cells > maximum_cells {
        return Err(GroebnerError::ResourceLimit {
            resource: "Macaulay matrix cells",
            limit: maximum_cells,
        });
    }
    let mut triples = Vec::<(usize, usize, i64)>::new();
    for (column, (generator, multiplier)) in unknowns.iter().enumerate() {
        for (monomial, coefficient) in generators[*generator].terms() {
            let product = multiplier
                .iter()
                .enumerate()
                .map(|(variable, exponent)| exponent.checked_add(monomial.exponent(variable)))
                .collect::<Option<Vec<_>>>()
                .ok_or(GroebnerError::ExponentOverflow)?;
            let row = *equation_index
                .get(&product)
                .ok_or(GroebnerError::ExponentOverflow)?;
            triples.push((row, column, i64::from(*coefficient)));
        }
    }
    let matrix = CsrFp::from_coo(
        equations.len(),
        unknowns.len(),
        &triples,
        u64::from(field.modulus()),
    );
    let constant_row = *equation_index
        .get(&vec![0; arity])
        .expect("bounded monomial enumeration contains one");
    let rhs = CsrFp::from_coo(
        equations.len(),
        1,
        &[(constant_row, 0, 1)],
        u64::from(field.modulus()),
    );
    let Some(solution) = matrix.solve(&rhs) else {
        return Ok(None);
    };
    let mut terms = vec![Vec::new(); generators.len()];
    for (column, (generator, monomial)) in unknowns.into_iter().enumerate() {
        if let Some(coefficient) = solution.get(column, 0) {
            terms[generator].push((Monomial::new(monomial.into_boxed_slice()), coefficient));
        }
    }
    let multipliers = terms
        .into_iter()
        .map(|terms| SparsePolynomial::new(arity, order, terms, field))
        .collect::<Vec<_>>();
    if verify_prime_certificate(generators, &multipliers, field) {
        Ok(Some(multipliers))
    } else {
        Err(GroebnerError::VerificationFailed)
    }
}

fn polynomial_family_maximum_degree<C: Clone + Eq>(
    generators: &[SparsePolynomial<C>],
) -> Result<Option<usize>, GroebnerError> {
    generators
        .iter()
        .map(polynomial_maximum_degree)
        .collect::<Result<Vec<_>, _>>()
        .map(|degrees| degrees.into_iter().flatten().max())
}

fn polynomial_maximum_degree<C: Clone + Eq>(
    polynomial: &SparsePolynomial<C>,
) -> Result<Option<usize>, GroebnerError> {
    polynomial
        .terms()
        .iter()
        .map(|(monomial, _)| {
            monomial
                .total_degree()
                .and_then(|degree| usize::try_from(degree).ok())
                .ok_or(GroebnerError::ExponentOverflow)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|degrees| degrees.into_iter().max())
}

fn bounded_monomials(
    arity: usize,
    degree: usize,
    maximum_count: usize,
) -> Result<Vec<Vec<u32>>, GroebnerError> {
    fn visit(
        variable: usize,
        remaining: usize,
        exponents: &mut [u32],
        output: &mut Vec<Vec<u32>>,
        maximum_count: usize,
    ) -> Result<(), GroebnerError> {
        if variable == exponents.len() {
            if output.len() == maximum_count {
                return Err(GroebnerError::ResourceLimit {
                    resource: "Macaulay monomial count",
                    limit: maximum_count,
                });
            }
            output.push(exponents.to_vec());
            return Ok(());
        }
        for exponent in 0..=remaining {
            exponents[variable] =
                u32::try_from(exponent).map_err(|_| GroebnerError::ExponentOverflow)?;
            visit(
                variable + 1,
                remaining - exponent,
                exponents,
                output,
                maximum_count,
            )?;
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(0, degree, &mut vec![0; arity], &mut output, maximum_count)?;
    Ok(output)
}

fn verify_prime_certificate(
    generators: &[SparsePolynomial<u32>],
    multipliers: &[SparsePolynomial<u32>],
    field: &PrimeField32,
) -> bool {
    let first = generators
        .first()
        .expect("a prime certificate has generators");
    let represented = multipliers.iter().zip(generators).fold(
        SparsePolynomial::new(first.arity(), first.order(), [], field),
        |sum, (multiplier, generator)| sum.add(&multiplier.multiply(generator, field, None), field),
    );
    represented
        == SparsePolynomial::new(
            first.arity(),
            first.order(),
            [(Monomial::one(first.arity()), 1)],
            field,
        )
}

fn is_unit_basis(basis: &[SparsePolynomial<Rational>], arity: usize) -> bool {
    basis.len() == 1
        && basis[0].terms().len() == 1
        && basis[0].terms()[0].0 == Monomial::new(vec![0; arity].into_boxed_slice())
        && basis[0].terms()[0].1 == Rational::one()
}

fn next_usable_prime(
    candidate: &mut u32,
    generators: &[SparsePolynomial<Rational>],
) -> Option<(u32, PrimeField32)> {
    for _ in 0..100_000 {
        *candidate = candidate.checked_sub(2)?;
        let Ok(field) = PrimeField32::new(*candidate) else {
            continue;
        };
        let usable = generators.iter().all(|polynomial| {
            polynomial
                .terms()
                .iter()
                .all(|(_, coefficient)| coefficient.denominator().mod_u(*candidate) != 0)
        });
        if usable {
            return Some((*candidate, field));
        }
    }
    None
}

fn rational_image(
    polynomial: &SparsePolynomial<Rational>,
    field: &PrimeField32,
) -> SparsePolynomial<u32> {
    let prime = field.modulus();
    SparsePolynomial::new(
        polynomial.arity(),
        polynomial.order(),
        polynomial.terms().iter().map(|(monomial, coefficient)| {
            let numerator = coefficient.numerator().mod_u(prime);
            let denominator = coefficient.denominator().mod_u(prime);
            let value = field.mul(&numerator, &field.inverse(&denominator));
            (monomial.clone(), value)
        }),
        field,
    )
}

fn rational_basis_matches_prime_image(
    candidate: &[SparsePolynomial<Rational>],
    image: &[SparsePolynomial<u32>],
    field: &PrimeField32,
) -> bool {
    candidate.len() == image.len()
        && candidate.iter().zip(image).all(|(candidate, image)| {
            candidate.terms().len() == image.terms().len()
                && candidate.terms().iter().zip(image.terms()).all(
                    |(
                        (candidate_monomial, candidate_coefficient),
                        (image_monomial, image_coefficient),
                    )| {
                        candidate_monomial == image_monomial
                            && rational_coefficient_mod_prime(
                                candidate_coefficient,
                                field.modulus(),
                            ) == Some(*image_coefficient)
                    },
                )
        })
}

fn verify_rational_certificate_at_fresh_prime(
    generators: &[SparsePolynomial<Rational>],
    multipliers: &[SparsePolynomial<Rational>],
    candidate: &mut u32,
) -> Option<bool> {
    for _ in 0..100_000 {
        *candidate = candidate.checked_sub(2)?;
        let Ok(field) = PrimeField32::new(*candidate) else {
            continue;
        };
        let usable = generators.iter().chain(multipliers).all(|polynomial| {
            polynomial
                .terms()
                .iter()
                .all(|(_, coefficient)| coefficient.denominator().mod_u(*candidate) != 0)
        });
        if !usable {
            continue;
        }
        let generator_images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        let multiplier_images = multipliers
            .iter()
            .map(|polynomial| rational_image(polynomial, &field))
            .collect::<Vec<_>>();
        return Some(verify_prime_certificate(
            &generator_images,
            &multiplier_images,
            &field,
        ));
    }
    None
}

fn crt_pair(
    residue: &Integer,
    modulus: &Integer,
    value: u32,
    prime: u32,
    modulus_inverse: u32,
) -> Integer {
    let old = residue.mod_u(prime);
    let difference = (u64::from(value) + u64::from(prime) - u64::from(old)) % u64::from(prime);
    let correction =
        ((u128::from(difference) * u128::from(modulus_inverse)) % u128::from(prime)) as u64;
    residue + &(modulus * &Integer::from(correction))
}

fn rational_reconstruct(residue: &Integer, modulus: &Integer) -> Option<Rational> {
    if let (Some(residue), Some(modulus)) = (residue.to_i64(), modulus.to_i64()) {
        if residue >= 0 && modulus > 0 {
            return rational_reconstruct_word(residue as u64, modulus as u64);
        }
    }
    rational_reconstruct_big(residue, modulus)
}

fn rational_reconstruct_candidate(residue: &Integer, modulus: &Integer) -> Option<Rational> {
    if modulus.to_i64().is_some() {
        rational_reconstruct(residue, modulus)
    } else {
        Rational::reconstruct_symmetric(residue, modulus)
            .or_else(|| rational_reconstruct_big(residue, modulus))
    }
}

fn rational_reconstruct_word(residue: u64, modulus: u64) -> Option<Rational> {
    if residue == 0 {
        return Some(Rational::zero());
    }
    let modulus_signed = i128::from(modulus);
    let mut old_remainder = modulus_signed;
    let mut remainder = i128::from(residue);
    let mut old_denominator = 0_i128;
    let mut denominator = 1_i128;
    let mut best = None::<(i128, i128, i128)>;
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        let mut numerator = remainder;
        let mut candidate_denominator = denominator;
        if candidate_denominator < 0 {
            numerator = -numerator;
            candidate_denominator = -candidate_denominator;
        }
        let numerator_abs = numerator.unsigned_abs();
        let denominator_abs = candidate_denominator.unsigned_abs();
        let numerator_modulus = numerator.rem_euclid(modulus_signed) as u128;
        let denominator_modulus = candidate_denominator.rem_euclid(modulus_signed) as u128;
        if gcd_word(numerator_abs, denominator_abs) == 1
            && (denominator_modulus * u128::from(residue)) % u128::from(modulus)
                == numerator_modulus
        {
            let score = numerator_abs * denominator_abs;
            if score < u128::from(modulus).div_ceil(2)
                && best
                    .as_ref()
                    .is_none_or(|(_, _, best_quotient)| quotient > *best_quotient)
            {
                best = Some((numerator, candidate_denominator, quotient));
            }
        }
        let next_remainder = old_remainder - quotient * remainder;
        let next_denominator = old_denominator - quotient * denominator;
        old_remainder = remainder;
        remainder = next_remainder;
        old_denominator = denominator;
        denominator = next_denominator;
    }
    best.and_then(|(numerator, denominator, _)| {
        Rational::new(
            Integer::from(i64::try_from(numerator).ok()?),
            Integer::from(i64::try_from(denominator).ok()?),
        )
        .ok()
    })
}

fn gcd_word(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn rational_reconstruct_big(residue: &Integer, modulus: &Integer) -> Option<Rational> {
    if residue.is_zero() {
        return Some(Rational::zero());
    }
    let mut old_remainder = modulus.clone();
    let mut remainder = residue.clone();
    let mut old_denominator = Integer::zero();
    let mut denominator = Integer::one();
    let mut best = None::<(Integer, Integer, Integer)>;
    while !remainder.is_zero() {
        let quotient = &old_remainder / &remainder;
        let mut numerator = remainder.clone();
        let mut candidate_denominator = denominator.clone();
        if candidate_denominator.is_negative() {
            numerator = -numerator;
            candidate_denominator = -candidate_denominator;
        }
        if numerator.gcd(&candidate_denominator) == Integer::one()
            && (&(&candidate_denominator * residue) - &numerator).is_divisible(modulus)
        {
            let score = numerator.abs() * candidate_denominator.abs();
            if &score + &score < *modulus
                && best
                    .as_ref()
                    .is_none_or(|(_, _, best_quotient)| quotient > *best_quotient)
            {
                best = Some((numerator, candidate_denominator, quotient.clone()));
            }
        }
        let next_remainder = &old_remainder - &(&quotient * &remainder);
        let next_denominator = &old_denominator - &(&quotient * &denominator);
        old_remainder = remainder;
        remainder = next_remainder;
        old_denominator = denominator;
        denominator = next_denominator;
    }
    best.and_then(|(numerator, denominator, _)| Rational::new(numerator, denominator).ok())
}

pub(crate) fn verify_candidate(
    generators: &[SparsePolynomial<Rational>],
    basis: &[SparsePolynomial<Rational>],
    options: &GroebnerOptions,
) -> Result<(), GroebnerError> {
    match verify_rational_candidate_primitive(generators, basis, options) {
        Ok(()) => return Ok(()),
        Err(GroebnerError::ResourceLimit { .. }) => {}
        Err(error) => return Err(error),
    }
    let field = RationalCoefficientField;
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let mut arena = MonomialArena::new(first.arity(), first.order());
    let generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let basis = basis
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_output(
        &field,
        &generators,
        &basis,
        options,
        &GenericReducer,
        &mut arena,
    )
}

fn verify_certificate_candidate(
    generators: &[SparsePolynomial<Rational>],
    multipliers: &[SparsePolynomial<Rational>],
) -> Result<(), GroebnerError> {
    if generators.len() != multipliers.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let field = RationalCoefficientField;
    let mut arena = MonomialArena::new(first.arity(), first.order());
    let generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let multipliers = multipliers
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut represented = ArenaPolynomial::zero(&arena);
    for (multiplier, generator) in multipliers.iter().zip(&generators) {
        let product = multiplier
            .multiply(generator, &mut arena, &field, None)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        represented = represented.add(&product, &arena, &field);
    }
    let one = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    if represented == ArenaPolynomial::new(&arena, [(one, Rational::one())], &field) {
        Ok(())
    } else {
        Err(GroebnerError::VerificationFailed)
    }
}

fn verify_transform_candidate(
    generators: &[SparsePolynomial<Rational>],
    basis: &[SparsePolynomial<Rational>],
    transformations: &[Vec<SparsePolynomial<Rational>>],
) -> Result<(), GroebnerError> {
    if basis.len() != transformations.len()
        || transformations
            .iter()
            .any(|row| row.len() != generators.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let first = generators.first().ok_or(GroebnerError::ContextMismatch)?;
    let field = RationalCoefficientField;
    for (expected, row) in basis.iter().zip(transformations) {
        let represented = row.iter().zip(generators).fold(
            SparsePolynomial::new(first.arity(), first.order(), [], &field),
            |sum, (multiplier, generator)| {
                sum.add(&multiplier.multiply(generator, &field, None), &field)
            },
        );
        if &represented != expected {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn absorb_metrics(total: &mut GroebnerMetrics, image: &GroebnerMetrics) {
    total.rounds = total.rounds.saturating_add(image.rounds);
    total.minimum_round_degree = match (total.minimum_round_degree, image.minimum_round_degree) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (left, right) => left.or(right),
    };
    total.maximum_round_degree = match (total.maximum_round_degree, image.maximum_round_degree) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    };
    total.round_degree_sum = total
        .round_degree_sum
        .saturating_add(image.round_degree_sum);
    total.pairs_generated = total.pairs_generated.saturating_add(image.pairs_generated);
    total.pairs_selected = total.pairs_selected.saturating_add(image.pairs_selected);
    total.product_criterion_pairs = total
        .product_criterion_pairs
        .saturating_add(image.product_criterion_pairs);
    total.chain_criterion_pairs = total
        .chain_criterion_pairs
        .saturating_add(image.chain_criterion_pairs);
    total.gebauer_moeller_pairs = total
        .gebauer_moeller_pairs
        .saturating_add(image.gebauer_moeller_pairs);
    total.equal_lcm_pairs = total.equal_lcm_pairs.saturating_add(image.equal_lcm_pairs);
    total.signature_pruned_pairs = total
        .signature_pruned_pairs
        .saturating_add(image.signature_pruned_pairs);
    total.matrix_rows = total.matrix_rows.saturating_add(image.matrix_rows);
    total.matrix_columns = total.matrix_columns.saturating_add(image.matrix_columns);
    total.matrix_nonzeros = total.matrix_nonzeros.saturating_add(image.matrix_nonzeros);
    total.peak_matrix_rows = total.peak_matrix_rows.max(image.peak_matrix_rows);
    total.peak_matrix_columns = total.peak_matrix_columns.max(image.peak_matrix_columns);
    total.peak_matrix_nonzeros = total.peak_matrix_nonzeros.max(image.peak_matrix_nonzeros);
    total.peak_monomial_bytes = total.peak_monomial_bytes.max(image.peak_monomial_bytes);
    total.peak_pair_bytes = total.peak_pair_bytes.max(image.peak_pair_bytes);
    total.peak_sparse_row_bytes = total.peak_sparse_row_bytes.max(image.peak_sparse_row_bytes);
    total.peak_dense_row_bytes = total.peak_dense_row_bytes.max(image.peak_dense_row_bytes);
    total.peak_coefficient_bytes = total
        .peak_coefficient_bytes
        .max(image.peak_coefficient_bytes);
    total.peak_workspace_bytes = total.peak_workspace_bytes.max(image.peak_workspace_bytes);
    total.generic_row_reductions = total
        .generic_row_reductions
        .saturating_add(image.generic_row_reductions);
    total.rational_dense_row_reductions = total
        .rational_dense_row_reductions
        .saturating_add(image.rational_dense_row_reductions);
    total.binary_sparse_row_reductions = total
        .binary_sparse_row_reductions
        .saturating_add(image.binary_sparse_row_reductions);
    total.binary_dense_row_reductions = total
        .binary_dense_row_reductions
        .saturating_add(image.binary_dense_row_reductions);
    total.binary_extension_dense_row_reductions = total
        .binary_extension_dense_row_reductions
        .saturating_add(image.binary_extension_dense_row_reductions);
    total.prime_sparse_row_reductions = total
        .prime_sparse_row_reductions
        .saturating_add(image.prime_sparse_row_reductions);
    total.prime_abcd_row_reductions = total
        .prime_abcd_row_reductions
        .saturating_add(image.prime_abcd_row_reductions);
    total.prime_dense_row_reductions = total
        .prime_dense_row_reductions
        .saturating_add(image.prime_dense_row_reductions);
    total.abcd_left_columns = total
        .abcd_left_columns
        .saturating_add(image.abcd_left_columns);
    total.abcd_right_columns = total
        .abcd_right_columns
        .saturating_add(image.abcd_right_columns);
    total.abcd_dense_cells = total
        .abcd_dense_cells
        .saturating_add(image.abcd_dense_cells);
    total.abcd_full_dense_cells = total
        .abcd_full_dense_cells
        .saturating_add(image.abcd_full_dense_cells);
    total.abcd_parallel_projection_rounds = total
        .abcd_parallel_projection_rounds
        .saturating_add(image.abcd_parallel_projection_rounds);
    total.abcd_parallel_projection_rows = total
        .abcd_parallel_projection_rows
        .saturating_add(image.abcd_parallel_projection_rows);
    total.last_row_backend = image.last_row_backend.or(total.last_row_backend);
    total.monomials_interned = total
        .monomials_interned
        .saturating_add(image.monomials_interned);
    total.signature_fallback |= image.signature_fallback;
    total.trace_learning_rounds = total
        .trace_learning_rounds
        .saturating_add(image.trace_learning_rounds);
    total.trace_application_rounds = total
        .trace_application_rounds
        .saturating_add(image.trace_application_rounds);
    total.trace_replayed_rows = total
        .trace_replayed_rows
        .saturating_add(image.trace_replayed_rows);
    total.trace_rejections = total
        .trace_rejections
        .saturating_add(image.trace_rejections);
    total.rational_backend = image.rational_backend.or(total.rational_backend);
    total.rational_transformation_backend = image
        .rational_transformation_backend
        .or(total.rational_transformation_backend);
}

#[cfg(test)]
mod tests {
    use super::*;
    use poly_core::MonomialOrder;

    fn monomial(values: &[u32]) -> Monomial {
        Monomial::new(values.to_vec().into_boxed_slice())
    }

    #[test]
    fn modular_basis_matches_direct_rational_f4_and_verifies() {
        let field = RationalCoefficientField;
        let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
        let third = Rational::new(Integer::one(), Integer::from(3)).unwrap();
        let arity = 9;
        let padded = |left: u32, right: u32| {
            let mut exponents = vec![0; arity];
            exponents[0] = left;
            exponents[1] = right;
            Monomial::new(exponents.into_boxed_slice())
        };
        let generators = vec![
            SparsePolynomial::new(
                arity,
                MonomialOrder::GradedReverseLexicographic,
                [(padded(2, 0), half), (padded(0, 1), Rational::one())],
                &field,
            ),
            SparsePolynomial::new(
                arity,
                MonomialOrder::GradedReverseLexicographic,
                [(padded(1, 1), Rational::one()), (padded(0, 0), third)],
                &field,
            ),
        ];
        let options = GroebnerOptions::default();
        let modular = groebner_basis_rational(&generators, &options).unwrap();
        let (profiled, _) = groebner_basis_rational_profiled(&generators, &options).unwrap();
        assert_eq!(profiled, modular);
        let direct = crate::groebner_basis(&field, &generators, &options).unwrap();
        assert_eq!(modular.basis, direct.basis);
        assert!(modular.metrics.modular_primes >= 2);
        assert!(modular.metrics.reconstruction_attempts >= 1);
        assert!(modular.metrics.trace_learning_rounds >= 1);
        assert!(modular.metrics.trace_application_rounds >= 1);
        assert!(modular.metrics.trace_replayed_rows >= 1);
        assert!(modular.metrics.peak_matrix_nonzeros > 0);
        assert!(modular.metrics.peak_workspace_bytes > 0);
        assert_eq!(modular.metrics.trace_rejections, 0);
        assert_eq!(
            modular.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
    }

    #[test]
    fn modular_tracked_basis_reconstructs_and_verifies_its_change_matrix() {
        let field = RationalCoefficientField;
        let generators = vec![
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[2, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (monomial(&[0, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                ],
                &field,
            ),
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[1, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (monomial(&[0, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                ],
                &field,
            ),
        ];
        let options = GroebnerOptions::default();
        let plain = groebner_basis_rational(&generators, &options).unwrap();
        let tracked = groebner_basis_with_transform_rational(&generators, &options).unwrap();
        assert_eq!(tracked.basis, plain.basis);
        assert_eq!(tracked.transformations.len(), tracked.basis.len());
        verify_transform_candidate(&generators, &tracked.basis, &tracked.transformations).unwrap();
        assert!(tracked.metrics.modular_primes >= 2);
        assert_eq!(
            tracked.metrics.rational_transformation_backend,
            Some(RationalTransformationBackend::Modular)
        );
    }

    #[test]
    fn semiregular_rational_change_matrix_uses_primitive_tracking() {
        let field = RationalCoefficientField;
        let mut state = 0x4D59_5DF4_D0F3_3173_u64;
        let mut coefficient = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let residue = state % 65_520 + 1;
            let signed = if residue <= 32_760 {
                residue as i64
            } else {
                residue as i64 - 65_521
            };
            Rational::from(signed)
        };
        let generators = (0..3)
            .map(|_| {
                let mut terms = Vec::new();
                for left in 0..3 {
                    for right in left..3 {
                        let mut exponent = vec![0_u32; 3];
                        exponent[left] += 1;
                        exponent[right] += 1;
                        terms.push((Monomial::new(exponent.into_boxed_slice()), coefficient()));
                    }
                }
                for variable in 0..3 {
                    let mut exponent = vec![0_u32; 3];
                    exponent[variable] = 1;
                    terms.push((Monomial::new(exponent.into_boxed_slice()), coefficient()));
                }
                terms.push((Monomial::one(3), coefficient()));
                SparsePolynomial::new(3, MonomialOrder::GradedReverseLexicographic, terms, &field)
            })
            .collect::<Vec<_>>();
        let output =
            groebner_basis_with_transform_rational(&generators, &GroebnerOptions::default())
                .unwrap();
        assert_eq!(output.basis.len(), 6);
        verify_transform_candidate(&generators, &output.basis, &output.transformations).unwrap();
        assert_eq!(
            output.metrics.rational_transformation_backend,
            Some(RationalTransformationBackend::PrimitiveInteger)
        );
    }

    #[test]
    fn modular_basis_reconstructs_a_large_denominator_after_repeated_trace_replay() {
        let field = RationalCoefficientField;
        let denominator = Integer::from(2).pow(180) + Integer::from(41);
        let coefficient = Rational::new(Integer::one(), denominator).unwrap();
        let generators = vec![
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[2, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (monomial(&[0, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                ],
                &field,
            ),
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[1, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (monomial(&[0, 0, 0, 0, 0, 0, 0, 0, 0]), coefficient),
                ],
                &field,
            ),
        ];
        let output = groebner_basis_rational(&generators, &GroebnerOptions::default()).unwrap();

        let direct =
            crate::groebner_basis(&field, &generators, &GroebnerOptions::default()).unwrap();
        assert_eq!(output.basis, direct.basis);
        assert_eq!(
            output.metrics.rational_backend,
            Some(RationalBackend::Modular)
        );
        assert!(output.metrics.modular_primes >= 4);
        assert!(output.metrics.trace_application_rounds > 0);
    }

    #[test]
    fn rational_reconstruction_recovers_signed_fractions() {
        let modulus = Integer::from(1_000_003_u32) * Integer::from(1_000_033_u32);
        for value in [
            Rational::new(Integer::from(17), Integer::from(31)).unwrap(),
            Rational::new(Integer::from(-29), Integer::from(37)).unwrap(),
            Rational::from(42),
        ] {
            let numerator = value.numerator();
            let denominator = value.denominator();
            let residue = (&numerator
                * denominator
                    .invert(&modulus)
                    .expect("the test denominator is invertible"))
                % &modulus;
            let residue = if residue.is_negative() {
                &residue + &modulus
            } else {
                residue
            };
            assert_eq!(rational_reconstruct(&residue, &modulus), Some(value));
        }
    }

    #[test]
    fn word_rational_reconstruction_matches_the_arbitrary_integer_floor() {
        let modulus_word = 1_000_003_u64 * 1_000_033_u64;
        let modulus = Integer::from(modulus_word);
        let mut residue_word = 17_u64;
        for _ in 0..512 {
            residue_word = (u128::from(residue_word) * 6_364_136_223_846_793_005_u128
                + 1_442_695_040_888_963_407_u128)
                .rem_euclid(u128::from(modulus_word)) as u64;
            let residue = Integer::from(residue_word);
            assert_eq!(
                rational_reconstruct_word(residue_word, modulus_word),
                rational_reconstruct_big(&residue, &modulus)
            );
        }
    }

    #[test]
    fn rational_reconstruction_accepts_asymmetric_bounds() {
        let modulus = Integer::from(2).pow(601) - Integer::one();
        let value = Rational::new(
            Integer::from(2).pow(360) + Integer::from(17),
            Integer::from(2).pow(120) + Integer::from(41),
        )
        .unwrap();
        let residue = (&value.numerator()
            * value
                .denominator()
                .invert(&modulus)
                .expect("the denominator is invertible"))
            % &modulus;
        let residue = if residue.is_negative() {
            residue + &modulus
        } else {
            residue
        };
        assert_eq!(rational_reconstruct(&residue, &modulus), Some(value));
    }

    #[test]
    fn reconstruction_state_retains_only_coefficients_witnessed_by_new_images() {
        let field = RationalCoefficientField;
        let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
        let rational_basis = vec![SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            [(monomial(&[1]), half.clone())],
            &field,
        )];
        let first_field = PrimeField32::new(101).unwrap();
        let first = rational_basis
            .iter()
            .map(|polynomial| rational_image(polynomial, &first_field))
            .collect::<Vec<_>>();
        let mut state = ReconstructionState::from_image(&first, 101);
        assert_eq!(
            state.reconstruct(1, MonomialOrder::Lexicographic),
            Some(rational_basis.clone())
        );
        assert_eq!(state.stable, vec![Some(half)]);

        let second_field = PrimeField32::new(103).unwrap();
        let second = rational_basis
            .iter()
            .map(|polynomial| rational_image(polynomial, &second_field))
            .collect::<Vec<_>>();
        state.incorporate(&second, 103);
        assert!(state.stable[0].is_some());

        let third_field = PrimeField32::new(107).unwrap();
        let expected_third = rational_basis
            .iter()
            .map(|polynomial| rational_image(polynomial, &third_field))
            .collect::<Vec<_>>();
        let incompatible = vec![SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            [(monomial(&[1]), (expected_third[0].terms()[0].1 + 1) % 107)],
            &third_field,
        )];
        state.incorporate(&incompatible, 107);
        assert!(state.stable[0].is_none());
    }

    #[test]
    fn trace_batch_replay_is_ordered_and_matches_independent_prime_images() {
        let field = RationalCoefficientField;
        let generators = [
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[2, 0]), Rational::one()),
                    (monomial(&[0, 1]), Rational::one()),
                ],
                &field,
            ),
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[1, 1]), Rational::one()),
                    (monomial(&[0, 0]), Rational::one()),
                ],
                &field,
            ),
        ];
        let options = GroebnerOptions::default();
        let learning_field = PrimeField32::new(101).unwrap();
        let learning_images = generators
            .iter()
            .map(|polynomial| rational_image(polynomial, &learning_field))
            .collect::<Vec<_>>();
        let (_, trace) = learn_prime_trace(&learning_field, &learning_images, &options).unwrap();
        let primes = [103, 107, 109, 113];
        let prepared = primes
            .iter()
            .map(|prime| {
                let field = PrimeField32::new(*prime).unwrap();
                let images = generators
                    .iter()
                    .map(|polynomial| rational_image(polynomial, &field))
                    .collect::<Vec<_>>();
                PreparedPrime {
                    prime: *prime,
                    field,
                    images,
                }
            })
            .collect();
        let replayed = replay_prime_batch(&trace, prepared, &options).unwrap();

        assert_eq!(
            replayed
                .iter()
                .map(|image| image.prepared.prime)
                .collect::<Vec<_>>(),
            primes
        );
        for image in replayed {
            let expected =
                groebner_basis_prime(&image.prepared.field, &image.prepared.images, &options)
                    .unwrap();
            let actual = image.replayed.expect("the shared trace is compatible").0;
            assert_eq!(actual, expected.basis);
        }
    }

    #[test]
    fn modular_unit_certificate_reconstructs_large_denominators() {
        let field = RationalCoefficientField;
        let scale = Integer::from(10_u32).pow(30) + Integer::from(151_u32);
        let generators = vec![
            SparsePolynomial::new(
                1,
                MonomialOrder::Lexicographic,
                [(monomial(&[1]), Rational::from(scale.clone()))],
                &field,
            ),
            SparsePolynomial::new(
                1,
                MonomialOrder::Lexicographic,
                [
                    (monomial(&[1]), Rational::from(-1)),
                    (monomial(&[0]), Rational::one()),
                ],
                &field,
            ),
        ];
        let options = GroebnerOptions {
            max_modular_primes: 8,
            ..GroebnerOptions::default()
        };
        let output = unit_ideal_certificate_rational(&generators, &options)
            .unwrap()
            .expect("the family generates one");
        verify_certificate_candidate(&generators, &output.multipliers).unwrap();
        assert!(output.metrics.modular_primes >= 4);
        assert!(output.metrics.reconstruction_attempts >= 1);
        assert_eq!(
            output.multipliers[0].terms()[0].1,
            Rational::new(Integer::one(), scale).unwrap()
        );
    }

    #[test]
    fn modular_macaulay_certificate_learns_degree_and_verifies() {
        let field = RationalCoefficientField;
        let scale = Integer::from(10_u32).pow(20) + Integer::from(39_u32);
        let generators = vec![
            SparsePolynomial::new(
                1,
                MonomialOrder::Lexicographic,
                [(monomial(&[1]), Rational::from(scale))],
                &field,
            ),
            SparsePolynomial::new(
                1,
                MonomialOrder::Lexicographic,
                [
                    (monomial(&[1]), Rational::from(-1)),
                    (monomial(&[0]), Rational::one()),
                ],
                &field,
            ),
        ];
        assert_eq!(
            rational_macaulay_certificate_degree(&generators, 1, 100).unwrap(),
            Some(1)
        );
        let options = GroebnerOptions {
            max_modular_primes: 16,
            ..GroebnerOptions::default()
        };
        let output = macaulay_unit_ideal_certificate_rational(&generators, 1, 100, &options)
            .unwrap()
            .expect("the bounded Macaulay system represents one");
        verify_certificate_candidate(&generators, &output.multipliers).unwrap();
        assert!(output.metrics.reconstruction_attempts >= 1);
    }

    #[test]
    fn modular_unit_certificate_distinguishes_a_proper_rational_ideal() {
        let field = RationalCoefficientField;
        let generators = vec![SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            [(monomial(&[1]), Rational::one())],
            &field,
        )];
        assert!(
            unit_ideal_certificate_rational(&generators, &GroebnerOptions::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn an_insufficient_prime_budget_uses_the_direct_exact_floor() {
        let field = RationalCoefficientField;
        let generators = vec![SparsePolynomial::new(
            9,
            MonomialOrder::Lexicographic,
            [
                (monomial(&[2, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                (monomial(&[0, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::from(2)),
            ],
            &field,
        )];
        let options = GroebnerOptions {
            max_modular_primes: 1,
            ..GroebnerOptions::default()
        };
        let output = groebner_basis_rational(&generators, &options).unwrap();
        assert_eq!(output.basis, generators);
        assert_eq!(output.metrics.modular_primes, 1);
        assert_eq!(output.evidence, GroebnerEvidence::ExactConstruction);
    }

    #[test]
    fn an_unlucky_learning_prime_rejects_the_trace_and_falls_back() {
        let field = RationalCoefficientField;
        let mut candidate = (1_u32 << 30) - 1;
        let (first_prime, _) = next_usable_prime(&mut candidate, &[]).unwrap();
        let generators = vec![
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[2, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (
                        monomial(&[1, 0, 0, 0, 0, 0, 0, 0, 0]),
                        Rational::from(first_prime),
                    ),
                    (monomial(&[0, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                ],
                &field,
            ),
            SparsePolynomial::new(
                9,
                MonomialOrder::GradedReverseLexicographic,
                [
                    (monomial(&[1, 1, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                    (monomial(&[0, 0, 0, 0, 0, 0, 0, 0, 0]), Rational::one()),
                ],
                &field,
            ),
        ];
        let options = GroebnerOptions {
            max_modular_primes: 4,
            ..GroebnerOptions::default()
        };
        let output = groebner_basis_rational(&generators, &options).unwrap();
        assert!(output.metrics.trace_learning_rounds > 0);
        assert!(output.metrics.trace_rejections > 0);
        verify_candidate(&generators, &output.basis, &options).unwrap();
    }
}
