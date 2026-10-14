// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strong Gröbner bases over the integers using S- and G-polynomials.

use std::collections::VecDeque;

use flint_core::Integer;
use poly_core::{CoefficientRing, SparsePolynomial};

use crate::engine::{GroebnerError, GroebnerOptions};

/// Validated arbitrary-precision integer coefficient lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegerCoefficientRing;

impl CoefficientRing for IntegerCoefficientRing {
    type Element = Integer;

    fn zero(&self) -> Self::Element {
        Integer::zero()
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

/// Resource bounds for an integral strong-basis computation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongGroebnerOptions {
    /// Maximum number of basis elements.
    pub max_basis_size: usize,
    /// Maximum number of pending pairs.
    pub max_pending_pairs: usize,
    /// Maximum processed pairs.
    pub max_processed_pairs: usize,
    /// Maximum reduction steps in one strong normal form.
    pub max_reduction_steps: usize,
}

impl Default for StrongGroebnerOptions {
    fn default() -> Self {
        let field_defaults = GroebnerOptions::default();
        Self {
            max_basis_size: field_defaults.max_basis_size,
            max_pending_pairs: field_defaults.max_pending_pairs,
            max_processed_pairs: 4_000_000,
            max_reduction_steps: field_defaults.max_reduction_steps,
        }
    }
}

/// Counters from an integral strong-basis computation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StrongGroebnerMetrics {
    /// Pairs processed.
    pub pairs_processed: usize,
    /// S-polynomials constructed.
    pub s_polynomials: usize,
    /// G-polynomials constructed.
    pub g_polynomials: usize,
    /// Nonzero remainders adjoined.
    pub basis_insertions: usize,
    /// Largest pending-pair queue.
    pub peak_pending_pairs: usize,
}

/// Verified normalized strong basis over `Z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongGroebnerOutput {
    /// Positive-leading-coefficient strong basis.
    pub basis: Vec<SparsePolynomial<Integer>>,
    /// Operation counters.
    pub metrics: StrongGroebnerMetrics,
}

/// Euclidean strong reduction together with exact quotient multipliers.
///
/// For input `value` and reducers `basis`, the output satisfies
/// `value = sum(quotients[i] * basis[i]) + remainder` exactly over `Z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongNormalFormOutput {
    /// Canonical Euclidean remainder for the declared reducer order.
    pub remainder: SparsePolynomial<Integer>,
    /// Polynomial multiplier for each reducer, in input order.
    pub quotients: Vec<SparsePolynomial<Integer>>,
}

/// Compute a normalized strong Gröbner basis over `Z`.
pub fn strong_groebner_basis(
    generators: &[SparsePolynomial<Integer>],
    options: &StrongGroebnerOptions,
) -> Result<StrongGroebnerOutput, GroebnerError> {
    validate_inputs(generators)?;
    let ring = IntegerCoefficientRing;
    let mut basis = generators
        .iter()
        .filter(|polynomial| !polynomial.is_zero())
        .cloned()
        .map(normalize_sign)
        .collect::<Vec<_>>();
    deduplicate(&mut basis);
    if basis.len() > options.max_basis_size {
        return Err(resource("basis size", options.max_basis_size));
    }

    let mut pending = VecDeque::new();
    for right in 0..basis.len() {
        for left in 0..right {
            push_pair(&mut pending, (left, right), options.max_pending_pairs)?;
        }
    }
    let mut metrics = StrongGroebnerMetrics {
        peak_pending_pairs: pending.len(),
        ..StrongGroebnerMetrics::default()
    };

    while let Some((left, right)) = pending.pop_front() {
        if metrics.pairs_processed == options.max_processed_pairs {
            return Err(resource("processed pairs", options.max_processed_pairs));
        }
        metrics.pairs_processed += 1;
        let s = integer_s_polynomial(&basis[left], &basis[right])?;
        metrics.s_polynomials += 1;
        let g = integer_g_polynomial(&basis[left], &basis[right])?;
        metrics.g_polynomials += 1;
        for candidate in [s, g] {
            let remainder = strong_normal_form(&candidate, &basis, options.max_reduction_steps)?;
            if remainder.is_zero() {
                continue;
            }
            let remainder = normalize_sign(remainder);
            if basis.contains(&remainder) {
                continue;
            }
            if basis.len() == options.max_basis_size {
                return Err(resource("basis size", options.max_basis_size));
            }
            let new_index = basis.len();
            basis.push(remainder);
            metrics.basis_insertions += 1;
            for old in 0..new_index {
                push_pair(&mut pending, (old, new_index), options.max_pending_pairs)?;
            }
            metrics.peak_pending_pairs = metrics.peak_pending_pairs.max(pending.len());
        }
    }

    autoreduce(&ring, &mut basis, options.max_reduction_steps)?;
    verify_strong_basis(&basis, options.max_reduction_steps)?;
    for generator in generators {
        if !strong_normal_form(generator, &basis, options.max_reduction_steps)?.is_zero() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(StrongGroebnerOutput { basis, metrics })
}

/// Euclidean strong normal form over `Z`.
pub fn strong_normal_form(
    value: &SparsePolynomial<Integer>,
    basis: &[SparsePolynomial<Integer>],
    max_steps: usize,
) -> Result<SparsePolynomial<Integer>, GroebnerError> {
    Ok(strong_normal_form_with_quotients(value, basis, max_steps)?.remainder)
}

/// Compute Euclidean strong normal form and exact reducer multipliers over `Z`.
pub fn strong_normal_form_with_quotients(
    value: &SparsePolynomial<Integer>,
    basis: &[SparsePolynomial<Integer>],
    max_steps: usize,
) -> Result<StrongNormalFormOutput, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let mut current = value.clone();
    let mut remainder = Vec::new();
    let mut quotients = basis
        .iter()
        .map(|_| SparsePolynomial::new(value.arity(), value.order(), [], &ring))
        .collect::<Vec<_>>();
    let mut steps = 0;
    while let Some((leading_monomial, leading_coefficient)) = current.leading_term() {
        if steps == max_steps {
            return Err(resource("strong normal-form steps", max_steps));
        }
        let reduction = basis.iter().enumerate().find_map(|(index, reducer)| {
            let (reducer_monomial, reducer_coefficient) = reducer.leading_term()?;
            let multiplier = leading_monomial.quotient(reducer_monomial)?;
            let quotient = leading_coefficient.div_floor(&reducer_coefficient.abs());
            (!quotient.is_zero()).then_some((index, reducer, multiplier, quotient))
        });
        if let Some((index, reducer, multiplier, quotient)) = reduction {
            current = current.sub(&reducer.term_mul(&multiplier, &quotient, &ring), &ring);
            let quotient_term = SparsePolynomial::new(
                value.arity(),
                value.order(),
                [(multiplier, quotient)],
                &ring,
            );
            quotients[index] = quotients[index].add(&quotient_term, &ring);
        } else {
            let leading = current.terms()[0].clone();
            remainder.push(leading.clone());
            let term = SparsePolynomial::new(current.arity(), current.order(), [leading], &ring);
            current = current.sub(&term, &ring);
        }
        steps += 1;
    }
    Ok(StrongNormalFormOutput {
        remainder: SparsePolynomial::new(value.arity(), value.order(), remainder, &ring),
        quotients,
    })
}

fn integer_s_polynomial(
    left: &SparsePolynomial<Integer>,
    right: &SparsePolynomial<Integer>,
) -> Result<SparsePolynomial<Integer>, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let (left_monomial, left_coefficient) = left.leading_term().expect("nonzero basis member");
    let (right_monomial, right_coefficient) = right.leading_term().expect("nonzero basis member");
    let monomial_lcm = left_monomial
        .lcm(right_monomial)
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let coefficient_lcm = left_coefficient.abs().lcm(&right_coefficient.abs());
    let left_scalar = &coefficient_lcm / left_coefficient;
    let right_scalar = &coefficient_lcm / right_coefficient;
    let left_multiplier = monomial_lcm
        .quotient(left_monomial)
        .expect("monomial lcm divisibility");
    let right_multiplier = monomial_lcm
        .quotient(right_monomial)
        .expect("monomial lcm divisibility");
    Ok(left.term_mul(&left_multiplier, &left_scalar, &ring).sub(
        &right.term_mul(&right_multiplier, &right_scalar, &ring),
        &ring,
    ))
}

fn integer_g_polynomial(
    left: &SparsePolynomial<Integer>,
    right: &SparsePolynomial<Integer>,
) -> Result<SparsePolynomial<Integer>, GroebnerError> {
    let ring = IntegerCoefficientRing;
    let (left_monomial, left_coefficient) = left.leading_term().expect("nonzero basis member");
    let (right_monomial, right_coefficient) = right.leading_term().expect("nonzero basis member");
    let monomial_lcm = left_monomial
        .lcm(right_monomial)
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let (_, left_bezout, right_bezout) = extended_gcd(left_coefficient, right_coefficient);
    let left_multiplier = monomial_lcm
        .quotient(left_monomial)
        .expect("monomial lcm divisibility");
    let right_multiplier = monomial_lcm
        .quotient(right_monomial)
        .expect("monomial lcm divisibility");
    Ok(left.term_mul(&left_multiplier, &left_bezout, &ring).add(
        &right.term_mul(&right_multiplier, &right_bezout, &ring),
        &ring,
    ))
}

pub(crate) fn extended_gcd(left: &Integer, right: &Integer) -> (Integer, Integer, Integer) {
    let mut old_remainder = left.abs();
    let mut remainder = right.abs();
    let mut old_left = Integer::one();
    let mut current_left = Integer::zero();
    let mut old_right = Integer::zero();
    let mut current_right = Integer::one();
    while !remainder.is_zero() {
        let quotient = &old_remainder / &remainder;
        let next_remainder = &old_remainder - &(&quotient * &remainder);
        old_remainder = remainder;
        remainder = next_remainder;

        let next_left = &old_left - &(&quotient * &current_left);
        old_left = current_left;
        current_left = next_left;

        let next_right = &old_right - &(&quotient * &current_right);
        old_right = current_right;
        current_right = next_right;
    }
    if left.is_negative() {
        old_left = -old_left;
    }
    if right.is_negative() {
        old_right = -old_right;
    }
    (old_remainder, old_left, old_right)
}

fn normalize_sign(polynomial: SparsePolynomial<Integer>) -> SparsePolynomial<Integer> {
    let ring = IntegerCoefficientRing;
    if polynomial
        .leading_term()
        .is_some_and(|(_, coefficient)| coefficient.is_negative())
    {
        polynomial.scalar_mul(&Integer::from(-1), &ring)
    } else {
        polynomial
    }
}

fn autoreduce(
    _: &IntegerCoefficientRing,
    basis: &mut Vec<SparsePolynomial<Integer>>,
    max_steps: usize,
) -> Result<(), GroebnerError> {
    let mut changed = true;
    let mut passes = 0;
    while changed && passes <= basis.len().saturating_add(1) {
        changed = false;
        let mut index = 0;
        while index < basis.len() {
            let reducers = basis
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            let reduced = normalize_sign(strong_normal_form(&basis[index], &reducers, max_steps)?);
            if reduced.is_zero() {
                basis.remove(index);
                changed = true;
            } else {
                if reduced != basis[index] {
                    basis[index] = reduced;
                    changed = true;
                }
                index += 1;
            }
        }
        deduplicate(basis);
        passes += 1;
    }
    basis.sort_by(|left, right| {
        right
            .leading_monomial()
            .expect("nonzero")
            .compare(left.leading_monomial().expect("nonzero"), left.order())
            .then_with(|| {
                left.leading_term()
                    .expect("nonzero")
                    .1
                    .cmp(right.leading_term().expect("nonzero").1)
            })
    });
    Ok(())
}

fn verify_strong_basis(
    basis: &[SparsePolynomial<Integer>],
    max_steps: usize,
) -> Result<(), GroebnerError> {
    for polynomial in basis {
        if polynomial.is_zero()
            || polynomial
                .leading_term()
                .is_some_and(|(_, coefficient)| !coefficient.is_positive())
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    for right in 0..basis.len() {
        for left in 0..right {
            let s = integer_s_polynomial(&basis[left], &basis[right])?;
            if !strong_normal_form(&s, basis, max_steps)?.is_zero() {
                return Err(GroebnerError::VerificationFailed);
            }
            let g = integer_g_polynomial(&basis[left], &basis[right])?;
            if !strong_normal_form(&g, basis, max_steps)?.is_zero() {
                return Err(GroebnerError::VerificationFailed);
            }
        }
    }
    Ok(())
}

fn validate_inputs(generators: &[SparsePolynomial<Integer>]) -> Result<(), GroebnerError> {
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    if generators
        .iter()
        .any(|value| value.arity() != first.arity() || value.order() != first.order())
    {
        Err(GroebnerError::ContextMismatch)
    } else {
        Ok(())
    }
}

fn push_pair(
    pending: &mut VecDeque<(usize, usize)>,
    pair: (usize, usize),
    limit: usize,
) -> Result<(), GroebnerError> {
    if pending.len() == limit {
        Err(resource("pending pairs", limit))
    } else {
        pending.push_back(pair);
        Ok(())
    }
}

fn deduplicate(basis: &mut Vec<SparsePolynomial<Integer>>) {
    let mut unique = Vec::with_capacity(basis.len());
    for polynomial in basis.drain(..) {
        if !unique.contains(&polynomial) {
            unique.push(polynomial);
        }
    }
    *basis = unique;
}

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use poly_core::{Monomial, MonomialOrder, SparsePolynomial};

    use super::*;

    fn monomial(exponents: &[u32]) -> Monomial {
        Monomial::new(exponents.to_vec().into_boxed_slice())
    }

    fn polynomial(terms: &[(i64, &[u32])]) -> SparsePolynomial<Integer> {
        SparsePolynomial::new(
            1,
            MonomialOrder::Lexicographic,
            terms
                .iter()
                .map(|(coefficient, exponents)| (monomial(exponents), Integer::from(*coefficient))),
            &IntegerCoefficientRing,
        )
    }

    #[test]
    fn g_polynomial_recovers_coefficient_gcd() {
        let generators = [polynomial(&[(2, &[1])]), polynomial(&[(3, &[1])])];
        let output = strong_groebner_basis(&generators, &StrongGroebnerOptions::default()).unwrap();
        assert!(output
            .basis
            .iter()
            .any(|value| { value.terms() == [(monomial(&[1]), Integer::one())].as_slice() }));
        assert!(output.metrics.g_polynomials > 0);
    }

    #[test]
    fn strong_basis_reduces_original_generators() {
        let generators = [
            polynomial(&[(2, &[2]), (4, &[0])]),
            polynomial(&[(3, &[1]), (6, &[0])]),
        ];
        let output = strong_groebner_basis(&generators, &StrongGroebnerOptions::default()).unwrap();
        for generator in &generators {
            assert!(strong_normal_form(generator, &output.basis, 1_000_000)
                .unwrap()
                .is_zero());
        }
    }

    #[test]
    fn extended_gcd_tracks_signs() {
        let (gcd, left, right) = extended_gcd(&Integer::from(-30), &Integer::from(18));
        assert_eq!(gcd, Integer::from(6));
        assert_eq!(
            &(&left * &Integer::from(-30)) + &(&right * &Integer::from(18)),
            gcd
        );
    }

    #[test]
    fn strong_normal_form_quotients_reconstruct_the_input() {
        let basis = [
            polynomial(&[(2, &[1]), (1, &[0])]),
            polynomial(&[(3, &[0])]),
        ];
        let value = polynomial(&[(7, &[2]), (-5, &[1]), (11, &[0])]);
        let output = strong_normal_form_with_quotients(&value, &basis, 1_000_000).unwrap();
        let reconstructed = output.quotients.iter().zip(&basis).fold(
            output.remainder.clone(),
            |sum, (quotient, reducer)| {
                sum.add(
                    &quotient.multiply(reducer, &IntegerCoefficientRing, None),
                    &IntegerCoefficientRing,
                )
            },
        );
        assert_eq!(reconstructed, value);
        assert_eq!(
            output.remainder,
            strong_normal_form(&value, &basis, 1_000_000).unwrap()
        );
    }
}
