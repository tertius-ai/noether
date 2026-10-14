// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bounded exact Hilbert invariants of monomial ideals.

use std::collections::BTreeMap;

use crate::{Integer, Rational};
use poly_core::{MonomialArena, MonomialArenaError, MonomialId, MonomialOrder};

use super::{Degree, FreeAbelianGrading, GroebnerError};

const MAX_HILBERT_DEGREE: usize = 4_096;
const MAX_HILBERT_RECURSION_DEPTH: usize = 512;
const MAX_HILBERT_STATES: usize = 100_000;
const MAX_HILBERT_TERMS: usize = 100_000;

pub(super) struct HilbertData {
    pub(super) numerator: Vec<Integer>,
    pub(super) reduced_numerator: Vec<Integer>,
    pub(super) dimension: i64,
    pub(super) degree: Integer,
}

pub(super) fn invariants(
    variable_count: usize,
    leading_monomials: &[Vec<Integer>],
) -> Result<HilbertData, GroebnerError> {
    let (arena, generators) = intern_leading_monomials(variable_count, leading_monomials)?;
    let mut computation = HilbertComputation {
        arena,
        memo: BTreeMap::new(),
        states: 0,
    };
    let numerator = computation.k_polynomial(generators, 0)?;
    let (vanishing_order, reduced_numerator) = reduce_at_one(&numerator);
    let dimension = if is_zero_polynomial(&numerator) {
        -1
    } else {
        let variable_count = i64::try_from(variable_count)
            .map_err(|_| resource_limit("variable count", i64::MAX as usize))?;
        let vanishing_order = i64::try_from(vanishing_order)
            .map_err(|_| resource_limit("vanishing order", i64::MAX as usize))?;
        variable_count - vanishing_order
    };
    if dimension < 0 && !is_zero_polynomial(&numerator) {
        return Err(GroebnerError::InvalidBasis(
            "the initial ideal has impossible negative Krull dimension",
        ));
    }
    let degree = if dimension < 0 {
        Integer::zero()
    } else {
        sum(&reduced_numerator)
    };
    if degree.sign() < 0 {
        return Err(GroebnerError::InvalidBasis(
            "the initial ideal produced a negative multiplicity",
        ));
    }
    Ok(HilbertData {
        numerator,
        reduced_numerator,
        dimension,
        degree,
    })
}

/// Numerator of the Hilbert series in the group algebra of the grading.
pub(super) fn multigraded_numerator(
    grading: &FreeAbelianGrading,
    leading_monomials: &[Vec<Integer>],
) -> Result<Vec<(Degree, Integer)>, GroebnerError> {
    let (arena, generators) = intern_leading_monomials(grading.arity(), leading_monomials)?;
    let mut computation = MultigradedHilbertComputation {
        grading,
        arena,
        memo: BTreeMap::new(),
        states: 0,
    };
    Ok(computation
        .k_polynomial(generators, 0)?
        .into_iter()
        .collect())
}

pub(super) fn hilbert_polynomial_coefficients(
    data: &HilbertData,
) -> Result<Vec<Rational>, GroebnerError> {
    if data.dimension <= 0 {
        return Ok(vec![Rational::zero()]);
    }
    let denominator_power = usize::try_from(data.dimension)
        .map_err(|_| resource_limit("Hilbert denominator power", MAX_HILBERT_DEGREE))?;
    if denominator_power > MAX_HILBERT_DEGREE {
        return Err(resource_limit(
            "Hilbert denominator power",
            MAX_HILBERT_DEGREE,
        ));
    }
    let polynomial_degree = denominator_power - 1;
    let mut result = vec![Rational::zero(); polynomial_degree + 1];
    for (index, coefficient) in data.reduced_numerator.iter().enumerate() {
        if coefficient.is_zero() {
            continue;
        }
        let offset = i64::try_from(denominator_power - 1)
            .and_then(|value| i64::try_from(index).map(|index| value - index))
            .map_err(|_| resource_limit("Hilbert numerator length", MAX_HILBERT_DEGREE))?;
        let binomial = binomial_polynomial(offset, polynomial_degree)?;
        let coefficient = Rational::from(coefficient.clone());
        for (target, value) in result.iter_mut().zip(binomial) {
            *target = &*target + &(&coefficient * &value);
        }
    }
    while result.len() > 1 && result.last().is_some_and(Rational::is_zero) {
        result.pop();
    }
    Ok(result)
}

type HilbertState = Vec<MonomialId>;

struct HilbertComputation {
    arena: MonomialArena,
    memo: BTreeMap<HilbertState, Vec<Integer>>,
    states: usize,
}

impl HilbertComputation {
    fn k_polynomial(
        &mut self,
        generators: HilbertState,
        depth: usize,
    ) -> Result<Vec<Integer>, GroebnerError> {
        if depth > MAX_HILBERT_RECURSION_DEPTH {
            return Err(resource_limit(
                "Hilbert recursion depth",
                MAX_HILBERT_RECURSION_DEPTH,
            ));
        }
        let generators = minimalize(&self.arena, generators);
        if let Some(cached) = self.memo.get(&generators) {
            return Ok(cached.clone());
        }
        self.states = self.states.saturating_add(1);
        if self.states > MAX_HILBERT_STATES {
            return Err(resource_limit(
                "Hilbert recursion state count",
                MAX_HILBERT_STATES,
            ));
        }

        let result = if generators.is_empty() {
            vec![Integer::one()]
        } else if generators
            .iter()
            .any(|generator| self.arena.total_degree(*generator) == 0)
        {
            vec![Integer::zero()]
        } else if generators.len() == 1 {
            one_minus_t_power(total_degree(&self.arena, generators[0])?)
        } else if let Some(components) = support_components(&self.arena, &generators) {
            let mut product = vec![Integer::one()];
            for component in components {
                product = polynomial_product(&product, &self.k_polynomial(component, depth + 1)?)?;
            }
            trim(product)
        } else {
            let (_, plus, colon) = choose_pivot(&mut self.arena, &generators)?;
            let plus_numerator = self.k_polynomial(plus, depth + 1)?;
            let colon_numerator = self.k_polynomial(colon, depth + 1)?;
            polynomial_add_shifted(plus_numerator, &colon_numerator, 1)?
        };
        self.memo.insert(generators, result.clone());
        Ok(result)
    }
}

type MultigradedPolynomial = BTreeMap<Degree, Integer>;

struct MultigradedHilbertComputation<'a> {
    grading: &'a FreeAbelianGrading,
    arena: MonomialArena,
    memo: BTreeMap<HilbertState, MultigradedPolynomial>,
    states: usize,
}

impl MultigradedHilbertComputation<'_> {
    fn k_polynomial(
        &mut self,
        generators: HilbertState,
        depth: usize,
    ) -> Result<MultigradedPolynomial, GroebnerError> {
        if depth > MAX_HILBERT_RECURSION_DEPTH {
            return Err(resource_limit(
                "multigraded Hilbert recursion depth",
                MAX_HILBERT_RECURSION_DEPTH,
            ));
        }
        let generators = minimalize(&self.arena, generators);
        if let Some(cached) = self.memo.get(&generators) {
            return Ok(cached.clone());
        }
        self.states = self.states.saturating_add(1);
        if self.states > MAX_HILBERT_STATES {
            return Err(resource_limit(
                "multigraded Hilbert recursion state count",
                MAX_HILBERT_STATES,
            ));
        }

        let zero = Degree::zero(self.grading.rank());
        let result = if generators.is_empty() {
            BTreeMap::from([(zero, Integer::one())])
        } else if generators
            .iter()
            .any(|generator| self.arena.total_degree(*generator) == 0)
        {
            BTreeMap::new()
        } else if generators.len() == 1 {
            let degree = self.monomial_degree(generators[0])?;
            let mut result = BTreeMap::from([(zero, Integer::one())]);
            add_multigraded_term(&mut result, degree, -Integer::one());
            result
        } else if let Some(components) = support_components(&self.arena, &generators) {
            let mut product = BTreeMap::from([(zero, Integer::one())]);
            for component in components {
                product = multigraded_product(&product, &self.k_polynomial(component, depth + 1)?)?;
            }
            product
        } else {
            let (pivot, plus, colon) = choose_pivot(&mut self.arena, &generators)?;
            let mut result = self.k_polynomial(plus, depth + 1)?;
            let colon = self.k_polynomial(colon, depth + 1)?;
            let shift = &self.grading.variable_degrees()[pivot];
            for (degree, coefficient) in colon {
                let degree = degree.add(shift).map_err(|_| {
                    GroebnerError::InvalidBasis("the grading rank changed during Hilbert recursion")
                })?;
                add_multigraded_term(&mut result, degree, coefficient);
                if result.len() > MAX_HILBERT_TERMS {
                    return Err(resource_limit(
                        "multigraded Hilbert numerator term count",
                        MAX_HILBERT_TERMS,
                    ));
                }
            }
            result
        };
        if result.len() > MAX_HILBERT_TERMS {
            return Err(resource_limit(
                "multigraded Hilbert numerator term count",
                MAX_HILBERT_TERMS,
            ));
        }
        self.memo.insert(generators, result.clone());
        Ok(result)
    }

    fn monomial_degree(&self, generator: MonomialId) -> Result<Degree, GroebnerError> {
        self.grading
            .degree(
                &(0..self.arena.arity())
                    .map(|variable| Integer::from(self.arena.exponent(generator, variable)))
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| {
                GroebnerError::InvalidBasis("the grading is incompatible with a Hilbert monomial")
            })
    }
}

fn multigraded_product(
    left: &MultigradedPolynomial,
    right: &MultigradedPolynomial,
) -> Result<MultigradedPolynomial, GroebnerError> {
    let work = left.len().checked_mul(right.len()).ok_or_else(|| {
        resource_limit("multigraded Hilbert numerator product", MAX_HILBERT_TERMS)
    })?;
    if work > MAX_HILBERT_TERMS {
        return Err(resource_limit(
            "multigraded Hilbert numerator product",
            MAX_HILBERT_TERMS,
        ));
    }
    let mut result = BTreeMap::new();
    for (left_degree, left_coefficient) in left {
        for (right_degree, right_coefficient) in right {
            let degree = left_degree.add(right_degree).map_err(|_| {
                GroebnerError::InvalidBasis(
                    "the grading rank changed during Hilbert multiplication",
                )
            })?;
            add_multigraded_term(&mut result, degree, left_coefficient * right_coefficient);
        }
    }
    Ok(result)
}

fn add_multigraded_term(
    polynomial: &mut MultigradedPolynomial,
    degree: Degree,
    coefficient: Integer,
) {
    let coefficient = polynomial
        .remove(&degree)
        .map_or(coefficient.clone(), |known| &known + &coefficient);
    if !coefficient.is_zero() {
        polynomial.insert(degree, coefficient);
    }
}

fn intern_leading_monomials(
    variable_count: usize,
    leading_monomials: &[Vec<Integer>],
) -> Result<(MonomialArena, HilbertState), GroebnerError> {
    let mut arena = MonomialArena::with_capacity(
        variable_count,
        MonomialOrder::GradedReverseLexicographic,
        leading_monomials.len(),
    );
    let mut generators = Vec::with_capacity(leading_monomials.len());
    for generator in leading_monomials {
        if generator.len() != variable_count {
            return Err(GroebnerError::InvalidBasis(
                "a Hilbert monomial has the wrong number of variables",
            ));
        }
        let exponents = generator
            .iter()
            .map(bounded_exponent)
            .collect::<Result<Vec<_>, _>>()?;
        generators.push(intern_hilbert_monomial(&mut arena, &exponents)?);
    }
    Ok((arena, generators))
}

fn intern_hilbert_monomial(
    arena: &mut MonomialArena,
    exponents: &[u32],
) -> Result<MonomialId, GroebnerError> {
    arena
        .intern_exponents(exponents)
        .map_err(|error| match error {
            MonomialArenaError::ArityMismatch { .. } => {
                GroebnerError::InvalidBasis("a Hilbert monomial has the wrong number of variables")
            }
            MonomialArenaError::CapacityExceeded => {
                resource_limit("Hilbert monomial arena size", MAX_HILBERT_STATES)
            }
            MonomialArenaError::ExponentOverflow => {
                resource_limit("Hilbert monomial degree", MAX_HILBERT_DEGREE)
            }
        })
}

fn choose_pivot(
    arena: &mut MonomialArena,
    generators: &[MonomialId],
) -> Result<(usize, HilbertState, HilbertState), GroebnerError> {
    let mut best = None;
    for pivot in 0..arena.arity() {
        let occurrence = generators
            .iter()
            .filter(|generator| arena.exponent(**generator, pivot) > 0)
            .count();
        if occurrence == 0 {
            continue;
        }
        let (plus, colon) = child_states(arena, generators, pivot)?;
        let score = (
            plus.len().saturating_add(colon.len()),
            core::cmp::Reverse(occurrence),
            pivot,
        );
        if best
            .as_ref()
            .is_none_or(|(best_score, _, _)| score < *best_score)
        {
            best = Some((score, plus, colon));
        }
    }
    let Some(((_, _, pivot), plus, colon)) = best else {
        return Err(GroebnerError::InvalidBasis(
            "a nonunit Hilbert monomial uses no variable",
        ));
    };
    Ok((pivot, plus, colon))
}

fn child_states(
    arena: &mut MonomialArena,
    generators: &[MonomialId],
    pivot: usize,
) -> Result<(HilbertState, HilbertState), GroebnerError> {
    let mut pivot_generator = vec![0; arena.arity()];
    pivot_generator[pivot] = 1;
    let pivot_generator = intern_hilbert_monomial(arena, &pivot_generator)?;
    let mut plus = Vec::with_capacity(generators.len() + 1);
    plus.extend_from_slice(generators);
    plus.push(pivot_generator);

    let mut colon = Vec::with_capacity(generators.len());
    for generator in generators {
        let mut quotient = arena.get(*generator).to_u32_vec();
        quotient[pivot] = quotient[pivot].saturating_sub(1);
        colon.push(intern_hilbert_monomial(arena, &quotient)?);
    }
    Ok((minimalize(arena, plus), minimalize(arena, colon)))
}

fn bounded_exponent(exponent: &Integer) -> Result<u32, GroebnerError> {
    let parsed = exponent
        .to_string()
        .parse::<u32>()
        .map_err(|_| resource_limit("Hilbert monomial degree", MAX_HILBERT_DEGREE))?;
    if u64::from(parsed) > MAX_HILBERT_DEGREE as u64 {
        return Err(resource_limit(
            "Hilbert monomial degree",
            MAX_HILBERT_DEGREE,
        ));
    }
    Ok(parsed)
}

fn minimalize(arena: &MonomialArena, mut generators: HilbertState) -> HilbertState {
    generators.sort_by_key(|generator| (arena.total_degree(*generator), *generator));
    generators.dedup();
    let mut antichain = HilbertState::with_capacity(generators.len());
    for generator in generators {
        if antichain
            .iter()
            .any(|known| arena.divides(*known, generator))
        {
            continue;
        }
        antichain.retain(|known| !arena.divides(generator, *known));
        antichain.push(generator);
    }
    antichain.sort_unstable();
    antichain
}

fn support_components(
    arena: &MonomialArena,
    generators: &[MonomialId],
) -> Option<Vec<HilbertState>> {
    let variable_count = arena.arity();
    let mut parents = (0..variable_count).collect::<Vec<_>>();
    fn find(parents: &mut [usize], mut value: usize) -> usize {
        while parents[value] != value {
            parents[value] = parents[parents[value]];
            value = parents[value];
        }
        value
    }
    for generator in generators {
        let support = (0..variable_count)
            .filter(|variable| arena.exponent(*generator, *variable) != 0)
            .collect::<Vec<_>>();
        if let Some((&first, rest)) = support.split_first() {
            for &variable in rest {
                let first_root = find(&mut parents, first);
                let variable_root = find(&mut parents, variable);
                if first_root != variable_root {
                    parents[variable_root] = first_root;
                }
            }
        }
    }
    let mut components = BTreeMap::<usize, HilbertState>::new();
    for generator in generators {
        let variable = (0..variable_count)
            .find(|variable| arena.exponent(*generator, *variable) != 0)
            .expect("unit generators were handled before support factorization");
        let root = find(&mut parents, variable);
        components.entry(root).or_default().push(*generator);
    }
    (components.len() > 1).then(|| components.into_values().collect())
}

fn total_degree(arena: &MonomialArena, generator: MonomialId) -> Result<usize, GroebnerError> {
    let degree = arena.total_degree(generator);
    if degree > MAX_HILBERT_DEGREE as u64 {
        return Err(resource_limit(
            "Hilbert polynomial degree",
            MAX_HILBERT_DEGREE,
        ));
    }
    usize::try_from(degree)
        .map_err(|_| resource_limit("Hilbert polynomial degree", MAX_HILBERT_DEGREE))
}

fn one_minus_t_power(degree: usize) -> Vec<Integer> {
    let mut polynomial = vec![Integer::zero(); degree + 1];
    polynomial[0] = Integer::one();
    polynomial[degree] = &polynomial[degree] - &Integer::one();
    polynomial
}

fn polynomial_product(left: &[Integer], right: &[Integer]) -> Result<Vec<Integer>, GroebnerError> {
    let length = left
        .len()
        .checked_add(right.len())
        .and_then(|length| length.checked_sub(1))
        .ok_or_else(|| resource_limit("Hilbert polynomial degree", MAX_HILBERT_DEGREE))?;
    if length > MAX_HILBERT_DEGREE + 1 {
        return Err(resource_limit(
            "Hilbert polynomial degree",
            MAX_HILBERT_DEGREE,
        ));
    }
    let mut product = vec![Integer::zero(); length];
    for (left_index, left_coefficient) in left.iter().enumerate() {
        if left_coefficient.is_zero() {
            continue;
        }
        for (right_index, right_coefficient) in right.iter().enumerate() {
            if right_coefficient.is_zero() {
                continue;
            }
            let term = left_coefficient * right_coefficient;
            let index = left_index + right_index;
            product[index] = &product[index] + &term;
        }
    }
    Ok(product)
}

fn polynomial_add_shifted(
    mut target: Vec<Integer>,
    source: &[Integer],
    shift: usize,
) -> Result<Vec<Integer>, GroebnerError> {
    let required = source
        .len()
        .checked_add(shift)
        .ok_or_else(|| resource_limit("Hilbert polynomial degree", MAX_HILBERT_DEGREE))?;
    if required > MAX_HILBERT_DEGREE + 1 {
        return Err(resource_limit(
            "Hilbert polynomial degree",
            MAX_HILBERT_DEGREE,
        ));
    }
    if target.len() < required {
        target.resize_with(required, Integer::zero);
    }
    for (index, coefficient) in source.iter().enumerate() {
        target[index + shift] = &target[index + shift] + coefficient;
    }
    Ok(trim(target))
}

fn trim(mut polynomial: Vec<Integer>) -> Vec<Integer> {
    while polynomial.len() > 1 && polynomial.last().is_some_and(Integer::is_zero) {
        polynomial.pop();
    }
    polynomial
}

fn is_zero_polynomial(polynomial: &[Integer]) -> bool {
    polynomial.len() == 1 && polynomial[0].is_zero()
}

fn divide_by_t_minus_one(polynomial: &[Integer]) -> (Vec<Integer>, Integer) {
    if polynomial.len() <= 1 {
        return (
            vec![Integer::zero()],
            polynomial.first().cloned().unwrap_or_else(Integer::zero),
        );
    }
    let mut quotient = vec![Integer::zero(); polynomial.len() - 1];
    quotient[polynomial.len() - 2] = polynomial[polynomial.len() - 1].clone();
    for index in (0..polynomial.len() - 2).rev() {
        quotient[index] = &polynomial[index + 1] + &quotient[index + 1];
    }
    let remainder = &polynomial[0] + &quotient[0];
    (quotient, remainder)
}

fn reduce_at_one(numerator: &[Integer]) -> (usize, Vec<Integer>) {
    let mut polynomial = trim(numerator.to_vec());
    if is_zero_polynomial(&polynomial) {
        return (0, vec![Integer::zero()]);
    }
    let mut order = 0;
    while sum(&polynomial).is_zero() {
        let (quotient, remainder) = divide_by_t_minus_one(&polynomial);
        debug_assert!(remainder.is_zero());
        polynomial = trim(quotient.into_iter().map(|value| -value).collect());
        order += 1;
    }
    (order, polynomial)
}

fn sum(values: &[Integer]) -> Integer {
    values
        .iter()
        .fold(Integer::zero(), |total, value| &total + value)
}

fn binomial_polynomial(offset: i64, degree: usize) -> Result<Vec<Rational>, GroebnerError> {
    let mut polynomial = vec![Rational::one()];
    for index in 0..degree {
        let index = i64::try_from(index)
            .map_err(|_| resource_limit("Hilbert polynomial degree", MAX_HILBERT_DEGREE))?;
        let constant = Rational::from(offset - index);
        let mut next = vec![Rational::zero(); polynomial.len() + 1];
        for (power, coefficient) in polynomial.iter().enumerate() {
            next[power] = &next[power] + &(coefficient * &constant);
            next[power + 1] = &next[power + 1] + coefficient;
        }
        polynomial = next;
    }
    let factorial = (2..=degree).fold(Integer::one(), |value, factor| {
        &value * &Integer::from(factor)
    });
    let factorial = Rational::from(factorial);
    polynomial
        .into_iter()
        .map(|coefficient| {
            coefficient
                .checked_div(&factorial)
                .map_err(|_| GroebnerError::CoefficientInvariant)
        })
        .collect()
}

fn resource_limit(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monomials(values: &[&[u64]]) -> Vec<Vec<Integer>> {
        values
            .iter()
            .map(|value| value.iter().copied().map(Integer::from).collect())
            .collect()
    }

    #[test]
    fn monomial_examples_have_exact_invariants() {
        let cases = [
            (1, monomials(&[&[2]]), vec![1, 0, -1], 0, 2),
            (2, monomials(&[&[1, 1]]), vec![1, 0, -1], 1, 2),
            (2, monomials(&[&[1, 0], &[0, 1]]), vec![1, -2, 1], 0, 1),
            (
                4,
                monomials(&[&[0, 2, 0, 0], &[0, 1, 1, 0], &[0, 0, 2, 0]]),
                vec![1, 0, -3, 2],
                2,
                3,
            ),
        ];
        for (variables, generators, numerator, dimension, degree) in cases {
            let data = invariants(variables, &generators).unwrap();
            assert_eq!(
                data.numerator,
                numerator.into_iter().map(Integer::from).collect::<Vec<_>>()
            );
            assert_eq!(data.dimension, dimension);
            assert_eq!(data.degree, Integer::from(degree));
        }
    }

    #[test]
    fn complete_intersection_and_unit_ideal_are_hand_checked() {
        let complete_intersection = invariants(2, &monomials(&[&[2, 0], &[0, 3]])).unwrap();
        assert_eq!(complete_intersection.dimension, 0);
        assert_eq!(complete_intersection.degree, Integer::from(6));

        let unit = invariants(2, &monomials(&[&[0, 0]])).unwrap();
        assert_eq!(unit.dimension, -1);
        assert_eq!(unit.degree, Integer::zero());
        assert_eq!(unit.numerator, vec![Integer::zero()]);
    }

    #[test]
    fn independent_support_components_factor_the_numerator() {
        let generators = monomials(&[
            &[2, 0, 0, 0],
            &[1, 1, 0, 0],
            &[0, 2, 0, 0],
            &[0, 0, 2, 0],
            &[0, 0, 1, 1],
            &[0, 0, 0, 2],
        ]);
        let (arena, bounded) = intern_leading_monomials(4, &generators).unwrap();
        assert_eq!(support_components(&arena, &bounded).unwrap().len(), 2);

        let data = invariants(4, &generators).unwrap();
        assert_eq!(
            data.numerator,
            [1, 0, -6, 4, 9, -12, 4]
                .into_iter()
                .map(Integer::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(data.dimension, 0);
        assert_eq!(data.degree, Integer::from(9));
    }

    #[test]
    fn deep_colon_recursion_fails_at_the_fixed_bound() {
        assert!(invariants(2, &monomials(&[&[514, 0], &[513, 1]])).is_err());
    }
}
