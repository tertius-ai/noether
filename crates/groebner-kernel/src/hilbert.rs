// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact standard-graded Hilbert numerators for completion guidance.

use std::collections::BTreeMap;

use flint_core::Integer;
use poly_core::{Monomial, MonomialArena, MonomialId, MonomialOrder};

use crate::GroebnerError;

const MAX_HILBERT_RECURSION_DEPTH: usize = 512;
const MAX_HILBERT_STATES: usize = 100_000;
const MAX_HILBERT_DEGREE: usize = 4_096;

type HilbertState = Vec<MonomialId>;

pub(crate) fn standard_hilbert_numerator(
    arity: usize,
    generators: &[Monomial],
) -> Result<Vec<Integer>, GroebnerError> {
    let mut arena = MonomialArena::with_capacity(
        arity,
        MonomialOrder::GradedReverseLexicographic,
        generators.len(),
    );
    let generators = generators
        .iter()
        .map(|generator| {
            if generator.arity() != arity {
                return Err(GroebnerError::ContextMismatch);
            }
            if generator
                .total_degree()
                .is_none_or(|degree| degree > MAX_HILBERT_DEGREE as u64)
            {
                return Err(resource("Hilbert monomial degree", MAX_HILBERT_DEGREE));
            }
            arena
                .intern(generator.clone())
                .map_err(|_| GroebnerError::ExponentOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    HilbertComputation {
        arena,
        memo: BTreeMap::new(),
        states: 0,
    }
    .numerator(generators, 0)
}

pub(crate) fn standard_module_quotient_hilbert_numerator(
    arity: usize,
    rank: usize,
    leading_terms: &[(Monomial, usize)],
    generator_degrees: &[usize],
) -> Result<Vec<Integer>, GroebnerError> {
    if generator_degrees.len() != rank
        || leading_terms
            .iter()
            .any(|(monomial, component)| monomial.arity() != arity || *component >= rank)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    let mut by_component = vec![Vec::new(); rank];
    for (monomial, component) in leading_terms {
        by_component[*component].push(monomial.clone());
    }
    let mut numerator = vec![Integer::zero()];
    for (component, generators) in by_component.iter().enumerate() {
        numerator = polynomial_add_shifted(
            numerator,
            &standard_hilbert_numerator(arity, generators)?,
            generator_degrees[component],
        )?;
    }
    Ok(trim(numerator))
}

pub(crate) fn standard_module_submodule_hilbert_numerator(
    arity: usize,
    ambient_generator_degrees: &[usize],
    leading_terms: &[(Monomial, usize)],
) -> Result<Vec<Integer>, GroebnerError> {
    let quotient = standard_module_quotient_hilbert_numerator(
        arity,
        ambient_generator_degrees.len(),
        leading_terms,
        ambient_generator_degrees,
    )?;
    let mut numerator = vec![Integer::zero()];
    for degree in ambient_generator_degrees {
        numerator = polynomial_add_shifted(numerator, &[Integer::one()], *degree)?;
    }
    let negated_quotient = quotient
        .into_iter()
        .map(|coefficient| -coefficient)
        .collect::<Vec<_>>();
    polynomial_add_shifted(numerator, &negated_quotient, 0)
}

struct HilbertComputation {
    arena: MonomialArena,
    memo: BTreeMap<HilbertState, Vec<Integer>>,
    states: usize,
}

impl HilbertComputation {
    fn numerator(
        &mut self,
        generators: HilbertState,
        depth: usize,
    ) -> Result<Vec<Integer>, GroebnerError> {
        if depth > MAX_HILBERT_RECURSION_DEPTH {
            return Err(resource(
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
            return Err(resource("Hilbert recursion states", MAX_HILBERT_STATES));
        }

        let result = if generators.is_empty() {
            vec![Integer::one()]
        } else if generators
            .iter()
            .any(|generator| self.arena.total_degree(*generator) == 0)
        {
            vec![Integer::zero()]
        } else if generators.len() == 1 {
            one_minus_t_power(bounded_degree(&self.arena, generators[0])?)
        } else if let Some(components) = support_components(&self.arena, &generators) {
            let mut product = vec![Integer::one()];
            for component in components {
                product = polynomial_product(&product, &self.numerator(component, depth + 1)?)?;
            }
            trim(product)
        } else {
            let (_, plus, colon) = choose_pivot(&mut self.arena, &generators)?;
            let plus = self.numerator(plus, depth + 1)?;
            let colon = self.numerator(colon, depth + 1)?;
            polynomial_add_shifted(plus, &colon, 1)?
        };
        self.memo.insert(generators, result.clone());
        Ok(result)
    }
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
        return Err(GroebnerError::VerificationFailed);
    };
    Ok((pivot, plus, colon))
}

fn child_states(
    arena: &mut MonomialArena,
    generators: &[MonomialId],
    pivot: usize,
) -> Result<(HilbertState, HilbertState), GroebnerError> {
    let mut variable = vec![0; arena.arity()];
    variable[pivot] = 1;
    let variable = arena
        .intern_exponents(&variable)
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut plus = generators.to_vec();
    plus.push(variable);
    let mut colon = Vec::with_capacity(generators.len());
    for generator in generators {
        let mut quotient = arena.get(*generator).to_u32_vec();
        quotient[pivot] = quotient[pivot].saturating_sub(1);
        colon.push(
            arena
                .intern_exponents(&quotient)
                .map_err(|_| GroebnerError::ExponentOverflow)?,
        );
    }
    Ok((minimalize(arena, plus), minimalize(arena, colon)))
}

fn minimalize(arena: &MonomialArena, mut generators: HilbertState) -> HilbertState {
    generators.sort_by_key(|generator| (arena.total_degree(*generator), *generator));
    generators.dedup();
    let mut antichain = Vec::with_capacity(generators.len());
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
    let mut parents = (0..arena.arity()).collect::<Vec<_>>();
    fn find(parents: &mut [usize], mut value: usize) -> usize {
        while parents[value] != value {
            parents[value] = parents[parents[value]];
            value = parents[value];
        }
        value
    }
    for generator in generators {
        let support = (0..arena.arity())
            .filter(|variable| arena.exponent(*generator, *variable) != 0)
            .collect::<Vec<_>>();
        if let Some((&first, rest)) = support.split_first() {
            for &variable in rest {
                let first = find(&mut parents, first);
                let variable = find(&mut parents, variable);
                if first != variable {
                    parents[variable] = first;
                }
            }
        }
    }
    let mut components = BTreeMap::<usize, HilbertState>::new();
    for generator in generators {
        let variable = (0..arena.arity())
            .find(|variable| arena.exponent(*generator, *variable) != 0)
            .expect("unit generators are handled before support factorization");
        let root = find(&mut parents, variable);
        components.entry(root).or_default().push(*generator);
    }
    (components.len() > 1).then(|| components.into_values().collect())
}

fn bounded_degree(arena: &MonomialArena, generator: MonomialId) -> Result<usize, GroebnerError> {
    let degree = arena.total_degree(generator);
    if degree > MAX_HILBERT_DEGREE as u64 {
        return Err(resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE));
    }
    usize::try_from(degree).map_err(|_| resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE))
}

fn one_minus_t_power(degree: usize) -> Vec<Integer> {
    let mut result = vec![Integer::zero(); degree + 1];
    result[0] = Integer::one();
    result[degree] = &result[degree] - &Integer::one();
    result
}

fn polynomial_product(left: &[Integer], right: &[Integer]) -> Result<Vec<Integer>, GroebnerError> {
    let length = left
        .len()
        .checked_add(right.len())
        .and_then(|length| length.checked_sub(1))
        .ok_or_else(|| resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE))?;
    if length > MAX_HILBERT_DEGREE + 1 {
        return Err(resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE));
    }
    let mut result = vec![Integer::zero(); length];
    for (left_index, left_coefficient) in left.iter().enumerate() {
        for (right_index, right_coefficient) in right.iter().enumerate() {
            if left_coefficient.is_zero() || right_coefficient.is_zero() {
                continue;
            }
            let index = left_index + right_index;
            result[index] = &result[index] + &(left_coefficient * right_coefficient);
        }
    }
    Ok(trim(result))
}

fn polynomial_add_shifted(
    mut target: Vec<Integer>,
    source: &[Integer],
    shift: usize,
) -> Result<Vec<Integer>, GroebnerError> {
    let required = source
        .len()
        .checked_add(shift)
        .ok_or_else(|| resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE))?;
    if required > MAX_HILBERT_DEGREE + 1 {
        return Err(resource("Hilbert polynomial degree", MAX_HILBERT_DEGREE));
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

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_numerators_are_exact_and_factor_components() {
        let generators = [
            Monomial::new(vec![2, 0, 0, 0].into_boxed_slice()),
            Monomial::new(vec![1, 1, 0, 0].into_boxed_slice()),
            Monomial::new(vec![0, 2, 0, 0].into_boxed_slice()),
            Monomial::new(vec![0, 0, 2, 0].into_boxed_slice()),
            Monomial::new(vec![0, 0, 1, 1].into_boxed_slice()),
            Monomial::new(vec![0, 0, 0, 2].into_boxed_slice()),
        ];
        assert_eq!(
            standard_hilbert_numerator(4, &generators).unwrap(),
            [1, 0, -6, 4, 9, -12, 4]
                .into_iter()
                .map(Integer::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn shifted_module_quotient_numerators_sum_components() {
        let leading = [
            (Monomial::new(vec![1, 0].into_boxed_slice()), 0),
            (Monomial::new(vec![0, 2].into_boxed_slice()), 1),
        ];
        assert_eq!(
            standard_module_quotient_hilbert_numerator(2, 2, &leading, &[2, 3]).unwrap(),
            vec![
                Integer::zero(),
                Integer::zero(),
                Integer::one(),
                Integer::zero(),
                Integer::zero(),
                -Integer::one(),
            ]
        );
        assert_eq!(
            standard_module_submodule_hilbert_numerator(2, &[2, 3], &leading).unwrap(),
            vec![
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::one(),
                Integer::zero(),
                Integer::one(),
            ]
        );
    }
}
