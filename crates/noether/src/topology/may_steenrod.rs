// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Odd-primary May--Steenrod operations on normalized simplicial cochains.
//!
//! The universal chains use the McClure--Smith surjection-operad realization
//! of the May--Steenrod map constructed by Kaufmann and Medina-Mardones,
//! <https://doi.org/10.1515/forum-2020-0296>.  This module evaluates those
//! universal chains directly; it does not expose an operad as a second public
//! algebra system.

use std::collections::BTreeMap;

use crate::linalg::{LinearOperator, Matrix, Vector};
use crate::numthy::FiniteField;
use crate::{Integer, Ring};

use super::{Simplex, SimplicialComplex, SimplicialSet, TopologyError};

type Surjection = Vec<usize>;
type SurjectionCombination = BTreeMap<Surjection, u64>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct MayChainKey {
    removed_vertices: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MayChainTerm {
    coefficient: u64,
    removed_vertices: Vec<Vec<usize>>,
}

impl SimplicialComplex {
    /// The odd-primary Steenrod reduced power `P^power` of a checked
    /// normalized cocycle over a prime field.
    ///
    /// A degree-`q` input produces a degree-`q + 2 power (p-1)` cocycle.
    /// The implementation uses the effective May--Steenrod structure on
    /// normalized simplicial cochains.
    pub fn steenrod_reduced_power(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        self.odd_primary_operation(power, degree, cocycle, false)
    }

    /// The odd-primary operation `beta P^power` of a checked normalized
    /// cocycle over a prime field.
    ///
    /// A degree-`q` input produces a degree-`q + 2 power (p-1) + 1`
    /// cocycle.  Here `beta` is the mod-`p` Bockstein belonging to the
    /// standard Steenrod-algebra convention.
    pub fn bockstein_reduced_power(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        self.odd_primary_operation(power, degree, cocycle, true)
    }

    fn odd_primary_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
        bockstein: bool,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        let field = cocycle.ring().clone();
        let target_degree = operation_target_degree(&field, power, degree, bockstein)?;
        let upper = degree
            .checked_add(1)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        require_odd_prime_cocycle(
            cocycle,
            degree,
            self.count(degree),
            self.boundary_matrix(field.clone(), upper)?,
        )?;
        odd_primary_operation(
            power,
            degree,
            cocycle,
            bockstein,
            self.count(target_degree),
            |simplex, removed| {
                let value = self.simplices(target_degree).get(simplex)?;
                let face = Simplex::new(
                    value
                        .vertices()
                        .iter()
                        .enumerate()
                        .filter_map(|(position, vertex)| {
                            removed.binary_search(&position).is_err().then_some(*vertex)
                        })
                        .collect(),
                );
                self.position(&face)
            },
        )
    }
}

impl SimplicialSet {
    /// The odd-primary Steenrod reduced power `P^power` of a checked
    /// normalized cocycle over a prime field.
    pub fn steenrod_reduced_power(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        self.odd_primary_operation(power, degree, cocycle, false)
    }

    /// The odd-primary operation `beta P^power` of a checked normalized
    /// cocycle over a prime field.
    pub fn bockstein_reduced_power(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        self.odd_primary_operation(power, degree, cocycle, true)
    }

    fn odd_primary_operation(
        &self,
        power: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
        bockstein: bool,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        let field = cocycle.ring().clone();
        let target_degree = operation_target_degree(&field, power, degree, bockstein)?;
        let upper = degree
            .checked_add(1)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        require_odd_prime_cocycle(
            cocycle,
            degree,
            self.count(degree),
            self.boundary_matrix(field.clone(), upper)?,
        )?;
        odd_primary_operation(
            power,
            degree,
            cocycle,
            bockstein,
            self.count(target_degree),
            |simplex, removed| self.normalized_iterated_face(target_degree, simplex, removed),
        )
    }
}

fn operation_target_degree(
    field: &FiniteField,
    power: usize,
    degree: usize,
    bockstein: bool,
) -> Result<usize, TopologyError> {
    require_odd_prime_field(field)?;
    let prime = field
        .characteristic()
        .to_u64_checked()
        .ok_or_else(|| invalid("finite-field characteristic does not fit u64"))?;
    let prime_minus_one = usize::try_from(prime - 1)
        .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
    let shift = power
        .checked_mul(2)
        .and_then(|value| value.checked_mul(prime_minus_one))
        .and_then(|value| value.checked_add(usize::from(bockstein)))
        .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
    degree
        .checked_add(shift)
        .ok_or(TopologyError::DimensionTooLarge { dimension: degree })
}

fn odd_primary_operation<F>(
    power: usize,
    degree: usize,
    cocycle: &Vector<FiniteField>,
    bockstein: bool,
    target_count: usize,
    face_index: F,
) -> Result<Vector<FiniteField>, TopologyError>
where
    F: Fn(usize, &[usize]) -> Option<usize>,
{
    let field = cocycle.ring().clone();
    if power == 0 && !bockstein {
        return Ok(cocycle.clone());
    }
    let prime = field
        .characteristic()
        .to_u64_checked()
        .ok_or_else(|| invalid("finite-field characteristic does not fit u64"))?;
    let prime_usize =
        usize::try_from(prime).map_err(|_| invalid("May--Steenrod arity does not fit usize"))?;
    let twice_power = power
        .checked_mul(2)
        .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
    let Some(excess) = degree.checked_sub(twice_power) else {
        return Ok(Vector::zeros(field, target_count));
    };
    let may_degree = excess
        .checked_mul(prime_usize - 1)
        .and_then(|value| value.checked_sub(usize::from(bockstein)));
    let Some(may_degree) = may_degree else {
        return Ok(Vector::zeros(field, target_count));
    };
    if target_count == 0 {
        return Ok(Vector::zeros(field, 0));
    }
    let target_degree = degree
        .checked_add(
            twice_power
                .checked_mul(prime_usize - 1)
                .and_then(|value| value.checked_add(usize::from(bockstein)))
                .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?,
        )
        .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
    let universal = may_chain(prime, may_degree, degree, target_degree, power)?;
    let mut output = Vec::with_capacity(target_count);
    for simplex in 0..target_count {
        let mut value = field.zero();
        for term in &universal {
            let mut summand = field.element(term.coefficient);
            let mut normalized = true;
            for removed in &term.removed_vertices {
                let Some(face) = face_index(simplex, removed) else {
                    normalized = false;
                    break;
                };
                let coefficient = cocycle.get(face).map_err(|_| {
                    invalid("May--Steenrod iterated face escaped its checked cochain")
                })?;
                summand = field.mul(&summand, &coefficient)?;
            }
            if normalized {
                value = field.add(&value, &summand)?;
            }
        }
        output.push(value);
    }
    Vector::from_values(field, output).map_err(Into::into)
}

fn require_odd_prime_cocycle(
    cocycle: &Vector<FiniteField>,
    degree: usize,
    expected: usize,
    upper_boundary: Matrix<FiniteField>,
) -> Result<(), TopologyError> {
    require_odd_prime_field(cocycle.ring())?;
    if cocycle.dimension() != expected {
        return Err(invalid(format!(
            "degree-{degree} cocycle has {} coefficients; expected {expected}",
            cocycle.dimension()
        )));
    }
    let coboundary = upper_boundary.transpose().apply(cocycle)?;
    if coboundary
        .values()
        .any(|coefficient| !coefficient.is_zero())
    {
        return Err(invalid("May--Steenrod input is not a cocycle"));
    }
    Ok(())
}

fn require_odd_prime_field(field: &FiniteField) -> Result<(), TopologyError> {
    if field.degree() != 1 || field.characteristic() == &Integer::from(2_u8) {
        return Err(invalid(
            "odd-primary May--Steenrod operations require a prime field of odd characteristic",
        ));
    }
    Ok(())
}

fn may_chain(
    prime: u64,
    may_degree: usize,
    source_degree: usize,
    target_degree: usize,
    power: usize,
) -> Result<Vec<MayChainTerm>, TopologyError> {
    let arity =
        usize::try_from(prime).map_err(|_| invalid("May--Steenrod arity does not fit usize"))?;
    let representative = may_steenrod_map(arity, may_degree, prime)?;
    let prefactor = steenrod_prefactor(prime, power, source_degree);
    let mut terms = BTreeMap::<MayChainKey, u64>::new();
    for (surjection, coefficient) in representative {
        let coefficient = mul_mod(coefficient, prefactor, prime);
        let cut_count = surjection.len().saturating_sub(1);
        weakly_increasing_sequences(
            target_degree.saturating_add(1),
            cut_count,
            &mut Vec::with_capacity(cut_count),
            &mut |cuts| {
                add_may_chain_summand(
                    &mut terms,
                    &surjection,
                    coefficient,
                    cuts,
                    arity,
                    source_degree,
                    target_degree,
                    prime,
                );
            },
        );
    }
    Ok(terms
        .into_iter()
        .filter_map(|(key, coefficient)| {
            (coefficient != 0).then_some(MayChainTerm {
                coefficient,
                removed_vertices: key.removed_vertices,
            })
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn add_may_chain_summand(
    terms: &mut BTreeMap<MayChainKey, u64>,
    surjection: &[usize],
    coefficient: u64,
    interior_cuts: &[usize],
    arity: usize,
    source_degree: usize,
    target_degree: usize,
    prime: u64,
) {
    let mut cuts = Vec::with_capacity(interior_cuts.len() + 2);
    cuts.push(0);
    cuts.extend_from_slice(interior_cuts);
    cuts.push(target_degree);
    let interval_degrees = cuts
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let sign_negative = may_action_sign(surjection, &interval_degrees);
    let mut kept = vec![Vec::new(); arity];
    for (interval, &fiber) in cuts.windows(2).zip(surjection) {
        kept[fiber].extend(interval[0]..=interval[1]);
    }
    if kept.iter().any(|vertices| {
        vertices.len() != source_degree.saturating_add(1)
            || vertices.windows(2).any(|pair| pair[0] >= pair[1])
    }) {
        return;
    }
    let removed_vertices = kept
        .into_iter()
        .map(|vertices| {
            (0..=target_degree)
                .filter(|vertex| vertices.binary_search(vertex).is_err())
                .collect::<Vec<_>>()
        })
        .collect();
    let signed = if sign_negative {
        neg_mod(coefficient, prime)
    } else {
        coefficient
    };
    add_mod(terms, MayChainKey { removed_vertices }, signed, prime);
}

fn may_action_sign(surjection: &[usize], interval_degrees: &[usize]) -> bool {
    let mut exponent = 0_usize;
    for left in 0..surjection.len() {
        for right in left + 1..surjection.len() {
            if surjection[left] > surjection[right] {
                exponent = exponent
                    .wrapping_add((interval_degrees[left] % 2) * (interval_degrees[right] % 2));
            }
        }
    }
    let mut ordered = (0..surjection.len()).collect::<Vec<_>>();
    ordered.sort_by_key(|&index| surjection[index]);
    let ordered_degrees = ordered
        .iter()
        .map(|&index| interval_degrees[index] % 2)
        .collect::<Vec<_>>();
    for position in 0..ordered.len().saturating_sub(1) {
        if surjection[ordered[position]] == surjection[ordered[position + 1]] {
            exponent = exponent.wrapping_add(ordered_degrees[..=position].iter().sum::<usize>());
        }
    }
    exponent % 2 == 1
}

fn may_steenrod_map(
    arity: usize,
    degree: usize,
    prime: u64,
) -> Result<SurjectionCombination, TopologyError> {
    if arity == 0 || arity as u128 != u128::from(prime) {
        return Err(invalid(
            "May--Steenrod arity and coefficient prime disagree",
        ));
    }
    let mut current = BTreeMap::from([((0..arity).collect::<Vec<_>>(), 1)]);
    for current_degree in 1..=degree {
        let acted = cyclic_resolution_operator(&current, arity, current_degree % 2, prime);
        current = contraction_h(&acted, arity, prime);
    }
    Ok(current)
}

fn cyclic_resolution_operator(
    source: &SurjectionCombination,
    arity: usize,
    parity: usize,
    prime: u64,
) -> SurjectionCombination {
    let rotations = if parity == 0 {
        (0..arity).map(|rotation| (rotation, 1)).collect()
    } else {
        vec![(1 % arity, 1), (0, prime - 1)]
    };
    let mut output = BTreeMap::new();
    for (surjection, coefficient) in source {
        for &(rotation, operator_coefficient) in &rotations {
            let permutation = (0..arity)
                .map(|value| (value + rotation) % arity)
                .collect::<Vec<_>>();
            let negative = permutation_action_sign(&permutation, surjection, arity);
            let mut term_coefficient = mul_mod(*coefficient, operator_coefficient, prime);
            if negative {
                term_coefficient = neg_mod(term_coefficient, prime);
            }
            let term = surjection
                .iter()
                .map(|&value| permutation[value])
                .collect::<Vec<_>>();
            add_surjection(&mut output, term, term_coefficient, arity, prime);
        }
    }
    output
}

fn permutation_action_sign(permutation: &[usize], surjection: &[usize], arity: usize) -> bool {
    let mut weights = vec![0_usize; arity];
    for &value in surjection {
        weights[value] += 1;
    }
    for weight in &mut weights {
        *weight = weight.saturating_sub(1);
    }
    let mut exponent = 0_usize;
    for left in 0..arity {
        for right in left + 1..arity {
            if permutation[left] > permutation[right] {
                exponent = exponent.wrapping_add((weights[left] % 2) * (weights[right] % 2));
            }
        }
    }
    exponent % 2 == 1
}

fn contraction_h(
    source: &SurjectionCombination,
    arity: usize,
    prime: u64,
) -> SurjectionCombination {
    let mut output = BTreeMap::new();
    for (surjection, coefficient) in source {
        add_surjection(
            &mut output,
            prepend_without_shift(surjection),
            *coefficient,
            arity,
            prime,
        );
        for projection_count in 1..arity.saturating_sub(1) {
            let Some(mut projected) = project_first_values(surjection, projection_count) else {
                continue;
            };
            projected = prepend_without_shift(&projected);
            for _ in 0..projection_count {
                projected = prepend_and_shift(&projected);
            }
            add_surjection(&mut output, projected, *coefficient, arity, prime);
        }
    }
    output
}

fn prepend_without_shift(surjection: &[usize]) -> Surjection {
    std::iter::once(0)
        .chain(surjection.iter().copied())
        .collect()
}

fn prepend_and_shift(surjection: &[usize]) -> Surjection {
    std::iter::once(0)
        .chain(surjection.iter().map(|value| value + 1))
        .collect()
}

fn project_first_values(surjection: &[usize], count: usize) -> Option<Surjection> {
    let mut projected = surjection.to_vec();
    for _ in 0..count {
        if projected.iter().filter(|&&value| value == 0).count() != 1 {
            return None;
        }
        projected.retain(|&value| value != 0);
        for value in &mut projected {
            *value -= 1;
        }
    }
    Some(projected)
}

fn add_surjection(
    output: &mut SurjectionCombination,
    surjection: Surjection,
    coefficient: u64,
    arity: usize,
    prime: u64,
) {
    if coefficient == 0
        || surjection.windows(2).any(|pair| pair[0] == pair[1])
        || (0..arity).any(|value| !surjection.contains(&value))
    {
        return;
    }
    add_mod(output, surjection, coefficient, prime);
}

fn steenrod_prefactor(prime: u64, power: usize, degree: usize) -> u64 {
    let negative = (degree.div_ceil(2) + power) % 2 == 1;
    let mut coefficient = if negative { prime - 1 } else { 1 };
    if degree % 2 == 1 {
        for factor in 2..=(prime - 1) / 2 {
            coefficient = mul_mod(coefficient, factor, prime);
        }
    }
    coefficient
}

fn weakly_increasing_sequences<F>(
    universe: usize,
    length: usize,
    current: &mut Vec<usize>,
    visit: &mut F,
) where
    F: FnMut(&[usize]),
{
    if current.len() == length {
        visit(current);
        return;
    }
    let start = current.last().copied().unwrap_or(0);
    for value in start..universe {
        current.push(value);
        weakly_increasing_sequences(universe, length, current, visit);
        current.pop();
    }
}

fn add_mod<K: Ord>(output: &mut BTreeMap<K, u64>, key: K, value: u64, prime: u64) {
    if value == 0 {
        return;
    }
    let previous = output.remove(&key).unwrap_or(0);
    let sum = ((u128::from(previous) + u128::from(value)) % u128::from(prime)) as u64;
    if sum != 0 {
        output.insert(key, sum);
    }
}

fn mul_mod(left: u64, right: u64, prime: u64) -> u64 {
    ((u128::from(left) * u128::from(right)) % u128::from(prime)) as u64
}

fn neg_mod(value: u64, prime: u64) -> u64 {
    if value == 0 {
        0
    } else {
        prime - value
    }
}

fn invalid(reason: impl Into<String>) -> TopologyError {
    TopologyError::InvalidCochain {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::SimplicialFace;

    fn basis(field: &FiniteField, dimension: usize, selected: usize) -> Vector<FiniteField> {
        Vector::from_values(
            field.clone(),
            (0..dimension).map(|index| {
                if index == selected {
                    field.one()
                } else {
                    field.zero()
                }
            }),
        )
        .unwrap()
    }

    fn mod_three_moore_space() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![]],
            vec![
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
            ],
            vec![
                vec![
                    SimplicialFace::Cell(0),
                    SimplicialFace::Cell(1),
                    SimplicialFace::Cell(0),
                ],
                vec![
                    SimplicialFace::Cell(0),
                    SimplicialFace::degenerate(0, 0, vec![0, 0]),
                    SimplicialFace::Cell(1),
                ],
            ],
        ])
        .unwrap()
    }

    #[test]
    fn beta_p_zero_has_the_expected_universal_degree_one_chain() {
        let chain = may_chain(3, 1, 1, 2, 0).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].coefficient, 2);
        assert_eq!(chain[0].removed_vertices, vec![vec![1], vec![2], vec![0]]);
    }

    #[test]
    fn intermediate_reduced_power_matches_a_hand_checked_universal_term() {
        let chain = may_chain(3, 2, 3, 7, 1).unwrap();
        assert_eq!(chain.len(), 19);
        let expected = vec![vec![1, 2, 6, 7], vec![1, 2, 3, 4], vec![4, 5, 6, 7]];
        assert_eq!(
            chain
                .iter()
                .find(|term| term.removed_vertices == expected)
                .map(|term| term.coefficient),
            Some(2)
        );
    }

    #[test]
    fn bockstein_universal_chains_match_odd_prime_reference_values() {
        let degree_two = may_chain(3, 3, 2, 3, 0).unwrap();
        assert_eq!(
            degree_two
                .iter()
                .map(|term| (term.removed_vertices.clone(), term.coefficient))
                .collect::<BTreeMap<_, _>>(),
            BTreeMap::from([
                (vec![vec![3], vec![3], vec![0]], 1),
                (vec![vec![1], vec![2], vec![0]], 1),
                (vec![vec![3], vec![0], vec![0]], 2),
                (vec![vec![3], vec![1], vec![2]], 2),
            ])
        );

        let at_five = may_chain(5, 3, 1, 2, 0).unwrap();
        assert_eq!(at_five.len(), 3);
        assert!(at_five.iter().all(|term| term.coefficient == 3));
        assert_eq!(
            at_five
                .iter()
                .map(|term| term.removed_vertices.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![vec![1], vec![1], vec![1], vec![2], vec![0]],
                vec![vec![1], vec![2], vec![0], vec![0], vec![0]],
                vec![vec![2], vec![2], vec![0], vec![1], vec![2]],
            ]
        );
    }

    #[test]
    fn beta_p_zero_detects_mod_three_torsion() {
        let space = mod_three_moore_space();
        let field = FiniteField::prime(3_u8).unwrap();
        let cocycle =
            Vector::from_values(field.clone(), [field.element(1_u8), field.element(2_u8)]).unwrap();
        let image = space.bockstein_reduced_power(0, 1, &cocycle).unwrap();
        assert!(image.values().any(|coefficient| !coefficient.is_zero()));
        let coboundary = space
            .boundary_matrix(field, 3)
            .unwrap()
            .transpose()
            .apply(&image)
            .unwrap();
        assert!(coboundary.values().all(|coefficient| coefficient.is_zero()));
    }

    #[test]
    fn p_zero_is_identity_and_instability_is_zero() {
        let space = mod_three_moore_space();
        let field = FiniteField::prime(3_u8).unwrap();
        let cocycle =
            Vector::from_values(field.clone(), [field.element(1_u8), field.element(2_u8)]).unwrap();
        assert_eq!(
            space.steenrod_reduced_power(0, 1, &cocycle).unwrap(),
            cocycle
        );
        assert_eq!(
            space.steenrod_reduced_power(1, 1, &cocycle).unwrap(),
            Vector::zeros(field, 0)
        );
    }

    #[test]
    fn intermediate_reduced_power_takes_cocycles_to_cocycles() {
        let simplex = SimplicialComplex::from_facets(&[(0..=8).collect::<Vec<_>>()]);
        let field = FiniteField::prime(3_u8).unwrap();
        let seed = basis(&field, simplex.count(2), 0);
        let cocycle = simplex
            .boundary_matrix(field.clone(), 3)
            .unwrap()
            .transpose()
            .apply(&seed)
            .unwrap();
        assert!(cocycle.values().any(|coefficient| !coefficient.is_zero()));
        let image = simplex.steenrod_reduced_power(1, 3, &cocycle).unwrap();
        let coboundary = simplex
            .boundary_matrix(field, 8)
            .unwrap()
            .transpose()
            .apply(&image)
            .unwrap();
        assert!(coboundary.values().all(|coefficient| coefficient.is_zero()));
    }

    #[test]
    fn odd_operations_reject_characteristic_two_and_extension_fields() {
        let interval = SimplicialComplex::from_facets(&[vec![0, 1]]);
        let binary = FiniteField::prime(2_u8).unwrap();
        let vertex = basis(&binary, interval.count(0), 0);
        assert!(matches!(
            interval.steenrod_reduced_power(0, 0, &vertex),
            Err(TopologyError::InvalidCochain { .. })
        ));
        let extension = FiniteField::extension(3, 2).unwrap();
        let vertex = basis(&extension, interval.count(0), 0);
        assert!(matches!(
            interval.steenrod_reduced_power(0, 0, &vertex),
            Err(TopologyError::InvalidCochain { .. })
        ));
    }
}
