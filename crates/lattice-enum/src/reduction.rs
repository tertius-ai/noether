// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact unimodular pre-reduction for enumeration bases.

use flint_core::{Integer, Rational};

/// Return `U G U^T` and the unimodular row transformation `U`.
///
/// Reduction is an implementation detail: stopping at the defensive operation
/// cap still leaves a valid unimodular change of basis and therefore cannot
/// affect the set enumerated later.
pub(crate) fn lll_reduce_gram(gram: &[Vec<Integer>]) -> (Vec<Vec<Integer>>, Vec<Vec<Integer>>) {
    let dimension = gram.len();
    let mut basis = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| {
                    if row == column {
                        Integer::one()
                    } else {
                        Integer::zero()
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if dimension < 2 {
        return (gram.to_vec(), basis);
    }

    let delta = Rational::new(Integer::from(3_u8), Integer::from(4_u8))
        .expect("three quarters is a rational");
    let mut reduced = transformed_gram(gram, &basis);
    let (mut mu, mut squared_lengths) = gram_schmidt(&reduced);
    let mut index = 1;
    let mut operations = 0_usize;
    const OPERATION_CAP: usize = 100_000;

    while index < dimension && operations < OPERATION_CAP {
        for previous in (0..index).rev() {
            let quotient = nearest_integer(&mu[index][previous]);
            if quotient.is_zero() {
                continue;
            }
            let (earlier, current_and_later) = basis.split_at_mut(index);
            for (current, previous) in current_and_later[0].iter_mut().zip(&earlier[previous]) {
                *current = &*current - &(&quotient * previous);
            }
            reduced = transformed_gram(gram, &basis);
            (mu, squared_lengths) = gram_schmidt(&reduced);
            operations += 1;
            if operations == OPERATION_CAP {
                break;
            }
        }
        if operations == OPERATION_CAP {
            break;
        }

        let correction = &delta - &(&mu[index][index - 1] * &mu[index][index - 1]);
        if squared_lengths[index] >= &correction * &squared_lengths[index - 1] {
            index += 1;
        } else {
            basis.swap(index, index - 1);
            reduced = transformed_gram(gram, &basis);
            (mu, squared_lengths) = gram_schmidt(&reduced);
            index = index.saturating_sub(1).max(1);
            operations += 1;
        }
    }
    (reduced, basis)
}

/// Map reduced coordinates `y` back to original coordinates `U^T y`.
pub(crate) fn original_coordinates(
    transformation: &[Vec<Integer>],
    reduced: &[i64],
) -> Option<Vec<i64>> {
    (0..reduced.len())
        .map(|coordinate| {
            transformation
                .iter()
                .zip(reduced)
                .fold(Integer::zero(), |sum, (row, coefficient)| {
                    &sum + &(&row[coordinate] * &Integer::from(*coefficient))
                })
                .to_i64()
        })
        .collect()
}

fn transformed_gram(gram: &[Vec<Integer>], basis: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    (0..basis.len())
        .map(|row| {
            (0..basis.len())
                .map(|column| {
                    (0..basis.len()).fold(Integer::zero(), |sum, left| {
                        (0..basis.len()).fold(sum, |sum, right| {
                            &sum + &(&(&basis[row][left] * &gram[left][right])
                                * &basis[column][right])
                        })
                    })
                })
                .collect()
        })
        .collect()
}

fn gram_schmidt(gram: &[Vec<Integer>]) -> (Vec<Vec<Rational>>, Vec<Rational>) {
    let dimension = gram.len();
    let mut mu = vec![vec![Rational::zero(); dimension]; dimension];
    let mut squared_lengths = vec![Rational::zero(); dimension];
    for row in 0..dimension {
        for column in 0..row {
            let numerator =
                (0..column).fold(Rational::from(gram[row][column].clone()), |value, inner| {
                    &value - &(&(&mu[row][inner] * &mu[column][inner]) * &squared_lengths[inner])
                });
            mu[row][column] = &numerator / &squared_lengths[column];
        }
        squared_lengths[row] =
            (0..row).fold(Rational::from(gram[row][row].clone()), |value, inner| {
                &value - &(&(&mu[row][inner] * &mu[row][inner]) * &squared_lengths[inner])
            });
        debug_assert!(squared_lengths[row].is_positive());
    }
    (mu, squared_lengths)
}

fn nearest_integer(value: &Rational) -> Integer {
    let denominator = value.denominator();
    let floor = value.numerator().div_floor(&denominator);
    let remainder = value.numerator() - &(&floor * &denominator);
    if &remainder * &Integer::from(2_u8) > denominator {
        floor + Integer::one()
    } else {
        floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_preserves_norms_under_the_reported_transform() {
        let gram = vec![
            vec![Integer::from(100_u8), Integer::from(99_u8)],
            vec![Integer::from(99_u8), Integer::from(100_u8)],
        ];
        let (reduced, transformation) = lll_reduce_gram(&gram);
        assert!(reduced[0][0] < gram[0][0]);
        let reduced_vector = vec![2_i64, -3_i64];
        let original = original_coordinates(&transformation, &reduced_vector).unwrap();
        let reduced_norm = crate::gso::exact_quadratic(&reduced, &reduced_vector).unwrap();
        let original_norm = crate::gso::exact_quadratic(&gram, &original).unwrap();
        assert_eq!(reduced_norm, original_norm);
    }
}
