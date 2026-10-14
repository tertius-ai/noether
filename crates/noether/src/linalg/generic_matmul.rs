// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generic dense matrix multiplication over the public [`Ring`] contract.

use crate::Ring;

use super::LinalgError;

/// Cache tile for the generic classical floor. Coefficients may be much larger
/// than machine words, so this is intentionally conservative.
const CLASSICAL_BLOCK: usize = 32;

/// Initial measured crossover for the allocation-based Strassen path. Keep
/// this local: backend matrices use their own tuned algorithms and never reach
/// the generic route.
pub(super) const STRASSEN_CROSSOVER: usize = 32;

pub(super) fn matmul<R: Ring>(
    ring: &R,
    left: &[R::Element],
    right: &[R::Element],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<R::Element>, LinalgError> {
    debug_assert_eq!(left.len(), rows * inner);
    debug_assert_eq!(right.len(), inner * cols);
    if rows == inner && inner == cols && rows.is_power_of_two() && rows >= STRASSEN_CROSSOVER {
        strassen_square(ring, left, right, rows, STRASSEN_CROSSOVER)
    } else {
        classical_blocked(ring, left, right, rows, inner, cols)
    }
}

pub(super) fn classical_blocked<R: Ring>(
    ring: &R,
    left: &[R::Element],
    right: &[R::Element],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<R::Element>, LinalgError> {
    let mut output = vec![ring.zero(); rows * cols];
    for row_block in (0..rows).step_by(CLASSICAL_BLOCK) {
        let row_end = (row_block + CLASSICAL_BLOCK).min(rows);
        for inner_block in (0..inner).step_by(CLASSICAL_BLOCK) {
            let inner_end = (inner_block + CLASSICAL_BLOCK).min(inner);
            for col_block in (0..cols).step_by(CLASSICAL_BLOCK) {
                let col_end = (col_block + CLASSICAL_BLOCK).min(cols);
                for row in row_block..row_end {
                    for index in inner_block..inner_end {
                        let left_value = &left[row * inner + index];
                        if ring.is_zero(left_value) {
                            continue;
                        }
                        for col in col_block..col_end {
                            let right_value = &right[index * cols + col];
                            if ring.is_zero(right_value) {
                                continue;
                            }
                            let product = ring.mul(left_value, right_value)?;
                            let position = row * cols + col;
                            output[position] = ring.add(&output[position], &product)?;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(super) fn strassen_square<R: Ring>(
    ring: &R,
    left: &[R::Element],
    right: &[R::Element],
    size: usize,
    crossover: usize,
) -> Result<Vec<R::Element>, LinalgError> {
    if size <= crossover || !size.is_multiple_of(2) {
        return classical_blocked(ring, left, right, size, size, size);
    }

    let half = size / 2;
    let [a11, a12, a21, a22] = split_quadrants(left, size);
    let [b11, b12, b21, b22] = split_quadrants(right, size);

    let m1 = strassen_square(
        ring,
        &elementwise_add(ring, &a11, &a22)?,
        &elementwise_add(ring, &b11, &b22)?,
        half,
        crossover,
    )?;
    let m2 = strassen_square(
        ring,
        &elementwise_add(ring, &a21, &a22)?,
        &b11,
        half,
        crossover,
    )?;
    let m3 = strassen_square(
        ring,
        &a11,
        &elementwise_sub(ring, &b12, &b22)?,
        half,
        crossover,
    )?;
    let m4 = strassen_square(
        ring,
        &a22,
        &elementwise_sub(ring, &b21, &b11)?,
        half,
        crossover,
    )?;
    let m5 = strassen_square(
        ring,
        &elementwise_add(ring, &a11, &a12)?,
        &b22,
        half,
        crossover,
    )?;
    let m6 = strassen_square(
        ring,
        &elementwise_sub(ring, &a21, &a11)?,
        &elementwise_add(ring, &b11, &b12)?,
        half,
        crossover,
    )?;
    let m7 = strassen_square(
        ring,
        &elementwise_sub(ring, &a12, &a22)?,
        &elementwise_add(ring, &b21, &b22)?,
        half,
        crossover,
    )?;

    let c11 = elementwise_add(
        ring,
        &elementwise_sub(ring, &elementwise_add(ring, &m1, &m4)?, &m5)?,
        &m7,
    )?;
    let c12 = elementwise_add(ring, &m3, &m5)?;
    let c21 = elementwise_add(ring, &m2, &m4)?;
    let c22 = elementwise_add(
        ring,
        &elementwise_add(ring, &elementwise_sub(ring, &m1, &m2)?, &m3)?,
        &m6,
    )?;
    Ok(join_quadrants([&c11, &c12, &c21, &c22], half))
}

fn split_quadrants<E: Clone>(matrix: &[E], size: usize) -> [Vec<E>; 4] {
    let half = size / 2;
    let mut quadrants = std::array::from_fn(|_| Vec::with_capacity(half * half));
    for row in 0..half {
        quadrants[0].extend_from_slice(&matrix[row * size..row * size + half]);
        quadrants[1].extend_from_slice(&matrix[row * size + half..(row + 1) * size]);
        let lower = row + half;
        quadrants[2].extend_from_slice(&matrix[lower * size..lower * size + half]);
        quadrants[3].extend_from_slice(&matrix[lower * size + half..(lower + 1) * size]);
    }
    quadrants
}

fn join_quadrants<E: Clone>(quadrants: [&[E]; 4], half: usize) -> Vec<E> {
    let size = half * 2;
    let mut output = Vec::with_capacity(size * size);
    for row in 0..half {
        output.extend_from_slice(&quadrants[0][row * half..(row + 1) * half]);
        output.extend_from_slice(&quadrants[1][row * half..(row + 1) * half]);
    }
    for row in 0..half {
        output.extend_from_slice(&quadrants[2][row * half..(row + 1) * half]);
        output.extend_from_slice(&quadrants[3][row * half..(row + 1) * half]);
    }
    output
}

fn elementwise_add<R: Ring>(
    ring: &R,
    left: &[R::Element],
    right: &[R::Element],
) -> Result<Vec<R::Element>, LinalgError> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| ring.add(left, right).map_err(Into::into))
        .collect()
}

fn elementwise_sub<R: Ring>(
    ring: &R,
    left: &[R::Element],
    right: &[R::Element],
) -> Result<Vec<R::Element>, LinalgError> {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| ring.sub(left, right).map_err(Into::into))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Integer, Z};

    #[test]
    fn strassen_matches_the_classical_floor_through_three_levels() {
        for size in [2, 4, 8] {
            let left = (0..size * size)
                .map(|index| Integer::from((index as i64 * 7 + 3) % 19 - 9))
                .collect::<Vec<_>>();
            let right = (0..size * size)
                .map(|index| Integer::from((index as i64 * 11 + 5) % 23 - 11))
                .collect::<Vec<_>>();
            let expected = classical_blocked(&Z, &left, &right, size, size, size).unwrap();
            let actual = strassen_square(&Z, &left, &right, size, 1).unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rectangular_and_non_power_of_two_inputs_keep_the_blocked_floor() {
        let left = (0..15).map(Integer::from).collect::<Vec<_>>();
        let right = (0..20).map(Integer::from).collect::<Vec<_>>();
        assert_eq!(
            matmul(&Z, &left, &right, 3, 5, 4).unwrap(),
            classical_blocked(&Z, &left, &right, 3, 5, 4).unwrap()
        );
    }
}
