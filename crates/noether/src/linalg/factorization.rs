// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact matrix factorizations over field contexts.

use core::fmt;

use crate::{Field, NoetherError, Ring, SemanticType, TypeDescription, VerificationError};

use super::{LinalgError, Matrix};

static PLUQ_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/PLUQFactorization@1",
    rust_path: "noether::linalg::PLUQFactorization",
    python_path: Some("noether.linalg.PLUQFactorization"),
};

/// A checked, rank-revealing `P * L * U * Q = A` factorization.
///
/// For an `m x n` source and `k = min(m, n)`, `P` and `Q` are `m x m` and
/// `n x n` permutation matrices, `L` is `m x k` unit lower trapezoidal, and
/// `U` is `k x n` upper trapezoidal. The first `rank` diagonal entries of `U`
/// are nonzero and every remaining row of `U` is zero.
#[derive(Clone, PartialEq, Eq)]
pub struct PLUQFactorization<R: Ring> {
    source: Matrix<R>,
    p: Matrix<R>,
    l: Matrix<R>,
    u: Matrix<R>,
    q: Matrix<R>,
    rank: usize,
}

impl<R: Ring> PLUQFactorization<R> {
    /// The coefficient field.
    pub fn ring(&self) -> &R {
        self.source.ring()
    }

    /// The matrix being factored.
    pub fn source(&self) -> &Matrix<R> {
        &self.source
    }

    /// The left permutation matrix.
    pub fn p(&self) -> &Matrix<R> {
        &self.p
    }

    /// The unit lower-trapezoidal factor.
    pub fn l(&self) -> &Matrix<R> {
        &self.l
    }

    /// The upper-trapezoidal factor.
    pub fn u(&self) -> &Matrix<R> {
        &self.u
    }

    /// The right permutation matrix.
    pub fn q(&self) -> &Matrix<R> {
        &self.q
    }

    /// The exact rank of the source matrix.
    pub fn rank(&self) -> usize {
        self.rank
    }
}

impl<R: Field> PLUQFactorization<R> {
    /// Recheck shapes, factor structure, rank, and `P * L * U * Q = source`.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let (rows, columns) = self.source.shape();
        let inner = rows.min(columns);
        if self.p.ring() != self.ring()
            || self.l.ring() != self.ring()
            || self.u.ring() != self.ring()
            || self.q.ring() != self.ring()
        {
            return invalid("PLUQ factors share the source field context");
        }
        if self.p.shape() != (rows, rows)
            || self.l.shape() != (rows, inner)
            || self.u.shape() != (inner, columns)
            || self.q.shape() != (columns, columns)
        {
            return invalid("PLUQ factors have the required rectangular shapes");
        }
        if !is_permutation_matrix(&self.p) || !is_permutation_matrix(&self.q) {
            return invalid("P and Q are permutation matrices");
        }
        for row in 0..rows {
            for column in 0..inner {
                let value = self
                    .l
                    .get(row, column)
                    .expect("checked PLUQ lower-factor shape");
                if column > row && value != self.ring().zero() {
                    return invalid("L is lower trapezoidal");
                }
                if column == row && value != self.ring().one() {
                    return invalid("L has a unit diagonal");
                }
            }
        }
        if self.rank > inner {
            return invalid("PLUQ rank does not exceed min(rows, columns)");
        }
        for row in 0..inner {
            for column in 0..columns {
                let value = self
                    .u
                    .get(row, column)
                    .expect("checked PLUQ upper-factor shape");
                if column < row && value != self.ring().zero() {
                    return invalid("U is upper trapezoidal");
                }
                if row >= self.rank && value != self.ring().zero() {
                    return invalid("the rows of U after the reported rank are zero");
                }
            }
            if row < self.rank
                && self.u.get(row, row).expect("diagonal lies inside U") == self.ring().zero()
            {
                return invalid("the pivot diagonal of U is nonzero");
            }
        }
        let reconstructed = self
            .p
            .matmul(&self.l)
            .and_then(|value| value.matmul(&self.u))
            .and_then(|value| value.matmul(&self.q))
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "PLUQ factors can be multiplied",
            })?;
        if reconstructed != self.source {
            return invalid("P * L * U * Q equals the source matrix");
        }
        if self
            .source
            .rank()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "the PLUQ source rank is independently computable",
            })?
            != self.rank
        {
            return invalid("the reported PLUQ rank equals the source rank");
        }
        Ok(())
    }
}

impl<R: Ring> fmt::Debug for PLUQFactorization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PLUQFactorization")
            .field("context", &R::TYPE_ID)
            .field("shape", &self.source.shape())
            .field("rank", &self.rank)
            .finish()
    }
}

impl<R: Ring> fmt::Display for PLUQFactorization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PLUQFactorization<{}>(shape={}x{}, rank={})",
            R::TYPE_ID,
            self.source.shape().0,
            self.source.shape().1,
            self.rank
        )
    }
}

impl<R: Ring> SemanticType for PLUQFactorization<R> {
    const TYPE_ID: &'static str = "noether/linalg/PLUQFactorization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        for component in [&self.source, &self.p, &self.l, &self.u, &self.q] {
            push_frame(&mut output, &component.canonical_bytes()?)?;
        }
        output.extend_from_slice(&(self.rank as u64).to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PLUQ_DESCRIPTION
    }
}

impl<R: Field> Matrix<R> {
    /// Compute a deterministic, rank-revealing PLUQ factorization.
    ///
    /// The generic field algorithm is the correctness floor. Specialized
    /// storage backends may replace it without changing this result type or
    /// reconstruction convention.
    pub fn pluq(&self) -> Result<PLUQFactorization<R>, LinalgError> {
        if let Some((p, l, u, q, rank)) = self.field_pluq_backend() {
            let result = PLUQFactorization {
                source: self.clone(),
                p,
                l,
                u,
                q,
                rank,
            };
            result.verify()?;
            return Ok(result);
        }
        let (rows, columns) = self.shape();
        let inner = rows.min(columns);
        let mut values = self.rows().collect::<Vec<_>>();
        let mut lower = vec![self.ring().zero(); rows * inner];
        for diagonal in 0..inner {
            lower[diagonal * inner + diagonal] = self.ring().one();
        }
        let mut row_order = (0..rows).collect::<Vec<_>>();
        let mut column_order = (0..columns).collect::<Vec<_>>();
        let mut rank = 0;

        for pivot in 0..inner {
            let selected = (pivot..rows).find_map(|row| {
                (pivot..columns)
                    .find(|column| values[row][*column] != self.ring().zero())
                    .map(|column| (row, column))
            });
            let Some((pivot_row, pivot_column)) = selected else {
                break;
            };

            if pivot_row != pivot {
                values.swap(pivot, pivot_row);
                row_order.swap(pivot, pivot_row);
                for column in 0..pivot {
                    lower.swap(pivot * inner + column, pivot_row * inner + column);
                }
            }
            if pivot_column != pivot {
                for row in &mut values {
                    row.swap(pivot, pivot_column);
                }
                column_order.swap(pivot, pivot_column);
            }

            let inverse = self.ring().inverse(&values[pivot][pivot])?;
            let pivot_values = values[pivot].clone();
            for row in pivot + 1..rows {
                if values[row][pivot] == self.ring().zero() {
                    continue;
                }
                let factor = self.ring().mul(&values[row][pivot], &inverse)?;
                lower[row * inner + pivot] = factor.clone();
                values[row][pivot] = self.ring().zero();
                for column in pivot + 1..columns {
                    let product = self.ring().mul(&factor, &pivot_values[column])?;
                    values[row][column] = self.ring().sub(&values[row][column], &product)?;
                }
            }
            rank += 1;
        }

        let mut p_entries = vec![self.ring().zero(); rows * rows];
        for (current, original) in row_order.into_iter().enumerate() {
            p_entries[original * rows + current] = self.ring().one();
        }
        let mut q_entries = vec![self.ring().zero(); columns * columns];
        for (current, original) in column_order.into_iter().enumerate() {
            q_entries[current * columns + original] = self.ring().one();
        }
        let upper = values.into_iter().take(inner).flatten().collect::<Vec<_>>();

        let result = PLUQFactorization {
            source: self.clone(),
            p: Matrix::from_entries(self.ring().clone(), rows, rows, p_entries)?,
            l: Matrix::from_entries(self.ring().clone(), rows, inner, lower)?,
            u: Matrix::from_entries(self.ring().clone(), inner, columns, upper)?,
            q: Matrix::from_entries(self.ring().clone(), columns, columns, q_entries)?,
            rank,
        };
        result.verify()?;
        Ok(result)
    }
}

fn is_permutation_matrix<R: Field>(matrix: &Matrix<R>) -> bool {
    let (rows, columns) = matrix.shape();
    if rows != columns {
        return false;
    }
    let zero = matrix.ring().zero();
    let one = matrix.ring().one();
    (0..rows).all(|row| {
        (0..columns)
            .filter(|column| matrix.get(row, *column).is_ok_and(|value| value == one))
            .count()
            == 1
            && (0..columns).all(|column| {
                matrix
                    .get(row, column)
                    .is_ok_and(|value| value == zero || value == one)
            })
    }) && (0..columns).all(|column| {
        (0..rows)
            .filter(|row| matrix.get(*row, column).is_ok_and(|value| value == one))
            .count()
            == 1
    })
}

fn invalid<T>(claim: &'static str) -> Result<T, VerificationError> {
    Err(VerificationError::InvalidWitness { claim })
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len())
        .map_err(|_| NoetherError::Serialization("PLUQ component is too large".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, Q};

    fn rational(value: i64) -> Rational {
        Rational::from(value)
    }

    #[test]
    fn full_pivot_factorization_reconstructs_a_rectangular_matrix() {
        let matrix = Matrix::from_rows(
            Q,
            [
                [rational(0), rational(0), rational(2)],
                [rational(0), rational(3), rational(4)],
            ],
        )
        .unwrap();
        let factorization = matrix.pluq().unwrap();

        assert_eq!(factorization.rank(), 2);
        assert_eq!(factorization.source(), &matrix);
        assert_eq!(factorization.p().shape(), (2, 2));
        assert_eq!(factorization.l().shape(), (2, 2));
        assert_eq!(factorization.u().shape(), (2, 3));
        assert_eq!(factorization.q().shape(), (3, 3));
        factorization.verify().unwrap();
    }

    #[test]
    fn rank_deficient_and_empty_shapes_verify() {
        let singular = Matrix::from_rows(
            Q,
            [
                [rational(1), rational(2)],
                [rational(2), rational(4)],
                [rational(0), rational(0)],
            ],
        )
        .unwrap();
        assert_eq!(singular.pluq().unwrap().rank(), 1);

        // The second pivot requires a row swap after the first elimination,
        // exercising the corresponding swap of the already-filled L column.
        let delayed_pivot = Matrix::from_rows(
            Q,
            [
                [rational(1), rational(1)],
                [rational(2), rational(2)],
                [rational(0), rational(1)],
            ],
        )
        .unwrap();
        let delayed_factorization = delayed_pivot.pluq().unwrap();
        assert_eq!(delayed_factorization.rank(), 2);
        delayed_factorization.verify().unwrap();

        for empty in [Matrix::zeros(Q, 0, 3), Matrix::zeros(Q, 3, 0)] {
            let factorization = empty.pluq().unwrap();
            assert_eq!(factorization.rank(), 0);
            factorization.verify().unwrap();
        }
    }
}
