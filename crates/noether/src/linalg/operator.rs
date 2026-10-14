// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The common linear-operator contract.

use crate::{Ring, SemanticType};

use super::{LinalgError, Matrix, Vector};

/// A linear map that can act on vectors and dense matrix blocks.
///
/// Matrices satisfy this trait directly. Sparse and matrix-free operators can
/// implement the same contract without pretending to be dense matrices.
pub trait LinearOperator<R: Ring>: SemanticType {
    /// The exact coefficient context.
    fn ring(&self) -> &R;

    /// Dimension of the operator's input space.
    fn input_dimension(&self) -> usize;

    /// Dimension of the operator's output space.
    fn output_dimension(&self) -> usize;

    /// Apply the operator to one vector.
    fn apply(&self, vector: &Vector<R>) -> Result<Vector<R>, LinalgError>;

    /// Apply the operator to the columns of one dense matrix block.
    fn apply_block(&self, block: &Matrix<R>) -> Result<Matrix<R>, LinalgError>;
}

impl<R: Ring> LinearOperator<R> for Matrix<R> {
    fn ring(&self) -> &R {
        Matrix::ring(self)
    }

    fn input_dimension(&self) -> usize {
        self.shape().1
    }

    fn output_dimension(&self) -> usize {
        self.shape().0
    }

    fn apply(&self, vector: &Vector<R>) -> Result<Vector<R>, LinalgError> {
        if self.input_dimension() != vector.dimension() {
            return Err(LinalgError::DimensionMismatch {
                operation: "linear-operator application",
                left: self.input_dimension(),
                right: vector.dimension(),
            });
        }
        Vector::from_column_matrix(self.matmul(vector.column_matrix())?)
    }

    fn apply_block(&self, block: &Matrix<R>) -> Result<Matrix<R>, LinalgError> {
        self.matmul(block)
    }
}
