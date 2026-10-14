// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Graph invariants owned by the Noether facade.

use core::fmt;

use graph_core::Graph;

/// Largest graph for which the dense spectral fallback allocates a
/// Laplacian and runs the cubic Jacobi eigensolver.
pub const MAX_DENSE_SPECTRUM_ORDER: usize = 4_096;

/// Failure to compute a facade-owned graph invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphInvariantError {
    /// A reduced Laplacian requested a vertex outside the graph.
    VertexOutOfRange {
        /// Rejected vertex.
        vertex: usize,
        /// Graph order.
        order: usize,
    },
    /// The dense spectral calculation exceeded its explicit allocation cap.
    DenseSpectrumOrderLimit {
        /// Requested graph order.
        order: usize,
        /// Largest accepted graph order.
        cap: usize,
    },
    /// The deterministic symmetric Jacobi iteration did not converge.
    SpectrumDidNotConverge,
}

impl fmt::Display for GraphInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange { vertex, order } => {
                write!(formatter, "vertex {vertex} is outside graph order {order}")
            }
            Self::DenseSpectrumOrderLimit { order, cap } => write!(
                formatter,
                "dense Laplacian spectrum for graph order {order} exceeds the explicit cap {cap}"
            ),
            Self::SpectrumDidNotConverge => {
                formatter.write_str("symmetric Jacobi eigensolver did not converge")
            }
        }
    }
}

impl std::error::Error for GraphInvariantError {}

/// Discrete and spectral invariants layered over the curated graph-core value.
///
/// These operations live in Noether rather than the PyO3 projection because
/// they are part of the mathematical surface, while `graph-core` deliberately
/// remains focused on storage, traversal, connectivity and shortest paths.
pub trait GraphInvariants {
    /// The unweighted Laplacian `D - A`, treating defining edges as undirected.
    fn laplacian(&self) -> Vec<Vec<i64>>;

    /// Delete one row and column from the unweighted Laplacian.
    fn reduced_laplacian(&self, delete_vertex: usize)
        -> Result<Vec<Vec<i64>>, GraphInvariantError>;

    /// Unweighted vertex degrees in nonincreasing order.
    fn degree_sequence(&self) -> Vec<i64>;

    /// Eigenvalues of the symmetric unweighted Laplacian in ascending order.
    fn laplacian_spectrum(&self) -> Result<Vec<f64>, GraphInvariantError>;
}

impl GraphInvariants for Graph {
    fn laplacian(&self) -> Vec<Vec<i64>> {
        let order = self.order();
        let mut laplacian = vec![vec![0_i64; order]; order];
        for edge in self.edges() {
            let source = edge.source as usize;
            let target = edge.target as usize;
            laplacian[source][target] -= 1;
            laplacian[target][source] -= 1;
            laplacian[source][source] += 1;
            laplacian[target][target] += 1;
        }
        laplacian
    }

    fn reduced_laplacian(
        &self,
        delete_vertex: usize,
    ) -> Result<Vec<Vec<i64>>, GraphInvariantError> {
        let order = self.order();
        if delete_vertex >= order {
            return Err(GraphInvariantError::VertexOutOfRange {
                vertex: delete_vertex,
                order,
            });
        }
        let laplacian = self.laplacian();
        Ok((0..order)
            .filter(|row| *row != delete_vertex)
            .map(|row| {
                (0..order)
                    .filter(|column| *column != delete_vertex)
                    .map(|column| laplacian[row][column])
                    .collect()
            })
            .collect())
    }

    fn degree_sequence(&self) -> Vec<i64> {
        let mut degrees = vec![0_i64; self.order()];
        for edge in self.edges() {
            degrees[edge.source as usize] += 1;
            degrees[edge.target as usize] += 1;
        }
        degrees.sort_unstable_by(|left, right| right.cmp(left));
        degrees
    }

    fn laplacian_spectrum(&self) -> Result<Vec<f64>, GraphInvariantError> {
        let order = self.order();
        if order == 0 {
            return Ok(Vec::new());
        }
        if order > MAX_DENSE_SPECTRUM_ORDER {
            return Err(GraphInvariantError::DenseSpectrumOrderLimit {
                order,
                cap: MAX_DENSE_SPECTRUM_ORDER,
            });
        }
        symmetric_jacobi_eigenvalues(
            self.laplacian()
                .into_iter()
                .map(|row| row.into_iter().map(|entry| entry as f64).collect())
                .collect(),
        )
    }
}

#[allow(clippy::needless_range_loop)]
fn symmetric_jacobi_eigenvalues(
    mut matrix: Vec<Vec<f64>>,
) -> Result<Vec<f64>, GraphInvariantError> {
    let order = matrix.len();
    if order < 2 {
        return Ok(matrix.into_iter().map(|row| row[0]).collect());
    }
    let scale = matrix
        .iter()
        .flatten()
        .map(|entry| entry.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = f64::EPSILON.sqrt() * scale;
    let max_steps = 64_usize.saturating_mul(order).saturating_mul(order);

    for _ in 0..max_steps {
        let mut pivot = (0, 1);
        let mut largest = matrix[0][1].abs();
        for (row, values) in matrix.iter().enumerate().take(order - 1) {
            for (column, entry) in values.iter().enumerate().skip(row + 1) {
                let magnitude = entry.abs();
                if magnitude > largest {
                    largest = magnitude;
                    pivot = (row, column);
                }
            }
        }
        if largest <= tolerance {
            let mut eigenvalues = (0..order)
                .map(|index| matrix[index][index])
                .collect::<Vec<_>>();
            for value in &mut eigenvalues {
                if value.abs() <= tolerance {
                    *value = 0.0;
                }
            }
            eigenvalues.sort_by(f64::total_cmp);
            return Ok(eigenvalues);
        }

        let (row, column) = pivot;
        let off_diagonal = matrix[row][column];
        let tau = (matrix[column][column] - matrix[row][row]) / (2.0 * off_diagonal);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;

        let diagonal_row = matrix[row][row];
        let diagonal_column = matrix[column][column];
        matrix[row][row] = diagonal_row - tangent * off_diagonal;
        matrix[column][column] = diagonal_column + tangent * off_diagonal;
        matrix[row][column] = 0.0;
        matrix[column][row] = 0.0;
        for index in 0..order {
            if index == row || index == column {
                continue;
            }
            let index_row = matrix[index][row];
            let index_column = matrix[index][column];
            let rotated_row = cosine * index_row - sine * index_column;
            let rotated_column = sine * index_row + cosine * index_column;
            matrix[index][row] = rotated_row;
            matrix[row][index] = rotated_row;
            matrix[index][column] = rotated_column;
            matrix[column][index] = rotated_column;
        }
    }
    Err(GraphInvariantError::SpectrumDidNotConverge)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: &[f64], right: &[f64]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-8)
    }

    #[test]
    fn path_laplacian_and_reduction_are_hand_checkable() {
        let graph = Graph::from_unweighted(3, &[(0, 1), (1, 2)], false).unwrap();

        assert_eq!(
            graph.laplacian(),
            vec![vec![1, -1, 0], vec![-1, 2, -1], vec![0, -1, 1]]
        );
        assert_eq!(
            graph.reduced_laplacian(2).unwrap(),
            vec![vec![1, -1], vec![-1, 2]]
        );
        assert_eq!(graph.degree_sequence(), vec![2, 1, 1]);
        assert!(close(
            &graph.laplacian_spectrum().unwrap(),
            &[0.0, 1.0, 3.0]
        ));
    }

    #[test]
    fn spectrum_detects_connected_components() {
        let graph = Graph::from_unweighted(4, &[(0, 1), (2, 3)], false).unwrap();
        assert!(close(
            &graph.laplacian_spectrum().unwrap(),
            &[0.0, 0.0, 2.0, 2.0]
        ));
    }

    #[test]
    fn invalid_reduction_fails_closed() {
        let graph = Graph::from_unweighted(1, &[], false).unwrap();
        assert_eq!(
            graph.reduced_laplacian(1),
            Err(GraphInvariantError::VertexOutOfRange {
                vertex: 1,
                order: 1,
            })
        );
    }
}
