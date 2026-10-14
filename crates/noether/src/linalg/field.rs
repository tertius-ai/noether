// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generic exact linear algebra over field contexts.

use core::fmt;

use crate::{Field, NoetherError, Ring, SemanticType, TypeDescription, VerificationError};

use super::{LinalgError, Matrix, Vector};

static SOLUTION_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/SolutionSpace@1",
    rust_path: "noether::linalg::SolutionSpace",
    python_path: Some("noether.linalg.SolutionSpace"),
};

#[derive(Clone, PartialEq, Eq)]
enum SolutionClaim<R: Ring> {
    Affine { operator: Matrix<R>, rhs: Vector<R> },
    Kernel { operator: Matrix<R> },
    Image { operator: Matrix<R> },
}

/// A checked affine solution set over one exact field context.
///
/// Every element has the form `particular + sum(c_i * basis_i)`. A linear
/// subspace uses the zero vector as its particular point. The originating
/// operator is retained as a verification witness, so [`verify`](Self::verify)
/// checks the mathematical claim rather than only the result's shape.
#[derive(Clone, PartialEq, Eq)]
pub struct SolutionSpace<R: Ring> {
    ring: R,
    particular: Vector<R>,
    basis: Vec<Vector<R>>,
    claim: SolutionClaim<R>,
}

impl<R: Ring> SolutionSpace<R> {
    /// Coefficient field.
    pub fn ring(&self) -> &R {
        &self.ring
    }

    /// One point in the affine solution set.
    pub fn particular(&self) -> &Vector<R> {
        &self.particular
    }

    /// Deterministically ordered basis of the translation subspace.
    pub fn basis(&self) -> &[Vector<R>] {
        &self.basis
    }

    /// Dimension of the translation subspace.
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Dimension of the ambient vector space.
    pub fn ambient_dimension(&self) -> usize {
        self.particular.dimension()
    }
}

impl<R: Field> SolutionSpace<R> {
    /// Recheck the defining equations, dimension, and linear independence.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.particular.ring() != &self.ring
            || self.basis.iter().any(|vector| {
                vector.ring() != &self.ring || vector.dimension() != self.ambient_dimension()
            })
        {
            return Err(VerificationError::InvalidWitness {
                claim: "solution-space vectors share one field and ambient dimension",
            });
        }
        if !basis_is_independent(&self.ring, &self.basis) {
            return Err(VerificationError::InvalidWitness {
                claim: "solution-space direction vectors are linearly independent",
            });
        }

        match &self.claim {
            SolutionClaim::Affine { operator, rhs } => {
                let image = apply(operator, &self.particular).map_err(|_| {
                    VerificationError::InvalidWitness {
                        claim: "particular solution can be applied to its operator",
                    }
                })?;
                if image != *rhs
                    || self.basis.iter().any(|vector| {
                        apply(operator, vector).map_or(true, |image| {
                            image != Vector::zeros(self.ring.clone(), operator.shape().0)
                        })
                    })
                    || self.basis.len()
                        + operator
                            .rank()
                            .map_err(|_| VerificationError::InvalidWitness {
                                claim: "solution-space operator rank is computable",
                            })?
                        != operator.shape().1
                {
                    return Err(VerificationError::InvalidWitness {
                        claim: "affine solution satisfies A*x=b with the full kernel",
                    });
                }
            }
            SolutionClaim::Kernel { operator } => {
                if !self
                    .particular
                    .values()
                    .all(|value| value == self.ring.zero())
                    || self.basis.iter().any(|vector| {
                        apply(operator, vector).map_or(true, |image| {
                            image != Vector::zeros(self.ring.clone(), operator.shape().0)
                        })
                    })
                    || self.basis.len()
                        + operator
                            .rank()
                            .map_err(|_| VerificationError::InvalidWitness {
                                claim: "kernel operator rank is computable",
                            })?
                        != operator.shape().1
                {
                    return Err(VerificationError::InvalidWitness {
                        claim: "reported vectors form the complete kernel",
                    });
                }
            }
            SolutionClaim::Image { operator } => {
                let expected = operator
                    .image()
                    .map_err(|_| VerificationError::InvalidWitness {
                        claim: "image basis is recomputable",
                    })?;
                if self.particular != expected.particular || self.basis != expected.basis {
                    return Err(VerificationError::InvalidWitness {
                        claim: "reported vectors form the complete image",
                    });
                }
            }
        }
        Ok(())
    }
}

impl<R: Ring> fmt::Debug for SolutionSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SolutionSpace")
            .field("context", &R::TYPE_ID)
            .field("ambient_dimension", &self.ambient_dimension())
            .field("dimension", &self.dimension())
            .finish()
    }
}

impl<R: Ring> fmt::Display for SolutionSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SolutionSpace<{}>(ambient={}, dimension={})",
            R::TYPE_ID,
            self.ambient_dimension(),
            self.dimension()
        )
    }
}

impl<R: Ring> SemanticType for SolutionSpace<R> {
    const TYPE_ID: &'static str = "noether/linalg/SolutionSpace@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ring.canonical_bytes()?;
        push_frame(&mut output, &self.particular.canonical_bytes()?)?;
        for vector in &self.basis {
            push_frame(&mut output, &vector.canonical_bytes()?)?;
        }
        let claim = match &self.claim {
            SolutionClaim::Affine { operator, rhs } => {
                let mut bytes = vec![0];
                push_frame(&mut bytes, &operator.canonical_bytes()?)?;
                push_frame(&mut bytes, &rhs.canonical_bytes()?)?;
                bytes
            }
            SolutionClaim::Kernel { operator } => {
                let mut bytes = vec![1];
                push_frame(&mut bytes, &operator.canonical_bytes()?)?;
                bytes
            }
            SolutionClaim::Image { operator } => {
                let mut bytes = vec![2];
                push_frame(&mut bytes, &operator.canonical_bytes()?)?;
                bytes
            }
        };
        push_frame(&mut output, &claim)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SOLUTION_SPACE_DESCRIPTION
    }
}

impl<R: Field> Matrix<R> {
    /// Return one simultaneous solution of `self * X = rhs`, if one exists.
    ///
    /// This is an operation-private certificate primitive: it deliberately
    /// avoids constructing the full affine kernel when a caller only needs
    /// exact witnesses for several right-hand sides. Rational matrices retain
    /// FLINT's rectangular solver; every other exact field uses the ordinary
    /// checked RREF floor.
    #[cfg(any(feature = "msolve-subprocess", test))]
    pub(crate) fn particular_solution_columns(
        &self,
        rhs: &Self,
    ) -> Result<Option<Self>, LinalgError> {
        let (rows, columns) = self.shape();
        if rhs.ring() != self.ring() {
            return Err(crate::ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "right-hand sides belong to another field context".to_owned(),
            }
            .into());
        }
        if rhs.shape().0 != rows {
            return Err(LinalgError::ShapeMismatch {
                operation: "simultaneous particular solve",
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        if (self.ring() as &dyn core::any::Any).is::<crate::RationalField>() {
            return self.rational_particular_solution(rhs);
        }

        let rhs_columns = rhs.shape().1;
        let augmented_columns =
            columns
                .checked_add(rhs_columns)
                .ok_or(LinalgError::ShapeTooLarge {
                    rows,
                    cols: usize::MAX,
                })?;
        let augmented = Matrix::from_entries(
            self.ring().clone(),
            rows,
            augmented_columns,
            (0..rows)
                .flat_map(|row| {
                    (0..columns)
                        .map(move |column| {
                            self.get(row, column)
                                .expect("an in-bounds coefficient exists")
                        })
                        .chain((0..rhs_columns).map(move |column| {
                            rhs.get(row, column)
                                .expect("an in-bounds right-hand coefficient exists")
                        }))
                })
                .collect(),
        )?;
        let (reduced, pivots) = rref_with_pivots(&augmented)?;
        if pivots.iter().any(|pivot| *pivot >= columns) {
            return Ok(None);
        }
        let solution_cells =
            columns
                .checked_mul(rhs_columns)
                .ok_or(LinalgError::ShapeTooLarge {
                    rows: columns,
                    cols: rhs_columns,
                })?;
        let mut solution = vec![self.ring().zero(); solution_cells];
        for (row, pivot) in pivots.into_iter().enumerate() {
            for rhs_column in 0..rhs_columns {
                solution[pivot * rhs_columns + rhs_column] = reduced
                    .get(row, columns + rhs_column)
                    .expect("an augmented right-hand coefficient exists");
            }
        }
        let solution = Matrix::from_entries(self.ring().clone(), columns, rhs_columns, solution)?;
        if self.matmul(&solution)? != *rhs {
            return Ok(None);
        }
        Ok(Some(solution))
    }

    /// Rank over the coefficient field.
    pub fn rank(&self) -> Result<usize, LinalgError> {
        if let Some(rank) = self.cached_rank() {
            return Ok(rank);
        }
        if let Some(rank) = self.field_rank_backend() {
            self.cache_rank(rank);
            return Ok(rank);
        }
        let rank = rref_with_pivots(self)?.1.len();
        self.cache_rank(rank);
        Ok(rank)
    }

    /// Reduced row-echelon form over the coefficient field.
    pub fn rref(&self) -> Result<Self, LinalgError> {
        if let Some(reduced) = self.field_rref_backend() {
            return Ok(reduced);
        }
        Ok(rref_with_pivots(self)?.0)
    }

    /// Solve `self * x = rhs`, returning the complete affine solution space.
    pub fn solve(&self, rhs: &Vector<R>) -> Result<SolutionSpace<R>, LinalgError> {
        let (rows, columns) = self.shape();
        if rhs.ring() != self.ring() {
            return Err(crate::ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "right-hand side belongs to another field context".to_owned(),
            }
            .into());
        }
        if rhs.dimension() != rows {
            return Err(LinalgError::DimensionMismatch {
                operation: "linear solve",
                left: rows,
                right: rhs.dimension(),
            });
        }
        if let Some(solution) = self.field_solve_backend(rhs.column_matrix()) {
            let solution = solution.ok_or(LinalgError::InconsistentSystem)?;
            let kernel = self
                .field_nullspace_backend()
                .expect("solve and nullspace routes exist for the same field storage");
            let result = SolutionSpace {
                ring: self.ring().clone(),
                particular: Vector::from_column_matrix(solution)?,
                basis: columns_as_vectors(kernel)?,
                claim: SolutionClaim::Affine {
                    operator: self.clone(),
                    rhs: rhs.clone(),
                },
            };
            result.verify()?;
            return Ok(result);
        }
        let mut augmented_rows = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut values = (0..columns)
                .map(|column| self.get(row, column).expect("in-bounds matrix entry"))
                .collect::<Vec<_>>();
            values.push(rhs.get(row).expect("in-bounds vector entry"));
            augmented_rows.push(values);
        }
        let augmented = Matrix::from_entries(
            self.ring().clone(),
            rows,
            columns + 1,
            augmented_rows.into_iter().flatten().collect(),
        )?;
        let (reduced, pivots) = rref_with_pivots(&augmented)?;
        if pivots.contains(&columns) {
            return Err(LinalgError::InconsistentSystem);
        }
        let mut particular = vec![self.ring().zero(); columns];
        for (row, pivot) in pivots.iter().enumerate() {
            particular[*pivot] = reduced
                .get(row, columns)
                .expect("augmented right-hand column exists");
        }
        let basis = kernel_basis_from_rref(self, &reduced, &pivots)?;
        let result = SolutionSpace {
            ring: self.ring().clone(),
            particular: Vector::from_values(self.ring().clone(), particular)?,
            basis,
            claim: SolutionClaim::Affine {
                operator: self.clone(),
                rhs: rhs.clone(),
            },
        };
        result.verify()?;
        Ok(result)
    }

    /// The complete right kernel as a checked linear subspace.
    pub fn kernel(&self) -> Result<SolutionSpace<R>, LinalgError> {
        if let Some(kernel) = self.field_nullspace_backend() {
            let result = SolutionSpace {
                ring: self.ring().clone(),
                particular: Vector::zeros(self.ring().clone(), self.shape().1),
                basis: columns_as_vectors(kernel)?,
                claim: SolutionClaim::Kernel {
                    operator: self.clone(),
                },
            };
            result.verify()?;
            return Ok(result);
        }
        let result = self.kernel_from_elimination()?;
        result.verify()?;
        Ok(result)
    }

    pub(crate) fn kernel_from_elimination(&self) -> Result<SolutionSpace<R>, LinalgError> {
        Ok(self.kernel_from_elimination_with_free_columns()?.0)
    }

    pub(crate) fn kernel_from_elimination_with_free_columns(
        &self,
    ) -> Result<(SolutionSpace<R>, Vec<usize>), LinalgError> {
        let (reduced, pivots) = rref_with_pivots(self)?;
        let free_columns = (0..self.shape().1)
            .filter(|column| !pivots.contains(column))
            .collect::<Vec<_>>();
        let basis = kernel_basis_from_rref(self, &reduced, &pivots)?;
        Ok((
            SolutionSpace {
                ring: self.ring().clone(),
                particular: Vector::zeros(self.ring().clone(), self.shape().1),
                basis,
                claim: SolutionClaim::Kernel {
                    operator: self.clone(),
                },
            },
            free_columns,
        ))
    }

    pub(crate) fn pivot_columns_from_elimination(&self) -> Result<Vec<usize>, LinalgError> {
        Ok(rref_with_pivots(self)?.1)
    }

    /// Column image as a checked linear subspace of the codomain.
    pub fn image(&self) -> Result<SolutionSpace<R>, LinalgError> {
        let (_, pivots) = rref_with_pivots(self)?;
        let rows = self.shape().0;
        let basis = pivots
            .into_iter()
            .map(|column| {
                Vector::from_values(
                    self.ring().clone(),
                    (0..rows).map(|row| {
                        self.get(row, column)
                            .expect("pivot column is inside the matrix")
                    }),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SolutionSpace {
            ring: self.ring().clone(),
            particular: Vector::zeros(self.ring().clone(), rows),
            basis,
            claim: SolutionClaim::Image {
                operator: self.clone(),
            },
        })
    }
}

fn rref_with_pivots<R: Field>(matrix: &Matrix<R>) -> Result<(Matrix<R>, Vec<usize>), LinalgError> {
    let (rows, columns) = matrix.shape();
    if let Some(reduced) = matrix.field_rref_backend() {
        let mut pivots = Vec::new();
        for row in 0..rows {
            if let Some(column) = (0..columns).find(|column| {
                reduced
                    .get(row, *column)
                    .expect("in-bounds reduced matrix entry")
                    != matrix.ring().zero()
            }) {
                pivots.push(column);
            }
        }
        return Ok((reduced, pivots));
    }
    let mut values = matrix.rows().collect::<Vec<_>>();
    let mut pivot_row = 0;
    let mut pivots = Vec::new();
    for column in 0..columns {
        let Some(selected) =
            (pivot_row..rows).find(|row| values[*row][column] != matrix.ring().zero())
        else {
            continue;
        };
        values.swap(selected, pivot_row);
        let inverse = matrix.ring().inverse(&values[pivot_row][column])?;
        for value in &mut values[pivot_row][column..] {
            *value = matrix.ring().mul(value, &inverse)?;
        }
        let normalized = values[pivot_row].clone();
        for (row, values_row) in values.iter_mut().enumerate() {
            if row == pivot_row || values_row[column] == matrix.ring().zero() {
                continue;
            }
            let factor = values_row[column].clone();
            for (value, normalized_value) in
                values_row[column..].iter_mut().zip(&normalized[column..])
            {
                let product = matrix.ring().mul(&factor, normalized_value)?;
                *value = matrix.ring().sub(value, &product)?;
            }
        }
        pivots.push(column);
        pivot_row += 1;
        if pivot_row == rows {
            break;
        }
    }
    Ok((
        Matrix::from_entries(
            matrix.ring().clone(),
            rows,
            columns,
            values.into_iter().flatten().collect(),
        )?,
        pivots,
    ))
}

fn kernel_basis_from_rref<R: Field>(
    matrix: &Matrix<R>,
    reduced: &Matrix<R>,
    pivots: &[usize],
) -> Result<Vec<Vector<R>>, LinalgError> {
    let columns = matrix.shape().1;
    (0..columns)
        .filter(|column| !pivots.contains(column))
        .map(|free| {
            let mut vector = vec![matrix.ring().zero(); columns];
            vector[free] = matrix.ring().one();
            for (row, pivot) in pivots.iter().enumerate() {
                vector[*pivot] = matrix.ring().neg(
                    &reduced
                        .get(row, free)
                        .expect("free column is inside the reduced matrix"),
                )?;
            }
            Vector::from_values(matrix.ring().clone(), vector)
        })
        .collect()
}

fn apply<R: Field>(matrix: &Matrix<R>, vector: &Vector<R>) -> Result<Vector<R>, LinalgError> {
    let columns = matrix.shape().1;
    if vector.ring() != matrix.ring() {
        return Err(crate::ContextMismatchError {
            expected: R::TYPE_ID.to_owned(),
            actual: "vector belongs to another field context".to_owned(),
        }
        .into());
    }
    if vector.dimension() != columns {
        return Err(LinalgError::DimensionMismatch {
            operation: "matrix-vector multiplication",
            left: columns,
            right: vector.dimension(),
        });
    }
    Vector::from_column_matrix(matrix.matmul(vector.column_matrix())?)
}

fn columns_as_vectors<R: Field>(matrix: Matrix<R>) -> Result<Vec<Vector<R>>, LinalgError> {
    let (rows, columns) = matrix.shape();
    (0..columns)
        .map(|column| {
            Vector::from_values(
                matrix.ring().clone(),
                (0..rows).map(|row| {
                    matrix
                        .get(row, column)
                        .expect("basis column is inside the kernel matrix")
                }),
            )
        })
        .collect()
}

fn basis_is_independent<R: Field>(ring: &R, basis: &[Vector<R>]) -> bool {
    if basis.is_empty() {
        return true;
    }
    let ambient = basis[0].dimension();
    let matrix = Matrix::from_entries(
        ring.clone(),
        ambient,
        basis.len(),
        (0..ambient)
            .flat_map(|row| {
                basis.iter().map(move |vector| {
                    vector
                        .get(row)
                        .expect("basis vector has the declared ambient dimension")
                })
            })
            .collect(),
    );
    matrix.is_ok_and(|matrix| matrix.rank().is_ok_and(|rank| rank == basis.len()))
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        NoetherError::Serialization("solution-space component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}
