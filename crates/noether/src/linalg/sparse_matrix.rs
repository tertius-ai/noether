// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrices over explicit coefficient contexts.

use core::any::{Any, TypeId};
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use spar_core::SparseRingStorage;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use crate::{
    numthy::{FiniteField, FiniteFieldElement},
    CoefficientMap, CoefficientMappingError, ComplexBall, ComplexBallField, ContextMismatchError,
    Integer, IntegerRing, NoetherError, RealBall, RealBallField, Ring, SemanticType,
    TypeDescription,
};

use super::{IndexError, LinalgError, LinearOperator, Matrix, Vector};

static SPARSE_MATRIX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/SparseMatrix@1",
    rust_path: "noether::linalg::SparseMatrix",
    python_path: Some("noether.linalg.SparseMatrix"),
};

enum SparseBackend {
    Integer(sparla::storage::CsrZ),
    Binary(mzs::Mzs),
    PrimeField(sparla::storage::CsrFp),
    RealBall(sparb::CsrArb),
    ComplexBall(sparb::CsrAcb),
}

#[derive(Clone, PartialEq, Eq)]
struct CanonicalCsr<E> {
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<E>,
}

/// An immutable sparse matrix over one explicit coefficient context.
///
/// The public identity is a canonical compressed-sparse-row matrix, not a
/// kernel backend. Duplicate triplets are added, exact zeros are removed, and
/// column indices are strictly increasing within each row. Specialized sparse
/// kernels may replace the private storage without changing this type.
///
/// # Examples
///
/// ```
/// use noether::linalg::{LinearOperator, SparseMatrix, Vector};
/// use noether::{Integer, Z};
///
/// let matrix = SparseMatrix::from_triplets(
///     Z,
///     2,
///     3,
///     [
///         (0, 0, Integer::from(2)),
///         (0, 2, Integer::from(3)),
///         (1, 1, Integer::from(5)),
///     ],
/// )?;
/// let vector = Vector::from_values(Z, [1.into(), 2.into(), 3.into()])?;
/// assert_eq!(
///     matrix.apply(&vector)?.values().collect::<Vec<_>>(),
///     [Integer::from(11), Integer::from(10)],
/// );
/// assert_eq!(matrix.nnz(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct SparseMatrix<R: Ring> {
    ring: Arc<R>,
    rows: usize,
    cols: usize,
    canonical: Arc<OnceLock<CanonicalCsr<R::Element>>>,
    backend: Option<Arc<SparseBackend>>,
}

impl<R: Ring> PartialEq for SparseMatrix<R> {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring
            && self.rows == other.rows
            && self.cols == other.cols
            && (Arc::ptr_eq(&self.canonical, &other.canonical)
                || match (&self.backend, &other.backend) {
                    (Some(left), Some(right)) if Arc::ptr_eq(left, right) => true,
                    (Some(_), Some(_)) => self
                        .backend_equal(other)
                        .unwrap_or_else(|| self.canonical() == other.canonical()),
                    _ => self.canonical() == other.canonical(),
                })
    }
}

impl<R: Ring> Eq for SparseMatrix<R> {}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, E: Deserialize<'de>")
)]
struct SparseMatrixWire<R: Ring, E> {
    #[serde(rename = "type")]
    type_id: String,
    ring: R,
    shape: [usize; 2],
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<E>,
}

impl<R: Ring> SparseMatrix<R> {
    /// Construct a canonical sparse matrix from coordinate triplets.
    ///
    /// Repeated coordinates are added in input order. Entries whose combined
    /// coefficient is exactly zero are omitted.
    pub fn from_triplets<I>(
        ring: R,
        rows: usize,
        cols: usize,
        entries: I,
    ) -> Result<Self, LinalgError>
    where
        I: IntoIterator<Item = (usize, usize, R::Element)>,
    {
        validate_shape(rows, cols)?;
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        for (row, col, value) in &entries {
            if *row >= rows || *col >= cols {
                return Err(LinalgError::SparseEntryOutsideShape {
                    row: *row,
                    col: *col,
                    shape: (rows, cols),
                });
            }
            if !ring.contains(value) {
                return Err(LinalgError::ElementOutsideContext {
                    row: *row,
                    col: *col,
                });
            }
        }
        entries.sort_by_key(|(row, col, _)| (*row, *col));

        let mut combined: Vec<(usize, usize, R::Element)> = Vec::with_capacity(entries.len());
        for (row, col, value) in entries {
            if let Some((previous_row, previous_col, previous_value)) = combined.last_mut() {
                if *previous_row == row && *previous_col == col {
                    *previous_value = ring.add(previous_value, &value)?;
                    continue;
                }
            }
            combined.push((row, col, value));
        }
        combined.retain(|(_, _, value)| !ring.is_zero(value));

        let mut row_offsets = vec![0usize; rows + 1];
        for (row, _, _) in &combined {
            row_offsets[*row + 1] += 1;
        }
        for row in 0..rows {
            row_offsets[row + 1] += row_offsets[row];
        }
        let mut column_indices = Vec::with_capacity(combined.len());
        let mut values = Vec::with_capacity(combined.len());
        for (_, col, value) in combined {
            column_indices.push(col);
            values.push(value);
        }

        Ok(Self::from_canonical(
            ring,
            rows,
            cols,
            CanonicalCsr {
                row_offsets,
                column_indices,
                values,
            },
        ))
    }

    /// Construct a sparse matrix from dense rows.
    pub fn from_rows<I, J>(ring: R, rows: I) -> Result<Self, LinalgError>
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = R::Element>,
    {
        let rows = rows
            .into_iter()
            .map(|row| row.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);
        for (row, values) in rows.iter().enumerate().skip(1) {
            if values.len() != col_count {
                return Err(LinalgError::RaggedRows {
                    row,
                    expected: col_count,
                    actual: values.len(),
                });
            }
        }
        let filter_ring = ring.clone();
        let entries = rows
            .into_iter()
            .enumerate()
            .flat_map(|(row, values)| {
                let ring = filter_ring.clone();
                values
                    .into_iter()
                    .enumerate()
                    .filter(move |(_, value)| !ring.is_zero(value))
                    .map(move |(col, value)| (row, col, value))
            })
            .collect::<Vec<_>>();
        Self::from_triplets(ring, row_count, col_count, entries)
    }

    /// Construct a canonical CSR matrix from its component arrays.
    ///
    /// The arrays must already be canonical: row offsets are nondecreasing,
    /// columns are strictly increasing within each row, and no value is zero.
    pub fn from_csr(
        ring: R,
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<R::Element>,
    ) -> Result<Self, LinalgError> {
        validate_shape(rows, cols)?;
        if row_offsets.len() != rows + 1 {
            return Err(invalid_sparse(format!(
                "row_offsets has length {}; expected {}",
                row_offsets.len(),
                rows + 1
            )));
        }
        if row_offsets.first() != Some(&0) {
            return Err(invalid_sparse("row_offsets must begin at zero"));
        }
        if column_indices.len() != values.len() {
            return Err(invalid_sparse(format!(
                "column_indices has length {} but values has length {}",
                column_indices.len(),
                values.len()
            )));
        }
        if row_offsets.last() != Some(&values.len()) {
            return Err(invalid_sparse(format!(
                "the final row offset must equal nnz {}",
                values.len()
            )));
        }
        for row in 0..rows {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            if start > end || end > values.len() {
                return Err(invalid_sparse(format!(
                    "row {row} has invalid offset range {start}..{end}"
                )));
            }
            let mut previous = None;
            for index in start..end {
                let col = column_indices[index];
                if col >= cols {
                    return Err(LinalgError::SparseEntryOutsideShape {
                        row,
                        col,
                        shape: (rows, cols),
                    });
                }
                if previous.is_some_and(|value| value >= col) {
                    return Err(invalid_sparse(format!(
                        "columns in row {row} are not strictly increasing"
                    )));
                }
                if !ring.contains(&values[index]) {
                    return Err(LinalgError::ElementOutsideContext { row, col });
                }
                if ring.is_zero(&values[index]) {
                    return Err(invalid_sparse(format!(
                        "stored entry ({row}, {col}) is exactly zero"
                    )));
                }
                previous = Some(col);
            }
        }
        Ok(Self::from_canonical(
            ring,
            rows,
            cols,
            CanonicalCsr {
                row_offsets,
                column_indices,
                values,
            },
        ))
    }

    /// Construct the sparse zero matrix with the requested shape.
    pub fn zeros(ring: R, rows: usize, cols: usize) -> Self {
        validate_shape(rows, cols).expect("sparse matrix shape exceeds supported dimensions");
        Self::from_canonical(
            ring,
            rows,
            cols,
            CanonicalCsr {
                row_offsets: vec![0; rows + 1],
                column_indices: Vec::new(),
                values: Vec::new(),
            },
        )
    }

    /// Losslessly convert a dense matrix into canonical sparse form.
    pub fn from_dense(matrix: &Matrix<R>) -> Result<Self, LinalgError> {
        let (rows, cols) = matrix.shape();
        let ring = matrix.ring().clone();
        let entries = matrix
            .rows()
            .enumerate()
            .flat_map(|(row, values)| {
                let ring = ring.clone();
                values
                    .into_iter()
                    .enumerate()
                    .filter(move |(_, value)| !ring.is_zero(value))
                    .map(move |(col, value)| (row, col, value))
            })
            .collect::<Vec<_>>();
        Self::from_triplets(matrix.ring().clone(), rows, cols, entries)
    }

    /// The exact coefficient context.
    pub fn ring(&self) -> &R {
        self.ring.as_ref()
    }

    /// `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Number of explicitly stored nonzero entries.
    pub fn nnz(&self) -> usize {
        match self.backend.as_deref() {
            Some(SparseBackend::Integer(matrix)) => matrix.nnz(),
            Some(SparseBackend::Binary(matrix)) => matrix.nnz(),
            Some(SparseBackend::PrimeField(matrix)) => matrix.nnz(),
            Some(SparseBackend::RealBall(matrix)) => matrix.nnz(),
            Some(SparseBackend::ComplexBall(matrix)) => matrix.nnz(),
            None => self.canonical().values.len(),
        }
    }

    /// Fraction of matrix cells that are explicitly stored.
    pub fn density(&self) -> f64 {
        SparseRingStorage::density(self)
    }

    /// Canonical CSR row offsets.
    pub fn row_offsets(&self) -> &[usize] {
        &self.canonical().row_offsets
    }

    /// Canonical CSR column indices.
    pub fn column_indices(&self) -> &[usize] {
        &self.canonical().column_indices
    }

    /// Canonical CSR nonzero coefficients.
    pub fn values(&self) -> &[R::Element] {
        &self.canonical().values
    }

    /// Materialize one coefficient, failing when its index is outside the shape.
    pub fn get(&self, row: usize, col: usize) -> Result<R::Element, IndexError> {
        if row >= self.rows || col >= self.cols {
            return Err(IndexError {
                row,
                col,
                shape: self.shape(),
            });
        }
        let range = self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1];
        match self.canonical().column_indices[range.clone()].binary_search(&col) {
            Ok(offset) => Ok(self.canonical().values[range.start + offset].clone()),
            Err(_) => Ok(self.ring.zero()),
        }
    }

    /// Losslessly materialize the corresponding dense matrix.
    pub fn to_dense(&self) -> Matrix<R> {
        if self.rows == 0 || self.cols == 0 {
            return Matrix::zeros(self.ring().clone(), self.rows, self.cols);
        }
        let mut entries = vec![self.ring.zero(); self.rows * self.cols];
        for row in 0..self.rows {
            for index in self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1] {
                entries[row * self.cols + self.canonical().column_indices[index]] =
                    self.canonical().values[index].clone();
            }
        }
        Matrix::from_validated_entries(self.ring().clone(), self.rows, self.cols, entries)
    }

    /// Apply one checked scalar map while retaining canonical sparse storage.
    ///
    /// Source nonzeros that map to zero are omitted from the target CSR value;
    /// the exact rectangular shape is preserved even when one dimension is
    /// zero.
    pub fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<SparseMatrix<M::Target>, CoefficientMappingError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        if self.ring() != map.source() {
            return Err(ContextMismatchError {
                expected: map.source().to_string(),
                actual: self.ring().to_string(),
            }
            .into());
        }
        let mut entries = Vec::with_capacity(self.nnz());
        for row in 0..self.rows {
            for index in self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1] {
                entries.push((
                    row,
                    self.canonical().column_indices[index],
                    map.apply(&self.canonical().values[index])
                        .map_err(|error| CoefficientMappingError::Coefficient(error.to_string()))?,
                ));
            }
        }
        SparseMatrix::from_triplets(map.target().clone(), self.rows, self.cols, entries)
            .map_err(|error| CoefficientMappingError::Value(error.to_string()))
    }

    /// Return the canonical sparse transpose.
    pub fn transpose(&self) -> Self {
        if let Some(transpose) = self.backend_transpose() {
            return transpose;
        }
        let entries = (0..self.rows).flat_map(|row| {
            (self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]).map(
                move |index| {
                    (
                        self.canonical().column_indices[index],
                        row,
                        self.canonical().values[index].clone(),
                    )
                },
            )
        });
        Self::from_triplets(self.ring().clone(), self.cols, self.rows, entries)
            .expect("transposing canonical sparse entries preserves every invariant")
    }

    /// Add sparse matrices with equal contexts and shapes.
    pub fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        self.require_same_shape(rhs, "sparse matrix addition")?;
        if let Some(sum) = self.backend_combine(rhs, false) {
            return sum;
        }
        self.combine(rhs, false)
    }

    /// Subtract sparse matrices with equal contexts and shapes.
    pub fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        self.require_same_shape(rhs, "sparse matrix subtraction")?;
        if let Some(difference) = self.backend_combine(rhs, true) {
            return difference;
        }
        self.combine(rhs, true)
    }

    /// Negate every stored coefficient.
    pub fn neg(&self) -> Result<Self, LinalgError> {
        if let Some(backend) = self.backend_neg() {
            return Self::from_backend(self.ring().clone(), backend);
        }
        Self::from_csr(
            self.ring().clone(),
            self.rows,
            self.cols,
            self.canonical().row_offsets.clone(),
            self.canonical().column_indices.clone(),
            self.canonical()
                .values
                .iter()
                .map(|value| self.ring.neg(value).map_err(LinalgError::from))
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Multiply every stored coefficient by a scalar in this context.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, LinalgError> {
        if !self.ring.contains(scalar) {
            return Err(ContextMismatchError {
                expected: format!("an element of the {} context", R::TYPE_ID),
                actual: "an element of a different context".to_owned(),
            }
            .into());
        }
        if let Some(backend) = self.backend_scalar_mul(scalar) {
            return Self::from_backend(self.ring().clone(), backend);
        }
        Self::from_triplets(
            self.ring().clone(),
            self.rows,
            self.cols,
            (0..self.rows)
                .flat_map(|row| {
                    (self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]).map(
                        move |index| {
                            self.ring
                                .mul(&self.canonical().values[index], scalar)
                                .map(|value| (row, self.canonical().column_indices[index], value))
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Compose two sparse matrices by exact sparse matrix multiplication.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        if self.cols != rhs.rows {
            return Err(LinalgError::ShapeMismatch {
                operation: "sparse matrix multiplication",
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        if let Some(product) = self.backend_matmul(rhs) {
            return product;
        }
        self.generic_matmul(rhs)
    }

    fn generic_matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if (0..self.rows).all(|row| {
            self.canonical().row_offsets[row + 1] - self.canonical().row_offsets[row] <= 1
        }) {
            let one = self.ring.one();
            let mut row_offsets = Vec::with_capacity(self.rows + 1);
            let mut column_indices = Vec::new();
            let mut values = Vec::new();
            row_offsets.push(0);
            for row in 0..self.rows {
                let left_index = self.canonical().row_offsets[row];
                if left_index == self.canonical().row_offsets[row + 1] {
                    row_offsets.push(values.len());
                    continue;
                }
                let inner = self.canonical().column_indices[left_index];
                let left = &self.canonical().values[left_index];
                for right_index in
                    rhs.canonical().row_offsets[inner]..rhs.canonical().row_offsets[inner + 1]
                {
                    let value = if left == &one {
                        rhs.canonical().values[right_index].clone()
                    } else {
                        self.ring.mul(left, &rhs.canonical().values[right_index])?
                    };
                    if !self.ring.is_zero(&value) {
                        column_indices.push(rhs.canonical().column_indices[right_index]);
                        values.push(value);
                    }
                }
                row_offsets.push(values.len());
            }
            return Self::from_csr(
                self.ring().clone(),
                self.rows,
                rhs.cols,
                row_offsets,
                column_indices,
                values,
            );
        }
        let mut row_offsets = Vec::with_capacity(self.rows + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in 0..self.rows {
            let mut accumulated = BTreeMap::<usize, Vec<(&R::Element, &R::Element)>>::new();
            for left_index in
                self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
            {
                let inner = self.canonical().column_indices[left_index];
                let left = &self.canonical().values[left_index];
                for right_index in
                    rhs.canonical().row_offsets[inner]..rhs.canonical().row_offsets[inner + 1]
                {
                    let col = rhs.canonical().column_indices[right_index];
                    accumulated
                        .entry(col)
                        .or_default()
                        .push((left, &rhs.canonical().values[right_index]));
                }
            }
            for (col, products) in accumulated {
                let value = self.ring.sum_products(&products)?;
                if !self.ring.is_zero(&value) {
                    column_indices.push(col);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
        }
        Self::from_csr(
            self.ring().clone(),
            self.rows,
            rhs.cols,
            row_offsets,
            column_indices,
            values,
        )
    }

    fn from_canonical(
        ring: R,
        rows: usize,
        cols: usize,
        canonical: CanonicalCsr<R::Element>,
    ) -> Self {
        let mut matrix = Self {
            ring: Arc::new(ring),
            rows,
            cols,
            canonical: Arc::new(OnceLock::from(canonical)),
            backend: None,
        };
        matrix.backend = matrix.build_backend().map(Arc::new);
        if matrix.backend.is_some() {
            // The specialized CSR becomes authoritative. Canonical public
            // arrays are reconstructed only if a caller explicitly inspects
            // or serializes them.
            matrix.canonical = Arc::new(OnceLock::new());
        }
        matrix
    }

    fn canonical(&self) -> &CanonicalCsr<R::Element> {
        self.canonical.get_or_init(|| self.canonical_from_backend())
    }

    fn canonical_from_backend(&self) -> CanonicalCsr<R::Element> {
        let backend = self
            .backend
            .as_deref()
            .expect("an empty canonical sparse cache requires a kernel backend");
        let entries = match backend {
            SparseBackend::Integer(matrix) => (0..matrix.rows())
                .flat_map(|row| {
                    matrix.iter_row(row).map(move |(col, value)| {
                        (row, col, cast_owned::<Integer, R::Element>(value.clone()))
                    })
                })
                .collect::<Vec<_>>(),
            SparseBackend::Binary(matrix) => {
                let field = (self.ring() as &dyn Any)
                    .downcast_ref::<FiniteField>()
                    .expect("binary sparse backend context invariant");
                let one = cast_owned::<FiniteFieldElement, R::Element>(field.one());
                (0..matrix.rows())
                    .flat_map(|row| {
                        let one = one.clone();
                        matrix
                            .row_indices(row)
                            .iter()
                            .map(move |column| (row, *column as usize, one.clone()))
                    })
                    .collect::<Vec<_>>()
            }
            SparseBackend::PrimeField(matrix) => {
                let field = (self.ring() as &dyn Any)
                    .downcast_ref::<FiniteField>()
                    .expect("prime-field sparse backend context invariant");
                (0..matrix.rows())
                    .flat_map(|row| {
                        matrix.iter_row(row).map(move |(col, value)| {
                            (
                                row,
                                col,
                                cast_owned::<FiniteFieldElement, R::Element>(
                                    field.element(u64::from(value)),
                                ),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            }
            SparseBackend::RealBall(matrix) => {
                let field = *(self.ring() as &dyn Any)
                    .downcast_ref::<RealBallField>()
                    .expect("real-ball sparse backend context invariant");
                (0..matrix.rows())
                    .flat_map(|row| {
                        matrix.iter_row(row).map(move |(col, value)| {
                            (
                                row,
                                col,
                                cast_owned::<RealBall, R::Element>(RealBall::from_raw(
                                    field,
                                    value.clone(),
                                )),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            }
            SparseBackend::ComplexBall(matrix) => {
                let field = *(self.ring() as &dyn Any)
                    .downcast_ref::<ComplexBallField>()
                    .expect("complex-ball sparse backend context invariant");
                (0..matrix.rows())
                    .flat_map(|row| {
                        matrix.iter_row(row).map(move |(col, value)| {
                            (
                                row,
                                col,
                                cast_owned::<ComplexBall, R::Element>(ComplexBall::from_raw(
                                    field,
                                    value.clone(),
                                )),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            }
        };
        let mut row_offsets = vec![0; self.rows + 1];
        let mut column_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for (row, column, value) in entries {
            row_offsets[row + 1] += 1;
            column_indices.push(column);
            values.push(value);
        }
        for row in 0..self.rows {
            row_offsets[row + 1] += row_offsets[row];
        }
        CanonicalCsr {
            row_offsets,
            column_indices,
            values,
        }
    }

    fn backend_equal(&self, other: &Self) -> Option<bool> {
        match (self.backend.as_deref()?, other.backend.as_deref()?) {
            (SparseBackend::Integer(left), SparseBackend::Integer(right)) => {
                Some((0..self.rows).all(|row| left.iter_row(row).eq(right.iter_row(row))))
            }
            (SparseBackend::Binary(left), SparseBackend::Binary(right)) => {
                Some((0..self.rows).all(|row| left.row_indices(row) == right.row_indices(row)))
            }
            (SparseBackend::PrimeField(left), SparseBackend::PrimeField(right)) => {
                Some((0..self.rows).all(|row| left.iter_row(row).eq(right.iter_row(row))))
            }
            (SparseBackend::RealBall(left), SparseBackend::RealBall(right)) => {
                Some((0..self.rows).all(|row| {
                    left.iter_row(row).zip(right.iter_row(row)).all(
                        |((left_col, left_value), (right_col, right_value))| {
                            left_col == right_col && left_value.same_ball(right_value)
                        },
                    ) && left.iter_row(row).count() == right.iter_row(row).count()
                }))
            }
            (SparseBackend::ComplexBall(left), SparseBackend::ComplexBall(right)) => {
                Some((0..self.rows).all(|row| {
                    left.iter_row(row).zip(right.iter_row(row)).all(
                        |((left_col, left_value), (right_col, right_value))| {
                            left_col == right_col && left_value.same_ball(right_value)
                        },
                    ) && left.iter_row(row).count() == right.iter_row(row).count()
                }))
            }
            _ => None,
        }
    }

    fn build_backend(&self) -> Option<SparseBackend> {
        if TypeId::of::<R>() == TypeId::of::<IntegerRing>()
            && TypeId::of::<R::Element>() == TypeId::of::<Integer>()
        {
            let mut entries = Vec::with_capacity(self.nnz());
            for row in 0..self.rows {
                for index in
                    self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
                {
                    let value = (&self.canonical().values[index] as &dyn Any)
                        .downcast_ref::<Integer>()?
                        .clone();
                    entries.push((row, self.canonical().column_indices[index], value));
                }
            }
            return Some(SparseBackend::Integer(
                sparla::storage::CsrZ::from_coo_owned(self.rows, self.cols, entries),
            ));
        }

        if TypeId::of::<R>() == TypeId::of::<FiniteField>()
            && TypeId::of::<R::Element>() == TypeId::of::<FiniteFieldElement>()
        {
            let field = (self.ring() as &dyn Any).downcast_ref::<FiniteField>()?;
            let prime = field.word_characteristic()?;
            if field.degree() != 1 || prime > sparla::storage::MAX_PRIME {
                return None;
            }
            if prime == 2 && self.cols <= u32::MAX as usize {
                let mut positions = Vec::with_capacity(self.nnz());
                for row in 0..self.rows {
                    for index in
                        self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
                    {
                        positions.push((row, self.canonical().column_indices[index]));
                    }
                }
                return Some(SparseBackend::Binary(mzs::Mzs::from_triples(
                    self.rows, self.cols, &positions,
                )));
            }
            let mut entries = Vec::with_capacity(self.nnz());
            for row in 0..self.rows {
                for index in
                    self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
                {
                    let value = (&self.canonical().values[index] as &dyn Any)
                        .downcast_ref::<FiniteFieldElement>()?
                        .representative()
                        .to_i64()?;
                    entries.push((row, self.canonical().column_indices[index], value));
                }
            }
            return Some(SparseBackend::PrimeField(sparla::storage::CsrFp::from_coo(
                self.rows, self.cols, &entries, prime,
            )));
        }

        if TypeId::of::<R>() == TypeId::of::<RealBallField>()
            && TypeId::of::<R::Element>() == TypeId::of::<RealBall>()
        {
            let field = (self.ring() as &dyn Any).downcast_ref::<RealBallField>()?;
            let mut entries = Vec::with_capacity(self.nnz());
            for row in 0..self.rows {
                for index in
                    self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
                {
                    let value = (&self.canonical().values[index] as &dyn Any)
                        .downcast_ref::<RealBall>()?
                        .raw()
                        .clone();
                    entries.push((row, self.canonical().column_indices[index], value));
                }
            }
            return Some(SparseBackend::RealBall(sparb::CsrArb::from_coo_balls(
                self.rows,
                self.cols,
                field.precision_bits() as i64,
                entries,
            )));
        }

        if TypeId::of::<R>() == TypeId::of::<ComplexBallField>()
            && TypeId::of::<R::Element>() == TypeId::of::<ComplexBall>()
        {
            let field = (self.ring() as &dyn Any).downcast_ref::<ComplexBallField>()?;
            let mut entries = Vec::with_capacity(self.nnz());
            for row in 0..self.rows {
                for index in
                    self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1]
                {
                    let value = (&self.canonical().values[index] as &dyn Any)
                        .downcast_ref::<ComplexBall>()?
                        .raw()
                        .clone();
                    entries.push((row, self.canonical().column_indices[index], value));
                }
            }
            return Some(SparseBackend::ComplexBall(sparb::CsrAcb::from_coo_balls(
                self.rows,
                self.cols,
                field.precision_bits() as i64,
                entries,
            )));
        }

        None
    }

    fn from_backend(ring: R, backend: SparseBackend) -> Result<Self, LinalgError> {
        let rows = match &backend {
            SparseBackend::Integer(matrix) => matrix.rows(),
            SparseBackend::Binary(matrix) => matrix.rows(),
            SparseBackend::PrimeField(matrix) => matrix.rows(),
            SparseBackend::RealBall(matrix) => matrix.rows(),
            SparseBackend::ComplexBall(matrix) => matrix.rows(),
        };
        let cols = match &backend {
            SparseBackend::Integer(matrix) => matrix.cols(),
            SparseBackend::Binary(matrix) => matrix.cols(),
            SparseBackend::PrimeField(matrix) => matrix.cols(),
            SparseBackend::RealBall(matrix) => matrix.cols(),
            SparseBackend::ComplexBall(matrix) => matrix.cols(),
        };
        Ok(Self {
            ring: Arc::new(ring),
            rows,
            cols,
            canonical: Arc::new(OnceLock::new()),
            backend: Some(Arc::new(backend)),
        })
    }

    fn backend_transpose(&self) -> Option<Self> {
        let backend = match self.backend.as_deref()? {
            SparseBackend::Integer(matrix) => SparseBackend::Integer(matrix.transpose()),
            SparseBackend::Binary(matrix) => SparseBackend::Binary(matrix.transpose()),
            SparseBackend::PrimeField(matrix) => SparseBackend::PrimeField(matrix.transpose()),
            SparseBackend::RealBall(matrix) => SparseBackend::RealBall(matrix.transpose()),
            SparseBackend::ComplexBall(matrix) => SparseBackend::ComplexBall(matrix.transpose()),
        };
        Some(
            Self::from_backend(self.ring().clone(), backend)
                .expect("a transposed kernel CSR matrix remains canonical"),
        )
    }

    fn backend_combine(&self, rhs: &Self, subtract: bool) -> Option<Result<Self, LinalgError>> {
        let backend = match (self.backend.as_deref()?, rhs.backend.as_deref()?) {
            (SparseBackend::Integer(left), SparseBackend::Integer(right)) => {
                SparseBackend::Integer(if subtract {
                    left.sub(right)
                } else {
                    left.add(right)
                })
            }
            (SparseBackend::Binary(left), SparseBackend::Binary(right)) => {
                SparseBackend::Binary(left.add(right))
            }
            (SparseBackend::PrimeField(left), SparseBackend::PrimeField(right)) => {
                SparseBackend::PrimeField(if subtract {
                    left.sub(right)
                } else {
                    left.add(right)
                })
            }
            (SparseBackend::RealBall(left), SparseBackend::RealBall(right)) => {
                SparseBackend::RealBall(if subtract {
                    left.sub(right)
                } else {
                    left.add(right)
                })
            }
            (SparseBackend::ComplexBall(left), SparseBackend::ComplexBall(right)) => {
                SparseBackend::ComplexBall(if subtract {
                    left.sub(right)
                } else {
                    left.add(right)
                })
            }
            _ => return None,
        };
        Some(Self::from_backend(self.ring().clone(), backend))
    }

    fn backend_neg(&self) -> Option<SparseBackend> {
        Some(match self.backend.as_deref()? {
            SparseBackend::Integer(matrix) => SparseBackend::Integer(matrix.neg()),
            SparseBackend::Binary(matrix) => SparseBackend::Binary(matrix.clone()),
            SparseBackend::PrimeField(matrix) => SparseBackend::PrimeField(matrix.neg()),
            SparseBackend::RealBall(matrix) => {
                let minus_one = sparb::Arb::from_f64(-1.0);
                SparseBackend::RealBall(matrix.scale(&minus_one))
            }
            SparseBackend::ComplexBall(matrix) => {
                let minus_one = sparb::Acb::from_re_im_rad(-1.0, 0.0, 0.0, 0.0);
                SparseBackend::ComplexBall(matrix.scale(&minus_one))
            }
        })
    }

    fn backend_scalar_mul(&self, scalar: &R::Element) -> Option<SparseBackend> {
        Some(match self.backend.as_deref()? {
            SparseBackend::Integer(matrix) => {
                let scalar = (scalar as &dyn Any).downcast_ref::<Integer>()?;
                SparseBackend::Integer(matrix.scale(scalar))
            }
            SparseBackend::Binary(matrix) => {
                let scalar = (scalar as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                SparseBackend::Binary(
                    matrix.scalar_mul(scalar.representative().modulo_u64(2) as u8),
                )
            }
            SparseBackend::PrimeField(matrix) => {
                let scalar = (scalar as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                SparseBackend::PrimeField(
                    matrix.scale(scalar.representative().modulo_u64(matrix.modulus())),
                )
            }
            SparseBackend::RealBall(matrix) => {
                let scalar = (scalar as &dyn Any).downcast_ref::<RealBall>()?;
                SparseBackend::RealBall(matrix.scale(scalar.raw()))
            }
            SparseBackend::ComplexBall(matrix) => {
                let scalar = (scalar as &dyn Any).downcast_ref::<ComplexBall>()?;
                SparseBackend::ComplexBall(matrix.scale(scalar.raw()))
            }
        })
    }

    fn backend_matmul(&self, rhs: &Self) -> Option<Result<Self, LinalgError>> {
        let backend = match (self.backend.as_deref()?, rhs.backend.as_deref()?) {
            (SparseBackend::Integer(left), SparseBackend::Integer(right)) => {
                SparseBackend::Integer(left.matmul(right))
            }
            (SparseBackend::Binary(left), SparseBackend::Binary(right)) => {
                SparseBackend::Binary(left.matmul(right))
            }
            (SparseBackend::PrimeField(left), SparseBackend::PrimeField(right)) => {
                SparseBackend::PrimeField(left.matmul(right))
            }
            (SparseBackend::RealBall(left), SparseBackend::RealBall(right)) => {
                SparseBackend::RealBall(left.matmul(right))
            }
            (SparseBackend::ComplexBall(left), SparseBackend::ComplexBall(right)) => {
                SparseBackend::ComplexBall(left.matmul(right))
            }
            _ => return None,
        };
        Some(Self::from_backend(self.ring().clone(), backend))
    }

    /// Serialize the complete coefficient context and canonical CSR arrays.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a sparse matrix and its coefficient context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn combine(&self, rhs: &Self, subtract: bool) -> Result<Self, LinalgError> {
        let mut entries = Vec::with_capacity(self.nnz() + rhs.nnz());
        for row in 0..self.rows {
            for index in self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1] {
                entries.push((
                    row,
                    self.canonical().column_indices[index],
                    self.canonical().values[index].clone(),
                ));
            }
            for index in rhs.canonical().row_offsets[row]..rhs.canonical().row_offsets[row + 1] {
                let value = if subtract {
                    self.ring.neg(&rhs.canonical().values[index])?
                } else {
                    rhs.canonical().values[index].clone()
                };
                entries.push((row, rhs.canonical().column_indices[index], value));
            }
        }
        Self::from_triplets(self.ring().clone(), self.rows, self.cols, entries)
    }

    fn require_same_context(&self, rhs: &Self) -> Result<(), LinalgError> {
        if self.ring == rhs.ring {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: format!("the same {} context as the left matrix", R::TYPE_ID),
                actual: "a different context value".to_owned(),
            }
            .into())
        }
    }

    fn require_same_shape(&self, rhs: &Self, operation: &'static str) -> Result<(), LinalgError> {
        if self.shape() == rhs.shape() {
            Ok(())
        } else {
            Err(LinalgError::ShapeMismatch {
                operation,
                left: self.shape(),
                right: rhs.shape(),
            })
        }
    }
}

impl SparseMatrix<IntegerRing> {
    /// Exact rank of an integral sparse matrix through the selected backend.
    pub(crate) fn integral_rank(&self) -> usize {
        match self.backend.as_deref() {
            Some(SparseBackend::Integer(matrix)) => matrix.rank(),
            _ => self.to_dense().smith_normal_form().rank(),
        }
    }

    /// Nonzero Smith invariant factors through the selected backend.
    pub(crate) fn integral_smith_invariant_factors(&self) -> Vec<Integer> {
        match self.backend.as_deref() {
            Some(SparseBackend::Integer(matrix)) => {
                let zero = Integer::zero();
                matrix
                    .snf()
                    .into_iter()
                    .filter(|factor| factor != &zero)
                    .collect()
            }
            _ => self
                .to_dense()
                .smith_normal_form()
                .invariant_factors()
                .to_vec(),
        }
    }
}

impl SparseMatrix<FiniteField> {
    pub(crate) fn prime_field_kernel_basis(&self) -> Result<Self, LinalgError> {
        match self.backend.as_deref() {
            Some(SparseBackend::PrimeField(matrix)) => Self::from_backend(
                self.ring().clone(),
                SparseBackend::PrimeField(matrix.kernel()),
            ),
            _ => {
                let kernel = self.to_dense().kernel()?;
                let columns = kernel.basis();
                let entries = columns.iter().enumerate().flat_map(|(column, vector)| {
                    vector.values().enumerate().filter_map(move |(row, value)| {
                        (!value.is_zero()).then_some((row, column, value))
                    })
                });
                Self::from_triplets(self.ring().clone(), self.cols, columns.len(), entries)
            }
        }
    }

    pub(crate) fn prime_field_image_basis(&self) -> Result<Self, LinalgError> {
        match self.backend.as_deref() {
            Some(SparseBackend::PrimeField(matrix)) => Self::from_backend(
                self.ring().clone(),
                SparseBackend::PrimeField(matrix.image()),
            ),
            _ => {
                let image = self.to_dense().image()?;
                let columns = image.basis();
                let entries = columns.iter().enumerate().flat_map(|(column, vector)| {
                    vector.values().enumerate().filter_map(move |(row, value)| {
                        (!value.is_zero()).then_some((row, column, value))
                    })
                });
                Self::from_triplets(self.ring().clone(), self.rows, columns.len(), entries)
            }
        }
    }

    pub(crate) fn prime_field_pivot_columns(&self) -> Result<Vec<usize>, LinalgError> {
        match self.backend.as_deref() {
            Some(SparseBackend::PrimeField(matrix)) => Ok(matrix.rref().pivot_cols),
            _ => self.to_dense().pivot_columns_from_elimination(),
        }
    }

    pub(crate) fn prime_field_particular_solutions(
        &self,
        right_hand_sides: &Self,
    ) -> Result<Self, LinalgError> {
        self.require_same_context(right_hand_sides)?;
        if self.rows != right_hand_sides.rows {
            return Err(LinalgError::ShapeMismatch {
                operation: "sparse block solve",
                left: self.shape(),
                right: right_hand_sides.shape(),
            });
        }
        match (self.backend.as_deref(), right_hand_sides.backend.as_deref()) {
            (
                Some(SparseBackend::PrimeField(matrix)),
                Some(SparseBackend::PrimeField(right_hand_sides)),
            ) => {
                let mut entries = Vec::new();
                for column in 0..right_hand_sides.cols() {
                    let rhs_entries = (0..right_hand_sides.rows()).filter_map(|row| {
                        right_hand_sides
                            .get(row, column)
                            .map(|value| (row, 0, i64::from(value)))
                    });
                    let rhs = sparla::storage::CsrFp::from_coo(
                        right_hand_sides.rows(),
                        1,
                        &rhs_entries.collect::<Vec<_>>(),
                        right_hand_sides.modulus(),
                    );
                    let solution = matrix.solve(&rhs).ok_or(LinalgError::InconsistentSystem)?;
                    for row in 0..solution.rows() {
                        if let Some(value) = solution.get(row, 0) {
                            entries.push((row, column, i64::from(value)));
                        }
                    }
                }
                let solutions = sparla::storage::CsrFp::from_coo(
                    self.cols,
                    right_hand_sides.cols(),
                    &entries,
                    matrix.modulus(),
                );
                Self::from_backend(self.ring().clone(), SparseBackend::PrimeField(solutions))
            }
            _ => {
                let operator = self.to_dense();
                let mut entries = Vec::new();
                for column in 0..right_hand_sides.cols {
                    let rhs = Vector::from_values(
                        self.ring().clone(),
                        (0..self.rows).map(|row| {
                            right_hand_sides
                                .get(row, column)
                                .expect("right-hand-side coordinate exists")
                        }),
                    )?;
                    let solution = operator.solve(&rhs)?;
                    entries.extend(solution.particular().values().enumerate().filter_map(
                        |(row, value)| (!value.is_zero()).then_some((row, column, value)),
                    ));
                }
                Self::from_triplets(
                    self.ring().clone(),
                    self.cols,
                    right_hand_sides.cols,
                    entries,
                )
            }
        }
    }
}

impl<R: Ring> SparseRingStorage for SparseMatrix<R> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        SparseMatrix::nnz(self)
    }
}

impl<R: Ring> LinearOperator<R> for SparseMatrix<R> {
    fn ring(&self) -> &R {
        self.ring()
    }

    fn input_dimension(&self) -> usize {
        self.cols
    }

    fn output_dimension(&self) -> usize {
        self.rows
    }

    fn apply(&self, vector: &Vector<R>) -> Result<Vector<R>, LinalgError> {
        if self.cols != vector.dimension() {
            return Err(LinalgError::DimensionMismatch {
                operation: "sparse linear-operator application",
                left: self.cols,
                right: vector.dimension(),
            });
        }
        if self.ring() != vector.ring() {
            return Err(ContextMismatchError {
                expected: format!("the same {} context as the operator", R::TYPE_ID),
                actual: "a different vector context".to_owned(),
            }
            .into());
        }
        let vector_values = vector.values().collect::<Vec<_>>();
        let maximum_row_nnz = self
            .canonical()
            .row_offsets
            .windows(2)
            .map(|offsets| offsets[1] - offsets[0])
            .max()
            .unwrap_or(0);
        let mut products = Vec::with_capacity(maximum_row_nnz);
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            products.clear();
            for index in self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1] {
                let value = &vector_values[self.canonical().column_indices[index]];
                if self.ring.is_zero(value) {
                    continue;
                }
                products.push((&self.canonical().values[index], value));
            }
            output.push(self.ring.sum_products(&products)?);
        }
        Vector::from_values(self.ring().clone(), output)
    }

    fn apply_block(&self, block: &Matrix<R>) -> Result<Matrix<R>, LinalgError> {
        if self.cols != block.shape().0 {
            return Err(LinalgError::ShapeMismatch {
                operation: "sparse linear-operator block application",
                left: self.shape(),
                right: block.shape(),
            });
        }
        if self.ring() != block.ring() {
            return Err(ContextMismatchError {
                expected: format!("the same {} context as the operator", R::TYPE_ID),
                actual: "a different matrix-block context".to_owned(),
            }
            .into());
        }
        let output_cols = block.shape().1;
        let mut rows = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut output = vec![self.ring.zero(); output_cols];
            for index in self.canonical().row_offsets[row]..self.canonical().row_offsets[row + 1] {
                let inner = self.canonical().column_indices[index];
                for (col, value) in output.iter_mut().enumerate() {
                    let right = block
                        .get(inner, col)
                        .expect("validated block shape must contain this coefficient");
                    let product = self.ring.mul(&self.canonical().values[index], &right)?;
                    *value = self.ring.add(value, &product)?;
                }
            }
            rows.push(output);
        }
        if self.rows == 0 {
            Ok(Matrix::zeros(self.ring().clone(), 0, output_cols))
        } else {
            Matrix::from_rows(self.ring().clone(), rows)
        }
    }
}

impl<R> Serialize for SparseMatrix<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SparseMatrix", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", self.ring())?;
        state.serialize_field("shape", &[self.rows, self.cols])?;
        state.serialize_field("row_offsets", &self.canonical().row_offsets)?;
        state.serialize_field("column_indices", &self.canonical().column_indices)?;
        state.serialize_field("values", &self.canonical().values)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for SparseMatrix<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SparseMatrixWire::<R, R::Element>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_csr(
            wire.ring,
            wire.shape[0],
            wire.shape[1],
            wire.row_offsets,
            wire.column_indices,
            wire.values,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for SparseMatrix<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SparseMatrix")
            .field("ring_type", &R::TYPE_ID)
            .field("shape", &self.shape())
            .field("nnz", &self.nnz())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for SparseMatrix<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SparseMatrix<{}>({}x{}, nnz={})",
            R::TYPE_ID,
            self.rows,
            self.cols,
            self.nnz()
        )
    }
}

impl<R: Ring> SemanticType for SparseMatrix<R> {
    const TYPE_ID: &'static str = "noether/linalg/SparseMatrix@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, R::TYPE_ID.as_bytes())?;
        push_frame(&mut output, &self.ring.canonical_bytes()?)?;
        output.extend_from_slice(&(self.rows as u64).to_be_bytes());
        output.extend_from_slice(&(self.cols as u64).to_be_bytes());
        for offset in &self.canonical().row_offsets {
            output.extend_from_slice(&(*offset as u64).to_be_bytes());
        }
        for (col, value) in self
            .canonical()
            .column_indices
            .iter()
            .zip(&self.canonical().values)
        {
            output.extend_from_slice(&(*col as u64).to_be_bytes());
            push_frame(&mut output, &value.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SPARSE_MATRIX_DESCRIPTION
    }
}

impl<R: Ring> Add<&SparseMatrix<R>> for &SparseMatrix<R> {
    type Output = Result<SparseMatrix<R>, LinalgError>;

    fn add(self, rhs: &SparseMatrix<R>) -> Self::Output {
        SparseMatrix::add(self, rhs)
    }
}

impl<R: Ring> Sub<&SparseMatrix<R>> for &SparseMatrix<R> {
    type Output = Result<SparseMatrix<R>, LinalgError>;

    fn sub(self, rhs: &SparseMatrix<R>) -> Self::Output {
        SparseMatrix::sub(self, rhs)
    }
}

impl<R: Ring> Neg for &SparseMatrix<R> {
    type Output = Result<SparseMatrix<R>, LinalgError>;

    fn neg(self) -> Self::Output {
        SparseMatrix::neg(self)
    }
}

impl<R: Ring> Mul<&R::Element> for &SparseMatrix<R> {
    type Output = Result<SparseMatrix<R>, LinalgError>;

    fn mul(self, scalar: &R::Element) -> Self::Output {
        self.scalar_mul(scalar)
    }
}

fn validate_shape(rows: usize, cols: usize) -> Result<(), LinalgError> {
    if i64::try_from(rows).is_err()
        || i64::try_from(cols).is_err()
        || rows.checked_mul(cols).is_none()
    {
        Err(LinalgError::ShapeTooLarge { rows, cols })
    } else {
        Ok(())
    }
}

fn invalid_sparse(message: impl Into<String>) -> LinalgError {
    LinalgError::InvalidSparseStructure {
        message: message.into(),
    }
}

fn cast_owned<T: Any, U: Any>(value: T) -> U {
    let value: Box<dyn Any> = Box::new(value);
    *value
        .downcast::<U>()
        .unwrap_or_else(|_| panic!("private sparse-matrix element invariant was violated"))
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len())
        .map_err(|_| NoetherError::Serialization("canonical frame is too large".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{MonomialOrder, PolynomialRing};

    #[test]
    fn integer_storage_and_products_stay_in_sparla() {
        let left = SparseMatrix::from_rows(
            crate::Z,
            [
                [Integer::from(1), Integer::from(0), Integer::from(2)],
                [Integer::from(0), Integer::from(-3), Integer::from(0)],
            ],
        )
        .unwrap();
        let right = left.transpose();
        let product = left.matmul(&right).unwrap();
        let sum = left.add(&left).unwrap();
        let difference = sum.sub(&left).unwrap();
        let negative = left.neg().unwrap();
        let scaled = left.scalar_mul(&Integer::from(3)).unwrap();

        assert!(matches!(
            left.backend.as_deref(),
            Some(SparseBackend::Integer(_))
        ));
        assert!(matches!(
            right.backend.as_deref(),
            Some(SparseBackend::Integer(_))
        ));
        assert!(matches!(
            product.backend.as_deref(),
            Some(SparseBackend::Integer(_))
        ));
        for matrix in [
            &left,
            &right,
            &product,
            &sum,
            &difference,
            &negative,
            &scaled,
        ] {
            assert!(matrix.canonical.get().is_none());
            let _ = matrix.nnz();
            assert!(matrix.canonical.get().is_none());
        }
        assert_eq!(
            product.to_dense(),
            left.to_dense().matmul(&right.to_dense()).unwrap()
        );
    }

    #[test]
    fn prime_field_storage_and_products_stay_in_sparla() {
        let field = FiniteField::prime(7_u8).unwrap();
        let left = SparseMatrix::from_rows(
            field.clone(),
            [
                [field.element(1_u8), field.element(0_u8)],
                [field.element(2_u8), field.element(3_u8)],
            ],
        )
        .unwrap();
        let right = left.transpose();
        let product = left.matmul(&right).unwrap();
        let sum = left.add(&left).unwrap();
        let difference = sum.sub(&left).unwrap();
        let negative = left.neg().unwrap();
        let scaled = left.scalar_mul(&field.element(3_u8)).unwrap();

        assert!(matches!(
            left.backend.as_deref(),
            Some(SparseBackend::PrimeField(_))
        ));
        assert!(matches!(
            right.backend.as_deref(),
            Some(SparseBackend::PrimeField(_))
        ));
        assert!(matches!(
            product.backend.as_deref(),
            Some(SparseBackend::PrimeField(_))
        ));
        for matrix in [
            &left,
            &right,
            &product,
            &sum,
            &difference,
            &negative,
            &scaled,
        ] {
            assert!(matrix.canonical.get().is_none());
            let _ = matrix.nnz();
            assert!(matrix.canonical.get().is_none());
        }
        assert_eq!(
            product.to_dense(),
            left.to_dense().matmul(&right.to_dense()).unwrap()
        );
    }

    #[test]
    fn equality_of_clones_does_not_materialize_canonical_storage() {
        let field = FiniteField::prime(2_u8).unwrap();
        let zero = field.element(0_u8);
        let one = field.element(1_u8);
        let matrix =
            SparseMatrix::from_rows(field, [[one.clone(), zero], [one.clone(), one]]).unwrap();
        let cloned = matrix.clone();
        assert!(matrix.canonical.get().is_none());
        assert_eq!(matrix, cloned);
        assert!(matrix.canonical.get().is_none());
    }

    #[test]
    fn equality_of_distinct_backends_does_not_materialize_canonical_storage() {
        let field = FiniteField::prime(2_u8).unwrap();
        let rows = [
            [field.element(1_u8), field.element(0_u8)],
            [field.element(1_u8), field.element(1_u8)],
        ];
        let left = SparseMatrix::from_rows(field.clone(), rows.clone()).unwrap();
        let right = SparseMatrix::from_rows(field, rows).unwrap();
        assert!(left.canonical.get().is_none());
        assert!(right.canonical.get().is_none());
        assert_eq!(left, right);
        assert!(left.canonical.get().is_none());
        assert!(right.canonical.get().is_none());
    }

    #[test]
    fn prime_field_sparse_elimination_adapters_return_checked_bases_and_solutions() {
        let field = FiniteField::prime(3_u8).unwrap();
        let operator = SparseMatrix::from_rows(
            field.clone(),
            [
                [field.element(1_u8), field.element(2_u8), field.zero()],
                [field.zero(), field.element(1_u8), field.element(1_u8)],
            ],
        )
        .unwrap();
        let kernel = operator.prime_field_kernel_basis().unwrap();
        let image = operator.prime_field_image_basis().unwrap();
        assert_eq!(kernel.shape(), (3, 1));
        assert_eq!(image.shape(), (2, 2));
        assert_eq!(operator.matmul(&kernel).unwrap().nnz(), 0);
        assert_eq!(operator.prime_field_pivot_columns().unwrap(), vec![0, 1]);

        let witnesses = SparseMatrix::from_rows(
            field.clone(),
            [
                [field.one(), field.zero()],
                [field.zero(), field.one()],
                [field.one(), field.one()],
            ],
        )
        .unwrap();
        let right_hand_sides = operator.matmul(&witnesses).unwrap();
        let solutions = operator
            .prime_field_particular_solutions(&right_hand_sides)
            .unwrap();
        assert_eq!(operator.matmul(&solutions).unwrap(), right_hand_sides);
    }

    #[test]
    fn ball_storage_and_arithmetic_stay_in_sparb() {
        let real = RealBallField::new(128).unwrap();
        let real_matrix = SparseMatrix::from_rows(
            real,
            [
                [real.element(1_i64), real.element(0_i64)],
                [real.element(2_i64), real.element(3_i64)],
            ],
        )
        .unwrap();
        let real_transpose = real_matrix.transpose();
        let real_product = real_matrix.matmul(&real_transpose).unwrap();
        let real_sum = real_matrix.add(&real_matrix).unwrap();
        let real_difference = real_sum.sub(&real_matrix).unwrap();
        assert!(matches!(
            real_matrix.backend.as_deref(),
            Some(SparseBackend::RealBall(_))
        ));
        assert!(matches!(
            real_transpose.backend.as_deref(),
            Some(SparseBackend::RealBall(_))
        ));
        assert!(matches!(
            real_product.backend.as_deref(),
            Some(SparseBackend::RealBall(_))
        ));
        assert!(matches!(
            real_sum.backend.as_deref(),
            Some(SparseBackend::RealBall(_))
        ));
        assert_eq!(real_difference, real_matrix);
        assert_eq!(
            real_product.to_dense(),
            real_matrix
                .to_dense()
                .matmul(&real_transpose.to_dense())
                .unwrap()
        );

        let complex = ComplexBallField::new(128).unwrap();
        let complex_matrix = SparseMatrix::from_rows(
            complex,
            [
                [complex.element_parts(1_i64, 1_i64), complex.element(0_i64)],
                [complex.element(2_i64), complex.element_parts(3_i64, -1_i64)],
            ],
        )
        .unwrap();
        let complex_transpose = complex_matrix.transpose();
        let complex_product = complex_matrix.matmul(&complex_transpose).unwrap();
        let complex_sum = complex_matrix.add(&complex_matrix).unwrap();
        let complex_difference = complex_sum.sub(&complex_matrix).unwrap();
        assert!(matches!(
            complex_matrix.backend.as_deref(),
            Some(SparseBackend::ComplexBall(_))
        ));
        assert!(matches!(
            complex_transpose.backend.as_deref(),
            Some(SparseBackend::ComplexBall(_))
        ));
        assert!(matches!(
            complex_product.backend.as_deref(),
            Some(SparseBackend::ComplexBall(_))
        ));
        assert!(matches!(
            complex_sum.backend.as_deref(),
            Some(SparseBackend::ComplexBall(_))
        ));
        assert_eq!(complex_difference, complex_matrix);
        assert_eq!(
            complex_product.to_dense(),
            complex_matrix
                .to_dense()
                .matmul(&complex_transpose.to_dense())
                .unwrap()
        );
    }

    #[test]
    fn binary_sparse_storage_stays_in_mzs_and_matches_the_generic_floor() {
        let field = FiniteField::prime(2_u8).unwrap();
        let one = field.one();
        let zero = field.zero();
        let left = SparseMatrix::from_rows(
            field.clone(),
            [
                [one.clone(), one.clone(), zero.clone()],
                [zero.clone(), one.clone(), zero.clone()],
            ],
        )
        .unwrap();
        let right = SparseMatrix::from_rows(
            field.clone(),
            [
                [one.clone(), zero.clone()],
                [one.clone(), one.clone()],
                [zero.clone(), one.clone()],
            ],
        )
        .unwrap();

        let routed = left.matmul(&right).unwrap();
        let transposed = left.transpose();
        let sum = left.add(&left).unwrap();
        let difference = sum.sub(&left).unwrap();
        let negative = left.neg().unwrap();
        let scaled = left.scalar_mul(&one).unwrap();
        for matrix in [
            &left,
            &right,
            &routed,
            &transposed,
            &sum,
            &difference,
            &negative,
            &scaled,
        ] {
            assert!(matches!(
                matrix.backend.as_deref(),
                Some(SparseBackend::Binary(_))
            ));
            assert!(matrix.canonical.get().is_none());
        }
        let generic = left.generic_matmul(&right).unwrap();
        assert_eq!(routed, generic);

        let tiny_left = SparseMatrix::from_rows(field.clone(), [[one.clone()]]).unwrap();
        let tiny_right = SparseMatrix::from_rows(field, [[one]]).unwrap();
        let tiny_product = tiny_left.matmul(&tiny_right).unwrap();
        assert!(matches!(
            tiny_product.backend.as_deref(),
            Some(SparseBackend::Binary(_))
        ));
        assert_eq!(tiny_product, tiny_left.generic_matmul(&tiny_right).unwrap());
    }

    #[test]
    fn generic_row_monomial_product_matches_dense_polynomial_arithmetic() {
        let ring = PolynomialRing::new(crate::Q, ["x"], MonomialOrder::GradedReverseLexicographic)
            .unwrap();
        let x = ring.generator(0).unwrap();
        let left = SparseMatrix::from_triplets(
            ring.clone(),
            3,
            3,
            [(0, 2, ring.one()), (1, 0, x.clone()), (2, 1, ring.one())],
        )
        .unwrap();
        let right = SparseMatrix::from_triplets(
            ring,
            3,
            2,
            [(0, 0, x.clone()), (1, 1, x.clone()), (2, 0, x)],
        )
        .unwrap();

        let product = left.generic_matmul(&right).unwrap();
        assert_eq!(
            product.to_dense(),
            left.to_dense().matmul(&right.to_dense()).unwrap()
        );
    }

    #[test]
    fn polynomial_sparse_accumulation_cancels_and_applies_exactly() {
        let ring = PolynomialRing::new(
            crate::Q,
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let left = SparseMatrix::from_rows(
            ring.clone(),
            [[x.clone(), x.clone()], [y.clone(), ring.one()]],
        )
        .unwrap();
        let right =
            SparseMatrix::from_rows(ring.clone(), [[x.clone()], [ring.neg(&x).unwrap()]]).unwrap();
        let vector = Vector::from_values(ring.clone(), [y.clone(), x.clone()]).unwrap();

        let product = left.generic_matmul(&right).unwrap();
        assert_eq!(product.nnz(), 1);
        assert_eq!(product.get(0, 0).unwrap(), ring.zero());
        assert_eq!(
            product.to_dense(),
            left.to_dense().matmul(&right.to_dense()).unwrap()
        );
        assert_eq!(
            left.apply(&vector).unwrap().values().collect::<Vec<_>>(),
            [
                ring.add(&ring.mul(&x, &y).unwrap(), &ring.mul(&x, &x).unwrap())
                    .unwrap(),
                ring.add(&ring.mul(&y, &y).unwrap(), &x).unwrap(),
            ]
        );
    }

    #[test]
    fn polynomial_sparse_products_keep_the_arbitrary_exponent_floor() {
        let ring = PolynomialRing::new(crate::Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let exponent = Integer::from(u128::from(u64::MAX) + 1);
        let huge = ring
            .from_terms([(vec![exponent], Integer::from(3))])
            .unwrap();
        let left = SparseMatrix::from_rows(ring.clone(), [[huge.clone(), ring.one()]]).unwrap();
        let right = SparseMatrix::from_rows(ring.clone(), [[ring.one()], [huge.clone()]]).unwrap();
        let vector = Vector::from_values(ring.clone(), [ring.one(), huge.clone()]).unwrap();
        let expected = ring.add(&huge, &huge).unwrap();

        assert_eq!(left.matmul(&right).unwrap().get(0, 0).unwrap(), expected);
        assert_eq!(left.apply(&vector).unwrap().get(0).unwrap(), expected);
    }
}
