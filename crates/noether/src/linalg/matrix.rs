// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Linear algebra over explicit coefficient contexts.
//!
//! [`Matrix<R>`] is parameterized by the coefficient context `R`, not by a
//! backend. Every [`Ring`] receives the generic dense implementation. The
//! canonical exact scalar, modular, finite-field, p-adic, and ball contexts
//! use private whole-matrix backends while retaining exactly the same public
//! type and semantics.

use core::any::{Any, TypeId};
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
#[cfg(feature = "distributed")]
use m4rst::Mzd;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::{Arc, OnceLock};

use crate::{
    commalg::{MonomialOrder, Polynomial, PolynomialRing, ResidueClass, ResidueRing},
    numthy::{FiniteField, FiniteFieldElement, PadicElement, PadicField, PadicRing},
    CoefficientMap, CoefficientMappingError, CommutativeRing, ComplexBall, ComplexBallField,
    ContextMismatchError, Integer, IntegerRing, InversionError, NoetherError, Rational,
    RationalField, RealBall, RealBallField, Ring, SemanticType, TypeDescription, VerificationError,
};

use super::flint::{
    ComplexBallMatrix, FiniteFieldMatrix, IntegerMatrix, ModularMatrix, PadicMatrix,
    PadicMatrixContext, RationalMatrix, RealBallMatrix,
};

static MATRIX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/Matrix@1",
    rust_path: "noether::linalg::Matrix",
    python_path: Some("noether.linalg.Matrix"),
};

/// Failure to construct or apply a linear-algebra object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinalgError {
    /// A row had a different length from the first row.
    RaggedRows {
        /// Zero-based row index.
        row: usize,
        /// Required number of columns.
        expected: usize,
        /// Received number of columns.
        actual: usize,
    },
    /// Matrix dimensions do not satisfy an operation's precondition.
    ShapeMismatch {
        /// Stable operation name.
        operation: &'static str,
        /// Left-hand shape.
        left: (usize, usize),
        /// Right-hand shape.
        right: (usize, usize),
    },
    /// Vector dimensions do not satisfy an operation's precondition.
    DimensionMismatch {
        /// Stable operation name.
        operation: &'static str,
        /// Left-hand dimension.
        left: usize,
        /// Right-hand dimension.
        right: usize,
    },
    /// The requested shape cannot be represented safely.
    ShapeTooLarge {
        /// Requested number of rows.
        rows: usize,
        /// Requested number of columns.
        cols: usize,
    },
    /// A coefficient did not belong to the matrix's ring context.
    ElementOutsideContext {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
    },
    /// A sparse entry was outside its declared matrix shape.
    SparseEntryOutsideShape {
        /// Zero-based row index.
        row: usize,
        /// Zero-based column index.
        col: usize,
        /// Declared sparse-matrix shape.
        shape: (usize, usize),
    },
    /// CSR arrays did not encode a canonical sparse matrix.
    InvalidSparseStructure {
        /// Stable explanation of the failed structural invariant.
        message: String,
    },
    /// Two values carried unequal coefficient contexts.
    ContextMismatch(ContextMismatchError),
    /// A field pivot unexpectedly failed inversion.
    Inversion(InversionError),
    /// A constructed linear-algebra witness did not verify.
    Verification(VerificationError),
    /// An operation required a square matrix.
    NotSquare {
        /// Requested operation.
        operation: &'static str,
        /// Actual matrix shape.
        shape: (usize, usize),
    },
    /// A linear system has no solution.
    InconsistentSystem,
    /// A smooth-torsion Smith strategy needs a prime bound of at least two.
    InvalidSmoothBound {
        /// Received inclusive bound.
        bound: u64,
    },
    /// A requested smooth-prime sweep exceeds the public bounded-work limit.
    SmoothBoundTooLarge {
        /// Received inclusive bound.
        bound: u64,
        /// Largest accepted inclusive bound.
        maximum: u64,
    },
    /// A local invariant needs more p-adic precision than word storage permits.
    SmoothValuationTooLarge {
        /// Prime whose local valuation did not fit.
        prime: u64,
        /// Largest modulus exponent checked.
        maximum_exponent: u32,
    },
    /// A residue-ring matrix operation requires a word-sized modulus.
    ResidueModulusTooLarge {
        /// Arbitrary-precision modulus that cannot use the FLINT `nmod` path.
        modulus: Integer,
    },
    /// A canonical polynomial result could not be constructed.
    PolynomialConstruction {
        /// Underlying checked polynomial error.
        message: String,
    },
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RaggedRows {
                row,
                expected,
                actual,
            } => write!(f, "row {row} has {actual} columns; expected {expected}"),
            Self::ShapeMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "shape mismatch for {operation}: left is {}x{}, right is {}x{}",
                left.0, left.1, right.0, right.1
            ),
            Self::DimensionMismatch {
                operation,
                left,
                right,
            } => write!(
                f,
                "dimension mismatch for {operation}: left is {left}, right is {right}"
            ),
            Self::ShapeTooLarge { rows, cols } => {
                write!(f, "matrix shape {rows}x{cols} is too large")
            }
            Self::ElementOutsideContext { row, col } => {
                write!(f, "matrix entry ({row}, {col}) is outside its ring context")
            }
            Self::SparseEntryOutsideShape { row, col, shape } => write!(
                f,
                "sparse entry ({row}, {col}) is outside shape {}x{}",
                shape.0, shape.1
            ),
            Self::InvalidSparseStructure { message } => {
                write!(f, "invalid sparse matrix structure: {message}")
            }
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Inversion(error) => error.fmt(f),
            Self::Verification(error) => error.fmt(f),
            Self::NotSquare { operation, shape } => write!(
                f,
                "{operation} requires a square matrix, got {}x{}",
                shape.0, shape.1
            ),
            Self::InconsistentSystem => f.write_str("linear system is inconsistent"),
            Self::InvalidSmoothBound { bound } => {
                write!(f, "smooth_bound must be at least 2, got {bound}")
            }
            Self::SmoothBoundTooLarge { bound, maximum } => write!(
                f,
                "smooth_bound {bound} exceeds the bounded-work maximum {maximum}"
            ),
            Self::SmoothValuationTooLarge {
                prime,
                maximum_exponent,
            } => write!(
                f,
                "the {prime}-adic Smith valuation reaches the word-precision cap {maximum_exponent}"
            ),
            Self::ResidueModulusTooLarge { modulus } => write!(
                f,
                "Howell normal form currently requires a word-sized residue modulus; received {modulus}"
            ),
            Self::PolynomialConstruction { message } => {
                write!(f, "could not construct the matrix polynomial result: {message}")
            }
        }
    }
}

impl std::error::Error for LinalgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Inversion(error) => Some(error),
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for LinalgError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<InversionError> for LinalgError {
    fn from(value: InversionError) -> Self {
        Self::Inversion(value)
    }
}

impl From<VerificationError> for LinalgError {
    fn from(value: VerificationError) -> Self {
        Self::Verification(value)
    }
}

/// A matrix index was outside the matrix shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexError {
    /// Requested zero-based row.
    pub row: usize,
    /// Requested zero-based column.
    pub col: usize,
    /// Matrix shape at the time of access.
    pub shape: (usize, usize),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "matrix index ({}, {}) is outside shape {}x{}",
            self.row, self.col, self.shape.0, self.shape.1
        )
    }
}

impl std::error::Error for IndexError {}

enum MatrixStorage<E> {
    Dense(Arc<Vec<E>>),
    Integer(IntegerMatrix),
    Rational(RationalMatrix),
    Residue(ModularMatrix),
    FiniteField(FiniteFieldMatrix),
    Padic(PadicMatrix),
    RealBall(RealBallMatrix),
    ComplexBall(ComplexBallMatrix),
}

impl<E> MatrixStorage<E> {
    fn dense(entries: Vec<E>) -> Self {
        Self::Dense(Arc::new(entries))
    }
}

impl<E: Clone> Clone for MatrixStorage<E> {
    fn clone(&self) -> Self {
        match self {
            Self::Dense(entries) => Self::Dense(Arc::clone(entries)),
            Self::Integer(matrix) => Self::Integer(matrix.clone()),
            Self::Rational(matrix) => Self::Rational(matrix.clone()),
            Self::Residue(matrix) => Self::Residue(matrix.clone()),
            Self::FiniteField(matrix) => Self::FiniteField(matrix.clone()),
            Self::Padic(matrix) => Self::Padic(matrix.clone()),
            Self::RealBall(matrix) => Self::RealBall(matrix.clone()),
            Self::ComplexBall(matrix) => Self::ComplexBall(matrix.clone()),
        }
    }
}

impl<E: Clone + 'static> MatrixStorage<E> {
    fn get(&self, row: usize, col: usize, cols: usize) -> E {
        match self {
            Self::Dense(entries) => entries[row * cols + col].clone(),
            Self::Integer(matrix) => cast_owned(matrix.get(row, col)),
            Self::Rational(matrix) => cast_owned(matrix.get(row, col)),
            Self::Residue(matrix) => cast_owned(matrix.get(row, col)),
            Self::FiniteField(matrix) => cast_owned(matrix.get(row, col)),
            Self::Padic(matrix) => cast_owned(matrix.get(row, col)),
            Self::RealBall(matrix) => cast_owned(matrix.get(row, col)),
            Self::ComplexBall(matrix) => cast_owned(matrix.get(row, col)),
        }
    }

    fn materialized_entries(&self, rows: usize, cols: usize) -> Vec<E> {
        match self {
            Self::Dense(entries) => entries.as_ref().clone(),
            Self::Residue(matrix) => {
                cast_owned::<Vec<ResidueClass>, Vec<E>>(matrix.materialized_values())
            }
            Self::FiniteField(matrix) => {
                cast_owned::<Vec<FiniteFieldElement>, Vec<E>>(matrix.materialized_values())
            }
            _ => (0..rows * cols)
                .map(|index| self.get(index / cols, index % cols, cols))
                .collect(),
        }
    }

    fn materialized_rows(&self, rows: usize, cols: usize) -> Vec<Vec<E>> {
        if cols == 0 {
            return vec![Vec::new(); rows];
        }
        let entries = self.materialized_entries(rows, cols);
        let mut entries = entries.into_iter();
        (0..rows)
            .map(|_| entries.by_ref().take(cols).collect())
            .collect()
    }
}

/// An immutable dense matrix over one explicit coefficient context.
///
/// `R` is the ring itself: `Matrix<IntegerRing>` is a matrix over `Z`, while
/// `Matrix<FiniteField>` carries the particular finite field.
/// Storage and implementation choices are private and do not change this
/// mathematical identity.
///
/// # Examples
///
/// ```
/// use noether::linalg::Matrix;
/// use noether::{Integer, Z};
///
/// let a = Matrix::from_rows(
///     Z,
///     [
///         [Integer::from(1), Integer::from(2)],
///         [Integer::from(3), Integer::from(4)],
///     ],
/// )?;
/// let identity = Matrix::identity(Z, 2);
/// assert_eq!(a.matmul(&identity)?, a);
///
/// let doubled = (&a + &a)?;
/// assert_eq!(doubled.get(1, 1)?, Integer::from(8));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Matrix<R: Ring> {
    ring: Arc<R>,
    rows: usize,
    cols: usize,
    storage: Arc<MatrixStorage<R::Element>>,
    rank: Arc<OnceLock<usize>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, E: Deserialize<'de>")
)]
struct MatrixWire<R: Ring, E> {
    #[serde(rename = "type")]
    type_id: String,
    ring: R,
    shape: [usize; 2],
    entries: Vec<E>,
}

impl<R: Ring> Matrix<R> {
    /// Construct a matrix from row-major coefficients.
    ///
    /// An empty outer vector constructs the `0 x 0` matrix. Use [`zeros`](Self::zeros)
    /// when a zero-row matrix with a nonzero column count is required.
    pub fn from_rows<I, J>(ring: R, rows: I) -> Result<Self, LinalgError>
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = R::Element>,
    {
        let rows: Vec<Vec<R::Element>> = rows
            .into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);
        for (index, row) in rows.iter().enumerate().skip(1) {
            if row.len() != col_count {
                return Err(LinalgError::RaggedRows {
                    row: index,
                    expected: col_count,
                    actual: row.len(),
                });
            }
        }
        validate_shape(row_count, col_count)?;
        Self::from_entries(
            ring,
            row_count,
            col_count,
            rows.into_iter().flatten().collect(),
        )
    }

    /// Construct the `rows x cols` zero matrix.
    pub fn zeros(ring: R, rows: usize, cols: usize) -> Self {
        let length = validate_shape(rows, cols).expect("matrix shape exceeds supported dimensions");
        if is_integer_context::<R>() {
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::Integer(IntegerMatrix::zeros(rows, cols)),
            );
        }
        if is_rational_context::<R>() {
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::Rational(RationalMatrix::zeros(rows, cols)),
            );
        }
        if is_residue_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<ResidueRing>()
                .expect("residue matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::Residue(ModularMatrix::zeros(context, rows, cols)),
            );
        }
        if is_finite_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<FiniteField>()
                .expect("finite-field matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::FiniteField(FiniteFieldMatrix::zeros(context, rows, cols)),
            );
        }
        if is_padic_ring_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicRing>()
                .expect("p-adic ring matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::Padic(PadicMatrix::zeros(
                    PadicMatrixContext::Ring(context),
                    rows,
                    cols,
                )),
            );
        }
        if is_padic_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicField>()
                .expect("p-adic field matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::Padic(PadicMatrix::zeros(
                    PadicMatrixContext::Field(context),
                    rows,
                    cols,
                )),
            );
        }
        if is_real_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<RealBallField>()
                .expect("real-ball matrix context invariant");
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::RealBall(RealBallMatrix::zeros(context, rows, cols)),
            );
        }
        if is_complex_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<ComplexBallField>()
                .expect("complex-ball matrix context invariant");
            return Self::from_storage(
                Arc::new(ring),
                rows,
                cols,
                MatrixStorage::ComplexBall(ComplexBallMatrix::zeros(context, rows, cols)),
            );
        }
        let zero = ring.zero();
        Self::from_entries(ring, rows, cols, vec![zero; length])
            .expect("a ring's zero belongs to that ring")
    }

    /// Construct the square identity matrix of the requested size.
    pub fn identity(ring: R, size: usize) -> Self {
        let length = validate_shape(size, size).expect("matrix shape exceeds supported dimensions");
        if is_integer_context::<R>() {
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::Integer(IntegerMatrix::identity(size)),
            );
        }
        if is_rational_context::<R>() {
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::Rational(RationalMatrix::identity(size)),
            );
        }
        if is_residue_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<ResidueRing>()
                .expect("residue matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::Residue(ModularMatrix::identity(context, size)),
            );
        }
        if is_finite_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<FiniteField>()
                .expect("finite-field matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::FiniteField(FiniteFieldMatrix::identity(context, size)),
            );
        }
        if is_padic_ring_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicRing>()
                .expect("p-adic ring matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::Padic(PadicMatrix::identity(
                    PadicMatrixContext::Ring(context),
                    size,
                )),
            );
        }
        if is_padic_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicField>()
                .expect("p-adic field matrix context invariant")
                .clone();
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::Padic(PadicMatrix::identity(
                    PadicMatrixContext::Field(context),
                    size,
                )),
            );
        }
        if is_real_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<RealBallField>()
                .expect("real-ball matrix context invariant");
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::RealBall(RealBallMatrix::identity(context, size)),
            );
        }
        if is_complex_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<ComplexBallField>()
                .expect("complex-ball matrix context invariant");
            return Self::from_storage(
                Arc::new(ring),
                size,
                size,
                MatrixStorage::ComplexBall(ComplexBallMatrix::identity(context, size)),
            );
        }
        let mut entries = vec![ring.zero(); length];
        for index in 0..size {
            entries[index * size + index] = ring.one();
        }
        Self::from_entries(ring, size, size, entries)
            .expect("a ring's zero and one belong to that ring")
    }

    /// The exact coefficient context.
    pub fn ring(&self) -> &R {
        self.ring.as_ref()
    }

    /// `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[cfg(test)]
    fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Materialize one coefficient, failing when the index is out of bounds.
    ///
    /// Backend-native matrices materialize only the requested scalar.
    pub fn get(&self, row: usize, col: usize) -> Result<R::Element, IndexError> {
        if row < self.rows && col < self.cols {
            Ok(self.storage.get(row, col, self.cols))
        } else {
            Err(IndexError {
                row,
                col,
                shape: self.shape(),
            })
        }
    }

    /// Iterate over materialized rows in deterministic order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = Vec<R::Element>> + '_ {
        (0..self.rows).map(|row| {
            (0..self.cols)
                .map(|col| self.storage.get(row, col, self.cols))
                .collect()
        })
    }

    pub(crate) fn into_entries(self) -> Vec<R::Element> {
        let Self {
            rows,
            cols,
            storage,
            ..
        } = self;
        match Arc::try_unwrap(storage) {
            Ok(MatrixStorage::Dense(entries)) => match Arc::try_unwrap(entries) {
                Ok(entries) => entries,
                Err(entries) => entries.as_ref().clone(),
            },
            Ok(storage) => (0..rows)
                .flat_map(|row| {
                    let storage = &storage;
                    (0..cols).map(move |col| storage.get(row, col, cols))
                })
                .collect(),
            Err(storage) => (0..rows)
                .flat_map(|row| {
                    let storage = storage.as_ref();
                    (0..cols).map(move |col| storage.get(row, col, cols))
                })
                .collect(),
        }
    }

    /// Sum the diagonal entries of a square matrix.
    pub fn trace(&self) -> Result<R::Element, LinalgError> {
        if self.rows != self.cols {
            return Err(LinalgError::NotSquare {
                operation: "matrix trace",
                shape: self.shape(),
            });
        }
        let mut trace = self.ring.zero();
        for index in 0..self.rows {
            trace = self
                .ring
                .add(&trace, &self.storage.get(index, index, self.cols))?;
        }
        Ok(trace)
    }

    /// Apply one checked scalar map entrywise while preserving the exact
    /// matrix shape, including zero-dimensional rectangular shapes.
    pub fn map_coefficients<M>(&self, map: &M) -> Result<Matrix<M::Target>, CoefficientMappingError>
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
        if self.rows == 0 || self.cols == 0 {
            return Ok(Matrix::zeros(map.target().clone(), self.rows, self.cols));
        }
        let mut rows = Vec::with_capacity(self.rows);
        for row in self.rows() {
            let mut mapped = Vec::with_capacity(self.cols);
            for coefficient in row {
                mapped
                    .push(map.apply(&coefficient).map_err(|error| {
                        CoefficientMappingError::Coefficient(error.to_string())
                    })?);
            }
            rows.push(mapped);
        }
        Matrix::from_rows(map.target().clone(), rows)
            .map_err(|error| CoefficientMappingError::Value(error.to_string()))
    }

    /// Materialize all rows in deterministic order.
    ///
    /// Unlike [`rows`](Self::rows), this is an eager terminal observation.
    /// Device-resident packed matrices download at most once before scalar
    /// conversion.
    pub fn to_rows(&self) -> Vec<Vec<R::Element>> {
        self.storage.materialized_rows(self.rows, self.cols)
    }

    pub(crate) fn materialized_entries(&self) -> Vec<R::Element> {
        self.storage.materialized_entries(self.rows, self.cols)
    }

    /// Serialize the complete coefficient context, shape, and row-major entries.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a matrix and its complete coefficient context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Return the transpose.
    pub fn transpose(&self) -> Self {
        let storage = match self.storage.as_ref() {
            MatrixStorage::Dense(_) => {
                let mut entries = Vec::with_capacity(self.rows * self.cols);
                for col in 0..self.cols {
                    for row in 0..self.rows {
                        entries.push(self.storage.get(row, col, self.cols));
                    }
                }
                MatrixStorage::dense(entries)
            }
            MatrixStorage::Integer(matrix) => MatrixStorage::Integer(matrix.transpose()),
            MatrixStorage::Rational(matrix) => MatrixStorage::Rational(matrix.transpose()),
            MatrixStorage::Residue(matrix) => MatrixStorage::Residue(matrix.transpose()),
            MatrixStorage::FiniteField(matrix) => MatrixStorage::FiniteField(matrix.transpose()),
            MatrixStorage::Padic(matrix) => MatrixStorage::Padic(matrix.transpose()),
            MatrixStorage::RealBall(matrix) => MatrixStorage::RealBall(matrix.transpose()),
            MatrixStorage::ComplexBall(matrix) => MatrixStorage::ComplexBall(matrix.transpose()),
        };
        Self::from_storage(self.ring.clone(), self.cols, self.rows, storage)
    }

    /// Add matrices with equal contexts and shapes.
    pub fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        self.require_same_shape(rhs, "matrix addition")?;
        let storage = match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (MatrixStorage::Integer(left), MatrixStorage::Integer(right)) => {
                MatrixStorage::Integer(left.add(right))
            }
            (MatrixStorage::Rational(left), MatrixStorage::Rational(right)) => {
                MatrixStorage::Rational(left.add(right))
            }
            (MatrixStorage::Residue(left), MatrixStorage::Residue(right)) => {
                MatrixStorage::Residue(left.add(right)?)
            }
            (MatrixStorage::FiniteField(left), MatrixStorage::FiniteField(right)) => {
                MatrixStorage::FiniteField(left.add(right)?)
            }
            (MatrixStorage::Padic(left), MatrixStorage::Padic(right)) => {
                MatrixStorage::Padic(left.add(right))
            }
            (MatrixStorage::RealBall(left), MatrixStorage::RealBall(right)) => {
                MatrixStorage::RealBall(left.add(right))
            }
            (MatrixStorage::ComplexBall(left), MatrixStorage::ComplexBall(right)) => {
                MatrixStorage::ComplexBall(left.add(right))
            }
            _ => {
                let mut entries = Vec::with_capacity(self.rows * self.cols);
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        entries.push(self.ring.add(
                            &self.storage.get(row, col, self.cols),
                            &rhs.storage.get(row, col, rhs.cols),
                        )?);
                    }
                }
                MatrixStorage::dense(entries)
            }
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            storage,
        ))
    }

    /// Subtract matrices with equal contexts and shapes.
    pub fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        self.require_same_shape(rhs, "matrix subtraction")?;
        let storage = match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (MatrixStorage::Integer(left), MatrixStorage::Integer(right)) => {
                MatrixStorage::Integer(left.sub(right))
            }
            (MatrixStorage::Rational(left), MatrixStorage::Rational(right)) => {
                MatrixStorage::Rational(left.sub(right))
            }
            (MatrixStorage::Residue(left), MatrixStorage::Residue(right)) => {
                MatrixStorage::Residue(left.sub(right)?)
            }
            (MatrixStorage::FiniteField(left), MatrixStorage::FiniteField(right)) => {
                MatrixStorage::FiniteField(left.sub(right)?)
            }
            (MatrixStorage::Padic(left), MatrixStorage::Padic(right)) => {
                MatrixStorage::Padic(left.sub(right))
            }
            (MatrixStorage::RealBall(left), MatrixStorage::RealBall(right)) => {
                MatrixStorage::RealBall(left.sub(right))
            }
            (MatrixStorage::ComplexBall(left), MatrixStorage::ComplexBall(right)) => {
                MatrixStorage::ComplexBall(left.sub(right))
            }
            _ => {
                let mut entries = Vec::with_capacity(self.rows * self.cols);
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        entries.push(self.ring.sub(
                            &self.storage.get(row, col, self.cols),
                            &rhs.storage.get(row, col, rhs.cols),
                        )?);
                    }
                }
                MatrixStorage::dense(entries)
            }
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            storage,
        ))
    }

    /// Negate every coefficient.
    pub fn neg(&self) -> Result<Self, LinalgError> {
        let storage = match self.storage.as_ref() {
            MatrixStorage::Integer(matrix) => MatrixStorage::Integer(matrix.neg()),
            MatrixStorage::Rational(matrix) => MatrixStorage::Rational(matrix.neg()),
            MatrixStorage::Residue(matrix) => MatrixStorage::Residue(matrix.neg()),
            MatrixStorage::FiniteField(matrix) => MatrixStorage::FiniteField(matrix.neg()),
            MatrixStorage::Padic(matrix) => MatrixStorage::Padic(matrix.neg()),
            MatrixStorage::RealBall(matrix) => MatrixStorage::RealBall(matrix.neg()),
            MatrixStorage::ComplexBall(matrix) => MatrixStorage::ComplexBall(matrix.neg()),
            MatrixStorage::Dense(_) => {
                let mut entries = Vec::with_capacity(self.rows * self.cols);
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        entries.push(self.ring.neg(&self.storage.get(row, col, self.cols))?);
                    }
                }
                MatrixStorage::dense(entries)
            }
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            storage,
        ))
    }

    /// Multiply every coefficient by a scalar in this context.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, LinalgError> {
        if !self.ring.contains(scalar) {
            return Err(LinalgError::ContextMismatch(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "scalar belongs to another context".to_owned(),
            }));
        }

        let storage = match self.storage.as_ref() {
            MatrixStorage::Integer(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<Integer>()
                    .expect("integer matrix storage invariant");
                MatrixStorage::Integer(matrix.scalar_mul(scalar))
            }
            MatrixStorage::Rational(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<Rational>()
                    .expect("rational matrix storage invariant");
                MatrixStorage::Rational(matrix.scalar_mul(scalar))
            }
            MatrixStorage::Residue(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<ResidueClass>()
                    .expect("residue matrix storage invariant");
                MatrixStorage::Residue(matrix.scalar_mul(scalar)?)
            }
            MatrixStorage::FiniteField(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<FiniteFieldElement>()
                    .expect("finite-field matrix storage invariant");
                MatrixStorage::FiniteField(matrix.scalar_mul(scalar)?)
            }
            MatrixStorage::Padic(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<PadicElement>()
                    .expect("p-adic matrix storage invariant");
                MatrixStorage::Padic(matrix.scalar_mul(scalar))
            }
            MatrixStorage::RealBall(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<RealBall>()
                    .expect("real-ball matrix storage invariant");
                MatrixStorage::RealBall(matrix.scalar_mul(scalar))
            }
            MatrixStorage::ComplexBall(matrix) => {
                let scalar = (scalar as &dyn Any)
                    .downcast_ref::<ComplexBall>()
                    .expect("complex-ball matrix storage invariant");
                MatrixStorage::ComplexBall(matrix.scalar_mul(scalar))
            }
            MatrixStorage::Dense(_) => {
                let mut entries = Vec::with_capacity(self.rows * self.cols);
                for row in 0..self.rows {
                    for col in 0..self.cols {
                        entries.push(
                            self.ring
                                .mul(&self.storage.get(row, col, self.cols), scalar)?,
                        );
                    }
                }
                MatrixStorage::dense(entries)
            }
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            storage,
        ))
    }

    /// Compose matrices by exact matrix multiplication.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_context(rhs)?;
        if self.cols != rhs.rows {
            return Err(LinalgError::ShapeMismatch {
                operation: "matrix multiplication",
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        if self.rows == 0 || self.cols == 0 || rhs.cols == 0 {
            return Ok(Self::zeros(self.ring.as_ref().clone(), self.rows, rhs.cols));
        }

        let storage = match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (MatrixStorage::Integer(left), MatrixStorage::Integer(right)) => {
                MatrixStorage::Integer(left.matmul(right))
            }
            (MatrixStorage::Rational(left), MatrixStorage::Rational(right)) => {
                MatrixStorage::Rational(left.matmul(right))
            }
            (MatrixStorage::Residue(left), MatrixStorage::Residue(right)) => {
                MatrixStorage::Residue(left.matmul(right)?)
            }
            (MatrixStorage::FiniteField(left), MatrixStorage::FiniteField(right)) => {
                MatrixStorage::FiniteField(left.matmul(right)?)
            }
            (MatrixStorage::Padic(left), MatrixStorage::Padic(right)) => {
                MatrixStorage::Padic(left.matmul(right))
            }
            (MatrixStorage::RealBall(left), MatrixStorage::RealBall(right)) => {
                MatrixStorage::RealBall(left.matmul(right))
            }
            (MatrixStorage::ComplexBall(left), MatrixStorage::ComplexBall(right)) => {
                MatrixStorage::ComplexBall(left.matmul(right))
            }
            _ => {
                let left = self.rows().flatten().collect::<Vec<_>>();
                let right = rhs.rows().flatten().collect::<Vec<_>>();
                MatrixStorage::dense(super::generic_matmul::matmul(
                    self.ring(),
                    &left,
                    &right,
                    self.rows,
                    self.cols,
                    rhs.cols,
                )?)
            }
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            self.rows,
            rhs.cols,
            storage,
        ))
    }

    pub(super) fn from_entries(
        ring: R,
        rows: usize,
        cols: usize,
        entries: Vec<R::Element>,
    ) -> Result<Self, LinalgError> {
        debug_assert_eq!(entries.len(), rows * cols);
        for (index, entry) in entries.iter().enumerate() {
            if !ring.contains(entry) {
                return Err(LinalgError::ElementOutsideContext {
                    row: index / cols,
                    col: index % cols,
                });
            }
        }

        Ok(Self::from_validated_entries(ring, rows, cols, entries))
    }

    pub(super) fn from_validated_entries(
        ring: R,
        rows: usize,
        cols: usize,
        entries: Vec<R::Element>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows.saturating_mul(cols));
        let storage = if is_integer_context::<R>() {
            let integer_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<Integer>()
                    .expect("integer context element invariant")
            });
            MatrixStorage::Integer(IntegerMatrix::from_entries(rows, cols, integer_entries))
        } else if is_rational_context::<R>() {
            let rational_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<Rational>()
                    .expect("rational context element invariant")
            });
            MatrixStorage::Rational(RationalMatrix::from_entries(rows, cols, rational_entries))
        } else if is_residue_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<ResidueRing>()
                .expect("residue matrix context invariant")
                .clone();
            let residue_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<ResidueClass>()
                    .expect("residue context element invariant")
            });
            MatrixStorage::Residue(ModularMatrix::from_entries(
                context,
                rows,
                cols,
                residue_entries,
            ))
        } else if is_finite_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<FiniteField>()
                .expect("finite-field matrix context invariant")
                .clone();
            let field_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<FiniteFieldElement>()
                    .expect("finite-field context element invariant")
            });
            MatrixStorage::FiniteField(FiniteFieldMatrix::from_entries(
                context,
                rows,
                cols,
                field_entries,
            ))
        } else if is_padic_ring_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicRing>()
                .expect("p-adic ring matrix context invariant")
                .clone();
            let padic_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<PadicElement>()
                    .expect("p-adic ring context element invariant")
            });
            MatrixStorage::Padic(PadicMatrix::from_entries(
                PadicMatrixContext::Ring(context),
                rows,
                cols,
                padic_entries,
            ))
        } else if is_padic_field_context::<R>() {
            let context = (&ring as &dyn Any)
                .downcast_ref::<PadicField>()
                .expect("p-adic field matrix context invariant")
                .clone();
            let padic_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<PadicElement>()
                    .expect("p-adic field context element invariant")
            });
            MatrixStorage::Padic(PadicMatrix::from_entries(
                PadicMatrixContext::Field(context),
                rows,
                cols,
                padic_entries,
            ))
        } else if is_real_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<RealBallField>()
                .expect("real-ball matrix context invariant");
            let ball_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<RealBall>()
                    .expect("real-ball context element invariant")
            });
            MatrixStorage::RealBall(RealBallMatrix::from_entries(
                context,
                rows,
                cols,
                ball_entries,
            ))
        } else if is_complex_ball_context::<R>() {
            let context = *(&ring as &dyn Any)
                .downcast_ref::<ComplexBallField>()
                .expect("complex-ball matrix context invariant");
            let ball_entries = entries.iter().map(|entry| {
                (entry as &dyn Any)
                    .downcast_ref::<ComplexBall>()
                    .expect("complex-ball context element invariant")
            });
            MatrixStorage::ComplexBall(ComplexBallMatrix::from_entries(
                context,
                rows,
                cols,
                ball_entries,
            ))
        } else {
            MatrixStorage::dense(entries)
        };

        Self::from_storage(Arc::new(ring), rows, cols, storage)
    }

    pub(crate) fn from_validated_columns(
        ring: R,
        rows: usize,
        columns: Vec<Vec<R::Element>>,
    ) -> Self {
        let cols = columns.len();
        debug_assert!(columns.iter().all(|column| column.len() == rows));
        if rows == 0 {
            return Self::from_validated_entries(ring, 0, cols, Vec::new());
        }
        let mut columns = columns.into_iter().map(Vec::into_iter).collect::<Vec<_>>();
        let mut entries = Vec::with_capacity(rows.saturating_mul(cols));
        for _ in 0..rows {
            for column in &mut columns {
                entries.push(
                    column
                        .next()
                        .expect("validated matrix column length invariant"),
                );
            }
        }
        Self::from_validated_entries(ring, rows, cols, entries)
    }

    fn from_storage(
        ring: Arc<R>,
        rows: usize,
        cols: usize,
        storage: MatrixStorage<R::Element>,
    ) -> Self {
        debug_assert!(match &storage {
            MatrixStorage::Dense(entries) => entries.len() == rows * cols,
            MatrixStorage::Integer(_) => is_integer_context::<R>(),
            MatrixStorage::Rational(_) => is_rational_context::<R>(),
            MatrixStorage::Residue(_) => is_residue_context::<R>(),
            MatrixStorage::FiniteField(_) => is_finite_field_context::<R>(),
            MatrixStorage::Padic(_) => {
                is_padic_ring_context::<R>() || is_padic_field_context::<R>()
            }
            MatrixStorage::RealBall(_) => is_real_ball_context::<R>(),
            MatrixStorage::ComplexBall(_) => is_complex_ball_context::<R>(),
        });
        Self {
            ring,
            rows,
            cols,
            storage: Arc::new(storage),
            rank: Arc::new(OnceLock::new()),
        }
    }

    fn require_same_context(&self, rhs: &Self) -> Result<(), LinalgError> {
        if Arc::ptr_eq(&self.ring, &rhs.ring) || self.ring == rhs.ring {
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

    pub(super) fn field_rref_backend(&self) -> Option<Self> {
        let (storage, rank) = match self.storage.as_ref() {
            MatrixStorage::Rational(matrix) => {
                let (matrix, rank) = matrix.rref();
                (MatrixStorage::Rational(matrix), rank)
            }
            MatrixStorage::FiniteField(matrix) => {
                let (matrix, rank) = matrix.rref();
                (MatrixStorage::FiniteField(matrix), rank)
            }
            _ => return None,
        };
        self.cache_rank(rank);
        let output = Self::from_storage(self.ring.clone(), self.rows, self.cols, storage);
        output.cache_rank(rank);
        Some(output)
    }

    pub(super) fn field_rank_backend(&self) -> Option<usize> {
        match self.storage.as_ref() {
            MatrixStorage::Rational(matrix) => Some(matrix.rank()),
            MatrixStorage::FiniteField(matrix) => Some(matrix.rank()),
            _ => None,
        }
    }

    pub(super) fn cached_rank(&self) -> Option<usize> {
        self.rank.get().copied()
    }

    pub(super) fn cache_rank(&self, rank: usize) {
        let _ = self.rank.set(rank);
    }

    /// Return one exact rational solution using FLINT's rectangular modular
    /// solver. This private operation avoids constructing the full affine
    /// kernel when a higher-level certificate needs only one witness.
    pub(crate) fn rational_particular_solution(
        &self,
        rhs: &Self,
    ) -> Result<Option<Self>, LinalgError> {
        self.require_same_context(rhs)?;
        if self.rows != rhs.rows {
            return Err(LinalgError::ShapeMismatch {
                operation: "rational particular solve",
                left: self.shape(),
                right: rhs.shape(),
            });
        }
        let rhs_columns = rhs.cols;
        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (MatrixStorage::Rational(matrix), MatrixStorage::Rational(rhs_matrix)) => {
                Ok(matrix.particular_solution(rhs_matrix).map(|solution| {
                    Self::from_storage(
                        self.ring.clone(),
                        self.cols,
                        rhs_columns,
                        MatrixStorage::Rational(solution),
                    )
                }))
            }
            _ => Ok(None),
        }
    }

    /// Solve through a retained general rectangular field kernel when one is
    /// available without changing representation.
    ///
    /// The outer option means the storage has no such route. The inner option
    /// is the mathematical answer for an inconsistent system.
    pub(super) fn field_solve_backend(&self, rhs: &Self) -> Option<Option<Self>> {
        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (MatrixStorage::FiniteField(left), MatrixStorage::FiniteField(right)) => {
                left.solve_left(right).map(|solution| {
                    solution.map(|matrix| {
                        Self::from_storage(
                            self.ring.clone(),
                            self.cols,
                            rhs.cols,
                            MatrixStorage::FiniteField(matrix),
                        )
                    })
                })
            }
            _ => None,
        }
    }

    pub(super) fn field_nullspace_backend(&self) -> Option<Self> {
        let MatrixStorage::FiniteField(matrix) = self.storage.as_ref() else {
            return None;
        };
        let matrix = matrix.nullspace()?;
        let nullity = matrix.cols();
        Some(Self::from_storage(
            self.ring.clone(),
            self.cols,
            nullity,
            MatrixStorage::FiniteField(matrix),
        ))
    }

    pub(super) fn field_pluq_backend(&self) -> Option<(Self, Self, Self, Self, usize)> {
        let MatrixStorage::FiniteField(matrix) = self.storage.as_ref() else {
            return None;
        };
        let (p, l, u, q, rank) = matrix.pluq()?;
        let inner = self.rows.min(self.cols);
        Some((
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                self.rows,
                MatrixStorage::FiniteField(p),
            ),
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                inner,
                MatrixStorage::FiniteField(l),
            ),
            Self::from_storage(
                self.ring.clone(),
                inner,
                self.cols,
                MatrixStorage::FiniteField(u),
            ),
            Self::from_storage(
                self.ring.clone(),
                self.cols,
                self.cols,
                MatrixStorage::FiniteField(q),
            ),
            rank,
        ))
    }
}

impl Matrix<FiniteField> {
    #[cfg(feature = "distributed")]
    pub(crate) fn packed_f2(&self) -> Option<Arc<Mzd>> {
        match self.storage.as_ref() {
            MatrixStorage::FiniteField(matrix) => matrix.packed_f2(),
            _ => None,
        }
    }

    #[cfg(feature = "distributed")]
    pub(crate) fn from_packed_f2(field: FiniteField, matrix: Mzd) -> Option<Self> {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let storage =
            MatrixStorage::FiniteField(FiniteFieldMatrix::from_packed_f2(field.clone(), matrix)?);
        Some(Self::from_storage(Arc::new(field), rows, cols, storage))
    }
}

impl<R> Serialize for Matrix<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let entries = self.materialized_entries();
        let mut state = serializer.serialize_struct("Matrix", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", self.ring())?;
        state.serialize_field("shape", &[self.rows, self.cols])?;
        state.serialize_field("entries", &entries)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for Matrix<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MatrixWire::<R, R::Element>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let expected =
            validate_shape(wire.shape[0], wire.shape[1]).map_err(serde::de::Error::custom)?;
        if wire.entries.len() != expected {
            return Err(serde::de::Error::custom(format!(
                "serialized matrix has {} entries; shape {}x{} requires {expected}",
                wire.entries.len(),
                wire.shape[0],
                wire.shape[1]
            )));
        }
        Self::from_entries(wire.ring, wire.shape[0], wire.shape[1], wire.entries)
            .map_err(serde::de::Error::custom)
    }
}

impl Matrix<IntegerRing> {
    /// Canonical row Hermite normal form over the integers.
    pub fn hermite_normal_form(&self) -> Self {
        let MatrixStorage::Integer(matrix) = self.storage.as_ref() else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            MatrixStorage::Integer(matrix.hermite_normal_form()),
        )
    }

    pub(super) fn smith_normal_form_backend(&self) -> Self {
        let MatrixStorage::Integer(matrix) = self.storage.as_ref() else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        Self::from_storage(
            self.ring.clone(),
            self.rows,
            self.cols,
            MatrixStorage::Integer(matrix.smith_normal_form()),
        )
    }

    pub(super) fn hermite_normal_form_with_transform_backend(&self) -> (Self, Self) {
        let MatrixStorage::Integer(matrix) = &*self.storage else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        let (form, left) = matrix.hermite_normal_form_with_transform();
        (
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                self.cols,
                MatrixStorage::Integer(form),
            ),
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                self.rows,
                MatrixStorage::Integer(left),
            ),
        )
    }

    pub(super) fn smith_normal_form_with_transforms_backend(&self) -> (Self, Self, Self) {
        let MatrixStorage::Integer(matrix) = self.storage.as_ref() else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        let (form, left, right) = matrix.smith_normal_form_with_transforms();
        (
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                self.cols,
                MatrixStorage::Integer(form),
            ),
            Self::from_storage(
                self.ring.clone(),
                self.rows,
                self.rows,
                MatrixStorage::Integer(left),
            ),
            Self::from_storage(
                self.ring.clone(),
                self.cols,
                self.cols,
                MatrixStorage::Integer(right),
            ),
        )
    }

    pub(super) fn determinant_backend(&self) -> Integer {
        let MatrixStorage::Integer(matrix) = self.storage.as_ref() else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        matrix.determinant()
    }

    pub(super) fn rank_backend(&self) -> usize {
        let MatrixStorage::Integer(matrix) = self.storage.as_ref() else {
            unreachable!("integer context always selects FLINT integer matrix storage")
        };
        matrix.rank()
    }
}

impl Matrix<ResidueRing> {
    pub(crate) fn from_integer_entry_refs<'a>(
        ring: ResidueRing,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a Integer>,
    ) -> Result<Self, LinalgError> {
        let length = validate_shape(rows, cols)?;
        debug_assert_eq!(entries.len(), length);
        Ok(Self::from_storage(
            Arc::new(ring.clone()),
            rows,
            cols,
            MatrixStorage::Residue(ModularMatrix::from_integer_entries(
                ring, rows, cols, entries,
            )),
        ))
    }

    /// Canonical row Howell form over `Z/nZ`.
    ///
    /// FLINT requires at least as many rows as columns, so a wide input is
    /// completed with zero rows. The returned matrix therefore has
    /// `max(rows, columns)` rows and generates exactly the same row submodule.
    pub fn howell_normal_form(&self) -> Result<Self, LinalgError> {
        let padded_rows = self.rows.max(self.cols);
        let MatrixStorage::Residue(matrix) = self.storage.as_ref() else {
            unreachable!("residue context always selects FLINT residue matrix storage")
        };
        Ok(Self::from_storage(
            self.ring.clone(),
            padded_rows,
            self.cols,
            MatrixStorage::Residue(matrix.howell(padded_rows)),
        ))
    }
}

impl<R: CommutativeRing> Matrix<R> {
    /// Determinant over the coefficient ring.
    ///
    /// Integer, rational, residue-ring, and finite-field matrices use their
    /// FLINT whole-matrix backends. Every other commutative ring uses the
    /// division-free Berkowitz algorithm, so no field division is required.
    pub fn det(&self) -> Result<R::Element, LinalgError> {
        if self.rows != self.cols {
            return Err(LinalgError::NotSquare {
                operation: "determinant",
                shape: self.shape(),
            });
        }
        let determinant = match self.storage.as_ref() {
            MatrixStorage::Integer(matrix) => cast_owned(matrix.determinant()),
            MatrixStorage::Rational(matrix) => cast_owned(matrix.determinant()),
            MatrixStorage::Residue(matrix) => cast_owned(matrix.determinant_element()),
            MatrixStorage::FiniteField(matrix) => cast_owned(matrix.determinant()),
            MatrixStorage::Dense(_)
            | MatrixStorage::Padic(_)
            | MatrixStorage::RealBall(_)
            | MatrixStorage::ComplexBall(_) => {
                let entries = self.rows().collect::<Vec<_>>();
                let mut coefficients = berkowitz_coefficients(self.ring(), &entries)?;
                let constant = coefficients
                    .pop()
                    .expect("Berkowitz always returns a constant coefficient");
                if self.rows.is_multiple_of(2) {
                    constant
                } else {
                    self.ring().neg(&constant)?
                }
            }
        };
        Ok(determinant)
    }

    /// The characteristic polynomial `det(tI - self)` in the canonical
    /// univariate polynomial ring `R[t]`.
    ///
    /// Integer, rational, residue-ring, and finite-field matrices use their
    /// FLINT whole-matrix backends. Every other commutative ring uses the
    /// division-free Berkowitz algorithm, so the operation remains valid in
    /// positive characteristic and over rings with zero divisors.
    pub fn charpoly(&self) -> Result<Polynomial<R>, LinalgError> {
        if self.rows != self.cols {
            return Err(LinalgError::NotSquare {
                operation: "characteristic polynomial",
                shape: self.shape(),
            });
        }
        let coefficients = match self.storage.as_ref() {
            MatrixStorage::Integer(matrix) => matrix
                .characteristic_polynomial()
                .into_iter()
                .map(cast_owned)
                .collect(),
            MatrixStorage::Rational(matrix) => matrix
                .characteristic_polynomial()
                .into_iter()
                .map(cast_owned)
                .collect(),
            MatrixStorage::Residue(matrix) => matrix
                .characteristic_polynomial()
                .into_iter()
                .map(cast_owned)
                .collect(),
            MatrixStorage::FiniteField(matrix) => matrix
                .characteristic_polynomial()
                .into_iter()
                .map(cast_owned)
                .collect(),
            MatrixStorage::Dense(_)
            | MatrixStorage::Padic(_)
            | MatrixStorage::RealBall(_)
            | MatrixStorage::ComplexBall(_) => {
                let entries = self.rows().collect::<Vec<_>>();
                let mut descending = berkowitz_coefficients(self.ring(), &entries)?;
                descending.reverse();
                descending
            }
        };
        let polynomial_ring =
            PolynomialRing::new(self.ring().clone(), ["t"], MonomialOrder::Lexicographic).map_err(
                |error| LinalgError::PolynomialConstruction {
                    message: error.to_string(),
                },
            )?;
        polynomial_ring
            .from_terms(
                coefficients
                    .into_iter()
                    .enumerate()
                    .map(|(degree, coefficient)| (vec![Integer::from(degree)], coefficient)),
            )
            .map_err(|error| LinalgError::PolynomialConstruction {
                message: error.to_string(),
            })
    }
}

/// Division-free Berkowitz coefficients, ordered from the leading monic
/// coefficient to the constant coefficient.
fn berkowitz_coefficients<R: CommutativeRing>(
    ring: &R,
    matrix: &[Vec<R::Element>],
) -> Result<Vec<R::Element>, LinalgError> {
    let size = matrix.len();
    if size == 0 {
        return Ok(vec![ring.one()]);
    }

    let minor = matrix[1..]
        .iter()
        .map(|row| row[1..].to_vec())
        .collect::<Vec<_>>();
    let minor_coefficients = berkowitz_coefficients(ring, &minor)?;

    // The lower-triangular Toeplitz multiplier has first column
    // 1, -a_00, -R*S, -R*M*S, ..., with M the trailing principal minor.
    let mut toeplitz_column = Vec::with_capacity(size + 1);
    toeplitz_column.push(ring.one());
    toeplitz_column.push(ring.neg(&matrix[0][0])?);
    if size > 1 {
        let row = matrix[0][1..].to_vec();
        let mut power_times_column = matrix[1..]
            .iter()
            .map(|values| values[0].clone())
            .collect::<Vec<_>>();
        for power in 0..size - 1 {
            let mut scalar = ring.zero();
            for (left, right) in row.iter().zip(&power_times_column) {
                scalar = ring.add(&scalar, &ring.mul(left, right)?)?;
            }
            toeplitz_column.push(ring.neg(&scalar)?);
            if power + 1 < size - 1 {
                power_times_column = minor
                    .iter()
                    .map(|minor_row| {
                        let mut value = ring.zero();
                        for (left, right) in minor_row.iter().zip(&power_times_column) {
                            value = ring.add(&value, &ring.mul(left, right)?)?;
                        }
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>, LinalgError>>()?;
            }
        }
    }

    let mut coefficients = Vec::with_capacity(size + 1);
    for output_index in 0..=size {
        let mut coefficient = ring.zero();
        for minor_index in 0..=output_index.min(size - 1) {
            let product = ring.mul(
                &toeplitz_column[output_index - minor_index],
                &minor_coefficients[minor_index],
            )?;
            coefficient = ring.add(&coefficient, &product)?;
        }
        coefficients.push(coefficient);
    }
    Ok(coefficients)
}

impl<R: Ring> Clone for Matrix<R> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring.clone(),
            rows: self.rows,
            cols: self.cols,
            storage: Arc::clone(&self.storage),
            rank: Arc::clone(&self.rank),
        }
    }
}

impl<R: Ring> PartialEq for Matrix<R> {
    fn eq(&self, rhs: &Self) -> bool {
        (Arc::ptr_eq(&self.ring, &rhs.ring) || self.ring == rhs.ring)
            && self.shape() == rhs.shape()
            && (Arc::ptr_eq(&self.storage, &rhs.storage)
                || (0..self.rows).all(|row| {
                    (0..self.cols).all(|col| {
                        self.storage.get(row, col, self.cols) == rhs.storage.get(row, col, rhs.cols)
                    })
                }))
    }
}

impl<R: Ring> Eq for Matrix<R> {}

impl<R: Ring> fmt::Debug for Matrix<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Matrix")
            .field("ring_type", &R::TYPE_ID)
            .field("shape", &self.shape())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for Matrix<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix<{}>({}x{})", R::TYPE_ID, self.rows, self.cols)
    }
}

impl<R: Ring> SemanticType for Matrix<R> {
    const TYPE_ID: &'static str = "noether/linalg/Matrix@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, R::TYPE_ID.as_bytes())?;
        push_frame(&mut output, &self.ring.canonical_bytes()?)?;
        output.extend_from_slice(&(self.rows as u64).to_be_bytes());
        output.extend_from_slice(&(self.cols as u64).to_be_bytes());
        for entry in self.materialized_entries() {
            push_frame(&mut output, &entry.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MATRIX_DESCRIPTION
    }
}

impl<R: Ring> Add<&Matrix<R>> for &Matrix<R> {
    type Output = Result<Matrix<R>, LinalgError>;

    fn add(self, rhs: &Matrix<R>) -> Self::Output {
        Matrix::add(self, rhs)
    }
}

impl<R: Ring> Sub<&Matrix<R>> for &Matrix<R> {
    type Output = Result<Matrix<R>, LinalgError>;

    fn sub(self, rhs: &Matrix<R>) -> Self::Output {
        Matrix::sub(self, rhs)
    }
}

impl<R: Ring> Neg for &Matrix<R> {
    type Output = Result<Matrix<R>, LinalgError>;

    fn neg(self) -> Self::Output {
        Matrix::neg(self)
    }
}

impl<R: Ring> Mul<&R::Element> for &Matrix<R> {
    type Output = Result<Matrix<R>, LinalgError>;

    fn mul(self, rhs: &R::Element) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

impl<R: Ring> Mul<R::Element> for Matrix<R> {
    type Output = Result<Matrix<R>, LinalgError>;

    fn mul(self, rhs: R::Element) -> Self::Output {
        self.scalar_mul(&rhs)
    }
}

fn validate_shape(rows: usize, cols: usize) -> Result<usize, LinalgError> {
    if i64::try_from(rows).is_err() || i64::try_from(cols).is_err() {
        return Err(LinalgError::ShapeTooLarge { rows, cols });
    }
    rows.checked_mul(cols)
        .ok_or(LinalgError::ShapeTooLarge { rows, cols })
}

fn is_integer_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<IntegerRing>()
        && TypeId::of::<R::Element>() == TypeId::of::<Integer>()
}

fn is_rational_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<RationalField>()
        && TypeId::of::<R::Element>() == TypeId::of::<Rational>()
}

fn is_residue_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<ResidueRing>()
        && TypeId::of::<R::Element>() == TypeId::of::<ResidueClass>()
}

fn is_finite_field_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<FiniteField>()
        && TypeId::of::<R::Element>() == TypeId::of::<FiniteFieldElement>()
}

fn is_padic_ring_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<PadicRing>()
        && TypeId::of::<R::Element>() == TypeId::of::<PadicElement>()
}

fn is_padic_field_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<PadicField>()
        && TypeId::of::<R::Element>() == TypeId::of::<PadicElement>()
}

fn is_real_ball_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<RealBallField>()
        && TypeId::of::<R::Element>() == TypeId::of::<RealBall>()
}

fn is_complex_ball_context<R: Ring>() -> bool {
    TypeId::of::<R>() == TypeId::of::<ComplexBallField>()
        && TypeId::of::<R::Element>() == TypeId::of::<ComplexBall>()
}

fn cast_owned<T: Any, U: Any>(value: T) -> U {
    let value: Box<dyn Any> = Box::new(value);
    *value
        .downcast::<U>()
        .unwrap_or_else(|_| panic!("private matrix storage element invariant was violated"))
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        NoetherError::Serialization("matrix component is too large for canonical framing".into())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{MonomialOrder, PolynomialRing, ResidueRing};
    use crate::numthy::{FiniteField, PadicField, PadicRing};
    use crate::{Q, Z};

    #[test]
    fn generic_dense_clones_share_immutable_entry_storage() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let matrix = Matrix::from_rows(ring.clone(), [[x.clone(), x]]).unwrap();
        let cloned = matrix.clone();
        assert!(matches!(matrix.storage.as_ref(), MatrixStorage::Dense(_)));
        if let (MatrixStorage::Dense(left), MatrixStorage::Dense(right)) =
            (matrix.storage.as_ref(), cloned.storage.as_ref())
        {
            assert!(Arc::ptr_eq(left, right));
        }
        assert_eq!(matrix, cloned);
    }

    #[test]
    fn canonical_exact_contexts_select_whole_matrix_storage() {
        let integers = Matrix::identity(Z, 2);
        assert!(matches!(
            integers.storage.as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers.add(&integers).unwrap().storage.as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers.sub(&integers).unwrap().storage.as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers.neg().unwrap().storage.as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers
                .scalar_mul(&Integer::from(3))
                .unwrap()
                .storage
                .as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers.matmul(&integers).unwrap().storage.as_ref(),
            MatrixStorage::Integer(_)
        ));
        assert!(matches!(
            integers.transpose().storage.as_ref(),
            MatrixStorage::Integer(_)
        ));

        let rationals = Matrix::identity(Q, 2);
        assert!(matches!(
            rationals.storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals.add(&rationals).unwrap().storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals.sub(&rationals).unwrap().storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals.neg().unwrap().storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals
                .scalar_mul(&Rational::from(3))
                .unwrap()
                .storage
                .as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals.matmul(&rationals).unwrap().storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
        assert!(matches!(
            rationals.transpose().storage.as_ref(),
            MatrixStorage::Rational(_)
        ));
    }

    #[test]
    fn power_of_two_zero_width_matrices_preserve_shape_under_arithmetic() {
        for modulus in [
            Integer::from(2),
            Integer::from(4),
            Integer::from(5),
            Integer::from(8),
            Integer::from(1_u8) << 80,
        ] {
            let ring = ResidueRing::new(modulus).unwrap();
            for (rows, columns) in [(2, 0), (0, 2), (0, 0)] {
                let matrix = Matrix::zeros(ring.clone(), rows, columns);
                assert_eq!(matrix.add(&matrix).unwrap(), matrix);
                assert_eq!(matrix.sub(&matrix).unwrap(), matrix);
                assert_eq!(matrix.neg().unwrap(), matrix);
                assert_eq!(matrix.scalar_mul(&ring.element(3_u8)).unwrap(), matrix);
                assert_eq!(matrix.transpose().shape(), (columns, rows));
            }
            assert_eq!(
                Matrix::zeros(ring.clone(), 2, 0)
                    .matmul(&Matrix::zeros(ring, 0, 3))
                    .unwrap()
                    .shape(),
                (2, 3)
            );
        }
    }

    #[test]
    fn immutable_matrix_clones_share_authoritative_storage() {
        let integers = Matrix::identity(Z, 128);
        let integer_clone = integers.clone();
        assert!(integers.shares_storage_with(&integer_clone));
        assert_eq!(integers, integer_clone);

        let field = FiniteField::prime(2_u8).unwrap();
        let binary = Matrix::identity(field, 257);
        let binary_clone = binary.clone();
        assert!(binary.shares_storage_with(&binary_clone));
        assert_eq!(binary, binary_clone);
    }

    #[test]
    fn immutable_field_matrix_clones_share_the_verified_rank_cache() {
        let field = FiniteField::prime(5_u8).unwrap();
        let matrix = Matrix::from_rows(
            field.clone(),
            [
                [
                    field.element(1_u8),
                    field.element(2_u8),
                    field.element(3_u8),
                ],
                [
                    field.element(2_u8),
                    field.element(4_u8),
                    field.element(1_u8),
                ],
            ],
        )
        .unwrap();
        let clone = matrix.clone();
        assert_eq!(matrix.cached_rank(), None);
        assert_eq!(clone.rank().unwrap(), 1);
        assert_eq!(matrix.cached_rank(), Some(1));
        let reduced = matrix.rref().unwrap();
        assert_eq!(reduced.cached_rank(), Some(1));
        assert_eq!(reduced.rank().unwrap(), 1);
    }

    #[test]
    fn rref_retains_rank_for_rational_and_extension_field_backends() {
        let rational = Matrix::from_rows(
            Q,
            [[1_i64.into(), 2_i64.into()], [2_i64.into(), 4_i64.into()]],
        )
        .unwrap();
        let rational_reduced = rational.rref().unwrap();
        assert_eq!(rational_reduced.cached_rank(), Some(1));
        assert_eq!(rational.cached_rank(), Some(1));

        let field = FiniteField::extension(5_u64, 2).unwrap();
        let extension = Matrix::from_rows(
            field.clone(),
            [
                [field.element(1_u8), field.element(2_u8)],
                [field.element(2_u8), field.element(4_u8)],
            ],
        )
        .unwrap();
        let extension_reduced = extension.rref().unwrap();
        assert_eq!(extension_reduced.cached_rank(), Some(1));
        assert_eq!(extension.cached_rank(), Some(1));
    }

    #[test]
    fn eager_rows_match_lazy_rows_including_empty_columns() {
        let field = FiniteField::prime(2_u8).unwrap();
        let zero = field.element(0_u8);
        let one = field.element(1_u8);
        let binary = Matrix::from_rows(field, [[zero, one.clone()], [one.clone(), one]]).unwrap();
        assert_eq!(binary.to_rows(), binary.rows().collect::<Vec<_>>());

        let integers = Matrix::from_rows(Z, [[1.into(), 2.into()], [3.into(), 4.into()]]).unwrap();
        assert_eq!(integers.to_rows(), integers.rows().collect::<Vec<_>>());
        assert_eq!(
            Matrix::zeros(Z, 3, 0).to_rows(),
            vec![Vec::<Integer>::new(); 3]
        );
    }

    #[test]
    fn integer_flint_operations_match_the_generic_ring_floor() {
        let entries = vec![
            Integer::from(2),
            Integer::from(-3),
            Integer::from(5),
            "100000000000000000003".parse().unwrap(),
        ];
        let flint = Matrix::from_entries(Z, 2, 2, entries.clone()).unwrap();
        let dense = Matrix::from_storage(Arc::new(Z), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(flint.storage.as_ref(), MatrixStorage::Integer(_)));
        assert!(matches!(dense.storage.as_ref(), MatrixStorage::Dense(_)));

        assert_eq!(flint.add(&flint), dense.add(&dense));
        assert_eq!(flint.sub(&flint), dense.sub(&dense));
        assert_eq!(flint.neg(), dense.neg());
        assert_eq!(
            flint.scalar_mul(&Integer::from(-7)),
            dense.scalar_mul(&Integer::from(-7))
        );
        assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
        assert_eq!(flint.transpose(), dense.transpose());
    }

    #[test]
    fn rational_flint_elimination_matches_generic_empty_and_thin_shapes() {
        for (rows, cols) in [(0, 0), (0, 3), (3, 0), (1, 4), (4, 1), (3, 3)] {
            let entries = (0..rows * cols)
                .map(|index| {
                    Rational::new(
                        Integer::from(index as i64 * 7 - 5),
                        Integer::from(index as u64 + 1),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let flint = Matrix::from_entries(Q, rows, cols, entries.clone()).unwrap();
            let dense =
                Matrix::from_storage(Arc::new(Q), rows, cols, MatrixStorage::dense(entries));

            assert_eq!(flint.rank(), dense.rank());
            assert_eq!(flint.rref(), dense.rref());
            if rows == cols {
                assert_eq!(flint.det(), dense.det());
            }
        }
    }

    #[test]
    fn rational_flint_particular_solver_handles_rectangular_systems() {
        let matrix = Matrix::from_rows(
            Q,
            [
                [Rational::from(1), Rational::from(2), Rational::from(3)],
                [Rational::from(0), Rational::from(1), Rational::from(1)],
            ],
        )
        .unwrap();
        let rhs = Matrix::from_rows(Q, [[Rational::from(1)], [Rational::from(2)]]).unwrap();
        let solution = matrix
            .rational_particular_solution(&rhs)
            .unwrap()
            .expect("the rectangular system is consistent");
        assert_eq!(solution.shape(), (3, 1));
        assert_eq!(matrix.matmul(&solution).unwrap(), rhs);

        let inconsistent =
            Matrix::from_rows(Q, [[Rational::from(0)], [Rational::from(1)]]).unwrap();
        let repeated = Matrix::from_rows(Q, [[Rational::from(1)], [Rational::from(1)]]).unwrap();
        assert!(repeated
            .rational_particular_solution(&inconsistent)
            .unwrap()
            .is_none());
    }

    #[test]
    fn integer_flint_charpoly_matches_berkowitz_through_dimension_five() {
        for size in 0..=5 {
            let entries = (0..size * size)
                .map(|index| Integer::from((index as i64 * 17 + size as i64 * 5) % 23 - 11))
                .collect::<Vec<_>>();
            let flint = Matrix::from_entries(Z, size, size, entries.clone()).unwrap();
            let dense =
                Matrix::from_storage(Arc::new(Z), size, size, MatrixStorage::dense(entries));

            assert_eq!(flint.det().unwrap(), dense.det().unwrap());
            assert_eq!(flint.charpoly().unwrap(), dense.charpoly().unwrap());
        }
    }

    #[test]
    fn rational_flint_operations_match_the_generic_ring_floor() {
        let entries = vec![
            "2/3".parse::<Rational>().unwrap(),
            "-5/7".parse().unwrap(),
            "11/13".parse().unwrap(),
            "17/19".parse().unwrap(),
        ];
        let flint = Matrix::from_entries(Q, 2, 2, entries.clone()).unwrap();
        let dense = Matrix::from_storage(Arc::new(Q), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(flint.storage.as_ref(), MatrixStorage::Rational(_)));
        assert!(matches!(dense.storage.as_ref(), MatrixStorage::Dense(_)));

        assert_eq!(flint.add(&flint), dense.add(&dense));
        assert_eq!(flint.sub(&flint), dense.sub(&dense));
        assert_eq!(flint.neg(), dense.neg());
        assert_eq!(
            flint.scalar_mul(&"23/29".parse().unwrap()),
            dense.scalar_mul(&"23/29".parse().unwrap())
        );
        assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
        assert_eq!(flint.transpose(), dense.transpose());
        assert_eq!(flint.rank(), dense.rank());
        assert_eq!(flint.rref(), dense.rref());
        assert_eq!(flint.det(), dense.det());
        assert_eq!(flint.charpoly(), dense.charpoly());
    }

    #[test]
    fn residue_flint_storage_matches_the_generic_ring_floor() {
        for ring in [
            ResidueRing::new(2_u8).unwrap(),
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new(256_u16).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            let entries = [2_u8, 3, 5, 7]
                .into_iter()
                .map(|value| ring.element(value))
                .collect::<Vec<_>>();
            let flint = Matrix::from_entries(ring.clone(), 2, 2, entries.clone()).unwrap();
            let dense =
                Matrix::from_storage(Arc::new(ring.clone()), 2, 2, MatrixStorage::dense(entries));
            assert!(matches!(flint.storage.as_ref(), MatrixStorage::Residue(_)));
            assert_eq!(flint.add(&flint), dense.add(&dense));
            assert_eq!(flint.sub(&flint), dense.sub(&dense));
            assert_eq!(flint.neg(), dense.neg());
            assert_eq!(
                flint.scalar_mul(&ring.element(11)),
                dense.scalar_mul(&ring.element(11))
            );
            assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
            assert_eq!(flint.transpose(), dense.transpose());
            assert_eq!(flint.det().unwrap(), dense.det().unwrap());
            assert_eq!(flint.charpoly().unwrap(), dense.charpoly().unwrap());
        }
    }

    #[test]
    fn finite_field_specialized_storage_matches_the_generic_field_floor() {
        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(2, 4).unwrap(),
            FiniteField::extension(5, 2).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
            FiniteField::with_modulus(7, vec![1, 0, 1]).unwrap(),
            FiniteField::with_modulus(11, vec![1, 0, 1]).unwrap(),
        ] {
            let entries = if field.degree() == 1 {
                [2_u8, 3, 5, 7]
                    .into_iter()
                    .map(|value| field.element(value))
                    .collect::<Vec<_>>()
            } else {
                let mixed = if field.characteristic() == &Integer::from(2_u8) {
                    field.element_from_coefficients(&[1, 1]).unwrap()
                } else {
                    field.element_from_coefficients(&[3, 1]).unwrap()
                };
                vec![field.generator(), field.one(), mixed, field.element(2_u8)]
            };
            let flint = Matrix::from_entries(field.clone(), 2, 2, entries.clone()).unwrap();
            let dense =
                Matrix::from_storage(Arc::new(field.clone()), 2, 2, MatrixStorage::dense(entries));
            assert!(matches!(
                flint.storage.as_ref(),
                MatrixStorage::FiniteField(_)
            ));
            assert_eq!(flint.add(&flint), dense.add(&dense));
            assert_eq!(flint.sub(&flint), dense.sub(&dense));
            assert_eq!(flint.neg(), dense.neg());
            assert_eq!(
                flint.scalar_mul(&field.element(3_u8)),
                dense.scalar_mul(&field.element(3_u8))
            );
            assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
            assert_eq!(flint.transpose(), dense.transpose());
            assert_eq!(flint.rank(), dense.rank());
            assert_eq!(flint.rref(), dense.rref());
            assert_eq!(flint.det(), dense.det());
            assert_eq!(flint.charpoly(), dense.charpoly());
        }
    }

    #[test]
    fn small_odd_extension_rref_matches_the_generic_field_floor() {
        for field in [
            FiniteField::extension(5, 2).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
            FiniteField::with_modulus(7, vec![1, 0, 1]).unwrap(),
        ] {
            for (rows, cols) in [(6, 6), (5, 8), (8, 5)] {
                let entries = (0..rows * cols)
                    .map(|index| {
                        let characteristic = field.word_characteristic().unwrap() as usize;
                        field
                            .element_from_coefficients(&[
                                ((index * 17 + index / cols * 11 + 3) % characteristic) as u64,
                                ((index * 7 + index / cols * 5 + 1) % characteristic) as u64,
                            ])
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let specialized =
                    Matrix::from_entries(field.clone(), rows, cols, entries.clone()).unwrap();
                let generic = Matrix::from_storage(
                    Arc::new(field.clone()),
                    rows,
                    cols,
                    MatrixStorage::dense(entries),
                );

                assert_eq!(specialized.rref(), generic.rref());
                assert_eq!(specialized.rank(), generic.rank());
            }
        }
    }

    #[test]
    fn small_odd_extension_dispatch_is_bounded_and_preserves_zero_shapes() {
        for field in [
            FiniteField::extension(5, 2).unwrap(),
            FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap(),
            FiniteField::with_modulus(7, vec![1, 0, 1]).unwrap(),
        ] {
            let matrix = Matrix::zeros(field.clone(), 2, 3);
            let MatrixStorage::FiniteField(backend) = matrix.storage.as_ref() else {
                panic!("finite-field matrices use specialized storage")
            };
            assert!(backend.uses_small_odd_extension_backend());

            for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
                let zero = Matrix::zeros(field.clone(), rows, cols);
                assert_eq!(zero.rref().unwrap().shape(), (rows, cols));
                assert_eq!(zero.rank().unwrap(), 0);
                assert_eq!(zero.add(&zero).unwrap(), zero);
                assert_eq!(zero.transpose().shape(), (cols, rows));
            }
        }

        for field in [
            FiniteField::extension(2, 5).unwrap(),
            FiniteField::with_modulus(11, vec![1, 0, 1]).unwrap(),
        ] {
            let matrix = Matrix::zeros(field, 2, 3);
            let MatrixStorage::FiniteField(backend) = matrix.storage.as_ref() else {
                panic!("finite-field matrices use specialized storage")
            };
            assert!(!backend.uses_small_odd_extension_backend());
        }
    }

    #[test]
    fn finite_field_flint_invariants_match_berkowitz_through_dimension_five() {
        for field in [
            FiniteField::prime(2_u8).unwrap(),
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(2, 4).unwrap(),
            FiniteField::extension(5, 2).unwrap(),
        ] {
            for size in 0..=5 {
                let entries = (0..size * size)
                    .map(|index| {
                        let scalar = field.element((index * 17 + size * 5 + 1) as u64);
                        if field.degree() > 1 && index % 2 == 0 {
                            field
                                .add(
                                    &scalar,
                                    &field.generator().pow((index % field.degree() + 1) as u64),
                                )
                                .unwrap()
                        } else {
                            scalar
                        }
                    })
                    .collect::<Vec<_>>();
                let flint =
                    Matrix::from_entries(field.clone(), size, size, entries.clone()).unwrap();
                let dense = Matrix::from_storage(
                    Arc::new(field.clone()),
                    size,
                    size,
                    MatrixStorage::dense(entries),
                );

                assert_eq!(flint.det(), dense.det());
                assert_eq!(flint.charpoly(), dense.charpoly());
            }
        }
    }

    #[test]
    fn padic_flint_storage_matches_the_generic_ring_and_field_floors() {
        let ring = PadicRing::new(5_u8, 8).unwrap();
        let entries = [2_u8, 3, 5, 7]
            .into_iter()
            .map(|value| ring.element(value))
            .collect::<Vec<_>>();
        let flint = Matrix::from_entries(ring.clone(), 2, 2, entries.clone()).unwrap();
        let dense =
            Matrix::from_storage(Arc::new(ring.clone()), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(flint.storage.as_ref(), MatrixStorage::Padic(_)));
        assert_eq!(flint.add(&flint), dense.add(&dense));
        assert_eq!(flint.sub(&flint), dense.sub(&dense));
        assert_eq!(flint.neg(), dense.neg());
        assert_eq!(
            flint.scalar_mul(&ring.element(11_u8)),
            dense.scalar_mul(&ring.element(11_u8))
        );
        assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
        assert_eq!(flint.transpose(), dense.transpose());
        assert_eq!(flint.charpoly(), dense.charpoly());

        let field = PadicField::new(5_u8, 8).unwrap();
        let entries = vec![
            field.element_from_unit_and_valuation(&Integer::one(), -1),
            field.element(1_u8),
            field.element(3_u8),
            field.element(5_u8),
        ];
        let flint = Matrix::from_entries(field.clone(), 2, 2, entries.clone()).unwrap();
        let dense =
            Matrix::from_storage(Arc::new(field.clone()), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(flint.storage.as_ref(), MatrixStorage::Padic(_)));
        assert_eq!(flint.add(&flint), dense.add(&dense));
        assert_eq!(flint.sub(&flint), dense.sub(&dense));
        assert_eq!(flint.neg(), dense.neg());
        assert_eq!(
            flint.scalar_mul(&field.element(11_u8)),
            dense.scalar_mul(&field.element(11_u8))
        );
        assert_eq!(flint.matmul(&flint), dense.matmul(&dense));
        assert_eq!(flint.transpose(), dense.transpose());
        assert_eq!(flint.charpoly(), dense.charpoly());
    }

    #[test]
    fn ball_flint_storage_encloses_the_generic_ring_floor() {
        fn real_overlaps(left: &Matrix<RealBallField>, right: &Matrix<RealBallField>) {
            assert_eq!(left.shape(), right.shape());
            for row in 0..left.rows {
                for col in 0..left.cols {
                    assert!(left
                        .get(row, col)
                        .unwrap()
                        .overlaps(&right.get(row, col).unwrap())
                        .unwrap());
                }
            }
        }

        fn complex_overlaps(left: &Matrix<ComplexBallField>, right: &Matrix<ComplexBallField>) {
            assert_eq!(left.shape(), right.shape());
            for row in 0..left.rows {
                for col in 0..left.cols {
                    assert!(left
                        .get(row, col)
                        .unwrap()
                        .overlaps(&right.get(row, col).unwrap())
                        .unwrap());
                }
            }
        }

        let reals = RealBallField::new(128).unwrap();
        let entries = ["2/3", "-5/7", "11/13", "17/19"]
            .into_iter()
            .map(|value| reals.element(value.parse::<Rational>().unwrap()))
            .collect::<Vec<_>>();
        let flint = Matrix::from_entries(reals, 2, 2, entries.clone()).unwrap();
        let dense = Matrix::from_storage(Arc::new(reals), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(flint.storage.as_ref(), MatrixStorage::RealBall(_)));
        real_overlaps(&flint.add(&flint).unwrap(), &dense.add(&dense).unwrap());
        real_overlaps(&flint.sub(&flint).unwrap(), &dense.sub(&dense).unwrap());
        real_overlaps(&flint.neg().unwrap(), &dense.neg().unwrap());
        let scalar = reals.element("23/29".parse::<Rational>().unwrap());
        real_overlaps(
            &flint.scalar_mul(&scalar).unwrap(),
            &dense.scalar_mul(&scalar).unwrap(),
        );
        real_overlaps(
            &flint.matmul(&flint).unwrap(),
            &dense.matmul(&dense).unwrap(),
        );
        real_overlaps(&flint.transpose(), &dense.transpose());

        let complexes = ComplexBallField::new(128).unwrap();
        let entries = [(2, 3), (-5, 7), (11, -13), (17, 19)]
            .into_iter()
            .map(|(real, imaginary)| complexes.element_parts(real, imaginary))
            .collect::<Vec<_>>();
        let flint = Matrix::from_entries(complexes, 2, 2, entries.clone()).unwrap();
        let dense = Matrix::from_storage(Arc::new(complexes), 2, 2, MatrixStorage::dense(entries));
        assert!(matches!(
            flint.storage.as_ref(),
            MatrixStorage::ComplexBall(_)
        ));
        complex_overlaps(&flint.add(&flint).unwrap(), &dense.add(&dense).unwrap());
        complex_overlaps(&flint.sub(&flint).unwrap(), &dense.sub(&dense).unwrap());
        complex_overlaps(&flint.neg().unwrap(), &dense.neg().unwrap());
        let scalar = complexes.element_parts(3, -2);
        complex_overlaps(
            &flint.scalar_mul(&scalar).unwrap(),
            &dense.scalar_mul(&scalar).unwrap(),
        );
        complex_overlaps(
            &flint.matmul(&flint).unwrap(),
            &dense.matmul(&dense).unwrap(),
        );
        complex_overlaps(&flint.transpose(), &dense.transpose());
    }
}
