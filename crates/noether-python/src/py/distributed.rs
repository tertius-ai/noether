// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Thin Python projection of distributed values owned by `noether::linalg`.

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::panic_in_result_fn
)]

use noether::linalg::{DistributedMatrix, DistributedPLUQFactorization, MpiThreadSupport, World};
use noether::numthy::FiniteField;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use super::errors::distributed as distributed_error;
use super::linalg::PyMatrix;
use super::numthy::{PyFiniteField, PyFiniteFieldElement};

/// MPI thread-support level supplied to Noether.
#[pyclass(
    module = "noether.linalg",
    name = "MpiThreadSupport",
    frozen,
    from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyMpiThreadSupport {
    value: MpiThreadSupport,
}

impl From<MpiThreadSupport> for PyMpiThreadSupport {
    fn from(value: MpiThreadSupport) -> Self {
        Self { value }
    }
}

#[pymethods]
impl PyMpiThreadSupport {
    /// MPI permits only one application thread.
    #[classattr]
    #[pyo3(name = "SINGLE")]
    fn single() -> Self {
        MpiThreadSupport::Single.into()
    }

    /// Only the initializing thread may call MPI.
    #[classattr]
    #[pyo3(name = "FUNNELED")]
    fn funneled() -> Self {
        MpiThreadSupport::Funneled.into()
    }

    /// MPI calls may originate on different threads but are serialized.
    #[classattr]
    #[pyo3(name = "SERIALIZED")]
    fn serialized() -> Self {
        MpiThreadSupport::Serialized.into()
    }

    /// MPI permits concurrent calls from multiple threads.
    #[classattr]
    #[pyo3(name = "MULTIPLE")]
    fn multiple() -> Self {
        MpiThreadSupport::Multiple.into()
    }

    /// Stable lowercase level name.
    #[getter]
    fn name(&self) -> &'static str {
        match self.value {
            MpiThreadSupport::Single => "single",
            MpiThreadSupport::Funneled => "funneled",
            MpiThreadSupport::Serialized => "serialized",
            MpiThreadSupport::Multiple => "multiple",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> isize {
        self.value as isize
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!("MpiThreadSupport.{}", self.name().to_uppercase()))
    }

    fn __str__(&self) -> &'static str {
        self.name()
    }
}

/// Explicit owner of one initialized MPI world.
#[pyclass(module = "noether.linalg", name = "World", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyWorld {
    inner: World,
}

#[pymethods]
impl PyWorld {
    /// Initialize MPI exactly once for this process.
    #[classmethod]
    fn initialize(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| {
            Ok(Self {
                inner: World::initialize().map_err(distributed_error)?,
            })
        })
    }

    /// Number of processes in the MPI world.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// This process's MPI rank.
    #[getter]
    fn mpi_rank(&self) -> usize {
        self.inner.mpi_rank()
    }

    /// Whether this process is rank zero.
    #[getter]
    fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    /// MPI implementation/version string.
    #[getter]
    fn library_version(&self) -> &str {
        self.inner.library_version()
    }

    /// Thread support supplied by MPI.
    #[getter]
    fn thread_support(&self) -> PyMpiThreadSupport {
        self.inner.thread_support().into()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "World(size={}, mpi_rank={}, thread_support={:?})",
            self.size(),
            self.mpi_rank(),
            self.thread_support().name()
        ))
    }
}

/// A dense exact `F_2` matrix distributed over an explicit MPI world.
#[pyclass(
    module = "noether.linalg",
    name = "DistributedMatrix",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDistributedMatrix {
    inner: DistributedMatrix<FiniteField>,
}

impl PyDistributedMatrix {
    fn from_inner(inner: DistributedMatrix<FiniteField>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDistributedMatrix {
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    /// Collective zero matrix. Every process must call this method in the
    /// same order with the same arguments.
    #[classmethod]
    fn zeros(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        world: &PyWorld,
        field: &PyFiniteField,
        rows: usize,
        cols: usize,
    ) -> PyResult<Self> {
        let world = world.inner.clone();
        let field = field.inner.clone();
        py.detach(move || {
            DistributedMatrix::zeros(&world, field, rows, cols)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective identity matrix.
    #[classmethod]
    fn identity(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        world: &PyWorld,
        field: &PyFiniteField,
        size: usize,
    ) -> PyResult<Self> {
        let world = world.inner.clone();
        let field = field.inner.clone();
        py.detach(move || {
            DistributedMatrix::identity(&world, field, size)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective process-grid-independent deterministic random matrix.
    #[classmethod]
    #[pyo3(signature = (world, field, rows, cols, *, seed))]
    fn random(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        world: &PyWorld,
        field: &PyFiniteField,
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let world = world.inner.clone();
        let field = field.inner.clone();
        py.detach(move || {
            DistributedMatrix::random(&world, field, rows, cols, seed)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collectively scatter one root-owned local matrix without unpacking its
    /// packed `F_2` storage.
    #[classmethod]
    #[pyo3(signature = (world, *, root=0, matrix=None, max_bytes))]
    fn distribute(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        world: &PyWorld,
        root: usize,
        matrix: Option<&PyMatrix>,
        max_bytes: usize,
    ) -> PyResult<Self> {
        let matrix = matrix
            .map(|value| {
                value.finite_field().ok_or_else(|| {
                    PyTypeError::new_err(
                        "DistributedMatrix.distribute requires Matrix[FiniteField]",
                    )
                })
            })
            .transpose()?;
        py.detach(|| {
            DistributedMatrix::distribute(&world.inner, root, matrix, max_bytes)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collectively reopen a process-grid-independent manifest.
    #[classmethod]
    fn open(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        world: &PyWorld,
        path: String,
    ) -> PyResult<Self> {
        let world = world.inner.clone();
        py.detach(move || {
            DistributedMatrix::open(&world, path)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Exact coefficient field (`F_2` in v0.2).
    #[getter]
    fn field(&self) -> PyFiniteField {
        PyFiniteField {
            inner: self.inner.field().clone(),
        }
    }

    /// Alias for the exact coefficient field.
    #[getter]
    fn ring(&self) -> PyFiniteField {
        self.field()
    }

    /// Explicit MPI world retained by the matrix.
    #[getter]
    fn world(&self) -> PyWorld {
        PyWorld {
            inner: self.inner.world().clone(),
        }
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn process_grid(&self) -> (usize, usize) {
        self.inner.process_grid()
    }

    #[getter]
    fn tile_shape(&self) -> (usize, usize) {
        self.inner.tile_shape()
    }

    #[getter]
    fn local_memory_bytes(&self) -> usize {
        self.inner.local_memory_bytes()
    }

    #[getter]
    fn global_memory_bytes(&self) -> Option<usize> {
        self.inner.global_memory_bytes()
    }

    /// Collective read of one coefficient.
    fn get(&self, py: Python<'_>, row: usize, col: usize) -> PyResult<PyFiniteFieldElement> {
        py.detach(|| {
            Ok(PyFiniteFieldElement {
                inner: self.inner.get(row, col).map_err(distributed_error)?,
            })
        })
    }

    /// Collective owned update of one coefficient.
    fn set(
        &self,
        py: Python<'_>,
        row: usize,
        col: usize,
        value: &PyFiniteFieldElement,
    ) -> PyResult<Self> {
        let value = value.inner.clone();
        py.detach(|| {
            self.inner
                .set(row, col, &value)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective owned toggle of one coefficient.
    fn flip(&self, py: Python<'_>, row: usize, col: usize) -> PyResult<Self> {
        py.detach(|| {
            self.inner
                .flip(row, col)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective owned row replacement ``row[dst] += row[src]`` over F2.
    fn xor_row(&self, py: Python<'_>, dst: usize, src: usize) -> PyResult<Self> {
        py.detach(|| {
            self.inner
                .xor_row(dst, src)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective owned row interchange.
    fn swap_rows(&self, py: Python<'_>, first: usize, second: usize) -> PyResult<Self> {
        py.detach(|| {
            self.inner
                .swap_rows(first, second)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective owned column interchange.
    fn swap_cols(&self, py: Python<'_>, first: usize, second: usize) -> PyResult<Self> {
        py.detach(|| {
            self.inner
                .swap_cols(first, second)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collectively extract at most 64 packed row bits.
    fn extract_bits(
        &self,
        py: Python<'_>,
        row: usize,
        col_start: usize,
        count: usize,
    ) -> PyResult<u64> {
        py.detach(|| {
            self.inner
                .extract_bits(row, col_start, count)
                .map_err(distributed_error)
        })
    }

    /// Collective owned transpose.
    fn transpose(&self, py: Python<'_>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_unary owns the collective.
        self.detached_unary(py, DistributedMatrix::transpose)
    }

    /// Collective owned half-open submatrix.
    fn submatrix(
        &self,
        py: Python<'_>,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
    ) -> PyResult<Self> {
        py.detach(|| {
            self.inner
                .submatrix(row_start, col_start, row_end, col_end)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    /// Collective paste into an owned copy of this matrix.
    fn paste(
        &self,
        py: Python<'_>,
        row_start: usize,
        col_start: usize,
        source: &Self,
    ) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_binary owns the collective.
        self.detached_binary(py, source, move |left, right| {
            left.paste(row_start, col_start, right)
        })
    }

    /// Collective exact matrix product.
    fn matmul(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_binary owns the collective.
        self.detached_binary(py, other, DistributedMatrix::matmul)
    }

    /// Collective row echelon form and rank.
    fn echelon(&self, py: Python<'_>) -> PyResult<(Self, usize)> {
        py.detach(|| {
            let (matrix, rank) = self.inner.echelon().map_err(distributed_error)?;
            Ok((Self::from_inner(matrix), rank))
        })
    }

    /// Collective reduced row echelon form and rank.
    fn rref(&self, py: Python<'_>) -> PyResult<(Self, usize)> {
        py.detach(|| {
            let (matrix, rank) = self.inner.rref().map_err(distributed_error)?;
            Ok((Self::from_inner(matrix), rank))
        })
    }

    /// Collective mathematical rank without retaining witnesses.
    fn rank(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.rank().map_err(distributed_error))
    }

    /// Collective PLE rank without retaining witnesses.
    fn ple_rank(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.ple_rank().map_err(distributed_error))
    }

    /// Collective deterministic PLUQ decomposition with ``P L U Q = A``.
    fn pluq(&self, py: Python<'_>) -> PyResult<PyDistributedPLUQFactorization> {
        py.detach(|| {
            Ok(PyDistributedPLUQFactorization {
                inner: self.inner.pluq().map_err(distributed_error)?,
            })
        })
    }

    /// Collectively solve ``self @ X = right`` or return ``None``.
    fn solve_left(&self, py: Python<'_>, right: &Self) -> PyResult<Option<Self>> {
        py.detach(|| {
            self.inner
                .solve_left(&right.inner)
                .map(|value| value.map(Self::from_inner))
                .map_err(distributed_error)
        })
    }

    /// Collective right-nullspace basis with basis vectors as columns.
    fn nullspace(&self, py: Python<'_>) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_unary owns the collective.
        self.detached_unary(py, DistributedMatrix::nullspace)
    }

    /// Collective multiplicative inverse or ``None`` when singular.
    fn inverse(&self, py: Python<'_>) -> PyResult<Option<Self>> {
        py.detach(|| {
            self.inner
                .inverse()
                .map(|value| value.map(Self::from_inner))
                .map_err(distributed_error)
        })
    }

    /// Collective exact equality without gathering either matrix.
    fn equals(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        py.detach(|| self.inner.equals(&other.inner).map_err(distributed_error))
    }

    /// Collectively write a process-grid-independent packed manifest.
    fn write(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.inner.write(path).map_err(distributed_error))
    }

    #[pyo3(signature = (*, root=0, max_bytes))]
    /// Collect a packed matrix on one root under an explicit byte ceiling.
    fn collect(&self, py: Python<'_>, root: usize, max_bytes: usize) -> PyResult<Option<PyMatrix>> {
        py.detach(|| {
            self.inner
                .collect(root, max_bytes)
                .map(|value| value.map(PyMatrix::from_finite_field))
                .map_err(distributed_error)
        })
    }

    fn __add__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_binary owns the collective.
        self.detached_binary(py, other, DistributedMatrix::add)
    }

    fn __sub__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: detached_binary owns the collective.
        self.detached_binary(py, other, DistributedMatrix::sub)
    }

    fn __matmul__(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        // NOETHER_RELEASES_GIL: matmul delegates to detached_binary.
        self.matmul(py, other)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DistributedMatrix[FiniteField](shape={:?}, mpi_rank={}, world_size={}, local_bytes={})",
            self.shape(),
            self.inner.world().mpi_rank(),
            self.inner.world().size(),
            self.local_memory_bytes()
        ))
    }
}

impl PyDistributedMatrix {
    fn detached_unary(
        &self,
        py: Python<'_>,
        operation: impl FnOnce(
                &DistributedMatrix<FiniteField>,
            )
                -> Result<DistributedMatrix<FiniteField>, noether::linalg::DistributedError>
            + Send,
    ) -> PyResult<Self> {
        py.detach(|| {
            operation(&self.inner)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }

    fn detached_binary(
        &self,
        py: Python<'_>,
        other: &Self,
        operation: impl FnOnce(
                &DistributedMatrix<FiniteField>,
                &DistributedMatrix<FiniteField>,
            )
                -> Result<DistributedMatrix<FiniteField>, noether::linalg::DistributedError>
            + Send,
    ) -> PyResult<Self> {
        py.detach(|| {
            operation(&self.inner, &other.inner)
                .map(Self::from_inner)
                .map_err(distributed_error)
        })
    }
}

/// Owned distributed PLUQ decomposition with convention `P L U Q = A`.
#[pyclass(
    module = "noether.linalg",
    name = "DistributedPLUQFactorization",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDistributedPLUQFactorization {
    inner: DistributedPLUQFactorization<FiniteField>,
}

#[pymethods]
impl PyDistributedPLUQFactorization {
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().unbind().into_any()
    }

    #[getter]
    fn p(&self) -> PyDistributedMatrix {
        PyDistributedMatrix::from_inner(self.inner.p().clone())
    }

    #[getter]
    fn l(&self) -> PyDistributedMatrix {
        PyDistributedMatrix::from_inner(self.inner.l().clone())
    }

    #[getter]
    fn u(&self) -> PyDistributedMatrix {
        PyDistributedMatrix::from_inner(self.inner.u().clone())
    }

    #[getter]
    fn q(&self) -> PyDistributedMatrix {
        PyDistributedMatrix::from_inner(self.inner.q().clone())
    }

    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn verify(&self, py: Python<'_>, source: &PyDistributedMatrix) -> PyResult<bool> {
        py.detach(|| self.inner.verify(&source.inner).map_err(distributed_error))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DistributedPLUQFactorization(rank={})",
            self.rank()
        ))
    }
}
