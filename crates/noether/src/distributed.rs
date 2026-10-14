// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! MPI-distributed exact linear algebra over `F_2`.
//!
//! The public value remains parameterized by its mathematical coefficient
//! context. Version 0.2 admits only `FiniteField::prime(2)`; every rank owns
//! packed `m4rst` tiles and no operation silently gathers a complete matrix.

use core::fmt;
use core::ops::{Add, Mul, Sub};
use std::path::Path;
use std::sync::Arc;

use m4rst_distributed::{DistributedMzd, World as KernelWorld};

use crate::linalg::Matrix;
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{Integer, Ring};

pub use m4rst_distributed::MpiThreadSupport;

/// Read-only execution facts for a currently retained MPI world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MpiRuntimeFacts {
    /// MPI implementation/version string.
    pub library_version: String,
    /// Number of processes in the world.
    pub world_size: usize,
    /// This process's MPI rank.
    pub mpi_rank: usize,
    /// Thread support supplied by MPI.
    pub thread_support: MpiThreadSupport,
}

/// Return MPI facts when an explicit [`World`] is currently alive.
#[must_use]
pub fn active_mpi_runtime() -> Option<MpiRuntimeFacts> {
    m4rst_distributed::active_mpi_runtime().map(|runtime| MpiRuntimeFacts {
        library_version: runtime.library_version,
        world_size: runtime.world_size,
        mpi_rank: runtime.mpi_rank,
        thread_support: runtime.thread_support,
    })
}

/// Failure from a distributed context, coefficient check, or MPI operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistributedError {
    /// The requested coefficient context is not supported by the distributed
    /// v0.2 kernel.
    UnsupportedCoefficientContext {
        /// Human-readable context supplied by the caller.
        context: String,
    },
    /// Values carrying unequal coefficient contexts were combined.
    ContextMismatch {
        /// Context required by the operation.
        expected: String,
        /// Context carried by the rejected value.
        actual: String,
    },
    /// The underlying packed MPI kernel rejected the operation.
    Kernel(m4rst_distributed::DistributedError),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCoefficientContext { context } => write!(
                formatter,
                "distributed matrices support F_2 in v0.2, not {context}"
            ),
            Self::ContextMismatch { expected, actual } => write!(
                formatter,
                "distributed coefficient context mismatch: expected {expected}, received {actual}"
            ),
            Self::Kernel(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DistributedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::UnsupportedCoefficientContext { .. } | Self::ContextMismatch { .. } => None,
        }
    }
}

impl From<m4rst_distributed::DistributedError> for DistributedError {
    fn from(error: m4rst_distributed::DistributedError) -> Self {
        Self::Kernel(error)
    }
}

/// Explicit owner of Noether's bounded MPI world context.
///
/// Initialization requests serialized MPI thread support. This type exposes
/// process facts needed by mathematical programs, not general MPI messaging.
#[derive(Clone)]
pub struct World {
    inner: KernelWorld,
}

impl World {
    /// Initialize MPI exactly once for this process.
    pub fn initialize() -> Result<Self, DistributedError> {
        Ok(Self {
            inner: KernelWorld::initialize()?,
        })
    }

    /// Number of MPI processes in this world.
    #[must_use]
    pub fn size(&self) -> usize {
        usize::try_from(self.inner.size()).expect("MPI world sizes are positive")
    }

    /// This process's MPI rank.
    #[must_use]
    pub fn mpi_rank(&self) -> usize {
        usize::try_from(self.inner.mpi_rank()).expect("MPI ranks are nonnegative")
    }

    /// Whether this process is rank zero.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.inner.is_root()
    }

    /// MPI implementation/version string reported by the linked runtime.
    #[must_use]
    pub fn library_version(&self) -> &str {
        self.inner.library_version()
    }

    /// Thread-support level supplied by the initialized MPI runtime.
    #[must_use]
    pub fn thread_support(&self) -> MpiThreadSupport {
        self.inner.thread_support()
    }
}

impl fmt::Debug for World {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("World")
            .field("mpi_rank", &self.mpi_rank())
            .field("size", &self.size())
            .field("library_version", &self.library_version())
            .field("thread_support", &self.thread_support())
            .finish_non_exhaustive()
    }
}

/// A dense exact matrix whose packed rank-local tiles are distributed over an
/// explicit [`World`].
///
/// The generic spelling preserves the mathematical identity. Version 0.2
/// provides constructors and operations only for `DistributedMatrix<FiniteField>`
/// carrying `F_2`.
#[derive(Clone)]
pub struct DistributedMatrix<R: Ring> {
    world: World,
    ring: Arc<R>,
    inner: Arc<DistributedMzd>,
}

impl DistributedMatrix<FiniteField> {
    /// Collective zero matrix over `field`.
    pub fn zeros(
        world: &World,
        field: FiniteField,
        rows: usize,
        cols: usize,
    ) -> Result<Self, DistributedError> {
        require_collective_f2(world, &field)?;
        Ok(Self::from_inner(
            world.clone(),
            field,
            DistributedMzd::new(&world.inner, rows, cols)?,
        ))
    }

    /// Collective identity matrix over `field`.
    pub fn identity(
        world: &World,
        field: FiniteField,
        size: usize,
    ) -> Result<Self, DistributedError> {
        require_collective_f2(world, &field)?;
        Ok(Self::from_inner(
            world.clone(),
            field,
            DistributedMzd::identity(&world.inner, size)?,
        ))
    }

    /// Collective deterministic random matrix over `field`.
    ///
    /// The logical bits are independent of process-grid shape.
    pub fn random(
        world: &World,
        field: FiniteField,
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> Result<Self, DistributedError> {
        require_collective_f2(world, &field)?;
        Ok(Self::from_inner(
            world.clone(),
            field,
            DistributedMzd::random(&world.inner, rows, cols, seed)?,
        ))
    }

    /// Collectively scatter a root-owned local `F_2` matrix.
    ///
    /// The root passes `Some(matrix)` and every other rank passes `None`.
    /// Packed storage is scattered directly; the matrix is never expanded to
    /// one scalar or byte per coefficient. `max_bytes` is checked before data
    /// movement.
    pub fn distribute(
        world: &World,
        root: usize,
        matrix: Option<&Matrix<FiniteField>>,
        max_bytes: usize,
    ) -> Result<Self, DistributedError> {
        let root_i32 = i32::try_from(root).map_err(|_| {
            m4rst_distributed::DistributedError::InvalidMpiRank {
                rank: i32::MAX,
                world_size: world.inner.size(),
            }
        })?;
        let role_is_valid = (world.mpi_rank() == root) == matrix.is_some();
        if !world.inner.collective_all(role_is_valid) {
            return Err(m4rst_distributed::DistributedError::RootBuffer(
                "the root must provide one local matrix and every non-root must pass None"
                    .to_owned(),
            )
            .into());
        }
        let local_is_f2 = matrix.is_none_or(|value| is_f2(value.ring()));
        if !world.inner.collective_all(local_is_f2) {
            return Err(DistributedError::UnsupportedCoefficientContext {
                context: "the root's local matrix context".to_owned(),
            });
        }
        let packed = matrix.and_then(Matrix::packed_f2);
        if !world
            .inner
            .collective_all(matrix.is_none() || packed.is_some())
        {
            return Err(DistributedError::UnsupportedCoefficientContext {
                context: "a non-packed local matrix representation".to_owned(),
            });
        }
        let inner = DistributedMzd::distribute_packed(
            &world.inner,
            packed.as_deref(),
            root_i32,
            max_bytes,
        )?;
        Ok(Self::from_inner(
            world.clone(),
            FiniteField::prime(2_u64).expect("two is prime"),
            inner,
        ))
    }

    /// Collectively reopen a process-grid-independent packed manifest.
    pub fn open(world: &World, path: impl AsRef<Path>) -> Result<Self, DistributedError> {
        let inner = DistributedMzd::open(&world.inner, path)?;
        Ok(Self::from_inner(
            world.clone(),
            FiniteField::prime(2_u64).expect("two is prime"),
            inner,
        ))
    }

    /// Exact coefficient field.
    #[must_use]
    pub fn field(&self) -> &FiniteField {
        self.ring.as_ref()
    }

    /// Collective read of one coefficient.
    pub fn get(&self, row: usize, col: usize) -> Result<FiniteFieldElement, DistributedError> {
        Ok(self.field().element(self.inner.get(row, col)?))
    }

    /// Collective owned update of one coefficient.
    pub fn set(
        &self,
        row: usize,
        col: usize,
        value: &FiniteFieldElement,
    ) -> Result<Self, DistributedError> {
        self.require_collective_element(value)?;
        let mut output = self.inner.as_ref().clone();
        let bit = value
            .representative()
            .to_u64_checked()
            .expect("an F_2 representative is word-sized") as u8;
        output.set(row, col, bit)?;
        self.owned_result(output)
    }

    /// Collective owned toggle of one coefficient.
    pub fn flip(&self, row: usize, col: usize) -> Result<Self, DistributedError> {
        let mut output = self.inner.as_ref().clone();
        output.flip(row, col)?;
        self.owned_result(output)
    }

    /// Collective owned row replacement `row[dst] += row[src]` over `F_2`.
    pub fn xor_row(&self, dst: usize, src: usize) -> Result<Self, DistributedError> {
        let mut output = self.inner.as_ref().clone();
        output.xor_row(dst, src)?;
        self.owned_result(output)
    }

    /// Collective owned row interchange.
    pub fn swap_rows(&self, first: usize, second: usize) -> Result<Self, DistributedError> {
        let mut output = self.inner.as_ref().clone();
        output.swap_rows(first, second)?;
        self.owned_result(output)
    }

    /// Collective owned column interchange.
    pub fn swap_cols(&self, first: usize, second: usize) -> Result<Self, DistributedError> {
        let mut output = self.inner.as_ref().clone();
        output.swap_cols(first, second)?;
        self.owned_result(output)
    }

    /// Collectively extract at most 64 row bits, least-significant bit first.
    pub fn extract_bits(
        &self,
        row: usize,
        col_start: usize,
        count: usize,
    ) -> Result<u64, DistributedError> {
        self.inner
            .extract_bits(row, col_start, count)
            .map_err(Into::into)
    }

    /// Exact element-wise addition/XOR without gathering either operand.
    pub fn add(&self, other: &Self) -> Result<Self, DistributedError> {
        self.binary_result(other, |left, right| left.add(right))
    }

    /// Exact element-wise subtraction, equal to addition over `F_2`.
    pub fn sub(&self, other: &Self) -> Result<Self, DistributedError> {
        self.binary_result(other, |left, right| left.sub(right))
    }

    /// Collective transpose without gathering the source.
    pub fn transpose(&self) -> Result<Self, DistributedError> {
        self.owned_result(self.inner.transpose()?)
    }

    /// Collective owned half-open submatrix.
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
    ) -> Result<Self, DistributedError> {
        self.owned_result(
            self.inner
                .submatrix(row_start, col_start, row_end, col_end)?,
        )
    }

    /// Collective paste into an owned copy of this destination.
    pub fn paste(
        &self,
        row_start: usize,
        col_start: usize,
        source: &Self,
    ) -> Result<Self, DistributedError> {
        self.require_collective_same_field(source)?;
        self.owned_result(self.inner.paste(row_start, col_start, &source.inner)?)
    }

    /// Collective exact matrix product without gathering either operand.
    pub fn matmul(&self, other: &Self) -> Result<Self, DistributedError> {
        self.binary_result(other, DistributedMzd::matmul)
    }

    /// Collective owned row echelon form and mathematical rank.
    pub fn echelon(&self) -> Result<(Self, usize), DistributedError> {
        let (matrix, rank) = self.inner.echelon()?;
        Ok((self.owned_result(matrix)?, rank))
    }

    /// Collective owned reduced row echelon form and mathematical rank.
    pub fn rref(&self) -> Result<(Self, usize), DistributedError> {
        let (matrix, rank) = self.inner.rref()?;
        Ok((self.owned_result(matrix)?, rank))
    }

    /// Collective no-witness mathematical rank.
    pub fn rank(&self) -> Result<usize, DistributedError> {
        self.inner.rank().map_err(Into::into)
    }

    /// Collective no-witness PLE rank.
    pub fn ple_rank(&self) -> Result<usize, DistributedError> {
        self.inner.ple_rank().map_err(Into::into)
    }

    /// Collective deterministic PLUQ decomposition with `P L U Q = self`.
    pub fn pluq(&self) -> Result<DistributedPLUQFactorization<FiniteField>, DistributedError> {
        let (p, l, u, q, rank) = self.inner.pluq()?.into_parts();
        Ok(DistributedPLUQFactorization {
            p: self.owned_result(p)?,
            l: self.owned_result(l)?,
            u: self.owned_result(u)?,
            q: self.owned_result(q)?,
            rank,
        })
    }

    /// Collective solution of `self * X = right`, or `None` when inconsistent.
    pub fn solve_left(&self, right: &Self) -> Result<Option<Self>, DistributedError> {
        self.require_collective_same_field(right)?;
        DistributedMzd::solve_left(&self.inner, &right.inner)?
            .map(|matrix| self.owned_result(matrix))
            .transpose()
    }

    /// Collective right-nullspace basis, with basis vectors as columns.
    pub fn nullspace(&self) -> Result<Self, DistributedError> {
        self.owned_result(self.inner.nullspace()?)
    }

    /// Collective multiplicative inverse, or `None` when singular.
    pub fn inverse(&self) -> Result<Option<Self>, DistributedError> {
        self.inner
            .inverse()?
            .map(|matrix| self.owned_result(matrix))
            .transpose()
    }

    /// Collective exact equality without gathering either operand.
    pub fn equals(&self, other: &Self) -> Result<bool, DistributedError> {
        self.require_collective_same_field(other)?;
        self.inner.equals(&other.inner).map_err(Into::into)
    }

    /// Collectively persist rank-owned packed tiles and a portable manifest.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), DistributedError> {
        self.inner.write(path).map(|_| ()).map_err(Into::into)
    }

    /// Collect a bounded packed matrix on `root`.
    ///
    /// The root receives `Some(Matrix)` and every other rank receives `None`.
    /// `max_bytes` is checked before communication.
    pub fn collect(
        &self,
        root: usize,
        max_bytes: usize,
    ) -> Result<Option<Matrix<FiniteField>>, DistributedError> {
        let root = i32::try_from(root).map_err(|_| {
            m4rst_distributed::DistributedError::InvalidMpiRank {
                rank: i32::MAX,
                world_size: self.world.inner.size(),
            }
        })?;
        self.inner
            .collect(root, max_bytes)?
            .map(|matrix| {
                Matrix::from_packed_f2(self.field().clone(), matrix).ok_or_else(|| {
                    DistributedError::UnsupportedCoefficientContext {
                        context: self.field().to_string(),
                    }
                })
            })
            .transpose()
    }

    fn from_inner(world: World, field: FiniteField, inner: DistributedMzd) -> Self {
        Self {
            world,
            ring: Arc::new(field),
            inner: Arc::new(inner),
        }
    }

    fn owned_result(&self, inner: DistributedMzd) -> Result<Self, DistributedError> {
        Ok(Self {
            world: self.world.clone(),
            ring: Arc::clone(&self.ring),
            inner: Arc::new(inner),
        })
    }

    fn binary_result(
        &self,
        other: &Self,
        operation: impl FnOnce(
            &DistributedMzd,
            &DistributedMzd,
        ) -> Result<DistributedMzd, m4rst_distributed::DistributedError>,
    ) -> Result<Self, DistributedError> {
        self.require_collective_same_field(other)?;
        self.owned_result(operation(self.inner.as_ref(), other.inner.as_ref())?)
    }

    fn require_collective_same_field(&self, other: &Self) -> Result<(), DistributedError> {
        if self
            .world
            .inner
            .collective_all(self.field() == other.field())
        {
            Ok(())
        } else {
            Err(DistributedError::ContextMismatch {
                expected: self.field().to_string(),
                actual: other.field().to_string(),
            })
        }
    }

    fn require_collective_element(
        &self,
        value: &FiniteFieldElement,
    ) -> Result<(), DistributedError> {
        if self
            .world
            .inner
            .collective_all(self.field().contains(value))
        {
            Ok(())
        } else {
            Err(DistributedError::ContextMismatch {
                expected: self.field().to_string(),
                actual: value.field().to_string(),
            })
        }
    }
}

impl<R: Ring> DistributedMatrix<R> {
    /// Exact coefficient context.
    #[must_use]
    pub fn ring(&self) -> &R {
        self.ring.as_ref()
    }

    /// Explicit MPI world retained by this value.
    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    /// Logical row count.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Logical column count.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Logical `(rows, columns)` shape.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Current process-grid shape `(process rows, process columns)`.
    #[must_use]
    pub fn process_grid(&self) -> (usize, usize) {
        let grid = self.inner.layout().process_grid();
        (grid.rows(), grid.cols())
    }

    /// Nominal rank-local tile shape `(tile rows, tile columns)`.
    #[must_use]
    pub fn tile_shape(&self) -> (usize, usize) {
        (
            self.inner.layout().tile_rows(),
            self.inner.layout().tile_cols(),
        )
    }

    /// Bytes of packed matrix payload owned by this process.
    #[must_use]
    pub fn local_memory_bytes(&self) -> usize {
        self.inner.local_memory_bytes()
    }

    /// Total packed payload bytes across all processes.
    #[must_use]
    pub fn global_memory_bytes(&self) -> Option<usize> {
        self.inner.global_memory_bytes()
    }
}

impl<R: Ring> fmt::Debug for DistributedMatrix<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DistributedMatrix")
            .field("context", &R::TYPE_ID)
            .field("shape", &self.shape())
            .field("mpi_rank", &self.world.mpi_rank())
            .field("world_size", &self.world.size())
            .field("local_memory_bytes", &self.local_memory_bytes())
            .finish_non_exhaustive()
    }
}

impl Add<&DistributedMatrix<FiniteField>> for &DistributedMatrix<FiniteField> {
    type Output = Result<DistributedMatrix<FiniteField>, DistributedError>;

    fn add(self, rhs: &DistributedMatrix<FiniteField>) -> Self::Output {
        DistributedMatrix::add(self, rhs)
    }
}

impl Sub<&DistributedMatrix<FiniteField>> for &DistributedMatrix<FiniteField> {
    type Output = Result<DistributedMatrix<FiniteField>, DistributedError>;

    fn sub(self, rhs: &DistributedMatrix<FiniteField>) -> Self::Output {
        DistributedMatrix::sub(self, rhs)
    }
}

impl Mul<&DistributedMatrix<FiniteField>> for &DistributedMatrix<FiniteField> {
    type Output = Result<DistributedMatrix<FiniteField>, DistributedError>;

    fn mul(self, rhs: &DistributedMatrix<FiniteField>) -> Self::Output {
        self.matmul(rhs)
    }
}

/// Owned distributed PLUQ decomposition with exact convention `P L U Q = A`.
#[derive(Clone, Debug)]
pub struct DistributedPLUQFactorization<R: Ring> {
    p: DistributedMatrix<R>,
    l: DistributedMatrix<R>,
    u: DistributedMatrix<R>,
    q: DistributedMatrix<R>,
    rank: usize,
}

impl<R: Ring> DistributedPLUQFactorization<R> {
    /// Distributed row-permutation factor.
    #[must_use]
    pub const fn p(&self) -> &DistributedMatrix<R> {
        &self.p
    }

    /// Distributed unit-lower factor.
    #[must_use]
    pub const fn l(&self) -> &DistributedMatrix<R> {
        &self.l
    }

    /// Distributed upper factor.
    #[must_use]
    pub const fn u(&self) -> &DistributedMatrix<R> {
        &self.u
    }

    /// Distributed column-permutation factor.
    #[must_use]
    pub const fn q(&self) -> &DistributedMatrix<R> {
        &self.q
    }

    /// Mathematical rank/pivot count.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }
}

impl DistributedPLUQFactorization<FiniteField> {
    /// Collectively verify `P L U Q = source` without gathering a factor.
    pub fn verify(
        &self,
        source: &DistributedMatrix<FiniteField>,
    ) -> Result<bool, DistributedError> {
        source.require_collective_same_field(&self.p)?;
        let lu = self.l.matmul(&self.u)?;
        let plu = self.p.matmul(&lu)?;
        plu.matmul(&self.q)?.equals(source)
    }
}

fn is_f2(field: &FiniteField) -> bool {
    field.degree() == 1 && field.characteristic() == &Integer::from(2_u64)
}

fn require_collective_f2(world: &World, field: &FiniteField) -> Result<(), DistributedError> {
    if world.inner.collective_all(is_f2(field)) {
        Ok(())
    } else {
        Err(DistributedError::UnsupportedCoefficientContext {
            context: field.to_string(),
        })
    }
}
