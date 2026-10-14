// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Private adapters for the retained pure-Rust dense matrix kernels.

use m4rst::{MatrixWrite, Mzd};
use m4rste::{Gf2e, Mzed};
use pwvla::{Mat, Z2n};
use std::sync::Arc;

use super::LinalgError;

/// One authoritative bit-packed payload for a local dense F_2 matrix.
pub(super) struct F2Matrix {
    inner: Arc<Mzd>,
}

impl F2Matrix {
    fn new(inner: Mzd) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    #[cfg(feature = "distributed")]
    pub(super) fn from_inner(inner: Mzd) -> Self {
        Self::new(inner)
    }

    #[cfg(feature = "distributed")]
    pub(super) fn packed_host(&self) -> Arc<Mzd> {
        self.inner.clone()
    }

    pub(super) fn rows(&self) -> usize {
        self.inner.rows()
    }

    pub(super) fn cols(&self) -> usize {
        self.inner.cols()
    }

    pub(super) fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(Mzd::new(rows, cols))
    }

    pub(super) fn identity(size: usize) -> Self {
        Self::new(Mzd::identity(size))
    }

    pub(super) fn from_values(
        rows: usize,
        cols: usize,
        values: impl ExactSizeIterator<Item = u64>,
    ) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut inner = Mzd::new(rows, cols);
        for (index, value) in values.enumerate() {
            inner.set(index / cols, index % cols, (value & 1) as u8);
        }
        Self::new(inner)
    }

    pub(super) fn get(&self, row: usize, col: usize) -> u64 {
        self.inner.get(row, col) as u64
    }

    pub(super) fn materialized_values(&self) -> Vec<u64> {
        (0..self.inner.rows())
            .flat_map(|row| {
                (0..self.inner.cols()).map(move |col| u64::from(self.inner.get(row, col)))
            })
            .collect()
    }

    pub(super) fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        let mut output = self.inner.as_ref().clone();
        output.xor_into(rhs.inner.as_ref());
        Ok(Self::new(output))
    }

    pub(super) fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.add(rhs)
    }

    pub(super) fn neg(&self) -> Self {
        self.clone()
    }

    pub(super) fn scalar_mul(&self, scalar: u64) -> Result<Self, LinalgError> {
        if scalar & 1 == 0 {
            Ok(Self::new(Mzd::new(self.rows(), self.cols())))
        } else {
            Ok(self.clone())
        }
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        Ok(Self::new(crate::concurrency::install(|| {
            Mzd::matmul(self.inner.as_ref(), rhs.inner.as_ref())
        })))
    }

    pub(super) fn transpose(&self) -> Self {
        Self::new(self.inner.transpose())
    }

    pub(super) fn with_zero_rows(&self, rows: usize) -> Self {
        debug_assert!(rows >= self.rows());
        if rows == self.rows() {
            return self.clone();
        }
        let mut inner = Mzd::new(rows, self.cols());
        inner.paste_into(0, 0, self.inner.as_ref());
        Self::new(inner)
    }

    pub(super) fn rank(&self) -> usize {
        self.inner.ple_rank()
    }

    pub(super) fn rref(&self) -> (Self, usize) {
        let mut inner = self.inner.as_ref().clone();
        let rank = inner.echelonize(true);
        (Self::new(inner), rank)
    }

    pub(super) fn solve_left(&self, rhs: &Self) -> Option<Self> {
        Mzd::solve_left(self.inner.as_ref(), rhs.inner.as_ref()).map(Self::new)
    }

    pub(super) fn nullspace(&self) -> Self {
        Self::new(Mzd::nullspace(self.inner.as_ref()))
    }

    pub(super) fn pluq(&self) -> (Self, Self, Self, Self, usize) {
        let (p, l, u, q, rank) = self.inner.pluq();
        (Self::new(p), Self::new(l), Self::new(u), Self::new(q), rank)
    }

    #[cfg(test)]
    pub(super) fn allocated_words(&self) -> usize {
        self.inner.as_words().len()
    }
}

impl Clone for F2Matrix {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

pub(super) struct M4rsteMatrix {
    inner: Mzed,
}

impl M4rsteMatrix {
    pub(super) fn rows(&self) -> usize {
        self.inner.rows()
    }

    pub(super) fn cols(&self) -> usize {
        self.inner.cols()
    }

    pub(super) fn zeros(minimal_polynomial: u64, rows: usize, cols: usize) -> Self {
        Self {
            inner: Mzed::zeros(Gf2e::new(minimal_polynomial), rows, cols),
        }
    }

    pub(super) fn identity(minimal_polynomial: u64, size: usize) -> Self {
        Self {
            inner: Mzed::identity(Gf2e::new(minimal_polynomial), size),
        }
    }

    pub(super) fn from_values(
        minimal_polynomial: u64,
        rows: usize,
        cols: usize,
        values: impl ExactSizeIterator<Item = u16>,
    ) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut inner = Mzed::zeros(Gf2e::new(minimal_polynomial), rows, cols);
        for (index, value) in values.enumerate() {
            inner.set(index / cols, index % cols, value);
        }
        Self { inner }
    }

    pub(super) fn get(&self, row: usize, col: usize) -> u16 {
        self.inner.get(row, col)
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        Self {
            inner: Mzed::add(&self.inner, &rhs.inner),
        }
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }

    pub(super) fn neg(&self) -> Self {
        self.clone()
    }

    pub(super) fn scalar_mul(&self, scalar: u16) -> Self {
        let context = self.inner.ctx();
        let mut output = Mzed::zeros(context, self.inner.rows(), self.inner.cols());
        for row in 0..self.inner.rows() {
            for col in 0..self.inner.cols() {
                output.set(row, col, context.mul(self.inner.get(row, col), scalar));
            }
        }
        Self { inner: output }
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        Self {
            inner: Mzed::matmul(&self.inner, &rhs.inner),
        }
    }

    pub(super) fn transpose(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    pub(super) fn rank(&self) -> usize {
        self.inner.rank()
    }

    pub(super) fn rref(&self) -> (Self, usize) {
        let (inner, rank) = self.inner.rref();
        (Self { inner }, rank)
    }

    pub(super) fn solve_left(&self, rhs: &Self) -> Option<Self> {
        Mzed::solve_left(&self.inner, &rhs.inner).map(|inner| Self { inner })
    }

    pub(super) fn nullspace(&self) -> Self {
        Self {
            inner: Mzed::nullspace(&self.inner),
        }
    }

    pub(super) fn pluq(&self) -> (Self, Self, Self, Self, usize) {
        let (p, l, u, q, rank) = self.inner.pluq();
        (
            Self { inner: p },
            Self { inner: l },
            Self { inner: u },
            Self { inner: q },
            rank,
        )
    }
}

impl Clone for M4rsteMatrix {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

enum PwvlaBackend {
    U8(Mat<u8>),
    U16(Mat<u16>),
    U32(Mat<u32>),
    U64(Mat<u64>),
}

pub(super) struct PwvlaMatrix {
    backend: PwvlaBackend,
    modulus: u64,
}

impl PwvlaMatrix {
    pub(super) fn rows(&self) -> usize {
        match &self.backend {
            PwvlaBackend::U8(matrix) => matrix.rows,
            PwvlaBackend::U16(matrix) => matrix.rows,
            PwvlaBackend::U32(matrix) => matrix.rows,
            PwvlaBackend::U64(matrix) => matrix.rows,
        }
    }

    pub(super) fn cols(&self) -> usize {
        match &self.backend {
            PwvlaBackend::U8(matrix) => matrix.cols,
            PwvlaBackend::U16(matrix) => matrix.cols,
            PwvlaBackend::U32(matrix) => matrix.cols,
            PwvlaBackend::U64(matrix) => matrix.cols,
        }
    }

    pub(super) fn zeros(rows: usize, cols: usize, modulus: u64) -> Self {
        Self::from_values(rows, cols, modulus, std::iter::repeat_n(0, rows * cols))
    }

    pub(super) fn identity(size: usize, modulus: u64) -> Self {
        Self::from_values(
            size,
            size,
            modulus,
            (0..size * size).map(|index| u64::from(index / size == index % size)),
        )
    }

    pub(super) fn from_values(
        rows: usize,
        cols: usize,
        modulus: u64,
        values: impl ExactSizeIterator<Item = u64>,
    ) -> Self {
        debug_assert!(modulus.is_power_of_two() && modulus > 2);
        debug_assert_eq!(values.len(), rows * cols);
        let bits = modulus.trailing_zeros();
        let backend = match bits {
            0..=8 => PwvlaBackend::U8(Mat::from_values(rows, cols, values.map(u8::from_u64))),
            9..=16 => PwvlaBackend::U16(Mat::from_values(rows, cols, values.map(u16::from_u64))),
            17..=32 => PwvlaBackend::U32(Mat::from_values(rows, cols, values.map(u32::from_u64))),
            33..=64 => PwvlaBackend::U64(Mat::from_values(rows, cols, values.map(u64::from_u64))),
            _ => unreachable!("a u64 power of two uses at most 64 bits"),
        };
        Self { backend, modulus }
    }

    pub(super) fn get(&self, row: usize, col: usize) -> u64 {
        let value = match &self.backend {
            PwvlaBackend::U8(matrix) => matrix.get(row, col).to_u64(),
            PwvlaBackend::U16(matrix) => matrix.get(row, col).to_u64(),
            PwvlaBackend::U32(matrix) => matrix.get(row, col).to_u64(),
            PwvlaBackend::U64(matrix) => matrix.get(row, col).to_u64(),
        };
        value % self.modulus
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        self.map_binary(rhs, Mat::add, Mat::add, Mat::add, Mat::add)
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        self.map_binary(rhs, Mat::sub, Mat::sub, Mat::sub, Mat::sub)
    }

    pub(super) fn neg(&self) -> Self {
        let backend = match &self.backend {
            PwvlaBackend::U8(matrix) => PwvlaBackend::U8(matrix.neg()),
            PwvlaBackend::U16(matrix) => PwvlaBackend::U16(matrix.neg()),
            PwvlaBackend::U32(matrix) => PwvlaBackend::U32(matrix.neg()),
            PwvlaBackend::U64(matrix) => PwvlaBackend::U64(matrix.neg()),
        };
        Self {
            backend,
            modulus: self.modulus,
        }
    }

    pub(super) fn scalar_mul(&self, scalar: u64) -> Self {
        let backend = match &self.backend {
            PwvlaBackend::U8(matrix) => PwvlaBackend::U8(matrix.scalar_mul(u8::from_u64(scalar))),
            PwvlaBackend::U16(matrix) => {
                PwvlaBackend::U16(matrix.scalar_mul(u16::from_u64(scalar)))
            }
            PwvlaBackend::U32(matrix) => {
                PwvlaBackend::U32(matrix.scalar_mul(u32::from_u64(scalar)))
            }
            PwvlaBackend::U64(matrix) => {
                PwvlaBackend::U64(matrix.scalar_mul(u64::from_u64(scalar)))
            }
        };
        Self {
            backend,
            modulus: self.modulus,
        }
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        self.map_binary(rhs, matmul_u8, matmul_u16, matmul_u32, matmul_u64)
    }

    pub(super) fn transpose(&self) -> Self {
        let backend = match &self.backend {
            PwvlaBackend::U8(matrix) => PwvlaBackend::U8(matrix.transpose()),
            PwvlaBackend::U16(matrix) => PwvlaBackend::U16(matrix.transpose()),
            PwvlaBackend::U32(matrix) => PwvlaBackend::U32(matrix.transpose()),
            PwvlaBackend::U64(matrix) => PwvlaBackend::U64(matrix.transpose()),
        };
        Self {
            backend,
            modulus: self.modulus,
        }
    }

    fn map_binary(
        &self,
        rhs: &Self,
        u8_operation: fn(&Mat<u8>, &Mat<u8>) -> Mat<u8>,
        u16_operation: fn(&Mat<u16>, &Mat<u16>) -> Mat<u16>,
        u32_operation: fn(&Mat<u32>, &Mat<u32>) -> Mat<u32>,
        u64_operation: fn(&Mat<u64>, &Mat<u64>) -> Mat<u64>,
    ) -> Self {
        let backend = match (&self.backend, &rhs.backend) {
            (PwvlaBackend::U8(left), PwvlaBackend::U8(right)) => {
                PwvlaBackend::U8(u8_operation(left, right))
            }
            (PwvlaBackend::U16(left), PwvlaBackend::U16(right)) => {
                PwvlaBackend::U16(u16_operation(left, right))
            }
            (PwvlaBackend::U32(left), PwvlaBackend::U32(right)) => {
                PwvlaBackend::U32(u32_operation(left, right))
            }
            (PwvlaBackend::U64(left), PwvlaBackend::U64(right)) => {
                PwvlaBackend::U64(u64_operation(left, right))
            }
            _ => unreachable!("equal power-of-two moduli select the same pwvla cell type"),
        };
        Self {
            backend,
            modulus: self.modulus,
        }
    }
}

impl Clone for PwvlaMatrix {
    fn clone(&self) -> Self {
        let backend = match &self.backend {
            PwvlaBackend::U8(matrix) => PwvlaBackend::U8(matrix.clone()),
            PwvlaBackend::U16(matrix) => PwvlaBackend::U16(matrix.clone()),
            PwvlaBackend::U32(matrix) => PwvlaBackend::U32(matrix.clone()),
            PwvlaBackend::U64(matrix) => PwvlaBackend::U64(matrix.clone()),
        };
        Self {
            backend,
            modulus: self.modulus,
        }
    }
}

fn matmul_u8(left: &Mat<u8>, right: &Mat<u8>) -> Mat<u8> {
    pwvla::matmul_u8(left, right)
}

fn matmul_u16(left: &Mat<u16>, right: &Mat<u16>) -> Mat<u16> {
    pwvla::matmul_u16(left, right)
}

fn matmul_u32(left: &Mat<u32>, right: &Mat<u32>) -> Mat<u32> {
    pwvla::matmul_u32(left, right)
}

fn matmul_u64(left: &Mat<u64>, right: &Mat<u64>) -> Mat<u64> {
    pwvla::matmul_u64(left, right)
}
