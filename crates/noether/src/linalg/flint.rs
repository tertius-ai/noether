// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Private whole-matrix FLINT storage for the canonical exact contexts.

#![allow(unsafe_code)]

use std::mem::MaybeUninit;

use flint_sys::{
    _perm_parity, acb_mat_add, acb_mat_clear, acb_mat_init, acb_mat_mul, acb_mat_neg, acb_mat_one,
    acb_mat_scalar_mul_acb, acb_mat_set, acb_mat_struct, acb_mat_sub, acb_mat_transpose, acb_set,
    arb_mat_add, arb_mat_clear, arb_mat_init, arb_mat_mul, arb_mat_neg, arb_mat_one,
    arb_mat_scalar_mul_arb, arb_mat_set, arb_mat_struct, arb_mat_sub, arb_mat_transpose, arb_set,
    fmpq, fmpq_mat_add, fmpq_mat_can_solve_multi_mod, fmpq_mat_charpoly, fmpq_mat_clear,
    fmpq_mat_det, fmpq_mat_init, fmpq_mat_mul, fmpq_mat_neg, fmpq_mat_rref,
    fmpq_mat_scalar_mul_fmpq, fmpq_mat_set, fmpq_mat_struct, fmpq_mat_sub, fmpq_mat_transpose,
    fmpq_poly_clear, fmpq_poly_get_coeff_fmpq, fmpq_poly_init, fmpq_poly_struct, fmpq_set,
    fmpz_mat_add, fmpz_mat_charpoly, fmpz_mat_clear, fmpz_mat_det, fmpz_mat_entry, fmpz_mat_hnf,
    fmpz_mat_hnf_transform, fmpz_mat_init, fmpz_mat_mul, fmpz_mat_neg, fmpz_mat_rank,
    fmpz_mat_scalar_mul_fmpz, fmpz_mat_set, fmpz_mat_snf, fmpz_mat_snf_transform, fmpz_mat_struct,
    fmpz_mat_sub, fmpz_mat_transpose, fmpz_mod_mat_add, fmpz_mod_mat_charpoly, fmpz_mod_mat_clear,
    fmpz_mod_mat_det, fmpz_mod_mat_get_entry, fmpz_mod_mat_howell_form, fmpz_mod_mat_init,
    fmpz_mod_mat_mul, fmpz_mod_mat_neg, fmpz_mod_mat_rank, fmpz_mod_mat_rref,
    fmpz_mod_mat_scalar_mul_fmpz, fmpz_mod_mat_set, fmpz_mod_mat_set_entry, fmpz_mod_mat_struct,
    fmpz_mod_mat_sub, fmpz_mod_mat_transpose, fmpz_mod_poly_clear, fmpz_mod_poly_get_coeff_fmpz,
    fmpz_mod_poly_init, fmpz_mod_poly_struct, fmpz_poly_clear, fmpz_poly_get_coeff_fmpz,
    fmpz_poly_init, fmpz_poly_struct, fmpz_set, fq_default_mat_add, fq_default_mat_charpoly,
    fq_default_mat_clear, fq_default_mat_entry, fq_default_mat_entry_set, fq_default_mat_init,
    fq_default_mat_lu, fq_default_mat_mul, fq_default_mat_neg, fq_default_mat_rank,
    fq_default_mat_rref, fq_default_mat_scalar_mul, fq_default_mat_set, fq_default_mat_struct,
    fq_default_mat_sub, fq_default_mat_transpose, fq_default_mul, fq_default_neg, fq_default_one,
    fq_default_poly_clear, fq_default_poly_get_coeff, fq_default_poly_init, fq_default_poly_struct,
    nmod_mat_add, nmod_mat_charpoly, nmod_mat_clear, nmod_mat_det, nmod_mat_get_entry,
    nmod_mat_howell_form, nmod_mat_init, nmod_mat_mul, nmod_mat_neg, nmod_mat_rank, nmod_mat_rref,
    nmod_mat_scalar_mul, nmod_mat_set, nmod_mat_set_entry, nmod_mat_struct, nmod_mat_sub,
    nmod_mat_transpose, nmod_poly_clear, nmod_poly_get_coeff_ui, nmod_poly_init, nmod_poly_struct,
    padic_mat_add, padic_mat_clear, padic_mat_get_entry_padic, padic_mat_init2, padic_mat_mul,
    padic_mat_neg, padic_mat_one, padic_mat_scalar_mul_padic, padic_mat_set,
    padic_mat_set_entry_padic, padic_mat_struct, padic_mat_sub, padic_mat_transpose, slong, ulong,
};

use crate::commalg::{ResidueClass, ResidueRing};
use crate::numthy::{
    DefaultExtensionElement, FiniteField, FiniteFieldElement, PadicElement, PadicField, PadicRing,
};
use crate::{ComplexBall, ComplexBallField, Integer, Rational, RealBall, RealBallField};

use super::kernels::{F2Matrix, M4rsteMatrix, PwvlaMatrix};
use super::LinalgError;

#[cfg(test)]
#[derive(Clone, Copy)]
enum DenseConversion {
    F2Terminal,
    PowerOfTwoTerminal,
    PowerOfTwoPersistent,
    BinaryExtensionTerminal,
}

#[cfg(test)]
thread_local! {
    static DENSE_CONVERSIONS: std::cell::Cell<[usize; 4]> = const {
        std::cell::Cell::new([0; 4])
    };
}

#[cfg(test)]
fn record_dense_conversion(kind: DenseConversion) {
    DENSE_CONVERSIONS.with(|counts| {
        let mut next = counts.get();
        next[kind as usize] += 1;
        counts.set(next);
    });
}

pub(super) struct IntegerMatrix {
    inner: fmpz_mat_struct,
    rows: usize,
    cols: usize,
}

impl IntegerMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        let rows_ffi = slong::try_from(rows).expect("matrix row count was validated");
        let cols_ffi = slong::try_from(cols).expect("matrix column count was validated");
        let mut inner = MaybeUninit::<fmpz_mat_struct>::uninit();
        unsafe { fmpz_mat_init(inner.as_mut_ptr(), rows_ffi, cols_ffi) };
        Self {
            inner: unsafe { inner.assume_init() },
            rows,
            cols,
        }
    }

    pub(super) fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub(super) fn identity(size: usize) -> Self {
        let matrix = Self::new(size, size);
        let one = Integer::one();
        for index in 0..size {
            unsafe { fmpz_set(matrix.entry_ptr(index, index), one.as_ptr()) };
        }
        matrix
    }

    pub(super) fn from_entries<'a>(
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a Integer>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let matrix = Self::new(rows, cols);
        for (index, entry) in entries.enumerate() {
            let row = index / cols;
            let col = index % cols;
            unsafe { fmpz_set(matrix.entry_ptr(row, col), entry.as_ptr()) };
        }
        matrix
    }

    pub(super) fn get(&self, row: usize, col: usize) -> Integer {
        unsafe { Integer::clone_from_ptr(self.entry_ptr(row, col)) }
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_add(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_sub(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn scalar_mul(&self, scalar: &Integer) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_scalar_mul_fmpz(output.as_mut_ptr(), self.as_ptr(), scalar.as_ptr()) };
        output
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, rhs.cols);
        unsafe { fmpz_mat_mul(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn transpose(&self) -> Self {
        let mut output = Self::new(self.cols, self.rows);
        unsafe { fmpz_mat_transpose(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn smith_normal_form(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_snf(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn hermite_normal_form(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_hnf(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn hermite_normal_form_with_transform(&self) -> (Self, Self) {
        let mut form = Self::new(self.rows, self.cols);
        let mut left = Self::new(self.rows, self.rows);
        unsafe {
            fmpz_mat_hnf_transform(form.as_mut_ptr(), left.as_mut_ptr(), self.as_ptr());
        }
        (form, left)
    }

    pub(super) fn smith_normal_form_with_transforms(&self) -> (Self, Self, Self) {
        let mut form = Self::new(self.rows, self.cols);
        let mut left = Self::new(self.rows, self.rows);
        let mut right = Self::new(self.cols, self.cols);
        unsafe {
            fmpz_mat_snf_transform(
                form.as_mut_ptr(),
                left.as_mut_ptr(),
                right.as_mut_ptr(),
                self.as_ptr(),
            )
        };
        (form, left, right)
    }

    pub(super) fn determinant(&self) -> Integer {
        debug_assert_eq!(self.rows, self.cols);
        let mut output = Integer::zero();
        unsafe { fmpz_mat_det(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn characteristic_polynomial(&self) -> Vec<Integer> {
        debug_assert_eq!(self.rows, self.cols);
        let mut polynomial = IntegerPolynomial::new();
        unsafe { fmpz_mat_charpoly(polynomial.as_mut_ptr(), self.as_ptr()) };
        (0..=self.rows)
            .map(|degree| polynomial.coefficient(degree))
            .collect()
    }

    pub(super) fn rank(&self) -> usize {
        unsafe { fmpz_mat_rank(self.as_ptr()) as usize }
    }

    fn entry_ptr(&self, row: usize, col: usize) -> *mut flint_sys::fmpz {
        unsafe { fmpz_mat_entry(self.as_ptr(), row as slong, col as slong) }
    }

    fn as_ptr(&self) -> *const fmpz_mat_struct {
        &self.inner
    }

    fn as_mut_ptr(&mut self) -> *mut fmpz_mat_struct {
        &mut self.inner
    }
}

struct NmodMatrix {
    inner: nmod_mat_struct,
    rows: usize,
    cols: usize,
    modulus: u64,
}

impl NmodMatrix {
    fn new(rows: usize, cols: usize, modulus: u64) -> Self {
        let rows = slong::try_from(rows).expect("matrix row count was validated");
        let cols = slong::try_from(cols).expect("matrix column count was validated");
        let mut inner = MaybeUninit::<nmod_mat_struct>::uninit();
        unsafe { nmod_mat_init(inner.as_mut_ptr(), rows, cols, modulus as ulong) };
        Self {
            inner: unsafe { inner.assume_init() },
            rows: rows as usize,
            cols: cols as usize,
            modulus,
        }
    }

    fn identity(size: usize, modulus: u64) -> Self {
        let mut output = Self::new(size, size, modulus);
        for index in 0..size {
            output.set(index, index, 1);
        }
        output
    }

    fn from_values(
        rows: usize,
        cols: usize,
        modulus: u64,
        values: impl ExactSizeIterator<Item = u64>,
    ) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut output = Self::new(rows, cols, modulus);
        for (index, value) in values.enumerate() {
            output.set(index / cols, index % cols, value);
        }
        output
    }

    fn set(&mut self, row: usize, column: usize, value: u64) {
        unsafe {
            nmod_mat_set_entry(
                &mut self.inner,
                row as slong,
                column as slong,
                value as ulong,
            )
        };
    }

    fn get(&self, row: usize, column: usize) -> u64 {
        unsafe { nmod_mat_get_entry(&self.inner, row as slong, column as slong) as u64 }
    }

    fn with_zero_rows(&self, rows: usize) -> Self {
        debug_assert!(rows >= self.rows);
        if rows == self.rows {
            return self.clone();
        }
        let mut output = Self::new(rows, self.cols, self.modulus);
        for row in 0..self.rows {
            for col in 0..self.cols {
                output.set(row, col, self.get(row, col));
            }
        }
        output
    }

    fn howell(&mut self) {
        unsafe { nmod_mat_howell_form(&mut self.inner) };
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols, self.modulus);
        unsafe { nmod_mat_add(&mut output.inner, &self.inner, &rhs.inner) };
        output
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols, self.modulus);
        unsafe { nmod_mat_sub(&mut output.inner, &self.inner, &rhs.inner) };
        output
    }

    fn neg(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols, self.modulus);
        unsafe { nmod_mat_neg(&mut output.inner, &self.inner) };
        output
    }

    fn scalar_mul(&self, scalar: u64) -> Self {
        let mut output = Self::new(self.rows, self.cols, self.modulus);
        unsafe { nmod_mat_scalar_mul(&mut output.inner, &self.inner, scalar as ulong) };
        output
    }

    fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, rhs.cols, self.modulus);
        unsafe { nmod_mat_mul(&mut output.inner, &self.inner, &rhs.inner) };
        output
    }

    fn transpose(&self) -> Self {
        let mut output = Self::new(self.cols, self.rows, self.modulus);
        unsafe { nmod_mat_transpose(&mut output.inner, &self.inner) };
        output
    }

    fn rank(&self) -> usize {
        unsafe { nmod_mat_rank(&self.inner) as usize }
    }

    fn rref(&self) -> (Self, usize) {
        let mut output = self.clone();
        let rank = unsafe { nmod_mat_rref(&mut output.inner) } as usize;
        (output, rank)
    }

    fn determinant(&self) -> u64 {
        unsafe { nmod_mat_det(&self.inner) as u64 }
    }

    fn characteristic_polynomial(&self) -> Vec<Integer> {
        debug_assert_eq!(self.rows, self.cols);
        let mut polynomial = NmodPolynomial::new(self.modulus);
        unsafe { nmod_mat_charpoly(&mut polynomial.inner, &self.inner) };
        (0..=self.rows)
            .map(|degree| Integer::from(polynomial.coefficient(degree)))
            .collect()
    }
}

impl Clone for NmodMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols, self.modulus);
        unsafe { nmod_mat_set(&mut output.inner, &self.inner) };
        output
    }
}

impl Drop for NmodMatrix {
    fn drop(&mut self) {
        unsafe { nmod_mat_clear(&mut self.inner) };
    }
}

// SAFETY: an nmod matrix owns its allocation; shared operations read it and
// every mutating FLINT call targets a fresh output matrix.
unsafe impl Send for NmodMatrix {}
unsafe impl Sync for NmodMatrix {}

struct NmodPolynomial {
    inner: nmod_poly_struct,
}

impl NmodPolynomial {
    fn new(modulus: u64) -> Self {
        let mut inner = MaybeUninit::<nmod_poly_struct>::uninit();
        unsafe { nmod_poly_init(inner.as_mut_ptr(), modulus as ulong) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn coefficient(&self, degree: usize) -> u64 {
        unsafe { nmod_poly_get_coeff_ui(&self.inner, degree as slong) as u64 }
    }
}

impl Drop for NmodPolynomial {
    fn drop(&mut self) {
        unsafe { nmod_poly_clear(&mut self.inner) };
    }
}

struct FmpzModMatrix {
    inner: fmpz_mod_mat_struct,
    ring: ResidueRing,
    rows: usize,
    cols: usize,
}

impl FmpzModMatrix {
    fn new(ring: ResidueRing, rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<fmpz_mod_mat_struct>::uninit();
        unsafe {
            fmpz_mod_mat_init(
                inner.as_mut_ptr(),
                rows as slong,
                cols as slong,
                ring.raw_context_ptr(),
            )
        };
        Self {
            inner: unsafe { inner.assume_init() },
            ring,
            rows,
            cols,
        }
    }

    fn identity(ring: ResidueRing, size: usize) -> Self {
        let mut output = Self::new(ring.clone(), size, size);
        let one = Integer::one();
        for index in 0..size {
            output.set(index, index, &one);
        }
        output
    }

    fn from_values<'a>(
        ring: ResidueRing,
        rows: usize,
        cols: usize,
        values: impl ExactSizeIterator<Item = &'a Integer>,
    ) -> Self {
        debug_assert_eq!(values.len(), rows * cols);
        let mut output = Self::new(ring, rows, cols);
        for (index, value) in values.enumerate() {
            output.set(index / cols, index % cols, value);
        }
        output
    }

    fn set(&mut self, row: usize, col: usize, value: &Integer) {
        unsafe {
            fmpz_mod_mat_set_entry(
                &mut self.inner,
                row as slong,
                col as slong,
                value.as_ptr(),
                self.ring.raw_context_ptr(),
            )
        };
    }

    fn get(&self, row: usize, col: usize) -> Integer {
        let mut output = Integer::zero();
        unsafe {
            fmpz_mod_mat_get_entry(
                output.as_mut_ptr(),
                &self.inner,
                row as slong,
                col as slong,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn with_zero_rows(&self, rows: usize) -> Self {
        debug_assert!(rows >= self.rows);
        if rows == self.rows {
            return self.clone();
        }
        let mut output = Self::new(self.ring.clone(), rows, self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                output.set(row, col, &self.get(row, col));
            }
        }
        output
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        unsafe {
            fmpz_mod_mat_add(
                &mut output.inner,
                &self.inner,
                &rhs.inner,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        unsafe {
            fmpz_mod_mat_sub(
                &mut output.inner,
                &self.inner,
                &rhs.inner,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn neg(&self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        unsafe { fmpz_mod_mat_neg(&mut output.inner, &self.inner, self.ring.raw_context_ptr()) };
        output
    }

    fn scalar_mul(&self, scalar: &Integer) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        // FLINT's C signature takes a mutable `fmpz_t` even though scalar
        // multiplication treats the coefficient as an input. Keep the public
        // scalar immutable and give FLINT a private owned copy.
        let mut scalar = scalar.clone();
        unsafe {
            fmpz_mod_mat_scalar_mul_fmpz(
                &mut output.inner,
                &self.inner,
                scalar.as_mut_ptr(),
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, rhs.cols);
        unsafe {
            fmpz_mod_mat_mul(
                &mut output.inner,
                &self.inner,
                &rhs.inner,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn transpose(&self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.cols, self.rows);
        unsafe {
            fmpz_mod_mat_transpose(&mut output.inner, &self.inner, self.ring.raw_context_ptr())
        };
        output
    }

    fn howell(&self) -> Self {
        let mut output = self.clone();
        unsafe { fmpz_mod_mat_howell_form(&mut output.inner, self.ring.raw_context_ptr()) };
        output
    }

    fn rank(&self) -> usize {
        unsafe { fmpz_mod_mat_rank(&self.inner, self.ring.raw_context_ptr()) as usize }
    }

    fn rref(&self) -> (Self, usize) {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        let rank = unsafe {
            fmpz_mod_mat_rref(&mut output.inner, &self.inner, self.ring.raw_context_ptr())
        } as usize;
        (output, rank)
    }

    fn determinant(&self) -> Integer {
        let mut output = Integer::zero();
        unsafe {
            fmpz_mod_mat_det(
                output.as_mut_ptr(),
                &self.inner,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }

    fn characteristic_polynomial(&self) -> Vec<Integer> {
        debug_assert_eq!(self.rows, self.cols);
        let mut polynomial = FmpzModPolynomial::new(self.ring.clone());
        unsafe {
            fmpz_mod_mat_charpoly(
                &mut polynomial.inner,
                &self.inner,
                self.ring.raw_context_ptr(),
            )
        };
        (0..=self.rows)
            .map(|degree| polynomial.coefficient(degree))
            .collect()
    }
}

impl Clone for FmpzModMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.ring.clone(), self.rows, self.cols);
        unsafe { fmpz_mod_mat_set(&mut output.inner, &self.inner, self.ring.raw_context_ptr()) };
        output
    }
}

impl Drop for FmpzModMatrix {
    fn drop(&mut self) {
        unsafe { fmpz_mod_mat_clear(&mut self.inner, self.ring.raw_context_ptr()) };
    }
}

unsafe impl Send for FmpzModMatrix {}
unsafe impl Sync for FmpzModMatrix {}

struct FmpzModPolynomial {
    inner: fmpz_mod_poly_struct,
    ring: ResidueRing,
}

impl FmpzModPolynomial {
    fn new(ring: ResidueRing) -> Self {
        let mut inner = MaybeUninit::<fmpz_mod_poly_struct>::uninit();
        unsafe { fmpz_mod_poly_init(inner.as_mut_ptr(), ring.raw_context_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
            ring,
        }
    }

    fn coefficient(&self, degree: usize) -> Integer {
        let mut output = Integer::zero();
        unsafe {
            fmpz_mod_poly_get_coeff_fmpz(
                output.as_mut_ptr(),
                &self.inner,
                degree as slong,
                self.ring.raw_context_ptr(),
            )
        };
        output
    }
}

impl Drop for FmpzModPolynomial {
    fn drop(&mut self) {
        unsafe { fmpz_mod_poly_clear(&mut self.inner, self.ring.raw_context_ptr()) };
    }
}

enum ModularMatrixStorage {
    F2(F2Matrix),
    PowerOfTwo(PwvlaMatrix),
    Word(NmodMatrix),
    Arbitrary(FmpzModMatrix),
}

pub(super) struct ModularMatrix {
    ring: ResidueRing,
    backend: ModularMatrixStorage,
}

impl ModularMatrix {
    fn rows(&self) -> usize {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => matrix.rows(),
            ModularMatrixStorage::PowerOfTwo(matrix) => matrix.rows(),
            ModularMatrixStorage::Word(matrix) => matrix.rows,
            ModularMatrixStorage::Arbitrary(matrix) => matrix.rows,
        }
    }

    fn cols(&self) -> usize {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => matrix.cols(),
            ModularMatrixStorage::PowerOfTwo(matrix) => matrix.cols(),
            ModularMatrixStorage::Word(matrix) => matrix.cols,
            ModularMatrixStorage::Arbitrary(matrix) => matrix.cols,
        }
    }

    pub(super) fn zeros(ring: ResidueRing, rows: usize, cols: usize) -> Self {
        let backend = match ring.modulus().to_u64_checked() {
            Some(2) => ModularMatrixStorage::F2(F2Matrix::zeros(rows, cols)),
            Some(modulus) if modulus.is_power_of_two() => {
                ModularMatrixStorage::PowerOfTwo(PwvlaMatrix::zeros(rows, cols, modulus))
            }
            Some(modulus) => ModularMatrixStorage::Word(NmodMatrix::new(rows, cols, modulus)),
            None => ModularMatrixStorage::Arbitrary(FmpzModMatrix::new(ring.clone(), rows, cols)),
        };
        Self { ring, backend }
    }

    #[cfg(feature = "distributed")]
    pub(super) fn from_packed_f2(ring: ResidueRing, matrix: m4rst::Mzd) -> Option<Self> {
        (ring.modulus().to_u64_checked() == Some(2)).then(|| Self {
            ring,
            backend: ModularMatrixStorage::F2(F2Matrix::from_inner(matrix)),
        })
    }

    #[cfg(feature = "distributed")]
    pub(super) fn packed_f2(&self) -> Option<std::sync::Arc<m4rst::Mzd>> {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => Some(matrix.packed_host()),
            _ => None,
        }
    }

    pub(super) fn identity(ring: ResidueRing, size: usize) -> Self {
        let backend = match ring.modulus().to_u64_checked() {
            Some(2) => ModularMatrixStorage::F2(F2Matrix::identity(size)),
            Some(modulus) if modulus.is_power_of_two() => {
                ModularMatrixStorage::PowerOfTwo(PwvlaMatrix::identity(size, modulus))
            }
            Some(modulus) => ModularMatrixStorage::Word(NmodMatrix::identity(size, modulus)),
            None => ModularMatrixStorage::Arbitrary(FmpzModMatrix::identity(ring.clone(), size)),
        };
        Self { ring, backend }
    }

    pub(super) fn from_entries<'a>(
        ring: ResidueRing,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a ResidueClass>,
    ) -> Self {
        Self::from_integer_entries(ring, rows, cols, entries.map(ResidueClass::representative))
    }

    pub(super) fn from_integer_entries<'a>(
        ring: ResidueRing,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a Integer>,
    ) -> Self {
        let backend = match ring.modulus().to_u64_checked() {
            Some(2) => ModularMatrixStorage::F2(F2Matrix::from_values(
                rows,
                cols,
                entries.map(|value| value.modulo_u64(2)),
            )),
            Some(modulus) if modulus.is_power_of_two() => {
                ModularMatrixStorage::PowerOfTwo(PwvlaMatrix::from_values(
                    rows,
                    cols,
                    modulus,
                    entries.map(|value| value.modulo_u64(modulus)),
                ))
            }
            Some(modulus) => ModularMatrixStorage::Word(NmodMatrix::from_values(
                rows,
                cols,
                modulus,
                entries.map(|value| value.modulo_u64(modulus)),
            )),
            None => ModularMatrixStorage::Arbitrary(FmpzModMatrix::from_values(
                ring.clone(),
                rows,
                cols,
                entries,
            )),
        };
        Self { ring, backend }
    }

    fn get_integer(&self, row: usize, col: usize) -> Integer {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => Integer::from(matrix.get(row, col)),
            ModularMatrixStorage::PowerOfTwo(matrix) => Integer::from(matrix.get(row, col)),
            ModularMatrixStorage::Word(matrix) => Integer::from(matrix.get(row, col)),
            ModularMatrixStorage::Arbitrary(matrix) => matrix.get(row, col),
        }
    }

    pub(super) fn get(&self, row: usize, col: usize) -> ResidueClass {
        self.ring.element(self.get_integer(row, col))
    }

    fn materialized_integers(&self) -> Vec<Integer> {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => matrix
                .materialized_values()
                .into_iter()
                .map(Integer::from)
                .collect(),
            _ => (0..self.cols() * self.rows())
                .map(|index| self.get_integer(index / self.cols(), index % self.cols()))
                .collect(),
        }
    }

    pub(super) fn materialized_values(&self) -> Vec<ResidueClass> {
        self.materialized_integers()
            .into_iter()
            .map(|value| self.ring.element(value))
            .collect()
    }

    pub(super) fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            F2Matrix::add,
            PwvlaMatrix::add,
            NmodMatrix::add,
            FmpzModMatrix::add,
        )
    }

    pub(super) fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            F2Matrix::sub,
            PwvlaMatrix::sub,
            NmodMatrix::sub,
            FmpzModMatrix::sub,
        )
    }

    pub(super) fn neg(&self) -> Self {
        let backend = match &self.backend {
            ModularMatrixStorage::F2(matrix) => ModularMatrixStorage::F2(matrix.neg()),
            ModularMatrixStorage::PowerOfTwo(matrix) => {
                ModularMatrixStorage::PowerOfTwo(matrix.neg())
            }
            ModularMatrixStorage::Word(matrix) => ModularMatrixStorage::Word(matrix.neg()),
            ModularMatrixStorage::Arbitrary(matrix) => {
                ModularMatrixStorage::Arbitrary(matrix.neg())
            }
        };
        Self {
            ring: self.ring.clone(),
            backend,
        }
    }

    /// Use a retained kernel's general rectangular solver without unpacking.
    ///
    /// The outer option reports whether this storage has such a kernel; the
    /// inner option reports whether the system is consistent.
    fn solve_left(&self, rhs: &Self) -> Option<Option<Self>> {
        match (&self.backend, &rhs.backend) {
            (ModularMatrixStorage::F2(left), ModularMatrixStorage::F2(right)) => {
                Some(left.solve_left(right).map(|matrix| Self {
                    ring: self.ring.clone(),
                    backend: ModularMatrixStorage::F2(matrix),
                }))
            }
            _ => None,
        }
    }

    fn nullspace(&self) -> Option<Self> {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => Some(Self {
                ring: self.ring.clone(),
                backend: ModularMatrixStorage::F2(matrix.nullspace()),
            }),
            _ => None,
        }
    }

    fn pluq(&self) -> Option<(Self, Self, Self, Self, usize)> {
        let ModularMatrixStorage::F2(matrix) = &self.backend else {
            return None;
        };
        let (p, l, u, q, rank) = matrix.pluq();
        let wrap = |matrix| Self {
            ring: self.ring.clone(),
            backend: ModularMatrixStorage::F2(matrix),
        };
        Some((wrap(p), wrap(l), wrap(u), wrap(q), rank))
    }

    pub(super) fn scalar_mul(&self, scalar: &ResidueClass) -> Result<Self, LinalgError> {
        self.scalar_mul_integer(scalar.representative())
    }

    fn scalar_mul_integer(&self, scalar: &Integer) -> Result<Self, LinalgError> {
        let modulus = self.ring.modulus().to_u64_checked();
        let backend = match &self.backend {
            ModularMatrixStorage::F2(matrix) => {
                ModularMatrixStorage::F2(matrix.scalar_mul(scalar.modulo_u64(2))?)
            }
            ModularMatrixStorage::PowerOfTwo(matrix) => ModularMatrixStorage::PowerOfTwo(
                matrix.scalar_mul(scalar.modulo_u64(modulus.expect("word-sized modulus"))),
            ),
            ModularMatrixStorage::Word(matrix) => {
                ModularMatrixStorage::Word(matrix.scalar_mul(scalar.modulo_u64(matrix.modulus)))
            }
            ModularMatrixStorage::Arbitrary(matrix) => {
                ModularMatrixStorage::Arbitrary(matrix.scalar_mul(scalar))
            }
        };
        Ok(Self {
            ring: self.ring.clone(),
            backend,
        })
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            F2Matrix::matmul,
            PwvlaMatrix::matmul,
            NmodMatrix::matmul,
            FmpzModMatrix::matmul,
        )
    }

    pub(super) fn transpose(&self) -> Self {
        let backend = match &self.backend {
            ModularMatrixStorage::F2(matrix) => ModularMatrixStorage::F2(matrix.transpose()),
            ModularMatrixStorage::PowerOfTwo(matrix) => {
                ModularMatrixStorage::PowerOfTwo(matrix.transpose())
            }
            ModularMatrixStorage::Word(matrix) => ModularMatrixStorage::Word(matrix.transpose()),
            ModularMatrixStorage::Arbitrary(matrix) => {
                ModularMatrixStorage::Arbitrary(matrix.transpose())
            }
        };
        Self {
            ring: self.ring.clone(),
            backend,
        }
    }

    pub(super) fn howell(&self, output_rows: usize) -> Self {
        debug_assert!(output_rows >= self.cols());
        let backend = match &self.backend {
            ModularMatrixStorage::F2(matrix) => {
                ModularMatrixStorage::F2(matrix.with_zero_rows(output_rows).rref().0)
            }
            ModularMatrixStorage::PowerOfTwo(matrix) => {
                #[cfg(test)]
                record_dense_conversion(DenseConversion::PowerOfTwoPersistent);
                let modulus = self
                    .ring
                    .modulus()
                    .to_u64_checked()
                    .expect("pwvla modulus is word-sized");
                let mut flint = NmodMatrix::from_values(
                    output_rows,
                    matrix.cols(),
                    modulus,
                    (0..output_rows * matrix.cols()).map(|index| {
                        let row = index / matrix.cols();
                        if row < matrix.rows() {
                            matrix.get(row, index % matrix.cols())
                        } else {
                            0
                        }
                    }),
                );
                flint.howell();
                // Howell reduction is the operation for which FLINT owns the
                // best retained kernel. Keep that result as the authoritative
                // payload instead of converting it back through row-major
                // values into pwvla storage.
                ModularMatrixStorage::Word(flint)
            }
            ModularMatrixStorage::Word(matrix) => {
                let mut output = matrix.with_zero_rows(output_rows);
                output.howell();
                ModularMatrixStorage::Word(output)
            }
            ModularMatrixStorage::Arbitrary(matrix) => {
                ModularMatrixStorage::Arbitrary(matrix.with_zero_rows(output_rows).howell())
            }
        };
        Self {
            ring: self.ring.clone(),
            backend,
        }
    }

    fn rank(&self) -> usize {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => matrix.rank(),
            ModularMatrixStorage::PowerOfTwo(_) => {
                unreachable!("a nonfield power-of-two residue ring has no field rank")
            }
            ModularMatrixStorage::Word(matrix) => matrix.rank(),
            ModularMatrixStorage::Arbitrary(matrix) => matrix.rank(),
        }
    }

    fn rref(&self) -> (Self, usize) {
        let (backend, rank) = match &self.backend {
            ModularMatrixStorage::F2(matrix) => {
                let (matrix, rank) = matrix.rref();
                (ModularMatrixStorage::F2(matrix), rank)
            }
            ModularMatrixStorage::PowerOfTwo(_) => {
                unreachable!("a nonfield power-of-two residue ring has no field RREF")
            }
            ModularMatrixStorage::Word(matrix) => {
                let (matrix, rank) = matrix.rref();
                (ModularMatrixStorage::Word(matrix), rank)
            }
            ModularMatrixStorage::Arbitrary(matrix) => {
                let (matrix, rank) = matrix.rref();
                (ModularMatrixStorage::Arbitrary(matrix), rank)
            }
        };
        (
            Self {
                ring: self.ring.clone(),
                backend,
            },
            rank,
        )
    }

    fn determinant(&self) -> Integer {
        match &self.backend {
            ModularMatrixStorage::F2(matrix) => {
                Integer::from(u8::from(matrix.rank() == matrix.rows()))
            }
            ModularMatrixStorage::PowerOfTwo(matrix) => Integer::from({
                #[cfg(test)]
                record_dense_conversion(DenseConversion::PowerOfTwoTerminal);
                NmodMatrix::from_values(
                    matrix.rows(),
                    matrix.cols(),
                    self.ring
                        .modulus()
                        .to_u64_checked()
                        .expect("power-of-two residue backend has a word modulus"),
                    (0..matrix.rows() * matrix.cols())
                        .map(|index| matrix.get(index / matrix.cols(), index % matrix.cols())),
                )
                .determinant()
            }),
            ModularMatrixStorage::Word(matrix) => Integer::from(matrix.determinant()),
            ModularMatrixStorage::Arbitrary(matrix) => matrix.determinant(),
        }
    }

    pub(super) fn determinant_element(&self) -> ResidueClass {
        self.ring.element(self.determinant())
    }

    pub(super) fn characteristic_polynomial(&self) -> Vec<ResidueClass> {
        let coefficients = match &self.backend {
            ModularMatrixStorage::F2(matrix) => {
                #[cfg(test)]
                record_dense_conversion(DenseConversion::F2Terminal);
                NmodMatrix::from_values(
                    matrix.rows(),
                    matrix.rows(),
                    2,
                    (0..matrix.rows() * matrix.rows())
                        .map(|index| matrix.get(index / matrix.rows(), index % matrix.rows())),
                )
                .characteristic_polynomial()
            }
            ModularMatrixStorage::PowerOfTwo(matrix) => {
                #[cfg(test)]
                record_dense_conversion(DenseConversion::PowerOfTwoTerminal);
                NmodMatrix::from_values(
                    matrix.rows(),
                    matrix.cols(),
                    self.ring
                        .modulus()
                        .to_u64_checked()
                        .expect("power-of-two residue backend has a word modulus"),
                    (0..matrix.rows() * matrix.cols())
                        .map(|index| matrix.get(index / matrix.cols(), index % matrix.cols())),
                )
                .characteristic_polynomial()
            }
            ModularMatrixStorage::Word(matrix) => matrix.characteristic_polynomial(),
            ModularMatrixStorage::Arbitrary(matrix) => matrix.characteristic_polynomial(),
        };
        coefficients
            .into_iter()
            .map(|coefficient| self.ring.element(coefficient))
            .collect()
    }

    fn map_binary(
        &self,
        rhs: &Self,
        binary: fn(&F2Matrix, &F2Matrix) -> Result<F2Matrix, LinalgError>,
        power_of_two: fn(&PwvlaMatrix, &PwvlaMatrix) -> PwvlaMatrix,
        word: fn(&NmodMatrix, &NmodMatrix) -> NmodMatrix,
        arbitrary: fn(&FmpzModMatrix, &FmpzModMatrix) -> FmpzModMatrix,
    ) -> Result<Self, LinalgError> {
        let backend = match (&self.backend, &rhs.backend) {
            (ModularMatrixStorage::F2(left), ModularMatrixStorage::F2(right)) => {
                ModularMatrixStorage::F2(binary(left, right)?)
            }
            (ModularMatrixStorage::PowerOfTwo(left), ModularMatrixStorage::PowerOfTwo(right)) => {
                ModularMatrixStorage::PowerOfTwo(power_of_two(left, right))
            }
            (ModularMatrixStorage::Word(left), ModularMatrixStorage::Word(right)) => {
                ModularMatrixStorage::Word(word(left, right))
            }
            (ModularMatrixStorage::PowerOfTwo(left), ModularMatrixStorage::Word(right)) => {
                let left = self.power_of_two_as_nmod(left);
                ModularMatrixStorage::Word(word(&left, right))
            }
            (ModularMatrixStorage::Word(left), ModularMatrixStorage::PowerOfTwo(right)) => {
                let right = self.power_of_two_as_nmod(right);
                ModularMatrixStorage::Word(word(left, &right))
            }
            (ModularMatrixStorage::Arbitrary(left), ModularMatrixStorage::Arbitrary(right)) => {
                ModularMatrixStorage::Arbitrary(arbitrary(left, right))
            }
            _ => unreachable!("equal residue contexts select the same matrix backend"),
        };
        Ok(Self {
            ring: self.ring.clone(),
            backend,
        })
    }

    fn power_of_two_as_nmod(&self, matrix: &PwvlaMatrix) -> NmodMatrix {
        #[cfg(test)]
        record_dense_conversion(DenseConversion::PowerOfTwoPersistent);
        let modulus = self
            .ring
            .modulus()
            .to_u64_checked()
            .expect("pwvla modulus is word-sized");
        NmodMatrix::from_values(
            matrix.rows(),
            matrix.cols(),
            modulus,
            (0..matrix.rows() * matrix.cols())
                .map(|index| matrix.get(index / matrix.cols(), index % matrix.cols())),
        )
    }
}

impl Clone for ModularMatrix {
    fn clone(&self) -> Self {
        let backend = match &self.backend {
            ModularMatrixStorage::F2(matrix) => ModularMatrixStorage::F2(matrix.clone()),
            ModularMatrixStorage::PowerOfTwo(matrix) => {
                ModularMatrixStorage::PowerOfTwo(matrix.clone())
            }
            ModularMatrixStorage::Word(matrix) => ModularMatrixStorage::Word(matrix.clone()),
            ModularMatrixStorage::Arbitrary(matrix) => {
                ModularMatrixStorage::Arbitrary(matrix.clone())
            }
        };
        Self {
            ring: self.ring.clone(),
            backend,
        }
    }
}

enum ExtensionMatrixStorage {
    Flint(fq_default_mat_struct),
    Small(Vec<u8>),
}

struct FqDefaultMatrix {
    storage: ExtensionMatrixStorage,
    field: FiniteField,
    context: std::sync::Arc<crate::numthy::ExtensionContext>,
    rows: usize,
    cols: usize,
}

impl FqDefaultMatrix {
    #[cfg(test)]
    fn uses_small_extension_storage(&self) -> bool {
        matches!(self.storage, ExtensionMatrixStorage::Small(_))
    }

    fn new(field: FiniteField, rows: usize, cols: usize) -> Self {
        let context = field
            .extension_context()
            .expect("fq_nmod matrix requires an extension field");
        if context.small_extension_tables().is_some() {
            return Self {
                storage: ExtensionMatrixStorage::Small(vec![0; rows * cols]),
                field,
                context,
                rows,
                cols,
            };
        }
        let mut inner = MaybeUninit::<fq_default_mat_struct>::uninit();
        unsafe {
            fq_default_mat_init(
                inner.as_mut_ptr(),
                rows as slong,
                cols as slong,
                context.default_as_ptr(),
            )
        };
        Self {
            storage: ExtensionMatrixStorage::Flint(unsafe { inner.assume_init() }),
            field,
            context,
            rows,
            cols,
        }
    }

    fn flint(field: FiniteField, rows: usize, cols: usize) -> Self {
        let context = field
            .extension_context()
            .expect("fq_nmod matrix requires an extension field");
        let mut inner = MaybeUninit::<fq_default_mat_struct>::uninit();
        unsafe {
            fq_default_mat_init(
                inner.as_mut_ptr(),
                rows as slong,
                cols as slong,
                context.default_as_ptr(),
            )
        };
        Self {
            storage: ExtensionMatrixStorage::Flint(unsafe { inner.assume_init() }),
            field,
            context,
            rows,
            cols,
        }
    }

    fn from_entries<'a>(
        field: FiniteField,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a FiniteFieldElement>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let mut output = Self::new(field, rows, cols);
        match &mut output.storage {
            ExtensionMatrixStorage::Small(values) => {
                for (target, entry) in values.iter_mut().zip(entries) {
                    *target = u8::try_from(
                        entry
                            .representative()
                            .to_u64_checked()
                            .expect("small extension-field element fits a machine word"),
                    )
                    .expect("small extension-field element fits one byte");
                }
            }
            ExtensionMatrixStorage::Flint(inner) => {
                for (index, entry) in entries.enumerate() {
                    let value = entry
                        .extension_value()
                        .expect("extension-field matrix element invariant")
                        .to_default();
                    unsafe {
                        fq_default_mat_entry_set(
                            inner,
                            (index / cols) as slong,
                            (index % cols) as slong,
                            &value.raw,
                            output.context.default_as_ptr(),
                        )
                    };
                }
            }
        }
        output
    }

    fn get(&self, row: usize, col: usize) -> FiniteFieldElement {
        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let tables = self
                .context
                .small_extension_tables()
                .expect("small extension matrix has arithmetic tables");
            return self
                .field
                .element_from_coefficients(&tables.decode(entries[row * self.cols + col]))
                .expect("small extension matrix stores a canonical field code");
        }
        let mut value = DefaultExtensionElement::initialized(self.context.clone());
        unsafe {
            fq_default_mat_entry(
                &mut value.raw,
                self.flint_inner(),
                row as slong,
                col as slong,
                self.context.default_as_ptr(),
            )
        };
        self.field.wrap_extension(value.to_extension())
    }

    fn binary(
        &self,
        rhs: &Self,
        operation: unsafe extern "C" fn(
            *mut fq_default_mat_struct,
            *const fq_default_mat_struct,
            *const fq_default_mat_struct,
            *const flint_sys::fq_default_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
        unsafe {
            operation(
                output.flint_inner_mut(),
                self.flint_inner(),
                rhs.flint_inner(),
                self.context.default_as_ptr(),
            )
        };
        output
    }

    fn add(&self, rhs: &Self) -> Self {
        if let (ExtensionMatrixStorage::Small(left), ExtensionMatrixStorage::Small(right)) =
            (&self.storage, &rhs.storage)
        {
            let tables = self.tables();
            let entries = left
                .iter()
                .zip(right)
                .map(|(&left, &right)| tables.add(left, right))
                .collect();
            return self.with_small(entries, self.rows, self.cols);
        }
        self.binary(rhs, fq_default_mat_add)
    }

    fn sub(&self, rhs: &Self) -> Self {
        if let (ExtensionMatrixStorage::Small(left), ExtensionMatrixStorage::Small(right)) =
            (&self.storage, &rhs.storage)
        {
            let tables = self.tables();
            let entries = left
                .iter()
                .zip(right)
                .map(|(&left, &right)| tables.subtract(left, right))
                .collect();
            return self.with_small(entries, self.rows, self.cols);
        }
        self.binary(rhs, fq_default_mat_sub)
    }

    fn neg(&self) -> Self {
        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let tables = self.tables();
            let entries = entries
                .iter()
                .map(|&value| tables.subtract(0, value))
                .collect();
            return self.with_small(entries, self.rows, self.cols);
        }
        let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
        unsafe {
            fq_default_mat_neg(
                output.flint_inner_mut(),
                self.flint_inner(),
                self.context.default_as_ptr(),
            )
        };
        output
    }

    fn scalar_mul(&self, scalar: &FiniteFieldElement) -> Self {
        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let scalar = u8::try_from(
                scalar
                    .representative()
                    .to_u64_checked()
                    .expect("small extension-field scalar fits a machine word"),
            )
            .expect("small extension-field scalar fits one byte");
            let tables = self.tables();
            let entries = entries
                .iter()
                .map(|&value| tables.multiply(value, scalar))
                .collect();
            return self.with_small(entries, self.rows, self.cols);
        }
        let scalar = scalar
            .extension_value()
            .expect("extension-field matrix scalar invariant")
            .to_default();
        let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
        unsafe {
            fq_default_mat_scalar_mul(
                output.flint_inner_mut(),
                self.flint_inner(),
                &scalar.raw,
                self.context.default_as_ptr(),
            )
        };
        output
    }

    fn matmul(&self, rhs: &Self) -> Self {
        if let (ExtensionMatrixStorage::Small(left), ExtensionMatrixStorage::Small(right)) =
            (&self.storage, &rhs.storage)
        {
            let tables = self.tables();
            let mut entries = vec![0; self.rows * rhs.cols];
            for row in 0..self.rows {
                for inner in 0..self.cols {
                    let left = left[row * self.cols + inner];
                    if left == 0 {
                        continue;
                    }
                    for column in 0..rhs.cols {
                        let target = row * rhs.cols + column;
                        let product = tables.multiply(left, right[inner * rhs.cols + column]);
                        entries[target] = tables.add(entries[target], product);
                    }
                }
            }
            return self.with_small(entries, self.rows, rhs.cols);
        }
        let mut output = Self::flint(self.field.clone(), self.rows, rhs.cols);
        unsafe {
            fq_default_mat_mul(
                output.flint_inner_mut(),
                self.flint_inner(),
                rhs.flint_inner(),
                self.context.default_as_ptr(),
            )
        };
        output
    }

    fn transpose(&self) -> Self {
        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let mut transposed = vec![0; entries.len()];
            for row in 0..self.rows {
                for column in 0..self.cols {
                    transposed[column * self.rows + row] = entries[row * self.cols + column];
                }
            }
            return self.with_small(transposed, self.cols, self.rows);
        }
        let mut output = Self::flint(self.field.clone(), self.cols, self.rows);
        unsafe {
            fq_default_mat_transpose(
                output.flint_inner_mut(),
                self.flint_inner(),
                self.context.default_as_ptr(),
            )
        };
        output
    }

    fn rank(&self) -> usize {
        match &self.storage {
            ExtensionMatrixStorage::Small(_) => self.rref().1,
            ExtensionMatrixStorage::Flint(inner) => unsafe {
                fq_default_mat_rank(inner, self.context.default_as_ptr()) as usize
            },
        }
    }

    fn rref(&self) -> (Self, usize) {
        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let mut entries = entries.clone();
            let rank = self.reduce_small(&mut entries);
            return (self.with_small(entries, self.rows, self.cols), rank);
        }
        let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
        let rank = unsafe {
            fq_default_mat_rref(
                output.flint_inner_mut(),
                self.flint_inner(),
                self.context.default_as_ptr(),
            )
        } as usize;
        (output, rank)
    }

    fn determinant(&self) -> FiniteFieldElement {
        debug_assert_eq!(self.rows, self.cols);
        if self.rows == 0 {
            return self.field.element(1_u8);
        }

        if let ExtensionMatrixStorage::Small(entries) = &self.storage {
            let tables = self.tables();
            let mut entries = entries.clone();
            let mut determinant = 1_u8;
            for column in 0..self.cols {
                let Some(pivot_row) =
                    (column..self.rows).find(|&row| entries[row * self.cols + column] != 0)
                else {
                    return self.field.element(0_u8);
                };
                if pivot_row != column {
                    for entry_column in 0..self.cols {
                        entries.swap(
                            pivot_row * self.cols + entry_column,
                            column * self.cols + entry_column,
                        );
                    }
                    determinant = tables.subtract(0, determinant);
                }
                let pivot = entries[column * self.cols + column];
                determinant = tables.multiply(determinant, pivot);
                let inverse = tables.inverse(pivot);
                for row in column + 1..self.rows {
                    let factor = tables.multiply(entries[row * self.cols + column], inverse);
                    if factor == 0 {
                        continue;
                    }
                    for entry_column in column + 1..self.cols {
                        let target = row * self.cols + entry_column;
                        entries[target] = tables.subtract_product(
                            entries[target],
                            factor,
                            entries[column * self.cols + entry_column],
                        );
                    }
                }
            }
            return self.decode_small(determinant);
        }

        let mut lu = self.clone();
        let mut permutation = vec![0; self.rows];
        let rank = unsafe {
            fq_default_mat_lu(
                permutation.as_mut_ptr(),
                lu.flint_inner_mut(),
                1,
                self.context.default_as_ptr(),
            )
        } as usize;
        if rank != self.rows {
            return self.field.element(0_u8);
        }

        let mut determinant = DefaultExtensionElement::initialized(self.context.clone());
        unsafe { fq_default_one(&mut determinant.raw, self.context.default_as_ptr()) };
        for index in 0..self.rows {
            let mut diagonal = DefaultExtensionElement::initialized(self.context.clone());
            unsafe {
                fq_default_mat_entry(
                    &mut diagonal.raw,
                    lu.flint_inner(),
                    index as slong,
                    index as slong,
                    self.context.default_as_ptr(),
                )
            };
            let mut product = DefaultExtensionElement::initialized(self.context.clone());
            unsafe {
                fq_default_mul(
                    &mut product.raw,
                    &determinant.raw,
                    &diagonal.raw,
                    self.context.default_as_ptr(),
                )
            };
            determinant = product;
        }
        if unsafe { _perm_parity(permutation.as_ptr(), self.rows as slong) } != 0 {
            let mut negative = DefaultExtensionElement::initialized(self.context.clone());
            unsafe {
                fq_default_neg(
                    &mut negative.raw,
                    &determinant.raw,
                    self.context.default_as_ptr(),
                )
            };
            determinant = negative;
        }
        self.field.wrap_extension(determinant.to_extension())
    }

    fn characteristic_polynomial(&self) -> Vec<FiniteFieldElement> {
        debug_assert_eq!(self.rows, self.cols);
        let materialized;
        let matrix = if matches!(self.storage, ExtensionMatrixStorage::Small(_)) {
            materialized = self.as_flint();
            &materialized
        } else {
            self
        };
        let mut polynomial = FqDefaultPolynomial::new(self.context.clone());
        unsafe {
            fq_default_mat_charpoly(
                &mut polynomial.inner,
                matrix.flint_inner(),
                self.context.default_as_ptr(),
            )
        };
        (0..=self.rows)
            .map(|degree| {
                self.field
                    .wrap_extension(polynomial.coefficient(degree).to_extension())
            })
            .collect()
    }

    fn tables(&self) -> &crate::numthy::SmallExtensionTables {
        self.context
            .small_extension_tables()
            .expect("small extension matrix has arithmetic tables")
    }

    fn with_small(&self, entries: Vec<u8>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        Self {
            storage: ExtensionMatrixStorage::Small(entries),
            field: self.field.clone(),
            context: self.context.clone(),
            rows,
            cols,
        }
    }

    fn decode_small(&self, value: u8) -> FiniteFieldElement {
        self.field
            .element_from_coefficients(&self.tables().decode(value))
            .expect("small extension matrix stores a canonical field code")
    }

    fn reduce_small(&self, entries: &mut [u8]) -> usize {
        let tables = self.tables();
        let mut rank = 0;
        for column in 0..self.cols {
            let Some(pivot_row) =
                (rank..self.rows).find(|&row| entries[row * self.cols + column] != 0)
            else {
                continue;
            };
            if pivot_row != rank {
                for entry_column in 0..self.cols {
                    entries.swap(
                        pivot_row * self.cols + entry_column,
                        rank * self.cols + entry_column,
                    );
                }
            }
            let inverse = tables.inverse(entries[rank * self.cols + column]);
            entries[rank * self.cols + column] = 1;
            for entry_column in column + 1..self.cols {
                let position = rank * self.cols + entry_column;
                entries[position] = tables.multiply(entries[position], inverse);
            }
            for row in 0..self.rows {
                if row == rank {
                    continue;
                }
                let factor = entries[row * self.cols + column];
                if factor == 0 {
                    continue;
                }
                entries[row * self.cols + column] = 0;
                for entry_column in column + 1..self.cols {
                    let position = row * self.cols + entry_column;
                    entries[position] = tables.subtract_product(
                        entries[position],
                        factor,
                        entries[rank * self.cols + entry_column],
                    );
                }
            }
            rank += 1;
            if rank == self.rows {
                break;
            }
        }
        rank
    }

    fn as_flint(&self) -> Self {
        let ExtensionMatrixStorage::Small(entries) = &self.storage else {
            return self.clone();
        };
        let values = entries
            .iter()
            .map(|&entry| self.decode_small(entry))
            .collect::<Vec<_>>();
        let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
        let inner = output.flint_inner_mut();
        for (index, entry) in values.iter().enumerate() {
            let value = entry
                .extension_value()
                .expect("extension-field matrix element invariant")
                .to_default();
            unsafe {
                fq_default_mat_entry_set(
                    inner,
                    (index / self.cols) as slong,
                    (index % self.cols) as slong,
                    &value.raw,
                    self.context.default_as_ptr(),
                )
            };
        }
        output
    }

    fn flint_inner(&self) -> &fq_default_mat_struct {
        let ExtensionMatrixStorage::Flint(inner) = &self.storage else {
            panic!("small extension matrix used as a FLINT matrix")
        };
        inner
    }

    fn flint_inner_mut(&mut self) -> &mut fq_default_mat_struct {
        let ExtensionMatrixStorage::Flint(inner) = &mut self.storage else {
            panic!("small extension matrix used as a FLINT matrix")
        };
        inner
    }
}

impl Clone for FqDefaultMatrix {
    fn clone(&self) -> Self {
        match &self.storage {
            ExtensionMatrixStorage::Small(entries) => {
                self.with_small(entries.clone(), self.rows, self.cols)
            }
            ExtensionMatrixStorage::Flint(_) => {
                let mut output = Self::flint(self.field.clone(), self.rows, self.cols);
                unsafe {
                    fq_default_mat_set(
                        output.flint_inner_mut(),
                        self.flint_inner(),
                        self.context.default_as_ptr(),
                    )
                };
                output
            }
        }
    }
}

impl Drop for FqDefaultMatrix {
    fn drop(&mut self) {
        if let ExtensionMatrixStorage::Flint(inner) = &mut self.storage {
            unsafe { fq_default_mat_clear(inner, self.context.default_as_ptr()) };
        }
    }
}

unsafe impl Send for FqDefaultMatrix {}
unsafe impl Sync for FqDefaultMatrix {}

struct FqDefaultPolynomial {
    inner: fq_default_poly_struct,
    context: std::sync::Arc<crate::numthy::ExtensionContext>,
}

impl FqDefaultPolynomial {
    fn new(context: std::sync::Arc<crate::numthy::ExtensionContext>) -> Self {
        let mut inner = MaybeUninit::<fq_default_poly_struct>::uninit();
        unsafe { fq_default_poly_init(inner.as_mut_ptr(), context.default_as_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
            context,
        }
    }

    fn coefficient(&self, degree: usize) -> DefaultExtensionElement {
        let mut output = DefaultExtensionElement::initialized(self.context.clone());
        unsafe {
            fq_default_poly_get_coeff(
                &mut output.raw,
                &self.inner,
                degree as slong,
                self.context.default_as_ptr(),
            )
        };
        output
    }
}

impl Drop for FqDefaultPolynomial {
    fn drop(&mut self) {
        unsafe { fq_default_poly_clear(&mut self.inner, self.context.default_as_ptr()) };
    }
}

enum FiniteFieldMatrixBackend {
    Prime(ModularMatrix),
    BinaryExtension(M4rsteMatrix),
    Extension(FqDefaultMatrix),
}

pub(super) struct FiniteFieldMatrix {
    field: FiniteField,
    backend: FiniteFieldMatrixBackend,
}

impl FiniteFieldMatrix {
    #[cfg(test)]
    pub(super) fn uses_small_odd_extension_backend(&self) -> bool {
        matches!(
            &self.backend,
            FiniteFieldMatrixBackend::Extension(matrix)
                if matrix.uses_small_extension_storage()
        )
    }

    fn rows(&self) -> usize {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix.rows(),
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => matrix.rows(),
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.rows,
        }
    }

    pub(super) fn cols(&self) -> usize {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix.cols(),
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => matrix.cols(),
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.cols,
        }
    }

    pub(super) fn zeros(field: FiniteField, rows: usize, cols: usize) -> Self {
        let backend = if field.degree() == 1 {
            FiniteFieldMatrixBackend::Prime(ModularMatrix::zeros(
                ResidueRing::new(field.characteristic().clone())
                    .expect("prime-field characteristic is a valid residue modulus"),
                rows,
                cols,
            ))
        } else if let Some(minimal_polynomial) = binary_extension_minimal_polynomial(&field) {
            FiniteFieldMatrixBackend::BinaryExtension(M4rsteMatrix::zeros(
                minimal_polynomial,
                rows,
                cols,
            ))
        } else {
            FiniteFieldMatrixBackend::Extension(FqDefaultMatrix::new(field.clone(), rows, cols))
        };
        Self { field, backend }
    }

    #[cfg(feature = "distributed")]
    pub(super) fn from_packed_f2(field: FiniteField, matrix: m4rst::Mzd) -> Option<Self> {
        if field.degree() != 1 || field.characteristic().to_u64_checked() != Some(2) {
            return None;
        }
        let residues = ResidueRing::new(field.characteristic().clone())
            .expect("prime-field characteristic is a valid residue modulus");
        Some(Self {
            field,
            backend: FiniteFieldMatrixBackend::Prime(ModularMatrix::from_packed_f2(
                residues, matrix,
            )?),
        })
    }

    #[cfg(feature = "distributed")]
    pub(super) fn packed_f2(&self) -> Option<std::sync::Arc<m4rst::Mzd>> {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix.packed_f2(),
            _ => None,
        }
    }

    pub(super) fn identity(field: FiniteField, size: usize) -> Self {
        let backend = if field.degree() == 1 {
            FiniteFieldMatrixBackend::Prime(ModularMatrix::identity(
                ResidueRing::new(field.characteristic().clone())
                    .expect("prime-field characteristic is a valid residue modulus"),
                size,
            ))
        } else if let Some(minimal_polynomial) = binary_extension_minimal_polynomial(&field) {
            FiniteFieldMatrixBackend::BinaryExtension(M4rsteMatrix::identity(
                minimal_polynomial,
                size,
            ))
        } else {
            let entries = (0..size * size)
                .map(|index| {
                    if index / size == index % size {
                        field.element(1_u8)
                    } else {
                        field.element(0_u8)
                    }
                })
                .collect();
            return Self::from_owned_entries(field, size, size, entries);
        };
        Self { field, backend }
    }

    pub(super) fn from_entries<'a>(
        field: FiniteField,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a FiniteFieldElement>,
    ) -> Self {
        let backend = if field.degree() == 1 {
            let residues = ResidueRing::new(field.characteristic().clone())
                .expect("prime-field characteristic is a valid residue modulus");
            FiniteFieldMatrixBackend::Prime(ModularMatrix::from_integer_entries(
                residues,
                rows,
                cols,
                entries.map(FiniteFieldElement::representative),
            ))
        } else if let Some(minimal_polynomial) = binary_extension_minimal_polynomial(&field) {
            FiniteFieldMatrixBackend::BinaryExtension(M4rsteMatrix::from_values(
                minimal_polynomial,
                rows,
                cols,
                entries.map(binary_extension_value),
            ))
        } else {
            FiniteFieldMatrixBackend::Extension(FqDefaultMatrix::from_entries(
                field.clone(),
                rows,
                cols,
                entries,
            ))
        };
        Self { field, backend }
    }

    fn from_owned_entries(
        field: FiniteField,
        rows: usize,
        cols: usize,
        entries: Vec<FiniteFieldElement>,
    ) -> Self {
        Self::from_entries(field, rows, cols, entries.iter())
    }

    pub(super) fn get(&self, row: usize, col: usize) -> FiniteFieldElement {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                self.field.element(matrix.get_integer(row, col))
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                binary_extension_element(&self.field, matrix.get(row, col))
            }
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.get(row, col),
        }
    }

    pub(super) fn materialized_values(&self) -> Vec<FiniteFieldElement> {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix
                .materialized_integers()
                .into_iter()
                .map(|value| self.field.element(value))
                .collect(),
            _ => (0..self.cols() * self.rows())
                .map(|index| self.get(index / self.cols(), index % self.cols()))
                .collect(),
        }
    }

    pub(super) fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            ModularMatrix::add,
            M4rsteMatrix::add,
            FqDefaultMatrix::add,
        )
    }

    pub(super) fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            ModularMatrix::sub,
            M4rsteMatrix::sub,
            FqDefaultMatrix::sub,
        )
    }

    pub(super) fn neg(&self) -> Self {
        let backend = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                FiniteFieldMatrixBackend::Prime(matrix.neg())
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                FiniteFieldMatrixBackend::BinaryExtension(matrix.neg())
            }
            FiniteFieldMatrixBackend::Extension(matrix) => {
                FiniteFieldMatrixBackend::Extension(matrix.neg())
            }
        };
        Self {
            field: self.field.clone(),
            backend,
        }
    }

    pub(super) fn scalar_mul(&self, scalar: &FiniteFieldElement) -> Result<Self, LinalgError> {
        let backend = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                FiniteFieldMatrixBackend::Prime(matrix.scalar_mul_integer(scalar.representative())?)
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                FiniteFieldMatrixBackend::BinaryExtension(
                    matrix.scalar_mul(binary_extension_value(scalar)),
                )
            }
            FiniteFieldMatrixBackend::Extension(matrix) => {
                FiniteFieldMatrixBackend::Extension(matrix.scalar_mul(scalar))
            }
        };
        Ok(Self {
            field: self.field.clone(),
            backend,
        })
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.map_binary(
            rhs,
            ModularMatrix::matmul,
            M4rsteMatrix::matmul,
            FqDefaultMatrix::matmul,
        )
    }

    pub(super) fn transpose(&self) -> Self {
        let backend = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                FiniteFieldMatrixBackend::Prime(matrix.transpose())
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                FiniteFieldMatrixBackend::BinaryExtension(matrix.transpose())
            }
            FiniteFieldMatrixBackend::Extension(matrix) => {
                FiniteFieldMatrixBackend::Extension(matrix.transpose())
            }
        };
        Self {
            field: self.field.clone(),
            backend,
        }
    }

    pub(super) fn rank(&self) -> usize {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix.rank(),
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => matrix.rank(),
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.rank(),
        }
    }

    pub(super) fn rref(&self) -> (Self, usize) {
        let (backend, rank) = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                let (matrix, rank) = matrix.rref();
                (FiniteFieldMatrixBackend::Prime(matrix), rank)
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                let (matrix, rank) = matrix.rref();
                (FiniteFieldMatrixBackend::BinaryExtension(matrix), rank)
            }
            FiniteFieldMatrixBackend::Extension(matrix) => {
                let (matrix, rank) = matrix.rref();
                (FiniteFieldMatrixBackend::Extension(matrix), rank)
            }
        };
        (
            Self {
                field: self.field.clone(),
                backend,
            },
            rank,
        )
    }

    pub(super) fn solve_left(&self, rhs: &Self) -> Option<Option<Self>> {
        match (&self.backend, &rhs.backend) {
            (FiniteFieldMatrixBackend::Prime(left), FiniteFieldMatrixBackend::Prime(right)) => {
                left.solve_left(right).map(|solution| {
                    solution.map(|matrix| Self {
                        field: self.field.clone(),
                        backend: FiniteFieldMatrixBackend::Prime(matrix),
                    })
                })
            }
            (
                FiniteFieldMatrixBackend::BinaryExtension(left),
                FiniteFieldMatrixBackend::BinaryExtension(right),
            ) => Some(left.solve_left(right).map(|matrix| Self {
                field: self.field.clone(),
                backend: FiniteFieldMatrixBackend::BinaryExtension(matrix),
            })),
            _ => None,
        }
    }

    pub(super) fn nullspace(&self) -> Option<Self> {
        let backend = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                FiniteFieldMatrixBackend::Prime(matrix.nullspace()?)
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                FiniteFieldMatrixBackend::BinaryExtension(matrix.nullspace())
            }
            FiniteFieldMatrixBackend::Extension(_) => return None,
        };
        Some(Self {
            field: self.field.clone(),
            backend,
        })
    }

    pub(super) fn pluq(&self) -> Option<(Self, Self, Self, Self, usize)> {
        let (p, l, u, q, rank) = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                let (p, l, u, q, rank) = matrix.pluq()?;
                (
                    FiniteFieldMatrixBackend::Prime(p),
                    FiniteFieldMatrixBackend::Prime(l),
                    FiniteFieldMatrixBackend::Prime(u),
                    FiniteFieldMatrixBackend::Prime(q),
                    rank,
                )
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                let (p, l, u, q, rank) = matrix.pluq();
                (
                    FiniteFieldMatrixBackend::BinaryExtension(p),
                    FiniteFieldMatrixBackend::BinaryExtension(l),
                    FiniteFieldMatrixBackend::BinaryExtension(u),
                    FiniteFieldMatrixBackend::BinaryExtension(q),
                    rank,
                )
            }
            FiniteFieldMatrixBackend::Extension(_) => return None,
        };
        let wrap = |backend| Self {
            field: self.field.clone(),
            backend,
        };
        Some((wrap(p), wrap(l), wrap(u), wrap(q), rank))
    }

    pub(super) fn determinant(&self) -> FiniteFieldElement {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => self.field.element(matrix.determinant()),
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                self.binary_extension_as_flint(matrix).determinant()
            }
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.determinant(),
        }
    }

    pub(super) fn characteristic_polynomial(&self) -> Vec<FiniteFieldElement> {
        match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => matrix
                .characteristic_polynomial()
                .into_iter()
                .map(|coefficient| self.field.element(coefficient.representative().clone()))
                .collect(),
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => self
                .binary_extension_as_flint(matrix)
                .characteristic_polynomial(),
            FiniteFieldMatrixBackend::Extension(matrix) => matrix.characteristic_polynomial(),
        }
    }

    fn binary_extension_as_flint(&self, matrix: &M4rsteMatrix) -> FqDefaultMatrix {
        #[cfg(test)]
        record_dense_conversion(DenseConversion::BinaryExtensionTerminal);
        let entries = (0..matrix.rows() * matrix.cols())
            .map(|index| {
                binary_extension_element(
                    &self.field,
                    matrix.get(index / matrix.cols(), index % matrix.cols()),
                )
            })
            .collect::<Vec<_>>();
        FqDefaultMatrix::from_entries(
            self.field.clone(),
            matrix.rows(),
            matrix.cols(),
            entries.iter(),
        )
    }

    fn map_binary(
        &self,
        rhs: &Self,
        prime: fn(&ModularMatrix, &ModularMatrix) -> Result<ModularMatrix, LinalgError>,
        binary_extension: fn(&M4rsteMatrix, &M4rsteMatrix) -> M4rsteMatrix,
        extension: fn(&FqDefaultMatrix, &FqDefaultMatrix) -> FqDefaultMatrix,
    ) -> Result<Self, LinalgError> {
        let backend = match (&self.backend, &rhs.backend) {
            (FiniteFieldMatrixBackend::Prime(left), FiniteFieldMatrixBackend::Prime(right)) => {
                FiniteFieldMatrixBackend::Prime(prime(left, right)?)
            }
            (
                FiniteFieldMatrixBackend::BinaryExtension(left),
                FiniteFieldMatrixBackend::BinaryExtension(right),
            ) => FiniteFieldMatrixBackend::BinaryExtension(binary_extension(left, right)),
            (
                FiniteFieldMatrixBackend::Extension(left),
                FiniteFieldMatrixBackend::Extension(right),
            ) => FiniteFieldMatrixBackend::Extension(extension(left, right)),
            _ => unreachable!("equal finite fields select the same matrix backend"),
        };
        Ok(Self {
            field: self.field.clone(),
            backend,
        })
    }
}

impl Clone for FiniteFieldMatrix {
    fn clone(&self) -> Self {
        let backend = match &self.backend {
            FiniteFieldMatrixBackend::Prime(matrix) => {
                FiniteFieldMatrixBackend::Prime(matrix.clone())
            }
            FiniteFieldMatrixBackend::BinaryExtension(matrix) => {
                FiniteFieldMatrixBackend::BinaryExtension(matrix.clone())
            }
            FiniteFieldMatrixBackend::Extension(matrix) => {
                FiniteFieldMatrixBackend::Extension(matrix.clone())
            }
        };
        Self {
            field: self.field.clone(),
            backend,
        }
    }
}

fn binary_extension_minimal_polynomial(field: &FiniteField) -> Option<u64> {
    if field.word_characteristic() != Some(2) || !(2..=16).contains(&field.degree()) {
        return None;
    }
    Some(
        field
            .defining_modulus()?
            .iter()
            .copied()
            .enumerate()
            .fold(0, |bits, (degree, coefficient)| {
                bits | ((coefficient & 1) << degree)
            }),
    )
}

fn binary_extension_value(element: &FiniteFieldElement) -> u16 {
    u16::try_from(
        element
            .representative()
            .to_u64_checked()
            .expect("degree-at-most-16 binary-field representative fits a machine word"),
    )
    .expect("degree-at-most-16 binary-field representative fits u16")
}

fn binary_extension_element(field: &FiniteField, value: u16) -> FiniteFieldElement {
    let coefficients = (0..field.degree())
        .map(|degree| u64::from((value >> degree) & 1))
        .collect::<Vec<_>>();
    field
        .element_from_coefficients(&coefficients)
        .expect("packed binary-extension element is canonical")
}

pub(super) enum PadicMatrixContext {
    Ring(PadicRing),
    Field(PadicField),
}

impl PadicMatrixContext {
    fn absolute_precision(&self) -> i64 {
        let precision = match self {
            Self::Ring(ring) => ring.absolute_precision(),
            Self::Field(field) => field.absolute_precision(),
        };
        i64::try_from(precision).expect("p-adic precision was validated at context construction")
    }

    fn raw_context_ptr(&self) -> *const flint_sys::padic_ctx_struct {
        match self {
            Self::Ring(ring) => ring.raw_context_ptr(),
            Self::Field(field) => field.raw_context_ptr(),
        }
    }

    fn element_from_flint_operation(
        &self,
        operation: impl FnOnce(*mut flint_sys::padic_struct),
    ) -> PadicElement {
        match self {
            Self::Ring(ring) => ring.element_with_flint(operation),
            Self::Field(field) => field.element_with_flint(operation),
        }
    }
}

impl Clone for PadicMatrixContext {
    fn clone(&self) -> Self {
        match self {
            Self::Ring(ring) => Self::Ring(ring.clone()),
            Self::Field(field) => Self::Field(field.clone()),
        }
    }
}

pub(super) struct PadicMatrix {
    inner: padic_mat_struct,
    context: PadicMatrixContext,
    rows: usize,
    cols: usize,
}

impl PadicMatrix {
    fn new(context: PadicMatrixContext, rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<padic_mat_struct>::uninit();
        unsafe {
            padic_mat_init2(
                inner.as_mut_ptr(),
                rows as slong,
                cols as slong,
                context.absolute_precision(),
            )
        };
        Self {
            inner: unsafe { inner.assume_init() },
            context,
            rows,
            cols,
        }
    }

    pub(super) fn zeros(context: PadicMatrixContext, rows: usize, cols: usize) -> Self {
        Self::new(context, rows, cols)
    }

    pub(super) fn identity(context: PadicMatrixContext, size: usize) -> Self {
        let mut output = Self::new(context, size, size);
        unsafe { padic_mat_one(&mut output.inner) };
        output
    }

    pub(super) fn from_entries<'a>(
        context: PadicMatrixContext,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a PadicElement>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let mut output = Self::new(context, rows, cols);
        for (index, entry) in entries.enumerate() {
            unsafe {
                padic_mat_set_entry_padic(
                    &mut output.inner,
                    (index / cols) as slong,
                    (index % cols) as slong,
                    entry.raw_ptr(),
                    output.context.raw_context_ptr(),
                )
            };
        }
        output
    }

    pub(super) fn get(&self, row: usize, col: usize) -> PadicElement {
        self.context.element_from_flint_operation(|output| unsafe {
            padic_mat_get_entry_padic(
                output,
                &self.inner,
                row as slong,
                col as slong,
                self.context.raw_context_ptr(),
            )
        })
    }

    fn binary(
        &self,
        rhs: &Self,
        rows: usize,
        cols: usize,
        operation: unsafe extern "C" fn(
            *mut padic_mat_struct,
            *const padic_mat_struct,
            *const padic_mat_struct,
            *const flint_sys::padic_ctx_struct,
        ),
    ) -> Self {
        let mut output = Self::new(self.context.clone(), rows, cols);
        unsafe {
            operation(
                &mut output.inner,
                &self.inner,
                &rhs.inner,
                self.context.raw_context_ptr(),
            )
        };
        output
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        self.binary(rhs, self.rows, self.cols, padic_mat_add)
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        self.binary(rhs, self.rows, self.cols, padic_mat_sub)
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::new(self.context.clone(), self.rows, self.cols);
        unsafe {
            padic_mat_neg(
                &mut output.inner,
                &self.inner,
                self.context.raw_context_ptr(),
            )
        };
        output
    }

    pub(super) fn scalar_mul(&self, scalar: &PadicElement) -> Self {
        let mut output = Self::new(self.context.clone(), self.rows, self.cols);
        unsafe {
            padic_mat_scalar_mul_padic(
                &mut output.inner,
                &self.inner,
                scalar.raw_ptr(),
                self.context.raw_context_ptr(),
            )
        };
        output
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        self.binary(rhs, self.rows, rhs.cols, padic_mat_mul)
    }

    pub(super) fn transpose(&self) -> Self {
        let mut output = Self::new(self.context.clone(), self.cols, self.rows);
        unsafe { padic_mat_transpose(&mut output.inner, &self.inner) };
        output
    }
}

impl Clone for PadicMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.context.clone(), self.rows, self.cols);
        unsafe {
            padic_mat_set(
                &mut output.inner,
                &self.inner,
                self.context.raw_context_ptr(),
            )
        };
        output
    }
}

impl Drop for PadicMatrix {
    fn drop(&mut self) {
        unsafe { padic_mat_clear(&mut self.inner) };
    }
}

// SAFETY: a matrix exclusively owns its FLINT allocation. Shared operations
// read their inputs and allocate a distinct output.
unsafe impl Send for PadicMatrix {}
unsafe impl Sync for PadicMatrix {}

impl Clone for IntegerMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpz_mat_set(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl Drop for IntegerMatrix {
    fn drop(&mut self) {
        unsafe { fmpz_mat_clear(self.as_mut_ptr()) };
    }
}

struct IntegerPolynomial {
    inner: fmpz_poly_struct,
}

impl IntegerPolynomial {
    fn new() -> Self {
        let mut inner = MaybeUninit::<fmpz_poly_struct>::uninit();
        unsafe { fmpz_poly_init(inner.as_mut_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn coefficient(&self, degree: usize) -> Integer {
        let mut output = Integer::zero();
        unsafe { fmpz_poly_get_coeff_fmpz(output.as_mut_ptr(), &self.inner, degree as slong) };
        output
    }

    fn as_mut_ptr(&mut self) -> *mut fmpz_poly_struct {
        &mut self.inner
    }
}

impl Drop for IntegerPolynomial {
    fn drop(&mut self) {
        unsafe { fmpz_poly_clear(&mut self.inner) };
    }
}

// SAFETY: the matrix owns every FLINT entry. Its public operations either read
// through shared references or mutate a freshly allocated output, and FLINT's
// per-object fmpz matrix routines are reentrant.
unsafe impl Send for IntegerMatrix {}
unsafe impl Sync for IntegerMatrix {}

pub(super) struct RationalMatrix {
    inner: fmpq_mat_struct,
    rows: usize,
    cols: usize,
}

impl RationalMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        let rows_ffi = slong::try_from(rows).expect("matrix row count was validated");
        let cols_ffi = slong::try_from(cols).expect("matrix column count was validated");
        let mut inner = MaybeUninit::<fmpq_mat_struct>::uninit();
        unsafe { fmpq_mat_init(inner.as_mut_ptr(), rows_ffi, cols_ffi) };
        Self {
            inner: unsafe { inner.assume_init() },
            rows,
            cols,
        }
    }

    pub(super) fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub(super) fn identity(size: usize) -> Self {
        let matrix = Self::new(size, size);
        let one = Rational::one();
        for index in 0..size {
            unsafe { fmpq_set(matrix.entry_ptr(index, index), one.as_ptr()) };
        }
        matrix
    }

    pub(super) fn from_entries<'a>(
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a Rational>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let matrix = Self::new(rows, cols);
        for (index, entry) in entries.enumerate() {
            let row = index / cols;
            let col = index % cols;
            unsafe { fmpq_set(matrix.entry_ptr(row, col), entry.as_ptr()) };
        }
        matrix
    }

    pub(super) fn get(&self, row: usize, col: usize) -> Rational {
        unsafe { Rational::clone_from_ptr(self.entry_ptr(row, col)) }
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpq_mat_add(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpq_mat_sub(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpq_mat_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn scalar_mul(&self, scalar: &Rational) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpq_mat_scalar_mul_fmpq(output.as_mut_ptr(), self.as_ptr(), scalar.as_ptr()) };
        output
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.rows, rhs.cols);
        unsafe { fmpq_mat_mul(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr()) };
        output
    }

    pub(super) fn transpose(&self) -> Self {
        let mut output = Self::new(self.cols, self.rows);
        unsafe { fmpq_mat_transpose(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn characteristic_polynomial(&self) -> Vec<Rational> {
        debug_assert_eq!(self.rows, self.cols);
        let mut polynomial = RationalPolynomial::new();
        unsafe { fmpq_mat_charpoly(polynomial.as_mut_ptr(), self.as_ptr()) };
        (0..=self.rows)
            .map(|degree| polynomial.coefficient(degree))
            .collect()
    }

    pub(super) fn rref(&self) -> (Self, usize) {
        let mut output = Self::new(self.rows, self.cols);
        let rank = unsafe { fmpq_mat_rref(output.as_mut_ptr(), self.as_ptr()) } as usize;
        (output, rank)
    }

    pub(super) fn particular_solution(&self, rhs: &Self) -> Option<Self> {
        debug_assert_eq!(self.rows, rhs.rows);
        let mut output = Self::new(self.cols, rhs.cols);
        let solvable = unsafe {
            fmpq_mat_can_solve_multi_mod(output.as_mut_ptr(), self.as_ptr(), rhs.as_ptr())
        } != 0;
        solvable.then_some(output)
    }

    pub(super) fn rank(&self) -> usize {
        self.rref().1
    }

    pub(super) fn determinant(&self) -> Rational {
        debug_assert_eq!(self.rows, self.cols);
        let mut output = Rational::zero();
        unsafe { fmpq_mat_det(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn entry_ptr(&self, row: usize, col: usize) -> *mut fmpq {
        let offset = row * self.inner.stride as usize + col;
        unsafe { self.inner.entries.add(offset) }
    }

    fn as_ptr(&self) -> *const fmpq_mat_struct {
        &self.inner
    }

    fn as_mut_ptr(&mut self) -> *mut fmpq_mat_struct {
        &mut self.inner
    }
}

impl Clone for RationalMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.rows, self.cols);
        unsafe { fmpq_mat_set(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl Drop for RationalMatrix {
    fn drop(&mut self) {
        unsafe { fmpq_mat_clear(self.as_mut_ptr()) };
    }
}

struct RationalPolynomial {
    inner: fmpq_poly_struct,
}

impl RationalPolynomial {
    fn new() -> Self {
        let mut inner = MaybeUninit::<fmpq_poly_struct>::uninit();
        unsafe { fmpq_poly_init(inner.as_mut_ptr()) };
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn coefficient(&self, degree: usize) -> Rational {
        let mut output = Rational::zero();
        unsafe { fmpq_poly_get_coeff_fmpq(output.as_mut_ptr(), &self.inner, degree as slong) };
        output
    }

    fn as_mut_ptr(&mut self) -> *mut fmpq_poly_struct {
        &mut self.inner
    }
}

impl Drop for RationalPolynomial {
    fn drop(&mut self) {
        unsafe { fmpq_poly_clear(&mut self.inner) };
    }
}

// SAFETY: as above, the matrix owns its entries and shared operations do not
// mutate an input matrix.
unsafe impl Send for RationalMatrix {}
unsafe impl Sync for RationalMatrix {}

pub(super) struct RealBallMatrix {
    inner: arb_mat_struct,
    field: RealBallField,
    rows: usize,
    cols: usize,
}

impl RealBallMatrix {
    fn new(field: RealBallField, rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<arb_mat_struct>::uninit();
        unsafe {
            arb_mat_init(
                inner.as_mut_ptr(),
                slong::try_from(rows).expect("matrix row count was validated"),
                slong::try_from(cols).expect("matrix column count was validated"),
            )
        };
        Self {
            inner: unsafe { inner.assume_init() },
            field,
            rows,
            cols,
        }
    }

    pub(super) fn zeros(field: RealBallField, rows: usize, cols: usize) -> Self {
        Self::new(field, rows, cols)
    }

    pub(super) fn identity(field: RealBallField, size: usize) -> Self {
        let mut output = Self::new(field, size, size);
        unsafe { arb_mat_one(output.as_mut_ptr()) };
        output
    }

    pub(super) fn from_entries<'a>(
        field: RealBallField,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a RealBall>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let output = Self::new(field, rows, cols);
        for (index, entry) in entries.enumerate() {
            unsafe { arb_set(output.entry_ptr(index / cols, index % cols), entry.as_ptr()) };
        }
        output
    }

    pub(super) fn get(&self, row: usize, col: usize) -> RealBall {
        RealBall::clone_from_ptr(self.field, self.entry_ptr(row, col))
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            arb_mat_add(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            arb_mat_sub(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe { arb_mat_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn scalar_mul(&self, scalar: &RealBall) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            arb_mat_scalar_mul_arb(
                output.as_mut_ptr(),
                self.as_ptr(),
                scalar.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, rhs.cols);
        unsafe {
            arb_mat_mul(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn transpose(&self) -> Self {
        let mut output = Self::new(self.field, self.cols, self.rows);
        unsafe { arb_mat_transpose(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn precision(&self) -> slong {
        self.field.precision_bits() as slong
    }

    fn entry_ptr(&self, row: usize, col: usize) -> *mut flint_sys::arb_struct {
        unsafe {
            self.inner
                .entries
                .add(row * self.inner.stride as usize + col)
        }
    }

    fn as_ptr(&self) -> *const arb_mat_struct {
        &self.inner
    }

    fn as_mut_ptr(&mut self) -> *mut arb_mat_struct {
        &mut self.inner
    }
}

impl Clone for RealBallMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe { arb_mat_set(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl Drop for RealBallMatrix {
    fn drop(&mut self) {
        unsafe { arb_mat_clear(self.as_mut_ptr()) };
    }
}

// SAFETY: each matrix owns its FLINT allocation; shared operations read inputs
// and write only to freshly allocated outputs.
unsafe impl Send for RealBallMatrix {}
unsafe impl Sync for RealBallMatrix {}

pub(super) struct ComplexBallMatrix {
    inner: acb_mat_struct,
    field: ComplexBallField,
    rows: usize,
    cols: usize,
}

impl ComplexBallMatrix {
    fn new(field: ComplexBallField, rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<acb_mat_struct>::uninit();
        unsafe {
            acb_mat_init(
                inner.as_mut_ptr(),
                slong::try_from(rows).expect("matrix row count was validated"),
                slong::try_from(cols).expect("matrix column count was validated"),
            )
        };
        Self {
            inner: unsafe { inner.assume_init() },
            field,
            rows,
            cols,
        }
    }

    pub(super) fn zeros(field: ComplexBallField, rows: usize, cols: usize) -> Self {
        Self::new(field, rows, cols)
    }

    pub(super) fn identity(field: ComplexBallField, size: usize) -> Self {
        let mut output = Self::new(field, size, size);
        unsafe { acb_mat_one(output.as_mut_ptr()) };
        output
    }

    pub(super) fn from_entries<'a>(
        field: ComplexBallField,
        rows: usize,
        cols: usize,
        entries: impl ExactSizeIterator<Item = &'a ComplexBall>,
    ) -> Self {
        debug_assert_eq!(entries.len(), rows * cols);
        let output = Self::new(field, rows, cols);
        for (index, entry) in entries.enumerate() {
            unsafe { acb_set(output.entry_ptr(index / cols, index % cols), entry.as_ptr()) };
        }
        output
    }

    pub(super) fn get(&self, row: usize, col: usize) -> ComplexBall {
        ComplexBall::clone_from_ptr(self.field, self.entry_ptr(row, col))
    }

    pub(super) fn add(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            acb_mat_add(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn sub(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            acb_mat_sub(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn neg(&self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe { acb_mat_neg(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    pub(super) fn scalar_mul(&self, scalar: &ComplexBall) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe {
            acb_mat_scalar_mul_acb(
                output.as_mut_ptr(),
                self.as_ptr(),
                scalar.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn matmul(&self, rhs: &Self) -> Self {
        let mut output = Self::new(self.field, self.rows, rhs.cols);
        unsafe {
            acb_mat_mul(
                output.as_mut_ptr(),
                self.as_ptr(),
                rhs.as_ptr(),
                self.precision(),
            )
        };
        output
    }

    pub(super) fn transpose(&self) -> Self {
        let mut output = Self::new(self.field, self.cols, self.rows);
        unsafe { acb_mat_transpose(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn precision(&self) -> slong {
        self.field.precision_bits() as slong
    }

    fn entry_ptr(&self, row: usize, col: usize) -> *mut flint_sys::acb_struct {
        unsafe {
            self.inner
                .entries
                .add(row * self.inner.stride as usize + col)
        }
    }

    fn as_ptr(&self) -> *const acb_mat_struct {
        &self.inner
    }

    fn as_mut_ptr(&mut self) -> *mut acb_mat_struct {
        &mut self.inner
    }
}

impl Clone for ComplexBallMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::new(self.field, self.rows, self.cols);
        unsafe { acb_mat_set(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl Drop for ComplexBallMatrix {
    fn drop(&mut self) {
        unsafe { acb_mat_clear(self.as_mut_ptr()) };
    }
}

// SAFETY: as for real-ball matrices, inputs are immutable and outputs own all
// FLINT storage they mutate.
unsafe impl Send for ComplexBallMatrix {}
unsafe impl Sync for ComplexBallMatrix {}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    fn reset_counts() {
        DENSE_CONVERSIONS.with(|counts| counts.set([0; 4]));
    }

    fn counts() -> [usize; 4] {
        DENSE_CONVERSIONS.with(std::cell::Cell::get)
    }

    #[test]
    fn f2_construction_and_matrix_chains_stay_bit_packed() {
        reset_counts();
        let ring = ResidueRing::new(2).unwrap();
        let entries = (0..195)
            .map(|index| Integer::from((index % 3 == 0) as u8))
            .collect::<Vec<_>>();
        let matrix = ModularMatrix::from_integer_entries(ring.clone(), 3, 65, entries.iter());
        let ModularMatrixStorage::F2(packed) = &matrix.backend else {
            panic!("modulus two must select F2 storage");
        };
        assert_eq!(packed.allocated_words(), 3 * 65_usize.div_ceil(64));

        let transposed = matrix.transpose();
        let product = matrix.matmul(&transposed).unwrap();
        let reduced = product.add(&product).unwrap().rref().0;
        let rhs = ModularMatrix::zeros(ring.clone(), 3, 1);
        let _ = reduced.solve_left(&rhs);
        let _ = reduced.nullspace();
        let _ = reduced.pluq();
        assert_eq!(counts(), [0; 4]);

        let _ = product.characteristic_polynomial();
        assert_eq!(counts(), [1, 0, 0, 0]);
    }

    #[test]
    fn binary_extension_rank_and_rref_do_not_visit_flint() {
        reset_counts();
        let field = FiniteField::with_modulus(2, vec![1, 1, 1]).unwrap();
        let entries = [
            field.element_from_coefficients(&[1, 0]).unwrap(),
            field.element_from_coefficients(&[0, 1]).unwrap(),
            field.element_from_coefficients(&[1, 1]).unwrap(),
            field.element_from_coefficients(&[1, 0]).unwrap(),
        ];
        let matrix = FiniteFieldMatrix::from_entries(field.clone(), 2, 2, entries.iter());
        let _ = matrix.rank();
        let reduced = matrix.rref().0;
        let _ = reduced.transpose().matmul(&matrix);
        let rhs = FiniteFieldMatrix::identity(field.clone(), 2);
        let _ = matrix.solve_left(&rhs);
        let _ = matrix.nullspace();
        let _ = matrix.pluq();
        assert_eq!(counts(), [0; 4]);

        let _ = matrix.determinant();
        let _ = matrix.characteristic_polynomial();
        assert_eq!(counts(), [0, 0, 0, 2]);
    }

    #[test]
    fn power_of_two_howell_changes_authoritative_storage_once() {
        reset_counts();
        let ring = ResidueRing::new(4).unwrap();
        let entries = [
            Integer::from(2),
            Integer::from(0),
            Integer::from(0),
            Integer::from(2),
        ];
        let matrix = ModularMatrix::from_integer_entries(ring, 2, 2, entries.iter());
        assert!(matches!(
            &matrix.backend,
            ModularMatrixStorage::PowerOfTwo(_)
        ));

        let howell = matrix.howell(2);
        assert!(matches!(&howell.backend, ModularMatrixStorage::Word(_)));
        let _ = howell.transpose().howell(2);
        assert_eq!(counts(), [0, 0, 1, 0]);
    }

    #[test]
    fn wide_howell_pads_inside_the_authoritative_transition() {
        reset_counts();
        let ring = ResidueRing::new(4).unwrap();
        let entries = [Integer::from(2), Integer::from(0), Integer::from(2)];
        let matrix = ModularMatrix::from_integer_entries(ring, 1, 3, entries.iter());

        let howell = matrix.howell(3);

        assert!(matches!(&howell.backend, ModularMatrixStorage::Word(_)));
        assert_eq!(howell.get_integer(0, 0), Integer::from(2));
        assert_eq!(counts(), [0, 0, 1, 0]);
    }
}
