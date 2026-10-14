// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Row-major dense matrix over Z/2^n for n ∈ {8, 16, 32, 64}.
//!
//! At these widths the native unsigned integer type *is* the ring:
//! wrapping arithmetic on `uN` equals arithmetic in Z/2^N. Smaller
//! prime-power moduli use the packed [`crate::zpn::MatZpn`] representation.

use rayon::prelude::*;

/// Bound on element types that *are* their own ring under wrapping
/// arithmetic — i.e. `Z/2^BITS` is just the native uN. Implemented for
/// `u8`, `u16`, `u32`, `u64`, `u128`.
pub trait Z2n: Copy + Default + Send + Sync + 'static + PartialEq {
    /// Bit-width of the ring; `2^BITS` is the modulus.
    const BITS: u32;
    /// Wrapping addition in `Z/2^BITS`.
    fn wrapping_add(self, other: Self) -> Self;
    /// Wrapping subtraction in `Z/2^BITS`.
    fn wrapping_sub(self, other: Self) -> Self;
    /// Wrapping multiplication in `Z/2^BITS`.
    fn wrapping_mul(self, other: Self) -> Self;
    /// Wrapping negation in `Z/2^BITS`.
    fn wrapping_neg(self) -> Self;
    /// Build a ring element from a `u64` by truncation to `BITS`.
    fn from_u64(x: u64) -> Self;
    /// Project a ring element back into a `u64` (zero-extended).
    fn to_u64(self) -> u64;
}

macro_rules! impl_z2n {
    ($t:ty, $bits:expr) => {
        impl Z2n for $t {
            const BITS: u32 = $bits;
            #[inline]
            fn wrapping_add(self, o: Self) -> Self {
                <$t>::wrapping_add(self, o)
            }
            #[inline]
            fn wrapping_sub(self, o: Self) -> Self {
                <$t>::wrapping_sub(self, o)
            }
            #[inline]
            fn wrapping_mul(self, o: Self) -> Self {
                <$t>::wrapping_mul(self, o)
            }
            #[inline]
            fn wrapping_neg(self) -> Self {
                <$t>::wrapping_neg(self)
            }
            #[inline]
            fn from_u64(x: u64) -> Self {
                x as $t
            }
            #[inline]
            fn to_u64(self) -> u64 {
                self as u64
            }
        }
    };
}

impl_z2n!(u8, 8);
impl_z2n!(u16, 16);
impl_z2n!(u32, 32);
impl_z2n!(u64, 64);
impl_z2n!(u128, 128);

/// Row-major dense matrix over `Z/2^BITS` for `T: Z2n`. The native uint
/// `T` is the cell type; wrapping arithmetic IS the ring operation.
#[derive(Clone, Debug)]
pub struct Mat<T: Z2n> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major backing buffer of length `rows * cols`.
    pub data: Vec<T>,
}

impl<T: Z2n> Mat<T> {
    /// Construct an all-zero `rows × cols` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }

    /// Construct a `rows × cols` matrix whose `(i, j)` cell is `f(i, j)`.
    pub fn from_fn<F: FnMut(usize, usize) -> T>(rows: usize, cols: usize, mut f: F) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i * cols + j] = f(i, j);
            }
        }
        m
    }

    /// Construct from one exact-size row-major stream without an intermediate
    /// matrix-sized staging buffer.
    pub fn from_values(rows: usize, cols: usize, values: impl ExactSizeIterator<Item = T>) -> Self {
        assert_eq!(values.len(), rows * cols, "from_values: dimension mismatch");
        Self {
            rows,
            cols,
            data: values.collect(),
        }
    }

    /// Read cell `(i, j)`. Panics on out-of-bounds in debug builds.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[i * self.cols + j]
    }

    /// Write `v` to cell `(i, j)`. Panics on out-of-bounds in debug builds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: T) {
        self.data[i * self.cols + j] = v;
    }
}

impl<T: Z2n> PartialEq for Mat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

/// Multiplicative inverse of an odd element of Z/2^BITS via Newton iteration.
/// Generic over all Z2n types (u8/u16/u32/u64).
///
/// Starting from y_0 = 1 (inverse mod 2), each step `y_{k+1} = y_k·(2 − x·y_k)`
/// doubles the precision. log₂(BITS) iterations reach full precision.
pub fn odd_inverse<T: Z2n>(x: T) -> T {
    debug_assert_eq!(x.to_u64() & 1, 1, "odd_inverse: x must be odd");
    let two = T::from_u64(2);
    let mut y = T::from_u64(1);
    let mut precision: u32 = 1;
    while precision < T::BITS {
        y = y.wrapping_mul(two.wrapping_sub(x.wrapping_mul(y)));
        precision *= 2;
    }
    y
}

impl<T: Z2n> Mat<T> {
    /// Borrow an immutable, full-extent strided view of this matrix.
    /// Cheap (no copy); used by Strassen for zero-copy quadrant access.
    #[inline]
    pub fn view(&self) -> MatView<'_, T> {
        MatView::from_mat(self)
    }

    /// Element-wise A + B (wrapping). Parallel over output rows.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "add: dimension mismatch"
        );
        let cols = self.cols;
        let mut out = Mat::<T>::zeros(self.rows, self.cols);
        if cols == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| {
                let a = &self.data[i * cols..(i + 1) * cols];
                let b = &other.data[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    row[j] = a[j].wrapping_add(b[j]);
                }
            });
        out
    }

    /// Element-wise A - B (wrapping).
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "sub: dimension mismatch"
        );
        let cols = self.cols;
        let mut out = Mat::<T>::zeros(self.rows, self.cols);
        if cols == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| {
                let a = &self.data[i * cols..(i + 1) * cols];
                let b = &other.data[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    row[j] = a[j].wrapping_sub(b[j]);
                }
            });
        out
    }

    /// Element-wise -A (wrapping). Identity on Z/2 (x = -x), two's-complement
    /// otherwise.
    pub fn neg(&self) -> Self {
        let cols = self.cols;
        let mut out = Mat::<T>::zeros(self.rows, self.cols);
        if cols == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| {
                let a = &self.data[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    row[j] = a[j].wrapping_neg();
                }
            });
        out
    }

    /// Broadcast scalar multiply: s · A (wrapping, element-wise).
    pub fn scalar_mul(&self, s: T) -> Self {
        let cols = self.cols;
        let mut out = Mat::<T>::zeros(self.rows, self.cols);
        if cols == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| {
                let a = &self.data[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    row[j] = a[j].wrapping_mul(s);
                }
            });
        out
    }

    /// Project each entry onto F_2 (= Z/2). Extracts the low bit;
    /// result is 0 or 1 as a T value. This is the "residue field"
    /// reduction used by the Hensel lifters.
    pub fn reduce_mod_2(&self) -> Self {
        let one = T::from_u64(1);
        let cols = self.cols;
        let mut out = Mat::<T>::zeros(self.rows, self.cols);
        if cols == 0 {
            return out;
        }
        out.data
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| {
                let src = &self.data[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    let bit = src[j].to_u64() & 1;
                    row[j] = if bit == 1 { one } else { T::default() };
                }
            });
        out
    }

    /// Trace: sum of the diagonal entries, wrapping in Z/2^BITS. For
    /// non-square A, sums along `min(rows, cols)`.
    pub fn trace(&self) -> T {
        let d = self.rows.min(self.cols);
        let mut acc = T::default();
        for i in 0..d {
            acc = acc.wrapping_add(self.data[i * self.cols + i]);
        }
        acc
    }

    /// Matrix-vector product `A · v`, row-by-row inner product with
    /// wrapping arithmetic in Z/2^BITS. Parallel over output rows via
    /// rayon. Primitive behind Wiedemann / Lanczos iteration.
    pub fn mat_vec(&self, v: &[T]) -> Vec<T> {
        assert_eq!(self.cols, v.len(), "mat_vec: A.cols != v.len()");
        let cols = self.cols;
        let rows = self.rows;
        let mut out = vec![T::default(); rows];
        out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
            let row = &self.data[i * cols..(i + 1) * cols];
            let mut acc = T::default();
            for j in 0..cols {
                acc = acc.wrapping_add(row[j].wrapping_mul(v[j]));
            }
            *out_i = acc;
        });
        out
    }

    /// Blocked parallel transpose. Input (rows × cols) → output (cols × rows).
    /// Tile size B=32 keeps each tile's working set in L1; outer parallelism
    /// partitions the output by row stripes (= input column stripes), which
    /// are disjoint and so safe for rayon.
    pub fn transpose(&self) -> Self {
        let (rows, cols) = (self.rows, self.cols);
        let mut out = Mat::<T>::zeros(cols, rows);
        if rows == 0 || cols == 0 {
            return out;
        }
        const B: usize = 32;

        out.data
            .par_chunks_mut(rows * B)
            .enumerate()
            .for_each(|(blk, chunk)| {
                let bj = blk * B;
                // Last block may be shorter.
                let this_b = chunk.len() / rows;
                for bi in (0..rows).step_by(B) {
                    let bi_end = (bi + B).min(rows);
                    for i in bi..bi_end {
                        let src_row = &self.data[i * cols + bj..i * cols + bj + this_b];
                        for (off, &v) in src_row.iter().enumerate() {
                            chunk[off * rows + i] = v;
                        }
                    }
                }
            });
        out
    }
}

/// CSR-format sparse matrix over Z/2^BITS. Used for block Wiedemann
/// and the sparse-to-dense lift paths, where the "A" input is sparse
/// but "B" (and hence "C") is dense.
///
/// Layout: `row_ptr[i]` is the starting index in `col_ind` / `values`
/// of row `i`; `row_ptr[rows]` is the total nnz. `col_ind` holds the
/// column index of each nonzero, `values` the corresponding element.
/// Zero entries are elided — iterating row i walks exactly the
/// nonzeros in that row.
#[derive(Clone, Debug)]
pub struct SparseCsrMat<T: Z2n> {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Starting index in `col_ind`/`values` for each row (length `rows + 1`).
    pub row_ptr: Vec<usize>,
    /// Column index of each stored nonzero, in row-major order.
    pub col_ind: Vec<usize>,
    /// Value of each stored nonzero, parallel to `col_ind`.
    pub values: Vec<T>,
}

impl<T: Z2n> SparseCsrMat<T> {
    /// Build a CSR from a dense `Mat<T>` by scanning row-major and
    /// pushing nonzero entries.
    pub fn from_dense(m: &Mat<T>) -> Self {
        let mut row_ptr = Vec::with_capacity(m.rows + 1);
        row_ptr.push(0);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        let zero = T::default();
        for i in 0..m.rows {
            for j in 0..m.cols {
                let v = m.data[i * m.cols + j];
                if v != zero {
                    col_ind.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_ind.len());
        }
        Self {
            rows: m.rows,
            cols: m.cols,
            row_ptr,
            col_ind,
            values,
        }
    }

    /// Number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse A · dense B → dense C over Z/2^BITS. Walks A row by row,
    /// accumulating `c[i, :] += A[i, col] · B[col, :]` per nonzero.
    /// Parallel across output rows via rayon. Cost: `O(nnz(A) · n)`
    /// multiply-adds, versus `O(m · k · n)` for the dense matmul.
    pub fn matmul_dense(&self, b: &Mat<T>) -> Mat<T> {
        assert_eq!(
            self.cols, b.rows,
            "sparse matmul: A.cols ({}) != B.rows ({})",
            self.cols, b.rows
        );
        let (m, n) = (self.rows, b.cols);
        let mut c = Mat::<T>::zeros(m, n);
        if n == 0 {
            return c;
        }
        c.data.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let row_start = self.row_ptr[i];
            let row_end = self.row_ptr[i + 1];
            for p in row_start..row_end {
                let col = self.col_ind[p];
                let val = self.values[p];
                let b_row = &b.data[col * n..(col + 1) * n];
                for j in 0..n {
                    c_row[j] = c_row[j].wrapping_add(val.wrapping_mul(b_row[j]));
                }
            }
        });
        c
    }
}

/// Read-only strided view into a dense matrix. Used by Strassen for
/// zero-copy quadrant access: `a.view().sub_view(0, 0, m/2, k/2)` etc.
#[derive(Clone, Copy)]
pub struct MatView<'a, T: Z2n> {
    data: &'a [T],
    /// Row stride of the parent buffer (elements per row).
    stride: usize,
    /// Offset into `data` where this view's (0,0) lives.
    offset: usize,
    /// Number of rows in this view.
    pub rows: usize,
    /// Number of columns in this view.
    pub cols: usize,
}

impl<'a, T: Z2n> MatView<'a, T> {
    /// Build a full-extent view over the rows and columns of `m`.
    pub fn from_mat(m: &'a Mat<T>) -> Self {
        Self {
            data: &m.data,
            stride: m.cols,
            offset: 0,
            rows: m.rows,
            cols: m.cols,
        }
    }

    /// Re-slice into a sub-view at row offset `r`, column offset `c`,
    /// with `rows × cols` extent. Coordinates are relative to this view.
    pub fn sub_view(&self, r: usize, c: usize, rows: usize, cols: usize) -> Self {
        debug_assert!(r + rows <= self.rows && c + cols <= self.cols);
        Self {
            data: self.data,
            stride: self.stride,
            offset: self.offset + r * self.stride + c,
            rows,
            cols,
        }
    }

    /// Read cell `(i, j)` via the view's stride. Panics on out-of-bounds
    /// in debug builds.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        debug_assert!(i < self.rows && j < self.cols);
        self.data[self.offset + i * self.stride + j]
    }

    /// Row i as a contiguous slice (independent of view stride).
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        debug_assert!(i < self.rows);
        let start = self.offset + i * self.stride;
        &self.data[start..start + self.cols]
    }
}
