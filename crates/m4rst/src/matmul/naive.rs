// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Naive bit-packed matmul over F_2.
//!
//! For each `(i, k)` with `A[i, k] = 1`, XOR row `k` of `B` into row
//! `i` of `C`. Word-stride inner XOR — `O(rows_a · cols_a · width_b)`
//! word ops vs schoolbook's `O(rows_a · cols_a · cols_b)` bit ops, a
//! 64× reduction for free. Equivalent and bit-identical to M4RI's
//! `mzd_mul_naive`.
//!
//! Generic over [`MatrixRead`] / [`MatrixWrite`] so the kernel covers
//! owned matrices, immutable views, and mutable views (Strassen
//! quadrants) without duplication.

use crate::mzd::{Mzd, RADIX};
use crate::view::{MatrixRead, MatrixWrite};

/// `C = A · B` over F_2. Freshly-allocated `C`. Panics on shape
/// mismatch.
pub fn matmul_naive<A: MatrixRead, B: MatrixRead>(a: &A, b: &B) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_naive_into(&mut c, a, b);
    c
}

/// `dst ^= a · b` over F_2 (XOR-accumulates into the destination).
/// For overwrite semantics, pass a zeroed `dst`. Caller-managed
/// destination, shape-checked.
pub fn matmul_naive_into<D, A, B>(dst: &mut D, a: &A, b: &B)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_naive_into: shape mismatch ({}×{}) · ({}×{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    assert_eq!(
        dst.rows(),
        a.rows(),
        "matmul_naive_into: dst.rows() != a.rows()"
    );
    assert_eq!(
        dst.cols(),
        b.cols(),
        "matmul_naive_into: dst.cols() != b.cols()"
    );
    if a.rows() == 0 || b.cols() == 0 || a.cols() == 0 {
        return;
    }
    let a_w = a.width();
    for i in 0..a.rows() {
        // Snapshot A's i-th row up front so the row-iteration borrow
        // doesn't overlap with `dst`'s mutable borrow inside the
        // xor_row_from call. The snapshot is a slice, so this is
        // a borrow-only — no data copy.
        let a_row_ptr = a.row_words(i).as_ptr();
        for wi in 0..a_w {
            // Safety: a_row has length a_w (== a.width()), iteration
            // index wi < a_w.
            let word = unsafe { *a_row_ptr.add(wi) };
            if word == 0 {
                continue;
            }
            let base_k = wi * RADIX;
            let mut bits = word;
            while bits != 0 {
                let k = base_k + bits.trailing_zeros() as usize;
                debug_assert!(k < a.cols());
                dst.xor_row_from(i, b, k);
                bits &= bits - 1;
            }
        }
    }
}
