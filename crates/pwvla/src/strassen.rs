// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strassen-Winograd recursion for u16 matmul, built on top of the
//! packed+parallel NEON kernel.
//!
//! 7 sub-products + 15 additions per level (vs 8+0 classical and
//! 7+18 classical-Strassen). Zero-copy quadrant access via MatView.
//!
//! Verified formulas (derived from the 16-term expansion, see below):
//!
//!   S1 = A21 + A22,  S2 = S1 - A11,  S3 = A11 - A21,  S4 = A12 - S2
//!   T1 = B12 - B11,  T2 = B22 - T1,  T3 = B22 - B12,  T4 = T2 - B21
//!
//!   M1 = S2 * T2
//!   M2 = A11 * B11
//!   M3 = A12 * B21
//!   M4 = S4 * B22
//!   M5 = S1 * T1
//!   M6 = A22 * T4
//!   M7 = S3 * T3
//!
//!   C11 = M2 + M3
//!   C12 = M1 + M2 + M4 + M5
//!   C21 = M1 + M2 + M7 - M6
//!   C22 = M1 + M2 + M5 + M7
//!
//! To avoid allocating U = M1+M2 and V = U+M5 as separate temps, we
//! stage through the output block itself:
//!
//!   out[C22] = M1 + M2           (temp U)
//!   out[C21] = out[C22] + M7 - M6   (finalize C21 using U)
//!   out[C22] += M5               (temp V)
//!   out[C12] = out[C22] + M4     (finalize C12 using V)
//!   out[C22] += M7               (finalize C22)
//!   out[C11] = M2 + M3
//!
//! Over Z/2^n the arithmetic is exact wrapping — Strassen-Winograd's
//! identities hold with no numerical concerns.

use crate::mat::{Mat, MatView};
use crate::neon;
use rayon::prelude::*;

/// Base-case threshold. Below this, we call the NEON kernel directly.
/// Tuned so that the Strassen overhead (15 adds on n²/4 elements + 7
/// recursive products + allocations) is smaller than one saved matmul
/// at this size. With our ~1 Tops/s base kernel, that crossover sits
/// around n≈2048–4096 — high because the base is nearly peak. At sizes
/// below this, matmul_u16 falls through to plain NEON and Strassen
/// adds zero overhead.
pub const CROSSOVER: usize = 2048;

/// Strassen-Winograd matmul over Z/2^16 — `C = A · B` — using the
/// default [`CROSSOVER`] threshold. Convenience wrapper over
/// [`matmul_u16_with_crossover`].
pub fn matmul_u16(a: &Mat<u16>, b: &Mat<u16>) -> Mat<u16> {
    matmul_u16_with_crossover(a, b, CROSSOVER)
}

/// Same as `matmul_u16` but with an explicit crossover threshold.
/// Useful for testing (smaller threshold → deeper recursion exercised).
pub fn matmul_u16_with_crossover(a: &Mat<u16>, b: &Mat<u16>, crossover: usize) -> Mat<u16> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    strassen_winograd(a.view(), b.view(), crossover)
}

fn strassen_winograd(a: MatView<u16>, b: MatView<u16>, crossover: usize) -> Mat<u16> {
    let (m, k, n) = (a.rows, a.cols, b.cols);

    // Empty matrices are valid mathematical inputs, including when a caller
    // uses an explicit crossover for testing.  Return before the row-chunked
    // assembly helpers, whose chunk widths are necessarily positive.
    if m == 0 || k == 0 || n == 0 {
        return Mat::<u16>::zeros(m, n);
    }
    assert!(crossover > 0, "Strassen crossover must be positive");

    // Base case: our fast kernel.
    if m < crossover || k < crossover || n < crossover {
        return neon::matmul_u16_view(a, b);
    }

    // Pad any odd dimension to even. We materialize into fresh owned
    // matrices once per level; sub-views below are then guaranteed to fit.
    if (m | k | n) & 1 == 1 {
        let pm = m + (m & 1);
        let pk = k + (k & 1);
        let pn = n + (n & 1);
        let ap = pad(a, pm, pk);
        let bp = pad(b, pk, pn);
        let cp = strassen_winograd(ap.view(), bp.view(), crossover);
        return trim(&cp, m, n);
    }

    let (m2, k2, n2) = (m / 2, k / 2, n / 2);

    // Zero-copy quadrant views.
    let a11 = a.sub_view(0, 0, m2, k2);
    let a12 = a.sub_view(0, k2, m2, k2);
    let a21 = a.sub_view(m2, 0, m2, k2);
    let a22 = a.sub_view(m2, k2, m2, k2);
    let b11 = b.sub_view(0, 0, k2, n2);
    let b12 = b.sub_view(0, n2, k2, n2);
    let b21 = b.sub_view(k2, 0, k2, n2);
    let b22 = b.sub_view(k2, n2, k2, n2);

    // 8 input sums (4 on A side, 4 on B side).
    let mut s1 = Mat::<u16>::zeros(m2, k2);
    let mut s2 = Mat::<u16>::zeros(m2, k2);
    let mut s3 = Mat::<u16>::zeros(m2, k2);
    let mut s4 = Mat::<u16>::zeros(m2, k2);
    add(a21, a22, &mut s1);
    sub(s1.view(), a11, &mut s2);
    sub(a11, a21, &mut s3);
    sub(a12, s2.view(), &mut s4);

    let mut t1 = Mat::<u16>::zeros(k2, n2);
    let mut t2 = Mat::<u16>::zeros(k2, n2);
    let mut t3 = Mat::<u16>::zeros(k2, n2);
    let mut t4 = Mat::<u16>::zeros(k2, n2);
    sub(b12, b11, &mut t1);
    sub(b22, t1.view(), &mut t2);
    sub(b22, b12, &mut t3);
    sub(t2.view(), b21, &mut t4);

    // 7 recursive products.
    let m1 = strassen_winograd(s2.view(), t2.view(), crossover);
    let m2_prod = strassen_winograd(a11, b11, crossover);
    let m3 = strassen_winograd(a12, b21, crossover);
    let m4 = strassen_winograd(s4.view(), b22, crossover);
    let m5 = strassen_winograd(s1.view(), t1.view(), crossover);
    let m6 = strassen_winograd(a22, t4.view(), crossover);
    let m7 = strassen_winograd(s3.view(), t3.view(), crossover);

    // Assemble output. Each block is written via rayon over its rows
    // (cols-wide chunks), autovectorizing the inner u16 wrapping ops.
    //
    // We build the four quadrants independently here rather than
    // staging through out[C22], because processing the 4 blocks in
    // parallel with rayon::join gives better throughput than the
    // "avoid temps" trick — the temp allocation cost is tiny compared
    // to the combination memory bandwidth.
    let mut out = Mat::<u16>::zeros(m, n);

    // C11 = M2 + M3              — writes top-left
    // C12 = M1 + M2 + M5 + M4    — writes top-right
    // C21 = M1 + M2 + M7 - M6    — writes bottom-left
    // C22 = M1 + M2 + M5 + M7    — writes bottom-right
    //
    // The 4 blocks write disjoint row ranges / column ranges, so we
    // can partition `out.data` into row stripes and fill quadrants
    // row-parallel.
    let (top, bot) = out.data.split_at_mut(m2 * n);
    rayon::join(
        || {
            // Top half: C11 (cols 0..n2) and C12 (cols n2..n) per row.
            top.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                let (c11_row, c12_row) = row.split_at_mut(n2);
                let off = i * n2;
                for j in 0..n2 {
                    c11_row[j] = m2_prod.data[off + j].wrapping_add(m3.data[off + j]);
                }
                for j in 0..n2 {
                    c12_row[j] = m1.data[off + j]
                        .wrapping_add(m2_prod.data[off + j])
                        .wrapping_add(m5.data[off + j])
                        .wrapping_add(m4.data[off + j]);
                }
            });
        },
        || {
            // Bottom half: C21 (cols 0..n2) and C22 (cols n2..n).
            bot.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                let (c21_row, c22_row) = row.split_at_mut(n2);
                let off = i * n2;
                for j in 0..n2 {
                    c21_row[j] = m1.data[off + j]
                        .wrapping_add(m2_prod.data[off + j])
                        .wrapping_add(m7.data[off + j])
                        .wrapping_sub(m6.data[off + j]);
                }
                for j in 0..n2 {
                    c22_row[j] = m1.data[off + j]
                        .wrapping_add(m2_prod.data[off + j])
                        .wrapping_add(m5.data[off + j])
                        .wrapping_add(m7.data[off + j]);
                }
            });
        },
    );

    out
}

// Row-wise parallel adds/subs. rustc autovectorizes u16 wrapping_add
// into NEON `vaddq_u16` when the inner loop sees contiguous slices.
fn add(a: MatView<u16>, b: MatView<u16>, out: &mut Mat<u16>) {
    let cols = out.cols;
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, out_row)| {
            let a_row = a.row(i);
            let b_row = b.row(i);
            for j in 0..cols {
                out_row[j] = a_row[j].wrapping_add(b_row[j]);
            }
        });
}

fn sub(a: MatView<u16>, b: MatView<u16>, out: &mut Mat<u16>) {
    let cols = out.cols;
    out.data
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, out_row)| {
            let a_row = a.row(i);
            let b_row = b.row(i);
            for j in 0..cols {
                out_row[j] = a_row[j].wrapping_sub(b_row[j]);
            }
        });
}

fn pad(v: MatView<u16>, rows: usize, cols: usize) -> Mat<u16> {
    let mut out = Mat::<u16>::zeros(rows, cols);
    for i in 0..v.rows {
        for j in 0..v.cols {
            out.data[i * cols + j] = v.get(i, j);
        }
    }
    out
}

fn trim(src: &Mat<u16>, rows: usize, cols: usize) -> Mat<u16> {
    let mut out = Mat::<u16>::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            out.data[i * cols + j] = src.data[i * src.cols + j];
        }
    }
    out
}
