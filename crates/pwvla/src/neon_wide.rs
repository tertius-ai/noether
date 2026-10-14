// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! NEON matmul kernels for u8, u32, u64 — the word-size siblings of the
//! u16 kernel in `neon.rs`.
//!
//! Design notes for each width:
//!
//! - **u32**: `uint32x4_t` holds 4 lanes. Register tile is **4×8** (2
//!   regs per row × 4 rows = 8 C regs, 4 A dup + 2 B = 14 live). Uses
//!   `vmlaq_u32` for wrapping fused multiply-add.
//! - **u8**: `uint8x16_t` holds 16 lanes. Register tile is **4×32** (2
//!   regs per row × 4 rows = 8 C regs, same shape as AVX2 u16). Uses
//!   `vmlaq_u8` for wrapping FMA.
//! - **u64**: AArch64 base NEON **lacks** `vmulq_u64` — 64-bit vector
//!   integer multiply needs SVE or PMULL extensions that aren't in the
//!   common baseline. The entry point therefore delegates to the
//!   portable `scalar::matmul`, which gives one wrapping mul per cycle
//!   (same as a degenerate-NEON emulation would). Stubbed-NEON would
//!   be dishonest; the API is here for symmetry so callers can migrate
//!   once a real SIMD path lands.
//!
//! The outer BLIS structure (pack, block, rayon, edge scratch) is a
//! faithful copy of the u16 path — changing only the microkernel + the
//! MR/NR constants per width.

use crate::mat::{Mat, MatView};
use core::arch::aarch64::*;
use rayon::prelude::*;

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

// ============================================================================
// u32 path
// ============================================================================

const MR32: usize = 4;
const NR32: usize = 8;
const MC32: usize = 128;
const KC32: usize = 256;
const NC32: usize = 256;

/// NEON dense matmul over Z/2^32 — `C = A · B`. Returns a freshly
/// allocated `C`. Convenience wrapper over [`matmul_u32_view`].
pub fn matmul_u32(a: &Mat<u32>, b: &Mat<u32>) -> Mat<u32> {
    matmul_u32_view(a.view(), b.view())
}

/// NEON dense matmul over Z/2^32 from strided views. Returns a freshly
/// allocated `C`.
pub fn matmul_u32_view(a: MatView<u32>, b: MatView<u32>) -> Mat<u32> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    let (m, _, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<u32>::zeros(m, n);
    addmul_u32_into(&mut c, a, b);
    c
}

/// NEON fused multiply-add over Z/2^32: `C += A · B` in-place.
/// Convenience wrapper over [`addmul_u32_into`].
pub fn addmul_u32(c: &mut Mat<u32>, a: &Mat<u32>, b: &Mat<u32>) {
    addmul_u32_into(c, a.view(), b.view());
}

/// NEON fused multiply-add over Z/2^32 from strided views: `C += A · B`
/// in-place. The kernel all other u32 entry points funnel into.
pub fn addmul_u32_into(c: &mut Mat<u32>, a: MatView<u32>, b: MatView<u32>) {
    assert_eq!(a.cols, b.rows, "addmul_u32: inner dimension mismatch");
    assert_eq!(
        (c.rows, c.cols),
        (a.rows, b.cols),
        "addmul_u32: C dims must match A·B"
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);

    if m == 0 || k == 0 || n == 0 {
        return;
    }
    if m * n * k < 32_768 || m < MR32 || n < NR32 {
        for i in 0..m {
            for p in 0..k {
                let aip = a.get(i, p);
                for j in 0..n {
                    let bij = b.get(p, j);
                    let cij = c.data[i * n + j];
                    c.data[i * n + j] = cij.wrapping_add(aip.wrapping_mul(bij));
                }
            }
        }
        return;
    }

    for jc in (0..n).step_by(NC32) {
        let nc = (jc + NC32).min(n) - jc;
        let nc_pad = round_up(nc, NR32);

        for pc in (0..k).step_by(KC32) {
            let kc = (pc + KC32).min(k) - pc;
            let b_block = b.sub_view(pc, jc, kc, nc);
            let mut b_tilde = vec![0u32; kc * nc_pad];
            pack_b_u32(b_block, &mut b_tilde);

            c.data
                .par_chunks_mut(MC32 * n)
                .enumerate()
                .for_each(|(ib, c_stripe)| {
                    let ic = ib * MC32;
                    let mc = c_stripe.len() / n;
                    let mc_pad = round_up(mc, MR32);
                    let a_block = a.sub_view(ic, pc, mc, kc);
                    let mut a_tilde = vec![0u32; mc_pad * kc];
                    pack_a_u32(a_block, &mut a_tilde);
                    run_kernels_u32(
                        &a_tilde, &b_tilde, c_stripe, mc, mc_pad, nc, nc_pad, kc, n, jc,
                    );
                });
        }
    }
}

fn pack_a_u32(block: MatView<u32>, a_tilde: &mut [u32]) {
    let (mc, kc) = (block.rows, block.cols);
    let num_panels = round_up(mc, MR32) / MR32;
    for panel in 0..num_panels {
        let panel_off = panel * MR32 * kc;
        let base_row = panel * MR32;
        for p in 0..kc {
            for r in 0..MR32 {
                let src_row = base_row + r;
                a_tilde[panel_off + p * MR32 + r] = if src_row < mc {
                    block.get(src_row, p)
                } else {
                    0
                };
            }
        }
    }
}

fn pack_b_u32(block: MatView<u32>, b_tilde: &mut [u32]) {
    let (kc, nc) = (block.rows, block.cols);
    let num_panels = round_up(nc, NR32) / NR32;
    for panel in 0..num_panels {
        let panel_off = panel * NR32 * kc;
        let base_col = panel * NR32;
        for p in 0..kc {
            for cc in 0..NR32 {
                let src_col = base_col + cc;
                b_tilde[panel_off + p * NR32 + cc] = if src_col < nc {
                    block.get(p, src_col)
                } else {
                    0
                };
            }
        }
    }
}

fn run_kernels_u32(
    a_tilde: &[u32],
    b_tilde: &[u32],
    c_stripe: &mut [u32],
    mc: usize,
    mc_pad: usize,
    nc: usize,
    nc_pad: usize,
    kc: usize,
    n: usize,
    jc: usize,
) {
    for jr in (0..nc_pad).step_by(NR32) {
        let b_panel = &b_tilde[(jr / NR32) * NR32 * kc..];
        let nr = (jr + NR32).min(nc) - jr;

        for ir in (0..mc_pad).step_by(MR32) {
            let a_panel = &a_tilde[(ir / MR32) * MR32 * kc..];
            let mr = (ir + MR32).min(mc) - ir;

            unsafe {
                if mr == MR32 && nr == NR32 {
                    let c_ptr = c_stripe.as_mut_ptr().add(ir * n + jc + jr);
                    kernel_packed_4x8_u32(a_panel.as_ptr(), b_panel.as_ptr(), c_ptr, kc, n);
                } else {
                    let mut tmp = [0u32; MR32 * NR32];
                    for r in 0..mr {
                        for cc in 0..nr {
                            tmp[r * NR32 + cc] = c_stripe[(ir + r) * n + jc + jr + cc];
                        }
                    }
                    kernel_packed_4x8_u32(
                        a_panel.as_ptr(),
                        b_panel.as_ptr(),
                        tmp.as_mut_ptr(),
                        kc,
                        NR32,
                    );
                    for r in 0..mr {
                        for cc in 0..nr {
                            c_stripe[(ir + r) * n + jc + jr + cc] = tmp[r * NR32 + cc];
                        }
                    }
                }
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn kernel_packed_4x8_u32(
    a_panel: *const u32,
    b_panel: *const u32,
    c: *mut u32,
    kc: usize,
    c_stride: usize,
) {
    let mut c00 = vld1q_u32(c);
    let mut c01 = vld1q_u32(c.add(4));
    let mut c10 = vld1q_u32(c.add(c_stride));
    let mut c11 = vld1q_u32(c.add(c_stride + 4));
    let mut c20 = vld1q_u32(c.add(2 * c_stride));
    let mut c21 = vld1q_u32(c.add(2 * c_stride + 4));
    let mut c30 = vld1q_u32(c.add(3 * c_stride));
    let mut c31 = vld1q_u32(c.add(3 * c_stride + 4));

    for p in 0..kc {
        let b0 = vld1q_u32(b_panel.add(p * NR32));
        let b1 = vld1q_u32(b_panel.add(p * NR32 + 4));

        let a0 = vdupq_n_u32(*a_panel.add(p * MR32));
        let a1 = vdupq_n_u32(*a_panel.add(p * MR32 + 1));
        let a2 = vdupq_n_u32(*a_panel.add(p * MR32 + 2));
        let a3 = vdupq_n_u32(*a_panel.add(p * MR32 + 3));

        c00 = vmlaq_u32(c00, b0, a0);
        c01 = vmlaq_u32(c01, b1, a0);
        c10 = vmlaq_u32(c10, b0, a1);
        c11 = vmlaq_u32(c11, b1, a1);
        c20 = vmlaq_u32(c20, b0, a2);
        c21 = vmlaq_u32(c21, b1, a2);
        c30 = vmlaq_u32(c30, b0, a3);
        c31 = vmlaq_u32(c31, b1, a3);
    }

    vst1q_u32(c, c00);
    vst1q_u32(c.add(4), c01);
    vst1q_u32(c.add(c_stride), c10);
    vst1q_u32(c.add(c_stride + 4), c11);
    vst1q_u32(c.add(2 * c_stride), c20);
    vst1q_u32(c.add(2 * c_stride + 4), c21);
    vst1q_u32(c.add(3 * c_stride), c30);
    vst1q_u32(c.add(3 * c_stride + 4), c31);
}

// ============================================================================
// u8 path
// ============================================================================

const MR8: usize = 4;
const NR8: usize = 32;
const MC8: usize = 128;
const KC8: usize = 256;
const NC8: usize = 1024;

/// NEON dense matmul over Z/2^8 — `C = A · B`. Returns a freshly
/// allocated `C`. Convenience wrapper over [`matmul_u8_view`].
pub fn matmul_u8(a: &Mat<u8>, b: &Mat<u8>) -> Mat<u8> {
    matmul_u8_view(a.view(), b.view())
}

/// NEON dense matmul over Z/2^8 from strided views. Returns a freshly
/// allocated `C`.
pub fn matmul_u8_view(a: MatView<u8>, b: MatView<u8>) -> Mat<u8> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    let (m, _, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<u8>::zeros(m, n);
    addmul_u8_into(&mut c, a, b);
    c
}

/// NEON fused multiply-add over Z/2^8: `C += A · B` in-place.
/// Convenience wrapper over [`addmul_u8_into`].
pub fn addmul_u8(c: &mut Mat<u8>, a: &Mat<u8>, b: &Mat<u8>) {
    addmul_u8_into(c, a.view(), b.view());
}

/// NEON fused multiply-add over Z/2^8 from strided views: `C += A · B`
/// in-place. The kernel all other u8 entry points funnel into.
pub fn addmul_u8_into(c: &mut Mat<u8>, a: MatView<u8>, b: MatView<u8>) {
    assert_eq!(a.cols, b.rows, "addmul_u8: inner dimension mismatch");
    assert_eq!(
        (c.rows, c.cols),
        (a.rows, b.cols),
        "addmul_u8: C dims must match A·B"
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);

    if m == 0 || k == 0 || n == 0 {
        return;
    }
    if m * n * k < 32_768 || m < MR8 || n < NR8 {
        for i in 0..m {
            for p in 0..k {
                let aip = a.get(i, p);
                for j in 0..n {
                    let bij = b.get(p, j);
                    let cij = c.data[i * n + j];
                    c.data[i * n + j] = cij.wrapping_add(aip.wrapping_mul(bij));
                }
            }
        }
        return;
    }

    for jc in (0..n).step_by(NC8) {
        let nc = (jc + NC8).min(n) - jc;
        let nc_pad = round_up(nc, NR8);

        for pc in (0..k).step_by(KC8) {
            let kc = (pc + KC8).min(k) - pc;
            let b_block = b.sub_view(pc, jc, kc, nc);
            let mut b_tilde = vec![0u8; kc * nc_pad];
            pack_b_u8(b_block, &mut b_tilde);

            c.data
                .par_chunks_mut(MC8 * n)
                .enumerate()
                .for_each(|(ib, c_stripe)| {
                    let ic = ib * MC8;
                    let mc = c_stripe.len() / n;
                    let mc_pad = round_up(mc, MR8);
                    let a_block = a.sub_view(ic, pc, mc, kc);
                    let mut a_tilde = vec![0u8; mc_pad * kc];
                    pack_a_u8(a_block, &mut a_tilde);
                    run_kernels_u8(
                        &a_tilde, &b_tilde, c_stripe, mc, mc_pad, nc, nc_pad, kc, n, jc,
                    );
                });
        }
    }
}

fn pack_a_u8(block: MatView<u8>, a_tilde: &mut [u8]) {
    let (mc, kc) = (block.rows, block.cols);
    let num_panels = round_up(mc, MR8) / MR8;
    for panel in 0..num_panels {
        let panel_off = panel * MR8 * kc;
        let base_row = panel * MR8;
        for p in 0..kc {
            for r in 0..MR8 {
                let src_row = base_row + r;
                a_tilde[panel_off + p * MR8 + r] = if src_row < mc {
                    block.get(src_row, p)
                } else {
                    0
                };
            }
        }
    }
}

fn pack_b_u8(block: MatView<u8>, b_tilde: &mut [u8]) {
    let (kc, nc) = (block.rows, block.cols);
    let num_panels = round_up(nc, NR8) / NR8;
    for panel in 0..num_panels {
        let panel_off = panel * NR8 * kc;
        let base_col = panel * NR8;
        for p in 0..kc {
            for cc in 0..NR8 {
                let src_col = base_col + cc;
                b_tilde[panel_off + p * NR8 + cc] = if src_col < nc {
                    block.get(p, src_col)
                } else {
                    0
                };
            }
        }
    }
}

fn run_kernels_u8(
    a_tilde: &[u8],
    b_tilde: &[u8],
    c_stripe: &mut [u8],
    mc: usize,
    mc_pad: usize,
    nc: usize,
    nc_pad: usize,
    kc: usize,
    n: usize,
    jc: usize,
) {
    for jr in (0..nc_pad).step_by(NR8) {
        let b_panel = &b_tilde[(jr / NR8) * NR8 * kc..];
        let nr = (jr + NR8).min(nc) - jr;

        for ir in (0..mc_pad).step_by(MR8) {
            let a_panel = &a_tilde[(ir / MR8) * MR8 * kc..];
            let mr = (ir + MR8).min(mc) - ir;

            unsafe {
                if mr == MR8 && nr == NR8 {
                    let c_ptr = c_stripe.as_mut_ptr().add(ir * n + jc + jr);
                    kernel_packed_4x32_u8(a_panel.as_ptr(), b_panel.as_ptr(), c_ptr, kc, n);
                } else {
                    let mut tmp = [0u8; MR8 * NR8];
                    for r in 0..mr {
                        for cc in 0..nr {
                            tmp[r * NR8 + cc] = c_stripe[(ir + r) * n + jc + jr + cc];
                        }
                    }
                    kernel_packed_4x32_u8(
                        a_panel.as_ptr(),
                        b_panel.as_ptr(),
                        tmp.as_mut_ptr(),
                        kc,
                        NR8,
                    );
                    for r in 0..mr {
                        for cc in 0..nr {
                            c_stripe[(ir + r) * n + jc + jr + cc] = tmp[r * NR8 + cc];
                        }
                    }
                }
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn kernel_packed_4x32_u8(
    a_panel: *const u8,
    b_panel: *const u8,
    c: *mut u8,
    kc: usize,
    c_stride: usize,
) {
    let mut c00 = vld1q_u8(c);
    let mut c01 = vld1q_u8(c.add(16));
    let mut c10 = vld1q_u8(c.add(c_stride));
    let mut c11 = vld1q_u8(c.add(c_stride + 16));
    let mut c20 = vld1q_u8(c.add(2 * c_stride));
    let mut c21 = vld1q_u8(c.add(2 * c_stride + 16));
    let mut c30 = vld1q_u8(c.add(3 * c_stride));
    let mut c31 = vld1q_u8(c.add(3 * c_stride + 16));

    for p in 0..kc {
        let b0 = vld1q_u8(b_panel.add(p * NR8));
        let b1 = vld1q_u8(b_panel.add(p * NR8 + 16));

        let a0 = vdupq_n_u8(*a_panel.add(p * MR8));
        let a1 = vdupq_n_u8(*a_panel.add(p * MR8 + 1));
        let a2 = vdupq_n_u8(*a_panel.add(p * MR8 + 2));
        let a3 = vdupq_n_u8(*a_panel.add(p * MR8 + 3));

        c00 = vmlaq_u8(c00, b0, a0);
        c01 = vmlaq_u8(c01, b1, a0);
        c10 = vmlaq_u8(c10, b0, a1);
        c11 = vmlaq_u8(c11, b1, a1);
        c20 = vmlaq_u8(c20, b0, a2);
        c21 = vmlaq_u8(c21, b1, a2);
        c30 = vmlaq_u8(c30, b0, a3);
        c31 = vmlaq_u8(c31, b1, a3);
    }

    vst1q_u8(c, c00);
    vst1q_u8(c.add(16), c01);
    vst1q_u8(c.add(c_stride), c10);
    vst1q_u8(c.add(c_stride + 16), c11);
    vst1q_u8(c.add(2 * c_stride), c20);
    vst1q_u8(c.add(2 * c_stride + 16), c21);
    vst1q_u8(c.add(3 * c_stride), c30);
    vst1q_u8(c.add(3 * c_stride + 16), c31);
}

// ============================================================================
// u64 path — delegates to scalar.
// ============================================================================

/// AArch64 base NEON has no `vmulq_u64`; a "NEON" u64 kernel would
/// serialize the multiplies back to scalar per lane and win nothing
/// over the portable loop. This entry point preserves API symmetry
/// and routes to the scalar reference. A real SIMD path would require
/// SVE or a decomposed 32×32→64 partial-product schedule.
pub fn matmul_u64(a: &Mat<u64>, b: &Mat<u64>) -> Mat<u64> {
    crate::scalar::matmul(a, b)
}
