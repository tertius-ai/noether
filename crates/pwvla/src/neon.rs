// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! NEON matmul for u16 on aarch64, BLIS-style: pack + block + parallel.
//!
//! Structure (outer to inner):
//!
//!   jc  (NC cols of B)            — serial
//!     pc  (KC rows of A / B)      — serial
//!       pack B[pc..pc+KC, jc..jc+NC] into contiguous B~
//!       par_chunks_mut over rows of C in stripes of MC rows:
//!         pack A[ic..ic+MC, pc..pc+KC] into contiguous A~
//!         jr  (NR cols) × ir (MR rows) → 4×16 microkernel on packed panels
//!
//! The microkernel keeps a 4×16 C tile in 8 NEON registers across the
//! kc-loop. Packing reorders A and B so each inner iteration sees
//! sequential memory. The same skeleton transplants to other rings:
//! swap the microkernel + packing dtype; everything else is ring-agnostic.

use crate::mat::{Mat, MatView};
use core::arch::aarch64::*;
use rayon::prelude::*;

// Register-level tile (must match the microkernel body).
const MR: usize = 4;
const NR: usize = 16;

// Cache-level blocks. Tuned for Apple-Silicon-ish:
//   A~ per stripe ≈ MC*KC*2 B =  64 KB  (L1)
//   B~ per block  ≈ KC*NC*2 B = 256 KB  (L2)
const MC: usize = 128;
const KC: usize = 256;
const NC: usize = 512;

/// NEON dense matmul over Z/2^16 — `C = A · B`. Returns a freshly
/// allocated `C`. Convenience wrapper over [`matmul_u16_view`].
pub fn matmul_u16(a: &Mat<u16>, b: &Mat<u16>) -> Mat<u16> {
    matmul_u16_view(a.view(), b.view())
}

/// Upper-triangular matrix multiply: T·B with T square upper-triangular.
///
/// Uses the recurrence
/// ```text
/// [T11 T12] [B1]   [T11·B1 + T12·B2]
/// [ 0  T22] [B2] = [    T22·B2      ]
/// ```
/// so each recursion level replaces one N-sized triangular multiply with
/// two (N/2)-sized triangular multiplies plus one (N/2)-sized matmul
/// (via `addmul_u16_into`, which uses the fast packed kernel). The
/// asymptotic work is ~1/3 that of a full matmul; at small n the base
/// case runs the unblocked scalar loop.
pub fn trmm_upper_u16(t: &Mat<u16>, b: &Mat<u16>) -> Mat<u16> {
    assert_eq!(t.rows, t.cols, "trmm: T must be square");
    assert_eq!(t.cols, b.rows, "trmm: inner dim mismatch");
    let mut c = Mat::<u16>::zeros(t.rows, b.cols);
    trmm_upper_into(t.view(), b.view(), &mut c);
    c
}

/// Lower-triangular variant: T square lower-triangular.
/// ```text
/// [T11  0 ] [B1]   [    T11·B1      ]
/// [T21 T22] [B2] = [T21·B1 + T22·B2 ]
/// ```
pub fn trmm_lower_u16(t: &Mat<u16>, b: &Mat<u16>) -> Mat<u16> {
    assert_eq!(t.rows, t.cols, "trmm: T must be square");
    assert_eq!(t.cols, b.rows, "trmm: inner dim mismatch");
    let mut c = Mat::<u16>::zeros(t.rows, b.cols);
    trmm_lower_into(t.view(), b.view(), &mut c);
    c
}

const TRMM_CROSSOVER: usize = 128;

fn trmm_upper_into(t: MatView<u16>, b: MatView<u16>, c: &mut Mat<u16>) {
    let n = t.rows;
    let m = c.cols;

    if n <= TRMM_CROSSOVER {
        // Assumes c was freshly zeroed by the caller.
        for i in 0..n {
            for k in i..n {
                let t_ik = t.get(i, k);
                if t_ik == 0 {
                    continue;
                }
                for j in 0..m {
                    let v = t_ik.wrapping_mul(b.get(k, j));
                    let idx = i * m + j;
                    c.data[idx] = c.data[idx].wrapping_add(v);
                }
            }
        }
        return;
    }

    let h = n / 2;
    let t11 = t.sub_view(0, 0, h, h);
    let t12 = t.sub_view(0, h, h, n - h);
    let t22 = t.sub_view(h, h, n - h, n - h);
    let b1 = b.sub_view(0, 0, h, m);
    let b2 = b.sub_view(h, 0, n - h, m);

    // Top block: T11·B1 + T12·B2
    let mut c_top = Mat::<u16>::zeros(h, m);
    trmm_upper_into(t11, b1, &mut c_top);
    addmul_u16_into(&mut c_top, t12, b2);

    // Bottom block: T22·B2
    let mut c_bot = Mat::<u16>::zeros(n - h, m);
    trmm_upper_into(t22, b2, &mut c_bot);

    // Copy into c (c is still zeros on entry at this level).
    for i in 0..h {
        for j in 0..m {
            c.data[i * m + j] = c_top.data[i * m + j];
        }
    }
    for i in 0..n - h {
        for j in 0..m {
            c.data[(h + i) * m + j] = c_bot.data[i * m + j];
        }
    }
}

fn trmm_lower_into(t: MatView<u16>, b: MatView<u16>, c: &mut Mat<u16>) {
    let n = t.rows;
    let m = c.cols;

    if n <= TRMM_CROSSOVER {
        for i in 0..n {
            for k in 0..=i {
                let t_ik = t.get(i, k);
                if t_ik == 0 {
                    continue;
                }
                for j in 0..m {
                    let v = t_ik.wrapping_mul(b.get(k, j));
                    let idx = i * m + j;
                    c.data[idx] = c.data[idx].wrapping_add(v);
                }
            }
        }
        return;
    }

    let h = n / 2;
    let t11 = t.sub_view(0, 0, h, h);
    let t21 = t.sub_view(h, 0, n - h, h);
    let t22 = t.sub_view(h, h, n - h, n - h);
    let b1 = b.sub_view(0, 0, h, m);
    let b2 = b.sub_view(h, 0, n - h, m);

    // Top block: T11·B1
    let mut c_top = Mat::<u16>::zeros(h, m);
    trmm_lower_into(t11, b1, &mut c_top);

    // Bottom block: T21·B1 + T22·B2
    let mut c_bot = Mat::<u16>::zeros(n - h, m);
    trmm_lower_into(t22, b2, &mut c_bot);
    addmul_u16_into(&mut c_bot, t21, b1);

    for i in 0..h {
        for j in 0..m {
            c.data[i * m + j] = c_top.data[i * m + j];
        }
    }
    for i in 0..n - h {
        for j in 0..m {
            c.data[(h + i) * m + j] = c_bot.data[i * m + j];
        }
    }
}

/// View-based entry. Strassen recursion calls this on strided sub-views
/// of its parent matrices — no need to materialize quadrants.
pub fn matmul_u16_view(a: MatView<u16>, b: MatView<u16>) -> Mat<u16> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    let (m, _, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<u16>::zeros(m, n);
    addmul_u16_into(&mut c, a, b);
    c
}

/// Fused `C += A · B` over Z/2^16 (wrapping).
///
/// The microkernel already loads the current C tile, accumulates A·B
/// into it, and stores — so calling this with a zero C gives matmul,
/// and calling with a non-zero C gives the fused-multiply-add that's
/// useful for LU trailing updates, iterative refinement, and Strassen
/// combination phases that want to avoid a temporary.
pub fn addmul_u16(c: &mut Mat<u16>, a: &Mat<u16>, b: &Mat<u16>) {
    addmul_u16_into(c, a.view(), b.view());
}

/// View-aware variant of `addmul_u16`.
pub fn addmul_u16_into(c: &mut Mat<u16>, a: MatView<u16>, b: MatView<u16>) {
    assert_eq!(a.cols, b.rows, "addmul: inner dimension mismatch");
    assert_eq!(
        (c.rows, c.cols),
        (a.rows, b.cols),
        "addmul: C dims must match A·B"
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);

    // Empty-dim: nothing to do. (C is already the correct empty shape.)
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Below-tile / tiny-problem scalar fallback. Packing + rayon overhead
    // dominates the microkernel gain when the problem is smaller than
    // roughly one cache-block worth of work; fall through to a simple
    // wrapping addmul. The threshold is chosen so real problems still
    // take the packed path: e.g., 64×64×64 = 262k ≥ 32k triggers packed.
    if m * n * k < 32_768 || m < MR || n < NR {
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

    for jc in (0..n).step_by(NC) {
        let nc = (jc + NC).min(n) - jc;
        let nc_pad = round_up(nc, NR);

        for pc in (0..k).step_by(KC) {
            let kc = (pc + KC).min(k) - pc;

            let b_block = b.sub_view(pc, jc, kc, nc);
            let mut b_tilde = vec![0u16; kc * nc_pad];
            pack_b(b_block, &mut b_tilde);

            c.data
                .par_chunks_mut(MC * n)
                .enumerate()
                .for_each(|(ib, c_stripe)| {
                    let ic = ib * MC;
                    let mc = c_stripe.len() / n;
                    let mc_pad = round_up(mc, MR);

                    let a_block = a.sub_view(ic, pc, mc, kc);
                    let mut a_tilde = vec![0u16; mc_pad * kc];
                    pack_a(a_block, &mut a_tilde);

                    run_kernels(
                        &a_tilde, &b_tilde, c_stripe, mc, mc_pad, nc, nc_pad, kc, n, jc,
                    );
                });
        }
    }
}

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

// Pack an mc×kc block of A as a stack of MR-row horizontal panels,
// padded to mc_pad rows. Inside a panel: for p in 0..kc, r in 0..MR,
// element (r, p) stored at offset p*MR + r.
fn pack_a(block: MatView<u16>, a_tilde: &mut [u16]) {
    let (mc, kc) = (block.rows, block.cols);
    let num_panels = round_up(mc, MR) / MR;
    for panel in 0..num_panels {
        let panel_off = panel * MR * kc;
        let base_row = panel * MR;
        for p in 0..kc {
            for r in 0..MR {
                let src_row = base_row + r;
                a_tilde[panel_off + p * MR + r] = if src_row < mc {
                    block.get(src_row, p)
                } else {
                    0
                };
            }
        }
    }
}

// Pack a kc×nc block of B as a row of NR-col vertical panels, padded
// to nc_pad cols. Inside a panel: for p in 0..kc, c in 0..NR, element
// (p, c) stored at offset p*NR + c.
fn pack_b(block: MatView<u16>, b_tilde: &mut [u16]) {
    let (kc, nc) = (block.rows, block.cols);
    let num_panels = round_up(nc, NR) / NR;
    for panel in 0..num_panels {
        let panel_off = panel * NR * kc;
        let base_col = panel * NR;
        for p in 0..kc {
            for cc in 0..NR {
                let src_col = base_col + cc;
                b_tilde[panel_off + p * NR + cc] = if src_col < nc {
                    block.get(p, src_col)
                } else {
                    0
                };
            }
        }
    }
}

fn run_kernels(
    a_tilde: &[u16],
    b_tilde: &[u16],
    c_stripe: &mut [u16],
    mc: usize,
    mc_pad: usize,
    nc: usize,
    nc_pad: usize,
    kc: usize,
    n: usize,
    jc: usize,
) {
    for jr in (0..nc_pad).step_by(NR) {
        let b_panel = &b_tilde[(jr / NR) * NR * kc..];
        let nr = (jr + NR).min(nc) - jr;

        for ir in (0..mc_pad).step_by(MR) {
            let a_panel = &a_tilde[(ir / MR) * MR * kc..];
            let mr = (ir + MR).min(mc) - ir;

            unsafe {
                if mr == MR && nr == NR {
                    // Fast path: write directly into C.
                    let c_ptr = c_stripe.as_mut_ptr().add(ir * n + jc + jr);
                    kernel_packed_4x16(a_panel.as_ptr(), b_panel.as_ptr(), c_ptr, kc, n);
                } else {
                    // Edge: kernel into a dense 4×16 scratch, then blit
                    // the valid mr×nr sub-tile back into C.
                    let mut tmp = [0u16; MR * NR];
                    for r in 0..mr {
                        for cc in 0..nr {
                            tmp[r * NR + cc] = c_stripe[(ir + r) * n + jc + jr + cc];
                        }
                    }
                    kernel_packed_4x16(
                        a_panel.as_ptr(),
                        b_panel.as_ptr(),
                        tmp.as_mut_ptr(),
                        kc,
                        NR,
                    );
                    for r in 0..mr {
                        for cc in 0..nr {
                            c_stripe[(ir + r) * n + jc + jr + cc] = tmp[r * NR + cc];
                        }
                    }
                }
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn kernel_packed_4x16(
    a_panel: *const u16,
    b_panel: *const u16,
    c: *mut u16,
    kc: usize,
    c_stride: usize,
) {
    // 8 NEON registers hold the 4×16 C tile across the kc-loop.
    let mut c00 = vld1q_u16(c);
    let mut c01 = vld1q_u16(c.add(8));
    let mut c10 = vld1q_u16(c.add(c_stride));
    let mut c11 = vld1q_u16(c.add(c_stride + 8));
    let mut c20 = vld1q_u16(c.add(2 * c_stride));
    let mut c21 = vld1q_u16(c.add(2 * c_stride + 8));
    let mut c30 = vld1q_u16(c.add(3 * c_stride));
    let mut c31 = vld1q_u16(c.add(3 * c_stride + 8));

    for p in 0..kc {
        // Sequential reads thanks to packing: stride 16 in B~, 4 in A~.
        let b0 = vld1q_u16(b_panel.add(p * NR));
        let b1 = vld1q_u16(b_panel.add(p * NR + 8));

        let a0 = vdupq_n_u16(*a_panel.add(p * MR));
        let a1 = vdupq_n_u16(*a_panel.add(p * MR + 1));
        let a2 = vdupq_n_u16(*a_panel.add(p * MR + 2));
        let a3 = vdupq_n_u16(*a_panel.add(p * MR + 3));

        c00 = vmlaq_u16(c00, b0, a0);
        c01 = vmlaq_u16(c01, b1, a0);
        c10 = vmlaq_u16(c10, b0, a1);
        c11 = vmlaq_u16(c11, b1, a1);
        c20 = vmlaq_u16(c20, b0, a2);
        c21 = vmlaq_u16(c21, b1, a2);
        c30 = vmlaq_u16(c30, b0, a3);
        c31 = vmlaq_u16(c31, b1, a3);
    }

    vst1q_u16(c, c00);
    vst1q_u16(c.add(8), c01);
    vst1q_u16(c.add(c_stride), c10);
    vst1q_u16(c.add(c_stride + 8), c11);
    vst1q_u16(c.add(2 * c_stride), c20);
    vst1q_u16(c.add(2 * c_stride + 8), c21);
    vst1q_u16(c.add(3 * c_stride), c30);
    vst1q_u16(c.add(3 * c_stride + 8), c31);
}
