// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! AVX2 matmul for u16 on x86_64.
//!
//! Mirror of `neon.rs` — the BLIS pack + block + rayon outer structure
//! and register-blocked microkernel are identical; only the intrinsics
//! change. YMM registers hold **16 u16 lanes** (vs NEON's 8), so the
//! natural tile widens from 4×16 to **4×32**:
//!
//!   - 4 rows × 32 cols of C = 8 YMM registers (2 per row)
//!   - 2 YMM for the B panel slab per `kc` step
//!   - 4 YMM for broadcast-A (one per row)
//!   - 14 live registers, well under AVX2's 16 YMM
//!
//! AVX2 has no fused u16 multiply-add; we use `_mm256_mullo_epi16` +
//! `_mm256_add_epi16` as the FMA-equivalent pair.
//!
//! The module is present on every x86-64 build. Its safe entry points detect
//! AVX2 at runtime and fall back to the scalar kernel before executing any
//! target-feature code. `tests/equivalence.rs` compares the dispatched result
//! with the scalar reference.

#![cfg(target_arch = "x86_64")]

use crate::mat::{Mat, MatView};
use core::arch::x86_64::*;
use rayon::prelude::*;

// Register-level tile.
const MR: usize = 4;
const NR: usize = 32;

// Cache-level blocks (same shape as NEON; sizing in bytes matches).
const MC: usize = 128;
const KC: usize = 256;
const NC: usize = 512;

/// AVX2 dense matmul over Z/2^16 — `C = A · B`. Returns a freshly
/// allocated `C`. Convenience wrapper over [`matmul_u16_view`].
pub fn matmul_u16(a: &Mat<u16>, b: &Mat<u16>) -> Mat<u16> {
    matmul_u16_view(a.view(), b.view())
}

/// AVX2 dense matmul over Z/2^16 from strided views. Returns a freshly
/// allocated `C`. Strassen-on-AVX2 invokes this at the recursion leaf.
pub fn matmul_u16_view(a: MatView<u16>, b: MatView<u16>) -> Mat<u16> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    let (m, _, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<u16>::zeros(m, n);
    addmul_u16_into(&mut c, a, b);
    c
}

/// AVX2 fused multiply-add over Z/2^16: `C += A · B` in-place.
/// Convenience wrapper over [`addmul_u16_into`].
pub fn addmul_u16(c: &mut Mat<u16>, a: &Mat<u16>, b: &Mat<u16>) {
    addmul_u16_into(c, a.view(), b.view());
}

/// AVX2 fused multiply-add over Z/2^16 from strided views:
/// `C += A · B` in-place. Caller-managed destination — the BLIS
/// micro-kernel that all the other entry points dispatch to. Panics
/// on dimension mismatch in debug + release.
pub fn addmul_u16_into(c: &mut Mat<u16>, a: MatView<u16>, b: MatView<u16>) {
    assert_eq!(a.cols, b.rows, "addmul: inner dimension mismatch");
    assert_eq!(
        (c.rows, c.cols),
        (a.rows, b.cols),
        "addmul: C dims must match A·B"
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);

    // Empty-dim shortcut — return the (already-zero) C of the correct shape.
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    if !std::arch::is_x86_feature_detected!("avx2") {
        addmul_u16_into_scalar(c, a, b);
        return;
    }

    // Tiny-problem scalar fallback: the packed path's overhead dominates
    // the microkernel gain for problems below ~one cache block of work,
    // or when any dim is below a single register tile (m < MR or n < NR).
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

fn addmul_u16_into_scalar(c: &mut Mat<u16>, a: MatView<u16>, b: MatView<u16>) {
    for i in 0..a.rows {
        for p in 0..a.cols {
            let aip = a.get(i, p);
            for j in 0..b.cols {
                let bij = b.get(p, j);
                let cij = c.data[i * b.cols + j];
                c.data[i * b.cols + j] = cij.wrapping_add(aip.wrapping_mul(bij));
            }
        }
    }
}

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

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
                    let c_ptr = c_stripe.as_mut_ptr().add(ir * n + jc + jr);
                    kernel_packed_4x32(a_panel.as_ptr(), b_panel.as_ptr(), c_ptr, kc, n);
                } else {
                    let mut tmp = [0u16; MR * NR];
                    for r in 0..mr {
                        for cc in 0..nr {
                            tmp[r * NR + cc] = c_stripe[(ir + r) * n + jc + jr + cc];
                        }
                    }
                    kernel_packed_4x32(
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

// The C-tile row offsets are written uniformly as `0/1/2/3 * c_stride`
// so the four rows of the register tile read as one block; clippy objects
// to the degenerate `0 *` and `1 *` factors, which is precisely the
// symmetry that makes a hand-tiled kernel legible. Keep the shape.
#[allow(clippy::erasing_op, clippy::identity_op)]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn kernel_packed_4x32(
    a_panel: *const u16,
    b_panel: *const u16,
    c: *mut u16,
    kc: usize,
    c_stride: usize,
) {
    // Load 4×32 C tile into 8 YMM registers.
    let mut c00 = _mm256_loadu_si256(c.add(0 * c_stride) as *const __m256i);
    let mut c01 = _mm256_loadu_si256(c.add(0 * c_stride + 16) as *const __m256i);
    let mut c10 = _mm256_loadu_si256(c.add(1 * c_stride) as *const __m256i);
    let mut c11 = _mm256_loadu_si256(c.add(1 * c_stride + 16) as *const __m256i);
    let mut c20 = _mm256_loadu_si256(c.add(2 * c_stride) as *const __m256i);
    let mut c21 = _mm256_loadu_si256(c.add(2 * c_stride + 16) as *const __m256i);
    let mut c30 = _mm256_loadu_si256(c.add(3 * c_stride) as *const __m256i);
    let mut c31 = _mm256_loadu_si256(c.add(3 * c_stride + 16) as *const __m256i);

    for p in 0..kc {
        let b0 = _mm256_loadu_si256(b_panel.add(p * NR) as *const __m256i);
        let b1 = _mm256_loadu_si256(b_panel.add(p * NR + 16) as *const __m256i);

        // Broadcast A scalars. We cast u16→i16 for the intrinsic; the
        // result is bitwise identical (the lane just carries the bit
        // pattern; `_mm256_mullo_epi16` gives low 16 bits of the i16
        // product = low 16 bits of the u16 product = Z/2^16 product).
        let a0 = _mm256_set1_epi16(*a_panel.add(p * MR) as i16);
        let a1 = _mm256_set1_epi16(*a_panel.add(p * MR + 1) as i16);
        let a2 = _mm256_set1_epi16(*a_panel.add(p * MR + 2) as i16);
        let a3 = _mm256_set1_epi16(*a_panel.add(p * MR + 3) as i16);

        // AVX2 has no fused u16 mul-add — emit mullo + add.
        c00 = _mm256_add_epi16(c00, _mm256_mullo_epi16(b0, a0));
        c01 = _mm256_add_epi16(c01, _mm256_mullo_epi16(b1, a0));
        c10 = _mm256_add_epi16(c10, _mm256_mullo_epi16(b0, a1));
        c11 = _mm256_add_epi16(c11, _mm256_mullo_epi16(b1, a1));
        c20 = _mm256_add_epi16(c20, _mm256_mullo_epi16(b0, a2));
        c21 = _mm256_add_epi16(c21, _mm256_mullo_epi16(b1, a2));
        c30 = _mm256_add_epi16(c30, _mm256_mullo_epi16(b0, a3));
        c31 = _mm256_add_epi16(c31, _mm256_mullo_epi16(b1, a3));
    }

    _mm256_storeu_si256(c.add(0 * c_stride) as *mut __m256i, c00);
    _mm256_storeu_si256(c.add(0 * c_stride + 16) as *mut __m256i, c01);
    _mm256_storeu_si256(c.add(1 * c_stride) as *mut __m256i, c10);
    _mm256_storeu_si256(c.add(1 * c_stride + 16) as *mut __m256i, c11);
    _mm256_storeu_si256(c.add(2 * c_stride) as *mut __m256i, c20);
    _mm256_storeu_si256(c.add(2 * c_stride + 16) as *mut __m256i, c21);
    _mm256_storeu_si256(c.add(3 * c_stride) as *mut __m256i, c30);
    _mm256_storeu_si256(c.add(3 * c_stride + 16) as *mut __m256i, c31);
}
