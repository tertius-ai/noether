// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bit-sliced GF(2^e) matrix multiplication — the m4rste → m4rst
//! reduction.
//!
//! This is the structural heart of the crate: a GF(2^e) matmul is
//! turned into a handful of **F_2** matmuls, each run by `m4rst`, followed by
//! a bilinear product over the bit planes.
//!
//! # The reduction
//!
//! Write a GF(2^e) matrix as a polynomial in `x` whose coefficients
//! are F_2 matrices (the *bit-planes*):
//!
//! ```text
//! A = A_0 + A_1·x + … + A_{e-1}·x^{e-1}
//! ```
//!
//! Then `A·B = Σ_m (Σ_{k+l=m} A_k·B_l) · x^m` for `m ∈ [0, 2e-2]`,
//! where each `A_k·B_l` is an F_2 matmul. Finally the degree-`≥ e`
//! coefficients are folded back down modulo the field's minimal
//! polynomial — a handful of plane XORs, since `x^e ≡ Σ_{j<e} c_j x^j`
//! with `c_j` the low bits of the minpoly.
//!
//! The bilinear product is **recursive Karatsuba** over the bit-planes
//! — `3^⌈log2 e⌉` leaf F_2 matmuls (27 for `e = 8`, vs 64 schoolbook),
//! each run by `m4rst`. Bodrato's optimal bilinear formulas for small
//! fixed `e` can reduce the leaf count further, while Karatsuba supplies the
//! general implementation for every supported degree.

use m4rst::{MatrixWrite, Mzd};

use crate::gf2e::Gf2e;
use crate::mzed::Mzed;

/// Slice a bit-packed [`Mzed`] into its `e` F_2 bit-planes.
///
/// Returns a `Vec<m4rst::Mzd>` of length `e`; plane `k` is an
/// `rows × cols` F_2 matrix holding the `x^k` coefficient bit of every
/// element. The inverse is [`cling`].
///
/// SWAR (bit-twiddling) fast paths cover every **power-of-two** `e`:
/// 1 (memcpy), 2, 4, 8 (multiply-trick), 16 (mask+shift extracts).
/// Non-power-of-two `e` (3, 5, 6, 7, …) fall through to the generic
/// per-bit path because those strides do not tile a `u64` cleanly.
pub fn slice(m: &Mzed) -> Vec<Mzd> {
    let e = m.ctx.degree() as usize;
    let rows = m.rows();
    let cols = m.cols();
    let mut planes: Vec<Mzd> = (0..e).map(|_| Mzd::new(rows, cols)).collect();
    if cols == 0 || e == 0 {
        return planes;
    }
    match e {
        1 => slice_e1(m, &mut planes),
        2 => slice_e2(m, &mut planes),
        4 => slice_e4(m, &mut planes),
        8 => slice_e8(m, &mut planes),
        16 => slice_e16(m, &mut planes),
        _ => slice_generic(m, &mut planes, e),
    }
    planes
}

/// Per-bit generic slice: one `packed.get(...)` + one `plane.set(...)`
/// per (element, bit). The SWAR path handles `e = 8`.
fn slice_generic(m: &Mzed, planes: &mut [Mzd], e: usize) {
    let rows = m.rows();
    let cols = m.cols();
    for i in 0..rows {
        for j in 0..cols {
            for (k, plane) in planes.iter_mut().enumerate() {
                plane.set(i, j, m.packed.get(i, j * e + k));
            }
        }
    }
}

/// SWAR slice for `e = 8`: one packed-row `u64` holds 8 GF(256)
/// elements (one byte each); plane `k` of that chunk is just bit `k`
/// of each byte, collected into a contiguous 8-bit plane-row byte via
/// the classic `(mask * 0x0102_0408_1020_4080) >> 56` multiply trick.
fn slice_e8(m: &Mzed, planes: &mut [Mzd]) {
    let rows = m.rows();
    let cols = m.cols();
    // Each packed-row word holds 8 elements (= 8 plane bytes per word).
    // The corresponding 8 plane bits live at bit positions w*8 .. w*8+7
    // of the plane row — always within one plane-row word since w*8
    // mod 64 ∈ {0, 8, …, 56} (alignment never crosses a u64).
    let packed_width = m.packed.width();
    // Iterate over packed-row words, scattering 8 plane bytes per chunk.
    const MASK: u64 = 0x0101_0101_0101_0101;
    const COLLECT_MUL: u64 = 0x0102_0408_1020_4080;
    for i in 0..rows {
        let src = m.packed.row_words(i);
        // Last packed word may have fewer than 8 meaningful elements
        // (when `cols` is not a multiple of 8). Compute meaningful
        // element count per word index.
        for w in 0..packed_width {
            let word = src[w];
            let bit_pos = w * 8;
            let dst_word_idx = bit_pos / 64;
            let dst_bit_off = bit_pos % 64;
            // How many elements (bytes) are meaningful in this packed
            // word? cols - 8*w, clamped to 8.
            let elems_here = (cols - w * 8).min(8);
            // Mask off bytes beyond `elems_here` so spurious bits in
            // the high tail of the packed word don't enter the planes.
            let live_mask: u64 = if elems_here == 8 {
                u64::MAX
            } else {
                (1u64 << (elems_here * 8)) - 1
            };
            let live = word & live_mask;
            for (k, plane) in planes.iter_mut().enumerate() {
                let masked = (live >> k) & MASK;
                let plane_byte = (masked.wrapping_mul(COLLECT_MUL) >> 56) & 0xFF;
                if plane_byte != 0 {
                    plane.row_words_mut(i)[dst_word_idx] |= plane_byte << dst_bit_off;
                }
            }
        }
    }
}

/// Reassemble `e` F_2 bit-planes into a bit-packed [`Mzed`] — the
/// inverse of [`slice()`].
///
/// SWAR fast paths cover every power-of-two `e` (1, 2, 4, 8, 16);
/// other `e` fall through to a per-element pack.
///
/// # Panics
///
/// Panics if `planes.len() != ctx.degree()` or the planes disagree on
/// shape.
pub fn cling(ctx: Gf2e, planes: &[Mzd]) -> Mzed {
    let (rows, cols) = if planes.is_empty() {
        (0, 0)
    } else {
        (planes[0].rows(), planes[0].cols())
    };
    let mut out = Mzed::zeros(ctx, rows, cols);
    cling_into(&mut out, planes);
    out
}

/// `_into` counterpart to [`cling`] — writes the unsliced matrix
/// directly into `dst.packed` instead of allocating a fresh `Mzed`
/// and copying. Mirrors libm4rie's own convention: `mzed_cling(C, Cs)`
/// (`conversion.h`'s `_mzed_addmul_karatsuba`) writes its unsliced
/// result straight into a caller-supplied `C`, not into a temporary
/// that then gets copied.
///
/// Every per-`e` dispatch function below (`cling_e1` … `cling_e16`,
/// `cling_generic`) already does a **full overwrite** of each output
/// word/bit — none of them accumulate onto a starting value — so this
/// needs no upfront zeroing of `dst`, even when `dst` is a reused
/// buffer holding stale data from a prior call.
///
/// # Panics
///
/// Panics if `planes.len() != dst.ctx.degree()`, the planes disagree
/// on shape, or `dst`'s shape doesn't match the planes'.
pub fn cling_into(dst: &mut Mzed, planes: &[Mzd]) {
    let e = dst.ctx.degree() as usize;
    assert_eq!(
        planes.len(),
        e,
        "cling_into: expected {e} bit-planes for GF(2^{e}), got {}",
        planes.len()
    );
    if e == 0 {
        return;
    }
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    for (k, p) in planes.iter().enumerate() {
        assert_eq!(
            (p.rows(), p.cols()),
            (rows, cols),
            "cling_into: bit-plane {k} shape mismatch"
        );
    }
    assert_eq!(
        (dst.rows(), dst.cols()),
        (rows, cols),
        "cling_into: dst shape mismatch"
    );
    if cols == 0 {
        return;
    }
    match e {
        1 => cling_e1(planes, &mut dst.packed),
        2 => cling_e2(planes, &mut dst.packed),
        4 => cling_e4(planes, &mut dst.packed),
        8 => cling_e8(planes, &mut dst.packed),
        16 => cling_e16(planes, &mut dst.packed),
        _ => cling_generic(planes, &mut dst.packed, e),
    }
}

fn cling_generic(planes: &[Mzd], packed: &mut Mzd, e: usize) {
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    for i in 0..rows {
        for j in 0..cols {
            for (k, plane) in planes.iter().enumerate() {
                packed.set(i, j * e + k, plane.get(i, j));
            }
        }
    }
}

/// SWAR cling for `e = 8`: spread each plane's 8-bit row chunk to
/// stride-8 (one bit per byte) and OR shifted-left-by-`k` into the
/// packed-row word.
fn cling_e8(planes: &[Mzd], packed: &mut Mzd) {
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    let packed_width = packed.width();
    for i in 0..rows {
        // Collect the 8 plane row-word slices for row `i` up front.
        // (Borrow-checker note: we can't take immutable borrows of all
        // planes while holding a mutable borrow of `packed`, but they're
        // disjoint allocations, so reading row_words from each plane
        // sequentially is fine.)
        let dst = packed.row_words_mut(i);
        for w in 0..packed_width {
            let bit_pos = w * 8;
            let plane_word_idx = bit_pos / 64;
            let plane_bit_off = bit_pos % 64;
            let elems_here = (cols - w * 8).min(8);
            let live_mask_8: u64 = if elems_here == 8 {
                0xFF
            } else {
                (1u64 << elems_here) - 1
            };
            let mut packed_word: u64 = 0;
            for (k, plane) in planes.iter().enumerate() {
                let plane_byte =
                    ((plane.row_words(i)[plane_word_idx] >> plane_bit_off) & live_mask_8) as u8;
                if plane_byte != 0 {
                    let spread = spread_byte_to_stride8(plane_byte);
                    packed_word |= spread << k;
                }
            }
            dst[w] = packed_word;
        }
    }
}

/// 8 input bits → 64-bit value with input bit `r` placed at output
/// bit `8 * r`. Three-step SWAR deinterleave; no per-byte variable
/// shifts.
#[inline]
fn spread_byte_to_stride8(b: u8) -> u64 {
    let mut x = b as u64;
    x = (x | (x << 28)) & 0x0000_000F_0000_000Fu64;
    x = (x | (x << 14)) & 0x0003_0003_0003_0003u64;
    x = (x | (x << 7)) & 0x0101_0101_0101_0101u64;
    x
}

// ── SWAR fast paths for e = 1, 2, 4, 16 ─────────────────────────────

/// e=1: the packed matrix *is* the single bit-plane (no interleave).
fn slice_e1(m: &Mzed, planes: &mut [Mzd]) {
    planes[0]
        .as_words_mut()
        .copy_from_slice(m.packed.as_words());
}

fn cling_e1(planes: &[Mzd], packed: &mut Mzd) {
    packed.as_words_mut().copy_from_slice(planes[0].as_words());
}

/// Collect 32 stride-2 bits at positions {0, 2, …, 62} of `x` into the
/// low 32 bits of the result.
#[inline]
fn collect_stride2_to_low(x: u64) -> u64 {
    let mut x = x & 0x5555_5555_5555_5555u64;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333u64;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0Fu64;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FFu64;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFFu64;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFFu64;
    x
}

/// Inverse: spread 32 contiguous bits in low 32 to positions
/// {0, 2, …, 62}.
#[inline]
fn spread_low_to_stride2(x: u64) -> u64 {
    let mut x = x & 0x0000_0000_FFFF_FFFFu64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFFu64;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FFu64;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0Fu64;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333u64;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555u64;
    x
}

/// e=2 (GF(4)): each packed `u64` holds 32 elements at bit positions
/// `{0,1, 2,3, 4,5, …}`. Plane `k` is the stride-2 bits offset by `k`.
fn slice_e2(m: &Mzed, planes: &mut [Mzd]) {
    let rows = m.rows();
    let cols = m.cols();
    let packed_width = m.packed.width();
    for i in 0..rows {
        let src = m.packed.row_words(i);
        for w in 0..packed_width {
            let word = src[w];
            let bit_pos = w * 32;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 32).min(32);
            let live: u64 = if elems_here == 32 {
                0xFFFF_FFFF
            } else {
                (1u64 << elems_here) - 1
            };
            let p0 = collect_stride2_to_low(word) & live;
            let p1 = collect_stride2_to_low(word >> 1) & live;
            if p0 != 0 {
                planes[0].row_words_mut(i)[pw] |= p0 << bo;
            }
            if p1 != 0 {
                planes[1].row_words_mut(i)[pw] |= p1 << bo;
            }
        }
    }
}

fn cling_e2(planes: &[Mzd], packed: &mut Mzd) {
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    let packed_width = packed.width();
    for i in 0..rows {
        let dst = packed.row_words_mut(i);
        for w in 0..packed_width {
            let bit_pos = w * 32;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 32).min(32);
            let live: u64 = if elems_here == 32 {
                0xFFFF_FFFF
            } else {
                (1u64 << elems_here) - 1
            };
            let p0 = (planes[0].row_words(i)[pw] >> bo) & live;
            let p1 = (planes[1].row_words(i)[pw] >> bo) & live;
            dst[w] = spread_low_to_stride2(p0) | (spread_low_to_stride2(p1) << 1);
        }
    }
}

/// Collect 16 stride-4 bits at positions {0, 4, 8, …, 60} into low 16.
#[inline]
fn collect_stride4_to_low(x: u64) -> u64 {
    let mut x = x & 0x1111_1111_1111_1111u64;
    x = (x | (x >> 3)) & 0x0303_0303_0303_0303u64;
    x = (x | (x >> 6)) & 0x000F_000F_000F_000Fu64;
    x = (x | (x >> 12)) & 0x0000_00FF_0000_00FFu64;
    x = (x | (x >> 24)) & 0x0000_0000_0000_FFFFu64;
    x
}

/// Inverse: spread 16 contiguous bits in low 16 to positions
/// {0, 4, 8, …, 60}.
#[inline]
fn spread_low_to_stride4(x: u64) -> u64 {
    let mut x = x & 0x0000_0000_0000_FFFFu64;
    x = (x | (x << 24)) & 0x0000_00FF_0000_00FFu64;
    x = (x | (x << 12)) & 0x000F_000F_000F_000Fu64;
    x = (x | (x << 6)) & 0x0303_0303_0303_0303u64;
    x = (x | (x << 3)) & 0x1111_1111_1111_1111u64;
    x
}

/// e=4 (GF(16)): each packed `u64` holds 16 elements at stride 4.
fn slice_e4(m: &Mzed, planes: &mut [Mzd]) {
    let rows = m.rows();
    let cols = m.cols();
    let packed_width = m.packed.width();
    for i in 0..rows {
        let src = m.packed.row_words(i);
        for w in 0..packed_width {
            let word = src[w];
            let bit_pos = w * 16;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 16).min(16);
            let live: u64 = if elems_here == 16 {
                0xFFFF
            } else {
                (1u64 << elems_here) - 1
            };
            for (k, plane) in planes.iter_mut().enumerate() {
                let collected = collect_stride4_to_low(word >> k) & live;
                if collected != 0 {
                    plane.row_words_mut(i)[pw] |= collected << bo;
                }
            }
        }
    }
}

fn cling_e4(planes: &[Mzd], packed: &mut Mzd) {
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    let packed_width = packed.width();
    for i in 0..rows {
        let dst = packed.row_words_mut(i);
        for w in 0..packed_width {
            let bit_pos = w * 16;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 16).min(16);
            let live: u64 = if elems_here == 16 {
                0xFFFF
            } else {
                (1u64 << elems_here) - 1
            };
            let mut packed_word: u64 = 0;
            for (k, plane) in planes.iter().enumerate() {
                let pb = (plane.row_words(i)[pw] >> bo) & live;
                if pb != 0 {
                    packed_word |= spread_low_to_stride4(pb) << k;
                }
            }
            dst[w] = packed_word;
        }
    }
}

/// e=16 (GF(2^16)): each packed `u64` holds exactly 4 elements at
/// positions `{0..16, 16..32, 32..48, 48..64}`. Plane `k` is bit `k`
/// of each 16-bit block — no actual deinterleave, just gather/scatter.
fn slice_e16(m: &Mzed, planes: &mut [Mzd]) {
    let rows = m.rows();
    let cols = m.cols();
    let packed_width = m.packed.width();
    for i in 0..rows {
        let src = m.packed.row_words(i);
        for w in 0..packed_width {
            let word = src[w];
            let bit_pos = w * 4;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 4).min(4);
            for (k, plane) in planes.iter_mut().enumerate() {
                let mut bits: u64 = 0;
                for r in 0..elems_here {
                    bits |= ((word >> (16 * r + k)) & 1) << r;
                }
                if bits != 0 {
                    plane.row_words_mut(i)[pw] |= bits << bo;
                }
            }
        }
    }
}

fn cling_e16(planes: &[Mzd], packed: &mut Mzd) {
    let rows = planes[0].rows();
    let cols = planes[0].cols();
    let packed_width = packed.width();
    for i in 0..rows {
        let dst = packed.row_words_mut(i);
        for w in 0..packed_width {
            let bit_pos = w * 4;
            let pw = bit_pos / 64;
            let bo = bit_pos % 64;
            let elems_here = (cols - w * 4).min(4);
            let mask: u64 = (1u64 << elems_here) - 1;
            let mut packed_word: u64 = 0;
            for (k, plane) in planes.iter().enumerate() {
                let pb = (plane.row_words(i)[pw] >> bo) & mask;
                if pb != 0 {
                    let b0 = pb & 1;
                    let b1 = (pb >> 1) & 1;
                    let b2 = (pb >> 2) & 1;
                    let b3 = (pb >> 3) & 1;
                    packed_word |=
                        (b0 << k) | (b1 << (16 + k)) | (b2 << (32 + k)) | (b3 << (48 + k));
                }
            }
            dst[w] = packed_word;
        }
    }
}

/// XOR `src` into `dst` (same-shape F_2 matrices) over the packed
/// word arrays.
fn xor_into(dst: &mut Mzd, src: &Mzd) {
    debug_assert_eq!((dst.rows(), dst.cols()), (src.rows(), src.cols()));
    for (d, &s) in dst.as_words_mut().iter_mut().zip(src.as_words()) {
        *d ^= s;
    }
}

/// Polynomial product of two equal-length coefficient vectors of F_2
/// matrices, via recursive **Karatsuba** with a **Cenk-Hasan 6-mult
/// base case at `e = 3`**.
///
/// `a` and `b` have the same length `e` (the bit-planes of the two
/// operands); the result has length `2e - 1` — the raw, unreduced
/// product coefficients. Each `a[k]` is `rows × inner`, each `b[l]` is
/// `inner × n`, and every result coefficient is `rows × n`.
///
/// Leaf F_2 matmuls per `e`, with the e=3 base case in effect:
/// 1, 3, 6, 9, 15, 18, 24, 27, 45, … (vs Karatsuba-only 1, 3, 7, 9,
/// 17, 21, 25, 27, 51, …). The improvement compounds whenever the
/// recursion hits `e = 3`; for `e ∈ {2, 4, 8, 16}` the leaf count is
/// unchanged (reducing those counts would need additional fixed-degree
/// bilinear schemes).
///
/// `scratch` (shape `rows × n`, allocated once by the top-level
/// caller — see `matmul_bitsliced_planes`) is the sole leaf-
/// multiplication buffer for the **entire** recursion: every `e = 3`
/// base case reuses it via `poly_mul_3`, which calls
/// `m4rst::Mzd::matmul_into` against it rather than allocating a
/// fresh `Mzd` per leaf product — the same "m4rste calls m4rst"
/// relationship the rest of the crate uses, and it mirrors libm4rie's
/// `karatsuba.c` (2-3 scratch `mzd_t`s reused across an entire
/// degree's leaf products via `mzd_mul(t0, ..., 0)`). The `e = 1`
/// base case is left as a plain `Mzd::matmul` call: it needs exactly
/// one owned result for its return, and `Mzd::matmul` is already
/// `Mzd::new` + `matmul_into` internally, so routing it through
/// `scratch` would cost a `.clone()` for no savings.
fn poly_mul(a: &[Mzd], b: &[Mzd], rows: usize, n: usize, scratch: &mut Mzd) -> Vec<Mzd> {
    let e = a.len();
    debug_assert_eq!(b.len(), e);

    // Base case: a single F_2 matmul.
    if e == 1 {
        return vec![Mzd::matmul(&a[0], &b[0])];
    }

    // Cenk-Hasan 6-mult bilinear scheme for degree-2 polynomials
    // (3 coefficients each) — saves one F_2 matmul vs Karatsuba's
    // uneven 2+1 split (which would take 7). Propagates through the
    // recursion to e ∈ {3, 5, 6, 7, 9, …}.
    if e == 3 {
        return poly_mul_3(a, b, scratch);
    }

    // Split each operand at `h`: `p = p_lo + p_hi · x^h`, with the low
    // half `h = ⌈e/2⌉` coefficients and the high half `lo = e - h ≤ h`.
    let h = e.div_ceil(2);
    let lo = e - h;
    let (a_lo, a_hi) = a.split_at(h);
    let (b_lo, b_hi) = b.split_at(h);

    // The three Karatsuba sub-products.
    let p_lo = poly_mul(a_lo, b_lo, rows, n, scratch); // len 2h-1
    let p_hi = poly_mul(a_hi, b_hi, rows, n, scratch); // len 2*lo-1

    // mid operands: (a_lo + a_hi), (b_lo + b_hi), each length `h`
    // (a_hi / b_hi are shorter — XOR covers their `lo` prefix).
    let mut mid_a: Vec<Mzd> = a_lo.to_vec();
    let mut mid_b: Vec<Mzd> = b_lo.to_vec();
    for i in 0..lo {
        xor_into(&mut mid_a[i], &a_hi[i]);
        xor_into(&mut mid_b[i], &b_hi[i]);
    }
    let mut p_mid = poly_mul(&mid_a, &mid_b, rows, n, scratch); // len 2h-1

    // middle = p_mid - p_lo - p_hi  (XOR in char 2).
    for (i, c) in p_lo.iter().enumerate() {
        xor_into(&mut p_mid[i], c);
    }
    for (i, c) in p_hi.iter().enumerate() {
        xor_into(&mut p_mid[i], c);
    }

    // Assemble: p_lo at x^0, middle at x^h, p_hi at x^{2h}.
    let mut out: Vec<Mzd> = (0..2 * e - 1).map(|_| Mzd::new(rows, n)).collect();
    for (m, c) in p_lo.iter().enumerate() {
        xor_into(&mut out[m], c);
    }
    for (m, c) in p_mid.iter().enumerate() {
        xor_into(&mut out[m + h], c);
    }
    for (m, c) in p_hi.iter().enumerate() {
        xor_into(&mut out[m + 2 * h], c);
    }
    out
}

/// Degree-2 polynomial product via Cenk-Hasan's 6-multiplication
/// bilinear scheme (vs Karatsuba's 7 for an uneven 2+1 split).
///
/// `a = a0 + a1·x + a2·x²` and `b = b0 + b1·x + b2·x²`; result has 5
/// coefficients. Six leaf F_2 matmuls — `a_i·b_i` for i ∈ {0,1,2} plus
/// the three sums `(a0+a1)·(b0+b1)`, `(a1+a2)·(b1+b2)`,
/// `(a0+a2)·(b0+b2)`. Reconstruction is XOR-only.
///
/// `p1 = a0·b0` and `p2 = a2·b2` each feed exactly one output
/// coefficient (`c0`, `c4`), so they're computed straight into that
/// slot via `m4rst::Mzd::matmul_into` — no scratch, no extra copy.
/// The other four products (`p3..p6`) each feed 2-3 coefficients, so
/// they're computed one at a time into the shared `scratch` buffer
/// and XORed out immediately, the way libm4rie's `karatsuba.c` reuses
/// its `t0`/`t1`/`t2` scratch `mzd_t`s across an entire degree's leaf
/// products instead of allocating one per product.
fn poly_mul_3(a: &[Mzd], b: &[Mzd], scratch: &mut Mzd) -> Vec<Mzd> {
    debug_assert_eq!(a.len(), 3);
    debug_assert_eq!(b.len(), 3);
    let (a0, a1, a2) = (&a[0], &a[1], &a[2]);
    let (b0, b1, b2) = (&b[0], &b[1], &b[2]);
    let rows = a0.rows();
    let n = b0.cols();

    // Six evaluation linear combinations on the operands.
    let mut a01 = a0.clone();
    xor_into(&mut a01, a1);
    let mut a12 = a1.clone();
    xor_into(&mut a12, a2);
    let mut a02 = a0.clone();
    xor_into(&mut a02, a2);
    let mut b01 = b0.clone();
    xor_into(&mut b01, b1);
    let mut b12 = b1.clone();
    xor_into(&mut b12, b2);
    let mut b02 = b0.clone();
    xor_into(&mut b02, b2);

    // c0 = p1, c4 = p2 — direct writes, no scratch needed.
    let mut c0 = Mzd::new(rows, n);
    Mzd::matmul_into(&mut c0, a0, b0); // p1 = a0·b0
    let mut c4 = Mzd::new(rows, n);
    Mzd::matmul_into(&mut c4, a2, b2); // p2 = a2·b2

    // c1, c2, c3 each combine multiple products — accumulate through
    // `scratch`, one leaf F_2 matmul at a time.
    //   c1 = p3 + p1 + p6           (= a0·b1 + a1·b0)
    //   c2 = p5 + p1 + p2 + p6      (= a0·b2 + a1·b1 + a2·b0)
    //   c3 = p4 + p2 + p6           (= a1·b2 + a2·b1)
    let mut c1 = Mzd::new(rows, n);
    let mut c2 = Mzd::new(rows, n);
    let mut c3 = Mzd::new(rows, n);

    Mzd::matmul_into(scratch, &a01, &b01); // p3
    xor_into(&mut c1, scratch);

    Mzd::matmul_into(scratch, &a12, &b12); // p4
    xor_into(&mut c3, scratch);

    Mzd::matmul_into(scratch, &a02, &b02); // p5
    xor_into(&mut c2, scratch);

    Mzd::matmul_into(scratch, a1, b1); // p6
    xor_into(&mut c1, scratch);
    xor_into(&mut c2, scratch);
    xor_into(&mut c3, scratch);

    // Fold in p1 (still exactly c0's value) and p2 (c4's value); c0
    // and c4 themselves receive no further terms.
    xor_into(&mut c1, &c0);
    xor_into(&mut c2, &c0);
    xor_into(&mut c2, &c4);
    xor_into(&mut c3, &c4);

    vec![c0, c1, c2, c3, c4]
}

/// GF(2^e) matmul `C = A · B` via the bit-sliced + Karatsuba reduction
/// onto `m4rst`. See module docs.
///
/// The GF(2^e) layer does not apply a second Strassen recursion. The bit-plane
/// reduction delegates every leaf to `m4rst`, whose dispatcher already owns
/// the shape-appropriate M4RM or Strassen choice without materializing
/// GF(2^e) quadrants.
///
/// # Panics
///
/// Panics on field mismatch or inner-dimension mismatch.
pub fn matmul(a: &Mzed, b: &Mzed) -> Mzed {
    assert_eq!(a.ctx, b.ctx, "matmul: field mismatch");
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul: inner-dim mismatch ({}x{} * {}x{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    let ctx = a.ctx;
    let rows = a.rows();
    let inner = a.cols();
    let n = b.cols();
    if rows == 0 || n == 0 || inner == 0 {
        return Mzed::zeros(ctx, rows, n);
    }
    matmul_bitsliced(a, b)
}

/// Caller-owned-output counterpart to [`matmul`]. Writes directly into
/// `dst.packed` via [`cling_into`]
/// instead of allocating a fresh `Mzed` and copying, and its leaf F_2
/// products are computed via `m4rst::Mzd::matmul_into` against one
/// shared scratch buffer rather than allocated fresh per leaf (see
/// `poly_mul`/`poly_mul_3`).
///
/// This does not remove the format-conversion allocations in `slice` (the `e` F_2
/// bit-planes for each operand) or the `O(log e)` `Vec<Mzd>` combine
/// buffers inside `poly_mul`'s recursive Karatsuba split.
///
/// # Panics
///
/// On field mismatch, inner-dimension mismatch, or `dst` shape
/// mismatch.
pub fn matmul_into(dst: &mut Mzed, a: &Mzed, b: &Mzed) {
    assert_eq!(a.ctx, b.ctx, "matmul_into: field mismatch");
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_into: inner-dim mismatch ({}x{} * {}x{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    assert_eq!(dst.ctx, a.ctx, "matmul_into: field mismatch");
    assert_eq!(
        (dst.rows(), dst.cols()),
        (a.rows(), b.cols()),
        "matmul_into: dst shape mismatch"
    );
    let rows = a.rows();
    let inner = a.cols();
    let n = b.cols();
    if rows == 0 || n == 0 || inner == 0 {
        // dst may be a reused buffer with stale data; the empty-dims
        // path never reaches cling_into, so clear explicitly.
        dst.packed.fill_zero();
        return;
    }
    matmul_bitsliced_into(dst, a, b);
}

/// Shared core for [`matmul_bitsliced`] / [`matmul_bitsliced_into`]:
/// slice both operands, Karatsuba-multiply, and reduce mod the
/// minimal polynomial. Returns the `e` reduced result bit-planes.
fn matmul_bitsliced_planes(a: &Mzed, b: &Mzed) -> Vec<Mzd> {
    let ctx = a.ctx;
    let e = ctx.degree() as usize;
    let rows = a.rows();
    let n = b.cols();

    let a_planes = slice(a); // e planes, each rows × inner
    let b_planes = slice(b); // e planes, each inner × n

    // Sole leaf-multiplication scratch for the entire recursion — see
    // `poly_mul`'s doc comment.
    let mut scratch = Mzd::new(rows, n);

    // Raw polynomial product: c_raw[m] = Σ_{k+l=m} A_k · B_l, an F_2
    // matrix, for m ∈ [0, 2e-2]. Karatsuba + Cenk-Hasan e=3 base.
    let mut c_raw = poly_mul(&a_planes, &b_planes, rows, n, &mut scratch);

    // Reduce mod the minimal polynomial. For m from 2e-2 down to e,
    // x^m = x^{m-e}·x^e ≡ x^{m-e}·Σ_{j<e, minpoly bit j} x^j, so the
    // degree-m coefficient folds into c_raw[m-e+j]. High-to-low order
    // makes the cascade resolve in a single pass.
    let minpoly = ctx.minpoly();
    for m in (e..2 * e - 1).rev() {
        for j in 0..e {
            if (minpoly >> j) & 1 == 1 {
                let (lo, hi) = c_raw.split_at_mut(m);
                xor_into(&mut lo[m - e + j], &hi[0]);
            }
        }
    }

    // c_raw[0..e] are now the reduced result bit-planes.
    c_raw.truncate(e);
    c_raw
}

/// Bit-sliced + Karatsuba matmul body — the actual workhorse.
pub(crate) fn matmul_bitsliced(a: &Mzed, b: &Mzed) -> Mzed {
    cling(a.ctx, &matmul_bitsliced_planes(a, b))
}

/// `_into` sibling of [`matmul_bitsliced`] — writes through
/// [`cling_into`] instead of allocating.
pub(crate) fn matmul_bitsliced_into(dst: &mut Mzed, a: &Mzed, b: &Mzed) {
    cling_into(dst, &matmul_bitsliced_planes(a, b));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference matmul straight off `Gf2e::{mul, add}` — the
    /// independent oracle the bit-sliced path is checked against.
    fn naive_matmul(a: &Mzed, b: &Mzed) -> Vec<u16> {
        let f = a.ctx();
        let (m, k, n) = (a.rows(), a.cols(), b.cols());
        let mut out = vec![0u16; m * n];
        for i in 0..m {
            for kk in 0..k {
                let aik = a.get(i, kk);
                if aik == 0 {
                    continue;
                }
                for j in 0..n {
                    let bkj = b.get(kk, j);
                    if bkj != 0 {
                        out[i * n + j] ^= f.mul(aik, bkj);
                    }
                }
            }
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn random(f: Gf2e, rows: usize, cols: usize, seed: u64) -> Mzed {
        let mask: u16 = if f.degree() >= 16 {
            u16::MAX
        } else {
            (1u16 << f.degree()) - 1
        };
        let mut rng = Rng(seed | 1);
        let buf: Vec<u16> = (0..rows * cols)
            .map(|_| (rng.next() as u16) & mask)
            .collect();
        Mzed::from_u16_buffer(f, rows, cols, &buf)
    }

    #[test]
    fn slice_cling_round_trip() {
        let f = Gf2e::new(0x11B); // GF(256)
        let m = random(f, 7, 5, 0xABCDEF);
        let planes = slice(&m);
        assert_eq!(planes.len(), 8);
        let back = cling(f, &planes);
        assert_eq!(back.to_u16_buffer(), m.to_u16_buffer());
    }

    /// Exercise the SWAR fast paths for every power-of-2 `e`. Each
    /// `e` has its own deinterleave structure (e=1 memcpy, e=2/4
    /// bit-shuffle, e=8 multiply-trick, e=16 mask+shift gather); the
    /// round-trip must reproduce the input bit-for-bit, and at the
    /// same time the naive per-bit path (used for non-power-of-2 e)
    /// must agree element-by-element with the SWAR result.
    #[test]
    fn slice_cling_round_trip_all_powers_of_two() {
        // (minpoly, e, label) tuples — irreducibles of each degree
        // we want to exercise.
        let cases: &[(u64, u32, &str)] = &[
            (0b10, 1, "GF(2)"),         // x (minpoly for F_2 itself)
            (0b111, 2, "GF(4)"),        // x^2 + x + 1
            (0b1_0011, 4, "GF(16)"),    // x^4 + x + 1
            (0x11B, 8, "GF(256)"),      // AES poly
            (0x1_002D, 16, "GF(2^16)"), // x^16+x^5+x^3+x+1 (a primitive 16-bit poly)
        ];
        // Shape mix: covers word-aligned (cols a multiple of 64/e)
        // and non-aligned tails, plus a thin and a wide matrix.
        let shapes: &[(usize, usize)] = &[(1, 1), (1, 64), (4, 7), (3, 33), (5, 64), (7, 130)];
        for &(minpoly, e_expected, label) in cases {
            let f = Gf2e::new(minpoly);
            assert_eq!(f.degree(), e_expected, "{label} minpoly degree");
            for &(rows, cols) in shapes {
                let m = random(
                    f,
                    rows,
                    cols,
                    0xC0FFEE ^ minpoly ^ (rows as u64) ^ ((cols as u64) << 32),
                );
                let planes = slice(&m);
                assert_eq!(planes.len(), e_expected as usize, "{label} planes len");
                let back = cling(f, &planes);
                assert_eq!(
                    back.to_u16_buffer(),
                    m.to_u16_buffer(),
                    "{label} round-trip shape {rows}x{cols}"
                );
            }
        }
    }

    /// matmul correctness at e=16 (the largest supported field). This
    /// stresses both the e=16 slice/cling path and the deep recursive
    /// Karatsuba in `poly_mul`. Sizes kept small — the bit-sliced
    /// product at e=16 fans out to many leaf F_2 matmuls.
    #[test]
    fn matmul_correct_at_e_16() {
        // x^16 + x^5 + x^3 + x + 1 — irreducible over F_2.
        let f = Gf2e::new(0x1_002D);
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (2, 3, 2), (4, 5, 3)] {
            let a = random(f, m, k, 0xBEEF ^ (m as u64));
            let b = random(f, k, n, 0xCAFE ^ (n as u64));
            let got = matmul(&a, &b).to_u16_buffer();
            let want = naive_matmul(&a, &b);
            assert_eq!(got, want, "matmul ≠ naive at e=16 shape {m}x{k}x{n}");
        }
    }

    #[test]
    fn known_gf4_product() {
        // GF(4), x^2 + x + 1. Let x = 0b10 = 2.
        //   x · x = x + 1 = 3
        //   x · (x+1) = x^2 + x = 1
        // A = [[x, 1], [1, x+1]], B = [[x+1, 0], [x, x]]
        // C[0][0] = x·(x+1) + 1·x  = 1 + 2       = 3
        // C[0][1] = x·0     + 1·x  = 0 + 2       = 2
        // C[1][0] = 1·(x+1) + (x+1)·x = 3 + 1    = 2
        // C[1][1] = 1·0     + (x+1)·x = 0 + 1    = 1
        let f = Gf2e::new(0b111);
        let a = Mzed::from_u16_buffer(f, 2, 2, &[2, 1, 1, 3]);
        let b = Mzed::from_u16_buffer(f, 2, 2, &[3, 0, 2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.to_u16_buffer(), vec![3, 2, 2, 1]);
    }

    #[test]
    fn bit_sliced_matches_naive_across_fields() {
        // Mix of even and ODD e — odd e exercises Karatsuba's uneven
        // split: GF(4) e=2, GF(8) e=3, GF(16) e=4, GF(32) e=5,
        // GF(256) e=8 (two polys, one non-primitive).
        for &minpoly in &[
            0b111u64,      // GF(4)   x^2+x+1
            0b1011,        // GF(8)   x^3+x+1
            0b1_0011,      // GF(16)  x^4+x+1
            0b10_0101,     // GF(32)  x^5+x^2+1
            0b1_0010_1011, // GF(256) x^8+x^5+x^3+x+1
            0x11B,         // GF(256) AES poly (x not a generator)
        ] {
            let f = Gf2e::new(minpoly);
            for &(m, k, n) in &[
                (1usize, 1usize, 1usize),
                (3, 4, 2),
                (16, 16, 16),
                (13, 9, 11),
            ] {
                let a = random(f, m, k, 0x100 ^ minpoly ^ (m as u64));
                let b = random(f, k, n, 0x200 ^ minpoly ^ (n as u64));
                let got = matmul(&a, &b).to_u16_buffer();
                let want = naive_matmul(&a, &b);
                assert_eq!(
                    got, want,
                    "bit-sliced ≠ naive for minpoly={minpoly:#x} shape {m}x{k}x{n}"
                );
            }
        }
    }

    #[test]
    fn empty_dims() {
        let f = Gf2e::new(0b111);
        // inner dim 0 → all-zero result.
        let a = Mzed::zeros(f, 3, 0);
        let b = Mzed::zeros(f, 0, 4);
        let c = matmul(&a, &b);
        assert_eq!((c.rows(), c.cols()), (3, 4));
        assert_eq!(c.to_u16_buffer(), vec![0u16; 12]);
    }

    #[test]
    fn matmul_matches_naive_at_larger_sizes() {
        // Odd and rectangular shapes exercise the bit-sliced, Karatsuba, and
        // Cenk-Hasan paths beyond trivial cases.
        let shapes: &[(usize, usize, usize)] = &[
            (32, 32, 32),
            (33, 33, 33),
            (40, 32, 48),
            (35, 41, 37),
            (64, 64, 64),
            (50, 50, 50),
        ];
        for &minpoly in &[
            0b111u64, // GF(4)   e=2
            0b1011,   // GF(8)   e=3
            0b1_0011, // GF(16)  e=4
            0x11B,    // GF(256) e=8
        ] {
            let f = Gf2e::new(minpoly);
            for &(m, k, n) in shapes {
                let a = random(f, m, k, 0x500 ^ minpoly ^ (m as u64));
                let b = random(f, k, n, 0x600 ^ minpoly ^ (n as u64));
                let got = matmul(&a, &b).to_u16_buffer();
                let want = naive_matmul(&a, &b);
                assert_eq!(
                    got, want,
                    "matmul ≠ naive for minpoly={minpoly:#x} shape {m}x{k}x{n}"
                );
            }
        }
    }
}
